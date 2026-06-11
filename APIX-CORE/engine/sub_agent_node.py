from uuid import uuid4

from openai import BadRequestError, RateLimitError

from langchain.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, AIMessageChunk, HumanMessage, ToolMessage, AIMessage
from langgraph.graph.state import Command
from langgraph.types import Overwrite

from ..event.stream_writer import AgentStreamWriter, AgentStreamEvent
from ..prompts.agent_prompts import *
from ..llm.llm_adapter import LlmNodeAdapter
from ..context.context_manager import ai_context_manager
from ..context.generating_cache import generating_cache
from ..commons.type_def import ConflictToolCalls, InvalidOutputsError, SubAgentState
from ..commons.logger import logger
from .agent_node_base import AgentNodeBase

MAX_RETRY = 8


def _guess_exception_type(err: str) -> str:
    """Classify an LLM error string into a retry category."""
    err_lower = err.lower()

    rate_limit_keywords = {
        "too many requests", "rate limit", "quota",
        "exceeded your current quota", "requests per min",
        "tokens per min", "rpm", "tpm", "concurrency",
    }
    if any(kw in err_lower for kw in rate_limit_keywords):
        return "rate_limit"

    token_keywords = {
        "context length", "maximum context", "token limit",
        "too many tokens", "max_tokens", "context_window",
        "request too large", "max context",
    }
    if any(kw in err_lower for kw in token_keywords):
        return "token_exceed"

    return "others"


class SubAgentNode(AgentNodeBase):

    def __init__(self, llm: BaseChatModel, tool_set: list[str]):
        super().__init__(llm, tool_set)

    # ------------------------------------------------------------------
    # History refresh (team_worker only)
    # ------------------------------------------------------------------

    async def _refresh_team_worker_history(
        self,
        *,
        state: SubAgentState,
        recent_messages,
        summary_text: str | None = None
    ):
        """Rewrite team worker history (optionally with summary)."""
        try:
            history_id = state.get("history_id")
            agent_name = state.get("agent_name")
            generation_id = state.get("generation_id")
            timestamp = state.get("timestamp")

            new_history = []

            if summary_text:
                new_history.append({
                    "role": "system",
                    "content": summary_text,
                    "timestamp": timestamp,
                    "generation_id": generation_id
                })

            for msg in recent_messages:
                if isinstance(msg, dict):
                    new_history.append(msg)
                else:
                    msg_dict = ai_context_manager.create_dict_message(
                        generation_id,
                        msg,
                        timestamp,
                        filter=True
                    )
                    if msg_dict:
                        new_history.append(msg_dict)

            await generating_cache.rewrite_history(
                history_id=history_id,
                agent_name=agent_name,
                messages=new_history
            )

        except Exception as e:
            logger.error(f"[context_summary] rewrite sub-agent history failed: {e}")

    # ------------------------------------------------------------------
    # Phase 1: context_prepare
    # ------------------------------------------------------------------

    async def context_prepare(self, state: SubAgentState) -> Command:
        """
        Extract input from state.
        For team_worker role, use generating_cache to load/append history.
        Create messages via ai_context_manager.create_agent_messages.
        No sandbox, no skills, no documents.
        """
        task_id = state.get("task_id") or str(uuid4())

        agent_role = state.get("agent_role")
        config = state.get("config", {})
        generation_id = state.get("generation_id")
        history_id = state.get("history_id")
        timestamp = state.get("timestamp")

        enable_think = config.get("enable_think", False)
        keep_tools_message = config.get("keep_tools_message")

        input_msg = state["input"]

        if not input_msg:
            raise RuntimeError("Error: Attempt invoke agent without input.")

        client_message = input_msg

        if client_message.get("role") == "human":
            client_message.update({
                "timestamp": timestamp,
                "generation_id": generation_id,
            })

            if agent_role == "team_worker":
                await generating_cache.append_dict_message(
                    history_id=history_id,
                    agent_name=state.get("agent_name"),
                    message_dict=client_message
                )

                history_messages = await generating_cache.load_history(
                    history_id=history_id,
                    agent_name=state.get("agent_name"),
                )

                client_messages = history_messages
            else:
                client_messages = [client_message]

            messages = ai_context_manager.create_agent_messages(
                client_messages,
                keep_tools_message,
                reasoning=enable_think
            )
            return Command(
                update={
                    "messages": messages,
                    "task_id": task_id,
                }
            )
        else:
            raise TypeError("Unknown role when invoke sub-agent.")

    # ------------------------------------------------------------------
    # Phase 2: context_summary
    # ------------------------------------------------------------------

    async def context_summary(self, state: SubAgentState) -> Command:
        """
        Context compression node (multi-level).

        Levels:
            0: no-op
            1: drop_tool_messages (light, reversible)
            2: LLM summary (lossy)
            3: drop_tool_messages(min_keep=2)
            4+: exponential truncate (reversible)
        """
        logger.trace('[sub_agent_node.py] [SubAgentNode] [context_summary] Enter')

        agent_role = state.get("agent_role")
        config = state.get("config", {})
        enable_shortterm_memory = config.get("enable_shortterm_memory")
        summary_trigger_threshold = config.get("summary_trigger_threshold")
        summary_exempt_tail_length = config.get("summary_exempt_tail_length")

        llm_retry_count = state.get("llm_retry_count", 0)
        last_error = state.get("error", "")
        context_compress_level = state.get("context_compress_level", 0)
        shortterm_memory = state.get("shortterm_memory", "")

        messages = state.get("messages", [])

        threshold = max(16, summary_trigger_threshold)
        keep_recent_base = max(8, summary_exempt_tail_length)

        should_trigger = (
            len(messages) >= threshold
            or (llm_retry_count > 0 and last_error == "token_exceed")
        )

        if len(messages) >= threshold:
            context_compress_level = max(context_compress_level, 2)

        if not should_trigger:
            return Command(update={})

        logger.info(
            f"[context_summary] Triggered. "
            f"len={len(messages)} level={context_compress_level} "
            f"retry={llm_retry_count} error={last_error}"
        )

        def calc_keep(level: int) -> int:
            keep = keep_recent_base // (2 ** max(1, level - 3))
            return max(2, keep)

        # Case 1: shortterm memory disabled — always truncate
        if not enable_shortterm_memory:
            keep_recent = calc_keep(context_compress_level)
            recent_messages = messages[-keep_recent:]

            if agent_role == "team_worker":
                await self._refresh_team_worker_history(
                    state=state,
                    recent_messages=recent_messages
                )

            logger.success(
                f"[context_summary] Truncate (memory disabled). keep={keep_recent}"
            )
            return Command(
                update={
                    "messages": Overwrite(recent_messages),
                }
            )

        # Level 0: no-op
        if context_compress_level <= 0:
            return Command(update={})

        # Level 1: drop tool messages
        if context_compress_level == 1:
            new_messages = [
                m for m in messages if not isinstance(m, ToolMessage)
            ]
            if len(new_messages) < keep_recent_base:
                new_messages = messages[-keep_recent_base:]

            logger.success(
                f"[context_summary] Level1 drop_tool_messages "
                f"(min_keep={keep_recent_base})"
            )
            return Command(
                update={
                    "messages": Overwrite(new_messages),
                }
            )

        # Level 2: LLM summary
        if context_compress_level == 2:
            # Split messages
            to_process = messages[:-keep_recent_base]
            recent_messages = messages[-keep_recent_base:]

            # Filter summarizable messages
            sys_msgs = [m for m in to_process if isinstance(m, SystemMessage)]
            to_summarize = [
                m for m in to_process
                if isinstance(m, (HumanMessage, AIMessage, AIMessageChunk))
            ]

            if not to_summarize:
                return Command(update={})

            # Limit summarize size
            max_summarize_messages = summary_trigger_threshold
            to_summarize = to_summarize[-max_summarize_messages:]

            # Build prompt
            summary_prompt = [
                SystemMessage(content=DEFAULT_SUMMARY_PROMPT)
            ]

            if shortterm_memory:
                summary_prompt.append(
                    HumanMessage(
                        content=self.SUMMARY_MEMORY_PREFIX + shortterm_memory
                    )
                )

            summary_prompt.extend(to_summarize)
            summary_prompt.append(
                HumanMessage(content=self.SUMMARY_INSTRUCTION_PROMPT)
            )

            # Call LLM
            try:
                summary_msg: AIMessage = await LlmNodeAdapter.ainvoke(
                    llm_node=self.llm,
                    input=summary_prompt,
                    reasoning=True,
                    fall_back_config=config
                )
            except Exception as e:
                logger.error(f"[context_summary] Summary failed: {e}")
                return Command(update={})

            summary_text = summary_msg.content.strip()

            if agent_role == "team_worker":
                await self._refresh_team_worker_history(
                    state=state,
                    recent_messages=recent_messages,
                    summary_text=summary_text
                )

            logger.success(
                f"[context_summary] Level2 summary done. "
                f"remain={len(recent_messages)}"
            )
            return Command(
                update={
                    "messages": Overwrite(sys_msgs + recent_messages),
                    "shortterm_memory": summary_text
                }
            )

        # Level 3: aggressive drop tool messages
        if context_compress_level == 3:
            new_messages = [
                m for m in messages if not isinstance(m, ToolMessage)
            ]
            if len(new_messages) < 2:
                new_messages = messages[-2:]

            logger.success("[context_summary] Level 3 drop_tool_messages(min_keep=2)")
            return Command(
                update={
                    "messages": Overwrite(new_messages),
                }
            )

        # Level >=4: exponential truncate
        keep_recent = calc_keep(context_compress_level)
        recent_messages = messages[-keep_recent:]

        logger.success(
            f"[context_summary] Level{context_compress_level} truncate "
            f"(keep={keep_recent})"
        )
        return Command(
            update={
                "messages": Overwrite(recent_messages),
            }
        )

    # ------------------------------------------------------------------
    # Phase 3: llm_call
    # ------------------------------------------------------------------

    async def llm_call(self, state: SubAgentState) -> Command:
        """
        Build system prompt with sub-agent specific prompt.
        Call LLM with current conversation state.
        """
        logger.trace("[sub_agent_node.py] [SubAgentNode] [llm_call] Enter")

        agent_role = state.get("agent_role")
        target = state.get("target")
        generation_id = state.get("generation_id")
        config = state.get("config", {})
        llm_calls_warning_threshold = config.get("llm_calls_warning_threshold", 8)
        enable_think = config.get("enable_think", False)

        event_writer = AgentStreamWriter(generation_id)
        messages = state["messages"]

        logger.info(
            f"[llm_call] Invoke llm with {len(messages)} messages"
        )

        # Build system prompt
        system_prompt_text = self._load_prompt(agent_role)
        llm_input = [SystemMessage(content=system_prompt_text)] + list(messages)

        # Inject alert if necessary
        need_alert = self._should_inject_alert(
            llm_calls=state.get("llm_calls", 0),
            threshold=llm_calls_warning_threshold,
        )
        if need_alert:
            logger.warning(
                f"[llm_call] Inject SYSTEM_ALERT_PROMPT: {self.SYSTEM_ALERT_PROMPT}"
            )
            llm_input.append(SystemMessage(self.SYSTEM_ALERT_PROMPT))

        if state.get("error_detail"):
            logger.warning(
                f"[llm_call] Inject CRITICAL WARN: {state.get('error_detail')}"
            )
            llm_input.append(
                SystemMessage(
                    f"CRITICAL WARN: {state.get('error_detail')}. "
                    "If you are trying to do that, stop immediately any way!"
                )
            )

        # Start streaming
        chunk_iterator = LlmNodeAdapter.astream(
            llm_node=self.llm,
            input=llm_input,
            reasoning=enable_think,
            fall_back_config=config,
        )

        event_writer.send_event(
            event=AgentStreamEvent.LLM_STREAM_START,
            target=target,
            data={
                "event_name": "node_stream_start",
                "content": "[Start LLM Response (sub-agent node)]",
            },
        )

        ai_msg_chunk = AIMessageChunk(content="")
        chunk_num = 0

        # Stream loop
        try:
            async for chunk in chunk_iterator:
                chunk_num += 1
                ai_msg_chunk = ai_msg_chunk + chunk

                think = (
                    chunk.additional_kwargs.get("reasoning_content")
                    if chunk.additional_kwargs
                    else None
                )
                content = chunk.text
                tool_calls = chunk.tool_calls or chunk.tool_call_chunks

                if think:
                    event_writer.send_event(
                        event=AgentStreamEvent.LLM_CHUNK_RETURN,
                        target=target,
                        data={"event_name": "think_chunk_rtn", "content": think},
                    )
                elif content:
                    event_writer.send_event(
                        event=AgentStreamEvent.LLM_CHUNK_RETURN,
                        target=target,
                        data={"event_name": "content_chunk_rtn", "content": content},
                    )
                if tool_calls:
                    event_writer.send_event(
                        event=AgentStreamEvent.LLM_CHUNK_RETURN,
                        target=target,
                        data={"event_name": "tool_chunk_rtn", "content": tool_calls},
                    )

            # Emit pending tool exec events
            if ai_msg_chunk.tool_calls:
                for tool_call in ai_msg_chunk.tool_calls:
                    event_writer.send_event(
                        event=AgentStreamEvent.TOOL_EXEC_START,
                        target=target,
                        data={
                            "event_name": "tool_exec_chunk_rtn",
                            "tool_name": tool_call.get("name"),
                            "tool_call_id": tool_call.get("id"),
                            "content": "Args: " + str(tool_call.get("args")),
                            "chunk_position": "pending",
                            "status": "success",
                        },
                    )

            ai_msg_chunk = self._ensure_agent_message(
                ai_msg_chunk, reasoning=enable_think
            )

        except ConflictToolCalls as e:
            llm_retry_count = state.get("llm_retry_count", 0) + 1
            logger.warning(
                f"[llm_call] Error: {type(e).__name__}; "
                f"Retry ({llm_retry_count}/{MAX_RETRY})..."
            )
            event_writer.send_event(
                event=AgentStreamEvent.RUNTIME_WARNING,
                target=target,
                data={
                    "event_name": "conflict_tool_calls_warning",
                    "content": {
                        "tool_name": " ".join(e.errors),
                        "retry": llm_retry_count,
                    },
                },
            )
            if llm_retry_count < MAX_RETRY:
                return Command(
                    update={
                        "llm_retry_count": llm_retry_count,
                        "error": "others",
                        "error_detail": e.message,
                    },
                )
            raise

        except InvalidOutputsError as e:
            llm_retry_count = state.get("llm_retry_count", 0) + 1
            logger.warning(
                f"[llm_call] Error: {type(e).__name__}; "
                f"Retry ({llm_retry_count}/{MAX_RETRY})..."
            )
            event_writer.send_event(
                event=AgentStreamEvent.RUNTIME_WARNING,
                target=target,
                data={
                    "event_name": "invalid_outputs_warning",
                    "content": {"retry": llm_retry_count},
                },
            )
            if llm_retry_count < MAX_RETRY:
                return Command(
                    update={
                        "llm_retry_count": llm_retry_count,
                        "error": "others",
                    },
                )
            raise

        except BadRequestError as e:
            llm_retry_count = state.get("llm_retry_count", 0) + 1
            context_compress_level = state.get("context_compress_level", 0) + 1
            logger.warning(
                f"[llm_call] Error: {type(e).__name__}; "
                f"Retry ({llm_retry_count}/{MAX_RETRY})..."
            )
            event_writer.send_event(
                event=AgentStreamEvent.RUNTIME_WARNING,
                target=target,
                data={
                    "event_name": "bad_request_warning",
                    "content": {"message": e.message, "retry": llm_retry_count},
                },
            )
            if (
                llm_retry_count < MAX_RETRY
                and _guess_exception_type(str(e)) == "token_exceed"
            ):
                return Command(
                    update={
                        "llm_retry_count": llm_retry_count,
                        "error": "token_exceed",
                        "context_compress_level": context_compress_level,
                    },
                )
            raise

        except RateLimitError as e:
            llm_retry_count = state.get("llm_retry_count", 0) + 1
            logger.warning(
                f"[llm_call] Error: {type(e).__name__}; "
                f"Retry ({llm_retry_count}/{MAX_RETRY})..."
            )
            event_writer.send_event(
                event=AgentStreamEvent.RUNTIME_WARNING,
                target=target,
                data={
                    "event_name": "rate_limit_warning",
                    "content": {"message": e.message, "retry": llm_retry_count},
                },
            )
            if (
                llm_retry_count < MAX_RETRY
                and _guess_exception_type(str(e)) == "rate_limit"
            ):
                return Command(
                    update={
                        "llm_retry_count": llm_retry_count,
                        "error": "rate_limit",
                    },
                )
            raise

        logger.info(f"[llm_call] Generate chunks num: {chunk_num}")

        # End streaming
        event_writer.send_event(
            event=AgentStreamEvent.LLM_STREAM_END,
            target=target,
            data={
                "event_name": "node_stream_end",
                "content": "[Finish LLM Response] (sub-agent node)",
            },
        )

        delta_msg = [ai_msg_chunk]
        if need_alert:
            delta_msg = [SystemMessage(self.SYSTEM_ALERT_PROMPT), ai_msg_chunk]

        return Command(
            update={
                "messages": delta_msg,
                "llm_calls": 1,
                "llm_retry_count": 0,
                "context_compress_level": (
                    0 if state.get("context_compress_level", 0) <= 2 else 3
                ),
                "error": "",
                "error_detail": "",
            }
        )

    # ------------------------------------------------------------------
    # Phase 4: messages_persist
    # ------------------------------------------------------------------

    async def messages_persist(self, state: SubAgentState) -> Command:
        """
        Persist messages for team_worker to generating_cache.
        For other roles, just emit stream events.
        """
        messages = state.get("messages", [])
        if not messages:
            return Command(update={})
        last_message = messages[-1]

        agent_role = state.get("agent_role")
        agent_name = state.get("agent_name")
        task_id = state.get("task_id")
        generation_id = state.get("generation_id")
        target = state.get("target")
        history_id = state.get("history_id")
        timestamp = state.get("timestamp")

        config = state.get("config", {})
        model_name = config.get("model_name")
        model_provider = config.get("models_provider")

        current_tool_calls = []
        event_writer = AgentStreamWriter(generation_id)

        # Case 1: AIMessage (may contain tool calls)
        if isinstance(last_message, (AIMessage, AIMessageChunk)):
            if last_message.tool_calls:
                current_tool_calls = last_message.tool_calls

            # Yield delta content output
            delta_outputs = last_message.content or ""
            event_writer.send_event(
                event=AgentStreamEvent.AI_MESSAGE_RETURN,
                target=target,
                data={
                    "event_name": "output_chunk_rtn",
                    "content": state.get("outputs", "") + delta_outputs
                }
            )

            # Persist single AI message for team worker
            if agent_role == "team_worker":
                client_message = ai_context_manager.create_dict_message(
                    generation_id,
                    last_message,
                    timestamp,
                    filter=True,
                    fallback_model_name=model_name,
                    fallback_model_provider=model_provider,
                    fallback_timestamp=timestamp
                )
                await generating_cache.append_dict_message(
                    history_id=history_id,
                    agent_name=agent_name,
                    message_dict=client_message
                )

            return Command(
                update={
                    "current_tool_calls": current_tool_calls,
                    "outputs": delta_outputs
                }
            )

        # Case 2: ToolMessage (batch return from ToolNode)
        if isinstance(last_message, ToolMessage):
            tool_calls = state.get("current_tool_calls", [])
            tool_call_ids = {call["id"] for call in tool_calls}

            # Filter relevant ToolMessage
            tool_msg_list = [
                msg for msg in messages
                if isinstance(msg, ToolMessage) and msg.tool_call_id in tool_call_ids
            ]

            # Deduplicate by tool_call_id (keep latest)
            dedup_map = {}
            for msg in tool_msg_list:
                dedup_map[msg.tool_call_id] = msg

            deduped_tool_msgs = list(dedup_map.values())

            # Persist all tool messages in batch
            for msg in deduped_tool_msgs:
                tool_message = ai_context_manager.create_dict_message(
                    generation_id,
                    msg,
                    timestamp,
                    filter=True,
                    fallback_model_name=model_name,
                    fallback_model_provider=model_provider,
                    fallback_timestamp=timestamp
                )
                # Write tool calls log
                await logger.write_log("sub_agent_logs", task_id, tool_message)

                # Persist tool message for team worker
                if agent_role == "team_worker":
                    await generating_cache.append_dict_message(
                        history_id=history_id,
                        agent_name=agent_name,
                        message_dict=tool_message
                    )

            return Command(
                update={
                    "current_tool_calls": []
                }
            )

        # Case 3: Other message types (Human/System/etc.), only persist for team worker
        elif agent_role == "team_worker":
            client_message = ai_context_manager.create_dict_message(
                generation_id,
                last_message,
                timestamp,
                filter=True,
                fallback_model_name=model_name,
                fallback_model_provider=model_provider,
                fallback_timestamp=timestamp
            )

            await generating_cache.append_dict_message(
                history_id=history_id,
                agent_name=agent_name,
                message_dict=client_message
            )

            return Command(
                update={
                    "current_tool_calls": current_tool_calls
                }
            )

        return Command(update={})
