from openai import BadRequestError, RateLimitError

from langchain.chat_models import BaseChatModel
from langchain_core.messages import (
    SystemMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
    AIMessage,
)
from langgraph.graph.state import Command
from langgraph.types import Overwrite

from ..prompts.agent_prompts import *
from ..llm.llm_adapter import LlmNodeAdapter
from ..event.stream_writer import AgentStreamWriter, AgentStreamEvent
from ..commons.type_def import MainAgentState, ConflictToolCalls, InvalidOutputsError
from ..commons.logger import logger
from .agent_node_base import AgentNodeBase
import time
from ..context.context_manager import ai_context_manager

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


class MainAgentNode(AgentNodeBase):
    """
    Primary agent node implementing the four-phase graph:
    context_prepare -> context_summary -> llm_call -> messages_persist
    """

    def __init__(self, llm: BaseChatModel, tool_set: list[str]):
        super().__init__(llm, tool_set)

    # ------------------------------------------------------------------
    # Phase 1: context_prepare
    # ------------------------------------------------------------------

    async def context_prepare(self, state: MainAgentState) -> Command:
        """
        Extract input message and create initial HumanMessage.
        Initialize memorandum list for memory-aware prompting.
        """
        logger.trace("[agent_node.py] [MainAgentNode] [context_prepare] Enter")

        config = state.get("config", {})

        input_msg = state.get("input")
        if not input_msg:
            raise RuntimeError("Error: Attempt invoke agent without input.")

        # Initialize memorandum
        ai_context_manager.init_memorandum_list(state)

        # Build a HumanMessage from the raw input dict
        content = input_msg.get("content", "") if isinstance(input_msg, dict) else str(input_msg)
        human_message = HumanMessage(content=content)

        return Command(
            update={
                "messages": [human_message],
                "sandbox": "",
            }
        )

    # ------------------------------------------------------------------
    # Phase 2: context_summary
    # ------------------------------------------------------------------

    async def context_summary(self, state: MainAgentState) -> Command:
        """
        Multi-level context compression (simplified).

        Levels:
            0: no-op
            1: drop tool messages
            2+: truncate to keep_recent messages

        Triggered when message count >= threshold OR retry from token_exceed.
        No LLM summary — just structural compression.
        """
        logger.trace("[agent_node.py] [MainAgentNode] [context_summary] Enter")

        config = state.get("config", {})
        summary_trigger_threshold = config.get("summary_trigger_threshold", 16)
        summary_exempt_tail_length = config.get("summary_exempt_tail_length", 8)

        llm_retry_count = state.get("llm_retry_count", 0)
        last_error = state.get("error", "")
        context_compress_level = state.get("context_compress_level", 0)

        messages = state.get("messages", [])
        threshold = max(16, summary_trigger_threshold)
        keep_recent_base = max(8, summary_exempt_tail_length)

        # Determine if compression should trigger
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

        # Level 0: no-op
        if context_compress_level <= 0:
            return Command(update={})

        # Level 1: drop tool messages
        if context_compress_level == 1:
            new_messages = [
                m for m in messages if not isinstance(m, ToolMessage)
            ]
            # Always keep at least keep_recent_base messages
            if len(new_messages) < keep_recent_base:
                new_messages = messages[-keep_recent_base:]

            logger.success(
                f"[context_summary] Level1 drop_tool_messages "
                f"(min_keep={keep_recent_base})"
            )
            return Command(update={"messages": Overwrite(new_messages)})

        # Level 2+: truncate to keep_recent
        keep_recent = calc_keep(context_compress_level)
        recent_messages = messages[-keep_recent:]

        logger.success(
            f"[context_summary] Level{context_compress_level} truncate "
            f"(keep={keep_recent})"
        )
        return Command(update={"messages": Overwrite(recent_messages)})

    # ------------------------------------------------------------------
    # Phase 3: llm_call
    # ------------------------------------------------------------------

    async def llm_call(self, state: MainAgentState) -> Command:
        """
        Build system prompt, stream LLM response, emit stream events.
        Handles ConflictToolCalls, InvalidOutputsError, BadRequestError,
        and RateLimitError with retry logic.
        """
        logger.trace("[agent_node.py] [MainAgentNode] [llm_call] Enter")

        # Config
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

        # Load base rule prompt
        state["rule_prompt"] = self._load_prompt(agent_role)

        # Build rich system prompts
        system_prompts = []

        # Role prompt
        role_msgs = ai_context_manager.create_role_prompt_list(state, agent_role)
        system_prompts.extend(role_msgs)

        # System prompt (rules + workflow)
        sys_msgs = ai_context_manager.create_system_prompt_list(state, agent_role)
        system_prompts.extend(sys_msgs)

        # Runtime prompts
        runtime_parts = []

        # Workspace info
        workspace_prompt = ai_context_manager.create_workspace_prompt(state, agent_role)
        if workspace_prompt:
            runtime_parts.append(workspace_prompt)

        # Skills
        skills_prompt = ai_context_manager.create_skills_prompt(state, agent_role)
        if skills_prompt:
            runtime_parts.append(skills_prompt)

        # Memorandum
        memo_prompt = ai_context_manager.create_memorandum_prompt(state, agent_role)
        if memo_prompt:
            runtime_parts.append(memo_prompt)

        # Todo list
        todo_prompt = ai_context_manager.create_todo_prompt(state, agent_role)
        if todo_prompt:
            runtime_parts.append(todo_prompt)

        # Documents
        docs_prompt = ai_context_manager.create_documents_prompt(state, agent_role)
        if docs_prompt:
            runtime_parts.append(docs_prompt)

        if runtime_parts:
            system_prompts.append(SystemMessage(content='# [RUNTIME STATE]\n' + '\n'.join(runtime_parts)))

        # Combine system prompts with conversation messages
        full_messages = system_prompts + list(messages)

        # Inject alert if necessary
        need_alert = self._should_inject_alert(
            llm_calls=state.get("llm_calls", 0),
            threshold=llm_calls_warning_threshold,
        )
        if need_alert:
            logger.warning(
                f"[llm_call] Inject SYSTEM_ALERT_PROMPT: {self.SYSTEM_ALERT_PROMPT}"
            )
            full_messages.append(SystemMessage(self.SYSTEM_ALERT_PROMPT))

        if state.get("error_detail"):
            logger.warning(
                f"[llm_call] Inject CRITICAL WARN: {state.get('error_detail')}"
            )
            full_messages.append(
                SystemMessage(
                    f"CRITICAL WARN: {state.get('error_detail')}. "
                    "If you are trying to do that, stop immediately any way!"
                )
            )

        llm_input = full_messages

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
                "content": "[Start LLM Response (single node)]",
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
                "content": "[Finish LLM Response] (single node)",
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

    async def messages_persist(self, state: MainAgentState) -> Command:
        """
        Persist the last message to local state.
        Converts AIMessage to dict and emits persistence event.
        """
        logger.trace("[agent_node.py] [MainAgentNode] [messages_persist] Enter")

        messages = state.get("messages", [])
        if not messages:
            return Command(update={})

        last_message = messages[-1]

        if isinstance(last_message, (AIMessage, AIMessageChunk)):
            generation_id = state.get("generation_id", "")
            timestamp = int(time.time() * 1000)
            msg_dict = ai_context_manager.create_dict_message(
                generation_id, last_message, timestamp
            )
            if msg_dict:
                event_writer = AgentStreamWriter(generation_id)
                event_writer.send_event(
                    event=AgentStreamEvent.AI_MESSAGE_RETURN,
                    data={
                        "event_name": "messages_persist_end",
                        "content": "",
                    },
                )

        return Command(update={})
