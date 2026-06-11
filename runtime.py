from __future__ import annotations

import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

# Add APIX-CORE to path for imports
_core_dir = Path(__file__).parent / "APIX-CORE"
if str(_core_dir) not in sys.path:
    sys.path.insert(0, str(_core_dir))

from tools.coding_tools import ToolError, execute_tool, get_tool_definitions
from tools.langchain_tools import set_workspace_root
from llm.llm_adapter import LlmNodeAdapter
from llm.llm_factory import BASE_URL

from .config import AgentConfig, AgentSpec, ModelConfig
from .events import RuntimeEvent
from .prompts import load_prompts

READ_ONLY_TOOLS = {"read", "grep", "find", "ls"}


class IntentOutput(BaseModel):
    """意图识别输出。

    用于识别用户请求的意图类型和相关元数据。
    """

    intent_type: str = "other"
    target_path: str | None = None
    required_reads: list[str] = Field(default_factory=list)
    success_criteria: str = "Fulfill the request with safe incremental steps."
    response_strategy: str = "Read first when uncertain, then edit."


def _extract_text_delta(content: Any) -> str:
    """从LLM响应内容中提取文本增量。

    Args:
        content: LLM响应内容，可能是字符串或列表

    Returns:
        str: 提取的文本增量
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("text"), str):
                    parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return ""


def _normalize_tool_calls(raw_calls: Any) -> list[dict[str, Any]]:
    """规范化工具调用数据。

    Args:
        raw_calls: 原始工具调用数据

    Returns:
        list[dict]: 规范化后的工具调用列表
    """
    tool_calls: list[dict[str, Any]] = []
    for call in raw_calls or []:
        name = call.get("name")
        call_id = call.get("id") or ""
        args = call.get("args")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if not isinstance(args, dict):
            args = {}
        if isinstance(name, str):
            tool_calls.append({"id": call_id, "name": name, "args": args})
    return tool_calls


def _resolve_model_api_key(model_config: ModelConfig) -> str:
    """Resolve API key for model config, raising if missing."""
    from .config import resolve_api_key
    import os

    api_key = resolve_api_key(model_config)
    if api_key:
        return api_key
    if model_config.provider.lower() in {"deepseek", "openai"}:
        fallback = os.getenv("OPENAI_API_KEY")
        if fallback:
            return fallback
    if model_config.api_key_env:
        raise RuntimeError(
            f"Missing API key: set {model_config.api_key_env} (or OPENAI_API_KEY for fallback)"
        )
    raise RuntimeError("Missing API key: no api_key_env configured")


class CodingRuntime:
    """编码运行时。

    管理LLM交互、工具执行和事件流生成。
    Uses APIX-CORE's LlmNodeAdapter for model creation and coding_tools for execution.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        agent_spec: AgentSpec,
        workspace_root: str,
        *,
        intent_recognition_enabled: bool = False,
        tool_execution_mode: str = "controlled",
        allow_unsafe_auto_exec: bool = False,
    ) -> None:
        """初始化运行时。

        Args:
            model_config: 模型配置
            agent_spec: Agent规格
            workspace_root: 工作空间根目录
            intent_recognition_enabled: 是否启用意图识别
            tool_execution_mode: 工具执行模式
            allow_unsafe_auto_exec: 是否允许不安全的自动执行
        """
        api_key = _resolve_model_api_key(model_config)
        self.model = LlmNodeAdapter.get_atapted_llm_node(
            provider=model_config.provider,
            model=model_config.model,
            api_key=api_key,
            config={"temperature": model_config.temperature},
        )
        self.prompts = load_prompts(workspace_root, agent_spec)
        self.workspace_root = workspace_root
        self.max_turns_default = 8
        self.agent_spec = agent_spec
        self.intent_recognition_enabled = intent_recognition_enabled
        self.tool_execution_mode = tool_execution_mode
        self.allow_unsafe_auto_exec = allow_unsafe_auto_exec
        self.allowed_tools = set(agent_spec.enabled_tools or [])
        self.tool_schemas = self._build_tool_schemas()

        # Set workspace root for langchain_tools compatibility
        set_workspace_root(workspace_root)

    def _build_tool_schemas(self) -> list[dict[str, Any]]:
        """Filter tool schemas based on configured policy."""
        if self.allow_unsafe_auto_exec:
            return get_tool_definitions()
        if self.allowed_tools:
            allowed = self.allowed_tools
        elif self.tool_execution_mode == "full_auto":
            allowed = set()
        else:
            allowed = READ_ONLY_TOOLS

        if not allowed:
            return get_tool_definitions()
        schemas = []
        for item in get_tool_definitions():
            fn = item.get("function", {})
            if fn.get("name") in allowed:
                schemas.append(item)
        return schemas

    def _is_tool_allowed(self, tool_name: str) -> bool:
        if self.allow_unsafe_auto_exec:
            return True
        if self.allowed_tools:
            return tool_name in self.allowed_tools
        if self.tool_execution_mode == "full_auto":
            return True
        if self.tool_execution_mode == "read_only":
            return tool_name in READ_ONLY_TOOLS
        return tool_name in READ_ONLY_TOOLS

    def recognize_intent(self, request: str, target_path: str) -> IntentOutput:
        """识别用户请求的意图。

        Args:
            request: 用户请求文本
            target_path: 目标路径提示

        Returns:
            IntentOutput: 识别出的意图信息
        """
        try:
            analyzer = self.model.with_structured_output(IntentOutput, method="function_calling")
            return analyzer.invoke(
                [
                    {
                        "role": "system",
                        "content": "Classify intent for a coding agent and provide actionable metadata.",
                    },
                    {
                        "role": "user",
                        "content": f"Request:\n{request}\n\nTarget hint:\n{target_path or '(none)'}",
                    },
                ]
            )
        except Exception:
            return IntentOutput()

    def run(self, request: str, target_path: str, max_turns: int | None = None) -> Iterator[RuntimeEvent]:
        """运行Agent工作流。

        Args:
            request: 用户请求
            target_path: 目标文件路径
            max_turns: 最大轮次，None时使用默认值

        Yields:
            RuntimeEvent: 运行时事件流
        """
        max_turns_value = max_turns or self.max_turns_default
        if self.intent_recognition_enabled:
            intent = self.recognize_intent(request, target_path)
        else:
            intent = IntentOutput()

        yield {"type": "agent_start"}
        if self.intent_recognition_enabled:
            yield {"type": "intent_recognized", "intent": intent.model_dump(mode="json")}
        else:
            yield {"type": "intent_skipped"}

        # 构建初始消息
        messages: list[Any] = [
            SystemMessage(content=self.prompts.agent),
            HumanMessage(
                content=(
                    f"User request:\n{request}\n\n"
                    f"Workspace root:\n{self.workspace_root}\n"
                    f"Target hint:\n{target_path or '(none)'}\n\n"
                    f"Intent:\n{intent.model_dump_json(indent=2)}"
                )
            ),
        ]

        # 绑定工具的LLM
        llm_with_tools = self.model.bind_tools(self.tool_schemas)

        # 执行多轮对话
        for turn in range(1, max_turns_value + 1):
            yield {"type": "turn_start", "turn": turn}
            yield {"type": "message_start", "turn": turn, "role": "assistant"}

            # 流式生成响应
            aggregate = None
            running_text = ""
            for chunk in llm_with_tools.stream(messages):
                aggregate = chunk if aggregate is None else aggregate + chunk
                delta = _extract_text_delta(chunk.content)
                if delta:
                    running_text += delta
                    yield {
                        "type": "message_delta",
                        "turn": turn,
                        "role": "assistant",
                        "delta": delta,
                        "text": running_text,
                    }

            # 检查是否有输出
            if aggregate is None:
                yield {"type": "error", "turn": turn, "error": "Model returned no output."}
                break

            # 提取工具调用
            tool_calls = _normalize_tool_calls(getattr(aggregate, "tool_calls", None))
            ai_message = AIMessage(content=aggregate.content or "", tool_calls=tool_calls)
            messages.append(ai_message)

            yield {
                "type": "message_end",
                "turn": turn,
                "role": "assistant",
                "text": running_text,
                "reason": "tool_calls" if tool_calls else "stop",
            }

            # 如果没有工具调用，结束
            if not tool_calls:
                final_response = running_text.strip() or "(empty response)"
                yield {"type": "turn_end", "turn": turn}
                yield {
                    "type": "agent_end",
                    "status": "DONE",
                    "turn": turn,
                    "final_response": final_response,
                }
                return

            # 执行工具调用
            for tool_call in tool_calls:
                tool_id = tool_call["id"] or f"call-{turn}-{tool_call['name']}"
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                yield {
                    "type": "tool_execution_start",
                    "turn": turn,
                    "tool_name": tool_name,
                    "tool_call_id": tool_id,
                    "args": tool_args,
                }

                if not self._is_tool_allowed(tool_name):
                    error_text = f"Tool blocked by policy: {tool_name}"
                    yield {
                        "type": "tool_blocked_by_policy",
                        "turn": turn,
                        "tool_name": tool_name,
                        "tool_call_id": tool_id,
                        "result": error_text,
                        "is_error": True,
                    }
                    messages.append(ToolMessage(content=error_text, tool_call_id=tool_id, name=tool_name))
                    continue

                update_queue: queue.Queue[str] = queue.Queue()
                try:
                    # Bash命令需要特殊处理以支持流式输出
                    if tool_name == "bash":
                        holder: dict[str, Any] = {}

                        def _run_bash() -> None:
                            try:
                                holder["result"] = execute_tool(
                                    self.workspace_root,
                                    tool_name,
                                    tool_args,
                                    on_update=lambda delta: update_queue.put(delta),
                                )
                            except Exception as exc:  # pragma: no cover
                                holder["error"] = exc

                        worker = threading.Thread(target=_run_bash, daemon=True)
                        worker.start()
                        while worker.is_alive() or not update_queue.empty():
                            while not update_queue.empty():
                                delta = update_queue.get_nowait()
                                yield {
                                    "type": "tool_execution_update",
                                    "turn": turn,
                                    "tool_name": tool_name,
                                    "tool_call_id": tool_id,
                                    "delta": delta,
                                }
                            time.sleep(0.02)
                        worker.join()
                        if "error" in holder:
                            raise holder["error"]
                        result = str(holder.get("result", ""))
                    else:
                        result = execute_tool(self.workspace_root, tool_name, tool_args, on_update=None)
                    yield {
                        "type": "tool_execution_end",
                        "turn": turn,
                        "tool_name": tool_name,
                        "tool_call_id": tool_id,
                        "result": result,
                        "is_error": False,
                    }
                    messages.append(ToolMessage(content=result, tool_call_id=tool_id, name=tool_name))
                except (ToolError, Exception) as exc:
                    error_text = f"Tool error: {exc}"
                    yield {
                        "type": "tool_execution_end",
                        "turn": turn,
                        "tool_name": tool_name,
                        "tool_call_id": tool_id,
                        "result": error_text,
                        "is_error": True,
                    }
                    messages.append(ToolMessage(content=error_text, tool_call_id=tool_id, name=tool_name))

            yield {"type": "turn_end", "turn": turn}

        # 达到最大轮次
        yield {
            "type": "agent_end",
            "status": "STOPPED",
            "reason": "max_turns_reached",
            "turn": max_turns_value,
            "final_response": "Stopped after reaching max_turns.",
        }


def create_runtime(
    model_config: ModelConfig,
    agent_config: AgentConfig,
    agent_spec: AgentSpec,
    workspace_root: str,
    *,
    intent_recognition_enabled: bool = False,
    tool_execution_mode: str = "controlled",
    allow_unsafe_auto_exec: bool = False,
) -> CodingRuntime:
    """创建运行时实例。

    Args:
        model_config: 模型配置
        agent_config: Agent配置
        agent_spec: Agent规格
        workspace_root: 工作空间根目录
        intent_recognition_enabled: 是否启用意图识别
        tool_execution_mode: 工具执行模式
        allow_unsafe_auto_exec: 是否允许不安全的自动执行

    Returns:
        CodingRuntime: 配置好的运行时实例
    """
    runtime = CodingRuntime(
        model_config=model_config,
        agent_spec=agent_spec,
        workspace_root=workspace_root,
        intent_recognition_enabled=intent_recognition_enabled,
        tool_execution_mode=tool_execution_mode,
        allow_unsafe_auto_exec=allow_unsafe_auto_exec,
    )
    runtime.max_turns_default = agent_config.max_turns
    return runtime
