from langgraph.prebuilt import ToolNode, ToolRuntime

from typing import Literal

from langchain_core.messages import (
    ToolCall,
    ToolMessage,
)
from langgraph.types import Command


class ApixToolNode(ToolNode):
    """
    Extended ToolNode with per-call error handling.
    Wraps _arun_one so a single tool failure returns a ToolMessage
    with status='error' instead of aborting the whole graph.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def _arun_one(
        self,
        call: ToolCall,
        input_type: Literal["list", "dict", "tool_calls"],
        tool_runtime: ToolRuntime,
    ) -> ToolMessage | Command:
        try:
            return await super()._arun_one(call, input_type, tool_runtime)
        except Exception as e:
            return ToolMessage(
                content=repr(e),
                name=call["name"],
                tool_call_id=call["id"],
                status="error",
            )
