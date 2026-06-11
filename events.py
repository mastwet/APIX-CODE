from __future__ import annotations

from typing import Any, Literal, TypedDict


class RuntimeEvent(TypedDict, total=False):
    """Runtime event payload."""

    type: Literal[
        "agent_start",
        "agent_end",
        "intent_recognized",
        "intent_skipped",
        "turn_start",
        "turn_end",
        "message_start",
        "message_delta",
        "message_end",
        "tool_execution_start",
        "tool_execution_update",
        "tool_execution_end",
        "tool_blocked_by_policy",
        "pipeline_start",
        "pipeline_stage_start",
        "pipeline_stage_end",
        "pipeline_end",
        "status",
        "error",
    ]
    turn: int
    role: str
    delta: str
    text: str
    final_response: str
    status: str
    reason: str
    intent: dict[str, Any]
    tool_name: str
    tool_call_id: str
    args: dict[str, Any]
    result: str
    error: str
    is_error: bool
    pipeline_name: str
    stage_count: int
    stage_id: str
    stage_agent: str
    stage_index: int
    stage_request: str
