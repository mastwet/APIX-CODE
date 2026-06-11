from __future__ import annotations

from langchain_core.tools.base import BaseTool


async def get_available_tools(
    permissions: list[str] = None,
    agent_role: str = '',
    workspace_configured: bool = True,
    client_id: str = '',
) -> list[BaseTool]:
    if permissions and 'forbidden' in permissions:
        return []

    from .file_ops import (
        read_workspace_file,
        write_workspace_file,
        delete_workspace_file,
        move_workspace_file,
        list_workspace_files,
        fetch_files,
    )
    from .code_runner import run_workspace_command, run_python_code
    from .todo import write_todos, update_memory, read_memory
    from .communication import request_user_input

    tools = [
        read_workspace_file,
        write_workspace_file,
        delete_workspace_file,
        move_workspace_file,
        list_workspace_files,
        run_workspace_command,
        run_python_code,
        write_todos,
        update_memory,
        read_memory,
        fetch_files,
    ]

    if workspace_configured:
        tools.append(request_user_input)

    # Sub-agent tools (only for main_agent/team_leader with agent_assign permission)
    if permissions and 'agent_assign' in permissions:
        from .sub_assistant import assign_sub_assistant, query_sub_assistant, stop_sub_assistant
        tools.extend([assign_sub_assistant, query_sub_assistant, stop_sub_assistant])

    return tools


conflict_tool_set: set[str] = set()
