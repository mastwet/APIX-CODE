from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional, TypedDict

import yaml

from langchain.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.messages import ToolMessage

from ..event.stream_writer import AgentStreamWriter, AgentStreamEvent
from .tool_descriptions import WRITE_TODOS_PROMPT, UPDATE_MEMORY_PROMPT, READ_MEMORY_PROMPT


# ------------------------------------------------------
# write_todos
# ------------------------------------------------------

@tool(description=WRITE_TODOS_PROMPT)
async def write_todos(
    todos: list[dict],
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:

    target = state.get("target", {})
    generation_id = state.get("generation_id")

    event_writer = AgentStreamWriter(generation_id)
    event_writer.send_event(
        event=AgentStreamEvent.TOOL_EXEC_START,
        data={
            "event_name": "tool_exec_chunk_rtn",
            "tool_name": "write_todos",
            "tool_call_id": tool_call_id,
            "content": str(todos),
            "chunk_position": "start",
            "status": "success",
        }
    )

    event_writer.send_event(
        event=AgentStreamEvent.TOOL_EXEC_END,
        data={
            "event_name": "tool_exec_chunk_rtn",
            "tool_name": "write_todos",
            "tool_call_id": tool_call_id,
            "content": "Finish",
            "chunk_position": "end",
            "status": "success",
        }
    )

    return Command(
        update={
            "todos": todos,
            "messages": [
                ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)
            ],
        }
    )


# ------------------------------------------------------
# Memory helpers
# ------------------------------------------------------

class Memory(TypedDict):
    title: str
    abstract: Optional[str]
    content: Optional[str]


def _get_memory_path(state: dict) -> Path:
    """Resolve the memory YAML file path from state."""
    work_dir = (state.get("config") or {}).get("work_dir", ".")
    memo_dir = Path(work_dir) / ".memo"
    memo_dir.mkdir(parents=True, exist_ok=True)
    return memo_dir / "memory.yaml"


def _load_memory(path: Path) -> list[dict]:
    """Load memory entries from YAML file."""
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or []
        if not isinstance(data, list):
            return []
        return data
    except Exception:
        return []


def _save_memory(path: Path, data: list[dict]):
    """Save memory entries to YAML file."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


# ------------------------------------------------------
# update_memory (local YAML, no external service)
# ------------------------------------------------------

@tool(description=UPDATE_MEMORY_PROMPT)
async def update_memory(
    memories: list[Memory],
    state: Annotated[dict, InjectedState] = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = None,
) -> Command:

    target = state.get("target", {})
    generation_id = state.get("generation_id")

    event_writer = AgentStreamWriter(generation_id)
    event_writer.send_event(
        event=AgentStreamEvent.TOOL_EXEC_START,
        data={
            "event_name": "tool_exec_chunk_rtn",
            "tool_name": "update_memory",
            "tool_call_id": tool_call_id,
            "content": "Update memories",
            "chunk_position": "start",
            "status": "success",
        }
    )

    if not memories:
        event_writer.send_event(
            event=AgentStreamEvent.TOOL_EXEC_END,
            data={
                "event_name": "tool_exec_chunk_rtn",
                "tool_name": "update_memory",
                "tool_call_id": tool_call_id,
                "content": "Empty memories",
                "chunk_position": "end",
                "status": "fail",
            }
        )
        return Command(update={
            "messages": [
                ToolMessage("Error: memories cannot be empty.", tool_call_id=tool_call_id)
            ]
        })

    # Validate titles
    for memory in memories:
        title = memory.get("title", "")
        if not title.strip():
            event_writer.send_event(
                event=AgentStreamEvent.TOOL_EXEC_END,
                data={
                    "event_name": "tool_exec_chunk_rtn",
                    "tool_name": "update_memory",
                    "tool_call_id": tool_call_id,
                    "content": "Empty title",
                    "chunk_position": "end",
                    "status": "fail",
                }
            )
            return Command(update={
                "messages": [
                    ToolMessage("Error: Title cannot be empty.", tool_call_id=tool_call_id)
                ]
            })

    try:
        memory_path = _get_memory_path(state)
        data = _load_memory(memory_path)
        actions: list[str] = []

        for memory in memories:
            title = memory["title"]
            abstract = memory.get("abstract")
            content = memory.get("content") or ""

            # Remove existing entries with same title
            data = [m for m in data if m.get("title") != title]

            if content.strip():
                # Create or update
                data.append({
                    "title": title,
                    "abstract": abstract,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "content": content,
                })
                actions.append(f"{title}: created/updated")
            else:
                # Delete (content empty)
                actions.append(f"{title}: deleted")

        _save_memory(memory_path, data)

        current_titles = [m.get("title", "") for m in data if m.get("title")]

        event_writer.send_event(
            event=AgentStreamEvent.TOOL_EXEC_END,
            data={
                "event_name": "tool_exec_chunk_rtn",
                "tool_name": "update_memory",
                "tool_call_id": tool_call_id,
                "content": "Batch update success",
                "chunk_position": "end",
                "status": "success",
            }
        )

        return Command(update={
            "messages": [
                ToolMessage(
                    "Memory operations completed:"
                    f"\n- " + "\n- ".join(actions)
                    + f"\n\n* Current available memory: {current_titles}.",
                    tool_call_id=tool_call_id,
                )
            ]
        })

    except Exception as e:
        event_writer.send_event(
            event=AgentStreamEvent.TOOL_EXEC_END,
            data={
                "event_name": "tool_exec_chunk_rtn",
                "tool_name": "update_memory",
                "tool_call_id": tool_call_id,
                "content": f"Error: {str(e)}",
                "chunk_position": "end",
                "status": "fail",
            }
        )
        return Command(update={
            "messages": [
                ToolMessage(f"Failed to update memories: {str(e)}", tool_call_id=tool_call_id)
            ]
        })


# ------------------------------------------------------
# read_memory (local YAML)
# ------------------------------------------------------

@tool(description=READ_MEMORY_PROMPT)
async def read_memory(
    title: Optional[str | list[str]],
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:

    target = state.get("target", {})
    generation_id = state.get("generation_id")

    event_writer = AgentStreamWriter(generation_id)
    event_writer.send_event(
        event=AgentStreamEvent.TOOL_EXEC_START,
        data={
            "event_name": "tool_exec_chunk_rtn",
            "tool_name": "read_memory",
            "tool_call_id": tool_call_id,
            "content": "Read memory",
            "chunk_position": "start",
            "status": "success",
        }
    )

    if not title:
        event_writer.send_event(
            event=AgentStreamEvent.TOOL_EXEC_END,
            data={
                "event_name": "tool_exec_chunk_rtn",
                "tool_name": "read_memory",
                "tool_call_id": tool_call_id,
                "content": "No title provided.",
                "chunk_position": "end",
                "status": "fail",
            }
        )
        return Command(update={
            "messages": [
                ToolMessage("A title is required.", tool_call_id=tool_call_id)
            ]
        })

    try:
        if isinstance(title, str):
            title = [title]

        memory_path = _get_memory_path(state)
        data = _load_memory(memory_path)

        memo_map = {m.get("title"): m for m in data if m.get("title")}
        contents: list[str] = []

        for t in title:
            memo = memo_map.get(t)
            if not memo:
                contents.append(f"No content found for title: {t}.")
            else:
                contents.append(
                    f"Title: {memo.get('title', '')}\n"
                    f"Date: {memo.get('date', '')}\n"
                    f"Content:\n{memo.get('content', '')}"
                )

        event_writer.send_event(
            event=AgentStreamEvent.TOOL_EXEC_END,
            data={
                "event_name": "tool_exec_chunk_rtn",
                "tool_name": "read_memory",
                "tool_call_id": tool_call_id,
                "content": f"Read {' '.join(title)}",
                "chunk_position": "end",
                "status": "success",
            }
        )

        return Command(update={
            "messages": [
                ToolMessage(
                    "\n\n---\n\n".join(contents) if contents else "No memory found.",
                    tool_call_id=tool_call_id,
                )
            ]
        })

    except Exception as e:
        event_writer.send_event(
            event=AgentStreamEvent.TOOL_EXEC_END,
            data={
                "event_name": "tool_exec_chunk_rtn",
                "tool_name": "read_memory",
                "tool_call_id": tool_call_id,
                "content": f"Error: {str(e)}",
                "chunk_position": "end",
                "status": "fail",
            }
        )
        return Command(update={
            "messages": [
                ToolMessage(f"Failed to read memory: {str(e)}", tool_call_id=tool_call_id)
            ]
        })
