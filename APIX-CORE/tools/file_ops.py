import os
import shutil
from pathlib import Path
from typing import Annotated, Optional

from langchain.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.messages import ToolMessage

from ..event.stream_writer import AgentStreamWriter, AgentStreamEvent
from .tool_descriptions import (
    FETCH_FILE_PROMPT,
    READ_WORKSPACE_FILE_PROMPT,
    WRITE_WORKSPACE_FILE_PROMPT,
    DELETE_WORKSPACE_FILE_PROMPT,
    MOVE_WORKSPACE_FILE_PROMPT,
    LIST_WORKSPACE_FILES_PROMPT,
)

# Maximum file size for reading (5 MB)
MAX_FILE_SIZE = 5 * 1024 * 1024
# Maximum output length for truncation
TOOLS_MAX_OUTPUT_LENGTH = 30000
# Max items for directory listing
MAX_FILES = 500
MAX_DEPTH = 6
# Directories ignored during recursive scanning
IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".pytest_cache", ".venv", "venv"}


# ------------------------------------------------------
# path sandbox check
# ------------------------------------------------------

def _get_work_dir(state: dict) -> str:
    """Extract work_dir from state config."""
    return (state.get('config') or {}).get('work_dir', '.')


def _sandbox_check(work_dir: str, file_path: str) -> Optional[str]:
    """
    Validate that file_path resolves within work_dir.
    Returns error message if path escapes workspace, None if OK.
    """
    try:
        resolved = (Path(work_dir) / file_path).resolve()
        work_resolved = Path(work_dir).resolve()
        if not str(resolved).startswith(str(work_resolved)):
            return "Error: Path escapes workspace."
    except Exception as e:
        return f"Error resolving path: {e}"
    return None


# ------------------------------------------------------
# fetch_files (stub - no file service in local mode)
# ------------------------------------------------------

@tool(description=FETCH_FILE_PROMPT)
async def fetch_files(
    file_ids: str | list[str],
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
            "tool_name": "fetch_files",
            "tool_call_id": tool_call_id,
            "content": str(file_ids),
            "chunk_position": "start",
            "status": "success",
        }
    )

    event_writer.send_event(
        event=AgentStreamEvent.TOOL_EXEC_END,
        data={
            "event_name": "tool_exec_chunk_rtn",
            "tool_name": "fetch_files",
            "tool_call_id": tool_call_id,
            "content": "File service not configured in local mode.",
            "chunk_position": "end",
            "status": "fail",
        }
    )

    return Command(update={
        "messages": [
            ToolMessage(
                "File service not configured in local mode.",
                tool_call_id=tool_call_id,
            )
        ]
    })


# ------------------------------------------------------
# read_workspace_file
# ------------------------------------------------------

@tool(description=READ_WORKSPACE_FILE_PROMPT)
async def read_workspace_file(
    file_path: str,
    start_line: Optional[int] = 0,
    end_line: Optional[int] = 0,
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
            "tool_name": "read_workspace_file",
            "tool_call_id": tool_call_id,
            "content": file_path,
            "chunk_position": "start",
            "status": "success",
        }
    )

    # Normalize line args
    if start_line in ("None", "0", None, "", 0):
        start_line = None
    if end_line in ("None", "0", None, "", 0):
        end_line = None

    work_dir = _get_work_dir(state)
    err = _sandbox_check(work_dir, file_path)
    if err:
        event_writer.send_event(
            event=AgentStreamEvent.TOOL_EXEC_END,
            data={
                "event_name": "tool_exec_chunk_rtn",
                "tool_name": "read_workspace_file",
                "tool_call_id": tool_call_id,
                "content": err,
                "chunk_position": "end",
                "status": "fail",
            }
        )
        return Command(update={
            "messages": [ToolMessage(err, tool_call_id=tool_call_id)]
        })

    try:
        host_path = (Path(work_dir) / file_path).resolve()

        if not host_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not host_path.is_file():
            raise IsADirectoryError(f"Path is not a file: {file_path}")
        if host_path.stat().st_size > MAX_FILE_SIZE:
            raise Exception("File too large (>5MB)")

        lines = host_path.read_text(encoding="utf-8").splitlines(keepends=False)
        total = len(lines)

        s = 1 if not start_line else max(1, start_line)
        e = total if not end_line else min(end_line, total)

        selected = lines[s - 1:e]

        numbered = "\n".join(
            f"[{i}] {line}"
            for i, line in zip(range(s, e + 1), selected)
        )

        # Truncate if output is too long
        if len(numbered) > TOOLS_MAX_OUTPUT_LENGTH:
            numbered = numbered[:TOOLS_MAX_OUTPUT_LENGTH] + "\n...[output truncated]"

        event_writer.send_event(
            event=AgentStreamEvent.TOOL_EXEC_END,
            data={
                "event_name": "tool_exec_chunk_rtn",
                "tool_name": "read_workspace_file",
                "tool_call_id": tool_call_id,
                "content": f"Read lines {s}-{e}",
                "chunk_position": "end",
                "status": "success",
            }
        )

        return Command(update={
            "messages": [ToolMessage(numbered, tool_call_id=tool_call_id)]
        })

    except Exception as e:
        event_writer.send_event(
            event=AgentStreamEvent.TOOL_EXEC_END,
            data={
                "event_name": "tool_exec_chunk_rtn",
                "tool_name": "read_workspace_file",
                "tool_call_id": tool_call_id,
                "content": str(e),
                "chunk_position": "end",
                "status": "fail",
            }
        )
        return Command(update={
            "messages": [ToolMessage(str(e), tool_call_id=tool_call_id)]
        })


# ------------------------------------------------------
# write_workspace_file
# ------------------------------------------------------

@tool(description=WRITE_WORKSPACE_FILE_PROMPT)
async def write_workspace_file(
    file_path: str,
    content: str,
    exist_ok: Optional[bool] = False,
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
            "tool_name": "write_workspace_file",
            "tool_call_id": tool_call_id,
            "content": file_path,
            "chunk_position": "start",
            "status": "success",
        }
    )

    work_dir = _get_work_dir(state)
    err = _sandbox_check(work_dir, file_path)
    if err:
        event_writer.send_event(
            event=AgentStreamEvent.TOOL_EXEC_END,
            data={
                "event_name": "tool_exec_chunk_rtn",
                "tool_name": "write_workspace_file",
                "tool_call_id": tool_call_id,
                "content": err,
                "chunk_position": "end",
                "status": "fail",
            }
        )
        return Command(update={
            "messages": [ToolMessage(err, tool_call_id=tool_call_id)]
        })

    try:
        host_path = (Path(work_dir) / file_path).resolve()

        # Ensure parent directory exists
        host_path.parent.mkdir(parents=True, exist_ok=True)

        if host_path.exists():
            if exist_ok:
                host_path.write_text(content, encoding="utf-8")
                action = "overwritten"
            else:
                raise FileExistsError(f"File already exists: {file_path}")
        else:
            host_path.write_text(content, encoding="utf-8")
            action = "created"

        log_line = f"File {action}: {file_path}"

        event_writer.send_event(
            event=AgentStreamEvent.TOOL_EXEC_END,
            data={
                "event_name": "tool_exec_chunk_rtn",
                "tool_name": "write_workspace_file",
                "tool_call_id": tool_call_id,
                "content": log_line,
                "chunk_position": "end",
                "status": "success",
            }
        )

        return Command(update={
            "messages": [ToolMessage(log_line, tool_call_id=tool_call_id)]
        })

    except FileExistsError as e:
        event_writer.send_event(
            event=AgentStreamEvent.TOOL_EXEC_END,
            data={
                "event_name": "tool_exec_chunk_rtn",
                "tool_name": "write_workspace_file",
                "tool_call_id": tool_call_id,
                "content": str(e),
                "chunk_position": "end",
                "status": "fail",
            }
        )
        return Command(update={
            "messages": [ToolMessage(f"File already exists: {file_path}", tool_call_id=tool_call_id)]
        })

    except Exception as e:
        event_writer.send_event(
            event=AgentStreamEvent.TOOL_EXEC_END,
            data={
                "event_name": "tool_exec_chunk_rtn",
                "tool_name": "write_workspace_file",
                "tool_call_id": tool_call_id,
                "content": str(e),
                "chunk_position": "end",
                "status": "fail",
            }
        )
        return Command(update={
            "messages": [ToolMessage(str(e), tool_call_id=tool_call_id)]
        })


# ------------------------------------------------------
# delete_workspace_file
# ------------------------------------------------------

@tool(description=DELETE_WORKSPACE_FILE_PROMPT)
async def delete_workspace_file(
    file_path: str,
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
            "tool_name": "delete_workspace_file",
            "tool_call_id": tool_call_id,
            "content": file_path,
            "chunk_position": "start",
            "status": "success",
        }
    )

    work_dir = _get_work_dir(state)
    err = _sandbox_check(work_dir, file_path)
    if err:
        event_writer.send_event(
            event=AgentStreamEvent.TOOL_EXEC_END,
            data={
                "event_name": "tool_exec_chunk_rtn",
                "tool_name": "delete_workspace_file",
                "tool_call_id": tool_call_id,
                "content": err,
                "chunk_position": "end",
                "status": "fail",
            }
        )
        return Command(update={
            "messages": [ToolMessage(err, tool_call_id=tool_call_id)]
        })

    try:
        host_path = (Path(work_dir) / file_path).resolve()

        if not host_path.exists():
            raise FileNotFoundError(f"Path not found: {file_path}")

        # Refuse to delete workspace root itself
        if host_path.resolve() == Path(work_dir).resolve():
            raise Exception("Refusing to delete workspace root directory.")

        if host_path.is_file():
            host_path.unlink()
            msg = "File deleted"
        elif host_path.is_dir():
            shutil.rmtree(host_path)
            msg = "Directory deleted"
        else:
            raise Exception("Target path is neither file nor directory")

        event_writer.send_event(
            event=AgentStreamEvent.TOOL_EXEC_END,
            data={
                "event_name": "tool_exec_chunk_rtn",
                "tool_name": "delete_workspace_file",
                "tool_call_id": tool_call_id,
                "content": msg,
                "chunk_position": "end",
                "status": "success",
            }
        )

        return Command(update={
            "messages": [ToolMessage(msg, tool_call_id=tool_call_id)]
        })

    except Exception as e:
        event_writer.send_event(
            event=AgentStreamEvent.TOOL_EXEC_END,
            data={
                "event_name": "tool_exec_chunk_rtn",
                "tool_name": "delete_workspace_file",
                "tool_call_id": tool_call_id,
                "content": str(e),
                "chunk_position": "end",
                "status": "fail",
            }
        )
        return Command(update={
            "messages": [ToolMessage(str(e), tool_call_id=tool_call_id)]
        })


# ------------------------------------------------------
# move_workspace_file
# ------------------------------------------------------

@tool(description=MOVE_WORKSPACE_FILE_PROMPT)
async def move_workspace_file(
    source_path: str,
    target_path: str,
    delete_source: Optional[bool] = True,
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
            "tool_name": "move_workspace_file",
            "tool_call_id": tool_call_id,
            "content": f"{source_path} -> {target_path}",
            "chunk_position": "start",
            "status": "success",
        }
    )

    work_dir = _get_work_dir(state)

    # Sandbox check both paths
    for p in (source_path, target_path):
        err = _sandbox_check(work_dir, p)
        if err:
            event_writer.send_event(
                event=AgentStreamEvent.TOOL_EXEC_END,
                data={
                    "event_name": "tool_exec_chunk_rtn",
                    "tool_name": "move_workspace_file",
                    "tool_call_id": tool_call_id,
                    "content": err,
                    "chunk_position": "end",
                    "status": "fail",
                }
            )
            return Command(update={
                "messages": [ToolMessage(err, tool_call_id=tool_call_id)]
            })

    try:
        source_host = (Path(work_dir) / source_path).resolve()
        dest_host = (Path(work_dir) / target_path).resolve()

        if not source_host.exists():
            raise FileNotFoundError(f"Source not found: {source_path}")

        if dest_host.exists():
            raise Exception("Target already exists")

        dest_host.parent.mkdir(parents=True, exist_ok=True)

        if delete_source:
            shutil.move(str(source_host), str(dest_host))
            msg = f"File moved to {target_path}"
        else:
            shutil.copy2(str(source_host), str(dest_host))
            msg = f"File copied to {target_path}"

        event_writer.send_event(
            event=AgentStreamEvent.TOOL_EXEC_END,
            data={
                "event_name": "tool_exec_chunk_rtn",
                "tool_name": "move_workspace_file",
                "tool_call_id": tool_call_id,
                "content": msg,
                "chunk_position": "end",
                "status": "success",
            }
        )

        return Command(update={
            "messages": [ToolMessage(msg, tool_call_id=tool_call_id)]
        })

    except Exception as e:
        event_writer.send_event(
            event=AgentStreamEvent.TOOL_EXEC_END,
            data={
                "event_name": "tool_exec_chunk_rtn",
                "tool_name": "move_workspace_file",
                "tool_call_id": tool_call_id,
                "content": str(e),
                "chunk_position": "end",
                "status": "fail",
            }
        )
        return Command(update={
            "messages": [ToolMessage(str(e), tool_call_id=tool_call_id)]
        })


# ------------------------------------------------------
# list_workspace_files
# ------------------------------------------------------

@tool(description=LIST_WORKSPACE_FILES_PROMPT)
async def list_workspace_files(
    path: Optional[str] = None,
    recursively_scan: Optional[bool] = False,
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
            "tool_name": "list_workspace_files",
            "tool_call_id": tool_call_id,
            "content": path or "/workspace",
            "chunk_position": "start",
            "status": "success",
        }
    )

    if path == "None":
        path = None
    if recursively_scan == "None":
        recursively_scan = False

    work_dir = _get_work_dir(state)

    # If a sub-path is given, sandbox check it
    if path:
        err = _sandbox_check(work_dir, path)
        if err:
            event_writer.send_event(
                event=AgentStreamEvent.TOOL_EXEC_END,
                data={
                    "event_name": "tool_exec_chunk_rtn",
                    "tool_name": "list_workspace_files",
                    "tool_call_id": tool_call_id,
                    "content": err,
                    "chunk_position": "end",
                    "status": "fail",
                }
            )
            return Command(update={
                "messages": [ToolMessage(err, tool_call_id=tool_call_id)]
            })

    try:
        fs_target = (Path(work_dir) / path).resolve() if path else Path(work_dir).resolve()

        if not fs_target.exists():
            raise FileNotFoundError("Directory not found")
        if not fs_target.is_dir():
            raise NotADirectoryError("Target is not a directory")

        lines: list[str] = []
        count = 0

        def scan_dir(current: Path, depth: int):
            nonlocal count

            with os.scandir(current) as entries:
                dirs: list[str] = []
                files: list[str] = []

                for entry in entries:
                    name = entry.name
                    if entry.is_dir(follow_symlinks=False):
                        if name.startswith(".") or name in IGNORE_DIRS:
                            continue
                        dirs.append(name)
                    else:
                        files.append(name)

                dirs.sort()
                files.sort()

                indent = "  " * depth

                for d in dirs:
                    lines.append(f"{indent}{d}/")
                    count += 1
                    if count > MAX_FILES:
                        raise Exception("Too many files (limit 500)")
                    if recursively_scan and depth < MAX_DEPTH:
                        scan_dir(Path(current) / d, depth + 1)

                for f in files:
                    lines.append(f"{indent}{f}")
                    count += 1
                    if count > MAX_FILES:
                        raise Exception("Too many files (limit 500)")

        scan_dir(fs_target, 0)
        result = "\n".join(lines) if lines else "(empty directory)"

        event_writer.send_event(
            event=AgentStreamEvent.TOOL_EXEC_END,
            data={
                "event_name": "tool_exec_chunk_rtn",
                "tool_name": "list_workspace_files",
                "tool_call_id": tool_call_id,
                "content": f"{count} items",
                "chunk_position": "end",
                "status": "success",
            }
        )

        return Command(update={
            "messages": [ToolMessage(result, tool_call_id=tool_call_id)]
        })

    except Exception as e:
        event_writer.send_event(
            event=AgentStreamEvent.TOOL_EXEC_END,
            data={
                "event_name": "tool_exec_chunk_rtn",
                "tool_name": "list_workspace_files",
                "tool_call_id": tool_call_id,
                "content": str(e),
                "chunk_position": "end",
                "status": "fail",
            }
        )
        return Command(update={
            "messages": [ToolMessage(str(e), tool_call_id=tool_call_id)]
        })
