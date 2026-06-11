import asyncio
import tempfile
from pathlib import Path
from typing import Annotated, Optional
from uuid import uuid4

from langchain.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.messages import ToolMessage

from ..event.stream_writer import AgentStreamWriter, AgentStreamEvent
from .tool_descriptions import RUN_WORKSPACE_COMMAND_PROMPT, RUN_PYTHON_CODE_PROMPT

# Maximum output length before truncation
TOOLS_MAX_OUTPUT_LENGTH = 30000
# Command execution timeout (seconds)
COMMAND_TIMEOUT = 600.0


# ------------------------------------------------------
# run_workspace_command (local, no Docker)
# ------------------------------------------------------

@tool(description=RUN_WORKSPACE_COMMAND_PROMPT)
async def run_workspace_command(
    command: str,
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
            "tool_name": "run_workspace_command",
            "tool_call_id": tool_call_id,
            "content": command,
            "chunk_position": "start",
            "status": "success",
        }
    )

    if not command.strip():
        event_writer.send_event(
            event=AgentStreamEvent.TOOL_EXEC_END,
            data={
                "event_name": "tool_exec_chunk_rtn",
                "tool_name": "run_workspace_command",
                "tool_call_id": tool_call_id,
                "content": "Error: command is empty.",
                "chunk_position": "end",
                "status": "fail",
            }
        )
        return Command(update={
            "messages": [ToolMessage("Error: command cannot be empty.", tool_call_id=tool_call_id)]
        })

    work_dir = (state.get("config") or {}).get("work_dir", ".")

    try:
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=work_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=COMMAND_TIMEOUT,
        )

        output = stdout.decode(errors="replace") + stderr.decode(errors="replace")

        # Truncate if output exceeds limit
        if len(output) > TOOLS_MAX_OUTPUT_LENGTH:
            half = TOOLS_MAX_OUTPUT_LENGTH // 2
            output = output[:half] + "\n\n...[output truncated]...\n\n" + output[-half:]

        status = "success" if process.returncode == 0 else "fail"
        summary = (
            f"{len(output.strip())} characters output."
            if output.strip()
            else "Command executed successfully (no output)."
        )

        event_writer.send_event(
            event=AgentStreamEvent.TOOL_EXEC_END,
            data={
                "event_name": "tool_exec_chunk_rtn",
                "tool_name": "run_workspace_command",
                "tool_call_id": tool_call_id,
                "content": summary,
                "chunk_position": "end",
                "status": status,
            }
        )

        return Command(update={
            "messages": [
                ToolMessage(
                    output.strip() or "Command executed successfully (no output).",
                    tool_call_id=tool_call_id,
                )
            ]
        })

    except asyncio.TimeoutError:
        try:
            process.kill()
            await process.wait()
        except Exception:
            pass

        event_writer.send_event(
            event=AgentStreamEvent.TOOL_EXEC_END,
            data={
                "event_name": "tool_exec_chunk_rtn",
                "tool_name": "run_workspace_command",
                "tool_call_id": tool_call_id,
                "content": "Error: Command execution timed out after 600 seconds.",
                "chunk_position": "end",
                "status": "fail",
            }
        )

        return Command(update={
            "messages": [
                ToolMessage(
                    "Error: Command execution timed out after 600 seconds.",
                    tool_call_id=tool_call_id,
                )
            ]
        })

    except Exception as e:
        event_writer.send_event(
            event=AgentStreamEvent.TOOL_EXEC_END,
            data={
                "event_name": "tool_exec_chunk_rtn",
                "tool_name": "run_workspace_command",
                "tool_call_id": tool_call_id,
                "content": f"Error: {str(e)}",
                "chunk_position": "end",
                "status": "fail",
            }
        )

        return Command(update={
            "messages": [ToolMessage(f"Error: {str(e)}", tool_call_id=tool_call_id)]
        })


# ------------------------------------------------------
# run_python_code (local, no Docker)
# ------------------------------------------------------

@tool(description=RUN_PYTHON_CODE_PROMPT)
async def run_python_code(
    code: str,
    run_args: Optional[list[str]] = None,
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
            "tool_name": "run_python_code",
            "tool_call_id": tool_call_id,
            "content": (
                "Running Python code\n\n"
                "```python\n"
                f"{code}\n"
                "```\n"
                f"With args: {run_args}"
            ),
            "chunk_position": "start",
            "status": "success",
        }
    )

    run_args = run_args or []
    work_dir = (state.get("config") or {}).get("work_dir", ".")

    if not code or not code.strip():
        event_writer.send_event(
            event=AgentStreamEvent.TOOL_EXEC_END,
            data={
                "event_name": "tool_exec_chunk_rtn",
                "tool_name": "run_python_code",
                "tool_call_id": tool_call_id,
                "content": "Error: Python code cannot be empty.",
                "chunk_position": "end",
                "status": "fail",
            }
        )
        return Command(update={
            "messages": [ToolMessage("Error: Python code cannot be empty.", tool_call_id=tool_call_id)]
        })

    # Create temp directory under work_dir for the script
    tmp_dir = Path(work_dir) / ".tmp_exec"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    script_path = tmp_dir / f"{uuid4().hex}.py"

    try:
        # Write code to temp file
        script_path.write_text(code, encoding="utf-8")

        # Build command
        cmd_args = ["python", str(script_path)] + run_args

        process = await asyncio.create_subprocess_exec(
            *cmd_args,
            cwd=work_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=COMMAND_TIMEOUT,
        )

        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")

        output_parts = []
        if stdout_text.strip():
            output_parts.append(stdout_text.rstrip())
        if stderr_text.strip():
            output_parts.append(stderr_text.rstrip())

        output = "\n".join(output_parts).strip()
        if not output:
            output = "Python code executed successfully (no output)."

        # Truncate if needed
        if len(output) > TOOLS_MAX_OUTPUT_LENGTH:
            output = output[:TOOLS_MAX_OUTPUT_LENGTH] + "\n...[output truncated]"

        status = "success" if process.returncode == 0 else "fail"

        event_writer.send_event(
            event=AgentStreamEvent.TOOL_EXEC_END,
            data={
                "event_name": "tool_exec_chunk_rtn",
                "tool_name": "run_python_code",
                "tool_call_id": tool_call_id,
                "content": (
                    "Result:\n"
                    "```\n"
                    "[STDOUT]\n"
                    f"{stdout_text}\n\n"
                    "[STDERR]\n"
                    f"{stderr_text}\n"
                    "```"
                ),
                "chunk_position": "end",
                "status": status,
            }
        )

        if process.returncode != 0:
            return Command(update={
                "messages": [
                    ToolMessage(
                        f"Python exited with code {process.returncode}.\n{output}",
                        tool_call_id=tool_call_id,
                    )
                ]
            })

        return Command(update={
            "messages": [ToolMessage(output, tool_call_id=tool_call_id)]
        })

    except asyncio.TimeoutError:
        try:
            process.kill()
            await process.wait()
        except Exception:
            pass

        event_writer.send_event(
            event=AgentStreamEvent.TOOL_EXEC_END,
            data={
                "event_name": "tool_exec_chunk_rtn",
                "tool_name": "run_python_code",
                "tool_call_id": tool_call_id,
                "content": "Error: Python execution timed out after 600 seconds.",
                "chunk_position": "end",
                "status": "fail",
            }
        )

        return Command(update={
            "messages": [
                ToolMessage(
                    "Error: Python execution timed out after 600 seconds.",
                    tool_call_id=tool_call_id,
                )
            ]
        })

    except Exception as e:
        event_writer.send_event(
            event=AgentStreamEvent.TOOL_EXEC_END,
            data={
                "event_name": "tool_exec_chunk_rtn",
                "tool_name": "run_python_code",
                "tool_call_id": tool_call_id,
                "content": f"Error executing Python code: {str(e)}",
                "chunk_position": "end",
                "status": "fail",
            }
        )
        return Command(update={
            "messages": [ToolMessage(f"Error executing Python code: {str(e)}", tool_call_id=tool_call_id)]
        })

    finally:
        # Clean up temp script
        try:
            if script_path.exists():
                script_path.unlink()
        except Exception:
            pass
