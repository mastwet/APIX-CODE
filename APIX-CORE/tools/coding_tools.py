"""
Backward-compatibility shim for the legacy CodingRuntime.

The real implementations have moved to:
  - tools/file_ops.py     (file tools)
  - tools/code_runner.py  (command/python execution)
  - tools/todo.py         (todo/memory tools)
  - tools/communication.py (user input)

This module preserves the old (workspace_root, name, args) -> str API
so that runtime.py continues to work without changes.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

# Maximum bytes for tool output
MAX_BYTES = 30 * 1024
READ_MAX_LINES = 250
GREP_MAX_RESULTS = 100
FIND_MAX_RESULTS = 1000
LS_MAX_RESULTS = 500

ToolUpdateFn = Callable[[str], None]


class ToolError(RuntimeError):
    """Tool execution error."""
    pass


def _resolve_path(workspace_root: str, file_path: str) -> Path:
    """Resolve a path and ensure it stays within the workspace."""
    ws = Path(workspace_root).resolve()
    candidate = (ws / file_path).resolve()
    if not str(candidate).startswith(str(ws)):
        raise ToolError(f"Path escapes workspace: {file_path}")
    return candidate


def _truncate_text(text: str, max_bytes: int = MAX_BYTES) -> tuple[str, bool]:
    """Truncate text to a maximum byte length."""
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text, False
    truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return truncated, True


def _skip_dir(name: str) -> bool:
    """Check if a directory should be skipped during traversal."""
    return name in {".git", "node_modules", "__pycache__", ".venv", "venv"}


def read_tool(workspace_root: str, args: dict[str, Any]) -> str:
    """Read file contents tool."""
    file_path = args.get("path", "")
    offset = int(args.get("offset", 1))
    limit = int(args.get("limit", READ_MAX_LINES))

    path = _resolve_path(workspace_root, file_path)
    if not path.exists():
        raise ToolError(f"File not found: {file_path}")
    if not path.is_file():
        raise ToolError(f"Not a file: {file_path}")
    if path.stat().st_size > 5 * 1024 * 1024:
        raise ToolError("File too large (>5MB)")

    lines = path.read_text(encoding="utf-8").splitlines(keepends=False)
    total = len(lines)
    s = max(1, offset)
    e = min(total, s + limit - 1)
    selected = lines[s - 1:e]

    body = "\n".join(f"[{i}] {line}" for i, line in zip(range(s, e + 1), selected))
    body, truncated = _truncate_text(body)
    if truncated:
        body += "\n[truncated]"
    return body


def write_tool(workspace_root: str, args: dict[str, Any]) -> str:
    """Write file contents tool."""
    file_path = args.get("path", "")
    content = args.get("content", "")
    path = _resolve_path(workspace_root, file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"Successfully wrote {len(content.encode('utf-8'))} bytes to {path}"


def edit_tool(workspace_root: str, args: dict[str, Any]) -> str:
    """Edit file tool (surgical text replacement)."""
    file_path = args.get("path", "")
    old_text = args.get("oldText", "")
    new_text = args.get("newText", "")
    path = _resolve_path(workspace_root, file_path)
    if not path.exists():
        raise ToolError(f"File not found: {file_path}")
    content = path.read_text(encoding="utf-8")
    if old_text not in content:
        raise ToolError("old_text not found in file")
    new_content = content.replace(old_text, new_text, 1)
    path.write_text(new_content, encoding="utf-8")
    diff = f"- {old_text}\n+ {new_text}"
    return f"Successfully replaced text in {path}\n\n{diff}" if diff else f"Successfully replaced text in {path}"


def ls_tool(workspace_root: str, args: dict[str, Any]) -> str:
    """List directory contents tool."""
    file_path = args.get("path", ".")
    max_depth = int(args.get("max_depth", 1))
    path = _resolve_path(workspace_root, file_path)
    if not path.exists():
        raise ToolError(f"Directory not found: {file_path}")
    if not path.is_dir():
        raise ToolError(f"Not a directory: {file_path}")

    lines: list[str] = []
    count = 0

    def scan(current: Path, depth: int):
        nonlocal count
        if depth > max_depth:
            return
        try:
            entries = sorted(os.scandir(current), key=lambda e: (not e.is_dir(), e.name))
        except PermissionError:
            return
        for entry in entries:
            if _skip_dir(entry.name):
                continue
            indent = "  " * depth
            if entry.is_dir():
                lines.append(f"{indent}{entry.name}/")
            else:
                lines.append(f"{indent}{entry.name}")
            count += 1
            if count >= LS_MAX_RESULTS:
                lines.append(f"  ... (truncated at {LS_MAX_RESULTS})")
                return
            if entry.is_dir(follow_symlinks=False) and depth < max_depth:
                scan(Path(entry.path), depth + 1)

    scan(path, 0)
    output = "\n".join(lines) if lines else "(empty directory)"
    output, _ = _truncate_text(output)
    return output


def find_tool(workspace_root: str, args: dict[str, Any]) -> str:
    """Find files by glob pattern tool."""
    import glob
    pattern = args.get("pattern", "**/*")
    file_path = args.get("path", ".")
    path = _resolve_path(workspace_root, file_path)
    if not path.exists():
        raise ToolError(f"Directory not found: {file_path}")

    matches = list(path.glob(pattern))
    if len(matches) > FIND_MAX_RESULTS:
        matches = matches[:FIND_MAX_RESULTS]

    lines = [str(m.relative_to(path)) for m in matches]
    output = "\n".join(lines) if lines else "(no matches found)"
    output, _ = _truncate_text(output)
    return output


def grep_tool(workspace_root: str, args: dict[str, Any]) -> str:
    """Search for text patterns in files tool."""
    import re
    pattern = args.get("pattern", "")
    file_path = args.get("path", ".")
    include = args.get("include", "")

    path = _resolve_path(workspace_root, file_path)
    if not path.exists():
        raise ToolError(f"Path not found: {file_path}")

    regex = re.compile(pattern)
    results: list[str] = []
    count = 0

    search_files = []
    if path.is_file():
        search_files.append(path)
    else:
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if not _skip_dir(d)]
            for f in files:
                if include and not f.endswith(include):
                    continue
                fp = Path(root) / f
                if fp.stat().st_size > 1024 * 1024:
                    continue
                search_files.append(fp)
                if len(search_files) >= GREP_MAX_RESULTS * 10:
                    break

    for fp in search_files:
        try:
            lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
            for i, line in enumerate(lines, 1):
                if regex.search(line):
                    results.append(f"{fp}:{i}: {line}")
                    count += 1
                    if count >= GREP_MAX_RESULTS:
                        results.append(f"... (truncated at {GREP_MAX_RESULTS} matches)")
                        break
        except (PermissionError, OSError):
            continue
        if count >= GREP_MAX_RESULTS:
            break

    text = "\n".join(results) if results else "(no matches found)"
    text, _ = _truncate_text(text)
    return text


def bash_tool(workspace_root: str, args: dict[str, Any], on_update: ToolUpdateFn | None = None) -> str:
    """Execute shell command tool."""
    command = args.get("command", "")
    timeout = int(args.get("timeout", 30))
    if not command.strip():
        raise ToolError("Command cannot be empty")

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=workspace_root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout + result.stderr
        if result.returncode != 0:
            output = f"Exit code: {result.returncode}\n{output}"
    except subprocess.TimeoutExpired:
        raise ToolError(f"Command timed out after {timeout}s")
    except Exception as e:
        raise ToolError(str(e))

    output, truncated = _truncate_text(output)
    if truncated:
        output += "\n[truncated]"
    return output.strip() or "(no output)"


def get_tool_definitions() -> list[dict[str, Any]]:
    """Get tool definitions list (legacy format)."""
    return [
        {"name": "read", "description": "Read file contents", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer", "default": 1}, "limit": {"type": "integer", "default": 250}}, "required": ["path"]}},
        {"name": "write", "description": "Write file contents", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
        {"name": "edit", "description": "Edit file (surgical replacement)", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "oldText": {"type": "string"}, "newText": {"type": "string"}}, "required": ["path", "oldText", "newText"]}},
        {"name": "bash", "description": "Execute shell command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer", "default": 30}}, "required": ["command"]}},
        {"name": "ls", "description": "List directory contents", "parameters": {"type": "object", "properties": {"path": {"type": "string", "default": "."}, "max_depth": {"type": "integer", "default": 1}}}},
        {"name": "find", "description": "Find files by glob pattern", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string", "default": "."}}}},
        {"name": "grep", "description": "Search for text pattern in files", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string", "default": "."}, "include": {"type": "string", "default": ""}}}},
    ]


def execute_tool(
    workspace_root: str,
    name: str,
    args: dict[str, Any],
    on_update: ToolUpdateFn | None = None,
) -> str:
    """Execute a named tool."""
    dispatch = {
        "read": read_tool,
        "write": write_tool,
        "edit": edit_tool,
        "ls": ls_tool,
        "find": find_tool,
        "grep": grep_tool,
    }

    fn = dispatch.get(name)
    if fn is None:
        if name == "bash":
            return bash_tool(workspace_root, args, on_update)
        raise ToolError(f"Unsupported tool: {name}")
    return fn(workspace_root, args)


# Needed for subprocess in bash_tool
import subprocess
