"""
Backward-compatibility shim for runtime.py.

The real LangGraph @tool implementations are in:
  - file_ops.py, code_runner.py, todo.py, communication.py

This module only provides set_workspace_root / get_workspace_root
so that the legacy CodingRuntime continues to import cleanly.
"""
from __future__ import annotations

# Global workspace root, set by the legacy runtime
_workspace_root: str = ''


def set_workspace_root(root: str):
    global _workspace_root
    _workspace_root = root


def get_workspace_root() -> str:
    return _workspace_root
