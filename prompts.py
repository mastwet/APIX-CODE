from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import AgentSpec


@dataclass(frozen=True)
class AgentPrompts:
    """Resolved prompts used by the runtime."""

    agent: str


def load_prompts(workspace_root: str, agent_spec: AgentSpec) -> AgentPrompts:
    """Load prompt files from an agent profile and join them."""
    root = Path(workspace_root).resolve()
    chunks: list[str] = []

    for rel_path in agent_spec.prompt_files:
        prompt_path = (root / rel_path).resolve()
        try:
            prompt_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Prompt path escapes workspace root: {rel_path}") from exc
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        content = prompt_path.read_text(encoding="utf-8").strip()
        if content:
            chunks.append(content)

    if not chunks:
        raise ValueError(f"Agent '{agent_spec.name}' has no usable prompt content")

    return AgentPrompts(agent="\n\n".join(chunks))
