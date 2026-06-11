from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """LLM model settings."""

    provider: str = "deepseek"
    api_base: str = "https://api.deepseek.com/v1"
    api_key_env: str = "DEEPSEEK_API_KEY"
    model: str = "deepseek-chat"
    temperature: float = 0.0


class AgentConfig(BaseModel):
    """Agent behavior settings."""

    max_turns: int = Field(default=8, ge=1, le=100)


class PlatformConfig(BaseModel):
    """Platform-level controls shared by CLI/TUI/runtime."""

    default_agent: str = "default"
    agents_dir: str = "agents"
    workflow_mode: Literal["single", "pipeline"] = "single"
    default_pipeline: str = "default-coding-pipeline"
    intent_recognition_enabled: bool = False
    tool_execution_mode: Literal["controlled", "full_auto", "read_only"] = "controlled"
    allow_unsafe_auto_exec: bool = False


class WorkflowConfig(BaseModel):
    """Workflow path settings."""

    target: str = "example-notes.md"
    prompts_dir: str = "prompts"
    workspace_root: str = "."


class TuiConfig(BaseModel):
    """TUI settings."""

    mode: Literal["chat", "agent"] = "agent"
    system_prompt: str = "You are a helpful assistant."


class AppConfig(BaseModel):
    """Top-level application configuration."""

    platform: PlatformConfig = Field(default_factory=PlatformConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    tui: TuiConfig = Field(default_factory=TuiConfig)


class AgentSpec(BaseModel):
    """Single agent specification loaded from agents/<name>/agent.yaml."""

    name: str
    description: str = ""
    prompt_files: list[str] = Field(default_factory=lambda: ["prompts/system.md"])
    enabled_tools: list[str] = Field(default_factory=list)
    model_overrides: dict[str, Any] = Field(default_factory=dict)


class PipelineStageSpec(BaseModel):
    """Single stage in a pipeline definition."""

    id: str
    agent: str
    task_template: str


class PipelineSpec(BaseModel):
    """Pipeline definition loaded from pipelines.yaml."""

    name: str
    description: str = ""
    stages: list[PipelineStageSpec] = Field(default_factory=list)


class PipelinesConfig(BaseModel):
    """Top-level pipeline config file schema."""

    pipelines: list[PipelineSpec] = Field(default_factory=list)


ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_config(path: str) -> AppConfig:
    """Load application config from YAML."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    raw = config_path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    return AppConfig.model_validate(data)


def _default_agent_spec(name: str) -> AgentSpec:
    """Fallback spec used when the agents directory is absent."""
    return AgentSpec(
        name=name,
        description="Default agent profile.",
        prompt_files=["prompts/agent.md"],
    )


def load_agent_specs(workspace_root: str, agents_dir: str) -> dict[str, AgentSpec]:
    """Load agent definitions from agents/<agent_name>/agent.yaml."""
    root = Path(workspace_root).resolve()
    base = (root / agents_dir).resolve()
    if not base.is_dir():
        return {}

    specs: dict[str, AgentSpec] = {}
    for entry in sorted(base.iterdir(), key=lambda p: p.name.lower()):
        if not entry.is_dir():
            continue
        config_path = entry / "agent.yaml"
        if not config_path.is_file():
            continue
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid agent config: {config_path}")
        raw.setdefault("name", entry.name)
        spec = AgentSpec.model_validate(raw)
        specs[spec.name] = spec
    return specs


def resolve_agent_spec(app_config: AppConfig, workspace_root: str, agent_name: str | None = None) -> AgentSpec:
    """Resolve active agent spec with compatibility fallback."""
    selected = (agent_name or app_config.platform.default_agent).strip()
    if not selected:
        selected = "default"
    specs = load_agent_specs(workspace_root, app_config.platform.agents_dir)
    if not specs:
        return _default_agent_spec(selected)
    if selected in specs:
        return specs[selected]
    available = ", ".join(sorted(specs))
    raise ValueError(f"Agent '{selected}' not found. Available agents: {available}")


def list_available_agents(app_config: AppConfig, workspace_root: str) -> list[str]:
    """List all configured agents in sorted order."""
    specs = load_agent_specs(workspace_root, app_config.platform.agents_dir)
    if not specs:
        return [app_config.platform.default_agent]
    return sorted(specs.keys())


def load_pipelines(workspace_root: str, file_name: str = "pipelines.yaml") -> dict[str, PipelineSpec]:
    """Load pipeline definitions from workspace root."""
    root = Path(workspace_root).resolve()
    path = (root / file_name).resolve()
    if not path.is_file():
        return {}
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Pipeline config escapes workspace root: {path}") from exc

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parsed = PipelinesConfig.model_validate(raw)
    specs: dict[str, PipelineSpec] = {}
    for pipeline in parsed.pipelines:
        if not pipeline.stages:
            raise ValueError(f"Pipeline '{pipeline.name}' has no stages.")
        specs[pipeline.name] = pipeline
    return specs


def resolve_pipeline_spec(
    app_config: AppConfig,
    workspace_root: str,
    pipeline_name: str | None = None,
) -> PipelineSpec:
    """Resolve active pipeline spec and validate stage agents."""
    selected = (pipeline_name or app_config.platform.default_pipeline).strip()
    if not selected:
        selected = app_config.platform.default_pipeline

    specs = load_pipelines(workspace_root)
    if not specs:
        raise ValueError("No pipelines found. Create pipelines.yaml in workspace root.")
    if selected not in specs:
        available = ", ".join(sorted(specs))
        raise ValueError(f"Pipeline '{selected}' not found. Available pipelines: {available}")

    pipeline = specs[selected]
    for stage in pipeline.stages:
        if not stage.id.strip():
            raise ValueError(f"Pipeline '{pipeline.name}' has a stage with empty id.")
        if not stage.task_template.strip():
            raise ValueError(f"Pipeline '{pipeline.name}' stage '{stage.id}' has empty task_template.")
        resolve_agent_spec(app_config, workspace_root, agent_name=stage.agent)
    return pipeline


def list_available_pipelines(workspace_root: str) -> list[str]:
    """List all configured pipelines in sorted order."""
    return sorted(load_pipelines(workspace_root).keys())


def resolve_workspace_paths(config: WorkflowConfig) -> tuple[str, str, str]:
    """Resolve workspace paths to absolute paths."""
    workspace_root = str(Path(config.workspace_root).resolve())
    prompts_dir = str((Path(workspace_root) / config.prompts_dir).resolve())
    target_path = str(Path(config.target))
    return workspace_root, prompts_dir, target_path


def resolve_api_key(model_config: ModelConfig) -> str | None:
    """Resolve API key from environment variable or literal value."""
    key_ref = model_config.api_key_env.strip()
    if not key_ref:
        return None
    if ENV_NAME_PATTERN.fullmatch(key_ref):
        return os.getenv(key_ref)
    return key_ref
