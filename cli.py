from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from config import (
    list_available_agents,
    list_available_pipelines,
    load_config,
    resolve_agent_spec,
    resolve_pipeline_spec,
    resolve_workspace_paths,
)
from pipeline_runtime import PipelineRuntime
from runtime import create_runtime


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the APIX agent workflow.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--request", required=False, help="User request for the agent.")
    parser.add_argument("--target", default=None, help="Override workflow.target.")
    parser.add_argument("--prompts-dir", default=None, help="Override workflow.prompts_dir.")
    parser.add_argument("--workspace-root", default=None, help="Override workflow.workspace_root.")
    parser.add_argument("--model", default=None, help="Override model.model.")
    parser.add_argument("--max-turns", type=int, default=None, help="Override agent.max_turns.")
    parser.add_argument("--agent", default=None, help="Select agent profile from agents directory.")
    parser.add_argument("--pipeline", default=None, help="Select pipeline from pipelines.yaml.")
    parser.add_argument("--list-agents", action="store_true", help="List available agent profiles and exit.")
    parser.add_argument("--list-pipelines", action="store_true", help="List available pipelines and exit.")
    parser.add_argument(
        "--workflow-mode",
        choices=["single", "pipeline"],
        default=None,
        help="Override platform.workflow_mode.",
    )
    parser.add_argument(
        "--intent-recognition",
        dest="intent_recognition",
        action="store_true",
        help="Enable intent recognition before tool execution.",
    )
    parser.add_argument(
        "--no-intent-recognition",
        dest="intent_recognition",
        action="store_false",
        help="Disable intent recognition before tool execution.",
    )
    parser.set_defaults(intent_recognition=None)
    parser.add_argument(
        "--unsafe-auto-exec",
        action="store_true",
        help="Allow write/edit/bash tools to auto-execute.",
    )
    return parser.parse_args()


def _load_env(config_path: str) -> None:
    config_file = Path(config_path).resolve()
    dotenv_path = config_file.with_name(".env")
    load_dotenv(dotenv_path=dotenv_path)


def main() -> None:
    args = _parse_args()
    _load_env(args.config)
    config = load_config(args.config)

    workflow_updates: dict[str, object] = {}
    if args.target is not None:
        workflow_updates["target"] = args.target
    if args.prompts_dir is not None:
        workflow_updates["prompts_dir"] = args.prompts_dir
    if args.workspace_root is not None:
        workflow_updates["workspace_root"] = args.workspace_root

    agent_updates: dict[str, object] = {}
    if args.max_turns is not None:
        agent_updates["max_turns"] = args.max_turns

    platform_updates: dict[str, object] = {}
    if args.intent_recognition is not None:
        platform_updates["intent_recognition_enabled"] = args.intent_recognition
    if args.unsafe_auto_exec:
        platform_updates["allow_unsafe_auto_exec"] = True
    if args.workflow_mode is not None:
        platform_updates["workflow_mode"] = args.workflow_mode

    workflow_config = config.workflow.model_copy(update=workflow_updates)
    agent_config = config.agent.model_copy(update=agent_updates)
    platform_config = config.platform.model_copy(update=platform_updates)
    app_config = config.model_copy(update={"platform": platform_config})

    workspace_root, _, target_path = resolve_workspace_paths(workflow_config)

    if args.list_agents:
        for name in list_available_agents(app_config, workspace_root):
            print(name)
        return

    if args.list_pipelines:
        for name in list_available_pipelines(workspace_root):
            print(name)
        return

    if not args.request:
        raise ValueError("--request is required unless --list-agents or --list-pipelines is used.")

    base_model_config = config.model
    if args.model is not None:
        base_model_config = base_model_config.model_copy(update={"model": args.model})

    workflow_mode = platform_config.workflow_mode
    if workflow_mode == "pipeline":
        pipeline_spec = resolve_pipeline_spec(app_config, workspace_root, pipeline_name=args.pipeline)
        runner = PipelineRuntime(
            app_config=app_config,
            model_config=base_model_config,
            agent_config=agent_config,
            pipeline_spec=pipeline_spec,
            workspace_root=workspace_root,
        )
        event_stream = runner.run(args.request, target_path=target_path, max_turns=agent_config.max_turns)
    else:
        agent_spec = resolve_agent_spec(app_config, workspace_root, agent_name=args.agent)
        model_config = base_model_config.model_copy(update=agent_spec.model_overrides)
        runtime = create_runtime(
            model_config,
            agent_config,
            agent_spec,
            workspace_root,
            intent_recognition_enabled=platform_config.intent_recognition_enabled,
            tool_execution_mode=platform_config.tool_execution_mode,
            allow_unsafe_auto_exec=platform_config.allow_unsafe_auto_exec,
        )
        event_stream = runtime.run(args.request, target_path=target_path, max_turns=agent_config.max_turns)

    for event in event_stream:
        event_type = event["type"]
        if event_type == "pipeline_start":
            print(
                f"[pipeline:start] {event.get('pipeline_name')} "
                f"({event.get('stage_count', 0)} stages)"
            )
        elif event_type == "pipeline_stage_start":
            print(
                f"\n[pipeline:stage:start] #{event.get('stage_index')} "
                f"{event.get('stage_id')} ({event.get('stage_agent')})"
            )
        elif event_type == "pipeline_stage_end":
            print(
                f"[pipeline:stage:end] #{event.get('stage_index')} "
                f"{event.get('stage_id')} status={event.get('status')}"
            )
            stage_response = event.get("final_response", "")
            if stage_response:
                print(stage_response)
        elif event_type == "pipeline_end":
            print(f"\n[pipeline:{event.get('status')}] {event.get('pipeline_name')}")
        elif event_type == "intent_recognized":
            intent = event.get("intent", {})
            print(f"[intent] type={intent.get('intent_type')} target={intent.get('target_path')}")
        elif event_type == "intent_skipped":
            print("[intent] skipped (disabled)")
        elif event_type == "message_delta":
            delta = event.get("delta", "")
            if delta:
                print(delta, end="", flush=True)
        elif event_type == "message_end":
            print()
        elif event_type == "tool_execution_start":
            print(f"\n[tool:start] {event.get('tool_name')} args={event.get('args')}")
        elif event_type == "tool_execution_update":
            delta = event.get("delta", "")
            if delta:
                print(delta, end="", flush=True)
        elif event_type == "tool_execution_end":
            marker = "error" if event.get("is_error") else "ok"
            print(f"\n[tool:end:{marker}] {event.get('tool_name')}")
            result = event.get("result", "")
            if result:
                print(result)
        elif event_type == "tool_blocked_by_policy":
            print(f"\n[tool:blocked] {event.get('tool_name')}")
            result = event.get("result", "")
            if result:
                print(result)
        elif event_type == "agent_end":
            print(f"\n[agent:{event.get('status')}] {event.get('final_response', '')}")


if __name__ == "__main__":
    main()
