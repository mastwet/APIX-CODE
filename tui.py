from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Footer, Input, RichLog, Static

from config import load_config, resolve_agent_spec, resolve_pipeline_spec, resolve_workspace_paths
from llm import create_chat_model
from pipeline_runtime import PipelineRuntime
from runtime import create_runtime


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run APIX agent in a simple TUI.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--mode", choices=["chat", "agent"], default=None, help="Override config.tui.mode.")
    parser.add_argument("--target", default=None, help="Override config.workflow.target.")
    parser.add_argument("--prompts-dir", default=None, help="Override config.workflow.prompts_dir.")
    parser.add_argument("--workspace-root", default=None, help="Override config.workflow.workspace_root.")
    parser.add_argument("--model", default=None, help="Override config.model.model.")
    parser.add_argument("--max-turns", type=int, default=None, help="Override config.agent.max_turns.")
    parser.add_argument("--agent", default=None, help="Select agent profile from agents directory.")
    parser.add_argument("--pipeline", default=None, help="Select pipeline from pipelines.yaml.")
    parser.add_argument(
        "--workflow-mode",
        choices=["single", "pipeline"],
        default=None,
        help="Override config.platform.workflow_mode.",
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


def _chat_response_to_lines(message: AIMessage) -> list[str]:
    content = message.content or ""
    if isinstance(content, str):
        return content.splitlines() or ["(empty response)"]
    return [str(content)]


def _load_env(config_path: str) -> None:
    config_file = Path(config_path).resolve()
    dotenv_path = config_file.with_name(".env")
    load_dotenv(dotenv_path=dotenv_path)


class PiTuiApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    #header {
        background: $surface;
        color: $text;
        padding: 1 2;
    }
    #status {
        padding: 0 2;
    }
    #log {
        height: 1fr;
        padding: 0 2;
    }
    #input {
        padding: 0 2;
    }
    """

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.mode = "chat"
        self.status = "idle"
        self.model_config = None
        self.workflow_config = None
        self.agent_config = None
        self.platform_config = None
        self.agent_spec = None
        self.llm = None
        self.system_prompt: str | None = None
        self.messages: list[Any] = []
        self.runtime = None
        self.pipeline_name: str | None = None
        self.workflow_mode = "single"
        self.workspace_root = ""
        self.target_path = ""
        self._log_lines: list[str] = []
        self._active_stream_line: int | None = None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(f"APIX TUI ({self.mode})", id="header")
            yield Static(self._status_text(), id="status")
            yield RichLog(id="log", wrap=True, markup=False)
            yield Input(placeholder="Type a message and press Enter...", id="input")
            yield Footer()

    async def on_mount(self) -> None:
        config = load_config(self.args.config)

        model_updates: dict[str, object] = {}
        if self.args.model is not None:
            model_updates["model"] = self.args.model

        workflow_updates: dict[str, object] = {}
        if self.args.target is not None:
            workflow_updates["target"] = self.args.target
        if self.args.prompts_dir is not None:
            workflow_updates["prompts_dir"] = self.args.prompts_dir
        if self.args.workspace_root is not None:
            workflow_updates["workspace_root"] = self.args.workspace_root

        agent_updates: dict[str, object] = {}
        if self.args.max_turns is not None:
            agent_updates["max_turns"] = self.args.max_turns

        platform_updates: dict[str, object] = {}
        if self.args.intent_recognition is not None:
            platform_updates["intent_recognition_enabled"] = self.args.intent_recognition
        if self.args.unsafe_auto_exec:
            platform_updates["allow_unsafe_auto_exec"] = True
        if self.args.workflow_mode is not None:
            platform_updates["workflow_mode"] = self.args.workflow_mode

        self.mode = self.args.mode or config.tui.mode
        self.workflow_config = config.workflow.model_copy(update=workflow_updates)
        self.agent_config = config.agent.model_copy(update=agent_updates)
        self.platform_config = config.platform.model_copy(update=platform_updates)
        self.workflow_mode = self.platform_config.workflow_mode
        self.pipeline_name = self.args.pipeline or self.platform_config.default_pipeline

        self.workspace_root, _, self.target_path = resolve_workspace_paths(self.workflow_config)
        app_config = config.model_copy(update={"platform": self.platform_config})
        self.model_config = config.model
        if model_updates:
            self.model_config = self.model_config.model_copy(update=model_updates)

        header = self.query_one("#header", Static)
        header.update(f"APIX TUI ({self.mode}) | Enter=send | Ctrl+C=exit")

        if self.mode == "agent":
            if self.workflow_mode == "pipeline":
                pipeline_spec = resolve_pipeline_spec(app_config, self.workspace_root, pipeline_name=self.pipeline_name)
                self.runtime = PipelineRuntime(
                    app_config=app_config,
                    model_config=self.model_config,
                    agent_config=self.agent_config,
                    pipeline_spec=pipeline_spec,
                    workspace_root=self.workspace_root,
                )
                self._log("Ready.")
                self._log(f"mode={self.mode}")
                self._log("workflow_mode=pipeline")
                self._log(f"pipeline={pipeline_spec.name}")
                self._log(f"target={self.target_path}")
            else:
                self.agent_spec = resolve_agent_spec(app_config, self.workspace_root, agent_name=self.args.agent)
                self.model_config = self.model_config.model_copy(update=self.agent_spec.model_overrides)
                self.runtime = create_runtime(
                    self.model_config,
                    self.agent_config,
                    self.agent_spec,
                    self.workspace_root,
                    intent_recognition_enabled=self.platform_config.intent_recognition_enabled,
                    tool_execution_mode=self.platform_config.tool_execution_mode,
                    allow_unsafe_auto_exec=self.platform_config.allow_unsafe_auto_exec,
                )
                self._log("Ready.")
                self._log(f"mode={self.mode}")
                self._log("workflow_mode=single")
                self._log(f"agent={self.agent_spec.name}")
                self._log(f"target={self.target_path}")
            self._log(
                f"intent_recognition={'on' if self.platform_config.intent_recognition_enabled else 'off'}"
            )
            self._log(
                f"unsafe_auto_exec={'on' if self.platform_config.allow_unsafe_auto_exec else 'off'}"
            )
            self._log("Streaming mode: token + tool events")
            self._log("Type a request and press Enter.")
        else:
            self.llm = create_chat_model(self.model_config)
            self.system_prompt = config.tui.system_prompt or None
            if self.system_prompt:
                self.messages.append(SystemMessage(content=self.system_prompt))
            self._log("Ready.")
            self._log(f"mode={self.mode}")
            self._log("Type a message and press Enter.")
            self._log("Commands: /reset, /exit")

        self.query_one("#input", Input).focus()

    def _status_text(self) -> str:
        return f"Status: {self.status}"

    def _set_status(self, status: str) -> None:
        self.status = status
        self.query_one("#status", Static).update(self._status_text())

    def _flush_log(self) -> None:
        log = self.query_one("#log", RichLog)
        log.clear()
        for line in self._log_lines:
            log.write(line)

    def _log(self, line: str) -> None:
        self._log_lines.append(line)
        self._flush_log()

    def _start_stream_line(self) -> None:
        if self._active_stream_line is not None:
            return
        self._log_lines.append("")
        self._active_stream_line = len(self._log_lines) - 1
        self._flush_log()

    def _append_stream_delta(self, delta: str) -> None:
        if not delta:
            return
        if self._active_stream_line is None:
            self._start_stream_line()
        if self._active_stream_line is None:
            return
        self._log_lines[self._active_stream_line] += delta
        self._flush_log()

    def _end_stream_line(self) -> None:
        self._active_stream_line = None

    def _chat_invoke(self, request: str) -> list[str]:
        if self.llm is None:
            return ["ERROR: LLM not initialized."]
        self.messages.append(HumanMessage(content=request))
        response = self.llm.invoke(self.messages)
        if isinstance(response, AIMessage):
            self.messages.append(response)
            return _chat_response_to_lines(response)
        return [str(response)]

    def _render_agent_event(self, event: dict[str, object]) -> None:
        event_type = event["type"]
        if event_type == "pipeline_start":
            self._log(f"[pipeline:start] {event.get('pipeline_name')} ({event.get('stage_count', 0)} stages)")
        elif event_type == "pipeline_stage_start":
            self._log(
                f"[pipeline:stage:start] #{event.get('stage_index')} "
                f"{event.get('stage_id')} ({event.get('stage_agent')})"
            )
        elif event_type == "pipeline_stage_end":
            self._log(
                f"[pipeline:stage:end] #{event.get('stage_index')} "
                f"{event.get('stage_id')} status={event.get('status')}"
            )
            response = event.get("final_response", "")
            if isinstance(response, str) and response:
                self._log(response)
        elif event_type == "pipeline_end":
            self._log(f"[pipeline:{event.get('status')}] {event.get('pipeline_name')}")
        elif event_type == "intent_recognized":
            intent = event.get("intent", {})
            if isinstance(intent, dict):
                self._log(f"[intent] {intent.get('intent_type')} target={intent.get('target_path')}")
        elif event_type == "intent_skipped":
            self._log("[intent] skipped (disabled)")
        elif event_type == "message_start":
            self._start_stream_line()
        elif event_type == "message_delta":
            delta = event.get("delta", "")
            if isinstance(delta, str) and delta:
                self._append_stream_delta(delta)
        elif event_type == "message_end":
            self._end_stream_line()
        elif event_type == "tool_execution_start":
            self._log(f"[tool:start] {event.get('tool_name')} args={event.get('args')}")
        elif event_type == "tool_execution_update":
            delta = event.get("delta", "")
            if isinstance(delta, str) and delta:
                self._log(delta)
        elif event_type == "tool_execution_end":
            marker = "error" if event.get("is_error") else "ok"
            self._log(f"[tool:end:{marker}] {event.get('tool_name')}")
            result = event.get("result", "")
            if isinstance(result, str) and result:
                self._log(result)
        elif event_type == "tool_blocked_by_policy":
            self._log(f"[tool:blocked] {event.get('tool_name')}")
            result = event.get("result", "")
            if isinstance(result, str) and result:
                self._log(result)
        elif event_type == "agent_end":
            self._log(f"[agent:{event.get('status')}] {event.get('final_response', '')}")

    def _run_agent_stream(self, request: str) -> None:
        if self.runtime is None:
            self.call_from_thread(self._log, "ERROR: runtime not initialized.")
            return
        for event in self.runtime.run(request, target_path=self.target_path, max_turns=self.agent_config.max_turns):
            self.call_from_thread(self._render_agent_event, event)

    @on(Input.Submitted)
    async def handle_submit(self, event: Input.Submitted) -> None:
        request = event.value.strip()
        event.input.value = ""
        if not request:
            return
        if request in ("/quit", "/exit"):
            self.exit()
            return
        if self.mode == "chat" and request == "/reset":
            self.messages = []
            if self.system_prompt:
                self.messages.append(SystemMessage(content=self.system_prompt))
            self._log("(conversation reset)")
            return

        self._set_status("running")
        self._log(f"> {request}")
        try:
            if self.mode == "agent":
                await asyncio.to_thread(self._run_agent_stream, request)
            else:
                lines = await asyncio.to_thread(self._chat_invoke, request)
                for line in lines:
                    self._log(line)
            self._set_status("done")
        except Exception as exc:
            self._log(f"ERROR: {exc}")
            self._set_status("error")
        finally:
            self._log("")


def main() -> None:
    args = _parse_args()
    _load_env(args.config)
    app = PiTuiApp(args)
    app.run()


if __name__ == "__main__":
    main()
