from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from .config import AgentConfig, AppConfig, ModelConfig, PipelineSpec, resolve_agent_spec
from .events import RuntimeEvent
from .runtime import create_runtime


@dataclass
class StageResult:
    stage_id: str
    stage_agent: str
    request: str
    status: str
    turns: int
    final_response: str


class PipelineRuntime:
    """Sequential multi-agent pipeline runtime."""

    def __init__(
        self,
        app_config: AppConfig,
        model_config: ModelConfig,
        agent_config: AgentConfig,
        pipeline_spec: PipelineSpec,
        workspace_root: str,
    ) -> None:
        self.app_config = app_config
        self.model_config = model_config
        self.agent_config = agent_config
        self.pipeline_spec = pipeline_spec
        self.workspace_root = workspace_root
        self.max_turns_default = agent_config.max_turns

    def _render_stage_request(
        self,
        task_template: str,
        request: str,
        target_path: str,
        previous: str,
    ) -> str:
        values = {
            "request": request,
            "target_path": target_path or "(none)",
            "workspace_root": self.workspace_root,
            "previous": previous,
        }
        try:
            rendered = task_template.format_map(values)
        except KeyError as exc:
            missing = str(exc).strip("'")
            raise ValueError(f"Unknown template variable '{{{missing}}}' in pipeline stage template.") from exc
        return rendered.strip() or request

    def _build_success_response(self, results: list[StageResult]) -> str:
        if not results:
            return "Pipeline completed with no stages."
        last = results[-1]
        lines = [f"Pipeline '{self.pipeline_spec.name}' completed.", "", f"Final stage: {last.stage_id} ({last.stage_agent})"]
        if last.final_response:
            lines.extend(["", "Final response:", last.final_response])
        return "\n".join(lines)

    def _build_failure_response(self, results: list[StageResult], failed: StageResult) -> str:
        lines = [
            f"Pipeline '{self.pipeline_spec.name}' stopped at stage '{failed.stage_id}' ({failed.stage_agent}).",
            f"Status: {failed.status}",
            f"Turns used: {failed.turns}/{self.max_turns_default}",
        ]
        if failed.final_response:
            lines.extend(["", "Stage output:", failed.final_response])
        if results:
            lines.extend(["", "Completed stages:"])
            for result in results:
                lines.append(f"- {result.stage_id} ({result.stage_agent}) -> {result.status}")
        return "\n".join(lines)

    def run(self, request: str, target_path: str, max_turns: int | None = None) -> Iterator[RuntimeEvent]:
        max_turns_value = max_turns or self.max_turns_default
        previous_output = ""
        stage_results: list[StageResult] = []

        yield {
            "type": "pipeline_start",
            "pipeline_name": self.pipeline_spec.name,
            "stage_count": len(self.pipeline_spec.stages),
        }

        for index, stage in enumerate(self.pipeline_spec.stages, start=1):
            stage_request = self._render_stage_request(stage.task_template, request, target_path, previous_output)
            yield {
                "type": "pipeline_stage_start",
                "stage_id": stage.id,
                "stage_agent": stage.agent,
                "stage_index": index,
                "stage_request": stage_request,
            }

            agent_spec = resolve_agent_spec(self.app_config, self.workspace_root, agent_name=stage.agent)
            stage_model_config = self.model_config.model_copy(update=agent_spec.model_overrides)
            runtime = create_runtime(
                model_config=stage_model_config,
                agent_config=self.agent_config,
                agent_spec=agent_spec,
                workspace_root=self.workspace_root,
                intent_recognition_enabled=self.app_config.platform.intent_recognition_enabled,
                tool_execution_mode=self.app_config.platform.tool_execution_mode,
                allow_unsafe_auto_exec=self.app_config.platform.allow_unsafe_auto_exec,
            )

            stage_status = "STOPPED"
            stage_turns = 0
            stage_final_response = ""

            for event in runtime.run(stage_request, target_path=target_path, max_turns=max_turns_value):
                if event["type"] == "agent_end":
                    stage_status = str(event.get("status", "STOPPED"))
                    stage_turns = int(event.get("turn", 0) or 0)
                    stage_final_response = str(event.get("final_response", ""))
                    continue

                enriched_event: RuntimeEvent = dict(event)
                enriched_event["stage_id"] = stage.id
                enriched_event["stage_agent"] = stage.agent
                enriched_event["stage_index"] = index
                yield enriched_event

            result = StageResult(
                stage_id=stage.id,
                stage_agent=stage.agent,
                request=stage_request,
                status=stage_status,
                turns=stage_turns,
                final_response=stage_final_response,
            )
            stage_results.append(result)

            yield {
                "type": "pipeline_stage_end",
                "stage_id": stage.id,
                "stage_agent": stage.agent,
                "stage_index": index,
                "status": stage_status,
                "turn": stage_turns,
                "final_response": stage_final_response,
            }

            if stage_status != "DONE":
                final_response = self._build_failure_response(stage_results[:-1], result)
                yield {
                    "type": "pipeline_end",
                    "pipeline_name": self.pipeline_spec.name,
                    "status": "STOPPED",
                    "reason": "stage_failed",
                    "stage_id": stage.id,
                    "stage_agent": stage.agent,
                    "final_response": final_response,
                }
                yield {
                    "type": "agent_end",
                    "status": "STOPPED",
                    "reason": "stage_failed",
                    "stage_id": stage.id,
                    "stage_agent": stage.agent,
                    "final_response": final_response,
                }
                return

            previous_output = stage_final_response

        final_response = self._build_success_response(stage_results)
        yield {
            "type": "pipeline_end",
            "pipeline_name": self.pipeline_spec.name,
            "status": "DONE",
            "final_response": final_response,
        }
        yield {
            "type": "agent_end",
            "status": "DONE",
            "turn": max((result.turns for result in stage_results), default=0),
            "final_response": final_response,
        }
