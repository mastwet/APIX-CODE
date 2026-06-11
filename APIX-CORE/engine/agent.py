import asyncio
import time
from typing import Optional

from langchain_core.messages import AIMessageChunk
from langgraph.graph.state import CompiledStateGraph

from .agent_creator import AgentCreator
from .task_manager import task_manager
from ..commons.logger import logger


class AgentRuntime:
    """
    Agent runtime with sub-agent worker loop.
    Creates agents via AgentCreator and manages their lifecycle.
    """

    def __init__(self):
        self._creator = AgentCreator()
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False

    async def submit_agent_task(
        self,
        agent_role: str,
        agent_name: str,
        config: dict,
    ) -> CompiledStateGraph:
        """
        Create and return a compiled agent graph ready for streaming.

        Args:
            agent_role: Role identifier (e.g. 'agent', 'main_agent', 'team_leader').
            agent_name: Unique agent name.
            config: AgentConfigSchema dict with model/provider/behavior settings.

        Returns:
            CompiledStateGraph ready to call .astream() or .ainvoke().

        Raises:
            RuntimeError: If graph creation fails.
        """
        logger.trace("[agent.py] [AgentRuntime] [submit_agent_task] Enter")

        agent = await self._creator.create_agent(agent_name, agent_role, config)

        if not isinstance(agent, CompiledStateGraph):
            raise RuntimeError(
                f"Failed to create agent. Please make sure your config is correct."
                f"\n\nDetail: {agent}"
            )

        logger.info(
            f"[submit_agent_task] Agent ready: {agent_role} - {agent_name}"
        )
        return agent

    async def _run_sub_agent(
        self,
        initial_state: dict,
        config: dict,
        agent_name: str,
    ):
        """Create and run a sub-agent graph for a single task."""
        task_id = initial_state.get("task_id", "unknown")
        history_id = initial_state.get("history_id", "")
        agent_role = initial_state.get("agent_role", "sub_agent")

        logger.info(f"[_run_sub_agent] Starting sub-agent {agent_name} task_id={task_id}")

        try:
            # Update status
            await task_manager.update_task_state_store(
                history_id, task_id, "status", "running"
            )

            # Create sub-agent graph
            agent_graph = await self._creator.create_sub_agent(
                agent_name, agent_role, config
            )

            if not isinstance(agent_graph, CompiledStateGraph):
                error_msg = f"Failed to create sub-agent graph: {agent_graph}"
                logger.error(f"[_run_sub_agent] {error_msg}")
                await task_manager.update_task_state_store(
                    history_id, task_id, "status", "failed"
                )
                await task_manager.update_task_state_store(
                    history_id, task_id, "errors", error_msg
                )
                return

            # Run the graph
            outputs = []
            async for event in agent_graph.astream(
                initial_state,
                {"recursion_limit": 256},
                stream_mode="custom",
            ):
                # Check for stop request
                if task_manager.stop_request_queue and not task_manager.stop_request_queue.empty():
                    try:
                        stop_id = task_manager.stop_request_queue.get_nowait()
                        if stop_id == task_id:
                            logger.warning(f"[_run_sub_agent] Task {task_id} stopped by request")
                            await task_manager.update_task_state_store(
                                history_id, task_id, "status", "cancelled"
                            )
                            return
                    except asyncio.QueueEmpty:
                        pass

                # Collect output chunks
                if isinstance(event, dict):
                    data = event.get("data", {})
                    content = data.get("content", "")
                    if content and isinstance(content, str):
                        outputs.append(content)

            # Mark completed
            final_output = "".join(outputs) if outputs else "Task completed."
            await task_manager.update_task_state_store(
                history_id, task_id, "status", "completed"
            )
            await task_manager.update_task_state_store(
                history_id, task_id, "outputs", final_output
            )
            await task_manager.update_task_state_store(
                history_id, task_id, "finish_timestamp", int(time.time())
            )

            logger.success(f"[_run_sub_agent] Task {task_id} completed")

        except Exception as e:
            error_msg = f"Sub-agent error: {str(e)}"
            logger.error(f"[_run_sub_agent] {error_msg}")
            await task_manager.update_task_state_store(
                history_id, task_id, "status", "failed"
            )
            await task_manager.update_task_state_store(
                history_id, task_id, "errors", error_msg
            )
            await task_manager.update_task_state_store(
                history_id, task_id, "finish_timestamp", int(time.time())
            )

    async def _sub_agent_worker_loop(self):
        """Background worker that processes sub-agent tasks from the queue."""
        logger.info("[_sub_agent_worker_loop] Worker started")

        while self._running:
            try:
                # Wait for task with timeout to allow periodic checks
                try:
                    agent_name, initial_state, config = await asyncio.wait_for(
                        task_manager.task_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Run sub-agent in background
                asyncio.create_task(
                    self._run_sub_agent(initial_state, config, agent_name)
                )

            except asyncio.CancelledError:
                logger.info("[_sub_agent_worker_loop] Worker cancelled")
                break
            except Exception as e:
                logger.error(f"[_sub_agent_worker_loop] Error: {e}")
                await asyncio.sleep(1)

        logger.info("[_sub_agent_worker_loop] Worker stopped")

    async def start(self):
        """Start the sub-agent worker loop."""
        if self._running:
            return

        self._running = True
        self._worker_task = asyncio.create_task(
            self._sub_agent_worker_loop(),
            name="sub-agent-worker",
        )
        logger.info("[AgentRuntime] Started sub-agent worker loop")

    async def stop(self):
        """Stop the sub-agent worker loop."""
        if not self._running:
            return

        self._running = False

        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

        logger.info("[AgentRuntime] Stopped sub-agent worker loop")

    async def stop_sub_agent(
        self,
        history_id: str,
        task_ids: list[str],
        reason: str = "",
    ) -> str:
        """Stop specified sub-agent tasks."""
        return await task_manager.stop_tasks(history_id, task_ids, reason=reason)

    async def done(self, agent_graph: CompiledStateGraph) -> None:
        """Mark a graph as done (no longer in active use)."""
        if agent_graph:
            await self._creator.done(agent_graph)


ai_agent = AgentRuntime()
