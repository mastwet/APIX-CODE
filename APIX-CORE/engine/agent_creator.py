import json
import time
import asyncio
from typing import Dict, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from dataclasses import asdict

from ..prompts.agent_prompts import *
from ..llm.llm_adapter import LlmNodeAdapter
from .agent_node import MainAgentNode
from .sub_agent_node import SubAgentNode
from .task_manager import task_manager
from ..tools.registry import get_available_tools
from ..tools.tool_node import ApixToolNode
from ..commons.type_def import MainAgentState, SubAgentState, AgentConfigSchema
from ..commons.logger import logger

GRAPH_CACHE_TTL = 600


class AgentCreator:
    """
    Singleton agent graph factory.
    Builds and caches compiled LangGraph StateGraphs for main and sub agents.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # key -> {"graph": CompiledStateGraph, "expire_at": float, "status": str}
        self.graph_cache: Dict[str, Dict] = {}
        self._graph_cache_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Permission helpers
    # ------------------------------------------------------------------

    def _collect_permission(
        self,
        config: AgentConfigSchema,
        permission_level: Literal["main", "sub"],
    ) -> list[str]:
        pure_chat_on = config.get("pure_chat_on", False)

        if pure_chat_on:
            return ["forbidden"]

        permissions = ["default", "file_operation", "command_operation"]
        if permission_level == "main":
            permissions.append("agent_assign")
        return permissions

    # ------------------------------------------------------------------
    # Internal unified builder
    # ------------------------------------------------------------------

    async def _create_agent_core(
        self,
        agent_name: str,
        agent_role: str,
        config: AgentConfigSchema,
        *,
        permission_level: Literal["main", "sub"],
        cache_prefix: str = "",
        log_prefix: str = "[create_agent]",
    ):
        """Unified agent builder for both main and sub agents."""

        logger.trace(f"[agent_creator.py] {log_prefix} Enter")

        config_dict = config if isinstance(config, dict) else asdict(config)

        hash_key = hash(
            cache_prefix
            + agent_name
            + json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
        )

        now = time.time()

        # Check cache
        async with self._graph_cache_lock:
            cached = self.graph_cache.get(hash_key)
            if cached and cached["expire_at"] > now:
                cached["status"] = "running"
                cached["expire_at"] = now + GRAPH_CACHE_TTL
                logger.success(f"{log_prefix} Get Agent From Cache (TTL refreshed).")
                return cached["graph"]

        # Config extraction
        try:
            provider = config.get("models_provider")
            model = config.get("model_name")
            api_key = config.get("api_key", "")
            pure_chat_on = config.get("pure_chat_on", False)
            agent_permission = self._collect_permission(config, permission_level)
        except Exception as e:
            return f"{e}"

        # LLM creation
        try:
            llm = LlmNodeAdapter.get_atapted_llm_node(
                provider=provider,
                model=model,
                api_key=api_key,
                config=config,
            )
            logger.success(f"{log_prefix} Get {model} from {provider}.")
        except Exception as e:
            return f"{e}"

        # Tools
        tools = await get_available_tools(
            agent_permission,
            agent_role,
            workspace_configured=bool(config.get("work_dir", "")),
            client_id=config.get("client_id", ""),
        )
        tool_set = [tool.name for tool in tools]

        if not pure_chat_on:
            if hasattr(llm, "bind_tools"):
                try:
                    llm = llm.bind_tools(tools)
                except NotImplementedError:
                    logger.warning(
                        f"{log_prefix} Binding tools to {model} from "
                        f"{provider} is not supported."
                    )

        # Graph construction - use SubAgentNode for sub-agent roles
        if agent_role in ('sub_agent', 'team_worker'):
            agent_node = SubAgentNode(llm=llm, tool_set=tool_set)
            state_schema = SubAgentState
        else:
            agent_node = MainAgentNode(llm=llm, tool_set=tool_set)
            state_schema = MainAgentState
        graph = StateGraph(state_schema)

        graph.add_node("context_prepare", agent_node.context_prepare)
        graph.add_edge(START, "context_prepare")

        graph.add_node("context_summary", agent_node.context_summary)
        graph.add_edge("context_prepare", "context_summary")

        graph.add_node("llm_call", agent_node.llm_call)
        graph.add_edge("context_summary", "llm_call")

        graph.add_node("messages_persist", agent_node.messages_persist)
        graph.add_conditional_edges(
            "llm_call",
            agent_node.route_after_llm,
            {
                "retry": "llm_call",
                "summary": "context_summary",
                "ok": "messages_persist",
            },
        )

        if not pure_chat_on:
            graph.add_node("tools", ApixToolNode(tools))
            graph.add_conditional_edges(
                "messages_persist",
                agent_node.should_continue,
                {
                    "llm": "context_summary",
                    "tools": "tools",
                    END: END,
                },
            )
            graph.add_edge("tools", "messages_persist")
        else:
            graph.add_edge("messages_persist", END)

        agent_graph = graph.compile()

        # Cache the compiled graph
        async with self._graph_cache_lock:
            self.graph_cache[hash_key] = {
                "graph": agent_graph,
                "expire_at": time.time() + GRAPH_CACHE_TTL,
                "status": "running",
            }

        logger.success(f"{log_prefix} Compile Agent Finish.")
        return agent_graph

    # ------------------------------------------------------------------
    # Public builders
    # ------------------------------------------------------------------

    async def create_agent(
        self, agent_name: str, agent_role: str, config: AgentConfigSchema
    ):
        """Create main agent."""
        return await self._create_agent_core(
            agent_name,
            agent_role,
            config,
            permission_level="main",
            cache_prefix="",
            log_prefix="[create_agent]",
        )

    async def create_sub_agent(
        self, agent_name: str, agent_role: str, config: AgentConfigSchema
    ):
        """Create sub agent."""
        return await self._create_agent_core(
            agent_name,
            agent_role,
            config,
            permission_level="sub",
            cache_prefix="sub_",
            log_prefix="[create_sub_agent]",
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def done(self, agent_graph: CompiledStateGraph) -> None:
        """Mark a graph as done (no longer in active use)."""
        async with self._graph_cache_lock:
            for entry in self.graph_cache.values():
                if entry["graph"] is agent_graph:
                    entry["status"] = "done"
                    return

    async def _clean_expired_graph_cache(self) -> int:
        """Remove expired + done graph cache entries."""
        now = time.time()
        removed = 0

        async with self._graph_cache_lock:
            expired_keys = [
                key
                for key, entry in self.graph_cache.items()
                if entry["expire_at"] <= now and entry["status"] == "done"
            ]
            for key in expired_keys:
                del self.graph_cache[key]
                removed += 1

        if removed:
            logger.info(f"[graph_cache] Cleaned {removed} expired graph(s).")
        return removed


agent_creator = AgentCreator()
