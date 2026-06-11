from __future__ import annotations

try:
    from langchain_core.tools import BaseTool
except ImportError:
    BaseTool = None  # type: ignore[assignment,misc]
try:
    from ..commons.logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    HAS_MCP = True
except ImportError:
    HAS_MCP = False


class McpManager:

    def __init__(self):
        self._clients: dict = {}

    async def load_all_mcp_tools(self, client_id: str = '') -> list[BaseTool]:
        if not HAS_MCP:
            return []
        # MCP tool loading will be implemented when MCP configs are provided
        return []


mcp_mgr = McpManager()
