import copy
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, TypedDict, NotRequired

import yaml
from uuid import uuid4

from langchain_core.messages import (
    SystemMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
    AIMessage,
    AnyMessage,
)

from ..commons.logger import logger
from ..commons.file_utils import load_from_yaml, write_to_yaml
from ..commons.type_def import MainAgentState

BASE_DIR = '.apix-code'


# ------------------------------------------------------------------
# MemoItem — local definition (not yet in type_def.py)
# ------------------------------------------------------------------

class MemoItem(TypedDict):
    memo_id: str
    content: str
    timestamp: NotRequired[float]


# ==================================================================
# AIContextManager
# ==================================================================

class AIContextManager:
    """
    Local version of APIX's AIContextManager.
    Handles message conversion, context management, prompt building,
    and file-backed memory (YAML).
    """

    # ------------------------------------------------------------------
    # Message Conversion (pure, no I/O)
    # ------------------------------------------------------------------

    def create_agent_messages(
        self,
        client_messages: list[dict],
        remain_tool_message: bool = True,
        *,
        after_index: str = None,
        reasoning: bool = False,
    ) -> list[AnyMessage]:
        """
        Convert a list of dict messages into LangChain message objects.

        Handles human/ai/system/tool roles, tool_calls, reasoning_content,
        and after_index filtering.

        Args:
            client_messages: List of message dicts with 'role', 'content', etc.
            remain_tool_message: If False, drop tool role messages.
            after_index: If provided, only include messages after this generation_id.
            reasoning: If True, include reasoning/think content.
        """
        agent_messages: list[AnyMessage] = []

        found_after = after_index is None
        for msg_dict in client_messages:
            role = msg_dict.get('role', '')
            content = msg_dict.get('content', '')
            generation_id = msg_dict.get('generation_id', '')
            think = msg_dict.get('think', '')

            # after_index filtering: skip until we find the target
            if not found_after:
                if generation_id == after_index:
                    found_after = True
                continue

            if role == 'human':
                agent_messages.append(HumanMessage(content=content))

            elif role == 'ai':
                extra = msg_dict.get('extra', {})
                tool_calls = extra.get('tool_calls', [])
                ai_msg = AIMessage(content=content, tool_calls=tool_calls or [])
                if think and reasoning:
                    ai_msg.additional_kwargs['reasoning_content'] = think
                agent_messages.append(ai_msg)

            elif role == 'tools':
                if not remain_tool_message:
                    continue
                info = msg_dict.get('info', {})
                tool_name = info.get('tool_name', '')
                task_id = info.get('task_id', '')
                agent_messages.append(
                    ToolMessage(
                        content=str(content),
                        name=tool_name,
                        tool_call_id=task_id,
                    )
                )

            elif role == 'system':
                agent_messages.append(SystemMessage(content=content))

        # Ensure tool messages have proper pairing
        agent_messages = self._ensure_tool_message(agent_messages)
        return agent_messages

    def create_dict_message(
        self,
        generation_id: str,
        message,
        timestamp: int,
        *,
        filter: bool = False,
    ) -> dict:
        """
        Convert a LangChain message object to a persistable dict.

        Args:
            generation_id: The generation ID to tag the message with.
            message: A LangChain message object.
            timestamp: Unix timestamp for the message.
            filter: If True, filter out empty/tool messages.
        """
        if isinstance(message, (AIMessage, AIMessageChunk)):
            think = (message.additional_kwargs or {}).get('reasoning_content', '')
            tool_calls = message.tool_calls or []
            result = {
                'role': 'ai',
                'content': message.content,
                'think': think,
                'extra': {'tool_calls': tool_calls} if tool_calls else {},
                'generation_id': generation_id,
                'timestamp': timestamp,
            }
            if filter and not message.content and not tool_calls:
                return {}
            return result

        elif isinstance(message, ToolMessage):
            result = {
                'role': 'tools',
                'content': str(message.content),
                'info': {
                    'tool_name': message.name,
                    'task_id': message.tool_call_id,
                },
                'generation_id': generation_id,
                'timestamp': timestamp,
            }
            if filter:
                return {}
            return result

        elif isinstance(message, HumanMessage):
            return {
                'role': 'human',
                'content': message.content,
                'generation_id': generation_id,
                'timestamp': timestamp,
            }

        elif isinstance(message, SystemMessage):
            return {
                'role': 'system',
                'content': message.content,
                'generation_id': generation_id,
                'timestamp': timestamp,
            }

        return {}

    def _ensure_tool_message(self, agent_messages: list) -> list:
        """
        Inject missing ToolMessages for tool_calls in AIMessages.

        If an AIMessage has tool_calls but no corresponding ToolMessage
        follows, inject a placeholder ToolMessage for each missing call.
        """
        result = []
        pending_tool_calls: dict[str, dict] = {}

        for msg in agent_messages:
            if isinstance(msg, (AIMessage, AIMessageChunk)) and msg.tool_calls:
                result.append(msg)
                for tc in msg.tool_calls:
                    tc_id = tc.get('id', '')
                    tc_name = tc.get('name', '')
                    if tc_id:
                        pending_tool_calls[tc_id] = {
                            'name': tc_name,
                            'args': tc.get('args', {}),
                        }

            elif isinstance(msg, ToolMessage):
                # This ToolMessage satisfies a pending tool_call
                pending_tool_calls.pop(msg.tool_call_id, None)
                result.append(msg)

            else:
                result.append(msg)

        # Inject placeholder ToolMessages for any remaining unmatched tool_calls
        for tc_id, tc_info in pending_tool_calls.items():
            placeholder = ToolMessage(
                content=f'[Tool call {tc_info["name"]} was not executed or response was lost]',
                name=tc_info['name'],
                tool_call_id=tc_id,
            )
            result.append(placeholder)

        return result

    def _extract_mes_info(
        self,
        message,
        *,
        fallback_model_provider: str = '',
        fallback_model_name: str = '',
        fallback_timestamp: int = 0,
    ) -> dict:
        """
        Extract metadata from a LangChain message object.

        Returns a dict with role, content, model info, timestamp, etc.
        """
        timestamp = fallback_timestamp or int(time.time())

        if isinstance(message, (AIMessage, AIMessageChunk)):
            think = (message.additional_kwargs or {}).get('reasoning_content', '')
            tool_calls = message.tool_calls or []
            response_metadata = getattr(message, 'response_metadata', {}) or {}
            model = response_metadata.get('model_name', fallback_model_name)
            provider = response_metadata.get('model_provider', fallback_model_provider)
            return {
                'role': 'ai',
                'content': message.content,
                'think': think,
                'tool_calls': tool_calls,
                'model': model,
                'provider': provider,
                'timestamp': timestamp,
            }

        elif isinstance(message, ToolMessage):
            return {
                'role': 'tools',
                'content': str(message.content),
                'tool_name': message.name,
                'tool_call_id': message.tool_call_id,
                'timestamp': timestamp,
            }

        elif isinstance(message, HumanMessage):
            return {
                'role': 'human',
                'content': message.content,
                'timestamp': timestamp,
            }

        elif isinstance(message, SystemMessage):
            return {
                'role': 'system',
                'content': message.content,
                'timestamp': timestamp,
            }

        return {'role': 'unknown', 'content': '', 'timestamp': timestamp}

    # ------------------------------------------------------------------
    # Context Management (local file I/O)
    # ------------------------------------------------------------------

    def drop_tool_messages(
        self,
        input_messages: list,
        *,
        split_by_todos: bool = True,
        min_keep: int = 16,
    ) -> list:
        """
        Drop outdated tool message content to reduce context size.

        Preserves the most recent tool messages (within min_keep tail),
        and for older tool messages replaces content with a placeholder.

        Args:
            input_messages: List of LangChain messages.
            split_by_todos: If True, respect todo boundaries.
            min_keep: Number of recent messages to keep intact.
        """
        if not input_messages:
            return input_messages

        total = len(input_messages)
        if total <= min_keep:
            return input_messages

        cutoff = total - min_keep
        result = []

        for i, msg in enumerate(input_messages):
            if i < cutoff and isinstance(msg, ToolMessage):
                # Replace tool content with a short placeholder
                placeholder = ToolMessage(
                    content='[Tool output removed to save context]',
                    name=msg.name,
                    tool_call_id=msg.tool_call_id,
                )
                result.append(placeholder)
            else:
                result.append(msg)

        return result

    def split_messages(
        self,
        input_messages: list,
        keep_recent: int = 14,
    ) -> Tuple[list, list]:
        """
        Split messages into (older, recent) for compression.

        The older portion is candidates for summary/compression.
        The recent portion is always kept intact.

        Args:
            input_messages: List of LangChain messages.
            keep_recent: Number of recent messages to keep.
        """
        if not input_messages:
            return [], []

        if len(input_messages) <= keep_recent:
            return [], list(input_messages)

        split_point = len(input_messages) - keep_recent
        older = list(input_messages[:split_point])
        recent = list(input_messages[split_point:])
        return older, recent

    def filter_agent_messages(self, input_messages: list) -> Tuple[list, list]:
        """
        Filter messages into (summary_safe, kept) groups.

        Summary-safe messages are Human/AI messages suitable for compression.
        Kept messages are Tool/System messages that should be preserved as-is.

        Returns:
            (summary_safe_messages, kept_messages)
        """
        summary_safe = []
        kept = []

        for msg in input_messages:
            if isinstance(msg, (HumanMessage, AIMessage, AIMessageChunk)):
                summary_safe.append(msg)
            else:
                kept.append(msg)

        return summary_safe, kept

    # ------------------------------------------------------------------
    # Runtime Prompt Builders
    # ------------------------------------------------------------------

    def create_skills_prompt(self, state: MainAgentState, agent_role: str = None) -> str:
        """Build the skills section of the runtime prompt."""
        config = state.get('config', {})
        enable_skill_load = config.get('enable_skill_load', False)

        if not enable_skill_load:
            return ''

        skills = config.get('skills', [])
        if not skills:
            return ''

        lines = ['## Loaded Skills\n']
        for skill in skills:
            name = skill.get('name', 'unknown')
            desc = skill.get('description', '')
            lines.append(f'- **{name}**: {desc}')
        lines.append('')
        return '\n'.join(lines)

    def create_documents_prompt(self, state: MainAgentState, agent_role: str = None) -> str:
        """Build the documents/RAG section of the runtime prompt."""
        config = state.get('config', {})
        enable_knowledge = config.get('enable_knowledge_retrieval', False)

        if not enable_knowledge:
            return ''

        documents = config.get('documents', [])
        if not documents:
            return ''

        lines = ['## Referenced Documents\n']
        for doc in documents:
            name = doc.get('name', 'unknown')
            desc = doc.get('description', '')
            lines.append(f'- **{name}**: {desc}')
        lines.append('')
        return '\n'.join(lines)

    def create_workflow_prompt(self, state: MainAgentState, agent_role: str = None) -> str:
        """Build workflow instructions based on agent role."""
        role = agent_role or state.get('agent_role', 'agent')

        if role in ('team_leader', 'main_agent'):
            return (
                '## Workflow\n'
                'You are a leader agent. When receiving a user request:\n'
                '1. Analyze the request and break it into sub-tasks.\n'
                '2. Assign sub-tasks to appropriate workers using TODO items.\n'
                '3. Format: `[TODO] worker_name → goal_description`\n'
                '4. Review worker outputs and synthesize a final response.\n'
                '5. If a worker fails, reassign or handle directly.\n\n'
            )
        elif role in ('team_worker', 'sub_agent'):
            return (
                '## Workflow\n'
                'You are a worker agent. When receiving a task:\n'
                '1. Execute the assigned task step by step.\n'
                '2. Use available tools as needed.\n'
                '3. Report your result clearly and concisely.\n'
                '4. If you encounter an error, try alternative approaches.\n\n'
            )
        else:
            return (
                '## Workflow\n'
                'You are a conversational agent. When receiving a user request:\n'
                '1. Understand the user\'s intent.\n'
                '2. Use available tools to fulfill the request.\n'
                '3. Provide a clear, helpful response.\n\n'
            )

    def create_shortterm_prompt(self, messages: list) -> str:
        """Build a short-term memory summary from recent messages."""
        if not messages:
            return ''

        summary_parts = []
        for msg in messages[-10:]:
            if isinstance(msg, HumanMessage):
                content = msg.content[:200] if isinstance(msg.content, str) else str(msg.content)[:200]
                summary_parts.append(f'User: {content}')
            elif isinstance(msg, (AIMessage, AIMessageChunk)):
                content = msg.content[:200] if isinstance(msg.content, str) else str(msg.content)[:200]
                if content:
                    summary_parts.append(f'Assistant: {content}')

        if not summary_parts:
            return ''

        return '## Recent Conversation Context\n' + '\n'.join(summary_parts) + '\n\n'

    def create_workspace_prompt(self, state: MainAgentState, agent_role: str = None) -> str:
        """Build workspace environment prompt for local mode (no Docker)."""
        config = state.get('config', {})
        work_dir = config.get('work_dir', '')

        if not work_dir:
            return '## No workspace directory has been specified by the user.\n\n'

        if not os.path.exists(work_dir):
            return f'## Workspace directory does not exist: {work_dir}\n\n'

        return f'''## Workspace Environment

Working directory: {work_dir}

Rules:
- Use relative paths when possible
- All file operations are relative to {work_dir}
'''

    def create_todo_prompt(self, state: MainAgentState, agent_role: str = None) -> str:
        """Build the TODO list section of the runtime prompt."""
        todos = state.get('todos', [])
        if not todos:
            return ''

        lines = ['## Current TODO List\n']
        for i, todo in enumerate(todos):
            status = todo.get('status', 'pending')
            content = todo.get('content', '')
            assignee = todo.get('assignee', '')
            status_icon = '✅' if status == 'done' else '⏳' if status == 'in_progress' else '⬜'
            assignee_str = f' [{assignee}]' if assignee else ''
            lines.append(f'{status_icon} {i + 1}. {content}{assignee_str}')
        lines.append('')
        return '\n'.join(lines)

    def create_memorandum_prompt(self, state: MainAgentState, agent_role: str = None) -> str:
        """Build the memorandum section of the runtime prompt."""
        memorandum = state.get('memorandum', '')
        if not memorandum:
            return ''

        return f'## Memorandum\n{memorandum}\n\n'

    def create_system_prompt_list(
        self,
        state: MainAgentState,
        agent_role: str = None,
    ) -> list[SystemMessage]:
        """
        Build the complete system prompt as a list of SystemMessages.
        Combines role prompt, runtime prompt sections, and agent prompt.
        """
        config = state.get('config', {})
        role = agent_role or state.get('agent_role', 'agent')

        # Role prompt from config
        role_prompt_schema = config.get('role_prompt', {})
        role_prompt = role_prompt_schema.get('definition', '') if role_prompt_schema else ''

        # Build runtime prompt sections
        sections = []

        # Date
        from ..commons.common_func import get_date_natural_language
        sections.append(get_date_natural_language())

        # Agent identity prompt
        from ..prompts.agent_prompts import (
            DEFAULT_AGENT_PROMPT,
            DEFAULT_LEADER_PROMPT,
            DEFAULT_WORKER_PROMPT,
        )
        if role in ('team_leader', 'main_agent'):
            sections.append(DEFAULT_LEADER_PROMPT)
        elif role in ('team_worker', 'sub_agent'):
            sections.append(DEFAULT_WORKER_PROMPT)
        else:
            sections.append(DEFAULT_AGENT_PROMPT)

        # Custom role prompt
        if role_prompt:
            sections.append(f'## Agent Role Definition\n{role_prompt}')

        # Workflow
        workflow = self.create_workflow_prompt(state, role)
        if workflow:
            sections.append(workflow)

        # Workspace
        workspace = self.create_workspace_prompt(state, role)
        if workspace:
            sections.append(workspace)

        # Skills
        skills = self.create_skills_prompt(state, role)
        if skills:
            sections.append(skills)

        # Documents
        docs = self.create_documents_prompt(state, role)
        if docs:
            sections.append(docs)

        # TODOs
        todos = self.create_todo_prompt(state, role)
        if todos:
            sections.append(todos)

        # Memorandum
        memo = self.create_memorandum_prompt(state, role)
        if memo:
            sections.append(memo)

        # Short-term memory
        messages = state.get('messages', [])
        shortterm = self.create_shortterm_prompt(messages)
        if shortterm:
            sections.append(shortterm)

        # Long-term memory
        longterm_memory = state.get('longterm_memory', '')
        if longterm_memory:
            sections.append(f'## Long-term Memory\n{longterm_memory}\n\n')

        # Tools prompt
        from ..prompts.agent_prompts import DEFAULT_TOOLS_PROMPT
        from ..tools.registry import conflict_tool_set

        tool_set = config.get('tool_set', [])
        unique_tools: list[str] = []
        for tool in tool_set:
            if tool not in unique_tools:
                unique_tools.append(tool)

        tool_list_text = (
            '\n'.join(f'- {name}' for name in unique_tools)
            if unique_tools
            else 'No tools available.'
        )
        tools_block = DEFAULT_TOOLS_PROMPT.format(
            tool_list=tool_list_text,
            conflict_tool_list=str(conflict_tool_set),
        )
        sections.append(tools_block)

        # Combine all sections
        full_prompt = '\n\n'.join(s for s in sections if s)
        return [SystemMessage(content=full_prompt)]

    def create_role_prompt_list(
        self,
        state: MainAgentState,
        agent_role: str = None,
    ) -> list:
        """
        Build role-specific prompt list.
        Returns a list of prompt strings for the given agent role.
        """
        config = state.get('config', {})
        role = agent_role or state.get('agent_role', 'agent')

        role_prompt_schema = config.get('role_prompt', {})
        role_prompt = role_prompt_schema.get('definition', '') if role_prompt_schema else ''

        prompts = []

        # Base agent prompt
        from ..prompts.agent_prompts import (
            DEFAULT_AGENT_PROMPT,
            DEFAULT_LEADER_PROMPT,
            DEFAULT_WORKER_PROMPT,
        )
        if role in ('team_leader', 'main_agent'):
            prompts.append(DEFAULT_LEADER_PROMPT)
        elif role in ('team_worker', 'sub_agent'):
            prompts.append(DEFAULT_WORKER_PROMPT)
        else:
            prompts.append(DEFAULT_AGENT_PROMPT)

        # Custom role prompt
        if role_prompt:
            prompts.append(role_prompt)

        return prompts

    # ------------------------------------------------------------------
    # Local Memory Methods (YAML-backed)
    # ------------------------------------------------------------------

    def _get_memory_dir(self, state: MainAgentState) -> Path:
        """Get or create the memory directory for the current workspace."""
        config = state.get('config', {})
        work_dir = config.get('work_dir', '.')
        memo_dir = Path(work_dir) / BASE_DIR / 'memory'
        memo_dir.mkdir(parents=True, exist_ok=True)
        return memo_dir

    def init_memorandum_list(self, state: MainAgentState) -> str:
        """
        Load memorandum from local YAML file.
        Returns a formatted string of all memorandum entries.
        """
        memo_dir = self._get_memory_dir(state)
        memo_file = memo_dir / 'memorandum.yaml'

        if not memo_file.exists():
            return ''

        try:
            data = load_from_yaml(str(memo_file))
            if not data:
                return ''

            items = data.get('items', [])
            if not items:
                return ''

            lines = []
            for item in items:
                content = item.get('content', '')
                ts = item.get('timestamp', 0)
                if ts:
                    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                    date_str = dt.strftime('%Y-%m-%d %H:%M')
                    lines.append(f'[{date_str}] {content}')
                else:
                    lines.append(content)

            return '\n'.join(lines)

        except Exception as e:
            logger.error(f'[AIContextManager] Error loading memorandum: {e}')
            return ''

    def insert_shortterm_memory(
        self,
        client_id: str,
        history_id: str,
        memory_id: str,
        content: str,
    ):
        """
        Save a short-term memory entry to local YAML.
        """
        # Use a global memory dir (not workspace-specific)
        memo_dir = Path(BASE_DIR) / 'memory' / 'shortterm'
        memo_dir.mkdir(parents=True, exist_ok=True)

        file_path = memo_dir / f'{client_id}_{history_id}.yaml'

        try:
            # Load existing
            existing = {}
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing = yaml.safe_load(f) or {}

            items = existing.get('items', [])

            # Add new item
            items.append({
                'memory_id': memory_id,
                'content': content,
                'timestamp': time.time(),
            })

            # Keep only last 50 items
            if len(items) > 50:
                items = items[-50:]

            existing['items'] = items

            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(existing, f, allow_unicode=True)

            logger.debug(f'[AIContextManager] Saved short-term memory: {memory_id}')

        except Exception as e:
            logger.error(f'[AIContextManager] Error saving short-term memory: {e}')

    def fetch_shortterm_memory(
        self,
        client_id: str,
        history_id: str,
    ) -> list[dict]:
        """
        Load short-term memory entries from local YAML.
        """
        memo_dir = Path(BASE_DIR) / 'memory' / 'shortterm'
        file_path = memo_dir / f'{client_id}_{history_id}.yaml'

        if not file_path.exists():
            return []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}

            items = data.get('items', [])
            return items

        except Exception as e:
            logger.error(f'[AIContextManager] Error loading short-term memory: {e}')
            return []


ai_context_manager = AIContextManager()
