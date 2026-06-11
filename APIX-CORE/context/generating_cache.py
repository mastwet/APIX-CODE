import os
import time
from pathlib import Path
from typing import Optional
from uuid import uuid4

import yaml
from langchain_core.messages import AIMessageChunk, ToolMessage, AIMessage, SystemMessage

from ..commons.logger import logger

BASE_DIR = '.apix-code'


class GeneratingCache:
    """
    Local file-backed cache for streaming AI message generation.
    Stores AIMessageChunk objects in memory during streaming,
    and persists completed messages as dicts to YAML files.
    """

    def __init__(self):
        # In-memory buffer: generation_id -> AIMessageChunk
        self._cache: dict[str, AIMessageChunk] = {}
        # Metadata: generation_id -> {client_id, history_id, ...}
        self._meta: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_cache_dir(self, client_id: str, history_id: str) -> Path:
        cache_dir = Path(BASE_DIR) / 'generating_cache' / client_id / history_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _message_to_dict(self, message, generation_id: str, timestamp: int) -> dict:
        """Convert a LangChain message to a persistable dict."""
        if isinstance(message, (AIMessage, AIMessageChunk)):
            think = (message.additional_kwargs or {}).get('reasoning_content', '')
            tool_calls = message.tool_calls or []
            return {
                'role': 'ai',
                'content': message.content,
                'think': think,
                'extra': {'tool_calls': tool_calls} if tool_calls else {},
                'generation_id': generation_id,
                'timestamp': timestamp,
            }
        elif isinstance(message, ToolMessage):
            return {
                'role': 'tools',
                'content': str(message.content),
                'info': {
                    'tool_name': message.name,
                    'task_id': message.tool_call_id,
                },
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

    def _dict_to_message(self, data: dict):
        """Convert a persisted dict back to a LangChain message."""
        role = data.get('role', '')
        content = data.get('content', '')

        if role == 'ai':
            think = data.get('think', '')
            extra = data.get('extra', {})
            tool_calls = extra.get('tool_calls', [])
            msg = AIMessage(
                content=content,
                tool_calls=tool_calls,
            )
            if think:
                msg.additional_kwargs['reasoning_content'] = think
            return msg
        elif role == 'tools':
            info = data.get('info', {})
            return ToolMessage(
                content=content,
                name=info.get('tool_name', ''),
                tool_call_id=info.get('task_id', ''),
            )
        elif role == 'system':
            return SystemMessage(content=content)
        return None

    # ------------------------------------------------------------------
    # In-memory cache operations (during streaming)
    # ------------------------------------------------------------------

    def append_message(
        self,
        generation_id: str,
        chunk: AIMessageChunk,
        client_id: str = '',
        history_id: str = '',
    ):
        """Append a streaming chunk to the in-memory cache."""
        if generation_id not in self._cache:
            self._cache[generation_id] = AIMessageChunk(content='')
            self._meta[generation_id] = {
                'client_id': client_id,
                'history_id': history_id,
                'start_time': time.time(),
            }
        self._cache[generation_id] = self._cache[generation_id] + chunk

    def append_dict_message(
        self,
        generation_id: str,
        message_dict: dict,
        client_id: str = '',
        history_id: str = '',
    ):
        """Append a pre-converted dict message to the cache (for replay)."""
        msg = self._dict_to_message(message_dict)
        if msg and isinstance(msg, (AIMessage, AIMessageChunk)):
            self.append_message(generation_id, msg, client_id, history_id)

    def get_cached_message(self, generation_id: str) -> Optional[AIMessageChunk]:
        """Get the accumulated message for a generation."""
        return self._cache.get(generation_id)

    def pop_cached_message(self, generation_id: str) -> Optional[AIMessageChunk]:
        """Pop the accumulated message and remove from cache."""
        self._meta.pop(generation_id, None)
        return self._cache.pop(generation_id, None)

    # ------------------------------------------------------------------
    # File persistence (YAML)
    # ------------------------------------------------------------------

    def _persist_to_file(
        self,
        client_id: str,
        history_id: str,
        generation_id: str,
        message_dict: dict,
    ):
        """Write a message dict to a YAML file."""
        cache_dir = self._get_cache_dir(client_id, history_id)
        file_path = cache_dir / f'{generation_id}.yaml'
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(message_dict, f, allow_unicode=True)
            logger.debug(f'[GeneratingCache] Persisted message to {file_path}')
        except Exception as e:
            logger.error(f'[GeneratingCache] Error persisting message: {e}')

    def load_history(
        self,
        client_id: str,
        history_id: str,
        generation_id: Optional[str] = None,
    ) -> list[dict]:
        """
        Load persisted message dicts from YAML files.
        If generation_id is provided, load only that file.
        Otherwise load all files in the history directory.
        """
        cache_dir = self._get_cache_dir(client_id, history_id)
        messages = []

        if generation_id:
            file_path = cache_dir / f'{generation_id}.yaml'
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                    if data:
                        messages.append(data)
                except Exception as e:
                    logger.error(f'[GeneratingCache] Error loading {file_path}: {e}')
            return messages

        # Load all files sorted by name (timestamp order)
        yaml_files = sorted(cache_dir.glob('*.yaml'))
        for file_path in yaml_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                if data:
                    messages.append(data)
            except Exception as e:
                logger.error(f'[GeneratingCache] Error loading {file_path}: {e}')

        return messages

    def rewrite_history(
        self,
        client_id: str,
        history_id: str,
        messages: list[dict],
    ):
        """
        Rewrite the entire history directory with new messages.
        Clears existing files and writes new ones.
        """
        cache_dir = self._get_cache_dir(client_id, history_id)

        # Clear existing files
        for file_path in cache_dir.glob('*.yaml'):
            try:
                file_path.unlink()
            except Exception as e:
                logger.error(f'[GeneratingCache] Error removing {file_path}: {e}')

        # Write new messages
        for i, msg_dict in enumerate(messages):
            gen_id = msg_dict.get('generation_id', str(uuid4()))
            file_path = cache_dir / f'{gen_id}.yaml'
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.safe_dump(msg_dict, f, allow_unicode=True)
            except Exception as e:
                logger.error(f'[GeneratingCache] Error writing {file_path}: {e}')

        logger.info(f'[GeneratingCache] Rewrote history for {client_id}/{history_id} ({len(messages)} messages)')

    def clear_history(
        self,
        client_id: str,
        history_id: str,
    ):
        """Delete all cached files for a given history."""
        cache_dir = self._get_cache_dir(client_id, history_id)
        for file_path in cache_dir.glob('*.yaml'):
            try:
                file_path.unlink()
            except Exception as e:
                logger.error(f'[GeneratingCache] Error removing {file_path}: {e}')
        logger.info(f'[GeneratingCache] Cleared history for {client_id}/{history_id}')

    def clear_all(self):
        """Clear all in-memory caches."""
        self._cache.clear()
        self._meta.clear()


generating_cache = GeneratingCache()
