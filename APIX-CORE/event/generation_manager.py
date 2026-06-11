import asyncio
import copy
import time
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional
from uuid import uuid4

from ..commons.logger import logger


GENERATION_TTL = 600


@dataclass
class GenerationState:
    """
    State for a single AI generation.
    """

    history_id: str
    generation_id: str
    client_id: str

    # running / finished / aborted
    status: Literal["running", "finished", "aborted"] = "running"

    cache_tokens: dict = field(default_factory=lambda: {
        "role": "ai",
        "content": "",
        "think": "",
        "extra": {},
        "info": {},
        "generation_id": "",
        "timestamp": 0
    })
    parent_node_id: str = field(default='-')

    created_at: float = field(default_factory=time.time)

    # Protect cache_tokens concurrent access
    gen_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class GenerationManager:

    def __init__(self):
        self._connections: Dict[str, Dict[str, GenerationState]] = {}  # {client_id: {generation_id: generation_state}}
        self._active_generation_ids: Dict[str, list[str]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    def _get_lock(self, client_id: str) -> asyncio.Lock:
        lock = self._locks.get(client_id)
        if not lock:
            lock = asyncio.Lock()
            self._locks[client_id] = lock
        return lock

    def _get_client_generations(self, client_id: str) -> Dict[str, GenerationState]:
        gens = self._connections.get(client_id)
        if gens is None:
            gens = {}
            self._connections[client_id] = gens
        return gens

    def _get_active_list(self, client_id: str) -> list[str]:
        active_list = self._active_generation_ids.get(client_id)
        if active_list is None:
            active_list = []
            self._active_generation_ids[client_id] = active_list
        return active_list

    def list_running_generations(self, client_id: str) -> list[str]:
        gens = self._connections.get(client_id, {})
        return [gid for gid, gen in gens.items() if gen.status == "running"]

    async def create_generation(
        self,
        client_id: str,
        history_id: str,
    ) -> str:
        new_gen_id = str(uuid4())

        # Abort existing generations for the same history
        async with self._get_lock(client_id):
            gens = self._get_client_generations(client_id)
            active_generation_ids = self._get_active_list(client_id)

            for gid in list(active_generation_ids):
                gen = gens.get(gid)
                if gen and gen.history_id == history_id and gen.status == "running":
                    await self.persist_cache_tokens(gen)
                    gen.status = "aborted"
                    try:
                        active_generation_ids.remove(gid)
                    except ValueError:
                        pass

        async with self._get_lock(client_id):
            gens = self._get_client_generations(client_id)
            active_generation_ids = self._get_active_list(client_id)

            gens[new_gen_id] = GenerationState(
                history_id=history_id,
                generation_id=new_gen_id,
                client_id=client_id
            )

            if new_gen_id not in active_generation_ids:
                active_generation_ids.append(new_gen_id)

        return new_gen_id

    def _ensure_code_block(self, content: str) -> str:
        """
        Ensure markdown code blocks in cache are properly closed.

        If the number of ``` is odd, append a closing ``` at the end.

        Args:
            content (str): streamed markdown content

        Returns:
            str: content with properly closed code block
        """

        if not content:
            return content

        in_code_block = False

        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code_block = not in_code_block

        if in_code_block:
            if not content.endswith("\n"):
                content += "\n"
            content += "```\n"

        return content

    async def update_cache_tokens(
        self,
        client_id: str,
        generation_id: str,
        envelope: dict,
    ):
        """
        Update cache_tokens based on streaming event.

        Behavior:
            - Maintains incremental content / think buffers
            - Resets buffers on specific lifecycle events
            - Thread-safe via gen_lock
        """

        gen = self.get_generation(client_id, generation_id)
        if not gen or gen.status != "running":
            return

        data = envelope.get("data") or {}
        action = data.get("event_name")
        content = data.get("content")

        async with gen.gen_lock:

            # Stream lifecycle start
            if action == "node_stream_start":
                gen.cache_tokens = {
                    "role": "ai",
                    "content": "",
                    "think": "",
                    "extra": {},
                    "info": {},
                    "generation_id": gen.generation_id,
                    "timestamp": time.time()
                }

            # Persist finished: reset buffer
            elif action == "messages_persist_end":
                gen.cache_tokens["content"] = ""
                gen.cache_tokens["think"] = ""

            # Content streaming
            elif action == "content_chunk_rtn":
                if content:
                    gen.cache_tokens["content"] += content

            # Think streaming
            elif action == "think_chunk_rtn":
                if content:
                    gen.cache_tokens["think"] += content

            # Tool execution: clear buffers
            elif action == "tool_exec_chunk_rtn":
                gen.cache_tokens["content"] = ""
                gen.cache_tokens["think"] = ""

    async def persist_cache_tokens(self, gen: GenerationState):
        async with gen.gen_lock:
            interrupted_msg = copy.deepcopy(gen.cache_tokens)

        if interrupted_msg:
            ts = int(time.time() * 1000)

            content = self._ensure_code_block(interrupted_msg.get("content", ""))
            think = self._ensure_code_block(interrupted_msg.get("think", ""))

            think_endswith = "[Conversation Abort]" if think and not content else ""
            content_endswith = "[Conversation Abort]" if content or not think_endswith else ""

            interrupted_msg.update({
                "content": content + content_endswith,
                "think": think + think_endswith,
                "generation_id": gen.generation_id,
                "timestamp": ts,
            })

            # Simplified: just log the persisted tokens (no external service call)
            logger.info(
                f"[persist_cache_tokens] generation={gen.generation_id}, "
                f"client={gen.client_id}, history={gen.history_id}, "
                f"content_len={len(interrupted_msg.get('content', ''))}, "
                f"think_len={len(interrupted_msg.get('think', ''))}"
            )

    async def abort_generation(self, client_id: str, generation_id: str):
        async with self._get_lock(client_id):
            gens = self._connections.get(client_id)
            if not gens:
                return

            gen = gens.get(generation_id)
            if not gen or gen.status != "running":
                return

            await self.persist_cache_tokens(gen)
            gen.status = "aborted"

            active_generation_ids = self._active_generation_ids.get(client_id)
            if active_generation_ids:
                try:
                    active_generation_ids.remove(generation_id)
                except ValueError:
                    pass

    async def is_generation_aborted(self, client_id: str, generation_id: str) -> bool:
        async with self._get_lock(client_id):
            gens = self._connections.get(client_id)
            if not gens:
                return True

            gen = gens.get(generation_id)
            return (not gen) or gen.status == "aborted"

    def get_generation(self, client_id: str, generation_id: str) -> Optional[GenerationState]:
        gens = self._connections.get(client_id)
        if not gens:
            logger.exception(f"[get_generation] Client {client_id} not register a generation state set")
            return None
        return gens.get(generation_id)

    async def clean_expired(self) -> int:
        """
        Clean expired generations across all clients.

        Returns:
            Total number of removed generations

        Behavior:
            - Removes finished/aborted generations older than TTL
            - Safe to call periodically by external scheduler
        """
        if not self._connections:
            return 0

        now = time.time()
        total_removed = 0

        for client_id, gens in list(self._connections.items()):
            if not gens:
                continue

            async with self._get_lock(client_id):
                to_delete = []

                for gen_id, gen in gens.items():
                    if gen.status == "running":
                        continue

                    age = now - gen.created_at
                    if age > GENERATION_TTL:
                        to_delete.append(gen_id)

                for gen_id in to_delete:
                    gens.pop(gen_id, None)

                    active_generation_ids = self._active_generation_ids.get(client_id, [])
                    try:
                        active_generation_ids.remove(gen_id)
                    except ValueError:
                        pass

                if to_delete:
                    removed = len(to_delete)
                    total_removed += removed
                    logger.info(f"[generation_cache] client={client_id}, removed={removed} generation(s)")

        return total_removed


generation_manager = GenerationManager()
