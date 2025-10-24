# -*- coding: utf-8 -*-
"""Graphiti MCP integration for LangGraph memory.

This module provides a thin wrapper around the ``langchain-mcp-adapters`` client
so the Agentic RAG graph can blend short-term conversation history with
Graphiti-backed long-term memory.

It is intentionally defensive: if Graphiti configuration is missing or the MCP
client is unavailable, the manager degrades gracefully and simply relies on
the in-process short-term buffer.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools
except ImportError:  # pragma: no cover - optional dependency path
    MultiServerMCPClient = None  # type: ignore[assignment]
    load_mcp_tools = None  # type: ignore[assignment]

try:
    from mcp.types import CallToolResult
except Exception:  # pragma: no cover - optional dependency path
    CallToolResult = Any  # type: ignore[assignment]


def _parse_json_env(value: str | None, default: Dict[str, Any]) -> Dict[str, Any]:
    if not value:
        return dict(default)
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return dict(default)


def _parse_list_env(value: str | None) -> List[str]:
    if not value:
        return []
    try:
        return [item.strip() for item in json.loads(value) if item]
    except json.JSONDecodeError:
        return [part.strip() for part in value.split(",") if part.strip()]


@dataclass(slots=True)
class GraphitiMemorySettings:
    """Configuration for connecting to a Graphiti MCP server."""

    server_name: str = field(default="graphiti")
    transport: str | None = field(default=None)
    command: str | None = field(default=None)
    args: List[str] = field(default_factory=list)
    url: str | None = field(default=None)
    headers: Dict[str, str] = field(default_factory=dict)
    env: Dict[str, str] = field(default_factory=dict)
    short_term_window: int = field(default=5)
    search_tool: str | None = field(default="graphiti.search_memories")
    upsert_tool: str | None = field(default="graphiti.upsert_memory")
    namespace: str = field(default="agentic-rag")
    metadata_tags: List[str] = field(default_factory=list)
    search_limit: int = field(default=5)
    enabled: bool = field(default=False)

    @classmethod
    def from_env(cls) -> "GraphitiMemorySettings":
        transport = os.getenv("GRAPHITI_MCP_TRANSPORT")
        command = os.getenv("GRAPHITI_MCP_COMMAND")
        args = shlex.split(os.getenv("GRAPHITI_MCP_ARGS", "")) if command else []
        url = os.getenv("GRAPHITI_MCP_URL")
        headers = _parse_json_env(os.getenv("GRAPHITI_MCP_HEADERS"), {})
        proc_env = _parse_json_env(os.getenv("GRAPHITI_MCP_ENV"), {})
        short_term_window = int(os.getenv("GRAPHITI_SHORT_TERM_WINDOW", "5") or "5")
        search_tool = os.getenv("GRAPHITI_MCP_SEARCH_TOOL", "graphiti.search_memories")
        upsert_tool = os.getenv("GRAPHITI_MCP_UPSERT_TOOL", "graphiti.upsert_memory")
        namespace = os.getenv("GRAPHITI_MEMORY_NAMESPACE", "agentic-rag")
        metadata_tags = _parse_list_env(os.getenv("GRAPHITI_MEMORY_TAGS"))
        search_limit = int(os.getenv("GRAPHITI_MEMORY_SEARCH_LIMIT", "5") or "5")

        enabled_flag = os.getenv("GRAPHITI_MCP_ENABLED")
        enabled = (
            bool(transport and (command or url))
            and MultiServerMCPClient is not None
            and (
                enabled_flag is None
                or enabled_flag.lower() not in {"0", "false", "off"}
            )
        )

        return cls(
            transport=transport,
            command=command,
            args=args,
            url=url,
            headers=headers,
            env=proc_env,
            short_term_window=short_term_window,
            search_tool=search_tool,
            upsert_tool=upsert_tool,
            namespace=namespace,
            metadata_tags=metadata_tags,
            search_limit=search_limit,
            enabled=enabled,
        )


@dataclass(slots=True)
class MemorySnapshot:
    """Return payload when fetching memory context."""

    short_term: List[str]
    long_term: List[str]
    raw_result: Dict[str, Any] = field(default_factory=dict)
    from_graphiti: bool = field(default=False)
    error: str | None = field(default=None)

    @property
    def has_long_term(self) -> bool:
        return bool(self.long_term)


class GraphitiMemoryManager:
    """Handle short-term buffering and Graphiti-backed long-term memory."""

    def __init__(self, settings: GraphitiMemorySettings | None = None) -> None:
        self.settings = settings or GraphitiMemorySettings.from_env()
        self._short_term = deque(maxlen=self.settings.short_term_window)
        self._client: MultiServerMCPClient | None = None
        self._client_lock = asyncio.Lock()
        self._tools_cache: Dict[str, Any] | None = None

    @property
    def is_enabled(self) -> bool:
        return bool(self.settings.enabled)

    def get_short_term(self) -> List[str]:
        return list(self._short_term)

    def _namespace(self, session_id: str | None = None) -> Sequence[str]:
        if session_id:
            return (self.settings.namespace, session_id)
        return (self.settings.namespace, "default")

    async def _ensure_client(self) -> MultiServerMCPClient:
        if not self.is_enabled:
            raise RuntimeError("Graphiti MCP is not enabled.")

        async with self._client_lock:
            if self._client is None:
                server_config: Dict[str, Any] = {
                    "transport": self.settings.transport,
                }

                if self.settings.transport == "stdio":
                    server_config.update(
                        {
                            "command": self.settings.command,
                            "args": self.settings.args,
                        }
                    )
                    if self.settings.env:
                        server_config["env"] = self.settings.env
                else:
                    if self.settings.url:
                        server_config["url"] = self.settings.url
                    if self.settings.headers:
                        server_config["headers"] = self.settings.headers

                self._client = MultiServerMCPClient(
                    {
                        self.settings.server_name: server_config,
                    }
                )

            return self._client

    async def _call_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> CallToolResult | None:
        client = await self._ensure_client()
        async with client.session(self.settings.server_name) as session:
            result = await session.call_tool(tool_name, arguments)
            return result

    def fetch_context(
        self,
        *,
        query: str,
        session_id: str | None = None,
        limit: int | None = None,
    ) -> MemorySnapshot:
        if not self.is_enabled or not self.settings.search_tool:
            return MemorySnapshot(
                short_term=self.get_short_term(),
                long_term=[],
                raw_result={},
                from_graphiti=False,
            )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.afetch_context(query=query, session_id=session_id, limit=limit)
            )
        else:  # pragma: no cover - not expected in sync graph execution
            return loop.run_until_complete(
                self.afetch_context(query=query, session_id=session_id, limit=limit)
            )

    async def afetch_context(
        self,
        *,
        query: str,
        session_id: str | None = None,
        limit: int | None = None,
    ) -> MemorySnapshot:
        if not self.is_enabled or not self.settings.search_tool:
            return MemorySnapshot(
                short_term=self.get_short_term(),
                long_term=[],
                raw_result={},
                from_graphiti=False,
            )

        try:
            result = await self._call_tool(
                self.settings.search_tool,
                {
                    "namespace": list(self._namespace(session_id)),
                    "query": query,
                    "limit": limit or self.settings.search_limit,
                },
            )
        except Exception as exc:  # pragma: no cover - network except path
            return MemorySnapshot(
                short_term=self.get_short_term(),
                long_term=[],
                raw_result={},
                from_graphiti=False,
                error=str(exc),
            )

        memories: List[str] = []
        raw_payload: Dict[str, Any] = {}

        if result is not None:
            raw_payload = {
                "content": getattr(result, "content", None),
                "diagnostics": getattr(result, "diagnostics", None),
            }

            content_items = getattr(result, "content", None)
            if isinstance(content_items, list):
                for item in content_items:
                    text = getattr(item, "text", None)
                    if text:
                        memories.append(text)
                    elif isinstance(item, dict):
                        txt = item.get("text")
                        if txt:
                            memories.append(str(txt))

        return MemorySnapshot(
            short_term=self.get_short_term(),
            long_term=memories,
            raw_result=raw_payload,
            from_graphiti=bool(memories),
        )

    def store_interaction(
        self,
        *,
        question: str,
        answer: str,
        metadata: Dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> None:
        """Persist the latest interaction in both buffers."""
        metadata = metadata or {}
        summary = self._summarize_interaction(question, answer)
        self._short_term.append(summary)

        if not self.is_enabled or not self.settings.upsert_tool:
            return

        payload = {
            "namespace": list(self._namespace(session_id)),
            "memory": {
                "summary": summary,
                "question": question,
                "answer": answer,
                "metadata": metadata,
                "tags": self.settings.metadata_tags,
            },
        }

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._astore_payload(payload))
        else:  # pragma: no cover - not expected in sync graph execution
            loop.run_until_complete(self._astore_payload(payload))

    async def _astore_payload(self, payload: Dict[str, Any]) -> None:
        try:
            await self._call_tool(self.settings.upsert_tool, payload)
        except Exception:  # pragma: no cover - network except path
            # Swallow errors to keep the agent running.
            return

    @staticmethod
    def _summarize_interaction(question: str, answer: str) -> str:
        q = question.strip().replace("\n", " ")
        a = answer.strip().replace("\n", " ")
        if len(q) > 300:
            q = q[:297] + "..."
        if len(a) > 500:
            a = a[:497] + "..."
        return f"Q: {q}\nA: {a}"


_GLOBAL_MANAGER: GraphitiMemoryManager | None = None


def get_memory_manager() -> GraphitiMemoryManager:
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is None:
        _GLOBAL_MANAGER = GraphitiMemoryManager()
    return _GLOBAL_MANAGER
