# -*- coding: utf-8 -*-
"""Graphiti MCP integration helpers and LangChain tool wrappers."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from langchain_core.tools import BaseTool, StructuredTool
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - optional dependency path
    BaseTool = None  # type: ignore[assignment]
    StructuredTool = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
except ImportError:  # pragma: no cover - optional dependency path
    MultiServerMCPClient = None  # type: ignore[assignment]

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
            search_tool=search_tool,
            upsert_tool=upsert_tool,
            namespace=namespace,
            metadata_tags=metadata_tags,
            search_limit=search_limit,
            enabled=enabled,
        )


class GraphitiMCPConnector:
    """Thin async wrapper over the Graphiti MCP server."""

    def __init__(self, settings: GraphitiMemorySettings | None = None) -> None:
        self.settings = settings or GraphitiMemorySettings.from_env()
        self._client: MultiServerMCPClient | None = None
        self._client_lock = asyncio.Lock()

    @property
    def is_enabled(self) -> bool:
        return bool(self.settings.enabled and MultiServerMCPClient is not None)

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
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> CallToolResult | None:
        client = await self._ensure_client()
        async with client.session(self.settings.server_name) as session:
            return await session.call_tool(tool_name, arguments)

    def _namespace(self, session_id: str | None) -> List[str]:
        suffix = session_id or "default"
        return [self.settings.namespace, suffix]

    async def search_memories(
        self,
        *,
        session_id: str | None,
        query: str,
        limit: int | None = None,
    ) -> Dict[str, Any]:
        if not self.is_enabled or not self.settings.search_tool:
            return {"memories": [], "diagnostics": None}

        try:
            result = await self._call_tool(
                self.settings.search_tool,
                {
                    "namespace": self._namespace(session_id),
                    "query": query,
                    "limit": limit or self.settings.search_limit,
                },
            )
        except Exception as exc:  # pragma: no cover - network except path
            return {"memories": [], "diagnostics": {"error": str(exc)}}

        texts: List[str] = []
        diagnostics: Dict[str, Any] | None = None
        if result is not None:
            diagnostics = getattr(result, "diagnostics", None)
            for content in getattr(result, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    texts.append(text)
                elif isinstance(content, dict) and content.get("text"):
                    texts.append(str(content["text"]))

        return {"memories": texts, "diagnostics": diagnostics}

    async def upsert_memory(
        self,
        *,
        session_id: str | None,
        question: str,
        answer: str,
        summary: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> bool:
        if not self.is_enabled or not self.settings.upsert_tool:
            return False

        payload = {
            "namespace": self._namespace(session_id),
            "memory": {
                "summary": summary
                or f"Q: {question.strip()}\nA: {answer.strip()}",
                "question": question,
                "answer": answer,
                "metadata": metadata or {},
                "tags": self.settings.metadata_tags,
            },
        }

        try:
            await self._call_tool(self.settings.upsert_tool, payload)
        except Exception:  # pragma: no cover - network except path
            return False
        return True

    def build_tools(self, session_id: str) -> List[BaseTool]:
        """Create LangChain tools that proxy Graphiti MCP operations."""
        if not self.is_enabled or StructuredTool is None:
            return []

        connector = self
        resolved_session = session_id

        class GraphitiSearchInput(BaseModel):  # type: ignore[misc,valid-type]
            query: str = Field(..., description="사용자 요청과 관련된 검색 질의")
            limit: Optional[int] = Field(
                default=None, description="검색 결과 개수 (기본값 5)"
            )

        class GraphitiUpsertInput(BaseModel):  # type: ignore[misc,valid-type]
            question: str = Field(..., description="사용자 질문 원문")
            answer: str = Field(..., description="AI가 제공한 답변 원문")
            summary: Optional[str] = Field(
                default=None, description="질문-답변에 대한 요약 (없으면 자동 생성)"
            )
            metadata: Optional[Dict[str, Any]] = Field(
                default=None, description="추가 메타데이터 (JSON 객체)"
            )

        async def search_tool(**kwargs: Any) -> str:
            result = await connector.search_memories(
                session_id=resolved_session,
                query=kwargs["query"],
                limit=kwargs.get("limit"),
            )
            payload = result.get("memories", [])
            if not payload:
                return "관련된 장기 기억을 찾지 못했습니다."
            return "\n\n".join(str(item) for item in payload)

        async def upsert_tool(**kwargs: Any) -> str:
            success = await connector.upsert_memory(
                session_id=resolved_session,
                question=kwargs["question"],
                answer=kwargs["answer"],
                summary=kwargs.get("summary"),
                metadata=kwargs.get("metadata"),
            )
            return (
                "장기 기억에 대화를 기록했습니다." if success else "장기 기억 저장에 실패했습니다."
            )

        search = StructuredTool(
            name="graphiti_search_memories",
            description=(
                "Graphiti MCP 그래프에서 사용자의 상담 기록과 관련된 장기 기억을 검색합니다."
            ),
            args_schema=GraphitiSearchInput,
            coroutine=search_tool,
        )
        upsert = StructuredTool(
            name="graphiti_upsert_memory",
            description=(
                "현재 상담 내용을 Graphiti MCP 장기 기억에 저장하거나 갱신합니다. "
                "최종 답변을 전달하기 전에 호출하여 로그를 남기세요."
            ),
            args_schema=GraphitiUpsertInput,
            coroutine=upsert_tool,
        )

        return [search, upsert]


_GLOBAL_CONNECTOR: GraphitiMCPConnector | None = None


def get_graphiti_connector() -> GraphitiMCPConnector:
    """Return a process-wide Graphiti MCP connector."""
    global _GLOBAL_CONNECTOR
    if _GLOBAL_CONNECTOR is None:
        _GLOBAL_CONNECTOR = GraphitiMCPConnector()
    return _GLOBAL_CONNECTOR

