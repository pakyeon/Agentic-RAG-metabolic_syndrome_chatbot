# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
from typing import List


def _reset_manager() -> None:
    from src.memory import graphiti

    graphiti._GLOBAL_MANAGER = None  # type: ignore[attr-defined]
    importlib.reload(graphiti)


def test_memory_manager_disabled_by_default(monkeypatch) -> None:
    """Without explicit Graphiti config the manager should disable safely."""
    for key in [
        "GRAPHITI_MCP_TRANSPORT",
        "GRAPHITI_MCP_COMMAND",
        "GRAPHITI_MCP_URL",
        "GRAPHITI_MCP_ENABLED",
    ]:
        monkeypatch.delenv(key, raising=False)

    _reset_manager()

    from src.memory import get_memory_manager

    manager = get_memory_manager()
    assert manager.is_enabled is False

    snapshot = manager.fetch_context(query="baseline question")
    assert snapshot.short_term == []
    assert snapshot.long_term == []
    assert snapshot.from_graphiti is False


def test_memory_manager_short_term_buffer(monkeypatch) -> None:
    monkeypatch.setenv("GRAPHITI_SHORT_TERM_WINDOW", "2")
    for key in [
        "GRAPHITI_MCP_TRANSPORT",
        "GRAPHITI_MCP_COMMAND",
        "GRAPHITI_MCP_URL",
        "GRAPHITI_MCP_ENABLED",
    ]:
        monkeypatch.delenv(key, raising=False)

    _reset_manager()

    from src.memory import get_memory_manager

    manager = get_memory_manager()

    interactions: List[tuple[str, str]] = [
        ("첫 번째 질문", "첫 번째 답변"),
        ("두 번째 질문", "두 번째 답변"),
        ("세 번째 질문", "세 번째 답변"),
    ]

    for question, answer in interactions:
        manager.store_interaction(question=question, answer=answer)

    short_term = manager.get_short_term()
    assert len(short_term) == 2
    assert "두 번째 질문" in short_term[0]
    assert "세 번째 질문" in short_term[1]
