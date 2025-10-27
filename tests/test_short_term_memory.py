# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

from src.memory.short_term import ShortTermMemoryStore


def test_short_term_memory_recent_and_history(tmp_path: Path) -> None:
    store = ShortTermMemoryStore(tmp_path / "memory.sqlite3")

    for idx in range(1, 7):
        store.record_interaction(
            session_id="session-a",
            question=f"대화 {idx}번 질문입니다?",
            answer=f"대화 {idx}번에 대한 상세한 답변입니다.",
            metadata={"turn": idx},
        )

    context = store.get_context("session-a")

    assert len(context.recent) == 3
    assert "최근 6턴" in context.recent[-1]
    assert context.history_summary is not None
    assert "4~6턴" in context.history_summary
    assert context.topic_summaries == []


def test_short_term_memory_topic_summary(tmp_path: Path) -> None:
    store = ShortTermMemoryStore(tmp_path / "memory.sqlite3")

    for idx in range(1, 12):
        store.record_interaction(
            session_id="session-b",
            question=f"식단 관리 방법 {idx}가 궁금해요",
            answer=f"식단 관리 방법 {idx}에 대한 답변",
        )

    context = store.get_context("session-b")
    assert len(context.recent) == 3
    assert context.history_summary is not None
    assert context.topic_summaries, "10턴 이후 대화는 주제 요약이 생성되어야 합니다."
    assert any("식단" in summary for summary in context.topic_summaries)

