# -*- coding: utf-8 -*-
"""SQLite-backed short-term memory store with summarization tiers."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, List, Tuple

DEFAULT_DB_FILENAME = "conversation_memory.sqlite3"


def _default_db_path() -> Path:
    """Determine the SQLite file path for short-term memory persistence."""
    custom_path = os.getenv("SHORT_TERM_MEMORY_DB")
    if custom_path:
        return Path(custom_path).expanduser()

    project_root = Path(os.getenv("PROJECT_ROOT", Path.cwd()))
    storage_dir = project_root / ".cache"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir / DEFAULT_DB_FILENAME


def _normalize_text(value: str, *, limit: int) -> str:
    """Trim whitespace and limit text length for summaries."""
    cleaned = re.sub(r"\s+", " ", value.strip())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _derive_topic(question: str) -> str:
    """Extract a coarse topic label from the question."""
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", question.lower())
    if not tokens:
        return "기타"
    return " ".join(tokens[:3])


@dataclass(frozen=True)
class ShortTermContext:
    """Structured payload returned when retrieving short-term memory."""

    recent: List[str]
    history_summary: str | None
    topic_summaries: List[str]


class ShortTermMemoryStore:
    """Persist conversation turns and build tiered summaries."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else _default_db_path()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            self._db_path,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._lock = Lock()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    session_id TEXT NOT NULL,
                    turn INTEGER NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, turn)
                )
                """
            )

    @property
    def db_path(self) -> Path:
        """Return the SQLite database path."""
        return self._db_path

    def record_interaction(
        self,
        *,
        session_id: str,
        question: str,
        answer: str,
        metadata: Dict[str, object] | None = None,
    ) -> int:
        """Persist a new conversation turn and return the turn index."""
        safe_metadata = json.dumps(metadata or {}, ensure_ascii=False)
        timestamp = datetime.now(timezone.utc).isoformat()

        with self._lock:
            cursor = self._connection.execute(
                "SELECT COALESCE(MAX(turn), 0) FROM interactions WHERE session_id = ?",
                (session_id,),
            )
            next_turn = (cursor.fetchone()[0] or 0) + 1
            self._connection.execute(
                """
                INSERT INTO interactions(session_id, turn, question, answer, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, next_turn, question, answer, safe_metadata, timestamp),
            )
        return next_turn

    def get_context(self, session_id: str) -> ShortTermContext:
        """Assemble the tiered memory context for a given session."""
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT turn, question, answer
                FROM interactions
                WHERE session_id = ?
                ORDER BY turn
                """,
                (session_id,),
            ).fetchall()

        if not rows:
            return ShortTermContext(recent=[], history_summary=None, topic_summaries=[])

        recent_segments = [
            self._format_recent(row["turn"], row["question"], row["answer"])
            for row in rows[-3:]
        ]

        history_rows = [row for row in rows if 4 <= row["turn"] <= 9]
        history_summary = (
            self._summarize_history(history_rows) if history_rows else None
        )

        topic_rows = [row for row in rows if row["turn"] >= 10]
        topic_summaries = self._summarize_topics(topic_rows) if topic_rows else []

        return ShortTermContext(
            recent=recent_segments,
            history_summary=history_summary,
            topic_summaries=topic_summaries,
        )

    def clear_session(self, session_id: str) -> None:
        """Remove all interactions for a session."""
        with self._lock:
            self._connection.execute(
                "DELETE FROM interactions WHERE session_id = ?", (session_id,)
            )

    @staticmethod
    def _format_recent(turn: int, question: str, answer: str) -> str:
        return (
            f"[최근 {turn}턴]\n"
            f"질문: {_normalize_text(question, limit=280)}\n"
            f"답변: {_normalize_text(answer, limit=420)}"
        )

    @staticmethod
    def _summarize_history(rows: List[sqlite3.Row]) -> str:
        parts = []
        for row in rows:
            question = _normalize_text(row["question"], limit=160)
            answer = _normalize_text(row["answer"], limit=220)
            parts.append(f"{row['turn']}턴 ▶ {question} → {answer}")
        header = f"히스토리 요약 (4~{rows[-1]['turn']}턴)"
        body = "\n".join(parts)
        return f"{header}\n{body}"

    @staticmethod
    def _summarize_topics(rows: List[sqlite3.Row]) -> List[str]:
        grouped: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)
        for row in rows:
            topic = _derive_topic(row["question"])
            grouped[topic].append((row["turn"], row["question"], row["answer"]))

        summaries: List[str] = []
        for topic, items in grouped.items():
            turns = ", ".join(str(turn) for turn, _, _ in items[:5])
            sample_answers = "; ".join(
                _normalize_text(answer, limit=160) for _, _, answer in items[:2]
            )
            summaries.append(
                f"주제 요약 - '{topic}' (총 {len(items)}회, 예시 턴: {turns})\n"
                f"핵심 내용: {sample_answers}"
            )
        return summaries


_GLOBAL_STORE: ShortTermMemoryStore | None = None


def get_short_term_store() -> ShortTermMemoryStore:
    """Return a process-wide short-term memory store."""
    global _GLOBAL_STORE
    if _GLOBAL_STORE is None:
        _GLOBAL_STORE = ShortTermMemoryStore()
    return _GLOBAL_STORE
