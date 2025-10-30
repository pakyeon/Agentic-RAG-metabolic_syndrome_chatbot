# -*- coding: utf-8 -*-
"""Unit tests for Graphiti long-term memory integration hooks."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.graph.nodes import evaluate_answer_node
from src.memory.short_term import ShortTermContext


class _FakeStore:
    def __init__(self) -> None:
        self.records = []

    def record_interaction(self, **kwargs):
        self.records.append(kwargs)
        return 1

    def get_context(self, session_id: str) -> ShortTermContext:  # pragma: no cover - simple path
        return ShortTermContext(recent=[], history_summary=None, topic_summaries=[])


class GraphitiIntegrationTests(unittest.TestCase):
    """Verify graph nodes cooperate with the Graphiti MCP connector."""

    def _build_state(self) -> dict:
        return {
            "question": "대사증후군 환자 상담 요약?",
            "answer": "생활습관 교정과 약물 요법을 병행해야 합니다.",
            "merged_context": "[내부 문서 1]\n대사증후군 관리 지침",
            "iteration": 1,
            "max_iterations": 2,
            "memory_session_id": "session-123",
            "metadata": {},
            "internal_docs": [object()],
            "external_docs": [],
        }

    @patch("src.graph.nodes.get_graphiti_connector")
    @patch("src.graph.nodes.get_short_term_store")
    @patch("src.evaluation.self_rag_evaluator.create_evaluator")
    def test_evaluate_answer_upserts_on_success(
        self,
        mock_create_evaluator: MagicMock,
        mock_get_store: MagicMock,
        mock_get_connector: MagicMock,
    ) -> None:
        """Successful answers trigger Graphiti upsert with metadata."""

        fake_quality = SimpleNamespace(
            support_results=[SimpleNamespace(support="fully_supported")],
            usefulness_score=4.2,
            usefulness_confidence=0.9,
            should_regenerate=False,
            regenerate_reason="",
        )

        evaluator = MagicMock()
        evaluator.assess_answer_quality.return_value = fake_quality
        mock_create_evaluator.return_value = evaluator

        store = _FakeStore()
        mock_get_store.return_value = store

        connector = MagicMock()
        connector.is_enabled = True
        connector.upsert_memory_sync.return_value = True
        mock_get_connector.return_value = connector

        result = evaluate_answer_node(self._build_state())

        connector.upsert_memory_sync.assert_called_once()
        self.assertTrue(result["metadata"]["graphiti"]["last_upsert"]["saved"])
        self.assertNotIn("last_upsert_error", result["metadata"]["graphiti"])
        self.assertEqual(len(store.records), 1)

    @patch("src.graph.nodes.get_graphiti_connector")
    @patch("src.graph.nodes.get_short_term_store")
    @patch("src.evaluation.self_rag_evaluator.create_evaluator")
    def test_evaluate_answer_skips_when_quality_low(
        self,
        mock_create_evaluator: MagicMock,
        mock_get_store: MagicMock,
        mock_get_connector: MagicMock,
    ) -> None:
        """Answers below quality threshold skip Graphiti upsert."""

        fake_quality = SimpleNamespace(
            support_results=[SimpleNamespace(support="not_supported")],
            usefulness_score=2.1,
            usefulness_confidence=0.6,
            should_regenerate=False,
            regenerate_reason="",
        )

        evaluator = MagicMock()
        evaluator.assess_answer_quality.return_value = fake_quality
        mock_create_evaluator.return_value = evaluator

        store = _FakeStore()
        mock_get_store.return_value = store

        connector = MagicMock()
        connector.is_enabled = True
        mock_get_connector.return_value = connector

        result = evaluate_answer_node(self._build_state())

        connector.upsert_memory_sync.assert_not_called()
        self.assertEqual(
            result["metadata"]["graphiti"].get("last_upsert_reason"),
            "quality_below_threshold",
        )


if __name__ == "__main__":  # pragma: no cover - manual run helper
    unittest.main()
