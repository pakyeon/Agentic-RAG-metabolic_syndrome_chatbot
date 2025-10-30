import os
import sys
import types

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.self_rag_evaluator import (  # noqa: E402
    RelevanceResult,
    SelfRAGEvaluator,
    SupportResult,
    DocumentEvaluationWithAction,
    CombinedEvaluationResult,
    BatchResultParseError,
)
from src.graph import nodes  # noqa: E402
from langchain_core.documents import Document  # noqa: E402


def _fake_response(payload: str) -> types.SimpleNamespace:
    """간단한 LLM 응답 객체."""
    return types.SimpleNamespace(content=payload)


def test_evaluate_relevance_batch_parses_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    evaluator = SelfRAGEvaluator()

    captured_prompt = {}

    def fake_invoke(prompt: str):
        captured_prompt["value"] = prompt
        return _fake_response(
            (
                '[{"doc_id": 1, "score": 5, "reason": "Direct match"},'
                ' {"doc_id": 2, "score": 2, "reason": "Off topic"}]'
            )
        )

    evaluator.llm = types.SimpleNamespace(invoke=fake_invoke)

    docs = ["첫 번째 문서", "두 번째 문서"]
    results = evaluator.evaluate_relevance_batch("대사증후군 진단?", docs)

    assert len(results) == 2
    assert isinstance(results[0], RelevanceResult)
    assert results[0].relevance == "relevant"
    assert results[1].relevance == "irrelevant"
    # 프롬프트가 문서 목록을 번호로 나열했는지 확인
    assert "1. 첫 번째 문서" in captured_prompt["value"]
    assert "Do not output anything other than valid JSON." in captured_prompt["value"]


def test_evaluate_relevance_batch_raises_on_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    evaluator = SelfRAGEvaluator()

    evaluator.llm = types.SimpleNamespace(invoke=lambda prompt: _fake_response("oops"))

    with pytest.raises(BatchResultParseError):
        evaluator.evaluate_relevance_batch("질문", ["문서 A", "문서 B"])


def test_evaluate_support_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    evaluator = SelfRAGEvaluator()

    def fake_invoke(prompt: str):
        return _fake_response(
            (
                '[{"doc_id": 1, "support": "fully_supported", "confidence": 0.9},'
                ' {"doc_id": 2, "support": "no_support", "confidence": 0.4}]'
            )
        )

    evaluator.llm = types.SimpleNamespace(invoke=fake_invoke)

    docs = ["문서 1", "문서 2"]
    results = evaluator.evaluate_support_batch("질문", docs, "답변")

    assert len(results) == 2
    assert isinstance(results[0], SupportResult)
    assert results[0].support == "fully_supported"
    assert results[1].support == "no_support"
    assert 0.0 <= results[1].confidence <= 1.0


def test_evaluate_documents_with_early_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    evaluator = SelfRAGEvaluator()

    call_sizes = []

    def fake_batch(query: str, documents: list) -> list:
        call_sizes.append(len(documents))
        return [
            RelevanceResult(relevance="relevant", confidence=0.9)
            if idx == 0
            else RelevanceResult(relevance="irrelevant", confidence=0.6)
            for idx in range(len(documents))
        ]

    monkeypatch.setattr(evaluator, "evaluate_relevance_batch", fake_batch)

    docs = ["문서 1", "문서 2", "문서 3"]
    results, stopped, count = evaluator.evaluate_documents_with_early_stop(
        "질문", docs, min_relevant_docs=1, enable_early_stop=True
    )

    assert stopped is True
    assert count == 1
    assert len(results) == 1
    assert call_sizes == [1]


def test_evaluate_documents_with_early_stop_no_trigger(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    evaluator = SelfRAGEvaluator()

    def fake_batch(query: str, documents: list) -> list:
        return [
            RelevanceResult(relevance="irrelevant", confidence=0.4)
            for _ in documents
        ]

    monkeypatch.setattr(evaluator, "evaluate_relevance_batch", fake_batch)

    docs = ["문서 1", "문서 2", "문서 3"]
    results, stopped, count = evaluator.evaluate_documents_with_early_stop(
        "질문",
        docs,
        min_relevant_docs=2,
        enable_early_stop=True,
        max_documents=2,
    )

    assert stopped is False
    assert count == 2
    assert len(results) == 2


def test_evaluate_retrieval_node_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeEvaluator:
        def evaluate_documents_with_early_stop(
            self, query, documents, **kwargs
        ):
            return (
                [RelevanceResult(relevance="relevant", confidence=0.95)],
                True,
                1,
            )

        def evaluate_relevance_batch(self, query, documents):
            return [
                RelevanceResult(relevance="relevant", confidence=0.95)
                for _ in documents
            ]

        def evaluate_retrieval_and_decide_action(
            self, query, documents, min_relevant_docs=1
        ):
            evaluations = []
            for idx, doc in enumerate(documents, start=1):
                evaluations.append(
                    DocumentEvaluationWithAction(
                        doc_id=idx,
                        document_content=getattr(doc, "page_content", str(doc)),
                        score=4.0,
                        relevance="relevant",
                        confidence=0.95,
                        reason="stub",
                    )
                )
            return CombinedEvaluationResult(
                document_evaluations=evaluations,
                crag_action="correct",
                reason="stub reason",
                min_relevant_docs=min_relevant_docs,
                relevant_count=len(evaluations),
            )

    monkeypatch.setattr(
        "src.evaluation.self_rag_evaluator.SelfRAGEvaluator",
        lambda *args, **kwargs: FakeEvaluator(),
    )

    state = {
        "question": "대사증후군 진단 기준?",
        "internal_docs": [Document(page_content="doc1")],
        "metadata": {},
        "min_relevant_docs": 1,
        "early_stop_enabled": True,
    }

    result = nodes.evaluate_retrieval_node(state)

    assert result["early_stopped"] is True
    assert result["evaluated_docs_count"] == 1
    assert result["total_evaluated_docs"] == 1
    assert len(result["relevance_scores"]) == 1
    assert len(result["document_evaluations"]) == 1
    assert result["document_evaluations"][0]["doc_id"] == 1
    meta = result["metadata"]
    assert meta["early_stop_enabled"] is True
    assert meta["early_stopped"] is True
