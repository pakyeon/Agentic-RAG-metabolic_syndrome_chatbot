import os
import sys
import types

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.self_rag_evaluator import (  # noqa: E402
    SelfRAGEvaluator,
    DocumentEvaluationWithAction,
    CombinedEvaluationResult,
    AnswerQualityResult,
    SupportResult,
    RelevanceResult,
    BatchResultParseError,
)


def _fake_response(payload: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(content=payload)


def test_evaluate_retrieval_and_decide_action(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    evaluator = SelfRAGEvaluator()

    def fake_invoke(prompt: str):
        return _fake_response(
            """{
              "document_evaluations": [
                {"doc_id": 1, "score": 4, "relevance": "relevant", "confidence": 0.8, "reason": "Matches diagnostic criteria"},
                {"doc_id": 2, "score": 2, "relevance": "irrelevant", "confidence": 0.6, "reason": "Different topic"}
              ],
              "crag_action": "CORRECT",
              "reason": "Two relevant documents"
            }"""
        )

    evaluator.llm = types.SimpleNamespace(invoke=fake_invoke)

    documents = ["문서1", "문서2"]
    result = evaluator.evaluate_retrieval_and_decide_action(
        "질문", documents, min_relevant_docs=1
    )

    assert isinstance(result, CombinedEvaluationResult)
    assert result.crag_action == "correct"
    assert result.reason == "Two relevant documents"
    assert result.relevant_count == 1
    assert len(result.document_evaluations) == 2
    assert result.document_evaluations[0].doc_id == 1
    assert result.document_evaluations[0].relevance == "relevant"


def test_evaluate_retrieval_and_decide_action_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    evaluator = SelfRAGEvaluator()

    def fake_invoke(prompt: str):
        raise ValueError("LLM error")

    evaluator.llm = types.SimpleNamespace(invoke=fake_invoke)

    documents = ["문서1", "문서2"]

    with pytest.raises(BatchResultParseError):
        evaluator.evaluate_retrieval_and_decide_action(
            "질문", documents, min_relevant_docs=1
        )


def test_evaluate_answer_quality(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    evaluator = SelfRAGEvaluator()

    def fake_invoke(prompt: str):
        return _fake_response(
            """{
              "support": [
                {"doc_id": 1, "support": "fully_supported", "confidence": 0.9},
                {"doc_id": 2, "support": "no_support", "confidence": 0.4}
              ],
              "usefulness_score": 4,
              "usefulness_confidence": 0.85,
              "should_regenerate": false,
              "regenerate_reason": "충분한 품질"
            }"""
        )

    evaluator.llm = types.SimpleNamespace(invoke=fake_invoke)

    result = evaluator.evaluate_answer_quality(
        "질문",
        "답변",
        ["문서1", "문서2"],
    )

    assert isinstance(result, AnswerQualityResult)
    assert result.usefulness_score == 4
    assert result.usefulness_confidence == 0.85
    assert result.should_regenerate is False
    assert isinstance(result.support_results[0], SupportResult)
