# -*- coding: utf-8 -*-
"""그래프 상태 정의 테스트"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.graph.state import RAGState
from langchain_core.documents import Document


def test_state_creation():
    """RAGState 생성 테스트"""
    state: RAGState = {
        "question": "대사증후군이란 무엇인가요?",
        "patient_id": "P001",
        "patient_context": None,
        "should_retrieve": True,
        "relevance_scores": [4.0, 3.5, 4.5],
        "support_score": 4.0,
        "usefulness_score": 4.5,
        "crag_action": "CORRECT",
        "crag_confidence": 0.85,
        "internal_docs": [
            Document(page_content="대사증후군은...", metadata={"source": "doc1"})
        ],
        "external_docs": [],
        "merged_context": "대사증후군 관련 정보...",
        "answer": "대사증후군은 복부비만, 고혈압...",
        "iteration": 1,
        "max_iterations": 2,
        "needs_regeneration": False,
        "error": None,
        "metadata": {"timestamp": "2025-10-20"},
    }

    assert state["question"] == "대사증후군이란 무엇인가요?"
    assert state["should_retrieve"] is True
    assert len(state["relevance_scores"]) == 3
    assert state["crag_action"] in ["CORRECT", "INCORRECT", "AMBIGUOUS"]

    print("✅ RAGState 생성 테스트 통과")


def test_state_fields():
    """필수 필드 존재 확인"""
    required_fields = [
        "question",
        "patient_id",
        "patient_context",
        "should_retrieve",
        "relevance_scores",
        "support_score",
        "usefulness_score",
        "crag_action",
        "crag_confidence",
        "internal_docs",
        "external_docs",
        "merged_context",
        "answer",
        "iteration",
        "max_iterations",
        "needs_regeneration",
        "error",
        "metadata",
    ]

    state_annotations = RAGState.__annotations__

    for field in required_fields:
        assert field in state_annotations, f"필드 누락: {field}"

    print(f"✅ 필수 필드 {len(required_fields)}개 모두 존재")


if __name__ == "__main__":
    test_state_creation()
    test_state_fields()
    print("\n✅ 모든 테스트 통과!")
