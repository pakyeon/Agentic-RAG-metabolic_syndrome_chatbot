# -*- coding: utf-8 -*-
"""그래프 상태 정의 테스트 (AgenticRAGState)"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.graph.state import AgenticRAGState
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


def test_state_creation():
    """AgenticRAGState 생성 테스트"""
    state: AgenticRAGState = {
        "messages": [
            HumanMessage(content="대사증후군이란 무엇인가요?"),
            SystemMessage(content="[검색 컨텍스트]\n대사증후군 관련 정보..."),
            AIMessage(content="대사증후군은 복부비만, 고혈압..."),
        ],
        "patient_id": 1,
        "patient_context": None,
        "short_term_memory": [],
        "long_term_memory": [],
        "memory_session_id": "test-session",
        "internal_docs": [
            Document(page_content="대사증후군은...", metadata={"source": "doc1"})
        ],
        "external_docs": [],
        "iteration": 1,
        "max_iterations": 2,
        "metadata": {
            "should_retrieve": True,
            "relevance_scores": [4.0, 3.5, 4.5],
            "support_score": 4.0,
            "usefulness_score": 4.5,
            "crag_action": "correct",
            "crag_confidence": 0.85,
            "timestamp": "2025-10-20",
        },
    }

    assert len(state["messages"]) == 3
    assert state["messages"][0].type == "human"
    assert state["messages"][1].type == "system"
    assert state["messages"][2].type == "ai"
    assert state["iteration"] == 1
    assert state["metadata"]["crag_action"] == "correct"

    print("✅ AgenticRAGState 생성 테스트 통과")


def test_state_fields():
    """필수 필드 존재 확인"""
    required_fields = [
        "messages",
        "patient_id",
        "patient_context",
        "short_term_memory",
        "long_term_memory",
        "memory_session_id",
        "internal_docs",
        "external_docs",
        "iteration",
        "max_iterations",
        "metadata",
    ]

    state_annotations = AgenticRAGState.__annotations__

    for field in required_fields:
        assert field in state_annotations, f"필드 누락: {field}"

    print(f"✅ 필수 필드 {len(required_fields)}개 모두 존재")


def test_message_adding():
    """메시지 추가 테스트"""
    from langgraph.graph.message import add_messages

    # 초기 메시지
    messages1 = [HumanMessage(content="안녕하세요")]

    # 새로운 메시지 추가
    messages2 = [AIMessage(content="안녕하세요! 무엇을 도와드릴까요?")]

    # add_messages로 병합
    merged = add_messages(messages1, messages2)

    assert len(merged) == 2
    assert merged[0].content == "안녕하세요"
    assert merged[1].content == "안녕하세요! 무엇을 도와드릴까요?"

    print("✅ 메시지 추가 테스트 통과")


if __name__ == "__main__":
    test_state_creation()
    test_state_fields()
    test_message_adding()
    print("\n✅ 모든 테스트 통과!")
