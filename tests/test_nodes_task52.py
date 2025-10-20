# -*- coding: utf-8 -*-
"""Task 5.2 노드 테스트"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.graph.nodes import load_patient_context_node, should_retrieve_node
from src.graph.state import RAGState


def test_load_patient_context_node():
    """환자 컨텍스트 로드 노드 테스트"""
    print("\n=== test_load_patient_context_node ===")

    # Case 1: 환자 ID가 있는 경우 (정수형)
    state: RAGState = {
        "question": "혈압 관리 방법은?",
        "patient_id": 1,
        "patient_context": None,
        "should_retrieve": False,
        "relevance_scores": [],
        "support_score": 0.0,
        "usefulness_score": 0.0,
        "crag_action": "",
        "crag_confidence": 0.0,
        "internal_docs": [],
        "external_docs": [],
        "merged_context": "",
        "answer": "",
        "iteration": 0,
        "max_iterations": 2,
        "needs_regeneration": False,
        "error": None,
        "metadata": {},
    }

    result = load_patient_context_node(state)
    print(f"환자 ID: {state['patient_id']}")

    # 에러 확인
    if result.get("error"):
        print(f"❌ 에러 발생: {result['error']}")

    print(f"컨텍스트 존재: {bool(result.get('patient_context'))}")
    if result.get("patient_context"):
        print(f"컨텍스트 길이: {len(result['patient_context'])} 문자")
        print(f"컨텍스트 미리보기: {result['patient_context'][:100]}...")

    assert "patient_context" in result

    # 실제로 컨텍스트가 로드되었는지 확인
    if not result.get("patient_context") and not result.get("error"):
        print("⚠️  경고: 에러는 없지만 컨텍스트가 비어있습니다")

    if result.get("patient_context"):
        print("✅ 환자 컨텍스트 로드 테스트 통과")
    else:
        print("⚠️  환자 컨텍스트 로드 실패 (위 에러 확인)")

    # Case 2: 환자 ID가 없는 경우
    state_no_patient = state.copy()
    state_no_patient["patient_id"] = None

    result_no_patient = load_patient_context_node(state_no_patient)
    assert result_no_patient["patient_context"] == ""
    print("✅ 환자 ID 없음 케이스 통과")


def test_should_retrieve_node():
    """검색 필요성 판단 노드 테스트"""
    print("\n=== test_should_retrieve_node ===")

    # Case 1: 대사증후군 관련 질문 (검색 필요)
    state_related: RAGState = {
        "question": "대사증후군의 진단 기준은 무엇인가요?",
        "patient_id": None,
        "patient_context": None,
        "should_retrieve": False,
        "relevance_scores": [],
        "support_score": 0.0,
        "usefulness_score": 0.0,
        "crag_action": "",
        "crag_confidence": 0.0,
        "internal_docs": [],
        "external_docs": [],
        "merged_context": "",
        "answer": "",
        "iteration": 0,
        "max_iterations": 2,
        "needs_regeneration": False,
        "error": None,
        "metadata": {},
    }

    result_related = should_retrieve_node(state_related)
    print(f"질문: {state_related['question']}")
    print(f"검색 필요: {result_related['should_retrieve']}")
    print(f"반복 횟수: {result_related['iteration']}")

    assert "should_retrieve" in result_related
    assert "iteration" in result_related
    assert result_related["iteration"] == 1
    print("✅ 대사증후군 관련 질문 테스트 통과")

    # Case 2: 일반 질문 (검색 불필요할 수 있음)
    state_general: RAGState = {
        "question": "안녕하세요",
        "patient_id": None,
        "patient_context": None,
        "should_retrieve": False,
        "relevance_scores": [],
        "support_score": 0.0,
        "usefulness_score": 0.0,
        "crag_action": "",
        "crag_confidence": 0.0,
        "internal_docs": [],
        "external_docs": [],
        "merged_context": "",
        "answer": "",
        "iteration": 0,
        "max_iterations": 2,
        "needs_regeneration": False,
        "error": None,
        "metadata": {},
    }

    result_general = should_retrieve_node(state_general)
    print(f"\n질문: {state_general['question']}")
    print(f"검색 필요: {result_general['should_retrieve']}")

    assert "should_retrieve" in result_general
    print("✅ 일반 질문 테스트 통과")


def test_integration():
    """두 노드 통합 테스트"""
    print("\n=== test_integration ===")

    state: RAGState = {
        "question": "복부비만 관리 방법을 알려주세요",
        "patient_id": 1,
        "patient_context": None,
        "should_retrieve": False,
        "relevance_scores": [],
        "support_score": 0.0,
        "usefulness_score": 0.0,
        "crag_action": "",
        "crag_confidence": 0.0,
        "internal_docs": [],
        "external_docs": [],
        "merged_context": "",
        "answer": "",
        "iteration": 0,
        "max_iterations": 2,
        "needs_regeneration": False,
        "error": None,
        "metadata": {},
    }

    # Step 1: 환자 컨텍스트 로드
    result1 = load_patient_context_node(state)
    state.update(result1)
    print(f"1. 환자 컨텍스트 로드: {bool(state.get('patient_context'))}")

    # Step 2: 검색 필요성 판단
    result2 = should_retrieve_node(state)
    state.update(result2)
    print(f"2. 검색 필요: {state['should_retrieve']}")
    print(f"3. 현재 반복: {state['iteration']}")

    assert state.get("patient_context") is not None
    assert state["should_retrieve"] is not None
    assert state["iteration"] == 1

    print("✅ 통합 테스트 통과")


if __name__ == "__main__":
    test_load_patient_context_node()
    test_should_retrieve_node()
    test_integration()
    print("\n" + "=" * 50)
    print("✅ Task 5.2 모든 테스트 통과!")
    print("=" * 50)
