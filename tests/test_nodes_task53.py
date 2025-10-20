# -*- coding: utf-8 -*-
"""Task 5.3 노드 테스트"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.graph.nodes import (
    retrieve_internal_node,
    evaluate_retrieval_node,
    decide_crag_action_node,
)
from src.graph.state import RAGState


def test_retrieve_internal_node():
    """내부 검색 노드 테스트"""
    print("\n=== test_retrieve_internal_node ===")

    state: RAGState = {
        "question": "대사증후군의 진단 기준은 무엇인가요?",
        "patient_id": None,
        "patient_context": None,
        "should_retrieve": True,
        "relevance_scores": [],
        "support_score": 0.0,
        "usefulness_score": 0.0,
        "crag_action": "",
        "crag_confidence": 0.0,
        "internal_docs": [],
        "external_docs": [],
        "merged_context": "",
        "answer": "",
        "iteration": 1,
        "max_iterations": 2,
        "needs_regeneration": False,
        "error": None,
        "metadata": {},
    }

    result = retrieve_internal_node(state)

    if result.get("error"):
        print(f"❌ 에러 발생: {result['error']}")

    docs = result.get("internal_docs", [])
    print(f"검색된 문서 수: {len(docs)}")

    if docs:
        print(f"첫 번째 문서 미리보기: {docs[0].page_content[:100]}...")
        print("✅ 내부 검색 테스트 통과")
    else:
        print("⚠️  검색된 문서가 없습니다 (VectorDB 빌드 확인 필요)")

    assert "internal_docs" in result


def test_evaluate_retrieval_node():
    """검색 품질 평가 노드 테스트"""
    print("\n=== test_evaluate_retrieval_node ===")

    # 먼저 검색 수행
    state: RAGState = {
        "question": "복부비만의 기준은?",
        "patient_id": None,
        "patient_context": None,
        "should_retrieve": True,
        "relevance_scores": [],
        "support_score": 0.0,
        "usefulness_score": 0.0,
        "crag_action": "",
        "crag_confidence": 0.0,
        "internal_docs": [],
        "external_docs": [],
        "merged_context": "",
        "answer": "",
        "iteration": 1,
        "max_iterations": 2,
        "needs_regeneration": False,
        "error": None,
        "metadata": {},
    }

    # Step 1: 검색
    search_result = retrieve_internal_node(state)
    state.update(search_result)

    if not state.get("internal_docs"):
        print("⚠️  검색 결과 없음, 평가 스킵")
        return

    # Step 2: 평가
    eval_result = evaluate_retrieval_node(state)

    if eval_result.get("error"):
        print(f"❌ 에러 발생: {eval_result['error']}")

    scores = eval_result.get("relevance_scores", [])
    print(f"관련성 점수: {scores}")

    if scores:
        print(f"평균 관련성: {sum(scores)/len(scores):.2f}/5.0")
        print("✅ 검색 품질 평가 테스트 통과")
    else:
        print("⚠️  평가 점수 없음")

    assert "relevance_scores" in eval_result


def test_decide_crag_action_node():
    """CRAG 액션 결정 노드 테스트"""
    print("\n=== test_decide_crag_action_node ===")

    state: RAGState = {
        "question": "고혈압 관리 방법은?",
        "patient_id": None,
        "patient_context": None,
        "should_retrieve": True,
        "relevance_scores": [],
        "support_score": 0.0,
        "usefulness_score": 0.0,
        "crag_action": "",
        "crag_confidence": 0.0,
        "internal_docs": [],
        "external_docs": [],
        "merged_context": "",
        "answer": "",
        "iteration": 1,
        "max_iterations": 2,
        "needs_regeneration": False,
        "error": None,
        "metadata": {},
    }

    # Step 1: 검색
    search_result = retrieve_internal_node(state)
    state.update(search_result)

    # Step 2: 평가
    eval_result = evaluate_retrieval_node(state)
    state.update(eval_result)

    # Step 3: CRAG 액션 결정
    crag_result = decide_crag_action_node(state)

    if crag_result.get("error"):
        print(f"❌ 에러 발생: {crag_result['error']}")

    action = crag_result.get("crag_action")
    confidence = crag_result.get("crag_confidence")

    print(f"CRAG 액션: {action}")
    print(f"신뢰도: {confidence:.2f}")

    assert action in ["CORRECT", "INCORRECT", "AMBIGUOUS"]
    print("✅ CRAG 액션 결정 테스트 통과")


def test_integration():
    """세 노드 통합 테스트"""
    print("\n=== test_integration ===")

    state: RAGState = {
        "question": "대사증후군 예방을 위한 식이요법은?",
        "patient_id": 1,
        "patient_context": None,
        "should_retrieve": True,
        "relevance_scores": [],
        "support_score": 0.0,
        "usefulness_score": 0.0,
        "crag_action": "",
        "crag_confidence": 0.0,
        "internal_docs": [],
        "external_docs": [],
        "merged_context": "",
        "answer": "",
        "iteration": 1,
        "max_iterations": 2,
        "needs_regeneration": False,
        "error": None,
        "metadata": {},
    }

    # 전체 파이프라인
    print("1. 내부 검색...")
    result1 = retrieve_internal_node(state)
    state.update(result1)
    print(f"   검색 문서 수: {len(state.get('internal_docs', []))}")

    if not state.get("internal_docs"):
        print("⚠️  검색 결과 없음, 통합 테스트 중단")
        return

    print("2. 검색 품질 평가...")
    result2 = evaluate_retrieval_node(state)
    state.update(result2)
    scores = state.get("relevance_scores", [])
    if scores:
        print(f"   평균 관련성: {sum(scores)/len(scores):.2f}/5.0")

    print("3. CRAG 액션 결정...")
    result3 = decide_crag_action_node(state)
    state.update(result3)
    print(f"   액션: {state['crag_action']}")
    print(f"   신뢰도: {state['crag_confidence']:.2f}")

    assert state.get("internal_docs") is not None
    assert state.get("relevance_scores") is not None
    assert state.get("crag_action") in ["CORRECT", "INCORRECT", "AMBIGUOUS"]

    print("✅ 통합 테스트 통과")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    test_retrieve_internal_node()
    test_evaluate_retrieval_node()
    test_decide_crag_action_node()
    test_integration()
    print("\n" + "=" * 50)
    print("✅ Task 5.3 모든 테스트 통과!")
    print("=" * 50)
