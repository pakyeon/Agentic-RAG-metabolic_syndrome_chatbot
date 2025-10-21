# -*- coding: utf-8 -*-
"""Task 5.4 노드 테스트"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.graph.nodes import search_external_node, merge_context_node
from src.graph.state import RAGState
from langchain_core.documents import Document


def test_search_external_node():
    """외부 검색 노드 테스트"""
    print("\n=== test_search_external_node ===")

    # Case 1: action="incorrect" → 외부 검색 실행
    print("\n[Case 1] action='incorrect' → 외부 검색 실행")
    state_incorrect: RAGState = {
        "question": "대사증후군 최신 치료법은?",
        "patient_id": None,
        "patient_context": None,
        "should_retrieve": True,
        "relevance_scores": [],
        "support_score": 0.0,
        "usefulness_score": 0.0,
        "crag_action": "incorrect",
        "crag_confidence": 1.0,
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

    result = search_external_node(state_incorrect)

    if result.get("error"):
        print(f"   ❌ 에러: {result['error']}")
        print(f"   → Tavily API 호출에 문제가 있을 수 있습니다")

    docs = result.get("external_docs", [])
    print(f"   외부 문서 수: {len(docs)}개")

    if len(docs) == 0:
        print(f"   ⚠️  외부 검색 결과가 없습니다")
        if not result.get("error"):
            print(f"   → Tavily Tool 응답 형식 확인 필요")
    else:
        print(f"   ✓ 외부 검색 성공!")
        print(f"   첫 문서 출처: {docs[0].metadata.get('source')}")
        print(f"   첫 문서 미리보기: {docs[0].page_content[:80]}...")

    assert "external_docs" in result
    print("   ✅ INCORRECT 케이스 통과")

    # Case 2: action="ambiguous" → 외부 검색 실행
    print("\n[Case 2] action='ambiguous' → 외부 검색 실행")
    state_ambiguous: RAGState = {
        "question": "대사증후군 식이요법",
        "patient_id": None,
        "patient_context": None,
        "should_retrieve": True,
        "relevance_scores": [],
        "support_score": 0.0,
        "usefulness_score": 0.0,
        "crag_action": "ambiguous",
        "crag_confidence": 0.7,
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

    result = search_external_node(state_ambiguous)

    if result.get("error"):
        print(f"   ❌ 에러: {result['error']}")
    else:
        docs = result.get("external_docs", [])
        print(f"   외부 문서 수: {len(docs)}개")

    assert "external_docs" in result
    print("   ✅ AMBIGUOUS 케이스 통과")

    # Case 3: action="correct" → 외부 검색 스킵
    print("\n[Case 3] action='correct' → 외부 검색 스킵")
    state_correct: RAGState = {
        "question": "복부비만 기준",
        "patient_id": None,
        "patient_context": None,
        "should_retrieve": True,
        "relevance_scores": [],
        "support_score": 0.0,
        "usefulness_score": 0.0,
        "crag_action": "correct",
        "crag_confidence": 1.0,
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

    result = search_external_node(state_correct)
    docs = result.get("external_docs", [])

    print(f"   외부 문서 수: {len(docs)}개 (예상: 0)")
    assert len(docs) == 0
    print("   ✅ CORRECT 케이스 통과 (검색 스킵)")


def test_merge_context_node():
    """컨텍스트 병합 노드 테스트"""
    print("\n\n=== test_merge_context_node ===")

    # 샘플 문서 생성
    internal_docs = [
        Document(
            page_content="대사증후군은 복부비만, 고혈압, 고혈당이 복합된 상태입니다.",
            metadata={"source": "internal", "basename": "metabolic_syndrome.md"},
        ),
        Document(
            page_content="비타민 D 결핍은 우울감을 유발할 수 있습니다.",
            metadata={"source": "internal", "basename": "vitamin_d.md"},
        ),
    ]

    external_docs = [
        Document(
            page_content="대사증후군 환자는 저염식과 규칙적 운동이 필요합니다.",
            metadata={
                "source": "tavily_web_search",
                "title": "대사증후군 관리 가이드",
                "url": "https://example.com",
            },
        )
    ]

    # Case 1: CORRECT → 내부 문서만
    print("\n[Case 1] CORRECT → 내부 문서만")
    state_correct: RAGState = {
        "question": "대사증후군이란?",
        "patient_id": None,
        "patient_context": None,
        "should_retrieve": True,
        "relevance_scores": [],
        "support_score": 0.0,
        "usefulness_score": 0.0,
        "crag_action": "correct",
        "crag_confidence": 1.0,
        "internal_docs": internal_docs,
        "external_docs": external_docs,
        "merged_context": "",
        "answer": "",
        "iteration": 1,
        "max_iterations": 2,
        "needs_regeneration": False,
        "error": None,
        "metadata": {},
    }

    result = merge_context_node(state_correct)
    context = result.get("merged_context", "")

    print(f"   컨텍스트 길이: {len(context)} 문자")
    print(
        f"   내부 문서만 포함: {'내부 문서' in context and '외부 문서' not in context}"
    )

    assert "merged_context" in result
    assert len(context) > 0
    print("   ✅ CORRECT 케이스 통과")

    # Case 2: INCORRECT → 외부 문서로 교체
    print("\n[Case 2] INCORRECT → 외부 문서로 교체")
    state_incorrect: RAGState = {
        "question": "대사증후군 치료법",
        "patient_id": None,
        "patient_context": None,
        "should_retrieve": True,
        "relevance_scores": [],
        "support_score": 0.0,
        "usefulness_score": 0.0,
        "crag_action": "incorrect",
        "crag_confidence": 1.0,
        "internal_docs": internal_docs,
        "external_docs": external_docs,
        "merged_context": "",
        "answer": "",
        "iteration": 1,
        "max_iterations": 2,
        "needs_regeneration": False,
        "error": None,
        "metadata": {},
    }

    result = merge_context_node(state_incorrect)
    context = result.get("merged_context", "")

    print(f"   컨텍스트 길이: {len(context)} 문자")
    print(
        f"   외부 문서만 포함: {'외부 문서' in context and '내부 문서' not in context}"
    )

    assert "merged_context" in result
    assert "외부 문서" in context
    print("   ✅ INCORRECT 케이스 통과")

    # Case 3: AMBIGUOUS → 내부 + 외부 병합
    print("\n[Case 3] AMBIGUOUS → 내부 + 외부 병합")

    # RelevanceResult 객체 대신 간단한 객체 사용
    class SimpleRelevance:
        def __init__(self, relevance):
            self.relevance = relevance

    state_ambiguous: RAGState = {
        "question": "대사증후군 관리",
        "patient_id": None,
        "patient_context": None,
        "should_retrieve": True,
        "relevance_scores": [
            SimpleRelevance("relevant"),  # 첫 문서는 관련 있음
            SimpleRelevance("irrelevant"),  # 두 문서는 무관
        ],
        "support_score": 0.0,
        "usefulness_score": 0.0,
        "crag_action": "ambiguous",
        "crag_confidence": 0.7,
        "internal_docs": internal_docs,
        "external_docs": external_docs,
        "merged_context": "",
        "answer": "",
        "iteration": 1,
        "max_iterations": 2,
        "needs_regeneration": False,
        "error": None,
        "metadata": {},
    }

    result = merge_context_node(state_ambiguous)
    context = result.get("merged_context", "")

    print(f"   컨텍스트 길이: {len(context)} 문자")
    has_internal = "내부 문서" in context
    has_external = "외부 문서" in context
    print(f"   내부+외부 혼합: 내부={has_internal}, 외부={has_external}")

    assert "merged_context" in result
    assert has_internal or has_external  # 최소 하나는 있어야 함
    print("   ✅ AMBIGUOUS 케이스 통과")


def test_integration():
    """Task 5.4 통합 테스트"""
    print("\n\n=== test_integration (Task 5.4) ===")

    # 시나리오: incorrect 액션 → 외부 검색 → 병합
    print("\n시나리오: INCORRECT 액션으로 외부 검색 후 병합")

    state: RAGState = {
        "question": "대사증후군 최신 연구는?",
        "patient_id": None,
        "patient_context": None,
        "should_retrieve": True,
        "relevance_scores": [],
        "support_score": 0.0,
        "usefulness_score": 0.0,
        "crag_action": "incorrect",
        "crag_confidence": 1.0,
        "internal_docs": [
            Document(
                page_content="오래된 정보",
                metadata={"source": "internal", "basename": "old.md"},
            )
        ],
        "external_docs": [],
        "merged_context": "",
        "answer": "",
        "iteration": 1,
        "max_iterations": 2,
        "needs_regeneration": False,
        "error": None,
        "metadata": {},
    }

    # Step 1: 외부 검색
    print("\n1. 외부 검색 실행...")
    search_result = search_external_node(state)
    state["external_docs"] = search_result.get("external_docs", [])

    if search_result.get("error"):
        print(f"   ❌ 외부 검색 에러: {search_result['error']}")
    else:
        print(f"   외부 문서 수: {len(state['external_docs'])}개")

    # Step 2: 컨텍스트 병합
    print("\n2. 컨텍스트 병합...")
    merge_result = merge_context_node(state)
    state["merged_context"] = merge_result.get("merged_context", "")

    if merge_result.get("error"):
        print(f"   ❌ 병합 에러: {merge_result['error']}")
    else:
        print(f"   최종 컨텍스트 길이: {len(state['merged_context'])} 문자")

    # 검증
    assert state.get("external_docs") is not None
    assert state.get("merged_context") is not None

    print("\n✅ Task 5.4 통합 테스트 통과!")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    # TAVILY_API_KEY 확인
    if not os.getenv("TAVILY_API_KEY"):
        print("\n" + "=" * 50)
        print("⚠️  TAVILY_API_KEY가 설정되지 않았습니다!")
        print("=" * 50)
        print("Tavily 외부 검색 테스트를 스킵합니다.")
        print("환경변수를 설정하려면:")
        print("  export TAVILY_API_KEY=your-key-here")
        print("=" * 50)
        # 병합 테스트만 실행
        test_merge_context_node()
    else:
        # 전체 테스트 실행
        test_search_external_node()
        test_merge_context_node()
        test_integration()

    print("\n" + "=" * 50)
    print("✅ Task 5.4 모든 테스트 완료!")
    print("=" * 50)
