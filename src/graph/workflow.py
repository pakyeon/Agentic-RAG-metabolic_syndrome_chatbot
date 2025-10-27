# -*- coding: utf-8 -*-
"""
LangGraph Workflow 구성

Self-RAG + CRAG = Self-CRAG 기반 Agentic RAG 그래프
"""
import sqlite3
from typing import Dict, Any, Literal, List

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from src.graph.state import RAGState
from src.graph.nodes import (
    load_patient_context_node,
    load_memory_context_node,
    should_retrieve_node,
    retrieve_internal_node,
    evaluate_retrieval_node,
    decide_crag_action_node,
    search_external_node,
    merge_context_node,
    generate_answer_node,
    evaluate_answer_node,
)
from src.memory import get_short_term_store

_CHECKPOINTER: SqliteSaver | None = None


def _get_checkpointer() -> SqliteSaver:
    """Return a shared SqliteSaver using the short-term memory database."""
    global _CHECKPOINTER
    if _CHECKPOINTER is None:
        store = get_short_term_store()
        connection = sqlite3.connect(store.db_path, check_same_thread=False)
        _CHECKPOINTER = SqliteSaver(connection)
    return _CHECKPOINTER


# === Task 6.2: Self-RAG 조건부 엣지 ===


def route_retrieve(state: RAGState) -> Literal["retrieve_internal", "generate_answer"]:
    """
    Self-RAG [Retrieve] 토큰 기반 분기

    - should_retrieve=True → 내부 검색 수행
    - should_retrieve=False → 검색 스킵하고 직접 답변

    Args:
        state: 현재 그래프 상태

    Returns:
        다음 노드 이름
    """
    if state.get("should_retrieve", True):
        return "retrieve_internal"
    else:
        return "generate_answer"


# === Task 6.3: CRAG 조건부 엣지 ===


def route_crag_action(state: RAGState) -> Literal["search_external", "merge_context"]:
    """
    CRAG 액션에 따른 분기

    - CORRECT: 내부 문서만 사용 → 외부 검색 스킵 → merge_context
    - INCORRECT: 외부 검색으로 교체 → search_external → merge_context
    - AMBIGUOUS: 내부+외부 병합 → search_external → merge_context

    Args:
        state: 현재 그래프 상태

    Returns:
        다음 노드 이름
    """
    action = state.get("crag_action", "correct").lower()

    if action == "correct":
        # CORRECT: 내부 문서 품질 충분 → 외부 검색 스킵
        return "merge_context"
    else:
        # INCORRECT or AMBIGUOUS: 외부 검색 필요
        return "search_external"


# === Task 6.4: Self-RAG 재생성 루프 ===


def route_regeneration(state: RAGState) -> Literal["retrieve_internal", "__end__"]:
    """
    Self-RAG 답변 품질 평가 후 재생성 판단

    - needs_regeneration=True AND iteration < max_iterations → 재검색
    - 아니면 → END

    Args:
        state: 현재 그래프 상태

    Returns:
        다음 노드 이름 또는 "__end__"
    """
    needs_regen = state.get("needs_regeneration", False)
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 2)

    if needs_regen and iteration < max_iterations:
        # 재생성 필요 & 최대 반복 횟수 미만 → 재검색
        return "retrieve_internal"
    else:
        # 품질 충분 또는 최대 반복 도달 → 종료
        return "__end__"


def build_rag_graph():
    """
    Self-CRAG 기반 Agentic RAG 그래프 생성

    Task 6.1: 기본 순차 연결
    Task 6.2: Self-RAG [Retrieve] 조건부 엣지 추가
    Task 6.3: CRAG 조건부 엣지 추가
    Task 6.4: Self-RAG 재생성 루프 추가

    Returns:
        CompiledGraph: 컴파일된 LangGraph
    """
    # StateGraph 생성
    workflow = StateGraph(RAGState)

    # 노드 추가 (9개)
    workflow.add_node("load_patient_context", load_patient_context_node)
    workflow.add_node("load_memory_context", load_memory_context_node)
    workflow.add_node("should_retrieve", should_retrieve_node)
    workflow.add_node("retrieve_internal", retrieve_internal_node)
    workflow.add_node("evaluate_retrieval", evaluate_retrieval_node)
    workflow.add_node("decide_crag_action", decide_crag_action_node)
    workflow.add_node("search_external", search_external_node)
    workflow.add_node("merge_context", merge_context_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("evaluate_answer", evaluate_answer_node)

    # 엣지 연결
    workflow.set_entry_point("load_patient_context")
    workflow.add_edge("load_patient_context", "load_memory_context")
    workflow.add_edge("load_memory_context", "should_retrieve")

    # Task 6.2: Self-RAG [Retrieve] 조건부 분기
    workflow.add_conditional_edges(
        "should_retrieve",
        route_retrieve,
        {
            "retrieve_internal": "retrieve_internal",
            "generate_answer": "generate_answer",
        },
    )

    # 검색 수행 경로
    workflow.add_edge("retrieve_internal", "evaluate_retrieval")
    workflow.add_edge("evaluate_retrieval", "decide_crag_action")

    # Task 6.3: CRAG 조건부 분기
    workflow.add_conditional_edges(
        "decide_crag_action",
        route_crag_action,
        {
            "search_external": "search_external",
            "merge_context": "merge_context",
        },
    )

    # 외부 검색 → 병합
    workflow.add_edge("search_external", "merge_context")

    # 병합 → 답변 생성
    workflow.add_edge("merge_context", "generate_answer")

    # 답변 생성 → 평가
    workflow.add_edge("generate_answer", "evaluate_answer")

    # Task 6.4: 재생성 루프
    workflow.add_conditional_edges(
        "evaluate_answer",
        route_regeneration,
        {
            "retrieve_internal": "retrieve_internal",
            "__end__": END,
        },
    )

    # 컴파일
    app = workflow.compile(checkpointer=_get_checkpointer())

    return app


def create_initial_state(
    question: str,
    patient_id: int | None = None,
    *,
    session_id: str | None = None,
    short_term: List[str] | None = None,
) -> RAGState:
    """
    초기 상태 생성 헬퍼 함수

    Args:
        question: 사용자 질문
        patient_id: 환자 ID (선택)
        session_id: 장기 기억을 위한 세션/사용자 식별자
        short_term: 이전 턴의 단기 기억(선택)

    Returns:
        RAGState: 초기화된 상태
    """
    return {
        "question": question,
        "patient_id": patient_id,
        "patient_context": None,
        "short_term_memory": list(short_term or []),
        "long_term_memory": [],
        "memory_session_id": session_id,
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


# 간단한 실행 함수
def run_rag(
    question: str,
    patient_id: int | None = None,
    *,
    session_id: str | None = None,
    short_term: List[str] | None = None,
) -> Dict[str, Any]:
    """
    RAG 실행 헬퍼 함수

    Args:
        question: 사용자 질문
        patient_id: 환자 ID (선택)
        session_id: 장기 기억 세션 ID (선택)
        short_term: 이전 턴 단기 기억 (선택)

    Returns:
        최종 상태 딕셔너리
    """
    graph = build_rag_graph()
    initial_state = create_initial_state(
        question,
        patient_id,
        session_id=session_id,
        short_term=short_term,
    )

    # 그래프 실행
    config = {"configurable": {"thread_id": session_id or "default"}}
    final_state = graph.invoke(initial_state, config)

    return final_state
