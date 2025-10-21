# -*- coding: utf-8 -*-
"""
LangGraph Workflow 구성

Self-RAG + CRAG = Self-CRAG 기반 Agentic RAG 그래프
"""

from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END

from src.graph.state import RAGState
from src.graph.nodes import (
    load_patient_context_node,
    should_retrieve_node,
    retrieve_internal_node,
    evaluate_retrieval_node,
    decide_crag_action_node,
    search_external_node,
    merge_context_node,
    generate_answer_node,
    evaluate_answer_node,
)


# === Task 6.2: 조건부 엣지 함수 ===


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


def build_rag_graph():
    """
    Self-CRAG 기반 Agentic RAG 그래프 생성

    Task 6.1: 기본 순차 연결
    Task 6.2: Self-RAG [Retrieve] 조건부 엣지 추가

    Returns:
        CompiledGraph: 컴파일된 LangGraph
    """
    # StateGraph 생성
    workflow = StateGraph(RAGState)

    # 노드 추가 (9개)
    workflow.add_node("load_patient_context", load_patient_context_node)
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
    workflow.add_edge("load_patient_context", "should_retrieve")

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
    workflow.add_edge("decide_crag_action", "search_external")
    workflow.add_edge("search_external", "merge_context")
    workflow.add_edge("merge_context", "generate_answer")

    # 답변 생성 → 평가 → 종료
    workflow.add_edge("generate_answer", "evaluate_answer")
    workflow.add_edge("evaluate_answer", END)

    # 컴파일
    app = workflow.compile()

    return app


def create_initial_state(question: str, patient_id: int = None) -> RAGState:
    """
    초기 상태 생성 헬퍼 함수

    Args:
        question: 사용자 질문
        patient_id: 환자 ID (선택)

    Returns:
        RAGState: 초기화된 상태
    """
    return {
        "question": question,
        "patient_id": patient_id,
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


# 간단한 실행 함수
def run_rag(question: str, patient_id: int = None) -> Dict[str, Any]:
    """
    RAG 실행 헬퍼 함수

    Args:
        question: 사용자 질문
        patient_id: 환자 ID (선택)

    Returns:
        최종 상태 딕셔너리
    """
    graph = build_rag_graph()
    initial_state = create_initial_state(question, patient_id)

    # 그래프 실행
    final_state = graph.invoke(initial_state)

    return final_state
