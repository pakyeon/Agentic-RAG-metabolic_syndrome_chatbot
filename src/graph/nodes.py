# -*- coding: utf-8 -*-
"""Agentic RAG 그래프 노드 함수들 (MessagesState 기반으로 전환)"""

from __future__ import annotations

import os
from typing import Dict, List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.graph.state import AgenticRAGState
from src.memory import get_graphiti_connector, get_short_term_store


# === 헬퍼 함수 ===
def _extract_question_from_messages(messages: List[BaseMessage]) -> str:
    """메시지 리스트에서 마지막 사용자 질문 추출"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content

    return ""


def _extract_last_ai_message(messages: List[BaseMessage]) -> str:
    """메시지 리스트에서 마지막 AI 답변 추출"""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg.content

    return ""


# === 환자 컨텍스트 로드 ===
def load_patient_context_node(state: AgenticRAGState) -> Dict:
    """환자 정보를 로드하여 컨텍스트 생성."""
    patient_id = state.get("patient_id")

    if not patient_id:
        return {"patient_context": ""}

    try:
        from src.data.patient_context import PatientContextProvider
        from src.data.patient_db import PatientDatabase

        db = PatientDatabase()
        provider = PatientContextProvider(db)
        context = provider.get_patient_context(patient_id, format="detailed")

        if not context:
            return {
                "patient_context": "",
                "metadata": {
                    **state.get("metadata", {}),
                    "error": f"환자 ID '{patient_id}' 정보를 찾을 수 없습니다",
                },
            }

        return {"patient_context": context}

    except Exception as exc:
        return {
            "patient_context": "",
            "metadata": {
                **state.get("metadata", {}),
                "error": f"환자 정보 로드 실패: {exc}",
            },
        }


def load_memory_context_node(state: AgenticRAGState) -> Dict:
    """SQLite 기반 단기 기억을 불러와 LangGraph 상태에 주입한다."""
    store = get_short_term_store()
    session_id = state.get("memory_session_id") or "default"
    context = store.get_context(session_id)

    segments: List[str] = []
    segments.extend(context.recent)
    if context.history_summary:
        segments.append(context.history_summary)
    segments.extend(context.topic_summaries)

    metadata = dict(state.get("metadata", {}))
    metadata["memory_recent_turns"] = len(context.recent)
    metadata["memory_history_summary"] = bool(context.history_summary)
    metadata["memory_topic_summaries"] = len(context.topic_summaries)

    return {
        "short_term_memory": segments,
        "long_term_memory": [],
        "metadata": metadata,
    }


# === Self-RAG [Retrieve] 토큰: 검색 필요성 판단 ===
def should_retrieve_node(state: AgenticRAGState) -> Dict:
    """검색이 필요한지 판단 (Self-RAG [Retrieve] 토큰)."""
    question = _extract_question_from_messages(state["messages"])
    patient_context = state.get("patient_context")
    iteration = state.get("iteration", 0)

    try:
        from src.evaluation import create_evaluator

        evaluator = create_evaluator()
        result = evaluator.evaluate_retrieve_need(
            query=question, patient_context=patient_context
        )

        metadata = dict(state.get("metadata", {}))
        metadata["should_retrieve"] = result.should_retrieve == "yes"
        metadata["retrieve_decision"] = result.should_retrieve
        metadata["retrieve_difficulty"] = result.difficulty
        metadata["retrieve_reason"] = result.reason

        return {
            "iteration": iteration + 1,
            "metadata": metadata,
        }

    except Exception as exc:
        metadata = dict(state.get("metadata", {}))
        metadata["should_retrieve"] = True
        metadata["error"] = f"검색 필요성 판단 실패: {exc}"

        return {
            "iteration": iteration + 1,
            "metadata": metadata,
        }


# === 내부 검색 ===
def retrieve_internal_node(state: AgenticRAGState) -> Dict:
    """내부 VectorDB에서 문서 검색."""
    question = _extract_question_from_messages(state["messages"])
    patient_context = state.get("patient_context")

    try:
        from src.data.vector_store import get_vector_store

        vector_store = get_vector_store()

        query = question
        if patient_context:
            query = f"{patient_context}\n\n{question}"

        docs = vector_store.search(query, k=5)

        return {"internal_docs": docs}

    except Exception as exc:
        metadata = dict(state.get("metadata", {}))
        metadata["error"] = f"내부 검색 실패: {exc}"

        return {"internal_docs": [], "metadata": metadata}


# === 검색 품질 평가 ===
def evaluate_retrieval_node(state: AgenticRAGState) -> Dict:
    """검색된 문서의 관련성 평가 (Self-RAG ISREL)."""
    question = _extract_question_from_messages(state["messages"])
    internal_docs = state.get("internal_docs", [])

    if not internal_docs:
        metadata = dict(state.get("metadata", {}))
        metadata["relevance_scores"] = []
        return {"metadata": metadata}

    try:
        from src.evaluation import create_evaluator

        evaluator = create_evaluator()
        evaluation = evaluator.assess_retrieval_quality(
            query=question, documents=internal_docs, min_relevant_docs=2
        )

        relevance_scores = [
            doc_eval.relevance.score for doc_eval in evaluation.document_evaluations
        ]

        metadata = dict(state.get("metadata", {}))
        metadata["relevance_scores"] = relevance_scores

        return {"metadata": metadata}

    except Exception as exc:
        metadata = dict(state.get("metadata", {}))
        metadata["relevance_scores"] = []
        metadata["error"] = f"검색 품질 평가 실패: {exc}"

        return {"metadata": metadata}


# === CRAG 액션 결정 ===
def decide_crag_action_node(state: AgenticRAGState) -> Dict:
    """CRAG 액션 결정 (CORRECT/INCORRECT/AMBIGUOUS)."""
    question = _extract_question_from_messages(state["messages"])
    internal_docs = state.get("internal_docs", [])
    relevance_scores = state.get("metadata", {}).get("relevance_scores", [])

    try:
        from src.strategies import create_corrective_rag

        crag = create_corrective_rag()
        result = crag.decide_action(
            query=question, documents=internal_docs, relevance_scores=relevance_scores
        )

        metadata = dict(state.get("metadata", {}))
        metadata["crag_action"] = result.action.value.lower()
        metadata["crag_confidence"] = result.confidence
        metadata["crag_reason"] = result.reason

        return {"metadata": metadata}

    except Exception as exc:
        metadata = dict(state.get("metadata", {}))
        metadata["crag_action"] = "correct"
        metadata["crag_confidence"] = 0.5
        metadata["error"] = f"CRAG 액션 결정 실패: {exc}"

        return {"metadata": metadata}


# === 외부 검색 ===
def search_external_node(state: AgenticRAGState) -> Dict:
    """Tavily를 사용한 외부 웹 검색."""
    question = _extract_question_from_messages(state["messages"])

    try:
        from src.tools.tavily import get_default_tool

        tool = get_default_tool()
        results = tool.invoke({"query": question})

        from langchain_core.documents import Document

        if isinstance(results, str):
            external_docs = [
                Document(page_content=results, metadata={"source": "tavily"})
            ]
        elif isinstance(results, list):
            external_docs = [
                Document(
                    page_content=item.get("content", ""),
                    metadata={"source": "tavily", "url": item.get("url", "")},
                )
                for item in results
            ]
        else:
            external_docs = []

        return {"external_docs": external_docs}

    except Exception as exc:
        metadata = dict(state.get("metadata", {}))
        metadata["error"] = f"외부 검색 실패: {exc}"

        return {"external_docs": [], "metadata": metadata}


# === 컨텍스트 병합 ===
def merge_context_node(state: AgenticRAGState) -> Dict:
    """CRAG 액션에 따라 컨텍스트 병합 후 SystemMessage로 추가."""
    crag_action = state.get("metadata", {}).get("crag_action", "correct")
    internal_docs = state.get("internal_docs", [])
    external_docs = state.get("external_docs", [])

    merged_parts = []

    if crag_action == "correct":
        # CORRECT: 내부 문서만 사용
        for i, doc in enumerate(internal_docs, 1):
            source = doc.metadata.get("basename", doc.metadata.get("source", "unknown"))
            merged_parts.append(f"[내부 문서 {i}] (출처: {source})\n{doc.page_content}")

    elif crag_action == "incorrect":
        # INCORRECT: 외부 문서로 완전 대체
        for i, doc in enumerate(external_docs, 1):
            source = doc.metadata.get("url", doc.metadata.get("source", "unknown"))
            merged_parts.append(f"[외부 문서 {i}] (출처: {source})\n{doc.page_content}")

    else:  # ambiguous
        # AMBIGUOUS: 내부+외부 혼합
        relevant_internal = [
            doc
            for i, doc in enumerate(internal_docs)
            if i < len(state.get("metadata", {}).get("relevance_scores", []))
            and state.get("metadata", {}).get("relevance_scores", [])[i] >= 3.0
        ]

        for i, doc in enumerate(relevant_internal, 1):
            source = doc.metadata.get("basename", doc.metadata.get("source", "unknown"))
            merged_parts.append(f"[내부 문서 {i}] (출처: {source})\n{doc.page_content}")

        for i, doc in enumerate(external_docs, 1):
            source = doc.metadata.get("url", doc.metadata.get("source", "unknown"))
            merged_parts.append(f"[외부 문서 {i}] (출처: {source})\n{doc.page_content}")

    merged_context = "\n\n".join(merged_parts)

    # 병합된 컨텍스트를 SystemMessage로 추가
    if merged_context:
        context_msg = SystemMessage(content=f"[검색 컨텍스트]\n\n{merged_context}")

        metadata = dict(state.get("metadata", {}))
        metadata["context_added"] = True

        return {"messages": [context_msg], "metadata": metadata}

    metadata = dict(state.get("metadata", {}))
    metadata["context_added"] = False

    return {"metadata": metadata}


# === 답변 생성 ===
def generate_answer_node(state: AgenticRAGState) -> Dict:
    """최종 답변 생성 (AIMessage로 반환)."""
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # messages를 그대로 LLM에 전달
        response = llm.invoke(state["messages"])

        return {
            "messages": [response],
        }

    except Exception as exc:
        # 에러 발생 시 에러 메시지를 AIMessage로 반환
        error_msg = AIMessage(content=f"답변 생성 중 오류가 발생했습니다: {exc}")

        metadata = dict(state.get("metadata", {}))
        metadata["error"] = f"답변 생성 실패: {exc}"

        return {"messages": [error_msg], "metadata": metadata}


# === 답변 품질 평가 ===
def evaluate_answer_node(state: AgenticRAGState) -> Dict:
    """답변 품질 평가 (Self-RAG ISSUP/ISUSE)."""
    question = _extract_question_from_messages(state["messages"])
    answer = _extract_last_ai_message(state["messages"])
    internal_docs = state.get("internal_docs", [])
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 2)

    if not answer:
        metadata = dict(state.get("metadata", {}))
        metadata["support_score"] = 0.0
        metadata["usefulness_score"] = 0.0
        metadata["needs_regeneration"] = False
        return {"metadata": metadata}

    try:
        from src.evaluation import create_evaluator

        evaluator = create_evaluator()

        # ISSUP: 답변 지원도 평가
        support_result = evaluator.evaluate_support(
            query=question, documents=internal_docs, answer=answer
        )

        # ISUSE: 답변 유용성 평가
        usefulness_result = evaluator.evaluate_usefulness(query=question, answer=answer)

        support_score = support_result.score
        usefulness_score = usefulness_result.score

        # 재생성 필요 여부 판단
        needs_regeneration = (
            support_score < 3.0 or usefulness_score < 3.0
        ) and iteration < max_iterations

        metadata = dict(state.get("metadata", {}))
        metadata["support_score"] = support_score
        metadata["usefulness_score"] = usefulness_score
        metadata["needs_regeneration"] = needs_regeneration

        # 단기 기억에 저장
        store = get_short_term_store()
        session_id = state.get("memory_session_id") or "default"
        store.add_turn(session_id, question, answer)

        return {"metadata": metadata}

    except Exception as exc:
        metadata = dict(state.get("metadata", {}))
        metadata["support_score"] = 3.0
        metadata["usefulness_score"] = 3.0
        metadata["needs_regeneration"] = False
        metadata["error"] = f"답변 품질 평가 실패: {exc}"

        return {"metadata": metadata}
