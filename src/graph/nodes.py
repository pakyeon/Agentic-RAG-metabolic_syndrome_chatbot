# -*- coding: utf-8 -*-
"""Agentic RAG 그래프 노드 함수들"""

from typing import Dict
from src.graph.state import RAGState


# === 환자 컨텍스트 로드 ===
def load_patient_context_node(state: RAGState) -> Dict:
    """환자 정보를 로드하여 컨텍스트 생성"""
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
                "error": f"환자 ID '{patient_id}' 정보를 찾을 수 없습니다",
            }

        return {"patient_context": context}

    except Exception as e:
        return {"patient_context": "", "error": f"환자 정보 로드 실패: {str(e)}"}


# === Self-RAG [Retrieve] 토큰: 검색 필요성 판단 ===
def should_retrieve_node(state: RAGState) -> Dict:
    """검색이 필요한지 판단 (Self-RAG [Retrieve] 토큰)"""
    try:
        from src.evaluation.self_rag_evaluator import SelfRAGEvaluator

        evaluator = SelfRAGEvaluator()
        should_retrieve = evaluator.evaluate_retrieve_need(state["question"])

        return {
            "should_retrieve": should_retrieve,
            "iteration": state.get("iteration", 0) + 1,
        }

    except Exception as e:
        return {
            "should_retrieve": True,  # 오류 시 검색 진행
            "iteration": state.get("iteration", 0) + 1,
            "error": f"검색 필요성 평가 실패: {str(e)}",
        }


# === 내부 검색 (VectorDB Hybrid Search) ===
def retrieve_internal_node(state: RAGState) -> Dict:
    """VectorDB에서 관련 문서 검색"""
    try:
        from src.data.vector_store import load_vector_db, HybridRetriever
        from langchain_core.documents import Document

        vector_store = load_vector_db()

        # VectorDB에서 모든 문서를 가져와서 BM25용으로 사용
        # Note: 실제로는 _chunks가 캐시되어 있어야 하지만,
        # 여기서는 VectorDB에서 모든 문서를 가져옴
        all_docs_data = vector_store.get()
        documents = [
            Document(page_content=content, metadata=metadata)
            for content, metadata in zip(
                all_docs_data.get("documents", []), all_docs_data.get("metadatas", [])
            )
        ]

        if not documents:
            return {"internal_docs": [], "error": "VectorDB에 문서가 없습니다"}

        retriever = HybridRetriever(vectorstore=vector_store, documents=documents)

        # 환자 컨텍스트가 있으면 쿼리에 포함
        query = state["question"]
        if state.get("patient_context"):
            query = f"{state['patient_context']}\n\n질문: {state['question']}"

        docs = retriever.search(query, k=5)

        return {"internal_docs": docs}

    except Exception as e:
        return {"internal_docs": [], "error": f"내부 검색 실패: {str(e)}"}


# === Self-RAG ISREL: 검색 품질 평가 ===
def evaluate_retrieval_node(state: RAGState) -> Dict:
    """검색된 문서의 관련성 평가 (Self-RAG ISREL)"""
    try:
        from src.evaluation.self_rag_evaluator import SelfRAGEvaluator

        evaluator = SelfRAGEvaluator()
        docs = state.get("internal_docs", [])

        if not docs:
            return {"relevance_scores": []}

        scores = [
            evaluator.evaluate_relevance(doc.page_content, state["question"])
            for doc in docs
        ]

        return {"relevance_scores": scores}

    except Exception as e:
        return {"relevance_scores": [], "error": f"검색 품질 평가 실패: {str(e)}"}


# === CRAG: 액션 결정 ===
def decide_crag_action_node(state: RAGState) -> Dict:
    """CRAG 전략에 따른 액션 결정"""
    try:
        from src.strategies.corrective_rag import CorrectiveRAG

        strategy = CorrectiveRAG()
        docs = state.get("internal_docs", [])

        if not docs:
            return {"crag_action": "incorrect", "crag_confidence": 0.0}

        action, reason = strategy.decide_action(query=state["question"], documents=docs)

        return {
            "crag_action": action.value,
            "crag_confidence": 1.0,  # CorrectiveRAG에서 confidence 제공 안함
        }

    except Exception as e:
        return {
            "crag_action": "correct",
            "crag_confidence": 0.0,
            "error": f"CRAG 액션 결정 실패: {str(e)}",
        }
