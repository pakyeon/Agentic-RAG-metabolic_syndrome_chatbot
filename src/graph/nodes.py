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
