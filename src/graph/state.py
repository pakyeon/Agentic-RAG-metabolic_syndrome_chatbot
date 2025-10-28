# -*- coding: utf-8 -*-
"""Agentic RAG 그래프 상태 정의"""

from typing import TypedDict, List, Optional, Dict, Any, Annotated
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgenticRAGState(TypedDict):
    """
    Agentic RAG를 위한 메시지 기반 상태

    기존 RAGState를 MessagesState 패턴으로 전환
    Self-RAG + CRAG 기법 유지, 상태 관리만 메시지 기반으로 변경
    """

    # 메시지 히스토리 (핵심: 질문, 답변, 컨텍스트 모두 메시지로 관리)
    messages: Annotated[List[BaseMessage], add_messages]

    # 환자 정보
    patient_id: Optional[int]
    patient_context: Optional[str]

    # 메모리
    short_term_memory: List[str]
    long_term_memory: List[str]
    memory_session_id: Optional[str]

    # 검색 결과
    internal_docs: List[Document]
    external_docs: List[Document]

    # 제어
    iteration: int
    max_iterations: int

    # 메타데이터 (평가 점수, 액션, 에러 등)
    metadata: Dict[str, Any]
