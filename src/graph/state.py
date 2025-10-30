# -*- coding: utf-8 -*-
"""Agentic RAG 그래프 상태 정의"""

from typing import TypedDict, List, Optional, Dict, Any, NotRequired
from langchain_core.documents import Document


class RAGState(TypedDict):
    """Agentic RAG 그래프 상태

    Self-RAG + CRAG 기법이 적용된 Agentic RAG 시스템의 상태를 관리합니다.
    """

    # 입력
    question: str
    patient_id: Optional[int]

    # 환자 정보
    patient_context: Optional[str]

    # 메모리
    short_term_memory: List[str]
    long_term_memory: List[str]
    memory_session_id: Optional[str]

    # Self-RAG Reflection Tokens
    should_retrieve: bool  # [Retrieve] 토큰
    relevance_scores: List[float]  # ISREL: 문서 관련성 (1-5)
    support_score: float  # ISSUP: 답변 지원도 (1-5)
    usefulness_score: float  # ISUSE: 답변 유용성 (1-5)
    min_relevant_docs: NotRequired[int]
    early_stop_enabled: NotRequired[bool]
    evaluated_docs_count: NotRequired[int]
    total_evaluated_docs: NotRequired[int]
    early_stopped: NotRequired[bool]

    # CRAG Strategy
    crag_action: str  # correct/incorrect/ambiguous
    crag_confidence: float  # 0-1

    # 검색 결과
    internal_docs: List[Document]  # VectorDB 검색
    external_docs: List[Document]  # Tavily 외부 검색
    merged_context: str  # 최종 컨텍스트

    # 답변
    answer: str

    # 제어
    iteration: int
    max_iterations: int
    needs_regeneration: bool

    # 메타
    error: Optional[str]
    metadata: Dict[str, Any]
