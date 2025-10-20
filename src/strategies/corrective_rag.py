"""
CRAG (Corrective RAG) 전략

Self-RAG 평가 결과를 바탕으로 보정 액션을 결정하고 실행합니다:
- Correct: 검색 품질 충분 → 지식 정제
- Incorrect: 검색 품질 낮음 → 웹 검색으로 대체
- Ambiguous: 불확실 → 웹 검색으로 보완
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document

from src.evaluation.self_rag_evaluator import (
    SelfRAGEvaluator,
    create_evaluator,
    OverallEvaluation,
)
from src.tools.tavily import get_tavily_tool


class CRAGAction(Enum):
    """CRAG 보정 액션"""

    CORRECT = "correct"  # 검색 품질 충분, 정제만 수행
    INCORRECT = "incorrect"  # 검색 품질 낮음, 웹 검색으로 대체
    AMBIGUOUS = "ambiguous"  # 불확실, 웹 검색으로 보완


@dataclass
class CRAGResult:
    """CRAG 실행 결과"""

    action: CRAGAction
    documents: List[Document]
    reason: str
    web_search_performed: bool = False
    original_doc_count: int = 0
    final_doc_count: int = 0


class CorrectiveRAG:
    """
    CRAG (Corrective RAG) 전략 구현

    Self-RAG 평가 결과를 바탕으로 보정 액션을 결정하고,
    필요 시 Tavily 웹 검색으로 결과를 보완/대체합니다.
    """

    def __init__(
        self,
        evaluator: Optional[SelfRAGEvaluator] = None,
        relevance_threshold: float = 0.5,
        min_relevant_docs: int = 2,
        max_web_results: int = 3,
    ):
        """
        Args:
            evaluator: Self-RAG 평가자 (None이면 자동 생성)
            relevance_threshold: 관련 문서 비율 임계값 (0.0~1.0)
            min_relevant_docs: 최소 관련 문서 수
            max_web_results: 웹 검색 시 최대 결과 수
        """
        self.evaluator = evaluator or create_evaluator()
        self.relevance_threshold = relevance_threshold
        self.min_relevant_docs = min_relevant_docs
        self.max_web_results = max_web_results
        self.tavily_tool = get_tavily_tool(max_results=max_web_results)

    def decide_action(
        self, query: str, documents: List[Document]
    ) -> Tuple[CRAGAction, str]:
        """
        Self-RAG 평가 결과를 바탕으로 CRAG 액션 결정

        Args:
            query: 사용자 질문
            documents: 검색된 문서들

        Returns:
            (CRAGAction, 이유)
        """
        if not documents:
            return (CRAGAction.INCORRECT, "검색 결과가 없어 웹 검색으로 대체합니다.")

        # Self-RAG ISREL 평가
        doc_contents = [doc.page_content for doc in documents]
        overall_eval = self.evaluator.evaluate_documents(
            query, doc_contents, min_relevant_docs=self.min_relevant_docs
        )

        relevant_count = sum(
            1
            for doc_eval in overall_eval.document_evaluations
            if doc_eval.relevance.relevance == "relevant"
        )
        relevance_ratio = relevant_count / len(documents)

        # 액션 결정
        if (
            relevant_count >= self.min_relevant_docs
            and relevance_ratio >= self.relevance_threshold
        ):
            # Correct: 충분히 관련성 있음
            return (
                CRAGAction.CORRECT,
                f"관련 문서 {relevant_count}개로 충분합니다. (비율: {relevance_ratio:.1%})",
            )
        elif relevant_count == 0:
            # Incorrect: 관련 문서가 하나도 없음
            return (CRAGAction.INCORRECT, "관련 문서가 없어 웹 검색으로 대체합니다.")
        else:
            # Ambiguous: 일부만 관련 있음
            return (
                CRAGAction.AMBIGUOUS,
                f"관련 문서가 {relevant_count}개로 불충분합니다. 웹 검색으로 보완합니다.",
            )

    def refine_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """
        문서 정제 (CORRECT 액션 시)

        관련 있는 문서만 필터링하고 순서를 최적화합니다.

        Args:
            query: 사용자 질문
            documents: 원본 문서들

        Returns:
            정제된 문서 리스트
        """
        refined_docs = []

        for doc in documents:
            relevance_result = self.evaluator.evaluate_relevance(
                query, doc.page_content
            )

            if relevance_result.relevance == "relevant":
                # 메타데이터에 평가 결과 추가
                doc_copy = Document(
                    page_content=doc.page_content,
                    metadata={
                        **doc.metadata,
                        "crag_relevance": "relevant",
                        "crag_confidence": relevance_result.confidence,
                    },
                )
                refined_docs.append(doc_copy)

        # 신뢰도 순으로 정렬
        refined_docs.sort(
            key=lambda d: d.metadata.get("crag_confidence", 0.0), reverse=True
        )

        return refined_docs

    def web_search(self, query: str) -> List[Document]:
        """
        Tavily 웹 검색 수행

        Args:
            query: 검색 쿼리

        Returns:
            웹 검색 결과 문서 리스트
        """
        try:
            # Tavily Tool 호출
            results = self.tavily_tool.invoke({"query": query})

            # 결과를 LangChain Document 형태로 변환
            web_docs = []

            if isinstance(results, list):
                for i, result in enumerate(results):
                    if isinstance(result, dict):
                        content = result.get("content", "")
                        url = result.get("url", "")
                        title = result.get("title", f"Web Result {i+1}")

                        doc = Document(
                            page_content=content,
                            metadata={
                                "source": "tavily_web_search",
                                "url": url,
                                "title": title,
                                "search_rank": i + 1,
                                "crag_relevance": "web_search",
                            },
                        )
                        web_docs.append(doc)

            return web_docs

        except Exception as e:
            print(f"[WARN] 웹 검색 실패: {e}")
            return []

    def merge_documents(
        self,
        internal_docs: List[Document],
        web_docs: List[Document],
        action: CRAGAction,
    ) -> List[Document]:
        """
        내부 검색 결과와 웹 검색 결과 병합

        Args:
            internal_docs: VectorDB 검색 결과
            web_docs: 웹 검색 결과
            action: CRAG 액션

        Returns:
            병합된 문서 리스트
        """
        if action == CRAGAction.CORRECT:
            # CORRECT: 정제된 내부 문서만 사용
            return internal_docs

        elif action == CRAGAction.INCORRECT:
            # INCORRECT: 웹 검색 결과로 대체
            return web_docs

        else:  # AMBIGUOUS
            # AMBIGUOUS: 내부 + 웹 결과 병합
            merged = []

            # 관련 있는 내부 문서 추가
            for doc in internal_docs:
                if doc.metadata.get("crag_relevance") == "relevant":
                    merged.append(doc)

            # 웹 검색 결과 추가
            merged.extend(web_docs)

            # 중복 제거 (content 기준)
            seen_contents = set()
            unique_docs = []

            for doc in merged:
                # 내용 정규화 (공백 제거하고 소문자화)
                normalized = doc.page_content.strip().lower()[:200]  # 앞 200자로 비교

                if normalized not in seen_contents:
                    seen_contents.add(normalized)
                    unique_docs.append(doc)

            return unique_docs

    def execute(self, query: str, documents: List[Document]) -> CRAGResult:
        """
        CRAG 전략 실행

        전체 워크플로우:
        1. Self-RAG 평가로 액션 결정
        2. 액션에 따라 문서 정제/웹검색/병합
        3. 최종 문서 반환

        Args:
            query: 사용자 질문
            documents: VectorDB 검색 결과

        Returns:
            CRAGResult (액션, 최종 문서, 메타정보)
        """
        original_count = len(documents)

        # 1단계: 액션 결정
        action, reason = self.decide_action(query, documents)

        # 2단계: 액션 실행
        web_search_performed = False

        if action == CRAGAction.CORRECT:
            # 문서 정제만 수행
            final_docs = self.refine_documents(query, documents)

        elif action == CRAGAction.INCORRECT:
            # 웹 검색으로 대체
            web_docs = self.web_search(query)
            final_docs = web_docs
            web_search_performed = True

        else:  # AMBIGUOUS
            # 내부 문서 정제 + 웹 검색 보완
            refined_internal = self.refine_documents(query, documents)
            web_docs = self.web_search(query)
            final_docs = self.merge_documents(refined_internal, web_docs, action)
            web_search_performed = True

        return CRAGResult(
            action=action,
            documents=final_docs,
            reason=reason,
            web_search_performed=web_search_performed,
            original_doc_count=original_count,
            final_doc_count=len(final_docs),
        )


# 편의 함수
def create_corrective_rag(
    relevance_threshold: float = 0.5,
    min_relevant_docs: int = 2,
    max_web_results: int = 3,
) -> CorrectiveRAG:
    """
    CRAG 전략 생성

    Example:
        >>> crag = create_corrective_rag()
        >>> result = crag.execute(
        ...     "대사증후군의 최신 치료법은?",
        ...     vector_db_documents
        ... )
        >>> print(result.action)  # CRAGAction.AMBIGUOUS
        >>> print(result.web_search_performed)  # True
        >>> print(len(result.documents))  # 5
    """
    return CorrectiveRAG(
        relevance_threshold=relevance_threshold,
        min_relevant_docs=min_relevant_docs,
        max_web_results=max_web_results,
    )
