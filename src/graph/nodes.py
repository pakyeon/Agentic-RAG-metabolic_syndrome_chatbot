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

        return {"crag_action": action.value, "crag_confidence": 1.0}

    except Exception as e:
        return {
            "crag_action": "correct",
            "crag_confidence": 0.0,
            "error": f"CRAG 액션 결정 실패: {str(e)}",
        }


# === Task 5.4: 외부 검색 및 컨텍스트 병합 ===


def search_external_node(state: RAGState) -> Dict:
    """
    Tavily 외부 검색 노드 (Task 5.4)

    CRAG 액션이 'incorrect' 또는 'ambiguous'일 때만 실행됩니다.
    """
    try:
        from src.tools.tavily import get_tavily_tool
        from langchain_core.documents import Document

        action = state.get("crag_action", "")

        # correct 액션이면 외부 검색 스킵
        if action == "correct":
            return {"external_docs": []}

        # incorrect 또는 ambiguous일 때만 외부 검색
        if action not in ["incorrect", "ambiguous"]:
            return {"external_docs": []}

        # Tavily 검색 실행
        tavily_tool = get_tavily_tool(max_results=3)
        query = state["question"]

        result = tavily_tool.invoke({"query": query})

        # 결과 파싱
        external_docs = []

        if isinstance(result, dict):
            # Tavily Tool은 dict 형식으로 반환
            # {'query': ..., 'results': [...], 'answer': ...}
            results_list = result.get("results", [])

            for i, item in enumerate(results_list):
                content = item.get("content", "")
                url = item.get("url", "")
                title = item.get("title", f"Web Result {i+1}")
                score = item.get("score", 0.0)

                if content.strip():  # 내용이 있는 경우만
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": "tavily_web_search",
                            "url": url,
                            "title": title,
                            "query": query,
                            "rank": i + 1,
                            "score": score,
                        },
                    )
                    external_docs.append(doc)

        elif isinstance(result, str):
            # 문자열인 경우 (하위 호환성)
            if result.strip():
                doc = Document(
                    page_content=result,
                    metadata={"source": "tavily_web_search", "query": query},
                )
                external_docs.append(doc)

        elif isinstance(result, list):
            # 리스트인 경우 (하위 호환성)
            for i, item in enumerate(result):
                if isinstance(item, dict):
                    content = item.get("content", str(item))
                    url = item.get("url", "")
                    title = item.get("title", f"Web Result {i+1}")

                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": "tavily_web_search",
                            "url": url,
                            "title": title,
                            "query": query,
                            "rank": i + 1,
                        },
                    )
                    external_docs.append(doc)
                else:
                    doc = Document(
                        page_content=str(item),
                        metadata={
                            "source": "tavily_web_search",
                            "query": query,
                            "rank": i + 1,
                        },
                    )
                    external_docs.append(doc)

        return {"external_docs": external_docs}

    except Exception as e:
        import traceback

        error_msg = f"외부 검색 실패: {str(e)}\n{traceback.format_exc()}"
        return {"external_docs": [], "error": error_msg}


def merge_context_node(state: RAGState) -> Dict:
    """
    내부 + 외부 문서 병합 노드 (Task 5.4)

    CRAG 액션에 따라 다른 병합 전략을 적용합니다:
    - correct: 내부 문서만 사용
    - incorrect: 외부 문서로 교체
    - ambiguous: 관련 있는 내부 + 외부 병합
    """
    try:
        action = state.get("crag_action", "correct")
        internal_docs = state.get("internal_docs", [])
        external_docs = state.get("external_docs", [])
        relevance_scores = state.get("relevance_scores", [])

        final_docs = []

        if action == "correct":
            # CORRECT: 내부 문서만 사용
            final_docs = internal_docs

        elif action == "incorrect":
            # INCORRECT: 외부 문서로 완전 교체
            final_docs = external_docs

        else:  # ambiguous
            # AMBIGUOUS: 관련 있는 내부 + 외부 병합

            # 1. 관련 있는 내부 문서 필터링
            relevant_internal = []
            for i, doc in enumerate(internal_docs):
                if i < len(relevance_scores):
                    score = relevance_scores[i]
                    # RelevanceResult 객체인 경우
                    if hasattr(score, "relevance"):
                        if score.relevance == "relevant":
                            relevant_internal.append(doc)
                    # 단순 점수인 경우 (숫자)
                    elif isinstance(score, (int, float)):
                        if score >= 3.0:  # 3점 이상을 관련 있음으로 판단
                            relevant_internal.append(doc)

            # 2. 내부 + 외부 병합
            final_docs = relevant_internal + external_docs

            # 3. 중복 제거 (content 기준, 앞 200자 비교)
            seen_contents = set()
            unique_docs = []

            for doc in final_docs:
                normalized = doc.page_content.strip().lower()[:200]
                if normalized not in seen_contents:
                    seen_contents.add(normalized)
                    unique_docs.append(doc)

            final_docs = unique_docs

        # 최종 컨텍스트 문자열 생성
        context_parts = []
        for i, doc in enumerate(final_docs, 1):
            source = doc.metadata.get("source", "N/A")
            title = doc.metadata.get("title", "")
            basename = doc.metadata.get("basename", "")

            # 출처 표시
            if source == "tavily_web_search":
                header = f"[외부 문서 {i}] {title}"
            else:
                header = f"[내부 문서 {i}] {basename or 'N/A'}"

            context_parts.append(f"{header}\n{doc.page_content}")

        merged_context = "\n\n".join(context_parts)

        return {"merged_context": merged_context}

    except Exception as e:
        return {"merged_context": "", "error": f"컨텍스트 병합 실패: {str(e)}"}
