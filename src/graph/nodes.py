# -*- coding: utf-8 -*-
"""Agentic RAG 그래프 노드 함수들."""

from __future__ import annotations

import os
from typing import Dict, List

from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from src.graph.state import RAGState
from src.memory import get_graphiti_connector, get_short_term_store


# === 환자 컨텍스트 로드 ===
def load_patient_context_node(state: RAGState) -> Dict:
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
                "error": f"환자 ID '{patient_id}' 정보를 찾을 수 없습니다",
            }

        return {"patient_context": context}

    except Exception as exc:  # pragma: no cover - defensive path
        return {"patient_context": "", "error": f"환자 정보 로드 실패: {exc}"}


def load_memory_context_node(state: RAGState) -> Dict:
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
        "long_term_memory": [],  # Graphiti는 도구 호출 시 사용
        "metadata": metadata,
    }


# === Self-RAG [Retrieve] 토큰: 검색 필요성 판단 ===
def should_retrieve_node(state: RAGState) -> Dict:
    """검색이 필요한지 판단 (Self-RAG [Retrieve] 토큰)."""
    try:
        from src.evaluation.self_rag_evaluator import SelfRAGEvaluator

        evaluator = SelfRAGEvaluator()
        result = evaluator.evaluate_retrieve_need(state["question"])

        should_retrieve = result.should_retrieve == "yes"
        documents_to_evaluate = result.documents_to_evaluate if should_retrieve else 0

        return {
            "should_retrieve": should_retrieve,
            "iteration": state.get("iteration", 0) + 1,
            "retrieve_difficulty": result.difficulty,
            "evaluation_doc_limit": documents_to_evaluate,
            "metadata": {
                **state.get("metadata", {}),
                "retrieve_decision": result.should_retrieve,
                "retrieve_reason": result.reason,
                "retrieve_difficulty": result.difficulty,
                "retrieve_doc_limit": documents_to_evaluate,
            },
        }

    except Exception as exc:
        return {
            "should_retrieve": True,
            "iteration": state.get("iteration", 0) + 1,
            "error": f"검색 필요성 평가 실패: {exc}",
        }


# === 내부 검색 (VectorDB Hybrid Search) ===
def retrieve_internal_node(state: RAGState) -> Dict:
    """VectorDB에서 관련 문서 검색."""
    try:
        from src.data.vector_store import get_cached_hybrid_retriever
        from src.data.path_utils import (
            DEFAULT_PERSIST_DIRECTORY,
            DEFAULT_PARSED_DIRECTORY,
            DEFAULT_RAW_DIRECTORY,
        )

        retriever = get_cached_hybrid_retriever(
            persist_directory=DEFAULT_PERSIST_DIRECTORY,
            parsed_dir=DEFAULT_PARSED_DIRECTORY,
            raw_dir=DEFAULT_RAW_DIRECTORY,
        )

        query = state["question"]
        if state.get("patient_context"):
            query = f"{state['patient_context']}\n\n질문: {state['question']}"

        docs = retriever.search(query, k=5)

        return {"internal_docs": docs}

    except Exception as exc:
        return {"internal_docs": [], "error": f"내부 검색 실패: {exc}"}


# === Self-RAG ISREL: 검색 품질 평가 ===
def evaluate_retrieval_node(state: RAGState) -> Dict:
    """검색된 문서의 관련성 평가 (Self-RAG ISREL)."""
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

    except Exception as exc:
        return {"relevance_scores": [], "error": f"검색 품질 평가 실패: {exc}"}


# === CRAG: 액션 결정 ===
def decide_crag_action_node(state: RAGState) -> Dict:
    """CRAG 전략에 따른 액션 결정."""
    try:
        from src.strategies.corrective_rag import CorrectiveRAG

        strategy = CorrectiveRAG()
        docs = state.get("internal_docs", [])

        if not docs:
            return {"crag_action": "incorrect", "crag_confidence": 0.0}

        action, _ = strategy.decide_action(
            query=state["question"],
            documents=docs,
            documents_to_evaluate=state.get("evaluation_doc_limit") or None,
        )

        return {"crag_action": action.value, "crag_confidence": 1.0}

    except Exception as exc:
        return {
            "crag_action": "correct",
            "crag_confidence": 0.0,
            "error": f"CRAG 액션 결정 실패: {exc}",
        }


# === Task 5.4: 외부 검색 및 컨텍스트 병합 ===
def search_external_node(state: RAGState) -> Dict:
    """
    Tavily 외부 검색 노드 (Task 5.4).

    CRAG 액션이 'incorrect' 또는 'ambiguous'일 때만 실행된다.
    """
    try:
        from src.tools.tavily import get_tavily_tool
        from langchain_core.documents import Document

        action = state.get("crag_action", "")

        if action == "correct":
            return {"external_docs": []}

        if action not in ["incorrect", "ambiguous"]:
            return {"external_docs": []}

        tavily_tool = get_tavily_tool(max_results=3)
        query = state["question"]

        result = tavily_tool.invoke({"query": query})

        external_docs: List[Document] = []
        if isinstance(result, dict):
            for i, item in enumerate(result.get("results", [])):
                content = item.get("content", "")
                url = item.get("url", "")
                title = item.get("title", f"Web Result {i+1}")
                score = item.get("score", 0.0)

                if content.strip():
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
            if result.strip():
                doc = Document(
                    page_content=result,
                    metadata={"source": "tavily_web_search", "query": query},
                )
                external_docs.append(doc)
        elif isinstance(result, list):
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

    except Exception as exc:
        import traceback

        error_msg = f"외부 검색 실패: {exc}\n{traceback.format_exc()}"
        return {"external_docs": [], "error": error_msg}


def merge_context_node(state: RAGState) -> Dict:
    """
    내부 + 외부 문서 병합 노드 (Task 5.4).

    CRAG 액션에 따라 다른 병합 전략을 적용한다.
    """
    try:
        action = state.get("crag_action", "correct")
        internal_docs = state.get("internal_docs", [])
        external_docs = state.get("external_docs", [])
        relevance_scores = state.get("relevance_scores", [])

        final_docs = []

        if action == "correct":
            final_docs = internal_docs
        elif action == "incorrect":
            final_docs = external_docs
        else:
            relevant_internal = []
            for i, doc in enumerate(internal_docs):
                if i < len(relevance_scores):
                    score = relevance_scores[i]
                    if hasattr(score, "relevance"):
                        if score.relevance == "relevant":
                            relevant_internal.append(doc)
                    elif isinstance(score, (int, float)) and score >= 3.0:
                        relevant_internal.append(doc)

            final_docs = relevant_internal + external_docs

            seen_contents = set()
            unique_docs = []

            for doc in final_docs:
                normalized = doc.page_content.strip().lower()[:200]
                if normalized not in seen_contents:
                    seen_contents.add(normalized)
                    unique_docs.append(doc)

            final_docs = unique_docs

        context_parts = []
        for i, doc in enumerate(final_docs, 1):
            source = doc.metadata.get("source", "N/A")
            title = doc.metadata.get("title", "")
            basename = doc.metadata.get("basename", "")

            if source == "tavily_web_search":
                header = f"[외부 문서 {i}] {title}"
            else:
                header = f"[내부 문서 {i}] {basename or 'N/A'}"

            context_parts.append(f"{header}\n{doc.page_content}")

        merged_context = "\n\n".join(context_parts)

        return {"merged_context": merged_context}

    except Exception as exc:
        return {"merged_context": "", "error": f"컨텍스트 병합 실패: {exc}"}


def generate_answer_node(state: RAGState) -> Dict:
    """
    LLM으로 최종 답변 생성 노드 (Task 5.5 + Task 6.2).

    단기 기억은 최근 3턴 원문 + 요약으로 구성되고,
    장기 기억은 Graphiti MCP 도구 호출을 통해 필요 시 사용된다.
    """
    try:
        session_id = state.get("memory_session_id") or "default"
        question = state["question"]
        merged_context = state.get("merged_context", "")
        patient_context = state.get("patient_context", "")
        short_term_memory = state.get("short_term_memory", [])

        connector = get_graphiti_connector()
        graphiti_tools = (
            connector.build_tools(session_id) if connector.is_enabled else []
        )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY가 설정되지 않아 답변을 생성할 수 없습니다."
            )

        llm = ChatOpenAI(
            model="gpt-5-mini",
            temperature=0.3,
            reasoning_effort="minimal",
            api_key=api_key,
        )

        system_prompt = """당신은 대사증후군 상담사를 어시스턴트하는 전문 AI입니다.

**역할:**
- 상담사가 환자 상담 시 필요한 전문 정보와 가이드라인 제공
- 대사증후군 관련 진단, 치료, 예방, 관리에 대한 근거 있는 정보 제공
- 생활습관 개선(식단, 운동, 금연, 스트레스 관리 등)에 대한 구체적 조언

**도구 사용 지침:**
- 장기 기억이 필요하면 `graphiti_search_memories` 도구를 호출하여 관련 정보를 탐색하세요.
- 답변 후에는 동일한 대화를 `graphiti_upsert_memory`로 저장해 향후 상담에 활용하세요.

**답변 원칙:**
1. 제공된 컨텍스트를 바탕으로 정확하고 근거 있는 답변
2. 상담사가 환자에게 설명하기 쉽도록 명확하고 구조화된 정보 제공
3. 환자 정보가 있으면 개인 맞춤형 조언 제시
4. 의학 용어 사용 시 간단한 설명 추가
5. 필요 시 추가 검사나 전문의 상담 권장 사항 포함"""

        context_segments: List[str] = []
        if patient_context:
            context_segments.append(f"**환자 정보:**\n{patient_context}")
        if short_term_memory:
            context_segments.append(
                "**단기 기억:**\n"
                + "\n\n".join(str(item) for item in short_term_memory)
            )
        if merged_context:
            context_segments.append(f"**참고 자료:**\n{merged_context}")

        context_payload = (
            "\n\n".join(context_segments) if context_segments else "컨텍스트 없음"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("system", "{context}"),
                ("human", "{question}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )

        if graphiti_tools:
            agent = create_agent(llm, graphiti_tools, prompt)
            response = agent.invoke(
                {
                    "context": context_payload,
                    "question": question,
                }
            )
            answer = response.get("output", "")
        else:
            messages = prompt.format_messages(
                context=context_payload,
                question=question,
                agent_scratchpad=[],
            )
            answer = llm.invoke(messages).content

        return {"answer": answer}

    except Exception as exc:
        import traceback

        error_msg = f"답변 생성 실패: {exc}\n{traceback.format_exc()}"
        return {
            "answer": "죄송합니다. 답변 생성 중 오류가 발생했습니다.",
            "error": error_msg,
        }


def evaluate_answer_node(state: RAGState) -> Dict:
    """
    Self-RAG 답변 평가 노드 (Task 5.5).

    ISSUP (지원도)와 ISUSE (유용성)를 평가하여 답변 품질을 판단하고,
    평가 결과를 단기 기억 저장소에 기록한다.
    """
    try:
        from src.evaluation.self_rag_evaluator import create_evaluator

        question = state["question"]
        answer = state.get("answer", "")
        merged_context = state.get("merged_context", "")
        iteration = state.get("iteration", 1)
        max_iterations = state.get("max_iterations", 2)

        if not answer or not merged_context:
            return {
                "support_score": 0.0,
                "usefulness_score": 0.0,
                "needs_regeneration": True,
                "short_term_memory": state.get("short_term_memory", []),
            }

        evaluator = create_evaluator()

        context_parts = merged_context.split("\n\n")
        documents = []
        for part in context_parts:
            lines = part.split("\n", 1)
            if len(lines) > 1:
                documents.append(lines[1])
            else:
                documents.append(part)

        documents = [doc for doc in documents if doc.strip()]

        if not documents:
            return {
                "support_score": 0.0,
                "usefulness_score": 0.0,
                "needs_regeneration": True,
            }

        answer_quality = evaluator.assess_answer_quality(
            query=question,
            answer=answer,
            documents=documents,
        )

        support_results = answer_quality["support_results"]
        usefulness_result = answer_quality["usefulness"]

        support_score = 0.0
        if support_results:
            support_scores = []
            for item in support_results:
                if item.support == "fully_supported":
                    support_scores.append(5.0)
                elif item.support == "partially_supported":
                    support_scores.append(3.0)
                else:
                    support_scores.append(1.0)
            support_score = sum(support_scores) / len(support_scores)

        usefulness_score = float(usefulness_result.score)

        needs_regeneration = False
        if (
            support_score < 3.0 or usefulness_score < 3.0
        ) and iteration < max_iterations:
            needs_regeneration = True

        session_id = state.get("memory_session_id") or "default"
        store = get_short_term_store()
        store.record_interaction(
            session_id=session_id,
            question=question,
            answer=answer,
            metadata={
                "support_score": support_score,
                "usefulness_score": usefulness_score,
                "iteration": iteration,
                "patient_id": state.get("patient_id"),
            },
        )
        updated_context = store.get_context(session_id)
        short_term_segments: List[str] = []
        short_term_segments.extend(updated_context.recent)
        if updated_context.history_summary:
            short_term_segments.append(updated_context.history_summary)
        short_term_segments.extend(updated_context.topic_summaries)

        return {
            "support_score": support_score,
            "usefulness_score": usefulness_score,
            "needs_regeneration": needs_regeneration,
            "short_term_memory": short_term_segments,
        }

    except Exception as exc:
        import traceback

        error_msg = f"답변 평가 실패: {exc}\n{traceback.format_exc()}"
        return {
            "support_score": 0.0,
            "usefulness_score": 0.0,
            "needs_regeneration": False,
            "error": error_msg,
        }
