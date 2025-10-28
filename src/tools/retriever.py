# src/tools/retriever.py
"""
Internal Knowledge Base Retriever Tool

HybridRetriever를 LangChain Tool로 래핑
"""

from langchain.tools import tool
from typing import List


@tool
def internal_retriever_tool(query: str, k: int = 5) -> str:
    """
    대사증후군 관련 내부 지식베이스 검색 도구

    Args:
        query: 검색할 질문 또는 키워드
        k: 반환할 문서 수 (기본값 5)

    Returns:
        검색된 문서 내용 (문자열로 병합)
    """
    from src.data.vector_store import get_cached_hybrid_retriever

    retriever = get_cached_hybrid_retriever()
    docs = retriever.search(query, k=k)

    if not docs:
        return "관련 문서를 찾을 수 없습니다."

    # 문서 내용을 문자열로 병합
    result = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("basename", "Unknown")
        content = doc.page_content
        result.append(f"[문서 {i}] 출처: {source}\n{content}\n")

    return "\n".join(result)
