"""
Tavily Search Tool 래퍼

LangGraph Agentic RAG에서 사용할 Tavily 검색 Tool
"""

import os
from typing import Optional
from langchain_tavily import TavilySearch


def get_tavily_tool(
    api_key: Optional[str] = None, max_results: int = 5, topic: str = "general"
) -> TavilySearch:
    """
    Tavily Search Tool 생성

    Args:
        api_key: Tavily API 키 (None이면 환경변수에서 로드)
        max_results: 최대 결과 수 (기본: 5)
        topic: 검색 주제
            - "general": 일반 검색 (기본, 의료 정보 포함)
            - "news": 뉴스
            - "finance": 금융

    Returns:
        TavilySearch Tool 인스턴스

    Example:
        >>> tool = get_tavily_tool()
        >>> result = tool.invoke({"query": "대사증후군 진단 기준"})
    """
    if api_key is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError(
                "TAVILY_API_KEY가 설정되지 않았습니다. "
                ".env 파일을 확인하거나 api_key 파라미터를 전달하세요."
            )

    tool = TavilySearch(
        api_key=api_key,
        max_results=max_results,
        topic=topic,
    )

    return tool


# 기본 Tool 인스턴스 (환경변수 사용)
def get_default_tool() -> TavilySearch:
    """기본 설정으로 Tavily Tool 생성"""
    return get_tavily_tool(max_results=5, topic="general")
