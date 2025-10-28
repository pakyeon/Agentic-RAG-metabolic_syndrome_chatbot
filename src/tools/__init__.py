# src/tools/__init__.py
"""
Tools 패키지

LangGraph Agentic RAG에서 사용할 도구들
"""

from .tavily import get_tavily_tool, get_default_tool
from .retriever import internal_retriever_tool
from .patient import patient_context_tool


def get_all_tools(include_tavily: bool = True):
    """
    모든 도구 반환

    Args:
        include_tavily: Tavily 도구 포함 여부 (기본값 True)

    Returns:
        Tool 리스트
    """
    tools = [
        internal_retriever_tool,
        patient_context_tool,
    ]

    if include_tavily:
        tools.append(get_default_tool())

    return tools


__all__ = [
    "get_tavily_tool",
    "get_default_tool",
    "internal_retriever_tool",
    "patient_context_tool",
    "get_all_tools",
]
