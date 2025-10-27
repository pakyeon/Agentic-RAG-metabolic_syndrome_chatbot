"""
CRAG (Corrective RAG) 전략 테스트
"""

import sys
import os
import pytest

if not os.getenv("OPENAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
    pytest.skip(
        "CRAG 테스트는 OpenAI 및 Tavily API 키가 필요하여 키가 없으면 건너뜁니다.",
        allow_module_level=True,
    )

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.documents import Document
from src.strategies import create_corrective_rag, CRAGAction


def create_sample_documents(relevant: int, irrelevant: int) -> list:
    """테스트용 샘플 문서 생성"""
    docs = []

    # 관련 있는 문서
    for i in range(relevant):
        docs.append(
            Document(
                page_content=f"""
            대사증후군 관련 정보 {i+1}:
            대사증후군은 복부 비만, 고혈압, 고혈당, 이상지질혈증 등이 
            복합적으로 나타나는 상태입니다. 진단 기준은 5가지 항목 중 
            3가지 이상을 충족할 때입니다.
            """,
                metadata={"doc_id": f"relevant_{i+1}"},
            )
        )

    # 관련 없는 문서
    for i in range(irrelevant):
        docs.append(
            Document(
                page_content=f"""
            무관한 정보 {i+1}:
            감기는 바이러스 감염으로 인한 상기도 감염입니다.
            주요 증상으로는 콧물, 기침, 인후통 등이 있습니다.
            """,
                metadata={"doc_id": f"irrelevant_{i+1}"},
            )
        )

    return docs


def test_correct_action():
    """CORRECT 액션 테스트 (충분한 관련 문서)"""
    print("=" * 60)
    print("CORRECT 액션 테스트")
    print("=" * 60)

    crag = create_corrective_rag(min_relevant_docs=2)
    query = "대사증후군의 진단 기준은 무엇인가요?"

    # 관련 문서 4개 (충분함)
    documents = create_sample_documents(relevant=4, irrelevant=0)

    print(f"\n질문: {query}")
    print(f"검색된 문서: {len(documents)}개")

    result = crag.execute(query, documents)

    print(f"\n액션: {result.action.value}")
    print(f"이유: {result.reason}")
    print(f"웹 검색 수행: {result.web_search_performed}")
    print(f"원본 문서 수: {result.original_doc_count}개")
    print(f"최종 문서 수: {result.final_doc_count}개")

    assert result.action == CRAGAction.CORRECT
    assert not result.web_search_performed
    print("\n✅ CORRECT 액션 테스트 통과")


def test_incorrect_action():
    """INCORRECT 액션 테스트 (관련 문서 없음)"""
    print("\n" + "=" * 60)
    print("INCORRECT 액션 테스트")
    print("=" * 60)

    crag = create_corrective_rag(min_relevant_docs=2)
    query = "대사증후군의 진단 기준은 무엇인가요?"

    # 관련 문서 0개 (모두 무관함)
    documents = create_sample_documents(relevant=0, irrelevant=3)

    print(f"\n질문: {query}")
    print(f"검색된 문서: {len(documents)}개")

    result = crag.execute(query, documents)

    print(f"\n액션: {result.action.value}")
    print(f"이유: {result.reason}")
    print(f"웹 검색 수행: {result.web_search_performed}")
    print(f"원본 문서 수: {result.original_doc_count}개")
    print(f"최종 문서 수: {result.final_doc_count}개")

    if result.final_doc_count > 0:
        print(f"\n웹 검색 결과 샘플:")
        for i, doc in enumerate(result.documents[:2], 1):
            print(f"  문서 {i}: {doc.metadata.get('title', 'N/A')[:50]}...")

    assert result.action == CRAGAction.INCORRECT
    assert result.web_search_performed
    print("\n✅ INCORRECT 액션 테스트 통과")


def test_ambiguous_action():
    """AMBIGUOUS 액션 테스트 (일부만 관련)"""
    print("\n" + "=" * 60)
    print("AMBIGUOUS 액션 테스트")
    print("=" * 60)

    crag = create_corrective_rag(min_relevant_docs=2)
    query = "대사증후군의 진단 기준은 무엇인가요?"

    # 관련 문서 1개 (불충분), 무관 문서 2개
    documents = create_sample_documents(relevant=1, irrelevant=2)

    print(f"\n질문: {query}")
    print(f"검색된 문서: {len(documents)}개")

    result = crag.execute(query, documents)

    print(f"\n액션: {result.action.value}")
    print(f"이유: {result.reason}")
    print(f"웹 검색 수행: {result.web_search_performed}")
    print(f"원본 문서 수: {result.original_doc_count}개")
    print(f"최종 문서 수: {result.final_doc_count}개")

    # 최종 문서 소스 분석
    internal_count = sum(
        1
        for doc in result.documents
        if doc.metadata.get("source") != "tavily_web_search"
    )
    web_count = sum(
        1
        for doc in result.documents
        if doc.metadata.get("source") == "tavily_web_search"
    )

    print(f"\n최종 문서 구성:")
    print(f"  내부 문서: {internal_count}개")
    print(f"  웹 문서: {web_count}개")

    assert result.action == CRAGAction.AMBIGUOUS
    assert result.web_search_performed
    assert web_count > 0  # 웹 검색 결과가 포함되어야 함
    print("\n✅ AMBIGUOUS 액션 테스트 통과")


def test_empty_documents():
    """빈 문서 리스트 테스트"""
    print("\n" + "=" * 60)
    print("빈 문서 리스트 테스트")
    print("=" * 60)

    crag = create_corrective_rag()
    query = "대사증후군이란 무엇인가요?"
    documents = []

    print(f"\n질문: {query}")
    print(f"검색된 문서: {len(documents)}개")

    result = crag.execute(query, documents)

    print(f"\n액션: {result.action.value}")
    print(f"이유: {result.reason}")
    print(f"웹 검색 수행: {result.web_search_performed}")
    print(f"최종 문서 수: {result.final_doc_count}개")

    assert result.action == CRAGAction.INCORRECT
    assert result.web_search_performed
    print("\n✅ 빈 문서 리스트 테스트 통과")


def test_document_refinement():
    """문서 정제 기능 테스트"""
    print("\n" + "=" * 60)
    print("문서 정제 기능 테스트")
    print("=" * 60)

    crag = create_corrective_rag(min_relevant_docs=2)
    query = "대사증후군의 진단 기준은?"

    # 관련 3개 + 무관 2개
    documents = create_sample_documents(relevant=3, irrelevant=2)

    print(f"\n질문: {query}")
    print(f"원본 문서: {len(documents)}개")

    result = crag.execute(query, documents)

    print(f"\n액션: {result.action.value}")
    print(f"정제 후 문서: {result.final_doc_count}개")

    # 무관한 문서가 제거되었는지 확인
    irrelevant_docs = [
        doc
        for doc in result.documents
        if "irrelevant" in doc.metadata.get("doc_id", "")
    ]

    print(f"제거된 무관 문서: {2 - len(irrelevant_docs)}개")

    assert result.action == CRAGAction.CORRECT
    assert len(irrelevant_docs) < 2  # 일부 무관 문서가 제거되어야 함
    print("\n✅ 문서 정제 기능 테스트 통과")


if __name__ == "__main__":
    print("\n🔍 CRAG 전략 테스트 시작\n")

    try:
        test_correct_action()
        test_incorrect_action()
        test_ambiguous_action()
        test_empty_documents()
        test_document_refinement()

        print("\n" + "=" * 60)
        print("✅ 모든 테스트 완료!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback

        traceback.print_exc()
