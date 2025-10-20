"""
CRAG (Corrective RAG) 실전 예제

Self-RAG + CRAG를 통합한 완전한 Agentic RAG 파이프라인
"""

import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.documents import Document
from src.evaluation import create_evaluator
from src.strategies import create_corrective_rag


def simulate_vector_db_search(query: str, quality: str) -> list:
    """
    VectorDB 검색 시뮬레이션

    Args:
        query: 검색 쿼리
        quality: "good" | "poor" | "mixed"
    """
    if quality == "good":
        # 관련성 높은 문서들
        return [
            Document(
                page_content="""
                대사증후군 진단 기준 (2024 가이드라인):
                1. 복부 비만: 허리둘레 남성 90cm, 여성 85cm 이상
                2. 고혈압: 수축기 130mmHg 또는 이완기 85mmHg 이상
                3. 공복혈당 장애: 100mg/dL 이상
                4. 고중성지방혈증: 150mg/dL 이상
                5. 낮은 HDL 콜레스테롤: 남성 40mg/dL, 여성 50mg/dL 미만
                
                위 5가지 중 3가지 이상 해당 시 대사증후군으로 진단합니다.
                """,
                metadata={"source": "metabolic_guidelines_2024.pdf", "page": 12},
            ),
            Document(
                page_content="""
                대사증후군의 병태생리학적 기전:
                인슐린 저항성이 핵심 원인이며, 복부 비만과 밀접한 관련이 있습니다.
                내장지방이 증가하면 염증성 사이토카인 분비가 증가하여
                인슐린 신호전달 경로를 방해합니다.
                """,
                metadata={"source": "metabolic_pathophysiology.pdf", "page": 45},
            ),
            Document(
                page_content="""
                대사증후군 관리 전략:
                - 생활습관 개선: 체중 감량 5-10%
                - 규칙적 운동: 주 150분 이상의 중등도 유산소 운동
                - 식이요법: 저염식, 저지방식, 고섬유질 식단
                - 약물치료: 필요 시 혈압강하제, 당뇨약, 지질강하제
                """,
                metadata={"source": "metabolic_management.pdf", "page": 78},
            ),
        ]

    elif quality == "poor":
        # 관련성 낮은 문서들
        return [
            Document(
                page_content="""
                감기의 증상과 치료:
                감기는 바이러스 감염으로 인한 상기도 감염입니다.
                주요 증상으로는 콧물, 재채기, 기침, 인후통이 있습니다.
                충분한 휴식과 수분 섭취가 중요합니다.
                """,
                metadata={"source": "common_cold.pdf", "page": 5},
            ),
            Document(
                page_content="""
                골다공증 예방법:
                칼슘과 비타민 D 섭취가 중요합니다.
                규칙적인 체중부하 운동을 실시하고,
                흡연과 과도한 음주를 피해야 합니다.
                """,
                metadata={"source": "osteoporosis.pdf", "page": 23},
            ),
        ]

    else:  # mixed
        # 일부만 관련 있음
        return [
            Document(
                page_content="""
                대사증후군은 심혈관 질환의 주요 위험 요인입니다.
                복부 비만, 고혈압, 고혈당이 복합적으로 나타납니다.
                """,
                metadata={"source": "cardiovascular_risk.pdf", "page": 34},
            ),
            Document(
                page_content="""
                비타민 D 결핍 증상:
                피로, 근육통, 우울감 등이 나타날 수 있습니다.
                햇빛 노출과 비타민 D 보충제 섭취가 도움이 됩니다.
                """,
                metadata={"source": "vitamin_d.pdf", "page": 12},
            ),
        ]


def demo_scenario_1_correct():
    """
    시나리오 1: CORRECT 액션
    검색 품질이 충분하여 정제만 수행
    """
    print("=" * 70)
    print("시나리오 1: CORRECT 액션 (검색 품질 충분)")
    print("=" * 70)

    query = "대사증후군의 진단 기준은 무엇인가요?"
    print(f"\n[질문] {query}")

    # 1. VectorDB 검색 (고품질)
    print("\n[1단계] VectorDB 검색")
    documents = simulate_vector_db_search(query, quality="good")
    print(f"  검색된 문서: {len(documents)}개")

    # 2. CRAG 실행
    print("\n[2단계] CRAG 전략 실행")
    crag = create_corrective_rag()
    result = crag.execute(query, documents)

    print(f"  액션: {result.action.value.upper()}")
    print(f"  이유: {result.reason}")
    print(f"  웹 검색: {'실행됨' if result.web_search_performed else '불필요'}")
    print(f"  문서 수: {result.original_doc_count} → {result.final_doc_count}")

    # 3. 최종 결과
    print("\n[3단계] 정제된 문서")
    for i, doc in enumerate(result.documents, 1):
        print(f"  문서 {i}: {doc.metadata.get('source', 'N/A')}")

    print("\n결론: 검색 품질이 충분하여 웹 검색 없이 진행 ✅")


def demo_scenario_2_incorrect():
    """
    시나리오 2: INCORRECT 액션
    검색 품질이 낮아 웹 검색으로 대체
    """
    print("\n\n" + "=" * 70)
    print("시나리오 2: INCORRECT 액션 (검색 품질 낮음)")
    print("=" * 70)

    query = "대사증후군의 최신 치료 가이드라인 2025년"
    print(f"\n[질문] {query}")

    # 1. VectorDB 검색 (저품질)
    print("\n[1단계] VectorDB 검색")
    documents = simulate_vector_db_search(query, quality="poor")
    print(f"  검색된 문서: {len(documents)}개")

    # 2. CRAG 실행
    print("\n[2단계] CRAG 전략 실행")
    crag = create_corrective_rag()
    result = crag.execute(query, documents)

    print(f"  액션: {result.action.value.upper()}")
    print(f"  이유: {result.reason}")
    print(f"  웹 검색: {'실행됨' if result.web_search_performed else '불필요'}")
    print(f"  문서 수: {result.original_doc_count} → {result.final_doc_count}")

    # 3. 최종 결과
    print("\n[3단계] 웹 검색 결과")
    if result.documents:
        for i, doc in enumerate(result.documents[:3], 1):
            print(f"  문서 {i}: {doc.metadata.get('title', 'N/A')[:60]}...")
    else:
        print("  (웹 검색 결과 없음)")

    print("\n결론: 내부 문서가 관련 없어 웹 검색으로 대체 🔄")


def demo_scenario_3_ambiguous():
    """
    시나리오 3: AMBIGUOUS 액션
    일부만 관련 있어 웹 검색으로 보완
    """
    print("\n\n" + "=" * 70)
    print("시나리오 3: AMBIGUOUS 액션 (일부만 관련)")
    print("=" * 70)

    query = "대사증후군 환자의 운동 요법과 식이 조절"
    print(f"\n[질문] {query}")

    # 1. VectorDB 검색 (혼합 품질)
    print("\n[1단계] VectorDB 검색")
    documents = simulate_vector_db_search(query, quality="mixed")
    print(f"  검색된 문서: {len(documents)}개")

    # 2. CRAG 실행
    print("\n[2단계] CRAG 전략 실행")
    crag = create_corrective_rag()
    result = crag.execute(query, documents)

    print(f"  액션: {result.action.value.upper()}")
    print(f"  이유: {result.reason}")
    print(f"  웹 검색: {'실행됨' if result.web_search_performed else '불필요'}")
    print(f"  문서 수: {result.original_doc_count} → {result.final_doc_count}")

    # 3. 최종 결과 구성
    internal_docs = [
        doc
        for doc in result.documents
        if doc.metadata.get("source") != "tavily_web_search"
    ]
    web_docs = [
        doc
        for doc in result.documents
        if doc.metadata.get("source") == "tavily_web_search"
    ]

    print("\n[3단계] 최종 문서 구성")
    print(f"  내부 문서: {len(internal_docs)}개")
    print(f"  웹 문서: {len(web_docs)}개")
    print(f"  총합: {len(result.documents)}개")

    print("\n결론: 내부 문서가 불충분하여 웹 검색으로 보완 ➕")


def demo_full_pipeline():
    """
    완전한 Self-RAG + CRAG 파이프라인
    """
    print("\n\n" + "=" * 70)
    print("완전한 Self-RAG + CRAG 파이프라인")
    print("=" * 70)

    query = "대사증후군의 예방법과 생활습관 개선 방법"
    print(f"\n[질문] {query}")

    # Self-RAG 평가자 생성
    evaluator = create_evaluator()

    # 0단계: [Retrieve] - 검색 필요성 판단
    print("\n[0단계] Self-RAG: 검색 필요성 판단")
    retrieve_result = evaluator.evaluate_retrieve_need(query)
    print(f"  판단: {retrieve_result.decision}")
    print(f"  이유: {retrieve_result.reason}")

    if retrieve_result.decision == "no":
        print("  → LLM 직접 답변으로 진행")
        return

    # 1단계: VectorDB 검색
    print("\n[1단계] VectorDB 검색")
    documents = simulate_vector_db_search(query, quality="mixed")
    print(f"  검색 결과: {len(documents)}개")

    # 2단계: CRAG 전략 실행
    print("\n[2단계] CRAG 전략 실행")
    crag = create_corrective_rag()
    crag_result = crag.execute(query, documents)

    print(f"  액션: {crag_result.action.value.upper()}")
    print(f"  웹 검색: {'실행' if crag_result.web_search_performed else '불필요'}")
    print(f"  최종 문서: {crag_result.final_doc_count}개")

    # 3단계: LLM 답변 생성 (시뮬레이션)
    print("\n[3단계] LLM 답변 생성")
    generated_answer = """
    대사증후군 예방을 위한 생활습관 개선 방법:
    
    1. 체중 관리: 현재 체중의 5-10% 감량 목표
    2. 규칙적 운동: 주 150분 이상 중등도 유산소 운동
    3. 건강한 식습관:
       - 저염식 (하루 소금 섭취 6g 이하)
       - 저지방, 고섬유질 식단
       - 가공식품 및 당류 섭취 제한
    4. 금연 및 절주
    5. 스트레스 관리 및 충분한 수면
    """
    print(f"  답변 길이: {len(generated_answer)}자")

    # 4단계: Self-RAG 답변 품질 평가
    print("\n[4단계] Self-RAG: 답변 품질 평가")
    answer_quality = evaluator.evaluate_answer_quality(
        query, generated_answer, [doc.page_content for doc in crag_result.documents[:2]]
    )

    print(f"  ISUSE (유용성): {answer_quality['usefulness'].score}/5")
    print(
        f"  ISSUP (지지도): {answer_quality['fully_supported_count']}개 문서 완전 지지"
    )
    print(f"  재생성 필요: {'예' if answer_quality['should_regenerate'] else '아니오'}")

    # 5단계: 최종 결과
    print("\n[5단계] 최종 결과")
    if not answer_quality["should_regenerate"]:
        print("  ✅ 답변 품질 충분, 사용자에게 반환")
        print(f"\n{generated_answer}")
    else:
        print("  ⚠️ 답변 품질 낮음, 재생성 또는 추가 검색 필요")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    print("\n🎯 CRAG (Corrective RAG) 실전 예제\n")

    try:
        demo_scenario_1_correct()
        demo_scenario_2_incorrect()
        demo_scenario_3_ambiguous()
        demo_full_pipeline()

        print("\n" + "=" * 70)
        print("✅ 모든 시나리오 완료!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ 예제 실행 실패: {e}")
        import traceback

        traceback.print_exc()
