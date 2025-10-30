"""
Self-RAG 평가자 테스트 및 예제
"""

import sys
import os
import pytest

if not os.getenv("OPENAI_API_KEY"):
    pytest.skip("OpenAI API 키가 설정되어 있지 않아 Self-RAG 평가자 테스트를 건너뜁니다.", allow_module_level=True)

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.self_rag_evaluator import create_evaluator, SelfRAGEvaluator


def test_retrieve():
    """[Retrieve] 평가 테스트"""
    print("=" * 60)
    print("[Retrieve] 검색 필요성 평가 테스트")
    print("=" * 60)

    evaluator = create_evaluator()

    # 검색이 필요한 질문들
    queries_need_retrieval = [
        "대사증후군의 최신 치료 가이드라인은?",
        "2024년 대사증후군 유병률 통계는?",
        "대사증후군 진단 기준 5가지는 무엇인가요?",
    ]

    # 검색이 불필요한 질문들
    queries_no_retrieval = [
        "안녕하세요",
        "오늘 날씨 어때요?",
        "1+1은?",
    ]

    print("\n[검색 필요한 질문들]")
    for query in queries_need_retrieval:
        result = evaluator.evaluate_retrieve_need(query)
        print(f"\n질문: {query}")
        print(f"  판단: {result.should_retrieve}")
        print(f"  난이도: {result.difficulty}")
        print(f"  평가 문서 수: {result.documents_to_evaluate}")
        print(f"  이유: {result.reason}")

    print("\n\n[검색 불필요한 질문들]")
    for query in queries_no_retrieval:
        result = evaluator.evaluate_retrieve_need(query)
        print(f"\n질문: {query}")
        print(f"  판단: {result.should_retrieve}")
        print(f"  난이도: {result.difficulty}")
        print(f"  평가 문서 수: {result.documents_to_evaluate}")
        print(f"  이유: {result.reason}")


def test_isrel():
    """ISREL (관련성) 평가 테스트"""
    print("=" * 60)
    print("ISREL (관련성) 평가 테스트")
    print("=" * 60)

    evaluator = create_evaluator()

    query = "대사증후군의 진단 기준은 무엇인가요?"

    # 관련 있는 문서
    relevant_doc = """
    대사증후군은 다음 5가지 위험 요인 중 3가지 이상을 충족할 때 진단됩니다:
    1. 복부 비만 (허리둘레: 남성 90cm, 여성 85cm 이상)
    2. 고혈압 (130/85 mmHg 이상)
    3. 공복혈당 장애 (100 mg/dL 이상)
    4. 고중성지방혈증 (150 mg/dL 이상)
    5. 낮은 HDL 콜레스테롤 (남성 40mg/dL, 여성 50mg/dL 미만)
    """

    # 관련 없는 문서
    irrelevant_doc = """
    감기는 바이러스 감염으로 인한 상기도 감염입니다.
    주요 증상으로는 콧물, 기침, 인후통 등이 있습니다.
    """

    print(f"\n질문: {query}\n")

    results = evaluator.evaluate_relevance_batch(
        query, [relevant_doc, irrelevant_doc]
    )

    print("관련 있는 문서 평가:")
    print(f"  결과: {results[0].relevance} (신뢰도: {results[0].confidence})")

    print("\n관련 없는 문서 평가:")
    print(f"  결과: {results[1].relevance} (신뢰도: {results[1].confidence})")


def test_issup():
    """ISSUP (지지도) 평가 테스트"""
    print("\n" + "=" * 60)
    print("ISSUP (지지도) 평가 테스트")
    print("=" * 60)

    evaluator = create_evaluator()

    query = "대사증후군의 진단 기준은?"

    document = """
    대사증후군 진단 기준:
    - 허리둘레: 남성 90cm, 여성 85cm 이상
    - 혈압: 130/85 mmHg 이상
    - 공복혈당: 100 mg/dL 이상
    """

    # 완전히 뒷받침되는 답변
    fully_supported_answer = """
    대사증후군은 허리둘레(남성 90cm, 여성 85cm 이상), 
    혈압(130/85 mmHg 이상), 공복혈당(100 mg/dL 이상) 등의 
    기준으로 진단됩니다.
    """

    # 부분적으로 뒷받침되는 답변
    partially_supported_answer = """
    대사증후군은 허리둘레와 혈압으로 진단되며, 
    최근 연구에 따르면 운동 부족도 주요 원인입니다.
    """

    # 뒷받침되지 않는 답변
    no_support_answer = """
    대사증후군은 유전적 요인이 가장 크며, 
    특별한 약물 치료가 필요합니다.
    """

    print(f"\n질문: {query}\n")

    print("완전히 뒷받침되는 답변 평가:")
    result1 = evaluator.evaluate_support_batch(
        query, [document], fully_supported_answer
    )[0]
    print(f"  결과: {result1.support} (신뢰도: {result1.confidence})")

    print("\n부분적으로 뒷받침되는 답변 평가:")
    result2 = evaluator.evaluate_support_batch(
        query, [document], partially_supported_answer
    )[0]
    print(f"  결과: {result2.support} (신뢰도: {result2.confidence})")

    print("\n뒷받침되지 않는 답변 평가:")
    result3 = evaluator.evaluate_support_batch(query, [document], no_support_answer)[0]
    print(f"  결과: {result3.support} (신뢰도: {result3.confidence})")


def test_isuse():
    """ISUSE (유용성) 평가 테스트"""
    print("\n" + "=" * 60)
    print("ISUSE (유용성) 평가 테스트")
    print("=" * 60)

    evaluator = create_evaluator()

    query = "대사증후군을 예방하려면 어떻게 해야 하나요?"

    # 매우 유용한 답변 (5점 예상)
    excellent_answer = """
    대사증후군 예방을 위해서는:
    1. 규칙적인 운동 (주 150분 이상)
    2. 건강한 식습관 (채소, 과일 중심)
    3. 체중 관리 (적정 BMI 유지)
    4. 금연 및 절주
    5. 정기적인 건강검진
    을 실천하시는 것이 중요합니다.
    """

    # 보통 답변 (3점 예상)
    average_answer = """
    대사증후군을 예방하려면 건강한 생활습관이 중요합니다.
    운동과 식이조절을 하시면 됩니다.
    """

    # 유용하지 않은 답변 (1-2점 예상)
    poor_answer = """
    대사증후군은 복잡한 질환입니다.
    """

    print(f"\n질문: {query}\n")

    documents = [
        "대사증후군 예방에는 규칙적인 운동, 건강한 식습관, 체중 관리가 중요합니다.",
        "대사증후군은 복부 비만과 관련이 있습니다.",
    ]

    print("매우 유용한 답변 평가:")
    result1 = evaluator.evaluate_answer_quality(query, excellent_answer, documents)
    print(f"  점수: {result1.usefulness_score}/5 (신뢰도: {result1.usefulness_confidence})")

    print("\n보통 답변 평가:")
    result2 = evaluator.evaluate_answer_quality(query, average_answer, documents)
    print(f"  점수: {result2.usefulness_score}/5 (신뢰도: {result2.usefulness_confidence})")

    print("\n유용하지 않은 답변 평가:")
    result3 = evaluator.evaluate_answer_quality(query, poor_answer, documents)
    print(f"  점수: {result3.usefulness_score}/5 (신뢰도: {result3.usefulness_confidence})")


def test_overall_evaluation():
    """전체 평가 워크플로우 테스트"""
    print("\n" + "=" * 60)
    print("전체 평가 워크플로우 테스트")
    print("=" * 60)

    evaluator = create_evaluator()

    query = "대사증후군의 주요 원인은 무엇인가요?"

    documents = [
        "대사증후군의 주요 원인은 비만, 운동 부족, 불규칙한 식습관입니다.",
        "대사증후군은 복부 비만과 인슐린 저항성이 핵심 원인입니다.",
        "감기 예방을 위해서는 손씻기가 중요합니다.",  # 관련 없는 문서
    ]

    print(f"\n질문: {query}")
    print(f"검색된 문서 수: {len(documents)}")

    # 1단계: 문서 평가
    overall_eval = evaluator.assess_retrieval_quality(query, documents)

    print(f"\n외부 검색 필요 여부: {overall_eval.should_retrieve_external}")
    print(f"사유: {overall_eval.reason}")

    print("\n문서별 평가:")
    for i, doc_eval in enumerate(overall_eval.document_evaluations, 1):
        print(f"  문서 {i}: {doc_eval.relevance.relevance}")

    # 2단계: 답변 품질 평가 (가정: 답변이 생성되었다고 가정)
    generated_answer = """
    대사증후군의 주요 원인은 복부 비만, 운동 부족, 불규칙한 식습관입니다.
    특히 인슐린 저항성이 핵심적인 역할을 합니다.
    """

    print(f"\n생성된 답변:")
    print(f"  {generated_answer}")

    answer_quality = evaluator.assess_answer_quality(
        query, generated_answer, documents[:2]  # 관련 있는 문서만 사용
    )

    fully_supported = sum(
        1 for item in answer_quality.support_results if item.support == "fully_supported"
    )
    print(f"\n답변 품질 평가:")
    print(f"  유용성: {answer_quality.usefulness_score}/5")
    print(f"  완전히 뒷받침되는 문서 수: {fully_supported}")
    print(f"  재생성 필요: {answer_quality.should_regenerate}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    print("\n🔍 Self-RAG 평가자 테스트 시작\n")

    try:
        test_retrieve()
        test_isrel()
        test_issup()
        test_isuse()
        test_overall_evaluation()

        print("\n" + "=" * 60)
        print("✅ 모든 테스트 완료!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback

        traceback.print_exc()
