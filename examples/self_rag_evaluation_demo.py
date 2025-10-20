"""
Self-RAG 평가 실전 예제

실제 RAG 시스템에서 Self-RAG 평가를 어떻게 사용하는지 보여주는 예제
"""

import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation import create_evaluator


def simulate_rag_pipeline():
    """
    실제 RAG 파이프라인을 시뮬레이션합니다.

    흐름:
    0. [Retrieve]: 검색 필요성 판단
    1. 사용자 질문
    2. VectorDB 검색
    3. ISREL: 검색 품질 평가
    4. 외부 검색 필요 판단
    5. LLM 답변 생성
    6. ISSUP & ISUSE: 답변 품질 평가
    7. 재생성 필요 판단
    """

    print("=" * 70)
    print("Self-RAG 기반 RAG 파이프라인 시뮬레이션")
    print("=" * 70)

    # 평가자 생성
    evaluator = create_evaluator()

    # 0단계: [Retrieve] - 검색 필요성 판단
    query = "대사증후군 환자에게 권장되는 식단은 무엇인가요?"
    print(f"\n[0단계] [Retrieve] - 검색 필요성 판단")
    print(f"  질문: {query}")

    retrieve_result = evaluator.evaluate_retrieve_need(query)
    print(f"  판단: {retrieve_result.decision}")
    print(f"  이유: {retrieve_result.reason}")

    if retrieve_result.decision == "no":
        print("  → LLM 자체 지식으로 답변 가능. 검색 건너뜀.")
        # 실제로는 여기서 바로 LLM에게 답변 요청
        return

    print("  → 검색 진행")

    # 1단계: 사용자 질문
    print(f"\n[1단계] 사용자 질문 확인")
    print(f"  {query}")

    # 2단계: VectorDB에서 문서 검색 (시뮬레이션)
    print(f"\n[2단계] VectorDB 검색")
    retrieved_documents = [
        """대사증후군 환자는 저염식, 저지방식을 섭취해야 합니다. 
        채소와 과일을 충분히 섭취하고, 가공식품을 피하는 것이 좋습니다.""",
        """규칙적인 운동이 대사증후군 관리에 중요합니다. 
        주 3-5회, 회당 30분 이상의 유산소 운동을 권장합니다.""",
        """대사증후군 환자는 혈당 관리가 중요합니다.
        저GI 식품을 선택하고, 식사는 규칙적으로 하세요.""",
    ]
    print(f"  검색된 문서 수: {len(retrieved_documents)}개")

    # 3단계: ISREL - 검색 품질 평가
    print(f"\n[3단계] ISREL - 검색 품질 평가")
    overall_eval = evaluator.evaluate_documents(
        query, retrieved_documents, min_relevant_docs=2
    )

    relevant_docs = []
    for i, doc_eval in enumerate(overall_eval.document_evaluations, 1):
        relevance = doc_eval.relevance.relevance
        print(f"  문서 {i}: {relevance}")
        if relevance == "relevant":
            relevant_docs.append(doc_eval.document_content)

    # 4단계: 외부 검색 필요 판단
    print(f"\n[4단계] 외부 검색 필요성 판단")
    print(f"  필요 여부: {overall_eval.should_retrieve_external}")
    print(f"  사유: {overall_eval.reason}")

    if overall_eval.should_retrieve_external:
        print("  → Tavily 외부 검색을 실행합니다...")
        # 실제로는 여기서 Tavily Tool 호출
        external_doc = (
            "외부 검색 결과: 대사증후군 식단은 DASH 식단을 따르는 것이 효과적입니다."
        )
        relevant_docs.append(external_doc)
        print(f"  외부 문서 추가: {len(relevant_docs)}개 → {len(relevant_docs)}개")

    # 5단계: LLM 답변 생성 (시뮬레이션)
    print(f"\n[5단계] LLM 답변 생성")
    generated_answer = """
    대사증후군 환자에게 권장되는 식단은 다음과 같습니다:
    
    1. 저염식, 저지방식 섭취
    2. 채소와 과일 충분히 섭취
    3. 가공식품 피하기
    4. 저GI(혈당지수) 식품 선택
    5. 규칙적인 식사
    
    이러한 식습관과 함께 주 3-5회, 회당 30분 이상의 
    유산소 운동을 병행하면 더욱 효과적입니다.
    """
    print(f"  답변 생성 완료")
    print(f"  답변 길이: {len(generated_answer)}자")

    # 6단계: ISSUP & ISUSE - 답변 품질 평가
    print(f"\n[6단계] ISSUP & ISUSE - 답변 품질 평가")
    answer_quality = evaluator.evaluate_answer_quality(
        query, generated_answer, relevant_docs
    )

    print(f"  ISSUP (지지도):")
    for i, support_result in enumerate(answer_quality["support_results"], 1):
        print(f"    문서 {i}: {support_result.support}")

    print(f"  ISUSE (유용성): {answer_quality['usefulness'].score}/5")
    print(f"  완전히 뒷받침되는 문서 수: {answer_quality['fully_supported_count']}")

    # 7단계: 재생성 필요 판단
    print(f"\n[7단계] 재생성 필요성 판단")
    should_regenerate = answer_quality["should_regenerate"]
    print(f"  재생성 필요: {should_regenerate}")

    if should_regenerate:
        print("  → 답변 품질이 낮습니다. 재생성 또는 추가 검색이 필요합니다.")
    else:
        print("  → 답변 품질이 충분합니다. 최종 답변으로 사용 가능합니다.")

    # 8단계: 최종 결과
    print(f"\n[8단계] 최종 결과")
    print(f"\n{generated_answer}")

    print("\n" + "=" * 70)
    print("파이프라인 완료")
    print("=" * 70)


def demonstrate_edge_cases():
    """
    엣지 케이스 시연
    """
    print("\n\n" + "=" * 70)
    print("엣지 케이스 시연")
    print("=" * 70)

    evaluator = create_evaluator()

    # Case 0: [Retrieve] - 검색 불필요한 경우
    print("\n[Case 0] 검색이 불필요한 일반 질문")
    general_query = "안녕하세요, 오늘 날씨 어때요?"
    retrieve_result = evaluator.evaluate_retrieve_need(general_query)
    print(f"  질문: {general_query}")
    print(f"  검색 필요: {retrieve_result.decision}")
    print(f"  이유: {retrieve_result.reason}")

    # Case 1: 모든 문서가 관련 없는 경우
    print("\n[Case 1] 모든 검색 문서가 관련 없는 경우")
    query = "대사증후군의 진단 기준은?"
    irrelevant_docs = [
        "감기는 바이러스 감염입니다.",
        "비타민 D는 뼈 건강에 중요합니다.",
        "스트레스 관리는 정신 건강에 도움이 됩니다.",
    ]

    eval_result = evaluator.evaluate_documents(
        query, irrelevant_docs, min_relevant_docs=1
    )
    print(f"  외부 검색 필요: {eval_result.should_retrieve_external}")
    print(f"  사유: {eval_result.reason}")

    # Case 2: 답변이 문서로 뒷받침되지 않는 경우
    print("\n[Case 2] 답변이 문서로 뒷받침되지 않는 경우")
    query = "대사증후군 예방법은?"
    document = "대사증후군은 복부 비만과 관련이 있습니다."
    wrong_answer = "대사증후군은 특별한 약물 치료가 필요합니다."

    support_result = evaluator.evaluate_support(query, document, wrong_answer)
    print(f"  지지도 평가: {support_result.support}")

    # Case 3: 답변이 유용하지 않은 경우
    print("\n[Case 3] 답변이 유용하지 않은 경우")
    query = "대사증후군 치료 방법을 자세히 알려주세요"
    bad_answer = "대사증후군은 복잡한 질환입니다."

    usefulness_result = evaluator.evaluate_usefulness(query, bad_answer)
    print(f"  유용성 점수: {usefulness_result.score}/5")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    print("\n🎯 Self-RAG 평가 실전 예제\n")

    try:
        # 메인 파이프라인 시뮬레이션
        simulate_rag_pipeline()

        # 엣지 케이스 시연
        demonstrate_edge_cases()

        print("\n✅ 예제 실행 완료!")

    except Exception as e:
        print(f"\n❌ 예제 실행 실패: {e}")
        import traceback

        traceback.print_exc()
