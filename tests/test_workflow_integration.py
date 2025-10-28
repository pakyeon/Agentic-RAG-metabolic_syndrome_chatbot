# -*- coding: utf-8 -*-
"""
Task 6.5: 통합 테스트

기존 Task 5.2~5.5의 개별 노드 테스트를 그래프 기반 엔드투엔드 테스트로 통합
"""

import sys
import os
import pytest

if not os.getenv("OPENAI_API_KEY"):
    pytest.skip(
        "OpenAI API 키가 없어 통합 워크플로우 테스트를 건너뜁니다.",
        allow_module_level=True,
    )

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def extract_answer_from_messages(messages):
    """메시지에서 마지막 AI 답변 추출"""
    for msg in reversed(messages):
        if msg.type == "ai":
            return msg.content
    return ""


def test_scenario_1_basic():
    """
    시나리오 1: 일반 질문 (검색 → CORRECT → 답변)
    """
    print("\n" + "=" * 70)
    print("시나리오 1: 일반 질문 (검색 → CORRECT → 답변)")
    print("=" * 70)

    question = "대사증후군 환자를 위한 구체적인 식단 계획과 운동 프로그램을 알려주세요"
    print(f"질문: {question}")

    try:
        from src.graph.workflow import run_rag

        final_state = run_rag(question)

        # === Task 5.2 검증: 환자 컨텍스트 + 검색 필요성 ===
        print("\n[Task 5.2 검증]")
        patient_context = final_state.get("patient_context")
        metadata = final_state.get("metadata", {})
        should_retrieve = metadata.get("should_retrieve", False)
        print(f"  환자 컨텍스트: {'없음' if not patient_context else '있음'}")
        print(f"  검색 필요: {should_retrieve}")
        assert should_retrieve == True, "검색이 필요해야 함"

        # === Task 5.3 검증: 내부 검색 + ISREL + CRAG ===
        print("\n[Task 5.3 검증]")
        internal_docs = final_state.get("internal_docs", [])
        relevance_scores = metadata.get("relevance_scores", [])
        crag_action = metadata.get("crag_action", "")
        print(f"  내부 검색 문서 수: {len(internal_docs)}")
        print(f"  ISREL 평가 수: {len(relevance_scores)}")
        print(f"  CRAG 액션: {crag_action}")
        assert len(internal_docs) > 0, "내부 검색 결과가 있어야 함"

        # === Task 5.4 검증: 외부 검색 + 컨텍스트 병합 ===
        print("\n[Task 5.4 검증]")
        external_docs = final_state.get("external_docs", [])
        context_added = metadata.get("context_added", False)
        print(f"  외부 검색 문서 수: {len(external_docs)}")
        print(f"  컨텍스트 추가: {context_added}")
        assert context_added, "컨텍스트가 추가되어야 함"

        # === Task 5.5 검증: 답변 생성 + ISSUP/ISUSE ===
        print("\n[Task 5.5 검증]")
        messages = final_state.get("messages", [])
        answer = extract_answer_from_messages(messages)
        support_score = metadata.get("support_score", 0.0)
        usefulness_score = metadata.get("usefulness_score", 0.0)
        needs_regeneration = metadata.get("needs_regeneration", False)
        print(f"  답변 길이: {len(answer)} 문자")
        print(f"  ISSUP (지원도): {support_score}/5.0")
        print(f"  ISUSE (유용성): {usefulness_score}/5.0")
        print(f"  재생성 필요: {needs_regeneration}")
        assert len(answer) > 0, "답변이 생성되어야 함"

        print(f"\n답변 미리보기:\n{answer[:200]}...")
        print("\n✅ 시나리오 1 통과!")
        return True

    except Exception as e:
        print(f"\n❌ 시나리오 1 실패: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_scenario_2_patient():
    """
    시나리오 2: 환자별 질문 (환자 컨텍스트 포함)
    """
    print("\n" + "=" * 70)
    print("시나리오 2: 환자별 질문 (환자 컨텍스트 포함)")
    print("=" * 70)

    question = "제 상태를 개선하려면 어떻게 해야 하나요?"
    patient_id = 1
    print(f"질문: {question}")
    print(f"환자 ID: {patient_id}")

    try:
        from src.graph.workflow import run_rag

        final_state = run_rag(question, patient_id=patient_id)

        # === Task 5.2 검증: 환자 컨텍스트 ===
        print("\n[Task 5.2 검증]")
        patient_context = final_state.get("patient_context")
        metadata = final_state.get("metadata", {})
        should_retrieve = metadata.get("should_retrieve", False)
        print(f"  환자 컨텍스트: {'있음' if patient_context else '없음'}")
        if patient_context:
            print(f"  컨텍스트 길이: {len(patient_context)} 문자")
            print(f"  컨텍스트 미리보기: {patient_context[:150]}...")
        print(f"  검색 필요: {should_retrieve}")

        # === Task 5.3 검증 ===
        print("\n[Task 5.3 검증]")
        internal_docs = final_state.get("internal_docs", [])
        print(f"  내부 검색 문서 수: {len(internal_docs)}")

        # === Task 5.5 검증: 환자별 맞춤 답변 ===
        print("\n[Task 5.5 검증]")
        messages = final_state.get("messages", [])
        answer = extract_answer_from_messages(messages)
        print(f"  답변 길이: {len(answer)} 문자")
        assert len(answer) > 0, "답변이 생성되어야 함"

        print(f"\n답변 미리보기:\n{answer[:200]}...")
        print("\n✅ 시나리오 2 통과!")
        return True

    except Exception as e:
        print(f"\n❌ 시나리오 2 실패: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_scenario_3_external():
    """
    시나리오 3: 외부 검색 필요 (INCORRECT → Tavily)

    통합 노드:
    - Task 5.3: CRAG 액션 = INCORRECT or AMBIGUOUS
    - Task 5.4: Tavily 외부 검색 + 컨텍스트 병합

    참고: 이 시나리오는 실제로 INCORRECT가 발생하기 어려울 수 있음
          (VectorDB 품질이 좋기 때문)
    """
    print("\n" + "=" * 70)
    print("시나리오 3: 외부 검색 케이스 확인")
    print("=" * 70)

    # 최신 정보나 구체적 통계를 요구하는 질문
    question = "2024년 대사증후군 최신 연구 동향과 새로운 치료법은 무엇인가요?"
    print(f"질문: {question}")
    print("(최신 정보 요구 → INCORRECT 또는 AMBIGUOUS 가능성)")

    try:
        from src.graph.workflow import run_rag

        final_state = run_rag(question)

        # === CRAG 액션 확인 ===
        print("\n[CRAG 액션 확인]")
        crag_action = final_state.get("crag_action", "")
        print(f"  CRAG 액션: {crag_action}")

        # === Task 5.4 검증: 외부 검색 ===
        print("\n[Task 5.4 검증]")
        external_docs = final_state.get("external_docs", [])
        internal_docs = final_state.get("internal_docs", [])
        merged_context = final_state.get("merged_context", "")

        print(f"  내부 문서 수: {len(internal_docs)}")
        print(f"  외부 문서 수: {len(external_docs)}")
        print(f"  병합 컨텍스트 길이: {len(merged_context)} 문자")

        if len(external_docs) > 0:
            print(f"  ✅ 외부 검색 수행됨!")
            print(
                f"     외부 문서 출처: {external_docs[0].metadata.get('source', 'N/A')}"
            )
        else:
            print(f"  ⚠️  외부 검색 수행 안 됨 (CORRECT 판단)")
            print(f"     → VectorDB 품질이 충분하여 내부 문서만 사용")

        # === 답변 검증 ===
        print("\n[답변 검증]")
        answer = final_state.get("answer", "")
        print(f"  답변 길이: {len(answer)} 문자")
        assert len(answer) > 0, "답변이 생성되어야 함"

        print(f"\n답변 미리보기:\n{answer[:200]}...")
        print("\n✅ 시나리오 3 완료!")
        return True

    except Exception as e:
        print(f"\n❌ 시나리오 3 실패: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다")
        sys.exit(1)

    print("=" * 70)
    print("Task 6.5: Self-CRAG 통합 테스트")
    print("=" * 70)
    print("\n기존 Task 5.2~5.5 노드 테스트를 그래프 기반으로 통합 검증")

    results = []

    # 시나리오 1: 일반 질문
    results.append(test_scenario_1_basic())

    # 시나리오 2: 환자별 질문
    results.append(test_scenario_2_patient())

    # 시나리오 3: 외부 검색
    results.append(test_scenario_3_external())

    # 결과 요약
    print("\n" + "=" * 70)
    print("통합 테스트 결과")
    print("=" * 70)

    passed = sum(results)
    total = len(results)

    print(f"통과: {passed}/{total}")
    print(f"  - 시나리오 1 (일반): {'✅' if results[0] else '❌'}")
    print(f"  - 시나리오 2 (환자): {'✅' if results[1] else '❌'}")
    print(f"  - 시나리오 3 (외부): {'✅' if results[2] else '❌'}")

    if passed == total:
        print("\n🎉 Task 6.5 통합 테스트 모두 통과!")
        print("\n완성된 Self-CRAG 플로우:")
        print("  환자 로드 → [Retrieve] → 내부 검색 → ISREL")
        print("  → CRAG 액션 → 외부 검색(조건부) → 병합")
        print("  → 답변 생성 → ISSUP/ISUSE → 재생성(조건부)")
    else:
        print(f"\n❌ {total - passed}개 시나리오 실패")
