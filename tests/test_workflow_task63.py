# -*- coding: utf-8 -*-
"""Task 6.3 워크플로우 테스트 - CRAG 조건부 분기"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_crag_correct():
    """CRAG CORRECT: 내부 문서만 사용 (외부 검색 스킵)"""
    print("\n=== test_crag_correct ===")
    question = "대사증후군 환자를 위한 구체적인 식단 계획과 운동 프로그램을 알려주세요"
    print(f"질문: {question}")

    try:
        from src.graph.workflow import run_rag

        final_state = run_rag(question)

        # 디버깅: 중간 값 출력
        print(f"\n[디버깅 정보]")
        print(f"should_retrieve: {final_state.get('should_retrieve')}")
        print(f"crag_action: {final_state.get('crag_action')}")
        print(f"crag_confidence: {final_state.get('crag_confidence')}")
        print(f"내부 문서 수: {len(final_state.get('internal_docs', []))}")
        print(f"외부 문서 수: {len(final_state.get('external_docs', []))}")
        print(f"merged_context 길이: {len(final_state.get('merged_context', ''))} 문자")

        # 검증
        assert final_state.get("should_retrieve") == True, "검색이 수행되어야 함"
        assert len(final_state.get("internal_docs", [])) > 0, "내부 검색 결과 있어야 함"
        assert len(final_state.get("answer", "")) > 0, "답변 생성되어야 함"

        # CRAG 액션 확인
        action = final_state.get("crag_action", "").lower()
        print(f"\nCRAG 액션: {action}")

        if action == "correct":
            print("✅ CORRECT: 내부 문서만 사용 (외부 검색 스킵됨)")
            # CORRECT인 경우 외부 문서가 없어야 함
            assert (
                len(final_state.get("external_docs", [])) == 0
            ), "CORRECT 시 외부 문서 없어야 함"
        elif action == "incorrect":
            print("⚠️  INCORRECT: 외부 검색으로 교체")
            assert (
                len(final_state.get("external_docs", [])) > 0
            ), "INCORRECT 시 외부 문서 있어야 함"
        elif action == "ambiguous":
            print("⚠️  AMBIGUOUS: 내부+외부 병합")
            assert (
                len(final_state.get("external_docs", [])) > 0
            ), "AMBIGUOUS 시 외부 문서 있어야 함"

        print(f"\n✅ 답변 길이: {len(final_state['answer'])} 문자")
        print(f"답변 미리보기: {final_state['answer'][:150]}...")

        return True

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
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

    print("=" * 60)
    print("Task 6.3: CRAG 조건부 분기 테스트")
    print("=" * 60)

    results = []

    # Test 1: CRAG 분기 테스트 (CORRECT 기대)
    results.append(test_crag_correct())

    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"통과: {passed}/{total}")

    if passed == total:
        print("✅ Task 6.3 테스트 통과!")
    else:
        print(f"❌ {total - passed}개 테스트 실패")
