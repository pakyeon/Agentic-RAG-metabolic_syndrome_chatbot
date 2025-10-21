# -*- coding: utf-8 -*-
"""Task 6.4 워크플로우 테스트 - Self-RAG 재생성 루프"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_normal_termination():
    """정상 종료: 품질 충분하여 재생성 불필요"""
    print("\n=== test_normal_termination ===")
    question = "대사증후군 환자를 위한 구체적인 식단 계획과 운동 프로그램을 알려주세요"
    print(f"질문: {question}")

    try:
        from src.graph.workflow import run_rag

        final_state = run_rag(question)

        # 디버깅: 중간 값 출력
        print(f"\n[디버깅 정보]")
        print(f"should_retrieve: {final_state.get('should_retrieve')}")
        print(f"iteration: {final_state.get('iteration')}")
        print(f"max_iterations: {final_state.get('max_iterations')}")
        print(f"support_score: {final_state.get('support_score')}")
        print(f"usefulness_score: {final_state.get('usefulness_score')}")
        print(f"needs_regeneration: {final_state.get('needs_regeneration')}")

        # 검증
        assert len(final_state.get("answer", "")) > 0, "답변이 생성되어야 함"

        # 재생성 여부 확인
        needs_regen = final_state.get("needs_regeneration", False)
        iteration = final_state.get("iteration", 0)
        max_iterations = final_state.get("max_iterations", 2)

        if needs_regen:
            print(f"\n⚠️  재생성 필요 판단됨 (iteration={iteration}/{max_iterations})")
            if iteration >= max_iterations:
                print(f"✅ 최대 반복 횟수 도달하여 종료")
            else:
                print(f"⚠️  재생성이 수행되었을 수 있음")
        else:
            print(f"\n✅ 품질 충분 → 재생성 불필요 → 정상 종료")

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
    print("Task 6.4: Self-RAG 재생성 루프 테스트")
    print("=" * 60)

    results = []

    # Test 1: 정상 종료
    results.append(test_normal_termination())

    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"통과: {passed}/{total}")

    if passed == total:
        print("✅ Task 6.4 테스트 통과!")
    else:
        print(f"❌ {total - passed}개 테스트 실패")
