# -*- coding: utf-8 -*-
"""Task 6.2 워크플로우 테스트 - Self-RAG [Retrieve] 조건부 분기"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_retrieve_yes():
    """검색 필요 O: 대사증후군 관련 질문"""
    print("\n=== test_retrieve_yes ===")
    print("질문: 대사증후군 진단 기준은 무엇인가요?")

    try:
        from src.graph.workflow import run_rag

        final_state = run_rag("대사증후군 진단 기준은 무엇인가요?")

        # 검증
        assert final_state.get("should_retrieve") == True, "검색 필요성 판단 실패"
        assert len(final_state.get("internal_docs", [])) > 0, "내부 검색 수행되지 않음"
        assert len(final_state.get("answer", "")) > 0, "답변 생성 실패"

        print(f"✅ should_retrieve: {final_state['should_retrieve']}")
        print(f"✅ 검색된 문서 수: {len(final_state['internal_docs'])}")
        print(f"✅ 답변 길이: {len(final_state['answer'])} 문자")
        print(f"답변 미리보기: {final_state['answer'][:100]}...")

        return True

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_retrieve_no():
    """검색 필요 X: 일반 인사"""
    print("\n=== test_retrieve_no ===")
    print("질문: 안녕하세요")

    try:
        from src.graph.workflow import run_rag

        final_state = run_rag("안녕하세요")

        # 검증
        assert final_state.get("should_retrieve") == False, "검색 스킵 실패"
        assert (
            len(final_state.get("internal_docs", [])) == 0
        ), "검색이 수행됨 (스킵되어야 함)"
        assert len(final_state.get("answer", "")) > 0, "답변 생성 실패"

        print(f"✅ should_retrieve: {final_state['should_retrieve']}")
        print(f"✅ 검색 스킵 확인: 내부 문서 {len(final_state['internal_docs'])}개")
        print(f"✅ 답변 길이: {len(final_state['answer'])} 문자")
        print(f"답변: {final_state['answer']}")

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
    print("Task 6.2: Self-RAG [Retrieve] 조건부 분기 테스트")
    print("=" * 60)

    results = []

    # Test 1: 검색 필요 O
    results.append(test_retrieve_yes())

    # Test 2: 검색 필요 X
    results.append(test_retrieve_no())

    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"통과: {passed}/{total}")

    if passed == total:
        print("✅ Task 6.2 모든 테스트 통과!")
    else:
        print(f"❌ {total - passed}개 테스트 실패")
