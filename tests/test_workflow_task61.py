# -*- coding: utf-8 -*-
"""Task 6.1 워크플로우 테스트 - 그래프 생성 및 컴파일 확인"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_graph_creation():
    """그래프 생성 테스트"""
    print("\n=== test_graph_creation ===")

    try:
        from src.graph.workflow import build_rag_graph

        graph = build_rag_graph()
        print("✅ 그래프 생성 성공")

        # 컴파일 확인
        assert graph is not None
        print("✅ 그래프 컴파일 성공")

        return True

    except Exception as e:
        print(f"❌ 그래프 생성 실패: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_initial_state_creation():
    """초기 상태 생성 테스트"""
    print("\n=== test_initial_state_creation ===")

    try:
        from src.graph.workflow import create_initial_state

        # Case 1: 환자 ID 없음
        state1 = create_initial_state("대사증후군이란?")
        assert state1["question"] == "대사증후군이란?"
        assert state1["patient_id"] is None
        print("✅ 초기 상태 생성 (환자 ID 없음)")

        # Case 2: 환자 ID 있음
        state2 = create_initial_state("제 상태는 어떤가요?", patient_id=1)
        assert state2["question"] == "제 상태는 어떤가요?"
        assert state2["patient_id"] == 1
        print("✅ 초기 상태 생성 (환자 ID 있음)")

        return True

    except Exception as e:
        print(f"❌ 초기 상태 생성 실패: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Task 6.1: 기본 그래프 구조 테스트")
    print("=" * 60)

    results = []

    # Test 1: 그래프 생성
    results.append(test_graph_creation())

    # Test 2: 초기 상태 생성
    results.append(test_initial_state_creation())

    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"통과: {passed}/{total}")

    if passed == total:
        print("✅ Task 6.1 모든 테스트 통과!")
    else:
        print(f"❌ {total - passed}개 테스트 실패")
