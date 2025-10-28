"""
Tool 통합 테스트

실행:
    python tests/test_tools.py
"""

import sys
import os

# src 디렉토리를 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_internal_retriever_tool():
    """Internal Retriever Tool 테스트"""
    print("\n=== Test 1: Internal Retriever Tool ===")

    try:
        from src.tools import internal_retriever_tool

        # Tool 기본 정보 확인
        print(f"✓ Tool 이름: {internal_retriever_tool.name}")
        print(f"✓ Tool 설명: {internal_retriever_tool.description[:80]}...")

        # 검색 실행
        query = "대사증후군 진단 기준"
        print(f"\n검색 쿼리: '{query}'")

        result = internal_retriever_tool.invoke({"query": query, "k": 3})

        # 결과 확인
        print(f"✓ 검색 완료")
        print(f"  - 결과 타입: {type(result)}")
        print(f"  - 결과 길이: {len(result)} 문자")
        print(f"  - 결과 미리보기:\n{result[:300]}...")

        assert isinstance(result, str)
        assert len(result) > 0
        assert "[문서" in result

        print("\n✅ Internal Retriever Tool 테스트 통과!")
        return True

    except Exception as e:
        print(f"\n✗ Internal Retriever Tool 테스트 실패: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_patient_context_tool():
    """Patient Context Tool 테스트"""
    print("\n=== Test 2: Patient Context Tool ===")

    try:
        from src.tools import patient_context_tool

        # Tool 기본 정보 확인
        print(f"✓ Tool 이름: {patient_context_tool.name}")
        print(f"✓ Tool 설명: {patient_context_tool.description[:80]}...")

        # 환자 정보 조회
        patient_id = 1
        print(f"\n조회할 환자 ID: {patient_id}")

        result = patient_context_tool.invoke({"patient_id": patient_id})

        # 결과 확인
        print(f"✓ 조회 완료")
        print(f"  - 결과 타입: {type(result)}")
        print(f"  - 결과 길이: {len(result)} 문자")
        print(f"  - 결과 미리보기:\n{result[:300]}...")

        assert isinstance(result, str)
        assert len(result) > 0
        assert "환자" in result or "Patient" in result

        print("\n✅ Patient Context Tool 테스트 통과!")
        return True

    except Exception as e:
        print(f"\n✗ Patient Context Tool 테스트 실패: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_get_all_tools():
    """전체 Tool 리스트 테스트"""
    print("\n=== Test 3: Get All Tools ===")

    try:
        from src.tools import get_all_tools

        # Tavily 포함
        tools_with_tavily = get_all_tools(include_tavily=True)
        print(f"✓ Tavily 포함: {len(tools_with_tavily)}개 도구")
        for tool in tools_with_tavily:
            print(f"  - {tool.name}")

        # Tavily 제외
        tools_without_tavily = get_all_tools(include_tavily=False)
        print(f"\n✓ Tavily 제외: {len(tools_without_tavily)}개 도구")
        for tool in tools_without_tavily:
            print(f"  - {tool.name}")

        assert len(tools_with_tavily) >= 2
        assert len(tools_without_tavily) == 2

        print("\n✅ Get All Tools 테스트 통과!")
        return True

    except Exception as e:
        print(f"\n✗ Get All Tools 테스트 실패: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_tool_with_agent():
    """Agent와 Tool 통합 테스트 (간단)"""
    print("\n=== Test 4: Agent Tool Integration (Basic) ===")

    try:
        from src.tools import get_all_tools
        from langchain_openai import ChatOpenAI

        # API 키 확인
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  OPENAI_API_KEY가 없어 Agent 테스트를 건너뜁니다")
            return True

        # Tool 준비
        tools = get_all_tools(include_tavily=False)  # API 비용 절감
        print(f"✓ {len(tools)}개 도구 준비 완료")

        # LLM 준비
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        print(f"✓ LLM 준비 완료: {llm.model_name}")

        # Tool binding 테스트
        llm_with_tools = llm.bind_tools(tools)
        print(f"✓ Tool binding 완료")

        # 간단한 호출 테스트
        response = llm_with_tools.invoke("대사증후군이란?")
        print(f"✓ LLM 호출 완료")
        print(f"  - 응답 타입: {type(response)}")

        print("\n✅ Agent Tool Integration 테스트 통과!")
        return True

    except Exception as e:
        print(f"\n✗ Agent Tool Integration 테스트 실패: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """테스트 실행"""
    from dotenv import load_dotenv

    load_dotenv()

    print("=" * 60)
    print("Tool 통합 테스트")
    print("=" * 60)

    results = []

    # Test 1: Internal Retriever Tool
    results.append(test_internal_retriever_tool())

    # Test 2: Patient Context Tool
    results.append(test_patient_context_tool())

    # Test 3: Get All Tools
    results.append(test_get_all_tools())

    # Test 4: Agent Tool Integration (선택)
    if os.getenv("OPENAI_API_KEY"):
        results.append(test_tool_with_agent())

    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"통과: {passed}/{total}")

    if passed == total:
        print("✓ 모든 테스트 통과!")
    else:
        print(f"✗ {total - passed}개 테스트 실패")


if __name__ == "__main__":
    main()
