"""
Tavily Tool 테스트

실행:
    python tests/test_tavily_tool.py
"""

import sys
import os

# src 디렉토리를 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_tool_creation():
    """Tool 생성 테스트 (API 호출 없음)"""
    print("\n=== Test 1: Tool 생성 ===")

    try:
        from src.tools import get_tavily_tool

        # API 키 확인
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            print("✗ TAVILY_API_KEY 환경변수가 설정되지 않았습니다")
            return False

        # Tool 생성
        tool = get_tavily_tool(max_results=2)  # 최소 결과 수

        print(f"✓ Tool 생성 성공")
        print(f"  - Tool 이름: {tool.name}")
        print(f"  - Tool 설명: {tool.description[:50]}...")

        return True

    except Exception as e:
        print(f"✗ Tool 생성 실패: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_single_search():
    """단일 검색 테스트 (API 호출 1회만)"""
    print("\n=== Test 2: 단일 검색 (API 호출 1회) ===")

    try:
        from src.tools import get_tavily_tool

        # Tool 생성 (최소 결과)
        tool = get_tavily_tool(max_results=2)

        # 검색 실행
        query = "대사증후군"
        print(f"검색 쿼리: '{query}'")

        result = tool.invoke({"query": query})

        # 결과 확인
        print(f"✓ 검색 완료")
        print(f"  - 결과 타입: {type(result)}")

        # 결과가 문자열이면 파싱
        if isinstance(result, str):
            print(f"  - 결과 길이: {len(result)} 문자")
            print(f"  - 결과 미리보기: {result[:200]}...")
        elif isinstance(result, dict):
            print(f"  - 결과 키: {list(result.keys())}")
        elif isinstance(result, list):
            print(f"  - 결과 개수: {len(result)}개")
            if result:
                print(f"  - 첫 번째 결과: {str(result[0])[:100]}...")

        return True

    except Exception as e:
        print(f"✗ 검색 실패: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    from dotenv import load_dotenv

    load_dotenv()
    """테스트 실행"""
    print("=" * 60)
    print("Tavily Tool 테스트")
    print("=" * 60)
    print()

    # 환경변수 확인
    if not os.getenv("TAVILY_API_KEY"):
        print("❌ TAVILY_API_KEY 환경변수가 설정되지 않았습니다")
        print("   .env 파일을 확인하거나 다음을 실행하세요:")
        print("   export TAVILY_API_KEY=your-key-here")
        return

    results = []

    # Test 1: Tool 생성 (API 호출 없음)
    results.append(test_tool_creation())

    # Test 2: 단일 검색 (API 호출 1회만)
    if results[0]:  # Test 1 성공 시에만
        results.append(test_single_search())

    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"통과: {passed}/{total}")
    print(f"총 API 호출: 1회 (검색 1회)")

    if passed == total:
        print("✓ 모든 테스트 통과!")
    else:
        print(f"✗ {total - passed}개 테스트 실패")


if __name__ == "__main__":
    main()
