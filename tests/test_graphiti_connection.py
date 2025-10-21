# -*- coding: utf-8 -*-
"""
Task 7.1: Graphiti MCP 연결 테스트

Graphiti MCP 서버가 정상적으로 실행되고 있는지 확인합니다.

사전 요구사항:
1. 프로젝트 외부에 Graphiti 레포지토리 클론
   cd .. && git clone https://github.com/getzep/graphiti.git

2. Graphiti MCP 서버 실행
   cd graphiti/mcp_server
   cp .env.example .env  # OPENAI_API_KEY 설정
   docker compose up -d
"""

import sys
import os
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_graphiti_mcp_connection():
    """Graphiti MCP 서버 연결 테스트"""
    print("\n" + "=" * 70)
    print("Task 7.1: Graphiti MCP 연결 테스트")
    print("=" * 70)

    mcp_url = "http://localhost:8000/sse"

    print(f"\n[1단계] MCP 서버 연결 확인")
    print(f"  URL: {mcp_url}")

    try:
        # SSE는 스트리밍 연결이므로 stream=True로 접근
        # 헤더만 확인하고 즉시 종료
        response = requests.get(
            mcp_url, timeout=3, stream=True, headers={"Accept": "text/event-stream"}
        )

        # 상태 코드만 확인하고 연결 종료
        if response.status_code == 200:
            # SSE 연결이 성공하면 즉시 닫기
            response.close()
            print(f"  ✅ SSE 연결 성공 (상태 코드: {response.status_code})")
            return True
        else:
            response.close()
            print(f"  ⚠️  응답 수신했으나 예상치 못한 상태 코드: {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"  ❌ 연결 실패: MCP 서버가 실행 중이지 않습니다")
        print(f"\n해결 방법:")
        print(f"  1. docker/ 디렉토리로 이동")
        print(f"  2. docker compose up -d 실행")
        print(f"  3. docker compose logs -f 로 로그 확인")
        return False

    except requests.exceptions.Timeout:
        print(f"  ❌ 타임아웃: 서버 응답이 없습니다")
        return False

    except Exception as e:
        print(f"  ❌ 예상치 못한 오류: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_neo4j_connection():
    """Neo4j 데이터베이스 연결 테스트"""
    print(f"\n[2단계] Neo4j 연결 확인")
    print(f"  URL: http://localhost:7474")

    try:
        response = requests.get("http://localhost:7474", timeout=5)

        if response.status_code == 200:
            print(f"  ✅ Neo4j 웹 인터페이스 접근 가능")
            print(f"  접속 정보:")
            print(f"    - URL: http://localhost:7474")
            print(f"    - Username: neo4j")
            print(f"    - Password: (docker-compose.yml에서 확인)")
            return True
        else:
            print(f"  ⚠️  응답 수신했으나 예상치 못한 상태 코드: {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"  ❌ 연결 실패: Neo4j가 실행 중이지 않습니다")
        return False

    except requests.exceptions.Timeout:
        print(f"  ❌ 타임아웃: Neo4j 응답이 없습니다")
        return False

    except Exception as e:
        print(f"  ❌ 예상치 못한 오류: {e}")
        return False


def check_docker_containers():
    """Docker 컨테이너 상태 확인"""
    print(f"\n[3단계] Docker 컨테이너 상태 확인")

    try:
        import subprocess

        result = subprocess.run(
            [
                "docker",
                "ps",
                "--filter",
                "name=metabolic",
                "--format",
                "table {{.Names}}\t{{.Status}}\t{{.Ports}}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print(f"  ⚠️  Docker 명령어 실행 실패")
            return False

    except FileNotFoundError:
        print(f"  ⚠️  Docker가 설치되어 있지 않거나 PATH에 없습니다")
        return False

    except Exception as e:
        print(f"  ⚠️  컨테이너 상태 확인 실패: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🧪 Graphiti MCP 환경 테스트")
    print("=" * 70)

    results = []

    # 1. Docker 컨테이너 확인
    results.append(("Docker 컨테이너", check_docker_containers()))

    # 2. Neo4j 연결 확인
    results.append(("Neo4j 연결", test_neo4j_connection()))

    # 3. Graphiti MCP 연결 확인
    results.append(("Graphiti MCP 연결", test_graphiti_mcp_connection()))

    # 결과 요약
    print("\n" + "=" * 70)
    print("테스트 결과 요약")
    print("=" * 70)

    for name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"  {name}: {status}")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    print(f"\n총 {passed}/{total} 테스트 통과")

    if passed == total:
        print("\n🎉 Task 7.1 환경 구축 완료!")
        print("\n다음 단계:")
        print("  - Task 7.2: langchain-mcp-adapter 통합")
        print("  - Task 7.3: 메모리 노드 구현")
    else:
        print("\n⚠️  일부 테스트 실패")
        print("위의 해결 방법을 참고하여 문제를 해결하세요.")
