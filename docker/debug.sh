#!/bin/bash

echo "============================================"
echo "Graphiti MCP 디버깅 스크립트"
echo "============================================"

echo -e "\n[1] 컨테이너 상태"
docker compose ps

echo -e "\n[2] Neo4j 상태"
docker compose logs neo4j | tail -20

echo -e "\n[3] Graphiti MCP 로그 (최근 30줄)"
docker compose logs graphiti-mcp | tail -30

echo -e "\n[4] 포트 확인"
echo "8000 포트:"
lsof -i :8000 || echo "  사용 중인 프로세스 없음"

echo "7474 포트:"
lsof -i :7474 || echo "  사용 중인 프로세스 없음"

echo "7687 포트:"
lsof -i :7687 || echo "  사용 중인 프로세스 없음"

echo -e "\n[5] 환경 변수 확인 (민감 정보 제외)"
if [ -f .env ]; then
    echo "OPENAI_API_KEY: $(grep OPENAI_API_KEY .env | sed 's/OPENAI_API_KEY=sk-.*/OPENAI_API_KEY=sk-***MASKED***/')"
    echo "MODEL_NAME: $(grep MODEL_NAME .env)"
    echo "NEO4J_PASSWORD: ***MASKED***"
else
    echo "  ⚠️  .env 파일이 없습니다!"
fi

echo -e "\n[6] Graphiti MCP 컨테이너 상세 상태"
docker inspect metabolic-graphiti-mcp --format='Status: {{.State.Status}}, ExitCode: {{.State.ExitCode}}, Error: {{.State.Error}}' 2>/dev/null || echo "  컨테이너를 찾을 수 없습니다"

echo -e "\n============================================"
echo "디버깅 완료"
echo "============================================"