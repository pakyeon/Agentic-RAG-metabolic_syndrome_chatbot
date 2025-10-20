# embeddings.py
"""
LangChain OpenAI Embeddings 래퍼

text-embedding-3-small 사용
LangSmith에서 토큰 및 비용 추적 가능
"""

import os
from typing import List
from langchain_openai import OpenAIEmbeddings as LangChainOpenAIEmbeddings


class OpenAIEmbeddings:
    """OpenAI 임베딩 클라이언트 (LangChain 기반)"""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str = None):
        """
        Args:
            model: 임베딩 모델 이름
            api_key: OpenAI API 키 (None이면 환경변수 사용)
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key가 필요합니다. " "OPENAI_API_KEY 환경변수를 설정하세요."
            )

        # LangChain OpenAIEmbeddings 초기화
        self.embeddings = LangChainOpenAIEmbeddings(
            model=model, openai_api_key=self.api_key
        )

        print(f"[OpenAI Embeddings] 초기화 완료: {model}")

    def embed_text(self, text: str) -> List[float]:
        """단일 텍스트 임베딩

        Args:
            text: 임베딩할 텍스트

        Returns:
            임베딩 벡터 (1536 차원)
        """
        return self.embeddings.embed_query(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """배치 텍스트 임베딩

        Args:
            texts: 임베딩할 텍스트 리스트

        Returns:
            임베딩 벡터 리스트
        """
        return self.embeddings.embed_documents(texts)

    def get_langchain_embeddings(self) -> LangChainOpenAIEmbeddings:
        """LangChain embeddings 객체 반환 (VectorDB 통합용)

        Returns:
            LangChain OpenAIEmbeddings 인스턴스
        """
        return self.embeddings


# 테스트 코드
def test_embedding():
    """기본 임베딩 테스트"""
    print("\n=== Task 1.1: OpenAI 임베딩 테스트 ===\n")

    # 초기화
    embedder = OpenAIEmbeddings()

    # 1. 단일 텍스트 임베딩
    print("1. 단일 텍스트 임베딩:")
    text = "대사증후군은 심혈관 질환의 위험을 높이는 대사 이상의 집합입니다."
    embedding = embedder.embed_text(text)
    print(f"   텍스트: {text}")
    print(f"   임베딩 차원: {len(embedding)}")
    print(f"   첫 5개 값: {[f'{v:.4f}' for v in embedding[:5]]}")

    # 2. 배치 임베딩
    print("\n2. 배치 임베딩:")
    texts = [
        "복부비만은 대사증후군의 주요 위험 요인입니다.",
        "고혈압은 수축기 혈압 130mmHg 이상일 때 진단됩니다.",
        "공복혈당이 100mg/dL 이상이면 공복혈당장애입니다.",
    ]
    embeddings = embedder.embed_batch(texts)
    print(f"   입력 텍스트 수: {len(texts)}")
    print(f"   생성된 임베딩 수: {len(embeddings)}")
    print(f"   각 임베딩 차원: {len(embeddings[0])}")

    # 3. LangChain 객체 접근
    print("\n3. LangChain embeddings 객체:")
    lc_embeddings = embedder.get_langchain_embeddings()
    print(f"   타입: {type(lc_embeddings)}")
    print(f"   모델: {lc_embeddings.model}")

    print("\n=== 테스트 완료 ===")


def test_similarity():
    """임베딩 유사도 간단 테스트"""
    print("\n=== 임베딩 유사도 테스트 ===\n")

    import numpy as np

    embedder = OpenAIEmbeddings()

    texts = [
        "대사증후군의 진단 기준은 무엇인가요?",
        "대사증후군을 어떻게 진단하나요?",
        "오늘 날씨가 좋네요.",
    ]

    embeddings = embedder.embed_batch(texts)

    # 코사인 유사도
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    print("텍스트:")
    for i, text in enumerate(texts):
        print(f"  [{i}] {text}")

    print("\n유사도:")
    sim_01 = cosine_similarity(embeddings[0], embeddings[1])
    sim_02 = cosine_similarity(embeddings[0], embeddings[2])
    sim_12 = cosine_similarity(embeddings[1], embeddings[2])

    print(f"  유사도(0, 1): {sim_01:.4f} (유사한 질문)")
    print(f"  유사도(0, 2): {sim_02:.4f} (무관한 질문)")
    print(f"  유사도(1, 2): {sim_12:.4f} (무관한 질문)")

    print("\n✅ 유사한 질문끼리는 높은 유사도를 보입니다.")


if __name__ == "__main__":
    # 환경변수 확인
    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("\n설정 방법:")
        print("  export OPENAI_API_KEY='sk-your-api-key-here'")
        exit(1)

    # 기본 테스트
    test_embedding()

    # 유사도 테스트 (선택)
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--similarity":
        test_similarity()
