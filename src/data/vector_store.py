"""
VectorDB 구축 및 하이브리드 검색

- BM25:Vector 가중치 1:1 → 2:3 (0.5:0.5 → 0.4:0.6)
- 총 검색 문서 수 명확히 5개로 제한
"""

import os
from pathlib import Path
from threading import Lock
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from src.data.embeddings import OpenAIEmbeddings
from src.data.document_loader import MetabolicSyndromeDocumentLoader
from src.data.document_loader import MetabolicSyndromeChunker
from src.data.path_utils import project_path

# 기본 경로 설정
DEFAULT_RAW_DIRECTORY = project_path("data/raw")
DEFAULT_PARSED_DIRECTORY = project_path("data/parsed")
DEFAULT_PERSIST_DIRECTORY = project_path("data/chroma_db")
_HYBRID_RETRIEVER_CACHE: "HybridRetriever | None" = None
_HYBRID_RETRIEVER_LOCK = Lock()


class VectorStoreBuilder:
    """ChromaDB VectorStore 구축 및 로드"""

    def __init__(
        self,
        persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
        collection_name: str = "metabolic_syndrome_docs",
        embedding_model: str = "text-embedding-3-small",
    ):
        """
        Args:
            persist_directory: ChromaDB 저장 경로
            collection_name: ChromaDB 컬렉션 이름
            embedding_model: 임베딩 모델 이름
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

        # 빌드 시 청크 캐시 (HybridRetriever 생성용)
        self._chunks: Optional[List[Document]] = None

    def build(
        self,
        parsed_dir: str,
        raw_dir: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 50,
        force_rebuild: bool = False,
    ) -> Chroma:
        """VectorDB 구축 또는 재구축

        Args:
            parsed_dir: 파싱된 MD 디렉토리
            raw_dir: 원본 PDF 디렉토리 (선택)
            chunk_size: 청크 크기
            chunk_overlap: 청크 오버랩
            batch_size: 배치 크기
            force_rebuild: 기존 DB 삭제 후 재구축 여부

        Returns:
            Chroma 인스턴스
        """
        parsed_path = project_path(parsed_dir)
        raw_path = project_path(raw_dir) if raw_dir else None

        if parsed_path is None:
            raise ValueError(
                f"parsed_dir을 찾을 수 없습니다: {parsed_dir}\n"
                "프로젝트 루트 기준 상대 경로를 입력하세요."
            )

        # 1. force_rebuild 시 기존 DB 삭제
        if force_rebuild and self.persist_directory.exists():
            print(f"[VectorStore] 기존 DB 삭제: {self.persist_directory}")
            import shutil

            shutil.rmtree(self.persist_directory)

        # 2. 문서 로드
        print(f"\n{'='*60}")
        print(f"[VectorStore] 문서 로드 중...")
        print(f"  - Parsed Dir: {parsed_path}")
        if raw_path:
            print(f"  - Raw Dir: {raw_path}")
        print(f"{'='*60}\n")

        loader = MetabolicSyndromeDocumentLoader(
            parsed_dir=parsed_path, raw_dir=raw_path
        )
        docs = loader.load_documents()

        # 3. 청킹
        print(
            f"\n[VectorStore] 청킹 중 (size={chunk_size}, overlap={chunk_overlap})..."
        )
        chunker = MetabolicSyndromeChunker()
        chunks = chunker.chunk_documents(docs, show_progress=True)

        # 청크 캐시 (HybridRetriever용)
        self._chunks = chunks

        print(f"\n[VectorStore] 총 {len(chunks)}개 청크 생성 완료\n")

        # 4. 임베딩 및 저장
        print(f"[VectorStore] ChromaDB 구축 중 (batch_size={batch_size})...")
        print(f"  - Persist Dir: {self.persist_directory}")
        print(f"  - Collection: {self.collection_name}")
        print(f"  - Embedding: {self.embedding_model}\n")

        lc_embeddings = self.embeddings.get_langchain_embeddings()

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=lc_embeddings,
            collection_name=self.collection_name,
            persist_directory=str(self.persist_directory),
        )

        print(f"\n{'='*60}")
        print("[VectorStore] 구축 완료!")
        print(f"{'='*60}\n")

        # 정보 출력
        info = self.get_db_info(vectorstore)
        self._print_info(info)

        return vectorstore

    def load(self) -> Chroma:
        """기존 VectorDB 로드

        Returns:
            Chroma 인스턴스
        """
        if not self.persist_directory.exists():
            raise ValueError(
                f"VectorDB가 존재하지 않습니다: {self.persist_directory}\n"
                "먼저 build() 메서드로 DB를 구축하세요."
            )

        print(f"\n[VectorStore] 기존 DB 로드: {self.persist_directory}")

        lc_embeddings = self.embeddings.get_langchain_embeddings()

        vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=lc_embeddings,
            persist_directory=str(self.persist_directory),
        )

        # 정보 출력
        info = self.get_db_info(vectorstore)
        self._print_info(info)

        return vectorstore

    def create_hybrid_retriever(
        self,
        vectorstore: Optional[Chroma] = None,
        documents: Optional[List[Document]] = None,
        parsed_dir: Optional[str | Path] = None,
        raw_dir: Optional[str | Path] = None,
        bm25_weight: float = 0.4,  # ⭐ 변경: 0.5 → 0.4 (2:3 비율)
        vector_weight: float = 0.6,  # ⭐ 변경: 0.5 → 0.6
        k: int = 5,  # ⭐ 총 5개 문서만 검색
    ) -> "HybridRetriever":
        """Hybrid Retriever 생성

        BM25:Vector = 2:3 (0.4:0.6), 총 5개 문서

        Args:
            vectorstore: Chroma 인스턴스 (None이면 자동 로드)
            documents: 문서 리스트 (None이면 자동 로드)
            parsed_dir: 문서 로드 시 필요 (documents가 None일 때)
            raw_dir: 문서 로드 시 필요 (선택)
            bm25_weight: BM25 가중치 (기본값 0.4)
            vector_weight: Vector 가중치 (기본값 0.6)
            k: 기본 반환 문서 수 (기본값 5)

        Returns:
            HybridRetriever 인스턴스
        """
        # VectorStore 준비
        if vectorstore is None:
            vectorstore = self.load()

        # Documents 준비
        if documents is None:
            # 빌드 시 저장한 청크가 있으면 사용
            if self._chunks is not None:
                documents = self._chunks
                print(f"[Info] 빌드 시 저장된 {len(documents)}개 청크 사용")
            else:
                # 없으면 다시 로드
                if parsed_dir is None:
                    raise ValueError(
                        "documents와 parsed_dir 둘 다 None입니다. "
                        "하나는 제공해야 합니다."
                    )

                parsed_path = project_path(parsed_dir)
                raw_path = project_path(raw_dir) if raw_dir else None

                if parsed_path is None:
                    raise ValueError("parsed_dir is required")

                print(f"[Info] 문서 재로드 중: {parsed_path}")
                loader = MetabolicSyndromeDocumentLoader(
                    parsed_dir=parsed_path, raw_dir=raw_path
                )
                docs = loader.load_documents()

                chunker = MetabolicSyndromeChunker()
                documents = chunker.chunk_documents(docs, show_progress=False)

                print(f"[Info] {len(documents)}개 청크 로드 완료")

        # Hybrid Retriever 생성
        return HybridRetriever(
            vectorstore=vectorstore,
            documents=documents,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
            k=k,
        )

    def get_db_info(self, vectorstore: Chroma) -> dict:
        """DB 정보 조회

        Returns:
            DB 통계 정보
        """
        try:
            count = vectorstore._collection.count()

            # 디렉토리 크기 계산
            total_size = 0
            if self.persist_directory.exists():
                for dirpath, dirnames, filenames in os.walk(self.persist_directory):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        if os.path.isfile(filepath):
                            total_size += os.path.getsize(filepath)

            size_mb = total_size / (1024 * 1024)

            return {
                "status": "OK",
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": str(self.persist_directory),
                "size_mb": round(size_mb, 2),
                "embedding_model": self.embedding_model,
            }
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}

    def _print_info(self, info: dict):
        """DB 정보 출력"""
        print(f"=== VectorDB 정보 ===")
        for key, value in info.items():
            print(f"  {key}: {value}")
        print()


class HybridRetriever:
    """LangChain EnsembleRetriever를 사용한 하이브리드 검색

    BM25:Vector = 2:3 (0.4:0.6), 총 5개 문서
    """

    def __init__(
        self,
        vectorstore: Chroma,
        documents: List[Document],
        bm25_weight: float = 0.4,  # ⭐ 기본값 변경: 0.5 → 0.4
        vector_weight: float = 0.6,  # ⭐ 기본값 변경: 0.5 → 0.6
        k: int = 5,  # ⭐ 기본값 5개 유지
    ):
        """
        Args:
            vectorstore: ChromaDB 벡터 스토어
            documents: 전체 문서 리스트 (BM25용)
            bm25_weight: BM25 가중치 (기본값 0.4)
            vector_weight: Vector 가중치 (기본값 0.6)
            k: 기본 반환 문서 수 (기본값 5)
        """
        self.vectorstore = vectorstore
        self.documents = documents
        self.k = k

        print(f"\n[Hybrid Retriever] 초기화 중...")

        # 1. BM25 Retriever 생성
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = k

        # 2. Vector Retriever 생성
        self.vector_retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
        )

        # 3. Ensemble Retriever 생성 (RRF 자동 적용)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[bm25_weight, vector_weight],
        )

        print(f"[Hybrid Retriever] 초기화 완료")
        print(f"  - BM25 가중치: {bm25_weight} (2)")
        print(f"  - Vector 가중치: {vector_weight} (3)")
        print(f"  - 비율: BM25:Vector = 2:3")
        print(f"  - 인덱싱된 문서: {len(documents)}개")
        print(f"  - 기본 k: {k}개\n")

    def search(self, query: str, k: int = None) -> List[Document]:
        """하이브리드 검색

        총 5개 문서만 반환

        Args:
            query: 검색 쿼리
            k: 반환할 문서 수 (None이면 기본값 5 사용)

        Returns:
            Document 리스트 (최대 k개)
        """
        _k = k or self.k

        if k and k != self.k:
            # k 값이 다르면 임시로 업데이트
            self.bm25_retriever.k = k
            self.vector_retriever.search_kwargs["k"] = k

        results = self.ensemble_retriever.invoke(query)

        # ⭐ 명확히 k개로 제한
        results = results[:_k]

        # k 값 복원
        if k and k != self.k:
            self.bm25_retriever.k = self.k
            self.vector_retriever.search_kwargs["k"] = self.k

        return results

    def search_with_scores(
        self, query: str, k: int = None
    ) -> List[tuple[Document, float]]:
        """하이브리드 검색 (점수 포함)

        총 5개 문서만 반환

        Args:
            query: 검색 쿼리
            k: 반환할 문서 수 (None이면 기본값 5 사용)

        Returns:
            [(Document, score), ...] (최대 k개)
        """
        _k = k or self.k

        # Vector 결과 (점수 포함)
        vector_results = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=_k
        )

        # Ensemble 결과
        ensemble_docs = self.ensemble_retriever.invoke(query)[:_k]

        # 점수 매핑 (Vector 결과에서 찾기)
        results = []
        for doc in ensemble_docs:
            score = 0.5  # 기본값
            for v_doc, v_score in vector_results:
                if doc.page_content == v_doc.page_content:
                    score = v_score
                    break
            results.append((doc, score))

        # ⭐ 명확히 k개로 제한
        return results[:_k]

    def get_retriever(self) -> EnsembleRetriever:
        """LangChain Retriever 객체 반환 (LangGraph 통합용)

        Returns:
            EnsembleRetriever 인스턴스
        """
        return self.ensemble_retriever


def get_cached_hybrid_retriever(
    persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
    parsed_dir: Optional[str | Path] = DEFAULT_PARSED_DIRECTORY,
    raw_dir: Optional[str | Path] = DEFAULT_RAW_DIRECTORY,
    bm25_weight: float = 0.4,
    vector_weight: float = 0.6,
    k: int = 5,
) -> HybridRetriever:
    """Return a cached HybridRetriever instance for reuse across requests."""
    global _HYBRID_RETRIEVER_CACHE

    if _HYBRID_RETRIEVER_CACHE is not None:
        return _HYBRID_RETRIEVER_CACHE

    with _HYBRID_RETRIEVER_LOCK:
        if _HYBRID_RETRIEVER_CACHE is None:
            builder = VectorStoreBuilder(persist_directory=persist_directory)
            _HYBRID_RETRIEVER_CACHE = builder.create_hybrid_retriever(
                parsed_dir=parsed_dir,
                raw_dir=raw_dir,
                bm25_weight=bm25_weight,
                vector_weight=vector_weight,
                k=k,
            )
    return _HYBRID_RETRIEVER_CACHE


# 유틸리티 함수
def build_vector_db(
    parsed_dir: str,
    raw_dir: Optional[str] = None,
    persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    force_rebuild: bool = False,
) -> Chroma:
    """VectorDB 구축 헬퍼 함수

    Args:
        parsed_dir: 파싱된 MD 디렉토리
        raw_dir: 원본 PDF 디렉토리
        persist_directory: ChromaDB 저장 경로
        chunk_size: 청크 크기
        chunk_overlap: 청크 오버랩
        force_rebuild: 기존 DB 삭제 후 재구축

    Returns:
        Chroma 인스턴스
    """
    builder = VectorStoreBuilder(persist_directory=persist_directory)

    return builder.build(
        parsed_dir=parsed_dir,
        raw_dir=raw_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        force_rebuild=force_rebuild,
    )


def load_vector_db(persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY) -> Chroma:
    """기존 VectorDB 로드 헬퍼 함수

    Args:
        persist_directory: ChromaDB 저장 경로

    Returns:
        Chroma 인스턴스
    """
    builder = VectorStoreBuilder(persist_directory=persist_directory)

    return builder.load()


def create_hybrid_retriever(
    persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
    parsed_dir: Optional[str | Path] = None,
    raw_dir: Optional[str | Path] = None,
    bm25_weight: float = 0.4,  # ⭐ 기본값 변경: 0.5 → 0.4 (2:3 비율)
    vector_weight: float = 0.6,  # ⭐ 기본값 변경: 0.5 → 0.6
    k: int = 5,  # ⭐ 총 5개 문서만 검색
) -> HybridRetriever:
    """Hybrid Retriever 생성 헬퍼 함수

    BM25:Vector = 2:3 (0.4:0.6), 총 5개 문서

    Args:
        persist_directory: ChromaDB 저장 경로
        parsed_dir: 문서 디렉토리 (필요시)
        raw_dir: 원본 PDF 디렉토리 (선택)
        bm25_weight: BM25 가중치 (기본값 0.4)
        vector_weight: Vector 가중치 (기본값 0.6)
        k: 기본 반환 문서 수 (기본값 5)

    Returns:
        HybridRetriever 인스턴스
    """
    builder = VectorStoreBuilder(persist_directory=persist_directory)

    return builder.create_hybrid_retriever(
        parsed_dir=parsed_dir,
        raw_dir=raw_dir,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
        k=k,
    )


# 테스트 코드


def test_build():
    """VectorDB 구축 테스트"""
    print("\n=== Task 1.3: VectorDB 구축 테스트 ===\n")

    # 경로 설정
    parsed_dir = DEFAULT_PARSED_DIRECTORY
    raw_dir = DEFAULT_RAW_DIRECTORY
    persist_dir = DEFAULT_PERSIST_DIRECTORY

    # 빌더 초기화
    builder = VectorStoreBuilder(persist_directory=persist_dir)

    # 구축
    vectorstore = builder.build(
        parsed_dir=parsed_dir,
        raw_dir=raw_dir,
        chunk_size=500,
        chunk_overlap=50,
        batch_size=50,
        force_rebuild=True,  # 테스트용
    )

    print("✅ VectorDB 구축 완료!")

    return vectorstore, builder


def test_vector_search(vectorstore: Chroma):
    """Vector 검색 테스트"""
    print("\n=== Vector Search 테스트 ===\n")

    query = "대사증후군 진단 기준은?"

    print(f"쿼리: '{query}'\n")

    results = vectorstore.similarity_search_with_relevance_scores(query, k=3)

    for i, (doc, score) in enumerate(results, 1):
        print(f"[{i}] Score: {score:.4f}")
        print(f"    출처: {doc.metadata.get('basename', 'N/A')[:50]}")
        print(f"    내용: {doc.page_content[:80]}...\n")


def test_hybrid_search(builder: VectorStoreBuilder):
    """Hybrid 검색 테스트"""
    print("\n=== Hybrid Search 테스트 (BM25:Vector = 2:3) ===\n")

    # Hybrid Retriever 생성 (Task 1.1 개선 설정 적용)
    hybrid = builder.create_hybrid_retriever(bm25_weight=0.4, vector_weight=0.6, k=5)

    queries = [
        "대사증후군 진단 기준은?",
        "복부비만 기준",
        "허리둘레 90cm",
    ]

    for query in queries:
        print(f"쿼리: '{query}'")

        results = hybrid.search_with_scores(query, k=5)

        print(f"  검색 결과: {len(results)}개 문서")
        for i, (doc, score) in enumerate(results, 1):
            print(f"  [{i}] Score: {score:.4f}")
            print(f"      출처: {doc.metadata.get('basename', 'N/A')[:40]}")
            print(f"      내용: {doc.page_content[:60]}...\n")

        print("-" * 70)


def test_load_and_hybrid():
    """기존 DB 로드 후 Hybrid 생성 테스트"""
    print("\n=== 로드 후 Hybrid 생성 테스트 ===\n")

    persist_dir = DEFAULT_PERSIST_DIRECTORY
    parsed_dir = DEFAULT_PARSED_DIRECTORY

    # 헬퍼 함수로 Hybrid 생성 (Task 1.1 개선 기본값 적용)
    hybrid = create_hybrid_retriever(
        persist_directory=persist_dir,
        parsed_dir=parsed_dir,  # 문서 재로드
    )

    query = "고혈압 수치"
    print(f"쿼리: '{query}'\n")

    results = hybrid.search(query, k=5)

    print(f"검색 결과: {len(results)}개 문서")
    for i, doc in enumerate(results, 1):
        print(f"[{i}] {doc.metadata.get('basename', 'N/A')[:40]}")
        print(f"    {doc.page_content[:80]}...\n")

    print("✅ 로드 및 Hybrid 생성 성공!")


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv()

    # OPENAI_API_KEY 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        sys.exit(1)

    # 경로 확인
    parsed_dir = DEFAULT_PARSED_DIRECTORY
    if not parsed_dir.exists():
        print(f"❌ Error: parsed/ 디렉토리를 찾을 수 없습니다: {parsed_dir}")
        sys.exit(1)

    # 구축 테스트
    vectorstore, builder = test_build()

    # Vector 검색 테스트
    if len(sys.argv) > 1 and sys.argv[1] == "--vector":
        test_vector_search(vectorstore)

    # Hybrid 검색 테스트 (Task 1.1 개선 확인)
    if len(sys.argv) > 1 and sys.argv[1] == "--hybrid":
        test_hybrid_search(builder)

    # 로드 후 Hybrid 테스트
    if len(sys.argv) > 1 and sys.argv[1] == "--load":
        test_load_and_hybrid()
