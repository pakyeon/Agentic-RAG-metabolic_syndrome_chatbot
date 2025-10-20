# vector_store.py
"""
ChromaDB VectorStore 구축 + Hybrid Search 통합

Task 1.1 (OpenAI 임베딩) + Task 1.2 (문서 로드/청킹) + Hybrid Search
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from tqdm import tqdm


from .embeddings import OpenAIEmbeddings
from .document_loader import (
    MetabolicSyndromeDocumentLoader,
    MetabolicSyndromeChunker,
)


class VectorStoreBuilder:
    """VectorDB 빌더 + Hybrid Search 통합"""

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        persist_directory: str = "../../chromadb/openai-small",
        collection_name: str = "metabolic_syndrome",
    ):
        """
        Args:
            embedding_model: OpenAI 임베딩 모델
            persist_directory: ChromaDB 저장 경로
            collection_name: 컬렉션 이름
        """
        self.embedding_model = embedding_model
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self._chunks = None  # Hybrid Search용 문서 저장

        # OpenAI 임베딩 초기화 (Task 1.1)
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

        print(f"[VectorStore Builder] 초기화 완료")
        print(f"  - 임베딩 모델: {embedding_model}")
        print(f"  - 저장 경로: {self.persist_directory}")
        print(f"  - 컬렉션: {collection_name}")

    def build(
        self,
        parsed_dir: str,
        raw_dir: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 100,
        force_rebuild: bool = False,
    ) -> Chroma:
        """VectorDB 구축

        Args:
            parsed_dir: 파싱된 MD 디렉토리
            raw_dir: 원본 PDF 디렉토리
            chunk_size: 청크 크기
            chunk_overlap: 청크 오버랩
            batch_size: 배치 크기 (한번에 추가할 문서 수)
            force_rebuild: 기존 DB 삭제 후 재구축

        Returns:
            Chroma 인스턴스
        """
        # 기존 DB 처리
        if self.persist_directory.exists():
            if force_rebuild:
                print(f"\n[Info] 기존 DB 삭제 중: {self.persist_directory}")
                shutil.rmtree(self.persist_directory)
            else:
                print(f"\n[Warning] 기존 DB가 존재합니다: {self.persist_directory}")
                print("기존 DB를 사용하려면 load() 메서드를 사용하세요.")
                print("재구축하려면 force_rebuild=True로 설정하세요.")
                return self.load()

        print(f"\n{'='*60}")
        print(f"VectorDB 구축 시작")
        print(f"{'='*60}\n")

        # Step 1: 문서 로드 (Task 1.2)
        print("Step 1/4: 문서 로드")
        loader = MetabolicSyndromeDocumentLoader(parsed_dir=parsed_dir, raw_dir=raw_dir)
        documents = loader.load_documents()

        if not documents:
            raise ValueError("로드된 문서가 없습니다. 경로를 확인하세요.")

        # Step 2: 청킹 (Task 1.2)
        print("Step 2/4: 문서 청킹")
        chunker = MetabolicSyndromeChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = chunker.chunk_documents(documents, show_progress=True)

        if not chunks:
            raise ValueError("생성된 청크가 없습니다.")

        # Hybrid Search용 청크 저장
        self._chunks = chunks

        # Step 3: ChromaDB 초기화 및 임베딩
        print("Step 3/4: ChromaDB 초기화 및 임베딩")
        print(f"  - 총 {len(chunks)}개 청크를 임베딩합니다...")
        print(f"  - 배치 크기: {batch_size}")

        # persist_directory 생성
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # LangChain embeddings 객체 가져오기
        lc_embeddings = self.embeddings.get_langchain_embeddings()

        # ChromaDB 초기화 (빈 상태)
        vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=lc_embeddings,
            persist_directory=str(self.persist_directory),
        )

        # Step 4: 배치로 문서 추가
        print("Step 4/4: 배치 인덱싱")

        total_batches = (len(chunks) + batch_size - 1) // batch_size

        with tqdm(total=len(chunks), desc="인덱싱 진행", unit="chunk") as pbar:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                batch_num = i // batch_size + 1

                # 배치 추가
                vectorstore.add_documents(documents=batch)

                pbar.update(len(batch))
                pbar.set_postfix({"배치": f"{batch_num}/{total_batches}"})

        print(f"\n{'='*60}")
        print(f"VectorDB 구축 완료!")
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
        parsed_dir: Optional[str] = None,
        raw_dir: Optional[str] = None,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        k: int = 5,
    ) -> "HybridRetriever":
        """Hybrid Retriever 생성

        Args:
            vectorstore: Chroma 인스턴스 (None이면 자동 로드)
            documents: 문서 리스트 (None이면 자동 로드)
            parsed_dir: 문서 로드 시 필요 (documents가 None일 때)
            raw_dir: 문서 로드 시 필요 (선택)
            bm25_weight: BM25 가중치
            vector_weight: Vector 가중치
            k: 기본 반환 문서 수

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

                print(f"[Info] 문서 재로드 중: {parsed_dir}")
                loader = MetabolicSyndromeDocumentLoader(
                    parsed_dir=parsed_dir, raw_dir=raw_dir
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
    """LangChain EnsembleRetriever를 사용한 하이브리드 검색"""

    def __init__(
        self,
        vectorstore: Chroma,
        documents: List[Document],
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        k: int = 5,
    ):
        """
        Args:
            vectorstore: ChromaDB 벡터 스토어
            documents: 전체 문서 리스트 (BM25용)
            bm25_weight: BM25 가중치 (0~1)
            vector_weight: Vector 가중치 (0~1)
            k: 기본 반환 문서 수
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
        print(f"  - BM25 가중치: {bm25_weight}")
        print(f"  - Vector 가중치: {vector_weight}")
        print(f"  - 인덱싱된 문서: {len(documents)}개")
        print(f"  - 기본 k: {k}\n")

    def search(self, query: str, k: int = None) -> List[Document]:
        """하이브리드 검색

        Args:
            query: 검색 쿼리
            k: 반환할 문서 수 (None이면 기본값 사용)

        Returns:
            Document 리스트
        """
        if k and k != self.k:
            # k 값이 다르면 임시로 업데이트
            self.bm25_retriever.k = k
            self.vector_retriever.search_kwargs["k"] = k

        results = self.ensemble_retriever.invoke(query)

        # k 값 복원
        if k and k != self.k:
            self.bm25_retriever.k = self.k
            self.vector_retriever.search_kwargs["k"] = self.k

        return results

    def search_with_scores(
        self, query: str, k: int = None
    ) -> List[tuple[Document, float]]:
        """하이브리드 검색 (점수 포함)

        Args:
            query: 검색 쿼리
            k: 반환할 문서 수

        Returns:
            [(Document, score), ...]
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

        return results

    def get_retriever(self) -> EnsembleRetriever:
        """LangChain Retriever 객체 반환 (LangGraph 통합용)

        Returns:
            EnsembleRetriever 인스턴스
        """
        return self.ensemble_retriever


# 유틸리티 함수
def build_vector_db(
    parsed_dir: str,
    raw_dir: Optional[str] = None,
    persist_directory: str = "../../chromadb/openai-small",
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


def load_vector_db(persist_directory: str = "../../chromadb/openai-small") -> Chroma:
    """기존 VectorDB 로드 헬퍼 함수

    Args:
        persist_directory: ChromaDB 저장 경로

    Returns:
        Chroma 인스턴스
    """
    builder = VectorStoreBuilder(persist_directory=persist_directory)

    return builder.load()


def create_hybrid_retriever(
    persist_directory: str = "../../chromadb/openai-small",
    parsed_dir: Optional[str] = None,
    raw_dir: Optional[str] = None,
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
    k: int = 5,
) -> HybridRetriever:
    """Hybrid Retriever 생성 헬퍼 함수

    Args:
        persist_directory: ChromaDB 저장 경로
        parsed_dir: 문서 디렉토리 (필요시)
        raw_dir: 원본 PDF 디렉토리 (선택)
        bm25_weight: BM25 가중치
        vector_weight: Vector 가중치
        k: 기본 반환 문서 수

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
    parsed_dir = "../../metabolic_syndrome_data/parsed"
    raw_dir = "../../metabolic_syndrome_data/raw"
    persist_dir = "../../chromadb/test-openai-small"

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
    print("\n=== Hybrid Search 테스트 ===\n")

    # Hybrid Retriever 생성 (빌드 시 저장된 청크 사용)
    hybrid = builder.create_hybrid_retriever(bm25_weight=0.5, vector_weight=0.5, k=5)

    queries = [
        "대사증후군 진단 기준은?",
        "복부비만 기준",
        "허리둘레 90cm",
    ]

    for query in queries:
        print(f"쿼리: '{query}'")

        results = hybrid.search_with_scores(query, k=3)

        for i, (doc, score) in enumerate(results, 1):
            print(f"  [{i}] Score: {score:.4f}")
            print(f"      출처: {doc.metadata.get('basename', 'N/A')[:40]}")
            print(f"      내용: {doc.page_content[:60]}...\n")

        print("-" * 70)


def test_load_and_hybrid():
    """기존 DB 로드 후 Hybrid 생성 테스트"""
    print("\n=== 로드 후 Hybrid 생성 테스트 ===\n")

    persist_dir = "../../chromadb/test-openai-small"
    parsed_dir = "../../metabolic_syndrome_data/parsed"

    # 헬퍼 함수로 Hybrid 생성
    hybrid = create_hybrid_retriever(
        persist_directory=persist_dir,
        parsed_dir=parsed_dir,  # 문서 재로드
        bm25_weight=0.5,
        vector_weight=0.5,
    )

    query = "고혈압 수치"
    print(f"쿼리: '{query}'\n")

    results = hybrid.search(query, k=3)

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
    parsed_dir = "../../metabolic_syndrome_data/parsed"
    if not os.path.exists(parsed_dir):
        print(f"❌ Error: parsed/ 디렉토리를 찾을 수 없습니다: {parsed_dir}")
        sys.exit(1)

    # 구축 테스트
    vectorstore, builder = test_build()

    # Vector 검색 테스트
    if len(sys.argv) > 1 and sys.argv[1] == "--vector":
        test_vector_search(vectorstore)

    # Hybrid 검색 테스트
    if len(sys.argv) > 1 and sys.argv[1] == "--hybrid":
        test_hybrid_search(builder)

    # 로드 후 Hybrid 테스트
    if len(sys.argv) > 1 and sys.argv[1] == "--load":
        test_load_and_hybrid()
