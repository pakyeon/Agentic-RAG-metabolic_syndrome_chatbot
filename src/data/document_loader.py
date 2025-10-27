# document_loader.py
"""
대사증후군 MD 문서 로드 및 청킹

기존 프로젝트의 parsed/ 폴더 구조 활용:
  parsed/
    ├── 문서1/
    │   ├── part-01.md
    │   ├── part-02.md
    │   └── ...
    └── 문서2/
        └── part-01.md
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from .path_utils import (
    DEFAULT_PARSED_DIRECTORY,
    DEFAULT_RAW_DIRECTORY,
    PathLike,
    project_path,
)


class MetabolicSyndromeDocumentLoader:
    """대사증후군 MD 문서 로더"""

    # part-XX.md 파일 패턴
    PART_FILE_PATTERN = re.compile(r"^part-(\d{2,3})\.md$", re.IGNORECASE)

    def __init__(
        self,
        parsed_dir: PathLike,
        raw_dir: Optional[PathLike] = None,
        min_content_length: int = 30,
    ):
        """
        Args:
            parsed_dir: 파싱된 MD 파일이 있는 디렉토리
            raw_dir: 원본 PDF 파일 디렉토리 (선택)
            min_content_length: 최소 컨텐츠 길이 (짧은 청크 필터링)
        """
        parsed_path = project_path(parsed_dir)
        if parsed_path is None:
            raise ValueError("parsed_dir is required")

        self.parsed_dir = parsed_path
        self.raw_dir = project_path(raw_dir) if raw_dir else None
        self.min_content_length = min_content_length

        if not self.parsed_dir.exists():
            raise ValueError(f"parsed_dir이 존재하지 않습니다: {self.parsed_dir}")

        print(f"[Document Loader] 초기화 완료")
        print(f"  - Parsed 디렉토리: {self.parsed_dir}")
        if self.raw_dir:
            print(f"  - Raw 디렉토리: {self.raw_dir}")

    def _collect_md_parts(self) -> Dict[str, List[Path]]:
        """Collect markdown part files grouped by basename."""
        grouped: Dict[str, List[Path]] = {}

        if not self.parsed_dir.is_dir():
            return grouped

        for md_path in sorted(self.parsed_dir.rglob("part-*.md")):
            if not md_path.is_file():
                continue

            match = self.PART_FILE_PATTERN.match(md_path.name)
            if not match:
                continue

            basename = md_path.parent.name
            grouped.setdefault(basename, []).append(md_path)

        # Ensure deterministic ordering
        for parts in grouped.values():
            parts.sort()

        return grouped

    def _resolve_pdf_path(self, basename: str) -> Optional[str]:
        """원본 PDF 경로 찾기

        Args:
            basename: 문서 베이스네임

        Returns:
            PDF 경로 또는 None
        """
        if not self.raw_dir:
            return None

        pdf_path = self.raw_dir / f"{basename}.pdf"
        return str(pdf_path) if pdf_path.exists() else None

    def load_documents(self) -> List[Document]:
        """모든 MD 파일 로드

        Returns:
            Document 리스트 (LangChain Document)
        """
        md_parts = self._collect_md_parts()

        if not md_parts:
            print(
                f"[Warning] parsed/ 디렉토리에서 문서를 찾을 수 없습니다: {self.parsed_dir}"
            )
            return []

        print(f"\n[문서 로드] 총 {len(md_parts)}개 문서 발견")

        part_counts = {basename: len(paths) for basename, paths in md_parts.items()}
        pdf_cache: Dict[str, Optional[str]] = {}
        missing_pdf_logged: set[str] = set()

        def _metadata_func(file_path: str, metadata: dict) -> dict:
            path = Path(file_path)
            match = self.PART_FILE_PATTERN.match(path.name)
            part_index = int(match.group(1)) if match else 0
            basename = path.parent.name

            if basename not in pdf_cache:
                pdf_cache[basename] = self._resolve_pdf_path(basename)
                if pdf_cache[basename] is None and basename not in missing_pdf_logged:
                    print(f"  ⚠️  {basename}: 원본 PDF 없음")
                    missing_pdf_logged.add(basename)

            enriched = {
                **metadata,
                "basename": basename,
                "part_index": part_index,
                "part_count": part_counts.get(basename, 0),
                "md_path": str(path),
                "source_id": f"{basename}#part-{part_index:02d}",
            }

            pdf_path = pdf_cache.get(basename)
            if pdf_path:
                enriched["source"] = pdf_path

            return enriched

        loader = DirectoryLoader(
            str(self.parsed_dir),
            glob="**/part-*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            use_multithreading=False,
            metadata_func=_metadata_func,
        )

        documents = loader.load()
        documents = [
            doc for doc in documents if len(doc.page_content.strip()) >= self.min_content_length
        ]
        documents.sort(
            key=lambda doc: (
                doc.metadata.get("basename", ""),
                int(doc.metadata.get("part_index", 0)),
            )
        )

        print(f"[완료] {len(documents)}개 MD 파트 파일 로드됨\n")
        return documents

    def get_summary(self) -> dict:
        """로드된 문서 요약 정보

        Returns:
            문서 통계 정보
        """
        md_parts = self._collect_md_parts()

        total_parts = sum(len(parts) for parts in md_parts.values())

        return {
            "document_count": len(md_parts),
            "part_count": total_parts,
            "basenames": list(md_parts.keys()),
        }


class MetabolicSyndromeChunker:
    """MD 문서 청킹 (마크다운 헤더 기반)"""

    def __init__(
        self, chunk_size: int = 500, chunk_overlap: int = 50, min_chunk_size: int = 100
    ):
        """
        Args:
            chunk_size: 청크 크기
            chunk_overlap: 청크 오버랩
            min_chunk_size: 최소 청크 크기 (너무 작은 청크 필터링)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        # 마크다운 헤더 기반 스플리터 (H1~H3)
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ],
            strip_headers=False,
        )

        # 재귀적 텍스트 스플리터 (크기 조정용)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        print(f"[Chunker] 초기화 완료")
        print(f"  - Chunk size: {chunk_size}")
        print(f"  - Chunk overlap: {chunk_overlap}")
        print(f"  - Min chunk size: {min_chunk_size}")

    def _extract_lower_headers(self, content: str) -> dict:
        """H4~H6 헤더 추출 (본문에서 직접)

        Args:
            content: 마크다운 텍스트

        Returns:
            {'Header 4': str, 'Header 5': str, ...}
        """
        headers = {}
        lines = content.split("\n")

        patterns = {
            "Header 4": re.compile(r"^\s*####\s+(.+)"),
            "Header 5": re.compile(r"^\s*#####\s+(.+)"),
            "Header 6": re.compile(r"^\s*######\s+(.+)"),
        }

        for key, pattern in patterns.items():
            for line in lines:
                match = pattern.match(line)
                if match:
                    headers[key] = match.group(1).strip()
                    break  # 첫 번째만

        return headers

    def _compose_header_path(self, metadata: dict) -> str:
        """헤더 경로 문자열 생성

        Args:
            metadata: 메타데이터 (Header 1~6 포함)

        Returns:
            "Header1 > Header2 > Header3" 형식
        """
        parts = []
        for i in range(1, 7):
            header = metadata.get(f"Header {i}")
            if header and isinstance(header, str):
                parts.append(header.strip())

        return " > ".join(parts) if parts else ""

    def chunk_documents(
        self, documents: List[Document], show_progress: bool = True
    ) -> List[Document]:
        """문서 청킹

        Args:
            documents: 원본 Document 리스트
            show_progress: 진행상황 표시

        Returns:
            청크된 Document 리스트
        """
        if not documents:
            return []

        all_chunks = []

        if show_progress:
            print(f"\n[청킹 시작] {len(documents)}개 문서")

        for i, doc in enumerate(documents, 1):
            if show_progress and i % 10 == 0:
                print(f"  진행: {i}/{len(documents)}")

            # 1. H1~H3 기준으로 분할
            md_splits = self.md_splitter.split_text(doc.page_content)

            for split in md_splits:
                # 2. 원본 메타데이터 + 헤더 메타데이터 병합
                merged_metadata = {**doc.metadata, **split.metadata}

                # 3. H4~H6 추출 (본문에서)
                lower_headers = self._extract_lower_headers(split.page_content)
                merged_metadata.update(lower_headers)

                # 4. 헤더 경로 생성
                merged_metadata["header_path"] = self._compose_header_path(
                    merged_metadata
                )

                # 5. 크기가 크면 재분할
                if len(split.page_content) > self.chunk_size:
                    sub_chunks = self.text_splitter.split_text(split.page_content)
                    for sub_chunk in sub_chunks:
                        if len(sub_chunk.strip()) >= self.min_chunk_size:
                            all_chunks.append(
                                Document(
                                    page_content=sub_chunk, metadata=merged_metadata
                                )
                            )
                else:
                    # 최소 크기 체크
                    if len(split.page_content.strip()) >= self.min_chunk_size:
                        all_chunks.append(
                            Document(
                                page_content=split.page_content,
                                metadata=merged_metadata,
                            )
                        )

        if show_progress:
            print(f"[완료] {len(all_chunks)}개 청크 생성\n")

        return all_chunks


# 테스트 코드
def test_document_loader():
    """문서 로더 테스트"""
    print("\n=== Task 1.2: 문서 로드 테스트 ===\n")

    # 경로 설정 (기존 프로젝트 구조)
    parsed_dir = DEFAULT_PARSED_DIRECTORY
    raw_dir = DEFAULT_RAW_DIRECTORY

    # 로더 초기화
    loader = MetabolicSyndromeDocumentLoader(parsed_dir=parsed_dir, raw_dir=raw_dir)

    # 요약 정보
    summary = loader.get_summary()
    print(f"문서 요약:")
    print(f"  - 문서 수: {summary['document_count']}")
    print(f"  - 파트 수: {summary['part_count']}")
    print(f"  - 베이스네임: {summary['basenames'][:3]}...\n")

    # 문서 로드
    documents = loader.load_documents()

    # 샘플 출력
    if documents:
        print(f"첫 번째 문서 샘플:")
        doc = documents[0]
        print(f"  - 베이스네임: {doc.metadata.get('basename')}")
        print(f"  - 파트 인덱스: {doc.metadata.get('part_index')}")
        print(f"  - Source ID: {doc.metadata.get('source_id')}")
        print(f"  - 내용 길이: {len(doc.page_content)} 문자")
        print(f"  - 내용 미리보기: {doc.page_content[:100]}...\n")

    return documents


def test_chunker(documents: List[Document]):
    """청킹 테스트"""
    print("\n=== 청킹 테스트 ===\n")

    # 청커 초기화
    chunker = MetabolicSyndromeChunker(
        chunk_size=500, chunk_overlap=50, min_chunk_size=100
    )

    # 청킹 실행 (샘플만)
    sample_docs = documents[:3] if len(documents) > 3 else documents
    print(f"샘플 문서 {len(sample_docs)}개로 테스트\n")

    chunks = chunker.chunk_documents(sample_docs, show_progress=True)

    # 결과 출력
    print(f"청킹 결과:")
    print(f"  - 입력 문서: {len(sample_docs)}")
    print(f"  - 출력 청크: {len(chunks)}")
    print(
        f"  - 평균 청크 크기: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} 문자\n"
    )

    # 샘플 청크 출력
    if chunks:
        print(f"첫 번째 청크 샘플:")
        chunk = chunks[0]
        print(f"  - 베이스네임: {chunk.metadata.get('basename')}")
        print(f"  - 헤더 경로: {chunk.metadata.get('header_path')}")
        print(f"  - 내용 길이: {len(chunk.page_content)} 문자")
        print(f"  - 내용:\n{chunk.page_content[:200]}...\n")


def test_full_pipeline():
    """전체 파이프라인 테스트"""
    print("\n=== 전체 파이프라인 테스트 ===\n")

    parsed_dir = DEFAULT_PARSED_DIRECTORY
    raw_dir = DEFAULT_RAW_DIRECTORY

    # 1. 로드
    loader = MetabolicSyndromeDocumentLoader(parsed_dir, raw_dir)
    documents = loader.load_documents()

    if not documents:
        print("❌ 문서를 찾을 수 없습니다. 경로를 확인하세요.")
        return

    # 2. 청킹
    chunker = MetabolicSyndromeChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_documents(documents)

    # 3. 통계
    print(f"=== 최종 통계 ===")
    print(f"  - 원본 문서: {len(documents)}")
    print(f"  - 생성된 청크: {len(chunks)}")
    print(f"  - 청크 증가율: {len(chunks) / len(documents):.1f}배")

    # 헤더 경로 분포
    header_paths = set(c.metadata.get("header_path", "") for c in chunks)
    print(f"  - 고유 헤더 경로: {len(header_paths)}개")

    # 베이스네임 분포
    basenames = set(c.metadata.get("basename") for c in chunks)
    print(f"  - 문서 종류: {len(basenames)}개")

    print("\n✅ 파이프라인 테스트 완료!")

    return chunks


if __name__ == "__main__":
    import sys

    # 경로 확인
    parsed_dir = DEFAULT_PARSED_DIRECTORY
    if not parsed_dir.exists():
        print(f"❌ Error: parsed/ 디렉토리를 찾을 수 없습니다: {parsed_dir}")
        print("\n경로를 수정하거나 심볼릭 링크를 생성하세요:")
        print(f"  ln -s {DEFAULT_PARSED_DIRECTORY.parent} ./metabolic_syndrome_data")
        sys.exit(1)

    # 기본 테스트
    documents = test_document_loader()

    if documents:
        test_chunker(documents)

        # 전체 테스트 (선택)
        if len(sys.argv) > 1 and sys.argv[1] == "--full":
            test_full_pipeline()
