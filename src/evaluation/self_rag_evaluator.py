"""
Self-RAG 평가자
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from langchain_openai import ChatOpenAI


@dataclass
class RelevanceResult:
    """ISREL 평가 결과"""

    relevance: str  # "relevant" | "irrelevant"
    confidence: float  # 0.0-1.0


@dataclass
class SupportResult:
    """ISSUP 평가 결과"""

    support: str  # "fully_supported" | "partially_supported" | "no_support"
    confidence: float  # 0.0-1.0


@dataclass
class UsefulnessResult:
    """ISUSE 평가 결과"""

    score: int  # 1-5
    confidence: float  # 0.0-1.0


@dataclass
class RetrieveResult:
    """[Retrieve] 평가 결과"""

    should_retrieve: (
        str  # "yes" | "no"  # TODO: support "continue" once contextual wiring is ready
    )
    difficulty: str  # "easy" | "normal" | "hard" | "none"
    documents_to_evaluate: int
    confidence: float  # 0.0-1.0
    reason: str  # 판단 이유

    @property
    def decision(self) -> str:
        """Backward compatibility alias."""
        return self.should_retrieve


@dataclass
class DocumentEvaluation:
    """단일 문서에 대한 전체 평가"""

    document_content: str
    relevance: RelevanceResult
    support: Optional[SupportResult] = None


@dataclass
class OverallEvaluation:
    """전체 검색 결과 평가"""

    query: str
    document_evaluations: List[DocumentEvaluation]
    should_retrieve_external: bool
    reason: str
    early_stopped: bool = False
    crag_action: Optional[str] = None


@dataclass
class DocumentEvaluationWithAction:
    """단일 문서 평가 + CRAG 메타데이터"""

    doc_id: int
    document_content: str
    score: float
    relevance: str  # "relevant" | "irrelevant"
    confidence: float
    reason: str


@dataclass
class CombinedEvaluationResult:
    """ISREL + CRAG 통합 결과"""

    document_evaluations: List[DocumentEvaluationWithAction]
    crag_action: str  # "correct" | "incorrect" | "ambiguous"
    reason: str
    min_relevant_docs: int
    relevant_count: int


@dataclass
class AnswerQualityResult:
    """ISSUP + ISUSE 통합 결과"""

    support_results: List[SupportResult]
    usefulness_score: int
    usefulness_confidence: float
    should_regenerate: bool
    regenerate_reason: str


DIFFICULTY_DOC_LIMITS = {
    "easy": 2,
    "normal": 5,
    "hard": 8,
}

BATCH_RELEVANCE_PROMPT = """You are a retrieval quality evaluator.

Question: {query}

Retrieved documents:
{documents}

Evaluation guidelines:
- Score each document from 1 (irrelevant) to 5 (highly relevant).
- Provide a brief reason for each score.

Output format (valid JSON only):
[
  {{"doc_id": 1, "score": 4, "reason": "High overlap with the question"}},
  {{"doc_id": 2, "score": 2, "reason": "Partially related"}}
]

Do not output anything other than valid JSON.
"""

BATCH_SUPPORT_PROMPT = """You are an answer support evaluator.

Question: {query}

Answer:
{answer}

Retrieved documents:
{documents}

Evaluation guidelines:
- For each document, classify support as fully_supported, partially_supported, or no_support.
- Provide a brief reason for each label.
- Include a confidence score between 0.0 and 1.0 for each decision.

Output format (valid JSON only):
[
  {{"doc_id": 1, "support": "fully_supported", "confidence": 0.9, "reason": "Directly cites diagnostic criteria"}},
  {{"doc_id": 2, "support": "partially_supported", "confidence": 0.7, "reason": "General lifestyle guidance"}}
]

Do not output anything other than valid JSON.
"""

RETRIEVAL_DECISION_PROMPT = """You evaluate retrieval quality and decide the next action.

Question: {query}
Documents:
{documents}

Tasks:
1. Score each document from 1–5 and label it relevant/irrelevant.
2. Choose the next action:
   - CORRECT: ≥{min_relevant_docs} relevant documents, no external search needed.
   - INCORRECT: no relevant documents, replace with external search.
   - AMBIGUOUS: not enough evidence, supplement with external search.

Output (valid JSON only):
{{
  "document_evaluations": [
    {{"doc_id": 1, "score": 4, "relevance": "relevant", "confidence": 0.8, "reason": "Direct diagnostic criteria"}}
  ],
  "crag_action": "CORRECT",
  "reason": "Two documents fully answer the question"
}}

Do not output anything other than valid JSON.
"""


class BatchResultParseError(RuntimeError):
    """Raised when the batch evaluator response cannot be parsed."""


def _format_documents_numbered(documents: Iterable[str]) -> str:
    """번호가 매겨진 문서 목록 문자열을 생성한다."""
    lines: List[str] = []
    for idx, raw in enumerate(documents, start=1):
        doc = (raw or "").strip()
        if len(doc) > 1200:
            doc = doc[:1200].rstrip() + "..."
        lines.append(f"{idx}. {doc}")
    return "\n".join(lines)


def _coerce_doc_id(value: Union[int, str, None], fallback: int) -> int:
    """LLM이 돌려준 doc_id를 정수 인덱스로 정규화한다."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if match:
            return int(match.group(0))
    return fallback


def _strip_json_fences(raw_text: str) -> str:
    """```json fences를 제거하고 양 끝 공백을 정리한다."""
    cleaned = raw_text.strip()
    cleaned = cleaned.replace("```json", "```")
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


class SelfRAGEvaluator:
    """
    Self-RAG 평가자 (배치 평가 + 조기 종료 + 단계 통합)
    """

    def __init__(
        self,
        model_name: str = "gpt-5-mini",
        reasoning_effort="minimal",
        temperature: float = 0.0,
    ):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self._prompts: Dict[str, str] = {}

    # ========== Batch parsing helper ==========

    def _parse_batch_evaluation_result(
        self, raw_text: str, expected_count: int
    ) -> List[Dict[str, Any]]:
        """LLM 응답에서 JSON 배열을 추출하고 문서 순서에 맞게 정렬한다."""
        if expected_count <= 0:
            return []

        if not raw_text or not raw_text.strip():
            raise BatchResultParseError("Empty batch response.")

        cleaned = raw_text.strip()
        cleaned = cleaned.replace("```json", "```")
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise BatchResultParseError("Failed parsing JSON output.") from exc

        if isinstance(payload, dict):
            for key in ("documents", "results", "items", "scores"):
                value = payload.get(key)
                if isinstance(value, list):
                    payload = value
                    break

        if not isinstance(payload, list):
            raise BatchResultParseError("Batch response is not a list.")

        by_doc_id: Dict[int, Dict[str, Any]] = {}

        for idx, item in enumerate(payload, start=1):
            if not isinstance(item, dict):
                continue
            doc_idx = _coerce_doc_id(item.get("doc_id"), idx)
            if 1 <= doc_idx <= expected_count and doc_idx not in by_doc_id:
                by_doc_id[doc_idx] = item

        normalized: List[Dict[str, Any]] = []
        for doc_idx in range(1, expected_count + 1):
            normalized.append(by_doc_id.get(doc_idx, {}))

        return normalized

    # ========== Batch ISREL/ISSUP ==========

    def evaluate_relevance_batch(
        self, query: str, documents: List[Union[str, Any]]
    ) -> List[RelevanceResult]:
        """여러 문서에 대한 ISREL 평가를 한 번에 수행한다."""
        if not documents:
            return []

        doc_texts = [
            doc
            if isinstance(doc, str)
            else getattr(doc, "page_content", str(doc))
            for doc in documents
        ]

        prompt = BATCH_RELEVANCE_PROMPT.format(
            query=query,
            documents=_format_documents_numbered(doc_texts),
        )

        response = self.llm.invoke(prompt)
        parsed = self._parse_batch_evaluation_result(
            getattr(response, "content", str(response)),
            expected_count=len(doc_texts),
        )

        results: List[RelevanceResult] = []

        for idx, item in enumerate(parsed, start=1):
            if not item:
                results.append(
                    RelevanceResult(relevance="irrelevant", confidence=0.0)
                )
                continue

            score_raw = item.get("score")
            try:
                score = float(score_raw)
            except (TypeError, ValueError):
                score = 0.0

            relevance = "relevant" if score >= 3 else "irrelevant"
            confidence = max(0.1, min(score / 5 if score else 0.5, 1.0))

            results.append(
                RelevanceResult(
                    relevance=relevance,
                    confidence=float(confidence),
                )
            )

        return results

    def evaluate_support_batch(
        self, query: str, documents: List[Union[str, Any]], answer: str
    ) -> List[SupportResult]:
        """여러 문서에 대한 ISSUP 평가를 한 번에 수행한다."""
        if not documents:
            return []

        doc_texts = [
            doc
            if isinstance(doc, str)
            else getattr(doc, "page_content", str(doc))
            for doc in documents
        ]

        prompt = BATCH_SUPPORT_PROMPT.format(
            query=query,
            answer=answer,
            documents=_format_documents_numbered(doc_texts),
        )

        response = self.llm.invoke(prompt)
        parsed = self._parse_batch_evaluation_result(
            getattr(response, "content", str(response)),
            expected_count=len(doc_texts),
        )

        results: List[SupportResult] = []
        for idx, item in enumerate(parsed, start=1):
            if not item:
                results.append(
                    SupportResult(support="no_support", confidence=0.0)
                )
                continue

            support = str(item.get("support", "")).strip().lower() or "no_support"
            if support not in {"fully_supported", "partially_supported", "no_support"}:
                support = "no_support"

            try:
                confidence = float(item.get("confidence", 0.7))
            except (TypeError, ValueError):
                confidence = 0.7

            confidence = max(0.0, min(confidence, 1.0))

            results.append(
                SupportResult(
                    support=support,
                    confidence=confidence,
                )
            )

        return results

    def evaluate_documents_with_early_stop(
        self,
        query: str,
        documents: List[Union[str, Any]],
        min_relevant_docs: int = 1,
        enable_early_stop: bool = True,
        max_documents: Optional[int] = None,
    ) -> Tuple[List[RelevanceResult], bool, int]:
        """
        문서 관련성 평가를 수행하면서 일정 조건에서 조기 종료한다.

        Returns:
            (평가 결과 리스트, 조기 종료 여부, 실제 평가한 문서 수)
        """
        if not documents:
            return [], False, 0

        total_limit = max_documents if max_documents else len(documents)
        total_limit = max(0, min(total_limit, len(documents)))

        if total_limit == 0:
            return [], False, 0

        batch_results: List[RelevanceResult] = []
        evaluated_count = 0

        for idx in range(1, total_limit + 1):
            subset = documents[:idx]
            batch_results = self.evaluate_relevance_batch(query, subset)
            evaluated_count = idx

            relevant_count = sum(
                1 for result in batch_results if result.relevance == "relevant"
            )

            if enable_early_stop and relevant_count >= max(1, min_relevant_docs):
                return batch_results, True, evaluated_count

        return batch_results, False, evaluated_count

    # ========== Relevance + CRAG 통합 평가 ==========

    def evaluate_retrieval_and_decide_action(
        self,
        query: str,
        documents: List[Union[str, Any]],
        min_relevant_docs: int = 2,
    ) -> CombinedEvaluationResult:
        """Prompt A2 기반으로 ISREL + CRAG 결정을 한 번에 수행한다."""
        if not documents:
            return CombinedEvaluationResult(
                document_evaluations=[],
                crag_action="incorrect",
                reason="검색 결과 없음",
                min_relevant_docs=min_relevant_docs,
                relevant_count=0,
            )

        doc_texts = [
            doc if isinstance(doc, str) else getattr(doc, "page_content", str(doc))
            for doc in documents
        ]

        prompt = RETRIEVAL_DECISION_PROMPT.format(
            query=query,
            documents=_format_documents_numbered(doc_texts),
            min_relevant_docs=min_relevant_docs,
        )

        try:
            response = self.llm.invoke(prompt)
            payload = json.loads(_strip_json_fences(response.content))
        except Exception as exc:  # pragma: no cover - surfaced to caller
            raise BatchResultParseError("Failed to parse combined retrieval evaluation.") from exc

        doc_items = payload.get("document_evaluations") or payload.get("documents") or []

        evaluations: List[DocumentEvaluationWithAction] = []
        seen_doc_ids = set()

        for fallback_idx, item in enumerate(doc_items, start=1):
            if not isinstance(item, dict):
                continue

            doc_id = _coerce_doc_id(item.get("doc_id"), fallback_idx)
            if doc_id in seen_doc_ids or not (1 <= doc_id <= len(doc_texts)):
                continue

            seen_doc_ids.add(doc_id)
            score_raw = item.get("score")
            try:
                score = float(score_raw)
            except (TypeError, ValueError):
                score = 0.0

            relevance = str(item.get("relevance", "irrelevant")).strip().lower()
            relevance = "relevant" if relevance == "relevant" else "irrelevant"

            try:
                confidence = float(item.get("confidence", 0.7))
            except (TypeError, ValueError):
                confidence = 0.7

            reason = str(item.get("reason", "")).strip() or ""

            evaluations.append(
                DocumentEvaluationWithAction(
                    doc_id=doc_id,
                    document_content=doc_texts[doc_id - 1],
                    score=score,
                    relevance=relevance,
                    confidence=max(0.0, min(confidence, 1.0)),
                    reason=reason,
                )
            )

        # Ensure docs without explicit entries are still represented
        for doc_id in range(1, len(doc_texts) + 1):
            if doc_id not in seen_doc_ids:
                evaluations.append(
                    DocumentEvaluationWithAction(
                        doc_id=doc_id,
                        document_content=doc_texts[doc_id - 1],
                        score=0.0,
                        relevance="irrelevant",
                        confidence=0.5,
                        reason="Not evaluated by LLM.",
                    )
                )

        evaluations.sort(key=lambda item: item.doc_id)

        crag_action = str(payload.get("crag_action", "ambiguous")).strip().lower()
        if crag_action not in {"correct", "incorrect", "ambiguous"}:
            crag_action = "ambiguous"

        relevant_count = sum(
            1 for item in evaluations if item.relevance == "relevant"
        )

        reason = str(payload.get("reason", "")).strip() or ""

        return CombinedEvaluationResult(
            document_evaluations=evaluations,
            crag_action=crag_action,
            reason=reason,
            min_relevant_docs=min_relevant_docs,
            relevant_count=relevant_count,
        )

    def evaluate_answer_combined(
        self, query: str, answer: str, documents: List[str]
    ) -> AnswerQualityResult:
        """
        ISSUP + ISUSE를 한 번에 평가

        답변의 지원도와 유용성을 동시에 평가하여 중복 제거

        Args:
            query: 사용자 질문
            answer: 생성된 답변
            documents: 참조한 문서들

        Returns:
            CombinedAnswerResult (지원도 + 유용성 + 재생성 여부)
        """
        if not documents:
            return AnswerQualityResult(
                support_results=[],
                usefulness_score=1,
                usefulness_confidence=0.5,
                should_regenerate=True,
                regenerate_reason="참조 문서 없음",
            )

        docs_text = ""
        for i, doc in enumerate(documents, 1):
            doc_preview = doc[:250] + "..." if len(doc) > 250 else doc
            docs_text += f"\n[문서 {i}]\n{doc_preview}\n"

        prompt = f"""당신은 답변 품질 평가자입니다.

**질문**: {query}

**생성된 답변**:
{answer}

**참조 문서들**:{docs_text}

**평가 기준**:
1. 각 문서가 답변을 얼마나 뒷받침하는지 평가 (ISSUP)
2. 답변이 질문에 얼마나 유용한지 평가 (ISUSE)
3. 답변 재생성이 필요한지 판단

**출력 형식 (JSON만 출력)**:
{{
  "support": [
    {{"doc_id": 1, "support": "fully_supported", "confidence": 0.9}},
    {{"doc_id": 2, "support": "partially_supported", "confidence": 0.7}}
  ],
  "usefulness_score": 4,
  "usefulness_confidence": 0.9,
  "should_regenerate": false,
  "regenerate_reason": "답변 품질 충분"
}}

**support 값**: "fully_supported", "partially_supported", "no_support"
**usefulness_score**: 1~5 (1=매우 나쁨, 5=매우 좋음)

DO NOT OUTPUT ANYTHING OTHER THAN VALID JSON.

출력:"""

        response = self.llm.invoke(prompt)
        result_text = response.content.strip()
        result_text = result_text.replace("```json", "").replace("```", "").strip()

        result_json = json.loads(result_text)

        support_results = []
        for item in result_json.get("support", []):
            support_results.append(
                SupportResult(
                    support=item.get("support", "no_support"),
                    confidence=float(item.get("confidence", 0.5)),
                )
            )

        # 문서 수에 맞춰 조정
        while len(support_results) < len(documents):
            support_results.append(
                SupportResult(support="no_support", confidence=0.5)
            )

        return AnswerQualityResult(
            support_results=support_results[: len(documents)],
            usefulness_score=int(result_json.get("usefulness_score", 3)),
            usefulness_confidence=float(
                result_json.get("usefulness_confidence", 0.5)
            ),
            should_regenerate=bool(result_json.get("should_regenerate", False)),
            regenerate_reason=result_json.get("regenerate_reason", ""),
        )

    # ========== 검색 필요성 판단 & 기본 평가기 ==========

    def evaluate_answer_quality(
        self,
        query: str,
        answer: str,
        documents: List[Union[str, Any]],
    ) -> AnswerQualityResult:
        """Prompt 기반 답변 품질 평가 래퍼."""
        doc_texts = [
            doc if isinstance(doc, str) else getattr(doc, "page_content", str(doc))
            for doc in documents
        ]
        return self.evaluate_answer_combined(query, answer, doc_texts)


    def evaluate_retrieve_need(
        self, query: str, context: Optional[str] = None
    ) -> RetrieveResult:
        """
        검색 필요성과 예상 난이도를 평가한다.

        질문이 추가 검색을 요구하는지, 요구한다면 몇 개의 문서를 검토해야 하는지를
        모델에게 판단하도록 한다.
        """
        context_block = (
            f"**현재 컨텍스트**:\n{context}\n\n" if context and context.strip() else ""
        )

        prompt = f"""당신은 대사증후군 상담 챗봇의 검색 및 난이도 평가자입니다.

**질문**:
{query}

{context_block}## 1단계: 검색 필요성 판단
- 대사증후군 관련 전문 정보 필요 → YES
- 단순 인사/일상 대화 또는 현재 컨텍스트만으로 충분 → NO
- TODO: CONTINUE(컨텍스트 유지) 결정을 지원하려면 세션/기억 정보를 전달해야 합니다.

## 2단계: 질문 난이도 분석 (검색이 필요한 경우만)
질문의 복잡도를 다음 기준으로 판단하세요. 개인화 관련 요소는 고려하지 말고 질문 자체만 평가합니다.

### [Easy] - 다음 중 하나에 해당
□ 용어 정의나 개념 설명 ("~이란?", "~무엇인가?")
□ 단순 사실 확인 ("~맞나요?", "~인가요?")
□ Yes/No로 답할 수 있는 질문

### [Normal] - 다음 중 하나에 해당
□ 구체적 정보 요청 ("~기준은?", "~수치는?")
□ 절차나 방법 질문 ("~어떻게?", "~방법은?")
□ 증상이나 원인 ("~증상은?", "~원인은?")
□ 단일 주제에 대한 상세 설명

### [Hard] - 다음 중 하나에 해당
□ 여러 질병/조건 동시 비교 ("당뇨와 고혈압이 있을 때")
□ 복잡한 상황 분석 (3개 이상의 요소 고려)
□ 장기적 관리 전략/계획 수립
□ 여러 치료법/방법 간 비교 분석
□ "종합적으로", "전반적으로" 등 다면적 평가

- 해당되는 가장 높은 난이도를 선택하세요.

## 출력 형식 (JSON만 출력)
{{
  "should_retrieve": "yes",
  "difficulty": "normal",
  "reason": "구체적 진단 기준 정보 필요"
}}
"""

        # 모델 출력은 현재 yes/no 두 가지 결정을 사용한다.
        # CONTINUE는 세션 컨텍스트 주입을 마친 뒤 TODO 섹션을 통해 재도입한다.
        try:
            response = self.llm.invoke(prompt)
            result_text = response.content.strip()
            clean_text = result_text.replace("```json", "").replace("```", "").strip()

            import json

            result_json = json.loads(clean_text)
        except Exception as exc:
            print(f"[Warning] 검색/난이도 평가 실패, 기본값 사용: {exc}")
            return RetrieveResult(
                should_retrieve="yes",
                difficulty="normal",
                documents_to_evaluate=DIFFICULTY_DOC_LIMITS["normal"],
                confidence=0.5,
                reason="평가 실패로 기본값 적용",
            )

        should_retrieve = str(result_json.get("should_retrieve", "yes")).strip().lower()
        if should_retrieve not in {"yes", "no"}:
            should_retrieve = "yes"

        if should_retrieve == "yes":
            difficulty = str(result_json.get("difficulty", "normal")).strip().lower()
            if difficulty not in DIFFICULTY_DOC_LIMITS:
                difficulty = "normal"
            documents_to_evaluate = DIFFICULTY_DOC_LIMITS[difficulty]
        else:
            difficulty = "none"
            documents_to_evaluate = 0

        reason = str(result_json.get("reason", "판단 완료")).strip()

        confidence = (
            0.9  # TODO: adjust confidence scaling when "continue" is supported again
        )

        return RetrieveResult(
            should_retrieve=should_retrieve,
            difficulty=difficulty,
            documents_to_evaluate=documents_to_evaluate,
            confidence=confidence,
            reason=reason or "판단 완료",
        )

    # ========== 통합 평가 메서드 ==========

    def assess_retrieval_quality(
        self,
        query: str,
        documents: List[str],
        min_relevant_docs: int = 2,
        enable_early_stopping: bool = False,
        documents_to_evaluate: Optional[int] = None,
    ) -> OverallEvaluation:
        """
        검색된 문서 평가 + CRAG 액션 결정

        ISREL+CRAG 통합 평가를 수행하고 결과를 LangGraph 노드에서 사용하기 쉬운 형태로 정리한다.
        """
        if not documents:
            return OverallEvaluation(
                query=query,
                document_evaluations=[],
                should_retrieve_external=True,
                reason="검색 결과 없음",
                early_stopped=False,
                crag_action="incorrect",
            )

        subset = documents
        quick_limit: Optional[int] = None

        if enable_early_stopping and documents_to_evaluate:
            quick_limit = max(1, min(len(documents), documents_to_evaluate))
            subset = documents[:quick_limit]

        combined = self.evaluate_retrieval_and_decide_action(
            query=query,
            documents=subset,
            min_relevant_docs=min_relevant_docs,
        )

        # DocumentEvaluation 리스트 생성
        doc_evals = []
        for item in combined.document_evaluations:
            doc_evals.append(
                DocumentEvaluation(
                    document_content=item.document_content,
                    relevance=RelevanceResult(
                        relevance=item.relevance, confidence=item.confidence
                    ),
                )
            )

        # 관련 문서 카운트
        relevant_count = sum(
            1 for doc_eval in doc_evals if doc_eval.relevance.relevance == "relevant"
        )

        early_stopped = False
        if enable_early_stopping and quick_limit:
            early_stopped = len(subset) < len(documents) and relevant_count >= min_relevant_docs
            if early_stopped:
                remaining_docs = documents[len(subset) :]
                for doc in remaining_docs:
                    doc_evals.append(
                        DocumentEvaluation(
                            document_content=doc,
                            relevance=RelevanceResult(
                                relevance="not_evaluated", confidence=0.0
                            ),
                        )
                    )

        # 외부 검색 필요 여부
        should_retrieve_external = combined.crag_action in ["incorrect", "ambiguous"]

        reason = combined.reason or ""
        if not reason:
            if combined.crag_action == "correct":
                reason = f"관련 문서 {relevant_count}개로 충분함"
            elif combined.crag_action == "ambiguous":
                reason = f"관련 문서 {relevant_count}개, 외부 검색으로 보완 필요"
            else:
                reason = f"관련 문서 부족, 외부 검색 필요"

        if early_stopped:
            reason += " (조기 종료)"

        return OverallEvaluation(
            query=query,
            document_evaluations=doc_evals,
            should_retrieve_external=should_retrieve_external,
            reason=reason,
            early_stopped=early_stopped,
            crag_action=combined.crag_action,
        )

    def assess_answer_quality(
        self,
        query: str,
        answer: str,
        documents: List[str],
    ) -> AnswerQualityResult:
        """
        답변 품질 평가

        ISSUP+ISUSE 통합 평가 결과를 사용하여 재생성 여부를 판단한다.
        """
        return self.evaluate_answer_combined(query, answer, documents)


def create_evaluator(
    model_name: str = "gpt-5-mini", reasoning_effort="minimal", temperature: float = 0.0
) -> SelfRAGEvaluator:
    """Self-RAG 평가자 생성"""
    return SelfRAGEvaluator(
        model_name=model_name,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
    )
