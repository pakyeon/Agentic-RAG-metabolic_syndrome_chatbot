"""
Self-RAG 평가자
"""

import os
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
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
class CombinedRelevanceResult:
    """ISREL + CRAG 통합 결과"""

    doc_id: int
    relevance: str  # "relevant" | "irrelevant"
    confidence: float
    crag_action: str  # "correct" | "incorrect" | "ambiguous"


@dataclass
class CombinedAnswerResult:
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

    # ========== Relevance + CRAG 통합 평가 ==========

    def evaluate_relevance_and_crag(
        self,
        query: str,
        documents: List[str],
        enable_early_stopping: bool = False,
        min_relevant_docs: int = 2,
        max_docs_for_quick_check: Optional[int] = None,
    ) -> Tuple[List[RelevanceResult], str]:
        """
        ISREL + CRAG를 한 번에 평가

        관련성 평가와 CRAG 액션 결정을 동시에 수행하여 중복 제거

        Args:
            query: 사용자 질문
            documents: 검색된 문서들
            enable_early_stopping: 조기 종료 활성화
            min_relevant_docs: 최소 관련 문서 수

        Returns:
            (RelevanceResult 리스트, CRAG 액션)
        """
        if not documents:
            return [], "incorrect"

        # 조기 종료 처리
        if enable_early_stopping:
            quick_target = (
                max_docs_for_quick_check
                if max_docs_for_quick_check and max_docs_for_quick_check > 0
                else min_relevant_docs + 1
            )
            quick_check_count = min(max(1, quick_target), len(documents))
            quick_docs = documents[:quick_check_count]

            quick_results, crag_action = self._evaluate_relevance_and_crag_batch(
                query, quick_docs, min_relevant_docs
            )

            relevant_count = sum(1 for r in quick_results if r.relevance == "relevant")

            if relevant_count >= min_relevant_docs:
                remaining = [
                    RelevanceResult(relevance="not_evaluated", confidence=0.0)
                    for _ in range(len(documents) - quick_check_count)
                ]
                return quick_results + remaining, crag_action

        # 전체 평가
        return self._evaluate_relevance_and_crag_batch(
            query, documents, min_relevant_docs
        )

    def _evaluate_relevance_and_crag_batch(
        self, query: str, documents: List[str], min_relevant_docs: int
    ) -> Tuple[List[RelevanceResult], str]:
        """내부 헬퍼: ISREL + CRAG 통합 평가"""
        docs_text = ""
        for i, doc in enumerate(documents, 1):
            doc_preview = doc[:250] + "..." if len(doc) > 250 else doc
            docs_text += f"\n[문서 {i}]\n{doc_preview}\n"

        template = self._prompts.get("relevance_crag")

        if template:
            prompt = template.format(
                query=query,
                documents=docs_text,
                min_relevant_docs=min_relevant_docs,
            )
        else:
            prompt = f"""당신은 검색 품질 평가자입니다.

**시스템 역할**:
이 시스템은 대사증후군 상담사를 어시스턴트하는 전문 챗봇입니다.

**질문**: {query}

**검색된 문서들**:{docs_text}

**평가 기준**:
1. 각 문서가 질문에 답하는 데 유용한 정보를 포함하는지 평가
2. 전체 검색 품질을 바탕으로 CRAG 액션 결정

**CRAG 액션**:
- "correct": 관련 문서가 {min_relevant_docs}개 이상이고 품질이 충분함
- "ambiguous": 일부 관련 문서가 있지만 불충분함
- "incorrect": 관련 문서가 거의 없거나 품질이 낮음

**출력 형식 (JSON만 출력)**:
{{
  "documents": [
    {{"doc_id": 1, "relevance": "relevant", "confidence": 0.9}},
    {{"doc_id": 2, "relevance": "irrelevant", "confidence": 0.8}}
  ],
  "crag_action": "correct",
  "reason": "관련 문서 2개로 충분함"
}}

DO NOT OUTPUT ANYTHING OTHER THAN VALID JSON.

출력:"""
        try:
            response = self.llm.invoke(prompt)
            result_text = response.content.strip()
            result_text = result_text.replace("```json", "").replace("```", "").strip()

            import json

            result_json = json.loads(result_text)

            # 관련성 결과 파싱
            relevance_results = []
            for item in result_json.get("documents", []):
                relevance_results.append(
                    RelevanceResult(
                        relevance=item.get("relevance", "irrelevant"),
                        confidence=float(item.get("confidence", 0.5)),
                    )
                )

            # 문서 수에 맞춰 조정
            while len(relevance_results) < len(documents):
                relevance_results.append(
                    RelevanceResult(relevance="irrelevant", confidence=0.5)
                )

            # CRAG 액션 추출
            crag_action = result_json.get("crag_action", "ambiguous")

            return relevance_results[: len(documents)], crag_action

        except Exception as e:
            print(f"[Warning] 통합 평가 실패, 폴백: {e}")
            # 폴백: 개별 평가 + 수동 CRAG 판단
            relevance_results = [
                self.evaluate_relevance(query, doc) for doc in documents
            ]
            relevant_count = sum(
                1 for r in relevance_results if r.relevance == "relevant"
            )

            if relevant_count >= min_relevant_docs:
                crag_action = "correct"
            elif relevant_count > 0:
                crag_action = "ambiguous"
            else:
                crag_action = "incorrect"

            return relevance_results, crag_action

    def evaluate_answer_combined(
        self, query: str, answer: str, documents: List[str]
    ) -> CombinedAnswerResult:
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
            return CombinedAnswerResult(
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

        try:
            response = self.llm.invoke(prompt)
            result_text = response.content.strip()
            result_text = result_text.replace("```json", "").replace("```", "").strip()

            import json

            result_json = json.loads(result_text)

            # 지원도 결과 파싱
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

            return CombinedAnswerResult(
                support_results=support_results[: len(documents)],
                usefulness_score=int(result_json.get("usefulness_score", 3)),
                usefulness_confidence=float(
                    result_json.get("usefulness_confidence", 0.5)
                ),
                should_regenerate=bool(result_json.get("should_regenerate", False)),
                regenerate_reason=result_json.get("regenerate_reason", ""),
            )

        except Exception as e:
            print(f"[Warning] 통합 답변 평가 실패, 폴백: {e}")
            # 폴백: 개별 평가
            support_results = [
                self.evaluate_support(query, doc, answer) for doc in documents
            ]
            usefulness = self.evaluate_usefulness(query, answer)

            fully_supported_count = sum(
                1 for s in support_results if s.support == "fully_supported"
            )
            should_regenerate = usefulness.score < 3 or fully_supported_count == 0

            return CombinedAnswerResult(
                support_results=support_results,
                usefulness_score=usefulness.score,
                usefulness_confidence=usefulness.confidence,
                should_regenerate=should_regenerate,
                regenerate_reason="품질 부족" if should_regenerate else "품질 충분",
            )

    # ========== 검색 필요성 판단 & 기본 평가기 ==========

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

    def evaluate_relevance(self, query: str, document: str) -> RelevanceResult:
        """ISREL: 개별 평가"""
        prompt = f"""**질문**: {query}
**문서**: {document}

출력: relevant / irrelevant"""

        response = self.llm.invoke(prompt)
        result = response.content.strip().lower()

        if "relevant" in result and "irrelevant" not in result:
            return RelevanceResult(relevance="relevant", confidence=0.9)
        elif "irrelevant" in result:
            return RelevanceResult(relevance="irrelevant", confidence=0.9)
        else:
            return RelevanceResult(relevance="irrelevant", confidence=0.5)

    def evaluate_support(self, query: str, document: str, answer: str) -> SupportResult:
        """ISSUP: 개별 평가"""
        prompt = f"""**질문**: {query}
**문서**: {document}
**답변**: {answer}

출력: fully_supported / partially_supported / no_support"""

        response = self.llm.invoke(prompt)
        result = response.content.strip().lower().replace(" ", "_")

        if "fully_supported" in result:
            return SupportResult(support="fully_supported", confidence=0.9)
        elif "partially_supported" in result:
            return SupportResult(support="partially_supported", confidence=0.9)
        else:
            return SupportResult(support="no_support", confidence=0.9)

    def evaluate_usefulness(self, query: str, answer: str) -> UsefulnessResult:
        """ISUSE: 개별 평가"""
        prompt = f"""**질문**: {query}
**답변**: {answer}

출력: 1~5 점수"""

        response = self.llm.invoke(prompt)
        result = response.content.strip()

        try:
            score = int(result)
            if 1 <= score <= 5:
                return UsefulnessResult(score=score, confidence=0.9)
            else:
                return UsefulnessResult(score=3, confidence=0.5)
        except ValueError:
            return UsefulnessResult(score=3, confidence=0.5)

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

        relevance_results, crag_action = self.evaluate_relevance_and_crag(
            query,
            documents,
            enable_early_stopping,
            min_relevant_docs,
            max_docs_for_quick_check=documents_to_evaluate,
        )

        # DocumentEvaluation 리스트 생성
        doc_evals = []
        for doc, rel_result in zip(documents, relevance_results):
            doc_evals.append(
                DocumentEvaluation(document_content=doc, relevance=rel_result)
            )

        # 관련 문서 카운트
        relevant_count = sum(
            1 for doc_eval in doc_evals if doc_eval.relevance.relevance == "relevant"
        )

        early_stopped = any(
            doc_eval.relevance.relevance == "not_evaluated" for doc_eval in doc_evals
        )

        # 외부 검색 필요 여부
        should_retrieve_external = crag_action in ["incorrect", "ambiguous"]

        if crag_action == "correct":
            reason = f"관련 문서 {relevant_count}개로 충분함"
        elif crag_action == "ambiguous":
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
            crag_action=crag_action,
        )

    def assess_answer_quality(
        self,
        query: str,
        answer: str,
        documents: List[str],
    ) -> Dict[str, Any]:
        """
        답변 품질 평가

        ISSUP+ISUSE 통합 평가 결과를 사용하여 재생성 여부를 판단한다.
        """
        combined_result = self.evaluate_answer_combined(query, answer, documents)

        return {
            "support_results": combined_result.support_results,
            "usefulness": UsefulnessResult(
                score=combined_result.usefulness_score,
                confidence=combined_result.usefulness_confidence,
            ),
            "fully_supported_count": sum(
                1
                for s in combined_result.support_results
                if s.support == "fully_supported"
            ),
            "should_regenerate": combined_result.should_regenerate,
            "regenerate_reason": combined_result.regenerate_reason,
        }


def create_evaluator(
    model_name: str = "gpt-5-mini", reasoning_effort="minimal", temperature: float = 0.0
) -> SelfRAGEvaluator:
    """Self-RAG 평가자 생성"""
    return SelfRAGEvaluator(
        model_name=model_name,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
    )
