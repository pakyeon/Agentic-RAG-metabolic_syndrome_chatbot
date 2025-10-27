"""
Self-RAG 평가자
"""

import os
from typing import Dict, List, Optional, Tuple
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

    decision: str  # "yes" | "no" | "continue"
    confidence: float  # 0.0-1.0
    reason: str  # 판단 이유


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

    # ========== Task 1.3: 평가 단계 통합 ==========

    def evaluate_relevance_and_crag(
        self,
        query: str,
        documents: List[str],
        enable_early_stopping: bool = False,
        min_relevant_docs: int = 2,
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
            quick_check_count = min(min_relevant_docs + 1, len(documents))
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

    # ========== Task 1.1 + 1.2: 기존 배치 평가 + 조기 종료 ==========

    def evaluate_relevance_batch(
        self,
        query: str,
        documents: List[str],
        enable_early_stopping: bool = False,
        min_relevant_docs: int = 2,
    ) -> List[RelevanceResult]:
        """ISREL: 배치 평가 + 조기 종료 (Task 1.3에서는 통합 버전 사용 권장)"""
        if not documents:
            return []

        if enable_early_stopping:
            quick_check_count = min(min_relevant_docs + 1, len(documents))
            quick_docs = documents[:quick_check_count]

            quick_results = self._batch_evaluate(query, quick_docs)

            relevant_count = sum(1 for r in quick_results if r.relevance == "relevant")

            if relevant_count >= min_relevant_docs:
                remaining = [
                    RelevanceResult(relevance="not_evaluated", confidence=0.0)
                    for _ in range(len(documents) - quick_check_count)
                ]
                return quick_results + remaining

        return self._batch_evaluate(query, documents)

    def _batch_evaluate(
        self, query: str, documents: List[str]
    ) -> List[RelevanceResult]:
        """내부 배치 평가 헬퍼"""
        docs_text = ""
        for i, doc in enumerate(documents, 1):
            doc_preview = doc[:250] + "..." if len(doc) > 250 else doc
            docs_text += f"\n[문서 {i}]\n{doc_preview}\n"

        prompt = f"""당신은 검색 품질 평가자입니다.

**질문**: {query}

**검색된 문서들**:{docs_text}

**출력 형식 (JSON만 출력)**:
[
  {{"doc_id": 1, "relevance": "relevant", "confidence": 0.9}},
  {{"doc_id": 2, "relevance": "irrelevant", "confidence": 0.8}}
]

DO NOT OUTPUT ANYTHING OTHER THAN VALID JSON.

출력:"""

        try:
            response = self.llm.invoke(prompt)
            result_text = response.content.strip()
            result_text = result_text.replace("```json", "").replace("```", "").strip()

            import json

            results_json = json.loads(result_text)

            relevance_results = []
            for item in results_json:
                relevance_results.append(
                    RelevanceResult(
                        relevance=item.get("relevance", "irrelevant"),
                        confidence=float(item.get("confidence", 0.5)),
                    )
                )

            while len(relevance_results) < len(documents):
                relevance_results.append(
                    RelevanceResult(relevance="irrelevant", confidence=0.5)
                )

            return relevance_results[: len(documents)]

        except Exception as e:
            print(f"[Warning] 배치 평가 실패: {e}")
            return [self.evaluate_relevance(query, doc) for doc in documents]

    def evaluate_support_batch(
        self, query: str, documents: List[str], answer: str
    ) -> List[SupportResult]:
        """ISSUP: 배치 평가 (Task 1.3에서는 통합 버전 사용 권장)"""
        if not documents:
            return []

        docs_text = ""
        for i, doc in enumerate(documents, 1):
            doc_preview = doc[:250] + "..." if len(doc) > 250 else doc
            docs_text += f"\n[문서 {i}]\n{doc_preview}\n"

        prompt = f"""당신은 답변 검증 평가자입니다.

**질문**: {query}
**생성된 답변**: {answer}
**참조 문서들**:{docs_text}

**출력 형식 (JSON만 출력)**:
[
  {{"doc_id": 1, "support": "fully_supported", "confidence": 0.9}},
  {{"doc_id": 2, "support": "partially_supported", "confidence": 0.7}}
]

DO NOT OUTPUT ANYTHING OTHER THAN VALID JSON.

출력:"""

        try:
            response = self.llm.invoke(prompt)
            result_text = response.content.strip()
            result_text = result_text.replace("```json", "").replace("```", "").strip()

            import json

            results_json = json.loads(result_text)

            support_results = []
            for item in results_json:
                support_results.append(
                    SupportResult(
                        support=item.get("support", "no_support"),
                        confidence=float(item.get("confidence", 0.5)),
                    )
                )

            while len(support_results) < len(documents):
                support_results.append(
                    SupportResult(support="no_support", confidence=0.5)
                )

            return support_results[: len(documents)]

        except Exception as e:
            print(f"[Warning] 배치 지원도 평가 실패: {e}")
            return [self.evaluate_support(query, doc, answer) for doc in documents]

    # ========== 개별 평가 메서드 (폴백용) ==========

    def evaluate_retrieve_need(
        self, query: str, context: Optional[str] = None
    ) -> RetrieveResult:
        """[Retrieve]: 검색이 필요한지 판단"""
        if context:
            prompt = f"""당신은 검색 필요성 판단자입니다.

**질문**: {query}
**현재 컨텍스트**: {context}

**판단 기준**:
1. 대사증후군 관련 전문 정보 필요 → YES
2. 단순 인사/일상 대화 → NO
3. 현재 컨텍스트로 충분 → CONTINUE

**출력**: yes / no / continue
그 다음 줄에 이유

출력:"""
        else:
            prompt = f"""당신은 검색 필요성 판단자입니다.

**질문**: {query}

**판단 기준**:
1. 대사증후군 관련 전문 정보 필요 → YES
2. 단순 인사/일상 대화 → NO

**출력**: yes / no
그 다음 줄에 이유

출력:"""

        response = self.llm.invoke(prompt)
        result = response.content.strip().lower()

        lines = result.split("\n", 1)
        decision = lines[0].strip()
        reason = lines[1].strip() if len(lines) > 1 else "판단 완료"

        if "yes" in decision:
            return RetrieveResult(decision="yes", confidence=0.9, reason=reason)
        elif "no" in decision:
            return RetrieveResult(decision="no", confidence=0.9, reason=reason)
        elif "continue" in decision:
            return RetrieveResult(decision="continue", confidence=0.9, reason=reason)
        else:
            return RetrieveResult(decision="yes", confidence=0.5, reason="판단 불명확")

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

    # ========== 통합 평가 메서드 (Task 1.3 적용) ==========

    def evaluate_documents(
        self,
        query: str,
        documents: List[str],
        min_relevant_docs: int = 2,
        enable_early_stopping: bool = False,
        use_integrated: bool = True,  # 통합 평가 사용
    ) -> OverallEvaluation:
        """
        검색된 문서 평가 + CRAG 액션 결정

        ISREL+CRAG 통합 평가 (use_integrated=True)
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

        if use_integrated:
            # 통합 평가
            relevance_results, crag_action = self.evaluate_relevance_and_crag(
                query, documents, enable_early_stopping, min_relevant_docs
            )
        else:
            # 기존 방식 (Task 1.1 + 1.2)
            relevance_results = self.evaluate_relevance_batch(
                query, documents, enable_early_stopping, min_relevant_docs
            )
            relevant_count = sum(
                1 for r in relevance_results if r.relevance == "relevant"
            )
            if relevant_count >= min_relevant_docs:
                crag_action = "correct"
            elif relevant_count > 0:
                crag_action = "ambiguous"
            else:
                crag_action = "incorrect"

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

    def evaluate_answer_quality(
        self,
        query: str,
        answer: str,
        documents: List[str],
        use_integrated: bool = True,  # 통합 평가 사용
    ) -> Dict[str, any]:
        """
        답변 품질 평가

        ISSUP+ISUSE 통합 평가 (use_integrated=True)
        """
        if use_integrated:
            # 통합 평가
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
        else:
            support_results = self.evaluate_support_batch(query, documents, answer)
            usefulness = self.evaluate_usefulness(query, answer)

            fully_supported_count = sum(
                1 for s in support_results if s.support == "fully_supported"
            )

            return {
                "support_results": support_results,
                "usefulness": usefulness,
                "fully_supported_count": fully_supported_count,
                "should_regenerate": (
                    usefulness.score < 3 or fully_supported_count == 0
                ),
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
