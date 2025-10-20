"""
Self-RAG 평가자

Self-RAG 논문의 4가지 Reflection Tokens를 프롬프트 기반으로 구현:
- [Retrieve]: 검색 필요성 판단
- ISREL: 검색 문서의 관련성 평가
- ISSUP: 생성 답변의 지지도 평가
- ISUSE: 답변의 유용성 평가
"""

import os
from typing import List, Dict, Optional
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
    support: Optional[SupportResult] = None  # 답변 생성 전에는 None


@dataclass
class OverallEvaluation:
    """전체 검색 결과 평가"""

    query: str
    document_evaluations: List[DocumentEvaluation]
    should_retrieve_external: bool
    reason: str


class SelfRAGEvaluator:
    """
    Self-RAG 평가자

    프롬프트 기반으로 Self-RAG의 reflection tokens를 구현합니다.
    원래 Self-RAG는 모델을 fine-tuning하지만, 우리는 일반 LLM을 사용합니다.
    """

    def __init__(
        self,
        model_name: str = "gpt-5-mini",
        reasoning_effort="minimal",
        temperature: float = 0.0,
    ):
        """
        Args:
            model_name: 평가에 사용할 LLM 모델
            reasoning_effort: LLM의 Thinking 수준
            temperature: 평가 일관성을 위해 0.0 권장
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    def evaluate_retrieve_need(
        self, query: str, context: Optional[str] = None
    ) -> RetrieveResult:
        """
        [Retrieve]: 검색이 필요한지 판단

        Self-RAG의 첫 번째 단계로, LLM이 자체 지식으로 답할 수 있는지
        아니면 외부 검색이 필요한지 판단합니다.

        Args:
            query: 사용자 질문
            context: 이전 대화 내용 또는 부분 생성 내용 (선택)

        Returns:
            RetrieveResult (yes/no/continue + 신뢰도 + 이유)
            - yes: 검색 필요
            - no: LLM 지식으로 충분
            - continue: 현재 진행 중, 추가 검색 불필요
        """
        if context:
            prompt = f"""당신은 검색 필요성 판단자입니다.

**질문**: {query}

**현재 컨텍스트**:
{context}

**판단 기준**:
1. 질문이 최신 정보나 특정 사실을 요구하는가?
2. LLM의 일반 지식으로 충분히 답변 가능한가?
3. 현재 컨텍스트에 추가 정보가 필요한가?

**출력 형식**:
반드시 다음 중 하나만 출력하세요:
- "yes" (외부 검색 필요)
- "no" (LLM 지식으로 충분)
- "continue" (현재 진행 중, 추가 검색 불필요)

그 다음 줄에 간단한 이유를 한 문장으로 작성하세요.

출력:"""
        else:
            prompt = f"""당신은 검색 필요성 판단자입니다.

**질문**: {query}

**판단 기준**:
1. 질문이 대사증후군 관련 의료 정보인가?
2. 최신 연구나 특정 통계를 요구하는가?
3. LLM의 일반 의료 지식으로 답변 가능한가?

**출력 형식**:
반드시 다음 중 하나만 출력하세요:
- "yes" (외부 검색 필요)
- "no" (LLM 지식으로 충분)

그 다음 줄에 간단한 이유를 한 문장으로 작성하세요.

출력:"""

        response = self.llm.invoke(prompt)
        result = response.content.strip().lower()

        # 파싱: 첫 줄은 decision, 나머지는 reason
        lines = result.split("\n", 1)
        decision_line = lines[0].strip()
        reason = lines[1].strip() if len(lines) > 1 else ""

        # decision 파싱
        if "yes" in decision_line and "no" not in decision_line:
            decision = "yes"
            confidence = 0.9
        elif "no" in decision_line:
            decision = "no"
            confidence = 0.9
        elif "continue" in decision_line:
            decision = "continue"
            confidence = 0.9
        else:
            # 파싱 실패 시 보수적으로 검색 필요로 판단
            decision = "yes"
            confidence = 0.5
            reason = "판단 실패, 보수적으로 검색 필요"

        return RetrieveResult(
            decision=decision,
            confidence=confidence,
            reason=reason if reason else f"검색 {decision}",
        )

    def evaluate_relevance(self, query: str, document: str) -> RelevanceResult:
        """
        ISREL: 검색된 문서가 질문과 관련 있는지 평가

        Args:
            query: 사용자 질문
            document: 검색된 문서 내용

        Returns:
            RelevanceResult (relevant/irrelevant + 신뢰도)
        """
        prompt = f"""당신은 검색 품질 평가자입니다.

**질문**: {query}

**검색된 문서**:
{document}

**평가 기준**:
검색된 문서가 질문에 답하는 데 유용한 정보를 포함하고 있는가?

**출력 형식**:
반드시 다음 중 하나만 출력하세요:
- "relevant" (관련 있음)
- "irrelevant" (관련 없음)

출력:"""

        response = self.llm.invoke(prompt)
        result = response.content.strip().lower()

        # 파싱
        if "relevant" in result and "irrelevant" not in result:
            return RelevanceResult(relevance="relevant", confidence=0.9)
        elif "irrelevant" in result:
            return RelevanceResult(relevance="irrelevant", confidence=0.9)
        else:
            # 파싱 실패 시 기본값
            return RelevanceResult(relevance="irrelevant", confidence=0.5)

    def evaluate_support(self, query: str, document: str, answer: str) -> SupportResult:
        """
        ISSUP: 생성된 답변이 검색 문서로 뒷받침되는지 평가

        Args:
            query: 사용자 질문
            document: 검색된 문서 내용
            answer: LLM이 생성한 답변

        Returns:
            SupportResult (fully_supported/partially_supported/no_support + 신뢰도)
        """
        prompt = f"""당신은 답변 검증 평가자입니다.

**질문**: {query}

**검색된 문서**:
{document}

**생성된 답변**:
{answer}

**평가 기준**:
답변의 모든 주장이 문서로 뒷받침되는가?

**출력 형식**:
반드시 다음 중 하나만 출력하세요:
- "fully_supported" (완전히 뒷받침됨)
- "partially_supported" (부분적으로 뒷받침됨)
- "no_support" (뒷받침되지 않음)

출력:"""

        response = self.llm.invoke(prompt)
        result = response.content.strip().lower().replace(" ", "_")

        # 파싱
        if "fully_supported" in result or "fully supported" in result:
            return SupportResult(support="fully_supported", confidence=0.9)
        elif "partially_supported" in result or "partially supported" in result:
            return SupportResult(support="partially_supported", confidence=0.9)
        elif "no_support" in result or "no support" in result:
            return SupportResult(support="no_support", confidence=0.9)
        else:
            # 파싱 실패 시 보수적 기본값
            return SupportResult(support="partially_supported", confidence=0.5)

    def evaluate_usefulness(self, query: str, answer: str) -> UsefulnessResult:
        """
        ISUSE: 답변이 질문에 유용한지 평가

        Args:
            query: 사용자 질문
            answer: LLM이 생성한 답변

        Returns:
            UsefulnessResult (1-5 점수 + 신뢰도)
        """
        prompt = f"""당신은 답변 품질 평가자입니다.

**질문**: {query}

**생성된 답변**:
{answer}

**평가 기준**:
답변이 질문에 유용하게 답변하고 있는가?

**점수 기준**:
- 5점: 매우 유용하고 완벽한 답변
- 4점: 유용하고 대부분 답변함
- 3점: 보통, 일부만 답변함
- 2점: 별로 유용하지 않음
- 1점: 전혀 유용하지 않음

**출력 형식**:
반드시 숫자 하나만 출력하세요 (1, 2, 3, 4, 또는 5):"""

        response = self.llm.invoke(prompt)
        result = response.content.strip()

        # 파싱
        try:
            score = int(result)
            if 1 <= score <= 5:
                return UsefulnessResult(score=score, confidence=0.9)
            else:
                return UsefulnessResult(score=3, confidence=0.5)
        except ValueError:
            # 파싱 실패 시 중간값
            return UsefulnessResult(score=3, confidence=0.5)

    def evaluate_documents(
        self, query: str, documents: List[str], min_relevant_docs: int = 2
    ) -> OverallEvaluation:
        """
        검색된 모든 문서 평가 및 외부 검색 필요성 판단

        Args:
            query: 사용자 질문
            documents: 검색된 문서 리스트
            min_relevant_docs: 최소 관련 문서 수

        Returns:
            OverallEvaluation (문서별 평가 + 외부 검색 필요 여부)
        """
        doc_evals = []
        relevant_count = 0

        for doc in documents:
            relevance_result = self.evaluate_relevance(query, doc)
            doc_eval = DocumentEvaluation(
                document_content=doc, relevance=relevance_result
            )
            doc_evals.append(doc_eval)

            if relevance_result.relevance == "relevant":
                relevant_count += 1

        # 외부 검색 필요성 판단
        should_retrieve_external = relevant_count < min_relevant_docs

        if should_retrieve_external:
            reason = f"관련 문서가 {relevant_count}개로 최소 {min_relevant_docs}개 미만입니다."
        else:
            reason = f"관련 문서가 {relevant_count}개로 충분합니다."

        return OverallEvaluation(
            query=query,
            document_evaluations=doc_evals,
            should_retrieve_external=should_retrieve_external,
            reason=reason,
        )

    def evaluate_answer_quality(
        self, query: str, answer: str, documents: List[str]
    ) -> Dict[str, any]:
        """
        생성된 답변의 전체 품질 평가

        Args:
            query: 사용자 질문
            answer: 생성된 답변
            documents: 참조한 문서들

        Returns:
            종합 평가 결과 딕셔너리
        """
        # ISSUP: 각 문서에 대한 지지도 평가
        support_results = []
        for doc in documents:
            support = self.evaluate_support(query, doc, answer)
            support_results.append(support)

        # ISUSE: 답변 유용성 평가
        usefulness = self.evaluate_usefulness(query, answer)

        # 종합 판단
        fully_supported_count = sum(
            1 for s in support_results if s.support == "fully_supported"
        )

        return {
            "support_results": support_results,
            "usefulness": usefulness,
            "fully_supported_count": fully_supported_count,
            "should_regenerate": (usefulness.score < 3 or fully_supported_count == 0),
        }


# 편의 함수
def create_evaluator(
    model_name: str = "gpt-5-mini", reasoning_effort="minimal", temperature: float = 0.0
) -> SelfRAGEvaluator:
    """
    Self-RAG 평가자 생성

    Example:
        >>> evaluator = create_evaluator()

        >>> # [Retrieve]: 검색 필요성 판단
        >>> retrieve_result = evaluator.evaluate_retrieve_need(
        ...     "대사증후군의 최신 치료법은?"
        ... )
        >>> print(retrieve_result.decision)  # "yes" | "no" | "continue"

        >>> # ISREL: 문서 관련성 평가
        >>> relevance = evaluator.evaluate_relevance(
        ...     "대사증후군이란?",
        ...     "대사증후군은 복부 비만, 고혈압 등이 동반되는 질환입니다."
        ... )
        >>> print(relevance.relevance)  # "relevant" | "irrelevant"
    """
    return SelfRAGEvaluator(
        model_name=model_name,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
    )
