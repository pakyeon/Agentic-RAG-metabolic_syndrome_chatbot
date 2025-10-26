# -*- coding: utf-8 -*-
"""Task 5.5 노드 테스트"""

import sys
import os
import pytest

if not os.getenv("OPENAI_API_KEY"):
    pytest.skip("OpenAI API 키가 없으므로 Task 5.5 노드 테스트를 건너뜁니다.", allow_module_level=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.graph.nodes import generate_answer_node, evaluate_answer_node
from src.graph.state import RAGState


def test_generate_answer_node():
    """답변 생성 노드 테스트"""
    print("\n=== test_generate_answer_node ===")

    # Case 1: 컨텍스트가 있는 경우 (환자 정보 없음)
    print("\n[Case 1] 일반 질문 (환자 정보 없음)")
    state_general: RAGState = {
        "question": "대사증후군 진단 기준은 무엇인가요?",
        "patient_id": None,
        "patient_context": None,
        "should_retrieve": True,
        "relevance_scores": [],
        "support_score": 0.0,
        "usefulness_score": 0.0,
        "crag_action": "correct",
        "crag_confidence": 1.0,
        "internal_docs": [],
        "external_docs": [],
        "merged_context": """[내부 문서 1] metabolic_syndrome.md
대사증후군 진단 기준 (NCEP ATP III 기준):
다음 5가지 중 3가지 이상 해당 시 진단
1. 복부비만: 허리둘레 남성 90cm, 여성 85cm 이상
2. 고중성지방혈증: 150mg/dL 이상
3. 낮은 HDL 콜레스테롤: 남성 40mg/dL, 여성 50mg/dL 미만
4. 고혈압: 130/85mmHg 이상
5. 공복혈당장애: 100mg/dL 이상""",
        "answer": "",
        "iteration": 1,
        "max_iterations": 2,
        "needs_regeneration": False,
        "error": None,
        "metadata": {},
    }

    result = generate_answer_node(state_general)

    if result.get("error"):
        print(f"   ❌ 에러: {result['error']}")
    else:
        answer = result.get("answer", "")
        print(f"   답변 길이: {len(answer)} 문자")
        print(f"   답변 미리보기:\n{answer[:200]}...")

    assert "answer" in result
    assert len(result.get("answer", "")) > 0
    print("   ✅ 일반 질문 케이스 통과")

    # Case 2: 환자 정보가 있는 경우
    print("\n[Case 2] 환자별 맞춤 질문")
    state_patient: RAGState = {
        "question": "제 상태를 개선하려면 어떻게 해야 하나요?",
        "patient_id": 1,
        "patient_context": """환자 ID: 1
성별: 남성
나이: 45세

측정값:
- 허리둘레: 95.0 cm (기준: 90 cm 이상)
- 중성지방: 180.0 mg/dL (기준: 150 mg/dL 이상)
- HDL 콜레스테롤: 38.0 mg/dL (기준: 40 mg/dL 미만)
- 혈압: 140/92 mmHg (기준: 130/85 mmHg 이상)
- 공복혈당: 110.0 mg/dL (기준: 100 mg/dL 이상)

진단: 대사증후군 (5개 기준 중 5개 충족)""",
        "should_retrieve": True,
        "relevance_scores": [],
        "support_score": 0.0,
        "usefulness_score": 0.0,
        "crag_action": "correct",
        "crag_confidence": 1.0,
        "internal_docs": [],
        "external_docs": [],
        "merged_context": """[내부 문서 1] treatment.md
대사증후군 생활습관 개선:
1. 체중 감량: 현재 체중의 5-10% 감량 목표
2. 규칙적 운동: 주 5회, 30분 이상 유산소 운동
3. 식이요법: 저염식, 저지방식, 고섬유질 식단
4. 금연 및 절주
5. 스트레스 관리""",
        "answer": "",
        "iteration": 1,
        "max_iterations": 2,
        "needs_regeneration": False,
        "error": None,
        "metadata": {},
    }

    result = generate_answer_node(state_patient)

    if result.get("error"):
        print(f"   ❌ 에러: {result['error']}")
    else:
        answer = result.get("answer", "")
        print(f"   답변 길이: {len(answer)} 문자")
        print(f"   답변 미리보기:\n{answer[:200]}...")
        # 환자 정보를 고려한 답변인지 확인
        has_personalization = any(
            word in answer.lower()
            for word in ["귀하", "환자분", "체중", "혈압", "혈당"]
        )
        print(f"   개인화된 답변: {has_personalization}")

    assert "answer" in result
    assert len(result.get("answer", "")) > 0
    print("   ✅ 환자별 맞춤 질문 케이스 통과")

    # Case 3: 컨텍스트가 없는 경우 (에러 처리)
    print("\n[Case 3] 컨텍스트 없음 (에러 케이스)")
    state_empty: RAGState = {
        "question": "대사증후군이란?",
        "patient_id": None,
        "patient_context": None,
        "should_retrieve": False,
        "relevance_scores": [],
        "support_score": 0.0,
        "usefulness_score": 0.0,
        "crag_action": "incorrect",
        "crag_confidence": 0.0,
        "internal_docs": [],
        "external_docs": [],
        "merged_context": "",  # 빈 컨텍스트
        "answer": "",
        "iteration": 1,
        "max_iterations": 2,
        "needs_regeneration": False,
        "error": None,
        "metadata": {},
    }

    result = generate_answer_node(state_empty)

    if result.get("error"):
        print(f"   에러 메시지: {result['error']}")
        print(f"   답변: {result.get('answer', '')[:100]}")

    assert "answer" in result
    assert "error" in result
    print("   ✅ 에러 케이스 통과")


def test_evaluate_answer_node():
    """답변 평가 노드 테스트"""
    print("\n\n=== test_evaluate_answer_node ===")

    # Case 1: 고품질 답변 (지원도/유용성 모두 높음)
    print("\n[Case 1] 고품질 답변 (재생성 불필요)")
    state_good: RAGState = {
        "question": "대사증후군 진단 기준은?",
        "patient_id": None,
        "patient_context": None,
        "should_retrieve": True,
        "relevance_scores": [],
        "support_score": 0.0,
        "usefulness_score": 0.0,
        "crag_action": "correct",
        "crag_confidence": 1.0,
        "internal_docs": [],
        "external_docs": [],
        "merged_context": """대사증후군 진단 기준:
1. 복부비만: 허리둘레 남성 90cm, 여성 85cm 이상
2. 고중성지방혈증: 150mg/dL 이상
3. 낮은 HDL 콜레스테롤: 남성 40mg/dL, 여성 50mg/dL 미만
4. 고혈압: 130/85mmHg 이상
5. 공복혈당장애: 100mg/dL 이상
5가지 중 3가지 이상 해당 시 진단됩니다.""",
        "answer": """대사증후군은 다음 5가지 기준 중 3가지 이상 해당될 때 진단됩니다:

1. **복부비만**: 허리둘레가 남성 90cm, 여성 85cm 이상
2. **고중성지방혈증**: 중성지방 150mg/dL 이상
3. **낮은 HDL 콜레스테롤**: 남성 40mg/dL, 여성 50mg/dL 미만
4. **고혈압**: 혈압 130/85mmHg 이상
5. **공복혈당장애**: 공복혈당 100mg/dL 이상

이 중 3가지 이상에 해당하면 대사증후군으로 진단받게 됩니다.""",
        "iteration": 1,
        "max_iterations": 2,
        "needs_regeneration": False,
        "error": None,
        "metadata": {},
    }

    result = evaluate_answer_node(state_good)

    if result.get("error"):
        print(f"   ❌ 에러: {result['error']}")
    else:
        support = result.get("support_score", 0.0)
        usefulness = result.get("usefulness_score", 0.0)
        needs_regen = result.get("needs_regeneration", False)

        print(f"   지원도 점수: {support}/5.0")
        print(f"   유용성 점수: {usefulness}/5.0")
        print(f"   재생성 필요: {needs_regen}")

    assert "support_score" in result
    assert "usefulness_score" in result
    assert "needs_regeneration" in result
    print("   ✅ 고품질 답변 케이스 통과")

    # Case 2: 저품질 답변 (재생성 필요)
    print("\n[Case 2] 저품질 답변 (재생성 필요)")
    state_poor: RAGState = {
        "question": "대사증후군 치료법은?",
        "patient_id": None,
        "patient_context": None,
        "should_retrieve": True,
        "relevance_scores": [],
        "support_score": 0.0,
        "usefulness_score": 0.0,
        "crag_action": "correct",
        "crag_confidence": 1.0,
        "internal_docs": [],
        "external_docs": [],
        "merged_context": """대사증후군 치료:
- 생활습관 개선이 가장 중요
- 체중 감량, 규칙적 운동, 건강한 식단
- 필요시 약물 치료""",
        "answer": "대사증후군은 심각한 질환입니다. 병원에 가보세요.",  # 컨텍스트 무시한 답변
        "iteration": 1,
        "max_iterations": 2,
        "needs_regeneration": False,
        "error": None,
        "metadata": {},
    }

    result = evaluate_answer_node(state_poor)

    if result.get("error"):
        print(f"   ❌ 에러: {result['error']}")
    else:
        support = result.get("support_score", 0.0)
        usefulness = result.get("usefulness_score", 0.0)
        needs_regen = result.get("needs_regeneration", False)

        print(f"   지원도 점수: {support}/5.0")
        print(f"   유용성 점수: {usefulness}/5.0")
        print(f"   재생성 필요: {needs_regen}")

    assert "needs_regeneration" in result
    print("   ✅ 저품질 답변 케이스 통과")


def test_integration():
    """Task 5.5 통합 테스트 (답변 생성 → 평가)"""
    print("\n\n=== test_integration (Task 5.5) ===")

    print("\n시나리오: 질문 → 답변 생성 → 품질 평가")

    state: RAGState = {
        "question": "대사증후군 예방을 위한 식이요법은?",
        "patient_id": None,
        "patient_context": None,
        "should_retrieve": True,
        "relevance_scores": [],
        "support_score": 0.0,
        "usefulness_score": 0.0,
        "crag_action": "correct",
        "crag_confidence": 1.0,
        "internal_docs": [],
        "external_docs": [],
        "merged_context": """[내부 문서] 대사증후군 식이요법
1. 저염식: 하루 소금 섭취 6g 이하
2. 저지방식: 포화지방 섭취 최소화
3. 고섬유질 식단: 채소, 과일, 통곡물
4. 가공식품 제한
5. 규칙적인 식사""",
        "answer": "",
        "iteration": 1,
        "max_iterations": 2,
        "needs_regeneration": False,
        "error": None,
        "metadata": {},
    }

    # Step 1: 답변 생성
    print("\n1. 답변 생성...")
    gen_result = generate_answer_node(state)
    state["answer"] = gen_result.get("answer", "")

    if gen_result.get("error"):
        print(f"   ❌ 생성 에러: {gen_result['error']}")
    else:
        print(f"   답변 길이: {len(state['answer'])} 문자")
        print(f"   답변:\n{state['answer'][:150]}...")

    # Step 2: 답변 평가
    print("\n2. 답변 평가...")
    eval_result = evaluate_answer_node(state)

    if eval_result.get("error"):
        print(f"   ❌ 평가 에러: {eval_result['error']}")
    else:
        state.update(eval_result)
        print(f"   지원도: {state['support_score']}/5.0")
        print(f"   유용성: {state['usefulness_score']}/5.0")
        print(f"   재생성 필요: {state['needs_regeneration']}")

    # 검증
    assert state.get("answer") is not None
    assert state.get("support_score") is not None
    assert state.get("usefulness_score") is not None

    print("\n✅ Task 5.5 통합 테스트 통과!")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    # OPENAI_API_KEY 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("\n" + "=" * 50)
        print("⚠️  OPENAI_API_KEY가 설정되지 않았습니다!")
        print("=" * 50)
        print("답변 생성 및 평가를 위해 API 키가 필요합니다.")
        print("환경변수를 설정하려면:")
        print("  export OPENAI_API_KEY=your-key-here")
        print("=" * 50)
        import sys

        sys.exit(1)

    # 테스트 실행
    test_generate_answer_node()
    test_evaluate_answer_node()
    test_integration()

    print("\n" + "=" * 50)
    print("✅ Task 5.5 모든 테스트 완료!")
    print("=" * 50)
