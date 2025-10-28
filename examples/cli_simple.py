# -*- coding: utf-8 -*-
"""
간단한 CLI 인터페이스

환자 선택 → 질문 → 답변 구조
"""

import sys
import os

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def extract_answer_from_state(final_state: dict) -> str:
    """메시지 기반 상태에서 답변 추출"""
    messages = final_state.get("messages", [])

    # 마지막 AI 메시지 찾기
    for msg in reversed(messages):
        if msg.type == "ai":
            return msg.content

    return "답변을 생성할 수 없습니다."


def display_patient_list(db):
    """
    환자 리스트 표시 (exam_at 기준 최신순 정렬)

    Args:
        db: PatientDatabase 인스턴스

    Returns:
        정렬된 환자 리스트
    """
    from src.data.patient_db import PatientDatabase

    # 모든 환자 조회
    all_patients = db.get_all_patients()

    # 각 환자의 최신 검진일 조회 및 정렬
    patients_with_exam = []
    for patient in all_patients:
        latest_exam = db.get_latest_exam(patient["patient_id"])
        if latest_exam:
            patients_with_exam.append(
                {
                    **patient,
                    "exam_at": latest_exam["exam_at"],
                }
            )

    # exam_at 기준 최신순 정렬 (내림차순)
    patients_sorted = sorted(
        patients_with_exam, key=lambda x: x["exam_at"], reverse=True
    )

    # 환자 리스트 출력
    print("\n" + "=" * 70)
    print("환자 목록 (최근 검진순)")
    print("=" * 70)

    for idx, patient in enumerate(patients_sorted, 1):
        print(
            f"[{idx}] {patient['name']} ({patient['sex']}, {patient['age']}세) - 검진일: {patient['exam_at']}"
        )

    print(f"[0] 환자 정보 없이 진행")
    print("=" * 70)

    return patients_sorted


def select_patient(patients_sorted):
    """
    사용자로부터 환자 선택 입력 받기

    Args:
        patients_sorted: 정렬된 환자 리스트

    Returns:
        선택된 patient_id 또는 None
    """
    while True:
        try:
            choice = input("\n환자 번호를 선택하세요: ").strip()

            if not choice:
                continue

            choice_num = int(choice)

            if choice_num == 0:
                print("\n환자 정보 없이 진행합니다.")
                return None

            if 1 <= choice_num <= len(patients_sorted):
                selected = patients_sorted[choice_num - 1]
                print(
                    f"\n선택된 환자: {selected['name']} ({selected['sex']}, {selected['age']}세)"
                )
                return selected["patient_id"]
            else:
                print(f"⚠️  1~{len(patients_sorted)} 또는 0을 입력하세요.")

        except ValueError:
            print("⚠️  숫자를 입력하세요.")
        except KeyboardInterrupt:
            print("\n\n프로그램을 종료합니다.")
            sys.exit(0)


def question_loop(patient_id):
    """
    질문 입력 루프

    Args:
        patient_id: 선택된 환자 ID (None 가능)
    """
    from src.graph.workflow import run_rag

    print("\n" + "=" * 70)
    print("질문을 입력하세요 (종료: exit 또는 quit)")
    print("=" * 70)

    while True:
        try:
            # 질문 입력
            question = input("\n질문> ").strip()

            if not question:
                continue

            # 종료 명령
            if question.lower() in ["exit", "quit", "종료"]:
                print("\n감사합니다! 👋")
                break

            # RAG 실행
            print("\n[처리 중...]")
            final_state = run_rag(question, patient_id=patient_id)

            # 답변 출력
            answer = extract_answer_from_state(final_state)
            if answer:
                print("\n" + "=" * 70)
                print("[답변]")
                print("=" * 70)
                print(answer)
                print("=" * 70)
            else:
                error = final_state.get("error")
                if error:
                    print(f"\n⚠️  오류 발생: {error}")
                else:
                    print("\n⚠️  답변을 생성할 수 없습니다.")

        except KeyboardInterrupt:
            print("\n\n감사합니다! 👋")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            import traceback

            traceback.print_exc()


def main():
    """메인 함수"""
    from dotenv import load_dotenv
    from src.data.patient_db import PatientDatabase

    # 환경변수 로드
    load_dotenv()

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        print("   .env 파일을 확인하거나 환경변수를 설정하세요.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("🏥 대사증후군 Agentic RAG 챗봇 (Self-CRAG)")
    print("=" * 70)

    try:
        # 환자 데이터베이스 초기화
        db = PatientDatabase()

        # 환자 리스트 표시
        patients_sorted = display_patient_list(db)

        # 환자 선택
        patient_id = select_patient(patients_sorted)

        # 질문 루프
        question_loop(patient_id)

    except FileNotFoundError as e:
        print(f"\n❌ 오류: {e}")
        print(
            "   build_health_scenarios_v2.py를 먼저 실행하여 데이터베이스를 생성하세요."
        )
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
