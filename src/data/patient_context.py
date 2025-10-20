# -*- coding: utf-8 -*-
"""
환자 컨텍스트 제공 모듈 (Patient Context Provider)

사용자 요청 시 특정 환자의 정보를 조회하여 RAG 시스템에 1차 정보로 제공합니다.
쿼리 증강이나 검색 변경은 하지 않고, 순수하게 환자 정보만 반환합니다.
"""

from typing import Dict, List, Optional
from .patient_db import PatientDatabase


class PatientContextProvider:
    """환자 컨텍스트 제공자 - RAG 시스템에 환자 정보를 제공"""

    def __init__(self, db: PatientDatabase):
        self.db = db

    def get_patient_context(
        self, patient_id: int, format: str = "standard"
    ) -> Optional[str]:
        """
        특정 환자의 컨텍스트 정보 조회

        Args:
            patient_id: 환자 ID
            format: 포맷 유형 ('standard', 'detailed', 'compact')

        Returns:
            환자 컨텍스트 문자열 또는 None
        """
        if format == "standard":
            return self._format_standard(patient_id)
        elif format == "detailed":
            return self._format_detailed(patient_id)
        elif format == "compact":
            return self._format_compact(patient_id)
        else:
            return self._format_standard(patient_id)

    def _format_standard(self, patient_id: int) -> Optional[str]:
        """
        표준 포맷: 진단 결과와 위험 요인 요약

        예시:
        [환자 정보 - ID: 5]
        이름: 이도윤 (여, 32세)
        대사증후군 진단: 있음 (5/5 기준 충족)
        위험도: 고위험

        위험 요인:
        ⚠️ 복부비만 (허리둘레 98.0cm)
        ⚠️ 고혈압 (145/92mmHg)
        ...
        """
        diagnosis = self.db.check_metabolic_syndrome(patient_id)
        if not diagnosis:
            return None

        risk_eval = self.db.evaluate_risk_level(patient_id)

        lines = []
        lines.append(f"[환자 정보 - ID: {patient_id}]")
        lines.append(
            f"이름: {diagnosis['name']} ({diagnosis['sex']}, {diagnosis['age']}세)"
        )
        lines.append(
            f"대사증후군 진단: {'있음' if diagnosis['has_metabolic_syndrome'] else '없음'} ({diagnosis['criteria_met']}/5 기준 충족)"
        )
        lines.append(f"위험도: {risk_eval['risk_label']}")
        lines.append("")

        # 위험 요인
        risk_labels = {
            "abdominal_obesity": "복부비만",
            "high_blood_pressure": "고혈압",
            "high_fasting_glucose": "공복혈당장애",
            "high_triglycerides": "고중성지방",
            "low_hdl": "저HDL콜레스테롤",
        }

        has_risk = False
        for key, label in risk_labels.items():
            if diagnosis["risk_factors"][key]:
                if not has_risk:
                    lines.append("위험 요인:")
                    has_risk = True

                m = diagnosis["measurements"]
                if key == "abdominal_obesity":
                    lines.append(f"⚠️ {label} (허리둘레 {m['waist_cm']:.1f}cm)")
                elif key == "high_blood_pressure":
                    lines.append(
                        f"⚠️ {label} ({m['systolic_mmHg']}/{m['diastolic_mmHg']}mmHg)"
                    )
                elif key == "high_fasting_glucose":
                    lines.append(f"⚠️ {label} (공복혈당 {m['fbg_mg_dl']:.1f}mg/dL)")
                elif key == "high_triglycerides":
                    lines.append(f"⚠️ {label} (중성지방 {m['tg_mg_dl']:.1f}mg/dL)")
                elif key == "low_hdl":
                    lines.append(f"⚠️ {label} (HDL {m['hdl_mg_dl']:.1f}mg/dL)")

        if not has_risk:
            lines.append("위험 요인: 없음 ✅")

        return "\n".join(lines)

    def _format_detailed(self, patient_id: int) -> Optional[str]:
        """
        상세 포맷: 진단 보고서 전체 (Task 2.2의 보고서 사용)
        """
        return self.db.generate_diagnostic_report(patient_id)

    def _format_compact(self, patient_id: int) -> Optional[str]:
        """
        간결 포맷: 한 줄 요약

        예시: "환자 5번 (이도윤, 여, 32세): 대사증후군 진단, 고위험, 위험요인 5개"
        """
        diagnosis = self.db.check_metabolic_syndrome(patient_id)
        if not diagnosis:
            return None

        risk_eval = self.db.evaluate_risk_level(patient_id)

        return (
            f"환자 {patient_id}번 ({diagnosis['name']}, {diagnosis['sex']}, {diagnosis['age']}세): "
            f"대사증후군 {'진단' if diagnosis['has_metabolic_syndrome'] else '없음'}, "
            f"{risk_eval['risk_label']}, "
            f"위험요인 {diagnosis['criteria_met']}개"
        )

    def list_patients(self, format: str = "compact") -> List[str]:
        """
        모든 환자 목록 조회

        Args:
            format: 포맷 유형 ('standard', 'compact')

        Returns:
            환자 정보 리스트
        """
        all_patients = self.db.get_all_patients()
        result = []

        for patient in all_patients:
            context = self.get_patient_context(patient["patient_id"], format=format)
            if context:
                result.append(context)

        return result

    def get_metabolic_syndrome_patients(self, format: str = "compact") -> List[str]:
        """
        대사증후군 진단 환자만 조회

        Args:
            format: 포맷 유형

        Returns:
            대사증후군 환자 정보 리스트
        """
        ms_patients = self.db.get_patients_with_metabolic_syndrome()
        result = []

        for patient_diagnosis in ms_patients:
            patient_id = patient_diagnosis["patient_id"]
            context = self.get_patient_context(patient_id, format=format)
            if context:
                result.append(context)

        return result

    def format_for_llm_context(self, patient_id: Optional[int] = None) -> str:
        """
        LLM 컨텍스트로 사용할 형태로 포맷
        (RAG 답변 생성 시 시스템 프롬프트에 추가)

        Args:
            patient_id: 환자 ID (None이면 빈 문자열 반환)

        Returns:
            LLM 프롬프트용 컨텍스트
        """
        if patient_id is None:
            return ""

        context = self.get_patient_context(patient_id, format="standard")
        if not context:
            return ""

        return f"""다음은 현재 상담 중인 환자의 정보입니다. 이 정보를 참고하여 답변해주세요:

{context}

※ 위 환자 정보를 고려하되, 검색된 문서 내용을 우선으로 답변하세요."""


class PatientSession:
    """환자 세션 관리 - 선택된 환자 정보 유지"""

    def __init__(self, provider: PatientContextProvider):
        self.provider = provider
        self.current_patient_id: Optional[int] = None

    def select_patient(self, patient_id: int) -> Optional[str]:
        """
        환자 선택 및 정보 반환

        Args:
            patient_id: 선택할 환자 ID

        Returns:
            환자 정보 문자열 또는 None (없을 경우)
        """
        context = self.provider.get_patient_context(patient_id, format="standard")
        if context:
            self.current_patient_id = patient_id
            return context
        return None

    def get_current_patient_id(self) -> Optional[int]:
        """현재 선택된 환자 ID 반환"""
        return self.current_patient_id

    def get_current_context(self) -> str:
        """현재 선택된 환자의 LLM 컨텍스트 반환"""
        if self.current_patient_id is None:
            return ""
        return self.provider.format_for_llm_context(self.current_patient_id)

    def is_patient_selected(self) -> bool:
        """환자가 선택되어 있는지 확인"""
        return self.current_patient_id is not None

    def clear_selection(self):
        """환자 선택 해제"""
        self.current_patient_id = None


class SimpleCLI:
    """간단한 대화형 CLI 인터페이스 (UI 적용 전 테스트용)"""

    def __init__(self, db: PatientDatabase):
        self.db = db
        self.provider = PatientContextProvider(db)
        self.session = PatientSession(self.provider)

    def run(self):
        """CLI 실행"""
        print("\n" + "=" * 60)
        print("🏥 대사증후군 상담 시스템 (PoC)")
        print("=" * 60)

        # 자동으로 환자 리스트 표시
        self._show_patient_list()

        # 환자 선택
        if not self._select_patient_interactive():
            print("\n프로그램을 종료합니다.")
            return

        # 질문 루프
        self._question_loop()

    def _show_patient_list(self):
        """환자 목록 표시"""
        print("\n📋 환자 목록:")
        print("-" * 60)

        all_patients = self.db.get_all_patients()
        for patient in all_patients:
            diagnosis = self.db.check_metabolic_syndrome(patient["patient_id"])
            if diagnosis:
                ms_status = (
                    "🔴 진단" if diagnosis["has_metabolic_syndrome"] else "🟢 정상"
                )
                print(
                    f"{patient['patient_id']:2d}. {patient['name']:6s} "
                    f"({patient['sex']}, {patient['age']:2d}세) - {ms_status}"
                )

    def _select_patient_interactive(self) -> bool:
        """대화형 환자 선택"""
        while True:
            try:
                user_input = input("\n환자 번호를 입력하세요 (취소: q): ").strip()

                if user_input.lower() == "q":
                    return False

                patient_id = int(user_input)
                context = self.session.select_patient(patient_id)

                if context:
                    print("\n" + "=" * 60)
                    print("✅ 환자 선택 완료")
                    print("=" * 60)
                    print(f"\n{context}")
                    return True
                else:
                    print("❌ 해당 환자를 찾을 수 없습니다. 다시 입력해주세요.")

            except ValueError:
                print("❌ 올바른 번호를 입력해주세요.")
            except KeyboardInterrupt:
                print("\n\n프로그램을 종료합니다.")
                return False

    def _question_loop(self):
        """질문 입력 루프"""
        print("\n" + "=" * 60)
        print("💬 질문을 입력하세요")
        print("=" * 60)
        print("명령어: /select (환자 재선택), /info (환자 정보), /exit (종료)")
        print("-" * 60)

        while True:
            try:
                question = input("\n질문> ").strip()

                if not question:
                    continue

                # 명령어 처리
                if question == "/exit":
                    print("\n👋 상담을 종료합니다.")
                    break

                elif question == "/select":
                    print()
                    self._show_patient_list()
                    if self._select_patient_interactive():
                        continue
                    else:
                        break

                elif question == "/info":
                    if self.session.is_patient_selected():
                        context = self.provider.get_patient_context(
                            self.session.get_current_patient_id(), format="standard"
                        )
                        print(f"\n{context}")
                    else:
                        print("\n⚠️ 선택된 환자가 없습니다.")
                    continue

                # 일반 질문 처리 (RAG 시뮬레이션)
                self._handle_question(question)

            except KeyboardInterrupt:
                print("\n\n👋 상담을 종료합니다.")
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}")

    def _handle_question(self, question: str):
        """
        질문 처리 (RAG 시뮬레이션)

        실제 구현에서는 여기서 RAG 시스템을 호출합니다.
        현재는 시뮬레이션으로 동작을 보여줍니다.
        """
        print("\n[시스템 동작 시뮬레이션]")
        print(f"1️⃣ 사용자 질문: {question}")

        # 환자 컨텍스트 조회
        patient_context = self.session.get_current_context()
        if patient_context:
            print(
                f"2️⃣ 환자 컨텍스트: 환자 {self.session.get_current_patient_id()}번 정보 로드 ✓"
            )
        else:
            print("2️⃣ 환자 컨텍스트: 없음")

        print(f"3️⃣ 대사증후군 문서 검색: '{question}' (쿼리 변경 없음) ✓")
        print("4️⃣ 검색 결과 + 환자 컨텍스트를 LLM에 전달 ✓")
        print("5️⃣ LLM 답변 생성 중... ✓")

        print("\n[답변 예시]")
        print("-" * 60)
        print(
            f"환자님의 상태를 고려하여 말씀드리겠습니다.\n"
            f"(실제 RAG 시스템에서는 여기에 검색된 문서 기반 답변이 생성됩니다.)"
        )
        print("-" * 60)


def main():
    """메인 함수 - CLI 실행 또는 테스트 선택"""
    import sys

    try:
        db = PatientDatabase()

        # 인자가 있으면 테스트 모드, 없으면 CLI 모드
        if len(sys.argv) > 1 and sys.argv[1] == "--test":
            _run_tests(db)
        else:
            # 기본: 대화형 CLI 실행
            cli = SimpleCLI(db)
            cli.run()

    except FileNotFoundError as e:
        print(f"❌ 오류: {e}")
        print("먼저 build_health_scenarios_v2.py를 실행하여 데이터베이스를 생성하세요.")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        import traceback

        traceback.print_exc()


def _run_tests(db: PatientDatabase):
    """테스트 함수"""
    provider = PatientContextProvider(db)

    print("=" * 60)
    print("📝 Task 2.3: 환자 컨텍스트 제공 + 세션 관리 테스트")
    print("=" * 60)

    # 테스트 환자 찾기
    test_patient_id = None
    for patient in db.get_all_patients():
        diagnosis = db.check_metabolic_syndrome(patient["patient_id"])
        if diagnosis and diagnosis["has_metabolic_syndrome"]:
            test_patient_id = patient["patient_id"]
            break

    if test_patient_id is None:
        test_patient_id = db.get_all_patients()[0]["patient_id"]

    print(f"\n테스트 환자 ID: {test_patient_id}")

    # 1. 포맷 테스트
    print("\n" + "=" * 60)
    print("1️⃣ 간결 포맷 (Compact)")
    print("=" * 60)
    print(provider.get_patient_context(test_patient_id, format="compact"))

    print("\n" + "=" * 60)
    print("2️⃣ 표준 포맷 (Standard)")
    print("=" * 60)
    print(provider.get_patient_context(test_patient_id, format="standard"))

    # 2. 세션 관리 테스트
    print("\n" + "=" * 60)
    print("3️⃣ 세션 관리 테스트")
    print("=" * 60)

    session = PatientSession(provider)
    print(f"환자 선택 전: {session.is_patient_selected()}")

    session.select_patient(test_patient_id)
    print(f"환자 선택 후: {session.is_patient_selected()}")
    print(f"선택된 환자 ID: {session.get_current_patient_id()}")

    context = session.get_current_context()
    print(f"\nLLM 컨텍스트 (처음 5줄):")
    print("\n".join(context.split("\n")[:5]) + "\n...")

    print("\n" + "=" * 60)
    print("4️⃣ CLI 실행 안내")
    print("=" * 60)
    print("대화형 CLI를 실행하려면:")
    print("  python src/data/patient_context.py")
    print("\n테스트 모드 재실행:")
    print("  python src/data/patient_context.py --test")


if __name__ == "__main__":
    main()
