# -*- coding: utf-8 -*-
"""
환자 데이터베이스 조회 및 대사증후군 진단 평가 모듈

대사증후군 진단 기준 (한국인):
- 복부비만: 허리둘레 남성 ≥90cm, 여성 ≥85cm
- 고혈압: 수축기 ≥130mmHg 또는 이완기 ≥85mmHg
- 공복혈당장애: 공복혈당 ≥100mg/dL
- 고중성지방: 중성지방 ≥150mg/dL
- 저HDL콜레스테롤: HDL 남성 <40mg/dL, 여성 <50mg/dL
※ 5개 항목 중 3개 이상 해당 시 대사증후군 진단
"""

import sqlite3
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime


class PatientDatabase:
    """환자 데이터베이스 조회 및 진단 평가 클래스"""

    # 대사증후군 진단 기준 (한국인)
    CRITERIA = {
        "waist": {"male": 90, "female": 85},  # cm
        "blood_pressure": {"systolic": 130, "diastolic": 85},  # mmHg
        "fasting_glucose": 100,  # mg/dL
        "triglycerides": 150,  # mg/dL
        "hdl": {"male": 40, "female": 50},  # mg/dL
    }

    def __init__(self, db_path: str = "health_scenarios_v2.sqlite"):
        """
        Args:
            db_path: SQLite 데이터베이스 파일 경로
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"데이터베이스 파일을 찾을 수 없습니다: {db_path}\n"
                "build_health_scenarios_v2.py를 먼저 실행하세요."
            )

    def _get_connection(self) -> sqlite3.Connection:
        """DB 연결 생성"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 컬럼명으로 접근 가능
        return conn

    def get_patient(self, patient_id: int) -> Optional[Dict]:
        """
        환자 기본 정보 조회

        Args:
            patient_id: 환자 ID

        Returns:
            환자 정보 딕셔너리 또는 None
        """
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT patient_id, name, sex, age, rrn_masked, registered_at
                FROM patients
                WHERE patient_id = ?
                """,
                (patient_id,),
            )
            row = cur.fetchone()

            if row:
                return dict(row)
            return None

    def get_all_patients(self) -> List[Dict]:
        """
        모든 환자 목록 조회

        Returns:
            환자 정보 리스트
        """
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT patient_id, name, sex, age, rrn_masked, registered_at
                FROM patients
                ORDER BY patient_id
                """
            )
            return [dict(row) for row in cur.fetchall()]

    def get_latest_exam(self, patient_id: int) -> Optional[Dict]:
        """
        환자의 최신 검진 데이터 조회

        Args:
            patient_id: 환자 ID

        Returns:
            검진 정보 딕셔너리 또는 None
        """
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT *
                FROM health_exams
                WHERE patient_id = ?
                ORDER BY exam_at DESC
                LIMIT 1
                """,
                (patient_id,),
            )
            row = cur.fetchone()

            if row:
                return dict(row)
            return None

    def get_exam_history(self, patient_id: int) -> List[Dict]:
        """
        환자의 모든 검진 이력 조회

        Args:
            patient_id: 환자 ID

        Returns:
            검진 이력 리스트 (최신순)
        """
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT *
                FROM health_exams
                WHERE patient_id = ?
                ORDER BY exam_at DESC
                """,
                (patient_id,),
            )
            return [dict(row) for row in cur.fetchall()]

    def check_metabolic_syndrome(self, patient_id: int) -> Optional[Dict]:
        """
        환자의 대사증후군 진단 기준 평가

        Args:
            patient_id: 환자 ID

        Returns:
            진단 평가 결과 딕셔너리:
            {
                'patient_id': int,
                'name': str,
                'sex': str,
                'age': int,
                'exam_at': str,
                'criteria_met': int (충족한 항목 수),
                'has_metabolic_syndrome': bool (3개 이상 시 True),
                'risk_factors': {
                    'abdominal_obesity': bool,
                    'high_blood_pressure': bool,
                    'high_fasting_glucose': bool,
                    'high_triglycerides': bool,
                    'low_hdl': bool
                },
                'measurements': Dict (측정값)
            }
        """
        # 환자 정보 조회
        patient = self.get_patient(patient_id)
        if not patient:
            return None

        # 최신 검진 데이터 조회
        exam = self.get_latest_exam(patient_id)
        if not exam:
            return None

        sex = patient["sex"]

        # 각 기준 평가
        risk_factors = {
            "abdominal_obesity": self._check_abdominal_obesity(exam["waist_cm"], sex),
            "high_blood_pressure": self._check_blood_pressure(
                exam["systolic_mmHg"], exam["diastolic_mmHg"]
            ),
            "high_fasting_glucose": self._check_fasting_glucose(exam["fbg_mg_dl"]),
            "high_triglycerides": self._check_triglycerides(exam["tg_mg_dl"]),
            "low_hdl": self._check_hdl(exam["hdl_mg_dl"], sex),
        }

        criteria_met = sum(risk_factors.values())
        has_metabolic_syndrome = criteria_met >= 3

        return {
            "patient_id": patient_id,
            "name": patient["name"],
            "sex": sex,
            "age": patient["age"],
            "exam_at": exam["exam_at"],
            "criteria_met": criteria_met,
            "has_metabolic_syndrome": has_metabolic_syndrome,
            "risk_factors": risk_factors,
            "measurements": {
                "waist_cm": exam["waist_cm"],
                "systolic_mmHg": exam["systolic_mmHg"],
                "diastolic_mmHg": exam["diastolic_mmHg"],
                "fbg_mg_dl": exam["fbg_mg_dl"],
                "tg_mg_dl": exam["tg_mg_dl"],
                "hdl_mg_dl": exam["hdl_mg_dl"],
                "bmi": exam["bmi"],
            },
        }

    def _check_abdominal_obesity(self, waist_cm: float, sex: str) -> bool:
        """복부비만 평가"""
        if waist_cm is None:
            return False
        threshold = (
            self.CRITERIA["waist"]["male"]
            if sex == "남"
            else self.CRITERIA["waist"]["female"]
        )
        return waist_cm >= threshold

    def _check_blood_pressure(self, systolic_mmHg: int, diastolic_mmHg: int) -> bool:
        """고혈압 평가"""
        if systolic_mmHg is None or diastolic_mmHg is None:
            return False
        return (
            systolic_mmHg >= self.CRITERIA["blood_pressure"]["systolic"]
            or diastolic_mmHg >= self.CRITERIA["blood_pressure"]["diastolic"]
        )

    def _check_fasting_glucose(self, fbg_mg_dl: float) -> bool:
        """공복혈당장애 평가"""
        if fbg_mg_dl is None:
            return False
        return fbg_mg_dl >= self.CRITERIA["fasting_glucose"]

    def _check_triglycerides(self, tg_mg_dl: float) -> bool:
        """고중성지방 평가"""
        if tg_mg_dl is None:
            return False
        return tg_mg_dl >= self.CRITERIA["triglycerides"]

    def _check_hdl(self, hdl_mg_dl: float, sex: str) -> bool:
        """저HDL콜레스테롤 평가"""
        if hdl_mg_dl is None:
            return False
        threshold = (
            self.CRITERIA["hdl"]["male"]
            if sex == "남"
            else self.CRITERIA["hdl"]["female"]
        )
        return hdl_mg_dl < threshold

    def get_patients_with_metabolic_syndrome(self) -> List[Dict]:
        """
        대사증후군으로 진단된 모든 환자 목록 조회

        Returns:
            대사증후군 환자 리스트
        """
        all_patients = self.get_all_patients()
        ms_patients = []

        for patient in all_patients:
            diagnosis = self.check_metabolic_syndrome(patient["patient_id"])
            if diagnosis and diagnosis["has_metabolic_syndrome"]:
                ms_patients.append(diagnosis)

        return ms_patients

    def get_statistics(self) -> Dict:
        """
        전체 환자 통계 정보

        Returns:
            통계 딕셔너리
        """
        all_patients = self.get_all_patients()
        ms_patients = self.get_patients_with_metabolic_syndrome()

        # 성별 통계
        male_count = sum(1 for p in all_patients if p["sex"] == "남")
        female_count = sum(1 for p in all_patients if p["sex"] == "여")

        # 연령대 통계
        age_groups = {}
        for p in all_patients:
            age_group = f"{p['age']//10*10}대"
            age_groups[age_group] = age_groups.get(age_group, 0) + 1

        return {
            "total_patients": len(all_patients),
            "male_patients": male_count,
            "female_patients": female_count,
            "metabolic_syndrome_patients": len(ms_patients),
            "metabolic_syndrome_rate": (
                len(ms_patients) / len(all_patients) * 100 if all_patients else 0
            ),
            "age_distribution": age_groups,
        }

    def evaluate_risk_level(self, patient_id: int) -> Optional[Dict]:
        """
        환자의 위험도 레벨 평가 (저위험/중위험/고위험)

        Args:
            patient_id: 환자 ID

        Returns:
            위험도 평가 결과:
            {
                'risk_level': str ('low', 'medium', 'high'),
                'risk_score': int (0-5, 충족한 기준 수),
                'risk_label': str ('저위험', '중위험', '고위험'),
                'risk_description': str (위험도 설명)
            }
        """
        diagnosis = self.check_metabolic_syndrome(patient_id)
        if not diagnosis:
            return None

        score = diagnosis["criteria_met"]

        if score == 0:
            level, label = "low", "저위험"
            desc = "현재 대사증후군 위험 요인이 없습니다. 건강한 생활습관을 유지하세요."
        elif score == 1:
            level, label = "low", "저위험"
            desc = "1개의 위험 요인이 있습니다. 예방적 관리가 필요합니다."
        elif score == 2:
            level, label = "medium", "중위험"
            desc = "2개의 위험 요인이 있어 대사증후군 전단계입니다. 적극적인 생활습관 개선이 필요합니다."
        elif score == 3:
            level, label = "high", "고위험"
            desc = "대사증후군으로 진단됩니다. 의료진 상담 및 치료적 개입이 필요합니다."
        else:  # score >= 4
            level, label = "high", "고위험"
            desc = f"{score}개의 위험 요인이 있어 심혈관질환 위험이 매우 높습니다. 즉각적인 의학적 관리가 필요합니다."

        return {
            "risk_level": level,
            "risk_score": score,
            "risk_label": label,
            "risk_description": desc,
        }

    def interpret_risk_factors(self, patient_id: int) -> Optional[Dict]:
        """
        각 위험 요인에 대한 상세 해석 제공

        Args:
            patient_id: 환자 ID

        Returns:
            위험 요인별 상세 해석
        """
        diagnosis = self.check_metabolic_syndrome(patient_id)
        if not diagnosis:
            return None

        patient = self.get_patient(patient_id)
        sex = patient["sex"]
        measurements = diagnosis["measurements"]
        risk_factors = diagnosis["risk_factors"]

        interpretations = {}

        # 1. 복부비만
        waist = measurements["waist_cm"]
        threshold = (
            self.CRITERIA["waist"]["male"]
            if sex == "남"
            else self.CRITERIA["waist"]["female"]
        )
        if risk_factors["abdominal_obesity"]:
            interpretations["abdominal_obesity"] = {
                "status": "위험",
                "value": f"{waist:.1f}cm",
                "threshold": f"{threshold}cm",
                "interpretation": f"허리둘레가 {waist:.1f}cm로 기준({threshold}cm)을 초과하여 복부비만입니다.",
                "recommendation": "내장지방 감소를 위해 유산소 운동과 식이조절이 필요합니다.",
            }
        else:
            interpretations["abdominal_obesity"] = {
                "status": "정상",
                "value": f"{waist:.1f}cm",
                "threshold": f"{threshold}cm",
                "interpretation": f"허리둘레가 {waist:.1f}cm로 정상 범위입니다.",
                "recommendation": "현재 상태를 유지하세요.",
            }

        # 2. 혈압
        sys = measurements["systolic_mmHg"]
        dia = measurements["diastolic_mmHg"]
        if risk_factors["high_blood_pressure"]:
            interpretations["high_blood_pressure"] = {
                "status": "위험",
                "value": f"{sys}/{dia}mmHg",
                "threshold": "130/85mmHg",
                "interpretation": f"혈압이 {sys}/{dia}mmHg로 고혈압 기준을 충족합니다.",
                "recommendation": "저염식이, 규칙적인 운동, 필요시 약물 치료가 필요합니다.",
            }
        else:
            interpretations["high_blood_pressure"] = {
                "status": "정상",
                "value": f"{sys}/{dia}mmHg",
                "threshold": "130/85mmHg",
                "interpretation": f"혈압이 {sys}/{dia}mmHg로 정상 범위입니다.",
                "recommendation": "저염식이와 규칙적인 운동으로 혈압을 유지하세요.",
            }

        # 3. 공복혈당
        fbg = measurements["fbg_mg_dl"]
        if risk_factors["high_fasting_glucose"]:
            if fbg >= 126:
                interpretations["high_fasting_glucose"] = {
                    "status": "위험",
                    "value": f"{fbg:.1f}mg/dL",
                    "threshold": "100mg/dL",
                    "interpretation": f"공복혈당이 {fbg:.1f}mg/dL로 당뇨병 진단 기준(126mg/dL)에 해당합니다.",
                    "recommendation": "즉시 의료진 상담이 필요하며 혈당 관리를 시작해야 합니다.",
                }
            else:
                interpretations["high_fasting_glucose"] = {
                    "status": "위험",
                    "value": f"{fbg:.1f}mg/dL",
                    "threshold": "100mg/dL",
                    "interpretation": f"공복혈당이 {fbg:.1f}mg/dL로 공복혈당장애(전단계) 상태입니다.",
                    "recommendation": "당질 섭취 조절과 체중 감량으로 당뇨병 진행을 예방할 수 있습니다.",
                }
        else:
            interpretations["high_fasting_glucose"] = {
                "status": "정상",
                "value": f"{fbg:.1f}mg/dL",
                "threshold": "100mg/dL",
                "interpretation": f"공복혈당이 {fbg:.1f}mg/dL로 정상 범위입니다.",
                "recommendation": "균형잡힌 식단으로 혈당을 유지하세요.",
            }

        # 4. 중성지방
        tg = measurements["tg_mg_dl"]
        if risk_factors["high_triglycerides"]:
            if tg >= 200:
                interpretations["high_triglycerides"] = {
                    "status": "위험",
                    "value": f"{tg:.1f}mg/dL",
                    "threshold": "150mg/dL",
                    "interpretation": f"중성지방이 {tg:.1f}mg/dL로 매우 높습니다 (200mg/dL 이상).",
                    "recommendation": "의료진 상담 및 약물 치료가 필요할 수 있습니다. 술, 단순당 섭취를 제한하세요.",
                }
            else:
                interpretations["high_triglycerides"] = {
                    "status": "위험",
                    "value": f"{tg:.1f}mg/dL",
                    "threshold": "150mg/dL",
                    "interpretation": f"중성지방이 {tg:.1f}mg/dL로 높습니다.",
                    "recommendation": "알코올 섭취를 줄이고 단순당 섭취를 제한하세요.",
                }
        else:
            interpretations["high_triglycerides"] = {
                "status": "정상",
                "value": f"{tg:.1f}mg/dL",
                "threshold": "150mg/dL",
                "interpretation": f"중성지방이 {tg:.1f}mg/dL로 정상 범위입니다.",
                "recommendation": "현재 수준을 유지하세요.",
            }

        # 5. HDL 콜레스테롤
        hdl = measurements["hdl_mg_dl"]
        hdl_threshold = (
            self.CRITERIA["hdl"]["male"]
            if sex == "남"
            else self.CRITERIA["hdl"]["female"]
        )
        if risk_factors["low_hdl"]:
            interpretations["low_hdl"] = {
                "status": "위험",
                "value": f"{hdl:.1f}mg/dL",
                "threshold": f"{hdl_threshold}mg/dL",
                "interpretation": f"HDL 콜레스테롤이 {hdl:.1f}mg/dL로 낮습니다 (기준: {hdl_threshold}mg/dL 이상).",
                "recommendation": "유산소 운동과 오메가-3 섭취로 HDL을 높이세요.",
            }
        else:
            interpretations["low_hdl"] = {
                "status": "정상",
                "value": f"{hdl:.1f}mg/dL",
                "threshold": f"{hdl_threshold}mg/dL",
                "interpretation": f"HDL 콜레스테롤이 {hdl:.1f}mg/dL로 정상 범위입니다.",
                "recommendation": "좋은 콜레스테롤 수준을 유지하세요.",
            }

        return interpretations

    def generate_diagnostic_report(self, patient_id: int) -> Optional[str]:
        """
        RAG 시스템에 활용 가능한 자연어 진단 보고서 생성

        Args:
            patient_id: 환자 ID

        Returns:
            자연어 형태의 진단 보고서
        """
        diagnosis = self.check_metabolic_syndrome(patient_id)
        if not diagnosis:
            return None

        risk_eval = self.evaluate_risk_level(patient_id)
        interpretations = self.interpret_risk_factors(patient_id)

        # 보고서 생성
        lines = []
        lines.append(f"환자명: {diagnosis['name']}")
        lines.append(f"성별/나이: {diagnosis['sex']}, {diagnosis['age']}세")
        lines.append(f"검진일: {diagnosis['exam_at']}")
        lines.append("")
        lines.append("=" * 60)
        lines.append("대사증후군 진단 평가 결과")
        lines.append("=" * 60)
        lines.append("")

        # 종합 진단
        if diagnosis["has_metabolic_syndrome"]:
            lines.append(
                f"⚠️ 대사증후군: 진단 ({diagnosis['criteria_met']}/5 기준 충족)"
            )
        else:
            lines.append(
                f"✅ 대사증후군: 해당 없음 ({diagnosis['criteria_met']}/5 기준)"
            )

        lines.append(f"위험도: {risk_eval['risk_label']} ({risk_eval['risk_score']}점)")
        lines.append(f"평가: {risk_eval['risk_description']}")
        lines.append("")

        # 각 위험 요인 상세
        lines.append("세부 평가:")
        lines.append("-" * 60)

        risk_labels = {
            "abdominal_obesity": "1. 복부비만",
            "high_blood_pressure": "2. 고혈압",
            "high_fasting_glucose": "3. 공복혈당장애",
            "high_triglycerides": "4. 고중성지방혈증",
            "low_hdl": "5. 저HDL콜레스테롤혈증",
        }

        for key, label in risk_labels.items():
            interp = interpretations[key]
            status_icon = "⚠️" if interp["status"] == "위험" else "✅"
            lines.append(f"\n{label}")
            lines.append(f"  {status_icon} 상태: {interp['status']}")
            lines.append(f"  측정값: {interp['value']} (기준: {interp['threshold']})")
            lines.append(f"  해석: {interp['interpretation']}")
            lines.append(f"  권장사항: {interp['recommendation']}")

        # 종합 권장사항
        lines.append("")
        lines.append("-" * 60)
        lines.append("종합 권장사항:")
        lines.append("-" * 60)

        if diagnosis["has_metabolic_syndrome"]:
            lines.append(
                "• 대사증후군으로 진단되어 의료진과의 정기적인 상담이 필요합니다."
            )
            lines.append(
                "• 생활습관 개선: 규칙적인 운동(주 5회, 30분 이상), 균형잡힌 식단"
            )
            lines.append("• 체중 감량: 현재 체중의 5-10% 감량 목표")
            lines.append("• 금연 및 절주")
            lines.append("• 스트레스 관리 및 충분한 수면")
        else:
            if diagnosis["criteria_met"] >= 2:
                lines.append("• 대사증후군 전단계로 예방적 관리가 중요합니다.")
                lines.append("• 위험 요인 개선을 위한 생활습관 교정이 필요합니다.")
            elif diagnosis["criteria_met"] == 1:
                lines.append("• 1개의 위험 요인이 있어 예방적 관리가 필요합니다.")
                lines.append(
                    "• 현재 상태가 악화되지 않도록 건강한 생활습관을 유지하세요."
                )
            else:
                lines.append(
                    "• 현재 대사증후군 위험이 없으나 정기적인 검진을 권장합니다."
                )
                lines.append("• 건강한 생활습관을 꾸준히 유지하세요.")

        return "\n".join(lines)


def main():
    """테스트 함수"""
    try:
        db = PatientDatabase()

        print("=" * 60)
        print("📊 환자 데이터베이스 테스트 (Task 2.1 + 2.2)")
        print("=" * 60)

        # 통계 정보
        stats = db.get_statistics()
        print(f"\n📈 전체 통계:")
        print(f"  - 총 환자 수: {stats['total_patients']}명")
        print(
            f"  - 남성: {stats['male_patients']}명, 여성: {stats['female_patients']}명"
        )
        print(
            f"  - 대사증후군 환자: {stats['metabolic_syndrome_patients']}명 ({stats['metabolic_syndrome_rate']:.1f}%)"
        )
        print(f"\n  연령대별 분포:")
        for age_group, count in sorted(stats["age_distribution"].items()):
            print(f"    {age_group}: {count}명")

        # Task 2.2 테스트: 상세 진단 보고서
        print(f"\n" + "=" * 60)
        print("🔍 Task 2.2: 상세 진단 보고서 (샘플 2명)")
        print("=" * 60)

        # 샘플 1: 대사증후군 환자 (고위험)
        test_patients = []
        for patient in db.get_all_patients():
            diagnosis = db.check_metabolic_syndrome(patient["patient_id"])
            if diagnosis and diagnosis["has_metabolic_syndrome"]:
                test_patients.append(patient["patient_id"])
                if len(test_patients) >= 1:
                    break

        # 샘플 2: 정상 환자 (저위험)
        for patient in db.get_all_patients():
            diagnosis = db.check_metabolic_syndrome(patient["patient_id"])
            if diagnosis and not diagnosis["has_metabolic_syndrome"]:
                test_patients.append(patient["patient_id"])
                if len(test_patients) >= 2:
                    break

        for patient_id in test_patients:
            print("\n" + "=" * 60)
            report = db.generate_diagnostic_report(patient_id)
            if report:
                print(report)
            print("\n")

        # Task 2.2 기능 요약 테스트
        print("=" * 60)
        print("🧪 Task 2.2 기능 테스트")
        print("=" * 60)

        test_id = test_patients[0] if test_patients else 1

        # 1. 위험도 평가
        print(f"\n1️⃣ 위험도 평가 (환자 #{test_id}):")
        risk_eval = db.evaluate_risk_level(test_id)
        if risk_eval:
            print(
                f"  - 위험 레벨: {risk_eval['risk_label']} ({risk_eval['risk_level']})"
            )
            print(f"  - 위험 점수: {risk_eval['risk_score']}/5")
            print(f"  - 설명: {risk_eval['risk_description']}")

        # 2. 위험 요인 해석
        print(f"\n2️⃣ 위험 요인 상세 해석:")
        interpretations = db.interpret_risk_factors(test_id)
        if interpretations:
            for key, interp in interpretations.items():
                if interp["status"] == "위험":
                    print(f"\n  ⚠️ {key}:")
                    print(f"    측정값: {interp['value']}")
                    print(f"    해석: {interp['interpretation']}")

    except FileNotFoundError as e:
        print(f"❌ 오류: {e}")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
