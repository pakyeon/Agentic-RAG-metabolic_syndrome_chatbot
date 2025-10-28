# src/tools/patient.py
"""
Patient Context Tool

환자 정보를 조회하는 Tool
"""

from langchain.tools import tool


@tool
def patient_context_tool(patient_id: int) -> str:
    """
    환자의 건강 검진 정보를 조회하는 도구

    Args:
        patient_id: 환자 ID (1-20)

    Returns:
        환자의 상세 건강 정보
    """
    from src.data.patient_context import PatientContextProvider
    from src.data.patient_db import PatientDatabase

    db = PatientDatabase()
    provider = PatientContextProvider(db)

    context = provider.get_patient_context(patient_id, format="detailed")

    if not context:
        return f"환자 ID {patient_id}의 정보를 찾을 수 없습니다."

    return context
