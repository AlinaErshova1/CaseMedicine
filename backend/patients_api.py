from sqlalchemy.orm import Session
from sqlalchemy import func
from models import Patient, Study
from security import encrypt_field, decrypt_field
from logger import write_log
from datetime import datetime, date
import uuid


def calculate_age(birth_date: date | None) -> int | None:
    if not birth_date:
        return None
    today = date.today()
    return today.year - birth_date.year - (
        (today.month, today.day) < (birth_date.month, birth_date.day)
    )


def create_patient(
    db: Session,
    full_name: str,
    sex: str,
    birth_date: date,
    doctor_id: str,
    phone: str = None,
    email: str = None,
    address: str = None,
    policy_oms: str = None,
    blood_type: str = None,
    height_cm: int = None,
    weight_kg: int = None,
    reason: str = None,
    complaints: str = None,
    allergies: str = None,
    attending_doctor_id: str = None,
    notes: str = None,
):
    anonymized_code = f"P-{datetime.now().year}-{str(uuid.uuid4())[:4].upper()}"

    patient = Patient(
        anonymized_code=anonymized_code,
        full_name_encrypted=encrypt_field(full_name),
        phone_encrypted=encrypt_field(phone),
        email_encrypted=encrypt_field(email),
        address_encrypted=encrypt_field(address),
        policy_oms_encrypted=encrypt_field(policy_oms),
        sex=sex,
        birth_date=birth_date,
        blood_type=blood_type,
        height_cm=height_cm,
        weight_kg=weight_kg,
        reason=reason,
        complaints=complaints,
        allergies=allergies,
        notes=notes,
        created_by=doctor_id,
        attending_doctor_id=attending_doctor_id or doctor_id,
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)

    write_log("create_patient", doctor_id, f"patient_id={patient.id}")
    return patient


def get_patient(db: Session, patient_id: str, user_id: str = None):
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if patient:
        write_log("view_patient", user_id, f"patient_id={patient_id}")
    return patient


def get_patient_full_info(db: Session, patient_id: str, user_id: str = None) -> dict | None:
    """Возвращает всё, что нужно для карточки пациента на фронте."""
    patient = get_patient(db, patient_id, user_id)
    if not patient:
        return None

    # Считаем статистику по исследованиям
    studies = (
        db.query(Study)
        .filter(Study.patient_id == patient_id)
        .order_by(Study.created_at.desc())
        .all()
    )
    total_studies = len(studies)
    first_study = studies[-1] if studies else None
    last_study = studies[0] if studies else None

    return {
        # Идентификация
        "id": str(patient.id),
        "anonymized_code": patient.anonymized_code,

        # Основная информация
        "full_name": decrypt_field(patient.full_name_encrypted),
        "birth_date": patient.birth_date.isoformat() if patient.birth_date else None,
        "age": calculate_age(patient.birth_date),
        "sex": patient.sex,
        "policy_oms": decrypt_field(patient.policy_oms_encrypted),
        "blood_type": patient.blood_type,
        "height_cm": patient.height_cm,
        "weight_kg": patient.weight_kg,

        # Контакты
        "phone": decrypt_field(patient.phone_encrypted),
        "email": decrypt_field(patient.email_encrypted),
        "address": decrypt_field(patient.address_encrypted),

        # Медицинская информация
        "attending_doctor": patient.attending_doctor.full_name if patient.attending_doctor else None,
        "reason": patient.reason,
        "complaints": patient.complaints,
        "allergies": patient.allergies,
        "created_at": patient.created_at.isoformat(),

        # Исследования (агрегаты)
        "studies_stats": {
            "total": total_studies,
            "first_date": first_study.created_at.isoformat() if first_study else None,
            "last_date": last_study.created_at.isoformat() if last_study else None,
            "last_prediction": last_study.prediction if last_study else None,
            "last_confidence": last_study.confidence if last_study else None,
            "last_status": last_study.status if last_study else None,
        },
    }


def list_patients_for_doctor(db: Session, doctor_id: str, user_id: str = None):
    """Список пациентов для таблицы на фронте."""
    patients = (
        db.query(Patient)
        .filter(
            (Patient.created_by == doctor_id) |
            (Patient.attending_doctor_id == doctor_id)
        )
        .order_by(Patient.created_at.desc())
        .all()
    )

    write_log("list_patients", user_id, f"count={len(patients)}")

    result = []
    for p in patients:
        # Последнее исследование
        last_study = (
            db.query(Study)
            .filter(Study.patient_id == p.id)
            .order_by(Study.created_at.desc())
            .first()
        )
        result.append({
            "id": str(p.id),
            "anonymized_code": p.anonymized_code,
            "full_name": decrypt_field(p.full_name_encrypted),
            "birth_date": p.birth_date.isoformat() if p.birth_date else None,
            "age": calculate_age(p.birth_date),
            "sex": p.sex,
            "last_prediction": last_study.prediction if last_study else None,
            "last_confidence": last_study.confidence if last_study else None,
        })
    return result