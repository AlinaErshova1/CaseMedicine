import os
import sys
import uuid
import shutil
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, Depends, HTTPException, Header, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from jose import jwt, JWTError

from database import get_db
from models import Doctor, Patient, Study, Admin
from patients_api import list_patients_for_doctor, get_patient_full_info, list_all_patients
from security import verify_password, hash_password
from logger import write_log

# === Конфигурация ===
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 8

# === Папки для загрузок ===
UPLOAD_DIR = Path(__file__).parent / "uploads"
ORIGINALS_DIR = UPLOAD_DIR / "originals"
PROCESSED_DIR = UPLOAD_DIR / "processed"
ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# === FastAPI app ===
app = FastAPI(title="Medical AI Case API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8001",
        "http://127.0.0.1:8001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Вспомогательные функции ===

def create_token(user_id: str, role: str) -> str:
    payload = {
        "sub": str(user_id),
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(authorization: str = Header(...)) -> dict:
    """Возвращает {'user_id': ..., 'role': 'doctor'|'admin'}."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Неверный формат токена")
    token = authorization.replace("Bearer ", "")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {"user_id": payload["sub"], "role": payload.get("role", "doctor")}
    except JWTError:
        raise HTTPException(status_code=401, detail="Токен недействителен")


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Требуются права администратора")
    return user


# === Базовые эндпоинты ===

@app.get("/")
def root():
    return {"status": "ok", "service": "Medical AI Case API"}


@app.post("/auth/login")
def login(login: str, password: str, db: Session = Depends(get_db)):
    # 1. Пробуем врача
    doctor = db.query(Doctor).filter(Doctor.login == login, Doctor.is_active == True).first()
    if doctor and verify_password(password, doctor.password_hash):
        token = create_token(str(doctor.id), role="doctor")
        write_log("login_success", str(doctor.id), f"role=doctor login={login}")
        return {
            "access_token": token,
            "token_type": "bearer",
            "role": "doctor",
            "user": {
                "id": str(doctor.id),
                "full_name": doctor.full_name,
                "specialty": doctor.specialty,
            },
        }

    # 2. Пробуем админа
    admin = db.query(Admin).filter(Admin.login == login, Admin.is_active == True).first()
    if admin and verify_password(password, admin.password_hash):
        token = create_token(str(admin.id), role="admin")
        write_log("login_success", str(admin.id), f"role=admin login={login}")
        return {
            "access_token": token,
            "token_type": "bearer",
            "role": "admin",
            "user": {
                "id": str(admin.id),
                "full_name": admin.full_name,
                "specialty": "Администратор",
            },
        }

    # 3. Никого не нашли
    write_log("login_failed", None, f"login={login}")
    raise HTTPException(status_code=401, detail="Неверный логин или пароль")

@app.get("/patients")
def list_patients(
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user_id = user["user_id"]
    if user["role"] == "admin":
        return list_all_patients(db, user_id)
    return list_patients_for_doctor(db, user_id, user_id)

@app.get("/patients/{patient_id}")
def get_patient(
    patient_id: str,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user_id = user["user_id"]
    info = get_patient_full_info(db, patient_id, user_id)
    if not info:
        raise HTTPException(status_code=404, detail="Пациент не найден")
    return info


# === Загрузка и анализ ===

@app.post("/upload")
async def upload_mri(
    patient_id: str = Form(...),
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user_id=user["user_id"]
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Пациент не найден")

    ext = Path(file.filename).suffix.lower()
    if ext not in [".png", ".jpg", ".jpeg"]:
        raise HTTPException(status_code=400, detail="Только PNG/JPG")

    file_uuid = str(uuid.uuid4())
    filename = f"{file_uuid}{ext}"
    filepath = ORIGINALS_DIR / filename

    with filepath.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Храним путь относительно backend/ (а не корня проекта)
    study = Study(
        patient_id=patient.id,
        uploaded_by=user_id,
        original_filename=file.filename,
        original_path=f"uploads/originals/{filename}",
        status="pending",
    )
    db.add(study)
    db.commit()
    db.refresh(study)

    write_log("upload_study", user_id, f"study_id={study.id}")

    return {
        "study_id": str(study.id),
        "filename": file.filename,
        "status": study.status,
        "created_at": study.created_at.isoformat(),
    }


@app.get("/patients/{patient_id}/studies")
def list_studies(
    patient_id: str,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user_id = user["user_id"]
    studies = (
        db.query(Study)
        .filter(Study.patient_id == patient_id)
        .order_by(Study.created_at.desc())
        .all()
    )
    write_log("list_studies", user_id, f"patient_id={patient_id}")
    return [
        {
            "id": str(s.id),
            "original_filename": s.original_filename,
            "original_url": f"/files/originals/{Path(s.original_path).name}",
            "result_url": f"/files/processed/{Path(s.result_path).name}" if s.result_path else None,
            "prediction": s.prediction,
            "confidence": s.confidence,
            "status": s.status,
            "created_at": s.created_at.isoformat(),
        }
        for s in studies
    ]


@app.get("/files/originals/{filename}")
def get_original_file(filename: str):
    filepath = ORIGINALS_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Файл не найден")
    return FileResponse(filepath)


@app.get("/files/processed/{filename}")
def get_processed_file(filename: str):
    filepath = PROCESSED_DIR / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Файл не найден")
    return FileResponse(filepath)


@app.post("/studies/{study_id}/analyze")
def analyze_study(
    study_id: str,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user_id=user["user_id"]
    study = db.query(Study).filter(Study.id == study_id).first()
    if not study:
        raise HTTPException(status_code=404, detail="Исследование не найдено")

    # === ЗДЕСЬ БУДЕТ ВЫЗОВ CV-МОДУЛЯ ===
    # Пока — заглушка: копируем оригинал в processed
    original = Path(__file__).parent / study.original_path
    if not original.exists():
        raise HTTPException(status_code=500, detail="Оригинальный файл не найден на диске")

    processed_name = f"{uuid.uuid4()}{original.suffix}"
    processed_path = PROCESSED_DIR / processed_name
    shutil.copyfile(original, processed_path)

    study.result_path = f"uploads/processed/{processed_name}"
    study.prediction = "glioma"
    study.confidence = 0.94
    study.status = "done"
    study.processed_at = datetime.utcnow()
    db.commit()

    write_log("analyze_study", user_id, f"study_id={study.id}")

    return {
        "study_id": str(study.id),
        "status": study.status,
        "prediction": study.prediction,
        "confidence": study.confidence,
        "result_url": f"/files/processed/{processed_name}",
    }

# === Админ-эндпоинты ===

from pydantic import BaseModel


class CreateDoctorRequest(BaseModel):
    full_name: str
    login: str
    password: str
    specialty: str | None = None
    hospital: str | None = None


@app.get("/admin/doctors")
def list_doctors(
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Список всех врачей (только для админа)."""
    doctors = db.query(Doctor).order_by(Doctor.created_at.desc()).all()
    write_log("list_doctors", admin["user_id"], f"count={len(doctors)}")
    return [
        {
            "id": str(d.id),
            "full_name": d.full_name,
            "login": d.login,
            "specialty": d.specialty,
            "hospital": d.hospital,
            "is_active": d.is_active,
            "created_at": d.created_at.isoformat(),
        }
        for d in doctors
    ]

@app.get("/doctors")
def list_doctors_public(
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Краткий список активных врачей для выпадающего списка (доступен всем авторизованным)."""
    doctors = (
        db.query(Doctor)
        .filter(Doctor.is_active == True)
        .order_by(Doctor.full_name)
        .all()
    )
    return [
        {
            "id": str(d.id),
            "full_name": d.full_name,
            "specialty": d.specialty,
        }
        for d in doctors
    ]

@app.post("/admin/doctors")
def create_doctor(
    payload: CreateDoctorRequest,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Создать нового врача (только админ)."""
    existing = db.query(Doctor).filter(Doctor.login == payload.login).first()
    if existing:
        raise HTTPException(status_code=400, detail="Логин уже занят")

    doctor = Doctor(
        full_name=payload.full_name,
        login=payload.login,
        password_hash=hash_password(payload.password),
        specialty=payload.specialty,
        hospital=payload.hospital,
    )
    db.add(doctor)
    db.commit()
    db.refresh(doctor)

    write_log("create_doctor", admin["user_id"], f"doctor_id={doctor.id}")

    return {
        "id": str(doctor.id),
        "full_name": doctor.full_name,
        "login": doctor.login,
        "specialty": doctor.specialty,
        "hospital": doctor.hospital,
        "is_active": doctor.is_active,
        "created_at": doctor.created_at.isoformat(),
    }


@app.patch("/admin/doctors/{doctor_id}/toggle-active")
def toggle_doctor_active(
    doctor_id: str,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Деактивировать/активировать врача (только админ)."""
    doctor = db.query(Doctor).filter(Doctor.id == doctor_id).first()
    if not doctor:
        raise HTTPException(status_code=404, detail="Врач не найден")

    doctor.is_active = not doctor.is_active
    db.commit()

    write_log(
        "toggle_doctor_active",
        admin["user_id"],
        f"doctor_id={doctor_id} is_active={doctor.is_active}",
    )
    return {"id": str(doctor.id), "is_active": doctor.is_active}


@app.get("/admin/patients")
def admin_list_patients(
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    return list_all_patients(db, admin["user_id"])
    # """Список ВСЕХ пациентов (только админ)."""
    # patients = db.query(Patient).order_by(Patient.created_at.desc()).all()
    # write_log("list_all_patients", admin["user_id"], f"count={len(patients)}")
    #
    # from patients_api import calculate_age, decrypt_field
    # result = []
    # for p in patients:
    #     last_study = (
    #         db.query(Study)
    #         .filter(Study.patient_id == p.id)
    #         .order_by(Study.created_at.desc())
    #         .first()
    #     )
    #     result.append({
    #         "id": str(p.id),
    #         "anonymized_code": p.anonymized_code,
    #         "full_name": decrypt_field(p.full_name_encrypted),
    #         "birth_date": p.birth_date.isoformat() if p.birth_date else None,
    #         "age": calculate_age(p.birth_date),
    #         "sex": p.sex,
    #         "last_prediction": last_study.prediction if last_study else None,
    #         "last_confidence": last_study.confidence if last_study else None,
    #     })
    # return result

class UpdatePatientRequest(BaseModel):
    full_name: str | None = None
    sex: str | None = None
    birth_date: str | None = None       # ISO "1983-05-17"
    phone: str | None = None
    email: str | None = None
    address: str | None = None
    policy_oms: str | None = None
    blood_type: str | None = None
    height_cm: int | None = None
    weight_kg: int | None = None
    reason: str | None = None
    complaints: str | None = None
    allergies: str | None = None


@app.patch("/admin/patients/{patient_id}")
def update_patient(
    patient_id: str,
    payload: UpdatePatientRequest,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Редактирование данных пациента (только админ)."""
    from datetime import date as date_type
    from security import encrypt_field

    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Пациент не найден")

    # Обновляем только те поля, которые пришли
    if payload.full_name is not None:
        patient.full_name_encrypted = encrypt_field(payload.full_name)
    if payload.sex is not None:
        patient.sex = payload.sex
    if payload.birth_date is not None:
        try:
            patient.birth_date = date_type.fromisoformat(payload.birth_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Неверный формат даты")
    if payload.phone is not None:
        patient.phone_encrypted = encrypt_field(payload.phone)
    if payload.email is not None:
        patient.email_encrypted = encrypt_field(payload.email)
    if payload.address is not None:
        patient.address_encrypted = encrypt_field(payload.address)
    if payload.policy_oms is not None:
        patient.policy_oms_encrypted = encrypt_field(payload.policy_oms)
    if payload.blood_type is not None:
        patient.blood_type = payload.blood_type
    if payload.height_cm is not None:
        patient.height_cm = payload.height_cm
    if payload.weight_kg is not None:
        patient.weight_kg = payload.weight_kg
    if payload.reason is not None:
        patient.reason = payload.reason
    if payload.complaints is not None:
        patient.complaints = payload.complaints
    if payload.allergies is not None:
        patient.allergies = payload.allergies

    db.commit()
    db.refresh(patient)

    write_log("update_patient", admin["user_id"], f"patient_id={patient_id}")

    return {"ok": True, "id": str(patient.id)}

@app.delete("/admin/patients/{patient_id}")
def delete_patient(
    patient_id: str,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Удаление пациента и всех его исследований (только админ)."""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Пациент не найден")

    # Удаляем файлы исследований с диска
    studies = db.query(Study).filter(Study.patient_id == patient_id).all()
    for s in studies:
        if s.original_path:
            p = Path(__file__).parent / s.original_path
            if p.exists():
                p.unlink()
        if s.result_path:
            p = Path(__file__).parent / s.result_path
            if p.exists():
                p.unlink()

    db.delete(patient)          # cascade удалит studies из БД
    db.commit()

    write_log("delete_patient", admin["user_id"], f"patient_id={patient_id}")
    return {"ok": True}

@app.delete("/admin/studies/{study_id}")
def delete_study(
    study_id: str,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Удаление исследования вместе с файлами (только админ)."""
    study = db.query(Study).filter(Study.id == study_id).first()
    if not study:
        raise HTTPException(status_code=404, detail="Исследование не найдено")

    # Удаляем файлы с диска
    if study.original_path:
        p = Path(__file__).parent / study.original_path
        if p.exists():
            p.unlink()
    if study.result_path:
        p = Path(__file__).parent / study.result_path
        if p.exists():
            p.unlink()

    db.delete(study)
    db.commit()

    write_log("delete_study", admin["user_id"], f"study_id={study_id}")
    return {"ok": True}

class UpdateDoctorRequest(BaseModel):
    full_name: str | None = None
    specialty: str | None = None
    hospital: str | None = None
    password: str | None = None       # если не пусто — меняем пароль


@app.patch("/admin/doctors/{doctor_id}")
def update_doctor(
    doctor_id: str,
    payload: UpdateDoctorRequest,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Редактирование данных врача (только админ)."""
    doctor = db.query(Doctor).filter(Doctor.id == doctor_id).first()
    if not doctor:
        raise HTTPException(status_code=404, detail="Врач не найден")

    if payload.full_name is not None:
        doctor.full_name = payload.full_name
    if payload.specialty is not None:
        doctor.specialty = payload.specialty
    if payload.hospital is not None:
        doctor.hospital = payload.hospital
    if payload.password:
        doctor.password_hash = hash_password(payload.password)

    db.commit()
    db.refresh(doctor)

    write_log("update_doctor", admin["user_id"], f"doctor_id={doctor_id}")
    return {"ok": True, "id": str(doctor.id)}

@app.delete("/admin/doctors/{doctor_id}")
def delete_doctor(
    doctor_id: str,
    admin: dict = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Удаление врача (только админ)."""
    doctor = db.query(Doctor).filter(Doctor.id == doctor_id).first()
    if not doctor:
        raise HTTPException(status_code=404, detail="Врач не найден")

    # Проверяем, есть ли у него пациенты / исследования
    patient_count = db.query(Patient).filter(Patient.created_by == doctor_id).count()
    study_count = db.query(Study).filter(Study.uploaded_by == doctor_id).count()

    if patient_count or study_count:
        raise HTTPException(
            status_code=400,
            detail=f"У врача есть {patient_count} пациентов и {study_count} исследований. Сначала передайте их другому врачу.",
        )

    db.delete(doctor)
    db.commit()

    write_log("delete_doctor", admin["user_id"], f"doctor_id={doctor_id}")
    return {"ok": True}

class CreatePatientRequest(BaseModel):
    full_name: str
    sex: str                       # "M" или "F"
    birth_date: str                # "1983-05-17"
    phone: str | None = None
    email: str | None = None
    address: str | None = None
    policy_oms: str | None = None
    blood_type: str | None = None
    height_cm: int | None = None
    weight_kg: int | None = None
    reason: str | None = None
    complaints: str | None = None
    allergies: str | None = None
    doctor_id: str | None = None       # ← для админа: какого врача привязать


@app.post("/patients")
def create_new_patient(
    payload: CreatePatientRequest,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Создание нового пациента. Доступно врачу и админу."""
    from datetime import date as date_type
    from patients_api import create_patient

    try:
        birth = date_type.fromisoformat(payload.birth_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Неверный формат даты")

    # Определяем, к какому врачу привязать пациента
    if user["role"] == "admin":
        # Админ: берём doctor_id из формы
        if not payload.doctor_id:
            raise HTTPException(
                status_code=400,
                detail="Выберите лечащего врача"
            )
        # Проверяем, что врач существует
        doctor = db.query(Doctor).filter(Doctor.id == payload.doctor_id).first()
        if not doctor:
            raise HTTPException(status_code=400, detail="Врач не найден")
        creator_id = payload.doctor_id
    else:
        # Врач: привязываем к себе
        creator_id = user["user_id"]

    patient = create_patient(
        db,
        full_name=payload.full_name,
        sex=payload.sex,
        birth_date=birth,
        doctor_id=creator_id,
        phone=payload.phone,
        email=payload.email,
        address=payload.address,
        policy_oms=payload.policy_oms,
        blood_type=payload.blood_type,
        height_cm=payload.height_cm,
        weight_kg=payload.weight_kg,
        reason=payload.reason,
        complaints=payload.complaints,
        allergies=payload.allergies,
    )

    write_log("create_patient_endpoint", user["user_id"], f"patient_id={patient.id}")

    return {
        "id": str(patient.id),
        "anonymized_code": patient.anonymized_code,
        "full_name": payload.full_name,
    }