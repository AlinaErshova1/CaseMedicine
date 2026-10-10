import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from jose import jwt, JWTError
from datetime import datetime, timedelta
import os

from database import get_db
from models import Doctor
from patients_api import list_patients_for_doctor, get_patient_full_info
from security import verify_password
from logger import write_log

SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 8

app = FastAPI(title="Medical AI Case API")

# Разрешаем фронтенду обращаться к API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # для разработки; в продакшене — конкретные домены
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Вспомогательные функции ===

def create_token(doctor_id: str) -> str:
    payload = {
        "sub": str(doctor_id),
        "exp": datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_doctor_id(authorization: str = Header(...)) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Неверный формат токена")
    token = authorization.replace("Bearer ", "")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload["sub"]
    except JWTError:
        raise HTTPException(status_code=401, detail="Токен недействителен")


# === Эндпоинты ===

@app.get("/")
def root():
    return {"status": "ok", "service": "Medical AI Case API"}


@app.post("/auth/login")
def login(login: str, password: str, db: Session = Depends(get_db)):
    """Логин врача. Возвращает JWT-токен."""
    doctor = db.query(Doctor).filter(Doctor.login == login).first()
    if not doctor or not verify_password(password, doctor.password_hash):
        write_log("login_failed", None, f"login={login}")
        raise HTTPException(status_code=401, detail="Неверный логин или пароль")

    token = create_token(str(doctor.id))
    write_log("login_success", str(doctor.id), f"login={login}")

    return {
        "access_token": token,
        "token_type": "bearer",
        "doctor": {
            "id": str(doctor.id),
            "full_name": doctor.full_name,
            "specialty": doctor.specialty,
        },
    }


@app.get("/patients")
def list_patients(
    user_id: str = Depends(get_current_doctor_id),
    db: Session = Depends(get_db),
):
    """Список пациентов текущего врача."""
    return list_patients_for_doctor(db, user_id, user_id)


@app.get("/patients/{patient_id}")
def get_patient(
    patient_id: str,
    user_id: str = Depends(get_current_doctor_id),
    db: Session = Depends(get_db),
):
    """Полная карточка пациента."""
    info = get_patient_full_info(db, patient_id, user_id)
    if not info:
        raise HTTPException(status_code=404, detail="Пациент не найден")
    return info