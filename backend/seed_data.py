from datetime import date
from database import SessionLocal, engine, Base
from models import Study
from patients_api import create_patient
from users_admin import create_doctor, create_admin
from datetime import datetime

Base.metadata.create_all(bind=engine)
db = SessionLocal()

# 1. Админ
admin = create_admin(db, full_name="Иванов Иван Иванович", login="admin", password="admin123")

# 2. Врачи
doctor_therapist = create_doctor(
    db, full_name="Иванова Анна Алексеевна", login="doctor1", password="doc123",
    specialty="Невролог", hospital="ГКБ №1", admin_id=admin.id
)
doctor_radiologist = create_doctor(
    db, full_name="Петров Пётр Петрович", login="doctor2", password="doc123",
    specialty="Рентгенолог", hospital="ГКБ №1", admin_id=admin.id
)

# 3. Пациент — как на вашем скрине
patient = create_patient(
    db,
    full_name="Ковалев Алексей Сергеевич",
    sex="M",
    birth_date=date(1983, 5, 17),
    doctor_id=doctor_therapist.id,
    attending_doctor_id=doctor_therapist.id,
    phone="+7 (913) 234-56-78",
    email="kovalev.as@mail.ru",
    address="г. Красноярск, ул. Мира, д. 12, кв. 45",
    policy_oms="7712 345 678 901",
    blood_type="II (A) Rh+",
    height_cm=182,
    weight_kg=84,
    reason="Направление на МРТ головного мозга с контрастом",
    complaints="Периодические головные боли, головокружение",
    allergies="Не выявлено",
)

# 4. Три исследования (как на скрине)
studies_data = [
    {"date": datetime(2026, 8, 21, 10, 0), "prediction": "no_tumor",   "confidence": 0.12, "status": "done"},
    {"date": datetime(2026, 9, 15, 11, 30), "prediction": "no_tumor",  "confidence": 0.08, "status": "done"},
    {"date": datetime(2026, 10, 5, 9, 15),  "prediction": "glioma",    "confidence": 0.94, "status": "done"},
]

for s in studies_data:
    study = Study(
        patient_id=patient.id,
        uploaded_by=doctor_radiologist.id,
        original_filename="mri_scan.nii.gz",
        original_path="/uploads/mri_scan.nii.gz",
        result_path="/results/mri_scan_result.png",
        prediction=s["prediction"],
        confidence=s["confidence"],
        status=s["status"],
        created_at=s["date"],
        processed_at=s["date"],
    )
    db.add(study)

db.commit()
db.close()
print("База заполнена.")