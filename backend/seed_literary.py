"""
Заполнение БД литературными персонажами.
Врачи и пациенты — узнаваемые, но данные вымышленные.
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from datetime import date, datetime
from database import SessionLocal, engine, Base
from models import Doctor, Patient, Study
from patients_api import create_patient
from users_admin import create_doctor
from security import hash_password
from logger import write_log

Base.metadata.create_all(bind=engine)
db = SessionLocal()


# ============================================================
# ВРАЧИ
# ============================================================
DOCTORS = [
    {
        "full_name": "Борменталь Иван Арнольдович",
        "login": "bormental",
        "password": "doctor123",
        "specialty": "Невролог",
        "hospital": "Клиника Преображенского",
    },
    {
        "full_name": "Вонзель Фёдор Фёдорович",
        "login": "vonzel",
        "password": "doctor123",
        "specialty": "Нейрохирург",
        "hospital": "ГКБ №1",
    },
    {
        "full_name": "Дорн Аркадий Иванович",
        "login": "dorn",
        "password": "doctor123",
        "specialty": "Терапевт",
        "hospital": "Городская больница",
    },
    {
        "full_name": "Чехов Антон Павлович",
        "login": "chekhov",
        "password": "doctor123",
        "specialty": "Врач общей практики",
        "hospital": "Земская больница",
    },
]


# ============================================================
# ПАЦИЕНТЫ
# ============================================================
PATIENTS = [
    {
        "full_name": "Болконская Наталья Ростовна",
        "sex": "F",
        "birth_date": date(1998, 8, 23),
        "phone": "+7 (916) 222-33-44",
        "email": "natasha.bolkonskaya@mail.ru",
        "address": "г. Москва, ул. Пречистенка, д. 8, кв. 12",
        "policy_oms": "7712 345 678 902",
        "blood_type": "I (0) Rh+",
        "height_cm": 165,
        "weight_kg": 55,
        "reason": "Головокружения после перенесённой простуды",
        "complaints": "Слабость, утомляемость",
        "allergies": "Пыльца берёзы",
    },
    {
        "full_name": "Печорин Григорий Александрович",
        "sex": "M",
        "birth_date": date(1992, 3, 15),
        "phone": "+7 (903) 333-44-55",
        "email": "pechorin.g@mail.ru",
        "address": "г. Пятигорск, ул. Лермонтова, д. 5, кв. 7",
        "policy_oms": "2612 345 678 903",
        "blood_type": "III (B) Rh+",
        "height_cm": 185,
        "weight_kg": 78,
        "reason": "Мигрени с аурой, участившиеся приступы",
        "complaints": "Головные боли в височной области, тошнота",
        "allergies": "Не выявлено",
    },
    {
        "full_name": "Онегин Евгений Александрович",
        "sex": "M",
        "birth_date": date(1990, 1, 9),
        "phone": "+7 (911) 444-55-66",
        "email": "onegin@mail.ru",
        "address": "г. Санкт-Петербург, наб. реки Мойки, д. 12, кв. 3",
        "policy_oms": "7811 234 567 904",
        "blood_type": "IV (AB) Rh-",
        "height_cm": 178,
        "weight_kg": 74,
        "reason": "Плановое обследование, хандра",
        "complaints": "Апатия, снижение интереса",
        "allergies": "Не выявлено",
    },
    {
        "full_name": "Ларина Татьяна Дмитриевна",
        "sex": "F",
        "birth_date": date(1997, 1, 25),
        "phone": "+7 (916) 555-66-77",
        "email": "tatyana.larina@mail.ru",
        "address": "г. Москва, ул. Онегина, д. 3, кв. 1",
        "policy_oms": "7712 345 678 905",
        "blood_type": "II (A) Rh+",
        "height_cm": 168,
        "weight_kg": 58,
        "reason": "Обращение по поводу частых головных болей",
        "complaints": "Головные боли при стрессе",
        "allergies": "Цитрусовые",
    },
    {
        "full_name": "Гринёв Пётр Андреевич",
        "sex": "M",
        "birth_date": date(2001, 5, 14),
        "phone": "+7 (926) 666-77-88",
        "email": "grinev.petr@mail.ru",
        "address": "г. Оренбург, ул. Белогорская, д. 1, кв. 8",
        "policy_oms": "5612 345 678 906",
        "blood_type": "I (0) Rh+",
        "height_cm": 176,
        "weight_kg": 70,
        "reason": "Направление после спортивной травмы",
        "complaints": "Периодические головокружения",
        "allergies": "Не выявлено",
    },
    {
        "full_name": "Мария Миронова (капитанская дочка)",
        "sex": "F",
        "birth_date": date(2002, 11, 2),
        "phone": "+7 (926) 777-88-99",
        "email": "maria.mironova@mail.ru",
        "address": "г. Оренбург, ул. Крепостная, д. 7, кв. 2",
        "policy_oms": "5612 345 678 907",
        "blood_type": "III (B) Rh+",
        "height_cm": 162,
        "weight_kg": 52,
        "reason": "Профилактический осмотр",
        "complaints": "Жалоб нет",
        "allergies": "Не выявлено",
    },
    {
        "full_name": "Чичиков Павел Иванович",
        "sex": "M",
        "birth_date": date(1985, 4, 20),
        "phone": "+7 (985) 888-99-00",
        "email": "chichikov.pavel@mail.ru",
        "address": "г. Москва, ул. Маниловская, д. 4, кв. 15",
        "policy_oms": "7712 345 678 908",
        "blood_type": "II (A) Rh-",
        "height_cm": 170,
        "weight_kg": 82,
        "reason": "Обращение по поводу головных болей напряжения",
        "complaints": "Стресс, усталость",
        "allergies": "Не выявлено",
    },
    {
        "full_name": "Обломов Илья Ильич",
        "sex": "M",
        "birth_date": date(1980, 6, 30),
        "phone": "+7 (985) 999-00-11",
        "email": "oblomov.ii@mail.ru",
        "address": "г. Санкт-Петербург, ул. Гороховая, д. 12, кв. 4",
        "policy_oms": "7811 234 567 909",
        "blood_type": "I (0) Rh+",
        "height_cm": 175,
        "weight_kg": 95,
        "reason": "Обследование по поводу апатии",
        "complaints": "Постоянная усталость, сонливость",
        "allergies": "Не выявлено",
    },
    {
        "full_name": "Штольц Андрей Иванович",
        "sex": "M",
        "birth_date": date(1978, 9, 12),
        "phone": "+7 (985) 000-11-22",
        "email": "shtolz@mail.ru",
        "address": "г. Санкт-Петербург, Невский пр., д. 100, кв. 10",
        "policy_oms": "7811 234 567 910",
        "blood_type": "III (B) Rh+",
        "height_cm": 180,
        "weight_kg": 76,
        "reason": "Плановое обследование",
        "complaints": "Жалоб нет",
        "allergies": "Не выявлено",
    },
    {
        "full_name": "Каренина Анна Аркадьевна",
        "sex": "F",
        "birth_date": date(1988, 12, 5),
        "phone": "+7 (903) 111-22-33",
        "email": "anna.karenina@mail.ru",
        "address": "г. Москва, ул. Вознесенская, д. 5, кв. 3",
        "policy_oms": "7712 345 678 911",
        "blood_type": "II (A) Rh+",
        "height_cm": 170,
        "weight_kg": 60,
        "reason": "Периодические головные боли на фоне стресса",
        "complaints": "Бессонница, тревожность",
        "allergies": "Лаванда",
    },
    {
        "full_name": "Левин Константин Дмитриевич",
        "sex": "M",
        "birth_date": date(1985, 3, 18),
        "phone": "+7 (903) 222-33-44",
        "email": "levin.kd@mail.ru",
        "address": "г. Москва, ул. Покровская, д. 8, кв. 2",
        "policy_oms": "7712 345 678 912",
        "blood_type": "IV (AB) Rh+",
        "height_cm": 182,
        "weight_kg": 80,
        "reason": "Обращение по поводу головных болей",
        "complaints": "Периодические мигрени",
        "allergies": "Мёд",
    },
    {
        "full_name": "Вронский Алексей Кириллович",
        "sex": "M",
        "birth_date": date(1986, 8, 25),
        "phone": "+7 (903) 333-44-55",
        "email": "vronsky.ak@mail.ru",
        "address": "г. Москва, ул. Тверская, д. 15, кв. 7",
        "policy_oms": "7712 345 678 913",
        "blood_type": "I (0) Rh+",
        "height_cm": 187,
        "weight_kg": 85,
        "reason": "Спортивная травма головы",
        "complaints": "Головокружения после падения",
        "allergies": "Не выявлено",
    },
    {
        "full_name": "Облонский Степан Аркадьевич",
        "sex": "M",
        "birth_date": date(1983, 10, 8),
        "phone": "+7 (903) 444-55-66",
        "email": "stiva@mail.ru",
        "address": "г. Москва, ул. Пречистенка, д. 21, кв. 4",
        "policy_oms": "7712 345 678 914",
        "blood_type": "II (A) Rh+",
        "height_cm": 179,
        "weight_kg": 88,
        "reason": "Головные боли после застолий",
        "complaints": "Утренние головные боли",
        "allergies": "Не выявлено",
    },
    {
        "full_name": "Дарья Александровна Облонская",
        "sex": "F",
        "birth_date": date(1985, 7, 14),
        "phone": "+7 (903) 555-66-77",
        "email": "dolly@mail.ru",
        "address": "г. Москва, ул. Пречистенка, д. 21, кв. 4",
        "policy_oms": "7712 345 678 915",
        "blood_type": "III (B) Rh+",
        "height_cm": 165,
        "weight_kg": 62,
        "reason": "Частые головные боли",
        "complaints": "Мигрени, усталость",
        "allergies": "Шоколад",
    },
]


# ============================================================
# ЗАПОЛНЕНИЕ
# ============================================================
print("=" * 60)
print("Создание врачей...")
print("=" * 60)

created_doctors = []
for d in DOCTORS:
    # Проверяем, нет ли уже такого логина
    existing = db.query(Doctor).filter(Doctor.login == d["login"]).first()
    if existing:
        print(f"  ⏭  {d['full_name']} (логин {d['login']}) — уже существует")
        created_doctors.append(existing)
        continue

    doctor = Doctor(
        full_name=d["full_name"],
        login=d["login"],
        password_hash=hash_password(d["password"]),
        specialty=d["specialty"],
        hospital=d["hospital"],
        is_active=True,
    )
    db.add(doctor)
    db.commit()
    db.refresh(doctor)
    created_doctors.append(doctor)
    print(f"  ✅ {d['full_name']} ({d['specialty']})")


print()
print("=" * 60)
print("Создание пациентов...")
print("=" * 60)

# Распределяем пациентов по врачам (по кругу)
for i, p in enumerate(PATIENTS):
    doctor = created_doctors[i % len(created_doctors)]

    patient = create_patient(
        db,
        full_name=p["full_name"],
        sex=p["sex"],
        birth_date=p["birth_date"],
        doctor_id=str(doctor.id),
        attending_doctor_id=str(doctor.id),
        phone=p["phone"],
        email=p["email"],
        address=p["address"],
        policy_oms=p["policy_oms"],
        blood_type=p["blood_type"],
        height_cm=p["height_cm"],
        weight_kg=p["weight_kg"],
        reason=p["reason"],
        complaints=p["complaints"],
        allergies=p["allergies"],
    )
    print(f"  ✅ {p['full_name']} → врач {doctor.full_name}")


print()
print("=" * 60)
print(f"ГОТОВО! Создано врачей: {len(created_doctors)}, пациентов: {len(PATIENTS)}")
print("=" * 60)
print()
print("Логины врачей (пароль у всех: doctor123):")
for d in DOCTORS:
    print(f"  {d['login']:15} — {d['full_name']} ({d['specialty']})")

db.close()