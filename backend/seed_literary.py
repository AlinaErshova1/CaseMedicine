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
PATIENTS =  [
    {
      "full_name": "Сато Кэнтаро",
      "sex": "M",
      "birth_date": "1996-05-14",
      "phone": "+7 (911) 401-12-34",
      "email": "postman.cabbage@mail.ru",
      "address": "г. Москва, ул. Почтовая, д. 14, кв. 3",
      "policy_oms": "7712 401 123 401",
      "blood_type": "II (A) Rh+",
      "height_cm": 174,
      "weight_kg": 62,
      "reason": "Длительные непрекращающиеся головные боли и эпизод потери сознания на рабочем месте",
      "complaints": "Распирающая головная боль, двоение в глазах, тошнота, субфебрильная температура, выраженная слабость",
      "allergies": "Не выявлено"
    },
    {
      "full_name": "Монро Ив",
      "sex": "F",
      "birth_date": "1974-12-12",
      "phone": "+7 (916) 402-23-45",
      "email": "eve.monroe.author@mail.ru",
      "address": "г. Москва, Патриарший пер., д. 6, кв. 19",
      "policy_oms": "7712 402 234 502",
      "blood_type": "I (0) Rh+",
      "height_cm": 168,
      "weight_kg": 58,
      "reason": "Внезапный приступ судорог и острая дезориентация во времени и пространстве",
      "complaints": "Провалы в памяти, спутанность речи, быстрая утомляемость, трудности с концентрацией внимания",
      "allergies": "Пенициллин"
    },
    {
      "full_name": "Кранц Чарльз",
      "sex": "M",
      "birth_date": "1987-09-21",
      "phone": "+7 (903) 403-34-56",
      "email": "chuck.krantz39@mail.ru",
      "address": "г. Санкт-Петербург, наб. реки Фонтанки, д. 42, кв. 8",
      "policy_oms": "7812 403 345 603",
      "blood_type": "III (B) Rh+",
      "height_cm": 180,
      "weight_kg": 76,
      "reason": "Учащающиеся приступы сильных головных болей и эпизоды спутанности сознания",
      "complaints": "Выраженная астения, нарушение координации движений, шаткость походки, головные боли",
      "allergies": "Не выявлено"
    },
    {
      "full_name": "Халлинг Фредерик",
      "sex": "M",
      "birth_date": "1978-04-03",
      "phone": "+7 (921) 404-45-67",
      "email": "f.halling.school@mail.ru",
      "address": "г. Санкт-Петербург, Большой пр. П.С., д. 58, кв. 14",
      "policy_oms": "7812 404 456 704",
      "blood_type": "IV (AB) Rh-",
      "height_cm": 184,
      "weight_kg": 82,
      "reason": "Резкое изменение поведения, импульсивность и когнитивные нарушения, замеченные семьёй",
      "complaints": "Снижение критики к собственному поведению, вспышки раздражительности, периодические головные боли",
      "allergies": "Пыльца злаковых трав"
    },
    {
      "full_name": "Барнс Дженнифер",
      "sex": "F",
      "birth_date": "1978-06-17",
      "phone": "+7 (915) 405-56-78",
      "email": "jennifer.barnes@mail.ru",
      "address": "г. Москва, ул. Остоженка, д. 25, кв. 11",
      "policy_oms": "7712 405 567 805",
      "blood_type": "II (A) Rh-",
      "height_cm": 167,
      "weight_kg": 61,
      "reason": "Внезапные эпизоды потери равновесия и ухудшение самочувствия неясного генеза",
      "complaints": "Сильные головные боли, выраженное головокружение, тошнота, общая слабость",
      "allergies": "Лидокаин"
    },
    {
      "full_name": "Квентин Дэниел",
      "sex": "M",
      "birth_date": "2009-07-08",
      "phone": "+7 (926) 406-67-89",
      "email": "dq.warrior@mail.ru",
      "address": "г. Москва, ул. Малая Бронная, д. 12, кв. 5",
      "policy_oms": "7712 406 678 906",
      "blood_type": "I (0) Rh-",
      "height_cm": 172,
      "weight_kg": 54,
      "reason": "Прогрессирующая слабость в нижних конечностях и интенсивные головные боли",
      "complaints": "Нарушение опорно-двигательной функции (передвигается в кресле-коляске), тошнота, утомляемость",
      "allergies": "Цитрусовые"
    },
    {
      "full_name": "Картер Уильям",
      "sex": "M",
      "birth_date": "1991-11-02",
      "phone": "+7 (905) 407-78-90",
      "email": "will.carter.uk@mail.ru",
      "address": "г. Москва, ул. Тверская, д. 19, кв. 31",
      "policy_oms": "7712 407 789 007",
      "blood_type": "III (B) Rh-",
      "height_cm": 182,
      "weight_kg": 79,

        "reason": "Острые приступы головокружения, расстройство памяти и эпизоды дезориентации",
        "complaints": "Провалы в памяти, интенсивные головные боли, повышенная тревожность",
        "allergies": "Не выявлено"
    },
    {
        "full_name": "Самойлов Глеб Игоревич",
        "sex": "M",
        "birth_date": "2001-02-19",
        "phone": "+7 (916) 408-89-01",
        "email": "gleb.drumir@mail.ru",
        "address": "г. Москва, Ленинградский пр-т, д. 37, кв. 44",
        "policy_oms": "7712 408 890 108",
        "blood_type": "II (A) Rh+",
        "height_cm": 178,
        "weight_kg": 66,
        "reason": "Некупируемые головные боли и нарастающее угнетение сознания",
        "complaints": "Интенсивная головная боль, выраженная кахексия, заторможенность, кратковременные потери сознания",
        "allergies": "Не выявлено"
    }
]


# ============================================================
# ЗАПОЛНЕНИЕ
# ============================================================
print("=" * 60)
print("Проверка и создание пациентов...")
print("=" * 60)

from security import decrypt_field

# Получаем всех существующих врачей из базы (чтобы к ним привязать пациентов)
db_doctors = db.query(Doctor).all()
if not db_doctors:
    print("❌ Ошибка: В базе нет врачей! Сначала создайте хотя бы одного врача.")
    db.close()
    sys.exit(1)

# Собираем ФИО существующих пациентов (расшифровываем), чтобы не создавать дубликаты
existing_patients_names = set()
all_patients = db.query(Patient).all()
for p in all_patients:
    existing_patients_names.add(decrypt_field(p.full_name_encrypted))

created_count = 0
skipped_count = 0

# Распределяем пациентов по врачам (по кругу)
for i, p in enumerate(PATIENTS):
    # Проверка на дубликат
    if p["full_name"] in existing_patients_names:
        print(f"  ⏭  {p['full_name']} — уже существует в базе")
        skipped_count += 1
        continue

    doctor = db_doctors[i % len(db_doctors)]

    # Преобразуем строку даты в объект date
    birth_date = datetime.strptime(p["birth_date"], "%Y-%m-%d").date()

    patient = create_patient(
        db,
        full_name=p["full_name"],
        sex=p["sex"],
        birth_date=birth_date,
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
    created_count += 1

print()
print("=" * 60)
print(f"ГОТОВО! Создано новых пациентов: {created_count}. Пропущено (уже были): {skipped_count}")
print("=" * 60)

db.close()