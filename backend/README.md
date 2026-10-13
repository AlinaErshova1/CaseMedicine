# Бэкенд — AI-MRI

REST API для системы интеллектуального анализа МРТ головного мозга. Реализован на **FastAPI** с использованием **PostgreSQL** и обеспечивает работу с врачами, администраторами, пациентами и МРТ-снимками.

## Стек

- **Python 3.12**
- **FastAPI** — веб-фреймворк
- **SQLAlchemy 2.x** — ORM
- **PostgreSQL 17** — БД
- **psycopg 3** — драйвер PostgreSQL
- **python-jose** — JWT-токены
- **passlib + bcrypt 4.0.1** — хеширование паролей
- **cryptography (Fernet)** — шифрование персональных данных
- **Pydantic** — валидация входящих данных
- **Uvicorn** — ASGI-сервер

## Структура

```
backend/
├── api.py                  # Все эндпоинты FastAPI + точка входа
├── database.py             # Подключение к PostgreSQL, get_db()
├── models.py               # SQLAlchemy: Doctor, Admin, Patient, Study
├── security.py             # Fernet (шифрование) + bcrypt (пароли)
├── logger.py               # JSON-аудит в logs.jsonl
├── patients_api.py         # Бизнес-логика работы с пациентами
├── users_admin.py          # Создание врачей и админов
├── seed_data.py            # Базовое заполнение БД
├── seed_literary.py        # Синтетические данные (литературные персонажи)
├── uploads/                # МРТ-снимки (не в Git)
│   ├── originals/
│   └── processed/
└── logs.jsonl              # Аудит-лог (не в Git)
```

## Установка

### 1. Зависимости

```bash
pip install fastapi "uvicorn[standard]" python-jose[cryptography] python-multipart
pip install sqlalchemy "psycopg[binary]" passlib[bcrypt] cryptography python-dotenv
pip install "bcrypt==4.0.1"
```

**Важно:** именно `bcrypt==4.0.1`. Более новые версии несовместимы с `passlib 1.7.4`.

### 2. PostgreSQL

```bash
psql -U postgres -c "CREATE DATABASE medical_ai;"
```

### 3. Файл `.env`

Создайте `Medicine/.env` **в корне проекта**:

```env
DATABASE_URL=postgresql://postgres:ТВОЙ_ПАРОЛЬ@localhost:5432/medical_ai
FERNET_KEY=СГЕНЕРИРОВАННЫЙ_КЛЮЧ
SECRET_KEY=СЛУЧАЙНАЯ_СТРОКА
```

**Генерация:**

```bash
# FERNET_KEY
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# SECRET_KEY
python -c "import secrets; print(secrets.token_hex(32))"
```

### 4. Заполнение БД

```bash
# Базовое (админ + 2 врача + 1 пациент)
python seed_data.py

# И синтетика с литературными персонажами (5 врачей + 15 пациентов)
python seed_literary.py
```

### 5. Запуск

```bash
uvicorn backend.api:app --reload --port 8000
```

- API: **http://localhost:8000**
- Swagger: **http://localhost:8000/docs**

## Роли и права доступа

| Действие | Врач | Админ |
|---|---|---|
| Логин | ✅ | ✅ |
| Смотреть **своих** пациентов | ✅ | ✅ |
| Смотреть **всех** пациентов | ❌ | ✅ |
| Создавать пациентов | ✅ (привязка к себе) | ✅ (выбор врача) |
| Редактировать пациента | ❌ | ✅ |
| Удалять пациента | ❌ | ✅ |
| Загружать МРТ-снимки | ✅ | ✅ |
| Запускать анализ | ✅ | ✅ |
| Удалять снимки | ❌ | ✅ |
| Создавать врачей /редактировать их данные | ❌ | ✅ |
| Смотреть аудит-лог | ❌ | ✅ (через `logs.jsonl`) |

## Эндпоинты

### Аутентификация

```
POST /auth/login?login=ЛОГИН&password=ПАРОЛЬ
```

Возвращает:
```json
{
  "access_token": "eyJhbGciOi...",
  "token_type": "bearer",
  "role": "doctor" | "admin",
  "user": { "id": "...", "full_name": "...", "specialty": "..." }
}
```

Токен действует **8 часов**. Передаётся в заголовке `Authorization: Bearer <token>`.

### Пациенты

| Метод | URL | Роли | Описание |
|---|---|---|---|
| GET | `/patients` | doctor, admin | Список пациентов (врач — свои, админ — все) |
| POST | `/patients` | doctor, admin | Создание пациента |
| GET | `/patients/{id}` | doctor, admin | Полная карточка |
| GET | `/patients/{id}/studies` | doctor, admin | Исследования пациента |

### Снимки

| Метод | URL | Роли | Описание |
|---|---|---|---|
| POST | `/upload` | doctor, admin | Загрузка PNG/JPG |
| POST | `/studies/{id}/analyze` | doctor, admin | Запуск CV-анализа (заглушка) |
| GET | `/files/originals/{filename}` | авторизованные | Оригинал снимка |
| GET | `/files/processed/{filename}` | авторизованные | Результат анализа |

### Админ

| Метод | URL | Описание |
|---|---|---|
| GET | `/admin/doctors` | Все врачи |
| POST | `/admin/doctors` | Создать врача |
| PATCH | `/admin/doctors/{id}` | Редактировать врача |
| PATCH | `/admin/doctors/{id}/toggle-active` | Активировать/отключить |
| DELETE | `/admin/doctors/{id}` | Удалить (если нет пациентов) |
| GET | `/admin/patients` | Все пациенты |
| PATCH | `/admin/patients/{id}` | Редактировать пациента |
| DELETE | `/admin/patients/{id}` | Удалить пациента + снимки |
| DELETE | `/admin/studies/{id}` | Удалить снимок |

### Служебные

| Метод | URL | Описание |
|---|---|---|
| GET | `/` | Health-check |
| GET | `/doctors` | Краткий список активных врачей (для dropdown) |
| GET | `/docs` | Swagger UI |

## Безопасность (152-ФЗ)

### Что шифруется

Симметричное шифрование **Fernet** (AES-128-CBC + HMAC-SHA256):

| Поле | Как хранится в БД |
|---|---|
| ФИО | `full_name_encrypted` |
| Телефон | `phone_encrypted` |
| Email | `email_encrypted` |
| Адрес | `address_encrypted` |
| Полис ОМС | `policy_oms_encrypted` |

**Ключ шифрования** (`FERNET_KEY`) хранится **только** в `.env` на сервере и **никогда** не попадает в клиентские запросы.

### Что хешируется

Пароли врачей и админов — **bcrypt** с солью. Необратимо.

### Обезличивание

Каждому пациенту присваивается **обезличенный код** вида `P-2026-A1B2`, который используется во фронтенде вместо UUID.

### Аудит

Все действия записываются в **`logs.jsonl`** в формате JSON-lines:

```json
{"timestamp": "2026-10-13T12:00:00Z", "level": "INFO", "action": "login_success", "user_id": "...", "details": "role=doctor login=bormental"}
{"timestamp": "2026-10-13T12:00:05Z", "level": "INFO", "action": "view_patient", "user_id": "...", "details": "patient_id=..."}
```

### Разграничение доступа

Врач видит **только своих** пациентов: тех, кого создал (`created_by`), и тех, кого лечит (`attending_doctor_id`).

## Тестовые данные

### Админ

| Логин | Пароль |
|---|---|
| `admin` | `admin123` |

### Врачи (пароль у всех — `doctor123`)

| Логин | ФИО | Специальность |
|---|---|---|
| `bormental` | Борменталь Иван Арнольдович | Невролог |
| `stravinsky` | Стравинский Афанасий Иванович | Рентгенолог |
| `vonzel` | Вонзель Фёдор Фёдорович | Нейрохирург |
| `dorn` | Дорн Аркадий Иванович | Терапевт |
| `chekhov` | Чехов Антон Павлович | Врач общей практики |

## Тестирование

### Через Swagger

1. Открой http://localhost:8000/docs
2. **POST /auth/login** → Try it out → введи `login=bormental`, `password=doctor123` → Execute
3. Скопируй `access_token`
4. Нажми **Authorize** (замок) → вставь токен → Authorize
5. Теперь можешь вызывать защищённые эндпоинты

### Через curl

```bash
# Логин
curl -X POST "http://localhost:8000/auth/login?login=bormental&password=doctor123"

# Список пациентов (подставь токен)
curl -H "Authorization: Bearer ТОКЕН" http://localhost:8000/patients
```

## Что НЕ попадает в Git

- `.env` — секреты
- `.venv/` — виртуальное окружение
- `db.sqlite3` — служебная БД Django
- `backend/uploads/` — медицинские снимки (персональные данные)
- `logs.jsonl` — аудит-лог
- `__pycache__/` — кэш Python
