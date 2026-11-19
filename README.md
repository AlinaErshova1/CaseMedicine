# AI-MRI: Система поддержки принятия врачебных решений

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Django](https://img.shields.io/badge/Backend-Django-darkgreen?logo=django)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)
![PostgreSQL](https://img.shields.io/badge/DB-PostgreSQL-blue?logo=postgresql)
![PyTorch](https://img.shields.io/badge/ML-PyTorch-EE4C2C?logo=pytorch)
![OpenCV](https://img.shields.io/badge/CV-OpenCV-white?logo=opencv)
![JavaScript](https://img.shields.io/badge/Frontend-JavaScript-yellow?logo=javascript)
![HTML5](https://img.shields.io/badge/Frontend-HTML5-E34F26?logo=html5)
![CSS3](https://img.shields.io/badge/Frontend-CSS3-1572B6?logo=css3)

**AI-MRI** — это комплексная веб-платформа для анализа МРТ-снимков головного мозга с использованием компьютерного зрения. Система помогает врачам выявлять новообразования, классифицировать их, сегментировать пораженные области и формировать итоговые протоколы диагностики.

Проект разработан в рамках кейс-чемпионата и представляет собой связку: **Frontend (Django)** + **Backend (FastAPI)** + **ML-модуль (PyTorch)**.

---

## Основные возможности

*   **Ролевая модель:** Разделение на администраторов и врачей.
*   **Анализ МРТ в один клик:**
    *   Определение проекции снимка.
    *   Классификация типа опухоли с указанием уверенности модели.
    *   Автоматическая сегментация опухоли и расчет площади поражения в процентах.
    *   Определение анатомической локализации.
*   **Инструменты врача:**
    *   Возможность **ручного выделения** опухоли (кистью или полигонами) прямо поверх снимка, если ИИ ошибся или нужно более точное выделение.
    *   **Верификация данных:** врач проверяет результаты ИИ, при необходимости корректирует их, ставит галочки подтверждения и система генерирует официальное заключение.
*   **Экспорт:** Скачивание готового заключения в формате Word (.doc).
*   **Безопасность (152-ФЗ):** Шифрование персональных данных пациентов с помощью Fernet.

---
## Безопасность

* **Персональные данные** пациентов не хранятся в открытом виде. Они шифруются алгоритмом Fernet перед сохранением в PostgreSQL.

* **Пароли** пользователей хэшируются и хранятся исключительно в виде bcrypt-хэшей.
 
* **Аудит всех действий** ведется в файле logs.jsonl.

---

## Установка и запуск

### 1. Клонирование репозитория
```bash
git clone https://github.com/AlinaErshova1/CaseMedicine.git
cd CaseMedicine
```

### 2. Настройка окружения
Создайте и активируйте виртуальное окружение:

Для Windows:

```
python -m venv .venv
.venv\Scripts\activate
```

Для Mac/Linux:

```
python -m venv .venv
source .venv/bin/activate
```

#### Установите зависимости:

```
pip install -r requirements.txt
```

(Убедитесь, что установлен bcrypt==4.0.1, иначе passlib может выдать ошибку).

### 3. Настройка .env
Создайте файл .env в корне проекта (рядом с manage.py) и заполните его:

```
DATABASE_URL=postgresql://postgres:ТВОЙ_ПАРОЛЬ@localhost:5432/medical_ai
FERNET_KEY=СГЕНЕРИРОВАННЫЙ_КЛЮЧ
SECRET_KEY=СЛУЧАЙНАЯ_СТРОКА
```

(Ключи можно сгенерировать через Python: from cryptography.fernet import Fernet; print(Fernet.generate_key().decode()))

### 4. Запуск базы данных и миграций
Создайте БД в PostgreSQL:

```
CREATE DATABASE medical_ai;
```

Примените миграции:

```
python manage.py migrate
python backend/seed_data.py
```

### 5. Запуск серверов (в двух терминалах)
Терминал 1 — FastAPI (Backend & ML):

```
uvicorn backend.api:app --reload --port 8000
```

Терминал 2 — Django (Frontend):

```
python manage.py runserver 8001
```

Интерфейс будет доступен по адресу: http://127.0.0.1:8001

---

Для демонстрации проекта жюри или заказчикам сайт был опубликован в интернет с помощью туннеля CloudPub.
Доступен по ссылке https://calmly-liked-midge.cloudpub.ru по предварительной договорённости.

---

### Состав команды MIX

*  [Важова Дарья Юрьевна](https://github.com/DariaVazh) (тг - @dariavazh) - специалист по ML и CV. (Разработка и обучение моделей для распознавания опухоли с последующим выделением. Разработка моделей для локализации опухолей)

* [Ершова Алина Дмитриевна](https://github.com/AlinaErshova1) (тг - @aalliinnaaaaaaa) - backend разработка. (Разработка REST API на базе FastAPI. Настройка взаимодействия с БД PostgreSQL через SQLAlchemy. Безопастность ПД пациентов. Аудит действий)

* [Мелешенко Юлия Андреевна](https://github.com/Ewryser) (тг - @Yolloid) - frontend разработка. (Разработка пользовательского интерфейса на базе Django Templates, HTML5, CSS3 и Vanilla JavaScript. Реализация адаптивной верстки личного кабинета врача и панели администратора. Настройка взаимодействия фронтенда с бэкендом через Fetch API)

* Колесник Алёна Александровна (тг - @soul1alenka) - капитан команды, идейный вдохновитель.

---

Все демонстранционные данные пациентов являются вымышленными.

