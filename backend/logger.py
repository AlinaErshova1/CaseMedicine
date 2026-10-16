import json
from datetime import datetime
from pathlib import Path

LOG_FILE = Path(__file__).parent / "logs.jsonl"


def _write(level: str, action: str, user_id: str = None, details: str = ""):
    """Внутренняя функция: формирует JSON и пишет одну строку в файл."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "level": level,
        "action": action,
        "user_id": str(user_id) if user_id else None,
        "details": details,
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def write_log(action: str, user_id: str = None, details: str = ""):
    """Информационное событие: создание, просмотр, обновление."""
    _write("INFO", action, user_id, details)


def write_warning(action: str, user_id: str = None, details: str = ""):
    """Предупреждение: что-то подозрительное, но не критичное."""
    _write("WARNING", action, user_id, details)


def write_error(action: str, user_id: str = None, details: str = ""):
    """Ошибка: операция не удалась."""
    _write("ERROR", action, user_id, details)