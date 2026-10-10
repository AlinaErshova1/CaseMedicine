import json
from pathlib import Path

LOG_FILE = Path("logs.jsonl")


def show_logs(level: str = None, user_id: str = None, limit: int = 20):
    if not LOG_FILE.exists():
        print("Файл логов пуст.")
        return

    entries = []
    with LOG_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    if level:
        entries = [e for e in entries if e["level"] == level]
    if user_id:
        entries = [e for e in entries if e["user_id"] == user_id]

    for entry in entries[-limit:]:
        print(f"[{entry['timestamp']}] {entry['level']:<7} "
              f"{entry['action']:<20} user={entry['user_id']} | {entry['details']}")


if __name__ == "__main__":
    print("=== Последние 10 INFO-записей ===")
    show_logs(level="INFO", limit=10)

    print("\n=== Все ERROR-записи ===")
    show_logs(level="ERROR")