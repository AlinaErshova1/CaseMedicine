from pathlib import Path
import hashlib
from tqdm import tqdm


def remove_duplicates_in_folder(root_dir):
    root = Path(root_dir)
    hashes = set()
    removed_count = 0

    img_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    files = [f for f in root.glob("**/*") if f.suffix.lower() in img_exts]

    print(f"[INFO] Сканирование {len(files)} файлов на дубликаты в {root_dir}...")
    for file_path in tqdm(files):
        try:
            with open(file_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            if file_hash in hashes:
                file_path.unlink()  # Удаляем дубликат
                removed_count += 1
            else:
                hashes.add(file_hash)
        except Exception as e:
            print(f"[WARN] Ошибка с файлом {file_path}: {e}")

    print(f"[DONE] Удалено дубликатов: {removed_count}\n")


if __name__ == "__main__":
    # Запуск очистки для папки с картинками и масками
    remove_duplicates_in_folder("data/train")
    remove_duplicates_in_folder("data/test")