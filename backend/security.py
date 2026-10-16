from passlib.context import CryptContext
from cryptography.fernet import Fernet
import os

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

FERNET_KEY = os.getenv("FERNET_KEY")
if not FERNET_KEY:
    raise ValueError("FERNET_KEY не установлен в переменных окружения")

fernet = Fernet(FERNET_KEY.encode() if isinstance(FERNET_KEY, str) else FERNET_KEY)

## Хэширование пароля. (Сверяем хэш вводимого пароля с хранящимся хэшем)
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

## Шифрование данных. С использованием переменной окружения
def _encrypt(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    return fernet.encrypt(value.encode()).decode()

def _decrypt(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    return fernet.decrypt(value.encode()).decode()


def encrypt_fio(fio: str) -> str:
    return _encrypt(fio)

def decrypt_fio(encrypted_fio: str) -> str:
    return _decrypt(encrypted_fio)


def encrypt_field(value: str | None) -> str | None:
    return _encrypt(value)

def decrypt_field(value: str | None) -> str | None:
    return _decrypt(value)