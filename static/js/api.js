// static/js/api.js
// Общий модуль для работы с FastAPI

const API_URL = "http://localhost:8000";

/**
 * Логин врача. Возвращает {access_token, token_type, doctor}.
 */
async function apiLogin(login, password) {
  const url = `${API_URL}/auth/login?login=${encodeURIComponent(login)}&password=${encodeURIComponent(password)}`;
  const res = await fetch(url, { method: "POST" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Неверный логин или пароль");
  }
  return res.json();
}

/**
 * Список пациентов текущего врача.
 */
async function apiGetPatients() {
  const token = localStorage.getItem("token");
  if (!token) {
    window.location.href = "/login/";
    throw new Error("Не авторизован");
  }
  const res = await fetch(`${API_URL}/patients`, {
    headers: { "Authorization": `Bearer ${token}` }
  });
  if (res.status === 401) {
    localStorage.clear();
    window.location.href = "/login/";
    throw new Error("Сессия истекла");
  }
  if (!res.ok) throw new Error("Ошибка загрузки пациентов");
  return res.json();
}

/**
 * Полная карточка пациента.
 */
async function apiGetPatient(id) {
  const token = localStorage.getItem("token");
  if (!token) {
    window.location.href = "/login/";
    throw new Error("Не авторизован");
  }
  const res = await fetch(`${API_URL}/patients/${id}`, {
    headers: { "Authorization": `Bearer ${token}` }
  });
  if (res.status === 401) {
    localStorage.clear();
    window.location.href = "/login/";
    throw new Error("Сессия истекла");
  }
  if (!res.ok) throw new Error("Пациент не найден");
  return res.json();
}

/**
 * Логаут: чистим localStorage и на логин.
 */
function apiLogout() {
  localStorage.clear();
  window.location.href = "/login/";
}

