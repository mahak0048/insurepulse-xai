const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000/api';

export async function api(path, options = {}) {
  const token = localStorage.getItem('insurepulse_token');
  const headers = new Headers(options.headers || {});
  if (!headers.has('Content-Type') && options.body && !(options.body instanceof FormData)) {
    headers.set('Content-Type', 'application/json');
  }
  if (token) headers.set('Authorization', `Bearer ${token}`);
  const res = await fetch(`${API_BASE}${path}`, { ...options, headers });
  let data = null;
  try { data = await res.json(); } catch { data = null; }
  if (res.status === 401) {
    localStorage.removeItem('insurepulse_token');
    localStorage.removeItem('insurepulse_user');
  }
  if (!res.ok) throw new Error(data?.detail || 'Request failed.');
  return data;
}

export function setAuth(result) {
  localStorage.setItem('insurepulse_token', result.token);
  localStorage.setItem('insurepulse_user', JSON.stringify(result.user));
}
export function getUser() {
  try { return JSON.parse(localStorage.getItem('insurepulse_user') || 'null'); } catch { return null; }
}
export function logout() {
  localStorage.removeItem('insurepulse_token');
  localStorage.removeItem('insurepulse_user');
}
