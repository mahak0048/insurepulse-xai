from __future__ import annotations

import base64
import csv
import hashlib
import hmac
import io
import json
import os
import random
import secrets
import smtplib
import string
import time
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from functools import lru_cache
from typing import Any, Optional

import mysql.connector
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(title="InsurePulse XAI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.getenv("FRONTEND_ORIGIN", "http://localhost:5173"),
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET = os.getenv("INSUREPULSE_SECRET", "dev-only-change-this")
TOKEN_TTL_MINUTES = 60 * 8
security = HTTPBearer(auto_error=False)

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "password"),
    "database": os.getenv("DB_NAME", "insurepulse_xai"),
}

SETTINGS_DEFAULTS = {
    "low_risk_max": 35.0,
    "high_risk_min": 65.0,
    "auto_approval_enabled": True,
    "session_timeout_minutes": 30,
    "require_reauth_sensitive": True,
    "notify_app_submitted": True,
    "notify_status_changed": True,
    "notify_review_alerts": True,
    "notification_email": "admin@insurepulse.com",
    "default_policy_type": "Health Insurance",
    "default_page_size": 10,
    "date_time_format": "DD-MM-YYYY (24h)",
    "theme": "System",
    "data_retention_period": "1 Year",
}
SETTINGS = dict(SETTINGS_DEFAULTS)

CONDITION_OPTIONS = [
    "Asthma", "Diabetes", "Hypertension", "Heart Disease", "Thyroid Disorder",
    "Kidney Disease", "Liver Disease", "Cancer", "Other",
]
COVERAGE_TIERS = [100000, 200000, 300000, 500000, 1000000, 1500000, 2000000, 2500000, 5000000, 10000000]
INCOME_BRACKETS = [
    ("Below ₹3,00,000", 150000),
    ("₹3,00,000 - ₹6,00,000", 450000),
    ("₹6,00,000 - ₹10,00,000", 800000),
    ("₹10,00,000 - ₹20,00,000", 1500000),
    ("₹20,00,000 - ₹50,00,000", 3500000),
    ("Above ₹50,00,000", 6000000),
]
POLICY_TYPES = ["Health Insurance"]
STATUS_LABEL = {
    "APPROVED": "APPROVED",
    "REJECTED": "REJECTED",
    "PENDING_ADMIN_REVIEW": "PENDING REVIEW",
    "MEDICAL_AUDIT_REQUIRED": "MEDICAL AUDIT",
}
ALCOHOL_MAP = {"None": 0, "Occasional": 1, "Regular": 2, "Heavy": 3}
ALCOHOL_REVERSE = {v: k for k, v in ALCOHOL_MAP.items()}
FEATURE_ORDER = [
    "age", "bmi", "smoker", "pre_existing_conditions", "family_history",
    "alcohol_score", "exercise_frequency",
]
FEATURE_DESCRIPTIONS = {
    "age": lambda v: f"Applicant age ({int(v)} years)",
    "bmi": lambda v: f"Body Mass Index ({v:.1f})",
    "smoker": lambda v: "Smoking history (Current smoker)" if v == 1 else "Smoking history (Non-smoker)",
    "pre_existing_conditions": lambda v: f"Pre-existing medical conditions ({int(v)} reported)",
    "family_history": lambda v: "Family history of major illness (Present)" if v == 1 else "Family history of major illness (None)",
    "alcohol_score": lambda v: f"Alcohol consumption ({ALCOHOL_REVERSE.get(int(v), 'Unknown')})",
    "exercise_frequency": lambda v: f"Exercise frequency ({int(v)} days/week)",
}

otp_store: dict[str, dict[str, Any]] = {}


def hash_password(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_conn(with_db: bool = True):
    cfg = dict(DB_CONFIG)
    kwargs = {
        "host": cfg["host"],
        "port": int(cfg["port"]),
        "user": cfg["user"],
        "password": cfg["password"],
        "autocommit": True,
        "connection_timeout": 6,
    }
    if with_db:
        kwargs["database"] = cfg["database"]
    return mysql.connector.connect(**kwargs)


def db_run(query: str, params: Optional[tuple] = None, fetch=False, fetchone=False):
    conn = get_conn()
    cur = conn.cursor(dictionary=True)
    cur.execute(query, params or ())
    result = cur.fetchone() if fetchone else cur.fetchall() if fetch else cur.lastrowid
    cur.close()
    conn.close()
    return result


CREATE_USERS_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL,
    full_name VARCHAR(150),
    email VARCHAR(150) UNIQUE NOT NULL,
    mobile VARCHAR(20) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""
CREATE_APPLICATIONS_SQL = """
CREATE TABLE IF NOT EXISTS applications (
    id INT AUTO_INCREMENT PRIMARY KEY,
    policy_reference VARCHAR(30) NOT NULL UNIQUE,
    applicant_username VARCHAR(50),
    full_name VARCHAR(150),
    age INT,
    gender VARCHAR(20),
    height_cm FLOAT,
    weight_kg FLOAT,
    bmi FLOAT,
    smoker TINYINT,
    alcohol_consumption VARCHAR(20),
    exercise_frequency INT,
    pre_existing_conditions INT,
    pre_existing_conditions_detail TEXT,
    family_history TINYINT,
    annual_income FLOAT,
    policy_type VARCHAR(30),
    coverage_amount FLOAT,
    risk_score FLOAT,
    premium_inr FLOAT NULL,
    status VARCHAR(40),
    shap_rationale TEXT,
    underwriter_notes TEXT,
    reviewed_by VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""
CREATE_SETTINGS_SQL = """
CREATE TABLE IF NOT EXISTS system_settings (
    setting_key VARCHAR(50) PRIMARY KEY,
    setting_value VARCHAR(255)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""
CREATE_AUDIT_SQL = """
CREATE TABLE IF NOT EXISTS audit_log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    admin_username VARCHAR(50),
    action VARCHAR(120),
    details TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""
CREATE_DRAFTS_SQL = """
CREATE TABLE IF NOT EXISTS application_drafts (
    applicant_username VARCHAR(50) PRIMARY KEY,
    draft_data TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


def init_db() -> tuple[bool, Optional[str]]:
    try:
        cfg = dict(DB_CONFIG)
        conn = get_conn(False)
        cur = conn.cursor()
        cur.execute(
            f"CREATE DATABASE IF NOT EXISTS `{cfg['database']}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        cur.close(); conn.close()
        conn = get_conn(True)
        cur = conn.cursor()
        for sql in [CREATE_USERS_SQL, CREATE_APPLICATIONS_SQL, CREATE_SETTINGS_SQL, CREATE_AUDIT_SQL, CREATE_DRAFTS_SQL]:
            cur.execute(sql)
        migrations = [
            "ALTER TABLE applications ADD COLUMN premium_inr FLOAT NULL AFTER risk_score",
            "ALTER TABLE applications ADD COLUMN pre_existing_conditions_detail TEXT AFTER pre_existing_conditions",
            "ALTER TABLE users ADD COLUMN mobile VARCHAR(20) AFTER email",
        ]
        for mig in migrations:
            try:
                cur.execute(mig)
            except Exception:
                pass
        cur.execute("SELECT COUNT(*) AS c FROM users WHERE role='admin'")
        if int(cur.fetchone()["c"]) == 0:
            cur.execute(
                "INSERT INTO users (username,password_hash,role,full_name,email,mobile) VALUES (%s,%s,%s,%s,%s,%s)",
                ("admin", hash_password("Admin@123"), "admin", "System Administrator", "admin@insurepulse.com", ""),
            )
        cur.execute("SELECT setting_key, setting_value FROM system_settings")
        existing = {r["setting_key"]: r["setting_value"] for r in cur.fetchall()}
        for key, val in SETTINGS_DEFAULTS.items():
            if key not in existing:
                sv = "1" if val is True else "0" if val is False else str(val)
                cur.execute("INSERT INTO system_settings (setting_key,setting_value) VALUES (%s,%s)", (key, sv))
            else:
                raw = existing[key]
                if isinstance(val, bool): SETTINGS[key] = raw == "1"
                elif isinstance(val, float): SETTINGS[key] = float(raw)
                elif isinstance(val, int): SETTINGS[key] = int(float(raw))
                else: SETTINGS[key] = raw
        cur.close(); conn.close()
        return True, None
    except Exception as exc:
        return False, str(exc)


@app.on_event("startup")
def startup_event():
    init_db()


def encode_token(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    body = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    sig = hmac.new(SECRET.encode(), body.encode(), hashlib.sha256).hexdigest()
    return f"{body}.{sig}"


def decode_token(token: str) -> dict[str, Any]:
    try:
        body, sig = token.split(".", 1)
        expected = hmac.new(SECRET.encode(), body.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            raise ValueError("bad signature")
        padded = body + "=" * (-len(body) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode()).decode())
        if int(payload.get("exp", 0)) < int(time.time()):
            raise ValueError("expired")
        return payload
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Session expired. Please sign in again.") from exc


def make_auth_token(row: dict[str, Any]) -> str:
    ttl = max(1, int(SETTINGS.get("session_timeout_minutes", 30)))
    return encode_token({
        "sub": row["username"],
        "role": row["role"],
        "full_name": row.get("full_name") or row["username"],
        "email": row.get("email") or "",
        "exp": int(time.time()) + ttl * 60,
    })


def current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict[str, Any]:
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return decode_token(credentials.credentials)


def require_role(role: str):
    def dep(user: dict[str, Any] = Depends(current_user)):
        if user.get("role") != role:
            raise HTTPException(status_code=403, detail="You are not authorized for this action.")
        return user
    return dep


def audit(admin_username: str, action: str, details: str = ""):
    try:
        db_run("INSERT INTO audit_log (admin_username, action, details) VALUES (%s,%s,%s)", (admin_username, action, details))
    except Exception:
        pass


def serialize(value: Any):
    if isinstance(value, (datetime,)):
        return value.isoformat(sep=" ")
    if isinstance(value, np.generic):
        return value.item()
    return value


def serialize_row(row: dict[str, Any] | None):
    if row is None:
        return None
    return {k: serialize(v) for k, v in row.items()}


class LoginIn(BaseModel):
    username: str
    password: str
    role: str = "applicant"

class RegisterIn(BaseModel):
    username: str
    password: str
    full_name: str
    email: str
    mobile: str

class ForgotStartIn(BaseModel):
    username: str
    email: str

class ForgotVerifyIn(BaseModel):
    challenge: str
    otp: str

class ForgotResetIn(BaseModel):
    username: str
    token: str
    password: str

class DraftIn(BaseModel):
    draft: dict[str, Any]

class ApplicationIn(BaseModel):
    personal: dict[str, Any]
    health: dict[str, Any]
    policy: dict[str, Any]

class StatusUpdateIn(BaseModel):
    status: str
    notes: str = ""

class NotesUpdateIn(BaseModel):
    notes: str = ""

class SettingsIn(BaseModel):
    settings: dict[str, Any]

class DBConfigIn(BaseModel):
    host: str
    port: int
    user: str
    password: str
    database: str


@app.get("/api/health")
def health():
    ok, err = init_db()
    return {"ok": ok, "error": err}

@app.post("/api/auth/login")
def login(payload: LoginIn):
    row = db_run("SELECT * FROM users WHERE username=%s AND role=%s", (payload.username.strip(), payload.role), fetchone=True)
    if not row or row["password_hash"] != hash_password(payload.password):
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    token = make_auth_token(row)
    audit(row["username"], "ADMIN_LOGIN" if row["role"] == "admin" else "APPLICANT_LOGIN", f"{row['role']} logged in.")
    return {"token": token, "user": {"username": row["username"], "role": row["role"], "full_name": row["full_name"], "email": row["email"]}}

@app.post("/api/auth/register")
def register(payload: RegisterIn):
    if not payload.full_name.strip() or not payload.email.strip() or not payload.mobile.strip() or not payload.username.strip():
        raise HTTPException(status_code=400, detail="All required fields must be filled.")
    p = payload.password
    if not (len(p) >= 8 and any(c.isupper() for c in p) and any(c.isdigit() for c in p) and any(not c.isalnum() for c in p)):
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters long, contain one capital letter, one number, and one special character.")
    if p != payload.password:
        raise HTTPException(status_code=400, detail="Passwords do not match.")
    try:
        existing = db_run("SELECT id FROM users WHERE username=%s", (payload.username.strip(),), fetchone=True)
        if existing:
            raise HTTPException(status_code=400, detail="That username is already taken.")
        db_run(
            "INSERT INTO users (username,password_hash,role,full_name,email,mobile) VALUES (%s,%s,'applicant',%s,%s,%s)",
            (payload.username.strip(), hash_password(p), payload.full_name.strip(), payload.email.strip(), payload.mobile.strip()),
        )
        return {"ok": True}
    except mysql.connector.Error as exc:
        if getattr(exc, "errno", None) == 1062:
            raise HTTPException(status_code=400, detail="This email or mobile number is already registered, please login.") from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/api/auth/forgot/start")
def forgot_start(payload: ForgotStartIn):
    user = db_run("SELECT id FROM users WHERE username=%s AND email=%s", (payload.username.strip(), payload.email.strip()), fetchone=True)
    if not user:
        raise HTTPException(status_code=404, detail="No account matches that username and email combination.")
    otp = str(random.randint(100000, 999999))
    smtp_email = os.getenv("SMTP_EMAIL", "")
    smtp_password = os.getenv("SMTP_APP_PASSWORD", "")
    if not smtp_email or not smtp_password:
        raise HTTPException(status_code=500, detail="Email service is not configured on the backend.")
    try:
        msg = MIMEMultipart()
        msg["From"] = f"InsurePulse Security <{smtp_email}>"
        msg["To"] = payload.email.strip()
        msg["Subject"] = "Password Reset OTP - InsurePulse"
        msg.attach(MIMEText(f"Your 6-digit OTP to reset your InsurePulse account password is:\n\n{otp}\n\nThis OTP is valid for 10 minutes. Please do not share this OTP with anyone.\n", "plain"))
        server = smtplib.SMTP(os.getenv("SMTP_HOST", "smtp.gmail.com"), int(os.getenv("SMTP_PORT", "587")))
        server.starttls(); server.login(smtp_email, smtp_password); server.send_message(msg); server.quit()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to send email OTP: {exc}") from exc
    challenge = secrets.token_urlsafe(24)
    otp_store[challenge] = {"username": payload.username.strip(), "email": payload.email.strip(), "otp": otp, "expires": time.time() + 600, "verified": False}
    return {"ok": True, "challenge": challenge}

@app.post("/api/auth/forgot/verify")
def forgot_verify(payload: ForgotVerifyIn):
    rec = otp_store.get(payload.challenge)
    if rec is None:
        raise HTTPException(status_code=400, detail="OTP session not found.")
    if rec["expires"] < time.time() or payload.otp != rec["otp"]:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP.")
    rec["verified"] = True
    reset_token = secrets.token_urlsafe(24)
    otp_store[reset_token] = {**rec, "expires": time.time() + 600, "verified": True}
    return {"ok": True, "reset_token": reset_token}

@app.post("/api/auth/forgot/reset")
def forgot_reset(payload: ForgotResetIn):
    rec = otp_store.get(payload.token)
    if not rec or not rec.get("verified") or rec["expires"] < time.time() or rec["username"] != payload.username:
        raise HTTPException(status_code=400, detail="Password reset session is invalid or expired.")
    p = payload.password
    if not (len(p) >= 8 and any(c.isupper() for c in p) and any(c.isdigit() for c in p) and any(not c.isalnum() for c in p)):
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters long, contain one capital letter, one number, and one special character.")
    db_run("UPDATE users SET password_hash=%s WHERE username=%s", (hash_password(p), payload.username))
    otp_store.pop(payload.token, None)
    return {"ok": True}


# -------------------------- AI / underwriting --------------------------
@lru_cache(maxsize=1)
def generate_training_data(n=4000, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, n).astype(float)
    bmi = np.clip(rng.normal(26, 5, n), 15, 50)
    smoker = rng.binomial(1, 0.25, n).astype(float)
    pre_existing = np.clip(rng.poisson(0.6, n), 0, 5).astype(float)
    family_history = rng.binomial(1, 0.30, n).astype(float)
    alcohol_score = rng.integers(0, 4, n).astype(float)
    exercise_frequency = rng.integers(0, 8, n).astype(float)
    risk = (
        0.30 * (age / 80.0)
        + 0.20 * np.clip((bmi - 22.0) / 20.0, 0, 1)
        + 0.20 * smoker
        + 0.12 * np.clip(pre_existing / 5.0, 0, 1)
        + 0.10 * family_history
        + 0.05 * (alcohol_score / 3.0)
        - 0.12 * (exercise_frequency / 7.0)
        + 0.05
    )
    noise = rng.normal(0, 0.04, n)
    risk = np.clip(risk + noise, 0.01, 0.99)
    df = pd.DataFrame({
        "age": age, "bmi": bmi, "smoker": smoker,
        "pre_existing_conditions": pre_existing,
        "family_history": family_history,
        "alcohol_score": alcohol_score,
        "exercise_frequency": exercise_frequency,
    })
    return df[FEATURE_ORDER], pd.Series(risk, name="risk")

@lru_cache(maxsize=1)
def train_model_and_explainer():
    X, y = generate_training_data()
    model = xgb.XGBRegressor(
        objective="reg:squarederror", n_estimators=180, max_depth=4,
        learning_rate=0.07, subsample=0.9, colsample_bytree=0.9, random_state=42,
    )
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    return model, explainer


def predict_risk(feature_values: dict):
    model, explainer = train_model_and_explainer()
    X_row = pd.DataFrame([feature_values])[FEATURE_ORDER]
    raw_pred = float(model.predict(X_row)[0])
    risk_score = round(float(np.clip(raw_pred, 0, 1)) * 100, 2)
    shap_row = explainer.shap_values(X_row)[0]
    base_value = float(explainer.expected_value)
    return risk_score, shap_row, base_value * 100


NIVA_BUPA_ZONES = {"Zone 1 (Metro)": 1.15, "Zone 2 (Non-Metro)": 1.00}
MEDICAL_SEVERITY_CONFIG = {
    "ICD_CHAPTER_NEOPLASMS": {"default_tier": 4, "action": "REJECT"},
    "ICD_CHAPTER_CIRCULATORY": {"default_tier": 3, "action": "LOADING_OR_WAITING_PERIOD"},
    "ICD_CHAPTER_ENDOCRINE": {"default_tier": 2, "action": "RATE_UP"},
    "ICD_CHAPTER_INFECTIOUS": {"default_tier": 1, "action": "STANDARD"},
}

def evaluate_niva_bupa_underwriting(applicant_profile: dict, ml_risk_score: float, condition_records: list[dict]):
    base_risk_points = 0.0
    declination_triggers = []
    loading_factors = 1.0
    for record in condition_records:
        condition_name = record.get("condition", "").lower()
        severity_tier = record.get("severity_tier", 1)
        debits = severity_tier * 25.0
        base_risk_points += debits
        if severity_tier >= 4:
            declination_triggers.append(f"Critical Medical Decline: {record.get('condition')}")
        elif "hypertension" in condition_name or "blood pressure" in condition_name:
            loading_factors += 0.20
    if declination_triggers:
        return "REJECTED", f"Declined under Niva Bupa guidelines: {'; '.join(declination_triggers)}", base_risk_points
    annual_income = float(applicant_profile.get("annual_income", 0))
    requested_coverage = float(applicant_profile.get("coverage_amount", 0))
    max_allowable_coverage = annual_income * 15.0
    if requested_coverage > max_allowable_coverage:
        return "REJECTED", f"Financial Underwriting Rejection: Requested coverage (₹{requested_coverage:,.2f}) exceeds the 15x income rule limit (₹{max_allowable_coverage:,.2f}).", base_risk_points
    age = int(applicant_profile.get("age", 30))
    composite_risk_index = (base_risk_points / 100.0) + float(ml_risk_score / 100.0)
    if age >= 60 and len(condition_records) > 0:
        return "PENDING_ADMIN_REVIEW", "Senior profile with pre-existing conditions routed for mandatory Niva Bupa medical underwriting review.", composite_risk_index
    if not SETTINGS["auto_approval_enabled"]:
        return "PENDING_ADMIN_REVIEW", "Application routed for manual review (auto-approval disabled).", composite_risk_index
    if composite_risk_index >= 3.5 or composite_risk_index >= (SETTINGS["high_risk_min"] / 100.0):
        return "REJECTED", f"Composite Risk Index ({composite_risk_index:.2f}) exceeds acceptance threshold.", composite_risk_index
    if composite_risk_index <= (SETTINGS["low_risk_max"] / 100.0):
        return "APPROVED", "Meets Niva Bupa standard automated underwriting criteria.", composite_risk_index
    return "PENDING_ADMIN_REVIEW", "Profile requires medical underwriting approval.", composite_risk_index

def health_decision_logic(risk_score: float, num_conditions: int, smoker: int, age: int) -> str:
    settings = SETTINGS
    if not settings["auto_approval_enabled"]: return "PENDING_ADMIN_REVIEW"
    if risk_score >= settings["high_risk_min"]: return "REJECTED"
    if risk_score >= 60: return "REJECTED"
    if num_conditions >= 3 and risk_score >= 45: return "REJECTED"
    if age >= 60 and risk_score >= 50: return "REJECTED"
    if risk_score > settings["low_risk_max"]: return "PENDING_ADMIN_REVIEW"
    return "APPROVED"


def calculate_health_premium(risk_score, age, bmi, smoker, num_conditions, family_history, alcohol_score, exercise_frequency, coverage_amount):
    coverage = float(coverage_amount)
    base_premium = coverage * 0.0085
    if age <= 30: age_factor = 0.96
    elif age <= 40: age_factor = 1.00
    elif age <= 50: age_factor = 1.04
    elif age <= 60: age_factor = 1.08
    else: age_factor = 1.12
    if bmi < 18.5: bmi_factor = 1.01
    elif bmi <= 24.9: bmi_factor = 1.00
    elif bmi <= 29.9: bmi_factor = 1.03
    elif bmi <= 34.9: bmi_factor = 1.06
    else: bmi_factor = 1.10
    smoker_factor = 1.08 if int(smoker) == 1 else 1.00
    condition_factor = {0:1.00, 1:1.03, 2:1.06, 3:1.09}.get(min(int(num_conditions),3),1.12)
    family_factor = 1.02 if int(family_history) == 1 else 1.00
    alcohol_factor = {0:1.00, 1:1.00, 2:1.03, 3:1.06}.get(int(alcohol_score),1.00)
    exercise_factor = {0:1.03, 1:1.02, 2:1.01, 3:1.00, 4:0.99, 5:0.98, 6:0.97, 7:0.96}.get(max(0,min(int(exercise_frequency),7)),1.00)
    adjusted = base_premium * age_factor * bmi_factor * smoker_factor * condition_factor * family_factor * alcohol_factor * exercise_factor
    return round(max(coverage * 0.0075, min(adjusted, coverage * 0.0100)), 2)


def build_shap_rationale(feature_values, shap_row, base_value_pct, risk_score, status):
    contributions = sorted(zip(FEATURE_ORDER, shap_row), key=lambda pair: abs(pair[1]), reverse=True)
    lines = []
    for feat, shap_val in contributions[:5]:
        label = FEATURE_DESCRIPTIONS[feat](feature_values[feat])
        direction = "increased" if shap_val > 0 else "decreased"
        pts = abs(shap_val) * 100
        lines.append(f"  • {label} {direction} the risk score by ~{pts:.1f} points.")
    footer_map = {
        "APPROVED": "Profile qualifies for instant Niva Bupa policy issuance.",
        "REJECTED": "Application declined due to financial or clinical underwriting constraints.",
        "PENDING_ADMIN_REVIEW": "Referred to manual underwriting due to senior age bracket or medical history flags.",
    }
    body = "\n".join(lines)
    return f"Niva Bupa XAI Risk Assessment — calculated risk score: {risk_score:.1f}% (model baseline: {base_value_pct:.1f}%). Decision: {status.replace('_',' ')}.\n\nKey contributing factors (SHAP AI):\n{body}\n\n{footer_map.get(status,'')}"


def clean_conditions(raw: list[str]) -> list[str]:
    values = [str(x).strip() for x in (raw or []) if str(x).strip()]
    if "None" in values:
        return [x for x in values if x != "None"] if len(values) > 1 else []
    return values



class ChangePasswordIn(BaseModel):
    current_password: str
    new_password: str

@app.post("/api/admin/password")
def admin_password(payload: ChangePasswordIn, user=Depends(require_role("admin"))):
    row = db_run("SELECT * FROM users WHERE username=%s AND role='admin'", (user["sub"],), fetchone=True)
    if not row or row["password_hash"] != hash_password(payload.current_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect.")
    p = payload.new_password
    if not (len(p) >= 8 and any(c.isupper() for c in p) and any(c.isdigit() for c in p) and any(not c.isalnum() for c in p)):
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters long, contain one capital letter, one number, and one special character.")
    db_run("UPDATE users SET password_hash=%s WHERE username=%s", (hash_password(p), user["sub"]))
    audit(user["sub"], "PASSWORD_CHANGED", "Admin password updated.")
    return {"ok": True}

@app.get("/api/app/config")
def public_config():
    return {
        "policy_types": POLICY_TYPES,
        "coverage_tiers": COVERAGE_TIERS,
        "income_brackets": [{"label": l, "value": v} for l, v in INCOME_BRACKETS],
        "condition_options": CONDITION_OPTIONS,
        "alcohol_options": list(ALCOHOL_MAP.keys()),
        "settings": {"theme": SETTINGS.get("theme", "System")},
    }

@app.post("/api/app/drafts")
def save_draft(payload: DraftIn, user=Depends(require_role("applicant"))):
    db_run("INSERT INTO application_drafts (applicant_username,draft_data) VALUES (%s,%s) ON DUPLICATE KEY UPDATE draft_data=VALUES(draft_data)", (user["sub"], json.dumps(payload.draft)))
    return {"ok": True}

@app.get("/api/app/drafts")
def get_draft(user=Depends(require_role("applicant"))):
    row = db_run("SELECT draft_data FROM application_drafts WHERE applicant_username=%s", (user["sub"],), fetchone=True)
    return {"draft": json.loads(row["draft_data"]) if row and row.get("draft_data") else None}

@app.delete("/api/app/drafts")
def delete_draft(user=Depends(require_role("applicant"))):
    db_run("DELETE FROM application_drafts WHERE applicant_username=%s", (user["sub"],))
    return {"ok": True}

@app.post("/api/applications")
def submit_application(payload: ApplicationIn, user=Depends(require_role("applicant"))):
    personal, health, policy = payload.personal, payload.health, payload.policy
    if policy.get("policy_type") != "Health Insurance":
        raise HTTPException(status_code=400, detail="The AI underwriting engine currently supports Health Insurance only.")
    conditions = clean_conditions(health.get("conditions_final", []))
    if "Other" in conditions:
        other = str(health.get("other_condition_text", "")).strip()
        if not other:
            raise HTTPException(status_code=400, detail="Please specify the 'Other' pre-existing condition.")
        conditions = [f"Other: {other}" if c == "Other" else c for c in conditions]
    height_m = float(personal["height_cm"]) / 100.0
    bmi = float(personal["weight_kg"]) / (height_m ** 2)
    num_conditions = len(conditions)
    alcohol_value = health.get("alcohol_consumption", "None") if health.get("alcohol_consumption") in ALCOHOL_MAP else "None"
    exercise = int(health.get("exercise_frequency", 0))
    family = 1 if health.get("family_history") == "Yes" else 0
    smoker = 1 if health.get("smoker") == "Yes" else 0
    features = {"age": float(personal["age"]), "bmi": float(bmi), "smoker": float(smoker), "pre_existing_conditions": float(num_conditions), "family_history": float(family), "alcohol_score": float(ALCOHOL_MAP[alcohol_value]), "exercise_frequency": float(exercise)}
    risk_score, shap_row, base_value = predict_risk(features)
    status_ = health_decision_logic(risk_score, num_conditions, smoker, int(personal["age"]))
    rationale = build_shap_rationale(features, shap_row, base_value, risk_score, status_)
    premium = calculate_health_premium(risk_score, int(personal["age"]), bmi, smoker, num_conditions, family, ALCOHOL_MAP[alcohol_value], exercise, float(policy["coverage_amount"])) if status_ == "APPROVED" else None
    ref = f"POL-{datetime.now().strftime('%Y%m')}-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    data = {
        "policy_reference": ref, "applicant_username": user["sub"], "full_name": personal["full_name"], "age": int(personal["age"]), "gender": personal["gender"],
        "height_cm": personal["height_cm"], "weight_kg": personal["weight_kg"], "bmi": round(bmi,2), "smoker": smoker, "alcohol_consumption": alcohol_value,
        "exercise_frequency": exercise, "pre_existing_conditions": num_conditions, "pre_existing_conditions_detail": json.dumps(conditions), "family_history": family,
        "annual_income": policy["annual_income"], "policy_type": policy["policy_type"], "coverage_amount": policy["coverage_amount"], "risk_score": risk_score,
        "premium_inr": premium, "status": status_, "shap_rationale": rationale, "underwriter_notes": rationale, "reviewed_by": None,
    }
    db_run("""INSERT INTO applications
        (policy_reference, applicant_username, full_name, age, gender, height_cm, weight_kg, bmi, smoker, alcohol_consumption, exercise_frequency,
         pre_existing_conditions, pre_existing_conditions_detail, family_history, annual_income, policy_type, coverage_amount, risk_score, premium_inr,
         status, shap_rationale, underwriter_notes, reviewed_by)
        VALUES (%(policy_reference)s,%(applicant_username)s,%(full_name)s,%(age)s,%(gender)s,%(height_cm)s,%(weight_kg)s,%(bmi)s,%(smoker)s,%(alcohol_consumption)s,%(exercise_frequency)s,
        %(pre_existing_conditions)s,%(pre_existing_conditions_detail)s,%(family_history)s,%(annual_income)s,%(policy_type)s,%(coverage_amount)s,%(risk_score)s,%(premium_inr)s,%(status)s,%(shap_rationale)s,%(underwriter_notes)s,%(reviewed_by)s)""", data)
    db_run("DELETE FROM application_drafts WHERE applicant_username=%s", (user["sub"],))
    audit(user["sub"], "APPLICATION_SUBMITTED", f"Policy {ref} submitted.")
    return {"application": serialize_row(db_run("SELECT * FROM applications WHERE policy_reference=%s", (ref,), fetchone=True))}

@app.get("/api/applications/mine")
def my_applications(user=Depends(require_role("applicant"))):
    rows = db_run("SELECT * FROM applications WHERE applicant_username=%s ORDER BY created_at DESC", (user["sub"],), fetch=True)
    return {"applications": [serialize_row(r) for r in rows]}

@app.get("/api/applications/track/{reference}")
def track_application(reference: str):
    row = db_run("SELECT * FROM applications WHERE policy_reference=%s", (reference.strip(),), fetchone=True)
    if not row:
        raise HTTPException(status_code=404, detail="No application found for that reference number.")
    return {"application": serialize_row(row)}

@app.get("/api/admin/dashboard")
def admin_dashboard(user=Depends(require_role("admin"))):
    rows = db_run("SELECT * FROM applications ORDER BY created_at DESC", fetch=True)
    total = len(rows)
    summary = {
        "total": total,
        "approved": sum(1 for r in rows if r["status"] == "APPROVED"),
        "rejected": sum(1 for r in rows if r["status"] == "REJECTED"),
        "pending": sum(1 for r in rows if r["status"] == "PENDING_ADMIN_REVIEW"),
        "audit": sum(1 for r in rows if r["status"] == "MEDICAL_AUDIT_REQUIRED"),
    }
    recent = [serialize_row(r) for r in rows[:10]]
    return {"summary": summary, "recent": recent}

@app.get("/api/admin/applications")
def admin_applications(search: str = "", status_filter: str = "", policy_filter: str = "", page: int = 1, page_size: int = 10, user=Depends(require_role("admin"))):
    rows = db_run("SELECT * FROM applications ORDER BY created_at DESC", fetch=True)
    f = rows
    if search.strip():
        s = search.strip().lower(); f = [r for r in f if s in (r["policy_reference"] or "").lower() or s in (r["full_name"] or "").lower()]
    if status_filter:
        rev = {v:k for k,v in STATUS_LABEL.items()}; f = [r for r in f if r["status"] == rev.get(status_filter, status_filter)]
    if policy_filter: f = [r for r in f if r["policy_type"] == policy_filter]
    total_pages = max(1, (len(f)+page_size-1)//page_size)
    page = max(1, min(page, total_pages))
    start=(page-1)*page_size
    return {"applications":[serialize_row(r) for r in f[start:start+page_size]], "page":page, "total_pages":total_pages, "total":len(f)}

@app.get("/api/admin/applications/{app_id}")
def admin_application_detail(app_id: int, user=Depends(require_role("admin"))):
    row = db_run("SELECT * FROM applications WHERE id=%s", (app_id,), fetchone=True)
    if not row: raise HTTPException(status_code=404, detail="Application not found.")
    return {"application": serialize_row(row)}

@app.post("/api/admin/applications/{app_id}/status")
def admin_update_status(app_id: int, payload: StatusUpdateIn, user=Depends(require_role("admin"))):
    row = db_run("SELECT * FROM applications WHERE id=%s", (app_id,), fetchone=True)
    if not row: raise HTTPException(status_code=404, detail="Application not found.")
    notes = payload.notes
    premium = None
    if payload.status == "APPROVED":
        detail = json.loads(row["pre_existing_conditions_detail"] or "[]") if row.get("pre_existing_conditions_detail") else None
        num_conditions = len([c for c in (detail or []) if str(c).strip().lower() != "none"]) if detail is not None else int(row.get("pre_existing_conditions") or 0)
        alcohol = row.get("alcohol_consumption") if row.get("alcohol_consumption") in ALCOHOL_MAP else "None"
        premium = calculate_health_premium(
            float(row.get("risk_score") or 0), int(row.get("age") or 0), float(row.get("bmi") or 0), int(row.get("smoker") or 0), num_conditions,
            int(row.get("family_history") or 0), ALCOHOL_MAP[alcohol], int(row.get("exercise_frequency") or 0), float(row.get("coverage_amount") or 0)
        )
    db_run("UPDATE applications SET status=%s, underwriter_notes=%s, reviewed_by=%s, premium_inr=%s WHERE id=%s", (payload.status, notes, user["sub"], premium, app_id))
    audit(user["sub"], f"APPLICATION_{payload.status}", f"{row['policy_reference']} updated.")
    updated = db_run("SELECT * FROM applications WHERE id=%s", (app_id,), fetchone=True)
    return {"application": serialize_row(updated)}

@app.post("/api/admin/applications/{app_id}/notes")
def admin_update_notes(app_id: int, payload: NotesUpdateIn, user=Depends(require_role("admin"))):
    row = db_run("SELECT * FROM applications WHERE id=%s", (app_id,), fetchone=True)
    if not row:
        raise HTTPException(status_code=404, detail="Application not found.")
    db_run("UPDATE applications SET underwriter_notes=%s WHERE id=%s", (payload.notes, app_id))
    audit(user["sub"], "APPLICATION_NOTES_UPDATED", f"{row['policy_reference']} notes updated.")
    updated = db_run("SELECT * FROM applications WHERE id=%s", (app_id,), fetchone=True)
    return {"application": serialize_row(updated)}

@app.get("/api/admin/audit-logs")
def audit_logs(user=Depends(require_role("admin"))):
    rows = db_run("SELECT * FROM audit_log ORDER BY created_at DESC LIMIT 100", fetch=True)
    return {"logs": [serialize_row(r) for r in rows]}

@app.get("/api/admin/settings")
def get_settings(user=Depends(require_role("admin"))):
    return {"settings": SETTINGS}

@app.post("/api/admin/settings")
def save_settings(payload: SettingsIn, user=Depends(require_role("admin"))):
    for key, val in payload.settings.items():
        if key in SETTINGS_DEFAULTS:
            SETTINGS[key] = val
            db_run("INSERT INTO system_settings (setting_key,setting_value) VALUES (%s,%s) ON DUPLICATE KEY UPDATE setting_value=VALUES(setting_value)", (key, "1" if val is True else "0" if val is False else str(val)))
    return {"settings": SETTINGS}

@app.post("/api/admin/db/test")
def test_db(payload: DBConfigIn, user=Depends(require_role("admin"))):
    old = dict(DB_CONFIG); DB_CONFIG.update(payload.model_dump())
    try:
        conn = get_conn(False); conn.close(); ok=True; err=None
    except Exception as exc:
        ok=False; err=str(exc)
    DB_CONFIG.update(old)
    return {"ok":ok,"error":err}

@app.post("/api/admin/db/save")
def save_db(payload: DBConfigIn, user=Depends(require_role("admin"))):
    DB_CONFIG.update(payload.model_dump())
    ok, err = init_db()
    if ok: audit(user["sub"], "DB_CONFIG_UPDATED", f"host={payload.host}, db={payload.database}")
    return {"ok":ok,"error":err}

@app.get("/api/admin/export.csv")
def export_csv(user=Depends(require_role("admin"))):
    rows = db_run("SELECT * FROM applications ORDER BY created_at DESC", fetch=True)
    output = io.StringIO()
    if rows:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows: writer.writerow({k: serialize(v) for k,v in r.items()})
    return Response(content=output.getvalue(), media_type="text/csv", headers={"Content-Disposition":"attachment; filename=insurepulse_applications_export.csv"})
