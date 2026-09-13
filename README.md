# InsurePulse XAI — React + FastAPI migration

This project migrates the uploaded Streamlit application to a browser-native React frontend with a FastAPI backend while keeping the existing MySQL data model and underwriting/ML logic.

## What was preserved
- Applicant registration/login and admin login.
- XGBoost + SHAP risk calculation and the existing synthetic training formula.
- Health underwriting decision thresholds.
- Health premium calculation and the ₹75,000–₹1,00,000 band for ₹1 crore coverage.
- None/condition handling, including Other condition text.
- Draft saving.
- Four-step application flow: Personal → Health & Lifestyle → Policy → Review.
- Applicant result, My Applications, and Track Policy Status.
- Admin dashboard, applications filtering, pagination, application detail, approval/rejection/medical audit.
- Approval-time premium calculation for manually reviewed applications.
- Settings, DB test/save, password change, notifications, theme, audit log and CSV export.

## Fixes requested
1. **Dark mode:** the frontend uses CSS variables and reacts to `Light`, `System`, or `Dark`; inputs, cards and fields switch with the theme.
2. **Refresh:** authentication is stored in browser localStorage using a signed backend token, so refreshing the page does not require logging in again.
3. **Browser back/forward:** React Router manages real browser history; going back changes route state instead of logging the user out. Authentication is independent of route history.

## Run backend
```bash
cd backend
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env
# edit .env with the MySQL + Gmail settings
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Run frontend
```bash
cd frontend
npm install
npm run dev
```

The frontend defaults to `http://localhost:5173` and the API to `http://localhost:8000/api`.
Set `VITE_API_BASE` in `frontend/.env` if the API is hosted elsewhere.

## Existing Streamlit code
`streamlit_reference_app8_7.py` is included untouched as the original uploaded reference.

## Deployment
For a real deployment, host the React build as a static frontend and the FastAPI app as a separate backend. Configure the backend environment variables and point `VITE_API_BASE` to the public API URL.
