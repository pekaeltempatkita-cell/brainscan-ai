"""
app.py — Entry point aplikasi NeuroCheck.
Login Google (untuk user biasa) + login username/password (untuk admin).

Cara jalanin (dari dalam folder src/):
    uvicorn app:app --reload
"""
import os
import re
import html

from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import RedirectResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from dotenv import load_dotenv
from markupsafe import Markup

import inference
import gemini_client
import api
from report_generator import generate_report_pdf
from disease_info import get_disease_info
from config import FIGURES_DIR
from database import (
    init_db, create_patient, update_patient, get_patient_by_nik,
    get_all_patients, get_medical_records_by_patient,
    save_prediction, add_medical_record, get_all_medical_records,
    get_prediction_by_id,
)

load_dotenv()

app = FastAPI(title="NeuroCheck")

# Session (buat nyimpen status login user & admin di cookie, terenkripsi)
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET_KEY", "dev-secret-ganti-ini"))

templates = Jinja2Templates(directory="templates")


def _format_explanation(text):
    """Filter Jinja: escape teks penjelasan AI, ubah **bold** -> <strong>, \\n -> <br>.
    Dipakai di template lewat {{ teks | explain_fmt }} biar aman dari HTML injection
    tapi tetap enak dibaca (heading bold & paragraf kebentuk rapi)."""
    if not text:
        return ""
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = escaped.replace("\n", "<br>")
    return Markup(escaped)


templates.env.filters["explain_fmt"] = _format_explanation

app.mount("/static", StaticFiles(directory="static"), name="static")
# Folder outputs/figures dipakai buat nyimpen heatmap Grad-CAM -> perlu bisa diakses browser
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")

# Endpoint JSON tipis (lihat api.py) buat kebutuhan integrasi programatik
app.include_router(api.router, prefix="/api")

# --- Setup Google OAuth ---
oauth = OAuth()
oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

init_db()


def _model_warning():
    """Pesan peringatan buat ditampilkan di UI kalau checkpoint model belum ada."""
    if inference.MODELS_READY:
        return None
    return inference.MODEL_LOAD_ERROR


@app.get("/", response_class=HTMLResponse)
def landing(request: Request):
    user = request.session.get("user")
    return templates.TemplateResponse(request, "landing.html", {"user": user})


@app.get("/informasi", response_class=HTMLResponse)
def informasi(request: Request):
    user = request.session.get("user")
    return templates.TemplateResponse(request, "informasi.html", {"user": user})


# --- Login Google (user biasa) ---
@app.get("/auth/google/login")
async def google_login(request: Request):
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI")
    return await oauth.google.authorize_redirect(request, redirect_uri)


@app.get("/auth/google/callback")
async def google_callback(request: Request):
    token = await oauth.google.authorize_access_token(request)
    user_info = token.get("userinfo")
    request.session["user"] = {"email": user_info["email"], "name": user_info["name"]}
    return RedirectResponse(url="/dashboard")


@app.get("/auth/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/")


# --- Login admin (username/password) ---
@app.get("/admin/login", response_class=HTMLResponse)
def admin_login_page(request: Request):
    return templates.TemplateResponse(request, "admin_login.html", {"error": None})


@app.post("/admin/login")
def admin_login(request: Request, username: str = Form(...), password: str = Form(...)):
    if username == os.getenv("ADMIN_USERNAME") and password == os.getenv("ADMIN_PASSWORD"):
        request.session["admin"] = True
        return RedirectResponse(url="/admin/dashboard", status_code=303)
    return templates.TemplateResponse(request, "admin_login.html", {"error": "Username atau password salah."})


@app.get("/admin/dashboard", response_class=HTMLResponse)
def admin_dashboard(request: Request):
    if not request.session.get("admin"):
        return RedirectResponse(url="/admin/login")
    records = get_all_medical_records()
    patients = get_all_patients()
    return templates.TemplateResponse(request, "admin_dashboard.html", {
        "records": records, "patients": patients, "model_warning": _model_warning(),
    })


@app.get("/admin/logout")
def admin_logout(request: Request):
    request.session.pop("admin", None)
    return RedirectResponse(url="/")


# --- Dashboard user setelah login Google (list + registrasi pasien = "Riwayat Pasien") ---
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    user = request.session.get("user")
    if not user:
        return RedirectResponse(url="/")
    patients = get_all_patients()
    return templates.TemplateResponse(request, "dashboard.html", {
        "user": user, "patients": patients, "model_warning": _model_warning(),
    })


@app.post("/patients/register")
def register_patient(
    request: Request,
    nik: str = Form(...),
    nama: str = Form(...),
    tanggal_lahir: str = Form(...),
    jenis_kelamin: str = Form(...),
    alamat: str = Form(""),
    no_telepon: str = Form(""),
):
    user = request.session.get("user")
    if not user:
        return RedirectResponse(url="/")

    clean_nik = nik.strip()
    new_id = create_patient(clean_nik, nama, tanggal_lahir, jenis_kelamin, alamat, no_telepon, user["email"])
    if new_id is None:
        update_patient(clean_nik, nama, tanggal_lahir, jenis_kelamin, alamat, no_telepon)

    return RedirectResponse(url=f"/patients/{clean_nik}", status_code=303)


@app.get("/patients/{nik}", response_class=HTMLResponse)
def patient_detail(request: Request, nik: str):
    user = request.session.get("user")
    if not user:
        return RedirectResponse(url="/")

    patient = get_patient_by_nik(nik)
    if not patient:
        return HTMLResponse("<h2>Pasien tidak ditemukan</h2><a href='/dashboard'>Kembali</a>", status_code=404)

    records = get_medical_records_by_patient(patient["id"])

    hasil_data = None
    hasil_id = request.query_params.get("hasil")
    if hasil_id:
        pred = get_prediction_by_id(int(hasil_id))
        if pred and pred.get("precheck_status") == "valid" and pred.get("prediction_label"):
            info = get_disease_info(pred["prediction_label"])
            hasil_data = {"prediction": pred, "info": info}

    return templates.TemplateResponse(request, "patient_detail.html", {
        "user": user, "patient": patient, "records": records, "model_warning": _model_warning(),
        "hasil_data": hasil_data,
    })


@app.post("/patients/{nik}/upload")
async def upload_scan(
    request: Request, nik: str, file: UploadFile = File(...),
    jenis_scan: str = Form(""), umur_saat_scan: str = Form(""), gejala: str = Form(""),
):
    user = request.session.get("user")
    if not user:
        return RedirectResponse(url="/")

    patient = get_patient_by_nik(nik)
    if not patient:
        return HTMLResponse("<h2>Pasien tidak ditemukan</h2>", status_code=404)

    file_bytes = await file.read()
    result = inference.run_pipeline(file_bytes)

    if result["status"] == "model_not_ready":
        return RedirectResponse(url=f"/patients/{nik}?model_error=1", status_code=303)

    if result["precheck_status"] != "valid":
        prediction_id = save_prediction(
            filename=file.filename, precheck_status="invalid",
            precheck_confidence=result["precheck_confidence"],
            jenis_scan=jenis_scan, umur_saat_scan=umur_saat_scan, gejala=gejala,
        )
        return RedirectResponse(url=f"/patients/{nik}?invalid=1&conf={result['precheck_confidence']}", status_code=303)

    explanation = gemini_client.get_explanation(result["prediction_label"], result["prediction_confidence"])

    prediction_id = save_prediction(
        filename=file.filename, precheck_status="valid",
        precheck_confidence=result["precheck_confidence"],
        prediction_label=result["prediction_label"],
        prediction_confidence=result["prediction_confidence"],
        all_probabilities=result["all_probabilities"],
        gradcam_path=result.get("gradcam_path"),
        gemini_explanation=explanation,
        jenis_scan=jenis_scan, umur_saat_scan=umur_saat_scan, gejala=gejala,
    )
    add_medical_record(patient_id=patient["id"], prediction_id=prediction_id)

    return RedirectResponse(url=f"/patients/{nik}?hasil={prediction_id}", status_code=303)


def _can_access_reports(request: Request) -> bool:
    return bool(request.session.get("user") or request.session.get("admin"))


@app.get("/patients/{nik}/report/{prediction_id}")
def download_report(request: Request, nik: str, prediction_id: int):
    """Generate (atau pakai cache) PDF laporan hasil pemeriksaan, lalu stream ke browser."""
    if not _can_access_reports(request):
        return RedirectResponse(url="/")

    patient = get_patient_by_nik(nik)
    if not patient:
        return HTMLResponse("<h2>Pasien tidak ditemukan</h2>", status_code=404)

    prediction = get_prediction_by_id(prediction_id)
    if not prediction:
        return HTMLResponse("<h2>Data pemeriksaan tidak ditemukan</h2>", status_code=404)

    pdf_path = generate_report_pdf(patient, prediction)
    filename = f"Laporan_{patient['nama'].replace(' ', '_')}_{prediction_id}.pdf"
    return FileResponse(str(pdf_path), media_type="application/pdf", filename=filename)