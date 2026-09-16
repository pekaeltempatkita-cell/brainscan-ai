"""
config.py — Pusat konfigurasi aplikasi.
Semua "angka ajaib" dan path ditaruh di sini, BUKAN di-hardcode di file lain.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import torch

# --- Load file .env (isinya API key, dll -- gak ikut ke-push ke GitHub) ---
load_dotenv()

# --- Path dasar proyek ---
BASE_DIR = Path(__file__).resolve().parent.parent   # folder root project (di atas src/)
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = SRC_DIR / "models"

OUTPUTS_DIR = BASE_DIR / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"        # FIX: nested di dalam outputs/, sesuai struktur asli
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "brainscan.db"            # BARU: riwayat prediksi user

TEMP_UPLOADS_DIR = BASE_DIR / "temp_uploads"

# Pastikan folder-folder ini ada (auto-buat kalau belum ada)
for folder in [CHECKPOINTS_DIR, TEMP_UPLOADS_DIR, FIGURES_DIR, REPORTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# --- Path checkpoint model (HASIL TRAINING KAMU, disalin manual dari Google Drive) ---
MAIN_MODEL_PATH = CHECKPOINTS_DIR / "hybrid_vit_efficientnet_brain_best.pth"
PRECHECK_MODEL_PATH = CHECKPOINTS_DIR / "precheck_brain_gate_best.pth"

# --- Kelas-kelas model (URUTAN HARUS SAMA PERSIS kayak pas training!) ---
MAIN_CLASSES = [
    "Alzheimer_Mild", "Alzheimer_Moderate", "Alzheimer_Very_Mild",
    "Intracranial_Hemorrhage", "Multiple_Sclerosis", "Normal_Healthy",
    "Stroke_Iskemik", "Tumor_Glioma", "Tumor_Meningioma", "Tumor_Pituitary",
]
PRECHECK_CLASSES = ["Non_Brain", "Brain"]

# --- Konfigurasi gambar (HARUS SAMA PERSIS kayak CFG.IMG_SIZE di notebook training!) ---
# Notebook training pakai IMG_SIZE=224. Kalau ini beda dengan training,
# shape positional-embedding ViT tidak akan cocok dan load_state_dict() akan ERROR.
IMG_SIZE = 300
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Threshold precheck (di bawah ini dianggap "bukan gambar otak") ---
PRECHECK_THRESHOLD = 0.5

# --- API Keys (dari .env, JANGAN ditulis langsung di sini) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# --- Device (otomatis pakai GPU kalau ada, CPU kalau enggak) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Info aplikasi ---
APP_NAME = "NeuroCheck"
APP_VERSION = "1.0.0"