"""
inference.py — Muat model precheck & model utama SEKALI saat startup,
sediakan fungsi run_pipeline() buat dipanggil dari route upload.

PENTING: kalau file checkpoint (.pth) belum ada di outputs/checkpoints/,
aplikasi TETAP BISA JALAN (biar UI/DB/PDF report bisa dites duluan), tapi
run_pipeline() akan return status "model_not_ready" -- bukan crash total.
"""
import time
import uuid

import torch
import torch.nn.functional as F

from config import (
    MAIN_MODEL_PATH, PRECHECK_MODEL_PATH,
    MAIN_CLASSES, PRECHECK_CLASSES, PRECHECK_THRESHOLD, DEVICE, FIGURES_DIR,
)
from models.precheck_model import PrecheckModel
from models.classifier_model import HybridViTEfficientNet
from preprocess import bytes_to_tensor, tensor_to_display_image
import explainability

precheck_model = None
main_model = None
MODELS_READY = False
MODEL_LOAD_ERROR = None


def _load_model(model, path):
    checkpoint = torch.load(path, map_location=DEVICE)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def load_models():
    """Coba load kedua model. Aman dipanggil ulang (idempotent-ish)."""
    global precheck_model, main_model, MODELS_READY, MODEL_LOAD_ERROR
    try:
        print("Memuat model precheck ...")
        precheck_model = _load_model(PrecheckModel(num_classes=len(PRECHECK_CLASSES)), PRECHECK_MODEL_PATH)
        print("Memuat model utama (ini bisa agak lama di CPU) ...")
        main_model = _load_model(HybridViTEfficientNet(num_classes=len(MAIN_CLASSES)), MAIN_MODEL_PATH)
        MODELS_READY = True
        MODEL_LOAD_ERROR = None
        print("Semua model siap dipakai.")
    except FileNotFoundError as e:
        MODELS_READY = False
        MODEL_LOAD_ERROR = (
            f"File checkpoint model tidak ditemukan ({e.filename}). "
            f"Taruh file .pth hasil training kamu di folder outputs/checkpoints/ "
            f"dengan nama persis seperti di config.py, lalu restart server."
        )
        print(f"[PERINGATAN] {MODEL_LOAD_ERROR}")
    except Exception as e:
        MODELS_READY = False
        MODEL_LOAD_ERROR = f"Gagal memuat model: {e}"
        print(f"[PERINGATAN] {MODEL_LOAD_ERROR}")


# Coba load pas modul ini di-import pertama kali (saat startup app).
load_models()


def run_pipeline(file_bytes: bytes, generate_heatmap: bool = True) -> dict:
    if not MODELS_READY:
        return {
            "status": "model_not_ready",
            "error": MODEL_LOAD_ERROR or "Model belum dimuat.",
            "precheck_status": None, "precheck_confidence": None,
            "prediction_label": None, "prediction_confidence": None,
            "all_probabilities": None, "gradcam_path": None,
        }

    x = bytes_to_tensor(file_bytes)

    with torch.no_grad():
        brain_prob = F.softmax(precheck_model(x), dim=1)[0, PRECHECK_CLASSES.index("Brain")].item()

    if brain_prob < PRECHECK_THRESHOLD:
        return {
            "status": "ok",
            "precheck_status": "invalid", "precheck_confidence": round(brain_prob, 4),
            "prediction_label": None, "prediction_confidence": None,
            "all_probabilities": None, "gradcam_path": None,
        }

    with torch.no_grad():
        probs = F.softmax(main_model(x), dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())

    gradcam_relative_path = None
    if generate_heatmap:
        try:
            target_layer = main_model.features[-1]   # blok conv terakhir EfficientNet-B3
            original_rgb = tensor_to_display_image(file_bytes)
            filename = f"gradcam_{uuid.uuid4().hex[:10]}_{int(time.time())}.png"
            save_path = FIGURES_DIR / filename
            ok = explainability.generate_gradcam_overlay(
                main_model, x, original_rgb, class_idx=idx,
                target_layer=target_layer, save_path=save_path,
            )
            if ok:
                gradcam_relative_path = filename   # disimpan relatif, gampang di-mount ke URL
        except Exception as e:
            print(f"[GradCAM skip] {e}")

    return {
        "status": "ok",
        "precheck_status": "valid", "precheck_confidence": round(brain_prob, 4),
        "prediction_label": MAIN_CLASSES[idx], "prediction_confidence": round(float(probs[idx]), 4),
        "all_probabilities": {cls: round(float(p), 4) for cls, p in zip(MAIN_CLASSES, probs)},
        "gradcam_path": gradcam_relative_path,
    }