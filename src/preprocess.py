"""
preprocess.py — Siapkan gambar upload user supaya persis sama seperti
format yang dipakai saat training (CLAHE + resize + normalize).
JANGAN pakai augmentasi random di sini -- ini bukan training, ini inferensi.
"""
import cv2
import numpy as np
import torch
from PIL import Image
import io

from config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, DEVICE

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


def get_inference_transform():
    """Transform SAMA PERSIS seperti get_val_transforms() saat training."""
    if HAS_ALBUMENTATIONS:
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE, interpolation=cv2.INTER_CUBIC),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        raise ImportError("albumentations wajib untuk hasil yang konsisten dengan training")


def validate_image_bytes(file_bytes: bytes) -> bool:
    """Cek apakah file yang diupload beneran gambar valid (bukan file rusak/bukan gambar)."""
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()
        return True
    except Exception:
        return False


def bytes_to_tensor(file_bytes: bytes) -> torch.Tensor:
    """
    Konversi bytes gambar (dari upload FastAPI) -> tensor siap masuk model.
    Return shape: [1, 3, IMG_SIZE, IMG_SIZE] (sudah ada batch dimension).
    """
    # Baca via PIL dulu (lebih toleran macam-macam format: PNG/JPG/BMP/dll)
    pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img_array = np.array(pil_img)   # RGB, shape (H, W, 3)

    transform = get_inference_transform()
    transformed = transform(image=img_array)
    tensor = transformed["image"]                # shape (3, IMG_SIZE, IMG_SIZE)
    tensor = tensor.unsqueeze(0)                  # tambah batch dim -> (1, 3, IMG_SIZE, IMG_SIZE)
    return tensor.to(DEVICE)


def tensor_to_display_image(file_bytes: bytes) -> np.ndarray:
    """
    Untuk keperluan TAMPILAN (GradCAM overlay, dll) -- gambar di-resize
    ke IMG_SIZE tapi TIDAK dinormalisasi, biar masih enak dilihat mata manusia.
    """
    pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img_array = np.array(pil_img)
    resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    return resized