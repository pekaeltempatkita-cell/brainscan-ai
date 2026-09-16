"""
explainability.py — Grad-CAM untuk model utama (HybridViTEfficientNet).
Menghasilkan gambar heatmap yang menunjukkan region citra otak yang
paling mempengaruhi keputusan model (dipakai untuk laporan & tampilan hasil).

Portingan dari class GradCAM di notebook training -- targetnya sengaja
diarahkan ke feature map CNN terakhir (bukan branch ViT) supaya heatmap-nya
tetap punya bentuk spasial (H x W) yang gampang di-overlay ke gambar asli.
"""
import uuid

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from config import FIGURES_DIR, IMAGENET_MEAN, IMAGENET_STD


class GradCAM:
    """Grad-CAM generik: hook ke satu target_layer, backward dari skor kelas target."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int):
        """input_tensor: [1, 3, H, W], sudah requires_grad tidak perlu diset manual."""
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, target_class]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)
        cam = cam.squeeze(0).cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))
        return cam

    def remove_hooks(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()


def denormalize_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Balikin tensor yang sudah dinormalisasi ImageNet -> gambar RGB uint8 biasa."""
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def overlay_heatmap(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    cam_uint8 = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 1 - alpha, heatmap, alpha, 0)


def generate_gradcam_image(model, input_tensor: torch.Tensor, target_class: int) -> str:
    """
    Jalankan Grad-CAM pada satu prediksi, simpan hasil overlay ke FIGURES_DIR,
    dan return path relatif filenya (buat disimpan ke kolom gradcam_path di DB).

    CATATAN: butuh gradient, jadi tensor input di sini TIDAK boleh dalam blok
    `with torch.no_grad()` -- panggil fungsi ini terpisah dari forward pass biasa.
    """
    target_layer = model.features[-1]  # blok konvolusi terakhir EfficientNet-B3
    cam_engine = GradCAM(model, target_layer)
    try:
        input_tensor = input_tensor.clone().requires_grad_(True)
        cam = cam_engine.generate(input_tensor, target_class)
    finally:
        cam_engine.remove_hooks()
        model.zero_grad()

    base_image = denormalize_to_uint8(input_tensor.detach())
    overlay = overlay_heatmap(base_image, cam)

    filename = f"gradcam_{uuid.uuid4().hex[:12]}.png"
    save_path = FIGURES_DIR / filename
    cv2.imwrite(str(save_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return f"figures/{filename}"  # path relatif, biar gampang di-serve lewat static route
