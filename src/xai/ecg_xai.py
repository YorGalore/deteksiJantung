"""XAI untuk M1 (ResNet-1D): Grad-CAM 1D dan oklusi per-lead.

Hanya fungsi murni di atas tensor; tidak membaca data pasien dan tidak menulis berkas.
- grad_cam_1d: peta kepentingan temporal (panjang = input) dari keluaran layer4, kelas positif (logit).
- lead_occlusion: penurunan logit saat satu lead diganti 0 (= rata-rata training setelah standardisasi).
Model harus dalam mode eval; fungsi memanggil model.eval() dan mengembalikannya ke mode semula.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def grad_cam_1d(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    """x: (B, 12, T) -> (B, T), dinormalisasi 0..1 per rekaman. Target: logit tunggal (kelas AMI)."""
    was_training = model.training
    model.eval()
    store: dict[str, torch.Tensor] = {}

    def fwd_hook(_m, _i, out):
        out.retain_grad()
        store["act"] = out

    handle = model.layer4.register_forward_hook(fwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        logit = model(x.clone().requires_grad_(False))
        logit.sum().backward()
        act, grad = store["act"], store["act"].grad
        weights = grad.mean(dim=2, keepdim=True)                    # (B, C, 1)
        cam = F.relu((weights * act).sum(dim=1, keepdim=True))      # (B, 1, L)
        cam = F.interpolate(cam, size=x.shape[-1], mode="linear", align_corners=False)[:, 0]
        cam = cam.detach().cpu().numpy()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
        model.train(was_training)
    lo = cam.min(axis=1, keepdims=True)
    hi = cam.max(axis=1, keepdims=True)
    return np.where(hi - lo > 1e-12, (cam - lo) / np.maximum(hi - lo, 1e-12), 0.0)


@torch.no_grad()
def lead_occlusion(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    """x: (B, 12, T) -> (B, 12): logit asli - logit tanpa lead i (positif = lead mendukung prediksi AMI)."""
    was_training = model.training
    model.eval()
    try:
        base = model(x)[:, 0]
        out = []
        for i in range(x.shape[1]):
            xo = x.clone()
            xo[:, i, :] = 0.0
            out.append((base - model(xo)[:, 0]).cpu().numpy())
    finally:
        model.train(was_training)
    return np.stack(out, axis=1)
