"""Uji faithfulness penjelasan M1: apakah wilayah/lead yang ditandai XAI benar-benar penting bagi logit?

Fungsi murni di atas tensor (tidak membaca data pasien, tidak menulis berkas).
- Waktu: bagi sinyal menjadi 20 blok (0,5 detik); nolkan k blok dengan Grad-CAM tertinggi vs k blok acak vs k blok terendah.
- Lead : nolkan k lead dengan skor oklusi tertinggi vs k lead acak vs k lead terendah.
Nol = rata-rata training setelah standardisasi. Keluaran per rekaman: penurunan logit = logit asli - logit setelah dinolkan.
Penjelasan dikatakan faithful bila penurunan 'tinggi' > 'acak' > 'rendah' secara konsisten.
"""
from __future__ import annotations

import numpy as np
import torch

N_BLOCKS = 20


@torch.no_grad()
def _logit(model, x: torch.Tensor) -> np.ndarray:
    was = model.training
    model.eval()
    try:
        return model(x)[:, 0].cpu().numpy()
    finally:
        model.train(was)


def _zero_blocks(x: torch.Tensor, idx: np.ndarray) -> torch.Tensor:
    """idx: (B, k) indeks blok yang dinolkan pada semua lead."""
    B, _, T = x.shape
    L = T // N_BLOCKS
    xm = x.clone()
    for b in range(B):
        for j in idx[b]:
            xm[b, :, j * L:(j + 1) * L] = 0.0
    return xm


def _zero_leads(x: torch.Tensor, idx: np.ndarray) -> torch.Tensor:
    xm = x.clone()
    for b in range(x.shape[0]):
        xm[b, idx[b], :] = 0.0
    return xm


def _random_idx(rng, n: int, B: int, k: int) -> np.ndarray:
    return np.stack([rng.choice(n, size=k, replace=False) for _ in range(B)])


def _drops(model, x, base, idx_top, idx_bottom, zero_fn, n_items, k, n_random, seed):
    rng = np.random.default_rng(seed)
    B = x.shape[0]
    rand = np.mean([base - _logit(model, zero_fn(x, _random_idx(rng, n_items, B, k)))
                    for _ in range(n_random)], axis=0)
    return {"top": base - _logit(model, zero_fn(x, idx_top)), "random": rand,
            "bottom": base - _logit(model, zero_fn(x, idx_bottom))}


def time_faithfulness(model, x: torch.Tensor, cam: np.ndarray, k: int = 4, n_random: int = 5, seed: int = 0) -> dict:
    """cam: (B, T) dari grad_cam_1d. Mengembalikan {'top','random','bottom'} masing-masing (B,)."""
    score = cam.reshape(cam.shape[0], N_BLOCKS, -1).mean(2)
    order = np.argsort(score, axis=1)
    base = _logit(model, x)
    return _drops(model, x, base, order[:, -k:], order[:, :k], _zero_blocks, N_BLOCKS, k, n_random, seed)


def lead_faithfulness(model, x: torch.Tensor, occ: np.ndarray, k: int = 3, n_random: int = 5, seed: int = 0) -> dict:
    """occ: (B, 12) dari lead_occlusion (lebih besar = lebih mendukung logit)."""
    order = np.argsort(occ, axis=1)
    base = _logit(model, x)
    return _drops(model, x, base, order[:, -k:], order[:, :k], _zero_leads, x.shape[1], k, n_random, seed)


def mean_ci(v: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Rata-rata dan selang 95% bootstrap."""
    rng = np.random.default_rng(seed)
    v = np.asarray(v, dtype=float)
    boots = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(1)
    return float(v.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))