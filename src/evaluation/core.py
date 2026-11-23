"""Metrik klinis, aturan threshold, dan kalibrasi (NumPy/SciPy saja, tanpa scikit-learn).

AUROC dan AUPRC memakai fungsi yang sudah diuji di src/model/metrics.py (Orang 2).

Aturan proyek (PROJECT_BRIEF 11.3 dan 11.5):
- Threshold utama (rule-out): threshold TERBESAR di fold 18 yang memberi sensitivitas >= 0,90.
- Threshold sekunder (rule-in): threshold TERKECIL di fold 18 yang memberi spesifisitas >= 0,90
  (terkecil = sensitivitas setinggi mungkin dengan syarat spesifisitas terpenuhi).
- Prediksi positif bila y_prob >= threshold.
- Kalibrasi (Platt) dan threshold dipilih HANYA di fold 18, lalu dikunci untuk fold 19.
Akurasi sengaja tidak dihitung (data sangat tidak seimbang).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.special import logit as _logit

from src.model.metrics import pr_auc, roc_auc

EPS = 1e-6


def _arrays(y_true, y_prob) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    p = np.asarray(y_prob, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError("y_true dan y_prob harus 1D, tidak kosong, dan sama panjang.")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y_true harus 0/1.")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("y_prob harus probabilitas 0-1 yang hingga.")
    return y.astype(np.int64), p


def safe_logit(p) -> np.ndarray:
    return _logit(np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS))


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


# ---------------------------------------------------------------------------
# Metrik
# ---------------------------------------------------------------------------
def brier(y_true, y_prob) -> float:
    y, p = _arrays(y_true, y_prob)
    return float(np.mean((p - y) ** 2))


def binary_metrics(y_true, y_prob, threshold: float) -> dict:
    """Metrik bergantung threshold. Positif bila y_prob >= threshold."""
    y, p = _arrays(y_true, y_prob)
    pred = p >= threshold
    tp = int(((y == 1) & pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    tn = int(((y == 0) & ~pred).sum())
    sens, ppv = _ratio(tp, tp + fn), _ratio(tp, tp + fp)
    f1 = float("nan") if (np.isnan(sens) or np.isnan(ppv) or sens + ppv == 0) else 2 * sens * ppv / (sens + ppv)
    return {"threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": sens, "specificity": _ratio(tn, tn + fp), "ppv": ppv,
            "npv": _ratio(tn, tn + fn), "f1": f1, "n_pred_pos": int(pred.sum())}


def _count_needed(target: float, n: int) -> int:
    # round(...) mencegah galat floating point (mis. 0.9*70 = 63.00000000000001 -> 64)
    return max(1, min(n, math.ceil(round(target * n, 9))))


def threshold_for_sensitivity(y_true, y_prob, target: float = 0.90) -> float:
    """Threshold TERBESAR dengan sensitivitas >= target (rule-out)."""
    y, p = _arrays(y_true, y_prob)
    pos = np.sort(p[y == 1])[::-1]
    if len(pos) == 0:
        raise ValueError("Tidak ada kasus positif.")
    return float(pos[_count_needed(target, len(pos)) - 1])


def threshold_for_specificity(y_true, y_prob, target: float = 0.90) -> float:
    """Threshold TERKECIL dengan spesifisitas >= target (rule-in)."""
    y, p = _arrays(y_true, y_prob)
    neg = np.sort(p[y == 0])
    if len(neg) == 0:
        raise ValueError("Tidak ada kasus negatif.")
    q = neg[_count_needed(target, len(neg)) - 1]
    return float(np.nextafter(q, np.inf))


# ---------------------------------------------------------------------------
# Kalibrasi
# ---------------------------------------------------------------------------
def _fit_logistic_on_logit(y: np.ndarray, x: np.ndarray, fix_slope: bool = False) -> tuple[float, float]:
    """Regresi logistik y ~ a + b*x (atau a + 1*x bila fix_slope) dengan log-loss rata-rata."""
    def unpack(w):
        return (w[0], 1.0) if fix_slope else (w[0], w[1])

    def nll(w):
        a, b = unpack(w)
        z = a + b * x
        return float(np.mean(np.logaddexp(0.0, z) - y * z))

    def grad(w):
        a, b = unpack(w)
        r = expit(a + b * x) - y
        g = [r.mean()] if fix_slope else [r.mean(), (r * x).mean()]
        return np.asarray(g)

    x0 = np.array([0.0] if fix_slope else [0.0, 1.0])
    res = minimize(nll, x0, jac=grad, method="BFGS", options={"gtol": 1e-8, "maxiter": 1000})
    if not res.success and res.status != 2:      # status 2 = presisi tercapai (masih valid)
        raise RuntimeError(f"Fit kalibrasi gagal: {res.message}")
    a, b = unpack(res.x)
    return float(a), float(b)


@dataclass
class PlattCalibrator:
    """p_kalibrasi = sigmoid(a + b * logit(p)). Dilatih HANYA di fold 18."""
    a: float
    b: float

    @classmethod
    def fit(cls, y_true, y_prob) -> "PlattCalibrator":
        y, p = _arrays(y_true, y_prob)
        if y.min() == y.max():
            raise ValueError("Kalibrasi butuh kedua kelas.")
        a, b = _fit_logistic_on_logit(y.astype(float), safe_logit(p))
        if b <= 0:
            raise ValueError(f"Slope kalibrasi tidak positif (b={b:.3f}); urutan prediksi terbalik, periksa model.")
        return cls(a, b)

    def transform(self, y_prob) -> np.ndarray:
        return expit(self.a + self.b * safe_logit(y_prob))

    def to_dict(self) -> dict:
        return {"method": "platt", "a": self.a, "b": self.b}

    @classmethod
    def from_dict(cls, d: dict) -> "PlattCalibrator":
        if d.get("method") != "platt":
            raise ValueError("Bukan parameter Platt.")
        return cls(float(d["a"]), float(d["b"]))


def calibration_intercept_slope(y_true, y_prob) -> dict:
    """Calibration-in-the-large (intercept dengan slope tetap 1) dan calibration slope.

    Ideal: intercept ~ 0 dan slope ~ 1. Intercept negatif = model terlalu tinggi (overestimate).
    """
    y, p = _arrays(y_true, y_prob)
    x = safe_logit(p)
    citl, _ = _fit_logistic_on_logit(y.astype(float), x, fix_slope=True)
    _, slope = _fit_logistic_on_logit(y.astype(float), x)
    return {"calibration_intercept": citl, "calibration_slope": slope}


def calibration_table(y_true, y_prob, n_bins: int = 10) -> list[dict]:
    """Bin berisi jumlah kunjungan yang (hampir) sama, diurutkan menurut y_prob."""
    y, p = _arrays(y_true, y_prob)
    order = np.argsort(p, kind="mergesort")
    rows = []
    for i, idx in enumerate(np.array_split(order, n_bins), start=1):
        if len(idx) == 0:
            continue
        rows.append({"bin": i, "n": int(len(idx)), "p_min": float(p[idx].min()), "p_max": float(p[idx].max()),
                     "mean_pred": float(p[idx].mean()), "observed": float(y[idx].mean())})
    return rows


def summarize(y_true, y_prob) -> dict:
    """Metrik yang tidak bergantung threshold."""
    y, p = _arrays(y_true, y_prob)
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "prevalence": float(y.mean()),
           "auroc": roc_auc(y, p), "auprc": pr_auc(y, p), "brier": brier(y, p)}
    out.update(calibration_intercept_slope(y, p))
    return out
