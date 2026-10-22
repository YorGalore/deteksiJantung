"""Metrik evaluasi klasifikasi biner (NumPy + SciPy saja, tanpa scikit-learn).

Catatan penggunaan: ambang keputusan (threshold) harus dipilih pada fold 18
(kalibrasi & threshold), BUKAN pada fold 19 (test). Fold 19 hanya dipakai satu
kali di akhir dengan ambang yang sudah terkunci.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def _validate(y_true, y_prob) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    p = np.asarray(y_prob, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p):
        raise ValueError("y_true dan y_prob harus 1D dengan panjang sama.")
    if len(y) == 0:
        raise ValueError("Input kosong.")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y_true harus berisi hanya 0 dan 1.")
    if not np.isfinite(p).all():
        raise ValueError("y_prob mengandung NaN atau infinity.")
    return y.astype(np.int64), p


def confusion_matrix(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Prediksi positif jika y_prob >= threshold."""
    y, p = _validate(y_true, y_prob)
    pred = p >= threshold
    return {
        "tn": int(((y == 0) & ~pred).sum()),
        "fp": int(((y == 0) & pred).sum()),
        "fn": int(((y == 1) & ~pred).sum()),
        "tp": int(((y == 1) & pred).sum()),
    }


def _ratio(num: int, den: int) -> float:
    return float(num / den) if den > 0 else float("nan")


def roc_auc(y_true, y_prob) -> float:
    """AUROC lewat statistik Mann-Whitney (ties dihitung rata-rata)."""
    y, p = _validate(y_true, y_prob)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(p)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pr_auc(y_true, y_prob) -> float:
    """Average precision (PR-AUC), sama dengan definisi scikit-learn."""
    y, p = _validate(y_true, y_prob)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    y, p = y[order], p[order]
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    last_of_group = np.r_[np.where(np.diff(p))[0], len(p) - 1]
    tps, fps = tps[last_of_group], fps[last_of_group]
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def evaluate(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Semua metrik sekaligus. Metrik yang tak terdefinisi bernilai NaN."""
    cm = confusion_matrix(y_true, y_prob, threshold)
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    sensitivity = _ratio(tp, tp + fn)
    precision = _ratio(tp, tp + fp)
    f1 = (float("nan") if np.isnan(sensitivity) or np.isnan(precision) or (precision + sensitivity) == 0
          else 2 * precision * sensitivity / (precision + sensitivity))
    return {
        **cm,
        "threshold": float(threshold),
        "sensitivity": sensitivity,
        "specificity": _ratio(tn, tn + fp),
        "precision": precision,
        "f1": f1,
        "roc_auc": roc_auc(y_true, y_prob),
        "pr_auc": pr_auc(y_true, y_prob),
        "n": int(len(np.asarray(y_true))),
        "n_pos": int(np.asarray(y_true).sum()),
    }
