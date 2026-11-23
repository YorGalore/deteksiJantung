"""Bootstrap berbasis pasien (cluster = subject_id), berpasangan antar-model (PROJECT_BRIEF 11.3).

Di setiap iterasi, PASIEN diambil ulang dengan pengembalian (semua kunjungan pasien itu ikut),
lalu SEMUA model dinilai pada sampel yang sama. Selisih metrik antar-model dihitung per iterasi,
sehingga CI selisih memperhitungkan korelasi antar-model. Perbedaan dianggap bermakna bila CI 95%
selisih tidak melewati 0.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

MetricFn = Callable[[np.ndarray, np.ndarray, str], float]   # (y, p, nama_model) -> nilai


def cluster_resample_indices(clusters, n_boot: int, seed: int):
    """Generator indeks baris untuk tiap iterasi bootstrap berbasis cluster (vektorisasi)."""
    codes, uniques = pd.factorize(pd.Series(clusters), sort=True)
    n_clusters = len(uniques)
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes, minlength=n_clusters)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        pick = rng.integers(0, n_clusters, n_clusters)
        c = counts[pick]
        total = int(c.sum())
        offsets = np.arange(total) - np.repeat(np.cumsum(c) - c, c)
        yield order[np.repeat(starts[pick], c) + offsets]


def paired_bootstrap(y_true, probs: dict[str, np.ndarray], clusters, metrics: dict[str, MetricFn],
                     n_boot: int = 2000, seed: int = 42, diffs: list[tuple[str, str]] = (),
                     alpha: float = 0.05) -> pd.DataFrame:
    """CI persentil untuk tiap (model, metrik) dan tiap selisih (A - B).

    Kembalian kolom: target, metric, estimate, ci_low, ci_high, n_valid.
    Iterasi tanpa kasus positif/negatif menghasilkan NaN dan tidak dihitung (lihat n_valid).
    """
    y = np.asarray(y_true).astype(np.int64)
    probs = {m: np.asarray(p, dtype=np.float64) for m, p in probs.items()}
    models = list(probs)
    for a, b in diffs:
        if a not in probs or b not in probs:
            raise ValueError(f"Selisih {a} - {b}: model tidak ada di probs.")

    point = {(m, k): f(y, probs[m], m) for m in models for k, f in metrics.items()}
    boots = {key: np.empty(n_boot) for key in point}
    for i, idx in enumerate(cluster_resample_indices(clusters, n_boot, seed)):
        yb = y[idx]
        both = 0 < yb.sum() < len(yb)
        for (m, k) in point:
            boots[(m, k)][i] = metrics[k](yb, probs[m][idx], m) if both else np.nan

    def ci(v):
        v = v[np.isfinite(v)]
        if len(v) == 0:
            return np.nan, np.nan, 0
        return float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2)), int(len(v))

    rows = []
    for (m, k), est in point.items():
        lo, hi, n = ci(boots[(m, k)])
        rows.append({"target": m, "metric": k, "estimate": est, "ci_low": lo, "ci_high": hi, "n_valid": n})
    for a, b in diffs:
        for k in metrics:
            lo, hi, n = ci(boots[(a, k)] - boots[(b, k)])
            rows.append({"target": f"{a} - {b}", "metric": k, "estimate": point[(a, k)] - point[(b, k)],
                         "ci_low": lo, "ci_high": hi, "n_valid": n})
    out = pd.DataFrame(rows)
    is_diff = out["target"].str.contains(" - ", regex=False)
    out["ci_excludes_0"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    out.loc[is_diff, "ci_excludes_0"] = ((out["ci_low"] > 0) | (out["ci_high"] < 0))[is_diff]
    return out
