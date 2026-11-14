"""Membaca dan memvalidasi berkas prediksi sesuai kontrak PROJECT_BRIEF 12.1.

predictions/{KODE_MODEL}_fold{N}.parquet
kolom: ed_stay_id, subject_id, study_id, fold, y_true, y_prob

Fold 19 (test) dikunci: hanya bisa dibaca dengan allow_test=True secara eksplisit,
dan itu hanya boleh dilakukan setelah rencana analisis disepakati (brief 11.5).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import FOLD_TEST

CONTRACT = ["ed_stay_id", "subject_id", "study_id", "fold", "y_true", "y_prob"]
ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = ROOT / "predictions"


def pred_path(model: str, fold: int, pred_dir: Path = PRED_DIR) -> Path:
    return Path(pred_dir) / f"{model}_fold{fold}.parquet"


def validate_predictions(df: pd.DataFrame, expected_fold: int | None = None, name: str = "") -> pd.DataFrame:
    """Periksa kontrak; kembalikan salinan dengan tipe yang rapi. Error yang jelas bila melanggar."""
    tag = f"[{name}] " if name else ""
    missing = [c for c in CONTRACT if c not in df.columns]
    if missing:
        raise ValueError(f"{tag}kolom kontrak hilang: {missing}")
    d = df[CONTRACT].copy()
    if d.empty:
        raise ValueError(f"{tag}berkas prediksi kosong")
    if d["ed_stay_id"].duplicated().any():
        raise ValueError(f"{tag}ed_stay_id duplikat: {int(d['ed_stay_id'].duplicated().sum())} baris")
    if d[["ed_stay_id", "subject_id", "y_true", "y_prob", "fold"]].isna().any().any():
        raise ValueError(f"{tag}ada nilai kosong di kolom wajib")
    if not d["y_true"].isin([0, 1]).all():
        raise ValueError(f"{tag}y_true harus 0/1")
    p = d["y_prob"].to_numpy(dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError(f"{tag}y_prob harus probabilitas 0-1 yang hingga")
    folds = set(int(f) for f in d["fold"].unique())
    if expected_fold is not None and folds != {int(expected_fold)}:
        raise ValueError(f"{tag}fold berkas {sorted(folds)} tidak sama dengan yang diharapkan ({expected_fold})")
    d["y_true"] = d["y_true"].astype("int8")
    d["y_prob"] = p
    return d


def load_predictions(model: str, fold: int, pred_dir: Path = PRED_DIR, allow_test: bool = False) -> pd.DataFrame:
    if int(fold) == FOLD_TEST and not allow_test:
        raise SystemExit(f"Fold {FOLD_TEST} dikunci (brief 11.5). Evaluasi fold 19 hanya lewat skrip evaluasi akhir.")
    path = pred_path(model, fold, pred_dir)
    if not path.exists():
        raise FileNotFoundError(f"Tidak ada berkas prediksi: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    return validate_predictions(pd.read_parquet(path), expected_fold=fold, name=f"{model}_fold{fold}")


def align(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Gabungkan beberapa model pada kunjungan yang SAMA (inner join ed_stay_id) untuk perbandingan berpasangan.

    Kembalian: tabel lebar (ed_stay_id, subject_id, y_true, p_<model>...) dan info jumlah baris.
    Error bila y_true/subject_id/fold berbeda antar-model untuk kunjungan yang sama.
    """
    if not frames:
        raise ValueError("Tidak ada model untuk digabung.")
    info = {"n_per_model": {m: len(f) for m, f in frames.items()}}
    wide = None
    for m, f in frames.items():
        part = f[["ed_stay_id", "subject_id", "fold", "y_true", "y_prob"]].rename(
            columns={"y_prob": f"p_{m}", "subject_id": f"subject_{m}", "fold": f"fold_{m}", "y_true": f"y_{m}"})
        wide = part if wide is None else wide.merge(part, on="ed_stay_id", how="inner", validate="one_to_one")
    models = list(frames)
    first = models[0]
    for m in models[1:]:
        for col in ("y", "subject", "fold"):
            if not (wide[f"{col}_{m}"] == wide[f"{col}_{first}"]).all():
                raise ValueError(f"{col} berbeda antara {first} dan {m} pada kunjungan yang sama")
    out = wide.rename(columns={f"y_{first}": "y_true", f"subject_{first}": "subject_id", f"fold_{first}": "fold"})
    out = out[["ed_stay_id", "subject_id", "fold", "y_true"] + [f"p_{m}" for m in models]]
    info["n_aligned"] = len(out)
    info["dropped_per_model"] = {m: n - len(out) for m, n in info["n_per_model"].items()}
    return out.reset_index(drop=True), info
