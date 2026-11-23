"""Model tabular XGBoost (Orang 3): M2 (EMR), M3 (machine measurements), M4 (EMR + MM, Jalur B).

Aturan (PROJECT_BRIEF 7.2, 11.1; rencana proyek 6.3):
- Fitur HANYA dari src/constants.py (EMR_FEATURES, MM_FEATURES). Kolom label/analisis/split = kebocoran.
- M2, M3: train = fold 0-16 & internal_val_0_16 == 0; validasi (early stopping + tuning) = internal_val_0_16 == 1.
- M4    : train = fold 0-17 & internal_val_0_17 == 0; validasi = internal_val_0_17 == 1.
- Fold 17/18/19 tidak pernah dipakai untuk early stopping atau tuning model dasar.
- scale_pos_weight = n_negatif / n_positif pada data training.
- NaN dibiarkan (XGBoost menanganinya). arrival_transport di-one-hot dengan daftar kategori tetap.
- Prediksi: M2/M3 untuk fold 17, 18 (19 hanya dengan flag eksplisit). M4 untuk fold 18 (19 dengan flag);
  M4 TIDAK memprediksi fold 17 karena fold 17 adalah data latihnya.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.constants import (
    ANALYSIS_COLS, EMR_FEATURES, FOLD_COL, FOLD_META, FOLD_TEST, FOLD_THRESHOLD, FORBIDDEN_COLUMNS,
    LABEL_COLS, MM_FEATURES, SPLIT_COLS,
)

ARRIVAL_COL = "arrival_transport"
# Kategori tetap (PROJECT_BRIEF 7.3, dicek pada master_table_draft 2026-11-14). Kategori baru -> error.
ARRIVAL_CATEGORIES = ["AMBULANCE", "HELICOPTER", "OTHER", "UNKNOWN", "WALK IN"]

MODEL_SPECS = {
    "M2": {"desc": "XGBoost EMR", "features": list(EMR_FEATURES),
           "train_folds": list(range(0, FOLD_META)), "val_col": "internal_val_0_16",
           "pred_folds": (FOLD_META, FOLD_THRESHOLD)},
    "M3": {"desc": "XGBoost machine measurements", "features": list(MM_FEATURES),
           "train_folds": list(range(0, FOLD_META)), "val_col": "internal_val_0_16",
           "pred_folds": (FOLD_META, FOLD_THRESHOLD)},
    "M4": {"desc": "XGBoost concat EMR + MM (Jalur B)", "features": list(EMR_FEATURES) + list(MM_FEATURES),
           "train_folds": list(range(0, FOLD_THRESHOLD)), "val_col": "internal_val_0_17",
           "pred_folds": (FOLD_THRESHOLD,)},
}

_NOT_FEATURES = set(FORBIDDEN_COLUMNS) | set(LABEL_COLS) | set(ANALYSIS_COLS) | set(SPLIT_COLS)


def arrival_dummy_name(cat: str) -> str:
    return "arrival_" + cat.replace(" ", "_")


def check_feature_list(features: list[str]) -> None:
    """Gagal keras bila ada kolom terlarang/label/analisis/split di daftar fitur."""
    bad = sorted(set(features) & _NOT_FEATURES)
    if bad:
        raise ValueError(f"Kolom terlarang di daftar fitur: {bad}")
    allowed = set(EMR_FEATURES) | set(MM_FEATURES)
    extra = sorted(set(features) - allowed)
    if extra:
        raise ValueError(f"Kolom di luar EMR_FEATURES/MM_FEATURES: {extra}")


def build_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Matriks fitur float32. arrival_transport -> one-hot (kosong = semua 0). NaN lain dibiarkan."""
    check_feature_list(features)
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom fitur tidak ada di tabel master: {missing}")
    cols = {}
    for c in features:
        if c == ARRIVAL_COL:
            s = df[c].astype("string").str.upper().str.strip()
            unknown = sorted(set(s.dropna().unique()) - set(ARRIVAL_CATEGORIES))
            if unknown:
                raise ValueError(f"Kategori arrival_transport tidak dikenal: {unknown}")
            for cat in ARRIVAL_CATEGORIES:
                cols[arrival_dummy_name(cat)] = (s == cat).fillna(False).astype(np.float32).to_numpy()
        else:
            cols[c] = pd.to_numeric(df[c], errors="raise").astype(np.float32).to_numpy()
    X = pd.DataFrame(cols, index=df.index)
    leaked = sorted(set(X.columns) & _NOT_FEATURES)
    assert not leaked, f"Kolom terlarang masuk matriks fitur: {leaked}"
    return X


def split_train_val(master: pd.DataFrame, model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pembagian train/validasi internal sesuai brief untuk model `model`."""
    spec = MODEL_SPECS[model]
    base = master[master[FOLD_COL].isin(spec["train_folds"])]
    train = base[base[spec["val_col"]] == 0]
    val = base[base[spec["val_col"]] == 1]
    max_fold = max(spec["train_folds"])
    assert train[FOLD_COL].max() <= max_fold and val[FOLD_COL].max() <= max_fold
    assert FOLD_TEST not in set(train[FOLD_COL]) and FOLD_THRESHOLD not in set(train[FOLD_COL])
    overlap = set(train["subject_id"]) & set(val["subject_id"])
    if overlap:
        raise ValueError(f"{len(overlap)} pasien ada di training dan validasi sekaligus.")
    for name, d in (("training", train), ("validasi", val)):
        if d.empty or d["label_ami"].nunique() < 2:
            raise ValueError(f"Data {name} {model} kosong atau hanya satu kelas.")
    return train, val


def sample_params(n_trials: int, seed: int) -> list[dict]:
    """Random search ringan (rencana 6.3, +-20 kombinasi). Trial pertama = konfigurasi dasar yang wajar."""
    space = {
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.02, 0.03, 0.05, 0.1],
        "min_child_weight": [1, 3, 5, 10],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 1.0],
    }
    trials = [{"max_depth": 4, "learning_rate": 0.05, "min_child_weight": 1, "subsample": 0.8, "colsample_bytree": 0.8}]
    rng = np.random.default_rng(seed)
    seen = {tuple(trials[0].values())}
    while len(trials) < n_trials:
        t = {k: v[int(rng.integers(len(v)))] for k, v in space.items()}
        key = tuple(t.values())
        if key not in seen:
            seen.add(key)
            trials.append(t)
    return trials[:n_trials]
