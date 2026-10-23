"""Dataset PyTorch untuk EKG 12-lead dari cache WFDB lokal (pipeline sesuai brief).

Tidak mengunduh apa pun. Urutan: baca -> prepare_ecg (susun lead, interpolasi NaN,
resample, clip) -> standardisasi dengan statistik DARI DATA TRAINING SAJA.
Fold 17-19 (meta, threshold, test) dikunci: hanya bisa dimuat dengan
allow_locked=True secara eksplisit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.constants import FOLD_COL, FOLD_META
from src.dataset.ecg_loader import read_ecg_record
from src.model.ecg_preprocessing import LeadStandardizer, prepare_ecg

VAL_COL = "internal_val_0_16"


def relative_path(file_name, prefix: str) -> str:
    return str(file_name).lstrip("/").removeprefix(prefix)


def load_record_table(qc_csv: Path, master_parquet: Path, folds,
                      allow_locked: bool = False) -> pd.DataFrame:
    """Gabungkan hasil QC (signal_ok == 1) dengan master; saring menurut fold."""
    folds = sorted(set(int(f) for f in folds))
    if not allow_locked and any(f >= FOLD_META for f in folds):
        raise ValueError(f"Fold {[f for f in folds if f >= FOLD_META]} dikunci (>= {FOLD_META}).")

    qc = pd.read_csv(qc_csv)
    qc = qc[qc["signal_ok"] == 1][["ed_stay_id"]]
    master = pd.read_parquet(master_parquet,
                             columns=["ed_stay_id", "file_name", FOLD_COL, "label_ami", VAL_COL])
    df = qc.merge(master, on="ed_stay_id", how="left", validate="one_to_one")
    if df["file_name"].isna().any():
        raise ValueError("Ada ed_stay_id di hasil QC yang tidak ada di tabel master.")
    return df[df[FOLD_COL].isin(folds)].reset_index(drop=True)


def split_train_val(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pembagian sesuai brief 7.2 untuk model fold 0-16: validasi = internal_val_0_16 == 1."""
    if (table[FOLD_COL] >= FOLD_META).any():
        raise ValueError("Tabel berisi fold >= 17; tidak boleh dipakai untuk training/validasi.")
    val = table[table[VAL_COL] == 1].reset_index(drop=True)
    train = table[table[VAL_COL] == 0].reset_index(drop=True)
    return train, val


class ECGDataset(Dataset):
    """Mengembalikan (x, y): x float32 (12, 1000) terstandardisasi, y float32 (1,).

    `standardizer` harus di-set (dari data training) sebelum item diambil.
    """

    def __init__(self, table: pd.DataFrame, cache_dir: Path, prefix: str,
                 standardizer: LeadStandardizer | None = None):
        self.labels = table["label_ami"].to_numpy(dtype=np.float32)
        self.paths = [Path(cache_dir) / relative_path(f, prefix) for f in table["file_name"]]
        self.standardizer = standardizer
        self._raw = [self._load(p) for p in self.paths]   # (12, 1000), belum distandardisasi

    @staticmethod
    def _load(path: Path) -> np.ndarray:
        signal, fs, names = read_ecg_record(path)
        return prepare_ecg(signal, fs, names)

    def raw_records(self):
        """Iterator rekaman (12, 1000) sebelum standardisasi, untuk LeadStandardizer.fit."""
        return iter(self._raw)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        if self.standardizer is None:
            raise RuntimeError("Standardizer belum di-set: fit dulu dari data training.")
        x = self.standardizer.transform(self._raw[i])
        return torch.from_numpy(x), torch.tensor([self.labels[i]], dtype=torch.float32)
