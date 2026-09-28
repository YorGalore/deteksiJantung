"""Langkah 8: pemeriksaan otomatis tabel master (struktur, label, kebocoran).

Jalankan dari root repo (setelah build_master):
    python -m src.dataset.check_master

Output: reports/master_checks.md (hanya angka agregat, aman di-commit).
Program berhenti dengan error jika ada pemeriksaan WAJIB yang gagal.

Cek kebocoran memakai AUROC satu fitur di fold 0-16 saja (fold 17-19 tidak disentuh).
Fitur dengan AUROC > 0.85 (atau < 0.15) ditandai untuk diperiksa manual.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import (
    ANALYSIS_COLS, EMR_FEATURES, FOLD_COL, FOLD_META, FORBIDDEN_COLUMNS, LABEL_COLS,
    MM_FEATURES, SPLIT_COLS,
)
from src.dataset.cohort import md_table

ROOT = Path(__file__).resolve().parents[2]
SUSPICIOUS_AUROC = 0.85
ALLOWED_TEXT_COLS = {"file_name", "arrival_transport", "label_source"}


def auroc(y: pd.Series, x: pd.Series) -> float:
    """AUROC lewat statistik Mann-Whitney (tanpa sklearn). NaN diabaikan."""
    m = x.notna()
    y, x = y[m], x[m]
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = x.rank(method="average")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def run_checks(master: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    checks = []

    def check(name, ok, detail="", wajib=True):
        checks.append({"pemeriksaan": name, "hasil": "LOLOS" if ok else ("GAGAL" if wajib else "PERIKSA"),
                       "wajib": "ya" if wajib else "tidak", "detail": detail})

    feats = EMR_FEATURES + MM_FEATURES

    # Struktur
    check("ed_stay_id unik", master["ed_stay_id"].is_unique)
    n_fold = master.groupby("subject_id")[FOLD_COL].nunique()
    check("Setiap pasien hanya di satu fold", bool((n_fold <= 1).all()), f"{int((n_fold > 1).sum())} pasien melanggar")
    check("fold hanya 0-19", bool(master[FOLD_COL].between(0, 19).all()))
    for col, max_fold in [("internal_val_0_16", FOLD_META), ("internal_val_0_17", FOLD_META + 1)]:
        bad = int(((master[col] == 1) & (master[FOLD_COL] >= max_fold)).sum())
        check(f"{col} tidak menyentuh fold >= {max_fold}", bad == 0, f"{bad} baris")
        per_subject = master[master[FOLD_COL] < max_fold].groupby("subject_id")[col].nunique()
        check(f"{col} berbasis pasien (tidak terbelah)", bool((per_subject <= 1).all()))

    # Label
    check("label_ami hanya 0/1 tanpa NaN", set(master["label_ami"].dropna().unique()) <= {0, 1}
          and master["label_ami"].notna().all())

    # Kebocoran kolom
    forbidden_in_feats = sorted(set(feats) & set(FORBIDDEN_COLUMNS))
    check("Tidak ada kolom terlarang di daftar fitur", not forbidden_in_feats, ", ".join(forbidden_in_feats))
    overlap = sorted(set(feats) & set(ANALYSIS_COLS + LABEL_COLS + SPLIT_COLS))
    check("Kolom analisis/label/split tidak ada di daftar fitur", not overlap, ", ".join(overlap))
    text_cols = [c for c in master.columns
                 if (master[c].dtype == object or str(master[c].dtype) == "string") and c not in ALLOWED_TEXT_COLS]
    check("Tidak ada kolom teks bebas di tabel master", not text_cols, ", ".join(text_cols))
    raw_cols = [c for c in ["chiefcomplaint", "p_onset", "qrs_onset", "t_end", "ecg_time"] + [f"report_{i}" for i in range(18)]
                if c in master.columns]
    check("Tidak ada teks keluhan/laporan mesin/onset mentah", not raw_cols, ", ".join(raw_cols))

    # AUROC satu fitur (hanya fold 0-16)
    train = master[master[FOLD_COL] < FOLD_META]
    y = train["label_ami"].astype(int)
    rows = []
    for c in feats:
        x = train[c]
        if c == "arrival_transport":
            x = x.map(train.groupby(c)["label_ami"].mean())   # target rate per kategori (hanya untuk cek)
        rows.append({"fitur": c, "auroc_fold0_16": round(auroc(y, pd.to_numeric(x, errors="coerce")), 3)})
    auc = pd.DataFrame(rows)
    auc["jarak_dari_0.5"] = (auc["auroc_fold0_16"] - 0.5).abs().round(3)
    auc = auc.sort_values("jarak_dari_0.5", ascending=False)
    sus = auc[(auc["auroc_fold0_16"] > SUSPICIOUS_AUROC) | (auc["auroc_fold0_16"] < 1 - SUSPICIOUS_AUROC)]
    check(f"Tidak ada fitur tunggal dengan AUROC > {SUSPICIOUS_AUROC}", sus.empty,
          ", ".join(sus["fitur"]), wajib=False)
    return checks, auc


def main() -> None:
    processed = ROOT / "data" / "processed"
    path = processed / "master_table_v1.parquet"
    if not path.exists():
        path = processed / "master_table_draft.parquet"
    master = pd.read_parquet(path)
    print("Memeriksa", path.relative_to(ROOT))

    checks, auc = run_checks(master)
    tbl = pd.DataFrame(checks)
    lines = ["# Pemeriksaan Tabel Master", "",
             f"Dihasilkan otomatis oleh `src/dataset/check_master.py` untuk `{path.name}`. Hanya angka agregat.", "",
             md_table(tbl), "",
             "## AUROC satu fitur (fold 0-16), 15 teratas", "",
             "AUROC mendekati 1 atau 0 pada satu fitur saja adalah tanda kebocoran label yang perlu diperiksa.", "",
             md_table(auc.head(15)), ""]
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "master_checks.md").write_text("\n".join(lines), encoding="utf-8")

    print(tbl[["pemeriksaan", "hasil"]].to_string(index=False))
    print("Laporan: reports/master_checks.md")
    if (tbl["hasil"] == "GAGAL").any():
        print("ADA PEMERIKSAAN WAJIB YANG GAGAL. Jangan serahkan tabel ini dulu.")
        sys.exit(1)


if __name__ == "__main__":
    main()
