"""Langkah 7: gabungkan kohort, label, fitur EMR, dan fitur MM menjadi tabel master.

Jalankan dari root repo (setelah cohort, labels, features_emr, features_mm):
    python -m src.dataset.build_master

Output:
    data/processed/master_table_draft.parquet   (sebelum hasil cek sinyal dari Orang 2)
    data/processed/master_table_v1.parquet      (jika data/interim/signal_qc.csv sudah ada)
    data/processed/internal_val_0_16.csv        (daftar subject_id validasi internal)
    data/processed/internal_val_0_17.csv
    data/processed/ecg_download_urls.txt        (untuk Orang 2: URL .hea dan .dat kohort)
    reports/master_summary.md                   (angka agregat + kamus kolom, aman di-commit)

Hasil cek sinyal dari Orang 2 disimpan sebagai data/interim/signal_qc.csv
dengan kolom: ed_stay_id, signal_ok (1 = lolos, 0 = gagal).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import (
    ANALYSIS_COLS, EMR_FEATURES, FOLD_COL, FOLD_META, FOLD_TEST, FOLD_THRESHOLD,
    ID_COLS, LABEL_COLS, MM_FEATURES, SPLIT_COLS,
)
from src.dataset.cohort import md_table

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
ECG_BASE_URL = "https://physionet.org/files/mimic-iv-ecg/1.0/"
INTERNAL_VAL_FRACTION = 0.10


def _seed() -> int:
    cfg = ROOT / "config.yaml"
    if cfg.exists():
        try:
            import yaml
            return int((yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}).get("seed", 42))
        except Exception:
            pass
    return 42


def internal_val_subjects(df: pd.DataFrame, folds: list[int], fraction: float, seed: int) -> np.ndarray:
    """Pilih acak `fraction` pasien (bukan kunjungan) dari fold yang diberikan."""
    subjects = np.sort(df.loc[df[FOLD_COL].isin(folds), "subject_id"].unique())
    rng = np.random.default_rng(seed)
    n = int(round(len(subjects) * fraction))
    return np.sort(rng.choice(subjects, size=n, replace=False))


def build_master(labeled: pd.DataFrame, emr: pd.DataFrame, mm: pd.DataFrame,
                 seed: int = 42, signal_qc: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    info = {}
    base_cols = ["ed_stay_id", "subject_id", "study_id", "file_name", FOLD_COL] + [
        c for c in LABEL_COLS if c in labeled.columns] + ["first_ecg_matches_no0"]
    master = labeled[[c for c in base_cols if c in labeled.columns]]

    emr_cols = ["ed_stay_id"] + [c for c in emr.columns if c not in ("ed_stay_id", "subject_id")]
    master = master.merge(emr[emr_cols], on="ed_stay_id", how="left", validate="one_to_one")
    master = master.merge(mm, on="ed_stay_id", how="left", validate="one_to_one")
    assert len(master) == len(labeled), "Jumlah baris berubah saat menggabungkan fitur!"

    # Validasi internal berbasis pasien
    val_016 = internal_val_subjects(master, list(range(0, FOLD_META)), INTERNAL_VAL_FRACTION, seed)
    val_017 = internal_val_subjects(master, list(range(0, FOLD_THRESHOLD)), INTERNAL_VAL_FRACTION, seed + 1)
    master["internal_val_0_16"] = (master["subject_id"].isin(val_016) & (master[FOLD_COL] < FOLD_META)).astype("int8")
    master["internal_val_0_17"] = (master["subject_id"].isin(val_017) & (master[FOLD_COL] < FOLD_THRESHOLD)).astype("int8")
    info["val_016"], info["val_017"] = val_016, val_017

    # Hasil cek kualitas sinyal (Orang 2)
    if signal_qc is not None:
        master = master.merge(signal_qc[["ed_stay_id", "signal_ok"]], on="ed_stay_id", how="left", validate="one_to_one")
        info["signal_missing"] = int(master["signal_ok"].isna().sum())
        info["signal_fail"] = int((master["signal_ok"] == 0).sum())
        master = master[master["signal_ok"] == 1].copy()
        master["signal_ok"] = master["signal_ok"].astype("int8")
    else:
        master["signal_ok"] = pd.Series(pd.NA, index=master.index, dtype="Int8")

    ordered = (ID_COLS + SPLIT_COLS + LABEL_COLS + ANALYSIS_COLS + EMR_FEATURES + MM_FEATURES)
    missing = [c for c in ordered if c not in master.columns]
    assert not missing, f"Kolom hilang dari tabel master: {missing}"
    return master[ordered].reset_index(drop=True), info


def ecg_download_urls(master: pd.DataFrame) -> list[str]:
    urls = []
    for f in master["file_name"].dropna().astype(str):
        f = f.lstrip("/")
        urls += [ECG_BASE_URL + f + ".hea", ECG_BASE_URL + f + ".dat"]
    return urls


def column_dictionary() -> pd.DataFrame:
    groups = [("ID", ID_COLS), ("Split", SPLIT_COLS), ("Label", LABEL_COLS),
              ("Analisis (BUKAN fitur)", ANALYSIS_COLS), ("Fitur EMR", EMR_FEATURES),
              ("Fitur MM", MM_FEATURES)]
    return pd.DataFrame([{"kelompok": g, "kolom": ", ".join(cols)} for g, cols in groups])


def write_master_report(master: pd.DataFrame, info: dict, final: bool, path: Path) -> None:
    per_fold = master.groupby(FOLD_COL).agg(n=("label_ami", "size"), positif=("label_ami", "sum"),
                                            internal_val_0_16=("internal_val_0_16", "sum"),
                                            internal_val_0_17=("internal_val_0_17", "sum")).reset_index()
    peran = {FOLD_META: "meta-learner", FOLD_THRESHOLD: "threshold", FOLD_TEST: "TEST (dikunci)"}
    per_fold["peran"] = per_fold[FOLD_COL].map(peran).fillna("training")
    feats = EMR_FEATURES + MM_FEATURES
    miss = pd.DataFrame({"fitur": feats, "missing_%": [round(master[c].isna().mean() * 100, 2) for c in feats]})
    miss = miss[miss["missing_%"] > 0].sort_values("missing_%", ascending=False)

    lines = [
        "# Ringkasan Tabel Master", "",
        "Dihasilkan otomatis oleh `src/dataset/build_master.py`. Hanya angka agregat.", "",
        f"- Status: **{'FINAL (v1)' if final else 'DRAFT (menunggu hasil cek sinyal Orang 2)'}**",
        f"- Jumlah kunjungan: {len(master):,} | pasien: {master['subject_id'].nunique():,}",
        f"- Positif: {int(master['label_ami'].sum()):,} ({master['label_ami'].mean()*100:.2f}%)",
        f"- Jumlah fitur: {len(EMR_FEATURES)} EMR + {len(MM_FEATURES)} MM",
        f"- Validasi internal: {len(info['val_016']):,} pasien (fold 0-16), {len(info['val_017']):,} pasien (fold 0-17)",
    ]
    if final:
        lines += [f"- Dikeluarkan karena sinyal gagal: {info['signal_fail']:,} | tanpa hasil cek: {info['signal_missing']:,}"]
    lines += ["", "## Per fold", "", md_table(per_fold), "",
              "## Fitur dengan nilai kosong", "", md_table(miss), "",
              "## Kamus kolom (pakai daftar di `src/constants.py`, jangan memilih kolom sendiri)", "",
              md_table(column_dictionary()), ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    labeled = pd.read_parquet(INTERIM / "cohort_labeled.parquet")
    emr = pd.read_parquet(INTERIM / "features_emr.parquet")
    mm = pd.read_parquet(INTERIM / "features_mm.parquet")
    qc_path = INTERIM / "signal_qc.csv"
    signal_qc = pd.read_csv(qc_path) if qc_path.exists() else None
    final = signal_qc is not None

    master, info = build_master(labeled, emr, mm, seed=_seed(), signal_qc=signal_qc)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    name = "master_table_v1.parquet" if final else "master_table_draft.parquet"
    master.to_parquet(PROCESSED / name, index=False)
    pd.Series(info["val_016"], name="subject_id").to_csv(PROCESSED / "internal_val_0_16.csv", index=False)
    pd.Series(info["val_017"], name="subject_id").to_csv(PROCESSED / "internal_val_0_17.csv", index=False)
    urls = ecg_download_urls(master)
    (PROCESSED / "ecg_download_urls.txt").write_text("\n".join(urls) + "\n", encoding="utf-8")
    write_master_report(master, info, final, ROOT / "reports" / "master_summary.md")

    print(f"Tersimpan: data/processed/{name} ({len(master):,} kunjungan, {master.shape[1]} kolom)")
    print(f"Tersimpan: data/processed/ecg_download_urls.txt ({len(urls):,} URL untuk Orang 2)")
    print("Laporan  : reports/master_summary.md")
    print("Langkah berikutnya: python -m src.dataset.check_master")


if __name__ == "__main__":
    main()
