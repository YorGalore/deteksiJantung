"""Langkah 6: fitur machine measurements (MIMIC-IV-ECG).

Jalankan dari root repo (setelah cohort.py dan labels.py):
    python -m src.dataset.features_mm

Output:
    data/interim/features_mm.parquet   (fitur numerik, satu baris per ed_stay_id)
    data/interim/mm_reports.parquet    (teks report_* KHUSUS untuk baseline M0;
                                        tidak boleh digabung ke tabel master)
    reports/mm_summary.md              (hanya angka agregat, aman di-commit)

Aturan (dari eksplorasi 28-09-2026):
- Nilai >= 29999 adalah sentinel -> NaN.
- Nilai onset/end mentah bergantung jenis mesin, jadi hanya selisihnya yang dipakai.
- Nilai di luar rentang valid -> NaN (termasuk qrs_duration negatif).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import MM_FEATURES, MM_RAW_COLS, MM_SENTINEL_MIN, MM_VALID_RANGE
from src.dataset.cohort import _config_path, md_table

ROOT = Path(__file__).resolve().parents[2]
KEYS = ["subject_id", "study_id"]
REPORT_COLS = [f"report_{i}" for i in range(18)]


def clean_mm(mm: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Bersihkan kolom mentah dan hitung fitur turunan.

    Mengembalikan (tabel fitur dengan kolom KEYS + MM_FEATURES, catatan jumlah NaN per alasan).
    """
    raw = mm[MM_RAW_COLS].apply(pd.to_numeric, errors="coerce")
    notes = {"sentinel": {}, "luar_rentang": {}}

    # 1. Sentinel -> NaN
    is_sentinel = raw >= MM_SENTINEL_MIN
    notes["sentinel"] = {c: int(is_sentinel[c].sum()) for c in MM_RAW_COLS}
    p_axis_sentinel = is_sentinel["p_axis"]
    p_axis_raw_missing = raw["p_axis"].isna()
    raw = raw.mask(is_sentinel)

    # 2. RR di luar rentang -> NaN sebelum dipakai untuk turunan
    rr = raw["rr_interval"]

    out = mm[KEYS].copy()
    out["rr_interval"] = rr
    rr_pos = rr.where(rr > 0)
    out["hr_ecg"] = 60000 / rr_pos
    out["pr_interval"] = raw["qrs_onset"] - raw["p_onset"]
    out["qrs_duration"] = raw["qrs_end"] - raw["qrs_onset"]
    out["qt_interval"] = raw["t_end"] - raw["qrs_onset"]
    out["qtc_bazett"] = out["qt_interval"] / np.sqrt(rr_pos / 1000)
    for ax in ["p_axis", "qrs_axis", "t_axis"]:
        out[ax] = raw[ax]

    # 3. Rentang valid (rr dulu, lalu turunan yang memakai rr dihitung ulang dari rr bersih)
    lo, hi = MM_VALID_RANGE["rr_interval"]
    bad_rr = out["rr_interval"].notna() & ~out["rr_interval"].between(lo, hi)
    notes["luar_rentang"]["rr_interval"] = int(bad_rr.sum())
    out.loc[bad_rr, ["rr_interval", "hr_ecg", "qtc_bazett"]] = np.nan

    for col, (lo, hi) in MM_VALID_RANGE.items():
        if col == "rr_interval":
            continue
        bad = out[col].notna() & ~out[col].between(lo, hi)
        notes["luar_rentang"][col] = int(bad.sum())
        out.loc[bad, col] = np.nan

    # Normalisasi axis ke (-180, 180]: mesin berbeda memakai konvensi berbeda (mis. 197 = -163)
    for ax in ["p_axis", "qrs_axis", "t_axis"]:
        a = out[ax]
        out[ax] = -((180 - a) % 360) + 180

    # 4. P wave tidak terdeteksi: 1 jika p_axis sentinel, 0 jika ada nilai, NaN jika memang kosong
    out["p_wave_absent"] = np.where(p_axis_sentinel, 1.0, np.where(p_axis_raw_missing, np.nan, 0.0))

    out["mm_missing_count"] = out[MM_FEATURES].isna().sum(axis=1).astype("int16")
    return out, notes


def build_mm_features(cohort: pd.DataFrame, mm: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    assert not mm.duplicated(subset=KEYS).any(), "subject_id + study_id tidak unik di machine_measurements!"
    feat, notes = clean_mm(mm)
    out = cohort[["ed_stay_id"] + KEYS].merge(feat, on=KEYS, how="left")
    assert len(out) == len(cohort), "Jumlah baris berubah setelah join!"
    notes["tanpa_data_mm"] = int(out[MM_FEATURES].isna().all(axis=1).sum())
    out["mm_missing_count"] = out["mm_missing_count"].fillna(len(MM_FEATURES)).astype("int16")
    return out.drop(columns=KEYS), notes


def write_mm_report(feat: pd.DataFrame, notes: dict, path: Path) -> None:
    rows = []
    for c in MM_FEATURES:
        rows.append({
            "fitur": c,
            "missing_%": round(feat[c].isna().mean() * 100, 2),
            "p1": round(feat[c].quantile(0.01), 1),
            "median": round(feat[c].median(), 1),
            "p99": round(feat[c].quantile(0.99), 1),
            "luar_rentang_->NaN": notes["luar_rentang"].get(c, 0),
        })
    sent = pd.DataFrame({"kolom_mentah": list(notes["sentinel"]), "n_sentinel_(semua_EKG)": list(notes["sentinel"].values())})
    qrs0 = (feat["qrs_axis"] == 0).mean() * 100
    lines = [
        "# Ringkasan Fitur Machine Measurements", "",
        "Dihasilkan otomatis oleh `src/dataset/features_mm.py`. Hanya angka agregat.", "",
        f"- Jumlah kunjungan: {len(feat):,}",
        f"- Kunjungan tanpa data MM sama sekali: {notes['tanpa_data_mm']:,}",
        f"- p_wave_absent = 1: {feat['p_wave_absent'].mean()*100:.2f}%",
        f"- qrs_axis tepat 0: {qrs0:.2f}% (jika jauh lebih tinggi dari nilai 1 atau -1, curigai sentinel)",
        f"- qrs_axis tepat 1: {(feat['qrs_axis'] == 1).mean()*100:.2f}% | tepat -1: {(feat['qrs_axis'] == -1).mean()*100:.2f}%", "",
        "## Fitur (kohort)", "", md_table(pd.DataFrame(rows)), "",
        "## Sentinel (>= 29999) di seluruh machine_measurements", "", md_table(sent), "",
        "Nilai onset/end mentah tidak dijadikan fitur karena titik acuannya bergantung jenis mesin.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    mm_path = _config_path("mm", "data/raw/mimic-iv-ecg/machine_measurements.csv")
    cohort = pd.read_parquet(ROOT / "data" / "interim" / "cohort_labeled.parquet")
    print("Membaca", mm_path)

    mm = pd.read_csv(mm_path, usecols=KEYS + MM_RAW_COLS)
    feat, notes = build_mm_features(cohort, mm)
    out = ROOT / "data" / "interim" / "features_mm.parquet"
    feat.to_parquet(out, index=False)

    # Teks interpretasi mesin, disimpan terpisah hanya untuk baseline klinis M0
    header = pd.read_csv(mm_path, nrows=0).columns
    rep_cols = [c for c in REPORT_COLS if c in header]
    reports = pd.read_csv(mm_path, usecols=KEYS + rep_cols, dtype={c: "string" for c in rep_cols})
    reports = cohort[["ed_stay_id"] + KEYS].merge(reports, on=KEYS, how="left").drop(columns=KEYS)
    reports.to_parquet(ROOT / "data" / "interim" / "mm_reports.parquet", index=False)

    write_mm_report(feat, notes, ROOT / "reports" / "mm_summary.md")
    print(f"Tersimpan: {out.relative_to(ROOT)} ({len(feat):,} kunjungan)")
    print("Tersimpan: data/interim/mm_reports.parquet (khusus baseline M0)")
    print("Laporan  : reports/mm_summary.md")


if __name__ == "__main__":
    main()
