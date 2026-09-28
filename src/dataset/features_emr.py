"""Langkah 5: fitur EMR dari Ext-ICD dan MIMIC-IV-ED (edstays, triage, medrecon).

Jalankan dari root repo (setelah cohort.py dan labels.py):
    python -m src.dataset.features_emr

Output:
    data/interim/features_emr.parquet   (satu baris per ed_stay_id)
    reports/emr_summary.md              (hanya angka agregat, aman di-commit)

Aturan waktu: hanya informasi yang tersedia saat EKG pertama di IGD.
Teks keluhan utama asli TIDAK disimpan; yang disimpan hanya flag 0/1.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import CC_EXPLICIT_MI_PATTERN, CC_PATTERNS, MED_PATTERNS, TRIAGE_VALID_RANGE
from src.dataset.cohort import _config_path, md_table

ROOT = Path(__file__).resolve().parents[2]

TRIAGE_NUMERIC = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity"]
CC_FLAGS = list(CC_PATTERNS)
MED_FLAGS = list(MED_PATTERNS)


# ---------------------------------------------------------------------------
# Triase
# ---------------------------------------------------------------------------
def clean_triage(triage: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Ubah vital ke angka dan jadikan NaN nilai di luar rentang valid.

    Mengembalikan (tabel bersih, jumlah nilai yang dijadikan NaN per kolom).
    """
    out = triage[["stay_id"]].copy()
    n_invalid = {}
    for col in TRIAGE_NUMERIC:
        if col not in triage.columns:
            out[col] = np.nan
            continue
        v = pd.to_numeric(triage[col], errors="coerce")   # teks seperti 'unable' -> NaN
        lo, hi = TRIAGE_VALID_RANGE[col]
        bad = v.notna() & ((v < lo) | (v > hi))
        n_invalid[col] = int(bad.sum())
        out[col] = v.mask(bad)
    return out, n_invalid


def chief_complaint_flags(cc: pd.Series) -> pd.DataFrame:
    """Flag 0/1 dari teks keluhan utama. Teks aslinya tidak ikut dikembalikan."""
    text = cc.astype("string").str.lower()
    out = pd.DataFrame(index=cc.index)
    for name, pattern in CC_PATTERNS.items():
        out[name] = text.str.contains(pattern, regex=True, na=False).astype("int8")
    out["cc_explicit_mi"] = text.str.contains(CC_EXPLICIT_MI_PATTERN, regex=True, na=False).astype("int8")
    out["cc_missing"] = (text.isna() | (text.str.strip() == "")).astype("int8")
    return out


# ---------------------------------------------------------------------------
# Obat rutin (medrecon)
# ---------------------------------------------------------------------------
def medication_flags(medrecon: pd.DataFrame, stay_ids: pd.Series) -> pd.DataFrame:
    """Flag golongan obat per kunjungan.

    Kunjungan tanpa catatan medrecon: flag dan n_meds = NaN (bukan 0),
    karena 'tidak tercatat' tidak sama dengan 'tidak minum obat'.
    """
    med = medrecon.loc[medrecon["stay_id"].isin(stay_ids), ["stay_id", "name"]].copy()
    med["name"] = med["name"].astype("string").str.lower().str.strip()

    # Cocokkan pola pada nama obat unik saja (jauh lebih cepat daripada per baris)
    names = pd.Series(med["name"].dropna().unique(), dtype="string")
    lookup = pd.DataFrame({"name": names})
    for flag, pattern in MED_PATTERNS.items():
        lookup[flag] = names.str.contains(pattern, regex=True, na=False).astype("int8")
    med = med.merge(lookup, on="name", how="left")
    med[MED_FLAGS] = med[MED_FLAGS].fillna(0)

    per_stay = med.groupby("stay_id")[MED_FLAGS].max()
    per_stay["n_meds"] = med.groupby("stay_id")["name"].nunique()

    out = pd.DataFrame({"stay_id": stay_ids.unique()}).merge(
        per_stay.reset_index(), on="stay_id", how="left")
    out["medrecon_available"] = out["stay_id"].isin(per_stay.index).astype("int8")
    return out


# ---------------------------------------------------------------------------
# Gabungan
# ---------------------------------------------------------------------------
def build_emr_features(cohort: pd.DataFrame, edstays: pd.DataFrame,
                       triage: pd.DataFrame, medrecon: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    base = cohort[["ed_stay_id", "subject_id", "age", "gender"]].copy()
    base["is_male"] = (base["gender"].astype("string").str.upper().str.strip() == "M").astype("int8")
    base = base.drop(columns="gender")

    # Kedatangan (diketahui saat pasien tiba)
    arr = edstays[["stay_id", "arrival_transport"]].copy()
    arr["arrival_transport"] = arr["arrival_transport"].astype("string").str.upper().str.strip()

    tri, n_invalid = clean_triage(triage)
    cc = chief_complaint_flags(triage["chiefcomplaint"]) if "chiefcomplaint" in triage else pd.DataFrame(index=triage.index)
    tri = pd.concat([tri, cc], axis=1)

    meds = medication_flags(medrecon, cohort["ed_stay_id"])

    out = base
    for tbl in (arr, tri, meds):
        out = out.merge(tbl, left_on="ed_stay_id", right_on="stay_id", how="left").drop(columns="stay_id")

    assert len(out) == len(cohort), "Jumlah baris berubah setelah join (ada kunci ganda)!"
    assert out["ed_stay_id"].is_unique
    return out, n_invalid


def write_emr_report(feat: pd.DataFrame, n_invalid: dict, path: Path) -> None:
    num_cols = ["age"] + TRIAGE_NUMERIC + ["n_meds"]
    num = pd.DataFrame({
        "fitur": num_cols,
        "missing_%": [round(feat[c].isna().mean() * 100, 2) for c in num_cols],
        "median": [round(feat[c].median(), 2) for c in num_cols],
        "dijadikan_NaN_(luar_rentang)": [n_invalid.get(c, 0) for c in num_cols],
    })
    flag_cols = CC_FLAGS + ["cc_explicit_mi", "cc_missing"] + MED_FLAGS + ["medrecon_available", "is_male"]
    flags = pd.DataFrame({
        "flag": flag_cols,
        "%_bernilai_1": [round(feat[c].mean() * 100, 2) for c in flag_cols],
        "missing_%": [round(feat[c].isna().mean() * 100, 2) for c in flag_cols],
    })
    arr = feat["arrival_transport"].fillna("(kosong)").value_counts()
    arr = arr[arr >= 10].rename_axis("arrival_transport").reset_index(name="n")

    lines = [
        "# Ringkasan Fitur EMR", "",
        "Dihasilkan otomatis oleh `src/dataset/features_emr.py`. Hanya angka agregat.", "",
        f"- Jumlah kunjungan: {len(feat):,}", "",
        "## Fitur numerik", "", md_table(num), "",
        "## Flag (persentase kunjungan bernilai 1; flag obat dihitung dari kunjungan yang punya medrecon)", "",
        md_table(flags), "",
        "## Cara kedatangan (kategori >= 10)", "", md_table(arr), "",
        "Catatan: `cc_explicit_mi`, `cc_missing`, dan `medrecon_available` adalah kolom analisis, bukan fitur model.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ed_dir = _config_path("ed_dir", "data/raw/mimic-iv-ed/ed")
    cohort = pd.read_parquet(ROOT / "data" / "interim" / "cohort_labeled.parquet")
    print("Membaca tabel MIMIC-IV-ED dari", ed_dir)
    edstays = pd.read_csv(ed_dir / "edstays.csv.gz", usecols=["stay_id", "arrival_transport"])
    triage = pd.read_csv(ed_dir / "triage.csv.gz")
    medrecon = pd.read_csv(ed_dir / "medrecon.csv.gz", usecols=["stay_id", "name"])

    feat, n_invalid = build_emr_features(cohort, edstays, triage, medrecon)

    out = ROOT / "data" / "interim" / "features_emr.parquet"
    feat.to_parquet(out, index=False)
    write_emr_report(feat, n_invalid, ROOT / "reports" / "emr_summary.md")
    print(f"Tersimpan: {out.relative_to(ROOT)} ({len(feat):,} kunjungan, {feat.shape[1]} kolom)")
    print("Laporan  : reports/emr_summary.md")


if __name__ == "__main__":
    main()
