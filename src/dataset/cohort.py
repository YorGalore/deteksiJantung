"""Langkah 3: bangun kohort (satu baris per kunjungan IGD, EKG pertama).

Jalankan dari root repo:
    python -m src.dataset.cohort

Output:
    data/interim/cohort.parquet
    reports/cohort_flow.md   (hanya angka agregat, aman di-commit)
"""
from __future__ import annotations

import numbers
from pathlib import Path

import pandas as pd

from src.constants import FOLD_COL
from src.dataset.labels import parse_codes

ROOT = Path(__file__).resolve().parents[2]

KEEP_COLS = [
    "ed_stay_id", "subject_id", "study_id", "file_name", "ecg_time",
    FOLD_COL, "age", "gender", "all_diag_all", "all_diag_hosp", "ed_diag_ed",
    "ecg_no_within_stay",
]


def _config_path(key: str, default: str) -> Path:
    """Ambil path dari config.yaml (paths.<key>) jika ada, selain itu default."""
    cfg = ROOT / "config.yaml"
    if cfg.exists():
        try:
            import yaml
            data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
            val = (data.get("paths") or {}).get(key)
            if val:
                return ROOT / val
        except Exception:
            pass
    return ROOT / default


def _count(name: str, d: pd.DataFrame) -> dict:
    return {
        "langkah": name,
        "n_ekg": len(d),
        "n_kunjungan_ed": int(d["ed_stay_id"].nunique()),
        "n_pasien": int(d["subject_id"].nunique()),
    }


def build_cohort(ext: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter Ext-ICD menjadi kohort. Mengembalikan (kohort, tabel alur)."""
    # Pasien tidak boleh ada di lebih dari satu fold
    n_fold = ext.groupby("subject_id")[FOLD_COL].nunique()
    assert (n_fold <= 1).all(), "Ada pasien di lebih dari satu fold!"

    flow = []
    d = ext
    flow.append(_count("0. Semua EKG di Ext-ICD", d))

    d = d[d["ecg_taken_in_ed"].astype(bool)]
    flow.append(_count("1. EKG diambil di IGD", d))

    d = d[d["ed_stay_id"].notna()]
    flow.append(_count("2. Punya ed_stay_id", d))

    d = d[d["all_diag_all"].map(lambda x: len(parse_codes(x)) > 0)]
    flow.append(_count("3. Punya diagnosis (all_diag_all)", d))

    # EKG pertama per kunjungan: ecg_time paling awal, seri -> study_id terkecil.
    # ecg_no_within_stay tidak dipakai karena nilai 0 tidak unik per kunjungan.
    d = d.copy()
    d["ecg_time"] = pd.to_datetime(d["ecg_time"], errors="coerce")
    d = d.sort_values(["ed_stay_id", "ecg_time", "study_id"], na_position="last")
    d = d.drop_duplicates(subset="ed_stay_id", keep="first")
    flow.append(_count("4. EKG pertama per kunjungan (ecg_time terawal)", d))

    d["ed_stay_id"] = d["ed_stay_id"].astype("int64")
    d["first_ecg_matches_no0"] = (d["ecg_no_within_stay"] == 0).astype(int)

    keep = [c for c in KEEP_COLS if c in d.columns] + ["first_ecg_matches_no0"]
    cohort = d[keep].reset_index(drop=True)

    assert cohort["ed_stay_id"].is_unique, "ed_stay_id tidak unik setelah filter!"
    return cohort, pd.DataFrame(flow)


def md_table(df: pd.DataFrame) -> str:
    """Tabel markdown sederhana tanpa dependensi tambahan (tabulate)."""
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        rows.append("| " + " | ".join(f"{v:,}" if isinstance(v, numbers.Integral) and not isinstance(v, bool) else str(v) for v in r) + " |")
    return "\n".join(rows)


def write_flow_report(flow: pd.DataFrame, cohort: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    per_fold = cohort.groupby(FOLD_COL).size().rename("n_kunjungan")
    match = cohort["first_ecg_matches_no0"].mean() * 100
    lines = [
        "# Alur Kohort",
        "",
        "Dihasilkan otomatis oleh `src/dataset/cohort.py`. Hanya angka agregat.",
        "",
        md_table(flow),
        "",
        f"EKG pertama (berdasarkan ecg_time) yang juga bernilai `ecg_no_within_stay == 0`: {match:.2f}%",
        "",
        "## Kunjungan per fold",
        "",
        md_table(per_fold.reset_index()),
        "",
        "Filter 5 (kualitas sinyal) ditambahkan setelah Orang 2 menyerahkan hasil cek sinyal.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ext_path = _config_path("ext_icd", "data/raw/records_w_diag_icd10.csv")
    print("Membaca", ext_path)
    ext = pd.read_csv(ext_path, low_memory=False)

    cohort, flow = build_cohort(ext)
    print(flow.to_string(index=False))

    out = ROOT / "data" / "interim" / "cohort.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(out, index=False)
    write_flow_report(flow, cohort, ROOT / "reports" / "cohort_flow.md")
    print(f"Tersimpan: {out.relative_to(ROOT)} ({len(cohort):,} kunjungan)")
    print("Laporan  : reports/cohort_flow.md")


if __name__ == "__main__":
    main()
