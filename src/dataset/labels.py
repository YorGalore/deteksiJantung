"""Langkah 4: label infark miokard akut (I21/I22) dari all_diag_all.

Jalankan dari root repo (setelah cohort.py):
    python -m src.dataset.labels

Output:
    data/interim/cohort_labeled.parquet
    reports/label_summary.md   (hanya angka agregat, aman di-commit)
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.constants import (
    AMI_OTHER_PREFIXES, AMI_PREFIXES, AMI_SUBSEQUENT_PREFIXES, AMI_TYPE2_PREFIXES,
    AMI_UNSPECIFIED_PREFIXES, FOLD_COL, FOLD_META, FOLD_THRESHOLD, NSTEMI_PREFIXES,
    STEMI_PREFIXES,
)

ROOT = Path(__file__).resolve().parents[2]
_CODE_RE = re.compile(r"[A-Za-z][0-9][0-9A-Za-z.]*")
MIN_POSITIVE_PER_FOLD = 100
MIN_REPORT_COUNT = 10


def parse_codes(value) -> list[str]:
    """Ubah isi kolom diagnosis menjadi list kode ICD yang dinormalisasi.

    Menerima string list seperti "['I21.4', 'E119']", list Python, atau NaN.
    Hasil: kode huruf besar tanpa titik/spasi, misalnya ['I214', 'E119'].
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        text = " ".join(map(str, value))
    else:
        if pd.isna(value):
            return []
        text = str(value)
    return [c.replace(".", "").upper() for c in _CODE_RE.findall(text)]


def _any_prefix(codes: list[str], prefixes: tuple[str, ...]) -> int:
    return int(any(c.startswith(prefixes) for c in codes))


def label_from_codes(codes: list[str]) -> dict:
    mi_codes = [c for c in codes if c.startswith(AMI_PREFIXES)]
    return {
        "label_ami": int(len(mi_codes) > 0),
        "ami_stemi": _any_prefix(codes, STEMI_PREFIXES),
        "ami_nstemi": _any_prefix(codes, NSTEMI_PREFIXES),
        "ami_unspecified": _any_prefix(codes, AMI_UNSPECIFIED_PREFIXES),
        "ami_type2": _any_prefix(codes, AMI_TYPE2_PREFIXES),
        "ami_other": _any_prefix(codes, AMI_OTHER_PREFIXES),
        "ami_subsequent": _any_prefix(codes, AMI_SUBSEQUENT_PREFIXES),
        # MI tipe 2 saja: semua kode MI adalah I21A1 (untuk analisis sensitivitas)
        "ami_type2_only": int(len(mi_codes) > 0 and all(c.startswith(AMI_TYPE2_PREFIXES) for c in mi_codes)),
    }


def add_labels(cohort: pd.DataFrame) -> pd.DataFrame:
    codes = cohort["all_diag_all"].map(parse_codes)
    labels = pd.DataFrame([label_from_codes(c) for c in codes], index=cohort.index)
    out = pd.concat([cohort, labels], axis=1)
    hosp = cohort["all_diag_hosp"].map(parse_codes) if "all_diag_hosp" in cohort else pd.Series([[]] * len(cohort))
    out["label_source"] = ["hosp" if len(c) > 0 else "ed" for c in hosp]
    return out


def _safe(n: int) -> str:
    return str(n) if n >= MIN_REPORT_COUNT else f"<{MIN_REPORT_COUNT}"


def write_label_report(df: pd.DataFrame, path: Path) -> list[str]:
    """Tulis ringkasan label; kembalikan daftar peringatan gerbang."""
    warnings = []
    per_fold = df.groupby(FOLD_COL)["label_ami"].agg(n="size", positif="sum")
    per_fold["prevalensi_%"] = (per_fold["positif"] / per_fold["n"] * 100).round(2)
    for f in (FOLD_META, FOLD_THRESHOLD):
        if f in per_fold.index and per_fold.loc[f, "positif"] < MIN_POSITIVE_PER_FOLD:
            warnings.append(f"Fold {f} hanya punya {int(per_fold.loc[f, 'positif'])} positif (< {MIN_POSITIVE_PER_FOLD}).")

    src = df.groupby("label_source")["label_ami"].agg(n="size", positif="sum")
    src["prevalensi_%"] = (src["positif"] / src["n"] * 100).round(2)

    pos = df[df["label_ami"] == 1]
    subtype_cols = ["ami_stemi", "ami_nstemi", "ami_unspecified", "ami_type2",
                    "ami_other", "ami_subsequent", "ami_type2_only"]

    lines = [
        "# Ringkasan Label",
        "",
        "Dihasilkan otomatis oleh `src/dataset/labels.py`. Hanya angka agregat.",
        "",
        f"- Total kunjungan: {len(df):,}",
        f"- Positif (I21/I22): {int(df['label_ami'].sum()):,} ({df['label_ami'].mean()*100:.2f}%)",
        "",
        "## Per fold",
        "",
        "| fold | n | positif | prevalensi_% |",
        "|---|---|---|---|",
    ]
    lines += [f"| {f} | {int(r['n']):,} | {int(r['positif']):,} | {r['prevalensi_%']} |" for f, r in per_fold.iterrows()]
    lines += ["", "## Per sumber label", "", "| sumber | n | positif | prevalensi_% |", "|---|---|---|---|"]
    lines += [f"| {s} | {int(r['n']):,} | {int(r['positif']):,} | {r['prevalensi_%']} |" for s, r in src.iterrows()]
    lines += ["", "## Subtipe di antara kasus positif (satu kasus bisa punya >1 subtipe)", "",
              "| subtipe | n |", "|---|---|"]
    lines += [f"| {c} | {_safe(int(pos[c].sum()))} |" for c in subtype_cols]
    lines += ["", "## Gerbang jumlah positif (fold 17 & 18 >= 100)", ""]
    lines += [f"- ⚠️ {w}" for w in warnings] or ["- Lolos."]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return warnings


def main() -> None:
    src = ROOT / "data" / "interim" / "cohort.parquet"
    cohort = pd.read_parquet(src)
    df = add_labels(cohort)
    out = ROOT / "data" / "interim" / "cohort_labeled.parquet"
    df.to_parquet(out, index=False)
    warnings = write_label_report(df, ROOT / "reports" / "label_summary.md")
    print(f"Positif: {int(df['label_ami'].sum()):,} dari {len(df):,} ({df['label_ami'].mean()*100:.2f}%)")
    print(f"Tersimpan: {out.relative_to(ROOT)}")
    print("Laporan  : reports/label_summary.md")
    for w in warnings:
        print("PERINGATAN:", w)


if __name__ == "__main__":
    main()
