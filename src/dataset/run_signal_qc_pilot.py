"""Pilot QC (maks. 20 rekaman, HANYA fold 0-16) lewat jalur brief yang sama dengan pipeline.

Jalankan dari root repo:
    python -u -m src.dataset.run_signal_qc_pilot

Output lokal (di-ignore git): data/interim/signal_qc_pilot.csv (kolom signal_qc_core.COLUMNS).
Fold 17-19 tidak boleh dipakai untuk menetapkan/menyetel aturan QC.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.constants import FOLD_COL, FOLD_META
from src.dataset.signal_qc_core import COLUMNS, download_and_process

ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "data/processed/master_table_draft.parquet"
CACHE = ROOT / "data/raw/ecg_cache"
OUTPUT = ROOT / "data/interim/signal_qc_pilot.csv"

BASE_URL = "https://physionet.org/files/mimic-iv-ecg/1.0/"
PREFIX = "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/"
LIMIT = 20
WORKERS = 4


def relative_path(file_name) -> str:
    return str(file_name).lstrip("/").removeprefix(PREFIX)


def draw_pilot(master: pd.DataFrame) -> pd.DataFrame:
    """Maks. LIMIT rekaman unik, hanya dari fold 0-16."""
    d = master[master[FOLD_COL] < FOLD_META].drop_duplicates("file_name").head(LIMIT)
    assert (d[FOLD_COL] < FOLD_META).all(), "Pilot menyentuh fold >= 17!"
    return d


def main() -> None:
    master = pd.read_parquet(MASTER, columns=["ed_stay_id", "file_name", FOLD_COL, "label_ami"])
    sample = draw_pilot(master)

    def progress(i, n, rec):
        if rec["status"] != "pass":
            print(f"[{i}/{n}] {rec['status']}: {rec['qc_reason'] or rec['error']}")

    results = download_and_process(sample.itertuples(index=False), BASE_URL, CACHE, relative_path,
                                   WORKERS, on_row=progress)
    report = pd.DataFrame(results, columns=COLUMNS).sort_values("ed_stay_id")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT, index=False)

    print("\n=== RINGKASAN PILOT (fold 0-16) ===")
    print(report["status"].value_counts().to_string())
    print("Hasil tersimpan:", OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
