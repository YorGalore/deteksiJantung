"""QC sinyal untuk SEMUA rekaman master (fold 0-18; fold 19 hanya dengan --include-fold-19).

Jalankan dari root repo setelah ekstraksi ZIP (resumable):
    python -u -m src.dataset.run_full_qc

Aturan QC/praproses TIDAK diubah (signal_qc_core.process_record). Rekaman yang sudah lengkap di cache
tidak diunduh ulang. Hasil lokal (di-ignore git): data/interim/signal_qc_full.csv (kolom signal_qc_core.COLUMNS),
satu baris per file_name unik (ed_stay_id wakil). ed_stay_id lain yang berbagi file_name yang sama
diperluas saat membangun kontrak (lihat expand_to_all_ids).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.constants import FOLD_COL, FOLD_TEST
from src.dataset.run_signal_qc_pilot import MASTER
from src.dataset.run_signal_qc_sample import WORKERS, run_sample

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/interim/signal_qc_full.csv"


def select_records(master: pd.DataFrame, include_fold_19: bool) -> pd.DataFrame:
    """Satu baris per file_name unik; fold 19 hanya jika diminta eksplisit."""
    d = master if include_fold_19 else master[master[FOLD_COL] != FOLD_TEST]
    return d.drop_duplicates("file_name").sort_values("ed_stay_id").reset_index(drop=True)


def expand_to_all_ids(results: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """Beri hasil rekaman yang sama ke semua ed_stay_id yang berbagi file_name."""
    rep = master[["ed_stay_id", "file_name"]].merge(
        master.drop_duplicates("file_name")[["ed_stay_id", "file_name"]].rename(columns={"ed_stay_id": "rep_id"}),
        on="file_name")
    out = rep.merge(results.rename(columns={"ed_stay_id": "rep_id"}).drop(columns=["fold", "label_ami"], errors="ignore"),
                    on="rep_id", how="inner").drop(columns=["file_name", "rep_id"])
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-fold-19", action="store_true",
                    help="WAJIB agar fold 19 (test terkunci) diproses; hanya sesuai kesepakatan tim (D5)")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--output", type=Path, default=OUTPUT)
    a = ap.parse_args(argv)
    master = pd.read_parquet(MASTER, columns=["ed_stay_id", "file_name", FOLD_COL, "label_ami"])
    sample = select_records(master, a.include_fold_19)
    shared = len(master) - master["file_name"].nunique()
    print(f"Master {len(master)} | rekaman unik diproses {len(sample)} | "
          f"ed_stay_id berbagi file_name: {shared} | fold 19 {'IKUT' if a.include_fold_19 else 'TIDAK ikut'}")
    allowed = set(range(FOLD_TEST)) | ({FOLD_TEST} if a.include_fold_19 else set())
    run_sample(sample, output=a.output, workers=a.workers, allowed_folds=allowed)


if __name__ == "__main__":
    main()