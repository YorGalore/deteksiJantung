"""Unduh + QC fold evaluasi (17, 18, opsional 19) dengan aturan QC brief yang TETAP.

Jalankan dari root repo (resumable; jangan jalankan bersamaan dengan unduhan lain):
    python -u -m src.dataset.run_eval_qc --folds 17 18
    python -u -m src.dataset.run_eval_qc --folds 19 --include-fold-19      # hanya sesuai brief 11.5

Hasil lokal (di-ignore git): data/interim/signal_qc_eval.csv (kolom signal_qc_core.COLUMNS).
Aturan: hasil QC fold evaluasi TIDAK boleh dipakai untuk mengubah aturan QC, clipping, preprocessing,
threshold, atau model. Cukup dilaporkan sebagai angka agregat (jumlah gagal per fold dan label).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.constants import FOLD_COL
from src.dataset.run_signal_qc_pilot import MASTER
from src.dataset.run_signal_qc_sample import WORKERS, run_sample

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/interim/signal_qc_eval.csv"
EVAL_FOLDS = (17, 18, 19)


def draw_eval(master: pd.DataFrame, folds) -> pd.DataFrame:
    """Semua rekaman unik pada `folds` (hanya fold evaluasi), terurut menurut ed_stay_id."""
    folds = sorted(set(folds))
    if not folds or any(f not in EVAL_FOLDS for f in folds):
        raise SystemExit(f"Fold harus bagian dari {EVAL_FOLDS}; diminta {folds}.")
    d = master[master[FOLD_COL].isin(folds)].drop_duplicates("file_name")
    assert set(d[FOLD_COL].unique()) <= set(folds)
    return d.sort_values("ed_stay_id").reset_index(drop=True)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[17, 18])
    ap.add_argument("--include-fold-19", action="store_true",
                    help="WAJIB agar fold 19 (test terkunci) diproses; hanya setelah disepakati tim")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--output", type=Path, default=OUTPUT)
    return ap.parse_args(argv)


def main(argv=None) -> None:
    a = parse_args(argv)
    if 19 in a.folds and not a.include_fold_19:
        raise SystemExit("Fold 19 dikunci (brief 11.5). Tambahkan --include-fold-19 hanya jika disepakati tim.")
    master = pd.read_parquet(MASTER, columns=["ed_stay_id", "file_name", FOLD_COL, "label_ami"])
    sample = draw_eval(master, a.folds)
    run_sample(sample, output=a.output, workers=a.workers, allowed_folds=set(a.folds))


if __name__ == "__main__":
    main()
