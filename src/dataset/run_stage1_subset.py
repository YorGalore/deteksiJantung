"""Tahap 1: subset kecil untuk menguji pipeline (loader, QC, preprocessing, training, checkpoint).

Jalankan dari root repo (bisa dihentikan dengan Ctrl+C dan dilanjutkan):
    python -u -m src.dataset.run_stage1_subset --n-pos 300 --n-neg 700
    python -u -m src.dataset.run_stage1_subset --n-pos 4134 --n-neg 12402 \\
        --output data/interim/signal_qc_stage2a.csv          # tahap 2a (seed 42, negatif 3x)

HANYA fold 0-16. Sampel = 300 rekaman sampel lama (sudah di cache) + tambahan acak (seed tetap)
sampai jumlah yang diminta. Ukuran ini angka awal untuk pengujian, BUKAN ukuran final.
Hasil lokal (di-ignore git): data/interim/signal_qc_stage1.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.constants import FOLD_COL, FOLD_META
from src.dataset.run_signal_qc_pilot import MASTER
from src.dataset.run_signal_qc_sample import N_NEG as BASE_NEG
from src.dataset.run_signal_qc_sample import N_POS as BASE_POS
from src.dataset.run_signal_qc_sample import SEED, WORKERS, draw_sample, run_sample

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/interim/signal_qc_stage1.csv"


def draw_stage1(master: pd.DataFrame, n_pos: int, n_neg: int, seed: int = SEED) -> pd.DataFrame:
    """Sampel acak bertingkat dari fold 0-16 yang MEMUAT sampel lama (hemat unduhan)."""
    if n_pos < BASE_POS or n_neg < BASE_NEG:
        raise ValueError(f"n_pos >= {BASE_POS} dan n_neg >= {BASE_NEG} (sampel lama harus termuat)")
    d = master[master[FOLD_COL] < FOLD_META].drop_duplicates("file_name")
    base = draw_sample(master)
    rest = d[~d["ed_stay_id"].isin(base["ed_stay_id"])]
    extra_pos = rest[rest["label_ami"] == 1].sample(n_pos - BASE_POS, random_state=seed)
    extra_neg = rest[rest["label_ami"] == 0].sample(n_neg - BASE_NEG, random_state=seed)
    out = pd.concat([base, extra_pos, extra_neg]).sort_values("ed_stay_id").reset_index(drop=True)
    assert (out[FOLD_COL] < FOLD_META).all() and out["file_name"].is_unique
    return out


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pos", type=int, default=300)
    ap.add_argument("--n-neg", type=int, default=700)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--output", type=Path, default=OUTPUT,
                    help="berkas hasil QC; pakai nama berbeda untuk tiap tahap agar tidak tercampur")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    a = parse_args(argv)
    master = pd.read_parquet(MASTER, columns=["ed_stay_id", "file_name", FOLD_COL, "label_ami"])
    run_sample(draw_stage1(master, a.n_pos, a.n_neg), output=a.output, workers=a.workers)


if __name__ == "__main__":
    main()
