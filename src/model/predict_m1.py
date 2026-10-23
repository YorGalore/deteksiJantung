"""Prediksi M1 untuk fold 17-19 -> predictions/M1_fold{N}.parquet (kontrak PROJECT_BRIEF 12.1).

Jalankan dari root repo (setelah train_m1 dan QC fold evaluasi):
    python -u -m src.model.predict_m1 --qc-csv data/interim/signal_qc_eval.csv --folds 17 18

Kolom: ed_stay_id, subject_id, study_id, fold, y_true, y_prob (probabilitas mentah, belum di-threshold).
Skrip ini HANYA menulis prediksi; tidak menghitung metrik dan tidak mengubah model.
Fold 19 (test terkunci, brief 11.5) hanya diproses dengan --include-fold-19 secara eksplisit.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.dataset.run_signal_qc_pilot import CACHE, MASTER, PREFIX
from src.model.ecg_dataset import ECGDataset, load_record_table
from src.model.ecg_preprocessing import LeadStandardizer
from src.model.resnet1d import resnet1d
from src.model.train_smoke import load_checkpoint, pick_device, predict

ROOT = Path(__file__).resolve().parents[2]
QC_CSV = ROOT / "data/interim/signal_qc_eval.csv"
OUT_DIR = ROOT / "data/interim/m1"
PRED_DIR = ROOT / "predictions"
EVAL_FOLDS = (17, 18, 19)
COLUMNS = ["ed_stay_id", "subject_id", "study_id", "fold", "y_true", "y_prob"]


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc-csv", type=Path, default=QC_CSV)
    ap.add_argument("--folds", type=int, nargs="+", default=[17, 18])
    ap.add_argument("--include-fold-19", action="store_true",
                    help="WAJIB agar fold 19 diproses; hanya setelah rencana analisis disepakati (brief 11.5)")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--stats", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    bad = [f for f in args.folds if f not in EVAL_FOLDS]
    if bad:
        raise SystemExit(f"Hanya fold {EVAL_FOLDS} yang valid untuk prediksi evaluasi; diminta {bad}.")
    if 19 in args.folds and not args.include_fold_19:
        raise SystemExit("Fold 19 dikunci (brief 11.5). Tambahkan --include-fold-19 hanya jika rencana "
                         "analisis sudah disepakati dan di-commit.")

    device = pick_device(args.device)
    model = resnet1d().to(device)
    load_checkpoint(args.checkpoint or OUT_DIR / "m1_best.pt", model)
    stats = LeadStandardizer.load(args.stats or OUT_DIR / "lead_stats.json")
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    for fold in sorted(set(args.folds)):
        table = load_record_table(args.qc_csv, MASTER, [fold], allow_locked=True,
                                  extra_cols=("subject_id", "study_id"))
        if table.empty:
            raise SystemExit(f"Tidak ada rekaman signal_ok == 1 untuk fold {fold} di {args.qc_csv}.")
        ds = ECGDataset(table, CACHE, PREFIX, standardizer=stats)
        y_true, y_prob = predict(model, DataLoader(ds, batch_size=args.batch_size), device)
        assert len(y_prob) == len(table) and np.isfinite(y_prob).all() and ((y_prob >= 0) & (y_prob <= 1)).all()
        out = pd.DataFrame({"ed_stay_id": table["ed_stay_id"].to_numpy(),
                            "subject_id": table["subject_id"].to_numpy(),
                            "study_id": table["study_id"].to_numpy(),
                            "fold": fold, "y_true": y_true.astype("int8"), "y_prob": y_prob.astype("float64")})
        path = PRED_DIR / f"M1_fold{fold}.parquet"
        out[COLUMNS].to_parquet(path, index=False)
        print(f"fold {fold}: {len(out)} prediksi ditulis -> {path.name} (tanpa metrik)")


if __name__ == "__main__":
    main()
