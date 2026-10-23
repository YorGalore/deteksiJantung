"""Training M1 (ResNet-1D) sesuai PROJECT_BRIEF 7.2 dan 10.3-10.4. HANYA fold 0-16.

Jalankan dari root repo:
    python -u -m src.model.train_m1 --qc-csv data/interim/signal_qc_stage2a.csv

- Train = fold <= 16 & internal_val_0_16 == 0; validasi = internal_val_0_16 == 1 (early stopping, AUPRC).
- Statistik standardisasi per lead dihitung dari data training saja dan disimpan.
- Loss BCEWithLogits dengan pos_weight = n_neg/n_pos pada data training (setelah subsampling).
- Fold 17-19 TIDAK dibaca di sini (load_record_table menolaknya).
Keluaran lokal (di-ignore git): data/interim/m1/{m1_best.pt, lead_stats.json, m1_history.json}

Catatan: dengan negatif disubsampel, y_prob TIDAK terkalibrasi terhadap prevalensi asli; kalibrasi dan
threshold dilakukan di fold 18 oleh Orang 3. AUPRC validasi di sini hanya untuk early stopping.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset.run_signal_qc_pilot import CACHE, MASTER, PREFIX
from src.model.ecg_dataset import ECGDataset, load_record_table, split_train_val
from src.model.ecg_preprocessing import LeadStandardizer
from src.model.metrics import evaluate
from src.model.resnet1d import resnet1d
from src.model.train_smoke import load_checkpoint, pick_device, predict, save_checkpoint, set_seed  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
QC_CSV = ROOT / "data/interim/signal_qc_stage2a.csv"
OUT_DIR = ROOT / "data/interim/m1"


class EarlyStopper:
    """Skor lebih besar = lebih baik. update() -> True jika membaik lebih dari min_delta."""

    def __init__(self, patience: int, min_delta: float = 1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.bad = -np.inf, 0

    def update(self, score: float) -> bool:
        if np.isfinite(score) and score > self.best + self.min_delta:
            self.best, self.bad = float(score), 0
            return True
        self.bad += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad >= self.patience


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc-csv", type=Path, default=QC_CSV)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)
    device = pick_device(args.device)
    print("device:", device, f"({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")

    table = load_record_table(args.qc_csv, MASTER, range(0, 17))
    train_t, val_t = split_train_val(table)
    n_pos = int(train_t["label_ami"].sum())
    n_neg = len(train_t) - n_pos
    n_pos_val = int(val_t["label_ami"].sum())
    print(f"train: {len(train_t)} (positif {n_pos}) | val: {len(val_t)} (positif {n_pos_val})")
    if n_pos == 0 or n_neg == 0 or n_pos_val == 0 or n_pos_val == len(val_t):
        raise SystemExit("Training dan validasi butuh kedua kelas.")

    t0 = time.time()
    train_ds = ECGDataset(train_t, CACHE, PREFIX)
    val_ds = ECGDataset(val_t, CACHE, PREFIX)
    stats = LeadStandardizer.fit(train_ds.raw_records())          # HANYA data training
    train_ds.standardizer = stats
    val_ds.standardizer = stats
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats_path = OUT_DIR / "lead_stats.json"
    stats.save(stats_path)
    print(f"Memuat dan preprocessing selesai dalam {time.time() - t0:.0f} s")

    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=128)

    model = resnet1d().to(device)
    pos_weight = n_neg / n_pos
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    stopper = EarlyStopper(args.patience)
    ckpt_path = OUT_DIR / "m1_best.pt"
    history, best_epoch = [], 0

    print(f"pos_weight = {pos_weight:.2f}\n\nepoch | train_loss | val_auroc | val_auprc | simpan")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
            count += len(x)
        y_va, p_va = predict(model, val_loader, device)
        va = evaluate(y_va, p_va)
        improved = stopper.update(va["pr_auc"])
        if improved:
            best_epoch = epoch
            save_checkpoint(ckpt_path, model, epoch, vars(args), {
                "val_pr_auc": va["pr_auc"], "val_roc_auc": va["roc_auc"], "pos_weight": pos_weight,
                "n_train": len(train_t), "n_pos_train": n_pos, "stats_file": stats_path.name})
        history.append({"epoch": epoch, "train_loss": total / count, "val_roc_auc": va["roc_auc"],
                        "val_pr_auc": va["pr_auc"], "saved": improved})
        print(f"{epoch:5d} | {total / count:10.4f} | {va['roc_auc']:9.3f} | {va['pr_auc']:9.3f} | {'*' if improved else ''}")
        if stopper.should_stop:
            print(f"Early stopping: tidak membaik {stopper.patience} epoch berturut-turut.")
            break

    (OUT_DIR / "m1_history.json").write_text(json.dumps(
        {"args": vars(args), "best_epoch": best_epoch, "best_val_pr_auc": stopper.best, "history": history},
        indent=2, default=str))
    print(f"\nCheckpoint terbaik: epoch {best_epoch} (AUPRC validasi {stopper.best:.3f}) -> {ckpt_path}")
    print("AUPRC validasi dihitung pada validasi yang diperkaya positif; BUKAN performa pada prevalensi asli.")


if __name__ == "__main__":
    main()
