"""Uji-asap pipeline training pada rekaman yang sudah di-QC (BUKAN evaluasi model).

Tujuan: memastikan Dataset -> LeadStandardizer -> ResNet1D -> loss -> backward ->
metrik berjalan ujung ke ujung dan loss turun. Data sedikit, jadi angka validasi
tidak bermakna sebagai performa. Hanya fold 0-16 yang dipakai.

Jalankan dari root repo (setelah run_signal_qc_brief):
    python -u -m src.model.train_smoke
    python -u -m src.model.train_smoke --shuffle-labels      # kontrol: label diacak
    python -u -m src.model.train_smoke --val-source folds    # validasi = fold 14-16
"""
from __future__ import annotations

import argparse
import json
import random
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

ROOT = Path(__file__).resolve().parents[2]
QC_CSV = ROOT / "data/interim/signal_qc_brief.csv"
OUT_DIR = ROOT / "data/interim"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(torch.sigmoid(model(x)).squeeze(1).numpy())
        ys.append(y.squeeze(1).numpy())
    return np.concatenate(ys).astype(int), np.concatenate(ps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle-labels", action="store_true", help="kontrol negatif: acak label training")
    ap.add_argument("--val-source", choices=["internal", "folds"], default="internal",
                    help="internal = internal_val_0_16 (sesuai brief); folds = fold 14-16 sebagai validasi")
    args = ap.parse_args()
    set_seed(args.seed)

    table = load_record_table(QC_CSV, MASTER, range(0, 17))
    if args.val_source == "internal":
        train_t, val_t = split_train_val(table)
    else:
        train_t = table[table["fold"] <= 13].reset_index(drop=True)
        val_t = table[table["fold"] >= 14].reset_index(drop=True)
    if args.shuffle_labels:
        rng = np.random.default_rng(args.seed)
        train_t["label_ami"] = rng.permutation(train_t["label_ami"].to_numpy())
    n_pos = int(train_t["label_ami"].sum())
    n_neg = len(train_t) - n_pos
    print(f"train: {len(train_t)} (positif {n_pos}) | val: {len(val_t)} (positif {int(val_t['label_ami'].sum())}) "
          f"[val={args.val_source}]" + ("  [LABEL DIACAK]" if args.shuffle_labels else ""))
    if n_pos == 0 or n_neg == 0 or int(val_t["label_ami"].sum()) == 0:
        raise SystemExit("Training dan validasi butuh kedua kelas.")

    t0 = time.time()
    train_ds = ECGDataset(train_t, CACHE, PREFIX)
    val_ds = ECGDataset(val_t, CACHE, PREFIX)
    stats = LeadStandardizer.fit(train_ds.raw_records())      # HANYA data training
    train_ds.standardizer = stats
    val_ds.standardizer = stats
    tag = ("shuffled" if args.shuffle_labels else "real") + f"_{args.val_source}"
    stats.save(OUT_DIR / f"smoke_lead_stats_{tag}.json")
    print(f"Memuat, preprocessing, dan standardisasi selesai dalam {time.time() - t0:.1f} s")

    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=gen)
    train_eval = DataLoader(train_ds, batch_size=64)
    val_loader = DataLoader(val_ds, batch_size=64)

    model = resnet1d()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    print("\nepoch | train_loss | train_auroc | val_auroc | val_prauc")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
            count += len(x)
        y_tr, p_tr = predict(model, train_eval)
        y_va, p_va = predict(model, val_loader)
        tr, va = evaluate(y_tr, p_tr), evaluate(y_va, p_va)
        history.append({"epoch": epoch, "train_loss": total / count,
                        "train_auroc": tr["roc_auc"], "val_auroc": va["roc_auc"], "val_prauc": va["pr_auc"]})
        print(f"{epoch:5d} | {total / count:10.4f} | {tr['roc_auc']:11.3f} | {va['roc_auc']:9.3f} | {va['pr_auc']:9.3f}")

    final_val = evaluate(y_va, p_va, threshold=0.5)
    out_json = OUT_DIR / f"smoke_train_metrics_{tag}.json"
    out_json.write_text(json.dumps({"args": vars(args), "history": history, "val_at_0.5": final_val}, indent=2))

    first, last = history[0], history[-1]
    print("\n=== RINGKASAN UJI-ASAP (bukan performa model) ===")
    print(f"Loss training: {first['train_loss']:.4f} -> {last['train_loss']:.4f} "
          f"({'TURUN' if last['train_loss'] < first['train_loss'] else 'TIDAK TURUN'})")
    print(f"AUROC training akhir: {last['train_auroc']:.3f} | AUROC validasi akhir: {last['val_auroc']:.3f} "
          f"(n_val={final_val['n']}, positif={final_val['n_pos']}; terlalu kecil untuk disimpulkan)")
    print("Validasi @0.5 -> tn/fp/fn/tp:", final_val["tn"], final_val["fp"], final_val["fn"], final_val["tp"])
    print("Detail lokal:", out_json.relative_to(ROOT))


if __name__ == "__main__":
    main()
