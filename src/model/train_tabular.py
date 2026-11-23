"""Training dan prediksi M2 / M3 / M4 (XGBoost). Jalankan dari root repo:

    python -m src.model.train_tabular --model M2
    python -m src.model.train_tabular --model M3
    python -m src.model.train_tabular --model M4
    # nanti, HANYA setelah rencana analisis disepakati dan di-commit (brief 11.5):
    python -m src.model.train_tabular --model M2 --predict-only --folds 19 --include-fold-19

Tabel master: data/processed/master_table_v1.parquet bila ada; selain itu master_table_draft.parquet.
Bila memakai draft dan data/interim/signal_qc.csv ada, hanya baris signal_ok == 1 yang dipakai
(setara v1), supaya semua model dinilai pada kunjungan yang sama dengan M1.

Keluaran:
  predictions/{model}_fold{N}.parquet          kontrak PROJECT_BRIEF 12.1 (berisi ID, jangan di-commit)
  data/interim/tabular/{model}_model.json      model XGBoost (lokal)
  data/interim/tabular/{model}_meta.json       parameter, iterasi terbaik, metrik validasi, hash tabel master
  reports/tabular_{model}.md                   ringkasan agregat (aman di-commit)
Metrik validasi di sini memakai validasi internal fold 0-16/0-17 (prevalensi asli), BUKAN hasil akhir.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import FOLD_COL, FOLD_TEST
from src.evaluation.predictions import CONTRACT, validate_predictions
from src.model.metrics import pr_auc, roc_auc
from src.model.tabular import MODEL_SPECS, build_matrix, sample_params, split_train_val

ROOT = Path(__file__).resolve().parents[2]
MASTER_V1 = ROOT / "data/processed/master_table_v1.parquet"
MASTER_DRAFT = ROOT / "data/processed/master_table_draft.parquet"
SIGNAL_QC = ROOT / "data/interim/signal_qc.csv"
OUT_DIR = ROOT / "data/interim/tabular"
PRED_DIR = ROOT / "predictions"
REPORTS = ROOT / "reports"
MAX_TREES = 3000
EARLY_STOP = 100


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_master() -> tuple[pd.DataFrame, dict]:
    """Muat tabel master (v1 diutamakan). Kembalikan (tabel, info sumber)."""
    path = MASTER_V1 if MASTER_V1.exists() else MASTER_DRAFT
    if not path.exists():
        raise SystemExit("Tabel master tidak ditemukan di data/processed/.")
    m = pd.read_parquet(path)
    info = {"master_file": path.name, "master_sha256": sha256(path), "n_master": int(len(m)), "signal_filter": "tidak"}
    if path == MASTER_DRAFT and SIGNAL_QC.exists():
        qc = pd.read_csv(SIGNAL_QC)
        ok = set(qc.loc[qc["signal_ok"] == 1, "ed_stay_id"])
        m = m[m["ed_stay_id"].isin(ok)].copy()
        info["signal_filter"] = f"signal_ok == 1 dari signal_qc.csv ({len(m):,} baris tersisa)"
    return m.reset_index(drop=True), info


def warn_if_torch_loaded() -> None:
    """macOS: torch dan xgboost membawa libomp sendiri; di satu proses bisa segmentation fault."""
    import sys
    if sys.platform == "darwin" and "torch" in sys.modules:
        print("PERINGATAN: torch sudah termuat di proses ini. Di macOS, training XGBoost bersama torch bisa "
              "segmentation fault. Jalankan model tabular di proses/terminal terpisah (python -m ...).", flush=True)


def fit_trial(params: dict, Xtr, ytr, Xva, yva, spw: float, seed: int, n_jobs):
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        n_estimators=MAX_TREES, early_stopping_rounds=EARLY_STOP, eval_metric="aucpr",
        tree_method="hist", objective="binary:logistic", scale_pos_weight=spw,
        random_state=seed, n_jobs=n_jobs, **params)
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    best = int(clf.best_iteration)
    p = clf.predict_proba(Xva, iteration_range=(0, best + 1))[:, 1]
    return clf, {"best_iteration": best, "val_auprc": pr_auc(yva, p), "val_auroc": roc_auc(yva, p)}


def write_predictions(clf, best_iteration: int, master: pd.DataFrame, model: str, folds) -> list[str]:
    spec = MODEL_SPECS[model]
    written = []
    for f in folds:
        d = master[master[FOLD_COL] == f]
        if d.empty:
            raise SystemExit(f"Tidak ada baris fold {f} di tabel master.")
        p = clf.predict_proba(build_matrix(d, spec["features"]), iteration_range=(0, best_iteration + 1))[:, 1]
        out = pd.DataFrame({"ed_stay_id": d["ed_stay_id"].to_numpy(), "subject_id": d["subject_id"].to_numpy(),
                            "study_id": d["study_id"].to_numpy(), "fold": int(f),
                            "y_true": d["label_ami"].astype("int8").to_numpy(), "y_prob": p.astype("float64")})
        out = validate_predictions(out[CONTRACT], expected_fold=f, name=f"{model}_fold{f}")
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        path = PRED_DIR / f"{model}_fold{f}.parquet"
        out.to_parquet(path, index=False)
        written.append(f"{path.name} ({len(out):,} baris)")
    return written


def check_folds(model: str, folds, include_19: bool) -> list[int]:
    spec = MODEL_SPECS[model]
    folds = sorted(set(int(f) for f in folds))
    allowed = set(spec["pred_folds"]) | {FOLD_TEST}
    bad = [f for f in folds if f not in allowed]
    if bad:
        raise SystemExit(f"{model} tidak boleh memprediksi fold {bad} (data latih atau bukan fold evaluasi). "
                         f"Fold yang boleh: {sorted(allowed)}.")
    if FOLD_TEST in folds and not include_19:
        raise SystemExit("Fold 19 dikunci (brief 11.5). Tambahkan --include-fold-19 hanya setelah rencana analisis "
                         "disepakati dan di-commit.")
    return folds


def write_report(model: str, meta: dict, trials: pd.DataFrame) -> Path:
    t = trials.copy()
    for c in ("val_auprc", "val_auroc"):
        t[c] = t[c].map(lambda v: f"{v:.4f}")
    cols = ["trial", "max_depth", "learning_rate", "min_child_weight", "subsample", "colsample_bytree",
            "best_iteration", "val_auprc", "val_auroc"]
    header = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    rows = "\n".join("| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in t.iterrows())
    d = meta["data"]
    lines = [
        f"# {model}: {meta['desc']}", "",
        "Dihasilkan otomatis oleh `src/model/train_tabular.py`. Hanya angka agregat.", "",
        f"- Tabel master: `{d['master_file']}` (sha256 {d['master_sha256'][:12]}...) | filter sinyal: {d['signal_filter']}",
        f"- Fitur: {meta['n_features_raw']} kolom asal -> {meta['n_features_matrix']} kolom matriks (arrival_transport one-hot)",
        f"- Training: {d['n_train']:,} kunjungan ({d['n_train_pos']:,} positif) | validasi internal: {d['n_val']:,} "
        f"({d['n_val_pos']:,} positif) | scale_pos_weight = {meta['scale_pos_weight']:.2f}",
        f"- Terpilih: trial {meta['chosen_trial']} | iterasi terbaik {meta['best_iteration']} | "
        f"AUPRC validasi {meta['val_auprc']:.4f} | AUROC validasi {meta['val_auroc']:.4f}",
        "- Metrik validasi internal dipakai untuk memilih model, BUKAN hasil akhir (sedikit optimistis karena dipakai memilih).",
        "", "## Random search", "", header + rows, "",
    ]
    path = REPORTS / f"tabular_{model}.md"
    REPORTS.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def train(model: str, n_trials: int, seed: int, n_jobs, include_19: bool) -> None:
    warn_if_torch_loaded()
    import xgboost as xgb
    spec = MODEL_SPECS[model]
    master, info = load_master()
    train_df, val_df = split_train_val(master, model)
    Xtr, ytr = build_matrix(train_df, spec["features"]), train_df["label_ami"].astype(int).to_numpy()
    Xva, yva = build_matrix(val_df, spec["features"]), val_df["label_ami"].astype(int).to_numpy()
    n_pos, n_neg = int(ytr.sum()), int(len(ytr) - ytr.sum())
    spw = n_neg / n_pos
    print(f"[{model}] {spec['desc']} | master {info['master_file']} | filter sinyal: {info['signal_filter']}")
    print(f"[{model}] train {len(ytr):,} (positif {n_pos:,}) | validasi {len(yva):,} (positif {int(yva.sum()):,})"
          f" | fitur {Xtr.shape[1]} | scale_pos_weight {spw:.2f}")

    rows, best = [], None
    t0 = time.time()
    for i, params in enumerate(sample_params(n_trials, seed), start=1):
        clf, res = fit_trial(params, Xtr, ytr, Xva, yva, spw, seed, n_jobs)
        rows.append({"trial": i, **params, **res})
        print(f"  trial {i:2d}/{n_trials}: AUPRC val {res['val_auprc']:.4f} | AUROC val {res['val_auroc']:.4f}"
              f" | iterasi {res['best_iteration']} | {time.time() - t0:.0f} s", flush=True)
        if best is None or res["val_auprc"] > best[1]["val_auprc"]:
            best = (clf, res, i, params)
    clf, res, i_best, params = best
    trials = pd.DataFrame(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clf.save_model(OUT_DIR / f"{model}_model.json")
    meta = {
        "model": model, "desc": spec["desc"], "features_raw": spec["features"], "features_matrix": list(Xtr.columns),
        "n_features_raw": len(spec["features"]), "n_features_matrix": int(Xtr.shape[1]),
        "params": params, "chosen_trial": i_best, "best_iteration": res["best_iteration"],
        "val_auprc": res["val_auprc"], "val_auroc": res["val_auroc"], "scale_pos_weight": spw,
        "seed": seed, "n_trials": n_trials, "max_trees": MAX_TREES, "early_stopping_rounds": EARLY_STOP,
        "xgboost_version": xgb.__version__, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": {**info, "n_train": len(ytr), "n_train_pos": n_pos, "n_val": len(yva), "n_val_pos": int(yva.sum()),
                 "train_folds": spec["train_folds"], "val_col": spec["val_col"]},
    }
    (OUT_DIR / f"{model}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    report = write_report(model, meta, trials)

    folds = list(spec["pred_folds"]) + ([FOLD_TEST] if include_19 else [])
    written = write_predictions(clf, res["best_iteration"], master, model, check_folds(model, folds, include_19))
    print(f"\n[{model}] TERPILIH trial {i_best}: {params}")
    print(f"[{model}] iterasi terbaik {res['best_iteration']} | AUPRC val {res['val_auprc']:.4f}"
          f" | AUROC val {res['val_auroc']:.4f} (validasi internal; bukan hasil akhir)")
    print(f"[{model}] prediksi: {', '.join(written)}")
    print(f"[{model}] laporan: {report.relative_to(ROOT)} | model: {(OUT_DIR / f'{model}_model.json').relative_to(ROOT)}")


def predict_only(model: str, folds, include_19: bool) -> None:
    warn_if_torch_loaded()
    import xgboost as xgb
    meta_path, model_path = OUT_DIR / f"{model}_meta.json", OUT_DIR / f"{model}_model.json"
    if not (meta_path.exists() and model_path.exists()):
        raise SystemExit(f"Model {model} belum dilatih (tidak ada {model_path.relative_to(ROOT)}).")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    folds = check_folds(model, folds, include_19)
    master, info = load_master()
    if info["master_sha256"] != meta["data"]["master_sha256"]:
        raise SystemExit("Tabel master berbeda dari yang dipakai saat training. Latih ulang model ini dulu.")
    clf = xgb.XGBClassifier()
    clf.load_model(model_path)
    written = write_predictions(clf, int(meta["best_iteration"]), master, model, folds)
    print(f"[{model}] prediksi: {', '.join(written)}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Training/prediksi model tabular M2, M3, M4")
    ap.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=None, help="jumlah thread XGBoost (default: semua)")
    ap.add_argument("--predict-only", action="store_true", help="pakai model tersimpan, tanpa training")
    ap.add_argument("--folds", type=int, nargs="+", default=None, help="fold untuk --predict-only")
    ap.add_argument("--include-fold-19", action="store_true", help="WAJIB untuk fold 19 (brief 11.5)")
    a = ap.parse_args(argv)
    os.environ.setdefault("PYTHONHASHSEED", str(a.seed))
    if a.predict_only:
        predict_only(a.model, a.folds or MODEL_SPECS[a.model]["pred_folds"], a.include_fold_19)
    else:
        train(a.model, a.n_trials, a.seed, a.n_jobs, a.include_fold_19)


if __name__ == "__main__":
    main()
