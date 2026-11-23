"""Training meta-learner M5 / M6 di fold 17 dan prediksi fold 18. Jalankan dari root repo:

    python -m src.model.train_meta --model M5
    python -m src.model.train_meta --model M6
    # nanti, HANYA setelah rencana analisis disepakati dan prediksi fold 19 model dasar tersedia (brief 11.5):
    python -m src.model.train_meta --model M5 --predict-only --folds 19 --include-fold-19

Masukan : predictions/{M1,M2,M3}_fold17.parquet (latih) dan _fold18.parquet (prediksi).
Keluaran: predictions/{model}_fold18.parquet       kontrak 12.1 (berisi ID, jangan di-commit)
          data/interim/meta/{model}_meta.json        parameter meta-learner (lokal)
          reports/meta_{model}.md                    koefisien + CI bootstrap (agregat, aman di-commit)
Fold 17 TIDAK diprediksi (data latih meta-learner). Tidak membaca fold 19 tanpa --include-fold-19.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import FOLD_META, FOLD_TEST, FOLD_THRESHOLD
from src.evaluation import predictions as P
from src.evaluation.predictions import CONTRACT, align, validate_predictions
from src.model.meta_learner import C_DEFAULT, META_SPECS, MetaModel, bootstrap_coefficients
from src.model.metrics import pr_auc, roc_auc

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data/interim/meta"
REPORTS = ROOT / "reports"
MIN_POS_FOLD17 = 100          # rencana proyek 3.4: < 100 positif di fold 17 -> perlu cross-fitting


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_inputs(model: str, fold: int, allow_test: bool = False) -> tuple[pd.DataFrame, dict]:
    """Gabungkan prediksi model dasar pada kunjungan yang sama; study_id juga harus sama antar-input."""
    inputs = META_SPECS[model]["inputs"]
    frames = {m: P.load_predictions(m, fold, pred_dir=P.PRED_DIR, allow_test=allow_test) for m in inputs}
    wide, info = align(frames)
    studies = [frames[m].set_index("ed_stay_id")["study_id"].reindex(wide["ed_stay_id"]).to_numpy() for m in inputs]
    for m, s in zip(inputs[1:], studies[1:]):
        if not np.array_equal(s, studies[0]):
            raise ValueError(f"study_id berbeda antara {inputs[0]} dan {m} (EKG pertama tidak sama).")
    wide["study_id"] = studies[0]
    return wide, info


def write_predictions(model: str, meta: MetaModel, fold: int, allow_test: bool) -> str:
    wide, info = load_inputs(model, fold, allow_test)
    p = meta.predict(wide)
    out = pd.DataFrame({"ed_stay_id": wide["ed_stay_id"], "subject_id": wide["subject_id"],
                        "study_id": wide["study_id"], "fold": int(fold),
                        "y_true": wide["y_true"].astype("int8"), "y_prob": p})
    out = validate_predictions(out[CONTRACT], expected_fold=fold, name=f"{model}_fold{fold}")
    P.PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = P.PRED_DIR / f"{model}_fold{fold}.parquet"
    out.to_parquet(path, index=False)
    dropped = {k: v for k, v in info["dropped_per_model"].items() if v}
    return f"{path.name} ({len(out):,} baris" + (f"; dibuang saat join: {dropped})" if dropped else ")")


def write_report(model: str, d: dict, coefs: pd.DataFrame) -> Path:
    t = coefs.copy()
    for c in ("estimate", "ci_low", "ci_high", "odds_ratio_per_sd"):
        t[c] = t[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    rows = "\n".join(f"| {r.term} | {r.estimate} | [{r.ci_low}, {r.ci_high}] | {r.odds_ratio_per_sd} |"
                     for r in t.itertuples())
    std = coefs[coefs.term != "intercept"]
    share = (std["estimate"].abs() / std["estimate"].abs().sum()).round(3)
    lines = [
        f"# {model}: {d['desc']}", "",
        "Dihasilkan otomatis oleh `src/model/train_meta.py`. Hanya angka agregat.", "",
        f"- Dilatih di fold {FOLD_META}: {d['n']:,} kunjungan ({d['n_pos']:,} positif); input: "
        + ", ".join(f"logit p_{m}" for m in d["inputs"]) + " (distandarkan)",
        f"- Regresi logistik L2, C = {d['C']}; bootstrap per subject_id {d['n_boot']}x (fit ulang di setiap sampel)",
        f"- Korelasi antar-input (logit) di fold 17: {d['input_corr']:.3f}",
        f"- AUROC/AUPRC in-sample fold 17 (hanya cek kewajaran, bukan hasil): {d['auroc_fold17']:.3f} / {d['auprc_fold17']:.3f}",
        "", "| term | koefisien (per 1 SD) | CI 95% | odds ratio per 1 SD |", "|---|---|---|---|", rows, "",
        "Porsi |koefisien terstandar| (gambaran bobot relatif): "
        + ", ".join(f"{term}: {s}" for term, s in zip(std["term"], share)),
        "",
        "Catatan tafsir: input saling berkorelasi, jadi porsi bobot bukan dekomposisi kausal. Koefisien menjawab: "
        "seberapa besar log-odds AMI berubah per 1 SD logit model dasar, dengan input lain tetap.", "",
    ]
    path = REPORTS / f"meta_{model}.md"
    REPORTS.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def train(model: str, n_boot: int, seed: int, C: float) -> None:
    spec = META_SPECS[model]
    wide, info = load_inputs(model, FOLD_META)
    y = wide["y_true"].to_numpy()
    n_pos = int(y.sum())
    print(f"[{model}] {spec['desc']}")
    print(f"[{model}] fold {FOLD_META}: {len(wide):,} kunjungan bersama ({n_pos} positif) | per model: {info['n_per_model']}")
    if n_pos < MIN_POS_FOLD17:
        raise SystemExit(f"Fold 17 hanya {n_pos} positif (< {MIN_POS_FOLD17}); rencana 3.4 meminta cross-fitting.")

    meta = MetaModel.fit(wide, spec["inputs"], C)
    p17 = meta.predict(wide)
    X = MetaModel.design(wide, spec["inputs"])
    print(f"[{model}] bootstrap koefisien {n_boot}x ...", flush=True)
    coefs = bootstrap_coefficients(wide, spec["inputs"], n_boot=n_boot, seed=seed, C=C)
    d = {"model": model, "desc": spec["desc"], "inputs": spec["inputs"], "n": int(len(wide)), "n_pos": n_pos,
         "C": C, "n_boot": n_boot, "seed": seed, "input_corr": float(np.corrcoef(X.T)[0, 1]),
         "auroc_fold17": roc_auc(y, p17), "auprc_fold17": pr_auc(y, p17),
         "input_sha256_fold17": {m: sha256(P.PRED_DIR / f"{m}_fold{FOLD_META}.parquet") for m in spec["inputs"]},
         "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    meta.extra = {k: v for k, v in d.items() if k not in ("inputs", "C")}
    meta.extra["coef_ci"] = coefs.to_dict(orient="records")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{model}_meta.json").write_text(json.dumps(meta.to_dict(), indent=2, default=float), encoding="utf-8")
    report = write_report(model, d, coefs)
    written = write_predictions(model, meta, FOLD_THRESHOLD, allow_test=False)

    print(f"[{model}] korelasi input (logit) fold 17: {d['input_corr']:.3f}")
    for r in coefs.itertuples():
        orr = "" if pd.isna(r.odds_ratio_per_sd) else f" | OR per SD {r.odds_ratio_per_sd:.2f}"
        print(f"[{model}]   {r.term:14s} {r.estimate:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]{orr}")
    print(f"[{model}] prediksi: {written}")
    print(f"[{model}] laporan: {report.relative_to(ROOT)} | parameter: {(OUT_DIR / f'{model}_meta.json').relative_to(ROOT)}")


def predict_only(model: str, folds, include_19: bool) -> None:
    path = OUT_DIR / f"{model}_meta.json"
    if not path.exists():
        raise SystemExit(f"Meta-learner {model} belum dilatih.")
    meta = MetaModel.from_dict(json.loads(path.read_text(encoding="utf-8")))
    for f in sorted(set(int(x) for x in folds)):
        if f == FOLD_META:
            raise SystemExit("Fold 17 adalah data latih meta-learner; prediksinya tidak sah.")
        if f not in (FOLD_THRESHOLD, FOLD_TEST):
            raise SystemExit(f"Fold {f} bukan fold evaluasi meta-learner (hanya 18 dan 19).")
        if f == FOLD_TEST and not include_19:
            raise SystemExit("Fold 19 dikunci (brief 11.5). Tambahkan --include-fold-19 hanya setelah rencana "
                             "analisis disepakati dan di-commit.")
        print(f"[{model}] prediksi: {write_predictions(model, meta, f, allow_test=(f == FOLD_TEST))}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Meta-learner late fusion M5/M6 (fold 17)")
    ap.add_argument("--model", required=True, choices=sorted(META_SPECS))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--C", type=float, default=C_DEFAULT)
    ap.add_argument("--predict-only", action="store_true")
    ap.add_argument("--folds", type=int, nargs="+", default=[FOLD_THRESHOLD])
    ap.add_argument("--include-fold-19", action="store_true")
    a = ap.parse_args(argv)
    if a.predict_only:
        predict_only(a.model, a.folds, a.include_fold_19)
    else:
        train(a.model, a.n_boot, a.seed, a.C)


if __name__ == "__main__":
    main()
