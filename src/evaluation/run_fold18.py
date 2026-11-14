"""Kalibrasi, threshold, dan evaluasi pengembangan di FOLD 18 (PROJECT_BRIEF 11.3, 11.5).

Jalankan dari root repo:
    python -m src.evaluation.run_fold18 --models M1
    python -m src.evaluation.run_fold18 --models M1 M2 --compare M1:M2      # nanti, bila M2 sudah ada

Untuk setiap model:
  1. Baca predictions/{model}_fold18.parquet (kontrak 12.1).
  2. Metrik tanpa threshold pada probabilitas mentah: AUROC, AUPRC, Brier, intercept/slope kalibrasi.
  3. Fit kalibrasi Platt di fold 18 -> probabilitas terkalibrasi.
  4. Threshold rule-out (sensitivitas >= 0,90, terbesar) dan rule-in (spesifisitas >= 0,90, terkecil).
  5. Bootstrap per subject_id (CI 95%), berpasangan bila ada --compare.
Keluaran (hanya angka agregat, aman di-commit):
  reports/eval_fold18.md        ringkasan
  reports/locked_fold18.json    parameter kalibrasi + threshold yang DIKUNCI untuk fold 19

PENTING: kalibrasi dan threshold dipilih DAN dinilai di fold 18 yang sama, jadi metrik bergantung
threshold di sini (mis. sensitivitas ~0,90) bersifat in-sample. Ini untuk menguji kode dan mengunci
parameter, BUKAN hasil akhir. Hasil akhir hanya dari fold 19, dijalankan satu kali.
Fold 19 tidak dibaca oleh skrip ini.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import FOLD_THRESHOLD
from src.evaluation.bootstrap import paired_bootstrap
from src.evaluation.core import (
    PlattCalibrator, binary_metrics, brier, calibration_table, summarize,
    threshold_for_sensitivity, threshold_for_specificity,
)
from src.evaluation.predictions import ROOT, align, load_predictions, pred_path
from src.model.metrics import pr_auc, roc_auc

TARGET = 0.90
MIN_REPORT_COUNT = 10        # rencana proyek Bagian 2.5: jangan laporkan sel < 10
REPORT = ROOT / "reports" / "eval_fold18.md"
LOCKED = ROOT / "reports" / "locked_fold18.json"


def cnt(n: int) -> str:
    return f"{n:,}" if n >= MIN_REPORT_COUNT else f"<{MIN_REPORT_COUNT}"


def f3(x) -> str:
    return "NaN" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def evaluate_model(model: str, df: pd.DataFrame, n_boot: int, seed: int) -> dict:
    y, p = df["y_true"].to_numpy(), df["y_prob"].to_numpy()
    raw = summarize(y, p)
    cal = PlattCalibrator.fit(y, p)
    pc = cal.transform(p)
    thr_out = threshold_for_sensitivity(y, p, TARGET)
    thr_in = threshold_for_specificity(y, p, TARGET)
    m_out, m_in = binary_metrics(y, p, thr_out), binary_metrics(y, p, thr_in)

    metrics = {
        "AUROC": lambda yy, pp, _m: roc_auc(yy, pp),
        "AUPRC": lambda yy, pp, _m: pr_auc(yy, pp),
        "Brier_kalibrasi": lambda yy, pp, _m: brier(yy, cal.transform(pp)),
        "Sens@rule-out": lambda yy, pp, _m: binary_metrics(yy, pp, thr_out)["sensitivity"],
        "Spes@rule-out": lambda yy, pp, _m: binary_metrics(yy, pp, thr_out)["specificity"],
        "NPV@rule-out": lambda yy, pp, _m: binary_metrics(yy, pp, thr_out)["npv"],
        "PPV@rule-out": lambda yy, pp, _m: binary_metrics(yy, pp, thr_out)["ppv"],
    }
    boot = paired_bootstrap(y, {model: p}, df["subject_id"].to_numpy(), metrics, n_boot=n_boot, seed=seed)
    return {"raw": raw, "brier_cal": brier(y, pc), "cal": cal, "thr_out": thr_out, "thr_in": thr_in,
            "m_out": m_out, "m_in": m_in, "boot": boot,
            "cal_table_raw": calibration_table(y, p), "cal_table_cal": calibration_table(y, pc)}


def locked_entry(model: str, df: pd.DataFrame, r: dict, path: Path) -> dict:
    cal: PlattCalibrator = r["cal"]
    return {
        "fold": FOLD_THRESHOLD, "n": int(len(df)), "n_pos": int(df["y_true"].sum()),
        "predictions_sha256": sha256(path),
        "calibration": cal.to_dict(),
        "threshold_rule_out": {"target_sensitivity": TARGET, "y_prob_raw": r["thr_out"],
                               "y_prob_calibrated": float(cal.transform([r["thr_out"]])[0])},
        "threshold_rule_in": {"target_specificity": TARGET, "y_prob_raw": r["thr_in"],
                              "y_prob_calibrated": float(cal.transform([r["thr_in"]])[0])},
        "rule": "positif bila y_prob_raw >= threshold (setara dengan skala terkalibrasi karena Platt monoton naik)",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def model_section(model: str, r: dict) -> list[str]:
    raw, cal = r["raw"], r["cal"]
    b = r["boot"].set_index("metric")
    ci = lambda k: f"{f3(b.loc[k, 'estimate'])} [{f3(b.loc[k, 'ci_low'])}, {f3(b.loc[k, 'ci_high'])}]"
    thr_rows = []
    for name, m in (("Rule-out (sens >= 0,90)", r["m_out"]), ("Rule-in (spes >= 0,90)", r["m_in"])):
        thr_rows.append({"threshold": name, "y_prob_raw": f"{m['threshold']:.6g}",
                         "sens": f3(m["sensitivity"]), "spes": f3(m["specificity"]), "PPV": f3(m["ppv"]),
                         "NPV": f3(m["npv"]), "F1": f3(m["f1"]), "TP": cnt(m["tp"]), "FP": cnt(m["fp"]),
                         "FN": cnt(m["fn"]), "TN": cnt(m["tn"])})
    ct = pd.DataFrame(r["cal_table_cal"])[["bin", "n", "mean_pred", "observed"]].round(4)
    return [
        f"## {model}", "",
        f"- Kunjungan: {raw['n']:,} | positif: {cnt(raw['n_pos'])} | prevalensi: {raw['prevalence'] * 100:.2f}%"
        f" (garis dasar AUPRC = prevalensi)",
        f"- AUROC: {ci('AUROC')} | AUPRC: {ci('AUPRC')}  (CI 95% bootstrap per subject_id)",
        f"- Probabilitas mentah: Brier {f3(raw['brier'])}, intercept kalibrasi {f3(raw['calibration_intercept'])},"
        f" slope {f3(raw['calibration_slope'])} (ideal 0 dan 1)",
        f"- Platt (fit di fold 18): a = {cal.a:.4f}, b = {cal.b:.4f} | Brier setelah kalibrasi {f3(r['brier_cal'])}"
        " (in-sample)", "",
        "**Threshold (dipilih di fold 18; metrik di bawah in-sample, bukan hasil akhir)**", "",
        md_table(pd.DataFrame(thr_rows)), "",
        f"CI bootstrap @rule-out: sens {ci('Sens@rule-out')}, spes {ci('Spes@rule-out')}, "
        f"NPV {ci('NPV@rule-out')}, PPV {ci('PPV@rule-out')}", "",
        "**Tabel kalibrasi setelah Platt (10 bin berisi jumlah kunjungan sama)**", "",
        md_table(ct), "",
    ]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Kalibrasi + threshold + evaluasi pengembangan di fold 18")
    ap.add_argument("--models", nargs="+", required=True, help="kode model, mis. M1 M2")
    ap.add_argument("--compare", nargs="*", default=[], help="pasangan A:B untuk selisih berpasangan, mis. M5:M4")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    frames, results = {}, {}
    for m in a.models:
        frames[m] = load_predictions(m, FOLD_THRESHOLD)
        print(f"[{m}] fold 18: {len(frames[m]):,} kunjungan, bootstrap {a.n_boot}x ...", flush=True)
        results[m] = evaluate_model(m, frames[m], a.n_boot, a.seed)

    lines = ["# Evaluasi pengembangan di fold 18", "",
             "Dihasilkan otomatis oleh `src/evaluation/run_fold18.py`. Hanya angka agregat.", "",
             "> Kalibrasi dan threshold dipilih DAN dinilai di fold 18. Metrik bergantung threshold di sini "
             "bersifat in-sample. Hasil akhir hanya dari fold 19 (sekali jalan) dengan parameter di "
             "`reports/locked_fold18.json`.", ""]
    for m in a.models:
        lines += model_section(m, results[m])

    pairs = [tuple(x.split(":")) for x in a.compare]
    if pairs:
        wide, info = align({m: frames[m] for m in sorted({x for p in pairs for x in p})})
        probs = {c[2:]: wide[c].to_numpy() for c in wide.columns if c.startswith("p_")}
        metrics = {"AUROC": lambda yy, pp, _m: roc_auc(yy, pp), "AUPRC": lambda yy, pp, _m: pr_auc(yy, pp)}
        comp = paired_bootstrap(wide["y_true"].to_numpy(), probs, wide["subject_id"].to_numpy(), metrics,
                                n_boot=a.n_boot, seed=a.seed, diffs=pairs)
        comp = comp[comp["target"].str.contains(" - ", regex=False)].copy()
        for c in ("estimate", "ci_low", "ci_high"):
            comp[c] = comp[c].map(f3)
        lines += ["## Perbandingan berpasangan (kunjungan yang sama)", "",
                  f"Kunjungan yang dinilai bersama: {info['n_aligned']:,} | dibuang per model: {info['dropped_per_model']}",
                  "", md_table(comp[["target", "metric", "estimate", "ci_low", "ci_high", "ci_excludes_0"]]), ""]
        print("\n" + comp[["target", "metric", "estimate", "ci_low", "ci_high", "ci_excludes_0"]].to_string(index=False))

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    locked = json.loads(LOCKED.read_text(encoding="utf-8")) if LOCKED.exists() else {}
    for m in a.models:
        locked[m] = locked_entry(m, frames[m], results[m], pred_path(m, FOLD_THRESHOLD))
    LOCKED.write_text(json.dumps(locked, indent=2), encoding="utf-8")

    print("\n=== RINGKASAN FOLD 18 (pengembangan; metrik threshold in-sample) ===")
    for m in a.models:
        r = results[m]
        b = r["boot"].set_index("metric")
        print(f"{m}: n={r['raw']['n']:,} positif={cnt(r['raw']['n_pos'])} prevalensi={r['raw']['prevalence'] * 100:.2f}%")
        print(f"  AUROC {f3(b.loc['AUROC', 'estimate'])} [{f3(b.loc['AUROC', 'ci_low'])}, {f3(b.loc['AUROC', 'ci_high'])}]"
              f" | AUPRC {f3(b.loc['AUPRC', 'estimate'])} [{f3(b.loc['AUPRC', 'ci_low'])}, {f3(b.loc['AUPRC', 'ci_high'])}]")
        print(f"  Mentah: Brier {f3(r['raw']['brier'])}, intercept {f3(r['raw']['calibration_intercept'])},"
              f" slope {f3(r['raw']['calibration_slope'])} | Platt a={r['cal'].a:.4f} b={r['cal'].b:.4f},"
              f" Brier terkalibrasi {f3(r['brier_cal'])}")
        for name, mm in (("rule-out", r["m_out"]), ("rule-in ", r["m_in"])):
            print(f"  {name}: thr={mm['threshold']:.6g} sens={f3(mm['sensitivity'])} spes={f3(mm['specificity'])}"
                  f" PPV={f3(mm['ppv'])} NPV={f3(mm['npv'])} F1={f3(mm['f1'])}")
    print(f"\nTersimpan: {REPORT.relative_to(ROOT)} dan {LOCKED.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
