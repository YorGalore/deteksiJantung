"""QC sinyal sesuai PROJECT_BRIEF pada sampel yang sama, TANPA mengunduh apa pun.

Jalankan dari root repo:
    python -u -m src.dataset.run_signal_qc_brief

Hanya memproses rekaman yang sudah lengkap di cache (ukuran .dat tervalidasi);
sisanya berstatus not_cached. Aturan: gagal jika ada lead dengan > 20% NaN atau
>= 3 lead datar. Amplitudo > 20 mV hanya menjadi penanda (amplitude_flag).
Hasil lokal (di-ignore git): data/interim/signal_qc_brief.csv
Jika data/interim/signal_qc_sample.csv (QC lama) ada, ditampilkan perbandingannya.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import FOLD_COL, FOLD_META
from src.dataset.ecg_download import is_valid_record
from src.dataset.ecg_loader import read_ecg_record
from src.dataset.run_signal_qc_pilot import CACHE, MASTER
from src.dataset.run_signal_qc_sample import draw_sample, relative_path
from src.model.ecg_preprocessing import prepare_ecg
from src.model.ecg_quality import check_ecg_quality_brief

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/interim/signal_qc_brief.csv"
OLD_OUTPUT = ROOT / "data/interim/signal_qc_sample.csv"


def process(row) -> dict:
    rec = {
        "ed_stay_id": row.ed_stay_id,
        "fold": getattr(row, FOLD_COL),
        "label_ami": row.label_ami,
        "status": "",
        "signal_ok": 0,
        "qc_reason": "",
        "amplitude_flag": 0,
        "max_nan_fraction": np.nan,
        "n_flat_leads": np.nan,
        "max_abs": np.nan,
        "error": "",
    }
    base = CACHE / relative_path(row.file_name)
    if not is_valid_record(base):
        rec["status"] = "not_cached"
        return rec

    status = "load_fail"
    try:
        sig, fs, names = read_ecg_record(base)
        qc = check_ecg_quality_brief(sig)
        finite = np.isfinite(sig)
        rec.update(
            qc_reason=qc["qc_reason"],
            amplitude_flag=qc["amplitude_flag"],
            max_nan_fraction=qc["max_nan_fraction"],
            n_flat_leads=qc["n_flat_leads"],
            max_abs=float(np.abs(sig[finite]).max()) if finite.any() else np.nan,
        )
        status = "qc_fail"
        if qc["signal_ok"] != 1:
            rec["status"] = status
            return rec

        status = "preprocess_fail"
        x = prepare_ecg(sig, fs, names)
        if x.shape != (12, 1000) or not np.isfinite(x).all():
            raise ValueError(f"bad_preprocessed_output:{x.shape}")
        rec["status"] = "ok"
        rec["signal_ok"] = 1
    except Exception as exc:
        rec["status"] = status
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:100]}"
    return rec


def summarize(res: pd.DataFrame) -> None:
    res = res.fillna({"qc_reason": "", "error": ""})
    print("\n=== RINGKASAN QC VERSI BRIEF (agregat saja) ===")
    print("Rekaman:", len(res))
    print("\nStatus:")
    print(res["status"].value_counts().to_string())
    print("\nAlasan QC:")
    print(res["qc_reason"].replace("", "(tidak sampai QC)").value_counts().to_string())
    ok = res[res["signal_ok"] == 1]
    print(f"\nLolos QC dan preprocessing: {len(ok)} | dengan amplitude_flag (>20 mV): {int(ok['amplitude_flag'].sum())}")
    print("\nLolos per label (0=negatif, 1=AMI):")
    print(res.groupby("label_ami")["signal_ok"].agg(n="size", lolos="sum", rate="mean").round(3).to_string())
    if OLD_OUTPUT.exists():
        old = pd.read_csv(OLD_OUTPUT)[["ed_stay_id", "status"]].rename(columns={"status": "status_lama"})
        both = res.merge(old, on="ed_stay_id", how="left").fillna({"status_lama": "(tidak ada)"})
        print("\nPerbandingan status lama (baris) vs versi brief (kolom):")
        print(pd.crosstab(both["status_lama"], both["status"]).to_string())


def main() -> None:
    master = pd.read_parquet(MASTER, columns=["ed_stay_id", "file_name", FOLD_COL, "label_ami"])
    sample = draw_sample(master)
    assert (sample[FOLD_COL] < FOLD_META).all(), "Sampel menyentuh fold >= 17!"

    rows = [process(r) for r in sample.itertuples(index=False)]
    res = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUTPUT, index=False)
    summarize(res)
    print("\nHasil lokal:", OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
