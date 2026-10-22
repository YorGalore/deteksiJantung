"""QC bertahap pada sampel kecil (hanya fold 0-16) dengan metrik amplitudo.

Jalankan dari root repo:
    python -u -m src.dataset.run_signal_qc_sample

Output lokal (di-ignore git): data/interim/signal_qc_sample.csv
JANGAN ganti namanya menjadi signal_qc.csv: build_master membuang semua baris
yang tidak ada di berkas itu. Fold 17-19 sengaja tidak disentuh (19 = test).

Skrip bisa dilanjutkan. Baris berstatus download_fail / load_fail dari run
sebelumnya dihapus dan dicoba lagi; rekaman yang sudah lengkap di cache tidak
diunduh ulang. Unduhan memakai paling banyak WORKERS koneksi paralel.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import FOLD_COL, FOLD_META
from src.dataset.ecg_download import fetch_record
from src.dataset.ecg_loader import load_ecg_record
from src.dataset.run_signal_qc_pilot import BASE_URL, CACHE, MASTER, PREFIX
from src.model.ecg_preprocessing import preprocess_ecg
from src.model.ecg_quality import check_ecg_quality

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/interim/signal_qc_sample.csv"
SEED = 42
N_POS = 100
N_NEG = 200
WORKERS = 8          # jangan dinaikkan: server publik
RETRY_STATUSES = {"download_fail", "load_fail"}


def draw_sample(master: pd.DataFrame) -> pd.DataFrame:
    """Sampel acak bertingkat dari fold 0-16: N_POS positif + N_NEG negatif."""
    d = master[master[FOLD_COL] < FOLD_META].drop_duplicates("file_name")
    pos = d[d["label_ami"] == 1].sample(N_POS, random_state=SEED)
    neg = d[d["label_ami"] == 0].sample(N_NEG, random_state=SEED)
    return pd.concat([pos, neg]).sort_values("ed_stay_id").reset_index(drop=True)


def relative_path(file_name) -> str:
    return str(file_name).lstrip("/").removeprefix(PREFIX)


def amplitude_metrics(sig: np.ndarray) -> dict:
    a = np.abs(sig)
    lead_max = a.max(axis=0)
    return {
        "max_abs": float(a.max()),
        "p999_abs": float(np.percentile(a, 99.9)),
        "worst_lead": int(lead_max.argmax()),
        "n_leads_over_20": int((lead_max > 20).sum()),
        "n_samples_over_20": int((a > 20).sum()),
        "min_lead_std": float(sig.std(axis=0).min()),
    }


def process(row, fetched_ok: bool, fetch_msg: str) -> dict:
    rec = {
        "ed_stay_id": row.ed_stay_id,
        "fold": getattr(row, FOLD_COL),
        "label_ami": row.label_ami,
        "status": "",
        "signal_ok": 0,
        "qc_reason": "",
        "error": "",
    }
    if not fetched_ok:
        rec["status"] = "download_fail"
        rec["error"] = fetch_msg
        return rec

    base = CACHE / relative_path(row.file_name)
    status = "load_fail"
    try:
        sig, fs = load_ecg_record(base)
        rec.update(fs=fs, n_samples=sig.shape[0], n_leads=sig.shape[1])
        rec.update(amplitude_metrics(sig))

        status = "qc_fail"
        qc = check_ecg_quality(sig)
        rec["qc_reason"] = qc["qc_reason"]
        if qc["signal_ok"] != 1:
            rec["status"] = status
            return rec

        status = "preprocess_fail"
        processed = preprocess_ecg(sig, fs)
        if processed.shape != (12, 1000) or not np.isfinite(processed).all():
            raise ValueError(f"bad_preprocessed_output:{processed.shape}")

        rec["status"] = "ok"
        rec["signal_ok"] = 1
    except Exception as exc:
        rec["status"] = status
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:100]}"
    return rec


def summarize(res: pd.DataFrame) -> None:
    res = res.fillna({"qc_reason": "", "error": ""})  # CSV membaca "" sebagai NaN
    print("\n=== RINGKASAN (agregat saja) ===")
    print("Rekaman diproses:", len(res))
    print("\nStatus:")
    print(res["status"].value_counts().to_string())
    print("\nAlasan QC:")
    print(res["qc_reason"].replace("", "(tidak sampai QC)").value_counts().to_string())
    print("\nTingkat lolos per label (0=negatif, 1=AMI):")
    print(res.groupby("label_ami")["signal_ok"].agg(n="size", lolos="sum", rate="mean").round(3).to_string())
    loaded = res.dropna(subset=["max_abs"])
    if len(loaded):
        print("\nDistribusi max_abs (mV) pada rekaman yang terbaca:")
        print(loaded["max_abs"].quantile([0.5, 0.9, 0.99, 1.0]).round(2).to_string())
        for t in (5, 10, 20):
            print(f"max_abs > {t} mV: {int((loaded['max_abs'] > t).sum())}")
        print("Rekaman dengan >=1 lead di atas 20 mV, jumlah lead:",
              loaded.loc[loaded["n_leads_over_20"] > 0, "n_leads_over_20"].tolist())
    if (res["error"] != "").any():
        print("\nJenis error:")
        print(res.loc[res["error"] != "", "error"].str.split(":").str[0].value_counts().to_string())


def main() -> None:
    master = pd.read_parquet(MASTER, columns=["ed_stay_id", "file_name", FOLD_COL, "label_ami"])
    sample = draw_sample(master)
    assert (sample[FOLD_COL] < FOLD_META).all(), "Sampel menyentuh fold >= 17!"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        prev = pd.read_csv(OUTPUT)
        keep = prev[~prev["status"].isin(RETRY_STATUSES)]
        if len(keep) != len(prev):
            keep.to_csv(OUTPUT, index=False)
            print(f"Dicoba ulang: {len(prev) - len(keep)} rekaman download_fail/load_fail dari run sebelumnya")
        done = set(keep["ed_stay_id"])
    else:
        done = set()

    todo = sample[~sample["ed_stay_id"].isin(done)]
    print(f"Sampel: {len(sample)} rekaman ({len(done)} sudah selesai, {len(todo)} akan diproses, "
          f"{WORKERS} koneksi paralel)")

    rows = list(todo.itertuples(index=False))
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(fetch_record, BASE_URL, relative_path(r.file_name), CACHE): r for r in rows}
        for i, future in enumerate(as_completed(futures), start=1):
            ok, msg = future.result()
            rec = process(futures[future], ok, msg)
            pd.DataFrame([rec]).to_csv(OUTPUT, mode="a", header=not OUTPUT.exists(), index=False)
            if i % 10 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)} diproses")

    summarize(pd.read_csv(OUTPUT))
    print("\nHasil lokal:", OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
