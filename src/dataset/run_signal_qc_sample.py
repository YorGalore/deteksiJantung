"""QC bertahap pada sampel (hanya fold 0-16): unduh tervalidasi + jalur brief.

Jalankan dari root repo:
    python -u -m src.dataset.run_signal_qc_sample

Output lokal (di-ignore git): data/interim/signal_qc_sample.csv (kolom signal_qc_core.COLUMNS).
JANGAN ganti namanya menjadi signal_qc.csv: berkas itu harus berisi SEMUA ed_stay_id master
(lihat signal_qc_core.build_contract). Fold 17-19 sengaja tidak disentuh (19 = test).

Skrip bisa dilanjutkan. Baris download_failed dari run sebelumnya dicoba lagi; rekaman yang sudah
lengkap di cache tidak diunduh ulang. Berkas hasil lama (status ok/download_fail/load_fail dari
jalur QC lama) DITOLAK: pindahkan/hapus dulu, jangan dicampur.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.constants import FOLD_COL, FOLD_META
from src.dataset.run_signal_qc_pilot import BASE_URL, CACHE, MASTER, relative_path  # noqa: F401 (re-export)
from src.dataset.signal_qc_core import (
    COLUMNS, DOWNLOAD_FAILED, PASS, STATUSES, download_and_process,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/interim/signal_qc_sample.csv"
SEED = 42
N_POS = 100
N_NEG = 200
WORKERS = 8          # jangan dinaikkan: server publik
RETRY_STATUSES = {DOWNLOAD_FAILED}


def draw_sample(master: pd.DataFrame) -> pd.DataFrame:
    """Sampel acak bertingkat dari fold 0-16: N_POS positif + N_NEG negatif."""
    d = master[master[FOLD_COL] < FOLD_META].drop_duplicates("file_name")
    pos = d[d["label_ami"] == 1].sample(N_POS, random_state=SEED)
    neg = d[d["label_ami"] == 0].sample(N_NEG, random_state=SEED)
    return pd.concat([pos, neg]).sort_values("ed_stay_id").reset_index(drop=True)


def summarize(res: pd.DataFrame) -> None:
    res = res.fillna({"qc_reason": "", "error": ""})  # CSV membaca "" sebagai NaN
    print("\n=== RINGKASAN (agregat saja) ===")
    print("Rekaman:", len(res))
    print("\nStatus:")
    print(res["status"].value_counts().to_string())
    print("\nAlasan QC:")
    print(res["qc_reason"].replace("", "(tidak sampai QC)").value_counts().to_string())
    print("\nLolos per label (0=negatif, 1=AMI):")
    print(res.groupby("label_ami")["signal_ok"].agg(n="size", lolos="sum", rate="mean").round(3).to_string())
    print("\nStatus per fold (jumlah rekaman):")
    print(pd.crosstab(res["fold"], res["status"]).to_string())
    ok = res[res["status"] == PASS]
    print(f"\nLolos: {len(ok)} | dengan amplitude_flag (>20 mV): {int(ok['amplitude_flag'].sum())}")
    loaded = res.dropna(subset=["max_abs"])
    if len(loaded):
        print("\nDistribusi max_abs (mV) pada rekaman yang terbaca:")
        print(loaded["max_abs"].quantile([0.5, 0.9, 0.99, 1.0]).round(2).to_string())
    if (res["error"] != "").any():
        print("\nJenis error:")
        print(res.loc[res["error"] != "", "error"].str.split(":").str[0].value_counts().to_string())


def run_sample(sample: pd.DataFrame, output: Path = OUTPUT, workers: int = WORKERS,
               allowed_folds=None) -> None:
    """Unduh + proses `sample`, bisa dilanjutkan, tulis ke `output`.

    Default hanya fold 0-16. Fold lain hanya lewat `allowed_folds` eksplisit (lihat run_eval_qc).
    """
    allowed = set(range(FOLD_META)) if allowed_folds is None else set(allowed_folds)
    assert set(sample[FOLD_COL].unique()) <= allowed, "Sampel memuat fold di luar yang diizinkan!"

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        prev = pd.read_csv(output)
        legacy = set(prev["status"].dropna()) - set(STATUSES)
        if legacy:
            raise SystemExit(f"{output.name} berisi status jalur QC lama {sorted(legacy)}. "
                             "Pindahkan/hapus berkas itu dulu (jangan dicampur), lalu jalankan ulang.")
        keep = prev[~prev["status"].isin(RETRY_STATUSES)]
        if len(keep) != len(prev):
            keep.to_csv(output, index=False)
            print(f"Dicoba ulang: {len(prev) - len(keep)} rekaman download_failed dari run sebelumnya")
        done = set(keep["ed_stay_id"])
    else:
        done = set()

    todo = sample[~sample["ed_stay_id"].isin(done)]
    print(f"Sampel: {len(sample)} rekaman ({len(done)} sudah selesai, {len(todo)} akan diproses, "
          f"{workers} koneksi paralel)")

    def save(i, n, rec):
        pd.DataFrame([rec], columns=COLUMNS).to_csv(output, mode="a", header=not output.exists(), index=False)
        if i % 10 == 0 or i == n:
            print(f"  {i}/{n} diproses")

    download_and_process(todo.itertuples(index=False), BASE_URL, CACHE, relative_path, workers, on_row=save)
    summarize(pd.read_csv(output))
    print("\nHasil lokal:", output)


def main() -> None:
    master = pd.read_parquet(MASTER, columns=["ed_stay_id", "file_name", FOLD_COL, "label_ami"])
    run_sample(draw_sample(master))


if __name__ == "__main__":
    main()
