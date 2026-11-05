"""Susun kontrak signal_qc.csv untuk Orang 1 (build_master) dari hasil QC penuh.

Jalankan dari root repo:
    python -m src.dataset.build_signal_qc

Masukan : data/interim/signal_qc_full.csv (run_full_qc) + master draft.
Keluaran: data/interim/signal_qc.csv         (ed_stay_id, signal_ok)  -> kontrak brief
          data/interim/signal_qc_status.csv  (ed_stay_id,status,signal_ok,qc_reason,amplitude_flag,error)
Aturan D3: kontrak HANYA ditulis bila SETIAP ed_stay_id master punya status akhir
(pass/qc_fail/preprocess_fail/load_fail). Jika masih ada not_processed/download_failed, berhenti tanpa menulis.
Berkas bersifat lokal (di-ignore git); hanya jumlah agregat dicetak.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.dataset.run_full_qc import expand_to_all_ids
from src.dataset.run_signal_qc_pilot import MASTER
from src.dataset.signal_qc_core import DOWNLOAD_FAILED, NOT_PROCESSED, build_contract

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "data/interim/signal_qc_full.csv"
CONTRACT = ROOT / "data/interim/signal_qc.csv"
STATUS = ROOT / "data/interim/signal_qc_status.csv"


def build(master: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Kontrak penuh (satu baris per ed_stay_id master); error bila ada yang belum berstatus akhir."""
    full = build_contract(expand_to_all_ids(results, master), master["ed_stay_id"])
    pending = full["status"].isin([NOT_PROCESSED, DOWNLOAD_FAILED])
    if pending.any():
        raise SystemExit(f"{int(pending.sum())} ed_stay_id belum berstatus akhir "
                         f"({full.loc[pending, 'status'].value_counts().to_dict()}). Kontrak tidak ditulis.")
    return full


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", type=Path, default=FULL)
    ap.add_argument("--contract", type=Path, default=CONTRACT)
    ap.add_argument("--status", type=Path, default=STATUS)
    a = ap.parse_args(argv)
    master = pd.read_parquet(MASTER, columns=["ed_stay_id", "file_name"])
    full = build(master, pd.read_csv(a.full))
    full.to_csv(a.status, index=False)
    full[["ed_stay_id", "signal_ok"]].astype({"signal_ok": "int8"}).to_csv(a.contract, index=False)
    print(f"Master: {len(master)} | kontrak: {len(full)} baris | signal_ok=1: {int((full.signal_ok == 1).sum())} | "
          f"signal_ok=0: {int((full.signal_ok == 0).sum())}")
    print(full["status"].value_counts().to_string())
    print("Ditulis:", a.contract, "dan", a.status)


if __name__ == "__main__":
    main()