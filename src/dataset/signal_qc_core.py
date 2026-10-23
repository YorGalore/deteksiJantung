"""Pemrosesan QC sinyal yang dipakai SEMUA runner + kontrak status signal_qc.

Jalur tunggal sesuai PROJECT_BRIEF 10.2/10.3 dan identik dengan ECGDataset:
read_ecg_record (validasi mV, NaN tidak ditolak) -> check_ecg_quality_brief ->
prepare_ecg. Fungsi lama (load_ecg_record, check_ecg_quality, preprocess_ecg) TIDAK
dipakai di sini.

Kontrak status (kolom `status` -> `signal_ok`):
  pass             -> 1
  qc_fail          -> 0   terbaca, gagal aturan QC brief
  preprocess_fail  -> 0   lolos QC, gagal di prepare_ecg / bentuk keluaran
  load_fail        -> 0   berkas lengkap tetapi tidak terbaca / satuan bukan mV
  download_failed  -> kosong (tidak pernah diperiksa)
  not_processed    -> kosong (belum diperiksa)
Hanya `pass` yang boleh dihitung lolos; build_master membuang semua selain signal_ok == 1.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import FOLD_COL
from src.dataset.ecg_download import fetch_record, is_valid_record
from src.dataset.ecg_loader import read_ecg_record
from src.model.ecg_preprocessing import prepare_ecg
from src.model.ecg_quality import check_ecg_quality_brief

PASS = "pass"
QC_FAIL = "qc_fail"
PREPROCESS_FAIL = "preprocess_fail"
LOAD_FAIL = "load_fail"
DOWNLOAD_FAILED = "download_failed"
NOT_PROCESSED = "not_processed"
STATUSES = (PASS, QC_FAIL, PREPROCESS_FAIL, LOAD_FAIL, DOWNLOAD_FAILED, NOT_PROCESSED)
_CHECKED_FAIL = (QC_FAIL, PREPROCESS_FAIL, LOAD_FAIL)

COLUMNS = ["ed_stay_id", "fold", "label_ami", "status", "signal_ok", "qc_reason",
           "amplitude_flag", "max_nan_fraction", "n_flat_leads", "max_abs", "error"]
CONTRACT_COLUMNS = ["ed_stay_id", "status", "signal_ok", "qc_reason", "amplitude_flag", "error"]


def signal_ok_for(status: str):
    """1 untuk pass, 0 untuk gagal setelah diperiksa, None (kosong) untuk belum diperiksa."""
    if status == PASS:
        return 1
    if status in _CHECKED_FAIL:
        return 0
    if status in (DOWNLOAD_FAILED, NOT_PROCESSED):
        return None
    raise ValueError(f"status_tidak_dikenal:{status}")


def _blank(status: str, error: str = "") -> dict:
    return {"status": status, "signal_ok": signal_ok_for(status), "qc_reason": "",
            "amplitude_flag": np.nan, "max_nan_fraction": np.nan, "n_flat_leads": np.nan,
            "max_abs": np.nan, "error": error}


def process_record(base: Path) -> dict:
    """Proses satu rekaman yang sudah ada di cache. TIDAK mengunduh apa pun."""
    if not is_valid_record(base):
        return _blank(NOT_PROCESSED, "not_cached_or_incomplete")

    out = _blank(LOAD_FAIL)
    stage = LOAD_FAIL
    try:
        sig, fs, names = read_ecg_record(base)
        qc = check_ecg_quality_brief(sig)
        finite = np.isfinite(sig)
        out.update(
            qc_reason=qc["qc_reason"],
            amplitude_flag=int(qc["amplitude_flag"]),
            max_nan_fraction=float(qc["max_nan_fraction"]),
            n_flat_leads=int(qc["n_flat_leads"]),
            max_abs=float(np.abs(sig[finite]).max()) if finite.any() else np.nan,
        )
        stage = QC_FAIL
        if qc["signal_ok"] != 1:
            out["status"] = QC_FAIL
            out["signal_ok"] = 0
            return out

        stage = PREPROCESS_FAIL
        x = prepare_ecg(sig, fs, names)
        if x.shape != (12, 1000) or not np.isfinite(x).all():
            raise ValueError(f"bad_preprocessed_output:{x.shape}")
        out["status"], out["signal_ok"] = PASS, 1
    except Exception as exc:
        out["status"], out["signal_ok"] = stage, 0
        out["error"] = f"{type(exc).__name__}: {str(exc)[:100]}"
    return out


def download_and_process(rows, base_url: str, cache: Path, rel_fn, workers: int, on_row=None) -> list[dict]:
    """Unduh (paralel, tervalidasi) lalu proses lewat process_record. Mengembalikan daftar baris.

    `rows`: itertuples dengan atribut ed_stay_id, file_name, fold, label_ami.
    Kegagalan unduh -> download_failed (signal_ok kosong), TIDAK pernah dianggap lolos.
    """
    rows = list(rows)
    results = []
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {pool.submit(fetch_record, base_url, rel_fn(r.file_name), cache): r for r in rows}
        for i, future in enumerate(as_completed(futures), start=1):
            r = futures[future]
            ok, msg = future.result()
            fields = _blank(DOWNLOAD_FAILED, msg) if not ok else process_record(cache / rel_fn(r.file_name))
            rec = {"ed_stay_id": r.ed_stay_id, "fold": getattr(r, FOLD_COL), "label_ami": r.label_ami, **fields}
            results.append(rec)
            if on_row is not None:
                on_row(i, len(rows), rec)
    except BaseException:
        # Ctrl+C / error: batalkan antrean yang belum jalan. Tanpa ini, `with ThreadPoolExecutor`
        # menunggu SELURUH antrean selesai dan Ctrl+C tampak tidak berfungsi. Unduhan yang sedang
        # berjalan (maks. `workers`) diselesaikan; rekaman yang sudah di cache tidak hilang.
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return results


def build_contract(results: pd.DataFrame, ed_stay_ids) -> pd.DataFrame:
    """Satu baris per ed_stay_id master; yang tidak punya hasil -> not_processed.

    signal_ok SELALU dihitung ulang dari status (kolom signal_ok masukan diabaikan).
    """
    ids = pd.DataFrame({"ed_stay_id": list(ed_stay_ids)})
    if ids["ed_stay_id"].duplicated().any():
        raise ValueError("ed_stay_id master tidak unik.")
    res = results.drop_duplicates("ed_stay_id", keep="last")
    res = res[res["ed_stay_id"].isin(ids["ed_stay_id"])]
    bad = set(res["status"].dropna()) - set(STATUSES)
    if bad:
        raise ValueError(f"status_tidak_dikenal:{sorted(bad)}")

    out = ids.merge(res.reindex(columns=["ed_stay_id", "status", "qc_reason", "amplitude_flag", "error"]),
                    on="ed_stay_id", how="left")
    out["status"] = out["status"].fillna(NOT_PROCESSED)
    out["signal_ok"] = pd.array([signal_ok_for(s) for s in out["status"]], dtype="Int8")
    return out[CONTRACT_COLUMNS]
