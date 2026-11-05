"""Ekstrak rekaman kohort dari ZIP resmi MIMIC-IV-ECG ke cache lokal (data/raw/ecg_cache).

Jalankan dari root repo:
    python -u -m src.dataset.extract_zip_records --zip D:/mimic_zip/mimic-iv-ecg-1.0.zip
    python -u -m src.dataset.extract_zip_records --zip ... --verify-existing     # bandingkan cache lama dengan ZIP

- Pemetaan: anggota ZIP = `file_name` master + ".hea" / ".dat" (awalan PREFIX sama dengan file_name);
  tujuan = cache / relative_path(file_name), struktur yang sama dengan unduhan per rekaman.
- Keutuhan: CRC ZIP diperiksa saat anggota dibaca penuh; ukuran .dat harus sama dengan nilai dari header
  (is_valid_record). Tulis atomik (.part -> rename); rekaman yang sudah valid di cache dilewati.
- Fold 19 hanya diekstrak dengan --include-fold-19 (aturan kunci fold 19 yang sama dengan runner QC).
- Keluaran lokal (tidak di-commit): data/interim/zip_extract_log.csv (ed_stay_id, fold, status). Ringkasan agregat dicetak.
Status: extracted, already_valid, missing_in_zip, crc_fail, size_mismatch, verify_identical, verify_different.
"""
from __future__ import annotations

import argparse
import time
import zipfile
from pathlib import Path

import pandas as pd

from src.constants import FOLD_COL, FOLD_TEST
from src.dataset.ecg_download import _path, is_valid_record
from src.dataset.run_signal_qc_pilot import CACHE, MASTER, relative_path

ROOT = Path(__file__).resolve().parents[2]
LOG_CSV = ROOT / "data/interim/zip_extract_log.csv"
DEFAULT_ZIP = ROOT.parent / "mimic_zip" / "mimic-iv-ecg-1.0.zip"
STATUSES = ("extracted", "already_valid", "missing_in_zip", "crc_fail", "size_mismatch",
            "verify_identical", "verify_different")


def load_targets(master_path: Path, include_fold_19: bool) -> pd.DataFrame:
    m = pd.read_parquet(master_path, columns=["ed_stay_id", "file_name", FOLD_COL])
    if not include_fold_19:
        m = m[m[FOLD_COL] != FOLD_TEST]
    return m.reset_index(drop=True)


def _write_atomic(dest: Path, data: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / (dest.name + ".part")
    try:
        tmp.write_bytes(data)
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink()


def _read_member(z: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    return z.read(info)               # membaca penuh -> CRC diperiksa; BadZipFile jika tidak cocok


def extract_all(zip_path: Path, targets: pd.DataFrame, cache: Path, verify_existing: bool = False,
                limit: int | None = None, progress_every: int = 5000, quiet: bool = False) -> pd.DataFrame:
    rows = []
    t0 = time.time()
    with zipfile.ZipFile(zip_path) as z:
        info = z.NameToInfo

        def offset(fn):
            i = info.get(str(fn).lstrip("/") + ".dat")
            return i.header_offset if i is not None else float("inf")

        order = sorted(targets.itertuples(index=False), key=lambda r: offset(r.file_name))   # baca ZIP berurutan
        if limit:
            order = order[:limit]
        for k, r in enumerate(order, start=1):
            base_name = str(r.file_name).lstrip("/")
            hea_i, dat_i = info.get(base_name + ".hea"), info.get(base_name + ".dat")
            base = cache / relative_path(r.file_name)
            if hea_i is None or dat_i is None:
                status = "missing_in_zip"
            else:
                try:
                    if is_valid_record(base):
                        if verify_existing:
                            same = (_path(base, ".hea").read_bytes() == _read_member(z, hea_i)
                                    and _path(base, ".dat").read_bytes() == _read_member(z, dat_i))
                            status = "verify_identical" if same else "verify_different"
                        else:
                            status = "already_valid"
                    else:
                        _write_atomic(_path(base, ".hea"), _read_member(z, hea_i))
                        _write_atomic(_path(base, ".dat"), _read_member(z, dat_i))
                        if is_valid_record(base):
                            status = "extracted"
                        else:
                            for ext in (".hea", ".dat"):
                                _path(base, ext).unlink(missing_ok=True)
                            status = "size_mismatch"
                except zipfile.BadZipFile:
                    for ext in (".hea", ".dat"):
                        if not verify_existing:
                            _path(base, ext).unlink(missing_ok=True)
                    status = "crc_fail"
            rows.append({"ed_stay_id": r.ed_stay_id, FOLD_COL: getattr(r, FOLD_COL), "status": status})
            if not quiet and k % progress_every == 0:
                el = time.time() - t0
                print(f"  {k}/{len(order)} | {k / el:.0f} rekaman/detik | sisa ~{(len(order) - k) / (k / el) / 60:.0f} menit",
                      flush=True)
    return pd.DataFrame(rows, columns=["ed_stay_id", FOLD_COL, "status"])


def summarize(log: pd.DataFrame) -> str:
    lines = ["=== RINGKASAN (agregat saja) ===", f"Rekaman: {len(log)}", "", "Status:",
             log["status"].value_counts().to_string(), ""]
    bad = log[log["status"].isin(["missing_in_zip", "crc_fail", "size_mismatch", "verify_different"])]
    if len(bad):
        lines += ["Bermasalah per fold:", bad.groupby([FOLD_COL, "status"]).size().unstack(fill_value=0).to_string()]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    ap.add_argument("--master", type=Path, default=MASTER)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--output", type=Path, default=LOG_CSV)
    ap.add_argument("--include-fold-19", action="store_true",
                    help="juga ekstrak fold 19 (test terkunci); wajib eksplisit")
    ap.add_argument("--verify-existing", action="store_true",
                    help="bandingkan rekaman yang sudah ada di cache dengan ZIP (tidak menimpa)")
    ap.add_argument("--limit", type=int, default=None, help="proses hanya N rekaman pertama (uji)")
    a = ap.parse_args(argv)
    targets = load_targets(a.master, a.include_fold_19)
    print(f"Target: {len(targets)} rekaman (fold 19 {'ikut' if a.include_fold_19 else 'TIDAK ikut'}); cache: {a.cache}")
    log = extract_all(a.zip, targets, a.cache, a.verify_existing, a.limit)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(a.output, index=False)
    print(summarize(log))
    print(f"\nLog lokal: {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())