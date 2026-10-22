"""Pilot QC untuk maksimal 20 rekaman ECG MIMIC-IV-ECG."""

from pathlib import Path
import time
import urllib.request

import pandas as pd

from src.dataset.ecg_loader import load_ecg_record
from src.model.ecg_preprocessing import preprocess_ecg
from src.model.ecg_quality import check_ecg_quality

ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "data/processed/master_table_draft.parquet"
CACHE = ROOT / "data/raw/ecg_cache"
OUTPUT = ROOT / "data/interim/signal_qc_pilot.csv"

BASE_URL = "https://physionet.org/files/mimic-iv-ecg/1.0/"
PREFIX = "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/"
LIMIT = 20


def download_file(url: str, destination: Path) -> None:
    """Unduh satu file; gunakan file cache jika sudah tersedia."""
    if destination.exists() and destination.stat().st_size > 0:
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url, headers={"User-Agent": "ECG-Research/1.0"}
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with temporary.open("wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)

        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    master = pd.read_parquet(MASTER)

    required = {"ed_stay_id", "file_name"}
    if not required.issubset(master.columns):
        raise ValueError(f"Kolom master tidak lengkap: {required}")

    # Hindari pengujian berulang terhadap rekaman yang sama.
    sample = master.drop_duplicates("file_name").head(LIMIT)
    results = []

    for i, row in enumerate(sample.itertuples(index=False), start=1):
        stay_id = row.ed_stay_id
        original_path = str(row.file_name).lstrip("/")
        relative_path = original_path.removeprefix(PREFIX)

        record_rel = Path(relative_path)
        cache_base = CACHE / record_rel

        result = {
            "ed_stay_id": stay_id,
            "signal_ok": 0,
            "qc_reason": "",
            "fs": None,
            "n_samples": None,
            "n_leads": None,
            "preprocess_ok": 0,
        }

        try:
            for ext in (".hea", ".dat"):
                destination = cache_base.with_suffix(ext)
                url = BASE_URL + relative_path + ext
                download_file(url, destination)

            signal, fs = load_ecg_record(cache_base)
            result["fs"] = fs
            result["n_samples"] = signal.shape[0]
            result["n_leads"] = signal.shape[1]

            qc = check_ecg_quality(signal)
            result["qc_reason"] = qc["qc_reason"]

            if qc["signal_ok"] != 1:
                raise ValueError(qc["qc_reason"])

            processed = preprocess_ecg(signal, fs)

            if processed.shape != (12, 1000):
                raise ValueError(
                    f"unexpected_preprocessed_shape:{processed.shape}"
                )

            result["preprocess_ok"] = 1
            result["signal_ok"] = 1
            result["qc_reason"] = "passed_basic_checks"

        except Exception as exc:
            if not result["qc_reason"]:
                result["qc_reason"] = (
                    f"{type(exc).__name__}: {str(exc)[:180]}"
                )
            print(f"[{i}/{len(sample)}] GAGAL stay_id={stay_id}: "
                  f"{result['qc_reason']}")

        results.append(result)

        if result["signal_ok"] == 1:
            print(
                f"[{i}/{len(sample)}] LULUS "
                f"fs={result['fs']} shape=({result['n_samples']},"
                f" {result['n_leads']})"
            )

        # Jeda kecil agar tidak mengirim permintaan terlalu cepat.
        time.sleep(0.2)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT, index=False)

    report = pd.DataFrame(results)
    print("\n=== RINGKASAN PILOT ===")
    print("Jumlah diuji:", len(report))
    print("QC lolos:", int((report["signal_ok"] == 1).sum()))
    print("QC gagal:", int((report["signal_ok"] == 0).sum()))
    print("Preprocessing berhasil:", int(report["preprocess_ok"].sum()))
    print("Hasil tersimpan:", OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
