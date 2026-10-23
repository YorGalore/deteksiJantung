"""QC sinyal sesuai PROJECT_BRIEF pada sampel yang sama, TANPA mengunduh apa pun.

Jalankan dari root repo:
    python -u -m src.dataset.run_signal_qc_brief

Hanya memproses rekaman yang sudah lengkap di cache (ukuran .dat tervalidasi); sisanya
berstatus not_processed. Aturan: gagal jika ada lead dengan > 20% NaN atau >= 3 lead datar;
amplitudo > 20 mV hanya menjadi penanda (amplitude_flag). Jalur pemrosesan = signal_qc_core.
Hasil lokal (di-ignore git): data/interim/signal_qc_brief.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.constants import FOLD_COL, FOLD_META
from src.dataset.run_signal_qc_pilot import CACHE, MASTER
from src.dataset.run_signal_qc_sample import draw_sample, relative_path, summarize
from src.dataset.signal_qc_core import COLUMNS, process_record

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/interim/signal_qc_brief.csv"


def main() -> None:
    master = pd.read_parquet(MASTER, columns=["ed_stay_id", "file_name", FOLD_COL, "label_ami"])
    sample = draw_sample(master)
    assert (sample[FOLD_COL] < FOLD_META).all(), "Sampel menyentuh fold >= 17!"

    rows = []
    for r in sample.itertuples(index=False):
        rec = process_record(CACHE / relative_path(r.file_name))
        rows.append({"ed_stay_id": r.ed_stay_id, "fold": getattr(r, FOLD_COL),
                     "label_ami": r.label_ami, **rec})
    res = pd.DataFrame(rows, columns=COLUMNS)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUTPUT, index=False)
    summarize(res)
    print("\nHasil lokal:", OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
