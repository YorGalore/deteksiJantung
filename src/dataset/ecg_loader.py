
"""Loader sinyal EKG dari berkas WFDB."""

from pathlib import Path

import numpy as np
import wfdb


def load_ecg_record(record_path: str | Path) -> tuple[np.ndarray, int]:
    """
    Membaca satu rekaman EKG WFDB.

    Parameters
    ----------
    record_path:
        Path rekaman tanpa ekstensi, misalnya:
        data/raw/example_record

    Returns
    -------
    signal:
        Sinyal dengan bentuk (samples, leads).
    fs:
        Sampling rate dalam Hz.
    """
    record_path = Path(record_path)

    if not record_path.with_suffix(".hea").exists():
        raise FileNotFoundError(
            f"Header WFDB tidak ditemukan: {record_path.with_suffix('.hea')}"
        )

    record = wfdb.rdrecord(str(record_path))

    signal = np.asarray(record.p_signal, dtype=np.float32)
    fs = int(record.fs)

    if signal.ndim != 2:
        raise ValueError("Sinyal EKG harus berbentuk array 2D.")

    if not np.isfinite(signal).all():
        raise ValueError("Sinyal EKG mengandung NaN atau infinity.")

    if fs <= 0:
        raise ValueError("Sampling rate tidak valid.")

    return signal, fs


def header_units(record_path: str | Path) -> list[str | None]:
    """Satuan tiap lead dari baris sinyal .hea MENTAH (bukan dari wfdb).

    wfdb mengisi satuan yang kosong/tidak ada menjadi 'mV' secara diam-diam, sehingga
    record.units tidak bisa membedakan 'mV' dari 'tidak dinyatakan'. None = tidak ada.
    """
    lines = [l for l in Path(record_path).with_suffix(".hea").read_text().splitlines()
             if l.strip() and not l.lstrip().startswith("#")]
    n_sig = int(lines[0].split()[1])
    units: list[str | None] = []
    for line in lines[1:1 + n_sig]:
        tok = line.split()
        gain = tok[2] if len(tok) > 2 else ""
        units.append(gain.split("/", 1)[1].strip() if "/" in gain else None)
    return units


def validate_units_mv(units, n_leads: int) -> None:
    """Pastikan SEMUA lead berstatuan mV; selain itu error eksplisit (tidak diasumsikan).

    Clipping +-3 mV dan semua ambang amplitudo hanya bermakna dalam mV.
    """
    if units is None or len(units) != n_leads:
        raise ValueError(f"unit_invalid:jumlah satuan tidak sama dengan jumlah lead ({n_leads})")
    found = {("" if u is None else str(u).strip()) for u in units}
    if "" in found:
        raise ValueError("unit_invalid:satuan tidak dinyatakan/kosong di header")
    if {u.lower() for u in found} != {"mv"}:
        raise ValueError(f"unit_invalid:bukan mV semua ({sorted(found)})")


def read_ecg_record(record_path: str | Path) -> tuple[np.ndarray, int, list[str]]:
    """Seperti load_ecg_record, tetapi TIDAK menolak NaN dan juga mengembalikan nama lead.

    Dipakai pipeline sesuai PROJECT_BRIEF: NaN diinterpolasi lalu dinilai oleh QC.
    Mengembalikan (signal [samples, leads], fs, nama_lead).
    """
    record_path = Path(record_path)

    if not record_path.with_suffix(".hea").exists():
        raise FileNotFoundError(
            f"Header WFDB tidak ditemukan: {record_path.with_suffix('.hea')}"
        )

    record = wfdb.rdrecord(str(record_path))
    signal = np.asarray(record.p_signal, dtype=np.float32)
    fs = int(record.fs)

    if signal.ndim != 2:
        raise ValueError("Sinyal EKG harus berbentuk array 2D.")
    if fs <= 0:
        raise ValueError("Sampling rate tidak valid.")
    validate_units_mv(header_units(record_path), signal.shape[1])

    return signal, fs, [str(n) for n in record.sig_name]
