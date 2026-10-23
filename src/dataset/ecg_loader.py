
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

    return signal, fs, [str(n) for n in record.sig_name]
