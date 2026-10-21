
"""Preprocessing sinyal EKG 12-lead untuk eksperimen ResNet-1D."""

import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly


def preprocess_ecg(
    signal: np.ndarray,
    fs: int,
    target_fs: int = 100,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    duration_seconds: int = 10,
) -> np.ndarray:
    """Mengolah sinyal berbentuk (samples, 12) menjadi (12, 1000).

    Ini baseline preprocessing untuk eksperimen awal, bukan protokol klinis
    yang sudah tervalidasi.
    """
    x = np.asarray(signal, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError("Sinyal harus berupa array 2D.")
    if x.shape[1] != 12:
        raise ValueError(f"Diharapkan 12 lead; ditemukan {x.shape[1]}.")
    if fs <= 0 or target_fs <= 0:
        raise ValueError("Sampling rate harus positif.")
    if x.shape[0] < 2:
        raise ValueError("Sinyal terlalu pendek.")
    if not np.isfinite(x).all():
        raise ValueError("Sinyal mengandung NaN atau infinity; perlu QC.")

    # Resampling menggunakan rasio sampling rate yang disederhanakan.
    from fractions import Fraction
    ratio = Fraction(target_fs, fs).limit_denominator(10000)
    x = resample_poly(x, ratio.numerator, ratio.denominator, axis=0)

    if not 0 < lowcut < highcut < target_fs / 2:
        raise ValueError("Batas filter harus berada di bawah frekuensi Nyquist.")

    sos = butter(
        4, [lowcut, highcut],
        btype="bandpass",
        fs=target_fs,
        output="sos",
    )
    if x.shape[0] <= 30:
        raise ValueError("Sinyal terlalu pendek untuk filtering yang aman.")

    x = sosfiltfilt(sos, x, axis=0)

    expected_samples = target_fs * duration_seconds

    # Pangkas sinyal panjang; pad sinyal pendek hanya untuk uji pipeline.
    # Dalam pipeline klinis, sinyal pendek harus ditandai untuk ditinjau.
    if x.shape[0] < expected_samples:
        raise ValueError(
            f"Sinyal hanya {x.shape[0]} sampel setelah resampling; "
            f"diharapkan minimal {expected_samples}."
        )
    x = x[:expected_samples, :]

    # Normalisasi z-score per lead.
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    x = (x - mean) / std

    # Bentuk output: (lead, samples).
    return x.T.astype(np.float32)

#Notes: fungsi ini adalah baseline awal. Kita sengaja menolak sinyal yang mengandung nilai tidak valid atau panjangnya tidak sesuai, daripada diam-diam mengisi data yang rusak. Nantinya, kita akan menyesuaikan penanganan sinyal pendek berdasarkan karakteristik waveform yang benar-benar tersedia.