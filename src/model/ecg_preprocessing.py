
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

# ---------------------------------------------------------------------------
# Pipeline sesuai PROJECT_BRIEF Bagian 10.3 (tanpa bandpass, statistik dari
# data training saja). preprocess_ecg di atas dipertahankan hanya agar tes lama
# tetap berjalan; pipeline training memakai prepare_ecg + LeadStandardizer.
# ---------------------------------------------------------------------------
import json
from fractions import Fraction
from pathlib import Path

LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CLIP_MV = 3.0


def reorder_leads(signal: np.ndarray, sig_names) -> np.ndarray:
    """Susun kolom ke urutan LEAD_ORDER berdasarkan nama sinyal di header."""
    names = [str(n).strip().upper() for n in sig_names]
    if len(names) != 12 or len(set(names)) != 12:
        raise ValueError(f"Nama lead tidak valid: {len(names)} nama, {len(set(names))} unik.")
    lookup = {n: i for i, n in enumerate(names)}
    try:
        idx = [lookup[lead.upper()] for lead in LEAD_ORDER]
    except KeyError as exc:
        raise ValueError(f"lead_missing:{exc.args[0]}") from None
    return np.asarray(signal)[:, idx]


def interpolate_nans(signal: np.ndarray) -> np.ndarray:
    """Interpolasi linear per lead; NaN di ujung (sebelum/sesudah data valid) -> 0."""
    x = np.array(signal, dtype=np.float64, copy=True)
    for j in range(x.shape[1]):
        bad = ~np.isfinite(x[:, j])
        if not bad.any():
            continue
        good = np.flatnonzero(~bad)
        if good.size == 0:
            x[:, j] = 0.0
            continue
        first, last = good[0], good[-1]
        interior = np.arange(first, last + 1)
        x[interior, j] = np.interp(interior, good, x[good, j])
        x[:first, j] = 0.0
        x[last + 1:, j] = 0.0
    return x


def prepare_ecg(signal: np.ndarray, fs: int, sig_names, target_fs: int = 100,
                duration_seconds: int = 10, clip_mv: float = CLIP_MV) -> np.ndarray:
    """(samples, 12) -> (12, target_fs * duration), belum distandardisasi.

    Urutan: susun lead -> interpolasi NaN -> resample -> potong 10 s -> clip.
    Tanpa bandpass (high-pass mendistorsi segmen ST).
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("Sinyal harus berupa array 2D.")
    if fs <= 0 or target_fs <= 0:
        raise ValueError("Sampling rate harus positif.")

    x = reorder_leads(x, sig_names)
    x = interpolate_nans(x)

    ratio = Fraction(target_fs, fs).limit_denominator(10000)
    x = resample_poly(x, ratio.numerator, ratio.denominator, axis=0)

    expected = target_fs * duration_seconds
    if x.shape[0] < expected:
        raise ValueError(f"Sinyal hanya {x.shape[0]} sampel setelah resampling; diharapkan minimal {expected}.")

    x = np.clip(x[:expected], -clip_mv, clip_mv)
    return x.T.astype(np.float32)


class LeadStandardizer:
    """Rata-rata dan simpangan baku global per lead, dihitung dari data TRAINING saja."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float64).reshape(12)
        self.std = np.asarray(std, dtype=np.float64).reshape(12)

    @classmethod
    def fit(cls, records) -> "LeadStandardizer":
        """`records`: iterable array (12, T). Dihitung bertahap, hemat memori."""
        n = 0
        total = np.zeros(12)
        total_sq = np.zeros(12)
        for r in records:
            r = np.asarray(r, dtype=np.float64)
            total += r.sum(axis=1)
            total_sq += (r ** 2).sum(axis=1)
            n += r.shape[1]
        if n == 0:
            raise ValueError("Tidak ada data untuk fit.")
        mean = total / n
        std = np.sqrt(np.maximum(total_sq / n - mean ** 2, 0.0))
        return cls(mean, np.maximum(std, 1e-6))

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Menerima (12, T) atau (N, 12, T)."""
        return ((np.asarray(x) - self.mean.reshape(12, 1)) / self.std.reshape(12, 1)).astype(np.float32)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({"mean": self.mean.tolist(), "std": self.std.tolist()}), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "LeadStandardizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["mean"], d["std"])
