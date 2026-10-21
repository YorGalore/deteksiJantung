
import numpy as np

from src.model.ecg_preprocessing import preprocess_ecg


def test_synthetic_ecg_shape():
    fs = 500
    duration = 10
    t = np.arange(fs * duration) / fs

    # Data sintetis untuk menguji pipeline, bukan simulasi klinis AMI.
    lead_signals = [
        np.sin(2 * np.pi * (1 + i * 0.1) * t)
        + 0.1 * np.sin(2 * np.pi * 15 * t)
        for i in range(12)
    ]
    signal = np.stack(lead_signals, axis=1)

    result = preprocess_ecg(signal, fs=fs)

    assert result.shape == (12, 1000)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()


if __name__ == "__main__":
    test_synthetic_ecg_shape()
    print("LULUS: preprocessing menghasilkan array (12, 1000)")
