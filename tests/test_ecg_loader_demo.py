
import numpy as np
import wfdb

from src.dataset.ecg_loader import load_ecg_record


def test_load_synthetic_wfdb_record(tmp_path):
    fs = 500
    duration = 10
    t = np.arange(fs * duration) / fs

    signal = np.stack(
        [
            np.sin(2 * np.pi * (1 + i * 0.1) * t)
            for i in range(12)
        ],
        axis=1,
    ).astype(np.float32)

    record_path = str(tmp_path / "synthetic_ecg")

    wfdb.wrsamp(
        record_name="synthetic_ecg",
        fs=fs,
        units=["mV"] * 12,
        sig_name=[
            "I", "II", "III", "aVR", "aVL", "aVF",
            "V1", "V2", "V3", "V4", "V5", "V6",
        ],
        p_signal=signal,
        fmt=["16"] * 12,
        write_dir=str(tmp_path),
    )

    loaded_signal, loaded_fs = load_ecg_record(record_path)

    assert loaded_signal.shape == (5000, 12)
    assert loaded_fs == 500
    assert np.isfinite(loaded_signal).all()


if __name__ == "__main__":
    test_load_synthetic_wfdb_record(__import__("pathlib").Path("tests"))
    print("LULUS: loader berhasil membaca rekaman WFDB sintetis.")
