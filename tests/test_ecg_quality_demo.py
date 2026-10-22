import numpy as np

from src.model.ecg_quality import check_ecg_quality


def test_valid_synthetic_ecg_passes():
    rng = np.random.default_rng(42)
    signal = rng.normal(0, 0.1, size=(5000, 12))

    result = check_ecg_quality(signal)

    assert result["signal_ok"] == 1
    assert result["qc_reason"] == "passed_basic_checks"


def test_wrong_number_of_leads_fails():
    signal = np.random.default_rng(42).normal(size=(5000, 8))

    result = check_ecg_quality(signal)

    assert result["signal_ok"] == 0
    assert result["qc_reason"] == "unexpected_lead_count"


def test_nan_signal_fails():
    signal = np.random.default_rng(42).normal(size=(5000, 12))
    signal[100, 0] = np.nan

    result = check_ecg_quality(signal)

    assert result["signal_ok"] == 0
    assert result["qc_reason"] == "non_finite_values"


def test_flat_lead_fails():
    signal = np.random.default_rng(42).normal(size=(5000, 12))
    signal[:, 0] = 0

    result = check_ecg_quality(signal)

    assert result["signal_ok"] == 0
    assert result["qc_reason"] == "flat_lead"
