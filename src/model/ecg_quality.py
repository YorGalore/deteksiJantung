"""Basic technical quality checks for 12-lead ECG signals."""

import numpy as np


def check_ecg_quality(
    signal: np.ndarray,
    expected_leads: int = 12,
    min_samples: int = 250,
    max_abs_amplitude_mv: float = 20.0,
) -> dict:
    """
    Check basic technical validity of an ECG array shaped (samples, leads).

    This is a technical QC baseline, not a clinical quality assessment.
    """
    x = np.asarray(signal, dtype=np.float64)

    result = {
        "signal_ok": 0,
        "qc_reason": "",
        "n_samples": 0,
        "n_leads": 0,
    }

    if x.ndim != 2:
        result["qc_reason"] = "invalid_dimensions"
        return result

    n_samples, n_leads = x.shape
    result["n_samples"] = int(n_samples)
    result["n_leads"] = int(n_leads)

    if n_leads != expected_leads:
        result["qc_reason"] = "unexpected_lead_count"
        return result

    if n_samples < min_samples:
        result["qc_reason"] = "record_too_short"
        return result

    if not np.isfinite(x).all():
        result["qc_reason"] = "non_finite_values"
        return result

    # Detect completely flat leads.
    lead_std = np.std(x, axis=0)
    if np.any(lead_std < 1e-8):
        result["qc_reason"] = "flat_lead"
        return result

    # Conservative technical guard against extreme amplitudes.
    if np.max(np.abs(x)) > max_abs_amplitude_mv:
        result["qc_reason"] = "extreme_amplitude"
        return result

    result["signal_ok"] = 1
    result["qc_reason"] = "passed_basic_checks"
    return result
