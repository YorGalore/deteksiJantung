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


def check_ecg_quality_brief(
    signal: np.ndarray,
    expected_leads: int = 12,
    min_samples: int = 250,
    max_nan_fraction: float = 0.20,
    max_flat_leads: int = 2,
    amplitude_flag_mv: float = 20.0,
) -> dict:
    """QC sesuai PROJECT_BRIEF 10.2 pada array (samples, leads), BOLEH berisi NaN.

    Gagal jika ada lead dengan > max_nan_fraction sampel NaN, atau lebih dari
    max_flat_leads lead datar (default: >= 3 lead datar). Amplitudo di atas
    amplitude_flag_mv hanya menyalakan `amplitude_flag` (informatif, BUKAN
    alasan eksklusi) sampai tim memutuskan.
    """
    result = {
        "signal_ok": 0,
        "qc_reason": "",
        "n_samples": 0,
        "n_leads": 0,
        "max_nan_fraction": float("nan"),
        "n_flat_leads": -1,
        "amplitude_flag": 0,
    }
    x = np.asarray(signal, dtype=np.float64)

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

    finite = np.isfinite(x)
    nan_fraction = (~finite).mean(axis=0)
    result["max_nan_fraction"] = float(nan_fraction.max())

    lead_std = np.array([x[finite[:, j], j].std() if finite[:, j].any() else 0.0 for j in range(n_leads)])
    n_flat = int((lead_std < 1e-8).sum())
    result["n_flat_leads"] = n_flat

    if finite.any() and np.abs(x[finite]).max() > amplitude_flag_mv:
        result["amplitude_flag"] = 1

    if nan_fraction.max() > max_nan_fraction:
        result["qc_reason"] = "lead_nan_over_20pct"
        return result
    if n_flat > max_flat_leads:
        result["qc_reason"] = "three_or_more_flat_leads"
        return result

    result["signal_ok"] = 1
    result["qc_reason"] = "passed_brief_checks"
    return result
