"""Unit test fitur machine measurements. Jalankan: pytest -q"""
import numpy as np
import pandas as pd

from src.constants import MM_FEATURES
from src.dataset.features_mm import build_mm_features, clean_mm


def _mm():
    return pd.DataFrame({
        "subject_id": [1, 2, 3, 4],
        "study_id": [10, 20, 30, 40],
        "rr_interval": [1000, 800, 5000, 750],
        "p_onset": [40, 29999, 40, 345],
        "p_end": [150, 29999, 150, 450],
        "qrs_onset": [200, 200, 200, 500],
        "qrs_end": [294, 290, 150, 590],      # study 30: qrs_end < qrs_onset -> negatif
        "t_end": [596, 600, 600, 900],
        "p_axis": [60, 29999, 50, 45],
        "qrs_axis": [30, 0, 10, -20],
        "t_axis": [45, 40, 30, 50],
    })


def test_sentinel_and_derived():
    f, notes = clean_mm(_mm())
    f = f.set_index("study_id")
    assert f.loc[10, "pr_interval"] == 160
    assert f.loc[10, "qrs_duration"] == 94
    assert f.loc[10, "qt_interval"] == 396
    assert round(f.loc[10, "qtc_bazett"]) == 396          # RR 1000 ms -> QTc = QT
    assert f.loc[10, "hr_ecg"] == 60
    assert np.isnan(f.loc[20, "pr_interval"])             # p_onset sentinel
    assert np.isnan(f.loc[20, "p_axis"])
    assert notes["sentinel"]["p_onset"] == 1


def test_device_offset_gives_same_intervals():
    # Mesin dengan qrs_onset ~500: interval tetap masuk akal
    f, _ = clean_mm(_mm())
    row = f.set_index("study_id").loc[40]
    assert row["pr_interval"] == 155 and row["qrs_duration"] == 90 and row["qt_interval"] == 400


def test_out_of_range_to_nan():
    f, notes = clean_mm(_mm())
    f = f.set_index("study_id")
    assert np.isnan(f.loc[30, "rr_interval"]) and np.isnan(f.loc[30, "hr_ecg"]) and np.isnan(f.loc[30, "qtc_bazett"])
    assert np.isnan(f.loc[30, "qrs_duration"])            # negatif -> NaN
    assert notes["luar_rentang"]["rr_interval"] == 1


def test_p_wave_absent():
    f, _ = clean_mm(_mm())
    assert f.set_index("study_id")["p_wave_absent"].tolist() == [0, 1, 0, 0]


def test_no_raw_onset_columns_in_features():
    f, _ = clean_mm(_mm())
    for c in ["p_onset", "p_end", "qrs_onset", "qrs_end", "t_end"]:
        assert c not in f.columns


def test_build_mm_features_join():
    cohort = pd.DataFrame({"ed_stay_id": [1, 2, 3], "subject_id": [1, 2, 9], "study_id": [10, 20, 99]})
    feat, notes = build_mm_features(cohort, _mm())
    assert len(feat) == 3 and list(feat["ed_stay_id"]) == [1, 2, 3]
    assert notes["tanpa_data_mm"] == 1                     # stay 3 tidak punya data MM
    assert feat.loc[2, "mm_missing_count"] == len(MM_FEATURES)


def test_axis_normalized_to_minus180_180():
    mm = _mm()
    mm["t_axis"] = [197, 180, -180, 270]
    f, _ = clean_mm(mm)
    assert f["t_axis"].tolist() == [-163, 180, 180, -90]
    assert f["qrs_axis"].tolist()[3] == -20
