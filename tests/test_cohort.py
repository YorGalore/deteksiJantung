"""Unit test pembentukan kohort. Jalankan: pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src.dataset.cohort import build_cohort


def _ext():
    # Data buatan kecil (bukan data pasien)
    return pd.DataFrame({
        "file_name": [f"f{i}" for i in range(8)],
        "study_id": [10, 11, 12, 13, 14, 15, 16, 17],
        "subject_id": [1, 1, 2, 3, 4, 5, 5, 6],
        "ecg_time": ["2150-01-01 10:05", "2150-01-01 10:00", "2150-02-01 08:00",
                     "2150-03-01 09:00", "2150-04-01 09:00", "2150-05-01 07:00",
                     "2150-05-01 07:00", "2150-06-01 07:00"],
        "ed_stay_id": [100.0, 100.0, 200.0, np.nan, 400.0, 500.0, 500.0, 600.0],
        "all_diag_all": ["['I214']", "['I214']", "['I10']", "['I10']", "[]",
                         "['R079']", "['R079']", "['I219']"],
        "all_diag_hosp": ["['I214']", "['I214']", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        "ecg_no_within_stay": [0, 0, 0, -1, 0, 0, 0, 0],
        "ecg_taken_in_ed": [True, True, True, False, True, True, True, False],
        "fold": [3, 3, 7, 9, 17, 18, 18, 19],
        "age": [60, 60, 50, 40, 70, 55, 55, 65],
        "gender": ["M", "M", "F", "F", "M", "F", "F", "M"],
    })


def test_cohort_one_row_per_stay():
    cohort, flow = build_cohort(_ext())
    assert cohort["ed_stay_id"].is_unique
    # stay 100, 200, 500 lolos; 400 tanpa diagnosis; baris subject 3 & 6 bukan EKG IGD
    assert sorted(cohort["ed_stay_id"]) == [100, 200, 500]


def test_first_ecg_by_time():
    cohort, _ = build_cohort(_ext())
    # stay 100: EKG pukul 10:00 (study 11) lebih awal dari 10:05 (study 10)
    assert cohort.loc[cohort.ed_stay_id == 100, "study_id"].item() == 11


def test_tie_break_by_study_id():
    cohort, _ = build_cohort(_ext())
    # stay 500: waktu sama, pilih study_id terkecil
    assert cohort.loc[cohort.ed_stay_id == 500, "study_id"].item() == 15


def test_flow_is_monotonic():
    _, flow = build_cohort(_ext())
    assert flow["n_ekg"].is_monotonic_decreasing
    assert flow["n_ekg"].iloc[-1] == 3


def test_subject_in_two_folds_fails():
    ext = _ext()
    ext.loc[1, "fold"] = 4  # subject 1 kini di fold 3 dan 4
    with pytest.raises(AssertionError):
        build_cohort(ext)
