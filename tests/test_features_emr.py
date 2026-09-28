"""Unit test fitur EMR. Jalankan: pytest -q"""
import numpy as np
import pandas as pd

from src.constants import FORBIDDEN_COLUMNS
from src.dataset.features_emr import (
    build_emr_features, chief_complaint_flags, clean_triage, medication_flags,
)


def _triage():
    return pd.DataFrame({
        "subject_id": [1, 2, 3],
        "stay_id": [100, 200, 300],
        "temperature": [98.6, 37.0, 101.0],     # 37 (Celsius) di luar rentang -> NaN
        "heartrate": [80, 300, 110],            # 300 -> NaN
        "resprate": [16, 18, 22],
        "o2sat": [98, 95, 88],
        "sbp": [120, 140, 90],
        "dbp": [80, 90, 60],
        "pain": ["5", "unable", "12"],          # 'unable' -> NaN, 12 -> NaN
        "acuity": [3, 2, 1],
        "chiefcomplaint": ["Chest pain, SOB", None, "STEMI"],
    })


def test_clean_triage_ranges_and_pain():
    tri, n_invalid = clean_triage(_triage())
    assert tri["temperature"].isna().tolist() == [False, True, False]
    assert tri["heartrate"].isna().tolist() == [False, True, False]
    assert tri["pain"].tolist()[0] == 5 and np.isnan(tri["pain"].tolist()[1]) and np.isnan(tri["pain"].tolist()[2])
    assert n_invalid["temperature"] == 1 and n_invalid["pain"] == 1


def test_chief_complaint_flags():
    f = chief_complaint_flags(pd.Series(["Chest pain, SOB", None, "STEMI", "back pain", "Leg pain"]))
    assert f["cc_chest_pain"].tolist() == [1, 0, 0, 0, 0]
    assert f["cc_dyspnea"].tolist() == [1, 0, 0, 0, 0]
    assert f["cc_explicit_mi"].tolist() == [0, 0, 1, 0, 0]
    assert f["cc_missing"].tolist() == [0, 1, 0, 0, 0]
    assert f["cc_back_pain"].tolist() == [0, 0, 0, 1, 0]
    assert f["cc_arm_jaw_pain"].sum() == 0


def test_cc_word_boundary():
    # 'cp' tidak boleh cocok di dalam kata lain, 'mi' tidak boleh cocok di 'vomiting'
    f = chief_complaint_flags(pd.Series(["cpr in progress", "vomiting"]))
    assert f["cc_chest_pain"].sum() == 0
    assert f["cc_explicit_mi"].sum() == 0


def test_medication_flags_missing_is_nan():
    med = pd.DataFrame({"stay_id": [100, 100, 200],
                        "name": ["Aspirin 81 mg", "Atorvastatin", "Omeprazole"]})
    out = medication_flags(med, pd.Series([100, 200, 300])).set_index("stay_id")
    assert out.loc[100, "med_antiplatelet"] == 1 and out.loc[100, "med_statin"] == 1
    assert out.loc[200, "med_antiplatelet"] == 0
    assert np.isnan(out.loc[300, "med_antiplatelet"])        # tanpa medrecon -> NaN
    assert out["medrecon_available"].tolist() == [1, 1, 0]
    assert out.loc[100, "n_meds"] == 2


def test_build_emr_features_shape_and_no_text():
    cohort = pd.DataFrame({"ed_stay_id": [100, 200, 300], "subject_id": [1, 2, 3],
                           "age": [60, 50, 70], "gender": ["M", "F", "M"]})
    edstays = pd.DataFrame({"stay_id": [100, 200, 300], "arrival_transport": ["walk in", "AMBULANCE", "WALK IN"]})
    med = pd.DataFrame({"stay_id": [100], "name": ["metoprolol"]})
    feat, _ = build_emr_features(cohort, edstays, _triage(), med)
    assert len(feat) == 3 and feat["ed_stay_id"].is_unique
    assert feat["is_male"].tolist() == [1, 0, 1]
    assert feat["arrival_transport"].tolist() == ["WALK IN", "AMBULANCE", "WALK IN"]
    # Teks bebas dan kolom terlarang tidak boleh ada
    assert "chiefcomplaint" not in feat.columns
    leaked = set(feat.columns) & (set(FORBIDDEN_COLUMNS) - {"cc_explicit_mi", "cc_missing", "medrecon_available"})
    assert not leaked, leaked
