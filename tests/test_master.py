"""Unit test tabel master dan pemeriksaannya. Jalankan: pytest -q"""
import numpy as np
import pandas as pd

from src.constants import EMR_FEATURES, MM_FEATURES
from src.dataset.build_master import build_master, ecg_download_urls, internal_val_subjects
from src.dataset.check_master import auroc, run_checks


def _inputs(n_subj=60):
    rng = np.random.default_rng(0)
    subj = np.repeat(np.arange(n_subj), 2)                 # 2 kunjungan per pasien
    fold = subj % 20
    n = len(subj)
    labeled = pd.DataFrame({
        "ed_stay_id": np.arange(1000, 1000 + n), "subject_id": subj, "study_id": np.arange(n),
        "file_name": [f"files/p1000/p{s}/s{i}/{i}" for i, s in enumerate(subj)], "fold": fold,
        "label_ami": rng.integers(0, 2, n), "label_source": "hosp",
        "ami_stemi": 0, "ami_nstemi": 0, "ami_unspecified": 0, "ami_type2": 0,
        "ami_other": 0, "ami_subsequent": 0, "ami_type2_only": 0, "first_ecg_matches_no0": 1,
        "age": 60, "gender": "M",
    })
    emr = pd.DataFrame({"ed_stay_id": labeled["ed_stay_id"], "subject_id": subj})
    for c in EMR_FEATURES:
        emr[c] = rng.normal(size=n)
    emr["arrival_transport"] = rng.choice(["WALK IN", "AMBULANCE"], n)
    emr["cc_explicit_mi"], emr["cc_missing"], emr["medrecon_available"] = 0, 0, 1
    mm = pd.DataFrame({"ed_stay_id": labeled["ed_stay_id"]})
    for c in MM_FEATURES:
        mm[c] = rng.normal(size=n)
    mm["mm_missing_count"] = 0
    return labeled, emr, mm


def test_internal_val_is_patient_based_and_reproducible():
    labeled, _, _ = _inputs()
    a = internal_val_subjects(labeled, list(range(17)), 0.1, seed=42)
    b = internal_val_subjects(labeled, list(range(17)), 0.1, seed=42)
    assert np.array_equal(a, b)
    assert set(labeled.loc[labeled.subject_id.isin(a), "fold"]) <= set(range(17))


def test_build_master_draft():
    master, info = build_master(*_inputs(), seed=42)
    assert len(master) == 120 and master["ed_stay_id"].is_unique
    assert master.loc[master.fold >= 17, "internal_val_0_16"].sum() == 0
    assert master.loc[master.fold >= 18, "internal_val_0_17"].sum() == 0
    assert master["signal_ok"].isna().all()
    assert "gender" not in master.columns and "chiefcomplaint" not in master.columns


def test_build_master_with_signal_qc_drops_failed():
    labeled, emr, mm = _inputs()
    qc = pd.DataFrame({"ed_stay_id": labeled["ed_stay_id"], "signal_ok": 1})
    qc.loc[:4, "signal_ok"] = 0
    master, info = build_master(labeled, emr, mm, signal_qc=qc)
    assert len(master) == 115 and info["signal_fail"] == 5


def test_checks_pass_on_clean_table():
    master, _ = build_master(*_inputs())
    checks, _ = run_checks(master)
    assert all(c["hasil"] != "GAGAL" for c in checks), checks


def test_checks_flag_leaky_feature():
    master, _ = build_master(*_inputs())
    master["heartrate"] = master["label_ami"] * 10.0      # fitur yang membocorkan label
    checks, auc = run_checks(master)
    flagged = [c for c in checks if c["hasil"] == "PERIKSA"]
    assert flagged and "heartrate" in flagged[0]["detail"]


def test_auroc_simple():
    assert auroc(pd.Series([0, 0, 1, 1]), pd.Series([1, 2, 3, 4])) == 1.0
    assert auroc(pd.Series([0, 0, 1, 1]), pd.Series([4, 3, 2, 1])) == 0.0
    assert np.isnan(auroc(pd.Series([1, 1]), pd.Series([1.0, 2.0])))


def test_ecg_urls():
    master, _ = build_master(*_inputs())
    urls = ecg_download_urls(master.head(1))
    assert urls[0].endswith(".hea") and urls[1].endswith(".dat")
    assert urls[0].startswith("https://physionet.org/files/mimic-iv-ecg/1.0/files/")


def test_build_master_never_treats_unchecked_as_pass():
    from src.dataset.signal_qc_core import (
        DOWNLOAD_FAILED, NOT_PROCESSED, PASS, QC_FAIL, build_contract)
    labeled, emr, mm = _inputs()
    ids = list(labeled["ed_stay_id"])
    status = pd.DataFrame({"ed_stay_id": ids[:4], "status": [PASS, QC_FAIL, DOWNLOAD_FAILED, PASS]})
    qc = build_contract(status, ids)                      # sisanya not_processed
    master, info = build_master(labeled, emr, mm, signal_qc=qc)
    assert len(master) == 2 and (master["signal_ok"] == 1).all()
    assert info["signal_fail"] == 1 and info["signal_missing"] == len(ids) - 3
    assert (qc["status"] == NOT_PROCESSED).sum() == len(ids) - 4
