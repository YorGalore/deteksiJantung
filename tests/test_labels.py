"""Unit test parsing ICD dan label. Jalankan: pytest -q"""
import numpy as np
import pandas as pd

from src.dataset.labels import add_labels, label_from_codes, parse_codes


# ---------- parse_codes ----------
def test_parse_string_list():
    assert parse_codes("['I214', 'E119']") == ["I214", "E119"]


def test_parse_dots_and_lowercase():
    assert parse_codes("['i21.4', 'E11.9']") == ["I214", "E119"]


def test_parse_empty_values():
    assert parse_codes(np.nan) == []
    assert parse_codes(None) == []
    assert parse_codes("[]") == []
    assert parse_codes("") == []


def test_parse_python_list():
    assert parse_codes(["I21A1", "I10"]) == ["I21A1", "I10"]


# ---------- label_from_codes ----------
def test_nstemi_positive():
    lab = label_from_codes(["I214"])
    assert lab["label_ami"] == 1 and lab["ami_nstemi"] == 1 and lab["ami_stemi"] == 0


def test_stemi_subcodes_positive():
    for code in ["I2101", "I2109", "I2111", "I2129", "I213"]:
        lab = label_from_codes([code])
        assert lab["label_ami"] == 1 and lab["ami_stemi"] == 1, code


def test_old_mi_is_negative():
    # I252 (MI lama) dan I2510 (PJK) bukan AMI
    assert label_from_codes(["I252", "I2510"])["label_ami"] == 0


def test_similar_codes_negative():
    for code in ["I20", "I200", "I24", "I248", "I10", "R079"]:
        assert label_from_codes([code])["label_ami"] == 0, code


def test_subsequent_mi_positive():
    lab = label_from_codes(["I220"])
    assert lab["label_ami"] == 1 and lab["ami_subsequent"] == 1


def test_type2_only():
    lab = label_from_codes(["I21A1", "I10"])
    assert lab["label_ami"] == 1 and lab["ami_type2"] == 1 and lab["ami_type2_only"] == 1


def test_type2_with_nstemi_is_not_type2_only():
    lab = label_from_codes(["I21A1", "I214"])
    assert lab["ami_type2"] == 1 and lab["ami_type2_only"] == 0


def test_other_mi_is_not_type2():
    lab = label_from_codes(["I21A9"])
    assert lab["label_ami"] == 1 and lab["ami_other"] == 1 and lab["ami_type2"] == 0 and lab["ami_type2_only"] == 0


def test_unspecified():
    lab = label_from_codes(["I219"])
    assert lab["ami_unspecified"] == 1 and lab["ami_stemi"] == 0 and lab["ami_nstemi"] == 0


# ---------- add_labels ----------
def test_add_labels_and_source():
    df = pd.DataFrame({
        "all_diag_all": ["['I214']", "['I10']", "['I219']"],
        "all_diag_hosp": ["['I214']", np.nan, "[]"],
        "fold": [0, 1, 2],
    })
    out = add_labels(df)
    assert out["label_ami"].tolist() == [1, 0, 1]
    assert out["label_source"].tolist() == ["hosp", "ed", "ed"]
    assert len(out) == len(df)
