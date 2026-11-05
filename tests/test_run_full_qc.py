import pandas as pd
import pytest

from src.constants import FOLD_COL
from src.dataset import run_full_qc as rf


def _m():
    return pd.DataFrame({"ed_stay_id": [1, 2, 3, 4, 5], "file_name": ["a", "b", "b", "c", "d"],
                         FOLD_COL: [0, 5, 5, 18, 19], "label_ami": [0, 1, 1, 0, 0]})


def test_select_excludes_fold_19_and_dedups():
    s = rf.select_records(_m(), False)
    assert list(s["ed_stay_id"]) == [1, 2, 4]
    assert list(rf.select_records(_m(), True)["ed_stay_id"]) == [1, 2, 4, 5]


def test_expand_gives_result_to_all_ids_sharing_file():
    res = pd.DataFrame({"ed_stay_id": [1, 2, 4], "fold": [0, 5, 18], "label_ami": [0, 1, 0],
                        "status": ["pass", "qc_fail", "pass"]})
    out = rf.expand_to_all_ids(res, _m()).sort_values("ed_stay_id")
    assert list(out["ed_stay_id"]) == [1, 2, 3, 4]
    assert dict(zip(out["ed_stay_id"], out["status"])) == {1: "pass", 2: "qc_fail", 3: "qc_fail", 4: "pass"}


def test_run_refuses_fold_19_in_sample_without_flag(tmp_path, monkeypatch):
    from src.dataset import run_signal_qc_sample as rs
    sample = rf.select_records(_m(), True)
    with pytest.raises(AssertionError):
        rs.run_sample(sample, output=tmp_path / "o.csv", allowed_folds=set(range(19)))