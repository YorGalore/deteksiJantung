import pandas as pd
import pytest

from src.dataset import build_signal_qc as b


def _master():
    return pd.DataFrame({"ed_stay_id": [1, 2, 3, 4], "file_name": ["a", "b", "b", "c"]})


def _res(ids, statuses):
    return pd.DataFrame({"ed_stay_id": ids, "status": statuses, "qc_reason": "", "amplitude_flag": 0.0, "error": ""})


def test_complete_results_make_contract_with_expansion():
    full = b.build(_master(), _res([1, 2, 4], ["pass", "qc_fail", "pass"]))
    assert list(full["ed_stay_id"]) == [1, 2, 3, 4]
    assert list(full["signal_ok"]) == [1, 0, 0, 1]
    assert full["ed_stay_id"].is_unique


def test_missing_ids_block_contract():
    with pytest.raises(SystemExit):
        b.build(_master(), _res([1, 2], ["pass", "pass"]))


def test_download_failed_blocks_contract():
    with pytest.raises(SystemExit):
        b.build(_master(), _res([1, 2, 4], ["pass", "download_failed", "pass"]))


def test_main_writes_two_column_contract(tmp_path, monkeypatch):
    mp = tmp_path / "m.parquet"
    _master().to_parquet(mp)
    monkeypatch.setattr(b, "MASTER", mp)
    f = tmp_path / "full.csv"
    _res([1, 2, 4], ["pass", "qc_fail", "pass"]).to_csv(f, index=False)
    b.main(["--full", str(f), "--contract", str(tmp_path / "c.csv"), "--status", str(tmp_path / "s.csv")])
    c = pd.read_csv(tmp_path / "c.csv")
    assert list(c.columns) == ["ed_stay_id", "signal_ok"] and len(c) == 4 and set(c.signal_ok) <= {0, 1}