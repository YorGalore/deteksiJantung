import importlib

import numpy as np
import pandas as pd
import pytest
import wfdb

from src.constants import FOLD_COL
from src.dataset import signal_qc_core as core
from src.dataset.signal_qc_core import (
    DOWNLOAD_FAILED, LOAD_FAIL, NOT_PROCESSED, PASS, PREPROCESS_FAIL, QC_FAIL, STATUSES,
    build_contract, download_and_process, process_record, signal_ok_for,
)

NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _sig(n=5000):
    return np.random.default_rng(0).normal(0, 0.3, (n, 12)).astype("float32")


def _write(tmp_path, sig=None, units=None, name="rec"):
    wfdb.wrsamp(name, fs=500, units=units or ["mV"] * 12, sig_name=NAMES,
                p_signal=_sig() if sig is None else sig, fmt=["16"] * 12, write_dir=str(tmp_path))
    return tmp_path / name


# ---------- kontrak status ----------
@pytest.mark.parametrize("status,expected", [
    (PASS, 1), (QC_FAIL, 0), (PREPROCESS_FAIL, 0), (LOAD_FAIL, 0),
    (DOWNLOAD_FAILED, None), (NOT_PROCESSED, None),
])
def test_signal_ok_mapping(status, expected):
    assert signal_ok_for(status) == expected


def test_signal_ok_unknown_status_raises():
    with pytest.raises(ValueError):
        signal_ok_for("ok")          # status jalur lama tidak dikenal


# ---------- process_record: satu rekaman per status ----------
def test_pass(tmp_path):
    r = process_record(_write(tmp_path))
    assert (r["status"], r["signal_ok"], r["amplitude_flag"]) == (PASS, 1, 0)


def test_pass_with_nan_below_threshold(tmp_path):
    s = _sig(); s[100:300, 2] = np.nan
    r = process_record(_write(tmp_path, s))
    assert r["status"] == PASS and r["signal_ok"] == 1


def test_amplitude_flag_is_informative_only(tmp_path):
    s = _sig(); s[10, 4] = 25.0
    r = process_record(_write(tmp_path, s))
    assert r["status"] == PASS and r["signal_ok"] == 1 and r["amplitude_flag"] == 1


@pytest.mark.parametrize("mutate,reason", [
    (lambda s: s.__setitem__((slice(1500), 3), np.nan), "lead_nan_over_20pct"),
    (lambda s: s.__setitem__((slice(None), slice(0, 3)), 0.0), "three_or_more_flat_leads"),
])
def test_qc_fail_reasons(tmp_path, mutate, reason):
    s = _sig(); mutate(s)
    r = process_record(_write(tmp_path, s))
    assert (r["status"], r["signal_ok"], r["qc_reason"]) == (QC_FAIL, 0, reason)


def test_inf_is_treated_like_nan_at_array_level():
    # Inf tidak bisa ada di .dat 16-bit, tetapi QC + prepare_ecg harus tetap aman terhadapnya.
    from src.model.ecg_preprocessing import LEAD_ORDER, prepare_ecg
    from src.model.ecg_quality import check_ecg_quality_brief
    s = _sig(); s[100, 3] = np.inf; s[200:230, 5] = -np.inf
    assert check_ecg_quality_brief(s)["signal_ok"] == 1
    assert np.isfinite(prepare_ecg(s, 500, list(LEAD_ORDER))).all()
    for bad in (np.inf, np.nan):                         # lead seluruhnya non-finite -> gagal QC
        s = _sig(); s[:, 3] = bad
        r = check_ecg_quality_brief(s)
        assert r["signal_ok"] == 0 and r["qc_reason"] == "lead_nan_over_20pct"


def test_two_flat_leads_still_pass(tmp_path):
    s = _sig(); s[:, :2] = 0.0
    assert process_record(_write(tmp_path, s))["status"] == PASS


def test_load_fail_for_non_mv_units(tmp_path):
    r = process_record(_write(tmp_path, units=["uV"] * 12))
    assert (r["status"], r["signal_ok"]) == (LOAD_FAIL, 0) and "unit_invalid" in r["error"]


def test_preprocess_fail(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(core, "prepare_ecg", boom)
    r = process_record(_write(tmp_path))
    assert (r["status"], r["signal_ok"]) == (PREPROCESS_FAIL, 0) and "boom" in r["error"]


def test_preprocess_fail_on_bad_output_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "prepare_ecg", lambda *a, **k: np.zeros((12, 999), dtype="float32"))
    assert process_record(_write(tmp_path))["status"] == PREPROCESS_FAIL


def test_not_processed_when_missing_or_truncated(tmp_path):
    assert process_record(tmp_path / "ghost")["status"] == NOT_PROCESSED
    base = _write(tmp_path)
    dat = base.with_suffix(".dat")
    dat.write_bytes(dat.read_bytes()[:-1000])             # unduhan terpotong
    r = process_record(base)
    assert r["status"] == NOT_PROCESSED and r["signal_ok"] is None


# ---------- download_and_process ----------
def _rows(n):
    return pd.DataFrame({"ed_stay_id": range(n), "file_name": [f"f{i}" for i in range(n)],
                         FOLD_COL: [0] * n, "label_ami": [0] * n})


def test_download_failed_is_not_pass_and_not_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "fetch_record", lambda *a, **k: (False, "HTTPError: 502"))
    res = download_and_process(_rows(2).itertuples(index=False), "u/", tmp_path, str, 2)
    assert {r["status"] for r in res} == {DOWNLOAD_FAILED}
    assert all(r["signal_ok"] is None and r["error"] == "HTTPError: 502" for r in res)


def test_download_ok_then_processed(tmp_path, monkeypatch):
    _write(tmp_path, name="f0")
    monkeypatch.setattr(core, "fetch_record", lambda *a, **k: (True, "cached"))
    res = download_and_process(_rows(1).itertuples(index=False), "u/", tmp_path, str, 1)
    assert res[0]["status"] == PASS and res[0]["signal_ok"] == 1


# ---------- build_contract ----------
def test_contract_one_row_per_master_id_and_unchecked_never_pass():
    results = pd.DataFrame({
        "ed_stay_id": [1, 2, 3, 4],
        "status": [PASS, QC_FAIL, DOWNLOAD_FAILED, PREPROCESS_FAIL],
        "signal_ok": [1, 1, 1, 1],                        # sengaja salah: harus diabaikan
        "qc_reason": ["", "x", "", ""], "amplitude_flag": [0, 0, np.nan, 0], "error": ["", "", "e", ""],
    })
    c = build_contract(results, [1, 2, 3, 4, 5, 6])
    assert list(c["ed_stay_id"]) == [1, 2, 3, 4, 5, 6]
    assert list(c["status"]) == [PASS, QC_FAIL, DOWNLOAD_FAILED, PREPROCESS_FAIL, NOT_PROCESSED, NOT_PROCESSED]
    assert c["signal_ok"].isna().tolist() == [False, False, True, False, True, True]
    assert c["signal_ok"].dropna().tolist() == [1, 0, 0]
    assert str(c["signal_ok"].dtype) == "Int8"


def test_contract_rejects_legacy_status_and_duplicate_master_ids():
    bad = pd.DataFrame({"ed_stay_id": [1], "status": ["ok"]})
    with pytest.raises(ValueError):
        build_contract(bad, [1])
    with pytest.raises(ValueError):
        build_contract(bad.assign(status=PASS), [1, 1])


def test_contract_csv_roundtrip_keeps_empty_signal_ok(tmp_path):
    c = build_contract(pd.DataFrame({"ed_stay_id": [1], "status": [PASS]}), [1, 2])
    p = tmp_path / "signal_qc.csv"
    c.to_csv(p, index=False)
    back = pd.read_csv(p)
    assert back["signal_ok"].isna().tolist() == [False, True]


# ---------- konsistensi dengan pipeline dataset ----------
def test_dataset_loads_only_pass_rows(tmp_path):
    from src.model.ecg_dataset import load_record_table
    master = pd.DataFrame({"ed_stay_id": range(1, 7), "file_name": [f"f{i}" for i in range(1, 7)],
                           FOLD_COL: [0, 1, 2, 3, 4, 5], "label_ami": [0, 1, 0, 1, 0, 1],
                           "internal_val_0_16": 0})
    mp = tmp_path / "m.parquet"; master.to_parquet(mp)
    contract = build_contract(pd.DataFrame({
        "ed_stay_id": [1, 2, 3, 4], "status": [PASS, QC_FAIL, DOWNLOAD_FAILED, PASS]}), master["ed_stay_id"])
    qp = tmp_path / "signal_qc.csv"; contract.to_csv(qp, index=False)
    t = load_record_table(qp, mp, range(0, 17))
    assert sorted(t["ed_stay_id"]) == [1, 4]


def test_core_uses_same_functions_as_dataset():
    from src.model import ecg_dataset
    assert core.read_ecg_record is ecg_dataset.read_ecg_record
    assert core.prepare_ecg is ecg_dataset.prepare_ecg


@pytest.mark.parametrize("mod", ["run_signal_qc_pilot", "run_signal_qc_sample", "run_signal_qc_brief", "run_stage1_subset"])
def test_runners_have_no_legacy_path(mod):
    m = importlib.import_module(f"src.dataset.{mod}")
    for legacy in ("check_ecg_quality", "load_ecg_record", "preprocess_ecg", "download_file"):
        assert not hasattr(m, legacy), f"{mod} masih memakai {legacy}"


# ---------- pembatasan fold ----------
def test_pilot_only_takes_folds_below_17():
    from src.dataset.run_signal_qc_pilot import LIMIT, draw_pilot
    m = pd.DataFrame({"ed_stay_id": range(200), "file_name": [f"f{i}" for i in range(200)],
                      FOLD_COL: list(range(20)) * 10, "label_ami": 0})
    d = draw_pilot(m)
    assert len(d) == LIMIT and (d[FOLD_COL] < 17).all()
    assert draw_pilot(m[m[FOLD_COL] >= 17]).empty


def test_sample_only_takes_folds_below_17():
    from src.dataset.run_signal_qc_sample import N_NEG, N_POS, draw_sample
    n = 4000
    m = pd.DataFrame({"ed_stay_id": range(n), "file_name": [f"f{i}" for i in range(n)],
                      FOLD_COL: [i % 20 for i in range(n)], "label_ami": [i % 2 for i in range(n)]})
    s = draw_sample(m)
    assert len(s) == N_POS + N_NEG and (s[FOLD_COL] < 17).all()


# ---------- tahap 1 + runner yang bisa dilanjutkan ----------
def _master(n=6000):
    return pd.DataFrame({"ed_stay_id": range(n), "file_name": [f"f{i}" for i in range(n)],
                         FOLD_COL: [i % 20 for i in range(n)], "label_ami": [1 if i % 5 == 0 else 0 for i in range(n)]})


def test_stage1_subset_folds_size_and_contains_old_sample():
    from src.dataset.run_signal_qc_sample import draw_sample
    from src.dataset.run_stage1_subset import draw_stage1
    m = _master()
    s = draw_stage1(m, 150, 350)
    assert len(s) == 500 and (s[FOLD_COL] < 17).all() and s["file_name"].is_unique
    assert s["label_ami"].sum() == 150
    assert set(draw_sample(m)["ed_stay_id"]) <= set(s["ed_stay_id"])
    assert s.equals(draw_stage1(m, 150, 350))                  # deterministik
    with pytest.raises(ValueError):
        draw_stage1(m, 50, 350)


def test_run_sample_resume_retries_only_download_failed(tmp_path, monkeypatch):
    from src.dataset import run_signal_qc_sample as rs
    out = tmp_path / "o.csv"
    sample = _rows(4)
    old = pd.DataFrame({"ed_stay_id": [0, 1, 2], "fold": 0, "label_ami": 0,
                        "status": [PASS, QC_FAIL, DOWNLOAD_FAILED], "signal_ok": [1, 0, np.nan],
                        "qc_reason": "", "amplitude_flag": np.nan, "max_nan_fraction": np.nan,
                        "n_flat_leads": np.nan, "max_abs": np.nan, "error": ""})
    old.to_csv(out, index=False)
    seen = []

    def fake(rows, *a, on_row=None, **k):
        for i, r in enumerate(list(rows), 1):
            seen.append(r.ed_stay_id)
            on_row(i, 2, {"ed_stay_id": r.ed_stay_id, "fold": 0, "label_ami": 0, "status": PASS, "signal_ok": 1,
                          "qc_reason": "", "amplitude_flag": 0, "max_nan_fraction": 0.0,
                          "n_flat_leads": 0, "max_abs": 1.0, "error": ""})
        return []

    monkeypatch.setattr(rs, "download_and_process", fake)
    rs.run_sample(sample, output=out, workers=1)
    assert sorted(seen) == [2, 3]                              # pass/qc_fail tidak diulang
    assert sorted(pd.read_csv(out)["ed_stay_id"]) == [0, 1, 2, 3]


def test_run_sample_refuses_locked_folds_and_legacy_file(tmp_path):
    from src.dataset import run_signal_qc_sample as rs
    bad = _rows(2).assign(**{FOLD_COL: [0, 17]})
    with pytest.raises(AssertionError):
        rs.run_sample(bad, output=tmp_path / "o.csv")
    legacy = tmp_path / "legacy.csv"
    pd.DataFrame({"ed_stay_id": [0], "status": ["ok"]}).to_csv(legacy, index=False)
    with pytest.raises(SystemExit):
        rs.run_sample(_rows(2), output=legacy)


def test_stage1_runner_output_is_configurable_and_default_unchanged():
    from src.dataset.run_stage1_subset import OUTPUT, parse_args
    a = parse_args([])
    assert a.output == OUTPUT and (a.n_pos, a.n_neg) == (300, 700)
    b = parse_args(["--n-pos", "4134", "--n-neg", "12402", "--output", "x/stage2a.csv"])
    assert str(b.output).replace("\\", "/") == "x/stage2a.csv" and (b.n_pos, b.n_neg) == (4134, 12402)


def test_larger_draw_contains_smaller_draw_with_same_seed():
    from src.dataset.run_stage1_subset import draw_stage1
    m = _master(8000)
    small, large = draw_stage1(m, 300, 700), draw_stage1(m, 400, 1200)
    assert set(small["ed_stay_id"]) <= set(large["ed_stay_id"])


# ---------- QC fold evaluasi ----------
def test_eval_qc_only_draws_requested_eval_folds_and_locks_fold_19(tmp_path):
    from src.dataset import run_eval_qc as ev
    m = _master(3000)
    d = ev.draw_eval(m, [17, 18])
    assert set(d[FOLD_COL]) == {17, 18} and d["file_name"].is_unique
    with pytest.raises(SystemExit):
        ev.draw_eval(m, [3])                                    # bukan fold evaluasi
    with pytest.raises(SystemExit):
        ev.draw_eval(m, [16, 17])
    with pytest.raises(SystemExit):
        ev.main(["--folds", "19", "--output", str(tmp_path / "x.csv")])   # tanpa --include-fold-19


def test_run_sample_only_allows_explicit_extra_folds(tmp_path):
    from src.dataset import run_signal_qc_sample as rs
    s = _rows(2).assign(**{FOLD_COL: [17, 18]})
    with pytest.raises(AssertionError):
        rs.run_sample(s, output=tmp_path / "o.csv")                         # default: hanya fold 0-16
    with pytest.raises(AssertionError):
        rs.run_sample(s, output=tmp_path / "o.csv", allowed_folds={17})     # 18 tidak diizinkan


def test_summary_reports_status_per_fold(capsys):
    from src.dataset.run_signal_qc_sample import summarize
    res = pd.DataFrame({"ed_stay_id": [1, 2, 3], "fold": [0, 0, 5], "label_ami": [0, 1, 0],
                        "status": [PASS, QC_FAIL, PASS], "signal_ok": [1, 0, 1], "qc_reason": ["", "x", ""],
                        "amplitude_flag": [0, np.nan, 0], "max_abs": [1.0, np.nan, 2.0], "error": ["", "", ""]})
    summarize(res)
    assert "Status per fold" in capsys.readouterr().out


def test_interrupt_cancels_queue_instead_of_running_everything(tmp_path, monkeypatch):
    import time
    calls = []

    def slow_fetch(base_url, rel, cache, **k):
        calls.append(rel)
        time.sleep(0.2)
        return False, "x"

    monkeypatch.setattr(core, "fetch_record", slow_fetch)

    def stop_after_first(i, n, rec):
        raise KeyboardInterrupt

    t = time.time()
    with pytest.raises(KeyboardInterrupt):
        download_and_process(_rows(60).itertuples(index=False), "u/", tmp_path, str, 2, on_row=stop_after_first)
    time.sleep(0.6)                                   # beri waktu unduhan yang sedang jalan selesai
    assert len(calls) < 60 and time.time() - t < 5     # antrean dibatalkan, tidak berjalan sampai habis
