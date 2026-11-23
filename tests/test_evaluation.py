"""Tes modul evaluasi Orang 3 (data sintetis saja). Jalankan: python -m pytest -q tests/test_evaluation.py"""
import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation import run_fold18
from src.evaluation.bootstrap import cluster_resample_indices, paired_bootstrap
from src.evaluation.core import (
    PlattCalibrator, binary_metrics, brier, calibration_intercept_slope, calibration_table,
    threshold_for_sensitivity, threshold_for_specificity,
)
from src.evaluation.predictions import CONTRACT, align, load_predictions, validate_predictions
from src.model.metrics import roc_auc


def _synthetic(n=20000, seed=0, shift=-2.0):
    """y ~ Bernoulli(sigmoid(2x - 3)); model melaporkan sigmoid(2x - 3 - shift) (terlalu tinggi bila shift < 0)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(2 * x - 3)))).astype(int)
    p = 1 / (1 + np.exp(-(2 * x - 3 - shift)))
    return y, p


def _pred_df(y, p, fold=18, subjects=None):
    n = len(y)
    return pd.DataFrame({"ed_stay_id": np.arange(n) + 1000, "subject_id": subjects if subjects is not None else np.arange(n),
                         "study_id": np.arange(n) + 5000, "fold": fold, "y_true": y, "y_prob": p})


# --- threshold ---------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rule_out_threshold_is_largest_with_sensitivity_090(seed):
    y, p = _synthetic(3000, seed)
    t = threshold_for_sensitivity(y, p, 0.90)
    assert binary_metrics(y, p, t)["sensitivity"] >= 0.90
    assert binary_metrics(y, p, np.nextafter(t, 1.0))["sensitivity"] < 0.90     # sedikit lebih besar -> gagal


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rule_in_threshold_is_smallest_with_specificity_090(seed):
    y, p = _synthetic(3000, seed)
    t = threshold_for_specificity(y, p, 0.90)
    assert binary_metrics(y, p, t)["specificity"] >= 0.90
    assert binary_metrics(y, p, np.nextafter(t, 0.0))["specificity"] < 0.90     # sedikit lebih kecil -> gagal


def test_threshold_float_guard():
    # 70 positif: butuh tepat 63 (0.9*70 = 63.00000000000001 di floating point)
    y = np.r_[np.ones(70, int), np.zeros(10, int)]
    p = np.r_[np.linspace(0.99, 0.30, 70), np.full(10, 0.1)]
    t = threshold_for_sensitivity(y, p, 0.90)
    assert binary_metrics(y, p, t)["tp"] == 63


def test_binary_metrics_known_values():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1, 0.3])
    m = binary_metrics(y, p, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 2)
    assert m["sensitivity"] == 0.5 and m["specificity"] == pytest.approx(2 / 3)
    assert m["ppv"] == 0.5 and m["npv"] == pytest.approx(2 / 3)
    assert brier(y, p) == pytest.approx(np.mean((p - y) ** 2))


# --- kalibrasi -----------------------------------------------------------------
def test_platt_recovers_known_shift():
    y, p = _synthetic(60000, 3, shift=-2.0)       # logit model = logit benar + 2 -> a = -2, b = 1
    cal = PlattCalibrator.fit(y, p)
    assert cal.a == pytest.approx(-2.0, abs=0.12) and cal.b == pytest.approx(1.0, abs=0.06)
    assert brier(y, cal.transform(p)) < brier(y, p)
    assert roc_auc(y, cal.transform(p)) == pytest.approx(roc_auc(y, p))       # monoton: AUROC tetap
    assert PlattCalibrator.from_dict(cal.to_dict()) == cal


def test_calibration_intercept_slope_on_calibrated_probs():
    y, p = _synthetic(60000, 4, shift=0.0)        # sudah terkalibrasi
    r = calibration_intercept_slope(y, p)
    assert r["calibration_intercept"] == pytest.approx(0.0, abs=0.06)
    assert r["calibration_slope"] == pytest.approx(1.0, abs=0.06)


def test_calibration_table_bins():
    y, p = _synthetic(1000, 5)
    rows = calibration_table(y, p, n_bins=10)
    assert len(rows) == 10 and sum(r["n"] for r in rows) == 1000
    assert all(rows[i]["mean_pred"] <= rows[i + 1]["mean_pred"] for i in range(9))


# --- bootstrap ---------------------------------------------------------------
def test_cluster_bootstrap_keeps_patients_whole():
    clusters = np.array([1, 1, 1, 2, 3, 3, 4])
    for idx in cluster_resample_indices(clusters, 50, seed=0):
        picked = pd.Series(clusters[idx]).value_counts()
        sizes = pd.Series(clusters).value_counts()
        # setiap pasien muncul dalam kelipatan jumlah kunjungannya
        assert all(picked[c] % sizes[c] == 0 for c in picked.index)


def test_paired_bootstrap_reproducible_and_brackets_estimate():
    y, p = _synthetic(4000, 6)
    rng = np.random.default_rng(1)
    q = 0.5 * p + 0.5 * rng.random(len(p))            # B = setengah derau acak -> jelas lebih buruk
    subj = np.arange(len(y)) // 2                      # 2 kunjungan per pasien
    metrics = {"AUROC": lambda yy, pp, _m: roc_auc(yy, pp)}
    a = paired_bootstrap(y, {"A": p, "B": q}, subj, metrics, n_boot=200, seed=7, diffs=[("A", "B")])
    b = paired_bootstrap(y, {"A": p, "B": q}, subj, metrics, n_boot=200, seed=7, diffs=[("A", "B")])
    pd.testing.assert_frame_equal(a, b)
    for _, r in a.iterrows():
        assert r["ci_low"] <= r["estimate"] <= r["ci_high"]
    diff = a[a.target == "A - B"].iloc[0]
    assert diff["estimate"] > 0 and bool(diff["ci_excludes_0"])      # A (tanpa derau) lebih baik dari B


# --- kontrak prediksi ----------------------------------------------------------
def test_validate_predictions_rejects_bad_files():
    y, p = _synthetic(100, 7)
    good = _pred_df(y, p)
    assert list(validate_predictions(good, 18).columns) == CONTRACT
    with pytest.raises(ValueError):
        validate_predictions(good.drop(columns="subject_id"), 18)
    with pytest.raises(ValueError):
        validate_predictions(pd.concat([good, good.iloc[:1]]), 18)            # ID duplikat
    with pytest.raises(ValueError):
        validate_predictions(good.assign(y_prob=good.y_prob * 3), 18)       # bukan probabilitas
    with pytest.raises(ValueError):
        validate_predictions(good, 17)                                       # fold salah


def test_fold19_is_locked(tmp_path):
    y, p = _synthetic(100, 8)
    _pred_df(y, p, fold=19).to_parquet(tmp_path / "M1_fold19.parquet")
    with pytest.raises(SystemExit):
        load_predictions("M1", 19, pred_dir=tmp_path)


def test_align_inner_join_and_consistency():
    y, p = _synthetic(50, 9)
    a = _pred_df(y, p)
    b = _pred_df(y, p).iloc[5:]
    wide, info = align({"M1": a, "M2": b})
    assert len(wide) == 45 and info["dropped_per_model"] == {"M1": 5, "M2": 0}
    with pytest.raises(ValueError):
        align({"M1": a, "M2": a.assign(y_true=1 - a.y_true)})


# --- runner ujung ke ujung (repo tiruan) -----------------------------------------
def test_run_fold18_end_to_end(tmp_path, monkeypatch):
    y, p = _synthetic(3000, 10)
    pred_dir = tmp_path / "predictions"
    pred_dir.mkdir()
    _pred_df(y, p).to_parquet(pred_dir / "M1_fold18.parquet")
    _pred_df(y, np.clip(p * 0.8, 0, 1)).to_parquet(pred_dir / "M2_fold18.parquet")
    monkeypatch.setattr("src.evaluation.predictions.PRED_DIR", pred_dir)
    monkeypatch.setattr(run_fold18, "ROOT", tmp_path)
    monkeypatch.setattr(run_fold18, "REPORT", tmp_path / "reports" / "eval_fold18.md")
    monkeypatch.setattr(run_fold18, "LOCKED", tmp_path / "reports" / "locked_fold18.json")
    monkeypatch.setattr(run_fold18, "load_predictions",
                        lambda m, f: load_predictions(m, f, pred_dir=pred_dir))
    monkeypatch.setattr(run_fold18, "pred_path", lambda m, f: pred_dir / f"{m}_fold{f}.parquet")
    run_fold18.main(["--models", "M1", "M2", "--compare", "M1:M2", "--n-boot", "50"])
    locked = json.loads((tmp_path / "reports" / "locked_fold18.json").read_text())
    assert set(locked) == {"M1", "M2"}
    t = locked["M1"]["threshold_rule_out"]["y_prob_raw"]
    assert binary_metrics(y, p, t)["sensitivity"] >= 0.90
    report = (tmp_path / "reports" / "eval_fold18.md").read_text()
    assert "## M1" in report and "Perbandingan berpasangan" in report
    assert "ed_stay_id" not in report.replace("`", "")      # tidak ada ID di laporan
