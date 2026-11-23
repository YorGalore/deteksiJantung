"""Tes meta-learner M5/M6 dengan prediksi SINTETIS. Jalankan: python -m pytest -q tests/test_meta.py

Tidak memakai scikit-learn/xgboost/torch, jadi aman dijalankan dalam satu proses pytest di macOS.
"""
import json

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad
from scipy.special import expit

from src.evaluation import predictions as P
from src.model import train_meta
from src.model.meta_learner import MetaModel, bootstrap_coefficients, fit_l2_logistic
from src.model.metrics import roc_auc


def _write_synthetic_predictions(pred_dir, seed=0, n=7500, drop_m2=0):
    """Dua sinyal laten independen: s_ekg (dipakai M1, sebagian M3) dan s_emr (dipakai M2).

    Probabilitas model dasar sengaja terlalu tinggi (meniru scale_pos_weight / subsampling).
    """
    rng = np.random.default_rng(seed)
    pred_dir.mkdir(parents=True, exist_ok=True)
    for fold in (17, 18):
        s_ekg, s_emr = rng.normal(size=n), rng.normal(size=n)
        y = (rng.random(n) < expit(-4.0 + 1.0 * s_ekg + 1.0 * s_emr)).astype("int8")
        ids = np.arange(n) + fold * 100_000
        base = {"ed_stay_id": ids, "subject_id": rng.integers(0, int(n * 0.7), n) + fold * 100_000,
                "study_id": ids + 5_000_000, "fold": fold, "y_true": y}
        probs = {"M1": expit(1.0 * s_ekg + rng.normal(0, 0.5, n) + 2.5),
                 "M2": expit(0.9 * s_emr + rng.normal(0, 0.5, n) + 3.4),
                 "M3": expit(0.5 * s_ekg + rng.normal(0, 0.8, n) + 3.4)}
        for m, p in probs.items():
            d = pd.DataFrame({**base, "y_prob": p})
            if m == "M2" and drop_m2:
                d = d.iloc[drop_m2:]
            d.to_parquet(pred_dir / f"{m}_fold{fold}.parquet", index=False)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PRED_DIR", tmp_path / "predictions")
    monkeypatch.setattr(train_meta, "ROOT", tmp_path)
    monkeypatch.setattr(train_meta, "OUT_DIR", tmp_path / "data/interim/meta")
    monkeypatch.setattr(train_meta, "REPORTS", tmp_path / "reports")
    _write_synthetic_predictions(tmp_path / "predictions")
    return tmp_path


# --- regresi logistik L2 --------------------------------------------------------
def test_l2_logistic_recovers_coefficients_when_weakly_regularized():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(40000, 2))
    y = (rng.random(40000) < expit(-1.0 + 0.8 * Z[:, 0] - 0.5 * Z[:, 1])).astype(float)
    b, w = fit_l2_logistic(Z, y, C=1e6)
    assert b == pytest.approx(-1.0, abs=0.05) and w == pytest.approx([0.8, -0.5], abs=0.05)


def test_l2_penalty_shrinks_coefficients():
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(500, 2))
    y = (rng.random(500) < expit(0.5 + 1.0 * Z[:, 0])).astype(float)
    _, w_weak = fit_l2_logistic(Z, y, C=1e6)
    _, w_strong = fit_l2_logistic(Z, y, C=0.001)
    assert np.linalg.norm(w_strong) < np.linalg.norm(w_weak)


def test_gradient_is_correct():
    rng = np.random.default_rng(3)
    Z, y, C = rng.normal(size=(200, 2)), (rng.random(200) < 0.3).astype(float), 0.7

    def obj(t):
        z = t[0] + Z @ t[1:]
        return C * np.sum(np.logaddexp(0, z) - y * z) + 0.5 * t[1:] @ t[1:]

    def grad(t):
        r = expit(t[0] + Z @ t[1:]) - y
        return np.r_[C * r.sum(), C * (Z.T @ r) + t[1:]]

    assert check_grad(obj, grad, rng.normal(size=3)) < 1e-4
    b, w = fit_l2_logistic(Z, y, C)
    assert np.abs(grad(np.r_[b, w])).max() < 1e-4                  # optimum: gradien ~ 0


def _objective_gradient(Z, y, C, b, w):
    r = expit(b + Z @ w) - y
    return np.r_[C * r.sum(), C * (Z.T @ r) + w]


@pytest.mark.parametrize("C", [0.01, 1.0, 1e4])
@pytest.mark.parametrize("prev", [0.005, 0.03, 0.3])
def test_newton_reaches_stationary_point(C, prev):
    """Konvergensi dicek langsung lewat gradien di solusi (bukan status optimizer). Regresi untuk v3 (L-BFGS-B ABNORMAL di Mac)."""
    rng = np.random.default_rng(11)
    n = 7500
    Z = rng.normal(size=(n, 2))
    y = (rng.random(n) < expit(np.log(prev / (1 - prev)) + 0.9 * Z[:, 0] + 0.7 * Z[:, 1])).astype(float)
    b, w = fit_l2_logistic(Z, y, C)
    g = _objective_gradient(Z, y, C, b, w)
    assert np.abs(g).max() <= 1e-6 * max(1.0, C * n)


def test_many_bootstrap_fits_never_fail():
    """Kasus yang gagal di Mac (bootstrap seed 7) + ratusan resample lain: semua fit harus berhasil."""
    w = _wide()
    for seed in (7, 8, 9):
        out = bootstrap_coefficients(w, ["M1", "M2"], n_boot=200, seed=seed)
        assert out["n_valid"].iloc[0] == 200


# --- MetaModel -----------------------------------------------------------------
def _wide(n=3000, seed=4):
    rng = np.random.default_rng(seed)
    s1, s2 = rng.normal(size=n), rng.normal(size=n)
    y = (rng.random(n) < expit(-3 + s1 + s2)).astype("int8")
    return pd.DataFrame({"ed_stay_id": np.arange(n), "subject_id": np.arange(n) // 2, "y_true": y,
                         "p_M1": expit(s1 + 2.5), "p_M2": expit(s2 + 3.4)})


def test_roundtrip_and_calibration_invariance():
    w = _wide()
    m = MetaModel.fit(w, ["M1", "M2"])
    m2 = MetaModel.from_dict(json.loads(json.dumps(m.to_dict())))
    np.testing.assert_allclose(m.predict(w), m2.predict(w), atol=1e-12)
    # Platt pada input (a + b*logit, b > 0) tidak mengubah prediksi meta-learner -> kalibrasi model dasar tidak perlu
    w_cal = w.copy()
    w_cal["p_M1"] = expit(-3.4 + 0.73 * np.log(w["p_M1"] / (1 - w["p_M1"])))
    np.testing.assert_allclose(MetaModel.fit(w_cal, ["M1", "M2"]).predict(w_cal), m.predict(w), atol=1e-6)


def test_bootstrap_ci_contains_estimate_and_is_reproducible():
    w = _wide()
    a = bootstrap_coefficients(w, ["M1", "M2"], n_boot=100, seed=7)
    b = bootstrap_coefficients(w, ["M1", "M2"], n_boot=100, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert ((a.ci_low <= a.estimate) & (a.estimate <= a.ci_high)).all()
    assert (a.loc[a.term != "intercept", "ci_low"] > 0).all()          # kedua sinyal nyata


# --- ujung ke ujung (CLI) --------------------------------------------------------
def test_train_m5_end_to_end(repo):
    train_meta.main(["--model", "M5", "--n-boot", "50"])
    assert sorted(p.name for p in (repo / "predictions").glob("M5_*")) == ["M5_fold18.parquet"]   # bukan 17/19
    d = P.validate_predictions(pd.read_parquet(repo / "predictions/M5_fold18.parquet"), expected_fold=18)
    m1 = pd.read_parquet(repo / "predictions/M1_fold18.parquet")
    m2 = pd.read_parquet(repo / "predictions/M2_fold18.parquet")
    assert len(d) == len(m1)
    # dua sinyal independen -> fusion lebih baik dari masing-masing model dasar
    assert roc_auc(d.y_true, d.y_prob) > max(roc_auc(m1.y_true, m1.y_prob), roc_auc(m2.y_true, m2.y_prob)) + 0.02
    # probabilitas fusion terkalibrasi ke prevalensi fold 17 (bukan terlalu tinggi seperti input)
    assert abs(d.y_prob.mean() - d.y_true.mean()) < 0.015
    meta = json.loads((repo / "data/interim/meta/M5_meta.json").read_text())
    assert meta["inputs"] == ["M1", "M2"] and all(c > 0 for c in meta["coef_standardized"])
    report = (repo / "reports/meta_M5.md").read_text()
    assert "odds ratio" in report and "ed_stay_id" not in report


def test_m6_uses_m3_and_m2(repo):
    train_meta.main(["--model", "M6", "--n-boot", "20"])
    meta = json.loads((repo / "data/interim/meta/M6_meta.json").read_text())
    assert meta["inputs"] == ["M3", "M2"]
    assert (repo / "predictions/M6_fold18.parquet").exists()


def test_locked_and_invalid_folds(repo):
    train_meta.main(["--model", "M5", "--n-boot", "20"])
    with pytest.raises(SystemExit, match="dikunci"):
        train_meta.main(["--model", "M5", "--predict-only", "--folds", "19"])
    with pytest.raises(SystemExit, match="data latih"):
        train_meta.main(["--model", "M5", "--predict-only", "--folds", "17"])
    before = pd.read_parquet(repo / "predictions/M5_fold18.parquet")
    train_meta.main(["--model", "M5", "--predict-only", "--folds", "18"])
    after = pd.read_parquet(repo / "predictions/M5_fold18.parquet")
    np.testing.assert_allclose(before.y_prob, after.y_prob, atol=1e-12)


def test_inner_join_when_input_rows_differ(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "PRED_DIR", tmp_path / "predictions")
    monkeypatch.setattr(train_meta, "ROOT", tmp_path)
    monkeypatch.setattr(train_meta, "OUT_DIR", tmp_path / "meta")
    monkeypatch.setattr(train_meta, "REPORTS", tmp_path / "reports")
    _write_synthetic_predictions(tmp_path / "predictions", drop_m2=5)    # M2 kehilangan 5 baris
    train_meta.main(["--model", "M5", "--n-boot", "20"])
    d = pd.read_parquet(tmp_path / "predictions/M5_fold18.parquet")
    assert len(d) == len(pd.read_parquet(tmp_path / "predictions/M1_fold18.parquet")) - 5


def test_study_id_mismatch_is_rejected(repo):
    p = repo / "predictions/M2_fold17.parquet"
    d = pd.read_parquet(p)
    d.loc[0, "study_id"] = -1
    d.to_parquet(p, index=False)
    with pytest.raises(ValueError, match="study_id"):
        train_meta.main(["--model", "M5", "--n-boot", "20"])


def test_too_few_positives_in_fold17(repo):
    for m in ("M1", "M2"):
        p = repo / f"predictions/{m}_fold17.parquet"
        d = pd.read_parquet(p)
        d["y_true"] = 0
        d.loc[d.index[:50], "y_true"] = 1
        d.to_parquet(p, index=False)
    with pytest.raises(SystemExit, match="cross-fitting"):
        train_meta.main(["--model", "M5", "--n-boot", "20"])
