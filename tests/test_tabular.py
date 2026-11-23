"""Tes model tabular M2/M3/M4 dengan tabel master SINTETIS berskema sama. Jalankan: python -m pytest -q tests/test_tabular.py"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# PENTING (macOS): torch dan xgboost membawa pustaka OpenMP (libomp) sendiri-sendiri. Bila keduanya
# termuat di SATU proses Python, training XGBoost bisa segmentation fault. Tes lain di repo memuat torch,
# jadi semua training XGBoost di sini dijalankan di proses Python TERPISAH yang tidak memuat torch.
# xgboost juga tidak di-import di proses pytest ini.
if importlib.util.find_spec("xgboost") is None:
    pytest.skip("xgboost belum terpasang", allow_module_level=True)

REPO = Path(__file__).resolve().parents[1]

from src.constants import (
    ANALYSIS_COLS, EMR_FEATURES, FORBIDDEN_COLUMNS, ID_COLS, LABEL_COLS, MM_FEATURES, SPLIT_COLS,
)
from src.evaluation.predictions import validate_predictions
from src.model.metrics import roc_auc
from src.model.tabular import ARRIVAL_CATEGORIES, MODEL_SPECS, build_matrix, sample_params, split_train_val


def synthetic_master(n=6000, seed=0) -> pd.DataFrame:
    """Skema sama dengan master_table_draft (65 kolom, tipe data sama). Label bergantung pada beberapa fitur."""
    rng = np.random.default_rng(seed)
    subj = rng.integers(0, int(n * 0.6), n)
    fold = (subj * 7919) % 20
    m = pd.DataFrame({"ed_stay_id": np.arange(n, dtype="int64") + 30_000_000, "subject_id": subj.astype("int64"),
                      "study_id": np.arange(n, dtype="int64") + 40_000_000,
                      "file_name": pd.Series([f"files/p0/p{s}/s{i}/{i}" for i, s in enumerate(subj)], dtype="str"),
                      "fold": fold.astype("int64")})
    val16 = set(rng.choice(np.unique(subj[fold <= 16]), size=int(0.1 * len(np.unique(subj[fold <= 16]))), replace=False))
    val17 = set(rng.choice(np.unique(subj[fold <= 17]), size=int(0.1 * len(np.unique(subj[fold <= 17]))), replace=False))
    m["internal_val_0_16"] = (m.subject_id.isin(val16) & (m.fold <= 16)).astype("int8")
    m["internal_val_0_17"] = (m.subject_id.isin(val17) & (m.fold <= 17)).astype("int8")
    hr = rng.normal(85, 20, n)
    chest = (rng.random(n) < 0.18).astype("int8")
    qrs = rng.normal(95, 15, n)
    logit = -4.5 + 1.8 * chest + 0.025 * (hr - 85) + 0.03 * (qrs - 95) + rng.normal(0, 0.5, n)
    m["label_ami"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype("int64")
    m["label_source"] = pd.Series(np.where(rng.random(n) < 0.66, "hosp", "ed"), dtype="str")
    for c in LABEL_COLS[2:]:
        m[c] = 0
        m[c] = m[c].astype("int64")
    for c in ["cc_explicit_mi", "cc_missing", "medrecon_available"]:
        m[c] = (rng.random(n) < 0.1).astype("int8")
    m["mm_missing_count"] = rng.integers(0, 3, n).astype("int16")
    m["first_ecg_matches_no0"] = 1
    m["signal_ok"] = pd.array([pd.NA] * n, dtype="Int8")
    m["age"] = rng.normal(60, 18, n)
    m["is_male"] = (rng.random(n) < 0.5).astype("int8")
    arr = rng.choice(ARRIVAL_CATEGORIES + [None], n, p=[0.35, 0.002, 0.05, 0.05, 0.5, 0.048])
    m["arrival_transport"] = pd.Series(arr, dtype="string")
    for c in ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity"]:
        m[c] = rng.normal(50, 10, n)
    m["heartrate"] = np.where(rng.random(n) < 0.05, np.nan, hr)
    cc_cols = [c for c in EMR_FEATURES if c.startswith("cc_")]
    for c in cc_cols:
        m[c] = (rng.random(n) < 0.05).astype("int8")
    m["cc_chest_pain"] = chest
    med_cols = [c for c in EMR_FEATURES if c.startswith("med_")]
    no_med = rng.random(n) < 0.185
    for c in med_cols + ["n_meds"]:
        v = (rng.random(n) < 0.2).astype(float) if c != "n_meds" else rng.integers(0, 15, n).astype(float)
        m[c] = np.where(no_med, np.nan, v)
    for c in MM_FEATURES:
        m[c] = rng.normal(100, 20, n)
    m["qrs_duration"] = qrs
    m["p_wave_absent"] = (rng.random(n) < 0.05).astype(float)
    return m[ID_COLS + SPLIT_COLS + LABEL_COLS + ANALYSIS_COLS + EMR_FEATURES + MM_FEATURES]


@pytest.fixture
def repo(tmp_path):
    """Repo tiruan: tabel master sintetis + jalur keluaran di tmp_path."""
    (tmp_path / "data/processed").mkdir(parents=True)
    master = synthetic_master()
    master.to_parquet(tmp_path / "data/processed/master_table_draft.parquet", index=False)
    return tmp_path, master


def run_cli(root: Path, args: list[str]) -> subprocess.CompletedProcess:
    """Jalankan train_tabular.main(args) di proses Python TERPISAH (tanpa torch), dengan jalur ke repo tiruan."""
    code = "\n".join([
        "import sys",
        "from pathlib import Path",
        "from src.model import train_tabular as t",
        f"r = Path({str(root)!r})",
        "t.ROOT = r",
        "t.MASTER_V1 = r / 'data/processed/master_table_v1.parquet'",
        "t.MASTER_DRAFT = r / 'data/processed/master_table_draft.parquet'",
        "t.SIGNAL_QC = r / 'data/interim/signal_qc.csv'",
        "t.OUT_DIR = r / 'data/interim/tabular'",
        "t.PRED_DIR = r / 'predictions'",
        "t.REPORTS = r / 'reports'",
        "t.MAX_TREES = 300",
        "t.EARLY_STOP = 30",
        f"t.main({args!r})",
        "assert 'torch' not in sys.modules, 'torch termuat bersama xgboost'",
    ])
    return subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=600)


def run_ok(root: Path, args: list[str]) -> str:
    r = run_cli(root, args)
    assert r.returncode == 0, f"gagal (kode {r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
    return r.stdout


def run_fails(root: Path, args: list[str], message: str) -> None:
    r = run_cli(root, args)
    assert r.returncode != 0 and message in (r.stdout + r.stderr), f"seharusnya gagal dengan '{message}'"


def test_schema_matches_real_master():
    m = synthetic_master(200)
    assert m.shape[1] == 65
    assert str(m["arrival_transport"].dtype) == "string" and str(m["signal_ok"].dtype) == "Int8"


def test_build_matrix_one_hot_and_no_leak():
    m = synthetic_master(500)
    X = build_matrix(m, MODEL_SPECS["M4"]["features"])
    dummies = [c for c in X.columns if c.startswith("arrival_")]
    assert dummies == ["arrival_AMBULANCE", "arrival_HELICOPTER", "arrival_OTHER", "arrival_UNKNOWN", "arrival_WALK_IN"]
    assert "arrival_transport" not in X.columns and X.shape[1] == 33 - 1 + 5 + 10
    assert (X[dummies].sum(axis=1) <= 1).all()
    assert (X.loc[m["arrival_transport"].isna().to_numpy(), dummies].sum(axis=1) == 0).all()   # kosong -> semua 0
    assert X["heartrate"].isna().any()                     # NaN dibiarkan untuk XGBoost
    assert not set(X.columns) & set(FORBIDDEN_COLUMNS)
    assert set(X.dtypes.astype(str)) == {"float32"}


def test_unknown_arrival_category_raises():
    m = synthetic_master(100)
    m.loc[0, "arrival_transport"] = "TELEPORT"
    with pytest.raises(ValueError, match="tidak dikenal"):
        build_matrix(m, MODEL_SPECS["M2"]["features"])


@pytest.mark.parametrize("bad", ["label_ami", "cc_explicit_mi", "internal_val_0_16", "report_0", "medrecon_available"])
def test_forbidden_feature_rejected(bad):
    m = synthetic_master(100).assign(report_0="x")
    with pytest.raises(ValueError, match="terlarang"):
        build_matrix(m, MODEL_SPECS["M2"]["features"] + [bad])


@pytest.mark.parametrize("model,max_fold", [("M2", 16), ("M3", 16), ("M4", 17)])
def test_split_respects_brief(model, max_fold):
    m = synthetic_master()
    tr, va = split_train_val(m, model)
    col = MODEL_SPECS[model]["val_col"]
    assert tr["fold"].max() <= max_fold and va["fold"].max() <= max_fold
    assert (tr[col] == 0).all() and (va[col] == 1).all()
    assert not set(tr["subject_id"]) & set(va["subject_id"])
    assert len(tr) + len(va) == int((m["fold"] <= max_fold).sum())


def test_sample_params_reproducible_unique():
    a, b = sample_params(20, 42), sample_params(20, 42)
    assert a == b and len({tuple(t.values()) for t in a}) == 20


def test_train_m2_end_to_end_and_fold19_locked(repo):
    root, master = repo
    run_ok(root, ["--model", "M2", "--n-trials", "2"])
    preds = sorted(p.name for p in (root / "predictions").glob("*.parquet"))
    assert preds == ["M2_fold17.parquet", "M2_fold18.parquet"]                  # fold 19 TIDAK ditulis
    for f in (17, 18):
        d = validate_predictions(pd.read_parquet(root / f"predictions/M2_fold{f}.parquet"), expected_fold=f)
        assert len(d) == int((master.fold == f).sum())
    meta = json.loads((root / "data/interim/tabular/M2_meta.json").read_text())
    assert meta["data"]["train_folds"] == list(range(17)) and meta["n_features_matrix"] == 37
    report = (root / "reports/tabular_M2.md").read_text()
    assert "ed_stay_id" not in report and "Random search" in report
    run_fails(root, ["--model", "M2", "--predict-only", "--folds", "19"], "dikunci")
    run_ok(root, ["--model", "M2", "--predict-only", "--folds", "19", "--include-fold-19"])
    assert (root / "predictions/M2_fold19.parquet").exists()
    # predict-only menghasilkan prediksi yang SAMA dengan saat training
    before = pd.read_parquet(root / "predictions/M2_fold18.parquet")
    run_ok(root, ["--model", "M2", "--predict-only", "--folds", "18"])
    after = pd.read_parquet(root / "predictions/M2_fold18.parquet")
    np.testing.assert_allclose(before.y_prob, after.y_prob, rtol=0, atol=1e-7)


def test_model_learns_planted_signal(tmp_path):
    """Data sintetis BESAR (40.000 kunjungan) agar AUROC stabil lintas platform/CPU.

    Catatan: versi v2.1 memeriksa AUROC di fold 18 tabel kecil (hanya 6 positif) sehingga rapuh
    (0,643 di Linux vs 0,575 di Mac). Di sini fold 17+18 berisi sekitar 90-136 positif; AUROC
    terukur 0,70-0,77 pada 4 seed, jadi batas 0,6 punya jarak aman.
    """
    (tmp_path / "data/processed").mkdir(parents=True)
    synthetic_master(n=40000, seed=0).to_parquet(tmp_path / "data/processed/master_table_draft.parquet", index=False)
    run_ok(tmp_path, ["--model", "M2", "--n-trials", "1"])
    both = pd.concat([pd.read_parquet(tmp_path / f"predictions/M2_fold{f}.parquet") for f in (17, 18)])
    assert both["y_true"].sum() >= 50
    assert roc_auc(both["y_true"], both["y_prob"]) > 0.6


def test_m4_never_predicts_its_training_fold(repo):
    root, _ = repo
    run_ok(root, ["--model", "M4", "--n-trials", "1"])
    assert sorted(p.name for p in (root / "predictions").glob("*.parquet")) == ["M4_fold18.parquet"]
    run_fails(root, ["--model", "M4", "--predict-only", "--folds", "17"], "tidak boleh")


def test_signal_qc_filter_applied_to_draft(repo):
    root, master = repo
    (root / "data/interim").mkdir(parents=True)
    qc = pd.DataFrame({"ed_stay_id": master.ed_stay_id, "signal_ok": 1})
    qc.loc[qc.index[:50], "signal_ok"] = 0
    qc.to_csv(root / "data/interim/signal_qc.csv", index=False)
    run_ok(root, ["--model", "M3", "--n-trials", "1"])
    kept = set(qc.loc[qc.signal_ok == 1, "ed_stay_id"])
    d17 = pd.read_parquet(root / "predictions/M3_fold17.parquet")
    assert set(d17.ed_stay_id) <= kept


def test_reproducible_training(repo):
    root, _ = repo
    run_ok(root, ["--model", "M3", "--n-trials", "2", "--n-jobs", "2"])
    a = pd.read_parquet(root / "predictions/M3_fold18.parquet")
    run_ok(root, ["--model", "M3", "--n-trials", "2", "--n-jobs", "2"])
    b = pd.read_parquet(root / "predictions/M3_fold18.parquet")
    np.testing.assert_allclose(a.y_prob, b.y_prob, rtol=0, atol=1e-7)
