import numpy as np
import pandas as pd
import pytest
import wfdb

from src.model import predict_m1, train_m1
from src.model.train_m1 import EarlyStopper

NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def test_early_stopper():
    s = EarlyStopper(patience=2)
    assert s.update(0.5) and not s.should_stop
    assert not s.update(0.5) and not s.should_stop          # tidak membaik > min_delta
    assert not s.update(0.4) and s.should_stop
    assert s.best == 0.5
    s2 = EarlyStopper(patience=3)
    assert not s2.update(float("nan")) and s2.best == -np.inf


@pytest.fixture()
def world(tmp_path, monkeypatch):
    cache = tmp_path / "cache"; cache.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    folds = list(range(0, 17)) * 3 + [17] * 6 + [18] * 6 + [19] * 6          # 63 rekaman
    for i, fold in enumerate(folds):
        y = int(i % 3 == 0)
        wfdb.wrsamp(f"f{i}", fs=500, units=["mV"] * 12, sig_name=NAMES, fmt=["16"] * 12, write_dir=str(cache),
                    p_signal=(rng.normal(0, .3, (5000, 12)) + 0.1 * y).astype("float32"))
        rows.append(dict(ed_stay_id=i, subject_id=1000 + i, study_id=5000 + i, file_name=f"p/f{i}", fold=fold,
                         label_ami=y, internal_val_0_16=int(fold < 17 and i % 5 == 0)))
    master = pd.DataFrame(rows)
    master.to_parquet(tmp_path / "m.parquet")
    pd.DataFrame({"ed_stay_id": master.ed_stay_id, "signal_ok": 1}).to_csv(tmp_path / "qc.csv", index=False)
    for mod in (train_m1, predict_m1):
        monkeypatch.setattr(mod, "CACHE", cache)
        monkeypatch.setattr(mod, "MASTER", tmp_path / "m.parquet")
        monkeypatch.setattr(mod, "PREFIX", "p/")
        monkeypatch.setattr(mod, "OUT_DIR", tmp_path / "m1")
    monkeypatch.setattr(predict_m1, "PRED_DIR", tmp_path / "pred")
    return tmp_path


def test_train_then_predict_contract(world):
    train_m1.main(["--qc-csv", str(world / "qc.csv"), "--epochs", "3", "--patience", "2", "--batch-size", "16",
                   "--device", "cpu"])
    for name in ("m1_best.pt", "lead_stats.json", "m1_history.json"):
        assert (world / "m1" / name).exists()

    predict_m1.main(["--qc-csv", str(world / "qc.csv"), "--folds", "17", "18", "--device", "cpu"])
    for fold in (17, 18):
        p = pd.read_parquet(world / "pred" / f"M1_fold{fold}.parquet")
        assert list(p.columns) == predict_m1.COLUMNS and len(p) == 6 and (p["fold"] == fold).all()
        assert p["y_prob"].between(0, 1).all() and set(p["y_true"]) <= {0, 1}
        assert p["ed_stay_id"].is_unique and p["subject_id"].notna().all() and p["study_id"].notna().all()
    assert not (world / "pred" / "M1_fold19.parquet").exists()


def test_fold_19_is_locked_unless_explicit(world):
    train_m1.main(["--qc-csv", str(world / "qc.csv"), "--epochs", "1", "--device", "cpu"])
    with pytest.raises(SystemExit):
        predict_m1.main(["--qc-csv", str(world / "qc.csv"), "--folds", "19", "--device", "cpu"])
    assert not (world / "pred" / "M1_fold19.parquet").exists()
    predict_m1.main(["--qc-csv", str(world / "qc.csv"), "--folds", "19", "--include-fold-19", "--device", "cpu"])
    assert (world / "pred" / "M1_fold19.parquet").exists()
    with pytest.raises(SystemExit):
        predict_m1.main(["--qc-csv", str(world / "qc.csv"), "--folds", "3", "--device", "cpu"])   # bukan fold evaluasi


def test_training_never_reads_eval_folds(world, monkeypatch):
    seen = []
    orig = train_m1.ECGDataset

    class Spy(orig):
        def __init__(self, table, *a, **k):
            seen.extend(table["fold"].tolist())
            super().__init__(table, *a, **k)

    monkeypatch.setattr(train_m1, "ECGDataset", Spy)
    train_m1.main(["--qc-csv", str(world / "qc.csv"), "--epochs", "1", "--device", "cpu"])
    assert seen and max(seen) <= 16
