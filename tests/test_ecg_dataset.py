import numpy as np
import pandas as pd
import pytest
import torch
import wfdb

from src.model.ecg_dataset import ECGDataset, load_record_table, split_train_val
from src.model.ecg_preprocessing import LeadStandardizer

NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _write_record(directory, name, seed, names=NAMES):
    rng = np.random.default_rng(seed)
    t = np.arange(5000) / 500
    sig = np.stack([np.sin(2 * np.pi * (1 + 0.1 * i) * t) + 0.05 * rng.normal(size=5000) for i in range(12)], axis=1)
    directory.mkdir(parents=True, exist_ok=True)
    wfdb.wrsamp(name, fs=500, units=["mV"] * 12, sig_name=names, p_signal=sig,
                fmt=["16"] * 12, write_dir=str(directory))


@pytest.fixture
def tables(tmp_path):
    rows = []
    for i, (fold, val) in enumerate([(0, 0), (1, 1), (17, 0), (19, 0)]):
        _write_record(tmp_path / "cache" / f"rec{i}", f"r{i}", i)
        rows.append({"ed_stay_id": i + 1, "file_name": f"/rec{i}/r{i}", "fold": fold,
                     "label_ami": i % 2, "internal_val_0_16": val})
    master = tmp_path / "master.parquet"
    pd.DataFrame(rows).to_parquet(master)
    qc = tmp_path / "qc.csv"
    pd.DataFrame({"ed_stay_id": [1, 2, 3, 4], "signal_ok": [1, 1, 1, 0]}).to_csv(qc, index=False)
    return tmp_path, master, qc


def test_locked_folds_are_refused_by_default(tables):
    _, master, qc = tables
    with pytest.raises(ValueError):
        load_record_table(qc, master, folds=[0, 19])


def test_only_signal_ok_and_requested_folds_are_loaded(tables):
    _, master, qc = tables
    df = load_record_table(qc, master, folds=range(0, 17))
    assert sorted(df["ed_stay_id"]) == [1, 2]            # 3 = fold 17, 4 = signal_ok 0
    df = load_record_table(qc, master, folds=[17], allow_locked=True)
    assert list(df["ed_stay_id"]) == [3]


def test_split_follows_internal_val_column_and_refuses_locked(tables):
    _, master, qc = tables
    train, val = split_train_val(load_record_table(qc, master, folds=range(0, 17)))
    assert list(train["ed_stay_id"]) == [1] and list(val["ed_stay_id"]) == [2]
    locked = load_record_table(qc, master, folds=[17], allow_locked=True)
    with pytest.raises(ValueError):
        split_train_val(locked)


def test_dataset_requires_standardizer_then_returns_shapes_and_labels(tables):
    tmp, master, qc = tables
    df = load_record_table(qc, master, folds=range(0, 17))
    ds = ECGDataset(df, tmp / "cache", prefix="")
    with pytest.raises(RuntimeError):
        ds[0]
    ds.standardizer = LeadStandardizer.fit(ds.raw_records())
    x, y = ds[0]
    assert x.shape == (12, 1000) and x.dtype == torch.float32 and torch.isfinite(x).all()
    assert y.shape == (1,) and y.dtype == torch.float32
    assert [float(ds[i][1]) for i in range(2)] == list(df["label_ami"].astype(float))


def test_lead_names_are_reordered_from_header(tables, tmp_path):
    shuffled = NAMES[::-1]
    _write_record(tmp_path / "cache" / "a", "a", 5, names=NAMES)
    _write_record(tmp_path / "cache" / "b", "b", 5, names=shuffled)   # sinyal sama, nama lead terbalik
    df = pd.DataFrame({"file_name": ["/a/a", "/b/b"], "label_ami": [0, 1]})
    ds = ECGDataset(df, tmp_path / "cache", prefix="")
    raw_a, raw_b = list(ds.raw_records())
    assert not np.allclose(raw_a, raw_b)                  # kolom terurut menurut nama, bukan posisi
    assert np.allclose(raw_a[0], raw_b[-1], atol=1e-4)     # lead I (a) = kolom terakhir di file b
