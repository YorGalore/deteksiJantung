import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import wfdb

from src.constants import FOLD_COL
from src.dataset import extract_zip_records as ex
from src.dataset.ecg_download import _path, is_valid_record
from src.dataset.run_signal_qc_pilot import PREFIX

NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _make(tmp, name, seed):
    sig = np.random.default_rng(seed).normal(0, 0.3, (5000, 12)).astype("float32")
    wfdb.wrsamp(name, fs=500, units=["mV"] * 12, sig_name=NAMES, p_signal=sig, fmt=["16"] * 12, write_dir=str(tmp))
    return tmp / name


def _fname(i):
    return f"{PREFIX}files/p1000/p{i}/s{i}/rec{i}"


@pytest.fixture
def world(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    zp = tmp_path / "m.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as z:
        z.writestr(PREFIX + "LICENSE.txt", "x")
        for i in (1, 2, 3, 4):              # 5 tidak ada di ZIP
            b = _make(src, f"rec{i}", i)
            z.write(_path(b, ".hea"), _fname(i) + ".hea")
            if i == 4:                       # .dat terpotong tetapi CRC ZIP valid
                z.writestr(_fname(i) + ".dat", _path(b, ".dat").read_bytes()[:1000])
            else:
                z.write(_path(b, ".dat"), _fname(i) + ".dat")
    targets = pd.DataFrame({"ed_stay_id": [1, 2, 3, 4, 5, 6], "file_name": [_fname(i) for i in range(1, 7)],
                            FOLD_COL: [0, 1, 2, 3, 4, 19]})
    return zp, targets, tmp_path / "cache"


def _st(log):
    return dict(zip(log["ed_stay_id"], log["status"]))


def test_statuses_and_files(world):
    zp, targets, cache = world
    t = targets[targets[FOLD_COL] != 19]
    st = _st(ex.extract_all(zp, t, cache, quiet=True))
    assert st == {1: "extracted", 2: "extracted", 3: "extracted", 4: "size_mismatch", 5: "missing_in_zip"}
    base = cache / "files/p1000/p1/s1/rec1"
    assert is_valid_record(base)
    assert not list(cache.rglob("*.part"))
    assert not _path(cache / "files/p1000/p4/s4/rec4", ".dat").exists()      # rekaman rusak tidak ditinggalkan


def test_rerun_skips_valid_and_does_not_rewrite(world):
    zp, targets, cache = world
    t = targets[targets[FOLD_COL] != 19]
    ex.extract_all(zp, t, cache, quiet=True)
    f = _path(cache / "files/p1000/p1/s1/rec1", ".dat")
    mtime = f.stat().st_mtime_ns
    st = _st(ex.extract_all(zp, t, cache, quiet=True))
    assert st[1] == st[2] == st[3] == "already_valid"
    assert f.stat().st_mtime_ns == mtime


def test_verify_existing_detects_difference_without_overwriting(world):
    zp, targets, cache = world
    t = targets[targets[FOLD_COL].isin([0, 1])]
    ex.extract_all(zp, t, cache, quiet=True)
    f = _path(cache / "files/p1000/p2/s2/rec2", ".dat")
    data = bytearray(f.read_bytes())
    data[100] ^= 0xFF
    f.write_bytes(bytes(data))
    st = _st(ex.extract_all(zp, t, cache, verify_existing=True, quiet=True))
    assert st == {1: "verify_identical", 2: "verify_different"}
    assert f.read_bytes() == bytes(data)


def test_crc_corruption_detected(world):
    zp, targets, cache = world
    with zipfile.ZipFile(zp) as z:
        i = z.getinfo(_fname(1) + ".dat")
        pos = i.header_offset + 30 + len(i.filename.encode()) + len(i.extra) + 50
    raw = bytearray(zp.read_bytes())
    raw[pos] ^= 0xFF
    zp.write_bytes(bytes(raw))
    st = _st(ex.extract_all(zp, targets[targets[FOLD_COL] == 0], cache, quiet=True))
    assert st == {1: "crc_fail"}
    assert not _path(cache / "files/p1000/p1/s1/rec1", ".dat").exists()


def test_fold_19_excluded_unless_flag(tmp_path):
    m = pd.DataFrame({"ed_stay_id": [1, 2], "file_name": ["a", "b"], FOLD_COL: [18, 19]})
    p = tmp_path / "m.parquet"
    m.to_parquet(p)
    assert list(ex.load_targets(p, False)["ed_stay_id"]) == [1]
    assert list(ex.load_targets(p, True)["ed_stay_id"]) == [1, 2]


def test_summary_is_aggregate_only(world):
    zp, targets, cache = world
    log = ex.extract_all(zp, targets[targets[FOLD_COL] != 19], cache, quiet=True)
    out = ex.summarize(log)
    assert "missing_in_zip" in out and "size_mismatch" in out
    assert "rec1" not in out and "p1000" not in out