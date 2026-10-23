import numpy as np
import pytest
import wfdb

from src.dataset.ecg_loader import read_ecg_record
from src.model.ecg_preprocessing import (
    LEAD_ORDER, LeadStandardizer, interpolate_nans, prepare_ecg, reorder_leads,
)
from src.model.ecg_quality import check_ecg_quality_brief


def _signal(seed=0, n=5000):
    return np.random.default_rng(seed).normal(0, 0.2, size=(n, 12))


def test_reorder_leads_uses_header_names():
    names = ["V6", "V5", "V4", "V3", "V2", "V1", "aVF", "aVL", "aVR", "III", "II", "I"]
    sig = np.tile(np.arange(12), (10, 1)).astype(float)       # kolom j bernilai j
    out = reorder_leads(sig, names)
    assert out[0].tolist() == [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]   # I=kolom 11, ...
    with pytest.raises(ValueError):
        reorder_leads(sig, names[:-1] + ["V7"])


def test_interpolate_nans_interior_edges_and_all_nan():
    x = np.ones((10, 3))
    x[:, 0] = np.arange(10, dtype=float)
    x[3:6, 0] = np.nan                       # interior -> linear
    x[:2, 1] = np.nan                        # awal -> 0
    x[-2:, 1] = np.nan                       # akhir -> 0
    x[:, 2] = np.nan                         # semua NaN -> 0
    out = interpolate_nans(x)
    assert out[:, 0].tolist() == pytest.approx(list(range(10)))
    assert out[:2, 1].tolist() == [0, 0] and out[-2:, 1].tolist() == [0, 0] and out[5, 1] == 1
    assert (out[:, 2] == 0).all()
    assert np.isfinite(out).all()


def test_prepare_ecg_shape_clip_and_no_highpass():
    sig = _signal() + 1.0                    # offset DC 1 mV harus bertahan (tanpa bandpass)
    sig[100, 3] = 50.0                       # lonjakan harus di-clip
    out = prepare_ecg(sig, 500, LEAD_ORDER)
    assert out.shape == (12, 1000) and out.dtype == np.float32
    assert np.abs(out).max() <= 3.0
    assert np.median(out[:, 100:900]) == pytest.approx(1.0, abs=0.1)


def test_prepare_ecg_handles_nan_burst_and_short_signal():
    sig = _signal()
    sig[1000:1500, 2] = np.nan
    assert np.isfinite(prepare_ecg(sig, 500, LEAD_ORDER)).all()
    with pytest.raises(ValueError):
        prepare_ecg(_signal(n=2000), 500, LEAD_ORDER)


def test_standardizer_fit_transform_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    data = rng.normal(2.0, 0.5, size=(20, 12, 1000))
    st = LeadStandardizer.fit(iter(data))
    z = st.transform(data)
    assert z.mean() == pytest.approx(0.0, abs=1e-3) and z.std() == pytest.approx(1.0, abs=1e-2)
    assert st.transform(data[0]).shape == (12, 1000)
    p = tmp_path / "stats.json"
    st.save(p)
    st2 = LeadStandardizer.load(p)
    assert np.allclose(st.mean, st2.mean) and np.allclose(st.std, st2.std)


def test_quality_brief_rules():
    ok = check_ecg_quality_brief(_signal())
    assert ok["signal_ok"] == 1 and ok["qc_reason"] == "passed_brief_checks" and ok["amplitude_flag"] == 0

    s = _signal(); s[:500, 0] = np.nan                       # 10% NaN -> lolos
    assert check_ecg_quality_brief(s)["signal_ok"] == 1
    s = _signal(); s[:1500, 0] = np.nan                      # 30% NaN -> gagal
    assert check_ecg_quality_brief(s)["qc_reason"] == "lead_nan_over_20pct"

    s = _signal(); s[:, :2] = 0.0                            # 2 lead datar -> lolos
    assert check_ecg_quality_brief(s)["signal_ok"] == 1
    s = _signal(); s[:, :3] = 0.0                            # 3 lead datar -> gagal
    assert check_ecg_quality_brief(s)["qc_reason"] == "three_or_more_flat_leads"

    s = _signal(); s[10, 4] = 25.0                           # amplitudo: hanya penanda
    r = check_ecg_quality_brief(s)
    assert r["signal_ok"] == 1 and r["amplitude_flag"] == 1


def test_read_ecg_record_keeps_names_and_nan(tmp_path):
    sig = _signal()
    sig[200:300, 1] = np.nan
    names = ["V6", "V5", "V4", "V3", "V2", "V1", "aVF", "aVL", "aVR", "III", "II", "I"]
    wfdb.wrsamp("rec", fs=500, units=["mV"] * 12, sig_name=names, p_signal=sig,
                fmt=["16"] * 12, write_dir=str(tmp_path))
    loaded, fs, got = read_ecg_record(tmp_path / "rec")
    assert fs == 500 and got == names and loaded.shape == (5000, 12)
    assert np.isnan(loaded[250, 1])
    assert np.isfinite(prepare_ecg(loaded, fs, got)).all()
