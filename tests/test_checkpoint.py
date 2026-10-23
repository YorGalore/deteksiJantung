import numpy as np
import pytest
import torch

from src.model.resnet1d import resnet1d
from src.model.train_smoke import load_checkpoint, pick_device, predict, save_checkpoint


def test_checkpoint_roundtrip_gives_identical_predictions(tmp_path):
    torch.manual_seed(0)
    m = resnet1d().eval()
    x = torch.randn(4, 12, 1000)
    loader = [(x, torch.zeros(4, 1))]
    _, before = predict(m, loader)
    path = tmp_path / "c.pt"
    save_checkpoint(path, m, 3, {"lr": 1e-3, "qc_csv": tmp_path / "q.csv"}, {"n_train": 10})
    fresh = resnet1d()
    ckpt = load_checkpoint(path, fresh)          # weights_only=True: metadata harus sederhana
    _, after = predict(fresh, loader)
    assert np.allclose(before, after, atol=1e-6)
    assert ckpt["epoch"] == 3 and ckpt["meta"]["n_train"] == 10 and isinstance(ckpt["meta"]["qc_csv"], str)


def test_pick_device_cpu_and_unavailable_cuda(monkeypatch):
    assert pick_device("cpu").type == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert pick_device("auto").type == "cpu"
    with pytest.raises(SystemExit):
        pick_device("cuda")
