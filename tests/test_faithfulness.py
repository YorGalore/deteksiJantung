import numpy as np
import torch

from src.xai import faithfulness as fa


class BlockModel(torch.nn.Module):
    """Logit hanya bergantung pada blok waktu ke-4 dan lead 2: jawaban yang benar diketahui."""
    def forward(self, x):
        L = x.shape[-1] // fa.N_BLOCKS
        return (x[:, 2, 4 * L:5 * L].mean(1, keepdim=True) * 10.0)


def _x(B=6):
    torch.manual_seed(0)
    return torch.ones(B, 12, 1000) + 0.01 * torch.randn(B, 12, 1000)


def test_time_faithfulness_detects_true_block():
    x = _x()
    cam = np.zeros((6, 1000))
    cam[:, 200:250] = 1.0                      # peta menyorot blok yang benar
    r = fa.time_faithfulness(BlockModel(), x, cam, k=4, n_random=20, seed=1)
    assert (r["top"] > r["random"]).all() and (r["top"] > r["bottom"] + 1).all()
    wrong = np.zeros((6, 1000))
    wrong[:, 800:850] = 1.0                    # peta menyorot blok yang salah
    rw = fa.time_faithfulness(BlockModel(), x, wrong, k=4, n_random=20, seed=1)
    assert rw["top"].mean() < r["top"].mean()
    assert np.allclose(rw["top"], 0.0, atol=1e-5)


def test_lead_faithfulness_detects_true_lead():
    x = _x()
    occ = np.zeros((6, 12))
    occ[:, 2] = 1.0
    r = fa.lead_faithfulness(BlockModel(), x, occ, k=3, n_random=30, seed=1)
    assert (r["top"] > r["random"]).all() and np.allclose(r["bottom"], 0.0, atol=1e-5)


def test_input_not_modified_and_mean_ci():
    x = _x()
    x0 = x.clone()
    fa.time_faithfulness(BlockModel(), x, np.random.rand(6, 1000))
    assert torch.equal(x, x0)
    m, lo, hi = fa.mean_ci(np.arange(100.0))
    assert lo < m < hi