import numpy as np
import torch

from src.model.resnet1d import resnet1d
from src.xai.ecg_xai import grad_cam_1d, lead_occlusion


def _model():
    torch.manual_seed(0)
    return resnet1d().eval()


def test_grad_cam_shape_range_and_mode_restored():
    m = _model(); m.train()
    cam = grad_cam_1d(m, torch.randn(3, 12, 1000))
    assert cam.shape == (3, 1000)
    assert cam.min() >= 0 and cam.max() <= 1 + 1e-9
    assert m.training is True


def test_occlusion_shape_and_matches_manual():
    m = _model()
    x = torch.randn(2, 12, 1000)
    occ = lead_occlusion(m, x)
    assert occ.shape == (2, 12)
    xo = x.clone(); xo[:, 4, :] = 0
    with torch.no_grad():
        expected = (m(x)[:, 0] - m(xo)[:, 0]).numpy()
    assert np.allclose(occ[:, 4], expected, atol=1e-6)


def test_zero_lead_has_zero_occlusion():
    m = _model()
    x = torch.randn(1, 12, 1000); x[:, 7, :] = 0
    assert abs(lead_occlusion(m, x)[0, 7]) < 1e-6


def test_cam_localizes_planted_segment():
    # sanity: model dilatih singkat agar sinyal di segmen 400-600 menentukan label
    torch.manual_seed(1)
    m = resnet1d(); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for _ in range(60):
        y = (torch.rand(32) > 0.5).float()
        x = torch.randn(32, 12, 1000) * 0.3
        x[:, :, 400:600] += y[:, None, None] * 2.0
        loss = torch.nn.functional.binary_cross_entropy_with_logits(m(x)[:, 0], y)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    x = torch.randn(8, 12, 1000) * 0.3; x[:, :, 400:600] += 2.0
    cam = grad_cam_1d(m, x).mean(axis=0)
    assert cam[400:600].mean() > cam[:300].mean()
