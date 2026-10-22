import torch
import torch.nn as nn

from src.model.resnet1d import resnet1d


def test_resnet1d_training_step():
    torch.manual_seed(42)

    model = resnet1d()
    model.train()

    # Dua sampel sintetis, 12 lead, masing-masing 1000 titik.
    x = torch.randn(2, 12, 1000)
    y = torch.tensor([[1.0], [0.0]])

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)

    assert logits.shape == (2, 1)
    assert torch.isfinite(loss)

    loss.backward()

    gradients = [
        p.grad for p in model.parameters()
        if p.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(g).all() for g in gradients)

    optimizer.step()


if __name__ == "__main__":
    test_resnet1d_training_step()
    print("LULUS: forward, loss, backward, dan optimizer step.")
