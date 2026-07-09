"""Training-loop smoke test: gradients propagate through a multi-layer sparse
network and an optimizer reduces the loss, with no NaN/Inf along the way.

The golden suite verifies a single backward pass per case; it never runs an
optimizer step, so nothing there proves the network is actually *trainable*
(stable gradients across steps, loss that goes down). This fills that gap on a
small UNet exercising subm/regular/inverse convs, BatchNorm, ReLU, JoinTable and
the 1x1 head — at batch size 4. Implementation-agnostic.
"""

import torch
import torch.nn.functional as F

from helpers import DEVICE, build_unet3d

BATCH = 4
SPATIAL = [32, 32, 32]
IN_CH = 6
OUT_CH = 8


def _make_input(impl, seed):
    g = torch.Generator().manual_seed(seed)
    blocks = []
    for b in range(BATCH):
        c = torch.unique(
            torch.stack([torch.randint(0, s, (1200,), generator=g)
                         for s in SPATIAL], 1), dim=0)
        bcol = torch.full((c.shape[0], 1), b, dtype=torch.int32)
        blocks.append(torch.cat([bcol, c.int()], 1))
    indices = torch.cat(blocks).to(DEVICE)
    feats = torch.randn(indices.shape[0], IN_CH, generator=g)
    return indices, feats


def _coord_keyed_target(out_indices, channels):
    """A fixed regression target as a function of the output coordinate.

    Keyed on the coordinate (not on row order), so the objective is well-defined
    regardless of the implementation's output row order.
    """
    lin = torch.zeros(out_indices.shape[0], dtype=torch.long, device=out_indices.device)
    for i, s in enumerate(SPATIAL):
        lin = lin * int(s) + out_indices[:, i + 1].long()
    lin = lin.to(torch.float32)
    ch = torch.arange(channels, device=out_indices.device, dtype=torch.float32)
    return torch.sin(lin[:, None] * 0.011 + ch[None, :] * 0.9) \
        + 0.4 * torch.cos(lin[:, None] * 0.0005 - ch[None, :])


def test_training_reduces_loss(impl):
    torch.manual_seed(0)
    net = build_unet3d(impl, in_channels=IN_CH, base=16).to(DEVICE).train()
    indices, feats = _make_input(impl, seed=777)

    # ---- input gradient genuinely flows back to the features ----
    f_leaf = feats.clone().to(DEVICE).requires_grad_(True)
    x0 = impl.pytorch.SparseConvTensor(f_leaf, indices, SPATIAL, BATCH)
    out0 = net(x0)
    out0.features.square().mean().backward()
    assert f_leaf.grad is not None and torch.isfinite(f_leaf.grad).all()
    assert f_leaf.grad.abs().sum().item() > 0, "no gradient reached the input features"

    # ---- training loop ----
    feats_dev = feats.to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=5e-3)
    before = {nm: p.detach().clone() for nm, p in net.named_parameters()}

    losses = []
    for step in range(80):
        opt.zero_grad(set_to_none=True)
        x = impl.pytorch.SparseConvTensor(feats_dev, indices, SPATIAL, BATCH)
        out = net(x)
        target = _coord_keyed_target(out.indices, OUT_CH)
        loss = F.mse_loss(out.features, target)
        loss.backward()
        for nm, p in net.named_parameters():
            assert p.grad is not None, f"step {step}: {nm} has no grad"
            assert torch.isfinite(p.grad).all(), f"step {step}: {nm} grad is non-finite"
        opt.step()
        assert torch.isfinite(loss).item(), f"step {step}: loss is non-finite"
        losses.append(loss.item())

    # loss must drop substantially (Adam easily overfits this fixed objective)
    assert losses[-1] < 0.6 * losses[0], (
        f"loss did not converge: start {losses[0]:.4f} end {losses[-1]:.4f} "
        f"(min {min(losses):.4f})")

    # parameters must have actually moved
    moved = sum(1 for nm, p in net.named_parameters()
                if not torch.equal(p.detach(), before[nm]))
    total = sum(1 for _ in net.parameters())
    assert moved == total, f"only {moved}/{total} params changed after training"
