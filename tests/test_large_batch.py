"""Large-batch (batch size 4, bzyx) correctness via per-batch independence.

A single multi-batch forward/backward must produce, for every batch, exactly
what running that batch alone produces. This exercises the parts of the bzyx
index layout that a batch_size=1 test cannot reach:
  - batch-major coordinate linearization (output coords carry the right batch),
  - per-batch separation during pair generation (no point pairs across batches),
  - the absence of gradient leakage across the batch dimension.

Property-based and implementation-agnostic: the same assertions hold for the
reference and the port (run with ``SPCONV_TEST_IMPL=spconv`` to re-validate).
"""

import pytest
import torch

from helpers import DEVICE

BATCH = 4
SPATIAL = [32, 32, 32]
CIN = 8
COUT = 12
NPTS = 1500


def _spatial_lin(coords, spatial_shape):
    lin = torch.zeros(coords.shape[0], dtype=torch.long)
    for i, s in enumerate(spatial_shape):
        lin = lin * int(s) + coords[:, i].long()
    return lin


def _gen_batches(seed):
    """bs=4 input laid out batch-major (batch b owns a contiguous row slice)."""
    g = torch.Generator().manual_seed(seed)
    blocks, counts = [], []
    for b in range(BATCH):
        c = torch.unique(
            torch.stack([torch.randint(0, s, (NPTS,), generator=g)
                         for s in SPATIAL], 1), dim=0)
        counts.append(c.shape[0])
        bcol = torch.full((c.shape[0], 1), b, dtype=torch.int32)
        blocks.append(torch.cat([bcol, c.int()], 1))
    indices = torch.cat(blocks)
    feats = torch.randn(indices.shape[0], CIN, generator=g)
    return indices, feats, counts


def _coord_keyed_grad(out_indices, channels, spatial_shape):
    """Upstream gradient as a deterministic function of the spatial coordinate.

    Being coordinate-keyed (batch-independent), a given output coordinate
    receives the same upstream gradient in the full run and in the standalone
    per-batch run, so input/weight gradients are directly comparable.
    """
    lin = _spatial_lin(out_indices[:, 1:], spatial_shape).to(torch.float64)
    ch = torch.arange(channels, dtype=torch.float64)
    v = (torch.sin(lin[:, None] * 0.013 + ch[None, :] * 0.7)
         + 0.3 * torch.cos(lin[:, None] * 0.0007 - ch[None, :] * 0.21))
    return v.to(torch.float32)


def _by_coord(out_indices, features, batch, spatial_shape):
    """(coords, features) for `batch`, sorted by spatial coordinate."""
    sel = out_indices[:, 0] == batch
    coords = out_indices[sel][:, 1:]
    order = torch.argsort(_spatial_lin(coords, spatial_shape))
    return coords[order], features[sel][order]


def _close(a, b, msg):
    a = a.to(torch.float64)
    b = b.to(torch.float64)
    assert a.shape == b.shape, f"{msg}: shape {tuple(a.shape)} != {tuple(b.shape)}"
    if a.numel() == 0:
        return
    err = (a - b).abs().max().item()
    ref = max(a.abs().max().item(), 1.0)
    atol = 3e-3 * ref + 1e-5
    assert err <= atol, f"{msg}: max err {err:.3e} > atol {atol:.3e} (ref {ref:.3e})"


def _build(sp, name):
    if name == "subm3d":
        return sp.SubMConv3d(CIN, COUT, 3, padding=1, indice_key="s", bias=True)
    if name == "conv3d_s2":
        return sp.SparseConv3d(CIN, COUT, 3, stride=2, padding=1,
                               indice_key="d", bias=True)
    if name == "maxpool3d":
        return sp.SparseMaxPool3d(2, 2)
    raise ValueError(name)


@pytest.mark.parametrize("name", ["subm3d", "conv3d_s2", "maxpool3d"])
def test_per_batch_independence(impl, name):
    sp = impl.pytorch
    torch.manual_seed(0)
    layer = _build(sp, name).to(DEVICE).train()
    has_w = hasattr(layer, "weight") and layer.weight is not None

    indices, feats, counts = _gen_batches(seed=12345)

    # ---- full bs=4 forward + backward ----
    layer.zero_grad(set_to_none=True)
    f_full = feats.clone().to(DEVICE).requires_grad_(True)
    x_full = sp.SparseConvTensor(f_full, indices.to(DEVICE), SPATIAL, BATCH)
    o_full = layer(x_full)
    out_shape = list(o_full.spatial_shape)
    of_idx = o_full.indices.detach().cpu()
    of_feat = o_full.features.detach().cpu()

    # every output row must carry a batch index in [0, BATCH)
    assert int(of_idx[:, 0].min()) >= 0 and int(of_idx[:, 0].max()) < BATCH
    assert set(of_idx[:, 0].tolist()) == set(range(BATCH)), \
        f"{name}: full output is missing some batches"

    g_full = _coord_keyed_grad(of_idx, of_feat.shape[1], out_shape)
    o_full.features.backward(g_full.to(DEVICE))
    gin_full = f_full.grad.detach().cpu()
    gw_full = layer.weight.grad.detach().cpu().clone() if has_w else None

    # ---- per-batch bs=1 runs ----
    gw_sum = torch.zeros_like(gw_full) if has_w else None
    off = 0
    for b in range(BATCH):
        n = counts[b]
        sl = slice(off, off + n)
        off += n
        sub_idx = indices[sl].clone()
        sub_idx[:, 0] = 0
        layer.zero_grad(set_to_none=True)
        f_b = feats[sl].clone().to(DEVICE).requires_grad_(True)
        x_b = sp.SparseConvTensor(f_b, sub_idx.to(DEVICE), SPATIAL, 1)
        o_b = layer(x_b)
        ob_idx = o_b.indices.detach().cpu()
        g_b = _coord_keyed_grad(ob_idx, o_b.features.shape[1], out_shape)
        o_b.features.backward(g_b.to(DEVICE))

        # forward: batch b of the full run == the standalone run (by coordinate)
        cf, ff = _by_coord(of_idx, of_feat, b, out_shape)
        cb, fb = _by_coord(ob_idx, o_b.features.detach().cpu(), 0, out_shape)
        assert torch.equal(cf, cb), \
            f"{name} b{b}: output coordinate set differs (batch leakage)"
        _close(ff, fb, f"{name} b{b}: forward features")

        # input grad: the batch-b slice of the full run == the standalone run
        _close(gin_full[sl], f_b.grad.detach().cpu(), f"{name} b{b}: grad_input")

        if has_w:
            gw_sum += layer.weight.grad.detach().cpu()

    # weight grad of the full run == sum over the per-batch weight grads
    if has_w:
        _close(gw_full, gw_sum, f"{name}: grad_weight additive over batches")
