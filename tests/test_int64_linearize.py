"""int64 linearization fallback (pair generation) correctness.

Pair generation packs each (batch, *coords) index into a single linearized
scalar (batch-major: ``b * prod(grid) + ... + x``). When
``batch_size * prod(grid) >= 2**31`` the packed scalar can exceed INT32_MAX, so
the implementation upgrades that scalar from int32 to int64 to avoid wrap-around
(``spconv_triton/pytorch/_impl/pairs.py``). This path is reachable in mainstream
detection training (e.g. KITTI/Waymo grids ~41x1600x1408 at batch >= 24, or a
single high-resolution scene at batch 1), so it must stay correct.

Strategy: a submanifold convolution is translation-invariant — the same local
voxel pattern yields the same per-row output features whether it sits at the
origin on a small grid (int32 path) or at high coordinates on a huge grid where
the linearized coordinate provably exceeds INT32_MAX (int64 path). If the int64
upgrade were dropped, the high-coordinate placement would wrap int32 and mis-pair
points, breaking the match. Implementation-agnostic.
"""

import torch

from helpers import DEVICE

INT32_MAX = 2**31 - 1

# prod = 2048*2048*512 = 2**31 exactly -> int64 selected even at batch 1.
BIG_GRID = [2048, 2048, 512]
SMALL_GRID = [64, 64, 64]
# High-coordinate offset on the big grid (local box of size <=12 fits: 2036+11<2048,
# 500+11<512). For batches >= 1 the linearized coord exceeds INT32_MAX.
OFFSET = [2036, 2036, 500]
BATCH = 4
CIN = 8
COUT = 16


def _lin(indices, grid):
    lin = indices[:, 0].to(torch.int64)
    for i, s in enumerate(grid):
        lin = lin * int(s) + indices[:, i + 1].to(torch.int64)
    return lin


def _pattern(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.unique(
        torch.stack([torch.randint(0, 12, (90,), generator=g) for _ in range(3)], 1),
        dim=0)  # [m, 3] local coords in a 12^3 box


def _place(local, offset):
    blocks = []
    for b in range(BATCH):
        coords = (local + torch.tensor(offset, dtype=torch.long)).int()
        bcol = torch.full((local.shape[0], 1), b, dtype=torch.int32)
        blocks.append(torch.cat([bcol, coords], 1))
    return torch.cat(blocks)


def _run(impl, indices, feats, grid):
    sp = impl.pytorch
    x = sp.SparseConvTensor(feats.clone().to(DEVICE), indices.to(DEVICE),
                            list(grid), BATCH)
    out = layer_global(x)
    return out.indices.detach().cpu(), out.features.detach().cpu()


layer_global = None  # set inside the test so the same weights drive both runs


def test_int64_linearize_matches_int32(impl):
    global layer_global
    local = _pattern(seed=7)
    big_idx = _place(local, OFFSET)
    small_idx = _place(local, [0, 0, 0])

    # The batch-major layout is identical row-for-row, so shared features are fine.
    g = torch.Generator().manual_seed(11)
    feats = torch.randn(big_idx.shape[0], CIN, generator=g)

    # Sanity: the big-grid placement must genuinely overflow int32 (so only the
    # int64 path can compute it correctly). Selection is on grid volume; the
    # actual coordinates of batches >= 1 exceed INT32_MAX.
    assert big_idx.shape == small_idx.shape
    assert int(_lin(big_idx, BIG_GRID).max()) > INT32_MAX, \
        "test misconfigured: big-grid coords do not exceed INT32_MAX"
    assert int(_lin(small_idx, SMALL_GRID).max()) <= INT32_MAX

    sp = impl.pytorch
    torch.manual_seed(0)
    layer_global = sp.SubMConv3d(CIN, COUT, 3, padding=1, indice_key="k",
                                 bias=True).to(DEVICE).eval()

    big_oidx, big_of = _run(impl, big_idx, feats, BIG_GRID)
    small_oidx, small_of = _run(impl, small_idx, feats, SMALL_GRID)

    # subm: output coords == input coords; aligning big-OFFSET to small must match.
    off = torch.tensor([0] + OFFSET, dtype=big_oidx.dtype)
    assert torch.equal(
        torch.unique(big_oidx - off, dim=0),
        torch.unique(small_oidx, dim=0)), \
        "int64 path produced different (translated) output coordinates"

    # Align both runs by (batch, local coord) and compare features.
    def key(idx, grid):
        loc = idx.clone()
        loc[:, 1:] = loc[:, 1:] - torch.tensor(
            OFFSET if grid is BIG_GRID else [0, 0, 0])
        order = torch.argsort(
            loc[:, 0].to(torch.int64) * 10_000_000
            + loc[:, 1] * 10_000 + loc[:, 2] * 100 + loc[:, 3])
        return order

    bo = key(big_oidx, BIG_GRID)
    so = key(small_oidx, SMALL_GRID)
    err = (big_of[bo].to(torch.float64) - small_of[so].to(torch.float64)).abs().max().item()
    ref = max(small_of.abs().max().item(), 1.0)
    atol = 3e-3 * ref + 1e-5
    assert err <= atol, (
        f"int64 vs int32 feature mismatch: max err {err:.3e} > atol {atol:.3e} "
        f"(ref {ref:.3e}) — int32 wrap-around suspected")
