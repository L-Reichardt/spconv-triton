"""Large-batch validation for fp16 GEMM accumulation (the bounded dWeight path).

The weight-gradient is the only GEMM whose reduction length scales with batch x
points (``seglen ~ n_out``). spconv_triton accumulates it with BOUNDED fp16
(fp16 within each BLOCK_P tile, promoted to fp32 across tiles), so its error must
NOT grow as n_out grows -- unlike a naive fp16-across-tiles accumulator, which
swamps / overflows. This script proves that on the RTX 3060 (12 GB), staying
within the memory budget.

For a SubM 3x3x3 conv (n_out == n_in, so the dWeight reduction == total points)
it ramps the point count and, at each size, compares three runs on the SAME
fp16-exact data (``x.half().float()`` round-trip, isolating the accumulator):

  ref   : fp32 inputs, fp32 accumulate           (ground truth)
  acc16 : fp16 inputs, fp16 policy (flag/None)    (the bounded kernel under test)
  acc32 : fp16 inputs, fp32 accumulate (flag off) (the conservative fp16 path)

Assertions at every size:
  1. no NaN/Inf in any output or gradient (overflow guard);
  2. bounded-fp16 dWeight rel-error vs ref stays FLAT (<= BOUND) as n grows;
  3. forward / input-grad fp16 rel-error stays flat (they do not scale with n).

Then, decoupled from the kernel, a synthetic counterfactual on a reduction of the
SAME length n_out shows that an fp16-ACROSS-tiles accumulator DIVERGES with n
while fp32-across (what the bounded kernel does between tiles) stays flat --
i.e. why the bounded design is necessary. Plus a per-layer ``fp32_accum``
override dispatch check.

Run:
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python scripts/verify_fp16_accum_large_batch.py
"""

from __future__ import annotations

import sys

import torch

import spconv_triton.pytorch as spconv
from spconv_triton import constants

DEV = "cuda"
SEED = 1234
CIN = COUT = 64
KV_KERNEL = 3  # 3x3x3 -> kv = 27, reduction kv*C = 27*64 = 1728 <= 3456 (fp16)
SIZES = [30_000, 100_000, 300_000, 500_000]  # total points (batch x per-batch)
BATCH = 4

# Bounds: bounded-fp16 should stay a tight, n-independent fraction of signal.
DW_BOUND = 3.0e-2  # 3% of |dW|max -- well above fp16 input-rounding, below naive
FWD_BOUND = 2.0e-2
DIN_BOUND = 2.0e-2


def make_pointcloud(total: int, batch: int, seed: int):
    """``batch`` clouds of unique voxel coords + Gaussian features (CPU gen).

    Spatial grid is sized so ``torch.unique`` drops few points; the actual count
    may be slightly below ``total``."""
    per = max(1, total // batch)
    side = max(16, int((per * 4) ** (1 / 3)) + 1)  # ~25% fill -> few collisions
    ss = [side, side, side]
    g = torch.Generator().manual_seed(seed)
    rows = []
    for b in range(batch):
        c = torch.stack([torch.randint(0, s, (per,), generator=g) for s in ss], 1)
        c = torch.unique(c, dim=0)
        rows.append(torch.cat([torch.full((c.shape[0], 1), b), c], 1))
    idx = torch.cat(rows).int()
    feats = torch.randn(idx.shape[0], CIN, generator=g)
    return feats, idx, ss, batch


def build_subm(weight_src: torch.Tensor | None, fp32_accum, half: bool = False):
    """SubMConv3d with a fixed (optionally injected) KRSC weight, no bias.

    ``half=True`` casts the layer to fp16 so it matches fp16 inputs (the layer
    weight dtype must equal the feature dtype -- the GEMM requires both operands
    to share a dtype)."""
    layer = spconv.SubMConv3d(
        CIN,
        COUT,
        KV_KERNEL,
        padding=1,
        indice_key="bk",
        bias=False,
        fp32_accum=fp32_accum,
    ).to(DEV)
    if half:
        layer = layer.half()
    if weight_src is not None:
        with torch.no_grad():
            layer.weight.copy_(weight_src.to(layer.weight.dtype))
    return layer


def run_once(layer, feats, idx, ss, bs, upstream):
    """One fwd+bwd; returns (out_features, weight_grad, input_grad), grads cleared."""
    if layer.weight.grad is not None:
        layer.weight.grad = None
    x_feats = feats.to(DEV).clone().requires_grad_(True)
    x = spconv.SparseConvTensor(x_feats, idx.to(DEV), ss, bs)
    out = layer(x)
    (out.features * upstream).sum().backward()
    return (
        out.features.detach().float(),
        layer.weight.grad.detach().float(),
        x_feats.grad.detach().float(),
    )


def synthetic_accumulator_divergence(n: int, seed: int):
    """Counterfactual on a reduction of length ``n`` (the dWeight regime).

    Sum ``n`` fp16 scalar products into (a) an fp16 running accumulator
    ('across-tiles fp16', the danger) and (b) an fp32 running accumulator (what
    the bounded kernel does between tiles). Returns their rel-errors vs an fp64
    reference. Pure numerics -- no kernel dependency -- isolating the accumulator
    dtype as the variable."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    terms16 = torch.randn(n, generator=g, device=DEV, dtype=torch.float32).half()
    ref = terms16.double().sum().item()
    across_fp16 = torch.zeros((), dtype=torch.float16, device=DEV)
    across_fp32 = torch.zeros((), dtype=torch.float32, device=DEV)
    CHUNK = 2048  # tile granularity; each tile reduced exactly, folded across
    for o0 in range(0, n, CHUNK):
        tile = terms16[o0 : o0 + CHUNK]
        across_fp16 += tile.float().sum().half()  # fold tile sum in fp16 (danger)
        across_fp32 += tile.float().sum()  # fold tile sum in fp32 (bounded)
    denom = max(abs(ref), 1e-12)
    return (
        abs(across_fp16.double().item() - ref) / denom,
        abs(across_fp32.double().item() - ref) / denom,
    )


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """max|a-b| / max(|b|max, eps)."""
    denom = max(b.abs().max().item(), 1e-12)
    return (a - b).abs().max().item() / denom


def has_bad(*ts) -> bool:
    return any((not torch.isfinite(t).all().item()) for t in ts)


def main() -> int:
    if not torch.cuda.is_available():
        raise AssertionError("GPU required")
    torch.manual_seed(SEED)

    # Shared fp32 weight (KRSC) reused across all sizes for comparable grads.
    weight0 = torch.randn(COUT, KV_KERNEL, KV_KERNEL, KV_KERNEL, CIN)

    print(
        f"{'N':>8} {'peakMB':>8} {'dW16':>9} {'fwd16':>9} {'din16':>9} "
        f"{'naiveΣ':>9} {'boundΣ':>9} {'verdict':>8}"
    )
    failures: list[str] = []
    dw_rels: list[float] = []
    naive_rels: list[float] = []
    bound_rels: list[float] = []

    for total in SIZES:
        feats, idx, ss, bs = make_pointcloud(total, BATCH, SEED + total)
        n = idx.shape[0]
        # fp16-exact data: round once, share the fp16 and its fp32 upcast.
        f16 = feats.half()
        f_ref = f16.float()
        up = torch.randn(n, COUT, generator=torch.Generator().manual_seed(SEED + 1))
        up16 = up.half().to(DEV)
        up_ref = up16.float()

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # ref: fp32 inputs, fp32 accumulate (ground truth on fp16-exact values).
        ref_layer = build_subm(weight0, fp32_accum=None)
        out_ref, dw_ref, din_ref = run_once(ref_layer, f_ref, idx, ss, bs, up_ref)

        # acc16: fp16 inputs, fp16 policy ON (default flag).
        constants.SPCONV_ALLOW_FP16_ACCUM = True
        l16 = build_subm(weight0, fp32_accum=None, half=True)
        out16, dw16, din16 = run_once(l16, f16, idx, ss, bs, up16)

        # acc32: fp16 inputs, fp32 accumulate (flag OFF).
        constants.SPCONV_ALLOW_FP16_ACCUM = False
        l32 = build_subm(weight0, fp32_accum=None, half=True)
        out32, dw32, din32 = run_once(l32, f16, idx, ss, bs, up16)
        constants.SPCONV_ALLOW_FP16_ACCUM = True

        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / 2**20

        # Reference the fp16 grads against the fp32-of-fp16-exact ground truth.
        e_dw16 = rel_err(dw16, dw_ref)
        e_fwd16 = rel_err(out16, out_ref)
        e_din16 = rel_err(din16, din_ref)
        dw_rels.append(e_dw16)

        # Synthetic accumulator counterfactual at this reduction length.
        e_naive, e_bound = synthetic_accumulator_divergence(n, SEED + total)
        naive_rels.append(e_naive)
        bound_rels.append(e_bound)

        ok = True
        if has_bad(out16, dw16, din16, out32, dw32, din32):
            failures.append(f"N={n}: NaN/Inf in fp16 outputs/grads")
            ok = False
        if e_dw16 > DW_BOUND:
            failures.append(f"N={n}: bounded dW rel {e_dw16:.2e} > {DW_BOUND:.0e}")
            ok = False
        if e_fwd16 > FWD_BOUND:
            failures.append(f"N={n}: fwd rel {e_fwd16:.2e} > {FWD_BOUND:.0e}")
            ok = False
        if e_din16 > DIN_BOUND:
            failures.append(f"N={n}: din rel {e_din16:.2e} > {DIN_BOUND:.0e}")
            ok = False

        print(
            f"{n:>8} {peak_mb:>8.0f} {e_dw16:>9.2e} {e_fwd16:>9.2e} "
            f"{e_din16:>9.2e} {e_naive:>9.2e} {e_bound:>9.2e} "
            f"{'ok' if ok else 'FAIL':>8}"
        )

        del out_ref, dw_ref, din_ref, out16, dw16, din16, out32, dw32, din32
        del ref_layer, l16, l32
        torch.cuda.empty_cache()

    # 2. bounded dW must be FLAT: largest size not materially worse than smallest.
    if dw_rels[-1] > 1.5 * dw_rels[0] + 5e-3:
        failures.append(
            f"bounded dW rel grew with n: {dw_rels[0]:.2e} -> {dw_rels[-1]:.2e}"
        )
    # Counterfactual: fp16-across diverges with n; fp32-across (bounded) stays flat.
    if not (naive_rels[-1] > naive_rels[0] and naive_rels[-1] > bound_rels[-1] * 5):
        failures.append(
            f"synthetic control unconvincing: naive {naive_rels[0]:.2e}->"
            f"{naive_rels[-1]:.2e} vs bounded {bound_rels[-1]:.2e}"
        )
    if bound_rels[-1] > 1.5 * bound_rels[0] + 1e-6:
        failures.append(
            f"fp32-across drifted with n: {bound_rels[0]:.2e} -> {bound_rels[-1]:.2e}"
        )

    # Per-layer override dispatch: fp32_accum=True forces fp32 even with flag ON;
    # fp32_accum=False forces fp16 even with flag OFF.
    print("\n-- per-layer fp32_accum override dispatch --")
    feats, idx, ss, bs = make_pointcloud(60_000, BATCH, SEED)
    n = idx.shape[0]
    f16 = feats.half()
    f_ref = f16.float()
    up16 = (
        torch.randn(n, COUT, generator=torch.Generator().manual_seed(SEED + 1))
        .half()
        .to(DEV)
    )
    _, dw_truth, _ = run_once(
        build_subm(weight0, None), f_ref, idx, ss, bs, up16.float()
    )

    constants.SPCONV_ALLOW_FP16_ACCUM = False  # global OFF
    _, dw_force16, _ = run_once(
        build_subm(weight0, fp32_accum=False, half=True), f16, idx, ss, bs, up16
    )
    constants.SPCONV_ALLOW_FP16_ACCUM = True  # global ON
    _, dw_force32, _ = run_once(
        build_subm(weight0, fp32_accum=True, half=True), f16, idx, ss, bs, up16
    )

    e_force16 = rel_err(dw_force16, dw_truth)
    e_force32 = rel_err(dw_force32, dw_truth)
    # force32 (fp32 accumulate) should track the fp32 truth more tightly than
    # force16 (fp16 policy), proving the override actually switched paths.
    print(f"  fp32_accum=False (flag off) dW rel = {e_force16:.2e}  (fp16 policy)")
    print(f"  fp32_accum=True  (flag on)  dW rel = {e_force32:.2e}  (fp32 accum)")
    if not (e_force32 < e_force16):
        failures.append(
            f"override dispatch unclear: force32 {e_force32:.2e} !< force16 {e_force16:.2e}"
        )

    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("ALL CHECKS PASSED: bounded fp16 dWeight is NaN-safe and n-independent;")
    print("fp16-across-tiles counterfactual diverges; fp32_accum override dispatches.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
