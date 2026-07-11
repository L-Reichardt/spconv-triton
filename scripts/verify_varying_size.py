"""Regression guard: pair-gen must handle varying point count AND grid size.

A real backbone feeds a different point cloud (different N) and, across levels,
different spatial shapes every forward. This drives the SAME kernel
specialization (same kv, ndim) with a sequence of changing N / grid and checks:
  - results stay byte-identical to the torch reference for every call, and
  - after warmup, no call pays a re-autotune spike (autotune keys on kv/ndim
    only, so changing N / grid must NOT re-tune).
"""

import functools
import statistics

import torch

from scripts.verify_pairgen_kernel import (
    kernel_offsets,
    make_indices,
    ref_loop,
)
from spconv_triton.pytorch._impl import pair_kernels

DEV = "cuda"


def _prod(v):
    return functools.reduce(lambda a, b: a * b, v, 1)


def out_shape_conv(ss, ksize, stride, pad, dil):
    return [
        (ss[i] + 2 * pad[i] - dil[i] * (ksize[i] - 1) - 1) // stride[i] + 1
        for i in range(len(ss))
    ]


def one_call(N, ss, ksize, stride, pad, dil, subm):
    ndim = len(ss)
    idx = make_indices(N, ss, 2, seed=N % 9973 + 1)
    lin_dtype = torch.int32 if 2 * _prod(ss) < 2**31 else torch.int64
    if subm:
        eff_pad = [(ksize[i] // 2) * dil[i] for i in range(ndim)]
        eff_st = [1] * ndim
        out_shape = list(ss)
    else:
        eff_pad, eff_st = list(pad), list(stride)
        out_shape = out_shape_conv(ss, ksize, stride, pad, dil)
    kv = _prod(ksize)
    idx_lin = idx.to(lin_dtype).contiguous()
    offs = kernel_offsets(ksize, idx.device, lin_dtype)
    ref_lin, ref_valid = ref_loop(
        idx_lin, offs, eff_pad, eff_st, dil, out_shape, kv, False
    )

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    s.record()
    k_lin, k_valid = pair_kernels.candidate_out_lin_valid(
        idx_lin, offs, eff_pad, eff_st, list(dil), out_shape, kv, False
    )
    e.record()
    torch.cuda.synchronize()
    ok = torch.equal(ref_valid, k_valid) and torch.equal(
        ref_lin[ref_valid], k_lin[k_valid]
    )
    return ok, s.elapsed_time(e), int(idx.shape[0])


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    k3 = (3, 3, 3)
    # warmup the (kv=27, ndim=3) specialization
    one_call(20_000, (50, 50, 50), k3, [1, 1, 1], [1, 1, 1], [1, 1, 1], True)

    # A varying stream: point count AND grid size change every iteration.
    g = torch.Generator().manual_seed(0)
    times, all_ok = [], True
    print(
        f"  {'iter':>4s} {'N(req)':>8s} {'grid':>16s} {'subm':>5s} {'N(real)':>9s} {'ms':>8s} {'eq':>4s}"
    )
    for it in range(16):
        npb = int(torch.randint(2_000, 130_000, (1,), generator=g).item())
        side = int(torch.randint(24, 96, (1,), generator=g).item())
        ss = (side, side, side)
        subm = bool(it % 2)
        stride = [1, 1, 1] if subm else [2, 2, 2]
        ok, ms, nreal = one_call(npb, ss, k3, stride, [1, 1, 1], [1, 1, 1], subm)
        all_ok &= ok
        times.append(ms)
        print(
            f"  {it:>4d} {npb:>8d} {ss!s:>16s} {subm!s:>5s} {nreal:>9d} {ms:>8.3f} {'OK' if ok else 'BAD':>4s}"
        )

    med = statistics.median(times)
    mx = max(times)
    # no re-autotune spike: every post-warmup call within ~6x median (autotune of
    # the ~10-config space would be orders of magnitude slower than one launch).
    no_retune = mx < max(6 * med, med + 5.0)
    print(f"\n  median={med:.3f}ms  max={mx:.3f}ms  no-retune-spike={no_retune}")
    print("\n" + ("VARYING SIZE OK ✅" if (all_ok and no_retune) else "PROBLEM ❌"))
    raise SystemExit(0 if (all_ok and no_retune) else 1)


if __name__ == "__main__":
    main()
