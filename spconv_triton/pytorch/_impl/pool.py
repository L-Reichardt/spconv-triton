# SPDX-FileCopyrightText: Copyright contributors to the spconv-Triton project
# SPDX-FileCopyrightText: Derived from spconv, Copyright 2021 Yan Yan and contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Triton kernels for sparse max/avg pooling (gather via pair tables).

spconv parity quirks: max-pool accumulator inits at 0 (NOT -inf), so
all-negative windows yield 0; max-pool backward routes grad to every input
equal to the output; avg-pool divides by the count of present inputs.

Tiling/num_warps/num_stages are autotuned; the per-output kv reduction is
tile-independent, so autotuning never changes numerics.
"""

import torch
import triton
import triton.language as tl

from ._autotune import AUTOTUNE_CACHE_KW


def _pool_configs():
    """Autotune configs. BLOCK_C=16 group serves narrow-channel pooling (C=16),
    where a 32-floor would mask off half of every tile."""
    base = [
        (64, 32, 2, 2),
        (64, 64, 4, 2),
        (128, 32, 4, 2),
        (128, 64, 4, 3),
        (32, 64, 2, 2),
        (256, 32, 4, 3),
        (64, 128, 4, 3),
        (128, 128, 8, 3),
        (128, 16, 2, 2),
        (256, 16, 4, 2),
        (512, 16, 4, 3),
    ]
    return [
        triton.Config({"BLOCK_M": bm, "BLOCK_C": bc}, num_warps=w, num_stages=s)
        for (bm, bc, w, s) in base
    ]


@triton.autotune(
    configs=_pool_configs(), key=["kv", "C", "ACC_F64"], **AUTOTUNE_CACHE_KW
)
@triton.jit
def _maxpool_fwd_kernel(
    feats_ptr,
    pairs_ptr,
    out_ptr,
    M,
    C: tl.constexpr,
    kv: tl.constexpr,
    s_fr,
    s_fc,
    s_pk,
    s_pm,
    s_or,
    s_oc,
    ACC_F64: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rc = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    m_mask = rm < M
    c_mask = rc < C
    # fp64 for double inputs (exact, so backward value-equality vs input holds);
    # fp32/fp16 keep the fp32 accumulator byte-for-byte.
    acc_dtype = tl.float64 if ACC_F64 else tl.float32
    # spconv quirk: accumulator starts at 0, not -inf
    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=acc_dtype)
    for k in range(0, kv):
        idx = tl.load(pairs_ptr + k * s_pk + rm * s_pm, mask=m_mask, other=-1)
        ok = idx >= 0
        idx_safe = tl.where(ok, idx, 0).to(tl.int64)
        v = tl.load(
            feats_ptr + idx_safe[:, None] * s_fr + rc[None, :] * s_fc,
            mask=ok[:, None] & c_mask[None, :],
            other=float("-inf"),
        ).to(acc_dtype)
        acc = tl.maximum(acc, v)
    tl.store(
        out_ptr + rm[:, None] * s_or + rc[None, :] * s_oc,
        acc.to(out_ptr.dtype.element_ty),
        mask=m_mask[:, None] & c_mask[None, :],
    )


@triton.autotune(
    configs=_pool_configs(), key=["kv", "C", "ACC_F64"], **AUTOTUNE_CACHE_KW
)
@triton.jit
def _maxpool_bwd_kernel(
    feats_ptr,
    out_ptr,
    gout_ptr,
    pairs_ptr,
    din_ptr,
    N,
    C: tl.constexpr,
    kv: tl.constexpr,
    s_fr,
    s_fc,
    s_or,
    s_oc,
    s_gr,
    s_gc,
    s_pk,
    s_pm,
    s_dr,
    s_dc,
    ACC_F64: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    rn = (pid_n * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rc = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    n_mask = rn < N
    c_mask = rc < C
    acc_dtype = tl.float64 if ACC_F64 else tl.float32
    f = tl.load(
        feats_ptr + rn[:, None] * s_fr + rc[None, :] * s_fc,
        mask=n_mask[:, None] & c_mask[None, :],
        other=0.0,
    )
    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=acc_dtype)
    for k in range(0, kv):
        idx = tl.load(pairs_ptr + k * s_pk + rn * s_pm, mask=n_mask, other=-1)
        ok = idx >= 0
        idx_safe = tl.where(ok, idx, 0).to(tl.int64)
        o = tl.load(
            out_ptr + idx_safe[:, None] * s_or + rc[None, :] * s_oc,
            mask=ok[:, None] & c_mask[None, :],
            other=0.0,
        )
        g = tl.load(
            gout_ptr + idx_safe[:, None] * s_gr + rc[None, :] * s_gc,
            mask=ok[:, None] & c_mask[None, :],
            other=0.0,
        )
        hit = ok[:, None] & (o == f)
        acc += tl.where(hit, g.to(acc_dtype), 0.0)
    tl.store(
        din_ptr + rn[:, None] * s_dr + rc[None, :] * s_dc,
        acc.to(din_ptr.dtype.element_ty),
        mask=n_mask[:, None] & c_mask[None, :],
    )


@triton.autotune(
    configs=_pool_configs(), key=["kv", "C", "ACC_F64"], **AUTOTUNE_CACHE_KW
)
@triton.jit
def _avgpool_fwd_kernel(
    feats_ptr,
    pairs_ptr,
    out_ptr,
    count_ptr,
    M,
    C: tl.constexpr,
    kv: tl.constexpr,
    s_fr,
    s_fc,
    s_pk,
    s_pm,
    s_or,
    s_oc,
    ACC_F64: tl.constexpr,
    WRITE_COUNT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rc = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    m_mask = rm < M
    c_mask = rc < C
    acc_dtype = tl.float64 if ACC_F64 else tl.float32
    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=acc_dtype)
    cnt = tl.zeros((BLOCK_M,), dtype=tl.int32)
    for k in range(0, kv):
        idx = tl.load(pairs_ptr + k * s_pk + rm * s_pm, mask=m_mask, other=-1)
        ok = idx >= 0
        idx_safe = tl.where(ok, idx, 0).to(tl.int64)
        v = tl.load(
            feats_ptr + idx_safe[:, None] * s_fr + rc[None, :] * s_fc,
            mask=ok[:, None] & c_mask[None, :],
            other=0.0,
        )
        acc += v.to(acc_dtype)
        cnt += ok.to(tl.int32)
    cnt_safe = tl.maximum(cnt, 1)
    acc = acc / cnt_safe[:, None].to(acc_dtype)
    tl.store(
        out_ptr + rm[:, None] * s_or + rc[None, :] * s_oc,
        acc.to(out_ptr.dtype.element_ty),
        mask=m_mask[:, None] & c_mask[None, :],
    )
    if WRITE_COUNT and pid_c == 0:
        tl.store(count_ptr + rm, cnt, mask=m_mask)


@triton.autotune(
    configs=_pool_configs(), key=["kv", "C", "ACC_F64"], **AUTOTUNE_CACHE_KW
)
@triton.jit
def _avgpool_bwd_kernel(
    gout_ptr,
    count_ptr,
    pairs_ptr,
    din_ptr,
    N,
    C: tl.constexpr,
    kv: tl.constexpr,
    s_gr,
    s_gc,
    s_pk,
    s_pm,
    s_dr,
    s_dc,
    ACC_F64: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    rn = (pid_n * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rc = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    n_mask = rn < N
    c_mask = rc < C
    acc_dtype = tl.float64 if ACC_F64 else tl.float32
    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=acc_dtype)
    for k in range(0, kv):
        idx = tl.load(pairs_ptr + k * s_pk + rn * s_pm, mask=n_mask, other=-1)
        ok = idx >= 0
        idx_safe = tl.where(ok, idx, 0).to(tl.int64)
        g = tl.load(
            gout_ptr + idx_safe[:, None] * s_gr + rc[None, :] * s_gc,
            mask=ok[:, None] & c_mask[None, :],
            other=0.0,
        )
        cnt = tl.load(count_ptr + idx_safe, mask=n_mask & ok, other=1)
        # spconv quirk: avgpool backward MULTIPLIES grad by window count (not divides)
        acc += tl.where(ok[:, None], g.to(acc_dtype) * cnt[:, None].to(acc_dtype), 0.0)
    tl.store(
        din_ptr + rn[:, None] * s_dr + rc[None, :] * s_dc,
        acc.to(din_ptr.dtype.element_ty),
        mask=n_mask[:, None] & c_mask[None, :],
    )


def maxpool_forward(
    features: torch.Tensor, pair_fwd: torch.Tensor, n_out: int
) -> torch.Tensor:
    kv = pair_fwd.shape[0]
    feats = features.contiguous()
    C = feats.shape[1]
    out = torch.empty((n_out, C), dtype=feats.dtype, device=feats.device)
    if n_out == 0 or C == 0:
        return out
    grid = lambda META: (  # noqa: E731
        triton.cdiv(n_out, META["BLOCK_M"]),
        triton.cdiv(C, META["BLOCK_C"]),
    )
    _maxpool_fwd_kernel[grid](
        feats,
        pair_fwd,
        out,
        n_out,
        C,
        kv,
        feats.stride(0),
        feats.stride(1),
        pair_fwd.stride(0),
        pair_fwd.stride(1),
        out.stride(0),
        out.stride(1),
        ACC_F64=feats.dtype == torch.float64,
    )
    return out


def maxpool_backward(
    features: torch.Tensor,
    out_features: torch.Tensor,
    grad_out: torch.Tensor,
    pair_bwd: torch.Tensor,
) -> torch.Tensor:
    kv = pair_bwd.shape[0]
    feats = features.contiguous()
    out = out_features.contiguous()
    g = grad_out.contiguous()
    din = torch.zeros_like(feats)
    n_in, C = feats.shape
    if n_in == 0 or C == 0:
        return din
    grid = lambda META: (  # noqa: E731
        triton.cdiv(n_in, META["BLOCK_M"]),
        triton.cdiv(C, META["BLOCK_C"]),
    )
    _maxpool_bwd_kernel[grid](
        feats,
        out,
        g,
        pair_bwd,
        din,
        n_in,
        C,
        kv,
        feats.stride(0),
        feats.stride(1),
        out.stride(0),
        out.stride(1),
        g.stride(0),
        g.stride(1),
        pair_bwd.stride(0),
        pair_bwd.stride(1),
        din.stride(0),
        din.stride(1),
        ACC_F64=feats.dtype == torch.float64,
    )
    return din


def avgpool_forward(
    features: torch.Tensor, pair_fwd: torch.Tensor, n_out: int, calc_count: bool = True
):
    kv = pair_fwd.shape[0]
    feats = features.contiguous()
    C = feats.shape[1]
    out = torch.empty((n_out, C), dtype=feats.dtype, device=feats.device)
    count = torch.zeros((n_out,), dtype=torch.int32, device=feats.device)
    if n_out == 0 or C == 0:
        return out, count
    grid = lambda META: (  # noqa: E731
        triton.cdiv(n_out, META["BLOCK_M"]),
        triton.cdiv(C, META["BLOCK_C"]),
    )
    _avgpool_fwd_kernel[grid](
        feats,
        pair_fwd,
        out,
        count,
        n_out,
        C,
        kv,
        feats.stride(0),
        feats.stride(1),
        pair_fwd.stride(0),
        pair_fwd.stride(1),
        out.stride(0),
        out.stride(1),
        ACC_F64=feats.dtype == torch.float64,
        WRITE_COUNT=True,
    )
    return out, count


def avgpool_backward(
    grad_out: torch.Tensor, pair_bwd: torch.Tensor, count: torch.Tensor
) -> torch.Tensor:
    kv = pair_bwd.shape[0]
    g = grad_out.contiguous()
    n_in = pair_bwd.shape[1]
    C = g.shape[1]
    din = torch.zeros((n_in, C), dtype=g.dtype, device=g.device)
    if n_in == 0 or C == 0:
        return din
    grid = lambda META: (  # noqa: E731
        triton.cdiv(n_in, META["BLOCK_M"]),
        triton.cdiv(C, META["BLOCK_C"]),
    )
    _avgpool_bwd_kernel[grid](
        g,
        count,
        pair_bwd,
        din,
        n_in,
        C,
        kv,
        g.stride(0),
        g.stride(1),
        pair_bwd.stride(0),
        pair_bwd.stride(1),
        din.stride(0),
        din.stride(1),
        ACC_F64=g.dtype == torch.float64,
    )
    return din
