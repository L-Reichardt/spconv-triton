# SPDX-FileCopyrightText: Copyright contributors to the spconv-Triton project
# SPDX-FileCopyrightText: Derived from spconv, Copyright 2021 Yan Yan and contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Triton kernels for sparse convolution (implicit gather-GEMM-scatter).

Deterministic (no atomics): each program block owns an output tile and loops
over kernel offsets / reduction chunks.

fp32 uses IEEE dot products (spconv disables TF32); live flag
``constants.SPCONV_ALLOW_TF32`` switches the fp32 path to TF32, mirroring
spconv's ``use_tf32`` knob (read per call).

fp16 accumulates in fp32 by default; live flag ``SPCONV_ALLOW_FP16_ACCUM=1``
restores spconv's fp16-in-fp16 accumulation -- a win ONLY on consumer Ampere/Ada
(half-rate fp32-accumulate), no faster and less accurate on data-center GPUs
(see README). When on, fwd/input-grad accumulate fp16 only while ``kv*C`` stays
within the overflow guard (``_FP16_ACCUM_MAX_REDUCTION``), fp32 above it; the
weight-grad uses BOUNDED fp16 (fp16 per ``BLOCK_P`` tile, fp32 across tiles), safe
at any length. Per-layer ``fp32_accum`` arg overrides (True->fp32, False->fp16
policy, None->flag).

Tile sizes / warps / stages come from ``triton.autotune``. Channel counts
(``N_dim``/``R_dim``) and ``kv`` are ``tl.constexpr`` (reduction loops unroll,
loads software-pipeline); row count ``M`` stays runtime -- it varies per input
and must not key the autotune cache. Autotuning only reorders the reduction at
the fp-reassociation level (never for max/avg pooling), inside suite tolerance.
"""

import functools
import inspect

import torch
import triton
import triton.language as tl

from spconv_triton import constants

from ._autotune import AUTOTUNE_CACHE_KW

# tl.dot's fp32-precision kwarg was renamed across Triton versions: >=3.0 takes
# input_precision="ieee"/"tf32", 2.2/2.3 the older allow_tf32=False/True. Same
# 3-operand accumulator, pure kwarg rename -- identical math. Triton <=2.1 lacks
# the accumulator arg and is unsupported (needs torch >=2.2); _dot fails fast.
_DOT_PARAMS = inspect.signature(tl.dot).parameters
_TRITON_DOT_HAS_INPUT_PRECISION = "input_precision" in _DOT_PARAMS

if _TRITON_DOT_HAS_INPUT_PRECISION:

    @triton.jit
    def _dot(lhs, rhs, acc, IEEE: tl.constexpr, ALLOW_TF32: tl.constexpr):
        # out_dtype must match acc: tl.dot defaults to float32, which asserts when
        # acc is fp16 (fp16-accumulate path). fp32 paths pass float32 -> unchanged.
        if IEEE:
            return tl.dot(
                lhs,
                rhs,
                acc,
                input_precision="tf32" if ALLOW_TF32 else "ieee",
                out_dtype=acc.dtype,
            )
        return tl.dot(lhs, rhs, acc, out_dtype=acc.dtype)

else:

    @triton.jit
    def _dot(lhs, rhs, acc, IEEE: tl.constexpr, ALLOW_TF32: tl.constexpr):
        if IEEE:
            return tl.dot(lhs, rhs, acc, allow_tf32=ALLOW_TF32, out_dtype=acc.dtype)
        return tl.dot(lhs, rhs, acc, out_dtype=acc.dtype)


def _gemm_configs():
    """Autotune candidates (BLOCK_M, BLOCK_N, BLOCK_R, num_warps, num_stages)
    for the gather-GEMM (fwd + bwd-input). Safe superset -- losing/over-budget
    configs are pruned at no cost.

    Backend-spanning, not just Ampere:
    - num_stages 2..5: 3..4 for NVIDIA cp.async; 2 for AMD CDNA / Intel XPU
      pipeliner (also keeps large tiles within CDNA's 64 KB/CU LDS budget).
    - BLOCK_R up to 128 for the wide-channel compute-bound regime (larger K-tile;
      most impactful on TF32/fp16).
    - GROUP_SIZE_M (L2-reuse swizzle depth) is autotuned since its optimum scales
      with LLC size (~2 MB NVIDIA L2 to ~256 MB MI300X). Bit-identical regardless
      of value (only tile visit order changes).
    """
    # GROUP_SIZE_M defaults to 8 (Ampere sweet spot) unless varied in the sweep.
    base = [
        (64, 64, 32, 4, 3),
        (64, 64, 32, 4, 4),
        (128, 64, 32, 4, 3),
        (64, 128, 32, 4, 3),
        (128, 128, 32, 8, 3),
        (128, 128, 32, 4, 4),
        (128, 64, 64, 4, 3),
        (64, 64, 64, 4, 4),
        (32, 64, 32, 2, 4),
        (64, 32, 32, 2, 4),
        (128, 32, 32, 4, 4),
        (32, 32, 32, 2, 5),
        (256, 64, 32, 8, 3),
        (64, 256, 32, 8, 3),
        # Larger tiles / deeper reduction for the compute-bound large-C regime.
        (128, 256, 64, 8, 3),
        (256, 128, 32, 8, 3),
        (128, 128, 64, 8, 4),
        # num_stages=2 mid-tiles: AMD CDNA / Intel XPU pipeliner optimum.
        (128, 128, 32, 8, 2),
        (128, 64, 32, 4, 2),
        (64, 64, 32, 4, 2),
        # BLOCK_R=128 for the wide-channel compute-bound regime (TF32/fp16).
        (128, 128, 128, 8, 3),
        (128, 64, 128, 8, 3),
        (64, 128, 128, 8, 3),
        # BLOCK_N/BLOCK_R=16 (tl.dot minimum) for narrow channels (C=16 stage):
        # a 32-floor masks off half the tile there.
        (64, 16, 16, 2, 4),
        (128, 16, 16, 4, 4),
        (256, 16, 16, 4, 3),
        (128, 32, 16, 4, 3),
        (128, 16, 32, 4, 3),
    ]
    configs = [
        triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_R": br, "GROUP_SIZE_M": 8},
            num_warps=w,
            num_stages=s,
        )
        for (bm, bn, br, w, s) in base
    ]
    # Swizzle sweep: pick GROUP_SIZE_M only for wide tiles where L2 reuse pays;
    # narrow tiles have a single N-tile (swizzle is a no-op there).
    for bm, bn, br, w, s in [
        (128, 128, 32, 8, 3),
        (256, 128, 32, 8, 3),
        (128, 256, 64, 8, 3),
    ]:
        for g in (1, 4, 16):
            configs.append(
                triton.Config(
                    {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_R": br, "GROUP_SIZE_M": g},
                    num_warps=w,
                    num_stages=s,
                )
            )
    return configs


def _dweight_configs():
    """Autotune candidates (BLOCK_P, BLOCK_K, BLOCK_C, num_warps, num_stages)
    for the weight-gradient kernel, same idiom as ``_gemm_configs``.
    - Wide-channel (128) tiles + a num_stages=2 AMD CDNA / Intel XPU variant.
    - Deep pipelining (num_stages 6-7) for the small-tile path (many-trip,
      small-SMEM reduction). num_stages only changes prefetch depth -> bit-id.
    - BLOCK_K/BLOCK_C=16 for narrow channels (C=16); a 32-floor masks off 3/4.
    """
    base = [
        (64, 64, 64, 4, 3),
        (128, 64, 64, 4, 3),
        (128, 64, 64, 8, 4),
        (64, 32, 64, 4, 4),
        (64, 64, 32, 4, 4),
        (128, 32, 32, 4, 4),
        (256, 64, 64, 8, 3),
        (64, 32, 32, 2, 5),
        # wide-channel tiles + stages=2 (AMD/Intel pipeliner)
        (64, 128, 128, 8, 3),
        (128, 128, 64, 8, 3),
        (128, 64, 128, 8, 3),
        (64, 64, 64, 4, 2),
        # deep pipelining for the small-tile path
        (64, 32, 32, 2, 6),
        (64, 32, 32, 2, 7),
        (64, 32, 32, 4, 6),
        (64, 64, 32, 4, 6),
        (128, 32, 32, 4, 6),
        # narrow-channel (C=16) output tiles
        (128, 16, 16, 2, 4),
        (256, 16, 16, 4, 3),
        (128, 32, 16, 4, 3),
        (256, 16, 16, 2, 6),
    ]
    return [
        triton.Config(
            {"BLOCK_P": bp, "BLOCK_K": bk, "BLOCK_C": bc},
            num_warps=w,
            num_stages=s,
        )
        for (bp, bk, bc, w, s) in base
    ]


# Largest dW tile, derived (not hardcoded) so the _dweight_splits occupancy
# estimate can't drift when configs grow.
_DW_MAX_BLOCK_K = max(c.kwargs["BLOCK_K"] for c in _dweight_configs())
_DW_MAX_BLOCK_C = max(c.kwargs["BLOCK_C"] for c in _dweight_configs())


@triton.jit
def _or_combine(a, b):
    """Bitwise-OR reduction combiner (for tl.reduce over a tile's masks)."""
    return a | b


@triton.autotune(
    configs=_gemm_configs(),
    # HAS_BIAS is deliberately NOT a key: bias is a single post-reduction broadcast
    # add that does not shift the optimal tile/warp/stage, so keying on it would run
    # two identical sweeps (bias vs no-bias). It stays a constexpr compile specializer.
    key=[
        "kv",
        "N_dim",
        "R_dim",
        "IEEE",
        "ALLOW_TF32",
        "FP16_ACC",
        "USE_MASK",
    ],
    **AUTOTUNE_CACHE_KW,
)
@triton.jit
def _gather_gemm_kernel(
    feats_ptr,
    weight_ptr,
    pairs_ptr,
    out_ptr,
    bias_ptr,
    pmask_ptr,
    argsort_ptr,
    M,
    N_dim: tl.constexpr,
    R_dim: tl.constexpr,
    kv: tl.constexpr,
    s_fr,
    s_fc,
    s_wkv,
    s_wr,
    s_wn,
    s_pk,
    s_pm,
    s_or,
    s_oc,
    HAS_BIAS: tl.constexpr,
    IEEE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    FP16_ACC: tl.constexpr,
    USE_MASK: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """out[m, n] = sum_k sum_r feats[pairs[k, m], r] * weight[k][r, n].

    M: output rows (runtime); N_dim: out channels; R_dim: reduction dim; kv:
    kernel volume. weight[k, r, n] addressed via caller strides so the KRSC
    tensor views as W[k][C_in, C_out_T] (forward) or its transpose (input-grad).

    USE_MASK (spconv's MaskImplicitGemm): rows visited in ``argsort_ptr`` order
    (similar-bitmask rows grouped per tile); the tile's OR'd bitmask gates the
    offset loop, skipping any offset empty across the whole tile. Skipped terms
    gather all-zero ``a`` and contribute 0 -> bit-identical to the dense path,
    only zero-work removed. Output stored at real row ``rm = argsort[slot]``.
    """
    # L2-reuse program swizzle: remap the linear pid into super-groups of
    # GROUP_SIZE_M row-tiles so neighbouring output tiles reuse weight/feature
    # rows from L2. Pure index remap -- only tile visit order changes.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N_dim, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    # int64 row index so (row * channel_stride) doesn't overflow int32 for
    # tensors with >= 2**31 elements (large-GPU scale).
    row_slot = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = row_slot < M
    if USE_MASK:
        rm = tl.load(argsort_ptr + row_slot, mask=m_mask, other=0).to(tl.int64)
        tile_bits = tl.load(pmask_ptr + row_slot, mask=m_mask, other=0)
        tile_mask = tl.reduce(tile_bits, 0, _or_combine)
    else:
        rm = row_slot.to(tl.int64)
        tile_mask = -1
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = rn < N_dim
    # FP16_ACC: accumulate the reduction in fp16 (full-rate on consumer Ampere),
    # gated by the caller's overflow guard on kv*R_dim. Otherwise fp32.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16 if FP16_ACC else tl.float32)
    for k in range(0, kv):
        if (tile_mask >> k) & 1:
            idx = tl.load(pairs_ptr + k * s_pk + rm * s_pm, mask=m_mask, other=-1)
            row_ok = idx >= 0
            idx_safe = tl.where(row_ok, idx, 0).to(tl.int64)
            for r0 in range(0, R_dim, BLOCK_R):
                rr = r0 + tl.arange(0, BLOCK_R)
                r_mask = rr < R_dim
                a = tl.load(
                    feats_ptr + idx_safe[:, None] * s_fr + rr[None, :] * s_fc,
                    mask=row_ok[:, None] & r_mask[None, :],
                    other=0.0,
                )
                w = tl.load(
                    weight_ptr + k * s_wkv + rr[:, None] * s_wr + rn[None, :] * s_wn,
                    mask=r_mask[:, None] & n_mask[None, :],
                    other=0.0,
                )
                acc = _dot(a, w, acc, IEEE, ALLOW_TF32)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + rn, mask=n_mask, other=0.0).to(acc.dtype)
        acc += bias[None, :]
    out_ptrs = out_ptr + rm[:, None] * s_or + rn[None, :] * s_oc
    tl.store(
        out_ptrs,
        acc.to(out_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.autotune(
    configs=_dweight_configs(),
    key=["kv", "K_dim", "C_dim", "IEEE", "ALLOW_TF32", "FP16_ACC"],
    **AUTOTUNE_CACHE_KW,
)
@triton.jit
def _dweight_masked_kernel(
    feats_ptr,
    gout_ptr,
    sig_i_ptr,
    sig_o_ptr,
    seg_ptr,
    dw_ptr,
    K_dim: tl.constexpr,
    C_dim: tl.constexpr,
    kv: tl.constexpr,
    splits,  # split-P factor (runtime): grid axis 0 is kv * splits
    s_fr,
    s_fc,
    s_gr,
    s_gc,
    s_dk,
    s_dkv,
    s_dc,
    s_ds,  # split-axis stride of dw_ptr ([splits|1, K, kv, C])
    IEEE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    FP16_ACC: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """dW[ko][c, n] = sum over COMPACTED contributions of offset ko of
    feats[in_row, c] * gout[out_row, n].

    ``valid_signal`` ((sig_i, sig_o) sliced by ``seg``) lists per offset only the
    existing (input_row, output_row) pairs, output rows ascending (dense order).
    Skips the ~80% of ``-1`` (zero-gather) slots a dense loop would discard; the
    skipped terms are exactly zero, so the result equals the dense one up to
    reduction REGROUPING -- within the calibrated gradient atol.

    Split-P (``splits > 1``): the per-offset reduction is chunked across
    ``splits`` programs writing fp32 PARTIAL tiles at ``split * s_ds``, summed on
    the host (one deterministic torch kernel); see ``_dweight_splits``.
    ``splits == 1`` reduces to the unsplit form (byte-identical). Splitting only
    regroups fp32 partials -- same reassociation class as the BLOCK_P choice."""
    pid_kvs = tl.program_id(0)
    pid_kv = pid_kvs // splits
    split = pid_kvs - pid_kv * splits
    pid_c = tl.program_id(1)
    pid_k = tl.program_id(2)
    rc = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    c_mask = rc < C_dim
    k_mask = rk < K_dim
    seg_start = tl.load(seg_ptr + pid_kv)
    seg_end = tl.load(seg_ptr + pid_kv + 1)
    seglen = seg_end - seg_start
    chunk = tl.cdiv(seglen, splits)
    p_lo = split * chunk
    p_hi = tl.minimum(seglen, p_lo + chunk)
    acc = tl.zeros((BLOCK_C, BLOCK_K), dtype=tl.float32)
    for p0 in range(p_lo, p_hi, BLOCK_P):
        rp = p0 + tl.arange(0, BLOCK_P)
        p_mask = rp < p_hi
        in_rows = tl.load(sig_i_ptr + seg_start + rp, mask=p_mask, other=0).to(tl.int64)
        out_rows = tl.load(sig_o_ptr + seg_start + rp, mask=p_mask, other=0).to(
            tl.int64
        )
        a = tl.load(
            feats_ptr + in_rows[:, None] * s_fr + rc[None, :] * s_fc,
            mask=p_mask[:, None] & c_mask[None, :],
            other=0.0,
        )
        g = tl.load(
            gout_ptr + out_rows[:, None] * s_gr + rk[None, :] * s_gc,
            mask=p_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        if FP16_ACC:
            # Bounded fp16: accumulate this BLOCK_P tile in fp16 (full-rate MMA on
            # consumer Ampere), then promote into the fp32 ``acc`` so error doesn't
            # grow with reduction length. Finer than spconv's split-K bound.
            part = _dot(
                tl.trans(a),
                g,
                tl.zeros((BLOCK_C, BLOCK_K), dtype=tl.float16),
                IEEE,
                ALLOW_TF32,
            )
            acc += part.to(tl.float32)
        else:
            acc = _dot(tl.trans(a), g, acc, IEEE, ALLOW_TF32)
    dw_ptrs = (
        dw_ptr
        + split.to(tl.int64) * s_ds
        + rk[None, :] * s_dk
        + pid_kv * s_dkv
        + rc[:, None] * s_dc
    )
    tl.store(
        dw_ptrs, acc.to(dw_ptr.dtype.element_ty), mask=c_mask[:, None] & k_mask[None, :]
    )


@functools.cache
def _multiprocessor_count(device: torch.device) -> int:
    """SM/CU count, cached per device (get_device_properties is measurable in the
    backward hot path). ``get_device_module`` keeps it portable beyond torch.cuda
    (CUDA/ROCm/XPU); backends reporting no count fall back to 16 (perf-only
    default -- only shrinks the split factor)."""
    get_mod = getattr(torch, "get_device_module", None)  # absent on torch < 2.3
    mod = get_mod(device.type) if get_mod is not None else torch.cuda
    props = mod.get_device_properties(device)
    return int(getattr(props, "multi_processor_count", 16))


def _dweight_splits(device, kv: int, K: int, C: int, nnz: int) -> int:
    """Split-P factor for the weight-gradient grid.

    The natural grid ``kv x ceil(C/BC) x ceil(K/BK)`` can be as few as ``kv``
    programs when C, K fit one tile (27 for k3 at C<=128), stranding SMs on large
    devices (132-SM H100 ~20% occupancy). Split so axis 0 reaches ~2 CTAs/SM even
    under the largest tiles. Capped at 16, and so each split keeps >= ~1k pairs
    (tiny reductions don't amortize a second pass)."""
    target = _multiprocessor_count(device) * 2
    n_tiles_min = kv * triton.cdiv(C, _DW_MAX_BLOCK_C) * triton.cdiv(K, _DW_MAX_BLOCK_K)
    splits = min(16, max(1, target // max(1, n_tiles_min)))
    avg_seglen = max(1, nnz // max(1, kv))
    return min(splits, max(1, avg_seglen // 1024))


def _is_fp32(t: torch.Tensor) -> bool:
    return t.dtype == torch.float32


def _allow_tf32(t: torch.Tensor) -> bool:
    """TF32 only on the fp32 path, and only when live ``SPCONV_ALLOW_TF32`` is set
    (read per call, like spconv)."""
    return _is_fp32(t) and bool(constants.SPCONV_ALLOW_TF32)


# spconv's fp16 overflow guard (spconv/algo.py:700-707): accumulate fp16 while
# reduction length ``C*kv`` stays <= this, fp32 above ("too large may cause nan").
# 128*27 = the 3x3x3, C=128 boundary.
_FP16_ACCUM_MAX_REDUCTION = 128 * 27


def _use_fp16_accum(dtype: torch.dtype, fp32_accum: bool | None) -> bool:
    """Whether the fp16-accumulate policy applies. Mirrors spconv's ``fp32_accum``
    (True->fp32, False->fp16, None->live ``SPCONV_ALLOW_FP16_ACCUM``, default OFF);
    non-fp16 inputs always False. The fwd/input-grad reduction-length threshold is
    applied by the caller; the weight-grad uses bounded fp16 (no threshold)."""
    if dtype != torch.float16:
        return False
    if fp32_accum is True:
        return False
    if fp32_accum is False:
        return True
    return bool(constants.SPCONV_ALLOW_FP16_ACCUM)


def _should_mask(
    kv: int,
    pmask: torch.Tensor | None,
    argsort: torch.Tensor | None,
) -> bool:
    """Whether the masked implicit-GEMM path is APPLICABLE (always taken when so;
    no perf gating -- bit-identical to dense, applied uniformly for hardware-
    agnostic behavior). Conditions are CORRECTNESS preconditions:
    - ``pmask``/``argsort`` exist: only single-split MaskImplicitGemm produces
      them; Native/pooling/MaskSplit pass None -> dense fallback.
    - ``1 < kv <= 32``: tile bitmask is one 32-bit word (kv<=32); kv==1 is conv1x1
      (handled before the GEMM)."""
    return pmask is not None and argsort is not None and 1 < kv <= 32


def conv_forward(
    features: torch.Tensor,
    filters_krsc: torch.Tensor,
    pair_fwd: torch.Tensor,
    n_out: int,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    pair_mask: torch.Tensor | None = None,
    mask_argsort: torch.Tensor | None = None,
    fp32_accum: bool | None = None,
) -> torch.Tensor:
    """Gather-GEMM forward. Returns [n_out, K].

    features: [N, C]. filters_krsc: [K, kv, C] (any strides). pair_fwd: [kv, n_out]
    int32 gather table. bias: [K] or None. out_dtype: defaults to features.dtype.
    pair_mask/mask_argsort: [n_out] int32 sorted-bitmask/permutation enabling the
    masked path (bit-identical to dense; None -> dense). fp32_accum: see
    ``_use_fp16_accum``."""
    if not features.is_cuda:
        raise AssertionError
    K, kv, C = filters_krsc.shape
    if out_dtype is None:
        out_dtype = features.dtype
    out = torch.empty((n_out, K), dtype=out_dtype, device=features.device)
    if n_out == 0:
        return out
    feats = features.contiguous()
    w = filters_krsc
    pf = pair_fwd
    use_mask = _should_mask(kv, pair_mask, mask_argsort)
    pm = mask_argsort if use_mask else pf  # int32 dummy when unused
    pk = pair_mask if use_mask else pf
    # Forward reduction length kv*C_in: fp16-accumulate only when policy on AND
    # within the overflow-guard threshold.
    fp16_acc = _use_fp16_accum(feats.dtype, fp32_accum) and (
        kv * C <= _FP16_ACCUM_MAX_REDUCTION
    )
    grid = lambda META: (  # noqa: E731
        triton.cdiv(n_out, META["BLOCK_M"]) * triton.cdiv(K, META["BLOCK_N"]),
    )
    _gather_gemm_kernel[grid](
        feats,
        w,
        pf,
        out,
        bias if bias is not None else feats,
        pk,
        pm,
        n_out,
        K,
        C,
        kv,
        feats.stride(0),
        feats.stride(1),
        w.stride(1),
        w.stride(2),
        w.stride(0),
        pf.stride(0),
        pf.stride(1),
        out.stride(0),
        out.stride(1),
        HAS_BIAS=bias is not None,
        IEEE=_is_fp32(feats),
        ALLOW_TF32=_allow_tf32(feats),
        FP16_ACC=fp16_acc,
        USE_MASK=use_mask,
    )
    return out


def conv_backward_input(
    grad_out: torch.Tensor,
    filters_krsc: torch.Tensor,
    pair_bwd: torch.Tensor,
    n_in: int,
    pair_mask: torch.Tensor | None = None,
    mask_argsort: torch.Tensor | None = None,
    fp32_accum: bool | None = None,
) -> torch.Tensor:
    """Input-gradient GEMM. Returns din [n_in, C].
    din[j, c] = sum_k grad_out[pair_bwd[k, j], :] @ W[k][:, c].

    grad_out: [M, K]. filters_krsc: [K, kv, C]. pair_bwd: [kv, n_in].
    pair_mask/mask_argsort: BACKWARD-table sorted bitmask/permutation over n_in
    rows, enabling the masked path as in ``conv_forward``. Subm has no backward
    mask -> dense."""
    K, kv, C = filters_krsc.shape
    din = torch.empty((n_in, C), dtype=grad_out.dtype, device=grad_out.device)
    if n_in == 0:
        return din
    g = grad_out.contiguous()
    w = filters_krsc
    pb = pair_bwd
    use_mask = _should_mask(kv, pair_mask, mask_argsort)
    pm = mask_argsort if use_mask else pb  # int32 dummy when unused
    pk = pair_mask if use_mask else pb
    # Input-grad reduction length kv*C_out (= kv*K); threshold-gated like forward.
    fp16_acc = _use_fp16_accum(g.dtype, fp32_accum) and (
        kv * K <= _FP16_ACCUM_MAX_REDUCTION
    )
    grid = lambda META: (  # noqa: E731
        triton.cdiv(n_in, META["BLOCK_M"]) * triton.cdiv(C, META["BLOCK_N"]),
    )
    _gather_gemm_kernel[grid](
        g,
        w,
        pb,
        din,
        g,
        pk,
        pm,
        n_in,
        C,
        K,
        kv,
        g.stride(0),
        g.stride(1),
        w.stride(1),
        w.stride(0),
        w.stride(2),
        pb.stride(0),
        pb.stride(1),
        din.stride(0),
        din.stride(1),
        HAS_BIAS=False,
        IEEE=_is_fp32(g),
        ALLOW_TF32=_allow_tf32(g),
        FP16_ACC=fp16_acc,
        USE_MASK=use_mask,
    )
    return din


def conv_backward_weight(
    features: torch.Tensor,
    grad_out: torch.Tensor,
    pair_fwd: torch.Tensor,
    weight_shape_krsc,
    fp32_accum: bool | None = None,
) -> torch.Tensor:
    """Weight-gradient GEMM. Returns dW [K, kv, C] (contiguous).

    Builds the compacted ``valid_signal`` from ``pair_fwd`` [kv, n_out] (row-major
    nonzero -> output rows ascending per offset, the dense order), skipping the
    ~80% of ``-1`` zero-gather slots. Works for both implicit-gemm and Native
    tables. Within calibrated gradient atol (reduction regrouping vs dense; see
    ``_dweight_masked_kernel``)."""
    K, kv, C = weight_shape_krsc
    n_out = grad_out.shape[0]
    dw = torch.empty((K, kv, C), dtype=grad_out.dtype, device=grad_out.device)
    if n_out == 0 or kv == 0:
        dw.zero_()
        return dw
    feats = features.contiguous()
    g = grad_out.contiguous()
    pf = pair_fwd
    # Memoized on the pair_fwd tensor (same convention as _spconv_triton_tables in
    # pairs.py: immutable source, cache lives as long as it). The signal is shared
    # by every layer with the same indice_key; without the cache each backward
    # repeats the identical nonzero (a device->host sync) + bincount + cumsum.
    cached = getattr(pf, "_spconv_triton_dw_sig", None)
    if cached is None:
        valid_flat = (pf >= 0).reshape(-1).nonzero(as_tuple=True)[0]
        kk = torch.div(valid_flat, n_out, rounding_mode="floor")
        sig_o = (valid_flat - kk * n_out).to(torch.int32).contiguous()  # out rows
        sig_i = pf.reshape(-1)[valid_flat].contiguous()  # in rows (int32)
        seg = torch.zeros(kv + 1, dtype=torch.int32, device=pf.device)
        seg[1:] = torch.bincount(kk, minlength=kv).cumsum(0).to(torch.int32)
        pf._spconv_triton_dw_sig = (sig_i, sig_o, seg)  # type: ignore[attr-defined]
    else:
        sig_i, sig_o, seg = cached
    # Weight-grad reduction (seglen ~ n_out) has no fixed threshold; kernel uses
    # BOUNDED fp16 (fp16 per BLOCK_P tile, fp32 across), safe at any length.
    fp16_acc = _use_fp16_accum(feats.dtype, fp32_accum)
    splits = _dweight_splits(feats.device, kv, K, C, int(sig_i.shape[0]))
    # splits == 1: store straight into dw via a [1, K, kv, C] view (byte-identical
    # to unsplit; split always 0). splits > 1: fp32 partials, summed below.
    part = (
        torch.empty((splits, K, kv, C), dtype=torch.float32, device=dw.device)
        if splits > 1
        else dw.view(1, K, kv, C)
    )
    grid = lambda META: (  # noqa: E731
        kv * splits,
        triton.cdiv(C, META["BLOCK_C"]),
        triton.cdiv(K, META["BLOCK_K"]),
    )
    _dweight_masked_kernel[grid](
        feats,
        g,
        sig_i,
        sig_o,
        seg,
        part,
        K,
        C,
        kv,
        splits,
        feats.stride(0),
        feats.stride(1),
        g.stride(0),
        g.stride(1),
        part.stride(1),
        part.stride(2),
        part.stride(3),
        part.stride(0),
        IEEE=_is_fp32(feats),
        ALLOW_TF32=_allow_tf32(feats),
        FP16_ACC=fp16_acc,
    )
    if splits > 1:
        dw.copy_(part.sum(0))
    return dw
