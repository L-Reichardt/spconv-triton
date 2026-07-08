# SPDX-FileCopyrightText: Copyright contributors to the spconv-Triton project
# SPDX-FileCopyrightText: Derived from spconv, Copyright 2021 Yan Yan and contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Fused Triton kernels for sparse-conv pair generation (NVIDIA/AMD/Intel).

Collapses the per-dimension torch-op loop that built the per-(kernel-offset,
input-point) output-coord + validity grids into single fused passes, plus the
mask bit-pack and the igemm/regular/subm membership back-halves.

Byte-identical contract: only VALID entries are consumed downstream; for those
the kernels reproduce the torch chains exactly -- same integer dtype (int32 when
``batch * volume < 2**31``, else int64) so overflow wrap matches, and validity
always gates ``o >= 0`` so truncating vs floor division agree on survivors.
Invalid entries' ``out_lin`` is don't-care (masked before read).
"""

import triton
import triton.language as tl

from ._autotune import AUTOTUNE_CACHE_KW


def _candidate_configs():
    """Block/warp configs for the memory-bound elementwise sweep.

    Autotune keys on layer-fixed (kv, ndim) only -- N is absorbed by the grid, so
    a scene-size change every forward never re-triggers tuning.
    """
    return [
        triton.Config({"BLOCK": bs}, num_warps=w)
        for bs in (256, 512, 1024, 2048, 4096)
        for w in (4, 8)
    ]


@triton.autotune(configs=_candidate_configs(), key=["KV", "NDIM"], **AUTOTUNE_CACHE_KW)
@triton.jit
def _candidates_kernel(
    idx_ptr,  # [N, NDIM+1] lin_dtype, row-major; column 0 = batch index
    offs_ptr,  # [KV, NDIM] lin_dtype kernel offsets (row-major, last dim fastest)
    params_ptr,  # [4*NDIM] lin_dtype, packed pad | stride | dilation | out_shape
    # (one packed tensor -> one host->device copy; scalar args would value-
    # specialize + recompile per scene since out_shape varies)
    out_lin_ptr,  # [KV*N] lin_dtype (output: batch-major linearized out coord)
    valid_ptr,  # [KV*N] int8 (output: 1 if geometrically valid else 0)
    N,
    KV,
    KV_N,
    NDIM: tl.constexpr,
    IDXW: tl.constexpr,  # NDIM + 1 (row stride of idx_ptr)
    TRANSPOSE: tl.constexpr,
    STRIDED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    # Contiguity hint to widen the stores/loads; no-op on values, most helps the
    # weaker AMD/Intel contiguity inference.
    off = tl.max_contiguous(tl.multiple_of(off, BLOCK), BLOCK)
    m = off < KV_N
    k = off // N  # kernel-offset index
    j = off - k * N  # input-point index

    # acc starts at the batch index (batch-major linearization).
    acc = tl.load(idx_ptr + j * IDXW, mask=m, other=0)
    valid = m
    for d in tl.static_range(NDIM):
        c_d = tl.load(idx_ptr + j * IDXW + (d + 1), mask=m, other=0)
        off_d = tl.load(offs_ptr + k * NDIM + d, mask=m, other=0)
        dil_d = tl.load(params_ptr + 2 * NDIM + d)
        pad_d = tl.load(params_ptr + d)
        osh_d = tl.load(params_ptr + 3 * NDIM + d)
        off_d = off_d * dil_d
        if TRANSPOSE:
            st_d = tl.load(params_ptr + NDIM + d)
            o = c_d * st_d + off_d - pad_d
            valid = valid & (o >= 0) & (o < osh_d)
            contrib = o
        else:
            o = c_d + pad_d - off_d  # numerator
            if STRIDED:
                st_d = tl.load(params_ptr + NDIM + d)
                q = o // st_d  # floor == trunc for surviving o>=0 entries
                valid = valid & (o >= 0) & (o - q * st_d == 0) & (q < osh_d)
                contrib = q
            else:
                valid = valid & (o >= 0) & (o < osh_d)
                contrib = o
        acc = acc * osh_d + contrib

    tl.store(out_lin_ptr + off, acc, mask=m)
    tl.store(valid_ptr + off, valid.to(tl.int8), mask=m)


def candidate_out_lin_valid(
    idx_lin,  # [N, ndim+1] lin_dtype, contiguous (col 0 = batch)
    offs,  # [kv, ndim] lin_dtype, contiguous
    pad,  # list[int] length ndim
    stride,  # list[int] length ndim
    dilation,  # list[int] length ndim
    out_shape,  # list[int] length ndim
    kv: int,
    transpose: bool,
):
    """Run the candidate kernel; return (out_lin [kv, N], valid [kv, N] bool).

    Per-dim params travel as one packed tensor in idx_lin's integer dtype so the
    arithmetic matches the torch loop bit-for-bit (one host->device copy).
    """
    import torch  # local: keep the module import-light

    device = idx_lin.device
    lin_dtype = idx_lin.dtype
    N = idx_lin.shape[0]
    kv_n = kv * N
    ndim = len(out_shape)

    out_lin = torch.empty(kv_n, dtype=lin_dtype, device=device)
    valid_i8 = torch.empty(kv_n, dtype=torch.int8, device=device)
    params_t = torch.tensor(
        [*pad, *stride, *dilation, *out_shape], dtype=lin_dtype, device=device
    )
    strided = any(int(s) != 1 for s in stride)

    def grid(meta):
        return (triton.cdiv(kv_n, meta["BLOCK"]),)

    _candidates_kernel[grid](
        idx_lin,
        offs,
        params_t,
        out_lin,
        valid_i8,
        N,
        kv,
        kv_n,
        NDIM=ndim,
        IDXW=ndim + 1,
        TRANSPOSE=transpose,
        STRIDED=strided,
    )
    return out_lin.view(kv, N), valid_i8.view(torch.bool).view(kv, N)


def _mask_pack_configs():
    """Row-tile/warp configs for the mask bit-pack sweep.

    Autotune keys on KV only (MIC follows from KV; N is absorbed by the grid).
    """
    return [
        triton.Config({"BLOCK": bs}, num_warps=w)
        for bs in (256, 512, 1024, 2048)
        for w in (4, 8)
    ]


@triton.autotune(configs=_mask_pack_configs(), key=["KV"], **AUTOTUNE_CACHE_KW)
@triton.jit
def _mask_pack_kernel(
    pair_ptr,  # [KV, N] int32 pair table; value >= 0 means offset valid for row
    words_ptr,  # [N, MIC] int64 (output): bit k of word (k // 32) set iff valid
    N,
    KV: tl.constexpr,
    MIC: tl.constexpr,  # mask_int_count = (KV + 31) // 32
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK + tl.arange(0, BLOCK)
    # Contiguity hint (no-op on values); see the candidate kernel.
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK), BLOCK)
    m = rows < N
    # One word (32 offsets) at a time; constexpr (k // 32 == w) guard drops loads
    # outside this word (total loads == KV). Distinct bits per word, so OR == the
    # index_add reduction bit-for-bit.
    for w in tl.static_range(MIC):
        acc = tl.zeros([BLOCK], tl.int64)
        for k in tl.static_range(KV):
            if k // 32 == w:
                v = tl.load(pair_ptr + k * N + rows, mask=m, other=-1) >= 0
                acc |= v.to(tl.int64) << (k % 32)
        tl.store(words_ptr + rows * MIC + w, acc, mask=m)


def pack_mask_words(pair_table, kv: int, mask_int_count: int):
    """Build per-row offset bitmask words [n, mask_int_count] (int64) for a
    ``pair_table`` [kv, n], one fused kernel.

    Byte-identical to the torch ``arange`` + ``<<`` + ``index_add_`` build (OR ==
    the additive reduction within a word). Caller folds in the per-split & mval,
    argsort, and int32 wrap.
    """
    import torch  # local: keep the module import-light

    n = pair_table.shape[1]
    words = torch.zeros(
        (n, mask_int_count), dtype=torch.int64, device=pair_table.device
    )
    if n == 0:
        return words

    def grid(meta):
        return (triton.cdiv(n, meta["BLOCK"]),)

    _mask_pack_kernel[grid](
        pair_table,
        words,
        n,
        KV=kv,
        MIC=mask_int_count,
    )
    return words


# Reuses the candidate kernel's flat-grid config set (same launch geometry). Keyed
# only on NEED_BWD (layer-fixed: train vs eval store traffic differs -- bwd writes an
# extra [kv, N] tensor); N/n_out are data-dependent and stay out of the key so a scene
# change never re-tunes. Byte-identical for any BLOCK (masked elementwise scatter).
@triton.autotune(configs=_candidate_configs(), key=["NEED_BWD"], **AUTOTUNE_CACHE_KW)
@triton.jit
def _scatter_igemm_kernel(
    valid_ptr,  # [kv, N] int8 (bool view); 1 == offset k valid for input point j
    out_row_ptr,  # [kv, N] int32; output row for (k, j) (don't-care if invalid)
    pair_fwd_ptr,  # [kv, n_out] int32 (output, pre-filled -1): pair_fwd[k, out_row] = j
    pair_bwd_ptr,  # [kv, N] int32 (output, fully written): pair_bwd[k, j] = out_row | -1
    N,
    n_out,
    KV_N,
    NEED_BWD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Single-pass scatter of the implicit-gemm gather tables.

    Flat grid ``off == k*N + j``. Each valid lane writes:
      pair_fwd[k, out_row[k, j]] = j   -- data-dependent SCATTER; collision-free
        because for fixed k the map j -> out_row is injective on the valid set.
      pair_bwd[k, j]            = out_row -- dense store at the lane's own cell.
    Invalid lanes skip pair_fwd (keeps its -1 pre-fill; destination unknowable),
    but store -1 into pair_bwd directly, so pair_bwd may be allocated
    UNINITIALIZED (one host fill pass saved). Replaces the torch nonzero + gather
    + 2x index_put_ back-half.
    """
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    off = tl.max_contiguous(tl.multiple_of(off, BLOCK), BLOCK)  # contiguity hint
    m = off < KV_N
    k = off // N
    j = off - k * N
    v = tl.load(valid_ptr + off, mask=m, other=0)
    orow = tl.load(out_row_ptr + off, mask=m, other=0)
    store_mask = m & (v != 0)
    # int64 pointer math (mirrors gemm.py); clamp invalid out_row to 0 so masked
    # lanes never form a wild address.
    orow_safe = tl.where(store_mask, orow, 0).to(tl.int64)
    pf_off = k.to(tl.int64) * n_out + orow_safe
    tl.store(pair_fwd_ptr + pf_off, j.to(tl.int32), mask=store_mask)
    if NEED_BWD:
        # pair_bwd[k, j] at flat off; write the -1 fill for invalid lanes here.
        bwd_val = tl.where(store_mask, orow, -1)
        tl.store(pair_bwd_ptr + off, bwd_val.to(tl.int32), mask=m)


def scatter_igemm_pairs(valid, out_row, pair_fwd, pair_bwd, need_bwd: bool):
    """Fill igemm gather tables pair_fwd [kv, n_out] and (when training)
    pair_bwd [kv, N] from valid [kv, N] bool and out_row [kv, N] int32.

    pair_fwd must be pre-filled -1 (only valid cells written); pair_bwd may be
    uninitialized (kernel writes every cell). Byte-identical to the nonzero +
    index_put_ scatter (see ``_scatter_igemm_kernel``). No-op on empty grid.
    """
    import torch  # local: keep the module import-light

    kv, N = valid.shape
    n_out = pair_fwd.shape[1]
    kv_n = kv * N
    if kv_n == 0:
        return
    valid_i8 = valid.contiguous().view(torch.int8)
    out_row = out_row.contiguous()
    pb = pair_bwd if need_bwd else pair_fwd  # dummy ptr when unused (NEED_BWD False)

    def grid(meta):
        return (triton.cdiv(kv_n, meta["BLOCK"]),)

    _scatter_igemm_kernel[grid](
        valid_i8,
        out_row,
        pair_fwd,
        pb,
        N,
        n_out,
        kv_n,
        NEED_BWD=need_bwd,
    )


@triton.jit
def _lower_bound(sorted_ptr, target, hi, steps):
    """Masked per-lane lower-bound binary search over ``sorted_ptr[0:hi]``.

    Lanes with hi==0 are pre-converged (result 0); callers zero hi to gate
    out-of-bounds/invalid lanes. ``steps`` is a RUNTIME bound (>= bit_length(max
    hi)) so a changing row count never recompiles; trimmed iterations are
    all-inactive no-ops (byte-identical to any larger bound). mid is in-bounds
    whenever active, so gathers are safe.
    """
    lo = tl.zeros_like(hi)
    for _ in range(steps):
        active = lo < hi
        mid = (lo + hi) // 2
        mid_safe = tl.where(active, mid, 0).to(tl.int64)
        sval = tl.load(sorted_ptr + mid_safe, mask=active, other=0)
        cond = active & (sval < target)
        lo = tl.where(cond, mid + 1, lo)
        hi = tl.where(active & ~cond, mid, hi)
    return lo


def _search_steps(n: int) -> int:
    """Iteration bound for ``_lower_bound``: [0, n) empties after bit_length(n)
    halvings."""
    return max(1, int(n).bit_length())


# Autotuned on the shared flat-grid config set. Empty key: every dim it sees (M, KV_N,
# steps) is data-dependent, so it is tuned once on first call and reused for all scenes
# -- never re-tuned per forward (the pair-gen variable-input invariant). Byte-identical
# for any BLOCK (masked lower-bound search).
@triton.autotune(configs=_candidate_configs(), key=[], **AUTOTUNE_CACHE_KW)
@triton.jit
def _lower_bound_rows_kernel(
    keys_ptr,  # [KV_N] lin_dtype (contiguous): candidate output linear coords
    pvalid_ptr,  # [KV_N] int8 (bool view): geometric validity; gates the search
    sorted_ptr,  # [M] lin_dtype: sorted unique output coords (torch.unique)
    out_row_ptr,  # [KV_N] int32 (output): lower-bound position (0 on invalid)
    M,
    KV_N,
    steps,  # see _lower_bound (runtime bound, >= bit_length(M))
    BLOCK: tl.constexpr,
):
    """Fused membership for the REGULAR-conv path: replaces the torch clamp +
    searchsorted back-half and the uniq[[0,-1]].tolist() device->host sync.

    Every VALID lane's value is a member of ``sorted`` (the unique of exactly
    those), so its lower bound == torch.searchsorted on the clamped input --
    byte-identical on every consumed entry. Invalid lanes skip the search (hi
    pre-gated to 0) and store 0 (masked out downstream)."""
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    off = tl.max_contiguous(tl.multiple_of(off, BLOCK), BLOCK)  # contiguity hint
    m = off < KV_N
    pv = tl.load(pvalid_ptr + off, mask=m, other=0)
    lane = m & (pv != 0)
    target = tl.load(keys_ptr + off, mask=lane, other=0)
    hi = tl.where(lane, M, 0).to(tl.int32)
    pos = _lower_bound(sorted_ptr, target, hi, steps)
    tl.store(out_row_ptr + off, pos, mask=m)


def lower_bound_rows(out_lin, valid, uniq):
    """Regular-conv membership: out_row[k, j] = lower-bound position of
    out_lin[k, j] in ``uniq`` for valid entries (0 for invalid).

    Returns out_row [kv, N] int32. Byte-identical to the searchsorted(uniq,
    out_lin.clamp(...)) chain on every valid entry, without the clamp pass or the
    host read of the bounds.
    """
    import torch  # local: keep the module import-light

    kv, N = out_lin.shape
    M = int(uniq.numel())
    out_row = torch.empty((kv, N), dtype=torch.int32, device=out_lin.device)
    kv_n = kv * N
    if kv_n == 0:
        return out_row
    ol = out_lin.contiguous()
    pv = valid.contiguous().view(torch.int8)

    def grid(meta):
        return (triton.cdiv(kv_n, meta["BLOCK"]),)

    _lower_bound_rows_kernel[grid](
        ol,
        pv,
        uniq,
        out_row,
        M,
        kv_n,
        _search_steps(M),
    )
    return out_row


# Autotuned on the shared flat-grid config set; empty key for the same reason as
# _lower_bound_rows_kernel (N/KV_N/steps are all data-dependent -> tune once, never
# re-tune per scene). Byte-identical for any BLOCK (masked membership search).
@triton.autotune(configs=_candidate_configs(), key=[], **AUTOTUNE_CACHE_KW)
@triton.jit
def _subm_membership_kernel(
    out_lin_ptr,  # [kv, N] lin_dtype (contiguous); candidate output linear coord
    sorted_lin_ptr,  # [N] lin_dtype; sorted input linear coords
    sort_idx_ptr,  # [N] int64; permutation from torch.sort(in_lin)
    pvalid_ptr,  # [kv, N] int8 (bool view); prior (geometric) validity
    out_row_ptr,  # [kv, N] int32 (output); input row matching the output coord
    valid_ptr,  # [kv, N] int8 (output); prior_valid AND found-in-input
    N,
    KV_N,
    steps,  # see _lower_bound (runtime bound, >= bit_length(N))
    BLOCK: tl.constexpr,
):
    """Fused subm membership test (replaces the searchsorted + clamp + gather +
    eq + lt + and chain).

    Flat grid ``off == k*N + j``. Lower-bound search out_lin[k, j] in the sorted
    input coords; ``found`` iff it lands on an exact in-range match. out_row =
    sort_idx[pos] (original input row, don't-care if not found); validity =
    prior_valid AND found. Byte-identical to the torch chain on valid entries.
    """
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    off = tl.max_contiguous(tl.multiple_of(off, BLOCK), BLOCK)  # contiguity hint
    m = off < KV_N
    target = tl.load(out_lin_ptr + off, mask=m, other=0)
    pv = tl.load(pvalid_ptr + off, mask=m, other=0)

    # OOB lanes pre-converge via hi==0 (target loads as 0) -- matches the old
    # in-loop `& m` masking on every stored (m-gated) lane.
    hi = tl.where(m, N, 0).to(tl.int32)
    pos = _lower_bound(sorted_lin_ptr, target, hi, steps)  # lower_bound in [0, N]

    # found = pos < N && sorted_lin[pos] == target (pos==N gated off by pos<N).
    in_range = m & (pos < N)
    pos_in = tl.where(in_range, pos, 0).to(tl.int64)
    sp = tl.load(sorted_lin_ptr + pos_in, mask=in_range, other=0)
    found = in_range & (sp == target)

    # out_row = sort_idx[min(pos, N-1)] (clamp matches torch; N>=1 for subm).
    pos_c = tl.where(pos < N, pos, N - 1)
    pos_c_safe = tl.where(m, pos_c, 0).to(tl.int64)
    orow = tl.load(sort_idx_ptr + pos_c_safe, mask=m, other=0)

    out_valid = (pv != 0) & found
    # out_row is a row index (< N), int32-safe even when sort_idx is int64.
    tl.store(out_row_ptr + off, orow.to(tl.int32), mask=m)
    tl.store(valid_ptr + off, out_valid.to(tl.int8), mask=m)


def subm_membership(out_lin, sorted_lin, sort_idx, prior_valid):
    """Subm membership: map each candidate output coord to its input row (fused
    binary search into the sorted input coords) and AND prior validity with found.

    Returns (out_row [kv, N] int32, valid [kv, N] bool). Caller keeps the
    data-dependent torch.sort. Byte-identical to the torch searchsorted + clamp +
    eq + lt + and back-half.
    """
    import torch  # local: keep the module import-light

    kv, N = out_lin.shape
    out_row = torch.empty((kv, N), dtype=torch.int32, device=out_lin.device)
    valid = torch.empty((kv, N), dtype=torch.int8, device=out_lin.device)
    kv_n = kv * N
    if kv_n == 0:
        return out_row, valid.view(torch.bool)
    ol = out_lin.contiguous()
    pv = prior_valid.contiguous().view(torch.int8)

    def grid(meta):
        return (triton.cdiv(kv_n, meta["BLOCK"]),)

    _subm_membership_kernel[grid](
        ol,
        sorted_lin,
        sort_idx,
        pv,
        out_row,
        valid,
        N,
        kv_n,
        _search_steps(N),
    )
    return out_row, valid.view(torch.bool)
