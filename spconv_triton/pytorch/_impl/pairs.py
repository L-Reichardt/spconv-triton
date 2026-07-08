# SPDX-FileCopyrightText: Copyright contributors to the spconv-Triton project
# SPDX-FileCopyrightText: Derived from spconv, Copyright 2021 Yan Yan and contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Index-pair generation for sparse convolution (GPU, vendor-agnostic).

Reproduces spconv's pair semantics exactly (verified against reference dumps).
Kernel offsets are enumerated row-major (last dim fastest), NO kernel flip for
transpose/inverse. Output-coord formulas:
- regular conv:  o = (x + padding - k_pos * dilation) / stride   (exact div)
- transpose:     o = x * stride - padding + k_pos * dilation
- subm:          o = x + (ksize//2) * dilation - k_pos * dilation
  (stride/padding ignored; odd kernels required; out coords must exist in input)

Native format [2, kv, N] int32, -1 fill: pair[0,k,:n_k]=input rows,
pair[1,k,:n_k]=output rows. subm: indice_num_per_loc[k]=count for k<center else
0, but pair CONTENT is filled for all offsets (center = full identity).

Implicit-gemm format: pair_fwd[k, out_row]=input row (-1 fill),
pair_bwd[k, in_row]=out row; indice_num_per_loc=zeros (reference direct-table
path); pair_mask returned SORTED with mask_argsort[i]=original row.
"""

import functools

import numpy as np
import torch

from spconv_triton.pytorch._impl import pair_kernels

_POINT_VANISH_MSG = (
    "Your points vanished here, this usually happens when you provide "
    "conv params that may ignore some input points. Example: "
    "spatial_shape={}, ksize={}, stride={}, padding={}, dilation={}"
)


def _prod(vals):
    return functools.reduce(lambda a, b: a * b, vals, 1)


def _kernel_offsets(ksize: list[int], device) -> torch.Tensor:
    grids = torch.meshgrid(
        *[torch.arange(k, device=device, dtype=torch.int64) for k in ksize],
        indexing="ij",
    )
    return torch.stack([g.reshape(-1) for g in grids], 1)  # [kv, ndim]


@functools.cache
def _kernel_offsets_cached(ksize: tuple[int, ...], device, dtype) -> torch.Tensor:
    """Cached kernel-offset grid (layer-fixed, keyed by ksize/device/dtype).
    Key set is bounded by distinct layer configs, not scene count. Callers must
    NOT mutate the returned tensor."""
    return _kernel_offsets(list(ksize), device).to(dtype)


def get_conv_output_size(input_size, kernel_size, stride, padding, dilation):
    ndim = len(input_size)
    output_size = []
    for i in range(ndim):
        size = (
            input_size[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1
        ) // stride[i] + 1
        if kernel_size[i] == -1:
            output_size.append(1)
        else:
            output_size.append(size)
    return output_size


def get_deconv_output_size(
    input_size, kernel_size, stride, padding, dilation, output_padding
):
    ndim = len(input_size)
    output_size = []
    for i in range(ndim):
        if kernel_size[i] == -1:
            raise ValueError("deconv don't support kernel_size < 0")
        size = (
            (input_size[i] - 1) * stride[i]
            - 2 * padding[i]
            + kernel_size[i]
            + output_padding[i]
        )
        output_size.append(size)
    return output_size


def _linearize(b: torch.Tensor, coords: torch.Tensor, shape: list[int]) -> torch.Tensor:
    lin = b
    for i, s in enumerate(shape):
        lin = lin * int(s) + coords[..., i]
    return lin


def _delinearize(
    lin: torch.Tensor, shape: list[int], device, dtype=torch.int64
) -> torch.Tensor:
    # Out coords + residual batch are bounded by shape/batch, so int32-safe;
    # allocating in dtype avoids an int64->int32 copy. div/mod run on lin (may
    # be int64 in large-grid regime) and narrow on store.
    ndim = len(shape)
    out = torch.empty((lin.numel(), ndim + 1), dtype=dtype, device=device)
    rem = lin
    for i in range(ndim - 1, -1, -1):
        out[:, i + 1] = rem % shape[i]
        rem = rem // shape[i]
    out[:, 0] = rem
    return out


class Candidates:
    """Per (kernel offset, input point) candidate output rows."""

    def __init__(self, valid, out_row, out_inds, out_shape, n_out, kv):
        self.valid = valid  # [kv, N] bool
        self.out_row = out_row  # [kv, N] int32 (only valid entries used)
        self.out_inds = out_inds  # [n_out, ndim+1] int32 (or input indices for subm)
        self.out_shape = out_shape
        self.n_out = n_out
        self.kv = kv


def compute_candidates(
    indices: torch.Tensor,
    batch_size: int,
    spatial_shape: list[int],
    ksize: list[int],
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    out_padding: list[int],
    subm: bool,
    transpose: bool,
) -> Candidates:
    device = indices.device
    ndim = len(spatial_shape)
    N = indices.shape[0]
    kv = _prod(ksize)
    if N == 0:
        # spconv parity: pair-gen kernels assert N > 0
        raise ValueError("N > 0 assert: sparse tensor has no points")
    if any(k <= 0 for k in ksize):
        raise ValueError(f"kernel size {ksize} not supported by pair gen")

    if subm:
        if any(k % 2 == 0 for k in ksize):
            raise NotImplementedError("subm convolution requires odd kernel sizes")
        out_shape = list(spatial_shape)
    elif transpose:
        out_shape = get_deconv_output_size(
            spatial_shape, ksize, stride, padding, dilation, out_padding
        )
    else:
        out_shape = get_conv_output_size(
            spatial_shape, ksize, stride, padding, dilation
        )
    if any(x <= 0 for x in out_shape):
        raise ValueError(
            f"your out spatial shape {out_shape} reach zero!!! "
            f"input shape: {spatial_shape}"
        )

    # int32 when batch*volume < 2**31: halves memory traffic of the O(kv*N)
    # build and is byte-identical to int64 (valid entries never overflow).
    max_lin = batch_size * _prod(out_shape)
    if subm:
        max_lin = max(max_lin, batch_size * _prod(spatial_shape))
    lin_dtype = torch.int32 if max_lin < 2**31 else torch.int64

    offs = _kernel_offsets_cached(tuple(ksize), device, lin_dtype)  # [kv, ndim]
    # Contiguous [N, ndim+1]: fused kernel reads col 0 as batch, cols 1: as
    # coords; coords/b are zero-copy views reused by the subm path.
    idx_lin = indices.to(lin_dtype)
    if not idx_lin.is_contiguous():
        idx_lin = idx_lin.contiguous()
    coords = idx_lin[:, 1:]  # [N, ndim]
    b = idx_lin[:, 0]  # [N]

    if subm:
        pad = [(ksize[i] // 2) * dilation[i] for i in range(ndim)]
        st = [1] * ndim
    else:
        pad = list(padding)
        st = list(stride)

    # Per-(offset, point) out coord + geometric validity, fused into one Triton
    # kernel. Byte-identical for every VALID entry (validity gates o>=0 so the
    # kernel's integer div agrees with torch floor div on survivors).
    out_lin, valid = pair_kernels.candidate_out_lin_valid(
        idx_lin, offs, pad, st, list(dilation), out_shape, kv, transpose
    )

    if subm:
        in_lin = _linearize(b, coords, spatial_shape)
        sorted_lin, sort_idx = torch.sort(in_lin)
        # Fused membership: binary-search each candidate out coord in sorted
        # input coords -> out row + found-in-input, ANDed with geometric validity.
        out_row, valid = pair_kernels.subm_membership(
            out_lin, sorted_lin, sort_idx, valid
        )
        return Candidates(valid, out_row, indices, out_shape, N, kv)

    lin_valid = out_lin[valid]
    uniq = torch.unique(lin_valid)
    del lin_valid
    n_out = int(uniq.numel())
    if n_out == 0:
        raise ValueError(
            _POINT_VANISH_MSG.format(spatial_shape, ksize, stride, padding, dilation)
        )
    # Fused lower-bound membership: positions index into uniq (< n_out),
    # int32-safe even when uniq is int64. Valid entries are members of uniq by
    # construction (invalid lanes' out_row is a masked-off don't-care).
    out_row = pair_kernels.lower_bound_rows(out_lin, valid, uniq)
    del out_lin
    out_inds = _delinearize(uniq, out_shape, device, dtype=torch.int32)
    return Candidates(valid, out_row, out_inds, out_shape, n_out, kv)


def _scatter_pairs_native(
    cand: Candidates, N: int, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the Native [2, kv, N] pair tensor (-1 fill) and per-loc counts."""
    kv = cand.kv
    pair = torch.full((2, kv, N), -1, dtype=torch.int32, device=device)
    valid = cand.valid
    counts = valid.sum(1)
    pos = torch.cumsum(valid, dim=1, dtype=torch.int32) - 1
    k_idx = torch.arange(kv, device=device)[:, None].expand(kv, N)
    j_idx = torch.arange(N, device=device)[None, :].expand(kv, N)
    kk = k_idx[valid]
    pp = pos[valid].long()
    del pos
    pair[0].index_put_((kk, pp), j_idx[valid].to(torch.int32))
    pair[1].index_put_((kk, pp), cand.out_row[valid].to(torch.int32))
    return pair, counts.to(torch.int32)


def native_pairs(
    indices,
    batch_size,
    spatial_shape,
    ksize,
    stride,
    padding,
    dilation,
    out_padding,
    subm,
    transpose,
):
    device = indices.device
    N = indices.shape[0]
    cand = compute_candidates(
        indices,
        batch_size,
        spatial_shape,
        ksize,
        stride,
        padding,
        dilation,
        out_padding,
        subm,
        transpose,
    )
    pair, counts = _scatter_pairs_native(cand, N, device)
    if subm:
        kv = cand.kv
        center = kv // 2
        k_range = torch.arange(kv, device=device)
        npl = torch.where(k_range < center, counts, torch.zeros_like(counts)).to(
            torch.int32
        )
    else:
        npl = counts.to(torch.int32)
    return cand.out_inds, pair, npl


def igemm_pairs(
    indices,
    batch_size,
    spatial_shape,
    ksize,
    stride,
    padding,
    dilation,
    out_padding,
    subm,
    transpose,
    is_train,
    is_mask_split,
    do_sort=True,
    num_out_act_bound=-1,
):
    device = indices.device
    N = indices.shape[0]
    cand = compute_candidates(
        indices,
        batch_size,
        spatial_shape,
        ksize,
        stride,
        padding,
        dilation,
        out_padding,
        subm,
        transpose,
    )
    kv = cand.kv
    # spconv parity: MaskSplitImplicitGemm has no kernels for kv > 32 (asserts
    # mask_int_count == 1); we raise instead of emitting truncated split masks.
    if is_mask_split and kv > 32:
        raise AssertionError("Not Implemented")
    n_out = cand.n_out
    valid = cand.valid
    if not subm and num_out_act_bound > 0 and n_out > num_out_act_bound:
        # spconv parity: only igemm enforces the bound; survivors are impl-defined
        # (we keep the first in canonical order). Dropped-output candidates -> invalid.
        n_out = num_out_act_bound
        cand.out_inds = cand.out_inds[:n_out]
        valid = valid & (cand.out_row < n_out)

    need_pair_bwd = is_train  # backward gather table only consumed when training
    pair_fwd = torch.full((kv, n_out), -1, dtype=torch.int32, device=device)
    # pair_bwd needs no -1 pre-fill: scatter kernel writes every [kv, N] cell.
    pair_bwd = (
        torch.empty((kv, N), dtype=torch.int32, device=device)
        if need_pair_bwd
        else torch.Tensor()
    )
    if N > 0:
        # Fused scatter: per VALID (k, j) write pair_fwd[k, out_row[k,j]]=j and
        # (train) pair_bwd[k, j]=out_row. Forward is collision-free: for fixed k,
        # j -> out_row is injective on the valid set (candidate fold is a bijection
        # while coords stay in [0, out_shape)).
        pair_kernels.scatter_igemm_pairs(
            valid, cand.out_row, pair_fwd, pair_bwd, need_pair_bwd
        )

    npl = torch.zeros(kv, dtype=torch.int32, device=device)

    if is_mask_split:
        kv_div_2 = kv // 2
        remain = kv - kv_div_2
        mask_np_1 = np.array([1], dtype=np.uint64)
        first = (mask_np_1 << remain) - 1
        second = ((mask_np_1 << kv_div_2) - 1) << remain
        masks = [first.astype(np.uint32), second.astype(np.uint32)]
    else:
        masks = [np.array([0xFFFFFFFF], dtype=np.uint32)]

    mask_int_count = (kv + 31) // 32

    def make_mask_splits(pair_table):
        """Per-row offset bitmasks [n, mask_int_count] int32 (bit k in word
        k//32), returned sorted with mask_argsort[i] = original row of the
        i-th sorted entry."""
        n = pair_table.shape[1]
        # Fused bitmask build. Within a 32-bit word each offset k owns a distinct
        # bit (k % 32), so the kernel's OR reproduces the additive index_add
        # reduction byte-for-byte (verified across kv in {1..125}).
        words = pair_kernels.pack_mask_words(pair_table, kv, mask_int_count)
        pm_splits, args_splits = [], []
        for m in masks:
            mval = int(m.astype(np.int64)[0])
            # Single-mask path consumes words once (mutate in place); 2-mask split
            # needs a fresh copy per m.
            mwords = words if len(masks) == 1 else words.clone()
            mwords[:, 0] &= mval  # split masks only defined for kv <= 32
            if do_sort:
                if mask_int_count == 1:
                    # Single-word masks (kv <= 32, production regime): stable sort
                    # returns values + permutation in one pass.
                    vals_col, args = torch.sort(mwords[:, 0], stable=True)
                    vals = vals_col.unsqueeze(1)
                else:
                    # lexicographic LSD radix: least-significant word first,
                    # stability carries order (high word most significant)
                    args = torch.arange(n, device=device)
                    for w in range(mask_int_count):
                        args = args[torch.argsort(mwords[args, w], stable=True)]
                    vals = mwords[args]
            else:
                vals = mwords
                args = torch.arange(n, device=device)
            pm = vals & 0xFFFFFFFF
            pm = pm.where(pm < 2**31, pm - 2**32).to(torch.int32)
            pm_splits.append(pm.reshape(n, mask_int_count))
            args_splits.append(args.to(torch.int32))
        return pm_splits, args_splits

    pm_fwd_splits, ma_fwd_splits = make_mask_splits(pair_fwd)
    if is_train and not subm:
        pm_bwd_splits, ma_bwd_splits = make_mask_splits(pair_bwd)
    else:
        pm_bwd_splits, ma_bwd_splits = [], []

    if subm:
        out_inds = indices
        pb = pair_bwd if is_train else torch.Tensor()
        return (
            out_inds,
            npl,
            pair_fwd,
            pb,
            pm_fwd_splits,
            [],
            ma_fwd_splits,
            [],
            masks,
        )
    else:
        pb = pair_bwd if is_train else torch.Tensor()
        return (
            cand.out_inds,
            npl,
            pair_fwd,
            pb,
            pm_fwd_splits,
            pm_bwd_splits,
            ma_fwd_splits,
            ma_bwd_splits,
            masks,
        )


def native_pair_to_tables(
    pair: torch.Tensor,
    indice_pair_num: torch.Tensor,
    n_in: int,
    n_out: int,
    inverse: bool,
    need_bwd: bool = True,
):
    """Convert a Native [2, kv, N] pair tensor into igemm gather tables
    (pair_fwd [kv, n_out], pair_bwd [kv, n_in]); pair_bwd is None when
    need_bwd is False. Memoized on the pair tensor, keyed by (n_in, n_out, inverse).

    Validity is content-based (-1 fill), transparently covering the subm layout.
    indice_pair_num: accepted for spconv call-shape parity but ignored.
    Each table built at most once; compaction re-runs only when a later call
    needs a table an earlier one skipped (e.g. pair_bwd after an eval forward).
    """
    key = (int(n_in), int(n_out), bool(inverse))
    cache = getattr(pair, "_spconv_triton_tables", None)
    if cache is None:
        cache = {}
        pair._spconv_triton_tables = cache  # type: ignore[attr-defined]
    pf, pb = cache.get(key, (None, None))
    build_fwd = pf is None
    build_bwd = need_bwd and pb is None
    if build_fwd or build_bwd:
        device = pair.device
        pair_in = pair[int(inverse)]
        pair_out = pair[int(not inverse)]
        kv = pair_in.shape[0]
        valid = (pair_in >= 0) & (pair_out >= 0)
        k_idx = torch.arange(kv, device=device)[:, None].expand_as(pair_in)
        kk = k_idx[valid]
        if build_fwd:
            pf = torch.full((kv, n_out), -1, dtype=torch.int32, device=device)
            pf.index_put_((kk, pair_out[valid].long()), pair_in[valid].int())
        if build_bwd:
            pb = torch.full((kv, n_in), -1, dtype=torch.int32, device=device)
            pb.index_put_((kk, pair_in[valid].long()), pair_out[valid].int())
        cache[key] = (pf, pb)
    return pf, pb
