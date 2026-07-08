# SPDX-FileCopyrightText: Copyright contributors to the spconv-Triton project
# SPDX-FileCopyrightText: Derived from spconv, Copyright 2021 Yan Yan and contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Primitive sparse-convolution ops (API-compatible with spconv.pytorch.ops).

Pair generation is vectorized torch (device-agnostic); the heavy compute
(gather-GEMM-scatter, pooling) runs in Triton kernels.
"""

import numpy as np
import torch

from spconv_triton.compat import tv
from spconv_triton.constants import (  # noqa: F401 (unused names are parity re-exports)
    ALL_WEIGHT_IS_KRSC,
    FILTER_HWIO,
    SPCONV_DO_SORT,
    SPCONV_USE_DIRECT_TABLE,
    AllocKeys,
)
from spconv_triton.core import AlgoHint, ConvAlgo  # noqa: F401 (parity)
from spconv_triton.pytorch._impl import gemm as _gemm
from spconv_triton.pytorch._impl import pairs as _pairs
from spconv_triton.pytorch._impl import pool as _pool
from spconv_triton.pytorch._impl.pairs import (
    get_conv_output_size,
    get_deconv_output_size,
)
from spconv_triton.pytorch.core import ThrustSortAllocator
from spconv_triton.pytorch.cppcore import get_current_stream  # noqa: F401
from spconv_triton.tools import CPU_ONLY_BUILD, CUDAKernelTimer  # noqa: F401
from spconv_triton.utils import nullcontext  # noqa: F401 (parity re-export)

INT32_MAX = 2147483647
# Public API surface mirrored from spconv.pytorch.ops (asserted by
# tests/test_ext_misc.py); never read by the port.
DEBUG = False
DEBUG_INT64_HASH_K = False

__all__ = [
    "ConvAlgo",
    "get_conv_output_size",
    "get_deconv_output_size",
    "get_indice_pairs",
    "get_indice_pairs_implicit_gemm",
    "global_pool_rearrange",
    "implicit_gemm",
    "implicit_gemm_backward",
    "indice_avgpool_implicit_gemm",
    "indice_avgpool_implicit_gemm_backward",
    "indice_conv",
    "indice_conv_backward",
    "indice_maxpool",
    "indice_maxpool_backward",
    "indice_maxpool_implicit_gemm",
    "indice_maxpool_implicit_gemm_backward",
    "maximum_value_int_",
]


def _apply_act_inplace(out: torch.Tensor, act_type, act_alpha: float, act_beta: float):
    name = getattr(act_type, "name", "None_")
    if name == "None_":
        return out
    if name == "ReLU":
        return torch.relu_(out)
    if name == "Sigmoid":
        return torch.sigmoid_(out)
    if name == "LeakyReLU":
        return torch.nn.functional.leaky_relu_(out, act_alpha)
    raise NotImplementedError(f"activation {name} not supported")


def _filters_krsc(filters: torch.Tensor):
    """Reshape a KRSC weight [K, *ksize, C] to [K, kv, C] (contiguous)."""
    out_channels = filters.shape[0]
    in_channels = filters.shape[-1]
    f = filters.reshape(out_channels, -1, in_channels)
    if not f.is_contiguous():
        f = f.contiguous()
    return f


def get_indice_pairs(
    indices: torch.Tensor,
    batch_size: int,
    spatial_shape: list[int],
    algo: ConvAlgo,
    ksize: list[int],
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    out_padding: list[int],
    subm: bool = False,
    transpose: bool = False,
    num_out_act_bound: int = -1,
):
    if algo != ConvAlgo.Native:
        raise AssertionError("TODO")
    return _pairs.native_pairs(
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


def get_indice_pairs_implicit_gemm(
    indices: torch.Tensor,
    batch_size: int,
    spatial_shape: list[int],
    algo: ConvAlgo,
    ksize: list[int],
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    out_padding: list[int],
    subm: bool = False,
    transpose: bool = False,
    is_train: bool = True,
    alloc: ThrustSortAllocator | None = None,
    timer: CUDAKernelTimer = CUDAKernelTimer(False),
    num_out_act_bound: int = -1,
    direct_table: bool = SPCONV_USE_DIRECT_TABLE,
    do_sort: bool = SPCONV_DO_SORT,
):
    if not indices.is_cuda:
        raise AssertionError("implicit gemm only support gpu tensors")
    if algo not in (ConvAlgo.MaskImplicitGemm, ConvAlgo.MaskSplitImplicitGemm):
        raise AssertionError("TODO")
    is_mask_split = algo == ConvAlgo.MaskSplitImplicitGemm
    return _pairs.igemm_pairs(
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
        do_sort,
        num_out_act_bound,
    )


# ---------------------------------------------------------------------------
# convolution
# ---------------------------------------------------------------------------


def indice_conv(
    features: torch.Tensor,
    filters: torch.Tensor,
    indice_pairs: torch.Tensor,
    indice_pair_num: torch.Tensor,
    num_activate_out: int,
    inverse: bool = False,
    subm: bool = False,
    algo: ConvAlgo = ConvAlgo.Native,
    timer: CUDAKernelTimer = CUDAKernelTimer(False),
    bias: torch.Tensor | None = None,
    act_alpha: float = 0.0,
    act_beta: float = 0.0,
    act_type=tv.gemm.Activation.None_,
):
    if not features.is_contiguous():
        features = features.contiguous()
    if features.dtype in (torch.int8, torch.qint8):
        raise NotImplementedError("int8 is not supported by spconv_triton")
    w = _filters_krsc(filters)
    n_in = features.shape[0]
    pf, _ = _pairs.native_pair_to_tables(
        indice_pairs,
        indice_pair_num,
        n_in,
        num_activate_out,
        inverse,
        need_bwd=False,
    )
    out = _gemm.conv_forward(features, w, pf, num_activate_out, bias=bias)
    _apply_act_inplace(out, act_type, act_alpha, act_beta)
    return out


def fused_indice_conv(
    features,
    filters,
    bias,
    indice_pairs,
    indice_pair_num,
    num_activate_out,
    inverse,
    subm,
):
    raise NotImplementedError


def indice_conv_backward(
    features: torch.Tensor,
    filters: torch.Tensor,
    out_bp: torch.Tensor,
    indice_pairs: torch.Tensor,
    indice_pair_num: torch.Tensor,
    inverse: bool = False,
    subm: bool = False,
    algo: ConvAlgo = ConvAlgo.Native,
    timer: CUDAKernelTimer = CUDAKernelTimer(False),
):
    if not features.is_contiguous():
        features = features.contiguous()
    if not out_bp.is_contiguous():
        out_bp = out_bp.contiguous()
    filters_shape = filters.shape
    w = _filters_krsc(filters)
    n_in = features.shape[0]
    n_out = out_bp.shape[0]
    pf, pb = _pairs.native_pair_to_tables(
        indice_pairs, indice_pair_num, n_in, n_out, inverse
    )
    din = _gemm.conv_backward_input(out_bp, w, pb, n_in)
    dfilters = _gemm.conv_backward_weight(features, out_bp, pf, w.shape)
    return (din, dfilters.reshape(filters_shape))


def _single_split_mask(
    pair_mask_splits: list[torch.Tensor],
    mask_argsort_splits: list[torch.Tensor],
    masks: list[np.ndarray],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Extract 1-D (pmask, argsort) for the masked implicit-GEMM path.

    Returns ``(None, None)`` when masking doesn't apply (GEMM falls back to dense):
    MaskSplitImplicitGemm's two complementary splits aren't a full active-offset
    bitmask, and subm backward splits are empty. pair_mask words are stored
    [n, mask_int_count]; for the single-split kv<=32 case count==1 so column 0 is
    the full per-row active-offset bitmask (kernel re-checks kv)."""
    if len(masks) != 1 or len(pair_mask_splits) != 1 or len(mask_argsort_splits) != 1:
        return None, None
    pm = pair_mask_splits[0]
    ar = mask_argsort_splits[0]
    if pm.numel() == 0 or ar.numel() == 0:
        return None, None
    pm1 = pm[:, 0].contiguous() if pm.dim() == 2 else pm.contiguous()
    return pm1, ar.contiguous()


def implicit_gemm(
    features: torch.Tensor,
    filters: torch.Tensor,
    pair_fwd: torch.Tensor,
    pair_mask_fwd_splits: list[torch.Tensor],
    mask_argsort_fwd_splits: list[torch.Tensor],
    num_activate_out: int,
    masks: list[np.ndarray],
    is_train: bool,
    is_subm: bool,
    timer: CUDAKernelTimer = CUDAKernelTimer(False),
    fp32_accum: bool | None = None,
    bias: torch.Tensor | None = None,
    act_alpha: float = 0.0,
    act_beta: float = 0.0,
    act_type=tv.gemm.Activation.None_,
    output_scale: float = 1.0,
    scale: torch.Tensor | None = None,
    output_add: torch.Tensor | None = None,
    output_add_scale: float = 0.0,
    output_dtype: torch.dtype | None = None,
):
    if output_add is not None and features.dtype != torch.qint8:
        raise AssertionError("fused residual add only support int8")
    if features.dtype in (torch.int8, torch.qint8):
        raise NotImplementedError("int8 is not supported by spconv_triton")
    if not features.is_contiguous():
        features = features.contiguous()
    w = _filters_krsc(filters)
    pm_fwd, ar_fwd = _single_split_mask(
        pair_mask_fwd_splits, mask_argsort_fwd_splits, masks
    )
    out = _gemm.conv_forward(
        features,
        w,
        pair_fwd,
        num_activate_out,
        bias=bias,
        out_dtype=output_dtype,
        pair_mask=pm_fwd,
        mask_argsort=ar_fwd,
        fp32_accum=fp32_accum,
    )
    _apply_act_inplace(out, act_type, act_alpha, act_beta)
    mask_output_fwd = torch.Tensor()
    mask_width = -1
    return out, mask_output_fwd, mask_width


def implicit_gemm_backward(
    features: torch.Tensor,
    filters: torch.Tensor,
    grad_output: torch.Tensor,
    pair_fwd: torch.Tensor,
    pair_bwd: torch.Tensor,
    pair_mask_fwd_splits: list[torch.Tensor],
    pair_mask_bwd_splits: list[torch.Tensor],
    mask_argsort_fwd_splits: list[torch.Tensor],
    mask_argsort_bwd_splits: list[torch.Tensor],
    mask_output_fwd: torch.Tensor,
    masks: list[np.ndarray],
    mask_width: int,
    is_subm: bool,
    timer: CUDAKernelTimer = CUDAKernelTimer(False),
    fp32_accum: bool | None = None,
):
    if not features.is_contiguous():
        features = features.contiguous()
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()
    filters_shape = filters.shape
    w = _filters_krsc(filters)
    n_in = features.shape[0]
    pm_bwd, ar_bwd = _single_split_mask(
        pair_mask_bwd_splits, mask_argsort_bwd_splits, masks
    )
    din = _gemm.conv_backward_input(
        grad_output,
        w,
        pair_bwd,
        n_in,
        pair_mask=pm_bwd,
        mask_argsort=ar_bwd,
        fp32_accum=fp32_accum,
    )
    dfilters = _gemm.conv_backward_weight(
        features, grad_output, pair_fwd, w.shape, fp32_accum=fp32_accum
    )
    return (din, dfilters.reshape(filters_shape))


# ---------------------------------------------------------------------------
# pooling
# ---------------------------------------------------------------------------


def indice_maxpool(
    features: torch.Tensor,
    indice_pairs: torch.Tensor,
    indice_pair_num: torch.Tensor,
    num_activate_out,
):
    if not features.is_contiguous():
        features = features.contiguous()
    pf, _ = _pairs.native_pair_to_tables(
        indice_pairs,
        indice_pair_num,
        features.shape[0],
        int(num_activate_out),
        False,
        need_bwd=False,
    )
    return _pool.maxpool_forward(features, pf, int(num_activate_out))


def indice_maxpool_backward(
    features, out_features, out_bp, indice_pairs, indice_pair_num
):
    _, pb = _pairs.native_pair_to_tables(
        indice_pairs,
        indice_pair_num,
        features.shape[0],
        out_features.shape[0],
        False,
    )
    return _pool.maxpool_backward(features, out_features, out_bp, pb)


def indice_maxpool_implicit_gemm(
    features: torch.Tensor, indice_pairs: torch.Tensor, num_activate_out
):
    if not features.is_cuda:
        raise AssertionError
    if not features.is_contiguous():
        features = features.contiguous()
    return _pool.maxpool_forward(features, indice_pairs, int(num_activate_out))


def indice_maxpool_implicit_gemm_backward(features, out_features, out_bp, indice_pairs):
    if not features.is_cuda:
        raise AssertionError
    return _pool.maxpool_backward(features, out_features, out_bp, indice_pairs)


def indice_avgpool_implicit_gemm(
    features: torch.Tensor,
    indice_pairs: torch.Tensor,
    num_activate_out,
    calc_count: bool,
):
    if not features.is_cuda:
        raise AssertionError
    if not features.is_contiguous():
        features = features.contiguous()
    out, count = _pool.avgpool_forward(
        features, indice_pairs, int(num_activate_out), calc_count
    )
    if not calc_count:
        count = torch.Tensor()
    return out, count


def indice_avgpool_implicit_gemm_backward(out_bp, indice_pairs, count_out):
    if not out_bp.is_cuda:
        raise AssertionError
    return _pool.avgpool_backward(out_bp, indice_pairs, count_out)


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------


def maximum_value_int_(ten: torch.Tensor, value: int):
    ten.clamp_(min=int(value))


def global_pool_rearrange(coords: torch.Tensor, batch_size: int):
    n = coords.shape[0]
    device = coords.device
    out_indices = torch.empty((batch_size, n), dtype=torch.int32, device=device)
    counts = torch.zeros((batch_size,), dtype=torch.int32, device=device)
    if n == 0:
        return out_indices, counts
    b = coords[:, 0].long()
    counts = torch.bincount(b, minlength=batch_size).to(torch.int32)
    order = torch.argsort(b, stable=True)
    starts = torch.zeros(batch_size, dtype=torch.int64, device=device)
    starts[1:] = torch.cumsum(counts.long(), 0)[:-1]
    rank = torch.arange(n, device=device) - starts[b[order]]
    out_indices.index_put_((b[order], rank), order.to(torch.int32))
    return out_indices, counts
