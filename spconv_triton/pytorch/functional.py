# SPDX-FileCopyrightText: Copyright contributors to the spconv-Triton project
# SPDX-FileCopyrightText: Derived from spconv, Copyright 2021 Yan Yan and contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Autograd functions and functional helpers (mirrors
spconv.pytorch.functional)."""

from functools import reduce
from typing import TypeVar

import numpy as np
import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from spconv_triton.compat import tv
from spconv_triton.pytorch import ops
from spconv_triton.pytorch.constants import PYTORCH_VERSION
from spconv_triton.pytorch.core import SparseConvTensor
from spconv_triton.pytorch.hash import HashTable
from spconv_triton.tools import CUDAKernelTimer

_MAX_INT32 = 2147483647


_T = TypeVar("_T")


def identity_decorator(func: _T) -> _T:
    return func


if PYTORCH_VERSION >= [2, 5, 0]:
    import torch.amp as amp

    _TORCH_CUSTOM_FWD = amp.custom_fwd(cast_inputs=torch.float16, device_type="cuda")
    _TORCH_CUSTOM_BWD = amp.custom_bwd(device_type="cuda")
elif PYTORCH_VERSION >= [1, 6, 0]:
    import torch.cuda.amp as amp  # type: ignore[no-redef]

    # Legacy (<2.5) amp.custom_fwd has no device_type kwarg; mypy sees only the
    # ≥2.5 signature from torch's stubs.
    _TORCH_CUSTOM_FWD = amp.custom_fwd(cast_inputs=torch.float16)  # type: ignore[call-arg]
    _TORCH_CUSTOM_BWD = amp.custom_bwd
else:
    _TORCH_CUSTOM_FWD = identity_decorator
    _TORCH_CUSTOM_BWD = identity_decorator


class SparseConvFunction(Function):
    @staticmethod
    @_TORCH_CUSTOM_FWD
    def forward(
        ctx,
        features,
        filters,
        indice_pairs,
        indice_pair_num,
        num_activate_out,
        algo,
        timer: CUDAKernelTimer = CUDAKernelTimer(False),
        bias: torch.Tensor | None = None,
        act_alpha: float = 0.0,
        act_beta: float = 0.0,
        act_type=tv.gemm.Activation.None_,
    ):
        ctx.save_for_backward(indice_pairs, indice_pair_num, features, filters)
        ctx.algo = algo
        ctx.timer = timer
        return ops.indice_conv(
            features,
            filters,
            indice_pairs,
            indice_pair_num,
            num_activate_out,
            False,
            algo=algo,
            timer=timer,
            bias=bias,
            act_alpha=act_alpha,
            act_beta=act_beta,
            act_type=act_type,
        )

    @staticmethod
    @once_differentiable
    @_TORCH_CUSTOM_BWD
    def backward(ctx, grad_output):
        indice_pairs, indice_pair_num, features, filters = ctx.saved_tensors
        timer = ctx.timer
        input_bp, filters_bp = ops.indice_conv_backward(
            features,
            filters,
            grad_output,
            indice_pairs,
            indice_pair_num,
            False,
            algo=ctx.algo,
            timer=timer,
        )
        return (
            input_bp,
            filters_bp,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class SparseInverseConvFunction(Function):
    @staticmethod
    @_TORCH_CUSTOM_FWD
    def forward(
        ctx,
        features,
        filters,
        indice_pairs,
        indice_pair_num,
        num_activate_out,
        algo,
        timer: CUDAKernelTimer = CUDAKernelTimer(False),
        bias: torch.Tensor | None = None,
        act_alpha: float = 0.0,
        act_beta: float = 0.0,
        act_type=tv.gemm.Activation.None_,
    ):
        ctx.save_for_backward(indice_pairs, indice_pair_num, features, filters)
        ctx.algo = algo
        ctx.timer = timer
        return ops.indice_conv(
            features,
            filters,
            indice_pairs,
            indice_pair_num,
            num_activate_out,
            True,
            False,
            algo=algo,
            timer=timer,
            bias=bias,
            act_alpha=act_alpha,
            act_beta=act_beta,
            act_type=act_type,
        )

    @staticmethod
    @once_differentiable
    @_TORCH_CUSTOM_BWD
    def backward(ctx, grad_output):
        indice_pairs, indice_pair_num, features, filters = ctx.saved_tensors
        timer = ctx.timer
        input_bp, filters_bp = ops.indice_conv_backward(
            features,
            filters,
            grad_output,
            indice_pairs,
            indice_pair_num,
            True,
            False,
            algo=ctx.algo,
            timer=timer,
        )
        return (
            input_bp,
            filters_bp,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class SparseImplicitGemmFunction(Function):
    @staticmethod
    @_TORCH_CUSTOM_FWD
    def forward(
        ctx,
        features: torch.Tensor,
        filters: torch.Tensor,
        pair_fwd: torch.Tensor,
        pair_bwd: torch.Tensor,
        pair_mask_fwd_splits: list[torch.Tensor],
        pair_mask_bwd_splits: list[torch.Tensor],
        mask_argsort_fwd_splits: list[torch.Tensor],
        mask_argsort_bwd_splits: list[torch.Tensor],
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
    ):
        out, mask_out, mask_width = ops.implicit_gemm(
            features,
            filters,
            pair_fwd,
            pair_mask_fwd_splits,
            mask_argsort_fwd_splits,
            num_activate_out,
            masks,
            is_train,
            is_subm,
            timer,
            fp32_accum,
            bias,
            act_alpha,
            act_beta,
            act_type,
        )
        ctx.save_for_backward(features, filters, pair_fwd, pair_bwd)
        ctx.mask_width = mask_width
        ctx.mask_out = mask_out
        ctx.timer = timer
        ctx.pair_mask_fwd_splits = pair_mask_fwd_splits
        ctx.mask_argsort_fwd_splits = mask_argsort_fwd_splits
        ctx.pair_mask_bwd_splits = pair_mask_bwd_splits
        ctx.mask_argsort_bwd_splits = mask_argsort_bwd_splits
        ctx.masks = masks
        ctx.is_subm = is_subm
        ctx.fp32_accum = fp32_accum
        return out

    @staticmethod
    @once_differentiable
    @_TORCH_CUSTOM_BWD
    def backward(ctx, grad_output):
        features, filters, pair_fwd, pair_bwd = ctx.saved_tensors
        input_bp, filters_bp = ops.implicit_gemm_backward(
            features,
            filters,
            grad_output,
            pair_fwd,
            pair_bwd,
            ctx.pair_mask_fwd_splits,
            ctx.pair_mask_bwd_splits,
            ctx.mask_argsort_fwd_splits,
            ctx.mask_argsort_bwd_splits,
            mask_output_fwd=ctx.mask_out,
            masks=ctx.masks,
            mask_width=ctx.mask_width,
            is_subm=ctx.is_subm,
            timer=ctx.timer,
            fp32_accum=ctx.fp32_accum,
        )
        return (input_bp, filters_bp, *[None] * 16)


class SubMConvFunction(Function):
    @staticmethod
    @_TORCH_CUSTOM_FWD
    def forward(
        ctx,
        features,
        filters,
        indice_pairs,
        indice_pair_num,
        num_activate_out,
        algo,
        timer: CUDAKernelTimer = CUDAKernelTimer(False),
        bias: torch.Tensor | None = None,
        act_alpha: float = 0.0,
        act_beta: float = 0.0,
        act_type=tv.gemm.Activation.None_,
    ):
        ctx.save_for_backward(indice_pairs, indice_pair_num, features, filters)
        ctx.algo = algo
        ctx.timer = timer
        return ops.indice_conv(
            features,
            filters,
            indice_pairs,
            indice_pair_num,
            num_activate_out,
            False,
            True,
            algo=algo,
            timer=timer,
            bias=bias,
            act_alpha=act_alpha,
            act_beta=act_beta,
            act_type=act_type,
        )

    @staticmethod
    @once_differentiable
    @_TORCH_CUSTOM_BWD
    def backward(ctx, grad_output):
        indice_pairs, indice_pair_num, features, filters = ctx.saved_tensors
        timer = ctx.timer
        input_bp, filters_bp = ops.indice_conv_backward(
            features,
            filters,
            grad_output,
            indice_pairs,
            indice_pair_num,
            False,
            True,
            algo=ctx.algo,
            timer=timer,
        )
        return (
            input_bp,
            filters_bp,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class SparseMaxPoolFunction(Function):
    @staticmethod
    @_TORCH_CUSTOM_FWD
    def forward(ctx, features, indice_pairs, indice_pair_num, num_activate_out):
        out = ops.indice_maxpool(
            features, indice_pairs, indice_pair_num, num_activate_out
        )
        ctx.save_for_backward(indice_pairs, indice_pair_num, features, out)
        return out

    @staticmethod
    @once_differentiable
    @_TORCH_CUSTOM_BWD
    def backward(ctx, grad_output):
        indice_pairs, indice_pair_num, features, out = ctx.saved_tensors
        input_bp = ops.indice_maxpool_backward(
            features, out, grad_output, indice_pairs, indice_pair_num
        )
        return input_bp, None, None, None


class SparseMaxPoolImplicitGemmFunction(Function):
    @staticmethod
    @_TORCH_CUSTOM_FWD
    def forward(
        ctx,
        features: torch.Tensor,
        indice_pairs_fwd: torch.Tensor,
        indice_pairs_bwd: torch.Tensor,
        num_activate_out: int,
    ):
        out = ops.indice_maxpool_implicit_gemm(
            features, indice_pairs_fwd, num_activate_out
        )
        ctx.save_for_backward(indice_pairs_bwd, features, out)
        return out

    @staticmethod
    @once_differentiable
    @_TORCH_CUSTOM_BWD
    def backward(ctx, grad_output):
        indice_pairs_bwd, features, out = ctx.saved_tensors
        input_bp = ops.indice_maxpool_implicit_gemm_backward(
            features, out, grad_output, indice_pairs_bwd
        )
        return input_bp, None, None, None


class SparseAvgPoolImplicitGemmFunction(Function):
    @staticmethod
    @_TORCH_CUSTOM_FWD
    def forward(
        ctx,
        features: torch.Tensor,
        indice_pairs_fwd: torch.Tensor,
        indice_pairs_bwd: torch.Tensor,
        num_activate_out: int,
        calc_count,
    ):
        out, count = ops.indice_avgpool_implicit_gemm(
            features, indice_pairs_fwd, num_activate_out, calc_count
        )
        ctx.save_for_backward(indice_pairs_bwd, features, out, count)
        return out

    @staticmethod
    @once_differentiable
    @_TORCH_CUSTOM_BWD
    def backward(ctx, grad_output):
        indice_pairs_bwd, _features, _out, count = ctx.saved_tensors
        input_bp = ops.indice_avgpool_implicit_gemm_backward(
            grad_output, indice_pairs_bwd, count
        )
        return input_bp, None, None, None, None


indice_conv = SparseConvFunction.apply
implicit_gemm = SparseImplicitGemmFunction.apply
indice_inverse_conv = SparseInverseConvFunction.apply
indice_subm_conv = SubMConvFunction.apply
indice_maxpool = SparseMaxPoolFunction.apply
indice_maxpool_implicit_gemm = SparseMaxPoolImplicitGemmFunction.apply
indice_avgpool_implicit_gemm = SparseAvgPoolImplicitGemmFunction.apply


def _indice_to_scalar(indices: torch.Tensor, shape: list[int]):
    if indices.shape[1] != len(shape):
        raise AssertionError
    stride = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        stride[i] = stride[i + 1] * shape[i + 1]
    scalar_inds = indices[:, -1].clone()
    for i in range(len(shape) - 1):
        scalar_inds += stride[i] * indices[:, i]
    return scalar_inds.contiguous()


def sparse_add_hash_based(*tens: SparseConvTensor):
    """sparse add with misaligned indices (hash-table based)."""
    table_size = 0
    max_num_indices = 0
    max_num_indices_idx = 0
    for i, ten in enumerate(tens):
        if ten.spatial_shape != tens[0].spatial_shape:
            raise AssertionError
        if ten.batch_size != tens[0].batch_size:
            raise AssertionError
        if ten.features.shape[1] != tens[0].features.shape[1]:
            raise AssertionError
        table_size += ten.features.shape[0]
        if max_num_indices < ten.features.shape[0]:
            max_num_indices_idx = i
            max_num_indices = ten.features.shape[0]

    first = tens[0]
    feat = first.features
    shape = [first.batch_size, *first.spatial_shape]
    whole_shape = int(np.prod(shape))
    table_size *= 2
    k_type = torch.int32
    if whole_shape >= _MAX_INT32:
        k_type = torch.int64
    table = HashTable(first.features.device, k_type, torch.int32, table_size)
    scalars: list[torch.Tensor] = []
    for ten in tens:
        indices = ten.indices
        if whole_shape >= _MAX_INT32:
            indices = indices.long()
        scalar = _indice_to_scalar(indices, shape)
        scalars.append(scalar)
        table.insert(scalar)
    count = table.assign_arange_()
    count_val = count.item()
    out_features = torch.zeros(
        [int(count_val), feat.shape[1]], dtype=feat.dtype, device=feat.device
    )
    out_indices = torch.zeros(
        [int(count_val), first.indices.shape[1]],
        dtype=first.indices.dtype,
        device=first.indices.device,
    )
    for ten, scalar in zip(tens, scalars, strict=False):
        out_inds, _ = table.query(scalar)
        out_inds = out_inds.long()
        out_features[out_inds] += ten.features
        out_indices[out_inds] = ten.indices
    res = SparseConvTensor(
        out_features,
        out_indices,
        first.spatial_shape,
        first.batch_size,
        benchmark=first.benchmark,
    )
    if count_val == max_num_indices:
        res.indice_dict = tens[max_num_indices_idx].indice_dict
    res.benchmark_record = first.benchmark_record
    res._timer = first._timer
    res.thrust_allocator = first.thrust_allocator
    return res


def sparse_add(*tens: SparseConvTensor):
    """Sparse add via torch.sparse (sort + unique internally)."""
    max_num_indices = 0
    max_num_indices_idx = 0
    ten_ths: list[torch.Tensor] = []
    first = tens[0]
    res_shape = [first.batch_size, *first.spatial_shape, first.features.shape[1]]

    for i, ten in enumerate(tens):
        if ten.spatial_shape != tens[0].spatial_shape:
            raise AssertionError
        if ten.batch_size != tens[0].batch_size:
            raise AssertionError
        if ten.features.shape[1] != tens[0].features.shape[1]:
            raise AssertionError
        if max_num_indices < ten.features.shape[0]:
            max_num_indices_idx = i
            max_num_indices = ten.features.shape[0]
        ten_ths.append(
            torch.sparse_coo_tensor(
                ten.indices.T, ten.features, res_shape, requires_grad=True
            )
        )

    c_th = reduce(lambda x, y: x + y, ten_ths).coalesce()
    c_th_inds = c_th.indices().T.contiguous().int()
    c_th_values = c_th.values()
    if not c_th_values.is_contiguous():
        raise AssertionError

    res = SparseConvTensor(
        c_th_values,
        c_th_inds,
        first.spatial_shape,
        first.batch_size,
        benchmark=first.benchmark,
    )
    if c_th_values.shape[0] == max_num_indices:
        res.indice_dict = tens[max_num_indices_idx].indice_dict
    res.benchmark_record = first.benchmark_record
    res._timer = first._timer
    res.thrust_allocator = first.thrust_allocator
    return res
