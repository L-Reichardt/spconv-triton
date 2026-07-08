# SPDX-FileCopyrightText: Copyright contributors to the spconv-Triton project
# SPDX-FileCopyrightText: Derived from spconv, Copyright 2021 Yan Yan and contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Union

import numpy as np
import torch

from spconv_triton.constants import SPCONV_FX_TRACE_MODE
from spconv_triton.core import ConvAlgo
from spconv_triton.pytorch.constants import PYTORCH_VERSION
from spconv_triton.tools import CUDAKernelTimer

if PYTORCH_VERSION >= [1, 8, 0]:
    try:
        import torch.fx

        if PYTORCH_VERSION >= [1, 10, 0]:
            from torch.fx import ProxyableClassMeta
        else:
            from torch.fx.symbolic_trace import (  # type: ignore[no-redef]
                ProxyableClassMeta,
            )
        SpConvTensorMeta = ProxyableClassMeta
    except Exception:

        class SpConvTensorMeta(type):  # type: ignore[no-redef]
            pass
else:

    class SpConvTensorMeta(type):  # type: ignore[no-redef]
        pass


class ThrustSortAllocator:
    """API-parity shim: upstream's thrust scratch allocator. Triton pair-gen
    never allocates through it; only the constructor surface is kept."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.alloced_objs: dict = {}
        self.device = device


class IndiceData:
    def __init__(
        self,
        out_indices,
        indices,
        indice_pairs,
        indice_pair_num,
        spatial_shape,
        out_spatial_shape,
        is_subm: bool,
        algo: ConvAlgo,
        ksize: list[int],
        stride: list[int],
        dilation: list[int],
        padding: list[int],
        voxel_num: Any | None = None,
    ):
        self.out_indices = out_indices
        self.indices = indices
        self.indice_pairs = indice_pairs
        self.indice_pair_num = indice_pair_num
        self.spatial_shape = spatial_shape
        self.out_spatial_shape = out_spatial_shape
        self.is_subm = is_subm
        self.algo = algo
        self.ksize = ksize
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.voxel_num = voxel_num


class ImplicitGemmIndiceData:
    def __init__(
        self,
        out_indices: torch.Tensor,
        indices: torch.Tensor,
        pair_fwd: torch.Tensor,
        pair_bwd: torch.Tensor,
        pair_mask_fwd_splits: list[torch.Tensor],
        pair_mask_bwd_splits: list[torch.Tensor],
        mask_argsort_fwd_splits: list[torch.Tensor],
        mask_argsort_bwd_splits: list[torch.Tensor],
        masks: list[np.ndarray],
        spatial_shape,
        out_spatial_shape,
        is_subm: bool,
        algo: ConvAlgo,
        ksize: list[int],
        stride: list[int],
        dilation: list[int],
        padding: list[int],
        in_voxel_num: Any | None = None,
        out_voxel_num: Any | None = None,
    ):
        self.out_indices = out_indices
        self.indices = indices
        self.pair_fwd = pair_fwd
        self.pair_bwd = pair_bwd
        self.pair_mask_fwd_splits = pair_mask_fwd_splits
        self.pair_mask_bwd_splits = pair_mask_bwd_splits
        self.mask_argsort_fwd_splits = mask_argsort_fwd_splits
        self.mask_argsort_bwd_splits = mask_argsort_bwd_splits
        self.masks = masks
        self.spatial_shape = spatial_shape
        self.out_spatial_shape = out_spatial_shape
        self.is_subm = is_subm
        self.algo = algo
        self.ksize = ksize
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.in_voxel_num = in_voxel_num
        self.out_voxel_num = out_voxel_num


def register_implicit_gemm_indice_data(
    indice_dict,
    indice_key,
    res,
    indices,
    *,
    is_subm,
    spatial_shape,
    out_spatial_shape,
    algo,
    ksize,
    stride,
    dilation,
    padding,
):
    """Build an ImplicitGemmIndiceData from a get_indice_pairs_implicit_gemm
    result tuple and register it under indice_key (raises on collision).
    Shared by the conv and pool implicit-gemm forward paths."""
    indice_data = ImplicitGemmIndiceData(
        res[0],
        indices,
        res[2],
        res[3],
        pair_mask_fwd_splits=res[4],
        pair_mask_bwd_splits=res[5],
        mask_argsort_fwd_splits=res[6],
        mask_argsort_bwd_splits=res[7],
        masks=res[8],
        is_subm=is_subm,
        spatial_shape=spatial_shape,
        out_spatial_shape=out_spatial_shape,
        algo=algo,
        ksize=ksize,
        stride=stride,
        dilation=dilation,
        padding=padding,
    )
    if indice_key in indice_dict:
        raise AssertionError(
            f"your indice key {indice_key} already exists in this sparse tensor."
        )
    indice_dict[indice_key] = indice_data


def scatter_nd(indices, updates, shape):
    """pytorch edition of tensorflow scatter_nd.
    No error handling; repeated indices overwrite (no repeat-add unlike tf).
    """
    ret = torch.zeros(*shape, dtype=updates.dtype, device=updates.device)
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + shape[indices.shape[-1] :]
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]
    ret[tuple(slices)] = updates.view(*output_shape)
    return ret


class SparseConvTensor(metaclass=SpConvTensorMeta):
    def __init__(
        self,
        features: torch.Tensor,
        indices: torch.Tensor,
        spatial_shape: list[int] | np.ndarray,
        batch_size: int,
        grid: torch.Tensor | None = None,
        voxel_num: torch.Tensor | None = None,
        indice_dict: dict | None = None,
        benchmark: bool = False,
        permanent_thrust_allocator: bool = False,
        enable_timer: bool = False,
        force_algo: ConvAlgo | None = None,
    ):
        """Sparse tensor container: features + indices + spatial_shape, with
        indice-pair caching by indice_key.

        Args:
            features: [num_points, num_features] feature tensor.
            indices: [num_points, ndim + 1]; batch index in indices[:, 0].
            spatial_shape: spatial shape of the sparse data.
            batch_size: batch size of the sparse data.
            grid: pre-allocated grid tensor.
            benchmark: enable benchmark recording.
            enable_timer: record internal op times in _timer.
            force_algo: vestigial - stored/propagated but never read in conv/pool forward (upstream parity).
        """
        ndim = indices.shape[1] - 1
        if not SPCONV_FX_TRACE_MODE:
            if features.ndim != 2:
                raise AssertionError
            if indices.ndim != 2:
                raise AssertionError
            if len(spatial_shape) != ndim:
                raise AssertionError("spatial shape must equal to ndim")
            if indices.dtype != torch.int32:
                raise AssertionError("only support int32")
            if batch_size <= 0:
                raise AssertionError
        self._features = features
        self.indices = indices
        self.spatial_shape = [int(v) for v in spatial_shape]
        self.batch_size = batch_size
        if indice_dict is None:
            indice_dict = {}
        self.indice_dict = indice_dict
        if grid is None:
            grid = torch.Tensor()
        self.grid: torch.Tensor | None = grid
        self.voxel_num = voxel_num
        self.benchmark = benchmark
        self.benchmark_record: dict = {}
        self.thrust_allocator: ThrustSortAllocator | None = None
        if permanent_thrust_allocator:
            self.thrust_allocator = ThrustSortAllocator(features.device)
        self._timer = CUDAKernelTimer(enable_timer)
        self.force_algo = force_algo
        self.int8_scale: np.ndarray | None = None

    def __repr__(self):
        return f"SparseConvTensor[shape={self._features.shape}]"

    @property
    def is_quantized(self):
        return self.features.dtype == torch.qint8

    def q_scale(self):
        if self.is_quantized:
            return self.features.q_scale()
        raise ValueError("sparse tensor must be quantized")

    def replace_feature(self, feature: torch.Tensor):
        """Return a new tensor with features replaced; use instead of setting
        .features directly (required by torch.fx)."""
        new_spt = SparseConvTensor(
            feature,
            self.indices,
            self.spatial_shape,
            self.batch_size,
            self.grid,
            self.voxel_num,
            self.indice_dict,
        )
        new_spt.benchmark = self.benchmark
        new_spt.benchmark_record = self.benchmark_record
        new_spt.thrust_allocator = self.thrust_allocator
        new_spt._timer = self._timer
        new_spt.force_algo = self.force_algo
        new_spt.int8_scale = self.int8_scale
        return new_spt

    def select_by_index(self, valid_indices: torch.Tensor):
        # Upstream parity: assigns to read-only `features` property -> always raises ValueError.
        new_spt = self.shadow_copy()
        new_spt.indices = self.indices[valid_indices]
        new_spt.features = self.features[valid_indices]
        new_spt.indice_dict.clear()
        return new_spt

    def minus(self):
        return self.replace_feature(-self.features)

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, val):
        msg = (
            "you can't set feature directly, use 'x = x.replace_feature(your_new_feature)'"
            " to generate new SparseConvTensor instead."
        )
        raise ValueError(msg)

    @classmethod
    def from_dense(cls, x: torch.Tensor):
        """Create a sparse tensor from a channel-last (NHWC) dense tensor."""
        x_sp = x.to_sparse(x.ndim - 1)
        spatial_shape = x_sp.shape[1:-1]
        batch_size = x_sp.shape[0]
        indices_th = x_sp.indices().permute(1, 0).contiguous().int()
        features_th = x_sp.values()
        return cls(features_th, indices_th, list(spatial_shape), batch_size)

    def dequantize(self):
        return self.replace_feature(self.features.dequantize())

    @property
    def spatial_size(self):
        return np.prod(self.spatial_shape)

    def find_indice_pair(self, key) -> IndiceData | ImplicitGemmIndiceData | None:
        if key is None:
            return None
        if key in self.indice_dict:
            return self.indice_dict[key]
        return None

    def dense(self, channels_first: bool = True):
        output_shape = [
            self.batch_size,
            *list(self.spatial_shape),
            self.features.shape[1],
        ]
        res = scatter_nd(
            self.indices.to(self.features.device).long(), self.features, output_shape
        )
        if not channels_first:
            return res
        ndim = len(self.spatial_shape)
        trans_params = list(range(0, ndim + 1))
        trans_params.insert(1, ndim + 1)
        return res.permute(*trans_params).contiguous()

    def __add__(self, other: Union["SparseConvTensor", torch.Tensor]):
        if not isinstance(other, (SparseConvTensor, torch.Tensor)):
            raise AssertionError
        other_features = other if isinstance(other, torch.Tensor) else other.features
        return self.replace_feature(self.features + other_features)

    def __iadd__(self, other: Union["SparseConvTensor", torch.Tensor]):
        if not isinstance(other, (SparseConvTensor, torch.Tensor)):
            raise AssertionError
        other_features = other if isinstance(other, torch.Tensor) else other.features
        self.features += other_features
        return self

    def __radd__(self, other: Union["SparseConvTensor", torch.Tensor]):
        if not isinstance(other, (SparseConvTensor, torch.Tensor)):
            raise AssertionError
        other_features = other if isinstance(other, torch.Tensor) else other.features
        return self.replace_feature(self.features + other_features)

    def shadow_copy(self) -> "SparseConvTensor":
        """Create a new tensor sharing all members with this one."""
        tensor = SparseConvTensor(
            self.features,
            self.indices,
            self.spatial_shape,
            self.batch_size,
            self.grid,
            self.voxel_num,
            self.indice_dict,
            self.benchmark,
        )
        tensor.benchmark_record = self.benchmark_record
        tensor.thrust_allocator = self.thrust_allocator
        tensor._timer = self._timer
        tensor.force_algo = self.force_algo
        tensor.int8_scale = self.int8_scale
        return tensor


def expand_nd(
    ndim: int, val: int | list[int] | tuple[int, ...] | np.ndarray
) -> list[int]:
    if isinstance(val, int):
        res = [val] * ndim
    elif isinstance(val, (tuple, np.ndarray)):
        res = list(val)
    else:
        res = val
    if len(res) != ndim:
        raise AssertionError
    return [int(v) for v in res]
