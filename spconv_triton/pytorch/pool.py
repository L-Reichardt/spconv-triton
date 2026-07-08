# SPDX-FileCopyrightText: Copyright contributors to the spconv-Triton project
# SPDX-FileCopyrightText: Derived from spconv, Copyright 2021 Yan Yan and contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Sparse pooling layers (mirrors spconv.pytorch.pool)."""

import time as _time

import numpy as np
import torch

from spconv_triton import pytorch as spconv
from spconv_triton.core import ConvAlgo
from spconv_triton.pytorch import functional as Fsp
from spconv_triton.pytorch import ops
from spconv_triton.pytorch.conv import _MAX_NUM_VOXELS_DURING_TRAINING
from spconv_triton.pytorch.core import (
    IndiceData,
    expand_nd,
    register_implicit_gemm_indice_data,
)
from spconv_triton.pytorch.modules import SparseModule

CPU_ONLY_BUILD = False


class SparseMaxPool(SparseModule):
    def __init__(
        self,
        ndim,
        kernel_size: int | list[int] | tuple[int, ...] = 3,
        stride: int | list[int] | tuple[int, ...] | None = 1,
        padding: int | list[int] | tuple[int, ...] = 0,
        dilation: int | list[int] | tuple[int, ...] = 1,
        indice_key: str | None = None,
        subm: bool = False,
        algo: ConvAlgo | None = None,
        record_voxel_count: bool = False,
        name=None,
    ):
        super().__init__(name=name)
        self.ndim = ndim
        self.kernel_size = expand_nd(ndim, kernel_size)
        if stride is None:
            self.stride = self.kernel_size.copy()
        else:
            self.stride = expand_nd(ndim, stride)
        self.padding = expand_nd(ndim, padding)
        self.subm = subm
        if record_voxel_count and not self.subm:
            self.register_buffer(
                _MAX_NUM_VOXELS_DURING_TRAINING, torch.zeros(1, dtype=torch.int32)
            )
        self.record_voxel_count = record_voxel_count
        self.dilation = expand_nd(ndim, dilation)
        self.indice_key = indice_key
        kv = int(np.prod(kernel_size))
        if algo is None:
            if kv <= 128:
                algo = ConvAlgo.MaskImplicitGemm
            else:
                algo = ConvAlgo.Native
        if kv > 128 and algo != ConvAlgo.Native:
            raise AssertionError("implicit gemm don't support kv >= 32 for now")
        self.algo = algo

    def extra_repr(self):
        s = "kernel_size={kernel_size}, stride={stride}"
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.algo is not None:
            s += f", algo={self.algo}"
        return s.format(**self.__dict__)

    def get_max_num_voxels(self) -> torch.Tensor | None:
        if hasattr(self, _MAX_NUM_VOXELS_DURING_TRAINING):
            return getattr(self, _MAX_NUM_VOXELS_DURING_TRAINING)
        return None

    # Graph-break boundary: data-dependent sparse path, un-torch.compile-able. See README.
    @torch.compiler.disable
    def forward(self, x: "spconv.SparseConvTensor"):
        if x.is_quantized:
            raise NotImplementedError("int8 is not supported by spconv_triton")
        if not isinstance(x, spconv.SparseConvTensor):
            raise AssertionError
        features = x.features
        device = features.device
        indices = x.indices
        spatial_shape = x.spatial_shape
        batch_size = x.batch_size
        if not self.subm:
            out_spatial_shape = ops.get_conv_output_size(
                spatial_shape,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
            )
        else:
            out_spatial_shape = spatial_shape
        out_tensor = x.shadow_copy()
        if x.benchmark:
            if self.name is None:
                raise ValueError(
                    "you need to assign name to spmodules before benchmark "
                    "(spconv.utils.bench.assign_name_to_spmod)"
                )
            if self.name not in x.benchmark_record:
                x.benchmark_record[self.name] = {
                    "type": "SparseMaxPool",
                    "indice_gen_time": [],
                    "time": [],
                    "num_points": [],
                    "num_out_points": [],
                    "params": {
                        "kernel_size": self.kernel_size,
                        "stride": self.stride,
                        "padding": self.padding,
                        "dilation": self.dilation,
                        "channels": features.shape[1],
                    },
                }
        if x.benchmark:
            torch.cuda.synchronize()
            _bench_t = _time.time()
        out_padding = [0] * self.ndim
        indice_dict = x.indice_dict.copy()
        if self.algo == ConvAlgo.Native:
            outids, indice_pairs, indice_pairs_num = ops.get_indice_pairs(
                indices,
                batch_size,
                spatial_shape,
                ConvAlgo.Native,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                out_padding,
                False,
            )
            if x.benchmark:
                # spconv parity: Native pool "time" excludes pair gen
                torch.cuda.synchronize()
                out_tensor.benchmark_record[self.name]["indice_gen_time"].append(
                    _time.time() - _bench_t
                )
                _bench_t = _time.time()
            if self.indice_key is not None:
                datas = x.find_indice_pair(self.indice_key)
                if datas is None:
                    indice_data = IndiceData(
                        outids,
                        indices,
                        indice_pairs,
                        indice_pairs_num,
                        spatial_shape,
                        out_spatial_shape,
                        is_subm=False,
                        algo=self.algo,
                        ksize=self.kernel_size,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                    )
                    indice_dict[self.indice_key] = indice_data
                else:
                    raise ValueError(f"indice key {self.indice_key} exists")

            out_features = Fsp.indice_maxpool(
                features,
                indice_pairs.to(device),
                indice_pairs_num.to(device),
                outids.shape[0],
            )
        else:
            res = ops.get_indice_pairs_implicit_gemm(
                indices,
                batch_size,
                spatial_shape,
                self.algo,
                ksize=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                out_padding=out_padding,
                subm=self.subm,
                is_train=(not self.subm) or self.training,
                alloc=x.thrust_allocator,
                timer=x._timer,
            )
            outids = res[0]
            pair_fwd = res[2]
            pair_bwd = res[3]
            if self.indice_key is not None:
                register_implicit_gemm_indice_data(
                    indice_dict,
                    self.indice_key,
                    res,
                    indices,
                    is_subm=self.subm,
                    spatial_shape=spatial_shape,
                    out_spatial_shape=out_spatial_shape,
                    algo=self.algo,
                    ksize=self.kernel_size,
                    stride=self.stride,
                    dilation=self.dilation,
                    padding=self.padding,
                )
            out_features = Fsp.indice_maxpool_implicit_gemm(
                features, pair_fwd, pair_bwd, outids.shape[0]
            )

        if x.benchmark:
            torch.cuda.synchronize()
            rec = out_tensor.benchmark_record[self.name]
            rec["time"].append(_time.time() - _bench_t)
            rec["num_points"].append(features.shape[0])
            rec["num_out_points"].append(out_features.shape[0])
        if (
            not self.subm
            and self.record_voxel_count
            and hasattr(self, _MAX_NUM_VOXELS_DURING_TRAINING)
        ):
            ops.maximum_value_int_(
                getattr(self, _MAX_NUM_VOXELS_DURING_TRAINING), outids.shape[0]
            )
        out_tensor = out_tensor.replace_feature(out_features)
        out_tensor.indices = outids
        out_tensor.indice_dict = indice_dict
        out_tensor.spatial_shape = out_spatial_shape
        return out_tensor


class SparseGlobalMaxOrAvgPool(SparseModule):
    def __init__(self, is_mean: bool, name=None):
        super().__init__(name=name)
        self.is_mean = is_mean

    # Graph-break boundary: data-dependent sparse path, un-torch.compile-able. See README.
    @torch.compiler.disable
    def forward(self, x: "spconv.SparseConvTensor"):
        if x.is_quantized:
            raise NotImplementedError("int8 is not supported by spconv_triton")
        if not isinstance(x, spconv.SparseConvTensor):
            raise AssertionError
        out_indices, counts = ops.global_pool_rearrange(x.indices, x.batch_size)
        counts_cpu = counts.cpu()
        counts_cpu_np = counts_cpu.numpy()
        res_features_list: list[torch.Tensor] = []
        for i in range(x.batch_size):
            real_inds = out_indices[i, : counts_cpu_np[i]]
            real_features = x.features[real_inds.long()]
            if self.is_mean:
                # spconv quirk: [0] selects channel 0 of the mean vector, so avg
                # pool returns a [batch_size] tensor (not [batch_size, channels]).
                real_features_reduced = torch.mean(real_features, dim=0)[0]
            else:
                real_features_reduced = torch.max(real_features, dim=0)[0]
            res_features_list.append(real_features_reduced)
        res = torch.stack(res_features_list)
        return res


class SparseGlobalAvgPool(SparseGlobalMaxOrAvgPool):
    def __init__(self, name=None):
        super().__init__(is_mean=True, name=name)


class SparseGlobalMaxPool(SparseGlobalMaxOrAvgPool):
    def __init__(self, name=None):
        super().__init__(is_mean=False, name=name)


class SparseAvgPool(SparseModule):
    def __init__(
        self,
        ndim,
        kernel_size: int | list[int] | tuple[int, ...] = 3,
        stride: int | list[int] | tuple[int, ...] | None = 1,
        padding: int | list[int] | tuple[int, ...] = 0,
        dilation: int | list[int] | tuple[int, ...] = 1,
        indice_key: str | None = None,
        subm: bool = False,
        algo: ConvAlgo | None = None,
        record_voxel_count: bool = False,
        name=None,
    ):
        super().__init__(name=name)
        self.ndim = ndim
        self.kernel_size = expand_nd(ndim, kernel_size)
        if stride is None:
            self.stride = self.kernel_size.copy()
        else:
            self.stride = expand_nd(ndim, stride)
        self.padding = expand_nd(ndim, padding)
        self.subm = subm
        if record_voxel_count and not self.subm:
            self.register_buffer(
                _MAX_NUM_VOXELS_DURING_TRAINING, torch.zeros(1, dtype=torch.int32)
            )
        self.record_voxel_count = record_voxel_count
        self.dilation = expand_nd(ndim, dilation)
        self.indice_key = indice_key
        kv = int(np.prod(kernel_size))
        if kv > 32:
            raise AssertionError(
                "avg pool only support implicit-gemm style indice gen with kv <= 32 limit"
            )
        self.algo = ConvAlgo.MaskImplicitGemm

    def extra_repr(self):
        s = "kernel_size={kernel_size}, stride={stride}"
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.algo is not None:
            s += f", algo={self.algo}"
        return s.format(**self.__dict__)

    def get_max_num_voxels(self) -> torch.Tensor | None:
        if hasattr(self, _MAX_NUM_VOXELS_DURING_TRAINING):
            return getattr(self, _MAX_NUM_VOXELS_DURING_TRAINING)
        return None

    # Graph-break boundary: data-dependent sparse path, un-torch.compile-able. See README.
    @torch.compiler.disable
    def forward(self, x):
        if x.is_quantized:
            raise NotImplementedError("int8 is not supported by spconv_triton")
        if not isinstance(x, spconv.SparseConvTensor):
            raise AssertionError
        features = x.features
        indices = x.indices
        spatial_shape = x.spatial_shape
        batch_size = x.batch_size
        if not self.subm:
            out_spatial_shape = ops.get_conv_output_size(
                spatial_shape,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
            )
        else:
            out_spatial_shape = spatial_shape
        out_tensor = x.shadow_copy()
        out_padding = [0] * self.ndim
        indice_dict = x.indice_dict.copy()
        res = ops.get_indice_pairs_implicit_gemm(
            indices,
            batch_size,
            spatial_shape,
            self.algo,
            ksize=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            out_padding=out_padding,
            subm=self.subm,
            is_train=(not self.subm) or self.training,
            alloc=x.thrust_allocator,
            timer=x._timer,
        )
        outids = res[0]
        pair_fwd = res[2]
        pair_bwd = res[3]
        if self.indice_key is not None:
            register_implicit_gemm_indice_data(
                indice_dict,
                self.indice_key,
                res,
                indices,
                is_subm=self.subm,
                spatial_shape=spatial_shape,
                out_spatial_shape=out_spatial_shape,
                algo=self.algo,
                ksize=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation,
                padding=self.padding,
            )
        out_features = Fsp.indice_avgpool_implicit_gemm(
            features, pair_fwd, pair_bwd, outids.shape[0], self.training
        )

        if (
            not self.subm
            and self.record_voxel_count
            and hasattr(self, _MAX_NUM_VOXELS_DURING_TRAINING)
        ):
            ops.maximum_value_int_(
                getattr(self, _MAX_NUM_VOXELS_DURING_TRAINING), outids.shape[0]
            )
        out_tensor = out_tensor.replace_feature(out_features)
        out_tensor.indices = outids
        out_tensor.indice_dict = indice_dict
        out_tensor.spatial_shape = out_spatial_shape
        return out_tensor


class SparseMaxPool1d(SparseMaxPool):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        indice_key=None,
        algo: ConvAlgo | None = None,
        record_voxel_count: bool = False,
        name=None,
    ):
        super().__init__(
            1,
            kernel_size,
            stride,
            padding,
            dilation,
            indice_key=indice_key,
            algo=algo,
            record_voxel_count=record_voxel_count,
            name=name,
        )


class SparseMaxPool2d(SparseMaxPool):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        indice_key=None,
        algo: ConvAlgo | None = None,
        record_voxel_count: bool = False,
        name=None,
    ):
        super().__init__(
            2,
            kernel_size,
            stride,
            padding,
            dilation,
            indice_key=indice_key,
            algo=algo,
            record_voxel_count=record_voxel_count,
            name=name,
        )


class SparseMaxPool3d(SparseMaxPool):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        indice_key=None,
        algo: ConvAlgo | None = None,
        record_voxel_count: bool = False,
        name=None,
    ):
        super().__init__(
            3,
            kernel_size,
            stride,
            padding,
            dilation,
            indice_key=indice_key,
            algo=algo,
            record_voxel_count=record_voxel_count,
            name=name,
        )


class SparseMaxPool4d(SparseMaxPool):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        indice_key=None,
        algo: ConvAlgo | None = None,
        record_voxel_count: bool = False,
        name=None,
    ):
        super().__init__(
            4,
            kernel_size,
            stride,
            padding,
            dilation,
            indice_key=indice_key,
            algo=algo,
            record_voxel_count=record_voxel_count,
            name=name,
        )


class SparseAvgPool1d(SparseAvgPool):
    """avg pool that use real point count instead of kernel size."""

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        indice_key=None,
        algo: ConvAlgo | None = None,
        record_voxel_count: bool = False,
        name=None,
    ):
        super().__init__(
            1,
            kernel_size,
            stride,
            padding,
            dilation,
            indice_key=indice_key,
            algo=algo,
            record_voxel_count=record_voxel_count,
            name=name,
        )


class SparseAvgPool2d(SparseAvgPool):
    """avg pool that use real point count instead of kernel size."""

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        indice_key=None,
        algo: ConvAlgo | None = None,
        record_voxel_count: bool = False,
        name=None,
    ):
        super().__init__(
            2,
            kernel_size,
            stride,
            padding,
            dilation,
            indice_key=indice_key,
            algo=algo,
            record_voxel_count=record_voxel_count,
            name=name,
        )


class SparseAvgPool3d(SparseAvgPool):
    """avg pool that use real point count instead of kernel size."""

    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        indice_key=None,
        algo: ConvAlgo | None = None,
        record_voxel_count: bool = False,
        name=None,
    ):
        super().__init__(
            3,
            kernel_size,
            stride,
            padding,
            dilation,
            indice_key=indice_key,
            algo=algo,
            record_voxel_count=record_voxel_count,
            name=name,
        )


ALL_POOL_LAYERS = set(
    [
        SparseAvgPool3d,
        SparseAvgPool2d,
        SparseAvgPool1d,
        SparseMaxPool1d,
        SparseMaxPool2d,
        SparseMaxPool3d,
        SparseMaxPool4d,
        SparseAvgPool,
        SparseMaxPool,
    ]
)
