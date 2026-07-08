# SPDX-FileCopyrightText: Copyright contributors to the spconv-Triton project
# SPDX-FileCopyrightText: Derived from spconv, Copyright 2021 Yan Yan and contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Sparse convolution layers (API/behavior-compatible with
spconv.pytorch.conv)."""

import math
import time as _time

import numpy as np
import torch
from torch.nn import init
from torch.nn.init import calculate_gain
from torch.nn.parameter import Parameter

from spconv_triton import SPCONV_VERSION_NUMBERS  # noqa: F401 (parity)
from spconv_triton.compat import Activation, tv
from spconv_triton.constants import (
    SAVED_WEIGHT_LAYOUT,
    SPCONV_DEBUG_WEIGHT,
)
from spconv_triton.core import ConvAlgo
from spconv_triton.pytorch import functional as Fsp
from spconv_triton.pytorch import ops
from spconv_triton.pytorch.core import (
    ImplicitGemmIndiceData,
    IndiceData,
    SparseConvTensor,
    expand_nd,
    register_implicit_gemm_indice_data,
)
from spconv_triton.pytorch.modules import SparseModule
from spconv_triton.utils import nullcontext  # noqa: F401 (parity re-export)

# Parity re-exports (asserted by tests): always these values upstream too;
# the branches they guarded were dead and removed.
FILTER_HWIO = False
CPU_ONLY_BUILD = False


_MAX_NUM_VOXELS_DURING_TRAINING = "max_num_voxels_during_training"


class SparseConvolutionBase:
    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int | list[int] | tuple[int, ...] = 3,
        stride: int | list[int] | tuple[int, ...] = 1,
        padding: int | list[int] | tuple[int, ...] = 0,
        dilation: int | list[int] | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
        subm: bool = False,
        output_padding: int | list[int] | tuple[int, ...] = 0,
        transposed: bool = False,
        inverse: bool = False,
        indice_key: str | None = None,
        algo: ConvAlgo | None = None,
        fp32_accum: bool | None = None,
        record_voxel_count: bool = False,
        act_type: Activation = tv.gemm.Activation.None_,
        act_alpha: float = 0,
        act_beta: float = 0,
        large_kernel_fast_algo: bool = False,
    ):
        if groups != 1:
            raise AssertionError("don't support groups for now")
        self.ndim = ndim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = expand_nd(ndim, kernel_size)
        self.stride = expand_nd(ndim, stride)
        kv = int(np.prod(self.kernel_size))
        kv_stride = int(np.prod(self.stride))
        self.dilation = expand_nd(ndim, dilation)
        self.padding = expand_nd(ndim, padding)
        self.conv1x1 = kv == 1
        if not subm:
            self.conv1x1 &= kv_stride == 1
            if self.conv1x1:
                if self.padding != [0] * ndim:
                    raise AssertionError("padding must be zero for 1x1 conv (k=1,s=1)")
        self.transposed = transposed
        self.inverse = inverse
        self.output_padding = expand_nd(ndim, output_padding)
        self.groups = groups
        self.subm = subm
        self.indice_key = indice_key
        self.record_voxel_count = record_voxel_count
        if algo is None:
            limit = 32
            if large_kernel_fast_algo:
                limit = 128
            if kv <= limit:
                algo = ConvAlgo.MaskImplicitGemm
            else:
                algo = ConvAlgo.Native
        self.algo = algo
        self.fp32_accum = fp32_accum

        # Weight is always KRSC; upstream's non-KRSC layouts are dead (guard constants False everywhere).
        weight_shape = [out_channels, *self.kernel_size, in_channels]
        self.weight_shape = weight_shape
        self.act_type = act_type
        self.act_alpha = act_alpha
        self.act_beta = act_beta
        if self.conv1x1:
            if act_type != tv.gemm.Activation.None_:
                raise AssertionError("conv1x1 don't support fused act")

    def is_inverseable(self):
        return self.indice_key is not None and not self.subm

    def _conv_forward(
        self,
        training: bool,
        x: SparseConvTensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        add_input: SparseConvTensor | None = None,
        channel_scale: torch.Tensor | None = None,
        output_scale: float | None = None,
        name: str | None = None,
        act_type: Activation = tv.gemm.Activation.None_,
        act_alpha: float = 0,
        act_beta: float = 0,
    ):
        if x.is_quantized and weight.is_quantized:
            raise NotImplementedError("int8 is not supported by spconv_triton")
        if x.features.shape[1] != self.in_channels:
            raise AssertionError("channel size mismatch")
        features = x.features
        indices = x.indices
        spatial_shape = x.spatial_shape
        batch_size = x.batch_size
        bias_for_training = bias if training else None
        bias_for_infer = bias if not training else None
        if training:
            msg = "act don't support backward, only used in inference"
            if self.act_type != tv.gemm.Activation.None_:
                raise AssertionError(msg)

        if not self.subm:
            if self.transposed:
                out_spatial_shape = ops.get_deconv_output_size(
                    spatial_shape,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.output_padding,
                )
            else:
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
            if name is None:
                raise ValueError(
                    "you need to assign name to spmodules before benchmark "
                    "(spconv.utils.bench.assign_name_to_spmod)"
                )
            if name not in x.benchmark_record:
                x.benchmark_record[name] = {
                    "type": "SparseConvolution",
                    "indice_gen_time": [],
                    "time": [],
                    "num_points": [],
                    "num_out_points": [],
                    "params": {
                        "kernel_size": self.kernel_size,
                        "stride": self.stride,
                        "padding": self.padding,
                        "dilation": self.dilation,
                        "output_padding": self.output_padding,
                        "subm": self.subm,
                        "transposed": self.transposed,
                        "input_channels": self.in_channels,
                        "out_channels": self.out_channels,
                    },
                }
        if self.conv1x1:
            # Quirk: conv1x1 (plain torch.mm) ignores SPCONV_ALLOW_TF32, follows
            # torch.backends.cuda.matmul.allow_tf32 instead. weight.view is a raw
            # KRSC memory reinterpretation, not a transpose.
            features = torch.mm(
                x.features,
                weight.view(self.in_channels, self.out_channels),
            )
            if bias is not None:
                features += bias
            out_tensor = out_tensor.replace_feature(features)
            out_tensor.spatial_shape = out_spatial_shape
            return out_tensor
        indice_dict = x.indice_dict.copy()
        if not features.is_contiguous():
            features = features.contiguous()
        algo = self.algo
        if self.indice_key is not None:
            datas = x.find_indice_pair(self.indice_key)
            if datas is not None:
                msg = (
                    "due to limitation of pytorch, you must provide same "
                    "algo to layers share same indice key."
                )
                if algo != datas.algo:
                    raise AssertionError(msg)
        if algo == ConvAlgo.Native:
            datas = x.find_indice_pair(self.indice_key)
            if datas is not None and not isinstance(datas, IndiceData):
                raise AssertionError
            if self.inverse:
                if datas is None:
                    raise AssertionError
                if self.indice_key is None:
                    raise AssertionError
                if datas.is_subm is not False:
                    raise AssertionError(
                        "inverse conv can only be used with standard conv and pool ops."
                    )
                outids = datas.indices
                indice_pairs = datas.indice_pairs
                indice_pair_num = datas.indice_pair_num
                out_spatial_shape = datas.spatial_shape
                self._check_inverse_reuse_valid(x, spatial_shape, datas)
            else:
                if self.indice_key is not None and datas is not None:
                    outids = datas.out_indices
                    indice_pairs = datas.indice_pairs
                    indice_pair_num = datas.indice_pair_num
                    if not self.subm:
                        raise AssertionError("only support reuse subm indices")
                    self._check_subm_reuse_valid(x, spatial_shape, datas)
                else:
                    if x.benchmark:
                        torch.cuda.synchronize()
                        _t_gen = _time.time()
                    outids, indice_pairs, indice_pair_num = ops.get_indice_pairs(
                        indices,
                        batch_size,
                        spatial_shape,
                        algo,
                        self.kernel_size,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.output_padding,
                        self.subm,
                        self.transposed,
                    )
                    if x.benchmark:
                        torch.cuda.synchronize()
                        out_tensor.benchmark_record[name]["indice_gen_time"].append(
                            _time.time() - _t_gen
                        )
                    indice_data = IndiceData(
                        outids,
                        indices,
                        indice_pairs,
                        indice_pair_num,
                        spatial_shape,
                        out_spatial_shape,
                        is_subm=self.subm,
                        algo=algo,
                        ksize=self.kernel_size,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                    )
                    if self.indice_key is not None:
                        msg = (
                            f"your indice key {self.indice_key} already "
                            f"exists in this sparse tensor."
                        )
                        if self.indice_key in indice_dict:
                            raise AssertionError(msg)
                        indice_dict[self.indice_key] = indice_data
            if x.benchmark:
                # Parity: "time" records the conv kernel only (pair gen -> "indice_gen_time").
                torch.cuda.synchronize()
                _bench_t = _time.time()
            indice_pairs_calc = indice_pairs
            if indice_pairs.device != features.device:
                indice_pairs_calc = indice_pairs.to(features.device)
            if self.subm:
                out_features = Fsp.indice_subm_conv(
                    features,
                    weight,
                    indice_pairs_calc,
                    indice_pair_num,
                    outids.shape[0],
                    algo,
                    x._timer,
                    bias_for_infer,
                    act_alpha,
                    act_beta,
                    act_type,
                )
            else:
                if self.inverse:
                    out_features = Fsp.indice_inverse_conv(
                        features,
                        weight,
                        indice_pairs_calc,
                        indice_pair_num,
                        outids.shape[0],
                        algo,
                        x._timer,
                        bias_for_infer,
                        act_alpha,
                        act_beta,
                        act_type,
                    )
                else:
                    # Argument order replicated from upstream, including its
                    # act_type-as-act_alpha quirk (inert when act_type is None_,
                    # which is asserted for training).
                    out_features = Fsp.indice_conv(
                        features,
                        weight,
                        indice_pairs_calc,
                        indice_pair_num,
                        outids.shape[0],
                        algo,
                        x._timer,
                        bias_for_infer,
                        act_type,
                        act_beta,
                        act_type,
                    )
        else:
            datas = x.find_indice_pair(self.indice_key)
            if datas is not None and not isinstance(datas, ImplicitGemmIndiceData):
                raise AssertionError
            if self.inverse:
                if datas is None:
                    raise AssertionError
                if self.indice_key is None:
                    raise AssertionError
                if datas.is_subm is not False:
                    raise AssertionError(
                        "inverse conv can only be used with standard conv and pool ops."
                    )
                outids = datas.indices
                pair_fwd = datas.pair_bwd
                pair_bwd = datas.pair_fwd
                pair_mask_fwd_splits = datas.pair_mask_bwd_splits
                pair_mask_bwd_splits = datas.pair_mask_fwd_splits
                mask_argsort_fwd_splits = datas.mask_argsort_bwd_splits
                mask_argsort_bwd_splits = datas.mask_argsort_fwd_splits
                masks = datas.masks
                out_spatial_shape = datas.spatial_shape
                self._check_inverse_reuse_valid(x, spatial_shape, datas)
            else:
                if self.indice_key is not None and datas is not None:
                    outids = datas.out_indices
                    pair_fwd = datas.pair_fwd
                    pair_bwd = datas.pair_bwd
                    pair_mask_fwd_splits = datas.pair_mask_fwd_splits
                    pair_mask_bwd_splits = datas.pair_mask_bwd_splits
                    mask_argsort_fwd_splits = datas.mask_argsort_fwd_splits
                    mask_argsort_bwd_splits = datas.mask_argsort_bwd_splits
                    masks = datas.masks
                    if not self.subm:
                        raise AssertionError("only support reuse subm indices")
                    self._check_subm_reuse_valid(x, spatial_shape, datas)
                else:
                    if x.benchmark:
                        torch.cuda.synchronize()
                        _t_gen = _time.time()
                    res = ops.get_indice_pairs_implicit_gemm(
                        indices,
                        batch_size,
                        spatial_shape,
                        algo,
                        ksize=self.kernel_size,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        out_padding=self.output_padding,
                        subm=self.subm,
                        transpose=self.transposed,
                        is_train=(not self.subm) or training,
                        alloc=x.thrust_allocator,
                        timer=x._timer,
                    )
                    if x.benchmark:
                        torch.cuda.synchronize()
                        out_tensor.benchmark_record[name]["indice_gen_time"].append(
                            _time.time() - _t_gen
                        )
                    outids = res[0]
                    pair_fwd = res[2]
                    pair_bwd = res[3]
                    pair_mask_fwd_splits = res[4]
                    pair_mask_bwd_splits = res[5]
                    mask_argsort_fwd_splits = res[6]
                    mask_argsort_bwd_splits = res[7]
                    masks = res[8]
                    if self.indice_key is not None:
                        register_implicit_gemm_indice_data(
                            indice_dict,
                            self.indice_key,
                            res,
                            indices,
                            is_subm=self.subm,
                            spatial_shape=spatial_shape,
                            out_spatial_shape=out_spatial_shape,
                            algo=algo,
                            ksize=self.kernel_size,
                            stride=self.stride,
                            dilation=self.dilation,
                            padding=self.padding,
                        )
            if x.benchmark:
                # Parity: "time" records the conv kernel only.
                torch.cuda.synchronize()
                _bench_t = _time.time()
            num_activate_out = outids.shape[0]
            if training:
                out_features = Fsp.implicit_gemm(
                    features,
                    weight,
                    pair_fwd,
                    pair_bwd,
                    pair_mask_fwd_splits,
                    pair_mask_bwd_splits,
                    mask_argsort_fwd_splits,
                    mask_argsort_bwd_splits,
                    num_activate_out,
                    masks,
                    training,
                    self.subm,
                    x._timer,
                    self.fp32_accum,
                    bias_for_infer,
                    act_alpha,
                    act_beta,
                    act_type,
                )
            else:
                output_dtype = None
                if output_scale is None:
                    output_dtype = weight.dtype
                out_features, _, _ = ops.implicit_gemm(
                    features,
                    weight,
                    pair_fwd,
                    pair_mask_fwd_splits,
                    mask_argsort_fwd_splits,
                    num_activate_out,
                    masks,
                    training,
                    self.subm,
                    x._timer,
                    self.fp32_accum,
                    bias_for_infer,
                    act_alpha,
                    act_beta,
                    act_type,
                    1.0 if output_scale is None else output_scale,
                    channel_scale,
                    output_add=add_input.features if add_input is not None else None,
                    output_add_scale=0.0,
                    output_dtype=output_dtype,
                )

        if bias_for_training is not None:
            out_features += bias_for_training
        if x.benchmark:
            torch.cuda.synchronize()
            interval = _time.time() - _bench_t
            out_tensor.benchmark_record[name]["time"].append(interval)
            out_tensor.benchmark_record[name]["num_points"].append(features.shape[0])
            out_tensor.benchmark_record[name]["num_out_points"].append(
                out_features.shape[0]
            )
        if (
            not self.subm
            and not self.inverse
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
        if add_input is not None:
            out_tensor = out_tensor.replace_feature(
                ops._apply_act_inplace(
                    out_tensor.features + add_input.features,
                    self.act_type,
                    self.act_alpha,
                    self.act_beta,
                )
            )

        return out_tensor

    def _check_subm_reuse_valid(
        self,
        inp: SparseConvTensor,
        spatial_shape: list[int],
        datas: ImplicitGemmIndiceData | IndiceData,
    ):
        if not datas.is_subm:
            raise AssertionError("only support reuse subm indices")
        if self.kernel_size != datas.ksize:
            raise ValueError(
                f"subm with same indice_key must have same kernel"
                f" size, expect {datas.ksize}, this layer {self.kernel_size}"
            )
        if self.dilation != datas.dilation:
            raise ValueError(
                f"subm with same indice_key must have same dilation"
                f", expect {datas.dilation}, this layer {self.dilation}"
            )
        if inp.spatial_shape != datas.spatial_shape:
            raise ValueError(
                f"subm with same indice_key must have same spatial structure"
                f", expect {datas.spatial_shape}, input {spatial_shape}"
            )
        if inp.indices.shape[0] != datas.indices.shape[0]:
            raise ValueError(
                f"subm with same indice_key must have same num of indices"
                f", expect {datas.indices.shape[0]}, input {inp.indices.shape[0]}"
            )

    def _check_inverse_reuse_valid(
        self,
        inp: SparseConvTensor,
        spatial_shape: list[int],
        datas: ImplicitGemmIndiceData | IndiceData,
    ):
        if self.kernel_size != datas.ksize:
            raise ValueError(
                f"Inverse with same indice_key must have same kernel"
                f" size, expect {datas.ksize}, this layer {self.kernel_size}, "
                "please check Inverse Convolution in docs/USAGE.md."
            )
        if inp.spatial_shape != datas.out_spatial_shape:
            raise ValueError(
                f"Inverse with same indice_key must have same spatial structure (spatial shape)"
                f", expect {datas.spatial_shape}, input {spatial_shape}, "
                "please check Inverse Convolution in docs/USAGE.md."
            )
        if inp.indices.shape[0] != datas.out_indices.shape[0]:
            raise ValueError(
                f"Inverse with same indice_key must have same num of indices"
                f", expect {datas.indices.shape[0]}, input {inp.indices.shape[0]}, "
                "please check Inverse Convolution in ."
            )


class SparseConvolution(SparseConvolutionBase, SparseModule):
    __constants__ = [  # noqa: RUF012 (parity: nn.Module convention)
        "stride",
        "padding",
        "dilation",
        "groups",
        "bias",
        "subm",
        "inverse",
        "transposed",
        "output_padding",
    ]

    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int | list[int] | tuple[int, ...] = 3,
        stride: int | list[int] | tuple[int, ...] = 1,
        padding: int | list[int] | tuple[int, ...] = 0,
        dilation: int | list[int] | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
        subm: bool = False,
        output_padding: int | list[int] | tuple[int, ...] = 0,
        transposed: bool = False,
        inverse: bool = False,
        indice_key: str | None = None,
        algo: ConvAlgo | None = None,
        fp32_accum: bool | None = None,
        record_voxel_count: bool = False,
        act_type: Activation = tv.gemm.Activation.None_,
        act_alpha: float = 0,
        act_beta: float = 0,
        large_kernel_fast_algo: bool = False,
        name=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        SparseConvolutionBase.__init__(
            self,
            ndim,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=False,
            subm=subm,
            output_padding=output_padding,
            transposed=transposed,
            inverse=inverse,
            indice_key=indice_key,
            algo=algo,
            fp32_accum=fp32_accum,
            record_voxel_count=record_voxel_count,
            act_type=act_type,
            act_alpha=act_alpha,
            act_beta=act_beta,
            large_kernel_fast_algo=large_kernel_fast_algo,
        )
        SparseModule.__init__(self, name=name)
        if record_voxel_count and not self.subm and not self.inverse:
            self.register_buffer(
                _MAX_NUM_VOXELS_DURING_TRAINING,
                torch.zeros(1, dtype=torch.int32, device=device),
            )
        self.weight = Parameter(torch.zeros(*self.weight_shape, **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.zeros(out_channels, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        if hasattr(self, "_register_load_state_dict_pre_hook"):
            self._register_load_state_dict_pre_hook(self._load_weight_different_layout)

    def get_max_num_voxels(self) -> torch.Tensor | None:
        if hasattr(self, _MAX_NUM_VOXELS_DURING_TRAINING):
            return getattr(self, _MAX_NUM_VOXELS_DURING_TRAINING)
        return None

    def _load_weight_different_layout(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        name = prefix + _MAX_NUM_VOXELS_DURING_TRAINING
        if (
            self.record_voxel_count
            and not self.subm
            and not self.inverse
            and name not in state_dict
        ):
            state_dict[name] = torch.zeros(1, dtype=torch.int32)
        if not SAVED_WEIGHT_LAYOUT:
            return
        key = prefix + "weight"
        if key not in state_dict:
            raise AssertionError
        ndim = self.ndim
        if SAVED_WEIGHT_LAYOUT == "RSKC":
            state_dict[key] = (
                state_dict[key].permute(ndim, *range(ndim), ndim + 1).contiguous()
            )
        elif SAVED_WEIGHT_LAYOUT == "RSCK":
            state_dict[key] = (
                state_dict[key].permute(ndim + 1, *range(ndim), ndim).contiguous()
            )

        # Upstream bug replicated: it permutes AGAIN here, so
        # SPCONV_SAVED_WEIGHT_LAYOUT loading always fails on a shape mismatch.
        # Only the dead non-KRSC else-branch is dropped.
        if SAVED_WEIGHT_LAYOUT == "RSKC":
            state_dict[key] = (
                state_dict[key].permute(ndim, *range(ndim), ndim + 1).contiguous()
            )
        elif SAVED_WEIGHT_LAYOUT == "RSCK":
            state_dict[key] = (
                state_dict[key].permute(ndim + 1, *range(ndim), ndim).contiguous()
            )

    def extra_repr(self):
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}"
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.algo is not None:
            s += f", algo={self.algo}"
        if self.act_type != tv.gemm.Activation.None_:
            s += f", act={self.act_type}"

        return s.format(**self.__dict__)

    def _calculate_fan_in_and_fan_out(self):
        receptive_field_size = 1
        for s in self.kernel_size:
            receptive_field_size *= s
        fan_in = self.in_channels * receptive_field_size
        fan_out = self.out_channels * receptive_field_size
        return fan_in, fan_out

    def _calculate_correct_fan(self, mode):
        mode = mode.lower()
        valid_modes = ["fan_in", "fan_out"]
        if mode not in valid_modes:
            raise ValueError(
                f"Mode {mode} not supported, please use one of {valid_modes}"
            )
        fan_in, fan_out = self._calculate_fan_in_and_fan_out()
        return fan_in if mode == "fan_in" else fan_out

    def _custom_kaiming_uniform_(
        self, tensor, a=0, mode="fan_in", nonlinearity="leaky_relu"
    ):
        r"""Like torch.init.kaiming_uniform_, with KRSC-layout fan calculation."""
        fan = self._calculate_correct_fan(mode)
        gain = calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std
        with torch.no_grad():
            return tensor.uniform_(-bound, bound)

    def reset_parameters(self):
        if SPCONV_DEBUG_WEIGHT:
            self._custom_kaiming_uniform_(self.weight, a=math.sqrt(0.005))
        else:
            self._custom_kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = self._calculate_fan_in_and_fan_out()
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    # Graph-break boundary: the data-dependent pair-gen + Triton kernels are
    # opaque to inductor, so disabling lets a wrapping model torch.compile without
    # error (dense layers around it still fuse). No-op in eager. See README.
    @torch.compiler.disable
    def forward(self, x: SparseConvTensor, add_input: SparseConvTensor | None = None):
        return self._conv_forward(
            self.training,
            x,
            self.weight,
            self.bias,
            add_input,
            name=self.name,
            act_type=self.act_type,
            act_alpha=self.act_alpha,
            act_beta=self.act_beta,
        )


class SparseConv1d(SparseConvolution):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        indice_key=None,
        algo: ConvAlgo | None = None,
        fp32_accum: bool | None = None,
        record_voxel_count: bool = False,
        large_kernel_fast_algo: bool = False,
        name=None,
    ):
        super().__init__(
            1,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            indice_key=indice_key,
            algo=algo,
            fp32_accum=fp32_accum,
            record_voxel_count=record_voxel_count,
            large_kernel_fast_algo=large_kernel_fast_algo,
            name=name,
        )


class SparseConv2d(SparseConvolution):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        indice_key=None,
        algo: ConvAlgo | None = None,
        fp32_accum: bool | None = None,
        record_voxel_count: bool = False,
        large_kernel_fast_algo: bool = False,
        name=None,
    ):
        super().__init__(
            2,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            indice_key=indice_key,
            algo=algo,
            fp32_accum=fp32_accum,
            large_kernel_fast_algo=large_kernel_fast_algo,
            record_voxel_count=record_voxel_count,
            name=name,
        )


class SparseConv3d(SparseConvolution):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        indice_key=None,
        algo: ConvAlgo | None = None,
        fp32_accum: bool | None = None,
        record_voxel_count: bool = False,
        large_kernel_fast_algo: bool = False,
        name=None,
    ):
        super().__init__(
            3,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            indice_key=indice_key,
            algo=algo,
            fp32_accum=fp32_accum,
            large_kernel_fast_algo=large_kernel_fast_algo,
            record_voxel_count=record_voxel_count,
            name=name,
        )


class SparseConv4d(SparseConvolution):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        indice_key=None,
        algo: ConvAlgo | None = None,
        fp32_accum: bool | None = None,
        record_voxel_count: bool = False,
        large_kernel_fast_algo: bool = False,
        name=None,
    ):
        super().__init__(
            4,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            indice_key=indice_key,
            algo=algo,
            fp32_accum=fp32_accum,
            large_kernel_fast_algo=large_kernel_fast_algo,
            record_voxel_count=record_voxel_count,
            name=name,
        )


class SparseConvTranspose1d(SparseConvolution):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        indice_key=None,
        algo: ConvAlgo | None = None,
        fp32_accum: bool | None = None,
        record_voxel_count: bool = False,
        large_kernel_fast_algo: bool = False,
        name=None,
    ):
        super().__init__(
            1,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            transposed=True,
            indice_key=indice_key,
            algo=algo,
            fp32_accum=fp32_accum,
            large_kernel_fast_algo=large_kernel_fast_algo,
            record_voxel_count=record_voxel_count,
            name=name,
        )


class SparseConvTranspose2d(SparseConvolution):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        indice_key=None,
        algo: ConvAlgo | None = None,
        fp32_accum: bool | None = None,
        record_voxel_count: bool = False,
        large_kernel_fast_algo: bool = False,
        name=None,
    ):
        super().__init__(
            2,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            transposed=True,
            indice_key=indice_key,
            algo=algo,
            fp32_accum=fp32_accum,
            large_kernel_fast_algo=large_kernel_fast_algo,
            record_voxel_count=record_voxel_count,
            name=name,
        )


class SparseConvTranspose3d(SparseConvolution):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        indice_key=None,
        algo: ConvAlgo | None = None,
        fp32_accum: bool | None = None,
        record_voxel_count: bool = False,
        large_kernel_fast_algo: bool = False,
        name=None,
    ):
        super().__init__(
            3,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            transposed=True,
            indice_key=indice_key,
            algo=algo,
            fp32_accum=fp32_accum,
            large_kernel_fast_algo=large_kernel_fast_algo,
            record_voxel_count=record_voxel_count,
            name=name,
        )


class SparseConvTranspose4d(SparseConvolution):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        indice_key=None,
        algo: ConvAlgo | None = None,
        fp32_accum: bool | None = None,
        record_voxel_count: bool = False,
        large_kernel_fast_algo: bool = False,
        name=None,
    ):
        super().__init__(
            4,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            transposed=True,
            indice_key=indice_key,
            algo=algo,
            fp32_accum=fp32_accum,
            large_kernel_fast_algo=large_kernel_fast_algo,
            record_voxel_count=record_voxel_count,
            name=name,
        )


class SparseInverseConv1d(SparseConvolution):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        indice_key,
        bias=True,
        algo: ConvAlgo | None = None,
        fp32_accum: bool | None = None,
        large_kernel_fast_algo: bool = False,
        name=None,
    ):
        super().__init__(
            1,
            in_channels,
            out_channels,
            kernel_size,
            bias=bias,
            inverse=True,
            indice_key=indice_key,
            algo=algo,
            fp32_accum=fp32_accum,
            large_kernel_fast_algo=large_kernel_fast_algo,
            name=name,
        )


class SparseInverseConv2d(SparseConvolution):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        indice_key,
        bias=True,
        algo: ConvAlgo | None = None,
        fp32_accum: bool | None = None,
        large_kernel_fast_algo: bool = False,
        name=None,
    ):
        super().__init__(
            2,
            in_channels,
            out_channels,
            kernel_size,
            bias=bias,
            inverse=True,
            indice_key=indice_key,
            algo=algo,
            fp32_accum=fp32_accum,
            large_kernel_fast_algo=large_kernel_fast_algo,
            name=name,
        )


class SparseInverseConv3d(SparseConvolution):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        indice_key,
        bias=True,
        algo: ConvAlgo | None = None,
        fp32_accum: bool | None = None,
        large_kernel_fast_algo: bool = False,
        name=None,
    ):
        super().__init__(
            3,
            in_channels,
            out_channels,
            kernel_size,
            bias=bias,
            inverse=True,
            indice_key=indice_key,
            algo=algo,
            fp32_accum=fp32_accum,
            large_kernel_fast_algo=large_kernel_fast_algo,
            name=name,
        )


class SparseInverseConv4d(SparseConvolution):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        indice_key,
        bias=True,
        algo: ConvAlgo | None = None,
        fp32_accum: bool | None = None,
        large_kernel_fast_algo: bool = False,
        name=None,
    ):
        super().__init__(
            4,
            in_channels,
            out_channels,
            kernel_size,
            bias=bias,
            inverse=True,
            indice_key=indice_key,
            algo=algo,
            fp32_accum=fp32_accum,
            large_kernel_fast_algo=large_kernel_fast_algo,
            name=name,
        )


class SubMConv1d(SparseConvolution):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        indice_key=None,
        algo: ConvAlgo | None = None,
        fp32_accum: bool | None = None,
        large_kernel_fast_algo: bool = False,
        name=None,
    ):
        super().__init__(
            1,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            True,
            indice_key=indice_key,
            algo=algo,
            fp32_accum=fp32_accum,
            large_kernel_fast_algo=large_kernel_fast_algo,
            name=name,
        )


class SubMConv2d(SparseConvolution):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        indice_key=None,
        algo: ConvAlgo | None = None,
        fp32_accum: bool | None = None,
        large_kernel_fast_algo: bool = False,
        name=None,
    ):
        super().__init__(
            2,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            True,
            indice_key=indice_key,
            algo=algo,
            fp32_accum=fp32_accum,
            large_kernel_fast_algo=large_kernel_fast_algo,
            name=name,
        )


class SubMConv3d(SparseConvolution):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        indice_key=None,
        algo: ConvAlgo | None = None,
        fp32_accum: bool | None = None,
        large_kernel_fast_algo: bool = False,
        name=None,
    ):
        super().__init__(
            3,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            True,
            indice_key=indice_key,
            algo=algo,
            fp32_accum=fp32_accum,
            large_kernel_fast_algo=large_kernel_fast_algo,
            name=name,
        )


class SubMConv4d(SparseConvolution):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        indice_key=None,
        algo: ConvAlgo | None = None,
        fp32_accum: bool | None = None,
        large_kernel_fast_algo: bool = False,
        name=None,
    ):
        super().__init__(
            4,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            True,
            indice_key=indice_key,
            algo=algo,
            fp32_accum=fp32_accum,
            large_kernel_fast_algo=large_kernel_fast_algo,
            name=name,
        )


DEFAULT_SPARSE_CONV_TYPES = {
    SubMConv1d,
    SubMConv2d,
    SubMConv3d,
    SubMConv4d,
    SparseConv1d,
    SparseConv2d,
    SparseConv3d,
    SparseConv4d,
    SparseInverseConv1d,
    SparseInverseConv2d,
    SparseInverseConv3d,
    SparseInverseConv4d,
    SparseConvTranspose1d,
    SparseConvTranspose2d,
    SparseConvTranspose3d,
    SparseConvTranspose4d,
}
