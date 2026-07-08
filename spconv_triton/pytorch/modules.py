# SPDX-FileCopyrightText: Copyright contributors to the spconv-Triton project
# SPDX-FileCopyrightText: Derived from spconv, Copyright 2021 Yan Yan and contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Module containers and wrappers (mirrors spconv.pytorch.modules)."""

import time
from collections import OrderedDict

import torch
from torch import nn

from spconv_triton import pytorch as spconv


def is_spconv_module(module):
    spconv_modules = (SparseModule, SparseBatchNorm, SparseReLU)
    return isinstance(module, spconv_modules)


def is_sparse_conv(module):
    from spconv_triton.pytorch.conv import SparseConvolution

    return isinstance(module, SparseConvolution)


class SparseModule(nn.Module):
    """Base class marking a module as sparse-aware for SparseSequential dispatch."""

    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self._sparse_unique_name = ""


class SparseSequential(SparseModule):
    r"""Sequential container. Modules run in insertion order; pass positionally,
    as an OrderedDict, or as keyword args."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError(f"index {idx} is out of range")
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for _i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, x):
        for _k, module in self._modules.items():
            if is_spconv_module(module):
                x = module(x)
            else:
                if isinstance(x, spconv.SparseConvTensor):
                    if x.indices.shape[0] != 0:
                        x = x.replace_feature(module(x.features))
                else:
                    x = module(x)
        return x


def assign_name_for_sparse_modules(module: nn.Module):
    for k, n in module.named_modules():
        if isinstance(n, SparseModule):
            n._sparse_unique_name = k


class SparseBatchNorm(nn.BatchNorm1d):
    """exists only for torch.fx transformation for quantization."""

    def forward(self, x):
        if isinstance(x, spconv.SparseConvTensor):
            return x.replace_feature(super().forward(x.features))
        return super().forward(x)


class SparseSyncBatchNorm(nn.SyncBatchNorm):
    """exists only for torch.fx transformation for quantization."""

    def forward(self, x):
        if isinstance(x, spconv.SparseConvTensor):
            return x.replace_feature(super().forward(x.features))
        return super().forward(x)


class SparseReLU(nn.ReLU):
    """exists only for torch.fx transformation for quantization."""

    def forward(self, x):
        if isinstance(x, spconv.SparseConvTensor):
            return x.replace_feature(super().forward(x.features))
        return super().forward(x)


class SparseIdentity(nn.Identity):
    """exists only for torch.fx transformation for quantization."""

    def forward(self, x):
        if isinstance(x, spconv.SparseConvTensor):
            return x.replace_feature(super().forward(x.features))
        return super().forward(x)


class PrintTensorMeta(nn.Module):
    def forward(self, x):
        if isinstance(x, torch.Tensor):
            print(x.min(), x.max(), x.mean())
        elif isinstance(x, spconv.SparseConvTensor):
            ft = x.features
            print(ft.min(), ft.max(), ft.mean())
        return x


class PrintCurrentTime(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first_time = time.time()

    def forward(self, x, msg="", reset: bool = False):
        if reset:
            self.first_time = time.time()
        torch.cuda.synchronize()
        print(msg, time.time() - self.first_time)
        return x
