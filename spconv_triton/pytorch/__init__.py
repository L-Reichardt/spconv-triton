# SPDX-FileCopyrightText: Copyright contributors to the spconv-Triton project
# SPDX-FileCopyrightText: Derived from spconv, Copyright 2021 Yan Yan and contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Public PyTorch API (drop-in for spconv.pytorch)."""

from spconv_triton.pytorch import functional, ops
from spconv_triton.pytorch.conv import (
    SparseConv1d,
    SparseConv2d,
    SparseConv3d,
    SparseConv4d,
    SparseConvTranspose1d,
    SparseConvTranspose2d,
    SparseConvTranspose3d,
    SparseConvTranspose4d,
    SparseInverseConv1d,
    SparseInverseConv2d,
    SparseInverseConv3d,
    SparseInverseConv4d,
    SubMConv1d,
    SubMConv2d,
    SubMConv3d,
    SubMConv4d,
)
from spconv_triton.pytorch.core import SparseConvTensor
from spconv_triton.pytorch.identity import Identity
from spconv_triton.pytorch.modules import (
    SparseBatchNorm,
    SparseIdentity,
    SparseModule,
    SparseReLU,
    SparseSequential,
    assign_name_for_sparse_modules,
)
from spconv_triton.pytorch.ops import ConvAlgo
from spconv_triton.pytorch.pool import (
    SparseAvgPool1d,
    SparseAvgPool2d,
    SparseAvgPool3d,
    SparseGlobalAvgPool,
    SparseGlobalMaxPool,
    SparseMaxPool1d,
    SparseMaxPool2d,
    SparseMaxPool3d,
    SparseMaxPool4d,
)
from spconv_triton.pytorch.tables import AddTable, ConcatTable, JoinTable


class ToDense(SparseModule):
    """convert SparseConvTensor to NCHW dense tensor."""

    def forward(self, x: SparseConvTensor):
        return x.dense()


class RemoveGrid(SparseModule):
    """remove pre-allocated grid buffer."""

    def forward(self, x: SparseConvTensor):
        x.grid = None
        return x
