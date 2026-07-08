# SPDX-FileCopyrightText: Copyright contributors to the spconv-Triton project
# SPDX-FileCopyrightText: Derived from spconv, Copyright 2021 Yan Yan and contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Table modules (mirrors spconv.pytorch.tables)."""

import torch

from spconv_triton.pytorch import functional as F
from spconv_triton.pytorch.core import SparseConvTensor
from spconv_triton.pytorch.modules import SparseModule


class JoinTable(SparseModule):
    def forward(self, inputs: list[SparseConvTensor]):
        msg = "you can't use JoinTable in two sptensor with different indices."
        for ten in inputs:
            if ten.spatial_shape != inputs[0].spatial_shape:
                raise AssertionError(msg)
            if ten.batch_size != inputs[0].batch_size:
                raise AssertionError(msg)
            if ten.features.shape[1] != inputs[0].features.shape[1]:
                raise AssertionError(msg)
            if ten.indices.shape[0] != inputs[0].indices.shape[0]:
                raise AssertionError(msg)
        output = SparseConvTensor(
            torch.cat([i.features for i in inputs], 1),
            inputs[0].indices,
            inputs[0].spatial_shape,
            inputs[0].batch_size,
            inputs[0].grid,
            inputs[0].voxel_num,
            inputs[0].indice_dict,
        )
        output.benchmark_record = inputs[1].benchmark_record
        output.thrust_allocator = inputs[1].thrust_allocator
        output._timer = inputs[1]._timer
        return output

    def input_spatial_size(self, out_size):
        return out_size


class AddTable(SparseModule):
    def forward(self, inputs: list[SparseConvTensor]):
        msg = (
            "you can't use AddTable in two sptensor with different "
            "indices. use AddTableMisaligned instead."
        )
        for ten in inputs:
            if ten.spatial_shape != inputs[0].spatial_shape:
                raise AssertionError(msg)
            if ten.batch_size != inputs[0].batch_size:
                raise AssertionError(msg)
            if ten.features.shape[1] != inputs[0].features.shape[1]:
                raise AssertionError(msg)
            if ten.indices.shape[0] != inputs[0].indices.shape[0]:
                raise AssertionError(msg)
        output = SparseConvTensor(
            sum([i.features for i in inputs]),
            inputs[0].indices,
            inputs[0].spatial_shape,
            inputs[0].batch_size,
            inputs[0].grid,
            inputs[0].voxel_num,
            inputs[0].indice_dict,
        )
        output.benchmark_record = inputs[1].benchmark_record
        output.thrust_allocator = inputs[1].thrust_allocator
        output._timer = inputs[1]._timer
        return output

    def input_spatial_size(self, out_size):
        return out_size


class AddTableMisaligned(SparseModule):
    """add sptensors with same shape but different indices."""

    def forward(self, inputs: list[SparseConvTensor]):
        return F.sparse_add_hash_based(*inputs)

    def input_spatial_size(self, out_size):
        return out_size


class ConcatTable(SparseModule):
    def forward(self, x):
        return [module(x) for module in self._modules.values()]

    def add(self, module):
        self._modules[str(len(self._modules))] = module
        return self

    def input_spatial_size(self, out_size):
        return self._modules["0"].input_spatial_size(out_size)
