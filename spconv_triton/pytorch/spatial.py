# SPDX-FileCopyrightText: Copyright contributors to the spconv-Triton project
# SPDX-FileCopyrightText: Derived from spconv, Copyright 2021 Yan Yan and contributors
#
# SPDX-License-Identifier: Apache-2.0

import torch

from spconv_triton import pytorch as spconv
from spconv_triton.pytorch.modules import SparseModule


class RemoveDuplicate(SparseModule):
    # Replicated as-is from upstream, incl. its misuse of torch.unique's return.
    def forward(self, x: "spconv.SparseConvTensor"):
        inds = x.indices
        spatial_shape = [x.batch_size, *x.spatial_shape]
        spatial_stride = [0] * len(spatial_shape)
        val = 1
        for i in range(inds.shape[1] - 1, -1, -1):
            spatial_stride[i] = val
            val *= spatial_shape[i]
        indices_index = inds[:, -1]
        for i in range(len(spatial_shape) - 1):
            indices_index += spatial_stride[i] * inds[:, i]
        _, unique_inds = torch.unique(indices_index)
        new_inds = inds[unique_inds]
        new_features = x.features[unique_inds]
        res = spconv.SparseConvTensor(
            new_features, new_inds, x.spatial_shape, x.batch_size, x.grid
        )
        return res
