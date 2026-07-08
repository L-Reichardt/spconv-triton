# SPDX-FileCopyrightText: Copyright contributors to the spconv-Triton project
# SPDX-FileCopyrightText: Derived from spconv, Copyright 2021 Yan Yan and contributors
#
# SPDX-License-Identifier: Apache-2.0

from torch.nn import Module


class Identity(Module):
    def forward(self, x):
        return x

    def input_spatial_size(self, out_size):
        return out_size
