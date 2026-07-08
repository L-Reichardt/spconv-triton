# SPDX-FileCopyrightText: Copyright contributors to the spconv-Triton project
# SPDX-FileCopyrightText: Derived from spconv, Copyright 2021 Yan Yan and contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Compatibility shim replacing the small slice of `cumm.tensorview` that the
public spconv API leaks (activation enum used by conv layer parameters).

`spconv_triton` must not depend on cumm/CUDA, so we provide an equivalent
namespace: ``from spconv_triton.compat import tv; tv.gemm.Activation.ReLU``.
"""

from enum import Enum


class Activation(Enum):
    None_ = 0
    ReLU = 1
    Sigmoid = 2
    LeakyReLU = 3


class _GemmNamespace:
    Activation = Activation


class _TensorViewNamespace:
    gemm = _GemmNamespace


tv = _TensorViewNamespace()
