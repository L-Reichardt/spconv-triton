# SPDX-FileCopyrightText: Copyright contributors to the spconv-Triton project
# SPDX-FileCopyrightText: Derived from spconv, Copyright 2021 Yan Yan and contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Core enums (mirrors spconv.core)."""

from enum import Enum

from spconv_triton.constants import NDIM_DONT_CARE  # noqa: F401 (parity re-export)


class ConvAlgo(Enum):
    Native = 0
    MaskImplicitGemm = 1
    MaskSplitImplicitGemm = 2


class AlgoHint(Enum):
    NoHint = 0
    Fowrard = 1  # [sic] - kept identical to upstream spelling
    BackwardInput = 2
    BackwardWeight = 4
