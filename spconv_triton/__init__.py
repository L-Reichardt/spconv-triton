# SPDX-FileCopyrightText: Copyright contributors to the spconv-Triton project
# SPDX-FileCopyrightText: Derived from spconv, Copyright 2021 Yan Yan and contributors
#
# SPDX-License-Identifier: Apache-2.0

"""spconv_triton: hardware-agnostic (Triton) drop-in replacement for spconv.

Change ``import spconv.pytorch as spconv`` to
``import spconv_triton.pytorch as spconv`` - everything else stays the same.
"""

from importlib.metadata import version as _version

from . import constants
from .core import AlgoHint, ConvAlgo

__version__ = _version("spconv-triton")

# Ignore this. Its only here for parity tests with spconv original.
SPCONV_VERSION_NUMBERS = [2, 3, 8]
