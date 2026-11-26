# SPDX-FileCopyrightText: Copyright contributors to the spconv-Triton project
# SPDX-FileCopyrightText: Derived from spconv, Copyright 2021 Yan Yan and contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Constants that mirror spconv.constants. There are three groups:
  1. LIVE: the port reads these. Environment variables are described in
     docs/ENVIRONMENT_VARIABLES.md.
  2. PINNED: inert constants. The frozen API test tests/test_ext_misc.py
     checks their names. Do not remove them.
  3. UNPINNED: inert constants. They exist only so that
     ``from spconv.constants import X`` works.

SPCONV_NVRTC_MODE is the only upstream name that is not mirrored. Its value
is a cumm enum. spconv_triton must not import cumm.
"""

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# 1. LIVE -- read by the port
# ---------------------------------------------------------------------------

# Weight layout of a checkpoint (KRSC, RSKC, or RSCK). The state-dict load hook reads it.
# This replicates the double-permute bug of upstream. Loading a non-KRSC layout always fails.
SAVED_WEIGHT_LAYOUT = os.getenv("SPCONV_SAVED_WEIGHT_LAYOUT", "")
if SAVED_WEIGHT_LAYOUT != "":
    if SAVED_WEIGHT_LAYOUT not in ["KRSC", "RSKC", "RSCK"]:
        raise AssertionError("please set SAVED_WEIGHT_LAYOUT to KRSC, RSKC or RSCK")

# When 1, SparseConvTensor skips shape assertions, so that torch.fx can trace the model.
SPCONV_FX_TRACE_MODE = os.getenv("SPCONV_FX_TRACE_MODE", "0") == "1"

# Default for ``do_sort`` in get_indice_pairs_implicit_gemm. 0 skips the
# pair-mask sort. This is a performance option from upstream.
SPCONV_DO_SORT = os.getenv("SPCONV_DO_SORT", "1") == "1"

# Default for ``direct_table`` in get_indice_pairs_implicit_gemm. It exists only
# for signature parity. The port always builds one direct table.
SPCONV_USE_DIRECT_TABLE = True

# Selects the kaiming ``a`` value in reset_parameters. Upstream never enables this debug option.
SPCONV_DEBUG_WEIGHT = False

# Read at each call by the convolution GEMM path. When True, the fp32 GEMM uses
# TF32 instead of IEEE fp32. This mirrors upstream ``use_tf32``. You can change it
# at runtime. The conv1x1 path (torch.mm) ignores this flag. It follows
# torch.backends.cuda.matmul.allow_tf32.
SPCONV_ALLOW_TF32 = False

# Read at each call by the convolution GEMM path. When True, fp16 GEMMs accumulate
# in fp16 for short reductions (kv*C <= 128*27). This mirrors the upstream
# fp32_accum=None heuristic (spconv/algo.py:700-707). The default is OFF, so results
# match across hardware. The option is faster only on consumer Ampere and Ada GPUs.
SPCONV_ALLOW_FP16_ACCUM = os.getenv("SPCONV_ALLOW_FP16_ACCUM", "0") == "1"

# spconv_triton.core re-exports this constant. The frozen test suite checks it there.
NDIM_DONT_CARE = 3

# ---------------------------------------------------------------------------
# Inert constants. They are kept so that existing code that sets them still runs.
# ---------------------------------------------------------------------------

ALL_WEIGHT_IS_KRSC = True  # weights are always KRSC
FILTER_HWIO = False  # legacy flag for the pre-KRSC layout
DISABLE_JIT = False
EDITABLE_INSTALLED = False
BOOST_ROOT = None
SPCONV_DEBUG_SAVE_PATH = ""
SPCONV_BWD_SPLITK = [1, 2, 4, 8, 16, 32, 64]
SPCONV_DIRECT_TABLE_HASH_SIZE_SCALE = 1.1
SPCONV_INT8_DEBUG = False


class AllocKeys:
    """Allocator keys of upstream. They exist for import parity only. The port
    allocates memory through torch instead."""

    PairBwd = "PairBwd"
    IndiceNumPerLoc = "IndiceNumPerLoc"
    PairMask = "PairMask"
    MaskArgSort = "MaskArgSort"
    OutIndices = "OutIndices"
    PairFwd = "PairFwd"
    PairMaskBwd = "PairMaskBwd"
    MaskArgSortBwd = "MaskArgSortBwd"
    MaskOutputFwd = "MaskOutputFwd"
    OutFeatures = "OutFeatures"
    Features = "Features"
    Filters = "Filters"
    OutBp = "OutBp"
    DIn = "DIn"
    DFilters = "DFilters"
    InpBuffer = "InpBuffer"
    OutBuffer = "OutBuffer"
    IndicePairsUniq = "IndicePairsUniq"
    IndicePairsUniqBackup = "IndicePairsUniqBackup"
    HashKOrKV = "HashKOrKV"
    HashV = "HashV"
    ThrustTemp = "ThrustTemp"
    TightUniqueCount = "TightUniqueCount"


# ---------------------------------------------------------------------------
# Inert constants. They are kept so that existing code that sets them still runs.
# ---------------------------------------------------------------------------

PACKAGE_NAME = "spconv_triton"
SPCONV_DEBUG_CPP_ONLY = False
SPCONV_DEBUG_NVRTC_KERNELS = False
SPCONV_CPP_INDICE_PAIRS = False
SPCONV_CPP_INDICE_PAIRS_IGEMM = False
SPCONV_CPP_GEMM = False
