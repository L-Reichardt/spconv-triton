# SPDX-FileCopyrightText: Copyright contributors to the spconv-Triton project
# SPDX-FileCopyrightText: Derived from spconv, Copyright 2021 Yan Yan and contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Constants mirroring spconv.constants, in three tiers:
  1. LIVE     -- read by the port (env vars: docs/ENVIRONMENT_VARIABLES.md).
  2. PINNED   -- inert mirrors whose names are asserted by the frozen
                 API-surface test (tests/test_ext_misc.py). Do not remove.
  3. UNPINNED -- inert mirrors kept only for ``from spconv.constants import X``
                 import parity.

SPCONV_NVRTC_MODE is the only upstream name NOT mirrored: its value is a cumm
enum, and spconv_triton must not import cumm.
"""

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parent.resolve()

# ---------------------------------------------------------------------------
# 1. LIVE -- read by the port
# ---------------------------------------------------------------------------

# Checkpoint weight layout (KRSC/RSKC/RSCK), read by the state-dict load hook.
# Replicates upstream's double-permute bug: loading non-KRSC always fails.
SAVED_WEIGHT_LAYOUT = os.getenv("SPCONV_SAVED_WEIGHT_LAYOUT", "")
if SAVED_WEIGHT_LAYOUT != "":
    if SAVED_WEIGHT_LAYOUT not in ["KRSC", "RSKC", "RSCK"]:
        raise AssertionError("please set SAVED_WEIGHT_LAYOUT to KRSC, RSKC or RSCK")

# When 1, SparseConvTensor skips shape asserts so models trace under torch.fx.
SPCONV_FX_TRACE_MODE = os.getenv("SPCONV_FX_TRACE_MODE", "0") == "1"

# Default for ``do_sort`` in get_indice_pairs_implicit_gemm; 0 skips the
# pair-mask sort (upstream perf knob).
SPCONV_DO_SORT = os.getenv("SPCONV_DO_SORT", "1") == "1"

# Default for ``direct_table`` in get_indice_pairs_implicit_gemm. Accepted for
# signature parity only; the port has a single (direct) table build.
SPCONV_USE_DIRECT_TABLE = True

# kaiming ``a`` selector in reset_parameters; debug knob upstream never ships on.
SPCONV_DEBUG_WEIGHT = False

# Read live by the conv GEMM path. When True, fp32 GEMM uses TF32 instead of
# IEEE fp32 (mirrors upstream ``use_tf32``). Runtime-toggleable. The conv1x1
# (torch.mm) path ignores this and follows torch.backends.cuda.matmul.allow_tf32.
SPCONV_ALLOW_TF32 = False

# Read live by the conv GEMM path. When True, fp16 GEMMs accumulate in fp16 for
# short reductions (kv*C <= 128*27), mirroring upstream's fp32_accum=None
# heuristic (spconv/algo.py:700-707). DEFAULT OFF for cross-hardware correctness.
# Faster ONLY on consumer Ampere/Ada.
SPCONV_ALLOW_FP16_ACCUM = os.getenv("SPCONV_ALLOW_FP16_ACCUM", "0") == "1"

# Re-exported by spconv_triton.core, where the frozen suite pins it.
NDIM_DONT_CARE = 3

# ---------------------------------------------------------------------------
# Dead constants, kept as some projects might explicitly set them
# ---------------------------------------------------------------------------

ALL_WEIGHT_IS_KRSC = True  # weights are always KRSC
FILTER_HWIO = False  # pre-KRSC legacy layout flag
DISABLE_JIT = False
EDITABLE_INSTALLED = False
BOOST_ROOT = None
SPCONV_DEBUG_SAVE_PATH = ""
SPCONV_BWD_SPLITK = [1, 2, 4, 8, 16, 32, 64]
SPCONV_DIRECT_TABLE_HASH_SIZE_SCALE = 1.1
SPCONV_INT8_DEBUG = False


class AllocKeys:
    """Upstream allocator dict keys; kept for import parity (the port allocates
    through torch instead)."""

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
# Dead constants, kept as some projects might explicitly set them
# ---------------------------------------------------------------------------

PACKAGE_NAME = "spconv_triton"
SPCONV_DEBUG_CPP_ONLY = False
SPCONV_DEBUG_NVRTC_KERNELS = False
SPCONV_CPP_INDICE_PAIRS = False
SPCONV_CPP_INDICE_PAIRS_IGEMM = False
SPCONV_CPP_GEMM = False
