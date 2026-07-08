# SPDX-FileCopyrightText: Copyright contributors to the spconv-Triton project
# SPDX-FileCopyrightText: Derived from spconv, Copyright 2021 Yan Yan and contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Torch version constants (mirrors spconv.pytorch.constants)."""

import torch

try:
    # torch.__version__ is a TorchVersion (str subclass); keep the parsing on a
    # plain str so the public constant is unambiguously list[int] (drives the
    # `>= [major, minor, patch]` gates across the package).
    _version = str(torch.__version__)
    remove_plus = _version.find("+")
    remove_dotdev = _version.find(".dev")

    _clean = _version
    if remove_plus != -1:
        _clean = _version[:remove_plus]
    if remove_dotdev != -1:
        _clean = _version[:remove_dotdev]

    PYTORCH_VERSION: list[int] = list(map(int, _clean.split(".")))
except Exception:
    PYTORCH_VERSION = [1, 8, 0]
