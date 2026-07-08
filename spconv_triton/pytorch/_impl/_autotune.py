# SPDX-FileCopyrightText: Copyright contributors to the spconv-Triton project
# SPDX-FileCopyrightText: Derived from spconv, Copyright 2021 Yan Yan and contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Version-portable opt-in to Triton's on-disk autotune cache.

Exposes ``AUTOTUNE_CACHE_KW`` to splat into ``@triton.autotune(...)``; enables
disk-persisted config selection where supported, no-op otherwise.
"""

import inspect

import triton

# Splat into @triton.autotune(...): {"cache_results": True} on Triton >= 3.3.0,
# else {} (the kwarg is absent and would raise TypeError).
AUTOTUNE_CACHE_KW: dict[str, bool] = (
    {"cache_results": True}
    if "cache_results" in inspect.signature(triton.autotune).parameters
    else {}
)
