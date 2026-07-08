# SPDX-FileCopyrightText: Copyright contributors to the spconv-Triton project
# SPDX-FileCopyrightText: Derived from spconv, Copyright 2021 Yan Yan and contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Timer shim (API-compatible no-op version of spconv.tools)."""

import contextlib
from contextlib import AbstractContextManager

CPU_ONLY_BUILD = False


class nullcontext(AbstractContextManager):
    """No-op context manager. Defined locally to avoid an import cycle through
    the pytorch subpackage (vs importing from spconv_triton.utils)."""

    def __init__(self, enter_result=None):
        self.enter_result = enter_result

    def __enter__(self):
        return self.enter_result

    def __exit__(self, *excinfo):
        pass


class CUDAKernelTimer:
    def __init__(self, enable: bool = True):
        self.enable = enable
        self._timer = None

    @contextlib.contextmanager
    def namespace(self, name: str):
        yield

    @contextlib.contextmanager
    def record(self, name: str, stream: int = 0):
        yield

    def get_all_pair_time(self):
        return {}
