# SPDX-FileCopyrightText: Copyright contributors to the spconv-Triton project
# SPDX-FileCopyrightText: Derived from spconv, Copyright 2021 Yan Yan and contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Compatibility stub for spconv.pytorch.cppcore.

spconv's cppcore bridges torch tensors to cumm tensorview objects. There is
no cumm in spconv_triton, so only the torch-level helpers are provided;
tensorview conversion raises with a clear message.
"""

import torch


def get_current_stream() -> int:
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return 0


def torch_tensor_to_tv(*args, **kwargs):
    raise NotImplementedError(
        "spconv_triton has no cumm/tensorview backend; "
        "torch_tensor_to_tv is unavailable. Use plain torch tensors."
    )
