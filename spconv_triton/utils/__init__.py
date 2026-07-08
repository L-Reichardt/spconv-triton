# SPDX-FileCopyrightText: Copyright contributors to the spconv-Triton project
# SPDX-FileCopyrightText: Derived from spconv, Copyright 2021 Yan Yan and contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Utility namespace (mirrors the relevant parts of spconv.utils)."""

from contextlib import AbstractContextManager

import torch

CPU_ONLY_BUILD = False


class nullcontext(AbstractContextManager):
    """Context manager that does no additional processing."""

    def __init__(self, enter_result=None):
        self.enter_result = enter_result

    def __enter__(self):
        return self.enter_result

    def __exit__(self, *excinfo):
        pass


# imported AFTER nullcontext: keeps the partial module usable during circular
# import re-entry from the pytorch subpackage.
from spconv_triton.pytorch.utils import PointToVoxel as _PointToVoxel


def _make_p2v(ndim: int, device_type: str):
    """Build a Point2Voxel{DEVICE}{ndim}d subclass fixing device and ndim."""

    class _Point2Voxel(_PointToVoxel):
        def __init__(
            self,
            vsize_xyz,
            coors_range_xyz,
            num_point_features,
            max_num_voxels,
            max_num_points_per_voxel,
        ):
            if len(vsize_xyz) != ndim:
                raise AssertionError
            super().__init__(
                vsize_xyz,
                coors_range_xyz,
                num_point_features,
                max_num_voxels,
                max_num_points_per_voxel,
                device=torch.device(device_type),
            )

    _Point2Voxel.__name__ = f"Point2Voxel{device_type.upper()}{ndim}d"
    return _Point2Voxel


Point2VoxelCPU1d = _make_p2v(1, "cpu")
Point2VoxelCPU2d = _make_p2v(2, "cpu")
Point2VoxelCPU3d = _make_p2v(3, "cpu")
Point2VoxelCPU4d = _make_p2v(4, "cpu")
Point2VoxelGPU1d = _make_p2v(1, "cuda")
Point2VoxelGPU2d = _make_p2v(2, "cuda")
Point2VoxelGPU3d = _make_p2v(3, "cuda")
Point2VoxelGPU4d = _make_p2v(4, "cuda")
