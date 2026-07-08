# SPDX-FileCopyrightText: Copyright contributors to the spconv-Triton project
# SPDX-FileCopyrightText: Derived from spconv, Copyright 2021 Yan Yan and contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Point cloud voxelization utilities (torch-based, device-agnostic).

API-compatible with spconv.pytorch.utils. Voxel assignment is
first-come-first-served by point order; num_per_voxel is clamped at
max_num_points_per_voxel (both match reference semantics).
"""

import torch


class PointToVoxel:
    """WARNING: you MUST construct PointToVoxel AFTER set device."""

    def __init__(
        self,
        vsize_xyz: list[float],
        coors_range_xyz: list[float],
        num_point_features: int,
        max_num_voxels: int,
        max_num_points_per_voxel: int,
        device: torch.device = torch.device("cpu:0"),
    ):
        self.ndim = len(vsize_xyz)
        self.device = device
        self.num_point_features = num_point_features
        self.max_num_voxels = max_num_voxels
        self.max_num_points_per_voxel = max_num_points_per_voxel

        cmin = coors_range_xyz[: self.ndim]
        cmax = coors_range_xyz[self.ndim :]
        grid_size_xyz = [
            round((cmax[i] - cmin[i]) / vsize_xyz[i]) for i in range(self.ndim)
        ]
        # Internal math stays XYZ: keeps voxel/index outputs bitwise-identical to reference.
        self._vsize_xyz = list(vsize_xyz)
        self._grid_size_xyz = grid_size_xyz
        self._coors_range_xyz = list(coors_range_xyz)

        # Public metadata stored ZYX-reversed, matching spconv (calc_point2voxel_meta_data):
        # downstream spatial_shape derivation relies on this order.
        self.vsize = list(reversed(vsize_xyz))
        self.grid_size = list(reversed(grid_size_xyz))
        self.coors_range = list(reversed(cmin)) + list(reversed(cmax))
        stride = [1] * self.ndim
        for i in range(self.ndim - 2, -1, -1):
            stride[i] = stride[i + 1] * self.grid_size[i + 1]
        self.grid_stride = stride

        self.voxels = torch.zeros(
            [max_num_voxels, max_num_points_per_voxel, num_point_features],
            dtype=torch.float32,
            device=device,
        )
        self.indices = torch.zeros(
            [max_num_voxels, self.ndim], dtype=torch.int32, device=device
        )
        self.num_per_voxel = torch.zeros(
            [max_num_voxels], dtype=torch.int32, device=device
        )

    def __call__(
        self, pc: torch.Tensor, clear_voxels: bool = True, empty_mean: bool = False
    ):
        res = self.generate_voxel_with_id(pc, clear_voxels, empty_mean)
        return res[0], res[1], res[2]

    def generate_voxel_with_id(
        self, pc: torch.Tensor, clear_voxels: bool = True, empty_mean: bool = False
    ):
        if pc.device.type != self.device.type:
            raise AssertionError("your pc device is wrong")
        ndim = self.ndim
        device = self.device
        n = pc.shape[0]
        maxpts = self.max_num_points_per_voxel
        with torch.no_grad():
            pc_voxel_id = torch.full([n], -1, dtype=torch.int64, device=device)
            cmin = torch.tensor(
                self._coors_range_xyz[:ndim], dtype=pc.dtype, device=device
            )
            vsize = torch.tensor(self._vsize_xyz, dtype=pc.dtype, device=device)
            gsize = torch.tensor(self._grid_size_xyz, dtype=torch.int64, device=device)
            q = torch.floor((pc[:, :ndim] - cmin) / vsize).long()
            in_range = ((q >= 0) & (q < gsize)).all(1)

            if clear_voxels:
                self.voxels.zero_()

            valid_idx = torch.nonzero(in_range).view(-1)
            if valid_idx.numel() == 0:
                return (
                    self.voxels[:0].clone(),
                    self.indices[:0].clone(),
                    self.num_per_voxel[:0].clone(),
                    pc_voxel_id,
                )
            qv = q[valid_idx]
            lin = qv[:, 0]
            for d in range(1, ndim):
                lin = lin * int(self._grid_size_xyz[d]) + qv[:, d]

            uniq, inverse = torch.unique(lin, return_inverse=True)
            # FCFS voxel ids: order voxels by first point occurrence
            first_seen = torch.full(
                (uniq.numel(),), n, dtype=torch.int64, device=device
            )
            first_seen.scatter_reduce_(0, inverse, valid_idx, reduce="amin")
            voxel_order = torch.argsort(first_seen)
            vid_of_uniq = torch.empty_like(voxel_order)
            vid_of_uniq[voxel_order] = torch.arange(uniq.numel(), device=device)
            vid = vid_of_uniq[inverse]  # FCFS voxel id per valid point

            num_voxels = min(int(uniq.numel()), self.max_num_voxels)
            kept = vid < num_voxels

            counts = torch.bincount(vid[kept], minlength=num_voxels)
            self.num_per_voxel[:num_voxels] = counts.clamp(max=maxpts).to(torch.int32)

            # voxel coords (ZYX-reversed quantized coords)
            uniq_sorted_by_vid = uniq[voxel_order][:num_voxels]
            coords = torch.empty((num_voxels, ndim), dtype=torch.int64, device=device)
            rem = uniq_sorted_by_vid
            for d in range(ndim - 1, -1, -1):
                coords[:, d] = rem % int(self._grid_size_xyz[d])
                rem = rem // int(self._grid_size_xyz[d])
            self.indices[:num_voxels] = coords.flip(1).to(torch.int32)

            # slot of each point inside its voxel (arrival order)
            kept_idx = valid_idx[kept]
            kept_vid = vid[kept]
            order = torch.argsort(kept_vid * (n + 1) + kept_idx)
            sorted_vid = kept_vid[order]
            seg_start = torch.zeros(num_voxels, dtype=torch.int64, device=device)
            seg_start[1:] = torch.cumsum(counts, 0)[:-1]
            slot = (
                torch.arange(sorted_vid.numel(), device=device) - seg_start[sorted_vid]
            )
            stored = slot < maxpts
            self.voxels[sorted_vid[stored], slot[stored]] = pc[
                kept_idx[order][stored], : self.num_point_features
            ].float()
            pc_voxel_id[kept_idx] = kept_vid

            if empty_mean:
                npv = self.num_per_voxel[:num_voxels].long()
                sums = self.voxels[:num_voxels].sum(dim=1)
                means = sums / npv.clamp(min=1).unsqueeze(1).float()
                slot_idx = torch.arange(maxpts, device=device)
                empty_mask = slot_idx[None, :] >= npv[:, None]
                vox = self.voxels[:num_voxels]
                vox[empty_mask] = means.unsqueeze(1).expand_as(vox)[empty_mask]

            return (
                self.voxels[:num_voxels].clone(),
                self.indices[:num_voxels].clone(),
                self.num_per_voxel[:num_voxels].clone(),
                pc_voxel_id,
            )


def gather_features_by_pc_voxel_id(
    seg_res_features: torch.Tensor,
    pc_voxel_id: torch.Tensor,
    invalid_value: int | float = 0,
):
    """Gather segmentation results back to the original point cloud."""
    if seg_res_features.device != pc_voxel_id.device:
        pc_voxel_id = pc_voxel_id.to(seg_res_features.device)
    res_feature_shape = (pc_voxel_id.shape[0], *seg_res_features.shape[1:])
    if invalid_value == 0:
        res = torch.zeros(
            res_feature_shape,
            dtype=seg_res_features.dtype,
            device=seg_res_features.device,
        )
    else:
        res = torch.full(
            res_feature_shape,
            invalid_value,
            dtype=seg_res_features.dtype,
            device=seg_res_features.device,
        )
    pc_voxel_id_valid = pc_voxel_id != -1
    pc_voxel_id_valid_ids = torch.nonzero(pc_voxel_id_valid).view(-1)
    seg_res_features_valid = seg_res_features[pc_voxel_id[pc_voxel_id_valid_ids]]
    res[pc_voxel_id_valid_ids] = seg_res_features_valid
    return res
