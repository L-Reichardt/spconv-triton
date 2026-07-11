"""Extension PointToVoxel coverage: CPU (bitwise), cap path, 1d/4d,
pc_voxel_id truncation semantics."""

import pytest
import torch

from helpers import assert_close, assert_tensor_equal, canon_voxel_result
from helpers_golden import aux_case, aux_case_ids


@pytest.mark.parametrize("case_id", aux_case_ids("golden_ext_p2v.pt",
                                                 "p2v_cpu"))
def test_p2v_cpu_bitwise(impl, case_id):
    """CPU voxelization is first-come-first-served deterministic: outputs
    must match the reference bitwise (incl. truncation and voxel cap)."""
    case = aux_case("golden_ext_p2v.pt", case_id)
    p2v = impl.putils.PointToVoxel(device=torch.device("cpu"),
                                   **case["ctor"])
    vox, idx, npv, pcvid = p2v.generate_voxel_with_id(case["points"])
    expect = case["expect"]
    assert_tensor_equal(idx, expect["indices"], f"{case_id}: indices")
    assert_tensor_equal(npv, expect["num_per_voxel"],
                        f"{case_id}: num_per_voxel")
    assert_tensor_equal(pcvid, expect["pc_voxel_id"],
                        f"{case_id}: pc_voxel_id")
    assert_close(vox, expect["voxels"], 0.0, f"{case_id}: voxels")


@pytest.mark.parametrize("case_id", aux_case_ids("golden_ext_p2v.pt",
                                                 "p2v_gpu_canon"))
def test_p2v_gpu_canonical(impl, case_id):
    case = aux_case("golden_ext_p2v.pt", case_id)
    p2v = impl.putils.PointToVoxel(device=torch.device("cuda"),
                                   **case["ctor"])
    vox, idx, npv, _ = p2v.generate_voxel_with_id(case["points"].cuda())
    idx_c, vox_c, npv_c = canon_voxel_result(idx, vox, npv)
    expect = case["expect"]
    assert_tensor_equal(idx_c, expect["indices_canon"],
                        f"{case_id}: coords")
    assert_tensor_equal(npv_c, expect["num_per_voxel_canon"],
                        f"{case_id}: num_per_voxel")
    if expect["check_contents"]:
        assert_close(vox_c, expect["voxels_canon"], 1e-5,
                     f"{case_id}: contents")


def test_p2v_gpu_cap_structural(impl):
    """max_num_voxels cap on GPU: exactly cap voxels survive; dropped points
    get -1; surviving ids quantize to their voxel coords."""
    g = torch.Generator().manual_seed(3)
    pc = torch.rand(500, 3, generator=g).cuda()
    cap = 20
    p2v = impl.putils.PointToVoxel([0.05] * 3, [0, 0, 0, 1, 1, 1], 3, cap, 4,
                                   torch.device("cuda"))
    vox, idx, npv, pcvid = p2v.generate_voxel_with_id(pc)
    assert idx.shape[0] == cap
    assert int(pcvid.max()) == cap - 1
    assert int((pcvid < 0).sum()) > 0
    valid = pcvid >= 0
    q = (pc[valid] / 0.05).floor().long().flip(1)  # ZYX
    assert torch.equal(idx.long()[pcvid[valid]], q)


def test_p2v_truncation_keeps_voxel_id(impl):
    """Points dropped by max_points_per_voxel truncation still carry the
    voxel id (verified reference semantics); per-voxel id counts equal the
    TRUE (unclamped) point counts."""
    pc = torch.tensor([[0.05, 0.05, 0.05], [0.051, 0.05, 0.05],
                       [0.052, 0.05, 0.05], [0.3, 0.3, 0.3]]).cuda()
    p2v = impl.putils.PointToVoxel([0.1] * 3, [0, 0, 0, 1, 1, 1], 3, 10, 2,
                                   torch.device("cuda"))
    vox, idx, npv, pcvid = p2v.generate_voxel_with_id(pc)
    counts = torch.bincount(pcvid[pcvid >= 0], minlength=idx.shape[0])
    by_coord = {tuple(c.tolist()): int(n)
                for c, n in zip(idx.cpu(), counts.cpu())}
    assert by_coord[(0, 0, 0)] == 3  # true count, not the clamped 2
    assert by_coord[(3, 3, 3)] == 1
    npv_by_coord = {tuple(c.tolist()): int(n)
                    for c, n in zip(idx.cpu(), npv.cpu())}
    assert npv_by_coord[(0, 0, 0)] == 2  # clamped
