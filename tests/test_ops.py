"""Primitive ops vs golden data and structural properties."""

import pytest
import torch

from helpers import (DEVICE, assert_close, assert_tensor_equal,
                     canon_igemm_pairs, canon_native_pairs,
                     canon_voxel_result, canonical_order,
                     check_mask_properties, expected_out_shape, golden_case,
                     load_golden)


def ops_ids(kind):
    return [c["id"] for c in load_golden("golden_ops.pt")["cases"]
            if c["kind"] == kind]


# ---------------------------------------------------------------------------
# output-size helpers (pure functions, literal expectations)
# ---------------------------------------------------------------------------

def test_get_conv_output_size(impl):
    f = impl.ops.get_conv_output_size
    assert f([32, 32, 32], [3, 3, 3], [2, 2, 2], [1, 1, 1], [1, 1, 1]) == \
        [16, 16, 16]
    assert f([7, 8], [3, 3], [2, 2], [1, 1], [1, 1]) == [4, 4]
    assert f([24], [3], [1], [2], [2]) == [24]
    assert f([13, 7], [-1, 3], [1, 1], [0, 0], [1, 1]) == [1, 5]
    assert f([10], [2], [2], [0], [1]) == [5]


def test_get_deconv_output_size(impl):
    f = impl.ops.get_deconv_output_size
    assert f([16, 16, 16], [2, 2, 2], [2, 2, 2], [0, 0, 0], [1, 1, 1],
             [0, 0, 0]) == [32, 32, 32]
    assert f([16], [3], [2], [1], [1], [1]) == [32]
    with pytest.raises(ValueError):
        f([8], [-1], [1], [0], [1], [0])


def test_maximum_value_int_(impl):
    ten = torch.tensor([5], dtype=torch.int32, device=DEVICE)
    impl.ops.maximum_value_int_(ten, 9)
    assert int(ten.item()) == 9
    impl.ops.maximum_value_int_(ten, 3)
    assert int(ten.item()) == 9


# ---------------------------------------------------------------------------
# pair generation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case_id", ops_ids("native_pairs"))
def test_native_pairs(impl, case_id):
    case = golden_case("golden_ops.pt", case_id)
    inp, args = case["input"], case["args"]
    indices = inp["indices"].to(DEVICE)
    out_inds, pair, npl = impl.ops.get_indice_pairs(
        indices, inp["batch_size"], inp["spatial_shape"],
        impl.core.ConvAlgo.Native, args["ksize"], args["stride"],
        args["padding"], args["dilation"], args["out_padding"], args["subm"],
        args["transpose"])
    assert_tensor_equal(out_inds.cpu(), case["expect"]["out_inds"],
                        f"{case_id}: out_inds")
    assert_tensor_equal(npl.cpu(), case["expect"]["indice_num_per_loc"],
                        f"{case_id}: indice_num_per_loc")
    actual_pairs = canon_native_pairs(pair, npl)
    expected_pairs = case["expect"]["pairs_canon"]
    assert len(actual_pairs) == len(expected_pairs)
    for k, (a, e) in enumerate(zip(actual_pairs, expected_pairs)):
        assert_tensor_equal(a, e, f"{case_id}: pair offset {k}")


@pytest.mark.parametrize("case_id", ops_ids("igemm_pairs"))
def test_igemm_pairs(impl, case_id):
    case = golden_case("golden_ops.pt", case_id)
    inp, args = case["input"], case["args"]
    ss = inp["spatial_shape"]
    indices = inp["indices"].to(DEVICE)
    res = impl.ops.get_indice_pairs_implicit_gemm(
        indices, inp["batch_size"], ss,
        getattr(impl.core.ConvAlgo, args["algo"]), args["ksize"],
        args["stride"], args["padding"], args["dilation"],
        args["out_padding"], args["subm"], args["transpose"],
        args["is_train"])
    (out_inds, npl, pair_fwd, pair_bwd, pm_fwd, pm_bwd, ma_fwd, ma_bwd,
     masks) = res
    expect = case["expect"]
    out_ss = expected_out_shape(ss, args)
    order = canonical_order(out_inds.cpu(), out_ss)
    assert_tensor_equal(out_inds.cpu()[order], expect["out_inds_canon"],
                        f"{case_id}: out_inds")
    assert_tensor_equal(npl.cpu(), expect["indice_num_per_loc"],
                        f"{case_id}: indice_num_per_loc")
    has_bwd = (isinstance(pair_bwd, torch.Tensor) and pair_bwd.numel() > 0)
    assert has_bwd == expect["has_pair_bwd"], f"{case_id}: pair_bwd presence"
    canon = canon_igemm_pairs(out_inds, out_ss, pair_fwd,
                              pair_bwd if has_bwd else None)
    assert_tensor_equal(canon["pair_fwd"], expect["pair_fwd_canon"],
                        f"{case_id}: pair_fwd")
    if has_bwd:
        assert_tensor_equal(canon["pair_bwd"], expect["pair_bwd_canon"],
                            f"{case_id}: pair_bwd")
    assert len(pm_fwd) == expect["n_mask_splits"]
    assert len(ma_fwd) == expect["n_mask_splits"]
    for m_actual, m_expected in zip(masks, expect["masks"]):
        got = torch.from_numpy(m_actual.astype("int64"))
        assert_tensor_equal(got, m_expected, f"{case_id}: mask values")
    # structural mask properties (single-split, kv <= 32 cases)
    if expect["n_mask_splits"] == 1:
        check_mask_properties(pair_fwd, pm_fwd[0], ma_fwd[0])
        if has_bwd and len(pm_bwd) == 1:
            check_mask_properties(pair_bwd, pm_bwd[0], ma_bwd[0])


# ---------------------------------------------------------------------------
# direct functional calls
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case_id", ops_ids("func_conv"))
def test_func_conv(impl, case_id):
    case = golden_case("golden_ops.pt", case_id)
    t = case["tensors"]
    f = t["features"].to(DEVICE).clone().requires_grad_(True)
    w = t["filters"].to(DEVICE).clone().requires_grad_(True)
    pair = t["pair"].to(DEVICE)
    npl = t["npl"].to(DEVICE)
    n_out = t["num_activate_out"]
    if case["mode"] == "subm":
        out = impl.functional.indice_subm_conv(f, w, pair, npl, n_out,
                                               impl.core.ConvAlgo.Native)
    elif case["mode"] == "regular":
        out = impl.functional.indice_conv(f, w, pair, npl, n_out,
                                          impl.core.ConvAlgo.Native)
    else:
        out = impl.functional.indice_inverse_conv(f, w, pair, npl, n_out,
                                                  impl.core.ConvAlgo.Native)
    expect = case["expect"]
    assert_close(out, expect["out"], expect["atol_out"], f"{case_id}: out")
    out.backward(t["grad_out"].to(DEVICE))
    assert_close(f.grad, expect["grad_features"],
                 expect["atol_grad_features"], f"{case_id}: grad_features")
    assert_close(w.grad, expect["grad_filters"],
                 expect["atol_grad_filters"], f"{case_id}: grad_filters")


def test_func_maxpool_native(impl):
    case = golden_case("golden_ops.pt", "maxpool_native_direct")
    t = case["tensors"]
    f = t["features"].to(DEVICE).clone().requires_grad_(True)
    out = impl.functional.indice_maxpool(
        f, t["pair"].to(DEVICE), t["npl"].to(DEVICE), t["num_activate_out"])
    assert_close(out, case["expect"]["out"], 1e-6, "maxpool out")
    out.backward(t["grad_out"].to(DEVICE))
    assert_close(f.grad, case["expect"]["grad_features"], 1e-6,
                 "maxpool grad")


def test_func_maxpool_igemm(impl):
    case = golden_case("golden_ops.pt", "maxpool_igemm_direct")
    t = case["tensors"]
    f = t["features"].to(DEVICE).clone().requires_grad_(True)
    out = impl.functional.indice_maxpool_implicit_gemm(
        f, t["pair_fwd"].to(DEVICE), t["pair_bwd"].to(DEVICE),
        t["num_activate_out"])
    assert_close(out, case["expect"]["out"], 1e-6, "maxpool igemm out")
    out.backward(t["grad_out"].to(DEVICE))
    assert_close(f.grad, case["expect"]["grad_features"], 1e-6,
                 "maxpool igemm grad")


def test_func_avgpool_igemm(impl):
    case = golden_case("golden_ops.pt", "avgpool_igemm_direct")
    t = case["tensors"]
    f = t["features"].to(DEVICE).clone().requires_grad_(True)
    out = impl.functional.indice_avgpool_implicit_gemm(
        f, t["pair_fwd"].to(DEVICE), t["pair_bwd"].to(DEVICE),
        t["num_activate_out"], True)
    assert_close(out, case["expect"]["out"], 1e-5, "avgpool igemm out")
    out.backward(t["grad_out"].to(DEVICE))
    assert_close(f.grad, case["expect"]["grad_features"], 1e-5,
                 "avgpool igemm grad")


# ---------------------------------------------------------------------------
# global pool rearrange + point2voxel
# ---------------------------------------------------------------------------

def test_global_pool_rearrange(impl):
    case = golden_case("golden_ops.pt", "gpr")
    coords = case["input"]["indices"].to(DEVICE)
    out_indices, counts = impl.ops.global_pool_rearrange(
        coords, case["batch_size"])
    counts_cpu = counts.cpu()
    assert_tensor_equal(counts_cpu, case["expect"]["counts"], "gpr counts")
    for b in range(case["batch_size"]):
        rows = out_indices[b, :counts_cpu[b]].cpu().long()
        rows = torch.sort(rows).values
        assert_tensor_equal(rows, case["expect"]["rows_sorted"][b],
                            f"gpr rows batch {b}")


@pytest.mark.parametrize("case_id", ops_ids("p2v"))
def test_point_to_voxel(impl, case_id):
    case = golden_case("golden_ops.pt", case_id)
    p2v = impl.putils.PointToVoxel(device=torch.device(DEVICE),
                                   **case["ctor"])
    pc = case["points"].to(DEVICE)
    vox, idx, npv, pcvid = p2v.generate_voxel_with_id(
        pc, empty_mean=case["empty_mean"])
    idx_c, vox_c, npv_c = canon_voxel_result(idx, vox, npv)
    expect = case["expect"]
    assert_tensor_equal(idx_c, expect["indices_canon"],
                        f"{case_id}: voxel coords")
    assert_tensor_equal(npv_c, expect["num_per_voxel_canon"],
                        f"{case_id}: num_per_voxel")
    if expect["check_contents"]:
        assert_close(vox_c, expect["voxels_canon"], 1e-5,
                     f"{case_id}: voxel contents")
    if expect["check_pc_voxel_id"]:
        # every point with a valid id must quantize to its voxel's coords
        pcvid_cpu = pcvid.cpu()
        valid = pcvid_cpu >= 0
        ndim = case["ndim"]
        vsize = case["ctor"]["vsize_xyz"]
        cmin = case["ctor"]["coors_range_xyz"][:ndim]
        pts = case["points"][:, :ndim]
        q = torch.stack([((pts[:, d] - cmin[d]) / vsize[d]).floor().long()
                         for d in range(ndim)], 1)
        q_zyx = q.flip(1)
        assigned = idx.cpu().long()[pcvid_cpu[valid]]
        assert torch.equal(assigned, q_zyx[valid]), \
            f"{case_id}: pc_voxel_id maps to wrong voxel coords"


def test_point_to_voxel_call_alias(impl):
    """__call__ returns the first three results of generate_voxel_with_id."""
    case = golden_case("golden_ops.pt", "p2v3d_basic")
    p2v = impl.putils.PointToVoxel(device=torch.device(DEVICE),
                                   **case["ctor"])
    pc = case["points"].to(DEVICE)
    res = p2v(pc)
    assert len(res) == 3


def test_gather_features_by_pc_voxel_id(impl):
    feats = torch.arange(12, dtype=torch.float32,
                         device=DEVICE).reshape(4, 3)
    ids = torch.tensor([0, -1, 3, 1, -1], device=DEVICE)
    res = impl.putils.gather_features_by_pc_voxel_id(feats, ids)
    expected = torch.stack([feats[0], torch.zeros(3, device=DEVICE),
                            feats[3], feats[1],
                            torch.zeros(3, device=DEVICE)])
    assert torch.equal(res, expected)
    res9 = impl.putils.gather_features_by_pc_voxel_id(feats, ids,
                                                      invalid_value=9)
    assert torch.equal(res9[1], torch.full((3,), 9.0, device=DEVICE))
