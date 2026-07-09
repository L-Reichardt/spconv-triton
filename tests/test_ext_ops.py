"""Extension ops coverage: kv>32 implicit-gemm pair generation."""

import torch

from helpers import (assert_tensor_equal, canon_igemm_pairs, canonical_order,
                     expected_out_shape)
from helpers_golden import aux_case


def test_igemm_pairs_kv125(impl):
    case = aux_case("golden_ext_ops.pt", "igemm_pairs_kv125")
    inp, args = case["input"], case["args"]
    ss = inp["spatial_shape"]
    indices = inp["indices"].cuda()
    res = impl.ops.get_indice_pairs_implicit_gemm(
        indices, inp["batch_size"], ss,
        getattr(impl.core.ConvAlgo, args["algo"]), args["ksize"],
        args["stride"], args["padding"], args["dilation"],
        args["out_padding"], args["subm"], args["transpose"],
        args["is_train"])
    out_inds, npl, pair_fwd, pair_bwd, pm_fwd = res[0], res[1], res[2], \
        res[3], res[4]
    expect = case["expect"]
    out_ss = expected_out_shape(ss, args)
    order = canonical_order(out_inds.cpu(), out_ss)
    assert_tensor_equal(out_inds.cpu()[order], expect["out_inds_canon"],
                        "kv125 out_inds")
    assert_tensor_equal(npl.cpu(), expect["indice_num_per_loc"],
                        "kv125 indice_num_per_loc")
    canon = canon_igemm_pairs(out_inds, out_ss, pair_fwd, pair_bwd)
    assert_tensor_equal(canon["pair_fwd"], expect["pair_fwd_canon"],
                        "kv125 pair_fwd")
    assert_tensor_equal(canon["pair_bwd"], expect["pair_bwd_canon"],
                        "kv125 pair_bwd")
    # multi-word mask layout: ceil(125/32) = 4 int32 words per row
    assert list(pm_fwd[0].shape) == expect["pm_fwd_shape"]
    assert pm_fwd[0].dtype == torch.int32


def test_pair_gen_determinism_of_port(impl):
    """Port-only stronger property: pair generation is fully deterministic
    including mask/argsort outputs."""
    import pytest
    if impl.name == "spconv":
        pytest.skip("reference pair-gen output order is hash-dependent")
    case = aux_case("golden_ext_ops.pt", "igemm_pairs_kv125")
    inp, args = case["input"], case["args"]
    indices = inp["indices"].cuda()

    def run():
        res = impl.ops.get_indice_pairs_implicit_gemm(
            indices, inp["batch_size"], inp["spatial_shape"],
            getattr(impl.core.ConvAlgo, args["algo"]), args["ksize"],
            args["stride"], args["padding"], args["dilation"],
            args["out_padding"], args["subm"], args["transpose"], True)
        return [res[0].cpu(), res[2].cpu(), res[3].cpu(),
                res[4][0].cpu(), res[6][0].cpu()]

    a, b = run(), run()
    for x, y in zip(a, b):
        assert torch.equal(x, y)
