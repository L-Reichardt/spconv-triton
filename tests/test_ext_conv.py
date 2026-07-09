"""Extension conv coverage: untested dims, MaskSplit regular, act on regular
conv, k1s2, stress scale, N=0, num_out_act_bound, benchmark mode."""

import pytest
import torch

from helpers import (DEVICE, build_layer, check_pipeline_case,
                     linearize_indices, make_sparse_tensor)
from helpers_golden import aux_case, aux_case_ids

CONV_CASES = aux_case_ids("golden_ext_conv.pt")


@pytest.mark.parametrize("case_id", CONV_CASES)
def test_ext_conv_case(impl, case_id):
    check_pipeline_case(impl, aux_case("golden_ext_conv.pt", case_id))


def _empty_input(impl, ndim=3):
    feats = torch.zeros(0, 4, device=DEVICE)
    idx = torch.zeros(0, ndim + 1, dtype=torch.int32, device=DEVICE)
    return impl.pytorch.SparseConvTensor(feats, idx, [16] * ndim, 1)


def test_empty_input_subm_raises(impl):
    layer = impl.pytorch.SubMConv3d(4, 8, 3).to(DEVICE)
    with pytest.raises(Exception):
        with torch.no_grad():
            layer(_empty_input(impl))


def test_empty_input_regular_raises(impl):
    layer = impl.pytorch.SparseConv3d(4, 8, 3, stride=2).to(DEVICE)
    with pytest.raises(Exception):
        with torch.no_grad():
            layer(_empty_input(impl))


def _bound_input(impl):
    g = torch.Generator().manual_seed(11)
    c = torch.unique(torch.stack(
        [torch.randint(0, 16, (200,), generator=g) for _ in range(3)], 1),
        dim=0)
    idx = torch.cat([torch.zeros(c.shape[0], 1, dtype=torch.long), c],
                    1).int().to(DEVICE)
    return idx


def test_num_out_act_bound_native_ignored(impl):
    """Reference behavior: the Native pair-gen path IGNORES the bound."""
    idx = _bound_input(impl)
    full = impl.ops.get_indice_pairs(
        idx, 1, [16, 16, 16], impl.core.ConvAlgo.Native, [3, 3, 3],
        [2, 2, 2], [1, 1, 1], [1, 1, 1], [0, 0, 0], False, False)
    bounded = impl.ops.get_indice_pairs(
        idx, 1, [16, 16, 16], impl.core.ConvAlgo.Native, [3, 3, 3],
        [2, 2, 2], [1, 1, 1], [1, 1, 1], [0, 0, 0], False, False,
        num_out_act_bound=10)
    assert bounded[0].shape[0] == full[0].shape[0] > 10


def test_num_out_act_bound_igemm_enforced(impl):
    """The implicit-gemm path truncates to the bound; surviving outputs are
    a subset of the unbounded output set."""
    idx = _bound_input(impl)
    algo = impl.core.ConvAlgo.MaskImplicitGemm
    args = ([3, 3, 3], [2, 2, 2], [1, 1, 1], [1, 1, 1], [0, 0, 0], False,
            False, True)
    full = impl.ops.get_indice_pairs_implicit_gemm(
        idx, 1, [16, 16, 16], algo, *args)
    bounded = impl.ops.get_indice_pairs_implicit_gemm(
        idx, 1, [16, 16, 16], algo, *args, num_out_act_bound=10)
    assert bounded[0].shape[0] == 10
    assert bounded[2].shape == (27, 10)  # pair_fwd truncated too
    lin_full = set(linearize_indices(full[0].cpu(), [8, 8, 8]).tolist())
    lin_bound = set(linearize_indices(bounded[0].cpu(), [8, 8, 8]).tolist())
    assert lin_bound.issubset(lin_full)
    # gather entries must reference valid input rows
    pf = bounded[2].cpu()
    assert int(pf.max()) < idx.shape[0]


def _small_x(impl, benchmark=False):
    g = torch.Generator().manual_seed(13)
    c = torch.unique(torch.stack(
        [torch.randint(0, 16, (60,), generator=g) for _ in range(3)], 1),
        dim=0)
    idx = torch.cat([torch.zeros(c.shape[0], 1, dtype=torch.long), c],
                    1).int().to(DEVICE)
    feats = torch.randn(idx.shape[0], 4, generator=g).to(DEVICE)
    x = impl.pytorch.SparseConvTensor(feats, idx, [16, 16, 16], 1)
    x.benchmark = benchmark
    return x


def test_benchmark_requires_name(impl):
    layer = impl.pytorch.SubMConv3d(4, 8, 3).to(DEVICE)
    with pytest.raises(ValueError):
        with torch.no_grad():
            layer(_small_x(impl, benchmark=True))
    pool = impl.pytorch.SparseMaxPool3d(2, 2).to(DEVICE)
    with pytest.raises(ValueError):
        with torch.no_grad():
            pool(_small_x(impl, benchmark=True))


def test_benchmark_record_structure(impl):
    layer = impl.pytorch.SubMConv3d(4, 8, 3).to(DEVICE)
    layer.name = "c0"
    x = _small_x(impl, benchmark=True)
    with torch.no_grad():
        out = layer(x)
    rec = out.benchmark_record["c0"]
    assert rec["type"] == "SparseConvolution"
    assert sorted(rec.keys()) == sorted(
        ["type", "indice_gen_time", "time", "num_points", "num_out_points",
         "params"])
    assert rec["params"] == {
        "kernel_size": [3, 3, 3], "stride": [1, 1, 1],
        "padding": [0, 0, 0], "dilation": [1, 1, 1],
        "output_padding": [0, 0, 0], "subm": True, "transposed": False,
        "input_channels": 4, "out_channels": 8}
    assert rec["num_points"] == [x.indices.shape[0]]
    assert rec["num_out_points"] == [out.indices.shape[0]]
    assert len(rec["time"]) == 1 and len(rec["indice_gen_time"]) == 1


def test_saved_weight_layout_rskc_raises(impl):
    """Upstream quirk: the SPCONV_SAVED_WEIGHT_LAYOUT load hook
    double-permutes the weight, so loading an RSKC checkpoint ALWAYS raises
    a size-mismatch RuntimeError. Behavior replicated verbatim.
    (Subprocess: the env var is read at import time.)"""
    import os
    import subprocess
    import sys
    from pathlib import Path

    runner = Path(__file__).parent / "_saved_layout_runner.py"
    env = dict(os.environ)
    env.update({"SPCONV_SAVED_WEIGHT_LAYOUT": "RSKC",
                "SPCONV_TEST_IMPL": impl.name,
                "PYTHONPATH": str(Path(__file__).parent.parent / "tests")})
    res = subprocess.run([sys.executable, str(runner)], env=env,
                         capture_output=True, text=True)
    assert res.returncode == 0 and "LOAD_RAISED" in res.stdout, \
        (res.stdout, res.stderr[-2000:])


def test_kernel_determinism_of_port(impl):
    """STRONGER property than reference parity: the port's conv kernels are
    deterministic (no atomics). Skipped for the reference implementation,
    whose default algo is nondeterministic by design."""
    if impl.name == "spconv":
        pytest.skip("reference implementation is nondeterministic")
    case = aux_case("golden_ext_conv.pt", "conv3d_msplit_regular")

    def run():
        layer = build_layer(impl, case["layers"][0], torch.float32, DEVICE)
        layer.train()
        x, feats = make_sparse_tensor(impl, case["input"], DEVICE,
                                      requires_grad=True)
        out = layer(x)
        out.features.square().sum().backward()
        return (out.indices.cpu().clone(),
                out.features.detach().cpu().clone(),
                feats.grad.cpu().clone(),
                layer.weight.grad.cpu().clone())

    a, b = run(), run()
    for x, y, what in zip(a, b, ["indices", "features", "grad_in", "grad_w"]):
        assert torch.equal(x, y), f"port kernels nondeterministic: {what}"
