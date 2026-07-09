"""PyTorch module protocol: init, state_dict, dtype casts, save/load, AMP."""

import io
import pickle

import pytest
import torch

from helpers import (DEVICE, assert_close, assert_sparse_output,
                     assert_tensor_equal, build_layer, golden_case,
                     load_golden, make_sparse_tensor)


def proto_cases(kind):
    return [c["id"] for c in load_golden("golden_proto.pt")["cases"]
            if c["kind"] == kind]


@pytest.mark.parametrize("case_id", proto_cases("init"))
def test_seeded_init_bitwise(impl, case_id):
    """Constructing a layer under a fixed seed must reproduce spconv's
    parameter initialization bit-for-bit (same RNG call sequence + bounds)."""
    case = golden_case("golden_proto.pt", case_id)
    torch.manual_seed(case["seed"])
    layer = getattr(impl.pytorch, case["cls"])(**case["ctor"])
    assert list(layer.weight.shape) == case["expect"]["weight_shape"]
    assert_tensor_equal(layer.weight.detach(), case["expect"]["weight"],
                        f"{case_id}: weight")
    if case["expect"]["bias"] is None:
        assert layer.bias is None
    else:
        assert_tensor_equal(layer.bias.detach(), case["expect"]["bias"],
                            f"{case_id}: bias")


@pytest.mark.parametrize("case_id", proto_cases("state_dict_keys"))
def test_state_dict_keys(impl, case_id):
    case = golden_case("golden_proto.pt", case_id)
    layer = getattr(impl.pytorch, case["cls"])(**case["ctor"])
    sd = layer.state_dict()
    assert sorted(sd.keys()) == case["expect"]["keys"]
    for k, shape in case["expect"]["shapes"].items():
        assert list(sd[k].shape) == shape, f"{case_id}: {k} shape"


@pytest.mark.parametrize("case_id", proto_cases("repr"))
def test_extra_repr(impl, case_id):
    case = golden_case("golden_proto.pt", case_id)
    layer = getattr(impl.pytorch, case["cls"])(**case["ctor"])
    assert layer.extra_repr() == case["expect"]["extra_repr"]


def test_pickle_roundtrip(impl):
    flags = golden_case("golden_proto.pt", "pickle_flags")["expect"]
    if flags["layer"]:
        layer = impl.pytorch.SubMConv3d(4, 8, 3)
        layer2 = pickle.loads(pickle.dumps(layer))
        assert torch.equal(layer.weight, layer2.weight)
    if flags["sptensor"]:
        feats = torch.randn(5, 3)
        idx = torch.zeros(5, 4, dtype=torch.int32)
        idx[:, 1] = torch.arange(5)
        x = impl.pytorch.SparseConvTensor(feats, idx, [8, 8, 8], 1)
        x2 = pickle.loads(pickle.dumps(x))
        assert torch.equal(x2.features, feats)
        assert torch.equal(x2.indices, idx)


def test_autocast_cast_inputs(impl):
    case = golden_case("golden_proto.pt", "autocast_subm3d")
    layer = build_layer(impl, case["layers"][0], torch.float32, DEVICE)
    layer.eval()
    x, _ = make_sparse_tensor(impl, case["input"], DEVICE)
    with torch.no_grad(), torch.autocast("cuda"):
        out = layer(x)
    assert str(out.features.dtype) == case["expect"]["out_dtype"]
    expected_dtype = getattr(torch,
                             case["expect"]["out_dtype"].split(".")[-1])
    assert_sparse_output(
        out,
        {"out_spatial_shape": list(x.spatial_shape),
         "out_indices": case["expect"]["out_indices"],
         "out_features": case["expect"]["out_features"],
         "atol_out": case["expect"]["atol_out"]},
        expected_dtype, msg="autocast")


def test_autocast_train_eval_dtype_matrix(impl):
    """Reference behavior on torch 2.12: under autocast, training-mode conv
    outputs are fp16 (custom_fwd cast, in-place bias add keeps fp16) while
    eval-mode outputs stay fp32 (inference path bypasses the autograd
    Function)."""
    torch.manual_seed(0)
    idx = torch.cat([torch.zeros(50, 1), torch.randint(0, 16, (50, 3))], 1)
    idx = torch.unique(idx, dim=0).int().to(DEVICE)
    feats = torch.randn(idx.shape[0], 8, device=DEVICE)
    for bias in [True, False]:
        for train, want in [(True, torch.float16), (False, torch.float32)]:
            layer = impl.pytorch.SubMConv3d(8, 16, 3, bias=bias).to(DEVICE)
            layer.train(train)
            x = impl.pytorch.SparseConvTensor(feats, idx, [16, 16, 16], 1)
            with torch.autocast("cuda"):
                out = layer(x)
            assert out.features.dtype == want, \
                f"bias={bias} train={train}: {out.features.dtype} != {want}"


def test_dtype_cast_roundtrip(impl):
    layer = impl.pytorch.SubMConv3d(4, 8, 3).to(DEVICE)
    w = layer.weight.detach().clone()
    layer.half()
    assert layer.weight.dtype == torch.float16
    assert layer.bias.dtype == torch.float16
    layer.float()
    assert layer.weight.dtype == torch.float32
    assert torch.allclose(layer.weight.detach(), w, atol=1e-3)
    layer.to(torch.float16)
    assert layer.weight.dtype == torch.float16
    layer.cpu()
    assert layer.weight.device.type == "cpu"
    layer.cuda()
    assert layer.weight.device.type == "cuda"


def test_half_cast_forward_matches_fp16_golden(impl):
    """Casting an fp32-constructed layer to .half() must reproduce the fp16
    golden case (parameters stored in fp16)."""
    case = golden_case("golden_conv.pt", "subm3d_fp16")
    layer = build_layer(impl, case["layers"][0], torch.float16, DEVICE)
    layer.eval()
    x, _ = make_sparse_tensor(impl, case["input"], DEVICE)
    with torch.no_grad():
        out = layer(x)
    assert_sparse_output(out, case["expect"], torch.float16,
                         msg="half-cast forward")


def test_state_dict_save_load_forward(impl, tmp_path):
    case = golden_case("golden_conv.pt", "subm3d_k3")
    layer = build_layer(impl, case["layers"][0], torch.float32, DEVICE)
    layer.eval()
    path = tmp_path / "ckpt.pt"
    torch.save(layer.state_dict(), path)
    spec_fresh = {"cls": case["layers"][0]["cls"],
                  "ctor": case["layers"][0]["ctor"]}
    fresh = build_layer(impl, spec_fresh, torch.float32, DEVICE)
    missing, unexpected = fresh.load_state_dict(
        torch.load(path, weights_only=True))
    assert not missing and not unexpected
    x, _ = make_sparse_tensor(impl, case["input"], DEVICE)
    with torch.no_grad():
        a = layer(x)
        x2, _ = make_sparse_tensor(impl, case["input"], DEVICE)
        b = fresh(x2)
    assert torch.equal(a.indices, b.indices)
    assert torch.equal(a.features, b.features)


def test_record_voxel_count_ckpt_injection(impl):
    """Loading a checkpoint without the voxel-count buffer into a layer with
    record_voxel_count=True must not fail (pre-hook injects it)."""
    plain = impl.pytorch.SparseConv3d(4, 8, 3)
    sd = plain.state_dict()
    rvc = impl.pytorch.SparseConv3d(4, 8, 3, record_voxel_count=True)
    missing, unexpected = rvc.load_state_dict(sd, strict=True)
    assert not missing and not unexpected


def test_optimizer_smoke(impl):
    case = golden_case("golden_conv.pt", "subm3d_k3")
    layer = build_layer(impl, case["layers"][0], torch.float32, DEVICE)
    layer.train()
    opt = torch.optim.SGD(layer.parameters(), lr=0.1)
    before = layer.weight.detach().clone()
    x, _ = make_sparse_tensor(impl, case["input"], DEVICE)
    out = layer(x)
    loss = out.features.square().mean()
    loss.backward()
    opt.step()
    after = layer.weight.detach()
    assert torch.isfinite(after).all()
    assert not torch.equal(before, after)


def test_once_differentiable_backward(impl):
    """Double differentiation through the conv op must raise (backward is
    decorated once_differentiable), matching spconv."""
    case = golden_case("golden_conv.pt", "subm3d_k3")
    layer = build_layer(impl, case["layers"][0], torch.float32, DEVICE)
    layer.train()
    x, feats = make_sparse_tensor(impl, case["input"], DEVICE,
                                  requires_grad=True)
    out = layer(x)
    loss = out.features.square().mean()
    g = torch.autograd.grad(loss, feats, create_graph=True)[0]
    with pytest.raises(RuntimeError):
        g.square().mean().backward()


def test_module_in_nn_sequential_print(impl):
    """repr(module) must not raise and contains the layer class name."""
    layer = impl.pytorch.SparseConv3d(4, 8, 3, stride=2, padding=1,
                                      bias=False)
    s = repr(layer)
    assert "SparseConv3d" in s


def test_named_buffers_record_voxel_count(impl):
    layer = impl.pytorch.SparseConv3d(4, 8, 3, record_voxel_count=True)
    names = [n for n, _ in layer.named_buffers()]
    assert "max_num_voxels_during_training" in names
    layer2 = impl.pytorch.SparseConv3d(4, 8, 3)
    assert layer2.get_max_num_voxels() is None
