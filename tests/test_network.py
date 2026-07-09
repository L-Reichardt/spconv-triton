"""End-to-end UNet-style network: forward+backward vs golden, including
loading a checkpoint that was produced by unchanged spconv (drop-in proof)."""

import pytest
import torch

from helpers import (DEVICE, assert_close, assert_tensor_equal,
                     build_unet3d, canonical_order, golden_case,
                     inverse_permutation, load_golden)

NET_CASES = [c["id"] for c in load_golden("golden_net.pt")["cases"]]


def run_net(impl, case, with_grad=True):
    dtype = getattr(torch, case["dtype"])
    net = build_unet3d(impl, in_channels=6, base=16).to(DEVICE)
    if dtype == torch.float16:
        net = net.half()
    # state_dict was produced by unchanged spconv -> drop-in checkpoint load
    missing, unexpected = net.load_state_dict(
        {k: v.to(DEVICE) for k, v in case["state_dict"].items()})
    assert not missing and not unexpected
    net.train()
    inp = case["input"]
    feats = inp["features"].to(DEVICE).clone().requires_grad_(with_grad)
    x = impl.pytorch.SparseConvTensor(feats, inp["indices"].to(DEVICE),
                                      list(inp["spatial_shape"]),
                                      int(inp["batch_size"]))
    out = net(x)
    return net, feats, out


@pytest.mark.parametrize("case_id", NET_CASES)
def test_unet_forward_backward(impl, case_id):
    case = golden_case("golden_net.pt", case_id)
    expect = case["expect"]
    net, feats, out = run_net(impl, case)
    assert list(out.spatial_shape) == list(expect["out_spatial_shape"])
    order = canonical_order(out.indices.cpu(), out.spatial_shape)
    assert_tensor_equal(out.indices.cpu()[order], expect["out_indices"],
                        f"{case_id}: out indices")
    assert_close(out.features.detach().cpu()[order], expect["out_features"],
                 expect["atol_out"], f"{case_id}: out features")

    inv = inverse_permutation(order.to(DEVICE))
    out.features.backward(
        expect["grad_out"].to(DEVICE)[inv].to(out.features.dtype))
    assert_close(feats.grad, expect["grad_input"],
                 expect["atol_grad_input"], f"{case_id}: grad_input")
    named = dict(net.named_parameters())
    expected_keys = set(expect["grad_params"].keys())
    actual_keys = {k for k, p in named.items() if p.requires_grad}
    assert expected_keys == actual_keys, (
        f"{case_id}: param name mismatch {expected_keys ^ actual_keys}")
    for k, ginfo in expect["grad_params"].items():
        assert named[k].grad is not None, f"{case_id}: {k} missing grad"
        assert_close(named[k].grad, ginfo["grad"], ginfo["atol"],
                     f"{case_id}: {k}.grad")


def test_unet_runs_twice_consistently(impl):
    """Two forward passes through a fresh network must produce the same
    canonicalized outputs (no cross-run cache pollution)."""
    case = golden_case("golden_net.pt", "unet3d_float32")
    _, _, out1 = run_net(impl, case, with_grad=False)
    _, _, out2 = run_net(impl, case, with_grad=False)
    o1 = canonical_order(out1.indices.cpu(), out1.spatial_shape)
    o2 = canonical_order(out2.indices.cpu(), out2.spatial_shape)
    assert torch.equal(out1.indices.cpu()[o1], out2.indices.cpu()[o2])
    err = (out1.features.detach().cpu()[o1]
           - out2.features.detach().cpu()[o2]).abs().max().item()
    assert err <= case["expect"]["atol_out"]


def test_unet_state_dict_keys_match_reference(impl):
    """The port's network must expose exactly the same state_dict keys as the
    spconv-built network (true drop-in checkpoints)."""
    case = golden_case("golden_net.pt", "unet3d_float32")
    net = build_unet3d(impl, in_channels=6, base=16)
    assert sorted(net.state_dict().keys()) == \
        sorted(case["state_dict"].keys())
