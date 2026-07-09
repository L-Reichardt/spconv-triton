"""Convolution layers: forward + backward vs golden data from unchanged spconv."""

import pytest
import torch

from helpers import (DEVICE, check_pipeline_case, golden_case,
                     golden_case_ids, build_layer, make_sparse_tensor)

CONV_CASES = golden_case_ids("golden_conv.pt")


@pytest.mark.parametrize("case_id", CONV_CASES)
def test_conv_case(impl, case_id):
    check_pipeline_case(impl, golden_case("golden_conv.pt", case_id))


def test_record_voxel_count(impl):
    case = golden_case("golden_conv.pt", "conv3d_k3s2p1")
    spec = dict(case["layers"][0])
    spec = {"cls": spec["cls"],
            "ctor": {**spec["ctor"], "record_voxel_count": True},
            "params": spec["params"]}
    layer = build_layer(impl, spec, torch.float32, DEVICE)
    layer.train()
    x, _ = make_sparse_tensor(impl, case["input"], DEVICE)
    with torch.no_grad():
        out = layer(x)
    n_out = case["expect"]["out_indices"].shape[0]
    buf = layer.get_max_num_voxels()
    assert buf is not None
    assert int(buf.item()) == n_out
    assert "max_num_voxels_during_training" in layer.state_dict()


def test_train_eval_equivalence(impl):
    """Train-mode (bias outside kernel) and eval-mode (bias fused) forward
    must agree within the case tolerance."""
    case = golden_case("golden_conv.pt", "subm3d_k3")
    layer = build_layer(impl, case["layers"][0], torch.float32, DEVICE)
    x, _ = make_sparse_tensor(impl, case["input"], DEVICE)
    layer.train()
    with torch.no_grad():
        out_train = layer(x)
    x2, _ = make_sparse_tensor(impl, case["input"], DEVICE)
    layer.eval()
    with torch.no_grad():
        out_eval = layer(x2)
    assert torch.equal(out_train.indices, out_eval.indices)
    err = (out_train.features.float()
           - out_eval.features.float()).abs().max().item()
    assert err <= max(case["expect"]["atol_out"], 1e-5), err


def test_indice_dict_population(impl):
    """A keyed conv must populate indice_dict with reusable pair data."""
    case = golden_case("golden_conv.pt", "inv3d_pair_conv")
    layer = build_layer(impl, case["layers"][0], torch.float32, DEVICE)
    x, _ = make_sparse_tensor(impl, case["input"], DEVICE)
    with torch.no_grad():
        out = layer(x)
    key = case["layers"][0]["ctor"]["indice_key"]
    data = out.find_indice_pair(key)
    assert data is not None
    assert data.is_subm is False
    assert list(data.ksize) == [3, 3, 3]
    assert list(data.stride) == [2, 2, 2]
    assert list(data.padding) == [1, 1, 1]
    assert list(data.dilation) == [1, 1, 1]
    assert list(data.spatial_shape) == list(case["input"]["spatial_shape"])
    assert data.out_indices is out.indices
    # unkeyed tensor must not have it
    assert x.find_indice_pair(key) is None


def test_inverse_restores_input_indices(impl):
    """conv(k3s2) followed by its inverse must restore the exact input
    indices (same rows, same order)."""
    case = golden_case("golden_conv.pt", "inv3d_pair_conv")
    layers = [build_layer(impl, s, torch.float32, DEVICE)
              for s in case["layers"]]
    x, _ = make_sparse_tensor(impl, case["input"], DEVICE)
    with torch.no_grad():
        mid = layers[0](x)
        out = layers[1](mid)
    assert torch.equal(out.indices.cpu(), case["input"]["indices"])
    assert list(out.spatial_shape) == list(case["input"]["spatial_shape"])


def test_subm_keeps_indices_object(impl):
    case = golden_case("golden_conv.pt", "subm3d_k3")
    layer = build_layer(impl, case["layers"][0], torch.float32, DEVICE)
    x, _ = make_sparse_tensor(impl, case["input"], DEVICE)
    with torch.no_grad():
        out = layer(x)
    assert out.indices is x.indices
    assert list(out.spatial_shape) == list(x.spatial_shape)
