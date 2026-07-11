"""Edge and error behavior of conv layers (matches unchanged spconv)."""

import pytest
import torch

from helpers import DEVICE, build_layer, golden_case, make_sparse_tensor


def test_groups_not_supported(impl):
    with pytest.raises(AssertionError):
        impl.pytorch.SubMConv3d(8, 16, 3, groups=2)


def test_conv1x1_padding_asserts(impl):
    with pytest.raises(AssertionError):
        impl.pytorch.SparseConv3d(8, 16, 1, stride=1, padding=1)


def test_channel_mismatch_raises(impl):
    layer = impl.pytorch.SubMConv3d(8, 16, 3).to(DEVICE)
    feats = torch.randn(10, 4, device=DEVICE)
    idx = torch.zeros(10, 4, dtype=torch.int32, device=DEVICE)
    idx[:, 1] = torch.arange(10)
    x = impl.pytorch.SparseConvTensor(feats, idx, [16, 16, 16], 1)
    with pytest.raises(AssertionError):
        layer(x)


def test_duplicate_indice_key_asserts(impl):
    case = golden_case("golden_conv.pt", "subm3d_k3")
    spec = case["layers"][0]
    keyed = {"cls": spec["cls"],
             "ctor": {**spec["ctor"], "indice_key": "dup"},
             "params": spec["params"]}
    conv_a = build_layer(impl, keyed, torch.float32, DEVICE)
    # same key, but NON-subm so the reuse path does not apply -> must assert
    bad = {"cls": "SparseConv3d",
           "ctor": dict(in_channels=16, out_channels=16, kernel_size=3,
                        stride=2, padding=1, indice_key="dup")}
    conv_b = build_layer(impl, bad, torch.float32, DEVICE)
    x, _ = make_sparse_tensor(impl, case["input"], DEVICE)
    with torch.no_grad():
        out = conv_a(x)
        with pytest.raises(AssertionError):
            conv_b(out)


def test_point_vanish_raises(impl):
    """All points stride out of the output grid -> ValueError."""
    layer = impl.pytorch.SparseConv1d(2, 4, kernel_size=1, stride=4).to(DEVICE)
    feats = torch.randn(1, 2, device=DEVICE)
    idx = torch.tensor([[0, 2]], dtype=torch.int32, device=DEVICE)
    x = impl.pytorch.SparseConvTensor(feats, idx, [4], 1)
    with pytest.raises(ValueError):
        layer(x)


def test_single_point_conv(impl):
    layer = impl.pytorch.SparseConv3d(4, 8, 3, stride=2, bias=False).to(DEVICE)
    feats = torch.randn(1, 4, device=DEVICE)
    idx = torch.tensor([[0, 5, 5, 5]], dtype=torch.int32, device=DEVICE)
    x = impl.pytorch.SparseConvTensor(feats, idx, [20, 20, 20], 1)
    with torch.no_grad():
        out = layer(x)
    assert out.indices.shape[0] >= 1
    assert list(out.spatial_shape) == [9, 9, 9]


def test_subm_even_kernel_rejected(impl):
    """spconv subm requires odd kernels (center must exist)."""
    layer = impl.pytorch.SubMConv3d(4, 8, 2).to(DEVICE)
    feats = torch.randn(10, 4, device=DEVICE)
    idx = torch.zeros(10, 4, dtype=torch.int32, device=DEVICE)
    idx[:, 1] = torch.arange(10)
    x = impl.pytorch.SparseConvTensor(feats, idx, [16, 16, 16], 1)
    with pytest.raises(Exception):
        with torch.no_grad():
            layer(x)


def test_add_input_eval_igemm_asserts_for_float(impl):
    """spconv's implicit-gemm inference path only fuses residual adds for
    int8; float add_input in eval mode raises. Replicated behavior."""
    layer = impl.pytorch.SubMConv3d(
        4, 4, 3, algo=impl.core.ConvAlgo.MaskImplicitGemm).to(DEVICE)
    layer.eval()
    feats = torch.randn(10, 4, device=DEVICE)
    idx = torch.zeros(10, 4, dtype=torch.int32, device=DEVICE)
    idx[:, 1] = torch.arange(10)
    x = impl.pytorch.SparseConvTensor(feats, idx, [16, 16, 16], 1)
    add = impl.pytorch.SparseConvTensor(torch.randn(10, 4, device=DEVICE),
                                        idx, [16, 16, 16], 1)
    with pytest.raises(AssertionError):
        with torch.no_grad():
            layer(x, add_input=add)


def test_is_inverseable(impl):
    assert impl.pytorch.SparseConv3d(2, 2, 2, indice_key="x").is_inverseable()
    assert not impl.pytorch.SparseConv3d(2, 2, 2).is_inverseable()
    assert not impl.pytorch.SubMConv3d(2, 2, 3, indice_key="x").is_inverseable()


def test_constants_attributes(impl):
    layer = impl.pytorch.SparseConv3d(4, 8, 3, stride=2, padding=1)
    assert layer.ndim == 3
    assert layer.in_channels == 4
    assert layer.out_channels == 8
    assert list(layer.kernel_size) == [3, 3, 3]
    assert list(layer.stride) == [2, 2, 2]
    assert list(layer.padding) == [1, 1, 1]
    assert list(layer.dilation) == [1, 1, 1]
    assert list(layer.output_padding) == [0, 0, 0]
    assert layer.groups == 1
    assert layer.subm is False
    assert layer.transposed is False
    assert layer.inverse is False
    assert layer.conv1x1 is False
    assert list(layer.weight_shape) == [8, 3, 3, 3, 4]
    assert layer.weight.shape == (8, 3, 3, 3, 4)
    assert layer.bias.shape == (8,)
    t = impl.pytorch.SubMConv3d(4, 8, 1)
    assert t.conv1x1 is True


def test_kernel_size_minus_one_global(impl):
    assert impl.ops.get_conv_output_size([13, 7], [-1, 3], [1, 1], [0, 0],
                                         [1, 1]) == [1, 5]
