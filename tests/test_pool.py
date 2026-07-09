"""Pooling layers vs golden data."""

import pytest
import torch

from helpers import (DEVICE, build_layer, check_pipeline_case, golden_case,
                     golden_case_ids, make_sparse_tensor)

POOL_CASES = golden_case_ids("golden_pool.pt")


@pytest.mark.parametrize("case_id", POOL_CASES)
def test_pool_case(impl, case_id):
    check_pipeline_case(impl, golden_case("golden_pool.pt", case_id))


def test_maxpool_default_stride_attribute(impl):
    layer = impl.pytorch.SparseMaxPool3d(kernel_size=3)
    assert list(layer.stride) == [3, 3, 3]
    layer = impl.pytorch.SparseMaxPool3d(kernel_size=2, stride=1)
    assert list(layer.stride) == [1, 1, 1]
    assert list(layer.kernel_size) == [2, 2, 2]
    layer = impl.pytorch.SparseAvgPool2d(kernel_size=4)
    assert list(layer.stride) == [4, 4, 4][:2]


def test_subm_maxpool_k1_clamps_at_zero(impl):
    """spconv's max pool accumulator initializes at 0 (NOT -inf), so a
    subm k1 max pool equals max(features, 0). Verified reference quirk;
    the port must replicate it."""
    pool_mod = impl.pool
    layer = pool_mod.SparseMaxPool(3, kernel_size=1, stride=1, padding=0,
                                   subm=True).to(DEVICE)
    case = golden_case("golden_pool.pt", "maxp3d_k2s2")
    x, _ = make_sparse_tensor(impl, case["input"], DEVICE)
    with torch.no_grad():
        out = layer(x)
    assert torch.equal(out.indices, x.indices)
    assert torch.equal(out.features, torch.clamp(x.features, min=0))
    assert list(out.spatial_shape) == list(x.spatial_shape)


def test_maxpool_all_negative_window_clamps_at_zero(impl):
    """Both algos clamp all-negative windows to 0 (reference quirk)."""
    idx = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1], [0, 4, 4, 4],
                        [0, 4, 4, 5]], dtype=torch.int32, device=DEVICE)
    feats = torch.tensor([[-1.0], [-2.0], [3.0], [-0.5]], device=DEVICE)
    for algo in [impl.core.ConvAlgo.MaskImplicitGemm,
                 impl.core.ConvAlgo.Native]:
        x = impl.pytorch.SparseConvTensor(feats, idx, [8, 8, 8], 1)
        pool = impl.pytorch.SparseMaxPool3d(2, 2, algo=algo).to(DEVICE)
        with torch.no_grad():
            out = pool(x)
        order = out.indices.cpu()[:, 1].argsort()
        vals = out.features.cpu().view(-1)[order]
        assert torch.equal(vals, torch.tensor([0.0, 3.0])), (algo, vals)


def test_global_avgpool_quirk_shape(impl):
    """spconv's SparseGlobalAvgPool returns a [batch_size] tensor
    (upstream quirk: torch.mean(...)[0]). The port must replicate it."""
    case = golden_case("golden_pool.pt", "gavgpool")
    assert list(case["expect"]["out_features"].shape) == \
        [case["input"]["batch_size"]]


def test_global_maxpool_shape(impl):
    case = golden_case("golden_pool.pt", "gmaxpool")
    assert list(case["expect"]["out_features"].shape) == \
        [case["input"]["batch_size"], case["input"]["features"].shape[1]]


def test_maxpool_stores_indice_data(impl):
    case = golden_case("golden_pool.pt", "maxp3d_k2s2")
    spec = {"cls": "SparseMaxPool3d",
            "ctor": {**case["layers"][0]["ctor"], "indice_key": "mp"}}
    layer = build_layer(impl, spec, torch.float32, DEVICE)
    x, _ = make_sparse_tensor(impl, case["input"], DEVICE)
    with torch.no_grad():
        out = layer(x)
    data = out.find_indice_pair("mp")
    assert data is not None
    assert data.is_subm is False
    assert list(data.ksize) == [2, 2, 2]
