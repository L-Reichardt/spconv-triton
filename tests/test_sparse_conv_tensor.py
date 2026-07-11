"""SparseConvTensor semantics."""

import pytest
import torch

from helpers import (DEVICE, assert_tensor_equal, golden_case,
                     make_sparse_tensor)


def small_input():
    return golden_case("golden_misc.pt", "dense_3d")["input"]


def test_ctor_asserts(impl):
    feats = torch.randn(5, 3, device=DEVICE)
    idx_i64 = torch.zeros(5, 4, dtype=torch.int64, device=DEVICE)
    with pytest.raises(AssertionError):
        impl.pytorch.SparseConvTensor(feats, idx_i64, [4, 4, 4], 1)
    idx = idx_i64.int()
    with pytest.raises(AssertionError):
        impl.pytorch.SparseConvTensor(feats, idx, [4, 4], 1)  # ndim mismatch
    with pytest.raises(AssertionError):
        impl.pytorch.SparseConvTensor(feats, idx, [4, 4, 4], 0)  # batch 0
    with pytest.raises(AssertionError):
        impl.pytorch.SparseConvTensor(feats.unsqueeze(0), idx, [4, 4, 4], 1)


def test_features_setter_raises(impl):
    x, _ = make_sparse_tensor(impl, small_input(), DEVICE)
    with pytest.raises(ValueError):
        x.features = torch.randn(3, 3)


def test_replace_feature(impl):
    x, _ = make_sparse_tensor(impl, small_input(), DEVICE)
    new_feats = torch.randn_like(x.features)
    y = x.replace_feature(new_feats)
    assert y is not x
    assert y.features is new_feats
    assert y.indices is x.indices
    assert y.spatial_shape == x.spatial_shape
    assert y.batch_size == x.batch_size
    assert y.indice_dict is x.indice_dict


def test_shadow_copy(impl):
    x, _ = make_sparse_tensor(impl, small_input(), DEVICE)
    y = x.shadow_copy()
    assert y is not x
    assert y.features is x.features
    assert y.indices is x.indices
    assert y.benchmark == x.benchmark


def test_add_operators(impl):
    x, _ = make_sparse_tensor(impl, small_input(), DEVICE)
    y = x.replace_feature(x.features * 2)
    z = x + y
    assert torch.allclose(z.features, x.features * 3)
    z2 = x + x.features
    assert torch.allclose(z2.features, x.features * 2)
    m = x.minus()
    assert torch.allclose(m.features, -x.features)


def test_select_by_index_upstream_bug(impl):
    """spconv 2.3.8's select_by_index assigns to the read-only `features`
    property and therefore always raises ValueError. Replicated as-is."""
    x, _ = make_sparse_tensor(impl, small_input(), DEVICE)
    sel = torch.arange(0, x.indices.shape[0], 2, device=DEVICE)
    with pytest.raises(ValueError):
        x.select_by_index(sel)


def test_spatial_size(impl):
    x, _ = make_sparse_tensor(impl, small_input(), DEVICE)
    assert int(x.spatial_size) == 8 * 12 * 10


def test_repr(impl):
    x, _ = make_sparse_tensor(impl, small_input(), DEVICE)
    n, c = x.features.shape
    assert repr(x) == f"SparseConvTensor[shape=torch.Size([{n}, {c}])]"


def test_dense_golden(impl):
    case = golden_case("golden_misc.pt", "dense_3d")
    x, _ = make_sparse_tensor(impl, case["input"], DEVICE)
    assert_tensor_equal(x.dense(True).cpu(), case["expect"]["dense_cf"],
                        "dense channels_first")
    assert_tensor_equal(x.dense(False).cpu(), case["expect"]["dense_cl"],
                        "dense channels_last")


def test_from_dense_golden(impl):
    case = golden_case("golden_misc.pt", "from_dense_3d")
    x = impl.pytorch.SparseConvTensor.from_dense(case["dense"].to(DEVICE))
    assert_tensor_equal(x.indices.cpu(), case["expect"]["indices"],
                        "from_dense indices")
    assert_tensor_equal(x.features.cpu(), case["expect"]["features"],
                        "from_dense features")
    assert list(x.spatial_shape) == case["expect"]["spatial_shape"]
    assert x.batch_size == case["expect"]["batch_size"]


def test_from_dense_dense_roundtrip(impl):
    case = golden_case("golden_misc.pt", "from_dense_3d")
    dense = case["dense"].to(DEVICE)
    x = impl.pytorch.SparseConvTensor.from_dense(dense)
    back = x.dense(channels_first=False)
    assert torch.equal(back, dense)


def test_scatter_nd_golden(impl):
    import importlib
    core_mod = importlib.import_module(f"{impl.name}.pytorch.core")
    case = golden_case("golden_misc.pt", "scatter_nd")
    out = core_mod.scatter_nd(case["indices"].to(DEVICE),
                              case["updates"].to(DEVICE), case["shape"])
    assert_tensor_equal(out.cpu(), case["expect"]["out"], "scatter_nd")


def test_fx_proxyable_metaclass(impl):
    try:
        from torch.fx import ProxyableClassMeta
    except ImportError:
        pytest.skip("torch.fx unavailable")
    assert type(impl.pytorch.SparseConvTensor) is ProxyableClassMeta


def test_find_indice_pair_none(impl):
    x, _ = make_sparse_tensor(impl, small_input(), DEVICE)
    assert x.find_indice_pair(None) is None
    assert x.find_indice_pair("nope") is None
