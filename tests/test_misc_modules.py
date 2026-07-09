"""Containers, tables, wrapper modules, sparse_add, HashTable."""

import pytest
import torch

from helpers import (DEVICE, assert_close, assert_tensor_equal, build_layer,
                     canonical_order, check_pipeline_case, golden_case,
                     make_sparse_tensor)


def test_seq2d_mixed_pipeline(impl):
    check_pipeline_case(impl, golden_case("golden_misc.pt", "seq2d_mixed"))


@pytest.mark.parametrize("case_id", ["tables_jointable", "tables_addtable"])
def test_tables_golden(impl, case_id):
    case = golden_case("golden_misc.pt", case_id)
    x, _ = make_sparse_tensor(impl, case["input"], DEVICE)
    branches = [build_layer(impl, s, torch.float32, DEVICE)
                for s in case["branch_specs"]]
    with torch.no_grad():
        outs = [b(x) for b in branches]
        table = getattr(impl.pytorch, case["table"])()
        out = table(outs)
    expect = case["expect"]
    order = canonical_order(out.indices.cpu(), out.spatial_shape)
    assert_tensor_equal(out.indices.cpu()[order], expect["out_indices"],
                        f"{case_id}: indices")
    assert_close(out.features.cpu()[order], expect["out_features"],
                 expect["atol_out"], f"{case_id}: features")
    assert list(out.spatial_shape) == expect["out_spatial_shape"]


@pytest.mark.parametrize("case_id", ["sparse_add", "sparse_add_hash_based",
                                     "add_table_misaligned"])
def test_sparse_add_golden(impl, case_id):
    case = golden_case("golden_misc.pt", case_id)
    xa, _ = make_sparse_tensor(impl, case["inputs"][0], DEVICE)
    xb, _ = make_sparse_tensor(impl, case["inputs"][1], DEVICE)
    if case["fn"] == "sparse_add":
        out = impl.functional.sparse_add(xa, xb)
    elif case["fn"] == "sparse_add_hash_based":
        out = impl.functional.sparse_add_hash_based(xa, xb)
    else:
        out = impl.tables.AddTableMisaligned()([xa, xb])
    expect = case["expect"]
    order = canonical_order(out.indices.cpu(), out.spatial_shape)
    assert_tensor_equal(out.indices.cpu()[order], expect["out_indices"],
                        f"{case_id}: indices")
    assert_close(out.features.detach().cpu()[order], expect["out_features"],
                 expect["atol_out"], f"{case_id}: features")


def test_sparse_add_keeps_largest_indice_dict(impl):
    """When the result has as many rows as the largest operand, its
    indice_dict is retained."""
    case = golden_case("golden_misc.pt", "sparse_add")
    xa, _ = make_sparse_tensor(impl, case["inputs"][0], DEVICE)
    conv = impl.pytorch.SubMConv3d(8, 8, 3, indice_key="keep").to(DEVICE)
    with torch.no_grad():
        xa2 = conv(xa)
    # adding a tensor with identical indices -> result rows == xa2 rows
    xb = xa2.replace_feature(xa2.features * 0.5)
    out = impl.functional.sparse_add(xa2, xb)
    assert out.find_indice_pair("keep") is not None


def test_sequential_container_protocol(impl):
    sp = impl.pytorch
    from collections import OrderedDict
    seq = sp.SparseSequential(OrderedDict([
        ("c1", sp.SubMConv2d(4, 8, 3)),
        ("r1", torch.nn.ReLU()),
    ]))
    assert len(seq) == 2
    assert seq[0] is seq._modules["c1"]
    assert seq[1] is seq._modules["r1"]
    seq2 = sp.SparseSequential(conv=sp.SubMConv2d(4, 8, 3),
                               relu=torch.nn.ReLU())
    assert "conv" in seq2._modules
    seq2.add(torch.nn.Identity(), name="extra")
    assert "extra" in seq2._modules
    with pytest.raises(IndexError):
        _ = seq[5]


def test_sequential_dense_module_on_features(impl):
    case = golden_case("golden_misc.pt", "tables_jointable")
    x, _ = make_sparse_tensor(impl, case["input"], DEVICE)
    seq = impl.pytorch.SparseSequential(torch.nn.ReLU())
    with torch.no_grad():
        out = seq(x)
    assert torch.equal(out.features, torch.relu(x.features))
    assert out.indices is x.indices


def test_concat_table(impl):
    sp = impl.pytorch
    case = golden_case("golden_misc.pt", "tables_jointable")
    x, _ = make_sparse_tensor(impl, case["input"], DEVICE)
    ct = sp.ConcatTable()
    ct.add(sp.Identity())
    ct.add(sp.Identity())
    res = ct(x)
    assert isinstance(res, list) and len(res) == 2
    assert res[0] is x and res[1] is x


def test_jointable_mismatch_asserts(impl):
    case = golden_case("golden_misc.pt", "tables_jointable")
    x, _ = make_sparse_tensor(impl, case["input"], DEVICE)
    y = impl.pytorch.SparseConvTensor(
        torch.randn(3, x.features.shape[1], device=DEVICE),
        x.indices[:3], x.spatial_shape, x.batch_size)
    with pytest.raises(AssertionError):
        impl.pytorch.JoinTable()([x, y])
    with pytest.raises(AssertionError):
        impl.pytorch.AddTable()([x, y])


def test_wrapper_modules(impl):
    sp = impl.pytorch
    case = golden_case("golden_misc.pt", "tables_jointable")
    x, _ = make_sparse_tensor(impl, case["input"], DEVICE)
    relu = sp.SparseReLU()
    out = relu(x)
    assert torch.equal(out.features, torch.relu(x.features))
    dense_in = torch.randn(4, 5, device=DEVICE)
    assert torch.equal(relu(dense_in), torch.relu(dense_in))

    ident = sp.SparseIdentity()
    assert torch.equal(ident(x).features, x.features)

    bn = sp.SparseBatchNorm(x.features.shape[1]).to(DEVICE)
    bn.eval()
    out_bn = bn(x)
    ref = torch.nn.functional.batch_norm(
        x.features, bn.running_mean, bn.running_var, bn.weight, bn.bias,
        False, 0.1, bn.eps)
    assert torch.allclose(out_bn.features, ref, atol=1e-6)


def test_todense_removegrid_identity(impl):
    sp = impl.pytorch
    case = golden_case("golden_misc.pt", "dense_3d")
    x, _ = make_sparse_tensor(impl, case["input"], DEVICE)
    td = sp.ToDense()
    assert torch.equal(td(x), x.dense())
    rg = sp.RemoveGrid()
    y = rg(x.shadow_copy())
    assert y.grid is None
    ident = sp.Identity()
    assert ident(x) is x
    assert ident.input_spatial_size([4, 4]) == [4, 4]


def test_module_predicates(impl):
    sp = impl.pytorch
    conv = sp.SubMConv2d(2, 2, 3)
    assert impl.modules.is_spconv_module(conv)
    assert impl.modules.is_sparse_conv(conv)
    assert not impl.modules.is_sparse_conv(torch.nn.ReLU())
    assert impl.modules.is_spconv_module(sp.SparseReLU())
    assert not impl.modules.is_spconv_module(torch.nn.ReLU())


def test_assign_name_for_sparse_modules(impl):
    sp = impl.pytorch
    net = sp.SparseSequential(sp.SubMConv2d(2, 2, 3), torch.nn.ReLU())
    impl.pytorch.assign_name_for_sparse_modules(net)
    assert net._modules["0"]._sparse_unique_name == "0"


def test_hash_table_semantics(impl):
    ht = impl.hash.HashTable(torch.device(DEVICE), torch.int32, torch.int32,
                             max_size=128)
    keys = torch.tensor([5, 9, 42, 7], dtype=torch.int32, device=DEVICE)
    vals = torch.tensor([50, 90, 420, 70], dtype=torch.int32, device=DEVICE)
    ht.insert(keys, vals)
    q = torch.tensor([9, 5, 100, 42], dtype=torch.int32, device=DEVICE)
    out_v, is_empty = ht.query(q)
    assert out_v.cpu()[0] == 90
    assert out_v.cpu()[1] == 50
    assert out_v.cpu()[3] == 420
    assert bool(is_empty.cpu()[2])
    assert not bool(is_empty.cpu()[0])


def test_hash_table_assign_arange(impl):
    ht = impl.hash.HashTable(torch.device(DEVICE), torch.int32, torch.int32,
                             max_size=64)
    keys = torch.tensor([3, 11, 200], dtype=torch.int32, device=DEVICE)
    ht.insert(keys)
    cnt = ht.assign_arange_()
    assert int(cnt.item()) == 3
    out_v, is_empty = ht.query(keys)
    got = sorted(out_v.cpu().tolist())
    assert got == [0, 1, 2]
