"""Extension misc coverage: HashTable extras, test_utils, cppcore,
table protocol methods, extended API surface."""

import importlib

import numpy as np
import pytest
import torch

from helpers import DEVICE, assert_tensor_equal
from helpers_golden import aux_case


def test_hash_table_insert_exist_keys(impl):
    ht = impl.hash.HashTable(torch.device(DEVICE), torch.int32, torch.int32,
                             max_size=64)
    keys = torch.tensor([3, 7, 11], dtype=torch.int32, device=DEVICE)
    vals = torch.tensor([30, 70, 110], dtype=torch.int32, device=DEVICE)
    ht.insert(keys, vals)
    upd_keys = torch.tensor([7, 99], dtype=torch.int32, device=DEVICE)
    upd_vals = torch.tensor([777, 999], dtype=torch.int32, device=DEVICE)
    ht.insert_exist_keys(upd_keys, upd_vals)
    out, is_empty = ht.query(
        torch.tensor([3, 7, 11, 99], dtype=torch.int32, device=DEVICE))
    assert out.cpu()[0] == 30
    assert out.cpu()[1] == 777  # updated
    assert out.cpu()[2] == 110
    assert bool(is_empty.cpu()[3])  # 99 was never inserted


def test_hash_table_items(impl):
    """items() returns (keys, values, count) buffers; the first `count`
    entries are the stored pairs (any order)."""
    ht = impl.hash.HashTable(torch.device(DEVICE), torch.int32, torch.int32,
                             max_size=64)
    keys = torch.tensor([5, 1, 9], dtype=torch.int32, device=DEVICE)
    vals = torch.tensor([50, 10, 90], dtype=torch.int32, device=DEVICE)
    ht.insert(keys, vals)
    k, v, cnt = ht.items()
    n = int(cnt.cpu().item())
    assert n == 3
    k, v = k.cpu()[:n], v.cpu()[:n]
    order = torch.argsort(k.long())
    assert k[order].tolist() == [1, 5, 9]
    assert v[order].tolist() == [10, 50, 90]


def test_test_utils_generate_sparse_data(impl):
    """Bitwise-reproducible under a fixed numpy seed (identical np.random
    call sequence as the reference)."""
    case = aux_case("golden_ext_misc.pt", "test_utils_gsd")
    tu = importlib.import_module(f"{impl.name}.test_utils")
    np.random.seed(case["seed"])
    d = tu.generate_sparse_data(case["shape"], case["num_points"],
                                case["num_channels"])
    assert sorted(d.keys()) == sorted(case["expect"].keys())
    for k, expected in case["expect"].items():
        assert_tensor_equal(torch.from_numpy(d[k].copy()), expected,
                            f"generate_sparse_data[{k}]")


def test_test_utils_params_grid(impl):
    tu = importlib.import_module(f"{impl.name}.test_utils")
    assert tu.params_grid([1, 2], ["a", "b"]) == [
        [1, "a"], [1, "b"], [2, "a"], [2, "b"]]
    assert hasattr(tu, "TestCase")


def test_cppcore(impl):
    cc = importlib.import_module(f"{impl.name}.pytorch.cppcore")
    assert isinstance(cc.get_current_stream(), int)


def test_tables_input_spatial_size(impl):
    sp = impl.pytorch
    for cls in [sp.JoinTable, sp.AddTable, impl.tables.AddTableMisaligned]:
        assert cls().input_spatial_size([4, 5]) == [4, 5]
    ct = sp.ConcatTable()
    ct.add(sp.Identity())
    assert ct.input_spatial_size([4, 5]) == [4, 5]


EXT_SURFACE = {
    "constants": ["ALL_WEIGHT_IS_KRSC", "FILTER_HWIO", "SAVED_WEIGHT_LAYOUT",
                  "DISABLE_JIT", "EDITABLE_INSTALLED", "BOOST_ROOT",
                  "SPCONV_DEBUG_SAVE_PATH", "SPCONV_BWD_SPLITK",
                  "SPCONV_ALLOW_TF32", "SPCONV_USE_DIRECT_TABLE",
                  "SPCONV_DIRECT_TABLE_HASH_SIZE_SCALE", "SPCONV_DO_SORT",
                  "SPCONV_INT8_DEBUG", "AllocKeys", "PACKAGE_ROOT"],
    "core": ["ConvAlgo", "AlgoHint", "NDIM_DONT_CARE"],
    "tools": ["CUDAKernelTimer", "CPU_ONLY_BUILD", "nullcontext"],
    "utils": ["CPU_ONLY_BUILD", "nullcontext",
              "Point2VoxelCPU1d", "Point2VoxelCPU2d", "Point2VoxelCPU3d",
              "Point2VoxelCPU4d", "Point2VoxelGPU1d", "Point2VoxelGPU2d",
              "Point2VoxelGPU3d", "Point2VoxelGPU4d"],
    "test_utils": ["TestCase", "params_grid", "generate_sparse_data"],
    "pytorch.cppcore": ["get_current_stream"],
    "pytorch.ops": ["ALL_WEIGHT_IS_KRSC", "FILTER_HWIO", "INT32_MAX",
                    "AlgoHint", "CPU_ONLY_BUILD", "get_current_stream",
                    "nullcontext", "SPCONV_DO_SORT",
                    "SPCONV_USE_DIRECT_TABLE", "AllocKeys", "DEBUG",
                    "DEBUG_INT64_HASH_K", "fused_indice_conv"],
    "pytorch.conv": ["SPCONV_VERSION_NUMBERS", "nullcontext", "FILTER_HWIO",
                     "DEFAULT_SPARSE_CONV_TYPES", "SparseConvolution"],
    "pytorch.pool": ["ALL_POOL_LAYERS", "SparseMaxPool", "SparseAvgPool"],
    "pytorch.modules": ["SparseSyncBatchNorm", "PrintTensorMeta",
                        "PrintCurrentTime", "is_spconv_module",
                        "is_sparse_conv"],
    "pytorch.tables": ["AddTableMisaligned"],
    "pytorch.spatial": ["RemoveDuplicate"],
}


@pytest.mark.parametrize("module", sorted(EXT_SURFACE.keys()))
def test_extended_api_surface(impl, module):
    mod = importlib.import_module(f"{impl.name}.{module}")
    missing = [n for n in EXT_SURFACE[module] if not hasattr(mod, n)]
    assert not missing, f"{impl.name}.{module} missing: {missing}"
