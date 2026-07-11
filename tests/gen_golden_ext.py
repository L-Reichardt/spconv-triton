"""Golden data generation for the EXTENSION suite, from UNCHANGED spconv.

Run once, with the reference implementation, on a GPU:

    SPCONV_TEST_IMPL=spconv uv run python tests_ext/gen_golden_ext.py

Reuses the frozen generator machinery from tests/gen_golden.py (which
asserts the reference implementation at import time).
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
sys.path.insert(0, str(Path(__file__).parent))

import gen_golden as G
import helpers
from helpers import canon_voxel_result, canonical_order

impl = G.impl
DEV = "cuda"
EXT_DATA_DIR = Path(__file__).parent / "data"
EXT_DATA_DIR.mkdir(parents=True, exist_ok=True)


def L(cls, **ctor):
    return {"cls": cls, "ctor": ctor}


def pipeline_case(
    cid,
    layers,
    inp,
    dtype="float32",
    training=True,
    alt=None,
    repeats=1,
    add_input=False,
):
    case = {
        "id": cid,
        "kind": "pipeline",
        "dtype": dtype,
        "training": training,
        "layers": layers,
        "input": inp,
        "add_input": add_input,
    }
    G.gen_params(cid, layers, dtype)
    algos = (alt,) if alt else ()
    G.finalize_case(case, alt_algos=algos, repeats=repeats)
    print(
        f"  [ext] {cid}: out={tuple(case['expect']['out_features'].shape)}"
        f" atol={case['expect']['atol_out']:.2e}"
    )
    return case


def conv_cases():
    cases = []
    # transpose / inverse in the previously untested dimensions
    cases.append(
        pipeline_case(
            "convt1d_k2s2",
            [
                L(
                    "SparseConvTranspose1d",
                    in_channels=6,
                    out_channels=12,
                    kernel_size=2,
                    stride=2,
                )
            ],
            G.gen_input("convt1d_k2s2", [64], 2, 6, 40),
            alt="Native",
        )
    )
    cases.append(
        pipeline_case(
            "convt4d_k2s2",
            [
                L(
                    "SparseConvTranspose4d",
                    in_channels=4,
                    out_channels=8,
                    kernel_size=2,
                    stride=2,
                )
            ],
            G.gen_input("convt4d_k2s2", [6, 12, 12, 12], 1, 4, 400),
            alt="Native",
        )
    )
    cases.append(
        pipeline_case(
            "inv1d_pair_conv",
            [
                L(
                    "SparseConv1d",
                    in_channels=6,
                    out_channels=12,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    indice_key="ds",
                ),
                L(
                    "SparseInverseConv1d",
                    in_channels=12,
                    out_channels=6,
                    kernel_size=3,
                    indice_key="ds",
                ),
            ],
            G.gen_input("inv1d_pair_conv", [64], 2, 6, 40),
            alt="Native",
        )
    )
    cases.append(
        pipeline_case(
            "inv4d_pair_conv",
            [
                L(
                    "SparseConv4d",
                    in_channels=4,
                    out_channels=8,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    indice_key="ds",
                ),
                L(
                    "SparseInverseConv4d",
                    in_channels=8,
                    out_channels=4,
                    kernel_size=3,
                    indice_key="ds",
                ),
            ],
            G.gen_input("inv4d_pair_conv", [6, 12, 12, 12], 1, 4, 400),
            repeats=2,
        )
    )  # kv=81 -> Native default on both layers
    # MaskSplitImplicitGemm on a REGULAR (non-subm) conv, fwd+bwd
    cases.append(
        pipeline_case(
            "conv3d_msplit_regular",
            [
                L(
                    "SparseConv3d",
                    in_channels=8,
                    out_channels=16,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    algo="MaskSplitImplicitGemm",
                )
            ],
            G.gen_input("conv3d_msplit_regular", [32, 32, 32], 2, 8, 1000),
            alt="Native",
        )
    )
    # act fusion on a regular conv, eval mode (base class)
    cases.append(
        pipeline_case(
            "conv3d_act_relu_regular_eval",
            [
                L(
                    "SparseConvolution",
                    ndim=3,
                    in_channels=8,
                    out_channels=16,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    subm=False,
                    act_type="ReLU",
                )
            ],
            G.gen_input("conv3d_act_relu_regular_eval", [32, 32, 32], 2, 8, 800),
            training=False,
            repeats=2,
        )
    )
    # k=1 with stride 2: kv==1 but NOT the conv1x1 fast path (pair-gen route)
    cases.append(
        pipeline_case(
            "conv2d_k1s2",
            [
                L(
                    "SparseConv2d",
                    in_channels=6,
                    out_channels=12,
                    kernel_size=1,
                    stride=2,
                )
            ],
            G.gen_input("conv2d_k1s2", [64, 64], 2, 6, 600),
            alt="Native",
        )
    )
    # stress scale: ~60k voxels
    cases.append(
        pipeline_case(
            "stress_subm3d_60k",
            [
                L(
                    "SubMConv3d",
                    in_channels=32,
                    out_channels=32,
                    kernel_size=3,
                    indice_key="s0",
                ),
                L(
                    "SparseConv3d",
                    in_channels=32,
                    out_channels=32,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ],
            G.gen_input(
                "stress_subm3d_60k", [96, 96, 96], 2, 32, 35000, mode="clustered"
            ),
            alt="Native",
        )
    )
    return cases


def pool_cases():
    cases = []
    cases.append(
        pipeline_case(
            "avgp3d_subm",
            [
                {
                    "cls": "SparseAvgPool",
                    "ctor": dict(ndim=3, kernel_size=3, stride=1, padding=1, subm=True),
                }
            ],
            G.gen_input("avgp3d_subm", [24, 24, 24], 2, 8, 900),
            repeats=2,
        )
    )
    cases.append(
        pipeline_case(
            "maxp3d_subm",
            [
                {
                    "cls": "SparseMaxPool",
                    "ctor": dict(ndim=3, kernel_size=3, stride=1, padding=1, subm=True),
                }
            ],
            G.gen_input("maxp3d_subm", [24, 24, 24], 2, 8, 900),
            repeats=2,
        )
    )
    cases.append(
        pipeline_case(
            "maxp3d_d2",
            [L("SparseMaxPool3d", kernel_size=3, stride=2, padding=2, dilation=2)],
            G.gen_input("maxp3d_d2", [24, 24, 24], 2, 8, 900),
            alt="Native",
        )
    )
    cases.append(
        pipeline_case(
            "maxp3d_k5_kv125",
            [L("SparseMaxPool3d", kernel_size=5, stride=2, padding=2)],
            G.gen_input("maxp3d_k5_kv125", [24, 24, 24], 2, 8, 1200),
            alt="Native",
        )
    )
    return cases


def ops_cases():
    cases = []
    ConvAlgo = impl.core.ConvAlgo
    # implicit-gemm pairs with kv=125 (multi-word masks)
    cid = "igemm_pairs_kv125"
    inp = G.gen_input(cid, [24, 24, 24], 2, 4, 700)
    indices = inp["indices"].to(DEV)
    args = dict(
        ksize=[5, 5, 5],
        stride=[2, 2, 2],
        padding=[2, 2, 2],
        dilation=[1, 1, 1],
        out_padding=[0, 0, 0],
        subm=False,
        transpose=False,
        is_train=True,
        algo="MaskImplicitGemm",
    )
    res = impl.ops.get_indice_pairs_implicit_gemm(
        indices,
        inp["batch_size"],
        [24, 24, 24],
        ConvAlgo.MaskImplicitGemm,
        args["ksize"],
        args["stride"],
        args["padding"],
        args["dilation"],
        args["out_padding"],
        False,
        False,
        True,
    )
    out_inds, npl, pair_fwd, pair_bwd, pm_fwd = res[0], res[1], res[2], res[3], res[4]
    out_ss = helpers.expected_out_shape([24, 24, 24], args)
    canon = helpers.canon_igemm_pairs(out_inds, out_ss, pair_fwd, pair_bwd)
    cases.append(
        {
            "id": cid,
            "kind": "igemm_pairs_kv125",
            "input": inp,
            "args": args,
            "expect": {
                "out_inds_canon": out_inds.cpu()[
                    canonical_order(out_inds.cpu(), out_ss)
                ],
                "indice_num_per_loc": npl.cpu(),
                "pair_fwd_canon": canon["pair_fwd"],
                "pair_bwd_canon": canon["pair_bwd"],
                "pm_fwd_shape": list(pm_fwd[0].shape),
            },
        }
    )
    print(f"  [ext] {cid}: nout={out_inds.shape[0]} pm_shape={list(pm_fwd[0].shape)}")
    return cases


def p2v_cases():
    cases = []
    from importlib import import_module

    putils = import_module(f"{impl.name}.pytorch.utils")

    def cpu_case(cid, vsize, crange, maxvox, maxpts, npts, ndim=3):
        g = torch.Generator().manual_seed(G.case_seed(cid))
        pc = torch.randn(npts, ndim + 1, generator=g) * 0.8
        p2v = putils.PointToVoxel(
            vsize, crange, ndim + 1, maxvox, maxpts, torch.device("cpu")
        )
        vox, idx, npv, pcvid = p2v.generate_voxel_with_id(pc)
        cases.append(
            {
                "id": cid,
                "kind": "p2v_cpu",
                "ndim": ndim,
                "ctor": {
                    "vsize_xyz": vsize,
                    "coors_range_xyz": crange,
                    "num_point_features": ndim + 1,
                    "max_num_voxels": maxvox,
                    "max_num_points_per_voxel": maxpts,
                },
                "points": pc,
                "expect": {
                    "voxels": vox.clone(),
                    "indices": idx.clone(),
                    "num_per_voxel": npv.clone(),
                    "pc_voxel_id": pcvid.clone(),
                },
            }
        )
        print(
            f"  [ext] {cid}: nvox={idx.shape[0]} "
            f"maxnpv={int(npv.max()) if npv.numel() else 0}"
        )

    # CPU: deterministic FCFS -> bitwise golden incl. truncation and cap
    cpu_case("p2v_cpu_basic", [0.1, 0.1, 0.1], [-2, -2, -2, 2, 2, 2], 8000, 6, 3000)
    cpu_case("p2v_cpu_trunc", [0.4, 0.4, 0.4], [-2, -2, -2, 2, 2, 2], 2000, 2, 3000)
    cpu_case("p2v_cpu_cap", [0.25, 0.25, 0.25], [-2, -2, -2, 2, 2, 2], 50, 4, 3000)

    # GPU canonical goldens for the untested 1d / 4d dims
    for cid, ndim, vsize, crange, maxvox, maxpts, npts in [
        ("p2v1d_basic", 1, [0.05], [-2, 2], 200, 8, 1500),
        (
            "p2v4d_basic",
            4,
            [0.2, 0.2, 0.2, 0.2],
            [-2, -2, -2, -2, 2, 2, 2, 2],
            4000,
            4,
            3000,
        ),
    ]:
        g = torch.Generator().manual_seed(G.case_seed(cid))
        pc = torch.randn(npts, ndim + 1, generator=g) * 0.8
        p2v = putils.PointToVoxel(
            vsize, crange, ndim + 1, maxvox, maxpts, torch.device(DEV)
        )
        vox, idx, npv, _pcvid = p2v.generate_voxel_with_id(pc.to(DEV))
        idx_c, vox_c, npv_c = canon_voxel_result(idx, vox, npv)
        truncated = bool((npv_c >= maxpts).any())
        cases.append(
            {
                "id": cid,
                "kind": "p2v_gpu_canon",
                "ndim": ndim,
                "ctor": {
                    "vsize_xyz": vsize,
                    "coors_range_xyz": crange,
                    "num_point_features": ndim + 1,
                    "max_num_voxels": maxvox,
                    "max_num_points_per_voxel": maxpts,
                },
                "points": pc,
                "expect": {
                    "indices_canon": idx_c,
                    "voxels_canon": vox_c,
                    "num_per_voxel_canon": npv_c,
                    "check_contents": not truncated,
                },
            }
        )
        print(f"  [ext] {cid}: nvox={idx_c.shape[0]} truncated={truncated}")
    return cases


def misc_cases():
    cases = []
    # test_utils.generate_sparse_data: bitwise reproducible under a fixed
    # numpy seed (identical np.random call sequence)
    from importlib import import_module

    tu = import_module(f"{impl.name}.test_utils")
    np.random.seed(7)
    d = tu.generate_sparse_data([10, 12, 14], [50, 60], 5)
    cases.append(
        {
            "id": "test_utils_gsd",
            "kind": "test_utils",
            "seed": 7,
            "shape": [10, 12, 14],
            "num_points": [50, 60],
            "num_channels": 5,
            "expect": {k: torch.from_numpy(v.copy()) for k, v in d.items()},
        }
    )
    print("  [ext] test_utils_gsd done")

    # SAVED_WEIGHT_LAYOUT=RSKC: upstream hook double-permutes the weight, so
    # the load ALWAYS raises. Verify the reference indeed raises (subprocess
    # because the env var is read at import time); the live test then asserts
    # behavior parity for the implementation under test.
    import os
    import subprocess

    script = Path(__file__).parent / "_saved_layout_runner.py"
    env = dict(os.environ)
    env.update(
        {
            "SPCONV_SAVED_WEIGHT_LAYOUT": "RSKC",
            "SPCONV_TEST_IMPL": impl.name,
            "PYTHONPATH": str(Path(__file__).parent.parent / "tests"),
        }
    )
    res = subprocess.run(
        [sys.executable, str(script)], env=env, capture_output=True, text=True
    )
    assert res.returncode == 0 and "LOAD_RAISED" in res.stdout, (
        "reference no longer raises on RSKC layout?!",
        res.stdout,
        res.stderr[-500:],
    )
    print("  [ext] saved_weight_layout reference raises: confirmed")
    return cases


def main():
    torch.backends.cudnn.deterministic = True
    print("generating golden_ext_conv.pt ...")
    torch.save({"cases": conv_cases()}, EXT_DATA_DIR / "golden_ext_conv.pt")
    print("generating golden_ext_pool.pt ...")
    torch.save({"cases": pool_cases()}, EXT_DATA_DIR / "golden_ext_pool.pt")
    print("generating golden_ext_ops.pt ...")
    torch.save({"cases": ops_cases()}, EXT_DATA_DIR / "golden_ext_ops.pt")
    print("generating golden_ext_p2v.pt ...")
    torch.save({"cases": p2v_cases()}, EXT_DATA_DIR / "golden_ext_p2v.pt")
    print("generating golden_ext_misc.pt ...")
    torch.save({"cases": misc_cases()}, EXT_DATA_DIR / "golden_ext_misc.pt")
    print("ALL EXT GOLDEN DATA GENERATED")
    for f in sorted(EXT_DATA_DIR.iterdir()):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
