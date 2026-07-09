"""Golden data generation from UNCHANGED spconv (reference implementation).

Run exactly once, with the reference implementation, on a GPU:

    SPCONV_TEST_IMPL=spconv uv run python tests/gen_golden.py

Tolerance calibration: for every float tensor we record
``atol = max(200 * noise, floor * max(|ref|max, 1))`` where ``noise`` is the
maximum deviation observed between the reference run and (a) the same case
computed with an alternative ConvAlgo and (b) a repeated run (spconv's default
algo is nondeterministic). floor = 1e-5 (fp32) / 1.5e-2 (fp16).
"""

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import helpers
from helpers import (DATA_DIR, canonical_order, canon_native_pairs,
                     canon_igemm_pairs, canon_voxel_result,
                     inverse_permutation, linearize_indices,
                     pipeline_named_params, run_pipeline_case)

impl = helpers.load_impl()
assert impl.name == "spconv", "golden data must come from unchanged spconv"
assert torch.cuda.is_available()

DEV = "cuda"
DATA_DIR.mkdir(parents=True, exist_ok=True)

FLOORS = {"float32": 1e-5, "float16": 1.5e-2}


def case_seed(case_id: str, salt: int = 0) -> int:
    return (zlib.crc32(case_id.encode()) + salt * 7919) % (2**31)


# ---------------------------------------------------------------------------
# input generation (CPU, fully seeded)
# ---------------------------------------------------------------------------

def gen_input(case_id, spatial_shape, batch_size, channels, npoints,
              dtype="float32", mode="uniform", empty_batches=(),
              quantize_feats=False, salt=0):
    # Batch coverage: every existing test runs at batch size 4 (bzyx, batch>1).
    # The per-call batch_size argument is overridden so the whole frozen suite
    # exercises batch-major coordinate linearization and per-batch separation
    # uniformly. empty_batches indices stay valid (all < 4).
    batch_size = 4
    g = torch.Generator().manual_seed(case_seed(case_id, salt))
    ndim = len(spatial_shape)
    all_coords = []
    for b in range(batch_size):
        if b in empty_batches:
            continue
        if mode == "uniform":
            c = torch.stack([torch.randint(0, s, (npoints,), generator=g)
                             for s in spatial_shape], 1)
        else:  # clustered: gaussian blobs, surface-like
            n_clusters = max(2, npoints // 200)
            centers = torch.stack(
                [torch.randint(0, s, (n_clusters,), generator=g).float()
                 for s in spatial_shape], 1)
            which = torch.randint(0, n_clusters, (npoints,), generator=g)
            spread = torch.randn(npoints, ndim, generator=g) * 3.0
            c = (centers[which] + spread).round().long()
            for d in range(ndim):
                c[:, d] = c[:, d].clamp(0, spatial_shape[d] - 1)
        c = torch.unique(c, dim=0)
        b_col = torch.full((c.shape[0], 1), b, dtype=torch.long)
        all_coords.append(torch.cat([b_col, c], 1))
    indices = torch.cat(all_coords).int()
    feats = torch.randn(indices.shape[0], channels, generator=g)
    if quantize_feats:
        feats = (feats * 2).round() * 0.5
    feats = feats.to(getattr(torch, dtype))
    return {"features": feats, "indices": indices,
            "spatial_shape": list(spatial_shape), "batch_size": batch_size}


def gen_params(case_id, layer_specs, dtype, salt=1):
    """Fill in 'params' for conv specs (KRSC weights, bias)."""
    g = torch.Generator().manual_seed(case_seed(case_id, salt))
    for spec in layer_specs:
        if spec["cls"] == "SparseSequential":
            gen_params(case_id, spec["children"], dtype,
                       salt + 13 + len(spec["children"]))
            continue
        params = {}
        if "Conv" in spec["cls"]:
            c = spec["ctor"]
            tmp = helpers.build_layer(impl, {**spec, "params": {}},
                                      torch.float32, "cpu")
            w = torch.empty(tmp.weight.shape)
            bound = 3.0 / max(float(torch.tensor(tmp.weight.shape[1:]).prod()),
                              1.0) ** 0.5
            w.uniform_(-bound, bound, generator=g)
            params["weight"] = w.to(getattr(torch, dtype))
            if tmp.bias is not None:
                b = torch.empty(tmp.weight.shape[0])
                b.uniform_(-bound, bound, generator=g)
                params["bias"] = b.to(getattr(torch, dtype))
        elif spec["cls"] == "nn.BatchNorm1d":
            nf = spec["ctor"]["num_features"]
            params["weight"] = torch.empty(nf).uniform_(0.6, 1.4, generator=g)
            params["bias"] = torch.empty(nf).uniform_(-0.2, 0.2, generator=g)
        spec["params"] = params


# ---------------------------------------------------------------------------
# pipeline execution -> canonical results
# ---------------------------------------------------------------------------

def execute(case, grad_canon=None):
    res = run_pipeline_case(impl, case, DEV)
    out, layers = res["out"], res["layers"]
    result = {}
    if case.get("returns_dense"):
        result["out_features"] = out.detach().cpu()
        result["out_shape"] = list(out.shape)
    else:
        order = canonical_order(out.indices.cpu(), out.spatial_shape)
        result["out_spatial_shape"] = list(out.spatial_shape)
        result["out_indices"] = out.indices.cpu()[order]
        result["out_features"] = out.features.detach().cpu()[order]
        result["order"] = order
    if case["training"] and grad_canon is not None:
        if case.get("returns_dense"):
            out.backward(grad_canon.to(DEV).to(out.dtype))
        else:
            inv = inverse_permutation(result["order"].to(DEV))
            out.features.backward(
                grad_canon.to(DEV)[inv].to(out.features.dtype))
        result["grad_input"] = res["feats_leaf"].grad.detach().cpu()
        result["grad_params"] = {
            k: p.grad.detach().cpu()
            for k, p in pipeline_named_params(layers).items()}
    torch.cuda.synchronize()
    return result


MULTIPLIERS = {"float32": 200.0, "float16": 500.0}


def atol_for(ref, alts, dtype):
    floor = FLOORS[dtype]
    ref64 = ref.detach().to(torch.float64)
    refmax = float(ref64.abs().max()) if ref.numel() else 1.0
    noise = 0.0
    for a in alts:
        noise = max(noise, float((a.detach().to(torch.float64) - ref64)
                                 .abs().max()))
    return float(max(MULTIPLIERS[dtype] * noise, floor * max(refmax, 1.0)))


def override_algo(case, algo_name):
    import copy
    c = copy.deepcopy(case)

    def _set(specs):
        for s in specs:
            if s["cls"] == "SparseSequential":
                _set(s["children"])
            elif "Conv" in s["cls"] or "Pool" in s["cls"]:
                s["ctor"]["algo"] = algo_name
    _set(c["layers"])
    return c


def finalize_case(case, alt_algos=("auto",), repeats=1):
    """Run reference + calibration runs, fill case['expect']."""
    dtype = case["dtype"]
    first = execute(case)
    grad_canon = None
    if case["training"]:
        g = torch.Generator().manual_seed(case_seed(case["id"], 2))
        if case.get("returns_dense"):
            shape = first["out_shape"]
        else:
            shape = list(first["out_features"].shape)
        grad_canon = torch.randn(*shape, generator=g).to(getattr(torch, dtype))
    ref = execute(case, grad_canon)

    alts = []
    for algo in alt_algos:
        if algo == "auto":
            continue
        alts.append(execute(override_algo(case, algo), grad_canon))
    for _ in range(repeats):
        alts.append(execute(case, grad_canon))

    if case.get("bitwise"):
        def tol(refT, key):
            return 0.0
    else:
        def tol(refT, key):
            return atol_for(refT, [a[key] for a in alts], dtype)

    expect = {"out_features": ref["out_features"],
              "atol_out": tol(ref["out_features"], "out_features")}
    if case.get("returns_dense"):
        expect["out_shape"] = ref["out_shape"]
    else:
        expect["out_spatial_shape"] = ref["out_spatial_shape"]
        expect["out_indices"] = ref["out_indices"]
        for a in alts:
            assert torch.equal(a["out_indices"], ref["out_indices"]), \
                f"{case['id']}: calibration runs disagree on indices!"
    if case["training"]:
        expect["grad_out"] = grad_canon
        expect["grad_input"] = ref["grad_input"]
        expect["atol_grad_input"] = tol(ref["grad_input"], "grad_input")
        expect["grad_params"] = {}
        for k, gref in ref["grad_params"].items():
            galts = [a["grad_params"][k] for a in alts]
            expect["grad_params"][k] = {
                "grad": gref,
                "atol": 0.0 if case.get("bitwise") else
                        atol_for(gref, galts, dtype)}
    case["expect"] = expect
    case.pop("bitwise", None)
    return case


# ---------------------------------------------------------------------------
# Conv cases
# ---------------------------------------------------------------------------

def conv_cases():
    cases = []

    def C(cid, layers, inp, dtype="float32", training=True, alt="auto2",
          add_input=False, bitwise=False, repeats=1):
        case = {"id": cid, "kind": "pipeline", "dtype": dtype,
                "training": training, "layers": layers, "input": inp,
                "add_input": add_input}
        if bitwise:
            case["bitwise"] = True
        gen_params(cid, layers, dtype)
        if alt == "auto2":
            algos = ("Native",) if all(
                l["ctor"].get("algo") is None for l in layers) else ()
        elif alt is None:
            algos = ()
        else:
            algos = (alt,)
        cases.append(finalize_case(case, alt_algos=algos, repeats=repeats))
        print(f"  [conv] {cid}: n_out={case['expect']['out_features'].shape}"
              f" atol_out={case['expect']['atol_out']:.2e}")

    def L(cls, **ctor):
        return {"cls": cls, "ctor": ctor}

    # --- SubM family ---
    C("subm1d_k3", [L("SubMConv1d", in_channels=6, out_channels=12,
                      kernel_size=3, padding=1)],
      gen_input("subm1d_k3", [64], 2, 6, 40))
    C("subm2d_k3", [L("SubMConv2d", in_channels=8, out_channels=16,
                      kernel_size=3, padding=1)],
      gen_input("subm2d_k3", [96, 96], 2, 8, 800, mode="clustered"))
    C("subm3d_k3", [L("SubMConv3d", in_channels=8, out_channels=16,
                      kernel_size=3, padding=1)],
      gen_input("subm3d_k3", [32, 32, 32], 2, 8, 1500, mode="clustered"))
    C("subm4d_k3", [L("SubMConv4d", in_channels=4, out_channels=8,
                      kernel_size=3, padding=1)],
      gen_input("subm4d_k3", [8, 16, 16, 16], 1, 4, 600), alt=None)
    C("subm3d_k133", [L("SubMConv3d", in_channels=8, out_channels=16,
                        kernel_size=(1, 3, 3), padding=(0, 1, 1))],
      gen_input("subm3d_k133", [32, 32, 32], 2, 8, 1200))
    C("subm3d_d2", [L("SubMConv3d", in_channels=8, out_channels=16,
                      kernel_size=3, dilation=2)],
      gen_input("subm3d_d2", [32, 32, 32], 2, 8, 1200))
    C("subm3d_nobias", [L("SubMConv3d", in_channels=8, out_channels=16,
                          kernel_size=3, bias=False)],
      gen_input("subm3d_nobias", [32, 32, 32], 2, 8, 1000))
    C("subm3d_c5_c7", [L("SubMConv3d", in_channels=5, out_channels=7,
                         kernel_size=3)],
      gen_input("subm3d_c5_c7", [24, 24, 24], 2, 5, 700))
    C("subm3d_fp16", [L("SubMConv3d", in_channels=16, out_channels=32,
                        kernel_size=3)],
      gen_input("subm3d_fp16", [32, 32, 32], 2, 16, 1500, dtype="float16"),
      dtype="float16")
    C("subm3d_share_key",
      [L("SubMConv3d", in_channels=8, out_channels=16, kernel_size=3,
         indice_key="sh"),
       L("SubMConv3d", in_channels=16, out_channels=16, kernel_size=3,
         indice_key="sh")],
      gen_input("subm3d_share_key", [32, 32, 32], 2, 8, 1000))
    C("subm2d_k13", [L("SubMConv2d", in_channels=8, out_channels=16,
                       kernel_size=(1, 3))],
      gen_input("subm2d_k13", [64, 64], 2, 8, 600))
    C("subm3d_c64", [L("SubMConv3d", in_channels=64, out_channels=64,
                       kernel_size=3)],
      gen_input("subm3d_c64", [64, 64, 64], 2, 64, 10000, mode="clustered"))
    C("subm3d_emptybatch", [L("SubMConv3d", in_channels=8, out_channels=16,
                              kernel_size=3)],
      gen_input("subm3d_emptybatch", [32, 32, 32], 3, 8, 800,
                empty_batches=(1,)))
    C("subm3d_native", [L("SubMConv3d", in_channels=8, out_channels=16,
                          kernel_size=3, algo="Native")],
      gen_input("subm3d_native", [32, 32, 32], 2, 8, 1000),
      alt="MaskImplicitGemm")
    C("subm3d_msplit", [L("SubMConv3d", in_channels=8, out_channels=16,
                          kernel_size=3, algo="MaskSplitImplicitGemm")],
      gen_input("subm3d_msplit", [32, 32, 32], 2, 8, 1000), alt="Native")
    C("subm3d_1x1", [L("SubMConv3d", in_channels=8, out_channels=16,
                       kernel_size=1)],
      gen_input("subm3d_1x1", [32, 32, 32], 2, 8, 1000),
      alt=None, bitwise=True)
    C("subm3d_1x1_fp16_nobias", [L("SubMConv3d", in_channels=8,
                                   out_channels=16, kernel_size=1,
                                   bias=False)],
      gen_input("subm3d_1x1_fp16_nobias", [32, 32, 32], 2, 8, 1000,
                dtype="float16"),
      dtype="float16", alt=None, bitwise=True)

    # --- regular conv family ---
    C("conv1d_k2s2", [L("SparseConv1d", in_channels=6, out_channels=12,
                        kernel_size=2, stride=2)],
      gen_input("conv1d_k2s2", [64], 2, 6, 40))
    C("conv2d_k3s2p1", [L("SparseConv2d", in_channels=8, out_channels=16,
                          kernel_size=3, stride=2, padding=1)],
      gen_input("conv2d_k3s2p1", [96, 96], 2, 8, 800, mode="clustered"))
    C("conv3d_k3s2p1", [L("SparseConv3d", in_channels=8, out_channels=16,
                          kernel_size=3, stride=2, padding=1)],
      gen_input("conv3d_k3s2p1", [32, 32, 32], 2, 8, 1500, mode="clustered"))
    C("conv3d_k2s2", [L("SparseConv3d", in_channels=8, out_channels=16,
                        kernel_size=2, stride=2)],
      gen_input("conv3d_k2s2", [32, 32, 32], 2, 8, 1200))
    C("conv4d_k2s2", [L("SparseConv4d", in_channels=4, out_channels=8,
                        kernel_size=2, stride=2)],
      gen_input("conv4d_k2s2", [8, 16, 16, 16], 1, 4, 600))
    C("conv3d_asym", [L("SparseConv3d", in_channels=8, out_channels=16,
                        kernel_size=(1, 3, 3), stride=(1, 2, 2),
                        padding=(0, 1, 1))],
      gen_input("conv3d_asym", [32, 32, 32], 2, 8, 1200))
    C("conv3d_d2", [L("SparseConv3d", in_channels=8, out_channels=16,
                      kernel_size=3, stride=1, padding=2, dilation=2)],
      gen_input("conv3d_d2", [24, 24, 24], 2, 8, 800))
    C("conv3d_fp16", [L("SparseConv3d", in_channels=16, out_channels=32,
                        kernel_size=3, stride=2, padding=1)],
      gen_input("conv3d_fp16", [32, 32, 32], 2, 16, 1500, dtype="float16"),
      dtype="float16")
    C("conv3d_native", [L("SparseConv3d", in_channels=8, out_channels=16,
                          kernel_size=3, stride=2, padding=1, algo="Native")],
      gen_input("conv3d_native", [32, 32, 32], 2, 8, 1000),
      alt="MaskImplicitGemm")
    C("conv3d_large", [L("SparseConv3d", in_channels=32, out_channels=48,
                         kernel_size=3, stride=2, padding=1)],
      gen_input("conv3d_large", [128, 128, 64], 2, 32, 16000,
                mode="clustered"))
    C("conv3d_s_mixed", [L("SparseConv3d", in_channels=8, out_channels=16,
                           kernel_size=3, stride=(2, 1, 1),
                           padding=(1, 0, 0))],
      gen_input("conv3d_s_mixed", [32, 32, 32], 2, 8, 1000))
    C("conv2d_1x1", [L("SparseConv2d", in_channels=8, out_channels=16,
                       kernel_size=1, stride=1)],
      gen_input("conv2d_1x1", [64, 64], 2, 8, 600), alt=None, bitwise=True)
    C("conv3d_emptybatch", [L("SparseConv3d", in_channels=8, out_channels=16,
                              kernel_size=3, stride=2, padding=1)],
      gen_input("conv3d_emptybatch", [32, 32, 32], 3, 8, 800,
                empty_batches=(1,)))
    C("conv3d_k5_native", [L("SparseConv3d", in_channels=4, out_channels=8,
                             kernel_size=5, stride=2, padding=2)],
      gen_input("conv3d_k5_native", [24, 24, 24], 1, 4, 500), alt=None)

    # --- transpose ---
    C("convt2d_k2s2", [L("SparseConvTranspose2d", in_channels=8,
                         out_channels=16, kernel_size=2, stride=2)],
      gen_input("convt2d_k2s2", [48, 48], 2, 8, 500))
    C("convt3d_k2s2", [L("SparseConvTranspose3d", in_channels=8,
                         out_channels=16, kernel_size=2, stride=2)],
      gen_input("convt3d_k2s2", [16, 16, 16], 2, 8, 600))
    C("convt3d_k3s2p1", [L("SparseConvTranspose3d", in_channels=8,
                           out_channels=16, kernel_size=3, stride=2,
                           padding=1)],
      gen_input("convt3d_k3s2p1", [16, 16, 16], 2, 8, 500))
    C("convt3d_fp16", [L("SparseConvTranspose3d", in_channels=16,
                         out_channels=16, kernel_size=2, stride=2)],
      gen_input("convt3d_fp16", [16, 16, 16], 2, 16, 600, dtype="float16"),
      dtype="float16")

    # --- inverse ---
    C("inv2d_pair_conv",
      [L("SparseConv2d", in_channels=8, out_channels=16, kernel_size=3,
         stride=2, padding=1, indice_key="ds"),
       L("SparseInverseConv2d", in_channels=16, out_channels=8,
         kernel_size=3, indice_key="ds")],
      gen_input("inv2d_pair_conv", [64, 64], 2, 8, 600))
    C("inv3d_pair_conv",
      [L("SparseConv3d", in_channels=8, out_channels=16, kernel_size=3,
         stride=2, padding=1, indice_key="ds"),
       L("SparseInverseConv3d", in_channels=16, out_channels=8,
         kernel_size=3, indice_key="ds")],
      gen_input("inv3d_pair_conv", [32, 32, 32], 2, 8, 1200,
                mode="clustered"))
    C("inv3d_pair_maxpool",
      [L("SparseMaxPool3d", kernel_size=2, stride=2, indice_key="mp"),
       L("SparseInverseConv3d", in_channels=8, out_channels=16,
         kernel_size=2, indice_key="mp")],
      gen_input("inv3d_pair_maxpool", [32, 32, 32], 2, 8, 1000))
    C("inv3d_fp16",
      [L("SparseConv3d", in_channels=16, out_channels=16, kernel_size=3,
         stride=2, padding=1, indice_key="ds"),
       L("SparseInverseConv3d", in_channels=16, out_channels=16,
         kernel_size=3, indice_key="ds")],
      gen_input("inv3d_fp16", [32, 32, 32], 2, 16, 1000, dtype="float16"),
      dtype="float16")
    C("inv3d_native",
      [L("SparseConv3d", in_channels=8, out_channels=16, kernel_size=3,
         stride=2, padding=1, indice_key="ds", algo="Native"),
       L("SparseInverseConv3d", in_channels=16, out_channels=8,
         kernel_size=3, indice_key="ds", algo="Native")],
      gen_input("inv3d_native", [32, 32, 32], 2, 8, 800),
      alt="MaskImplicitGemm")

    # --- eval-mode fusion (act_* only exists on the SparseConvolution base) ---
    C("subm3d_act_relu_eval", [L("SparseConvolution", ndim=3, in_channels=8,
                                 out_channels=16, kernel_size=3, subm=True,
                                 act_type="ReLU")],
      gen_input("subm3d_act_relu_eval", [32, 32, 32], 2, 8, 800),
      training=False, alt=None, repeats=2)
    C("subm3d_act_lrelu_eval", [L("SparseConvolution", ndim=3, in_channels=8,
                                  out_channels=16, kernel_size=3, subm=True,
                                  act_type="LeakyReLU", act_alpha=0.1)],
      gen_input("subm3d_act_lrelu_eval", [32, 32, 32], 2, 8, 800),
      training=False, alt=None, repeats=2)
    C("subm3d_act_sigmoid_eval", [L("SparseConvolution", ndim=3,
                                    in_channels=8, out_channels=16,
                                    kernel_size=3, subm=True,
                                    act_type="Sigmoid")],
      gen_input("subm3d_act_sigmoid_eval", [32, 32, 32], 2, 8, 800),
      training=False, alt=None, repeats=2)
    # float add_input is only supported on the Native eval path (the igemm
    # inference kernel asserts int8); replicated as-is.
    C("subm3d_addinput_eval", [L("SubMConv3d", in_channels=8,
                                 out_channels=16, kernel_size=3,
                                 algo="Native")],
      gen_input("subm3d_addinput_eval", [32, 32, 32], 2, 8, 800),
      training=False, add_input=True, alt=None, repeats=2)
    C("conv3d_eval_bias", [L("SparseConv3d", in_channels=8, out_channels=16,
                             kernel_size=3, stride=2, padding=1)],
      gen_input("conv3d_eval_bias", [32, 32, 32], 2, 8, 800),
      training=False)

    return cases


# ---------------------------------------------------------------------------
# Pool cases
# ---------------------------------------------------------------------------

def pool_cases():
    cases = []

    def C(cid, layers, inp, dtype="float32", training=True, alt="auto2",
          returns_dense=False, repeats=1):
        case = {"id": cid, "kind": "pipeline", "dtype": dtype,
                "training": training, "layers": layers, "input": inp,
                "add_input": False, "returns_dense": returns_dense}
        gen_params(cid, layers, dtype)
        if alt == "auto2":
            algos = ("Native",)
        elif alt is None:
            algos = ()
        else:
            algos = (alt,)
        cases.append(finalize_case(case, alt_algos=algos, repeats=repeats))
        print(f"  [pool] {cid}: out={tuple(case['expect']['out_features'].shape)}"
              f" atol={case['expect']['atol_out']:.2e}")

    def L(cls, **ctor):
        return {"cls": cls, "ctor": ctor}

    C("maxp1d_k2s2", [L("SparseMaxPool1d", kernel_size=2, stride=2)],
      gen_input("maxp1d_k2s2", [64], 2, 6, 40))
    C("maxp2d_k2s2", [L("SparseMaxPool2d", kernel_size=2, stride=2)],
      gen_input("maxp2d_k2s2", [64, 64], 2, 8, 600))
    C("maxp2d_k3s2p1", [L("SparseMaxPool2d", kernel_size=3, stride=2,
                          padding=1)],
      gen_input("maxp2d_k3s2p1", [64, 64], 2, 8, 600))
    C("maxp3d_k2s2", [L("SparseMaxPool3d", kernel_size=2, stride=2)],
      gen_input("maxp3d_k2s2", [32, 32, 32], 2, 8, 1200, mode="clustered"))
    C("maxp3d_k3s2p1", [L("SparseMaxPool3d", kernel_size=3, stride=2,
                          padding=1)],
      gen_input("maxp3d_k3s2p1", [32, 32, 32], 2, 8, 1200))
    C("maxp3d_defstride", [L("SparseMaxPool3d", kernel_size=2)],
      gen_input("maxp3d_defstride", [32, 32, 32], 2, 8, 800))
    C("maxp4d_k2s2", [L("SparseMaxPool4d", kernel_size=2, stride=2)],
      gen_input("maxp4d_k2s2", [8, 16, 16, 16], 1, 4, 500))
    C("maxp3d_fp16", [L("SparseMaxPool3d", kernel_size=2, stride=2)],
      gen_input("maxp3d_fp16", [32, 32, 32], 2, 16, 1000, dtype="float16"),
      dtype="float16")
    C("maxp3d_native", [L("SparseMaxPool3d", kernel_size=2, stride=2,
                          algo="Native")],
      gen_input("maxp3d_native", [32, 32, 32], 2, 8, 800),
      alt="MaskImplicitGemm")
    C("maxp3d_ties", [L("SparseMaxPool3d", kernel_size=3, stride=2,
                        padding=1)],
      gen_input("maxp3d_ties", [16, 16, 16], 2, 4, 1500,
                quantize_feats=True),
      alt=None, repeats=2)
    C("avgp1d_k2s2", [L("SparseAvgPool1d", kernel_size=2, stride=2)],
      gen_input("avgp1d_k2s2", [64], 2, 6, 40), alt=None, repeats=2)
    C("avgp2d_k2s2", [L("SparseAvgPool2d", kernel_size=2, stride=2)],
      gen_input("avgp2d_k2s2", [64, 64], 2, 8, 600), alt=None, repeats=2)
    C("avgp3d_k2s2", [L("SparseAvgPool3d", kernel_size=2, stride=2)],
      gen_input("avgp3d_k2s2", [32, 32, 32], 2, 8, 1200), alt=None,
      repeats=2)
    C("avgp3d_k3s2p1", [L("SparseAvgPool3d", kernel_size=3, stride=2,
                          padding=1)],
      gen_input("avgp3d_k3s2p1", [32, 32, 32], 2, 8, 1000), alt=None,
      repeats=2)
    C("avgp3d_fp16", [L("SparseAvgPool3d", kernel_size=2, stride=2)],
      gen_input("avgp3d_fp16", [32, 32, 32], 2, 16, 1000, dtype="float16"),
      dtype="float16", alt=None, repeats=2)
    C("gmaxpool", [L("SparseGlobalMaxPool")],
      gen_input("gmaxpool", [16, 16, 16], 3, 8, 300),
      returns_dense=True, alt=None, repeats=2)
    C("gavgpool", [L("SparseGlobalAvgPool")],
      gen_input("gavgpool", [16, 16, 16], 3, 8, 300),
      returns_dense=True, alt=None, repeats=2)
    C("gmaxpool_fp16", [L("SparseGlobalMaxPool")],
      gen_input("gmaxpool_fp16", [16, 16, 16], 2, 8, 300, dtype="float16"),
      dtype="float16", returns_dense=True, alt=None, repeats=2)

    return cases


# ---------------------------------------------------------------------------
# ops-level cases
# ---------------------------------------------------------------------------

def ops_cases():
    cases = []
    ConvAlgo = impl.core.ConvAlgo

    # ---- native pair generation ----
    native_specs = [
        ("native_pairs_subm3d", [32, 32, 32], dict(
            ksize=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1],
            dilation=[1, 1, 1], out_padding=[0, 0, 0], subm=True,
            transpose=False)),
        ("native_pairs_subm3d_d2", [32, 32, 32], dict(
            ksize=[3, 3, 3], stride=[1, 1, 1], padding=[2, 2, 2],
            dilation=[2, 2, 2], out_padding=[0, 0, 0], subm=True,
            transpose=False)),
        ("native_pairs_conv3d", [32, 32, 32], dict(
            ksize=[3, 3, 3], stride=[2, 2, 2], padding=[1, 1, 1],
            dilation=[1, 1, 1], out_padding=[0, 0, 0], subm=False,
            transpose=False)),
        ("native_pairs_conv2d", [64, 64], dict(
            ksize=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1],
            out_padding=[0, 0], subm=False, transpose=False)),
        ("native_pairs_convt3d", [16, 16, 16], dict(
            ksize=[2, 2, 2], stride=[2, 2, 2], padding=[0, 0, 0],
            dilation=[1, 1, 1], out_padding=[0, 0, 0], subm=False,
            transpose=True)),
        ("native_pairs_conv1d", [64], dict(
            ksize=[3], stride=[1], padding=[0], dilation=[1],
            out_padding=[0], subm=False, transpose=False)),
    ]
    for cid, ss, args in native_specs:
        inp = gen_input(cid, ss, 2, 4, 700)
        indices = inp["indices"].to(DEV)
        out_inds, pair, npl = impl.ops.get_indice_pairs(
            indices, inp["batch_size"], ss, ConvAlgo.Native, args["ksize"],
            args["stride"], args["padding"], args["dilation"],
            args["out_padding"], args["subm"], args["transpose"])
        cases.append({
            "id": cid, "kind": "native_pairs", "input": inp, "args": args,
            "expect": {
                "out_inds": out_inds.cpu(),
                "indice_num_per_loc": npl.cpu(),
                "pairs_canon": canon_native_pairs(pair, npl),
            },
            "raw": {"pair": pair.cpu(), "npl": npl.cpu(),
                    "nout": int(out_inds.shape[0])},
        })
        print(f"  [ops] {cid}: nout={out_inds.shape[0]}")

    # ---- implicit gemm pair generation ----
    igemm_specs = [
        ("igemm_pairs_subm3d", [32, 32, 32], dict(
            ksize=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1],
            dilation=[1, 1, 1], out_padding=[0, 0, 0], subm=True,
            transpose=False, is_train=True, algo="MaskImplicitGemm")),
        ("igemm_pairs_conv3d", [32, 32, 32], dict(
            ksize=[3, 3, 3], stride=[2, 2, 2], padding=[1, 1, 1],
            dilation=[1, 1, 1], out_padding=[0, 0, 0], subm=False,
            transpose=False, is_train=True, algo="MaskImplicitGemm")),
        ("igemm_pairs_conv3d_eval", [32, 32, 32], dict(
            ksize=[3, 3, 3], stride=[2, 2, 2], padding=[1, 1, 1],
            dilation=[1, 1, 1], out_padding=[0, 0, 0], subm=False,
            transpose=False, is_train=False, algo="MaskImplicitGemm")),
        ("igemm_pairs_convt2d", [48, 48], dict(
            ksize=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1],
            out_padding=[0, 0], subm=False, transpose=True, is_train=True,
            algo="MaskImplicitGemm")),
        ("igemm_pairs_subm3d_msplit", [32, 32, 32], dict(
            ksize=[3, 3, 3], stride=[1, 1, 1], padding=[1, 1, 1],
            dilation=[1, 1, 1], out_padding=[0, 0, 0], subm=True,
            transpose=False, is_train=True, algo="MaskSplitImplicitGemm")),
    ]
    for cid, ss, args in igemm_specs:
        inp = gen_input(cid, ss, 2, 4, 700)
        indices = inp["indices"].to(DEV)
        res = impl.ops.get_indice_pairs_implicit_gemm(
            indices, inp["batch_size"], ss, getattr(ConvAlgo, args["algo"]),
            args["ksize"], args["stride"], args["padding"], args["dilation"],
            args["out_padding"], args["subm"], args["transpose"],
            args["is_train"])
        (out_inds, npl, pair_fwd, pair_bwd, pm_fwd, pm_bwd, ma_fwd, ma_bwd,
         masks) = res
        pb = pair_bwd if (isinstance(pair_bwd, torch.Tensor)
                          and pair_bwd.numel() > 0) else None
        out_ss = helpers.expected_out_shape(ss, args)
        canon = canon_igemm_pairs(out_inds, out_ss, pair_fwd, pb)
        cases.append({
            "id": cid, "kind": "igemm_pairs", "input": inp, "args": args,
            "expect": {
                "out_inds_canon": out_inds.cpu()[
                    canonical_order(out_inds.cpu(), out_ss)],
                "indice_num_per_loc": npl.cpu(),
                "pair_fwd_canon": canon["pair_fwd"],
                "pair_bwd_canon": canon.get("pair_bwd"),
                "has_pair_bwd": pb is not None,
                "masks": [torch.from_numpy(np.ascontiguousarray(m).view(np.int64)
                          if m.dtype == np.uint64 else
                          np.ascontiguousarray(m).astype(np.int64))
                          for m in masks],
                "n_mask_splits": len(pm_fwd),
            },
        })
        print(f"  [ops] {cid}: nout={out_inds.shape[0]} splits={len(pm_fwd)}")

    # ---- direct functional conv calls (Native pairs) ----
    fdirect = [
        ("indice_conv_subm3d", "native_pairs_subm3d", "subm", 8, 16),
        ("indice_conv_regular3d", "native_pairs_conv3d", "regular", 8, 16),
        ("indice_conv_inverse3d", "native_pairs_conv3d", "inverse", 8, 16),
    ]
    case_by_id = {c["id"]: c for c in cases}
    for cid, src_id, mode, cin, cout in fdirect:
        src = case_by_id[src_id]
        inp = src["input"]
        n_in = inp["indices"].shape[0]
        nout_pairs = src["raw"]["nout"]
        kv = src["raw"]["pair"].shape[1]
        g = torch.Generator().manual_seed(case_seed(cid))
        if mode == "inverse":
            n_feat, n_out = nout_pairs, n_in
        else:
            n_feat, n_out = n_in, (n_in if mode == "subm" else nout_pairs)
        feats = torch.randn(n_feat, cin, generator=g)
        filters = torch.randn(cout, *([3, 3, 3] if kv == 27 else [2, 2, 2]),
                              cin, generator=g) * 0.1
        grad_out = torch.randn(n_out, cout, generator=g)

        def run_direct():
            f = feats.to(DEV).clone().requires_grad_(True)
            w = filters.to(DEV).clone().requires_grad_(True)
            pair = src["raw"]["pair"].to(DEV)
            npl_t = src["raw"]["npl"].to(DEV)
            if mode == "subm":
                out = impl.functional.indice_subm_conv(
                    f, w, pair, npl_t, n_out, impl.core.ConvAlgo.Native)
            elif mode == "regular":
                out = impl.functional.indice_conv(
                    f, w, pair, npl_t, n_out, impl.core.ConvAlgo.Native)
            else:
                out = impl.functional.indice_inverse_conv(
                    f, w, pair, npl_t, n_out, impl.core.ConvAlgo.Native)
            out.backward(grad_out.to(DEV))
            return (out.detach().cpu(), f.grad.cpu(), w.grad.cpu())

        o1, gf1, gw1 = run_direct()
        o2, gf2, gw2 = run_direct()
        cases.append({
            "id": cid, "kind": "func_conv", "mode": mode,
            "tensors": {"features": feats, "filters": filters,
                        "pair": src["raw"]["pair"], "npl": src["raw"]["npl"],
                        "num_activate_out": n_out, "grad_out": grad_out},
            "expect": {
                "out": o1, "atol_out": atol_for(o1, [o2], "float32"),
                "grad_features": gf1,
                "atol_grad_features": atol_for(gf1, [gf2], "float32"),
                "grad_filters": gw1,
                "atol_grad_filters": atol_for(gw1, [gw2], "float32"),
            },
        })
        print(f"  [ops] {cid}: out={tuple(o1.shape)}")

    # ---- direct pooling functional calls ----
    src = case_by_id["native_pairs_conv3d"]
    inp = src["input"]
    g = torch.Generator().manual_seed(case_seed("maxpool_native_direct"))
    feats = torch.randn(inp["indices"].shape[0], 8, generator=g)
    nout = src["raw"]["nout"]
    grad_out = torch.randn(nout, 8, generator=g)
    f = feats.to(DEV).clone().requires_grad_(True)
    out = impl.functional.indice_maxpool(
        f, src["raw"]["pair"].to(DEV), src["raw"]["npl"].to(DEV), nout)
    out.backward(grad_out.to(DEV))
    cases.append({
        "id": "maxpool_native_direct", "kind": "func_maxpool_native",
        "tensors": {"features": feats, "pair": src["raw"]["pair"],
                    "npl": src["raw"]["npl"], "num_activate_out": nout,
                    "grad_out": grad_out},
        "expect": {"out": out.detach().cpu(), "grad_features": f.grad.cpu()},
    })
    print("  [ops] maxpool_native_direct done")

    # igemm pooling direct: regenerate igemm pairs for a conv3d case and
    # store them (raw pair tables are valid inputs regardless of permutation).
    inp = gen_input("pool_igemm_direct", [32, 32, 32], 2, 8, 900)
    indices = inp["indices"].to(DEV)
    res = impl.ops.get_indice_pairs_implicit_gemm(
        indices, inp["batch_size"], [32, 32, 32], ConvAlgo.MaskImplicitGemm,
        [2, 2, 2], [2, 2, 2], [0, 0, 0], [1, 1, 1], [0, 0, 0], False, False,
        True)
    out_inds, _, pair_fwd, pair_bwd = res[0], res[1], res[2], res[3]
    nout = out_inds.shape[0]
    g = torch.Generator().manual_seed(case_seed("pool_igemm_direct"))
    feats = torch.randn(inp["indices"].shape[0], 8, generator=g)
    grad_out = torch.randn(nout, 8, generator=g)

    f = feats.to(DEV).clone().requires_grad_(True)
    out = impl.functional.indice_maxpool_implicit_gemm(
        f, pair_fwd, pair_bwd, nout)
    out.backward(grad_out.to(DEV))
    cases.append({
        "id": "maxpool_igemm_direct", "kind": "func_maxpool_igemm",
        "tensors": {"features": feats, "pair_fwd": pair_fwd.cpu(),
                    "pair_bwd": pair_bwd.cpu(), "num_activate_out": nout,
                    "grad_out": grad_out},
        "expect": {"out": out.detach().cpu(), "grad_features": f.grad.cpu()},
    })
    f = feats.to(DEV).clone().requires_grad_(True)
    out = impl.functional.indice_avgpool_implicit_gemm(
        f, pair_fwd, pair_bwd, nout, True)
    out.backward(grad_out.to(DEV))
    cases.append({
        "id": "avgpool_igemm_direct", "kind": "func_avgpool_igemm",
        "tensors": {"features": feats, "pair_fwd": pair_fwd.cpu(),
                    "pair_bwd": pair_bwd.cpu(), "num_activate_out": nout,
                    "grad_out": grad_out},
        "expect": {"out": out.detach().cpu(), "grad_features": f.grad.cpu()},
    })
    print("  [ops] igemm pool direct done")

    # ---- global_pool_rearrange ----
    inp = gen_input("gpr", [16, 16, 16], 4, 4, 300, empty_batches=(1,))
    coords = inp["indices"].to(DEV)
    out_indices, counts = impl.ops.global_pool_rearrange(coords, 4)
    counts_cpu = counts.cpu()
    rows = []
    for b in range(4):
        r = out_indices[b, :counts_cpu[b]].cpu().long()
        rows.append(torch.sort(r).values)
    cases.append({
        "id": "gpr", "kind": "global_pool_rearrange", "input": inp,
        "batch_size": 4,
        "expect": {"rows_sorted": rows, "counts": counts_cpu},
    })
    print("  [ops] global_pool_rearrange done")

    # ---- PointToVoxel ----
    p2v_specs = [
        ("p2v3d_basic", 3, [0.1, 0.1, 0.1], [-2, -2, -2, 2, 2, 2], 8000, 6,
         5000, False),
        ("p2v3d_trunc", 3, [0.4, 0.4, 0.4], [-2, -2, -2, 2, 2, 2], 2000, 2,
         4000, False),
        ("p2v3d_emptymean", 3, [0.2, 0.2, 0.2], [-2, -2, -2, 2, 2, 2], 4000,
         8, 2000, True),
        ("p2v2d_basic", 2, [0.1, 0.1], [-2, -2, 2, 2], 4000, 6, 3000, False),
    ]
    for cid, ndim, vsize, crange, maxvox, maxpts, npts, empty_mean in p2v_specs:
        g = torch.Generator().manual_seed(case_seed(cid))
        pc = torch.randn(npts, ndim + 1, generator=g) * 0.8
        p2v = impl.putils.PointToVoxel(
            vsize_xyz=vsize, coors_range_xyz=crange,
            num_point_features=ndim + 1, max_num_voxels=maxvox,
            max_num_points_per_voxel=maxpts,
            device=torch.device(DEV))
        vox, idx, npv, pcvid = p2v.generate_voxel_with_id(
            pc.to(DEV), empty_mean=empty_mean)
        idx_c, vox_c, npv_c = canon_voxel_result(idx, vox, npv)
        truncated = bool((npv_c > maxpts).any() or (npv_c == maxpts).any())
        cases.append({
            "id": cid, "kind": "p2v", "ndim": ndim,
            "ctor": {"vsize_xyz": vsize, "coors_range_xyz": crange,
                     "num_point_features": ndim + 1,
                     "max_num_voxels": maxvox,
                     "max_num_points_per_voxel": maxpts},
            "points": pc, "empty_mean": empty_mean,
            "expect": {"indices_canon": idx_c, "voxels_canon": vox_c,
                       "num_per_voxel_canon": npv_c,
                       "check_contents": not truncated,
                       "check_pc_voxel_id": not truncated},
        })
        print(f"  [ops] {cid}: nvox={idx_c.shape[0]} maxnpv={int(npv_c.max())}"
              f" truncated={truncated}")

    return cases


# ---------------------------------------------------------------------------
# misc cases (SparseConvTensor, tables, sequential, sparse_add)
# ---------------------------------------------------------------------------

def misc_cases():
    cases = []
    sp = impl.pytorch

    # dense / from_dense / scatter_nd
    inp = gen_input("dense_3d", [8, 12, 10], 2, 5, 150)
    x = sp.SparseConvTensor(inp["features"].to(DEV), inp["indices"].to(DEV),
                            inp["spatial_shape"], inp["batch_size"])
    cases.append({"id": "dense_3d", "kind": "dense", "input": inp,
                  "expect": {"dense_cf": x.dense(True).cpu(),
                             "dense_cl": x.dense(False).cpu()}})

    g = torch.Generator().manual_seed(case_seed("from_dense_3d"))
    dense = torch.zeros(4, 6, 7, 8, 4)
    mask = torch.rand(4, 6, 7, 8, generator=g) < 0.1
    vals = torch.randn(4, 6, 7, 8, 4, generator=g)
    dense[mask] = vals[mask]
    xs = sp.SparseConvTensor.from_dense(dense.to(DEV))
    cases.append({"id": "from_dense_3d", "kind": "from_dense",
                  "dense": dense,
                  "expect": {"indices": xs.indices.cpu(),
                             "features": xs.features.cpu(),
                             "spatial_shape": list(xs.spatial_shape),
                             "batch_size": xs.batch_size}})

    g = torch.Generator().manual_seed(case_seed("scatter_nd"))
    sn_idx = torch.stack([torch.randint(0, 5, (40,), generator=g),
                          torch.randint(0, 6, (40,), generator=g)], 1).long()
    sn_idx = torch.unique(sn_idx, dim=0)  # duplicates -> nondeterministic
    sn_upd = torch.randn(sn_idx.shape[0], 3, generator=g)
    import importlib
    core_mod = importlib.import_module(f"{impl.name}.pytorch.core")
    sn_out = core_mod.scatter_nd(sn_idx.to(DEV), sn_upd.to(DEV), [5, 6, 3])
    cases.append({"id": "scatter_nd", "kind": "scatter_nd",
                  "indices": sn_idx, "updates": sn_upd, "shape": [5, 6, 3],
                  "expect": {"out": sn_out.cpu()}})

    # SparseSequential with dense modules (handled as pipeline case)
    seq_case = {
        "id": "seq2d_mixed", "kind": "pipeline", "dtype": "float32",
        "training": True, "add_input": False,
        "layers": [{
            "cls": "SparseSequential",
            "children": [
                {"cls": "SubMConv2d",
                 "ctor": dict(in_channels=8, out_channels=16, kernel_size=3,
                              padding=1, bias=False)},
                {"cls": "nn.BatchNorm1d", "ctor": dict(num_features=16)},
                {"cls": "nn.ReLU", "ctor": {}},
                {"cls": "SparseConv2d",
                 "ctor": dict(in_channels=16, out_channels=16, kernel_size=3,
                              stride=2, padding=1)},
            ]}],
        "input": gen_input("seq2d_mixed", [64, 64], 2, 8, 600),
    }
    gen_params("seq2d_mixed", seq_case["layers"], "float32")
    finalize_case(seq_case, alt_algos=("Native",))
    cases.append(seq_case)
    print("  [misc] seq2d_mixed done")

    # tables
    for table in ["JoinTable", "AddTable"]:
        cid = f"tables_{table.lower()}"
        inp = gen_input(cid, [48, 48], 2, 8, 500)
        branch_specs = [
            {"cls": "SubMConv2d", "ctor": dict(in_channels=8, out_channels=8,
                                               kernel_size=3, bias=False)},
            {"cls": "SubMConv2d", "ctor": dict(in_channels=8, out_channels=8,
                                               kernel_size=3, bias=False)},
        ]
        gen_params(cid, branch_specs, "float32")

        def run_table():
            x, _ = helpers.make_sparse_tensor(impl, inp, DEV)
            br = [helpers.build_layer(impl, s, torch.float32, DEV)
                  for s in branch_specs]
            outs = [b(x) for b in br]
            t = getattr(sp, table)()
            return t(outs)
        o1 = run_table()
        o2 = run_table()
        order = canonical_order(o1.indices.cpu(), o1.spatial_shape)
        f1 = o1.features.detach().cpu()[order]
        order2 = canonical_order(o2.indices.cpu(), o2.spatial_shape)
        f2 = o2.features.detach().cpu()[order2]
        cases.append({
            "id": cid, "kind": "tables", "table": table, "input": inp,
            "branch_specs": branch_specs,
            "expect": {
                "out_indices": o1.indices.cpu()[order],
                "out_features": f1,
                "out_spatial_shape": list(o1.spatial_shape),
                "atol_out": atol_for(f1, [f2], "float32"),
            }})
        print(f"  [misc] {cid} done")

    # sparse_add / sparse_add_hash_based / AddTableMisaligned
    inp_a = gen_input("sparse_add_a", [32, 32, 32], 2, 8, 500)
    inp_b = gen_input("sparse_add_b", [32, 32, 32], 2, 8, 450)
    for fn_name in ["sparse_add", "sparse_add_hash_based",
                    "add_table_misaligned"]:
        def run_add():
            xa, _ = helpers.make_sparse_tensor(impl, inp_a, DEV)
            xb, _ = helpers.make_sparse_tensor(impl, inp_b, DEV)
            if fn_name == "sparse_add":
                return impl.functional.sparse_add(xa, xb)
            if fn_name == "sparse_add_hash_based":
                return impl.functional.sparse_add_hash_based(xa, xb)
            return impl.tables.AddTableMisaligned()([xa, xb])
        o1, o2 = run_add(), run_add()
        order = canonical_order(o1.indices.cpu(), o1.spatial_shape)
        order2 = canonical_order(o2.indices.cpu(), o2.spatial_shape)
        f1 = o1.features.detach().cpu()[order]
        f2 = o2.features.detach().cpu()[order2]
        cases.append({
            "id": fn_name, "kind": "sparse_add", "fn": fn_name,
            "inputs": [inp_a, inp_b],
            "expect": {
                "out_indices": o1.indices.cpu()[order],
                "out_features": f1,
                "out_spatial_shape": list(o1.spatial_shape),
                "atol_out": atol_for(f1, [f2], "float32"),
            }})
        print(f"  [misc] {fn_name} done")

    return cases


# ---------------------------------------------------------------------------
# protocol cases
# ---------------------------------------------------------------------------

def proto_cases():
    cases = []
    sp = impl.pytorch

    init_specs = [
        ("init_subm3d", "SubMConv3d",
         dict(in_channels=16, out_channels=32, kernel_size=3), 1234),
        ("init_conv2d_k23", "SparseConv2d",
         dict(in_channels=4, out_channels=8, kernel_size=(2, 3)), 99),
        ("init_convt3d", "SparseConvTranspose3d",
         dict(in_channels=8, out_channels=16, kernel_size=2, bias=False), 7),
        ("init_inv3d", "SparseInverseConv3d",
         dict(in_channels=8, out_channels=16, kernel_size=3,
              indice_key="ik"), 5),
    ]
    for cid, cls, ctor, seed in init_specs:
        torch.manual_seed(seed)
        layer = getattr(sp, cls)(**ctor)
        cases.append({
            "id": cid, "kind": "init", "cls": cls, "ctor": ctor, "seed": seed,
            "expect": {
                "weight": layer.weight.detach().clone(),
                "bias": (layer.bias.detach().clone()
                         if layer.bias is not None else None),
                "weight_shape": list(layer.weight.shape),
            }})

    sd_specs = [
        ("sd_subm3d", "SubMConv3d",
         dict(in_channels=4, out_channels=8, kernel_size=3)),
        ("sd_conv3d_nobias", "SparseConv3d",
         dict(in_channels=4, out_channels=8, kernel_size=3, bias=False)),
        ("sd_conv3d_rvc", "SparseConv3d",
         dict(in_channels=4, out_channels=8, kernel_size=3,
              record_voxel_count=True)),
        ("sd_maxpool", "SparseMaxPool3d", dict(kernel_size=2)),
    ]
    for cid, cls, ctor in sd_specs:
        layer = getattr(sp, cls)(**ctor)
        sd = layer.state_dict()
        cases.append({
            "id": cid, "kind": "state_dict_keys", "cls": cls, "ctor": ctor,
            "expect": {"keys": sorted(sd.keys()),
                       "shapes": {k: list(v.shape) for k, v in sd.items()}}})

    repr_specs = [
        ("repr_subm3d", "SubMConv3d",
         dict(in_channels=4, out_channels=8, kernel_size=3)),
        ("repr_conv3d_full", "SparseConv3d",
         dict(in_channels=4, out_channels=8, kernel_size=3, stride=2,
              padding=1, dilation=1, bias=False)),
        ("repr_maxpool", "SparseMaxPool2d", dict(kernel_size=3, stride=2)),
    ]
    for cid, cls, ctor in repr_specs:
        layer = getattr(sp, cls)(**ctor)
        cases.append({"id": cid, "kind": "repr", "cls": cls, "ctor": ctor,
                      "expect": {"extra_repr": layer.extra_repr()}})

    # pickle support flags (recorded from reference behavior)
    import pickle
    layer = sp.SubMConv3d(4, 8, 3)
    try:
        pickle.loads(pickle.dumps(layer))
        layer_pickle_ok = True
    except Exception:
        layer_pickle_ok = False
    inp = gen_input("proto_sptensor", [8, 8, 8], 1, 4, 50)
    x = sp.SparseConvTensor(inp["features"], inp["indices"],
                            inp["spatial_shape"], inp["batch_size"])
    try:
        pickle.loads(pickle.dumps(x))
        sptensor_pickle_ok = True
    except Exception:
        sptensor_pickle_ok = False
    cases.append({"id": "pickle_flags", "kind": "pickle_flags",
                  "expect": {"layer": layer_pickle_ok,
                             "sptensor": sptensor_pickle_ok}})
    print(f"  [proto] pickle: layer={layer_pickle_ok} "
          f"sptensor={sptensor_pickle_ok}")

    # autocast behavior on a small subm conv
    auto_case = {
        "id": "autocast_subm3d", "kind": "autocast",
        "layers": [{"cls": "SubMConv3d",
                    "ctor": dict(in_channels=8, out_channels=16,
                                 kernel_size=3)}],
        "input": gen_input("autocast_subm3d", [16, 16, 16], 2, 8, 400),
    }
    gen_params("autocast_subm3d", auto_case["layers"], "float32")
    layer = helpers.build_layer(impl, auto_case["layers"][0], torch.float32,
                                DEV)
    layer.eval()
    x, _ = helpers.make_sparse_tensor(impl, auto_case["input"], DEV)
    with torch.no_grad(), torch.autocast("cuda"):
        out = layer(x)
    order = canonical_order(out.indices.cpu(), out.spatial_shape)
    auto_case["expect"] = {
        "out_dtype": str(out.features.dtype),
        "out_indices": out.indices.cpu()[order],
        "out_features": out.features.detach().cpu()[order],
        "atol_out": FLOORS["float16"] * max(
            1.0, float(out.features.abs().max())),
    }
    cases.append(auto_case)
    print(f"  [proto] autocast dtype={auto_case['expect']['out_dtype']}")
    return cases


# ---------------------------------------------------------------------------
# network case
# ---------------------------------------------------------------------------

def net_cases():
    cases = []
    for dtype in ["float32", "float16"]:
        cid = f"unet3d_{dtype}"
        inp = gen_input(cid, [32, 32, 32], 2, 6, 2500, mode="clustered",
                        dtype=dtype)
        torch.manual_seed(case_seed(cid))
        net = helpers.build_unet3d(impl, in_channels=6, base=16).to(DEV)
        if dtype == "float16":
            net = net.half()
        sd = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        g = torch.Generator().manual_seed(case_seed(cid, 3))

        def run_net(grad_canon=None):
            torch.manual_seed(case_seed(cid))
            net2 = helpers.build_unet3d(impl, in_channels=6, base=16).to(DEV)
            if dtype == "float16":
                net2 = net2.half()
            net2.load_state_dict({k: v.to(DEV) for k, v in sd.items()})
            net2.train()
            feats = inp["features"].to(DEV).clone().requires_grad_(True)
            x = impl.pytorch.SparseConvTensor(
                feats, inp["indices"].to(DEV), inp["spatial_shape"],
                inp["batch_size"])
            out = net2(x)
            order = canonical_order(out.indices.cpu(), out.spatial_shape)
            res = {"out_indices": out.indices.cpu()[order],
                   "out_features": out.features.detach().cpu()[order],
                   "out_spatial_shape": list(out.spatial_shape)}
            if grad_canon is not None:
                inv = inverse_permutation(order.to(DEV))
                out.features.backward(
                    grad_canon.to(DEV)[inv].to(out.features.dtype))
                res["grad_input"] = feats.grad.detach().cpu()
                res["grad_params"] = {
                    k: p.grad.detach().cpu()
                    for k, p in net2.named_parameters() if p.grad is not None}
            return res

        first = run_net()
        grad_canon = torch.randn(*first["out_features"].shape,
                                 generator=g).to(getattr(torch, dtype))
        ref = run_net(grad_canon)
        alts = [run_net(grad_canon) for _ in range(3)]

        def grad_atol(v, alts_v):
            atol = atol_for(v, alts_v, dtype)
            if dtype == "float16":
                # batch-norm amplifies fp16 igemm nondeterminism in the
                # reference itself; observed up to ~5% of |grad|max.
                refmax = float(v.detach().to(torch.float64).abs().max()) \
                    if v.numel() else 1.0
                atol = max(atol, 0.06 * max(refmax, 1.0))
            return atol

        expect = {
            "out_indices": ref["out_indices"],
            "out_features": ref["out_features"],
            "out_spatial_shape": ref["out_spatial_shape"],
            "atol_out": atol_for(ref["out_features"],
                                 [a["out_features"] for a in alts], dtype),
            "grad_out": grad_canon,
            "grad_input": ref["grad_input"],
            "atol_grad_input": grad_atol(
                ref["grad_input"], [a["grad_input"] for a in alts]),
            "grad_params": {
                k: {"grad": v,
                    "atol": grad_atol(
                        v, [a["grad_params"][k] for a in alts])}
                for k, v in ref["grad_params"].items()},
        }
        cases.append({"id": cid, "kind": "net", "dtype": dtype, "input": inp,
                      "state_dict": sd, "expect": expect})
        print(f"  [net] {cid}: out={tuple(ref['out_features'].shape)} "
              f"atol={expect['atol_out']:.2e}")
    return cases


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------

def api_surface():
    import inspect
    surface = {"pytorch": {}, "ops": [], "functional": [], "top": [],
               "ctors": {}}
    pytorch_names = [
        "SparseConvTensor", "SparseModule", "SparseSequential",
        "SparseBatchNorm", "SparseReLU", "SparseIdentity",
        "assign_name_for_sparse_modules", "Identity", "ToDense", "RemoveGrid",
        "ConvAlgo", "AddTable", "ConcatTable", "JoinTable",
        "SparseConv1d", "SparseConv2d", "SparseConv3d", "SparseConv4d",
        "SparseConvTranspose1d", "SparseConvTranspose2d",
        "SparseConvTranspose3d", "SparseConvTranspose4d",
        "SparseInverseConv1d", "SparseInverseConv2d", "SparseInverseConv3d",
        "SparseInverseConv4d",
        "SubMConv1d", "SubMConv2d", "SubMConv3d", "SubMConv4d",
        "SparseMaxPool1d", "SparseMaxPool2d", "SparseMaxPool3d",
        "SparseMaxPool4d", "SparseAvgPool1d", "SparseAvgPool2d",
        "SparseAvgPool3d", "SparseGlobalMaxPool", "SparseGlobalAvgPool",
        "functional", "ops",
    ]
    for n in pytorch_names:
        assert hasattr(impl.pytorch, n), n
    surface["pytorch"] = pytorch_names
    surface["ops"] = [
        "get_conv_output_size", "get_deconv_output_size", "get_indice_pairs",
        "get_indice_pairs_implicit_gemm", "indice_conv",
        "indice_conv_backward", "implicit_gemm", "implicit_gemm_backward",
        "indice_maxpool", "indice_maxpool_backward",
        "indice_maxpool_implicit_gemm",
        "indice_maxpool_implicit_gemm_backward",
        "indice_avgpool_implicit_gemm",
        "indice_avgpool_implicit_gemm_backward", "global_pool_rearrange",
        "maximum_value_int_", "ConvAlgo"]
    for n in surface["ops"]:
        assert hasattr(impl.ops, n), n
    surface["functional"] = [
        "indice_conv", "implicit_gemm", "indice_inverse_conv",
        "indice_subm_conv", "indice_maxpool", "indice_maxpool_implicit_gemm",
        "indice_avgpool_implicit_gemm", "sparse_add", "sparse_add_hash_based"]
    for n in surface["functional"]:
        assert hasattr(impl.functional, n), n
    surface["top"] = ["ConvAlgo", "constants", "__version__",
                      "SPCONV_VERSION_NUMBERS"]
    for n in surface["top"]:
        assert hasattr(impl.root, n), n
    surface["pytorch_utils"] = ["PointToVoxel",
                                "gather_features_by_pc_voxel_id"]
    surface["hash"] = ["HashTable"]
    layer_classes = [n for n in pytorch_names
                     if ("Conv" in n or "Pool" in n) and n != "SparseConvTensor"]
    for n in layer_classes + ["SparseSequential"]:
        cls = getattr(impl.pytorch, n)
        try:
            sig = inspect.signature(cls.__init__)
            params = [p for p in sig.parameters if p != "self"]
        except (ValueError, TypeError):
            params = []
        surface["ctors"][n] = params
    return surface


# ---------------------------------------------------------------------------

def main():
    torch.backends.cudnn.deterministic = True
    print("generating golden_conv.pt ...")
    torch.save({"cases": conv_cases()}, DATA_DIR / "golden_conv.pt")
    print("generating golden_pool.pt ...")
    torch.save({"cases": pool_cases()}, DATA_DIR / "golden_pool.pt")
    print("generating golden_ops.pt ...")
    torch.save({"cases": ops_cases()}, DATA_DIR / "golden_ops.pt")
    print("generating golden_misc.pt ...")
    torch.save({"cases": misc_cases()}, DATA_DIR / "golden_misc.pt")
    print("generating golden_proto.pt ...")
    torch.save({"cases": proto_cases()}, DATA_DIR / "golden_proto.pt")
    print("generating golden_net.pt ...")
    torch.save({"cases": net_cases()}, DATA_DIR / "golden_net.pt")
    print("generating golden_api.json ...")
    (DATA_DIR / "golden_api.json").write_text(
        json.dumps(api_surface(), indent=2))
    print("ALL GOLDEN DATA GENERATED")
    for f in sorted(DATA_DIR.iterdir()):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
