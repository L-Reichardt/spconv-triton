"""Tight fp16 golden generation (audit remediation).

The main suites bound fp16 cases by 500x the reference's run-to-run noise,
which an audit showed is near-vacuous (atol up to 98% of max|ref|). This
suite instead bounds fp16 results against the FP32 REFERENCE computed on
bit-identical (fp16-exact) inputs/params:

    atol(tensor) = 3 x max-deviation(reference fp16 run vs fp32 run) + 1e-3

so any implementation must stay within a small factor of the legitimate
fp16 rounding error. Run once with the reference:

    SPCONV_TEST_IMPL=spconv uv run python tests_fp16/gen_golden_fp16.py
"""

import sys
import zlib
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

import helpers  # noqa: E402
from helpers import (  # noqa: E402
    canonical_order,
    inverse_permutation,
    pipeline_named_params,
    run_pipeline_case,
)

impl = helpers.load_impl()
assert impl.name == "spconv", "goldens must come from unchanged spconv"

DEV = "cuda"
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

FLOOR = 1e-3
K_MARGIN = 3.0


def seed_of(cid, salt=0):
    return (zlib.crc32(cid.encode()) + salt * 7919) % (2**31)


def fp16_exact(t):
    return t.half().float()


def gen_input(cid, ss, bs, channels, npoints):
    g = torch.Generator().manual_seed(seed_of(cid))
    rows = []
    for b in range(bs):
        c = torch.unique(
            torch.stack([torch.randint(0, s, (npoints,), generator=g) for s in ss], 1),
            dim=0,
        )
        rows.append(torch.cat([torch.full((c.shape[0], 1), b, dtype=torch.long), c], 1))
    idx = torch.cat(rows).int()
    feats = fp16_exact(torch.randn(idx.shape[0], channels, generator=g))
    return idx, feats


def run_variant(case_skel, dtype, grad_canon):
    """Run a pipeline case in the given dtype; returns canonical results."""
    case = dict(case_skel)
    case["dtype"] = dtype
    case["input"] = dict(case_skel["input"])
    case["input"]["features"] = case["input"]["features"].to(getattr(torch, dtype))
    res = run_pipeline_case(impl, case, DEV)
    out, layers = res["out"], res["layers"]
    order = canonical_order(out.indices.cpu(), out.spatial_shape)
    r = {
        "out_indices": out.indices.cpu()[order],
        "out_features": out.features.detach().cpu().float()[order],
        "out_spatial_shape": list(out.spatial_shape),
    }
    inv = inverse_permutation(order.to(DEV))
    out.features.backward(grad_canon.to(DEV)[inv].to(out.features.dtype))
    r["grad_input"] = res["feats_leaf"].grad.detach().cpu().float()
    r["grad_params"] = {
        k: p.grad.detach().cpu().float()
        for k, p in pipeline_named_params(layers).items()
    }
    torch.cuda.synchronize()
    return r


def dev_of(a, b):
    return float((a.double() - b.double()).abs().max())


def make_case(cid, layer_specs, ss, bs, channels, npoints):
    # Batch coverage: all existing tests run at batch size 4 (bzyx, batch>1).
    bs = 4
    idx, feats32 = gen_input(cid, ss, bs, channels, npoints)
    # fp16-exact parameters: generate fp32, round through fp16
    g = torch.Generator().manual_seed(seed_of(cid, 1))
    for spec in layer_specs:
        tmp = helpers.build_layer(impl, {**spec, "params": {}}, torch.float32, "cpu")
        bound = 3.0 / max(float(torch.tensor(tmp.weight.shape[1:]).prod()), 1.0) ** 0.5
        w = fp16_exact(torch.empty(tmp.weight.shape).uniform_(-bound, bound, generator=g))
        params = {"weight": w}
        if tmp.bias is not None:
            params["bias"] = fp16_exact(
                torch.empty(tmp.weight.shape[0]).uniform_(-bound, bound, generator=g)
            )
        spec["params"] = params

    skel = {
        "id": cid,
        "kind": "pipeline",
        "training": True,
        "add_input": False,
        "layers": layer_specs,
        "input": {
            "features": feats32,
            "indices": idx,
            "spatial_shape": list(ss),
            "batch_size": bs,
        },
    }
    # forward once to size the canonical upstream gradient
    probe = dict(skel)
    probe["dtype"] = "float32"
    res = run_pipeline_case(impl, probe, DEV)
    n_out, c_out = res["out"].features.shape
    g2 = torch.Generator().manual_seed(seed_of(cid, 2))
    grad_canon = fp16_exact(torch.randn(n_out, c_out, generator=g2))

    ref32 = run_variant(skel, "float32", grad_canon)
    devs = {"out_features": 0.0, "grad_input": 0.0}
    dev_params = dict.fromkeys(ref32["grad_params"], 0.0)
    for _ in range(2):  # reference fp16 igemm is nondeterministic
        h = run_variant(skel, "float16", grad_canon)
        assert torch.equal(h["out_indices"], ref32["out_indices"])
        devs["out_features"] = max(
            devs["out_features"], dev_of(h["out_features"], ref32["out_features"])
        )
        devs["grad_input"] = max(
            devs["grad_input"], dev_of(h["grad_input"], ref32["grad_input"])
        )
        for k in dev_params:
            dev_params[k] = max(
                dev_params[k], dev_of(h["grad_params"][k], ref32["grad_params"][k])
            )

    def atol(d):
        return K_MARGIN * d + FLOOR

    expect = {
        "out_spatial_shape": ref32["out_spatial_shape"],
        "out_indices": ref32["out_indices"],
        "out_features": ref32["out_features"],
        "atol_out": atol(devs["out_features"]),
        "grad_out": grad_canon,
        "grad_input": ref32["grad_input"],
        "atol_grad_input": atol(devs["grad_input"]),
        "grad_params": {
            k: {"grad": v, "atol": atol(dev_params[k])}
            for k, v in ref32["grad_params"].items()
        },
    }
    case = dict(skel)
    case["dtype"] = "float16"
    # store fp16 inputs/params (the values are fp16-exact by construction)
    case["input"]["features"] = feats32.half()
    for spec in case["layers"]:
        spec["params"] = {k: v.half() for k, v in spec["params"].items()}
    case["expect"] = expect

    refmax = float(ref32["out_features"].abs().max())
    sens = expect["atol_out"] / max(refmax, 1e-9)
    worst_g = max(
        info["atol"] / max(float(info["grad"].abs().max()), 1e-9)
        for info in expect["grad_params"].values()
    )
    print(
        f"  [fp16] {cid}: atol_out={expect['atol_out']:.3e} "
        f"({sens * 100:.2f}% of max|ref|), worst grad sens={worst_g * 100:.2f}%"
    )
    assert sens < 0.10, f"{cid}: fp16 out tolerance not meaningfully tight"
    return case


def L(cls, **ctor):
    return {"cls": cls, "ctor": ctor}


def main():
    cases = [
        make_case(
            "fp16_subm3d",
            [L("SubMConv3d", in_channels=16, out_channels=32, kernel_size=3)],
            [32, 32, 32], 2, 16, 1200,
        ),
        make_case(
            "fp16_conv3d_k3s2p1",
            [L("SparseConv3d", in_channels=16, out_channels=32, kernel_size=3,
               stride=2, padding=1)],
            [32, 32, 32], 2, 16, 1200,
        ),
        make_case(
            "fp16_convt3d_k2s2",
            [L("SparseConvTranspose3d", in_channels=16, out_channels=32,
               kernel_size=2, stride=2)],
            [16, 16, 16], 2, 16, 500,
        ),
        make_case(
            "fp16_inv3d_pair",
            [L("SparseConv3d", in_channels=16, out_channels=32, kernel_size=3,
               stride=2, padding=1, indice_key="ds"),
             L("SparseInverseConv3d", in_channels=32, out_channels=16,
               kernel_size=3, indice_key="ds")],
            [32, 32, 32], 2, 16, 1000,
        ),
        make_case(
            "fp16_subm3d_c32",
            [L("SubMConv3d", in_channels=32, out_channels=32, kernel_size=3,
               bias=False)],
            [32, 32, 32], 2, 32, 2000,
        ),
    ]
    torch.save({"cases": cases}, DATA_DIR / "golden_fp16.pt")
    print("FP16 GOLDEN DATA GENERATED")
    for f in sorted(DATA_DIR.iterdir()):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
