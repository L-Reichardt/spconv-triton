"""Tight TF32 golden generation (drop-in parity for SPCONV_ALLOW_TF32).

spconv exposes a runtime knob ``constants.SPCONV_ALLOW_TF32`` (default False).
When set, the CUTLASS/cumm conv GEMM path uses TF32 tensor ops
(``tf32,tf32,f32``) instead of IEEE fp32 (verified in
``spconv/pytorch/ops.py``: ``use_tf32=constants.SPCONV_ALLOW_TF32`` on every
conv gemm call). The conv1x1 path (``torch.mm``) is NOT governed by that flag --
upstream and the port both leave it to ``torch.backends.cuda.matmul.allow_tf32``
-- so conv1x1 is intentionally excluded from this suite (it has no
SPCONV_ALLOW_TF32-dependent behavior to pin).

Like the tight-fp16 suite, TF32 results are bounded against the FP32 (IEEE)
REFERENCE computed on identical inputs/params:

    atol(tensor) = 3 x max-deviation(reference tf32 vs ieee) + 1e-3

so any implementation must stay within a small factor of the legitimate TF32
rounding error of the IEEE result.

IMPORTANT: spconv's algo tuner caches the selected (tf32 vs ieee) kernel per
PROCESS on first use, so toggling SPCONV_ALLOW_TF32 in-process is a no-op. The
ieee and tf32 reference runs must therefore happen in SEPARATE processes, with
the flag set before the first conv. This script orchestrates that:

    SPCONV_TEST_IMPL=spconv uv run python tests/gen_golden_tf32.py
"""

import argparse
import os
import subprocess
import sys
import zlib
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

import helpers
from helpers import (
    build_layer,
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
SENS_FLOOR_REL = 1e-4  # a tensor's tf32 dev must exceed this fraction of refmax
TIGHT_REL = 0.05  # atol_out must stay below this fraction of refmax


def seed_of(cid, salt=0):
    return (zlib.crc32(cid.encode()) + salt * 7919) % (2**31)


def L(cls, **ctor):
    return {"cls": cls, "ctor": ctor}


# ---------------------------------------------------------------------------
# Case catalogue (shared by both phases and merge; deterministic from seeds)
# ---------------------------------------------------------------------------


def _case_specs():
    """Return [(cid, layer_specs, ss, bs, channels, npoints, training)]."""
    return [
        (
            "tf32_subm3d_c128",
            [L("SubMConv3d", in_channels=128, out_channels=128, kernel_size=3)],
            [32, 32, 32],
            2,
            128,
            1500,
            True,
        ),
        (
            "tf32_conv3d_k3s2p1_c128",
            [
                L(
                    "SparseConv3d",
                    in_channels=128,
                    out_channels=128,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            ],
            [32, 32, 32],
            2,
            128,
            1500,
            True,
        ),
        (
            "tf32_inv3d_c128",
            [
                L(
                    "SparseConv3d",
                    in_channels=128,
                    out_channels=128,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    indice_key="ds",
                    bias=False,
                ),
                L(
                    "SparseInverseConv3d",
                    in_channels=128,
                    out_channels=128,
                    kernel_size=3,
                    indice_key="ds",
                    bias=False,
                ),
            ],
            [32, 32, 32],
            2,
            128,
            1200,
            True,
        ),
    ]


def _gen_input(cid, ss, bs, channels, npoints):
    g = torch.Generator().manual_seed(seed_of(cid))
    rows = []
    for b in range(bs):
        c = torch.unique(
            torch.stack([torch.randint(0, s, (npoints,), generator=g) for s in ss], 1),
            dim=0,
        )
        rows.append(torch.cat([torch.full((c.shape[0], 1), b, dtype=torch.long), c], 1))
    idx = torch.cat(rows).int()
    feats = torch.randn(idx.shape[0], channels, generator=g)
    return idx, feats


def build_skeleton(cid, layer_specs, ss, bs, channels, npoints, training):
    """Deterministic pipeline-case skeleton (fp32 input + params), no expect."""
    # Batch coverage: all existing tests run at batch size 4 (bzyx, batch>1).
    bs = 4
    idx, feats = _gen_input(cid, ss, bs, channels, npoints)
    g = torch.Generator().manual_seed(seed_of(cid, 1))
    specs = [dict(s) for s in layer_specs]
    for spec in specs:
        tmp = build_layer(impl, {**spec, "params": {}}, torch.float32, "cpu")
        bound = 3.0 / max(float(torch.tensor(tmp.weight.shape[1:]).prod()), 1.0) ** 0.5
        params = {
            "weight": torch.empty(tmp.weight.shape).uniform_(-bound, bound, generator=g)
        }
        if tmp.bias is not None:
            params["bias"] = torch.empty(tmp.weight.shape[0]).uniform_(
                -bound, bound, generator=g
            )
        spec["params"] = params
    return {
        "id": cid,
        "kind": "pipeline",
        "training": training,
        "add_input": False,
        "dtype": "float32",
        "layers": specs,
        "input": {
            "features": feats,
            "indices": idx,
            "spatial_shape": list(ss),
            "batch_size": bs,
        },
    }


def _grad_canon(cid, n_out, c_out):
    g = torch.Generator().manual_seed(seed_of(cid, 2))
    return torch.randn(n_out, c_out, generator=g)


def run_canon(skel):
    """Run one pipeline case; return canonical out + (if training) grads."""
    res = run_pipeline_case(impl, skel, DEV)
    out, layers = res["out"], res["layers"]
    order = canonical_order(out.indices.cpu(), out.spatial_shape)
    r = {
        "out_indices": out.indices.cpu()[order],
        "out_features": out.features.detach().cpu().float()[order],
        "out_spatial_shape": list(out.spatial_shape),
    }
    if skel["training"]:
        n_out, c_out = out.features.shape
        grad_canon = _grad_canon(skel["id"], n_out, c_out)
        inv = inverse_permutation(order.to(DEV))
        out.features.backward(grad_canon.to(DEV)[inv].to(out.features.dtype))
        r["grad_out"] = grad_canon
        r["grad_input"] = res["feats_leaf"].grad.detach().cpu().float()
        r["grad_params"] = {
            k: p.grad.detach().cpu().float()
            for k, p in pipeline_named_params(layers).items()
        }
    torch.cuda.synchronize()
    return r


# ---------------------------------------------------------------------------
# Phase entry (separate process; flag fixed before any conv)
# ---------------------------------------------------------------------------


def run_phase(flag_on, out_path, n_runs):
    # Set the flag before the first conv: spconv's algo tuner caches the
    # tf32/ieee kernel choice per process, so this must be a fresh process.
    import spconv.constants as ref_constants

    ref_constants.SPCONV_ALLOW_TF32 = flag_on
    out = {}
    for cid, specs, ss, bs, ch, npts, training in _case_specs():
        skel = build_skeleton(cid, specs, ss, bs, ch, npts, training)
        runs = [run_canon(skel) for _ in range(n_runs)]
        out[cid] = runs
        print(
            f"  [{'tf32' if flag_on else 'ieee'}] {cid}: "
            f"{n_runs} run(s), out {tuple(runs[0]['out_features'].shape)}"
        )
    torch.save(out, out_path)


# ---------------------------------------------------------------------------
# Merge: build golden from ieee truth + tf32 deviation bands
# ---------------------------------------------------------------------------


def dev_of(a, b):
    return float((a.double() - b.double()).abs().max())


def atol(d):
    return K_MARGIN * d + FLOOR


def merge(ieee_path, tf32_path):
    ieee = torch.load(ieee_path, weights_only=False)
    tf32 = torch.load(tf32_path, weights_only=False)
    cases = []
    for cid, specs, ss, bs, ch, npts, training in _case_specs():
        skel = build_skeleton(cid, specs, ss, bs, ch, npts, training)
        truth = ieee[cid][0]
        # sanity: geometry identical across phases
        assert torch.equal(truth["out_indices"], tf32[cid][0]["out_indices"]), (
            f"{cid}: out indices differ between phases"
        )

        d_out = max(dev_of(r["out_features"], truth["out_features"]) for r in tf32[cid])

        refmax = float(truth["out_features"].abs().max())
        sens = d_out / max(refmax, 1e-9)
        expect = {
            "out_spatial_shape": truth["out_spatial_shape"],
            "out_indices": truth["out_indices"],
            "out_features": truth["out_features"],
            "atol_out": atol(d_out),
            "tf32_dev_out": d_out,
            "refmax_out": refmax,
        }
        print(
            f"  [merge] {cid}: tf32_dev_out={d_out:.3e} "
            f"({sens * 100:.3f}% of max|ref|), atol_out={atol(d_out):.3e}"
        )
        assert d_out >= SENS_FLOOR_REL * refmax, (
            f"{cid}: tf32 deviation {d_out:.3e} too small "
            f"(< {SENS_FLOOR_REL} x refmax {refmax:.3e}) -- tf32 not engaged "
            "in the reference; retune the case"
        )
        assert atol(d_out) < TIGHT_REL * refmax, (
            f"{cid}: atol_out {atol(d_out):.3e} not meaningfully tight "
            f"(>= {TIGHT_REL} x refmax {refmax:.3e})"
        )

        if training:
            d_gi = max(dev_of(r["grad_input"], truth["grad_input"]) for r in tf32[cid])
            expect["grad_out"] = truth["grad_out"]
            expect["grad_input"] = truth["grad_input"]
            expect["atol_grad_input"] = atol(d_gi)
            expect["tf32_dev_grad_input"] = d_gi
            expect["refmax_grad_input"] = float(truth["grad_input"].abs().max())
            gp = {}
            dev_gp = {}
            for k, gtruth in truth["grad_params"].items():
                d = max(dev_of(r["grad_params"][k], gtruth) for r in tf32[cid])
                gp[k] = {"grad": gtruth, "atol": atol(d)}
                dev_gp[k] = d
            expect["grad_params"] = gp
            expect["tf32_dev_grad_params"] = dev_gp

        case = dict(skel)
        case["expect"] = expect
        cases.append(case)

    torch.save({"cases": cases}, DATA_DIR / "golden_tf32.pt")
    print("TF32 GOLDEN DATA GENERATED")
    for f in sorted(DATA_DIR.iterdir()):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["ieee", "tf32"])
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.phase == "ieee":
        run_phase(False, args.out, n_runs=1)
        return
    if args.phase == "tf32":
        run_phase(True, args.out, n_runs=2)
        return

    # orchestrator: spawn both phases in fresh processes, then merge
    here = str(Path(__file__).resolve())
    ieee_path = "/tmp/tf32_dev/_phase_ieee.pt"
    tf32_path = "/tmp/tf32_dev/_phase_tf32.pt"
    Path("/tmp/tf32_dev").mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "SPCONV_TEST_IMPL": "spconv"}
    for phase, path in (("ieee", ieee_path), ("tf32", tf32_path)):
        print(f"=== spawning phase {phase} ===")
        subprocess.run(
            [sys.executable, here, "--phase", phase, "--out", path],
            env=env,
            check=True,
        )
    merge(ieee_path, tf32_path)


if __name__ == "__main__":
    main()
