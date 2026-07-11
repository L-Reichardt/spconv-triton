"""TF32 parity: SPCONV_ALLOW_TF32 conv results bounded against FP32 references.

spconv's ``constants.SPCONV_ALLOW_TF32`` (default False) switches the conv GEMM
path to TF32 tensor ops. The port mirrors this knob, read live in
``spconv_triton.pytorch._impl.gemm``. Like upstream, the conv1x1 ``torch.mm``
path does NOT read the flag (it follows torch.backends.cuda.matmul.allow_tf32),
so conv1x1 is intentionally excluded from this suite.

Two contracts, mirroring the tight-fp16 suite:

* ``test_tf32_case`` (both impls): with the flag ON, every result stays within
  ``3 x (reference tf32-vs-ieee deviation) + 1e-3`` of the IEEE reference.
* ``test_tf32_actually_engaged`` (port only): toggling the flag must actually
  change the port's output (proving the wiring is live), and the flag-OFF
  result must sit closer to the IEEE truth than the flag-ON result.
* ``test_tolerances_are_meaningful``: the stored bands are tight (small
  fraction of signal) and sensitive (tf32 deviation above a floor).
"""

import contextlib
import importlib
from pathlib import Path

import pytest
import torch
from helpers import (
    canonical_order,
    check_pipeline_case,
    inverse_permutation,
    pipeline_named_params,
    run_pipeline_case,
)

DATA = Path(__file__).parent / "data"
SENS_FLOOR_REL = 1e-4  # a tensor counts as tf32-sensitive above this x refmax
TIGHT_REL = 0.05  # atol_out must stay below this x refmax
CHANGE_FRAC = 0.3  # flag must change output by >= this x the measured dev

# TF32 is a tensor-core fp32 path that ``SPCONV_ALLOW_TF32`` switches the Triton
# GEMM into (``tl.dot(input_precision="tf32")``). It exists on:
#   * NVIDIA Ampere or newer (CUDA compute capability >= 8), and
#   * AMD MI300 (CDNA3, gfx94x) via hipblaslt — wired into PyTorch only in
#     torch >= 2.7, and only with HIPBLASLT_ALLOW_TF32=1 in the environment
#     (the ROCm tox rows set it; see tox.ini).
# Everywhere else (pre-Ampere NVIDIA, non-MI300 / old-torch ROCm) the flag is
# inert, so the flag-forcing tests below (assert TF32 engages / stays in band)
# have nothing to exercise and are skipped. ``test_tolerances_are_meaningful``
# is data-only and stays active everywhere.


def _torch_minor() -> tuple[int, int]:
    major, minor = torch.__version__.split("+")[0].split(".")[:2]
    return int(major), int(minor)


def _tf32_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    if torch.version.hip is not None:  # AMD/ROCm
        if _torch_minor() < (2, 7):
            return False
        return torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx94")
    if torch.version.cuda is not None:  # NVIDIA
        return torch.cuda.get_device_capability()[0] >= 8
    return False


TF32_SUPPORTED = _tf32_supported()
requires_tf32 = pytest.mark.skipif(
    not TF32_SUPPORTED,
    reason="TF32 needs NVIDIA Ampere+ or AMD MI300 (gfx94x) on torch>=2.7",
)

_GOLDEN = None


def golden():
    global _GOLDEN
    if _GOLDEN is None:
        _GOLDEN = torch.load(
            DATA / "golden_tf32.pt", map_location="cpu", weights_only=False
        )
    return _GOLDEN


def case_ids():
    return [c["id"] for c in golden()["cases"]]


def get_case(case_id):
    return next(c for c in golden()["cases"] if c["id"] == case_id)


@contextlib.contextmanager
def tf32(impl, on):
    """Toggle the implementation's live SPCONV_ALLOW_TF32 flag.

    NOTE for the reference: spconv's algo tuner caches the (tf32/ieee) kernel
    per process on first conv, so an in-process toggle is a no-op for its GEMM
    path. The band test still holds (its flag-on result equals the IEEE truth,
    trivially within band); engagement is asserted only for the port.
    """
    m = importlib.import_module(f"{impl.name}.constants")
    prev = m.SPCONV_ALLOW_TF32
    m.SPCONV_ALLOW_TF32 = on
    try:
        yield
    finally:
        m.SPCONV_ALLOW_TF32 = prev


def _run_canon(impl, case):
    """Forward (+backward for training) returning canonical tensors."""
    res = run_pipeline_case(impl, case)
    out, layers = res["out"], res["layers"]
    order = canonical_order(out.indices.cpu(), out.spatial_shape)
    r = {"out_features": out.features.detach().cpu().float()[order]}
    if case["training"]:
        inv = inverse_permutation(order.to("cuda"))
        grad = case["expect"]["grad_out"].to("cuda")[inv]
        out.features.backward(grad.to(out.features.dtype))
        r["grad_input"] = res["feats_leaf"].grad.detach().cpu().float()
        r["grad_params"] = {
            k: p.grad.detach().cpu().float()
            for k, p in pipeline_named_params(layers).items()
        }
    torch.cuda.synchronize()
    return r


@requires_tf32
@pytest.mark.parametrize("case_id", case_ids())
def test_tf32_case(impl, case_id):
    """With TF32 on, results stay within the calibrated band of the IEEE ref."""
    case = get_case(case_id)
    with tf32(impl, True):
        check_pipeline_case(impl, case)


def _check_engaged(name, on, off, truth, dev):
    refmax = float(truth.abs().max())
    if dev < SENS_FLOOR_REL * max(refmax, 1e-9):
        return  # tensor not tf32-sensitive enough to assert engagement
    d_change = float((on.double() - off.double()).abs().max())
    assert d_change >= CHANGE_FRAC * dev, (
        f"{name}: toggling SPCONV_ALLOW_TF32 changed output by {d_change:.3e}, "
        f"expected >= {CHANGE_FRAC} x dev ({CHANGE_FRAC * dev:.3e}) -- "
        "flag not wired into this path"
    )
    d_off = float((off.double() - truth.double()).abs().max())
    d_on = float((on.double() - truth.double()).abs().max())
    assert d_off < d_on, (
        f"{name}: flag-off deviation from IEEE truth ({d_off:.3e}) not below "
        f"flag-on deviation ({d_on:.3e}) -- flag-off should be IEEE"
    )


@requires_tf32
@pytest.mark.parametrize("case_id", case_ids())
def test_tf32_actually_engaged(impl, case_id):
    """Port-only: prove the flag actually drives the output (band alone can't:
    an IEEE result also passes the band)."""
    if impl.name == "spconv":
        pytest.skip(
            "reference tf32 selection is per-process; engagement is a "
            "property of the port's live-flag wiring"
        )
    case = get_case(case_id)
    e = case["expect"]
    with tf32(impl, False):
        off = _run_canon(impl, case)
    with tf32(impl, True):
        on = _run_canon(impl, case)

    _check_engaged(
        f"{case_id}:out",
        on["out_features"],
        off["out_features"],
        e["out_features"],
        e["tf32_dev_out"],
    )
    if case["training"]:
        _check_engaged(
            f"{case_id}:grad_input",
            on["grad_input"],
            off["grad_input"],
            e["grad_input"],
            e["tf32_dev_grad_input"],
        )
        for k, dev in e["tf32_dev_grad_params"].items():
            _check_engaged(
                f"{case_id}:{k}",
                on["grad_params"][k],
                off["grad_params"][k],
                e["grad_params"][k]["grad"],
                dev,
            )


def test_tolerances_are_meaningful():
    """Guard the band ratios forever: tight (small fraction of signal) and
    sensitive (tf32 deviation above a floor)."""
    for case in golden()["cases"]:
        e = case["expect"]
        refmax = e["refmax_out"]
        assert e["atol_out"] <= TIGHT_REL * refmax, (case["id"], "out tight")
        assert e["tf32_dev_out"] >= SENS_FLOOR_REL * refmax, (
            case["id"],
            "out sensitive",
        )
        if case["training"]:
            gmax = float(e["grad_input"].abs().max())
            assert e["atol_grad_input"] <= TIGHT_REL * max(gmax, 1.0), (
                case["id"],
                "grad_input tight",
            )
            for k, info in e["grad_params"].items():
                pmax = float(info["grad"].abs().max())
                assert info["atol"] <= TIGHT_REL * max(pmax, 1.0), (
                    case["id"],
                    k,
                    "tight",
                )
