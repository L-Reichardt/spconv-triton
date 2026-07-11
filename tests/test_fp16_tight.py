"""Tight fp16 parity: fp16 layer results bounded against FP32 references.

Tolerances are 3x the reference implementation's own fp16-vs-fp32 deviation
(plus a 1e-3 floor) on bit-identical fp16-exact inputs/params - orders of
magnitude tighter than the noise-calibrated fp16 cases in the main suite
(audit remediation; mutation-verified at freeze time)."""

from pathlib import Path

import pytest
import torch

from helpers import check_pipeline_case

FP16_DATA_DIR = Path(__file__).parent / "data"

_GOLDEN = None


def golden():
    global _GOLDEN
    if _GOLDEN is None:
        _GOLDEN = torch.load(
            FP16_DATA_DIR / "golden_fp16.pt", map_location="cpu", weights_only=False
        )
    return _GOLDEN


def case_ids():
    return [c["id"] for c in golden()["cases"]]


@pytest.mark.parametrize("case_id", case_ids())
def test_fp16_tight_case(impl, case_id):
    case = next(c for c in golden()["cases"] if c["id"] == case_id)
    check_pipeline_case(impl, case)


def test_tolerances_are_meaningful():
    """Guard against the audited failure mode: every stored atol must be a
    small fraction of the reference signal (no near-vacuous bounds)."""
    for case in golden()["cases"]:
        e = case["expect"]
        refmax = float(e["out_features"].abs().max())
        assert e["atol_out"] <= 0.10 * refmax, (case["id"], "out")
        gmax = float(e["grad_input"].abs().max())
        assert e["atol_grad_input"] <= 0.15 * max(gmax, 1.0), (
            case["id"], "grad_input")
        for k, info in e["grad_params"].items():
            pmax = float(info["grad"].abs().max())
            assert info["atol"] <= 0.20 * max(pmax, 1.0), (case["id"], k)
