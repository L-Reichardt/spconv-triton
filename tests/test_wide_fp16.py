"""Wide-channel fp16 conv parity (C=128/256), bounded against the FP32
reference on fp16-exact inputs (the tights_fp16 method), so the tolerances
are meaningful (a few % of signal), not the near-vacuous noise-calibrated
fp16 bounds of the main suites."""

import torch

import pytest

from helpers import check_pipeline_case
from helpers_golden import load_golden, aux_case, aux_case_ids

FP16_CASES = aux_case_ids("golden_wide_fp16.pt")


@pytest.mark.parametrize("case_id", FP16_CASES)
def test_wide_fp16_case(impl, case_id):
    check_pipeline_case(impl, aux_case("golden_wide_fp16.pt", case_id))


def test_wide_fp16_tolerances_are_meaningful():
    """Every stored fp16 atol must be a small fraction of the reference
    signal (guards against the near-vacuous-bound failure mode)."""
    for case in load_golden("golden_wide_fp16.pt")["cases"]:
        e = case["expect"]
        refmax = float(e["out_features"].abs().max())
        assert e["atol_out"] <= 0.10 * refmax, (case["id"], "out")
        gmax = float(e["grad_input"].abs().max())
        assert e["atol_grad_input"] <= 0.15 * max(gmax, 1.0), (
            case["id"], "grad_input")
        for k, info in e["grad_params"].items():
            pmax = float(info["grad"].abs().max())
            assert info["atol"] <= 0.20 * max(pmax, 1.0), (case["id"], k)
