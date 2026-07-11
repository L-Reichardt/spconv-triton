"""Wide-channel fp32 conv parity (C=128/256, subm + regular, fwd+bwd).

Pins the production-backbone channel-width region: a Triton GEMM multi-tile
or boundary-masking bug above C=64 (the previous max tested width) would pass
every frozen suite but fails here.
"""

import pytest

from helpers import check_pipeline_case
from helpers_golden import aux_case, aux_case_ids

CONV_CASES = aux_case_ids("golden_wide_conv.pt")


@pytest.mark.parametrize("case_id", CONV_CASES)
def test_wide_conv_case(impl, case_id):
    check_pipeline_case(impl, aux_case("golden_wide_conv.pt", case_id))
