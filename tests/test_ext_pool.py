"""Extension pooling coverage: subm pools, dilation, kv=125."""

import pytest

from helpers import check_pipeline_case
from helpers_golden import aux_case, aux_case_ids

POOL_CASES = aux_case_ids("golden_ext_pool.pt")


@pytest.mark.parametrize("case_id", POOL_CASES)
def test_ext_pool_case(impl, case_id):
    check_pipeline_case(impl, aux_case("golden_ext_pool.pt", case_id))
