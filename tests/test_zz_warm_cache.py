"""Warm-cache persistence witness (ACTIVE only under SPCONV_TEST_EXPECT_WARM=1).

Focused companion to the session-wide guard in ``conftest.py``. The ``zz`` name
makes it collect last, so every other test has already populated (cold) or
reused (warm) the shared Triton cache. It then re-exercises a representative
subm conv + a wide-channel conv and asserts ZERO new top-level kernel-hash dirs
appear — i.e. the cold env's compiled + autotuned kernels are reused with no
recompilation. This is exactly the persistence path (autotune ``cache_results``
reload + ``TRITON_CACHE_DIR`` cubin reuse) the user-facing autotune feature will
depend on.

The whole module is skipped unless ``SPCONV_TEST_EXPECT_WARM=1`` (so the default
matrix rows and a plain ``pytest tests/`` run do not collect it).
"""

import os

import pytest
from conftest import triton_kernel_hash_dirs
from helpers import check_pipeline_case
from helpers_golden import aux_case

pytestmark = pytest.mark.skipif(
    os.environ.get("SPCONV_TEST_EXPECT_WARM") != "1",
    reason="warm-cache witness runs only in the warm tox env (SPCONV_TEST_EXPECT_WARM=1)",
)

# (golden file, case id) — one representative subm conv + one wide-channel conv.
# Both are also exercised earlier in the suite, so on a warm cache re-running
# them must add no new compiled kernels.
WITNESS_CASES = [
    ("golden_conv.pt", "subm3d_k3"),
    ("golden_wide_conv.pt", "wide_conv3d_128_256"),
]


def test_warm_cache_no_recompile(impl):
    before = triton_kernel_hash_dirs()
    assert before, (
        "Triton cache is missing or empty — run the cold env first (tox -m warmcold)"
    )
    for golden, case_id in WITNESS_CASES:
        check_pipeline_case(impl, aux_case(golden, case_id))
    new = triton_kernel_hash_dirs() - before
    assert not new, (
        f"re-running {[c for _, c in WITNESS_CASES]} on a warm cache compiled "
        f"{len(new)} new Triton kernel(s): {sorted(new)}"
    )
