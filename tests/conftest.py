"""Pytest configuration: GPU guard, implementation fixture, warm-cache guard."""

import hashlib
import os
from pathlib import Path

import pytest
import torch

from helpers import load_impl, DATA_DIR


def pytest_configure(config):
    if not torch.cuda.is_available():
        raise pytest.UsageError(
            "These tests require a GPU (torch.cuda.is_available() == False). "
            "Full GPU support is a hard requirement of this project.")


def triton_kernel_hash_dirs() -> set[str]:
    """Top-level kernel-hash directory names under the active Triton cache.

    Triton persists each compiled kernel under ``TRITON_CACHE_DIR/<hash>/``. A
    new top-level dir appearing during a run means a kernel was (re)compiled.
    Lock-file / metadata churn inside existing dirs is ignored (names only).
    """
    cache = os.environ.get("TRITON_CACHE_DIR")
    if not cache or not Path(cache).is_dir():
        return set()
    return {p.name for p in Path(cache).iterdir() if p.is_dir()}


@pytest.fixture(scope="session", autouse=True)
def warm_cache_guard():
    """Under ``SPCONV_TEST_EXPECT_WARM=1`` (the warm tox env), assert the whole
    session reused the Triton compile cache the cold env produced: no new
    top-level kernel-hash dir may appear across the session (zero
    recompilation). Requires the cold env to have populated the shared cache
    first. No-op — and no cache dependency — unless the flag is set."""
    if os.environ.get("SPCONV_TEST_EXPECT_WARM") != "1":
        yield
        return
    before = triton_kernel_hash_dirs()
    if not before:
        pytest.fail(
            "SPCONV_TEST_EXPECT_WARM=1 but the Triton cache "
            f"({os.environ.get('TRITON_CACHE_DIR', '<unset>')}) is missing or "
            "empty — run the cold env first (tox -m warmcold)."
        )
    yield
    new = triton_kernel_hash_dirs() - before
    assert not new, (
        f"warm session compiled {len(new)} new Triton kernel(s) — expected zero "
        f"recompilation against the cold cache. New kernel-hash dirs: {sorted(new)}"
    )


@pytest.fixture(scope="session")
def impl():
    return load_impl()


@pytest.fixture(scope="session")
def manifest():
    path = DATA_DIR / "MANIFEST.sha256"
    if not path.exists():
        pytest.fail("tests/data/MANIFEST.sha256 missing - golden data not frozen")
    entries = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        digest, name = line.split(maxsplit=1)
        entries[name.strip()] = digest
    return entries


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
