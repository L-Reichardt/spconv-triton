"""Shared golden-loading helpers for the auxiliary case families (the former
ext / fp16 / wide suites, now merged into tests/).

One implementation, used by every auxiliary test module. Golden payloads live
in tests/data and share the dict-of-cases format produced by the gen_golden*
generators. The core feature/grad helpers (build_layer, check_pipeline_case,
canonical_order, ...) still come from the frozen ``helpers`` module.
"""

from pathlib import Path
from typing import Any

import torch

DATA_DIR = Path(__file__).parent / "data"

_CACHE: dict[str, Any] = {}


def load_golden(name: str):
    """Load (and cache) a golden ``.pt`` payload from tests/data by file name."""
    if name not in _CACHE:
        _CACHE[name] = torch.load(
            DATA_DIR / name, map_location="cpu", weights_only=False
        )
    return _CACHE[name]


def aux_case_ids(name: str, kind: str | None = None) -> list[str]:
    """Case ids in a golden file, optionally filtered by ``kind``."""
    cases = load_golden(name)["cases"]
    return [c["id"] for c in cases if kind is None or c["kind"] == kind]


def aux_case(name: str, case_id: str) -> dict[str, Any]:
    """Fetch a single case dict by id from a golden file."""
    for c in load_golden(name)["cases"]:
        if c["id"] == case_id:
            return c
    raise KeyError(case_id)
