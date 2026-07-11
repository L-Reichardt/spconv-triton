"""Drop-in API surface parity with spconv."""

import importlib
import inspect
import json

import pytest

from helpers import DATA_DIR


def surface():
    return json.loads((DATA_DIR / "golden_api.json").read_text())


def test_pytorch_exports(impl):
    missing = [n for n in surface()["pytorch"] if not hasattr(impl.pytorch, n)]
    assert not missing, f"missing from {impl.name}.pytorch: {missing}"


def test_ops_exports(impl):
    missing = [n for n in surface()["ops"] if not hasattr(impl.ops, n)]
    assert not missing, f"missing from {impl.name}.pytorch.ops: {missing}"


def test_functional_exports(impl):
    missing = [n for n in surface()["functional"]
               if not hasattr(impl.functional, n)]
    assert not missing, f"missing from functional: {missing}"


def test_top_level_exports(impl):
    missing = [n for n in surface()["top"] if not hasattr(impl.root, n)]
    assert not missing, f"missing from {impl.name}: {missing}"


def test_pytorch_utils_exports(impl):
    missing = [n for n in surface()["pytorch_utils"]
               if not hasattr(impl.putils, n)]
    assert not missing, f"missing from pytorch.utils: {missing}"


def test_hash_exports(impl):
    missing = [n for n in surface()["hash"] if not hasattr(impl.hash, n)]
    assert not missing, f"missing from pytorch.hash: {missing}"


@pytest.mark.parametrize("cls_name", sorted(surface()["ctors"].keys()))
def test_ctor_signatures(impl, cls_name):
    expected = surface()["ctors"][cls_name]
    cls = getattr(impl.pytorch, cls_name)
    try:
        sig = inspect.signature(cls.__init__)
        params = [p for p in sig.parameters if p != "self"]
    except (ValueError, TypeError):
        pytest.skip("signature not inspectable")
    assert params == expected, (
        f"{cls_name}.__init__ parameter mismatch:\n"
        f"  expected {expected}\n  actual   {params}")


def test_submodule_paths_importable(impl):
    for sub in ["pytorch.core", "pytorch.conv", "pytorch.pool",
                "pytorch.modules", "pytorch.tables", "pytorch.identity",
                "pytorch.functional", "pytorch.ops", "pytorch.utils",
                "pytorch.hash", "core", "constants"]:
        importlib.import_module(f"{impl.name}.{sub}")
