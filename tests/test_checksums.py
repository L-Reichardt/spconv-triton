"""Integrity of the merged frozen test contract.

Recursive: every golden file AND every test source file under tests/ (incl.
subdirs and non-Python files) must appear in data/MANIFEST.sha256 with a
matching sha256. ``__pycache__`` and the manifest itself are excluded.
"""

from pathlib import Path

from conftest import sha256_of

TESTS_DIR = Path(__file__).parent


def all_files() -> set[str]:
    out = set()
    for f in sorted(TESTS_DIR.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(TESTS_DIR).as_posix()
        if rel == "data/MANIFEST.sha256" or "__pycache__" in rel:
            continue
        out.add(rel)
    return out


def test_manifest_covers_everything(manifest):
    actual = all_files()
    assert actual == set(manifest.keys()), (
        f"file set drift: extra={actual - set(manifest)} "
        f"missing={set(manifest) - actual}"
    )


def test_checksums_match(manifest):
    for rel, digest in manifest.items():
        path = TESTS_DIR / rel
        assert path.exists(), f"{rel} missing"
        actual = sha256_of(path)
        assert actual == digest, (
            f"{rel} was modified! frozen={digest[:12]} actual={actual[:12]}"
        )
