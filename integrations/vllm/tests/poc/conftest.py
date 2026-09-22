from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from _helpers import make_tiny_run  # noqa: E402

VECTORS = Path(__file__).resolve().parent / "vectors"


from verity_vllm import relations as _relations  # noqa: E402

# review fix (W2c): the relation registry is a process-global dict.  `verity_vllm_numerics`
# fills W2's `gemm.bf16.v1` slot at *import* time, so once `test_numerics_relations.py` is
# collected the "slot is unfilled" assertions of `test_relations.py` failed in a full run
# (3 order-dependent failures).  conftest is imported before any test module: this is W2's
# registry as shipped, before any plugin.
PRISTINE_REGISTRY = dict(_relations.REGISTRY)


@pytest.fixture(autouse=True)
def _isolate_relation_registry():
    """Undo whatever a test registered (plugins installed inside a test do not leak)."""
    saved = dict(_relations.REGISTRY)
    yield
    _relations.REGISTRY.clear()
    _relations.REGISTRY.update(saved)


@pytest.fixture
def pristine_registry():
    """W2's registry exactly as the package ships it (no plugin filled any slot)."""
    saved = dict(_relations.REGISTRY)
    _relations.REGISTRY.clear()
    _relations.REGISTRY.update(PRISTINE_REGISTRY)
    yield _relations.REGISTRY
    _relations.REGISTRY.clear()
    _relations.REGISTRY.update(saved)


@pytest.fixture
def tiny_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    make_tiny_run(run_dir)
    return run_dir


@pytest.fixture(scope="session")
def hash_vectors() -> dict:
    return json.loads((VECTORS / "hash.json").read_text())


@pytest.fixture(scope="session")
def tree_vectors() -> dict:
    return json.loads((VECTORS / "tree.json").read_text())
