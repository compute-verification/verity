"""The primitives, composites and serving family's split (`integrations/vllm/tools/migrate/split_prims_serve.py`) moved no
Definition: at every statics circuit-check binds, the family's targets are the same tasks in the same order, with the same
descriptor digests, the same registered ids and the same pins as on the main it was cut from
(`fixtures/split/prims_serve.json`, the script's `pin` of that main).  A digest pins the whole descriptor, so a Definition
whose body, statics or callees changed in the move fails here; the import-boundary test (integrations/vllm
`tests/program/test_prims_serve_boundaries.py`) cannot see that."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from circuit_check import checks as CK
from circuit_check import cli
from circuit_check import targets as TG

REPO = Path(__file__).resolve().parents[4]
FIXTURE = Path(__file__).parent / "fixtures" / "split" / "prims_serve.json"
SCRIPT = REPO / "integrations" / "vllm" / "tools" / "migrate" / "split_prims_serve.py"


def _script():
    spec = importlib.util.spec_from_file_location("split_prims_serve", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("split_prims_serve", mod)
    spec.loader.exec_module(mod)
    return mod


def _suite() -> dict:
    """`TG.suite()`, shared with `test_circuit_check` when it is loaded (building it is about a minute)."""
    other = sys.modules.get("test_circuit_check")
    return other.SUITE if other is not None and hasattr(other, "SUITE") else TG.suite()


@pytest.fixture(scope="module")
def now() -> dict:
    S = _script()
    TG.load_registries()
    suite = _suite()
    real = TG.suite
    TG.suite = lambda: suite
    try:
        tasks = cli._tasks([], CK.Options(), True, None)
    finally:
        TG.suite = real
    family_ids = S.family_ids()
    families = {fid: TG.family(fn) for fid, fn in suite.items()}
    fam = set(family_ids)
    snap = {"family_ids": family_ids, "tasks": [[t[0], t[1]] for t in tasks], "families": families,
            "digests": {fid: S._digest(fn) for fid, fn in suite.items() if families[fid] in fam},
            "registered": list(TG.REGISTERED)}
    return S.pin(snap)


@pytest.fixture(scope="module")
def pinned() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.mark.slow
def test_family_definitions_are_the_same(now, pinned):
    assert now["family_ids"] == pinned["family_ids"]
    assert now["registered"] == pinned["registered"]


@pytest.mark.slow
def test_task_list_is_the_same(now, pinned):
    assert now["tasks"] == pinned["tasks"]
    assert now["pins"] == pinned["pins"]


@pytest.mark.slow
def test_descriptor_digests_are_the_same(now, pinned):
    changed = sorted(label for label in pinned["digests"] if now["digests"].get(label) != pinned["digests"][label])
    assert not changed, f"descriptor digests that changed: {changed[:20]}"
