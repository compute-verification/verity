"""`dead_code_census.py --tests-for`: the tests that go with a deleted set, and those to edit.

The rule on small maps (a test goes when every library module it reaches is deleted, a helper goes when only going tests reach
it), and the map on the tree: what a file names itself, through the helpers it names, and never through the library.
"""
from __future__ import annotations

import functools
import json

import pytest

from tests import dead_code_census as census

A, B, C = "lib.a", "lib.b", "lib.c"


def plan(targets: dict[str, set[str]], helpers: dict[str, set[str]], deleted: set[str]) -> dict:
    return census.tests_for(targets, {rel: helpers.get(rel, set()) for rel in targets}, deleted)


def test_a_test_goes_only_when_all_it_reaches_is_deleted():
    p = plan({"tests/test_a.py": {A}, "tests/test_ab.py": {A, B}, "tests/test_c.py": {C}}, {}, {A})
    assert p["goes"] == ["tests/test_a.py"]
    assert p["edit"] == {"tests/test_ab.py": {"deleted": [A], "others": [B], "named_for": None}}
    assert p["stay"] == 1


def test_a_helper_goes_only_with_every_test_that_reaches_it():
    targets = {"tests/test_a.py": {A}, "tests/test_b.py": {B}, "tests/_util.py": {A}, "tests/_shared.py": {A}}
    helpers = {"tests/test_a.py": {"tests/_util.py", "tests/_shared.py"}, "tests/test_b.py": {"tests/_shared.py"}}
    p = plan(targets, helpers, {A})
    assert p["goes"] == ["tests/test_a.py"]
    assert p["helpers_go"] == ["tests/_util.py"]
    assert list(p["edit"]) == ["tests/_shared.py"]


def test_a_helper_no_test_reaches_goes_when_its_targets_do():
    p = plan({"tests/_orphan.py": {A}, "tests/_other.py": {A, B}}, {}, {A})
    assert p["helpers_go"] == ["tests/_orphan.py"]
    assert list(p["edit"]) == ["tests/_other.py"]


def test_a_test_that_reaches_nothing_stays():
    p = plan({"tests/test_static.py": set()}, {}, {A})
    assert p["goes"] == [] and p["edit"] == {} and p["stay"] == 1


@pytest.mark.parametrize(("rel", "expected"), [
    ("tests/x/test_seed_rule.py", "lib.x.seed_rule"),
    ("tests/x/test_seed_rule_negative.py", "lib.x.seed_rule"),
    ("tests/x/test_seed.py", None),
])
def test_named_for_matches_the_module_a_file_is_named_after(rel, expected):
    assert census.named_for(rel, {"lib.x.seed_rule", "lib.x.other"}) == expected


@functools.lru_cache(maxsize=1)
def _tree():
    g = census.Graph()
    return g, *census.test_map(g)


def library_imports(g: census.Graph, rel: str) -> set[str]:
    return {m for m in g.edges.get(g.mod_of[rel], ()) if census.is_library(g, m) and not g.file_of[m].endswith("__init__.py")}


@pytest.mark.slow
def test_the_map_reaches_every_library_module_a_test_imports():
    g, targets, _ = _tree()
    missed = {rel: sorted(library_imports(g, rel) - ms) for rel, ms in targets.items() if library_imports(g, rel) - ms}
    assert not missed, dict(list(missed.items())[:5])


@pytest.mark.slow
def test_the_map_follows_the_helpers_a_test_names_and_never_the_library():
    g, targets, helpers = _tree()
    assert any(helpers.values()), "no test names a helper"
    for rel, hs in helpers.items():
        for h in hs:
            assert targets[h] <= targets[rel], (rel, h)
    open_ = [rel for rel, ms in targets.items() if {m for m in g.closure(ms) if census.is_library(g, m)} - ms]
    assert len(open_) > len(targets) // 2, "the targets are closed under the library's imports"


@pytest.mark.slow
def test_the_map_counts_only_library_modules():
    g, targets, _ = _tree()
    outside = {m for ms in targets.values() for m in ms if not census.is_library(g, m)}
    assert not outside, sorted(outside)[:10]


@pytest.mark.slow
def test_an_item_that_names_no_library_module_is_refused():
    g, _, _ = _tree()
    with pytest.raises(SystemExit, match="names no module of verity_vllm"):
        census.deleted_set(g, ["verity_vllm/no_such_module.py"])
    m = "verity_vllm.pipeline.cli"
    assert census.deleted_set(g, [g.file_of[m]]) == {m}
    assert m in census.deleted_set(g, ["verity_vllm/pipeline/"])


@pytest.mark.slow
def test_the_command_writes_its_plan(tmp_path, capsys):
    g, targets, helpers = _tree()
    rel, (m,) = next((rel, ms) for rel, ms in sorted(targets.items()) if len(ms) == 1 and rel.rsplit("/", 1)[-1].startswith("test_"))
    out = tmp_path / "plan.json"
    assert census.main(["--tests-for", m, "--out", str(out)]) == 0
    assert rel in json.loads(out.read_text())["goes"]
    assert capsys.readouterr().out.startswith("deleting 1 modules: ")
