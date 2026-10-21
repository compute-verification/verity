"""The catalog bindings' records (`circuit_check.units`), on the synthetic tree of `test_cache`: `cli.run`'s ``--all`` path with
the real `targets.catalog`, `targets.suite` and `targets.select` over the tree's bindings (its ``syn/catalog.py``), first as a
full build, which stores each binding's record, then from the records, building only the bindings whose record misses.  And
`units.call_families` against `targets.call_families`' rules for word Call families made on demand and unbound kinds."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from test_cache import REPO, _bindings, _edit, _tree

#: one cached `--all` run over the synthetic bindings with the catalog's records; prints its cache stats and each target's value
HARNESS = """
import json, sys
from pathlib import Path

root, repo = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(root))
from circuit_check import cache as CC, checks as CK, cli, sidecars as SC, targets as TG

CC.TARGETS, CC.PINS, CC._TARGETS_FILE = root / "targets.py", root / "pins", str(root / "targets.py")
CK.PINS = root / "pins"
SC.REPO, SC.PACKAGES = root, ("syn",)
cli.REPO = root
inner = CC._interpreter_dirs
CC._interpreter_dirs = lambda: (*inner(), str(repo) + "/")      # the checker's own modules come from the repository
CC.trace(root)

import runpy
runpy.run_path(str(root / "targets.py"))["load_registries"]()
from syn import catalog, check


def bindings():
    for units, make in catalog.BINDINGS:
        yield units, make, "model" if units[0].startswith("_model_roots:") else units[0], units[0]


TG.load_registries = lambda: None
TG._registered = getattr(catalog, "registered", lambda: [])
TG.catalog_units = bindings
TG.pouw_bindings = lambda: []
TG.templates = lambda: []
TG.word_call_families = lambda: set()
TG.on_demand_boolean_families = lambda words: set()
TG.boolean_call_candidates = lambda words: set()
CK.check_definition = check.check_definition
rep = cli.run([], CK.Options(), everything=True, log=None, cache=root / "cache", units=True)
print(json.dumps({"stats": rep["cache"], "hits": sorted(t["target"] for t in rep["targets"] if "cache" in t),
                  "values": {t["target"]: t["value"] for t in rep["targets"]}}))
"""

#: the bindings beside the core and model ones: SynK at a static `syn.ksize` gives, SynX over a primitive of its own
BINDINGS = ('(("_sized_roots",), sized)', '(("_xor_roots",), xor)')


def _run(root: Path) -> dict:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([sys.executable, "-c", HARNESS, str(root), str(REPO)], capture_output=True, text=True, cwd=REPO, env=env)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert not out["stats"]["untraced"] and not out["stats"]["errors"] and not out["stats"]["changed"], out
    return out


def _fresh(root: Path) -> dict:
    """``root``'s verdicts as a full build gives them now: a copy of the tree without its cache."""
    copy = root.parent / f"{root.name}-fresh"
    shutil.rmtree(copy, ignore_errors=True)
    shutil.copytree(root, copy, ignore=shutil.ignore_patterns("cache"))
    out = _run(copy)
    assert "reused" not in out["stats"]["units"]
    return out


def _records(root: Path, units: tuple[str, ...]) -> list[Path]:
    def target(p: Path) -> list:
        try:
            return json.loads(p.read_text())["key"]["request"]["target"]
        except ValueError:
            return []
    return [p for p in (root / "cache").glob("*/*/*.json") if (t := target(p))[:1] == ["catalog-unit"] and tuple(t[1]) == units]


def test_a_definition_module_edit_rebuilds_exactly_the_bindings_that_read_it(tmp_path):
    """A full build stores every binding's record once the suite they give is its own; a later run whose task list misses builds
    only the bindings that read an edited Definition module (SynX's, SynK's, SynSub's, which only the core binding's roots
    reach), rechecks what changed, and gives the verdicts a full build gives."""
    root = _tree(tmp_path / "t")
    _bindings(root, *BINDINGS)
    first = _run(root)
    assert first["hits"] == [] and first["stats"]["units"] == {"stored": 5}
    assert set(first["values"]) == {"SynAdd_v1", "SynK_v1{K=2}", "SynOther_v1", "SynSub_v1", "SynTop_v1", "SynX_v1", "SynXor_v1"}
    for rel, old, new, unit in [("syn/xdef.py", "B.call(SynXor, a, b)", "B.call(SynXor, b, a)", "_xor_roots"),
                                ("syn/kdef.py", "out = B.call(SynAdd, out, b)", "out = B.call(SynAdd, b, out)", "_sized_roots"),
                                ("syn/sub.py", "B.call(SynAdd, a, b)", "B.call(SynAdd, b, a)", "_core_roots")]:
        _edit(root, rel, old, new)
        out = _run(root)
        assert out["stats"]["units"]["rebuilt"] == [unit], (rel, out["stats"]["units"])
        assert out["stats"]["units"]["reused"] == 4 and out["stats"]["units"]["stored"] == 1
        assert out["values"] == _fresh(root)["values"]
    assert out["hits"] == ["SynAdd_v1", "SynK_v1{K=2}", "SynOther_v1", "SynX_v1", "SynXor_v1"]


def test_a_missing_or_unreadable_record_rebuilds_its_binding(tmp_path):
    """A record that cannot be read, or is gone, counts as a miss: its binding is built and its record stored again, and every
    target is reused as before."""
    root = _tree(tmp_path)
    _bindings(root, *BINDINGS)
    _run(root)
    (bad,) = _records(root, ("_xor_roots",))
    bad.write_text("{")
    (gone,) = _records(root, ("_sized_roots",))
    gone.unlink()
    _edit(root, "syn/idle.py", "B.call(SynAdd, a, a)", "B.call(SynAdd, b, b)")    # no binding reads it: the task list misses
    out = _run(root)
    assert sorted(out["stats"]["units"]["rebuilt"]) == ["_sized_roots", "_xor_roots"]
    assert out["stats"]["units"]["stored"] == 2 and out["hits"] == sorted(out["values"])
    assert len(_records(root, ("_sized_roots",))) == 1


#: SynLater, whose body build registers the primitive it calls (as a body on the real tree registers its constants' primitives),
#: its binding, and the registered roots the catalog starts from: that primitive, once something registered it
LATE = """
from verity.ml.circuits import composite, primitive

from syn.prims import V


@composite("SynLater", 1, [], lambda S: ((("a", V), ("b", V)), V))
def SynLater(B, S, a, b):
    @primitive("SynLate", 1, [("a", V), ("b", V)], V)
    def SynLate(a, b):
        return (a - b) & 0xFFFF
    return B.call(SynLate, a, b)
"""
LATE_BINDING = """def late():
    from syn import late
    return [bind(late.SynLater)]


def registered():
    from verity.ml.circuits.defs import REGISTRY
    return [d for d in REGISTRY.defs.values() if d.id == "SynLate_v1"]


"""


def test_what_a_body_build_registers_leaves_every_record_counting(tmp_path):
    """SynLater's body build registers SynLate after the catalog took its registered roots: the full build reads its Calls off
    the suite it built (a second catalog would start from SynLate), so it stores every record, and an edit of SynLate rebuilds
    only SynLater's binding, whose build registers SynLate again, and gives the verdicts a full build gives."""
    root = _tree(tmp_path / "t")
    (root / "syn/late.py").write_text(LATE.lstrip("\n"))
    _edit(root, "syn/catalog.py", "BINDINGS = [", LATE_BINDING + "BINDINGS = [")
    _bindings(root, '(("_late_roots",), late)')
    first = _run(root)
    assert first["stats"]["units"] == {"stored": 4}
    assert {"SynLate_v1", "SynLater_v1"} <= set(first["values"])
    _edit(root, "syn/late.py", "(a - b)", "(b - a)")
    out = _run(root)
    assert out["stats"]["units"]["rebuilt"] == ["_late_roots"] and out["stats"]["units"]["reused"] == 3, out["stats"]["units"]
    assert out["values"] == _fresh(root)["values"] and out["values"]["SynLate_v1"] == [0x0105 - 3]


def test_a_binding_that_makes_no_root_is_held_by_every_record(tmp_path):
    """SynKn's body reads `syn.knob.N`, which a binding that makes no root sets from `syn.ksize`, a module SynKn's binding never
    reads: every record holds that binding, so an edit of `syn.ksize` misses them all, the run builds the whole catalog, and
    SynKn is checked at its new body."""
    root = _tree(tmp_path / "t")
    _bindings(root, '(("_set_roots",), setknob)', '(("_knobbed_roots",), knobbed)')
    assert _run(root)["stats"]["units"] == {"stored": 5}
    _edit(root, "syn/ksize.py", "K = 1", "K = 2")
    out = _run(root)
    assert out["stats"]["units"] == {"stored": 5} and out["hits"] == ["SynAdd_v1", "SynOther_v1", "SynSub_v1", "SynTop_v1"]
    assert out["values"] == _fresh(root)["values"] and out["values"]["SynKn_v1"] == [3 + 3 * 0x0105]


#: SynStep at a static Definition its body doesn't call, bound by one binding and unbound by another (as `AttnDotStep_v1` takes
#: its DOT from `boolean_version` and from its sidecar): two Definitions of one id whose descriptors differ
STEP = """
from verity.ml.circuits import composite

from syn.prims import SynAdd, V


@composite("SynInner", 1, [], lambda S: ((("a", V), ("b", V)), V))
def SynInner(B, S, a, b):
    return B.call(SynAdd, a, b)


@composite("SynStep", 1, ["OP"], lambda S: ((("a", V), ("b", V)), V))
def SynStep(B, S, a, b):
    return B.call(SynAdd, b, a)
"""
STEP_BINDINGS = """def unbound():
    from syn import step
    return [bind(step.SynStep, OP=step.SynInner)]


def bound():
    from syn import step
    return [bind(step.SynStep, OP=bind(step.SynInner))]


"""


def test_two_definitions_of_one_id_are_the_one_the_catalog_keeps(tmp_path):
    """Two bindings reach two Definitions of one id (SynStep with its OP bound and unbound): the suite holds the one
    `targets.reachable` keeps, and so does the task list from the records, which takes each id from the record of the binding it
    is reached through, so a full build stores every record, a run from them reuses every target, and a target that misses (its
    pin changed) is made from that binding's root, not the other's."""
    from test_cache import _set_pin

    root = _tree(tmp_path / "t")
    (root / "syn/step.py").write_text(STEP.lstrip("\n"))
    _edit(root, "syn/catalog.py", "BINDINGS = [", STEP_BINDINGS + "BINDINGS = [")
    _bindings(root, '(("_unbound_roots",), unbound)', '(("_bound_roots",), bound)')
    first = _run(root)
    assert first["stats"]["units"] == {"stored": 5}, first["stats"]["units"]
    step = "SynStep_v1{OP=SynInner_v1}"
    assert step in first["values"]
    _edit(root, "syn/idle.py", "B.call(SynAdd, a, a)", "B.call(SynAdd, b, b)")    # no binding reads it: the task list misses
    out = _run(root)
    assert out["stats"]["units"].get("reused") == 5 and out["stats"]["units"].get("rebuilt") == [], out["stats"]["units"]
    assert out["hits"] == sorted(out["values"])
    _set_pin(root, step, {"and": 1})
    out = _run(root)
    assert out["stats"]["units"].get("reused") == 5 and "stale" not in out["stats"]["units"], out["stats"]["units"]
    assert out["hits"] == sorted(set(out["values"]) - {step})
    _edit(root, "syn/step.py", "B.call(SynAdd, b, a)", "B.call(SynAdd, a, b)")
    out = _run(root)
    assert sorted(out["stats"]["units"]["rebuilt"]) == ["_bound_roots", "_unbound_roots"], out["stats"]["units"]
    assert out["values"] == _fresh(root)["values"]


def test_the_calls_from_records_are_the_catalogs_for_made_on_demand_and_unbound_kinds(monkeypatch):
    """`units.call_families` gives `targets.call_families`' Calls: a word Call family made on demand (`Const32[0x..]_v1`) has
    its Boolean version among them though no record holds one, and a vocabulary kind that doesn't bind refuses them by name, as
    `targets.word_call_families` does (left out, its family's partition would be judged as no Call's)."""
    from circuit_check import targets as TG, units as UN

    nothing = UN.Derived([], {}, {})
    with monkeypatch.context() as m:
        m.setattr(TG, "word_call_families", lambda: {"Const32[0x00000001]_v1", "GatherI32x16_v1"})
        m.setattr(TG, "boolean_call_candidates", lambda words: set())
        assert UN.call_families([], nothing, {}) == {"Const32[0x00000001]_v1", "GatherI32x16_v1",
                                                    "Const32[0x00000001]_v2", "GatherI32x16_v2"}
    monkeypatch.setattr(TG, "_SHORT", {})
    monkeypatch.setattr(TG, "_BOUND", {})
    monkeypatch.setattr(TG, "load_registries", lambda: None)
    monkeypatch.setattr(TG, "templates", lambda: [])
    monkeypatch.setattr(TG, "_vocabulary_bindings", lambda: [("b1-eager-v3/dsv4_indexer_compress", "KeyError: 'ROPE'")])
    with pytest.raises(RuntimeError, match=r"b1-eager-v3/dsv4_indexer_compress: KeyError: 'ROPE'"):
        UN.call_families([], nothing, {})
