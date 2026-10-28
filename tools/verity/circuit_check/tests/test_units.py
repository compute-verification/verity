"""The catalog bindings' records (`circuit_check.units`), on the synthetic tree of `test_cache`: `cli.run`'s ``--all`` path with
the real `targets.catalog`, `targets.suite` and `targets.select` over the tree's bindings (its ``syn/catalog.py``), first as a
full build, which stores each binding's record, then from the records, building only the bindings whose record misses.  And
`units.call_families` against `targets.call_families`' rules for word Call families made on demand and unbound kinds.  And the
circuit_check suite's catalog, each binding built in a process of its own and made from what they reach in another
(`units.built_apart`, `units.from_manifest`), against `targets.suite` built whole."""
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


def _run(root: Path, harness: str = HARNESS) -> dict:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([sys.executable, "-c", harness, str(root), str(REPO)], capture_output=True, text=True, cwd=REPO, env=env)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert not out["stats"]["untraced"] and not out["stats"]["errors"] and not out["stats"]["changed"], out
    return out


def _fresh(root: Path, harness: str = HARNESS) -> dict:
    """``root``'s verdicts as a full build gives them now: a copy of the tree without its cache."""
    copy = root.parent / f"{root.name}-fresh"
    shutil.rmtree(copy, ignore_errors=True)
    shutil.copytree(root, copy, ignore=shutil.ignore_patterns("cache"))
    out = _run(copy, harness)
    assert "reused" not in out["stats"]["units"]
    return out


def _records(root: Path, units: tuple[str, ...] | None = None) -> list[Path]:
    def target(p: Path) -> list:
        try:
            return json.loads(p.read_text())["key"]["request"]["target"]
        except ValueError:
            return []
    return [p for p in (root / "cache").glob("*/*/*.json")
            if (t := target(p))[:1] == ["catalog-unit"] and (units is None or tuple(t[1]) == units)]


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


#: `units.store` with the `cache.seen` calls it makes counted (in ``<root>-seen.json``, beside the tree), and with PER_RECORD
#: digesting each record's reads on their own, as each record's `cache.seen` did before `cache.seen_all`
SEEN = """
from circuit_check import units as UN
PER_RECORD = {per_record}
seen, store, made = CC.seen, UN.store, []
CC.seen = lambda root_, got: made.append(1) or seen(root_, got)
if PER_RECORD:
    CC.seen_all = lambda root_, gots: [CC.seen(root_, g) for g in gots]


def counted(*a, **k):
    at = len(made)
    try:
        return store(*a, **k)
    finally:
        (root.parent / f"{{root.name}}-seen.json").write_text(json.dumps(len(made) - at))


UN.store = counted
"""


def test_records_digest_each_read_once_and_store_what_each_records_own_digest_gave(tmp_path):
    """`units.store` digests every read of the records it stores once, for them all (`cache.seen_all`), not once per record,
    and stores the bytes it stored when each record's reads were digested on their own: five bindings' records, the four
    that make roots each reading `syn.prims`."""
    root = _tree(tmp_path / "t")
    _bindings(root, *BINDINGS)
    made = tmp_path / "t-seen.json"
    harness = {p: HARNESS.replace("rep = cli.run(", SEEN.format(per_record=p) + "rep = cli.run(") for p in (True, False)}
    assert _run(root, harness[True])["stats"]["units"] == {"stored": 5}
    assert json.loads(made.read_text()) == 5
    before = {p.name: p.read_bytes() for p in _records(root)}
    shutil.rmtree(root / "cache")
    assert _run(root, harness[False])["stats"]["units"] == {"stored": 5}
    assert json.loads(made.read_text()) == 1
    after = {p.name: p.read_bytes() for p in _records(root)}
    assert len(after) == 5 and after == before
    assert sum("syn.prims" in json.loads(b)["reads"]["modules"] for b in after.values()) == 4


#: after the full build stored its records, `syn/prims.py` edited and every record stored again in the same process: the
#: digests of `syn.prims` (as a module, and in package `syn`'s module list) each record's `cache.store` was handed as seen
#: now, the ones the process holds, and the count stored
AGAIN = """
from circuit_check import units as UN
READ = {"files": [], "dirs": [], "modules": ["syn.prims"], "enums": ["syn"]}
def prims_of(d):
    return [d["modules"].get("syn.prims", {}).get("d"), (d["enums"].get("syn") or {}).get("prims")]
old = prims_of(CC._digests(root, READ))
prims = root / "syn/prims.py"
prims.write_text(prims.read_text() + "\\n_EDITED = 1\\n")
handed, store = [], CC.store
CC.store = lambda cache, k, root_, got, report, worker=None: handed.append(worker) or store(cache, k, root_, got, report, worker)
stored = UN.store(root / "cache", root, UN.built_entries(), only_rebuilt=False)
(root.parent / f"{root.name}-again.json").write_text(json.dumps({
    "stored": stored, "old": old, "held": prims_of(CC._digests(root, READ)), "new": prims_of(CC.seen(root, READ)),
    "seen": [prims_of(w) for w in handed]}))
"""


def test_a_file_edited_between_two_stores_in_one_run_is_seen_at_the_second(tmp_path):
    """Every `units.store` digests its records' reads afresh, so a module edited after one store is seen as it is now at the
    next one in the same process: each record that reads it is handed the new digest, which differs from the one it was
    built under (still the process's own), and none is stored.  All five read package `syn`'s module list, which digests
    `syn.prims` by content; the four that make roots read `syn.prims` itself too."""
    root = _tree(tmp_path / "t")
    _bindings(root, *BINDINGS)
    assert _run(root, HARNESS.replace("print(json.dumps(", AGAIN + "print(json.dumps("))["stats"]["units"] == {"stored": 5}
    got = json.loads((tmp_path / "t-again.json").read_text())
    assert got["old"] == got["held"] and all(o != n for o, n in zip(got["held"], got["new"]))
    assert [s[1] for s in got["seen"]] == [got["new"][1]] * 5
    assert sorted(s[0] for s in got["seen"] if s[0] is not None) == [got["new"][0]] * 4
    assert got["stored"] == 0


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


#: SynK at K = 2 and K = 5 inside another Definition's body, so a record holds K = 5 as reached, not as a root (a root is always
#: described), beside a smaller member of its family (a binding's smallest of a family is too)
KBIG = """
from verity.ml.circuits import composite
from verity.ml.circuits.defs import bind

from syn.kdef import SynK
from syn.prims import V


@composite("SynKBig", 1, [], lambda S: ((("a", V), ("b", V)), V))
def SynKBig(B, S, a, b):
    return B.call(bind(SynK, K=5), B.call(bind(SynK, K=2), a, b), b)
"""
KBIG_BINDING = """def kbig():
    from syn import kbig
    return [bind(kbig.SynKBig)]


"""


def test_a_pin_added_after_the_records_keeps_its_target_from_them(tmp_path):
    """SynK is reached at K = 2 (a root, and in SynKBig's body) and at K = 5 (in SynKBig's body), with `targets.MAX_GATES` at 2
    gates: the full build checks K = 2 alone, and its records describe no K = 5.  A pin of K = 5 then keeps it
    (`targets.pinned`): the run from the records reuses every one, makes K = 5 from SynKBig's binding and describes it (no
    `units.Stale`, which would build the whole catalog), checks it with its pin, reuses every other target, and gives the
    verdicts a full build gives."""
    from test_cache import _set_pin

    root = _tree(tmp_path / "t")
    (root / "syn/kbig.py").write_text(KBIG.lstrip("\n"))
    _edit(root, "syn/catalog.py", "BINDINGS = [", KBIG_BINDING + "BINDINGS = [")
    _bindings(root, *BINDINGS, '(("_kbig_roots",), kbig)')
    harness = HARNESS.replace("rep = cli.run(", "TG.MAX_GATES = 2\nrep = cli.run(")
    first = _run(root, harness)
    assert first["stats"]["units"] == {"stored": 6}, first["stats"]["units"]
    assert {"SynK_v1{K=2}", "SynKBig_v1"} <= set(first["values"]) and "SynK_v1{K=5}" not in first["values"]
    _set_pin(root, "SynK_v1{K=5}", {"and": 1})
    out = _run(root, harness)
    assert out["stats"]["units"] == {"reused": 6, "rebuilt": [], "stored": 0}, out["stats"]["units"]
    assert out["values"]["SynK_v1{K=5}"] == [3 + 5 * 0x0105]
    assert out["hits"] == sorted(set(out["values"]) - {"SynK_v1{K=5}"})
    assert out["values"] == _fresh(root, harness)["values"]


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


#: the catalog over the synthetic bindings, in a process of its own: built whole (``whole``, `targets.suite`), each binding built
#: in a process forked for it into a manifest (``build``, `units.built_apart`), or made from that manifest (``load``,
#: `units.from_manifest`, `units.make`); prints each suite target's descriptor and units, the catalog's errors and the ids made
#: along a binding's record (`units._materialize`), not from their own
APART = """
import json, sys
from pathlib import Path

root, mode, manifest = Path(sys.argv[1]), sys.argv[3], Path(sys.argv[4])
sys.path.insert(0, str(root))
from circuit_check import cache as CC, sidecars as SC, targets as TG, units as UN

SC.REPO, SC.PACKAGES = root, ("syn",)
import runpy
runpy.run_path(str(root / "targets.py"))["load_registries"]()
from syn import catalog


def bindings():
    for units, make in catalog.BINDINGS:
        yield units, make, "model" if units[0].startswith("_model_roots:") else units[0], units[0]


TG.load_registries = lambda: None
TG._registered = getattr(catalog, "registered", lambda: [])
TG.catalog_units = bindings
TG.pouw_bindings = lambda: []
along, materialize = [], UN._materialize
UN._materialize = lambda entries, objects, fid, *a, **k: along.append(fid) or materialize(entries, objects, fid, *a, **k)


def shown(suite, units):
    return {"targets": {fid: CC.descriptor(fn) for fid, fn in suite.items()},
            "units": {fid: sorted(units.get(fid, ())) for fid in suite}, "errors": sorted(map(list, TG.ROOT_ERRORS))}


if mode == "whole":
    out = shown(TG.suite(), TG.BINDINGS)
elif mode == "build":
    kept, dropped, _ = UN.built_apart(UN.made_entries(), 2)
    manifest.write_text(json.dumps(UN.manifest(kept, dropped)))
    out = {"errors": sorted(map(list, TG.ROOT_ERRORS)), "dropped": [list(u) for u in dropped]}
else:
    plan = UN.planned(UN.from_manifest(UN.made_entries(), json.loads(manifest.read_text())))
    out = {**shown({fid: UN.make(plan, fid) for fid in plan.derived.ids}, plan.derived.units), "along": along}
print(json.dumps(out))
"""

#: a model whose body raises, as a model the catalog can't build names its error
BAD = """
from verity.ml.circuits import composite

from syn.prims import V


@composite("SynBad", 1, [], lambda S: ((("a", V), ("b", V)), V))
def SynBad(B, S, a, b):
    raise ValueError("no body at these statics")
"""
#: a Definition whose body build kills the process building it, as the kernel's OOM killer does
DIE = """
import os
import signal

from verity.ml.circuits import composite

from syn.prims import V


@composite("SynDie", 1, [], lambda S: ((("a", V), ("b", V)), V))
def SynDie(B, S, a, b):
    os.kill(os.getpid(), signal.SIGKILL)
"""
APART_BINDINGS = """def bad():
    from syn import bad
    return [bind(bad.SynBad)]


def die():
    from syn import die
    return [bind(die.SynDie)]


"""


def _apart(root: Path, mode: str, ok: bool = True) -> dict | str:
    """``mode``'s output over ``root`` (`APART`, its manifest beside the tree), or its stderr when it fails (``ok`` false)."""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([sys.executable, "-c", APART, str(root), str(REPO), mode, str(root.parent / f"{root.name}-manifest.json")],
                       capture_output=True, text=True, cwd=REPO, env=env)
    if not ok:
        assert r.returncode != 0, r.stdout[-3000:]
        return r.stderr
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def _apart_tree(root: Path, *extra: str) -> Path:
    _tree(root)
    for rel, text in (("syn/step.py", STEP), ("syn/bad.py", BAD), ("syn/die.py", DIE)):
        (root / rel).write_text(text.lstrip("\n"))
    _edit(root, "syn/catalog.py", "BINDINGS = [", STEP_BINDINGS + APART_BINDINGS + "BINDINGS = [")
    _bindings(root, *BINDINGS, '(("_unbound_roots",), unbound)', '(("_bound_roots",), bound)', *extra)
    return root


def test_the_catalog_built_apart_and_made_from_its_manifest_is_the_one_built_whole(tmp_path):
    """Each binding built in a process of its own, and the catalog made in another from what they reach, is `targets.suite`
    built whole: the same targets at the same descriptors, with the same units, SynStep's the Definition of its id the catalog
    keeps, and a model whose body raises named in `targets.ROOT_ERRORS` as the catalog names it.  The process that makes it
    builds no binding's body: SynSub and SynAdd, which only a body reaches, are made from their ids."""
    root = _apart_tree(tmp_path / "t", '(("_model_roots:SynBad_v1",), bad)')
    whole = _apart(root, "whole")
    assert {"SynStep_v1{OP=SynInner_v1}", "SynSub_v1", "SynAdd_v1", "SynK_v1{K=2}"} <= set(whole["targets"])
    assert whole["errors"] == [["_model_roots:SynBad_v1", "ValueError: no body at these statics"]]
    assert _apart(root, "build") == {"errors": whole["errors"], "dropped": [["_model_roots:SynBad_v1"]]}
    load = _apart(root, "load")
    assert {k: load[k] for k in whole} == whole
    assert not {"SynSub_v1", "SynAdd_v1"} & set(load["along"]), load["along"]


def test_a_binding_whose_process_dies_is_named_and_a_model_is_dropped(tmp_path):
    """A binding's process killed while it builds (as by the kernel's OOM killer) leaves no result: a model's is dropped from the
    catalog and named in `targets.ROOT_ERRORS`, as the catalog names a model it can't build, and the rest is made as before;
    any other binding's fails the build by name."""
    root = _apart_tree(tmp_path / "t")
    whole = _apart(root, "whole")
    died = "its process died (signal 9), its result unwritten"
    _edit(root, "syn/catalog.py", "]\nFAMILIES", ', (("_model_roots:SynDie_v1",), die)]\nFAMILIES')
    assert _apart(root, "build") == {"errors": [["_model_roots:SynDie_v1", died]], "dropped": [["_model_roots:SynDie_v1"]]}
    load = _apart(root, "load")
    assert load["targets"] == whole["targets"] and load["errors"] == [["_model_roots:SynDie_v1", died]]
    _edit(root, "syn/catalog.py", '(("_model_roots:SynDie_v1",), die)', '(("_die_roots",), die)')
    assert f"RuntimeError: _die_roots: {died}" in _apart(root, "build", ok=False)


def test_one_xdist_worker_builds_the_catalog_and_the_others_take_it(tmp_path, monkeypatch):
    """Under xdist the first worker to ask builds the catalog apart and writes its manifest beside the workers' basetemps; a later
    one takes the manifest and builds nothing.  A builder that failed leaves its error there, and each later worker fails on it
    by name rather than build again."""
    import conftest
    from circuit_check import units as UN

    jobs: list[int] = []
    monkeypatch.setattr(UN, "built_apart",
                        lambda entries, n, budget, weights: jobs.append(n) or (entries[:1], [("gone",)], {}))
    monkeypatch.setattr(UN, "manifest", lambda kept, dropped: {"kept": len(kept), "dropped": dropped})
    monkeypatch.setattr(UN, "from_manifest", lambda entries, got: entries[:got["kept"]])
    assert conftest._built(["a", "b"], tmp_path, 3) == (["a"], "built apart")
    assert conftest._built(["a", "b"], tmp_path, 3) == (["a"], "taken from the worker that built it")
    assert jobs == [3]
    (tmp_path / conftest.MANIFEST).unlink()

    def dies(*_a):
        raise RuntimeError("_core_roots: its process died (signal 9), its result unwritten")

    monkeypatch.setattr(UN, "built_apart", dies)
    with pytest.raises(RuntimeError, match="signal 9"):
        conftest._built(["a"], tmp_path, 3)
    monkeypatch.setattr(UN, "built_apart", lambda *a: pytest.fail("built again"))
    with pytest.raises(RuntimeError, match="the xdist worker that built circuit-check's catalog failed: RuntimeError: _core"):
        conftest._built(["a"], tmp_path, 3)
