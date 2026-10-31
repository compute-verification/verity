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


#: the registered roots the catalog starts from, two, each of a module of its own
TWO_REGISTERED = """def registered():
    from verity.ml.circuits.defs import bind
    from syn import idle, xdef
    return [bind(idle.SynIdle), bind(xdef.SynX)]


"""


def test_the_registered_roots_are_runs_in_their_order_each_a_record_of_its_own():
    """`units.registered_entries`: at most `units.REGISTERED_RECORDS` runs of the registered roots, in their order, each keyed
    and named apart; one when there are none."""
    from circuit_check import units as UN

    roots = [f"r{i}" for i in range(3 * UN.REGISTERED_RECORDS - 1)]
    es = UN.registered_entries(roots)
    assert [r for e in es for r in e.roots] == roots and len(es) == UN.REGISTERED_RECORDS
    assert len({e.name for e in es}) == len({json.dumps(e.key, sort_keys=True) for e in es}) == len(es)
    assert all(e.units == UN.REGISTERED for e in es)
    assert [(e.name, e.roots) for e in UN.registered_entries([])] == [("<registered>:0", [])]


def test_an_edit_of_a_registered_roots_module_rebuilds_only_the_record_that_holds_it(tmp_path):
    """Two registered roots take a record each (`units.registered_entries`): an edit of SynX's module rebuilds SynX's record
    alone, SynIdle's counts, and the run gives the verdicts a full build gives."""
    root = _tree(tmp_path / "t")
    _edit(root, "syn/catalog.py", "BINDINGS = [", TWO_REGISTERED + "BINDINGS = [")
    first = _run(root)
    assert first["stats"]["units"] == {"stored": 4}
    assert {"SynIdle_v1", "SynX_v1", "SynXor_v1"} <= set(first["values"])
    _edit(root, "syn/xdef.py", "B.call(SynXor, a, b)", "B.call(SynXor, b, a)")
    out = _run(root)
    assert out["stats"]["units"]["rebuilt"] == ["<registered>:1"], out["stats"]["units"]
    assert out["stats"]["units"]["reused"] == 3 and out["stats"]["units"]["stored"] == 1
    assert out["values"] == _fresh(root)["values"]


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


#: `HARNESS` as `circuit-check --all --jobs 2` runs it: each target checked in a process of its own forked from this one, and
#: so each binding built in one; ``here`` lists each `units._reach` and `units._materialize` this process ran (a body built
#: here), ``keys`` the targets' keys, ``<root>-made.jsonl`` (beside the tree) the process, target and binding of each
#: target `units.make` made, and ``<root>-parsed.jsonl`` each check's process and the source files it parsed that no
#: process it was forked from had (`cache._parsed`, `cache._top_imports`)
FORKED = (HARNESS.replace("rep = cli.run(", "import hashlib, os\nfrom circuit_check import units as UN\n"
                          "parsed, parse, top, work, checking = root.parent / f'{root.name}-parsed.jsonl', CC._parsed, "
                          "CC._top_imports, cli._work, []\n"
                          "def parsing(p):\n"
                          "    with CC.internal():\n"
                          "        if checking and hashlib.sha512(p.read_bytes()).hexdigest() not in CC._PARSED:\n"
                          "            checking[-1].append(str(p))\n"
                          "    return parse(p)\n"
                          "def topping(path, name):\n"
                          "    if checking and path not in CC._IMPORTS:\n"
                          "        checking[-1].append(path)\n"
                          "    return top(path, name)\n"
                          "def working(i):\n"
                          "    checking.append([])\n"
                          "    try:\n"
                          "        return work(i)\n"
                          "    finally:\n"
                          "        with CC.internal(), parsed.open('a') as f:\n"
                          "            f.write(json.dumps([os.getpid(), checking.pop()]) + '\\n')\n"
                          "CC._parsed, CC._top_imports, cli._work = parsing, topping, working\n"
                          "here, reach, materialize = [], UN._reach, UN._materialize\n"
                          "UN._reach = lambda *a, **k: here.append('reach') or reach(*a, **k)\n"
                          "UN._materialize = lambda *a, **k: here.append('materialize') or materialize(*a, **k)\n"
                          "keyed, key = [], cli.target_key\n"
                          "cli.target_key = lambda *a, **k: keyed.append(key(*a, **k)) or keyed[-1]\n"
                          "made, make = root.parent / f'{root.name}-made.jsonl', UN.make\n"
                          "def logged(plan, fid):\n"
                          "    with CC.internal(), made.open('a') as f:\n"
                          "        f.write(json.dumps([os.getpid(), fid, plan.derived.via[fid].name]) + '\\n')\n"
                          "    return make(plan, fid)\n"
                          "UN.make = logged\n"
                          "rep = cli.run(")
          .replace("units=True)", "units=True, jobs=2, isolate=True, fork=True)")
          .replace('"values": {', '"here": here, "keys": sorted({json.dumps(k, sort_keys=True) for k in keyed}), "values": {'))


@pytest.mark.slow
def test_bindings_built_apart_give_a_full_builds_targets_keys_and_records(tmp_path):
    """With ``--jobs``, each binding is built in a process of its own and each target made by the zygote that forks the checks
    of its binding's targets, so this process builds no body, and no check parses a source file again that the process it
    was forked from parsed (the task list's digest, `cache.warm`): the verdicts and keys a full build here gives, every binding's
    record stored, and the task list, so the next run builds nothing.  Those records count for a run that builds here, and an
    edit rebuilds exactly the binding that reads it, apart, with the verdicts a full build gives, reusing the targets a run
    that builds here reuses."""
    root, here = _tree(tmp_path / "t"), _tree(tmp_path / "here")
    _bindings(root, *BINDINGS)
    _bindings(here, *BINDINGS)
    whole = _run(here, FORKED.replace(", jobs=2, isolate=True, fork=True)", ")"))
    first = _run(root, FORKED)
    assert first["values"] == whole["values"] and first["keys"] == whole["keys"] and first["hits"] == []
    assert first["here"] == [] and whole["here"], whole["here"]
    made = [json.loads(ln) for ln in (tmp_path / "t-made.jsonl").read_text().splitlines()]
    by = {}
    for pid, _, via in made:
        by.setdefault(via, set()).add(pid)
    assert len(made) > len(by) and all(len(p) == 1 for p in by.values()), made
    assert len({pid for pid, _, _ in made}) <= 2 and not {pid for pid, _, _ in made} & {os.getpid()}, made
    parsed = [json.loads(ln) for ln in (tmp_path / "t-parsed.jsonl").read_text().splitlines()]
    assert len(parsed) == len(first["values"]) and not [p for _, p in parsed if p], json.dumps(parsed)
    assert first["stats"]["units"]["reused"] == 0 and first["stats"]["units"]["stored"] == 5, first["stats"]["units"]
    again = _run(root, FORKED)
    assert again["stats"]["task_list"] and again["hits"] == sorted(again["values"]) and again["here"] == []
    _edit(root, "syn/idle.py", "B.call(SynAdd, a, a)", "B.call(SynAdd, b, b)")    # no binding reads it: the task list misses
    out = _run(root)
    assert out["stats"]["units"]["reused"] == 5 and out["hits"] == sorted(out["values"]), out["stats"]["units"]
    for tree in (root, here):
        _edit(tree, "syn/xdef.py", "B.call(SynXor, a, b)", "B.call(SynXor, b, a)")
    out, built = _run(root, FORKED), _run(here)
    assert out["stats"]["units"]["rebuilt"] == ["_xor_roots"] and out["stats"]["units"]["reused"] == 4, out["stats"]["units"]
    assert out["stats"]["units"]["stored"] == 1 and out["here"] == [] and out["values"] == _fresh(root)["values"]
    assert out["hits"] == built["hits"] and "SynX_v1" not in out["hits"], (out["hits"], built["hits"])


@pytest.mark.slow
def test_a_target_that_misses_through_a_record_that_counts_is_made_where_it_is_checked(tmp_path):
    """With ``--jobs``, a target whose result misses while its binding's record counts (SynX, its pin changed) is made by the
    zygote that forks its check, as one through a binding built apart is (`cli._planned`): this process builds and describes
    no body to check it, and every target's verdict is the one a full build gives."""
    from test_cache import _set_pin

    root = _tree(tmp_path / "t")
    _bindings(root, *BINDINGS)
    _run(root, FORKED)
    _set_pin(root, "SynX_v1", {"and": 1})
    (tmp_path / "t-made.jsonl").unlink()
    out = _run(root, FORKED)
    units = out["stats"]["units"]
    assert units["reused"] == 5 and units["rebuilt"] == [] and "stale" not in units, units
    assert out["here"] == [] and out["hits"] == sorted(set(out["values"]) - {"SynX_v1"}), (out["here"], out["hits"])
    made = [json.loads(ln) for ln in (tmp_path / "t-made.jsonl").read_text().splitlines()]
    assert [(fid, via) for _, fid, via in made] == [("SynX_v1", "_xor_roots")], made
    assert out["values"] == _fresh(root)["values"]


#: `HARNESS`'s setup, then ``circuit-check --all --records-only -j 2`` (`cli.records`) into the tree's cache; prints its report
RECORDS_ONLY = HARNESS[:HARNESS.index("rep = cli.run(")] + 'print(json.dumps(cli.records(root / "cache", log=None, jobs=2)))\n'


def _records_only(root: Path) -> dict:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([sys.executable, "-c", RECORDS_ONLY, str(root), str(REPO)], capture_output=True, text=True, cwd=REPO, env=env)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def _copied(root: Path, to: Path) -> Path:
    """Another tree as ``root`` was made, at ``to``, with a copy of its cache: the records on another machine."""
    tree = _tree(to)
    _bindings(tree, *BINDINGS)
    shutil.copytree(root / "cache", tree / "cache")
    return tree


@pytest.mark.slow
def test_records_only_stores_every_record_and_a_forked_run_from_a_copy_of_them_builds_no_binding(tmp_path):
    """``--records-only`` builds each binding apart and stores its record, keying and checking no target; run again, it
    reuses them all and stores none.  A forked ``--all`` in another tree from a copy of that cache builds no binding and no
    body in its own process, with the verdicts a full build gives; with a module a binding read edited there, or a record
    gone, it builds that binding again, as with no records."""
    root = _tree(tmp_path / "t")
    _bindings(root, *BINDINGS)
    first = _records_only(root)
    assert first["reused"] == 0 and first["stored"] == len(first["rebuilt"]) == 5 and first["scratch_bytes"] > 0, first
    assert len(_records(root)) == len(list((root / "cache").glob("*/*/*.json"))) == 5     # no target's entry, no task list
    assert first["bytes"] == sum(p.stat().st_size for p in _records(root))
    assert _records_only(root) == {"reused": 5, "rebuilt": [], "scratch_bytes": 0, "stored": 0, "bytes": first["bytes"]}

    warm = _copied(root, tmp_path / "warm")
    out = _run(warm, FORKED)
    assert out["stats"]["units"] == {"reused": 5, "rebuilt": [], "stored": 0} and out["here"] == [] and out["hits"] == []
    assert out["values"] == _fresh(warm)["values"]

    gone = _copied(root, tmp_path / "gone")
    (rec,) = _records(gone, ("_sized_roots",))
    rec.unlink()
    again = _run(gone, FORKED)
    assert again["stats"]["units"] == {"reused": 4, "rebuilt": ["_sized_roots"], "stored": 1} and again["values"] == out["values"]

    edited = _copied(root, tmp_path / "edited")
    _edit(edited, "syn/xdef.py", "B.call(SynXor, a, b)", "B.call(SynXor, b, a)")
    out = _run(edited, FORKED)
    assert out["stats"]["units"] == {"reused": 4, "rebuilt": ["_xor_roots"], "stored": 1}, out["stats"]["units"]
    assert out["values"] == _fresh(edited)["values"]


#: a reference two bindings bind at two statics (as `verity.ml.kernels.tc`'s sidecar and the gemm-coordinate template bind
#: the NVFP4 dot), a word primitive whose Boolean Definition (a ``word`` and a ``build``, as `verity.ml.circuits.boolean.
#: trace` makes one) only another binding's body build registers (as `boolean_marlin_moe`'s gathers), and one whose Boolean
#: Definition `WORDMOD` registers as it is imported (as `ops.boolean`'s pieces are), which only SynVTop's body imports
WORDS = """
from verity.ml.circuits import bind, composite, primitive
from verity.ml.circuits.evaluation import ScalarReference

from syn.prims import SynAdd, V

SCALE = ScalarReference("syn.scale", lambda a, b, *, K: (a * K + b) & 0xFFFF, lambda K: ((V, V), V), statics=("K",))
BITS = []


@primitive("SynW", 1, [("a", V), ("b", V)], V)
def SynW(a, b):
    return (a - b) & 0xFFFF


@primitive("SynV", 1, [("a", V), ("b", V)], V)
def SynV(a, b):
    return a ^ b


@composite("SynWTop", 1, [], lambda S: ((("a", V), ("b", V)), V))
def SynWTop(B, S, a, b):
    if not BITS:
        @composite("SynWBits", 1, [], lambda S: ((("a", V), ("b", V)), V))
        def SynWBits(B, S, a, b):
            return B.call(SynAdd, a, b)
        SynWBits.word, SynWBits.build = SynW, lambda C, *x: list(x)
        BITS.append(SynWBits)
    return B.call(bind(BITS[0]), a, b)


@composite("SynVTop", 1, [], lambda S: ((("a", V), ("b", V)), V))
def SynVTop(B, S, a, b):
    from syn import wordmod  # noqa: F401
    return B.call(SynAdd, a, b)
"""
WORDMOD = """
from verity.ml.circuits import composite

from syn.prims import SynAdd, V
from syn.words import SynV


@composite("SynVBits", 1, [], lambda S: ((("a", V), ("b", V)), V))
def SynVBits(B, S, a, b):
    return B.call(SynAdd, a, b)


SynVBits.word, SynVBits.build = SynV, lambda C, *x: list(x)
"""
WORDS_BINDINGS = """def scale2():
    from syn import words
    return [words.SCALE.bind(K=2)]


def scale3():
    from syn import words
    return [words.SCALE.bind(K=3)]


def wtop():
    from syn import words
    return [bind(words.SynWTop)]


def wprim():
    from syn import words
    return [words.SynW]


def vtop():
    from syn import words
    return [bind(words.SynVTop), words.SynV]


"""
#: the check's value also says whether a Boolean Definition of the target's word is registered where it is checked, as
#: C-Flock's lowering looks one up (`verity_flock.ir_lower`'s ``_boolean_build``)
WORDED = """def worded(fid):
    from verity.ml.circuits.defs import REGISTRY

    return any(getattr(getattr(d, "word", None), "id", None) == fid and callable(getattr(d, "build", None))
               for d in REGISTRY.defs.values())


"""
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


@pytest.mark.slow
def test_bindings_built_apart_check_the_reference_and_find_the_boolean_definitions_a_full_build_does(tmp_path):
    """Built apart, a reference two bindings bind at two statics is checked at the statics of the one `targets.reachable`
    keeps, not the first binding's (its descriptor holds them: `cache.descriptor`), and a word primitive's check finds the
    Boolean Definition of its word that only another binding's body build registers, or its import of a module no record
    reaches (`units.made_words`): the values and keys a full build gives, from the bindings built apart (no `units.Stale`).
    The records hold those Boolean Definitions, so a run from them finds them too."""
    from test_cache import _set_pin

    trees = []
    for name in ("t", "here"):
        root = _tree(tmp_path / name)
        (root / "syn/words.py").write_text(WORDS.lstrip("\n"))
        (root / "syn/wordmod.py").write_text(WORDMOD.lstrip("\n"))
        _edit(root, "syn/catalog.py", "BINDINGS = [", WORDS_BINDINGS + "BINDINGS = [")
        _bindings(root, '(("_scale2_roots",), scale2)', '(("_scale3_roots",), scale3)', '(("_wtop_roots",), wtop)',
                  '(("_wprim_roots",), wprim)', '(("_vtop_roots",), vtop)')
        _edit(root, "syn/check.py", "def check_definition(", WORDED + "def check_definition(")
        _edit(root, "syn/check.py", "evaluate(fn, 3, 0x0105)", "[evaluate(fn, 3, 0x0105), worded(fn.id)]")
        _edit(root, "syn/check.py", '"gates": fn.gates', '"gates": getattr(fn, "gates", 1)')
        trees.append(root)
    root, here = trees
    whole = _run(here, FORKED.replace(", jobs=2, isolate=True, fork=True)", ")"))
    values = whole["values"]
    assert values["syn.scale"] == [[3 * 3 + 0x0105], False] and values["SynW_v1"][1] and values["SynV_v1"][1], values
    first = _run(root, FORKED)
    assert first["stats"]["units"]["reused"] == 0 and "stale" not in first["stats"]["units"], first["stats"]["units"]
    assert first["values"] == values and first["keys"] == whole["keys"]
    _edit(root, "syn/idle.py", "B.call(SynAdd, a, a)", "B.call(SynAdd, b, b)")    # no binding reads it: the task list misses
    _set_pin(root, "SynW_v1", {"and": 1})
    out = _run(root, FORKED)
    assert out["stats"]["units"]["rebuilt"] == [] and "stale" not in out["stats"]["units"], out["stats"]["units"]
    assert out["values"] == values and "SynW_v1" not in out["hits"], out["hits"]


@pytest.mark.slow
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


@pytest.mark.slow
def test_bindings_built_apart_keep_a_pinned_target_and_read_no_pin(tmp_path):
    """Built apart, each binding's process describes the ids pinned here whatever their size (`targets.pinned`, handed down by
    `units.built_apart`): with `targets.MAX_GATES` at 2 gates, the pinned SynK at K = 5 is checked from what SynKBig's process
    reached, with the values and keys a full build gives and no body built here.  No binding's build reads a pin, so a new pin
    rebuilds no binding and changes only its own target's key."""
    from test_cache import _set_pin

    trees = []
    for name in ("t", "here"):
        root = _tree(tmp_path / name)
        (root / "syn/kbig.py").write_text(KBIG.lstrip("\n"))
        _edit(root, "syn/catalog.py", "BINDINGS = [", KBIG_BINDING + "BINDINGS = [")
        _bindings(root, *BINDINGS, '(("_kbig_roots",), kbig)')
        _set_pin(root, "SynK_v1{K=5}", {"and": 1})
        trees.append(root)
    root, here = trees
    forked = FORKED.replace("rep = cli.run(", "TG.MAX_GATES = 2\nrep = cli.run(")
    whole = _run(here, forked.replace(", jobs=2, isolate=True, fork=True)", ")"))
    assert whole["values"]["SynK_v1{K=5}"] == [3 + 5 * 0x0105], whole["values"]
    first = _run(root, forked)
    assert first["stats"]["units"]["reused"] == 0 and first["here"] == [], (first["stats"]["units"], first["here"])
    assert first["values"] == whole["values"] and first["keys"] == whole["keys"]
    _set_pin(root, "SynK_v1{K=2}", {"and": 1})
    out = _run(root, forked)
    units = out["stats"]["units"]
    assert units["rebuilt"] == [] and units["reused"] == first["stats"]["units"]["stored"], units
    assert out["values"] == whole["values"] and out["hits"] == sorted(set(out["values"]) - {"SynK_v1{K=2}"}), out["hits"]


#: a unit staged whole (`checks.UNIT_OWNERS` is ``syn.pouw.`` in `BLOCKED`): SynTile, made by `syn.build`'s function, which binds
#: it at no top level, and bound at the top level of `syn.pouw.tile` alone, which only SynHolder's body imports; the tiled
#: binding makes SynTile from `syn.build`, and the held one reaches it through SynHolder's body
TILE_FILES = {
    "syn/build.py": """
from verity.ml.circuits import composite

from syn.prims import SynAdd, V

_MADE = []


def tile():
    if not _MADE:
        @composite("SynTile", 1, [], lambda S: ((("a", V), ("b", V)), V))
        def SynTile(B, S, a, b):
            return B.call(SynAdd, B.call(SynAdd, a, b), b)
        _MADE.append(SynTile)
    return _MADE[0]
""",
    "syn/pouw/__init__.py": "",
    "syn/pouw/tile.py": "from syn.build import tile\n\nSynTile = tile()\n",
    "syn/holder.py": """
from verity.ml.circuits import composite

from syn.prims import V


@composite("SynHolder", 1, [], lambda S: ((("a", V), ("b", V)), V))
def SynHolder(B, S, a, b):
    from syn.pouw.tile import SynTile
    return B.call(SynTile, a, b)
""",
}
TILE_BINDINGS = """def tiled():
    from syn.build import tile
    return [bind(tile())]


def held():
    from syn import holder
    return [bind(holder.SynHolder)]


"""
#: each target's value, whether `checks.unit_block` applies to it, and the names of its failures; it reads the target's pin, so
#: a new pin reruns that target alone
BLOCK_CHECK = """
def check_definition(fn, opts, *, memo, pins, as_call=None, **kw):
    from circuit_check import checks as CK
    from verity.ml.circuits.evaluation.reference import evaluate

    F = CK.Findings()
    w = CK.unit_block(fn, F)
    pins.get("pieces", {}).get(fn.id)
    return {"failures": F.failures, "warnings": [], "gates": fn.gates,
            "value": [evaluate(fn, 3, 0x0105), w["applies"], [f["name"].split(" ")[0] for f in F.failures]]}
"""
#: `FORKED` with C-Flock's range at 2^5 witness bits, which SynTile's 48 port bits are past, and ``syn.pouw.`` PoUW's; a unit
#: counted here first, so what counting imports (the kernels its pieces register) is here before any check, as the catalog's
#: registries have it
BLOCKED = FORKED.replace("rep = cli.run(", "from verity_flock import circuit as FC\nfrom syn.prims import SynAdd\nFC.unit_witness(SynAdd)\n"
                         "FC.K_MAX, CK.UNIT_OWNERS = 5, ('syn.pouw.',)\nrep = cli.run(")


@pytest.mark.slow
def test_a_unit_past_the_range_fails_block_on_the_fast_path_as_in_a_full_build(tmp_path):
    """The fail-open case: from the bindings' records, built apart or reused, with ``--jobs`` or without, a unit staged whole
    and past C-Flock's range fails ``block/past-range`` as a full build fails it, though the module that makes it PoUW's
    (`sidecars.owner`, `checks.UNIT_OWNERS`) is imported only by another binding's body, which a full build builds before any
    check: once the records count and SynTile alone misses (its new pin), no process here builds SynHolder's body.  A lookup
    that read only what the checking process imported would skip the check."""
    from test_cache import _set_pin

    trees = []
    for name in ("t", "here"):
        root = _tree(tmp_path / name)
        for rel, text in TILE_FILES.items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(text.lstrip("\n"))
        (root / "syn/check.py").write_text(BLOCK_CHECK.lstrip("\n"))
        _edit(root, "syn/catalog.py", "BINDINGS = [", TILE_BINDINGS + "BINDINGS = [")
        _bindings(root, '(("_tiled_roots",), tiled)', '(("_held_roots",), held)')
        trees.append(root)
    root, here = trees
    in_turn = BLOCKED.replace(", jobs=2, isolate=True, fork=True)", ")")
    whole = _run(here, in_turn)
    assert whole["values"]["SynTile_v1"][1:] == [True, ["block/past-range"]], whole["values"]
    assert not [fid for fid, v in whole["values"].items() if v[2] and fid != "SynTile_v1"], whole["values"]
    first = _run(root, BLOCKED)
    assert first["here"] == [] and first["stats"]["units"]["reused"] == 0, (first["here"], first["stats"]["units"])
    assert first["values"] == whole["values"], (first["values"], whole["values"])
    for tree, harness in ((root, BLOCKED), (here, in_turn)):
        _set_pin(tree, "SynTile_v1", {"and": 1})
        out = _run(tree, harness)
        assert out["stats"]["units"]["rebuilt"] == [] and out["hits"] == sorted(set(whole["values"]) - {"SynTile_v1"}), out
        assert out["values"] == whole["values"], (out["values"], whole["values"])


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
    kept, dropped, _, _ = UN.built_apart(UN.made_entries(), 2)
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
#: a Definition whose body build kills the process that forked the one building it, as a node's lockup ends a whole run
ORPHAN = DIE.replace("SynDie", "SynOrphan").replace("os.kill(os.getpid()", "os.kill(os.getppid()")
APART_BINDINGS = """def bad():
    from syn import bad
    return [bind(bad.SynBad)]


def die():
    from syn import die
    return [bind(die.SynDie)]


def orphan():
    from syn import orphan
    return [bind(orphan.SynOrphan)]


"""


def _apart(root: Path, mode: str, ok: bool = True, tmp: Path | None = None) -> dict | str:
    """``mode``'s output over ``root`` (`APART`, its manifest beside the tree), or its stderr when it fails (``ok`` false);
    with ``tmp`` as its TMPDIR."""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **({"TMPDIR": str(tmp)} if tmp is not None else {})}
    r = subprocess.run([sys.executable, "-c", APART, str(root), str(REPO), mode, str(root.parent / f"{root.name}-manifest.json")],
                       capture_output=True, text=True, cwd=REPO, env=env)
    if not ok:
        assert r.returncode != 0, r.stdout[-3000:]
        return r.stderr
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def _apart_tree(root: Path, *extra: str) -> Path:
    _tree(root)
    for rel, text in (("syn/step.py", STEP), ("syn/bad.py", BAD), ("syn/die.py", DIE), ("syn/orphan.py", ORPHAN)):
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


def test_a_run_killed_while_its_bindings_build_leaves_no_file_behind(tmp_path):
    """The process that builds the bindings apart killed while they build (a node's lockup, the OOM killer) leaves nothing in
    its TMPDIR once the processes it forked end: each puts what it returns in a file of no name (`units._result`), which the
    kernel frees with the last process holding it; a directory of named files would stay, whatever they hold."""
    root = _apart_tree(tmp_path / "t", '(("_orphan_roots",), orphan)')
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    _apart(root, "build", ok=False, tmp=tmp)
    assert sorted(p.name for p in tmp.iterdir()) == []


#: a Definition whose body build runs a program, which the tracer cannot see (`cache.untraced`)
SPAWN = """
import os

from verity.ml.circuits import composite

from syn.prims import SynAdd, V


@composite("SynSpawn", 1, [], lambda S: ((("a", V), ("b", V)), V))
def SynSpawn(B, S, a, b):
    os.system("true")
    return B.call(SynAdd, a, b)
"""
SPAWN_BINDINGS = """def spawn():
    from syn import spawn
    return [bind(spawn.SynSpawn)]


"""
#: `HARNESS`'s tree with its bindings built apart one after another, traced; prints what the tracer could not see
APART_TRACED = HARNESS[:HARNESS.index("rep = cli.run(")] + (
    "from circuit_check import units as UN\nUN.built_apart(UN.made_entries(), 1, side=[])\nprint(json.dumps(CC.untraced()))\n")


def test_what_a_binding_built_apart_did_untraced_is_this_process_once(tmp_path):
    """Three bindings whose builds each run a program, built apart one after another: what each did that the tracer cannot see
    is this process's once.  A binding's process inherits what those that ended before it passed back, which it doesn't pass
    back again: passed again, the list doubles with each binding that ends."""
    root = _tree(tmp_path / "t")
    (root / "syn/spawn.py").write_text(SPAWN.lstrip("\n"))
    _edit(root, "syn/catalog.py", "BINDINGS = [", SPAWN_BINDINGS + "BINDINGS = [")
    _bindings(root, *BINDINGS, *(f'(("_spawn{k}_roots",), spawn)' for k in range(3)))
    r = subprocess.run([sys.executable, "-c", APART_TRACED, str(root), str(REPO)], capture_output=True, text=True, cwd=REPO,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert r.returncode == 0, r.stderr
    untraced = json.loads(r.stdout.strip().splitlines()[-1])
    assert len([u for u in untraced if u.startswith("os.system ")]) == 3, untraced


def test_one_xdist_worker_builds_the_catalog_and_the_others_take_it(tmp_path, monkeypatch):
    """Under xdist the first worker to ask builds the catalog apart and writes its manifest beside the workers' basetemps; a later
    one takes the manifest and builds nothing.  A builder that failed leaves its error there, and each later worker fails on it
    by name rather than build again."""
    import conftest
    from circuit_check import units as UN

    jobs: list[int] = []
    monkeypatch.setattr(UN, "built_apart",
                        lambda entries, n, budget, weights: jobs.append(n) or (entries[:1], [("gone",)], {}, 0))
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
