"""The catalog bindings' records (`circuit_check.units`), on the synthetic tree of `test_cache`: `cli.run`'s ``--all`` path with
the real `targets.catalog`, `targets.suite` and `targets.select` over the tree's bindings (its ``syn/catalog.py``), first as a
full build, which stores each binding's record, then from the records, building only the bindings whose record misses.  And
`units.call_families` against `targets.call_families`' rules for word Call families made on demand and unbound kinds.  And the
circuit_check suite's catalog, each binding built in a process of its own and made from what they reach in another
(`units.built_apart`, `units.from_manifest`), against `targets.suite` built whole."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import types
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


def _records(root: Path, units: tuple[str, ...] | None = None, skip: frozenset[Path] = frozenset()) -> list[Path]:
    """The cache's binding records, of ``units`` if given; ``skip``: files a test made unreadable, read as no record."""
    def target(p: Path) -> list:
        return json.loads(p.read_text())["key"]["request"]["target"]
    return [p for p in (root / "cache").glob("*/*/*.json")
            if p not in skip and (t := target(p))[:1] == ["catalog-unit"] and (units is None or tuple(t[1]) == units)]


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


#: a registry module that registers eight primitives, `SynBASE0_v1` to `SynBASE7_v1`, when it is imported
REGISTRY_MODULE = """
from verity.circuits import Value, primitive

V = Value(16)
for i in range(8):
    primitive(f"SynBASE{i}", 1, [("a", V)], V)(lambda a: a)
"""


def test_the_registered_records_hold_the_same_roots_whichever_registry_module_was_imported_first(tmp_path, monkeypatch):
    """`units.registered_entries` cuts `targets._registered` into records by position, and a record counts only for the roots
    it lists: so the cut is the same whichever of two registry modules made the registry first.  circuit-check's own process
    loads the registries before anything else, and a session of this suite imports its test modules first."""
    import importlib

    from circuit_check import targets as TG
    from circuit_check import units as UN
    from verity.circuits.defs import REGISTRY

    names = ("syn_registry_a", "syn_registry_b")
    for name in names:
        (tmp_path / f"{name}.py").write_text(REGISTRY_MODULE.replace("BASE", name[-1].upper()))
    monkeypatch.syspath_prepend(str(tmp_path))
    cuts = []
    for order in (names, names[::-1]):
        monkeypatch.setattr(REGISTRY, "defs", {})
        for name in order:
            monkeypatch.delitem(sys.modules, name, raising=False)
            importlib.import_module(name)
        cuts.append([e.root_ids for e in UN.registered_entries(TG._registered())])
    assert cuts[0] == cuts[1], cuts
    assert len(cuts[0]) == UN.REGISTERED_RECORDS and sum(map(len, cuts[0])) == 16


#: SynStep at a static Definition its body doesn't call, bound by one binding and unbound by another (as `AttnDotStep_v1` takes
#: its DOT from `boolean_version` and from its sidecar): two Definitions of one id whose descriptors differ
STEP = """
from verity.circuits import composite

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


#: `HARNESS`'s setup, then ``circuit-check --all --records-only -j 2`` (`cli.records`) into the tree's cache; prints its report
RECORDS_ONLY = HARNESS[:HARNESS.index("rep = cli.run(")] + 'print(json.dumps(cli.records(root / "cache", log=None, jobs=2)))\n'


def _records_only(root: Path) -> dict:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([sys.executable, "-c", RECORDS_ONLY, str(root), str(REPO)], capture_output=True, text=True, cwd=REPO, env=env)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_the_calls_from_records_are_the_catalogs_for_made_on_demand_and_unbound_kinds(monkeypatch):
    """`units.call_families` gives `targets.call_families`' Calls: a word Call family made on demand (`Const32[0x..]_v1`) has
    its Boolean version among them though no record holds one, and a vocabulary
    kind that doesn't bind refuses them by name, as `targets.word_call_families` does (left out, its family's partition would be
    judged as no Call's)."""
    from circuit_check import targets as TG, units as UN

    nothing = UN.Derived([], {}, {})
    with monkeypatch.context() as m:
        m.setattr(TG, "word_call_families", lambda: {"Const32[0x00000001]_v1", "GatherI32x16_v1"})
        m.setattr(TG, "boolean_call_candidates", lambda words: set())
        assert UN.call_families([], nothing, {}) == {"Const32[0x00000001]_v1", "GatherI32x16_v1",
                                                    "Const32[0x00000001]_v2", "GatherI32x16_v2"}
    from circuit_check import sidecars as SC

    syn = types.ModuleType("syn.circuit-check")
    vars(syn).update(__file__="syn.circuit-check.py", _vocabulary_roots=lambda: [],
                     _vocabulary_bindings=lambda: [("b1-eager-v3/dsv4_indexer_compress", "KeyError: 'ROPE'")])
    monkeypatch.setattr(TG, "_SHORT", {})
    monkeypatch.setattr(TG, "_BOUND", {})
    monkeypatch.setattr(TG, "load_registries", lambda: None)
    monkeypatch.setattr(TG, "templates", lambda: [])
    monkeypatch.setattr(SC, "binding_modules", lambda: [("syn", syn)])
    with pytest.raises(RuntimeError, match=r"b1-eager-v3/dsv4_indexer_compress: KeyError: 'ROPE'"):
        UN.call_families([], nothing, {})


#: what `targets.on_demand_boolean_families` registers, in order, given word Call families made on demand
ON_DEMAND = """
from circuit_check import targets as TG
from verity.circuits.defs import REGISTRY
import verity.circuits.ops.boolean.versions
import verity_vllm.definitions

before = set(dict.keys(REGISTRY.defs))
TG.on_demand_boolean_families({"Const8[0x0]_v1", "Const16[0x1]_v1", "Const32[0x2]_v1", "Const4[0x3]_v1", "GatherBf16x4_v1",
                               "GatherBf16x8_v1", "GatherI32x2_v1", "GatherI32x4_v1", "GatherI32x8_v1", "F32Add_v1"})
print([fid for fid in dict.keys(REGISTRY.defs) if fid not in before])
"""


def test_the_boolean_versions_made_on_demand_register_in_one_order_under_any_hash_seed():
    """`targets.on_demand_boolean_families` makes the Boolean versions in the same order whatever the process's hash seed: each
    one made registers, and the catalog's roots record notes the order (`units._noting`), so a set's order would give an
    unchanged catalog's rebuild other bytes."""
    got = {subprocess.run([sys.executable, "-c", ON_DEMAND], capture_output=True, text=True, check=True, cwd=REPO,
                          env={**os.environ, "PYTHONHASHSEED": str(seed)}).stdout for seed in (1, 2, 3, 4)}
    assert len(got) == 1, got


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
from verity.circuits import composite

from syn.prims import V


@composite("SynBad", 1, [], lambda S: ((("a", V), ("b", V)), V))
def SynBad(B, S, a, b):
    raise ValueError("no body at these statics")
"""
#: a Definition whose body build kills the process building it, as the kernel's OOM killer does
DIE = """
import os
import signal

from verity.circuits import composite

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

from verity.circuits import composite

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


#: `HARNESS`'s setup, then ``circuit-check --all --records-only -j 2`` as the command line runs it (`cli._records_main`) into the
#: tree's cache: its report at ``argv[3]``, its exit status the command's, and with ``argv[4]`` its ``--records-manifest`` there
RECORDS_MAIN = HARNESS[:HARNESS.index("rep = cli.run(")] + """
import argparse
sys.exit(cli._records_main(argparse.Namespace(quiet=True, mem_gb=None, slices=None, out=Path(sys.argv[3]),
                                              records_manifest=Path(sys.argv[4]) if len(sys.argv) > 4 else None),
                           root / "cache", 2))
"""

#: the suite's session over the synthetic bindings, in a process no tracer started in, as a pytest's is: conftest.py's catalog
#: from the records in the tree's cache (`conftest._taken`), its build of the records `RECORDS_MAIN`; prints the catalog's
#: line, how many times that build was started, and each suite target's descriptor and units
SESSION = HARNESS[:HARNESS.index("rep = cli.run(")].replace("CC.trace(root)\n", "") + """
sys.path.insert(0, sys.argv[3])
import conftest

ran = []
conftest._records_argv = lambda cache, jobs, mem_gb, out, listed: ran.append(1) or [
    sys.executable, "-c", sys.argv[4], str(root), str(repo), str(out), *([listed] if listed else [])]
built = conftest._taken(root / "cache", None, 2)
print(json.dumps({"how": built.how, "builds": len(ran), "targets": {fid: CC.descriptor(fn) for fid, fn in built.suite.items()},
                  "units": {fid: sorted(TG.BINDINGS.get(fid, ())) for fid in built.suite}}))
"""


def _session(root: Path, listed: Path | None = None, env: dict[str, str] | None = None) -> dict:
    env = {**{k: v for k, v in os.environ.items() if not k.startswith("CIRCUIT_CHECK_RECORDS")}, **(env or {})}
    r = subprocess.run([sys.executable, "-c", SESSION, str(root), str(REPO), str(Path(__file__).parent), RECORDS_MAIN],
                       capture_output=True, text=True, cwd=REPO,
                       env={**env, "PYTHONDONTWRITEBYTECODE": "1",
                            **({"CIRCUIT_CHECK_RECORDS_MANIFEST": str(listed)} if listed is not None else {})})
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.slow
def test_a_record_whose_bytes_are_not_the_ones_its_name_digests_is_never_taken(tmp_path):
    """A record a store handed back under its name with other bytes (here the same record, a newline after it, which still
    parses and whose reads still hold) is passed over, by a session and by a run: its binding is built, and the record stored
    then, under that name, is the one the name digests."""
    root = _tree(tmp_path / "t")
    _bindings(root, *BINDINGS)
    _records_only(root)
    (rec,) = _records(root, ("_sized_roots",))
    whole = rec.read_bytes()
    rec.write_bytes(whole + b"\n")
    out = _session(root)
    assert out["builds"] == 1 and "4 of 5 bindings reused, 1 built" in out["how"] and out["how"].endswith(": _sized_roots")
    assert rec.read_bytes() == whole
    rec.write_bytes(whole + b"\n")
    assert _records_only(root)["rebuilt"] == ["_sized_roots"] and rec.read_bytes() == whole


def _records_main(root: Path, out: Path, listed: Path | None = None, env: dict[str, str] | None = None) -> tuple[dict, str]:
    """``circuit-check --all --records-only`` over ``root`` as the command line runs it (`RECORDS_MAIN`), told to list its
    records at ``listed``: its report and the line it printed."""
    r = subprocess.run([sys.executable, "-c", RECORDS_MAIN, str(root), str(REPO), str(out), *([str(listed)] if listed else [])],
                       capture_output=True, text=True, cwd=REPO, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **(env or {})})
    assert r.returncode == 0, r.stderr
    (line,) = [ln for ln in r.stdout.splitlines() if ln.startswith("circuit-check --records-only: ")]
    return json.loads(out.read_text()), line


@pytest.mark.slow
def test_a_record_a_session_built_under_the_suite_s_read_tracer_counts_in_a_run_without_it(tmp_path):
    """The suite's runner puts its read tracer's directory, which is in the tree, on its tests' PYTHONPATH, and every Python
    started with it loads that directory's `sitecustomize`.  A binding whose build looks for its loaded modules' sources
    (`inspect.getmodule`, as torch does) then holds that module by its name, which resolves in no process without the
    directory on its path: a ``--records-only`` run with it stores a record of that binding no run without it takes, as the
    rule has it.  The session's build runs without it, with the path's other entries: every record it stores counts in a
    ``--records-only`` run that has no tracer, and in the next session, which starts no build."""
    tracer = {"guard/child/sitecustomize.py": "\n", "guard/child/verity_suite_reads.py": "\n"}
    looks = ("def xor():\n    from syn import xdef\n", "def xor():\n    from syn import xdef\n    import inspect\n"
             "    inspect.getmodule(inspect.currentframe())\n")

    def tree(name: str) -> tuple[Path, dict[str, str]]:
        root = _tree(tmp_path / name, tracer)
        _bindings(root, *BINDINGS)
        _edit(root, "syn/catalog.py", *looks)
        return root, {"PYTHONPATH": os.pathsep.join([str(root / "guard" / "child"), str(tmp_path / "elsewhere")]),
                      "VERITY_SUITE_READS": str(tmp_path / f"{name}.reads"), "VERITY_SUITE_ROOT": str(root)}

    root, traced = tree("with")
    assert _records_main(root, tmp_path / "out.json", env=traced)[0]["stored"] == 5
    without, _ = _records_main(root, tmp_path / "out.json")
    assert without["rebuilt"] == without["skipped"] == ["_xor_roots"] and without["reused"] == 4, without

    root, traced = tree("session")
    out = _session(root, env=traced)
    assert out["builds"] == 1 and out["how"].startswith("from records: 0 of 5 bindings reused, 5 built and stored in "), out
    step, line = _records_main(root, tmp_path / "out.json")
    assert step["reused"] == 5 and step["rebuilt"] == [] and step["stored"] == 0, line
    again = _session(root, env=traced)
    assert again["builds"] == 0 and again["how"] == "from records: 5 of 5 bindings reused, 0 built", again


def test_no_key_holds_the_variables_that_say_where_records_are_or_how_a_suite_runs(monkeypatch):
    """A record a session's build stores is the one `check`'s circuit-check takes only when both key it alike: the variables
    only the suite's runner and conftest.py read are in no key, and any other of the keyed prefixes still is."""
    from circuit_check import units as UN

    for k in [k for k in os.environ if k.startswith(("VERITY_", "CIRCUIT_CHECK_"))]:
        monkeypatch.delenv(k)
    bare = UN.key(("_x_roots",), 0)
    assert bare["environment"] == {}
    for k in ("CIRCUIT_CHECK_RECORDS", "CIRCUIT_CHECK_RECORDS_MANIFEST", "VERITY_SUITE_READS", "VERITY_SUITE_MODULE",
              "VERITY_SUITE_ROOT", "VERITY_SUITE_TEST", "VERITY_TEST_CACHE", "VERITY_CIRCUIT_CHECK_SLOW"):
        monkeypatch.setenv(k, "x")
    assert UN.key(("_x_roots",), 0) == bare
    monkeypatch.setenv("VERITY_REGRESSION", "1")
    monkeypatch.setenv("CIRCUIT_CHECK_OTHER", "1")
    assert UN.key(("_x_roots",), 0)["environment"] == {"CIRCUIT_CHECK_OTHER": "1", "VERITY_REGRESSION": "1"}


def _json_values(rng, depth: int = 0):
    """JSON values of every kind a slice holds (ids, paths, digests, counts, effects), as `json.loads` gives them."""
    leaves = [None, True, False, 0, -1, 2 ** 63 + 5, -0.0, 1.5, 1e-7, 1e21, 0.1 + 0.2, "", "é", "\u2028", "\U0001f600",
              "\x00\x1f\x7f", '"\\/', 'Gemm_v2{K=64,N=8,DOT={"fn":"HopperBF16WgmmaDot16_v1"}}', "sha512:" + "ab" * 64,
              "verity_vllm/definitions/registry/b1.py"]
    k = rng.randrange(8 if depth < 3 else 6)
    if k < 6:
        return rng.choice(leaves)
    n = rng.randrange(4)
    if k == 6:
        return [_json_values(rng, depth + 1) for _ in range(n)]
    return {rng.choice(["id", "node", "reads", "é", "\U0001f600", "a b", ""]) + str(i): _json_values(rng, depth + 1)
            for i in range(n)}


def test_every_slice_file_is_its_document_as_compact_sorted_ascii_json(tmp_path):
    import random

    from circuit_check import units as UN

    rng = random.Random(0)

    def old(doc):
        return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()

    for t in range(40):
        shared = {f"s{i}": _json_values(rng) for i in range(rng.randrange(8))}
        per = {f"binding-{t}-{i}-\u00e9": {f"p{j}": _json_values(rng) for j in range(rng.randrange(6))}
               for i in range(rng.randrange(1, 5))}
        to = tmp_path / str(t)
        UN.write_slices(to, shared, per)
        body = (to / "shared.json").read_bytes()
        assert body == old(json.loads(json.dumps(shared)))
        digest = "sha512:" + hashlib.sha512(body).hexdigest()
        index = json.loads((to / "index.json").read_bytes())
        assert (to / "index.json").read_bytes() == old(index)
        for b, f in index["bindings"].items():
            assert (to / f).read_bytes() == old(json.loads(json.dumps({**per[b], "shared": digest})))
