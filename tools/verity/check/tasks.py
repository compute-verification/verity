"""`tasks`: the check as a graph of tasks (`tasks/v1`), the whole contract between Verity and an executor.

    uv run tools/verity/check/tasks.py list --json [--kind after-landing|preflight] [--changed BASE] [--circuit-shards M]
                                               [--gather | --costs FILE] [--split-s S]
    uv run tools/verity/check/tasks.py run ID [--out FILE] [--work DIR] [--graph art:ID|FILE | [--circuit-shards M] [--timings FILE]]
    uv run tools/verity/check/tasks.py costs --graph art:ID|FILE [--graph ...] [--seconds FILE] [--out FILE]

`verity tasks ...` runs these.  The schema and what each field means is the "Tasks" section of this directory's README.md.

`list` prints the task graph of this checkout, uncommitted files included: one task per part of the shard planner's
(`check.auto_parts`), so each test suite (`pytest:SUITE`, `lean-suites:SUITE`; one that declares `[tool.verity.tests] parts =
P` in its P parts by test id, `lean-suites:SUITE:K/P`; one measured past `--split-s` seconds, default `shards.SPLIT_S`, in M
parts by test file, `pytest:SUITE:K/M`), each Lake package's audit (`lean-audit:PKG`), circuit-check's
`circuit-check:K/M`, the builds (`lean-build`, `flock-circuit-build`) and the other jobs whole.  `--kind
after-landing` (the default) is every job, `--kind preflight` those on the merge gate.  Each task's argv is the one `check.py`
builds for that part alone (`check.groups`), with this tree's path, the check's cache and the run's output directory written
`<tree>`, `<cache>` and `<out>`, and the interpreter `<python>`.  `--changed BASE` keeps the tasks whose inputs the change from
BASE's merge-base with HEAD to this tree touches (a suite by `suites.touched`, any other by its input paths), and the tasks
those depend on.  `--costs FILE` ({task id: seconds}, `costs`' output) prices tasks over the measured costs: a suite's whole
seconds there decide its M, and circuit-check's its shard count (`circuit_m`: ceil(seconds / S), never under ci.toml's), unless
`--circuit-shards` gives one.

`costs` writes what the executor took for the tasks of the graphs it ran (`--graph`, as `run` reads one, of any commit), as
`list --costs` reads it: each task's seconds from its newest passing `verdict` label on `task:<key>`, whose `ref` is the run
that earned it (its result's own steps' seconds, else the run's wall seconds), or `--seconds FILE`'s ({task id: seconds}, as
the executor's queue records them) over them; a split suite's parts and circuit-check's slices summed back to the whole job
over the share they cover, so the next listing splits it anew from the executor's seconds, not this machine's.  Several
graphs: each id's most.  What it priced and what it didn't goes to stderr.

`run ID` runs that task here as `check.py` runs that part in a shard of its own (`check.run` with ``only = [ID]``, without the
preflight), its deps beside it, and writes {verdict, outputs, seconds, key} as JSON to stdout or --out.  `--graph` names the
graph ID was listed in (`list --json`'s document: an `art:` id of the executor's `tasks-graph/v1` artifact, fetched from the
evidence store, or a file), which must be of this tree's HEAD and list ID: the run takes the graph's splits (each suite's M,
partition and timings) and circuit-check shards rather than this machine's, so any id it lists, whole or split, runs as listed
under the listed key.  Without it, a part of a suite split by file is of the partition `--timings` gives (the listing's
"splits" timings, `{suite: timings}`), and the rest of this machine's record.  The steps' progress goes to stderr.  Exit 0
when the verdict is passed or skipped, 1 when failed, 2 on a usage error or a refused graph.
"""
from __future__ import annotations

import argparse
import contextlib
import fnmatch
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = "tasks/v1"
GRAPHS = ("after-landing", "preflight")
#: the evidence store's kind for a graph the executor listed on a worker (`research.executor.TASKS_GRAPH_KIND`)
GRAPH_KIND = "tasks-graph/v1"
#: where an executor's run of `run ID --out "$OUT/task-result.json"` leaves its result among the run's files
RESULT = "out/task-result.json"
#: whether a task's audit regenerates lock records: never, so a lock record difference fails it, as it fails drive's
#: after-landing run (`check.py --no-regenerate`), whose verdicts a graph's stand beside and replace
REGENERATE = False


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"tasks_{name}", HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


C = _load("check")
S = C.SUITES
PLAN = C._load("plan")
LA = C._lean_audit()
ROOT = C.ROOT
VERIFIER = C.FLOCK_LEAN.relative_to(ROOT).as_posix()

#: what each build makes, under the artifact name its readers' `needs` list
OUTPUTS = {"lean-build": [f"lean-build:{VERIFIER}/.lake/build"],
           "flock-circuit-build": ["flock-circuit-build:$FLOCK_CHECKOUT"]}
#: the inputs of the jobs that run whole, as paths and globs from the root; "." (the whole tree) where none are declared.  A job
#: in `check.STEP_KEYS` is keyed by their git ids, which cover what its step key reads of the tree, under the environment a
#: suite's key reads; what its step key reads of the machine (the kernel release, `rustc -vV`, the slot's paths) check's own
#: cache keys where the task runs, as `PLACED` is set there
STEP_INPUTS = {
    "lean-build": [VERIFIER],
    "flock-circuit-build": ["verity/protocols/flock/prover/check_build.sh", "verity/protocols/flock/prover/live", "verity/protocols/flock/prover/flock-*-b684b12.patch"],
    "flock-rust-tests": ["verity/protocols/flock/prover/check_build.sh", "verity/protocols/flock/prover/live", "verity/protocols/flock/prover/flock-*-b684b12.patch"],
    "lean-agreement": ["verity/protocols/flock/agreement", VERIFIER],
}
#: a worker slot's size, by what a task asks: most tasks a 16-core, 64 GB slot; a Lake package's audit the same with a
#: Lean build's disk, or a 32-core, 128 GB one when it reads `security` (Mathlib, ArkLib: `lean_slot`); a suite that runs in
#: one process (not `parallel`) one core and what `suites.py` holds for one test process
SLOT = {"cores": 16, "mem_gb": 64, "disk_gb": 20}
LEAN_PKG_SLOT = {"cores": 16, "mem_gb": 64, "disk_gb": 60}
LEAN_SLOT = {"cores": 32, "mem_gb": 128, "disk_gb": 60}
ONE_PROCESS = {"cores": 1, "mem_gb": S.TEST_PROCESS_GB, "disk_gb": 10}
#: where a task's argv names the run's output directory before `<out>` stands for it
OUT = Path(tempfile.gettempdir()) / "verity-task-out"
#: what places a run (a recorded run's id, a check slot's memory) and so changes the argv `check.groups` builds: unset for a
#: task's argv and key, which are the tree's
PLACED = ("RESEARCH_RUN_ID", C.SLOT.SLOT_MEM_ENV)


def kind_of(part: str) -> str:
    job = part.partition(":")[0]
    if job in C.SHARDS.SUITE_JOBS:
        return "suite"
    if job == "lean-audit":
        return "lean-package"
    if job == "circuit-check":
        return "circuit-check"
    return "build" if job in OUTPUTS else "step"


def parts(graph: str = "after-landing", circuit_shards: int | None = None, splits: dict[str, dict] | None = None) -> list[str]:
    if graph not in GRAPHS:
        raise ValueError(f"--kind {graph}: one of {', '.join(GRAPHS)}")
    return C.auto_parts(gated=graph == "preflight", circuit_shards=circuit_shards,
                        splits={k: v["m"] for k, v in (splits or {}).items()})


def digest(doc) -> str:
    return hashlib.sha512(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@contextlib.contextmanager
def _unplaced():
    saved = os.environ.copy()
    for k in PLACED:
        os.environ.pop(k, None)
    try:
        yield
    finally:
        os.environ.update({k: saved[k] for k in PLACED if k in saved})


@contextlib.contextmanager
def _without_preflight():
    """`check.run` with no preflight steps, which it reads from its module: the preflight is no task."""
    was = C.preflight
    C.preflight = lambda: []
    try:
        yield
    finally:
        C.preflight = was


def _normal(argv: list[str]) -> list[str]:
    subs = [(sys.executable, "<python>"), (str(OUT), "<out>"), (str(C.cache_dir()), "<cache>"), (str(ROOT), "<tree>")]

    def one(a: str) -> str:
        for s, t in subs:
            a = a.replace(s, t)
        return a
    return [one(a) for a in argv]


def own_steps(part: str) -> set[str]:
    """The steps `check.run` reports for ``part`` that are its own, not a dep's: its job's, or for a Lean job on a tree without
    the verifier, the one skip `check.lean_steps` names `lean`."""
    job = part.partition(":")[0]
    return {job, *(["lean"] if job.startswith("lean") else [])} - (C.DECLARATION.needed(C.JOBS, [job]) - {job})


def pins(splits: dict[str, dict]) -> dict[str, dict]:
    """``splits`` without their timing records: what a task's argv names of each partition, its M and digest (`check.suite_args`
    pins a part to it, so a worker whose record partitions otherwise refuses it rather than run other files)."""
    return {k: {"m": v["m"], "digest": v["digest"]} for k, v in splits.items()}


def command(part: str, cores: int, splits: dict[str, dict] | None = None) -> tuple[list[str] | None, str, str | None]:
    """(``part``'s argv as `check.groups` builds it for a shard holding ``part`` alone on ``cores`` cores, its cwd, None), each
    written as the module docstring says, or (None, its cwd, why check can't run it on this tree: a reason starting "skipped:"
    is a skip)."""
    with _unplaced():
        try:
            gs = C.groups(OUT, cores, only=[part], regenerate=REGENERATE, splits=pins(splits or {}))
        except C.DECLARATION.DeclarationError as e:
            return None, "<tree>", str(e)
    mine = [(argv, cwd) for g in gs for name, argv, cwd in g if name in own_steps(part)]
    if not mine:
        return None, "<tree>", f"check.py builds no step for {part} on this tree"
    (argv, cwd), = mine
    where = _normal([str(cwd)])[0]
    return (_normal(argv), where, None) if isinstance(argv, list) else (None, where, argv)


class Facts:
    """What the records read of the tree, gathered once: the suites and their key's components, the tracked files."""

    def __init__(self, splits: dict[str, dict] | None = None) -> None:
        #: the suites split by test file (`check.suite_splits`), with the timing records their partitions are of
        self.splits = splits or {}
        e = C.env()
        #: the steps' environment as a suite's key reads it (`suites.environment`), which `pytest` and the jobs run whole run in
        with PLAN.environ(e):
            self.env = {"pytest": S.environment()}
        with PLAN.environ(PLAN.lean_env(e)):
            self.env["lean-suites"] = S.environment()
        self.pkgs, self.suites = S.members(), {s.name: s for s in S.discover()}
        self.tree, self.lock = S.worktree(), S.locked()
        self.inputs = {n: S.inputs_of(s, self.pkgs) for n, s in self.suites.items()}
        self.fixtures = S.fetched_fixtures(self.tree, {p for ps in self.inputs.values() for p in ps})
        self.files = C.tracked_files(ROOT) or []

    def suite_components(self, name: str, step: str) -> dict:
        """The components `suites.key` hashes for suite ``name`` under ``step``'s environment, as check's run of it keys it."""
        s, paths = self.suites[name], self.inputs[name]
        cut = S.dropped(s, paths)
        ids = S.input_ids(self.tree, sorted(set(paths) | set(S.RUNNER)), S.markdown(s), cut)
        ids.update({f"fixture:{p}": art for p, art in self.fixtures.items() if S.covered(p, paths, cut)})
        c = S.components(s, ids, self.env[step], [], S.resolved(S.roots(s, self.pkgs), self.lock, s.path))
        c.pop("trees", None)
        c.pop("nested", None)
        return c

    def part_key(self, name: str, spec: str, step: str) -> str:
        """`suites.part_key` of suite ``name``'s parts ``spec`` (K/M) under ``step``'s environment, as check's run of it keys it: by
        file of the partition ``splits`` names, or by test id where it declares `parts`."""
        s, paths = self.suites[name], self.inputs[name]
        cut = S.dropped(s, paths)
        ids = S.input_ids(self.tree, sorted(set(paths) | set(S.RUNNER)), S.markdown(s), cut)
        ids.update({f"fixture:{p}": art for p, art in self.fixtures.items() if S.covered(p, paths, cut)})
        k, m = (int(x) for x in spec.split("/"))
        part = S.part_of(s, (k,), m, self.tree, cut, {} if s.parts else self.splits[name]["timings"])
        mods, own, keep = S.part_context(s, part, self.tree, cut)
        return S.part_key(s, part, keep, mods, ids, own, self.env[step], [], S.resolved(S.roots(s, self.pkgs), self.lock, s.path))

    def path_ids(self, paths: list[str]) -> dict[str, str]:
        """The git object ids in this tree of what ``paths`` (paths and globs from the root, "." the tree) cover: a path's own (a
        directory's tree id), a glob's each file it matches."""
        globs = [p for p in paths if any(c in p for c in "*?[")]
        ids = S.object_ids(self.tree, [p for p in paths if p not in globs])
        for g in globs:
            ids.update({f: i for f, i in S.own_ids(self.tree, g.rpartition("/")[0] or ".").items() if fnmatch.fnmatchcase(f, g)})
        return ids

    def circuit_closure(self) -> list[str]:
        s = next((s for s in self.suites.values() if s.path == "tools/verity/circuit_check"), None)
        return sorted({s.path, *S.closure(s, self.pkgs), "uv.lock"}) if s else ["."]


def lean_cover(pkg: str) -> list[str]:
    """The directories and files that hold everything an audit of ``pkg`` reads (`lean_audit.inputs`)."""
    pkgs = [pkg, *LA.path_deps(pkg), *(d for q in LA.provers(pkg) for d in (q, *LA.path_deps(q)))]
    return [*dict.fromkeys(pkgs), LA.LEAN_TOOL.rstrip("/"), "tools/verity/check/common.py"]


def lean_slot(pkg: str) -> dict:
    """The slot ``pkg``'s audit asks: `LEAN_SLOT` when it or a package it requires is under `security`, else `LEAN_PKG_SLOT`."""
    return LEAN_SLOT if any(p.split("/")[0] == "security" for p in (pkg, *LA.path_deps(pkg))) else LEAN_PKG_SLOT


def record(part: str, facts: Facts, all_parts: list[str], costs: dict[str, float]) -> dict:
    """``part``'s `tasks/v1` record."""
    job, _, x = part.partition(":")
    kind = kind_of(part)
    j = next(j for j in C.JOBS if j.name == job)
    deps = sorted({*j.needs, *([j.after] if j.after else [])})
    needs = [a for d in deps for a in OUTPUTS.get(d, [])]
    beside: list[str] = []
    key_doc: dict | None = None
    unkeyed = None
    if kind == "suite":
        _job, name, spec = C.SHARDS.suite_part(part)
        s = facts.suites[name]
        resources = SLOT if s.parallel else ONE_PROCESS
        inputs = {"closure": "suites.inputs_of", "paths": sorted(set(facts.inputs[name]) | set(S.RUNNER))}
        key_doc = {"part": facts.part_key(name, spec, job)} if spec else {"suite": facts.suite_components(name, job)}
    elif kind == "lean-package":
        resources = lean_slot(x)
        inputs = {"closure": "lean_audit.inputs", "paths": lean_cover(x)}
        beside = [f"lean-audit:{q}" for q in LA.provers(x) if f"lean-audit:{q}" in all_parts]
        key_doc = {"lean": LA.components(x, facts.files, LA.FLAGS), "python": platform.python_version(), "image": S.base_image()}
    elif kind == "circuit-check":
        resources = SLOT
        inputs = {"closure": "the verity-circuit-check distribution's workspace closure", "paths": facts.circuit_closure()}
        unkeyed = ("circuit-check keys each target, not a slice of them: the targets' keys are in its task list, which only a "
                   "run that builds every Definition makes (its own cache reuses each target's pass)")
    else:
        resources = SLOT
        inputs = {"closure": None, "paths": STEP_INPUTS.get(job, ["."])}
        if job in C.STEP_KEYS:
            key_doc = {"step": {"inputs": facts.path_ids(inputs["paths"]), "environment": facts.env["pytest"]}}
        else:
            unkeyed = f"{job} keeps no verdict of its own"
            if kind == "build":
                unkeyed = f"a build's outputs are keyed by nothing yet: {job} has no declared inputs or read guard"
    argv, cwd, why = command(part, resources["cores"], facts.splits)
    if argv is None:
        key_doc, unkeyed = None, f"check can't run it on this tree: {why}"
    key = digest({"schema": SCHEMA, "id": part, "argv": argv, "cwd": cwd, **key_doc}) if key_doc is not None else None
    return {"id": part, "kind": kind, "argv": argv, "cwd": cwd, "reason": why, "inputs": inputs, "needs": needs,
            "outputs": OUTPUTS.get(job, []), "deps": deps, "beside": beside, "resources": dict(resources),
            "cost_s": costs.get(part), "key": key, "unkeyed": None if key else unkeyed}


def covers(f: str, paths: list[str]) -> bool:
    return any(p == "." or f == p or f.startswith(p.rstrip("/") + "/") or fnmatch.fnmatchcase(f, p) for p in paths)


def changed(base: str, facts: Facts) -> tuple[str, list[str], set[str]]:
    """(the commit the change is from, the files it changes, the suites it touches): from ``base``'s merge-base with HEAD to
    this tree, as `suites.py --changed` reads it."""
    b = S.merge_base(S.changed_base(base))
    out = subprocess.run(["git", "diff", "--no-renames", "--name-only", "-z", b, facts.tree], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout
    touched = {s.name for s in S.touched(list(facts.suites.values()), facts.pkgs, b, facts.tree, facts.lock)}
    return b, [f for f in out.split("\0") if f], touched


def narrowed(tasks: list[dict], files: list[str], touched: set[str]) -> list[dict]:
    """The tasks whose inputs ``files`` touch (a suite: one of ``touched``), and every task those depend on."""
    def hit(t: dict) -> bool:
        if t["kind"] == "suite":
            return C.SHARDS.suite_part(t["id"])[1] in touched
        return any(covers(f, t["inputs"]["paths"]) for f in files)
    keep = {t["id"] for t in tasks if hit(t)}
    by = {t["id"]: t for t in tasks}
    todo = list(keep)
    while todo:
        for d in by[todo.pop()]["deps"]:
            if d in by and d not in keep:
                keep.add(d)
                todo.append(d)
    return [t for t in tasks if t["id"] in keep]


def measured(gather: bool, given: dict[str, float] | None, split_s: float) -> tuple[tuple, dict[str, dict], list[str]]:
    """(`check.measure`'s costs: this machine's caches, and with ``gather`` the newest recorded checks', the suites split by test
    file past ``split_s`` (`check.suite_splits`), notes)."""
    with tempfile.TemporaryDirectory(prefix="verity-tasks-") as d:
        costs, notes = C.measure(Path(d), gather, split_s=split_s, given=given)
    splits = C.suite_splits(C.given_suites(costs[0], given), costs[3], split_s)
    return costs, splits, (notes or ["costs from this machine's caches alone"]) + (["and --costs"] if given else [])


def priced(names: list[str], costs: tuple, splits: dict[str, dict], given: dict[str, float] | None) -> dict[str, float]:
    """Each task's measured seconds (``costs``, `measured`'s), a split suite's part its share of its suite's, ``given``'s over
    them; a task nothing measured has none."""
    out = C.SHARDS.priced(names, C.given_suites(costs[0], given), costs[1], {**costs[2], **(given or {})}, C.split_shares(splits))
    return {**out, **{p: float(s) for p, s in (given or {}).items() if p in names}}


def graph(kind: str = "after-landing", circuit_shards: int | None = None, base: str | None = None, gather: bool = False,
          given: dict[str, float] | None = None, split_s: float = C.SHARDS.SPLIT_S) -> dict:
    costs, splits, notes = measured(gather, given, split_s)
    circuit_shards = circuit_shards or circuit_m(given, split_s)
    names = parts(kind, circuit_shards, splits)
    facts = Facts(splits)
    prices = priced(names, costs, splits, given)
    tasks = [record(p, facts, names, prices) for p in names]
    doc = {"schema": SCHEMA, "commit": head(), "tree": facts.tree, "kind": kind,
           "circuit_shards": circuit_shards or next(j for j in C.JOBS if j.name == "circuit-check").shards,
           "split_s": split_s,
           "splits": {k: {f: v[f] for f in ("m", "seconds", "digest", "shares", "files", "timings")} for k, v in splits.items()},
           "changed": None, "costs": notes}
    if base is not None:
        b, files, touched = changed(base, facts)
        tasks = narrowed(tasks, files, touched)
        doc["changed"] = {"base": b, "files": len(files)}
    return {**doc, "tasks": tasks}


def split_for(name: str, m: int, splits: dict[str, dict], timings: dict[str, dict[str, float]] | None = None) -> dict[str, dict]:
    """``splits`` with suite ``name`` in ``m`` parts partitioned by ``timings[name]`` (a lister's record of its files' seconds)
    or else the record `suites.py` reads here (`suites.file_times`), which the run's own `suites.py` is handed."""
    s = next(s for s in S.discover() if s.name == name)
    times = (timings or {}).get(name)
    times = S.file_times(s) if times is None else times
    files = S.test_files(s, S.worktree(), S.dropped(s, S.inputs_of(s, S.members())))
    mine = {f: v for f, v in times.items() if f in set(files)}
    return {**splits, name: {"m": m, "digest": S.partition_digest(S.partition(files, mine, m)), "timings": mine}}


def head() -> str | None:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or None


def document(ref: str) -> dict:
    """The `tasks/v1` graph ``ref`` names, `list --json`'s document: an `art:` id of a `GRAPH_KIND` artifact, fetched from the
    evidence store as `--gather` reads it, or a file.  Refused (ValueError) unless it is one."""
    if ref.startswith("art:"):
        from research.store.local import LocalStore, StoreError

        try:
            store = LocalStore()
            if (kind := store.get_manifest(ref).kind) != GRAPH_KIND:
                raise ValueError(f"--graph {ref}: a {kind} artifact, not a {GRAPH_KIND}")
            path = Path(store.fetch(ref))
        except StoreError as e:
            raise ValueError(f"--graph {ref}: {e}") from None
    else:
        path = Path(ref)
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"--graph {ref}: not JSON ({e})") from None
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA or not isinstance(doc.get("tasks"), list):
        raise ValueError(f"--graph {ref}: not a {SCHEMA} graph (`tasks.py list --json` writes one)")
    return doc


def listed(ref: str) -> dict:
    """`document`'s graph ``ref`` names, refused (ValueError) unless it is a graph of this tree's HEAD."""
    doc = document(ref)
    if doc.get("commit") != (h := head()):
        raise ValueError(f"--graph {ref} is a graph of commit {doc.get('commit')}, not of this tree's HEAD {h}: run a task of "
                         "the graph listed for the commit checked out")
    return doc


def ran_seconds(store, rid: str, part: str) -> tuple[float | None, str]:
    """(the seconds the run ``rid`` of task ``part`` took, how they were read): its `run`'s result (`RESULT` among the run's files)
    the sum of ``part``'s own steps' seconds (`own_steps`: not a dep's build run beside it), else its `seconds`; else the run's
    wall seconds, which count its wait for a slot and its `uv run` too; (None, "none") when neither can be read."""
    from research.store.local import StoreError

    try:
        a = store.get_attempt(rid) or {}
    except (StoreError, ValueError, OSError) as e:
        print(f"tasks: costs: {part}: its run {rid} can't be read ({type(e).__name__}: {e})", file=sys.stderr)
        return None, "none"
    out = a.get("outputs") or {}
    for art in dict.fromkeys(x for x in (out.get("run_files"), out.get("run_record")) if x):
        try:
            if not any(f["path"] == RESULT for f in store.listing(art)):
                continue
            with tempfile.TemporaryDirectory(prefix="verity-task-result-") as d:
                res = json.loads((Path(store.fetch(art, Path(d), paths=[RESULT])) / RESULT).read_text())
        except (StoreError, OSError, ValueError, KeyError) as e:
            print(f"tasks: costs: {part}: {rid}'s result in {art} can't be read ({type(e).__name__}: {e})", file=sys.stderr)
            continue
        own = [s["seconds"] for s in res.get("steps") or [] if isinstance(s, dict) and s.get("step") in own_steps(part)
               and isinstance(s.get("seconds"), (int, float))]
        if own:
            return float(sum(own)), "result"
        if isinstance(res.get("seconds"), (int, float)):
            return float(res["seconds"]), "result"
    wall = (a.get("execution") or {}).get("wall_s")
    return (float(wall), "wall") if isinstance(wall, (int, float)) and wall > 0 else (None, "none")


def executor_seconds(doc: dict, store) -> tuple[dict[str, float], dict[str, str]]:
    """({task id: seconds}, {task id: how they were read}) of ``doc``'s tasks the executor ran and passed: each one's newest `pass`
    verdict label on `task:<key>` (`research.executor`: an unkeyed task's key is its check's), its `ref` the run that earned it
    (`ran_seconds`).  A task with none is absent."""
    from research.executor import PASS, keyed, task_target
    from research.queue import PREFETCH_JOBS
    from research.store import labels_sync, vocab

    targets = {t["id"]: task_target(t["key"]) for t in keyed(doc)["tasks"]}
    if getattr(store, "remote", None) is not None:
        labels_sync.pull_targets(store, list(targets.values()), jobs=PREFETCH_JOBS)
    secs, how = {}, {}
    for part, target in targets.items():
        runs = [lab.ref for lab in store.labels(target, key=vocab.TASK_VERDICT) if lab.value == PASS and lab.ref]
        if runs and (got := ran_seconds(store, runs[-1], part))[0] is not None:
            secs[part], how[part] = got
    return secs, how


def folded(secs: dict[str, float], doc: dict) -> dict[str, float]:
    """``secs`` ({task id: seconds} of graph ``doc``'s tasks) with each job its graph cut in parts back whole, as `list --costs`
    reads it: a split suite's parts `JOB:SUITE:K/M` as `JOB:SUITE`, circuit-check's `circuit-check:K/M` as `circuit-check`, each
    the sum of its parts' seconds over the share of the job they cover (the graph's partition's shares; a circuit-check slice
    1/M), so parts that weren't priced don't shrink the whole."""
    shares = C.split_shares(doc.get("splits") or {})
    out: dict[str, float] = {}
    parts: dict[str, list[tuple[str, float]]] = {}
    for p, s in secs.items():
        job, name, spec = C.SHARDS.suite_part(p)
        if spec and job in C.SHARDS.SUITE_JOBS:
            parts.setdefault(f"{job}:{name}", []).append((p, s))
        elif kind_of(p) == "circuit-check" and ":" in p:
            parts.setdefault(job, []).append((p, s))
        else:
            out[p] = float(s)
    for whole, ps in parts.items():
        cover = sum(shares.get(p, 1 / int(p.rpartition("/")[2])) for p, _ in ps)
        out[whole] = round(sum(s for _, s in ps) / cover, 1) if cover > 0 else max(s for _, s in ps)
    return dict(sorted(out.items()))


def costs(refs: list[str], given: dict[str, float] | None = None) -> tuple[dict[str, float], list[str]]:
    """({task id: seconds}, notes): what the executor took for the tasks of the graphs ``refs`` (`document`'s), each graph's
    `executor_seconds` with ``given`` ({task id: seconds}, as the executor's queue records them) over them, `folded` to whole
    jobs; over several graphs, each id's most.  `list --costs` reads it."""
    from research.store.local import LocalStore, StoreError

    out: dict[str, float] = {}
    notes = []
    store = None
    for ref in refs:
        doc = document(ref)
        try:
            store = store or LocalStore()
            secs, how = executor_seconds(doc, store)
        except StoreError as e:
            secs, how = {}, {}
            notes.append(f"{ref}: the evidence store is not reachable ({e}): the seconds given alone")
        ids = [t["id"] for t in doc["tasks"]]
        mine_given = {p: float(s) for p, s in (given or {}).items() if p in ids}
        mine = {**secs, **mine_given}
        src = [("given" if p in mine_given else how[p]) for p in mine]
        by = {k: src.count(k) for k in ("result", "wall", "given")}
        missing = [p for p in ids if p not in mine]
        notes.append(f"{ref} (of {str(doc.get('commit'))[:12]}): {len(mine)} of {len(ids)} tasks priced ({by['result']} by their "
                     f"result's steps, {by['wall']} by their run's wall seconds, {by['given']} given)"
                     + (f"; none for {', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''}" if missing else ""))
        for p, s in folded(mine, doc).items():
            out[p] = max(out.get(p, 0.0), s)
    return dict(sorted(out.items())), notes


def circuit_m(given: dict[str, float] | None, split_s: float) -> int | None:
    """circuit-check's shard count from ``given``'s `circuit-check` seconds (`costs`' whole): ceil(seconds / ``split_s``), never
    under its ci.toml `shards`; None when ``given`` has none.  Its targets go to shards by the hash of each one's id, and it
    keeps each target's pass under that target's own key, so any count reuses every pass."""
    s = (given or {}).get("circuit-check")
    if not isinstance(s, (int, float)) or s <= 0:
        return None
    return max(next(j for j in C.JOBS if j.name == "circuit-check").shards or 1, math.ceil(s / split_s))


def run(part: str, work: Path | None = None, circuit_shards: int | None = None, split_s: float = C.SHARDS.SPLIT_S,
        timings: dict[str, dict[str, float]] | None = None, graph: dict | None = None) -> dict:
    """``part`` run here as `check.run` runs it in a shard of its own (its deps beside it, no preflight), in ``work`` (default:
    a new temporary directory).  With ``graph`` (`listed`'s), ``part`` is one of the tasks it lists, and runs as listed: of
    its splits (each suite's M, partition digest and timings) and its circuit-check shards, so a split suite's part runs the
    listed files under the listed key, and a suite listed whole runs whole, whatever this machine's record would split.
    Without it, the tasks are this machine's: a part of a suite split by file (`JOB:SUITE:K/M`, any K/M of a listed suite) is
    of the partition ``timings`` gives (`list`'s "splits" timings), else of this machine's record."""
    if graph is not None:
        if part not in {t.get("id") for t in graph["tasks"]}:
            raise ValueError(f"no task {part} in the graph given (of commit {graph['commit']}): its tasks are the ids it lists")
        splits = {k: {f: v[f] for f in ("m", "digest", "timings")} for k, v in (graph.get("splits") or {}).items()}
        names = parts(graph.get("kind", "after-landing"), graph.get("circuit_shards"), splits)
    else:
        _costs, splits, _notes = measured(False, None, split_s)
        names = parts("after-landing", circuit_shards, splits)
        job, name, spec = C.SHARDS.suite_part(part)
        if spec and int(spec.partition("/")[0]) <= (m := int(spec.partition("/")[2])) and any(
                C.SHARDS.suite_part(p)[:2] == (job, name) for p in names):
            splits = split_for(name, m, splits, timings)
            names = [p for p in names if C.SHARDS.suite_part(p)[:2] != (job, name)] + [part]
    if part not in names:
        raise ValueError(f"no task {part} on this tree: `tasks.py list --json` lists them")
    rec = record(part, Facts(splits), names, {})
    if graph is not None and (want := next(t.get("key") for t in graph["tasks"] if t.get("id") == part)) != rec["key"]:
        print(f"tasks: {part}'s key here differs from the graph's ({str(want)[:16]}, here {str(rec['key'])[:16]}): this "
              "worker's tree or environment is not the lister's", file=sys.stderr)
    work = work or Path(tempfile.mkdtemp(prefix="verity-task-"))
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    with contextlib.redirect_stdout(sys.stderr), _without_preflight():
        ok, steps = C.run(work, min(rec["resources"]["cores"], C.cores()), only=[part], regenerate=REGENERATE,
                          splits=splits)
    seconds = round(time.monotonic() - t0, 1)
    own = [s for s in steps if s["step"] in own_steps(part)]
    if not ok or not own:
        verdict = "failed"
    else:
        verdict = "skipped" if all(s["status"] == "skipped" for s in own) else "passed"
    files = sorted(str(p.relative_to(work)) for p in work.rglob("*") if p.is_file())
    return {"schema": SCHEMA, "id": part, "verdict": verdict, "key": rec["key"], "seconds": seconds,
            "outputs": {"dir": str(work), "files": files, "artifacts": rec["outputs"] if verdict == "passed" else []},
            "steps": steps}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="verity tasks", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list", help="print the task graph of this checkout")
    ls.add_argument("--json", action="store_true", help=f"as {SCHEMA} JSON (the default prints one line per task)")
    ls.add_argument("--kind", default="after-landing", choices=GRAPHS,
                    help="every job (after-landing, the default) or those on the merge gate (preflight)")
    ls.add_argument("--changed", metavar="BASE", help="only the tasks whose inputs the change from BASE touches, and their deps")
    ls.add_argument("--gather", action="store_true", help="price tasks from the newest recorded checks too (reads the store)")
    ls.add_argument("--costs", type=Path,
                    help="a JSON {task id: seconds} over the measured costs (`costs` writes the executor's for a graph it ran)")
    ls.add_argument("--split-s", type=float, default=C.SHARDS.SPLIT_S, metavar="S",
                    help="list a suite measured past S seconds in ceil(seconds / S) parts by test file (default %(default)s)")
    rn = sub.add_parser("run", help="run one task here and print its result")
    rn.add_argument("id", metavar="ID")
    rn.add_argument("--out", type=Path, help="write the result here instead of stdout")
    rn.add_argument("--work", type=Path, help="the steps' output directory (default: a new temporary one)")
    rn.add_argument("--timings", type=Path, metavar="FILE",
                    help="JSON {suite: {test file: seconds}} (list's splits' timings) a split suite's part is partitioned by")
    rn.add_argument("--graph", metavar="art:ID|FILE",
                    help=f"the {SCHEMA} graph ID was listed in (a {GRAPH_KIND} artifact or a file), of this HEAD: run ID as it "
                         "lists it (its splits, circuit-check shards and so its key)")
    for p in (ls, rn):
        p.add_argument("--circuit-shards", type=int, metavar="M",
                       help="circuit-check in M tasks (default: ci.toml's shards, or more as --costs's circuit-check seconds ask)")
    co = sub.add_parser("costs", help="write what the executor took for a listed graph's tasks, as list --costs reads it")
    co.add_argument("--graph", metavar="art:ID|FILE", action="append", required=True,
                    help=f"a {SCHEMA} graph the executor ran (a {GRAPH_KIND} artifact or a file); again for more, each id's most")
    co.add_argument("--seconds", type=Path, metavar="FILE",
                    help="a JSON {task id: seconds} of the graph's tasks (the executor's queue's) over the store's")
    co.add_argument("--out", type=Path, help="write the costs here instead of stdout")
    a = ap.parse_args(argv)
    if getattr(a, "circuit_shards", None) is not None and a.circuit_shards < 1:
        ap.error("--circuit-shards: at least 1")
    if a.cmd == "run" and a.graph and (a.timings or a.circuit_shards is not None):
        ap.error("--graph: the graph gives the splits' timings and the circuit-check shards, so no --timings or --circuit-shards")
    try:
        if a.cmd == "costs":
            got, notes = costs(a.graph, json.loads(a.seconds.read_text()) if a.seconds else None)
            for n in notes:
                print(f"tasks: costs: {n}", file=sys.stderr)
            text = json.dumps(got, indent=1) + "\n"
            if a.out:
                a.out.write_text(text)
            else:
                sys.stdout.write(text)
            return 0
        if a.cmd == "list":
            given = json.loads(a.costs.read_text()) if a.costs else None
            doc = graph(a.kind, a.circuit_shards, a.changed, a.gather, given, a.split_s)
            print(json.dumps(doc, indent=1) if a.json else "\n".join(
                f"{t['id']:60} {t['kind']:13} {'unkeyed' if t['key'] is None else t['key'][:16]}"
                + (f"  {t['cost_s']:.0f} s" if t["cost_s"] is not None else "") for t in doc["tasks"]))
            return 0
        res = run(a.id, a.work, a.circuit_shards, timings=json.loads(a.timings.read_text()) if a.timings else None,
                  graph=listed(a.graph) if a.graph else None)
    except (ValueError, C.DECLARATION.DeclarationError, S.Unresolved, OSError) as e:
        print(f"tasks: {e}", file=sys.stderr)
        return 2
    text = json.dumps(res, indent=1) + "\n"
    if a.out:
        a.out.write_text(text)
    else:
        sys.stdout.write(text)
    return 1 if res["verdict"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
