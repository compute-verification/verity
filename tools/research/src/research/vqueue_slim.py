"""The run beside each candidate of the verified queue (`research.vqueue`) of what its graph2 check doesn't check: no graph/v2
task runs lean-agreement or the preflights.  It is drive's run beside the executor's check (`Driver._start_slim`):
`train.sh SHA MACHINES --deferred STEPS` of `SLIM_STEPS`, lean-agreement left out when a run of every step would skip it,
whose check runs the preflights before any step.  It is judged as `Driver._follow_slim` judges it, on drive's merge gate:
the preflights and lean-agreement count, policy.toml's `off_gate` (ratchet-gate) doesn't.  A candidate passes only once its
graph2 check's gating tasks and this run both passed (`combine`).

What the loop does.  It starts the run with each candidate's build, beside each candidate's check, so up to its `--depth`
of them at once take drive's machines, or once for a candidate on `main`'s chain whose start its log doesn't hold; until
then the candidate's run verdict is pending (`Verdict(PENDING)`, not `judge` of None, which is infra), and a candidate dropped
while its run runs cancels it (one still launching has no run ids to cancel yet).  It calls `start` once a candidate, `judge` each tick, and saves the record after
every `judge`: `judge` fills it in (the run ids once a detached launch printed them, the verdict once the run ended or was
given up, which it returns from then on without reading the run again), and a launch read once is gone (`Driver._resolve`
removes its directory).  On
`vqueue.Retry` with this run's verdict infra, it calls `start` again and keeps the new record in place of the old.  It gives
the Driver the `on` of drive.toml at its `main` (`MAIN`): `start` refuses a Driver without machines.  Under `--dry-run` it
calls neither `start` nor `judge`.  The candidate's verdict for `vqueue.decide` is `combine` of graph2's and this one's.  The
loop's Driver outlives its ticks, so `judge` reads the runs' states afresh each call, never through `Driver._states`, which
reads each run once a tick.  Nothing here writes drive's state.

A failed step's `Failed.key` is its content key at the candidate (`key`): SHA-512 over the step's name and the files it
reads (`reads`) as git holds them in the candidate's tree; its `base_key` is the same at the base, computed from the base's
tree, with no run.  An equal key means the base fails the step too, as graph2's keys of its tasks by their inputs do, so
`vqueue.decide` takes the failure as inherited.  A step whose reads can't be bounded is keyed on the whole tree, so its
failure is never inherited by its key: preflight-lints (its lints read every file, `check.lints_key`) and ratchet-gate (it
judges the merges above the base, not a tree).  Each failed step's one test (`Failed.tests`) is the step's (`step_test`), so
the run beside the base's own check, which the loop keeps beside `main` (started on `main` against itself, so lean-agreement
never runs), says what the base fails (`based`): failed there too, a step's failure is inherited by `vqueue.decide`'s rule
of failing tests."""
from __future__ import annotations

import ast
import functools
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from .drive import (ACTIVE, SIGTERM_RC, SLIM_JUDGED, SLIM_STEPS, Driver, implicated, killed, not_run, step_paths,
                    step_records)
from .queue import _git
from .store.fixtures import REPLICA_MAP, entries
from .vqueue import FAIL, INFRA, LIST_FAILED, PASS, PENDING, STATES, Failed, Verdict, meets

#: a candidate's state from its two verdicts' (`combine`): the first of these either has
ORDER = (FAIL, INFRA, PENDING, LIST_FAILED, PASS)
#: what declares lean-agreement's reads: the graph build family, whose lean-agreement task reads `VERIFIER`'s files and
#: `CHECK_READS` (check.py, the modules it loads as it starts, and what they read), and the lean-build of `LEAN`, read from its
#: source at the commit keyed (`_declared`), never run
BUILD = "tools/verity/graph/verity_graph/families/build.py"
#: the environment every step runs in: the lock `uv run --locked` installs, and the interpreter it pins
ENV = ("uv.lock", ".python-version")
#: what `uv lock --check` (preflight-lock) reads: the lock and every file the workspace's metadata can come from
LOCK_READS = ("uv.lock", "uv.toml", ".python-version", "pyproject.toml", "**/pyproject.toml", "setup.py", "**/setup.py",
              "setup.cfg", "**/setup.cfg")
#: the replica map preflight-fixtures reads (`research.store.fixtures.load_map`), as a repository path
MAP = REPLICA_MAP.as_posix()
#: the code preflight-fixtures runs beside check.py: `check.fetchable_fixtures` imports `research.store` and what it imports
FIXTURES_CODE = "tools/research/src/research"
#: the tag a content key starts with (`key`)
KEY_TAG = "verity-vqueue-slim-step/v1"


def start(driver: Driver, sha: str, base: str) -> dict | None:
    """Launch the run beside candidate ``sha``'s check, ``base`` the commit of `main` below it, as `Driver._start_slim` launches
    it beside the executor's check of ``sha`` with ``base`` verified: `SLIM_STEPS` but those a run of every step of ``sha``
    above ``base`` skips by name (lean-agreement, unless what it checks changed), with that run's keywords (ratchet-gate
    judges the merges above ``base``), on drive's machines (`Driver.on`).  The record to keep (`Driver._started`): ``ids``,
    none yet while a detached launch runs (its ``launch`` then); the candidate (``tip``), ``base``, when it launched
    (``at``), the ``steps`` it runs and those it skips (``skip``).  None, logged, when it registered no run, or when
    ``driver`` launches nothing (a dry run, or its queue's shadow), as `Driver._start_slim` does."""
    if not driver.on:
        raise ValueError("the run beside a candidate's check needs drive's machines: build the Driver with drive.toml's `on` "
                         "(`research.queue.drive_flags(q.workdir)[0].on`)")
    kw, skip, _, flags = driver._postland_args({"verified": base}, sha, False)
    skip = [s for s in skip if s in SLIM_STEPS]
    steps = [s for s in SLIM_STEPS if s not in skip]
    what = f"train.sh {sha[:12]} {driver.on} --deferred {','.join(steps)}{flags} beside candidate {sha[:12]}'s check"
    if driver.dry_run or driver.q.shadow:
        driver.q.log(f"research queue vqueue: would launch {what}")
        return None
    rec, tail = driver._started(driver.launch_postland(driver.repo, sha, driver.on, ",".join(steps), **kw))
    if rec is None:
        driver.q.log(f"research queue vqueue: {what} registered no run: {tail}")
        return None
    driver.q.log(f"research queue vqueue: {what}: {' '.join(rec['ids']) or 'launching'}")
    return {**rec, "tip": sha, "base": base, "at": driver.now(), "steps": steps, **({"skip": skip} if skip else {})}


def judge(driver: Driver, rec: dict | None) -> Verdict:
    """The verdict of the run `start` recorded in ``rec``, read as `Driver._follow_slim` reads it, on the steps it judges on
    the merge gate: `SLIM_JUDGED` but a step it skipped and policy.toml's `off_gate` at ``tip``.  Pending while it launches
    or runs.  Infra when it registered no run (``rec`` None, or a detached launch that printed no run ids), never ran or was
    stopped, or was lost or killed before it passed each of those steps; infra's one `Failed` says which.  Once it ended,
    pass when it passed each of them, fail with each of them it failed (a `Failed`: ``step NAME``, the step's content key at
    the candidate, the paths drive names that failure's culprit by, `step_paths`, its content key at ``base``, `key`, and
    the step's one test, `step_test`), and infra when it failed none but passed fewer (one skipped, or absent from its
    steps.jsonl).  Each verdict but pending is kept in ``rec`` (``verdict``, with the steps it passed) and read from there."""
    if rec is None:
        return _down(None, "registered no run")
    if "verdict" in rec:
        return _kept(rec["verdict"])
    tip = rec["tip"]
    if (tail := driver._resolve(rec, f"the run of {', '.join(rec['steps'])} beside candidate {tip[:12]}'s check")) is not None:
        return _keep(rec, _down(rec, f"registered no run: {tail}"))
    if "launch" in rec:
        return Verdict(PENDING)
    if not (ids := rec["ids"]):
        return _keep(rec, _down(rec, "registered no run"))
    states = driver.states(driver.repo, ids)
    if lost := driver._lost(rec, states):
        states = {r: ("lost", -1) for r in states}
    if any(st in ACTIVE for st, *_ in states.values()):
        return Verdict(PENDING)
    why = "lost" if lost else ", ".join(f"{st} rc={rc}" for st, rc, *_ in states.values())
    if not lost and driver._never_ran(rec, states):
        return _keep(rec, _down(rec, f"never ran or was stopped ({why})"))
    off = driver._policy(tip).off_gate
    judged = [s for s in SLIM_JUDGED if (s in rec["steps"] or s not in SLIM_STEPS) and s not in off]
    if not (missed := driver._unrun(ids, judged, tip)):
        return _keep(rec, Verdict(PASS), judged)
    dirs = [d for ds in driver._shards(rec) for d in ds]
    ok = driver._every_shard(rec, states, lambda rc: rc == 0) or (
        driver._every_shard(rec, states, lambda rc: rc in (0, 1)) and driver._only_allowed(rec))
    if not ok and (lost or any(isinstance(rc, int) and (rc < 0 or (rc > 128 and rc != SIGTERM_RC))
                                   for _, rc, *_ in states.values())
                       or any(implicated(d) is None and killed(d, why) for d in dirs)):
        return _keep(rec, _down(rec, "was lost" if lost else f"was killed ({why})"))
    paths: dict[str, set[str]] = {}
    for d in dirs:
        for st in step_records(d):
            if (s := st.get("step")) in missed and st.get("status") == "failed" and not not_run(st):
                paths.setdefault(s, set()).update(step_paths(d, st) or ())
    passed = [s for s in judged if s not in missed]
    if paths:
        base = rec.get("base")
        return _keep(rec, Verdict(FAIL, tuple(Failed(f"step {s}", key(driver.repo, tip, s) or f"{tip}:{s}", _inputs(paths[s]),
                                                     key(driver.repo, base, s) if base else None, frozenset({step_test(s)}))
                                              for s in missed if s in paths), ended=True), passed)
    return _keep(rec, _down(rec, f"ended ({why}) without passing {', '.join(missed)}: each skipped, or absent from its "
                                 "steps.jsonl"), passed)


def combine(graph: Verdict, slim: Verdict) -> Verdict:
    """A candidate's verdict from its graph2 check's (``graph``: its gating tasks) and the run beside it's (``slim``,
    `judge`): fail when either failed; else infra when either ended in infra errors; else pending while either is; else
    list-failed when graph2's check failed to list; else pass, both passed.  It holds both verdicts' failures and infra
    errors."""
    if bad := sorted({graph.state, slim.state} - set(STATES)):
        raise ValueError(f"no verdict has state {', '.join(map(repr, bad))}: a state is one of {', '.join(STATES)}")
    state = next(s for s in ORDER if s in (graph.state, slim.state))
    return Verdict(state, graph.failed + slim.failed, graph.infra + slim.infra)


#: the steps whose reads can't be bounded, each keyed on the whole tree, and why
WHOLE = {"preflight-lints": "its lints read every file (`check.lints_key`)",
         "ratchet-gate": "it judges the merges above the base, not a tree"}


def reads(repo: Path, commit: str, step: str) -> frozenset[str] | str:
    """What ``step`` reads of ``commit``'s tree, as `vqueue.meets` reads paths, directories and globs; for a step read whole,
    why: one whose reads can't be bounded (`WHOLE`, or a step this doesn't know), or whose declaration ``commit`` doesn't hold
    as plain data.  Each runs under check.py, so each bounded one holds check's reads (`CHECK_READS` at ``commit``) and `ENV`:

    - lean-agreement: the graph build family's lean-agreement task's inputs, `VERIFIER`'s files beside check's, and the
      Lean package `LEAN` whose lean-build it reads, whole (its lock and nested packages too, which `package` cuts);
    - preflight-lock: `LOCK_READS`;
    - preflight-fixtures: the replica map (`MAP`), each path it registers (`research.store.fixtures.entries`), and the code
      its check imports (`FIXTURES_CODE`)."""
    if step in WHOLE:
        return WHOLE[step]
    if step not in ("lean-agreement", "preflight-lock", "preflight-fixtures"):
        return f"nothing declares what {step} reads"
    if isinstance(got := _declared(repo, commit), str):
        return got
    check = (*got["CHECK_READS"], *ENV)
    if step == "lean-agreement":
        return frozenset((got["VERIFIER"], got["LEAN"], *check))
    if step == "preflight-lock":
        return frozenset((*LOCK_READS, *check))
    if isinstance(doc := _map(repo, commit), str):
        return doc
    return frozenset((MAP, *entries(doc).values(), FIXTURES_CODE, *check))


def key(repo: Path, commit: str, step: str) -> str | None:
    """``step``'s content key at ``commit``: SHA-512 over `KEY_TAG`, the step's name and each file of ``commit``'s tree it
    reads (`reads`), as git holds it (mode, object and path), in path order; for a step read whole, over the tag, its name and
    the commit's tree.  Two commits with one key give the step the same files to read.  None when git can't read ``commit``."""
    r = _git(repo, "rev-parse", "--verify", "-q", f"{commit}^{{tree}}", check=False)
    if r.returncode:
        return None
    tree = r.stdout.strip()
    h = hashlib.sha512(f"{KEY_TAG}\0{step}\0".encode())
    if isinstance(want := reads(repo, tree, step), str):
        h.update(f"tree\0{tree}".encode())
        return h.hexdigest()
    for path, entry in _files(repo, tree):
        if meets((path,), want):
            h.update(f"{entry}\t{path}\0".encode(errors="surrogateescape"))
    return h.hexdigest()


@functools.lru_cache(maxsize=8)
def _files(repo: Path, tree: str) -> tuple[tuple[str, str], ...]:
    """(path, "MODE TYPE OBJECT") of each entry under ``tree`` (an object id, which never changes, so a listing is kept), in
    git's order, which is path order."""
    out = _git(repo, "ls-tree", "-r", "-z", "--full-tree", tree).stdout
    return tuple((path, entry) for entry, _, path in (line.partition("\t") for line in out.split("\0") if line))


#: the names `_declared` reads of `BUILD`
DECLARED = ("CHECK_READS", "VERIFIER", "LEAN")


def _declared(repo: Path, commit: str) -> dict | str:
    """`BUILD`'s `DECLARED` at ``commit``, read from its source without running it (`_data`): `VERIFIER` and `LEAN` strings,
    `CHECK_READS` a tuple of them; else why not."""
    r = _git(repo, "show", f"{commit}:{BUILD}", check=False)
    if r.returncode:
        return f"{commit[:12]} has no {BUILD}"
    try:
        body = ast.parse(r.stdout).body
    except SyntaxError as e:
        return f"{BUILD} at {commit[:12]} doesn't parse: {e}"
    got = {node.targets[0].id: _data(node.value, {}) for node in body
           if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
           and node.targets[0].id in DECLARED}
    check = got.get("CHECK_READS")
    if (isinstance(got.get("VERIFIER"), str) and isinstance(got.get("LEAN"), str) and isinstance(check, tuple)
            and all(isinstance(p, str) for p in check)):
        return got
    return f"{BUILD} at {commit[:12]} doesn't declare {', '.join(DECLARED)} as plain data"


def _map(repo: Path, commit: str) -> dict | str:
    """The replica map at ``commit``, else why not."""
    r = _git(repo, "show", f"{commit}:{MAP}", check=False)
    if r.returncode:
        return f"{commit[:12]} has no {MAP}"
    try:
        doc = json.loads(r.stdout)
    except ValueError as e:
        return f"{MAP} at {commit[:12]} isn't JSON: {e}"
    return doc if isinstance(doc, dict) else f"{MAP} at {commit[:12]} isn't a JSON object"


def _data(node: ast.AST, names: dict[str, str]) -> str | tuple | None:
    """The value of ``node``, an expression of string literals, f-strings of the names a comprehension binds (``names``),
    tuples and lists with starred items, and comprehensions over those: what `BUILD` writes its declarations in.  None for
    anything else, which is never run."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return names.get(node.id)
    if isinstance(node, ast.JoinedStr):
        parts = [_data(v.value if isinstance(v, ast.FormattedValue) and v.conversion == -1 and v.format_spec is None else v,
                       names) for v in node.values]
        return "".join(parts) if all(isinstance(p, str) for p in parts) else None
    if isinstance(node, (ast.Tuple, ast.List)):
        out: list = []
        for e in node.elts:
            got = _data(e.value if isinstance(e, ast.Starred) else e, names)
            if not isinstance(e, ast.Starred):
                out.append(got)
            elif isinstance(got, tuple):
                out.extend(got)
            else:
                return None
        return None if None in out else tuple(out)
    if isinstance(node, (ast.GeneratorExp, ast.ListComp)) and len(node.generators) == 1:
        g = node.generators[0]
        if isinstance(g.target, ast.Name) and not g.ifs and not g.is_async and isinstance(over := _data(g.iter, names), tuple):
            items = tuple(_data(node.elt, {**names, g.target.id: x}) for x in over)
            return None if None in items else items
    return None


def _inputs(paths: set[str]) -> frozenset[str]:
    """``paths`` as `vqueue.meets` reads them, meeting the files `drive.touches` matches them to: an absolute one (a traceback's
    frame) as each of its tails that has a directory, since a bare `conftest.py` would meet every conftest."""
    out: set[str] = set()
    for p in paths:
        parts = p.split("/")
        out.update(("/".join(parts[i:]) for i in range(1, len(parts) - 1)) if p.startswith("/") else (p,))
    return frozenset(out)


def _down(rec: dict | None, why: str) -> Verdict:
    """Infra: what became of the run ``rec`` records."""
    ids = " ".join(rec["ids"]) if rec and rec.get("ids") else ""
    return Verdict(INFRA, (), (Failed(f"the run beside its check{f' ({ids})' if ids else ''} {why}",
                                      f"{rec['tip'] if rec else ''}:{ids or 'no run'}", frozenset()),))


def _keep(rec: dict, v: Verdict, passed: list[str] | None = None) -> Verdict:
    rec["verdict"] = {"state": v.state, **{k: [{"task": f.task, "key": f.key, "inputs": sorted(f.inputs),
                                                **({"base_key": f.base_key} if f.base_key is not None else {}),
                                                **({"tests": sorted(f.tests)} if f.tests else {})} for f in fs]
                                           for k, fs in (("failed", v.failed), ("infra", v.infra)) if fs},
                      **({"passed": passed} if passed is not None else {})}
    return v


def _kept(got: dict) -> Verdict:
    return Verdict(got["state"], *(tuple(Failed(f["task"], f["key"], frozenset(f["inputs"]), f.get("base_key"),
                                                frozenset(f.get("tests") or ()))
                                         for f in got.get(k) or ()) for k in ("failed", "infra")), ended=got["state"] == FAIL)


def step_test(step: str) -> str:
    """The one test id of ``step`` of the run beside a check: its failures are compared by it (`vqueue.Failed.tests`), as a
    step read whole (`WHOLE`) is never at its base's key."""
    return f"step:{step}"


def based(v: Verdict, base: dict | None) -> Verdict:
    """``v`` with each failed step's `base_tests` read from the run beside the base's check (``base``, that run's record, as
    `judge` keeps it): the step's test (`step_test`) when that run failed it, none when it passed it, and None while it runs, or
    when it ended without judging it, so `vqueue.decide` takes a failure the base has too as inherited."""
    got = (base or {}).get("verdict")
    if not isinstance(got, dict) or not v.failed:
        return v
    failed = {f.get("task") for f in got.get("failed") or () if isinstance(f, dict)}
    passed = set(got.get("passed") or ())

    def of(f: Failed) -> Failed:
        s = f.task.removeprefix("step ")
        if s == f.task:
            return f
        return replace(f, base_tests=frozenset({step_test(s)}) if f.task in failed else frozenset() if s in passed else None)

    return replace(v, failed=tuple(of(f) for f in v.failed))
