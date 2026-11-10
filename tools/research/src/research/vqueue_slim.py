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
of failing tests.

The check that gates (`research queue vqueue --gate check`).  `start` with ``shards`` launches drive's after-landing run of
the candidate in place of the run of `SLIM_STEPS`: every step of `AFTER_LANDING` as that many shards, one launch a shard on
its own machines (`shard_machines`), with the keywords drive's own gets (`Driver._postland_args`).  Its record says so
(``gate``), and `judge` then judges every step `main` waits for at the candidate's base (`Driver._gating` there, the
preflights and a lean-agreement that ran, less policy.toml's `off_gate` and drive.toml's `warn_steps`), a step a shard holds
passed only once every shard holding it passed (`shard_passed`).  A failed step's tests are what its failure names
(`failing`), so `vqueue.decide` takes as inherited only a failure whose every failing test the run beside the base fails
too; what a failure doesn't name is one test no other key has (`unnamed`), so it is never inherited."""
from __future__ import annotations

import ast
import functools
import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

from .drive import (ACTIVE, AFTER_LANDING, LAUNCH_LOST_S, LEAN_AUDIT_RE, PREFLIGHT, SIGTERM_RC, SLIM_JUDGED, SLIM_STEPS,
                    SUITE_STEPS, Driver, _step_log, implicated, killed, not_run, shard_machines, shard_passed, shard_steps,
                    step_paths, step_records, suite_paths, suite_rows)
from .queue import CHECK_STEPS, _git
from .store.fixtures import REPLICA_MAP, entries
from .vqueue import FAIL, INFRA, LIST_FAILED, PASS, PENDING, STATES, Failed, Verdict, meets

#: a candidate's state from its two verdicts' (`combine`): the first of these either has
ORDER = (FAIL, INFRA, PENDING, LIST_FAILED, PASS)
#: the train.sh a candidate's check launches with: the loop's own, deployed with its pin, so a candidate's tree doesn't decide
#: how its launch is waited for
OWN_TRAIN = Path(__file__).resolve().parents[4] / "tools" / "verity" / "check" / "train.sh"
#: what a launch that registered no run says when it points at the machines, not the candidate's tree (`Failed.launch`):
#: train.sh stopped its launcher, still running, at the end of its wait; no run ids within `LAUNCH_LOST_S`; the pod lost its
#: process.  A launcher that exited without a run ran the candidate's own check.py, and says none of these
LAUNCH_ONLY = ("no run launched in the wait", "min after its launch", "its process is gone with no exit written")
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
#: a record's ``gate`` when its run is the check that gates (`start` with ``shards``)
CHECK = "check"
#: pytest's failing tests and the modules it couldn't collect, in a step's log's short summary
TEST_RE = re.compile(r"^(?:FAILED|ERROR) (\S+)", re.M)
#: lean-audit's own failures of no package, in its log
AUDIT_FAIL_RE = re.compile(r"^lean-audit: FAIL: .*$", re.M)
#: what the audit prints under a package's `AUDIT .../PKG: FAIL` that is no failure: counts and timings of its run
AUDIT_SAYS = ("closure:", "staged:", "trust gap:", "heartbeats:")
#: the audit's scratch tree, which each run names anew
SCRATCH_RE = re.compile(r"\S*?/lean-audit-scratch-[^/\s]+/")


def start(driver: Driver, sha: str, base: str, shards: int = 0) -> dict | None:
    """Launch the run beside candidate ``sha``'s check, ``base`` the commit of `main` below it, as `Driver._start_slim` launches
    it beside the executor's check of ``sha`` with ``base`` verified: `SLIM_STEPS` but those a run of every step of ``sha``
    above ``base`` skips by name (lean-agreement, unless what it checks changed), with that run's keywords (ratchet-gate
    judges the merges above ``base``), on drive's machines (`Driver.on`).  With ``shards``, the check that gates instead, as
    `Driver.start_postland` launches the after-landing run: `--deferred` every step of `AFTER_LANDING` (each check runs the
    preflights first), `--shard K/N` on shard K's machines (`shard_machines`) for each of that many, one whole run for 1.
    The record to keep (`Driver._started`): ``ids``, none yet while a detached launch runs (its ``launch`` then, or each
    shard's in ``shards``, which `judge` makes ``groups``, each shard's ids); the candidate (``tip``), ``base``, when it
    launched (``at``), the ``steps`` it runs and those it skips (``skip``), and ``gate`` (`CHECK`) with ``shards``.  None,
    logged, when it registered no run (a shard that registered none stops the others), or when ``driver`` launches nothing (a
    dry run, or its queue's shadow), as `Driver._start_slim` does."""
    if not driver.on:
        raise ValueError("the run beside a candidate's check needs drive's machines: build the Driver with drive.toml's `on` "
                         "(`research.queue.drive_flags(q.workdir)[0].on`)")
    kw, skip, _, flags = driver._postland_args({"verified": base}, sha, False)
    every = AFTER_LANDING if shards else SLIM_STEPS
    skip = [s for s in skip if s in every]
    steps = [s for s in every if s not in skip]
    sent = ",".join(every if shards else steps)
    what = (f"train.sh {sha[:12]} {driver.on} --deferred {sent}{flags}" + (f" as {shards} shards" if shards > 1 else "")
            + (f" as candidate {sha[:12]}'s check" if shards else f" beside candidate {sha[:12]}'s check"))
    if driver.dry_run or driver.q.shadow:
        driver.q.log(f"research queue vqueue: would launch {what}")
        return None
    if shards > 1:
        got = [driver._started(driver.launch_postland(driver.repo, sha, on, sent, **kw, shard=f"{k}/{shards}",
                                                      train_from=OWN_TRAIN))
               for k, on in enumerate(shard_machines(driver.on, shards), 1)]
        rec, tail = {"ids": [], "shards": [r for r, _ in got]}, next((t for r, t in got if r is None), "")
        if any(r is None for r, _ in got):
            if ids := [i for r, _ in got if r for i in r["ids"]]:
                driver.cancel(driver.repo, ids, f"vqueue: a shard of candidate {sha[:12]}'s check registered no run")
            rec = None
        elif not any("launch" in r for r, _ in got):
            rec = _grouped(rec)
    else:
        rec, tail = driver._started(driver.launch_postland(driver.repo, sha, driver.on, sent, **kw, train_from=OWN_TRAIN))
    if rec is None:
        driver.q.log(f"research queue vqueue: {what} registered no run: {tail}")
        return None
    driver.q.log(f"research queue vqueue: {what}: {' '.join(rec['ids']) or 'launching'}")
    return {**rec, "tip": sha, "base": base, "at": driver.now(), "steps": steps, **({"skip": skip} if skip else {}),
            **({"gate": CHECK} if shards else {})}


def _grouped(rec: dict) -> dict:
    """``rec`` once each of its ``shards`` has its run ids: they are its ``groups``, and its ``ids`` every one's."""
    rec["groups"] = [list(r["ids"]) for r in rec.pop("shards")]
    rec["ids"] = [i for g in rec["groups"] for i in g]
    return rec


def judge(driver: Driver, rec: dict | None) -> Verdict:
    """The verdict of the run `start` recorded in ``rec``, read as `Driver._follow_slim` reads it, on the steps it judges on
    the merge gate: `SLIM_JUDGED` but a step it skipped and policy.toml's `off_gate` at ``tip``; of the check that gates
    (``gate``), every step `main` waits for at ``base`` (`Driver._gating`), the preflights and a lean-agreement it ran, but
    policy.toml's `off_gate` there and drive.toml's `warn_steps`.  Pending while it launches or runs, each shard's launch
    read as the run's own is.  Infra when it registered no run (``rec`` None, or a detached launch that printed no run ids,
    a shard's stopping the other shards; `Failed.launch` when it says one of `LAUNCH_ONLY`), never ran or was stopped, or was lost or killed before it passed each of those
    steps; infra's one `Failed` says which.  Once it ended, pass when it passed each of them (the check that gates: in
    every shard that holds it, `shard_passed`), fail with each of them it failed (a `Failed`: ``step NAME``, the step's
    content key at the candidate, the paths drive names that failure's culprit by, `step_paths`, its content key at
    ``base``, `key`, and the step's one test, `step_test`; the check that gates: the tests and paths its failure names,
    `failing`), and infra when it failed none but passed fewer (one skipped, or absent from its steps.jsonl).  Each verdict
    but pending is kept in ``rec`` (``verdict``, with the steps it passed) and read from there."""
    if rec is None:
        return _down(None, "registered no run")
    if "verdict" in rec:
        return _kept(rec["verdict"])
    tip, check = rec["tip"], rec.get("gate") == CHECK
    what = f"the run of {', '.join(rec['steps'])} " + (f"as candidate {tip[:12]}'s check" if check else
                                                      f"beside candidate {tip[:12]}'s check")
    shards = rec.get("shards") or ()
    # every shard's launch is read in one call, since a launch read once is gone: the record never holds some read, some not
    printed = [driver.launched(r["launch"])[0] for r in shards if "launch" in r]
    if None in printed and [] not in printed and driver.now() - rec["at"] <= LAUNCH_LOST_S:
        return Verdict(PENDING)
    for k, shard in enumerate(shards, 1):
        shard.setdefault("at", rec["at"])
        if (tail := driver._resolve(shard, f"shard {k}/{len(shards)} of {what}")) is not None:
            if got := [i for r in shards for i in r["ids"]]:
                driver.cancel(driver.repo, got, f"vqueue: shard {k}/{len(shards)} of candidate {tip[:12]}'s check registered "
                                                "no run")
            return _keep(rec, _down(rec, f"registered no run: shard {k}/{len(shards)}: {tail}"))
    if any("launch" in r for r in shards):
        return Verdict(PENDING)
    if shards:
        _grouped(rec)
    if (tail := driver._resolve(rec, what)) is not None:
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
    base = rec.get("base")
    if check:
        gate = {*SLIM_JUDGED, *driver._gating(base or tip)} - {*driver._policy(base or tip).off_gate, *driver.warn_steps}
        judged = [s for s in CHECK_STEPS if s in gate and (s in rec["steps"] or s in PREFLIGHT)]
        view = shard_steps(driver._shards(rec), [any(states[r][1] == 0 for r in g) for g in rec.get("groups") or [ids]])
        passes = shard_passed(view, driver._allowed_steps(tip, [d for ds in driver._shards(rec) for d in ds]))
        missed = [s for s in judged if s not in passes]
    else:
        off = driver._policy(tip).off_gate
        judged = [s for s in SLIM_JUDGED if (s in rec["steps"] or s not in SLIM_STEPS) and s not in off]
        missed = driver._unrun(ids, judged, tip)
    if not missed:
        return _keep(rec, Verdict(PASS), judged)
    dirs = [d for ds in driver._shards(rec) for d in ds]
    ok = driver._every_shard(rec, states, lambda rc: rc == 0) or (
        driver._every_shard(rec, states, lambda rc: rc in (0, 1)) and driver._only_allowed(rec))
    if not ok and (lost or any(isinstance(rc, int) and (rc < 0 or (rc > 128 and rc != SIGTERM_RC))
                                   for _, rc, *_ in states.values())
                       or any(implicated(d) is None and killed(d, why) for d in dirs)):
        return _keep(rec, _down(rec, "was lost" if lost else f"was killed ({why})"))
    paths: dict[str, set[str]] = {}
    tests: dict[str, set[str]] = {}
    keys: dict[str, str] = {}
    for d in dirs:
        for st in step_records(d):
            if (s := st.get("step")) in missed and st.get("status") == "failed" and not not_run(st):
                if s not in keys:
                    keys[s] = key(driver.repo, tip, s) or f"{tip}:{s}"
                at, named = failing(driver, tip, d, st, unnamed(keys[s])) if check else (step_paths(d, st), {step_test(s)})
                paths.setdefault(s, set()).update(at or ())
                tests.setdefault(s, set()).update(named)
    passed = [s for s in judged if s not in missed]
    if paths:
        return _keep(rec, Verdict(FAIL, tuple(Failed(f"step {s}", keys[s], _inputs(paths[s]),
                                                     key(driver.repo, base, s) if base else None, frozenset(tests[s]))
                                              for s in missed if s in paths), ended=True), passed)
    return _keep(rec, _down(rec, f"ended ({why}) without passing {', '.join(missed)}: each skipped, or absent from its "
                                 "steps.jsonl"), passed)


def unnamed(key: str) -> str:
    """The one test of what a failure keyed ``key`` doesn't name: no failure under another key has it, so
    `vqueue.decide` never takes it as inherited by its tests."""
    return f"unnamed:{key}"


def failing(driver: Driver, tip: str, d: Path, st: dict, rest: str) -> tuple[set[str], set[str]]:
    """What failed step ``st`` of the check that gates names in its run dir ``d``, as drive reads it: the paths its culprit
    is named by, and its failing tests.  A step that runs suites (`SUITE_STEPS`): of each suite of its that failed
    (`suite_rows`) uncovered by ``tip``'s allowances, `suite_paths` and `SUITE::TEST` of each test it failed, and ``rest``
    under the suite's name when it names none or failed for more than its tests (its row's `beyond_tests` isn't false).
    Any other: `step_paths`, each `FAILED` and `ERROR` of its log's short summary, and of lean-audit each line under a
    failed package's `AUDIT .../PKG: FAIL` (`audit_failures`).  ``rest`` (`unnamed`) alone when it names no test."""
    step = st.get("step")
    if step not in SUITE_STEPS:
        log = _step_log(d, st)
        tests = set(TEST_RE.findall(log)) | (audit_failures(log, rest) if step == "lean-audit" else set())
        return step_paths(d, st) or set(), tests or {rest}
    paths: set[str] = set()
    tests = set()
    for name, row in suite_rows(d).items():
        if row["status"] != "failed" or row["step"] != step or driver._allowed(tip, f"suite {name}", d, row).covered:
            continue
        paths |= suite_paths(d, row) or set()
        tests |= {f"{name}::{t}" for t in row.get("failed_tests") or ()}
        if not row.get("failed_tests") or row.get("beyond_tests") is not False:
            tests.add(f"{name}::{rest}")
    return paths, tests or {rest}


def audit_failures(log: str, rest: str) -> set[str]:
    """The failures lean-audit's ``log`` names: `PKG::LINE` of each line the audit prints under a failed package's
    `AUDIT .../PKG: FAIL` (`LEAN_AUDIT_RE`) but its run's counts and timings (`AUDIT_SAYS`, with the lines under each), the
    scratch tree's name left out, ``rest`` under the name of a package it prints none under; and each `lean-audit: FAIL:`
    line."""
    out: dict[str, set[str]] = {}
    pkg, says = None, False
    for line in log.splitlines():
        if line.startswith("AUDIT"):
            pkg = m[1] if (m := LEAN_AUDIT_RE.match(line)) else None
            if pkg:
                out.setdefault(pkg, set())
        elif pkg and line.startswith(" ") and (text := line.strip()):
            if re.match(r"  \S", line):
                says = text.startswith(AUDIT_SAYS)
            if not says:
                out[pkg].add(SCRATCH_RE.sub("", text))
    return {f"{p}::{t}" for p, ts in out.items() for t in ts or {rest}} | set(AUDIT_FAIL_RE.findall(log))


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
                                      f"{rec['tip'] if rec else ''}:{ids or 'no run'}", frozenset(),
                                      launch=why.startswith("registered no run") and any(m in why for m in LAUNCH_ONLY)),))


def _keep(rec: dict, v: Verdict, passed: list[str] | None = None) -> Verdict:
    rec["verdict"] = {"state": v.state, **{k: [{"task": f.task, "key": f.key, "inputs": sorted(f.inputs),
                                                **({"base_key": f.base_key} if f.base_key is not None else {}),
                                                **({"tests": sorted(f.tests)} if f.tests else {}),
                                                **({"launch": True} if f.launch else {})} for f in fs]
                                           for k, fs in (("failed", v.failed), ("infra", v.infra)) if fs},
                      **({"passed": passed} if passed is not None else {})}
    return v


def _kept(got: dict) -> Verdict:
    return Verdict(got["state"], *(tuple(Failed(f["task"], f["key"], frozenset(f["inputs"]), f.get("base_key"),
                                                frozenset(f.get("tests") or ()), launch=bool(f.get("launch")))
                                         for f in got.get(k) or ()) for k in ("failed", "infra")), ended=got["state"] == FAIL)


def step_test(step: str) -> str:
    """The one test id of ``step`` of the run beside a check: its failures are compared by it (`vqueue.Failed.tests`), as a
    step read whole (`WHOLE`) is never at its base's key."""
    return f"step:{step}"


def based(v: Verdict, base: dict | None) -> Verdict:
    """``v`` with each failed step's `base_tests` read from the run beside the base's check (``base``, that run's record, as
    `judge` keeps it): the tests that run failed of it (the step's, `step_test`, where it kept none), none when it passed it,
    and None while it runs, or when it ended without judging it, so `vqueue.decide` takes a failure the base has too as
    inherited."""
    got = (base or {}).get("verdict")
    if not isinstance(got, dict) or not v.failed:
        return v
    failed = {f.get("task"): f.get("tests") for f in got.get("failed") or () if isinstance(f, dict)}
    passed = set(got.get("passed") or ())

    def of(f: Failed) -> Failed:
        s = f.task.removeprefix("step ")
        if s == f.task:
            return f
        return replace(f, base_tests=frozenset(failed[f.task] or {step_test(s)}) if f.task in failed else
                       frozenset() if s in passed else None)

    return replace(v, failed=tuple(of(f) for f in v.failed))
