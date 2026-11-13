"""Replay real PR traffic through the verified queue's `decide`, beside what the old path did with the same PRs.  This
module does the I/O that `sim` and `snapshots` don't.

    uv run python -m verity_landq.replay trace --since SINCE --until UNTIL --out trace.json
    uv run python -m verity_landq.replay tasks PLAN --out tasks.json
    uv run python -m verity_landq.replay run trace.json tasks.json --queue vqueue:SRC --batch 8,16 --depth 3 --nodes 32 \
        --out replay.json

`trace` reads a window of landings, in a clone with `checked` and `main` fetched, `gh` authenticated and `research` on the
PATH.  Each `Queue: #N` merge on `checked`'s first-parent chain is an arrival: its second parent is the PR's head, and its
diff against the first parent is the paths it touches.  Its ready mark is the first `ready` label at that head (its landing
on `checked` when there is none), its review the first `reviewed` land verdict there, and its old-path landing the first
push of `main` holding the merge, from GitHub's push activity, as the landing metric reads them.  A `Revert "Queue: #N`
on `checked` marks it as breaking.

`tasks` turns a shard plan (`check.py --shard-plan --json`, each part's measured seconds) into gating tasks with the inputs
each part is keyed on: a suite's `inputs_of`, its runner and the lock; a Lean package's files and those of the packages it
requires; and the whole tree for a `lean-suites` part, as graph/v2 keys one.

`run` replays the trace in every combination of batch size, depth and node count, with breaking PRs, infra errors and false
fails drawn at the rates given, `--seeds` times each, and prints one row per combination beside the old path's.  The queue
is `research.vqueue` from a checkout (`--queue vqueue:SRC`, SRC its `tools/research/src`), driven by `snapshots.Driver`,
whose snapshot fields `--set KEY=VALUE` sets (`--set lean=3`), and which starts the tasks `--defer TASK` names as the loop
starts the run beside a check (`--defer slim`).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tomllib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path

from verity_landq import model as lq
from verity_landq import sim, snapshots

ROOT = Path(__file__).resolve().parents[4]
REPO = "compute-verification/verity"
QUEUE = re.compile(r"Queue: #(\d+)")
REVERT = re.compile(r'Revert "Queue: #(\d+)')
LEAN_ENV = ("lean-toolchain", "lake-manifest.json")
#: the step between the numbers a PR's repeated landings arrive under, above every PR number
REPEAT = 1_000_000


def _git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def _when(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _minutes(s: str, t0: datetime) -> float:
    return (_when(s) - t0).total_seconds() / 60


def pushes(ref: str) -> list[tuple[datetime, str]]:
    url = f"repos/{REPO}/activity?ref=refs/heads/{ref}&per_page=100&activity_type=push&time_period=month"
    out = subprocess.run(["gh", "api", "--paginate", url, "--jq", ".[] | [.timestamp, .after] | @tsv"], capture_output=True,
                         text=True, check=True).stdout
    return sorted((_when(ts), sha) for ts, sha in (ln.split("\t") for ln in out.splitlines() if ln))


def first_seen(ps: list[tuple[datetime, str]], since: datetime) -> dict[str, datetime]:
    """{commit: the first push of the ref whose tip holds it on its first-parent chain}."""
    seen: dict[str, datetime] = {}
    prev = None
    for t, sha in ps:
        if subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=ROOT, capture_output=True).returncode:
            continue
        rng = [sha, f"^{prev}"] if prev else [sha, f"--since={since.isoformat()}"]
        for c in _git("rev-list", "--first-parent", *rng).split():
            seen.setdefault(c, t)
        prev = sha
    return seen


def _labels(target: str, key: str) -> list[dict]:
    r = subprocess.run(["research", "data", "labels", target, "--key", key, "--history", "--json"], capture_output=True,
                       text=True, env={**os.environ, "RESEARCH_AT_CHECKOUT": "1"})
    if r.returncode:
        raise SystemExit(f"research data labels {target} --key {key} failed ({r.returncode}): {r.stderr.strip()[-400:]}")
    return json.loads(r.stdout or "[]")


def _marks(pr: int, head: str) -> tuple[datetime | None, datetime | None]:
    """The first ready mark and the first land verdict at ``head``."""
    target = f"pr:{pr}@{head}"
    ready = [_when(x["ts"]) for x in _labels(target, "ready") if x.get("value") is True]
    land = [_when(x["ts"]) for x in _labels(target, "reviewed") if x.get("value") == "land"]
    return min(ready, default=None), min(land, default=None)


def trace(since: datetime, until: datetime, jobs: int = 12) -> dict:
    checked, main = pushes("checked"), pushes("main")
    on_checked, on_main = first_seen(checked, since - timedelta(days=6)), first_seen(main, since - timedelta(days=6))
    subjects: dict[str, tuple[list[str], str]] = {}
    for line in _git("log", "--first-parent", f"--since={(since - timedelta(days=6)).isoformat()}", "--format=%H %P%x09%s",
                     checked[-1][1]).splitlines():
        shas, _, subject = line.partition("\t")
        c, *parents = shas.split()
        subjects[c] = (parents, subject)
    reverted = {int(m.group(1)) for _, s in subjects.values() if (m := REVERT.match(s))}
    rows = []
    for c, t in on_checked.items():
        parents, subject = subjects.get(c, ([], ""))
        m = QUEUE.match(subject)
        if m and len(parents) > 1 and since - timedelta(days=1) <= t < until:
            rows.append({"pr": int(m.group(1)), "head": parents[1], "landing": c, "checked": t, "main": on_main.get(c)})
    with ThreadPoolExecutor(jobs) as ex:
        marks = list(ex.map(lambda r: _marks(r["pr"], r["head"]), rows))
        paths = list(ex.map(lambda r: _git("diff", "--no-renames", "--name-only", f"{r['landing']}^1", r["landing"]).split(),
                            rows))
    out = []
    for r, (ready, land), ps in zip(rows, marks, paths):
        marked = ready is not None and ready <= r["checked"]
        ready = ready if marked else r["checked"]
        if not since <= ready < until:
            continue
        reviewed = max(ready, land) if land is not None and land <= r["checked"] else ready
        out.append({"pr": r["pr"], "head": r["head"], "ready": ready.isoformat(), "reviewed": reviewed.isoformat(),
                    "marked": marked, "checked": r["checked"].isoformat(),
                    "main": r["main"].isoformat() if r["main"] else None, "reverted": r["pr"] in reverted,
                    "paths": sorted(ps)})
    return {"kind": "landq-trace/v1", "since": since.isoformat(), "until": until.isoformat(),
            "main_tip": main[-1][1], "checked_tip": checked[-1][1], "arrivals": sorted(out, key=lambda a: a["ready"])}


def _lean_packages() -> dict[str, list[str]]:
    """{Lean package directory: the package directories it requires}, from each tracked lakefile.toml's `[[require]] path`."""
    out = {}
    for f in _git("ls-files", "*lakefile.toml").split():
        d = Path(f).parent
        if "/tests/fixtures/" in f:
            continue
        reqs = tomllib.loads((ROOT / f).read_text()).get("require", [])
        out[d.as_posix()] = [os.path.normpath((d / r["path"]).as_posix()) for r in reqs if "path" in r]
    return out


def _owned(pkg: str, pkgs: set[str]) -> list[str]:
    """The paths of ``pkg``'s own files: its directory, less the packages nested in it."""
    nested = [p for p in pkgs if p != pkg and p.startswith(pkg + "/")]
    if not nested:
        return [pkg + "/"]
    out = []
    for child in sorted(_git("ls-tree", "--name-only", "HEAD", pkg + "/").split()):
        if child in nested:
            continue
        out += _owned(child, pkgs) if any(n.startswith(child + "/") for n in nested) else [child]
    return out


def tasks(plan: dict, lean_suites: str = "tree") -> list[dict]:
    """The plan's parts as gating tasks; a `lean-suites` part keyed on the whole tree (``lean_suites="tree"``, as graph/v2
    keys it) or on its suite's inputs (``"suite"``)."""
    sys.path.insert(0, str(ROOT / "tools/verity/check"))
    import suites as S
    found = {s.name: s for s in S.discover()}
    members = S.members()
    lean = _lean_packages()
    every = set(lean)

    def closure(pkg: str) -> set[str]:
        seen, todo = set(), [pkg]
        while todo:
            p = todo.pop()
            if p not in seen and p in lean:
                seen.add(p)
                todo += lean[p]
        return seen

    lean_inputs = {p: sorted({q for d in closure(p) for q in _owned(d, every)} | set(LEAN_ENV)) for p in lean}
    py_env = ["uv.lock", "pyproject.toml", *S.RUNNER]
    out = []
    for part, (seconds, _how) in sorted(plan["tasks"].items()):
        kind, _, name = part.partition(":")
        suite = name.rsplit(":", 1)[0] if re.search(r":\d+/\d+$", name) else name
        if kind == "pytest" and suite in found or kind == "lean-suites" and lean_suites == "suite" and suite in found:
            inputs = sorted({*S.inputs_of(found[suite], members), *py_env})
        elif kind == "lean-suites":
            inputs = ["."]
        elif kind == "lean-audit" and name in lean_inputs:
            inputs = lean_inputs[name]
        elif kind in ("lean-build", "lean-unit-cut", "lean-agreement"):
            inputs = sorted({*lean_inputs.get("verity/protocols/flock/verifier", []), "verity/protocols/flock/agreement/"})
        elif kind in ("flock-circuit-build", "flock-rust-tests"):
            inputs = ["verity/protocols/flock/prover/"]
        else:
            raise SystemExit(f"tasks: the plan's part {part} names no suite or Lean package this tree has")
        out.append({"name": part, "inputs": inputs, "minutes": round(seconds / 60, 2), "off_pool": False})
    return out


def arrivals(doc: dict, bad: float, seed: int, world: sim.World) -> tuple[list[sim.Arrival], datetime]:
    """The trace's arrivals in minutes from its start; each breaks one task it touches with probability ``bad``, and a
    reverted one always does.  Each landing is its own change, so a PR that landed k times before arrives k times, the
    i-th repeat under the number ``pr + i * REPEAT``, rather than as a new head of the first."""
    t0 = _when(doc["since"])
    rng = random.Random(seed)
    index = sim.Index(world.tasks)
    seen: Counter[int] = Counter()
    out = []
    for a in doc["arrivals"]:
        paths = frozenset(a["paths"])
        touched = sorted(index.touch(paths))
        breaks = frozenset()
        if touched and (a["reverted"] or rng.random() < bad):
            breaks = frozenset({world.tasks[rng.choice(touched)].name})
        number = a["pr"] + seen[a["pr"]] * REPEAT
        seen[a["pr"]] += 1
        out.append(sim.Arrival(number, a["head"], _minutes(a["ready"], t0), paths, _minutes(a["reviewed"], t0), breaks))
    return out, t0


def old_path(doc: dict, within: float = 120.0, lag: float = 180.0, every: float = 10.0) -> dict:
    """What the old path did with the same arrivals: ready-to-main from the ready mark to the first push of `main` holding
    the landing (a reverted one never counts as on `main`); main's lag as the simulator reads it, the age of the PR held
    longest off `main` since it was ready and reviewed; and main's lag as the age of the oldest landing on `checked` not yet
    on `main` (``checked_lag_*``), sampled only from the first push of `checked` the trace holds (``checked_from``), since
    a landing older than that reads as first seen there."""
    t0, end = _when(doc["since"]), _minutes(doc["until"], _when(doc["since"]))
    rows, held, landed = [], [], []
    for a in doc["arrivals"]:
        on_main = None if a["main"] is None or a["reverted"] else _minutes(a["main"], t0)
        rows.append(sim.Landing(a["pr"], a["head"], _minutes(a["ready"], t0), _minutes(a["reviewed"], t0), on_main, 0, 0))
        out = _minutes(a["main"], t0) if a["main"] else end
        held.append((_minutes(a["reviewed"], t0), out))
        landed.append((_minutes(a["checked"], t0), out))
    s = sim.summary(sim.Report(tuple(rows), tuple(held), end, 1), within=within, lag=lag, every=every)
    first = min((a["checked"] for a in doc["arrivals"]), default=doc["since"])
    c = sim.summary(sim.Report((), tuple(landed), end, 1), lag=lag, every=every, start=_minutes(first, t0))
    return {**{k: s[k] for k in ("prs", "landed", "p50_h", "p90_h", "review_p50_h", "review_p90_h", "within", "lag_max_h",
                                 "lag_ok")},
            "checked_lag_max_h": c["lag_max_h"], "checked_lag_ok": c["lag_ok"], "checked_from": first}


def queue(spec: str, fields: dict | None = None, deferred: frozenset[str] = frozenset()) -> tuple:
    """The queue a run replays, ``vqueue:SRC``: the snapshot queue `research.vqueue` imported from the directory SRC (a
    checkout's `tools/research/src`), driven by `snapshots.Driver` with snapshot ``fields`` and the ``deferred`` tasks it
    starts; as (step, start)."""
    kind, _, src = spec.partition(":")
    if kind != "vqueue" or not (Path(src) / "research" / "vqueue.py").is_file():
        raise SystemExit(f"--queue {spec}: give `vqueue:SRC`, SRC a directory holding research/vqueue.py")
    sys.path.insert(0, src)
    from research import vqueue
    driver = snapshots.Driver(vqueue, deferred=deferred, **(fields or {}))
    return driver.step, driver.start


def _field(s: str) -> tuple[str, object]:
    key, eq, value = s.partition("=")
    if not key or not eq:
        raise argparse.ArgumentTypeError(f"--set {s}: give KEY=VALUE")
    if re.fullmatch(r"-?\d+", value):
        return key, int(value)
    if re.fullmatch(r"-?(\d+\.\d*|\.\d+)([eE]-?\d+)?", value):
        return key, float(value)
    return key, value


def _one(job: tuple) -> dict:
    doc, task_rows, k, d, nodes, seed, rates, spec, fields, deferred = job
    world = sim.World(tasks=tuple(sim.Task(t["name"], frozenset(t["inputs"]), t["minutes"], t.get("off_pool", False),
                                           t["name"] in deferred) for t in task_rows),
                      nodes=nodes, build=rates["build"], listing=rates["listing"], infra=rates["infra"],
                      flaky=rates["flaky"], fix=rates["fix"], seed=seed)
    trace_, _ = arrivals(doc, rates["bad"], seed, world)
    step, start = queue(spec, fields, deferred)
    report = sim.simulate(trace_, world, lq.Policy(batch=k, depth=d), step=step, state=start("main"))
    return {"batch": k, "depth": d, "nodes": nodes, "seed": seed, **sim.summary(report)}


def _mean(rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if k not in ("batch", "depth", "nodes", "seed")]
    return {**{k: rows[0][k] for k in ("batch", "depth", "nodes")}, "seeds": len(rows),
            **{k: sum(r[k] for r in rows) / len(rows) for k in keys}}


def _table(old: dict, rows: list[dict]) -> str:
    head = ("k  d  nodes   p50 h   p90 h  from review: p50 h   p90 h  on main ≤2 h  lag max h  lag ≤3 h  ejects  false  "
            "cands  node use")
    lines = [head, f"old path       {old['p50_h']:6.2f}  {old['p90_h']:6.2f}  {old['review_p50_h']:19.2f}  "
                   f"{old['review_p90_h']:6.2f}  {old['within']:12.0%}  {old['lag_max_h']:9.1f}  {old['lag_ok']:8.0%}   (the "
                   f"oldest landing on checked not yet on main, from {old['checked_from'][:16]}: max "
                   f"{old['checked_lag_max_h']:.1f} h, ≤3 h {old['checked_lag_ok']:.0%})"]
    for r in rows:
        lines.append(f"{r['batch']:<2d} {r['depth']:<2d} {r['nodes']:<5d}  {r['p50_h']:6.2f}  {r['p90_h']:6.2f}  "
                     f"{r['review_p50_h']:19.2f}  {r['review_p90_h']:6.2f}  "
                     f"{r['within']:12.0%}  {r['lag_max_h']:9.1f}  {r['lag_ok']:8.0%}  {r['ejects']:6.1f}  "
                     f"{r['false_ejects']:5.1f}  {r['candidates']:5.0f}  {r['utilization']:8.0%}")
    return "\n".join(lines)


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trace")
    t.add_argument("--since", required=True)
    t.add_argument("--until", required=True)
    t.add_argument("--out", type=Path, required=True)
    k = sub.add_parser("tasks")
    k.add_argument("plan", type=Path)
    k.add_argument("--slim", type=float, default=20.0, help="minutes of the steps every candidate runs beside its tasks "
                                                            "(off the pool; 0: none)")
    k.add_argument("--lean-chain", type=float, default=None, help="minutes for each Security audit in place of the plan's")
    k.add_argument("--lean-suites-inputs", choices=("tree", "suite"), default="tree",
                   help="key each lean-suites part on the whole tree (graph/v2) or on its suite's inputs")
    k.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("trace", type=Path)
    r.add_argument("tasks", type=Path)
    r.add_argument("--batch", type=_ints, default=[1, 2, 4, 6, 8])
    r.add_argument("--depth", type=_ints, default=[1, 2, 3])
    r.add_argument("--nodes", type=_ints, default=[2, 4, 8])
    r.add_argument("--seeds", type=int, default=3)
    r.add_argument("--bad", type=float, default=0.02, help="the share of PRs that break one task they touch")
    r.add_argument("--infra", type=float, default=0.02, help="the probability a run ends in an infra error")
    r.add_argument("--flaky", type=float, default=0.0005, help="the probability a run fails falsely")
    r.add_argument("--fix", type=float, default=120.0, help="minutes until an ejected PR comes back fixed")
    r.add_argument("--build", type=float, default=2.0)
    r.add_argument("--listing", type=float, default=3.0)
    r.add_argument("--queue", required=True, metavar="vqueue:SRC",
                   help="research.vqueue from SRC, a checkout's tools/research/src")
    r.add_argument("--set", dest="fields", type=_field, action="append", default=[], metavar="KEY=VALUE",
                   help="a snapshot field for a snapshot queue, such as lean=3 (repeatable)")
    r.add_argument("--defer", action="append", default=[], metavar="TASK",
                   help="a task that runs only when the snapshot queue's driver starts it, as the loop starts the run "
                        "beside a check (`slim`; repeatable)")
    r.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    r.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "trace":
        doc = trace(_when(a.since), _when(a.until))
        a.out.write_text(json.dumps(doc, indent=1, default=str) + "\n")
        print(f"trace: {len(doc['arrivals'])} arrivals, {sum(x['marked'] for x in doc['arrivals'])} with a ready mark")
    elif a.cmd == "tasks":
        rows = tasks(json.loads(a.plan.read_text()), a.lean_suites_inputs)
        if a.lean_chain is not None:
            for row in rows:
                if row["name"].startswith("lean-audit:Security"):
                    row["minutes"] = a.lean_chain
        if a.slim:
            rows.append({"name": "slim", "inputs": ["."], "minutes": a.slim, "off_pool": True})
        a.out.write_text(json.dumps({"kind": "landq-tasks/v1", "tasks": rows}, indent=1) + "\n")
        print(f"tasks: {len(rows)}, {sum(x['minutes'] for x in rows):.0f} min in all")
    else:
        doc, task_rows = json.loads(a.trace.read_text()), json.loads(a.tasks.read_text())["tasks"]
        rates = {k: getattr(a, k) for k in ("bad", "infra", "flaky", "fix", "build", "listing")}
        fields, deferred = dict(a.fields), frozenset(a.defer)
        unknown = sorted(deferred - {t["name"] for t in task_rows})
        if unknown:
            raise SystemExit(f"--defer {', '.join(unknown)}: no such task in {a.tasks}")
        queue(a.queue, fields, deferred)
        jobs = [(doc, task_rows, k, d, n, s, rates, a.queue, fields, deferred)
                for k, d, n in product(a.batch, a.depth, a.nodes) for s in range(a.seeds)]
        with ProcessPoolExecutor(a.jobs) as ex:
            done = list(ex.map(_one, jobs))
        groups: dict[tuple, list[dict]] = {}
        for row in done:
            groups.setdefault((row["batch"], row["depth"], row["nodes"]), []).append(row)
        rows = [_mean(g) for _, g in sorted(groups.items())]
        old = old_path(doc)
        a.out.write_text(json.dumps({"kind": "landq-replay/v1", "queue": a.queue, "fields": fields,
                                     "deferred": sorted(deferred), "rates": rates, "old_path": old, "rows": rows,
                                     "runs": done}, indent=1) + "\n")
        print(_table(old, rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
