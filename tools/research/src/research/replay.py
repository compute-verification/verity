"""`research replay SET... --dry-run`: which attempts of a replay set can be launched again at a new commit, and how.

A replay set is everything labelled `replay_set=SET` (store/vocab.py `REPLAY_SET`, `AREA/re-verify` or `AREA/re-run`): an attempt
(a run id), an artifact (through the attempts that produced it) or a derivation (its attempts). Each attempt is classified from
its own records, never guessed:

  replayable      the records hold what `research run` needs to launch it again with only the source swapped: the runner's argv
                  (launch.json, which a `--on` run's machine writes), a shipped source tree, every `--send` input in the store,
                  no redacted value, and a machine still listed for its project. The relaunch is printed.
  needs window    as replayable, but its command held a GPU lease (`gpu-lease`): a timed GPU run goes through the node's queue in
                  a booked window, so the `research run --queue` an owner would queue is printed, never launched here.
  not replayable  one of those is missing; the reason names it.
  unknown         its records are in the store but not in this local one; `--fetch` reads them from the remote.

The dry run reads the local store, the local runs root and the machines table only (`--fetch` allows the store's remote reads).
A relaunch records `RESEARCH_REPLAY_OF=<original run id>` as an `--env`, which every tree's runner records as `params.env`.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import remote
from . import runs as R
from .store import ids
from .store import vocab as V
from .store.local import LocalStore, NotFound

REPLAYABLE, NEEDS_WINDOW, NOT_REPLAYABLE, UNKNOWN = "replayable", "needs window", "not replayable", "unknown"
CLASSES = (REPLAYABLE, NEEDS_WINDOW, NOT_REPLAYABLE, UNKNOWN)
REPLAY_OF_ENV = "RESEARCH_REPLAY_OF"
REDACTED = "<redacted>"
#: flags of launch.json's argv a relaunch makes its own (value count): the machine's runner's (remote._launch_request), those the
#: launcher adds for the tree it ships (cli._run_remote --source-identity, remote.launch --tree-sha), and --project, passed explicitly
DROPPED = {"--runs-dir": 1, "--id": 1, "--claimed": 0, "--cost-json": 1, "--remote-json": 1, "--exclusive-lock": 1,
           "--custody-dir": 1, "--source-identity": 1, "--tree-sha": 1, "--project": 1}
GPU_LEASE = re.compile(r"(?:^|[\s/])gpu[-_]lease(?:\.sh)?(?:\s|$)")
DEFAULT_CUSTODY_TTL_H = 6
CUSTODY_TTL_CAP_H = 24


@dataclass
class Plan:
    run: str | None
    targets: list[str]
    cls: str
    reason: str = ""
    machine: str | None = None
    commit: str | None = None
    command: list[str] | None = None
    prepare: list[list[str]] = field(default_factory=list)

    def to_doc(self) -> dict[str, Any]:
        d = asdict(self)
        d["class"] = d.pop("cls")
        return d


@dataclass
class Records:
    launch: dict[str, Any]
    job: dict[str, Any]
    sends: list[str]            # run-dir paths under inputs/
    sends_from: str             # the run record's art id, or the local run dir


class Missing(Exception):
    """A record a relaunch needs is not where it should be; `cls` is NOT_REPLAYABLE or UNKNOWN."""

    def __init__(self, cls: str, reason: str):
        super().__init__(reason)
        self.cls, self.reason = cls, reason


# ---- the set ---------------------------------------------------------------------------------------------------------------------------

def members(st: LocalStore, name: str) -> list[str]:
    """Every target labelled `replay_set=name`, sorted."""
    return [r[0] for r in st.index.sql("SELECT DISTINCT target FROM labels WHERE key=? AND value=? ORDER BY target", (V.REPLAY_SET, name))]


def attempts_of(st: LocalStore, target: str) -> tuple[list[str], str | None]:
    """(run ids, None) the target stands for, or ([], why none)."""
    if ids.is_run_id(target):
        return [target], None
    if ids.is_art(target):
        runs = st.index.attempts_by_output(target)
        return (runs, None) if runs else ([], "no attempt in this store's catalog produced it: an artifact put by hand is kept, not "
                                                 "replayed (if one did, `research data reindex --remote`)")
    if target.startswith("drv:"):
        runs = [d["id"] for d in st.index.attempts_by_derivation(target)]
        return (runs, None) if runs else ([], "no attempt of this derivation in this store's catalog")
    return [], f"{target!r} is neither a run, an artifact nor a derivation"


def _attempt(st: LocalStore, run_id: str, fetch: bool) -> dict[str, Any]:
    p = st.attempt_path(run_id)
    if p.exists():
        return json.loads(p.read_bytes())
    doc = st.get_attempt(run_id) if fetch else None
    if doc is None:
        raise Missing(NOT_REPLAYABLE if fetch else UNKNOWN,
                      "its attempt is not in the store" if fetch else "its attempt is not in the local store (--fetch)")
    return doc


# ---- its records -----------------------------------------------------------------------------------------------------------------------

def _json(data: bytes, what: str) -> dict[str, Any]:
    try:
        doc = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        raise Missing(NOT_REPLAYABLE, f"its {what} is not JSON") from None
    if not isinstance(doc, dict):
        raise Missing(NOT_REPLAYABLE, f"its {what} is not a JSON object")
    return doc


def _from_run_record(st: LocalStore, art: str, fetch: bool) -> Records:
    hex_ = ids.hex_of(art)
    if not st.manifest_path(hex_).exists() and not fetch:
        raise Missing(UNKNOWN, f"its run record {art} is not in the local store (--fetch)")
    try:
        m = st.get_manifest(art)
    except NotFound:
        raise Missing(NOT_REPLAYABLE, f"its run record {art} is not in the store") from None
    files = {f.path: f for f in (m.payload.files or [])} if m.payload and m.payload.type == "tree" else {}
    docs = {}
    for name in ("launch.json", "job.json"):
        f = files.get(name)
        if f is None:
            raise Missing(NOT_REPLAYABLE, f"its run record {art} holds no {name}"
                          + (" (the run was not launched with --on)" if name == "launch.json" else ""))
        blob = st.object_path(f.sha256)
        if not blob.exists():
            if not fetch:
                raise Missing(UNKNOWN, f"its run record's {name} ({art}) is not in the local store (--fetch)")
            blob = st.fetch(art, paths=[name]) / name
        docs[name] = _json(blob.read_bytes(), name)
    trimmed = [t.get("path") for t in (m.meta.get("trimmed") or []) if str(t.get("path", "")).startswith("inputs/")]
    if trimmed:
        raise Missing(NOT_REPLAYABLE, f"--send input(s) {', '.join(trimmed)} are not in the store (custody trimmed them from {art})")
    return Records(docs["launch.json"], docs["job.json"], sorted(p for p in files if p.startswith("inputs/")), art)


def _from_run_dir(d: Path) -> Records:
    docs = {}
    for name in ("launch.json", "job.json"):
        if not (d / name).is_file():
            raise Missing(NOT_REPLAYABLE, f"no run record in the store, and {d} holds no {name} (`research fetch {d.name}` from its machine, "
                                          f"then `research data custody {d.name} --publish`)")
        docs[name] = _json((d / name).read_bytes(), f"{d / name}")
    if not (d / "preserved.json").is_file():
        raise Missing(NOT_REPLAYABLE, f"no run record in the store, and {d} is not a full copy of its run dir, so whether it had --send "
                                      f"inputs is unknown (`research fetch {d.name} --all`, then `research data custody {d.name} --publish`)")
    sends = sorted(p.relative_to(d).as_posix() for p in (d / "inputs").rglob("*") if p.is_file()) if (d / "inputs").is_dir() else []
    return Records(docs["launch.json"], docs["job.json"], sends, str(d))


def records(st: LocalStore, runs_root: Path, attempt: dict[str, Any], fetch: bool) -> Records:
    """The attempt's launch.json, job.json and --send inputs: from its run record (`outputs.run_record`), else a full local copy of
    its run dir."""
    art = (attempt.get("outputs") or {}).get("run_record")
    if isinstance(art, str) and ids.is_art(art):
        return _from_run_record(st, art, fetch)
    return _from_run_dir(runs_root / attempt["id"])


# ---- the argv it ran under -------------------------------------------------------------------------------------------------------------

def runner_argv(argv: list[str]) -> tuple[list[str], list[str], list[str]]:
    """launch.json's argv -> (the queue's runner wrap, the `research run` options a relaunch passes on, the command after `--`).
    ValueError when it is not a runner's argv."""
    i = next((k for k in range(len(argv) - 2) if argv[k:k + 3] == ["-m", "research", "run"]), None)
    if i is None or i == 0:
        raise ValueError("launch.json's argv is not `python -m research run ...`")
    prefix, rest = argv[:i - 1], argv[i + 3:]
    if "disk-hold" in prefix:
        prefix = prefix[prefix.index("--") + 1:] if "--" in prefix else []
    if "--" not in rest:
        raise ValueError("launch.json's argv has no `--` before the command")
    k = rest.index("--")
    opts, command, out, j = rest[:k], rest[k + 1:], [], 0
    while j < len(opts):
        flag = opts[j].split("=", 1)[0]
        if flag in DROPPED:
            j += 1 + (DROPPED[flag] if "=" not in opts[j] else 0)
            continue
        out.append(opts[j])
        j += 1
    return prefix, out, command


def gpu_lease(command: list[str]) -> tuple[dict[str, Any] | None, list[str]]:
    """A command that starts with a GPU lease (`[env K=V...] gpu-lease N [--wait] [--max-min M] [--mem-gb G] [--preemptible] [--timed]
    [--on IDS] -- CMD`, cluster.submit.wrappers) -> (its arguments, CMD); else (None, command)."""
    k = 1 if command[:1] == ["env"] else 0
    while k < len(command) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", command[k]):
        k += 1
    if k >= len(command) or not GPU_LEASE.search(command[k]) or "--" not in command[k:]:
        return None, command
    end = command.index("--", k)
    args, lease = command[k + 1:end], {"preemptible": False, "timed": False}
    j = 0
    while j < len(args):
        a = args[j]
        if a in ("--max-min", "--mem-gb", "--on") and j + 1 < len(args):
            lease[a[2:].replace("-", "_")] = args[j + 1]
            j += 2
            continue
        if a in ("--preemptible", "--timed"):
            lease[a[2:]] = True
        elif a.isdigit() and "gpus" not in lease:
            lease["gpus"] = a
        j += 1
    return lease, command[end + 1:]


def queue_wrap(prefix: list[str]) -> dict[str, Any]:
    """The queue's runner wrap (cluster.submit.wrappers) -> {kind, mem_gb, cpus}: what the job was given."""
    out: dict[str, Any] = {}
    for tok in prefix:
        if tok.startswith("GPU_LEASE_WHO=") and ":" in tok:
            out["kind"] = tok.split("=", 1)[1].split(":", 1)[0] or None
        elif tok.startswith("MemoryMax="):
            out["mem_gb"] = tok.split("=", 1)[1].rstrip("GgB")
        elif tok.startswith("AllowedCPUs="):
            out["cpus"] = str(_count_cpus(tok.split("=", 1)[1]))
    return out


def _count_cpus(spec: str) -> int:
    n = 0
    for part in spec.split(","):
        a, _, b = part.partition("-")
        n += (int(b) - int(a) + 1) if b else 1
    return n


def _custody_ttl(opts: list[str]) -> list[str]:
    """`--custody-ttl` for a run whose --timeout outlives the default key (cli._custody_ttl_problem refuses it without one)."""
    t = remote.flag_value(opts, "--timeout")
    if t is None:
        return []
    h = remote.timeout_seconds(t) / 3600
    return [] if h <= DEFAULT_CUSTODY_TTL_H else ["--custody-ttl", f"{min(CUSTODY_TTL_CAP_H, math.ceil(h) + 1)}h"]


def secrets_needed(attempt: dict[str, Any], job: dict[str, Any] | None = None) -> list[str]:
    """What its record redacted: --env names, then the command's (job.json execution.command_redacted, else each redacted argv
    position)."""
    env = (attempt.get("params") or {}).get("env") or {}
    out = [f"--env {k}" for k, v in sorted(env.items()) if v == REDACTED]
    named = ((job or {}).get("execution") or {}).get("command_redacted") or []
    return out + ([str(x) for x in named] or [f"argv[{i}]" for i, t in enumerate(attempt.get("argv") or []) if isinstance(t, str) and REDACTED in t])


# ---- one attempt -----------------------------------------------------------------------------------------------------------------------

def plan(st: LocalStore, run_id: str, *, runs_root: Path, machines: dict[str, dict[str, Any]], on: str | None, source: str,
         fetch: bool = False, targets: list[str] | None = None) -> Plan:
    """Classify one attempt and, unless it is not replayable, build its relaunch: `research run --on` with the source swapped to
    `source` (a path, or a placeholder in a dry run), or for a GPU lease the `research run --queue` an owner would queue."""
    p = Plan(run=run_id, targets=targets or [run_id], cls=NOT_REPLAYABLE)
    try:
        a = _attempt(st, run_id, fetch)
        ex = a.get("execution") or {}
        p.commit = (a.get("source") or {}).get("commit")
        rem = ex.get("remote") or {}
        if not ex.get("remote"):
            raise Missing(NOT_REPLAYABLE, "it ran locally (no --on), so no launch.json records the flags it ran under")
        if not rem.get("source_sha"):
            raise Missing(NOT_REPLAYABLE, f"it shipped no source tree (no --source): it ran {rem.get('machine') or 'its machine'}'s own files, "
                                          f"which no commit replaces")
        if need := secrets_needed(a):
            raise Missing(NOT_REPLAYABLE, f"it needed secrets ({', '.join(need)} redacted from its record)")
        rec = records(st, runs_root, a, fetch)
        if need := secrets_needed(a, rec.job):
            raise Missing(NOT_REPLAYABLE, f"it needed secrets ({', '.join(need)} redacted from its record)")
        try:
            wrap, opts, command = runner_argv(list(rec.launch.get("argv") or []))
        except ValueError as e:
            raise Missing(NOT_REPLAYABLE, str(e)) from None
        project = ex.get("project") or (rec.job.get("ownership") or {}).get("project")
        p.machine = on or ex.get("machine") or rem.get("machine")
        if not p.machine:
            raise Missing(NOT_REPLAYABLE, "its record names no machine (pass --on MACHINE)")
        try:
            remote.resolve_machine(p.machine, project, machines)
        except remote.Refused as e:
            raise Missing(NOT_REPLAYABLE, f"{'--on ' if on else 'its machine '}{e}") from None
    except Missing as e:
        p.cls, p.reason = e.cls, e.reason
        return p
    env = {**{k: v for k, v in ((a.get("params") or {}).get("env") or {}).items()}, REPLAY_OF_ENV: run_id}
    fetch_dir = f"replay-inputs/{run_id}"
    if rec.sends:
        p.prepare.append(["research", "data", "fetch", rec.sends_from, "--path", "inputs", "--to", fetch_dir]
                         if ids.is_art(rec.sends_from) else ["cp", "-r", f"{rec.sends_from}/inputs", f"{fetch_dir}/"])
    sends = [x for s in rec.sends for x in ("--send", f"{fetch_dir}/{s}")]
    envs = [x for k, v in sorted(env.items()) for x in ("--env", f"{k}={v}")]
    lease, bare = gpu_lease(command)
    queued = queue_wrap(wrap) if wrap else None
    if lease is not None or any(GPU_LEASE.search(t) for t in command):
        p.cls = NEEDS_WINDOW
        if lease is None:
            p.reason = (f"its command holds a GPU lease inside it on {p.machine}: queue it in a booked window (the command below is "
                        f"its --on relaunch, lease included)")
        else:
            p.reason = f"it held a GPU lease ({lease.get('gpus', '?')} GPU) on {p.machine}: queue it in a booked window"
            q = queued or {}
            flags = ["--kind", q["kind"]] if q.get("kind") else []
            flags += ["--gpus", lease.get("gpus", "1")] + (["--max-min", lease["max_min"]] if lease.get("max_min") else [])
            flags += ["--mem-gb", q.get("mem_gb") or lease["mem_gb"]] if (q.get("mem_gb") or lease.get("mem_gb")) else []
            flags += ["--cpus", q["cpus"]] if q.get("cpus") else []
            flags += (["--preemptible"] if lease["preemptible"] else []) + (["--quiet"] if lease["timed"] else [])
            flags += ["--gpu-ids", lease["on"]] if lease.get("on") else []
            p.command = ["research", "run", "--queue", "--source", source, "--project", project, "--on", p.machine, *flags,
                         *envs, *sends, *_custody_ttl(opts), *opts, "--", *bare]
            return p
    else:
        p.cls = REPLAYABLE
    if queued is not None:
        flags = (["--kind", queued["kind"]] if queued.get("kind") else []) + (["--mem-gb", queued["mem_gb"]] if queued.get("mem_gb") else [])
        flags += ["--cpus", queued["cpus"]] if queued.get("cpus") else []
        p.command = ["research", "run", "--queue", "--source", source, "--project", project, "--on", p.machine, *flags]
    else:
        flags = (["--exclusive"] if ex.get("exclusive") else []) + (["--mem-gb", f"{ex['mem_gb_reserved']:g}"] if ex.get("mem_gb_reserved") else [])
        p.command = ["research", "run", "--on", p.machine, "--project", project, "--source", source, *flags]
    p.command += [*envs, *sends, *_custody_ttl(opts), *opts, "--", *command]
    return p


# ---- a set -----------------------------------------------------------------------------------------------------------------------------

def dry_run(st: LocalStore, name: str, *, runs_root: Path, machines: dict[str, dict[str, Any]], on: str | None = None,
            at: str | None = None, fetch: bool = False) -> dict[str, Any]:
    """Every member of the set, each attempt classified once (an attempt several members name is listed under all of them)."""
    source = f"<checkout of {at or 'REV'}>"
    order: dict[str, list[str]] = {}
    plans: list[Plan] = []
    targets = members(st, name)
    for target in targets:
        runs, why = attempts_of(st, target)
        if why:
            plans.append(Plan(run=None, targets=[target], cls=NOT_REPLAYABLE, reason=why))
        for r in runs:
            order.setdefault(r, []).append(target)
    for r, via in order.items():
        plans.append(plan(st, r, runs_root=runs_root, machines=machines, on=on, source=source, fetch=fetch, targets=via))
    plans.sort(key=lambda x: (CLASSES.index(x.cls), x.run or "", x.targets))
    return {"set": name, "at": at, "on": on, "members": len(targets), "attempts": [x.to_doc() for x in plans],
            "summary": {c: sum(1 for x in plans if x.cls == c) for c in CLASSES}}


def summary_line(rep: dict[str, Any]) -> str:
    c = rep["summary"]
    n = sum(1 for x in rep["attempts"] if x["run"])
    return (f"{rep['set']}: {rep['members']} member(s), {n} attempt(s): {c[REPLAYABLE]} replayable, {c[NEEDS_WINDOW]} needs window, "
            f"{c[NOT_REPLAYABLE]} not replayable" + (f", {c[UNKNOWN]} unknown (records not local: --fetch)" if c[UNKNOWN] else ""))


def format_report(rep: dict[str, Any]) -> list[str]:
    lines = []
    for x in rep["attempts"]:
        via = "" if x["targets"] == [x["run"]] else f"  (via {', '.join(x['targets'])})"
        commit = f"  {x['commit'][:12]}" if x.get("commit") else ""
        lines.append(f"  {x['run'] or '-':<24} {x['class']:<15}{x['machine'] or '-'}{commit}{via}")
        if x["reason"]:
            lines.append(f"      {x['reason']}")
        for c in x["prepare"]:
            lines.append(f"      $ {shlex.join(c)}")
        if x["command"]:
            lines.append(f"      $ {shlex.join(x['command'])}")
    return [*lines, summary_line(rep)]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research replay", description=__doc__.split("\n\n")[0])
    ap.add_argument("sets", nargs="+", metavar="SET", help="AREA/re-verify or AREA/re-run (the replay_set label's value)")
    ap.add_argument("--at", default=None, metavar="REV", help="the commit to replay at")
    ap.add_argument("--on", default=None, metavar="MACHINE", help="relaunch on this machine instead of each attempt's own")
    ap.add_argument("--dry-run", action="store_true", help="classify and print the relaunches; launch nothing")
    ap.add_argument("--fetch", action="store_true", help="read records the local store lacks from its remote")
    ap.add_argument("--runs-dir", default=None, help="the local runs root (default $RESEARCH_RUNS or ~/.research/runs)")
    ap.add_argument("--store", default=None, metavar="ROOT")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    key = V.VOCAB[V.REPLAY_SET]
    bad = [why for s in ns.sets if (why := key.check(s))]
    if bad:
        print(f"research replay: {'; '.join(bad)}", file=sys.stderr)
        return 2
    if not ns.dry_run:
        print("research replay: only --dry-run is built so far; the relaunch lands next", file=sys.stderr)
        return 2
    st = LocalStore(ns.store)
    machines = remote.load_machines()
    reps = [dry_run(st, s, runs_root=R.runs_root(ns.runs_dir), machines=machines, on=ns.on, at=ns.at, fetch=ns.fetch) for s in ns.sets]
    if ns.json:
        print(json.dumps(reps if len(reps) > 1 else reps[0], indent=2, sort_keys=True))
    else:
        for rep in reps:
            print(f"{rep['set']}" + (f" at {rep['at']}" if rep["at"] else "") + (f" on {rep['on']}" if rep["on"] else ""))
            print("\n".join(format_report(rep)))
    empty = [r["set"] for r in reps if not r["members"]]
    if empty:
        print(f"research replay: no target is labelled replay_set={', replay_set='.join(empty)}", file=sys.stderr)
    return 1 if empty else 0
