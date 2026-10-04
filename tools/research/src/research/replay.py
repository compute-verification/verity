"""`research replay SET... --at REV [--on M] [--dry-run]`: a replay set's attempts launched again at a new commit, and judged.

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

Without --dry-run (`relaunch`) every replayable attempt is launched from a detached worktree of REV as an ordinary `research run`
under a run id of its own, with `RESEARCH_REPLAY_OF=<original run id>` as an `--env` (every tree's runner records it as
`params.env`, so the catalog finds a replay by it); a not replayable one is labelled `replay=not-replayable`. Then, as each
replay's attempt reaches the store, `compare.verdict` judges it against the set's attempts of the same derivation (the band's
originals), honouring the original's `replay_expect`, and the original is labelled `replay=VERDICT --ref <replay>`. `--judge`
does only that last step, for every replay in the catalog its original has no verdict for (a window's run an owner queued).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
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
    inputs: str | None = None   # where its --send inputs come from: the run record's art id, or a local run dir

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
         fetch: bool = False, targets: list[str] | None = None, inputs_dir: str = "replay-inputs") -> Plan:
    """Classify one attempt and, unless it is not replayable, build its relaunch: `research run --on` with the source swapped to
    `source` (a path, or a placeholder in a dry run), or for a GPU lease the `research run --queue` an owner would queue.  Its
    --send inputs are sent from `inputs_dir/<run id>/inputs/`."""
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
    fetch_dir = f"{inputs_dir}/{run_id}"
    if rec.sends:
        p.inputs = rec.sends_from
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
            at: str | None = None, fetch: bool = False, source: str | None = None, inputs_dir: str = "replay-inputs") -> dict[str, Any]:
    """Every member of the set, each attempt classified once (an attempt several members name is listed under all of them).  The
    relaunches name `source` as their tree (a dry run: a placeholder for REV's checkout)."""
    source = source or f"<checkout of {at or 'REV'}>"
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
        plans.append(plan(st, r, runs_root=runs_root, machines=machines, on=on, source=source, fetch=fetch, targets=via, inputs_dir=inputs_dir))
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


# ---- the relaunch ----------------------------------------------------------------------------------------------------------------------

class ReplayError(Exception):
    pass


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise ReplayError(f"git {' '.join(args)}: {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout.strip()


def checkout(repo: Path, rev: str, into: Path) -> str:
    """A detached worktree of `repo` at `rev` in `into`: the clean tree `research run --source` ships.  Returns its commit."""
    sha = _git(repo, "rev-parse", "--verify", f"{rev}^{{commit}}")
    _git(repo, "worktree", "add", "--detach", "--quiet", str(into), sha)
    return sha


def materialise(st: LocalStore, x: dict[str, Any], inputs_dir: Path) -> None:
    """Put an attempt's --send inputs where its relaunch sends them from (`inputs_dir/<run id>/inputs/`)."""
    dst = inputs_dir / x["run"]
    if ids.is_art(x["inputs"]):
        st.fetch(x["inputs"], to=dst, paths=["inputs"])
    else:
        shutil.copytree(Path(x["inputs"]) / "inputs", dst / "inputs")


#: run states (runs.summarize) after which nothing more is published, and those after which custody still publishes the attempt
NEVER_PUBLISHED = frozenset({"refused", "launcher-exited"})
ENDED = frozenset({"done", "failed", "cancelled"})
POLL_S = 60.0
#: how many polls an ended run's attempt gets to reach the store (custody uploads it after the workload exits)
PUBLISH_POLLS = 30


def observe(st: LocalStore, runs_root: Path, run_id: str) -> tuple[dict[str, Any] | None, str | None]:
    """(its attempt, None) once the store or its remote has it, else (None, the run's state from its machine's records)."""
    doc = st.get_attempt(run_id)
    if doc is not None:
        return doc, None
    try:
        remote.fetch(runs_root / run_id)
    except remote.Refused as e:
        return None, f"unobservable: {e}"
    return None, R.summarize(runs_root / run_id)["state"]


def judge(st: LocalStore, originals: list[str], replay: str, expect: str | None) -> dict[str, Any]:
    """`compare.verdict` of the run `replay` against the runs `originals`, honouring `expect` (a replay_expect value): its document
    (`verdict` one of same | within-band | differs | expected-change, and `reasons`)."""
    from . import compare as C

    return C.verdict([C.load(st, o) for o in originals], C.load(st, replay), expect).to_doc()


def expect_of(st: LocalStore, run_id: str) -> str | None:
    labs = st.labels(run_id, key=V.REPLAY_EXPECT)
    return str(labs[-1].value) if labs else None


def originals_of(st: LocalStore, run_id: str, runs: list[str]) -> list[str]:
    """The attempts among `runs` (the set's) that are the computation `run_id` is (its derivation): the originals a band comes from."""
    drv = (st.get_attempt(run_id) or {}).get("derivation")
    return [r for r in runs if r == run_id or (drv is not None and (st.get_attempt(r) or {}).get("derivation") == drv)]


def replays_of(st: LocalStore, run_id: str) -> list[str]:
    """The attempts in the catalog that recorded themselves as replays of `run_id` (`--env RESEARCH_REPLAY_OF=<run id>`)."""
    return [r[0] for r in st.index.sql("SELECT DISTINCT attempt FROM attempt_params WHERE key=? AND value=? ORDER BY attempt",
                                       (f"env.{REPLAY_OF_ENV}", run_id))]


def label(st: LocalStore, target: str, value: str, by: str, ref: str | None) -> int:
    """`research data label TARGET replay VALUE` (its vocabulary check and write-through), its output on stderr."""
    from .store.cli import main as data_main

    with contextlib.redirect_stdout(sys.stderr):
        return data_main(["label", target, V.REPLAY, value, "--by", by, *(["--ref", ref] if ref else []), "--store", str(st.root)])


def _judge_one(st: LocalStore, x: dict[str, Any], new: str, runs: list[str], by: str) -> None:
    try:
        x["verdict"] = judge(st, originals_of(st, x["run"], runs), new, expect_of(st, x["run"]))
    except (LookupError, ValueError) as e:
        x["outcome"] = f"not judged: {e}"
        return
    if label(st, x["run"], x["verdict"]["verdict"], by, new) != 0:
        x["outcome"] = f"judged {x['verdict']['verdict']}, but its label did not reach the store's remote (`research data push --pending`)"


def wait_and_judge(st: LocalStore, rep: dict[str, Any], by: str, *, observe_fn: Callable[[str], tuple[dict | None, str | None]],
                   sleep: Callable[[float], None], poll_s: float = POLL_S) -> None:
    """Poll every launched replay until its attempt is in the store, then label its original with the verdict."""
    runs = [x["run"] for x in rep["attempts"] if x["run"]]
    pending = {x["replay"]: x for x in rep["attempts"] if x.get("replay") and x.get("launch_rc") == 0}
    grace = dict.fromkeys(pending, 0)
    while pending:
        for new, x in list(pending.items()):
            doc, state = observe_fn(new)
            if doc is None and state in ENDED and grace[new] < PUBLISH_POLLS:
                grace[new] += 1
                continue
            if doc is None and (state in ENDED or state in NEVER_PUBLISHED):
                x["outcome"] = f"{new} ended ({state}) without an attempt in the store (`research inspect {new}`, `research data custody {new}`)"
            elif doc is None:
                continue
            else:
                _judge_one(st, x, new, runs, by)
            del pending[new]
        if pending:
            sleep(poll_s)


def relaunch(st: LocalStore, name: str, *, at: str, repo: Path, runs_root: Path, machines: dict[str, dict[str, Any]], by: str,
             on: str | None = None, wait: bool = True, launch: Callable[[list[str]], int] | None = None,
             observe_fn: Callable[[str], tuple[dict | None, str | None]] | None = None, sleep: Callable[[float], None] = time.sleep,
             poll_s: float = POLL_S) -> dict[str, Any]:
    """Launch every replayable attempt of the set at `at` as an ordinary `research run` (its own run id, `RESEARCH_REPLAY_OF` in
    its env), label each not replayable one `replay=not-replayable` (once per asserter), then, with `wait`, judge each replay
    once its attempt is in the store and label its original `replay=VERDICT --ref <replay>`."""
    if launch is None:
        from .cli import main as research_main

        launch = research_main
    with tempfile.TemporaryDirectory(prefix="research-replay-") as tmp:
        tree, inputs = Path(tmp) / "tree", Path(tmp) / "inputs"
        rep = dry_run(st, name, runs_root=runs_root, machines=machines, on=on, at=at, fetch=True, source=str(tree), inputs_dir=str(inputs))
        todo = [x for x in rep["attempts"] if x["class"] == REPLAYABLE]
        rep["commit"] = None
        if todo:
            rep["commit"] = checkout(repo, at, tree)
            try:
                for x in todo:
                    if x["inputs"]:
                        materialise(st, x, inputs)
                    new = R.new_run_id()
                    while (runs_root / new).exists():
                        new = R.new_run_id()
                    x["replay"] = new
                    x["launch_rc"] = launch([x["command"][1], "--id", new, *x["command"][2:]])
                    if x["launch_rc"] != 0:
                        x["outcome"] = f"its relaunch {new} was refused (exit {x['launch_rc']}; `research inspect {new}`)"
            finally:
                _git(repo, "worktree", "remove", "--force", str(tree))
    for x in rep["attempts"]:
        if x["run"] and x["class"] == NOT_REPLAYABLE:
            mine = [lab for lab in st.labels(x["run"], key=V.REPLAY) if lab.by == by]
            if not mine or mine[-1].value != "not-replayable":
                label(st, x["run"], "not-replayable", by, None)
    if wait:
        wait_and_judge(st, rep, by, observe_fn=observe_fn or (lambda r: observe(st, runs_root, r)), sleep=sleep, poll_s=poll_s)
    return rep


def judge_set(st: LocalStore, name: str, *, by: str) -> dict[str, Any]:
    """`--judge`: every replay the catalog holds of the set's attempts that its original carries no verdict for yet (one an owner
    queued for a window, or one a relaunch stopped waiting for), judged and labelled.  `research data reindex --remote` first
    brings in the attempts other machines published."""
    runs = list(dict.fromkeys(r for t in members(st, name) for r in attempts_of(st, t)[0]))
    rep: dict[str, Any] = {"set": name, "judged": []}
    for r in runs:
        done = {lab.ref for lab in st.labels(r, key=V.REPLAY)}
        for new in replays_of(st, r):
            if new not in done:
                x = {"run": r, "replay": new}
                _judge_one(st, x, new, runs, by)
                rep["judged"].append(x)
    return rep


def format_relaunch(rep: dict[str, Any]) -> list[str]:
    lines = []
    for x in rep.get("attempts") or rep.get("judged") or []:
        if not x.get("replay"):
            if x.get("class") in (NOT_REPLAYABLE, NEEDS_WINDOW):
                lines.append(f"  {x['run'] or ', '.join(x['targets']):<24} {x['class']}: {x['reason']}")
                if x["class"] == NEEDS_WINDOW and x["command"]:
                    lines.append(f"      $ {shlex.join(x['command'])}")
            continue
        v = x.get("verdict")
        lines.append(f"  {x['run']:<24} -> {x['replay']}  " + (v["verdict"] if v else x.get("outcome") or "launched"))
        lines += [f"      - {r}" for r in (v or {}).get("reasons") or []]
    return lines


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research replay", description=__doc__.split("\n\n")[0])
    ap.add_argument("sets", nargs="+", metavar="SET", help="AREA/re-verify or AREA/re-run (the replay_set label's value)")
    ap.add_argument("--at", default=None, metavar="REV", help="the commit to replay at (required but for --dry-run and --judge)")
    ap.add_argument("--on", default=None, metavar="MACHINE", help="relaunch on this machine instead of each attempt's own")
    ap.add_argument("--dry-run", action="store_true", help="classify and print the relaunches; launch nothing")
    ap.add_argument("--judge", action="store_true", help="launch nothing: judge and label the replays in the catalog that have no verdict")
    ap.add_argument("--no-wait", action="store_true", help="launch and return; `--judge` labels the verdicts later")
    ap.add_argument("--poll", type=float, default=POLL_S, metavar="S", help=f"seconds between looks at the launched runs (default {POLL_S:g})")
    ap.add_argument("--fetch", action="store_true", help="--dry-run: read records the local store lacks from its remote")
    ap.add_argument("--by", default=os.environ.get("RESEARCH_LANE") or os.environ.get("USER") or "-", help="who asserts the verdicts")
    ap.add_argument("--repo", default=".", metavar="DIR", help="the git checkout REV is resolved in (default: the current directory)")
    ap.add_argument("--runs-dir", default=None, help="the local runs root (default $RESEARCH_RUNS or ~/.research/runs)")
    ap.add_argument("--store", default=None, metavar="ROOT")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    key = V.VOCAB[V.REPLAY_SET]
    bad = [why for s in ns.sets if (why := key.check(s))]
    if bad:
        print(f"research replay: {'; '.join(bad)}", file=sys.stderr)
        return 2
    if not (ns.dry_run or ns.judge or ns.at):
        print("research replay: --at REV names the commit to replay at (or --dry-run, --judge)", file=sys.stderr)
        return 2
    st = LocalStore(ns.store)
    runs_root = R.runs_root(ns.runs_dir)
    if ns.judge:
        reps = [judge_set(st, s, by=ns.by) for s in ns.sets]
    elif ns.dry_run:
        machines = remote.load_machines()
        reps = [dry_run(st, s, runs_root=runs_root, machines=machines, on=ns.on, at=ns.at, fetch=ns.fetch) for s in ns.sets]
    else:
        machines = remote.load_machines()
        try:
            reps = [relaunch(st, s, at=ns.at, repo=Path(ns.repo).expanduser(), runs_root=runs_root, machines=machines, by=ns.by, on=ns.on,
                             wait=not ns.no_wait, poll_s=ns.poll) for s in ns.sets]
        except ReplayError as e:
            print(f"research replay: {e}", file=sys.stderr)
            return 2
    if ns.json:
        print(json.dumps(reps if len(reps) > 1 else reps[0], indent=2, sort_keys=True))
    else:
        for rep in reps:
            print(f"{rep['set']}" + (f" at {rep['at']}" if rep.get("at") else "") + (f" ({rep['commit'][:12]})" if rep.get("commit") else "")
                  + (f" on {rep['on']}" if rep.get("on") else ""))
            print("\n".join(format_report(rep) if ns.dry_run else format_relaunch(rep)))
            if not (ns.dry_run or ns.judge):
                print(summary_line(rep))
    empty = [r["set"] for r in reps if "members" in r and not r["members"]]
    if empty:
        print(f"research replay: no target is labelled replay_set={', replay_set='.join(empty)}", file=sys.stderr)
    failed = [x for r in reps for x in r.get("attempts") or r.get("judged") or [] if x.get("outcome")]
    return 1 if empty or failed else 0
