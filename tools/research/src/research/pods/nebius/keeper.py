#!/usr/bin/env python3
"""Standing campaign targets on a Nebius node: keep N chunks of a lane's template leased here until a deadline, so no agent VM has
to stay awake to top them up (stdlib only; /usr/local/bin/vy-keeper, pods/nebius/deploy.toml; `research keeper` from any VM).

    vy-keeper tick                         vy-keeper.timer's oneshot, every minute, as research
    vy-keeper set NAME < TARGET.json       write a target, or re-arm it (`research keeper set` sends it over ssh)
    vy-keeper stop NAME --by WHO [--reason TEXT] [--cancel]
    vy-keeper status [NAME] [--json]
    vy-keeper hold REQUEST.json            a chunk's own transient unit: start its runner, then wait for it to exit

A target is KEEPER_DIR/targets/<name>.json (research/keeper-target/v0): its owner (a lane), this node (KEEPER_NODE), `n`, `until`
(UTC, at most KEEPER_MAX_AHEAD_H, 48, after it was set), `gpus` (default 1), an optional `max_min` per chunk, `command` (an argv
in which `{seed}` stands for the chunk's seed) and `request`: the launch request `research run --on` stages (the shipped tool
snapshot and source tree, the runner's cwd, the env and the `research run` flags), which `research keeper set` builds. Its state,
KEEPER_DIR/state/<name>.json, holds the seed counter, the live chunks, the failures in a row and why it starts nothing; every
change is a line of KEEPER_DIR/events.jsonl. `set` keeps the counter (it only rises, to `seed_from` if that is higher) and clears
a stop, a pause and the failure count; a target another owner set is refused.

Each tick, under KEEPER_DIR/.lock (which set and stop take too), for each target:
  - reaps the chunks whose unit has ended, reading the run dir: `done` (exit 0) clears the failure count, `preempted` (143: a
    waiter took the GPU, or a cancel) frees the place, and anything else (`failed`, `max_min` (124), `refused`, `not-started`,
    `lost`: no final status) is a failure; KEEPER_MAX_FAILURES (3) in a row pause the target until it is set again;
  - starts nothing for a target that is malformed (skipped and logged, never guessed at), stopped, paused or past its `until`,
    nor while a timed window booked in KEEPER_WINDOWS (node 2's fill windows file) or the daily KEEPER_DAILY_HOLD (node 1's
    quiet hold, HH:MM-HH:MM UTC) is on, or would be reached by a chunk started now within its max_min; a holds file that
    exists but can't be read holds everything;
  - otherwise tops the live chunks (running, or waiting for their lease) up to `n`. Each takes the next seed, written to the
    state before the chunk starts, so a seed is never handed out twice, even across a crash. It runs as a recorded `research
    run` on this machine with the claim and runner `research run --on` gets (`research launch-request`), in a transient user
    unit keeper-<run id> so that it outlives the tick; its workload is `gpu-lease G --wait --preemptible [--max-min M] --
    COMMAND`, so it waits for its GPUs like any request and a waiter can always take them back. job.json's run.remote.keeper
    names the target, owner and seed; the workload sees RESEARCH_KEEPER_TARGET and RESEARCH_KEEPER_SEED, and its lease
    who=keeper:<owner>.
A unit systemd can't report on counts as live (never more than n). The keeper signals only its own chunks' workloads: `stop
--cancel` sends SIGTERM to each live chunk's process group, recorded in its run dir as a manual cancel, as `research cancel` does.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import secrets
import shlex
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SCHEMA = "research/keeper-target/v0"
REQUEST_SCHEMA = "research/request/v0.1"
DIR = Path(os.environ.get("KEEPER_DIR", "/workspace/research/keep"))
ROOT = Path(os.environ.get("KEEPER_ROOT", "/workspace/research"))
NODE = os.environ.get("KEEPER_NODE", "")
GPU_LEASE = shlex.split(os.environ.get("KEEPER_GPU_LEASE", "gpu-lease"))
WINDOWS = os.environ.get("KEEPER_WINDOWS", "")
DAILY_HOLD = os.environ.get("KEEPER_DAILY_HOLD", "")
MAX_FAILURES = int(os.environ.get("KEEPER_MAX_FAILURES", "3"))
MAX_N = int(os.environ.get("KEEPER_MAX_N", "8"))
MAX_AHEAD_H = float(os.environ.get("KEEPER_MAX_AHEAD_H", "48"))
HOLD_POLL_S = float(os.environ.get("KEEPER_HOLD_POLL_S", "5"))
NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,39}")
OWNER = re.compile(r"[@\w.:-]{1,64}")
LIVE = {"active", "activating", "deactivating", "reloading", "refreshing"}
ENDED = {"inactive", "failed"}

clock = time.time


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc(s: Any, what: str) -> float:
    """A UTC time (`2026-10-05T12:00Z`) in epoch seconds; ValueError for anything else, a time without its zone included."""
    try:
        t = datetime.fromisoformat(s)
    except (TypeError, ValueError):
        raise ValueError(f"{what} {s!r} is not a UTC time like 2026-10-05T12:00Z") from None
    if t.utcoffset() != timedelta(0):
        raise ValueError(f"{what} {s!r} is not in UTC (end it with Z)")
    return t.timestamp()


def whole(doc: dict, key: str, lo: int, hi: int, default: int | None = None) -> int | None:
    v = doc.get(key, default)
    if v is None and default is None:
        return None
    if not isinstance(v, int) or isinstance(v, bool) or not lo <= v <= hi:
        raise ValueError(f"{key} {v!r} is not a whole number from {lo} to {hi}")
    return v


def parse_target(doc: Any, name: str) -> dict[str, Any]:
    """The target, normalized, with `until_t`; ValueError naming the first thing wrong with it."""
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise ValueError(f"schema is not {SCHEMA}")
    if not NAME.fullmatch(name) or doc.get("name") != name:
        raise ValueError(f"name {doc.get('name')!r} is not its file's ({name}: lower case, digits and -, at most 40)")
    if not isinstance(doc.get("owner"), str) or not OWNER.fullmatch(doc["owner"]):
        raise ValueError(f"owner {doc.get('owner')!r} is not a lane or handle")
    if not NODE or doc.get("node") != NODE:
        raise ValueError(f"node {doc.get('node')!r} is not this node ({NODE or 'KEEPER_NODE unset'})")
    n = whole(doc, "n", 1, MAX_N)
    gpus = whole(doc, "gpus", 1, 8, default=1)
    max_min = whole(doc, "max_min", 1, int(MAX_AHEAD_H * 60))
    whole(doc, "seed_from", 0, 2**62, default=0)
    until, set_at = utc(doc.get("until"), "until"), utc(doc.get("set_utc"), "set_utc")
    if until - set_at > MAX_AHEAD_H * 3600:
        raise ValueError(f"until {doc['until']} is more than {MAX_AHEAD_H:g} h after it was set ({doc['set_utc']})")
    cmd = doc.get("command")
    if not isinstance(cmd, list) or not cmd or not all(isinstance(a, str) for a in cmd):
        raise ValueError("command is not a list of strings")
    if not any("{seed}" in a for a in cmd):
        raise ValueError("command has no {seed}")
    req = doc.get("request")
    if not isinstance(req, dict) or req.get("schema") != REQUEST_SCHEMA:
        raise ValueError(f"request is not a {REQUEST_SCHEMA} request")
    if req.get("runs_root") != str(ROOT / "runs"):
        raise ValueError(f"request's runs_root {req.get('runs_root')!r} is not {ROOT / 'runs'}")
    if not isinstance(req.get("tool"), str) or Path(req["tool"]).parent != ROOT / "tool":
        raise ValueError(f"request's tool {req.get('tool')!r} is not a snapshot under {ROOT / 'tool'}")
    if not isinstance(req.get("python"), str) or not isinstance(req.get("env") or {}, dict) \
            or not isinstance(req.get("harness_args") or [], list) or not isinstance(req.get("remote") or {}, dict):
        raise ValueError("request's python, env, harness_args or remote is malformed")
    return {**doc, "n": n, "gpus": gpus, "max_min": max_min, "until_t": until}


# ---- files ------------------------------------------------------------------------------------------------------------------------------

def read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def write_json(p: Path, doc: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, p)


def target_path(name: str) -> Path:
    return DIR / "targets" / f"{name}.json"


def state_path(name: str) -> Path:
    return DIR / "state" / f"{name}.json"


def load_state(name: str) -> dict[str, Any]:
    """The target's state, a fresh one when it has none; OSError or ValueError when it exists but can't be trusted."""
    try:
        st = json.loads(state_path(name).read_text())
    except FileNotFoundError:
        return {"next_seed": 0, "chunks": {}, "failures": 0, "paused": None, "stopped": None, "note": None}
    if not isinstance(st, dict) or not isinstance(st.get("next_seed"), int) or not isinstance(st.get("chunks"), dict):
        raise ValueError(f"{state_path(name)} is not a keeper state")
    return st


def save_state(name: str, st: dict[str, Any]) -> None:
    write_json(state_path(name), st)


def log(target: str, event: str, **kv: Any) -> None:
    line = json.dumps({"t": iso(clock()), "target": target, "event": event, **kv}, sort_keys=True)
    DIR.mkdir(parents=True, exist_ok=True)
    with open(DIR / "events.jsonl", "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def note(st: dict[str, Any], name: str, event: str | None, why: str | None = None) -> None:
    """Why the target starts nothing, logged when it changes (a tick a minute would otherwise repeat it)."""
    text = f"{event}: {why}" if event else None
    if st.get("note") != text:
        st["note"] = text
        if event:
            log(name, event, why=why)


@contextmanager
def locked() -> Iterator[None]:
    DIR.mkdir(parents=True, exist_ok=True)
    with open(DIR / ".lock", "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


# ---- holds ------------------------------------------------------------------------------------------------------------------------------

def booked() -> list[tuple[float, float]]:
    """The timed windows in KEEPER_WINDOWS (fill_runner's file: `2026-10-01T10:00Z 30 # who`, start and minutes), none without
    the file; OSError when it exists but can't be read."""
    if not WINDOWS:
        return []
    try:
        lines = Path(WINDOWS).read_text().splitlines()
    except FileNotFoundError:
        return []
    out = []
    for line in lines:
        f = line.split("#")[0].split()
        try:
            t = datetime.fromisoformat(f[0]).timestamp()
            out.append((t, t + 60 * float(f[1] if len(f) > 1 else 30)))
        except (IndexError, ValueError):
            continue
    return out


def daily(spec: str, t: float) -> list[tuple[float, float]]:
    """KEEPER_DAILY_HOLD (`12:10-13:30`, UTC) from the day before `t` to three days after it."""
    if not spec:
        return []
    m = re.fullmatch(r"(\d\d):(\d\d)-(\d\d):(\d\d)", spec.strip())
    if not m:
        raise ValueError(f"KEEPER_DAILY_HOLD {spec!r} is not HH:MM-HH:MM")
    h0, m0, h1, m1 = map(int, m.groups())
    day = datetime.fromtimestamp(t, timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    out = []
    for d in range(-1, 4):
        s, e = day + timedelta(days=d, hours=h0, minutes=m0), day + timedelta(days=d, hours=h1, minutes=m1)
        out.append((s.timestamp(), (e if e > s else e + timedelta(days=1)).timestamp()))
    return out


def held(max_min: int | None, now: float) -> str | None:
    """Why no chunk may start now, else None: a booked window or the daily hold is on, or one would start within max_min."""
    try:
        spans = [("window", s, e) for s, e in booked()] + [("daily hold", s, e) for s, e in daily(DAILY_HOLD, now)]
    except (OSError, ValueError) as e:
        return f"the holds can't be read ({e})"
    for kind, s, e in sorted(spans, key=lambda x: x[1]):
        if s <= now < e:
            return f"in the {kind} {iso(s)}-{iso(e)}"
        if max_min and now < s < now + 60 * max_min:
            return f"a chunk of max_min {max_min} would reach the {kind} at {iso(s)}"
    return None


# ---- chunks: the systemd and research seams the tests replace ---------------------------------------------------------------------------

def unit_live(unit: str) -> bool | None:
    """True while the chunk's unit runs, False once it has ended (inactive, failed, or collected), None when systemd can't say."""
    r = subprocess.run(["systemctl", "--user", "show", "-p", "ActiveState", "--value", f"{unit}.service"],
                       capture_output=True, text=True)
    state = r.stdout.strip()
    if r.returncode == 0 and state in LIVE:
        return True
    if r.returncode == 0 and state in ENDED:
        return False
    return None


def launch(unit: str, request: Path, description: str) -> str | None:
    """Start the chunk's transient user unit, which runs `hold` (the user manager forks it, so it outlives the tick's service);
    None once started, else why not."""
    r = subprocess.run(["systemd-run", "--user", "--quiet", "--collect", "--unit", unit, "--description", description, "--",
                        sys.executable, str(Path(__file__).resolve()), "hold", str(request)], capture_output=True, text=True)
    return None if r.returncode == 0 else (r.stderr.strip() or f"systemd-run exited {r.returncode}")[-300:]


def cancel_chunk(chunk: dict[str, Any], reason: str) -> str | None:
    """SIGTERM to a live chunk's workload process group, recorded in its run dir as a manual cancel (`research cancel`'s
    cancel-intent, from the chunk's own tool snapshot); None once delivered, else why not."""
    rd = Path(chunk["run_dir"])
    st = read_json(rd / "status.json") or {}
    pgid = st.get("pgid")
    if st.get("state") != "running" or not isinstance(pgid, int) or pgid <= 1:
        return f"no running workload to signal (state {st.get('state')})"
    req = read_json(ROOT / "requests" / rd.name / "request.json") or {}
    if not req.get("tool"):
        return "its request names no tool snapshot"
    r = subprocess.run(["env", f"PYTHONPATH={req['tool']}", req.get("python") or "python3", "-m", "research.telemetry", "cancel-intent",
                        "--actor", "manual", "--reason", reason, "--target-pgid", str(pgid), "--expect-nonempty", "--attempt-dir", str(rd),
                        "--deliver"], capture_output=True, text=True)
    return None if r.returncode == 0 else (r.stderr.strip() or f"cancel-intent exited {r.returncode}")[-300:]


def hold(request: Path) -> int:
    """A chunk's unit: `research launch-request` claims the run dir and starts the detached runner, then the unit waits for the
    runner to exit (`tail --pid`, so no process holds this file open for the chunk's life and `research deploy drift` asks for
    no restart), so it is live exactly as long as the chunk."""
    req = json.loads(request.read_text())
    ident = (req.get("remote") or {}).get("tool_identity")
    r = subprocess.run([req["python"], "-m", "research", "launch-request", *(["--expect-tool", ident] if ident else []), str(request)],
                       env={**os.environ, "PYTHONPATH": req["tool"]}, capture_output=True, text=True)
    sys.stderr.write(r.stderr[-4000:])
    print(r.stdout.strip()[-2000:], flush=True)
    try:
        out = json.loads(r.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return r.returncode or 3
    if r.returncode or not out.get("started"):
        return r.returncode or 3
    os.execvp("tail", ["tail", f"--pid={int(out['runner_pid'])}", "-s", f"{HOLD_POLL_S:g}", "-f", "/dev/null"])


# ---- the tick ---------------------------------------------------------------------------------------------------------------------------

def outcome(run_dir: Path) -> tuple[str, int | None, str | None]:
    """How an ended chunk ended, from its run dir: (outcome, rc, why)."""
    refused = read_json(run_dir / "refused.json")
    if isinstance(refused, dict):
        return "refused", None, str(refused.get("reason"))[:300]
    if not run_dir.is_dir():
        return "not-started", None, "no run dir: launch-request never claimed it (the unit's journal says why)"
    st = read_json(run_dir / "status.json") or {}
    if st.get("state") not in ("done", "failed", "cancelled"):
        return "lost", None, f"its unit ended with the run {st.get('state') or 'not started'}"
    rc = st.get("rc")
    if st["state"] == "done" and rc == 0:
        return "done", 0, None
    if rc == 143:
        return "preempted", rc, None
    return ("max_min" if rc == 124 else "failed"), rc, None


def fail(name: str, st: dict[str, Any], why: str) -> None:
    st["failures"] = st.get("failures", 0) + 1
    if st["failures"] >= MAX_FAILURES and not st.get("paused"):
        st["paused"] = f"{st['failures']} failures in a row, the last: {why}"
        log(name, "paused", why=st["paused"])


def reap(name: str, st: dict[str, Any]) -> None:
    for rid, c in sorted(st["chunks"].items()):
        live = unit_live(c["unit"])
        if live is None:
            print(f"{name}: {c['unit']}: systemd can't say whether it runs; counted live", flush=True)
            continue
        if live:
            continue
        del st["chunks"][rid]
        if not c.get("launched"):
            log(name, "abandoned", seed=c["seed"], run=rid, why="the keeper stopped between taking the seed and starting the unit")
            continue
        out, rc, why = outcome(Path(c["run_dir"]))
        log(name, "exit", seed=c["seed"], run=rid, outcome=out, rc=rc, **({"why": why} if why else {}))
        if out == "done":
            st["failures"] = 0
        elif out != "preempted":
            fail(name, st, f"seed {c['seed']} ({rid}) {out}" + (f" rc={rc}" if rc is not None else ""))


def new_run_id(now: float) -> str:
    """`research run`'s id format (runs.new_run_id), unused under the runs root and the requests."""
    while True:
        rid = f"r{datetime.fromtimestamp(now, timezone.utc):%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
        if not (ROOT / "runs" / rid).exists() and not (ROOT / "requests" / rid).exists():
            return rid


def chunk_request(name: str, t: dict[str, Any], rid: str, seed: int, unit: str) -> dict[str, Any]:
    req = json.loads(json.dumps(t["request"]))
    lease = [*GPU_LEASE, str(t["gpus"]), "--wait", "--preemptible", *(["--max-min", str(t["max_min"])] if t["max_min"] else []), "--"]
    req.update(run_id=rid, runs_root=str(ROOT / "runs"), command=[*lease, *(a.replace("{seed}", str(seed)) for a in t["command"])],
               env={**(req.get("env") or {}), "GPU_LEASE_WHO": f"keeper:{t['owner']}", "RESEARCH_KEEPER_TARGET": name,
                    "RESEARCH_KEEPER_SEED": str(seed)},
               remote={**(req.get("remote") or {}), "keeper": {"target": name, "owner": t["owner"], "seed": seed, "node": NODE, "unit": unit}})
    return req


def start(name: str, t: dict[str, Any], st: dict[str, Any], now: float) -> bool:
    """Start one chunk with the next seed; False when it didn't start (a failure)."""
    seed = st["next_seed"]
    rid = new_run_id(now)
    unit = f"keeper-{rid}"
    request = ROOT / "requests" / rid / "request.json"
    write_json(request, chunk_request(name, t, rid, seed, unit))
    st["next_seed"] = seed + 1
    st["chunks"][rid] = {"seed": seed, "unit": unit, "run_dir": str(ROOT / "runs" / rid), "started": iso(now), "launched": False}
    save_state(name, st)
    why = launch(unit, request, f"keeper {name} seed {seed} ({t['owner']})")
    if why:
        del st["chunks"][rid]
        log(name, "launch-failed", seed=seed, run=rid, why=why)
        fail(name, st, f"seed {seed} ({rid}) did not start: {why}")
        save_state(name, st)
        return False
    st["chunks"][rid]["launched"] = True
    log(name, "start", seed=seed, run=rid, unit=unit)
    save_state(name, st)
    return True


def tick_target(name: str, now: float) -> None:
    try:
        st = load_state(name)
    except (OSError, ValueError) as e:
        log(name, "skipped", why=f"its state can't be read ({e}): nothing reaped or started")
        return
    reap(name, st)
    try:
        t = parse_target(json.loads(target_path(name).read_text()), name)
    except (OSError, ValueError) as e:
        note(st, name, "skipped", f"the target can't be used: {e}")
        save_state(name, st)
        return
    stopped, paused = st.get("stopped"), st.get("paused")
    why = (f"stopped by {stopped.get('by')} ({stopped.get('reason')})" if stopped else f"paused: {paused}" if paused
           else f"past its until {t['until']}" if now >= t["until_t"] else held(t["max_min"], now))
    if why:
        note(st, name, "idle", why)
    else:
        note(st, name, None)
        while len(st["chunks"]) < t["n"] and start(name, t, st, now):
            pass
    save_state(name, st)


def tick() -> int:
    now = clock()
    with locked():
        for p in sorted((DIR / "targets").glob("*.json")):
            try:
                tick_target(p.stem, now)
            except Exception as e:  # noqa: BLE001  (one target's fault starts nothing for it and stops no other)
                log(p.stem, "error", why=f"{type(e).__name__}: {e}")
    return 0


# ---- set, stop, status ------------------------------------------------------------------------------------------------------------------

def set_target(name: str, doc: Any) -> dict[str, Any]:
    now = clock()
    if not isinstance(doc, dict):
        raise ValueError("the target is not a JSON object")
    doc = {**doc, "schema": SCHEMA, "name": name, "set_utc": iso(now)}
    t = parse_target(doc, name)
    if t["until_t"] <= now:
        raise ValueError(f"until {doc['until']} has passed")
    with locked():
        old = read_json(target_path(name))
        if isinstance(old, dict) and old.get("owner") not in (None, doc["owner"]):
            raise ValueError(f"target {name} is {old['owner']}'s; pick another name")
        st = load_state(name)
        st["next_seed"] = max(st["next_seed"], int(doc.get("seed_from") or 0))
        st.update(failures=0, paused=None, stopped=None, note=None)
        write_json(target_path(name), doc)
        save_state(name, st)
        log(name, "set", owner=doc["owner"], n=t["n"], until=doc["until"], gpus=t["gpus"], max_min=t["max_min"], next_seed=st["next_seed"])
    return {"target": name, "owner": doc["owner"], "next_seed": st["next_seed"], "live": len(st["chunks"])}


def stop_target(name: str, by: str, reason: str, cancel: bool) -> dict[str, Any]:
    with locked():
        if not target_path(name).exists():
            raise ValueError(f"no target {name}")
        st = load_state(name)
        st["stopped"] = {"at": iso(clock()), "by": by, "reason": reason}
        log(name, "stop", by=by, reason=reason, cancel=cancel)
        cancels = {}
        for rid, c in sorted(st["chunks"].items()) if cancel else ():
            why = cancel_chunk(c, f"{reason} (vy-keeper stop {name} by {by})")
            cancels[rid] = why or "SIGTERM delivered"
            log(name, "cancel", seed=c["seed"], run=rid, result=cancels[rid])
        save_state(name, st)
    return {"target": name, "stopped": st["stopped"], "live": sorted(st["chunks"]), "cancels": cancels}


def status(name: str | None = None) -> list[dict[str, Any]]:
    out = []
    for p in sorted((DIR / "targets").glob("*.json")):
        if name and p.stem != name:
            continue
        doc = read_json(p)
        doc = doc if isinstance(doc, dict) else {}
        try:
            st = load_state(p.stem)
        except (OSError, ValueError) as e:
            st = {"note": f"state unreadable: {e}"}
        out.append({"name": p.stem, "owner": doc.get("owner"), "n": doc.get("n"), "until": doc.get("until"), "gpus": doc.get("gpus", 1),
                    "max_min": doc.get("max_min"), "next_seed": st.get("next_seed"), "failures": st.get("failures"),
                    "paused": st.get("paused"), "stopped": st.get("stopped"), "note": st.get("note"),
                    "live": [{"run": rid, "seed": c["seed"], "started": c.get("started")} for rid, c in sorted((st.get("chunks") or {}).items())]})
    if name and not out:
        raise ValueError(f"no target {name}")
    return out


def render(rows: list[dict[str, Any]]) -> str:
    lines = []
    for r in rows:
        lines.append(f"{r['name']}  owner={r['owner']}  live={len(r['live'])}/{r['n']}  until={r['until']}  gpus={r['gpus']}  "
                     f"max_min={r['max_min'] or '-'}  next_seed={r['next_seed']}  failures={r['failures']}"
                     + (f"  [{r['note']}]" if r.get("note") else ""))
        lines += [f"  {c['run']}  seed {c['seed']}  since {c['started']}" for c in r["live"]]
    return "\n".join(lines) or "no targets"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vy-keeper", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tick")
    sub.add_parser("set").add_argument("name")
    s = sub.add_parser("stop")
    s.add_argument("name")
    s.add_argument("--by", required=True)
    s.add_argument("--reason", default="stopped")
    s.add_argument("--cancel", action="store_true", help="also SIGTERM each live chunk's workload, recorded as a manual cancel")
    s = sub.add_parser("status")
    s.add_argument("name", nargs="?")
    s.add_argument("--json", action="store_true")
    sub.add_parser("hold").add_argument("request")
    a = ap.parse_args(argv)
    if a.cmd == "tick":
        return tick()
    if a.cmd == "hold":
        return hold(Path(a.request))
    try:
        if a.cmd == "set":
            try:
                doc = json.loads(sys.stdin.read())
            except ValueError as e:
                raise ValueError(f"the target on stdin is not JSON: {e}") from None
            out: Any = set_target(a.name, doc)
        elif a.cmd == "stop":
            out = stop_target(a.name, a.by, a.reason, a.cancel)
        else:
            out = status(a.name)
            if not a.json:
                print(render(out))
                return 0
    except (OSError, ValueError) as e:
        print(json.dumps({"refused": str(e)}))
        return 3
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
