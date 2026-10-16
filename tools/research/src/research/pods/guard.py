"""`research pods guard`: the one budgets guard.  It enforces every line of budgets.toml (research.pods.budgets), re-read each poll.

    research pods guard [--budgets NOTES|FILE] [--poll S] [--dry-run] [--log F] [--self-pod ID] [--detach]
    research pods guard status [--dry-run] [--json]
    research pods guard stop [--dry-run]

Scope.  Live pods (desiredStatus not EXITED/TERMINATED) whose name a budget line covers or a `[guard] project` prefix starts.  The
guard's own host and `[guard] exempt` names are never counted or terminated; every other pod on the account is invisible to it.

Every `--poll` seconds (default 60), from the budgets in memory (the last good read):
  tally      each pod's costPerHr x the time since the last poll (RunPod bills a running pod whether or not anything uses it), added
             to the pod and to the line that covers it.  A line's tally starts when the guard first sees the line and lives in the
             state file, so a restart continues it.
  cap        a line whose spend reaches its cap_usd trips: the pods under it then are terminated.  The trip holds until cap_usd is raised
             above the spend; pods that appear under a tripped line are counted, not terminated (`pods create` refuses them).
  per day    the same for cap_usd_per_day over the rolling 24 hours (spend kept in BUCKET_S buckets in the state); that trip clears by
             itself once the window's spend falls back under the cap.
  balance    the account's clientBalance (GraphQL; account-wide) at or under `[guard] balance_floor`: every pod in scope is terminated,
             each poll while it holds.
  max age    a pod under a line older than the line's max_pod_hours (RunPod's createdAt) is terminated: the bound for a pod whose lease
             never got armed.
  uncovered  a project pod no line covers is terminated once it has been uncovered for `uncovered_grace_min` (Daniel, 2026-09-29), or
             only reported with `terminate_uncovered = false`.
The guard never trips on the clock: each pod's lease (research.pods.lease) ends it on time.  A pod the guard terminated in the last
TERMINATING_S seconds is not terminated again (the API lists it for a while after DELETE) but still counts toward spend.  Each
termination is one log line and an entry in the state's `terminated`.  `--dry-run` logs WOULD-TERMINATE, calls no DELETE, and keeps
its own files.  After acting, the guard records, then reads budgets.toml again for the next poll (a notes clone: from its origin).

Fail closed.  A fault on the guard's own machine never stops enforcement and never loosens a limit (2026-09-28 18:10 PT: the control
pod's disk filled, every guard died on its state write, and nothing enforced any cap for 15 minutes).
  * Each poll decides and terminates from memory first, then records.  Every log, state and pid write is best effort: a failure is
    kept in memory (`faults`), printed to stderr when it starts and when it clears, and written into the state once a write succeeds.
  * A read that fails keeps the last value read: budgets.toml that cannot be fetched, parsed or validated keeps the last good budgets
    (the state carries their text, so a restart keeps them too); an unreadable balance keeps the last reading.  A state file that does
    not parse is set aside (`.corrupt-<epoch>`) and new tallies start, recorded in the state.
  * `--detach` starts a supervisor that runs the guard and starts it again one poll after any exit but a stop (0), bad arguments (2)
    or another live guard (3).  A guard with no RunPod key refuses to loop (EXIT_NO_KEY, 4); supervised, it is tried again each poll,
    so a key file that comes back brings it back.
  * `research pods create` refuses while the guard has not completed a poll in budgets.STALE_S, or while its last read of
    budgets.toml failed (research.pods.budgets.gate): nothing new starts unguarded or on budgets the guard can't refresh.

Files: $RESEARCH_GUARD_DIR (default ~/.research/pods)/guard-budgets[.dry-run].{json,log,pid}.  One guard per machine: a start refuses
(exit 3) while the recorded pid is alive.  `stop` signals exactly the recorded pid.  The guard's own host is `--self-pod ID`, else
RUNPOD_POD_ID (from /proc/1/environ when an ssh session did not inherit it), else none (a laptop).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import budgets as B

DEAD = ("EXITED", "TERMINATED")
TERMINATING_S = 600
KEEP_TERMINATED = 500
STOP_CODES = (0, 2, 3)          # the guard's exits the supervisor does not answer with a restart: stopped, bad arguments, another guard
EXIT_NO_KEY = 4                 # no RunPod API key here: the guard refuses to loop (it could terminate nothing); the supervisor retries
SCHEMA = "research/budgets-guard/v1"
WINDOW_S = 86400                # cap_usd_per_day's rolling window
BUCKET_S = 300                  # its spend is kept in buckets this long; a bucket that straddles the window's edge counts whole
SUPERVISED_ENV = "RESEARCH_GUARD_SUPERVISED"


class RunPodAPI:
    """The three provider calls the guard makes; tests substitute an object with the same methods."""

    def pods(self) -> list[dict[str, Any]]:
        from . import runpod
        return runpod._request("GET", "/pods") or []

    def terminate(self, pod_id: str) -> None:
        from . import runpod
        runpod._request("DELETE", f"/pods/{pod_id}")

    def balance(self) -> float:
        from . import runpod
        return float(runpod._graphql("{ myself { clientBalance } }")["data"]["myself"]["clientBalance"])


@dataclass
class Config:
    source: Path
    dry_run: bool = False
    self_pod: str | None = None


def guard_dir() -> Path:
    return Path(os.environ.get("RESEARCH_GUARD_DIR", Path.home() / ".research" / "pods"))


def paths(dry_run: bool = False, log: str | None = None) -> dict[str, Path]:
    stem = "guard-budgets" + (".dry-run" if dry_run else "")
    d = guard_dir()
    return {"state": d / f"{stem}.json", "log": Path(log) if log else d / f"{stem}.log", "pid": d / f"{stem}.pid"}


def parse_time(s: str | None) -> float | None:
    """Epoch seconds from ISO 8601 or RunPod's `2026-09-23 20:49:55.033 +0000 UTC`; None when absent or unparseable."""
    if not s:
        return None
    t = str(s).strip().removesuffix(" UTC")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f %z", "%Y-%m-%d %H:%M:%S %z"):
        try:
            return dt.datetime.strptime(t, fmt).timestamp()
        except ValueError:
            pass
    return B.parse_time(t)


iso = B.iso


def self_pod_id() -> str | None:
    if os.environ.get("RUNPOD_POD_ID"):
        return os.environ["RUNPOD_POD_ID"]
    try:
        for item in Path("/proc/1/environ").read_bytes().split(b"\0"):
            if item.startswith(b"RUNPOD_POD_ID="):
                return item.split(b"=", 1)[1].decode() or None
    except OSError:
        pass
    return None


def new_state(now: float) -> dict[str, Any]:
    return {"schema": SCHEMA, "started": now, "last_poll": None, "last_ok_poll": None, "budgets": None, "lines": {}, "pods": {},
            "rate_now": 0.0, "balance": None, "terminated": [], "terminating": {}}


def load_state(path: Path, now: float, log: Callable[[str], None]) -> dict[str, Any]:
    """The tallies in `path`, else new ones.  A file that cannot be read or parsed does not stop the guard: it is set aside and new
    tallies start, with `tally_restarted` saying why (the spend before it is not counted)."""
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return new_state(now)
    except OSError as err:
        why = f"unreadable ({type(err).__name__}: {err})"
    else:
        try:
            state = json.loads(raw)
            if isinstance(state, dict) and state.get("schema") == SCHEMA and isinstance(state.get("lines"), dict) \
                    and isinstance(state.get("pods"), dict):
                return state
            why = "not a budgets guard state"
        except ValueError as err:
            why = f"does not parse ({err})"
    aside = path.with_name(f"{path.name}.corrupt-{int(now)}")
    try:
        path.replace(aside)
    except OSError:
        aside = path
    log(f"STATE {path.name} {why}: kept as {aside.name}; new tallies start now")
    state = new_state(now)
    state["tally_restarted"] = {"at": iso(now), "why": why, "kept": str(aside)}
    return state


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w") as f:
            f.write(json.dumps(state, indent=1, sort_keys=True))
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _append(path: Path, text: str) -> None:
    with path.open("a") as f:
        f.write(text)


class Recorder:
    """The guard's writes: its log, state and pid files, all best effort.  A failure is kept in `faults` (per file kind: since, count,
    last error), printed to stderr when it starts and when it clears, and never raised, so it can neither end the loop nor delay a
    termination.  `clock` stamps the log lines."""

    def __init__(self, files: dict[str, Path], clock: Callable[[], float]):
        self.files, self.clock = files, clock
        self.faults: dict[str, dict[str, Any]] = {}

    def _say(self, msg: str) -> None:
        try:
            print(f"research pods guard: {msg}", file=sys.stderr, flush=True)
        except Exception:  # noqa: BLE001 -- stderr may be the log on the same full disk
            pass

    def attempt(self, what: str, fn: Callable[[], None]) -> bool:
        try:
            fn()
        except Exception as err:  # noqa: BLE001 -- no local fault may stop enforcement
            e = f"{type(err).__name__}: {err}"
            f = self.faults.get(what)
            if f is None:
                self.faults[what] = {"since": iso(self.clock()), "count": 1, "error": e}
                self._say(f"{what} write failed ({e}); enforcing from memory")
            else:
                f["count"] += 1
                f["error"] = e
            return False
        if what in self.faults:
            f = self.faults.pop(what)
            self._say(f"{what} writes work again after {f['count']} failure(s) since {f['since']} ({f['error']})")
        return True

    def log(self, msg: str) -> None:
        line = f"{iso(self.clock())} {msg}\n"
        self.attempt("log", lambda: _append(self.files["log"], line))

    def save(self, state: dict[str, Any]) -> None:
        state["faults"] = {k: dict(v) for k, v in self.faults.items()}
        self.attempt("state", lambda: save_state(self.files["state"], state))


# ---- the budgets in force -------------------------------------------------------------------------------------------------------------------

def refresh(cfg: Config, state: dict[str, Any], current: B.Budgets | None, now: float, log: Callable[[str], None]) -> B.Budgets | None:
    """Read budgets.toml again.  Success replaces the budgets in force (the state keeps their text and view, each change is logged);
    any failure keeps them and records the error, logged once per distinct error."""
    b = state.get("budgets") or {}
    try:
        text, where = B.read(cfg.source)
        new = B.parse(text)
    except Exception as err:  # noqa: BLE001 -- git, the network, the file, the parse: all keep the last good budgets
        e = f"{type(err).__name__}: {err}"
        if b.get("error") != e:
            log(f"BUDGETS unreadable ({e}): keeping " + (f"the budgets read {b.get('read_utc')} from {b.get('source')}" if current else "none: nothing is enforced yet"))
        state["budgets"] = {**b, "error": e, "error_utc": iso(now)}
        return current
    old = {p: ln.view() for p, ln in (current.lines.items() if current else [])}
    for p, v in sorted(new.view()["lines"].items()):
        if p not in old:
            caps = [f"{k} ${v[k]:g}" for k in B.CAP_KEYS if v.get(k) is not None]
            log(f"LINE {p!r} added: {', '.join(caps)}, expires {v['expires']}, max_pod_hours {v['max_pod_hours']:g}" + (f", by {v['by']}" if v.get("by") else ""))
        elif old[p] != v:
            log(f"LINE {p!r} changed: " + ", ".join(f"{k} {old[p][k]} -> {v[k]}" for k in v if old[p].get(k) != v[k]))
    for p in sorted(set(old) - set(new.lines)):
        log(f"LINE {p!r} removed: its pods are uncovered now")
    if b.get("error"):
        log(f"BUDGETS readable again ({where})")
    state["budgets"] = {**new.view(), "source": where, "read_utc": iso(now), "error": None, "text": text}
    return new


def budgets_from_state(state: dict[str, Any]) -> B.Budgets | None:
    try:
        return B.parse((state.get("budgets") or {})["text"])
    except (KeyError, TypeError, B.BudgetError):
        return None


# ---- one poll ---------------------------------------------------------------------------------------------------------------------------------

def poll(api: Any, budgets: B.Budgets, cfg: Config, state: dict[str, Any], now: float, log: Callable[[str], None]) -> list[dict[str, Any]]:
    """One guard step over the provider's current pod list: tally, decide, terminate, and only then log.  Returns the terminations it
    made (or would make)."""
    listed = [p for p in api.pods() if (p.get("desiredStatus") or "").upper() not in DEAD]
    hours = (now - state["last_poll"]) / 3600.0 if state.get("last_poll") else 0.0
    state["terminating"] = {k: t for k, t in (state.get("terminating") or {}).items() if now - t < TERMINATING_S}
    notes: list[str] = []
    scope: list[tuple[dict[str, Any], B.Line | None]] = []
    for p in listed:
        name = str(p.get("name") or "")
        if p["id"] == cfg.self_pod or budgets.is_exempt(name):
            continue
        line = budgets.covering(name)
        if line is not None or budgets.is_project(name):
            scope.append((p, line))
    for prefix in budgets.lines:
        state["lines"].setdefault(prefix, {"spent": 0.0, "first_seen": now, "tripped": None})
    bucket = str(int(now // BUCKET_S) * BUCKET_S)
    rate_now = 0.0
    for p, line in scope:
        r = float(p.get("costPerHr") or 0.0)
        rate_now += r
        e = state["pods"].setdefault(p["id"], {"spent": 0.0, "first_seen": now})
        e.update(name=p.get("name"), rate=r, last_seen=now, line=line.prefix if line else None,
                 created=parse_time(p.get("createdAt")) or e.get("created"))
        e["spent"] += r * hours
        if line is not None:
            t = state["lines"][line.prefix]
            t["spent"] += r * hours
            w = t.setdefault("window", {})
            w[bucket] = w.get(bucket, 0.0) + r * hours
    for t in state["lines"].values():
        w = t.setdefault("window", {})
        for k in [k for k in w if float(k) + BUCKET_S <= now - WINDOW_S]:
            del w[k]
        t["window_usd"] = sum(w.values())
    state["last_poll"], state["rate_now"] = now, rate_now

    balance = floor = None
    if budgets.balance_floor is not None:
        floor = budgets.balance_floor
        try:
            balance = state["balance"] = float(api.balance())
            state["balance_utc"] = iso(now)
        except Exception as err:  # noqa: BLE001 -- an unreadable balance keeps the last reading: a failed read never loosens the floor
            balance = state.get("balance")
            notes.append(f"balance read failed: {type(err).__name__}: {err}" + (f"; judging the last reading, ${balance:.2f}" if balance is not None else ""))
    below = balance is not None and floor is not None and balance <= floor

    kills: dict[str, tuple[dict[str, Any], str]] = {}
    for prefix, line in budgets.lines.items():
        t = state["lines"][prefix]
        members = [p for p, ln in scope if ln is line]
        over = []
        if line.cap_usd is not None and t["spent"] >= line.cap_usd:
            over.append(("cap", f"{prefix!r} CAP ${line.cap_usd:g} reached (spent ${t['spent']:.2f})"))
        if line.cap_usd_per_day is not None and t["window_usd"] >= line.cap_usd_per_day:
            over.append(("per_day", f"{prefix!r} DAILY CAP ${line.cap_usd_per_day:g} reached (${t['window_usd']:.2f} in the last 24 h)"))
        if t.get("tripped") and not over:
            notes.append(f"UNTRIPPED {prefix!r}: " + (f"${t['window_usd']:.2f} in the last 24 h is under the daily cap ${line.cap_usd_per_day:g}"
                                                     if t["tripped"].get("kind") == "per_day" else
                                                     f"cap ${line.cap_usd:g} is over the spend ${t['spent']:.2f}" if line.cap_usd is not None else
                                                     "no cap applies any more"))
            t["tripped"] = None
        if not t.get("tripped") and over:
            kind, why = over[0]
            t["tripped"] = {"at": iso(now), "why": why, "kind": kind, "pods": sorted(p["id"] for p in members)}
            notes.append(f"TRIPPED: {why}")
        elif t.get("tripped") and over and t["tripped"].get("kind", "cap") not in [k for k, _ in over]:
            t["tripped"].update(kind=over[0][0], why=over[0][1])          # still over, now on the other cap: the trip holds
            notes.append(f"STILL TRIPPED: {over[0][1]}")
        if t.get("tripped"):
            for p in members:
                if p["id"] in t["tripped"]["pods"]:
                    kills.setdefault(p["id"], (p, t["tripped"]["why"]))
                elif not state["pods"][p["id"]].get("over_cap_noted"):
                    state["pods"][p["id"]]["over_cap_noted"] = True
                    notes.append(f"OVER-CAP {p['id']} {p.get('name')} appeared under the tripped line {prefix!r}: counted, not terminated")
    for p, line in scope:
        e = state["pods"][p["id"]]
        if below:
            kills.setdefault(p["id"], (p, f"BALANCE ${balance:.2f} <= floor ${floor:g}"))
        if line is not None:
            e.pop("uncovered_since", None)
            age_h = (now - (e.get("created") or e["first_seen"])) / 3600.0
            if age_h > line.max_pod_hours:
                kills.setdefault(p["id"], (p, f"POD AGE {age_h:.2f} h > {line.prefix!r} max_pod_hours {line.max_pod_hours:g}"))
            continue
        since = e.setdefault("uncovered_since", now)
        waited_min = (now - since) / 60.0
        if budgets.terminate_uncovered and waited_min >= budgets.uncovered_grace_min:
            kills.setdefault(p["id"], (p, f"UNCOVERED: no budget line covers {p.get('name')!r} ({waited_min:.0f} min)"))
        elif not e.get("uncovered_noted"):
            e["uncovered_noted"] = True
            notes.append(f"UNCOVERED {p['id']} {p.get('name')}: no budget line covers it; "
                         + (f"terminated after {budgets.uncovered_grace_min:g} min" if budgets.terminate_uncovered else "reported only (terminate_uncovered = false)"))

    done = _terminate(api, cfg, state, [kv for k, kv in kills.items() if k not in state["terminating"]], now)
    state["last_ok_poll"] = now
    for n in notes:
        log(n)
    for d in done:
        log(d["line"])
    if scope:
        log(f"{len(scope)} pod(s) ${rate_now:.2f}/h: " + " ".join(f"{p['id']}:{p.get('name')}@{p.get('costPerHr')}" for p, _ in scope)
            + " | " + " ".join(_line_spend(pre, t, budgets.lines[pre]) for pre, t in sorted(state["lines"].items()) if pre in budgets.lines))
    return done


def _line_spend(prefix: str, t: dict[str, Any], line: B.Line) -> str:
    return (f"{prefix}=${t['spent']:.2f}" + (f"/{line.cap_usd:g}" if line.cap_usd is not None else "")
            + (f" 24h ${t.get('window_usd', 0.0):.2f}/{line.cap_usd_per_day:g}" if line.cap_usd_per_day is not None else ""))


def _terminate(api: Any, cfg: Config, state: dict[str, Any], kills: list[tuple[dict[str, Any], str]], now: float) -> list[dict[str, Any]]:
    done = []
    for p, why in kills:
        entry = {"at": iso(now), "id": p["id"], "name": p.get("name"), "rate": float(p.get("costPerHr") or 0.0), "why": why, "dry_run": cfg.dry_run}
        if cfg.dry_run:
            entry["line"] = f"WOULD-TERMINATE {p['id']} {p.get('name')} ${entry['rate']:g}/h -- {why}"
        else:
            try:
                api.terminate(p["id"])
                entry["line"] = f"TERMINATED {p['id']} {p.get('name')} ${entry['rate']:g}/h -- {why}"
            except Exception as err:  # noqa: BLE001 -- one failed DELETE must not stop the rest; it is tried again next poll
                entry["error"] = f"{type(err).__name__}: {err}"
                entry["line"] = f"terminate {p['id']} {p.get('name')} failed: {entry['error']} ({why})"
        if "error" not in entry:
            state["terminating"][p["id"]] = now
        state["terminated"] = (state.get("terminated") or [])[-(KEEP_TERMINATED - 1):] + [entry]
        done.append(entry)
    return done


# ---- the loop, its supervisor, and the process controls -------------------------------------------------------------------------------------------

def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    try:
        return b"guard" in Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return True                                                         # no /proc (macOS): the signal probe is all there is


def recorded_pid(pid_path: Path) -> int | None:
    try:
        return int(pid_path.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def api_problem() -> str | None:
    """Why the real RunPod API cannot be called here (no ~/.runpod/config.toml and no RUNPOD_API_KEY), or None."""
    from . import runpod

    try:
        runpod._config()
    except runpod.PodError as e:
        return str(e)
    return None


def run(cfg: Config, api: Any = None, *, poll_s: float = 60.0, log_path: str | None = None, clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep, iterations: int | None = None) -> int:
    """The guard loop (`iterations` bounds it for tests).  Exit 3 when another guard is alive here; EXIT_NO_KEY, before any poll, when
    it would call the real API and no RunPod key is configured.  No local fault ends it (module docstring, "Fail closed"): the
    directories, pid, log and state go through a `Recorder`."""
    no_key = api_problem() if api is None else None
    api = api or RunPodAPI()
    p = paths(cfg.dry_run, log_path)
    rec = Recorder(p, clock)
    for f in p.values():   # the pid and state files live in the guard dir even when --log points elsewhere
        rec.attempt("directory", lambda f=f: f.parent.mkdir(parents=True, exist_ok=True))
    log = rec.log
    other = recorded_pid(p["pid"])
    if other and other != os.getpid() and _alive(other):
        log(f"REFUSED to start: guard pid {other} is alive ({p['pid']})")
        print(f"research pods guard: guard pid {other} is alive ({p['pid']}); `research pods guard stop` first", file=sys.stderr)
        return 3
    if no_key:
        log(f"REFUSED to start: {no_key}")
        print(f"research pods guard: {no_key}: a guard without a RunPod key can terminate nothing, so it does not start", file=sys.stderr)
        return EXIT_NO_KEY
    rec.attempt("pid", lambda: p["pid"].write_text(f"{os.getpid()}\n"))
    state = load_state(p["state"], clock(), log)
    state.update(pid=os.getpid(), config={"source": str(cfg.source), "dry_run": cfg.dry_run, "self_pod": cfg.self_pod, "poll_s": poll_s})
    budgets = refresh(cfg, state, budgets_from_state(state), clock(), log)
    log(f"guard pid {os.getpid()} start budgets={cfg.source} self={cfg.self_pod} lines={len(budgets.lines) if budgets else 0}"
        + (" DRY-RUN" if cfg.dry_run else ""))
    rec.save(state)
    n = 0
    while iterations is None or n < iterations:
        if budgets is not None:
            try:
                poll(api, budgets, cfg, state, clock(), log)
            except Exception as err:  # noqa: BLE001 -- a safety loop must outlive a network blip (budget_cap.py, 2026-09-09)
                log(f"poll failed: {type(err).__name__}: {err}")
        rec.save(state)
        budgets = refresh(cfg, state, budgets, clock(), log)
        rec.save(state)
        n += 1
        if iterations is None or n < iterations:
            sleep(poll_s)
    return 0


def supervise(cmd: list[str], poll_s: float, *, call: Callable[[list[str]], int] | None = None,
              sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.time) -> int:
    """Run the guard `cmd` until it stops on purpose (STOP_CODES), starting it again `poll_s` after any other exit: a crash, a kill,
    an uncaught error.  A crash costs one poll, not the evening.  Returns the guard's last exit code."""
    def say(msg: str) -> None:
        try:
            print(f"{iso(clock())} supervisor pid {os.getpid()}: {msg}", flush=True)
        except Exception:  # noqa: BLE001 -- its output is the log, on a disk that may be full
            pass

    call = call or (lambda c: subprocess.call(c, env={**os.environ, SUPERVISED_ENV: "1"}))
    restarts = 0
    while True:
        try:
            rc = call(cmd)
        except OSError as err:           # the interpreter could not even be started (a transient fork failure)
            rc = -1
            say(f"could not start the guard: {type(err).__name__}: {err}")
        if rc in STOP_CODES:
            say(f"guard exited {rc}; not restarting")
            return rc
        restarts += 1
        say(f"guard exited {rc}: restart {restarts} in {poll_s:g} s")
        sleep(poll_s)


def detach(argv: list[str], dry_run: bool, log_path: str | None) -> int:
    """Start the supervisor (`--supervise`, which runs the guard and restarts it) in a new session, output to the log."""
    p = paths(dry_run, log_path)
    other = recorded_pid(p["pid"])
    if other and _alive(other):
        print(f"research pods guard: guard pid {other} is alive; `research pods guard stop` first", file=sys.stderr)
        return 3
    for f in p.values():
        f.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "research", "pods", "guard", "--supervise", *[a for a in argv if a != "--detach"]]
    if sys.platform == "darwin":
        cmd = ["caffeinate", "-is", *cmd]
    with p["log"].open("ab") as out:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=out, start_new_session=True)
    time.sleep(1.0)
    if proc.poll() is not None:
        print(f"research pods guard: the detached guard exited {proc.returncode}; see {p['log']}", file=sys.stderr)
        return 1
    print(f"research pods guard: supervisor pid {proc.pid} log={p['log']} (`research pods guard status`)")
    return 0


def status(dry_run: bool = False, as_json: bool = False, clock: Callable[[], float] = time.time) -> int:
    p = paths(dry_run)
    try:
        state = json.loads(p["state"].read_text())
    except (OSError, ValueError) as e:
        print(f"research pods guard: no readable guard state ({p['state']}: {type(e).__name__})", file=sys.stderr)
        return 2
    pid = recorded_pid(p["pid"])
    now = clock()
    b = state.get("budgets") or {}
    out = {"pid": pid, "alive": bool(pid and _alive(pid)), "started": iso(state.get("started")), "last_ok_poll": iso(state.get("last_ok_poll")),
           "stale": state.get("last_ok_poll") is None or now - state["last_ok_poll"] > B.STALE_S, "rate_now": state.get("rate_now"),
           "balance": state.get("balance"), "budgets": {k: v for k, v in b.items() if k != "text"}, "lines": state.get("lines") or {},
           "pods": {k: v for k, v in (state.get("pods") or {}).items() if state.get("last_poll") and v.get("last_seen") == state["last_poll"]},
           "faults": state.get("faults") or {}, "tally_restarted": state.get("tally_restarted"), "terminated": (state.get("terminated") or [])[-20:],
           "config": state.get("config"), "state_file": str(p["state"]), "log": str(p["log"])}
    if as_json:
        print(json.dumps(out, indent=1, sort_keys=True))
        return 0
    ago = f"{(now - state['last_ok_poll']) / 60:.1f} min ago" if state.get("last_ok_poll") else "never"
    print(f"budgets guard: {'alive' if out['alive'] else 'NOT RUNNING'} (pid {pid}), last completed poll {ago}"
          + (" -- STALE: `pods create` refuses" if out["stale"] else "") + (" DRY-RUN" if (out["config"] or {}).get("dry_run") else ""))
    print(f"  budgets from {b.get('source')} read {b.get('read_utc')}" + (f"; last read FAILED {b.get('error_utc')}: {b['error']}" if b.get("error") else ""))
    g = [f"project {','.join(b.get('project') or []) or '-'}"] + ([f"balance floor ${b['balance_floor']:g}"] if b.get("balance_floor") is not None else []) \
        + [f"uncovered: {'terminated after ' + format(b.get('uncovered_grace_min', 15), 'g') + ' min' if b.get('terminate_uncovered', True) else 'reported'}"]
    print("  " + ", ".join(g) + (f"; balance ${out['balance']:.2f}" if out["balance"] is not None else "") + f"; ${out['rate_now'] or 0:.2f}/h in scope")
    for pre, ln in sorted((b.get("lines") or {}).items()):
        t = out["lines"].get(pre) or {}
        caps = ([f"${t.get('spent', 0.0):.2f} of ${ln['cap_usd']:g}"] if ln.get("cap_usd") is not None else [f"${t.get('spent', 0.0):.2f}"]) \
            + ([f"${t.get('window_usd', 0.0):.2f} in 24 h of ${ln['cap_usd_per_day']:g}/day"] if ln.get("cap_usd_per_day") is not None else [])
        print(f"  {pre!r}: {', '.join(caps)}, expires {ln['expires']}, max_pod_hours {ln['max_pod_hours']:g}"
              + (f" -- TRIPPED {t['tripped']['at']} ({t['tripped'].get('kind', 'cap')})" if t.get("tripped") else "")
              + (f" (by {ln['by']})" if ln.get("by") else ""))
    for pid_, e in sorted(out["pods"].items(), key=lambda kv: kv[1].get("created") or 0):
        print(f"  {pid_} {e.get('name')} ${e.get('rate', 0):g}/h spent ${e.get('spent', 0):.2f} created {iso(e.get('created'))} "
              + (f"line {e['line']!r}" if e.get("line") else "UNCOVERED"))
    for what, f in out["faults"].items():
        print(f"  FAULT: {what} writes failing since {f.get('since')} ({f.get('count')}x, last {f.get('error')}); enforcing from memory")
    if out["tally_restarted"]:
        t = out["tally_restarted"]
        print(f"  tallies restarted {t.get('at')}: the state file {t.get('why')} (kept as {t.get('kept')})")
    for t in out["terminated"][-10:]:
        print(f"  {'would-terminate' if t.get('dry_run') else 'terminated'} {t['at']} {t['id']} {t.get('name')} -- {t['why']}" + (f" [FAILED {t['error']}]" if t.get("error") else ""))
    return 0


def stop(dry_run: bool = False) -> int:
    p = paths(dry_run)
    pid = recorded_pid(p["pid"])
    if not pid or not _alive(pid):
        print(f"research pods guard: no live guard ({p['pid']})", file=sys.stderr)
        return 1
    os.kill(pid, signal.SIGTERM)
    print(f"research pods guard: sent SIGTERM to guard pid {pid}")
    return 0


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="research pods guard", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budgets", default=None, metavar="NOTES|FILE",
                    help="a notes clone (budgets.toml read from its origin) or a file (default $RESEARCH_NOTES, else ~/.research/notes)")
    ap.add_argument("--poll", type=float, default=60.0, help="seconds between polls (default 60)")
    ap.add_argument("--dry-run", action="store_true", help="log WOULD-TERMINATE; call nothing; separate state/log/pid files")
    ap.add_argument("--log", help="log file (default $RESEARCH_GUARD_DIR/guard-budgets.log)")
    ap.add_argument("--self-pod", help="the guard's own pod id (default: RUNPOD_POD_ID, /proc/1/environ)")
    ap.add_argument("--detach", action="store_true", help="start the supervisor in a new session with output to the log, print its pid, return")
    ap.add_argument("--supervise", action="store_true", help=argparse.SUPPRESS)    # what --detach starts: run the guard, restart it after a crash
    return ap


def main(argv: list[str]) -> int:
    """`argv` after `guard`."""
    if argv[:1] in (["status"], ["stop"]):
        ap = argparse.ArgumentParser(prog=f"research pods guard {argv[0]}")
        ap.add_argument("--dry-run", action="store_true")
        if argv[0] == "status":
            ap.add_argument("--json", action="store_true")
        ns = ap.parse_args(argv[1:])
        return status(ns.dry_run, ns.json) if argv[0] == "status" else stop(ns.dry_run)
    ns = _parser().parse_args(argv)
    cfg = Config(source=Path(ns.budgets).expanduser() if ns.budgets else B.default_source(), dry_run=ns.dry_run,
                 self_pod=ns.self_pod or self_pod_id())
    if ns.supervise:
        return supervise([sys.executable, "-m", "research", "pods", "guard", *[a for a in argv if a != "--supervise"]], ns.poll)
    if not os.environ.get(SUPERVISED_ENV):
        try:
            prior = budgets_from_state(json.loads(paths(ns.dry_run)["state"].read_text()))
        except (OSError, ValueError):
            prior = None
        try:
            B.load(cfg.source)
        except Exception as e:  # noqa: BLE001
            if prior is None:
                print(f"research pods guard: budgets.toml at {cfg.source} cannot be read ({type(e).__name__}: {e}), and no earlier budgets "
                      "are in the state: there is nothing to enforce", file=sys.stderr)
                return 2
            print(f"research pods guard: budgets.toml at {cfg.source} cannot be read ({type(e).__name__}: {e}); the budgets in the state "
                  "stay in force until it can", file=sys.stderr)
        if why := api_problem():
            print(f"research pods guard: {why}: a guard without a RunPod key can terminate nothing, so it does not start", file=sys.stderr)
            return EXIT_NO_KEY
    if ns.detach:
        return detach(argv, ns.dry_run, ns.log)

    def on_term(signum, frame):  # noqa: ARG001
        Recorder(paths(cfg.dry_run, ns.log), time.time).log(f"guard pid {os.getpid()} stopped (signal {signum})")
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, on_term)
    return run(cfg, poll_s=ns.poll, log_path=ns.log)
