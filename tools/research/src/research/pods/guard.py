"""`research pods guard`: the fleet budget guard as one tested command (the loops of `budget_cap.py`, `sh/deadline.sh`, `sh/deadman.sh`).

    research pods guard --prefix vyv- [--deadline ISO] [--cap-usd N [--baseline USD] [--cap-file F]] [--rate-max USD/h]
                        [--balance-floor USD] [--pod-max-hours H] [--poll S] [--dry-run] [--log F] [--self-pod ID] [--detach]
    research pods guard status --prefix vyv- [--dry-run] [--json]
    research pods guard stop --prefix vyv- [--dry-run]
    research pods guard reset --prefix vyv- [--by WHO] [--dry-run]

Scope.  Only live pods (desiredStatus not EXITED/TERMINATED) whose NAME starts with `--prefix` are counted or touched.  Every other pod
on the account is invisible to the guard: not counted, never terminated.  The prefix is required.

Every `--poll` seconds (default 60):
  spend += the scoped pods' costPerHr x the time since the last poll (budget_cap.py's measure: RunPod bills a running pod whether or not
          anything uses it).  The tally lives in the state file, so a restart continues it; `--baseline USD` seeds a NEW tally with spend
          already incurred (switching over from budget_cap.py: its state's `spent`).  Account billing (GraphQL clientBalance) is
          account-wide, not prefix-scoped, so it feeds only `--balance-floor`.
  trips -- sticky: terminate EVERY scoped pod, the guard's own host last, then keep terminating scoped pods that appear:
    --deadline ISO        now >= the deadline (ISO 8601 with a zone: `2026-09-24T03:00Z`)
    --cap-usd N           spend >= N, or >= the number in `--cap-file` when that file is readable (read each poll: phase gates raise it)
    --balance-floor USD   the account's clientBalance <= USD
  shedding -- never the guard's own host, no trip:
    --pod-max-hours H     a scoped pod running longer than H hours (since RunPod's lastStartedAt) is terminated
    --rate-max USD/h      while the scoped rate exceeds it, terminate the NEWEST scoped pod (RunPod's createdAt), one at a time, until the
                          rate is at or under the limit.  budget_cap.py terminated every owned pod here, its own host included.
  A pod the guard terminated in the last TERMINATING_S seconds still counts toward spend but not toward the rate (the API lists it for a
  while after DELETE; counting it would shed one more pod per poll).
Each termination is one log line (pod id, name, $/h, reason) and an entry in the state's `terminated`.  `--dry-run` logs WOULD-TERMINATE,
calls no DELETE, and keeps its own state/log/pid files (a dry run's sticky trip must never arm the real guard).

The guard's own host is `--self-pod ID`, else RUNPOD_POD_ID (from /proc/1/environ when an ssh session did not inherit it), else
/root/dm/POD_ID, else none (a laptop).

Files: $RESEARCH_GUARD_DIR (default ~/.research/pods)/guard-<prefix>[.dry-run].{json,log,pid}.  One guard per prefix: a start refuses
(exit 3) while the recorded pid is alive.  `--detach` starts a supervisor in a new session, stdin closed, output to the log (setsid
nohup; under `caffeinate -is` on macOS), and prints its pid; the supervisor runs the guard and starts it again one poll after any exit
but a stop (0), bad arguments (2) or another live guard (3).  A failed list (network blip) is logged and retried: it never ends the
guard.  `stop` signals exactly the recorded pid (no pkill).  A trip is sticky across restarts; after a lapsed deadline `reset` clears it
in the state file, keeping the spend tally, and logs who cleared it.  It refuses (exit 3) while the recorded guard is alive, since a
live guard rewrites the file from memory each poll: stop, reset, then start again with the new limits.

Fail closed.  A fault on the guard's own machine never stops enforcement and never loosens a limit (the 2026-09-28 18:10 PT incident:
the control pod's disk filled, every guard died on its state write, and nothing enforced any cap for 15 minutes).
  * Each poll decides and terminates from memory first, then records.  Every log, state and pid write is best effort: a failure is
    kept in memory (`faults`), printed to stderr when it starts and when it clears, and written into the state once a write succeeds.
  * A read that fails keeps the last value read: an unreadable `--cap-file` keeps the cap in force, an unreadable balance keeps the
    last balance.  A state file that does not parse is set aside (`.corrupt-<epoch>`) and a new tally starts, recorded in the state.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import socket
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

DEAD = ("EXITED", "TERMINATED")
TERMINATING_S = 600
KEEP_TERMINATED = 500
STOP_CODES = (0, 2, 3)          # the guard's exits the supervisor does not answer with a restart: stopped, bad arguments, another guard


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
    prefix: str
    deadline: float | None = None
    cap_usd: float | None = None
    baseline: float = 0.0
    cap_file: str | None = None
    rate_max: float | None = None
    balance_floor: float | None = None
    pod_max_hours: float | None = None
    dry_run: bool = False
    self_pod: str | None = None

    def limits(self) -> list[str]:
        return [k for k in ("deadline", "cap_usd", "rate_max", "balance_floor", "pod_max_hours") if getattr(self, k) is not None]


def guard_dir() -> Path:
    return Path(os.environ.get("RESEARCH_GUARD_DIR", Path.home() / ".research" / "pods"))


def paths(prefix: str, dry_run: bool = False, log: str | None = None) -> dict[str, Path]:
    stem = "guard-" + re.sub(r"[^A-Za-z0-9_.-]", "_", prefix) + (".dry-run" if dry_run else "")
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
    try:
        d = dt.datetime.fromisoformat(t)
    except ValueError:
        return None
    return d.timestamp() if d.tzinfo else None


def iso(t: float | None) -> str | None:
    return None if t is None else dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def scoped(pods: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    """Live pods whose name starts with `prefix`; nothing else exists for the guard."""
    return [p for p in pods if prefix and str(p.get("name") or "").startswith(prefix) and (p.get("desiredStatus") or "").upper() not in DEAD]


def self_pod_id() -> str | None:
    if os.environ.get("RUNPOD_POD_ID"):
        return os.environ["RUNPOD_POD_ID"]
    try:
        for item in Path("/proc/1/environ").read_bytes().split(b"\0"):
            if item.startswith(b"RUNPOD_POD_ID="):
                return item.split(b"=", 1)[1].decode() or None
    except OSError:
        pass
    try:
        return Path("/root/dm/POD_ID").read_text().strip() or None
    except OSError:
        return None


def new_state(cfg: Config, now: float) -> dict[str, Any]:
    return {"prefix": cfg.prefix, "started": now, "spent": float(cfg.baseline), "baseline": float(cfg.baseline), "last_poll": None,
            "rate_now": 0.0, "pods": {}, "tripped": None, "terminated": [], "terminating": {}}


def load_state(path: Path, cfg: Config, now: float, log: Callable[[str], None]) -> dict[str, Any]:
    """The tally in `path`, else a new one.  A file that cannot be read or parsed does not stop the guard: it is set aside and a new
    tally starts from the baseline, with `tally_restarted` saying why (the spend before it is not counted)."""
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return new_state(cfg, now)
    except OSError as err:
        why = f"unreadable ({type(err).__name__}: {err})"
    else:
        try:
            state = json.loads(raw)
            if isinstance(state, dict) and isinstance(state.get("spent"), (int, float)) and isinstance(state.get("pods"), dict):
                if cfg.baseline:
                    log(f"--baseline {cfg.baseline:g} ignored: the tally in {path.name} continues (spent ${state['spent']:.2f})")
                return state
            why = "not a guard state"
        except ValueError as err:
            why = f"does not parse ({err})"
    aside = path.with_name(f"{path.name}.corrupt-{int(now)}")
    try:
        path.replace(aside)
    except OSError:
        aside = path
    log(f"STATE {path.name} {why}: kept as {aside.name}; a new tally starts at the baseline ${cfg.baseline:.2f}")
    state = new_state(cfg, now)
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


def cap_now(cfg: Config, state: dict[str, Any], log: Callable[[str], None]) -> float | None:
    """The cap in force: the number in `--cap-file` when it is readable, else the last cap in force (a failed read never loosens nor
    tightens it), else `--cap-usd`."""
    if cfg.cap_file:
        try:
            value: float | None = float(Path(cfg.cap_file).read_text().split()[0])
            err = None
        except (OSError, ValueError, IndexError) as e:
            value, err = None, f"{type(e).__name__}: {e}"
        if value is not None:
            if value != state.get("cap"):
                log(f"cap {state.get('cap')} -> {value:g} (from {cfg.cap_file})")
            state["cap"] = value
            state.pop("cap_file_error", None)
            return value
        if state.get("cap") is not None:
            if state.get("cap_file_error") is None:
                log(f"cap file {cfg.cap_file} unreadable ({err}): keeping the cap in force, ${state['cap']:g}")
            state["cap_file_error"] = err
            return state["cap"]
    state["cap"] = cfg.cap_usd
    return cfg.cap_usd


def poll(api: Any, cfg: Config, state: dict[str, Any], now: float, log: Callable[[str], None]) -> list[dict[str, Any]]:
    """One guard step over the provider's current pod list: tally, trip, shed.  Returns the terminations it made (or would make)."""
    pods = scoped(api.pods(), cfg.prefix)
    me = cfg.self_pod
    hours = (now - state["last_poll"]) / 3600.0 if state.get("last_poll") else 0.0
    state["terminating"] = {k: t for k, t in (state.get("terminating") or {}).items() if now - t < TERMINATING_S}
    rate = 0.0
    for p in pods:
        r = float(p.get("costPerHr") or 0.0)
        rate += r
        e = state["pods"].setdefault(p["id"], {"name": p.get("name"), "spent": 0.0, "first_seen": now})
        e.update(rate=r, last_seen=now, created=parse_time(p.get("createdAt")) or e.get("created"),
                 started=parse_time(p.get("lastStartedAt")) or e.get("started"))
        e["spent"] += r * hours
    state["spent"] += rate * hours
    state["last_poll"], state["rate_now"] = now, rate
    cap = cap_now(cfg, state, log)

    balance = None
    if cfg.balance_floor is not None and not state.get("tripped"):
        try:
            balance = state["balance"] = api.balance()
        except Exception as err:  # noqa: BLE001 -- an unreadable balance keeps the last reading: a failed read never loosens the floor
            balance = state.get("balance")
            log(f"balance read failed: {type(err).__name__}: {err}" + (f"; judging the last reading, ${balance:.2f}" if balance is not None else ""))
    if not state.get("tripped"):
        why = None
        if cfg.deadline is not None and now >= cfg.deadline:
            why = f"DEADLINE {iso(cfg.deadline)} reached"
        elif cap is not None and state["spent"] >= cap:
            why = f"CAP ${cap:g} reached (spent ${state['spent']:.2f})"
        elif balance is not None and balance <= cfg.balance_floor:
            why = f"BALANCE ${balance:.2f} <= floor ${cfg.balance_floor:g}"
        if why:
            state["tripped"] = {"at": iso(now), "why": why}
            log(f"TRIPPED: {why}")

    done: list[dict[str, Any]] = []
    try:
        _act(cfg, state, pods, now, me, log, done, api)
    finally:
        state["rate_kept"] = sum(float(p.get("costPerHr") or 0.0) for p in pods if p["id"] not in state["terminating"])
    if pods:
        log(f"{len(pods)} pod(s) ${rate:.2f}/h spent=${state['spent']:.2f}" + (f" of ${cap:g}" if cap is not None else "")
            + " " + " ".join(f"{p['id']}:{p.get('name')}@{p.get('costPerHr')}" for p in pods))
    return done


def _act(cfg: Config, state: dict[str, Any], pods: list[dict[str, Any]], now: float, me: str | None, log: Callable[[str], None],
         done: list[dict[str, Any]], api: Any) -> None:
    def kill(p: dict[str, Any], why: str) -> None:
        entry = {"at": iso(now), "id": p["id"], "name": p.get("name"), "rate": float(p.get("costPerHr") or 0.0), "why": why, "dry_run": cfg.dry_run}
        if cfg.dry_run:
            log(f"WOULD-TERMINATE {p['id']} {p.get('name')} ${entry['rate']:g}/h -- {why}")
        else:
            try:
                api.terminate(p["id"])
                log(f"TERMINATED {p['id']} {p.get('name')} ${entry['rate']:g}/h -- {why}")
            except Exception as err:  # noqa: BLE001 -- one failed DELETE must not stop the rest
                entry["error"] = f"{type(err).__name__}: {err}"
                log(f"terminate {p['id']} {p.get('name')} failed: {entry['error']}")
        if "error" not in entry:
            state["terminating"][p["id"]] = now
        state["terminated"] = (state.get("terminated") or [])[-(KEEP_TERMINATED - 1):] + [entry]
        done.append(entry)

    if state.get("tripped"):
        for p in sorted((p for p in pods if p["id"] not in state["terminating"]), key=lambda p: p["id"] == me):   # own host last
            kill(p, state["tripped"]["why"])
        return

    live = [p for p in pods if p["id"] not in state["terminating"]]
    if cfg.pod_max_hours is not None:
        for p in list(live):
            e = state["pods"][p["id"]]
            age_h = (now - (e.get("started") or e.get("created") or e["first_seen"])) / 3600.0
            if p["id"] != me and age_h > cfg.pod_max_hours:
                kill(p, f"POD AGE {age_h:.2f} h > {cfg.pod_max_hours:g} h")
                live.remove(p)
    if cfg.rate_max is not None:
        r = sum(float(p.get("costPerHr") or 0.0) for p in live)
        if r > cfg.rate_max:
            newest_first = sorted((p for p in live if p["id"] != me),
                                  key=lambda p: (state["pods"][p["id"]].get("created") or state["pods"][p["id"]]["first_seen"], p["id"]), reverse=True)
            for p in newest_first:
                if r <= cfg.rate_max:
                    break
                kill(p, f"RATE ${r:.2f}/h > ${cfg.rate_max:g}/h, newest first")
                r -= float(p.get("costPerHr") or 0.0)
            if r > cfg.rate_max:
                log(f"RATE ${r:.2f}/h still > ${cfg.rate_max:g}/h: only the guard's own host ({me}) is left, and it is never shed")


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


def run(cfg: Config, api: Any = None, *, poll_s: float = 60.0, log_path: str | None = None, clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep, iterations: int | None = None) -> int:
    """The guard loop (`iterations` bounds it for tests).  Exit 3 when another guard for this prefix is alive.  No local fault ends
    it (module docstring, "Fail closed"): the directories, pid, log and state go through a `Recorder`."""
    api = api or RunPodAPI()
    p = paths(cfg.prefix, cfg.dry_run, log_path)
    rec = Recorder(p, clock)
    for f in p.values():   # the pid and state files live in the guard dir even when --log points elsewhere
        rec.attempt("directory", lambda f=f: f.parent.mkdir(parents=True, exist_ok=True))
    log = rec.log

    other = recorded_pid(p["pid"])
    if other and other != os.getpid() and _alive(other):
        log(f"REFUSED to start: guard pid {other} for prefix {cfg.prefix!r} is alive ({p['pid']})")
        print(f"research pods guard: guard pid {other} for prefix {cfg.prefix!r} is alive ({p['pid']}); `research pods guard stop --prefix "
              f"{cfg.prefix}` first", file=sys.stderr)
        return 3
    rec.attempt("pid", lambda: p["pid"].write_text(f"{os.getpid()}\n"))
    state = load_state(p["state"], cfg, clock(), log)
    state["config"] = {**asdict(cfg), "deadline": iso(cfg.deadline), "poll_s": poll_s}
    state["pid"] = os.getpid()
    log(f"guard pid {os.getpid()} start prefix={cfg.prefix!r} self={cfg.self_pod} spent_so_far=${state['spent']:.2f} "
        + " ".join(f"{k}={state['config'][k]}" for k in cfg.limits()) + (" cap_file=" + cfg.cap_file if cfg.cap_file else "")
        + (" DRY-RUN" if cfg.dry_run else ""))
    rec.save(state)
    n = 0
    while iterations is None or n < iterations:
        try:
            poll(api, cfg, state, clock(), log)
        except Exception as err:  # noqa: BLE001 -- a safety loop must outlive a network blip (budget_cap.py, 2026-09-09)
            log(f"poll failed: {type(err).__name__}: {err}")
        rec.save(state)
        n += 1
        if iterations is None or n < iterations:
            sleep(poll_s)
    return 0


def supervise(cmd: list[str], poll_s: float, *, call: Callable[[list[str]], int] = subprocess.call,
              sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.time) -> int:
    """Run the guard `cmd` until it stops on purpose (STOP_CODES), starting it again `poll_s` after any other exit: a crash, a kill,
    an uncaught error.  A crash costs one poll, not the evening.  Returns the guard's last exit code."""
    def say(msg: str) -> None:
        try:
            print(f"{iso(clock())} supervisor pid {os.getpid()}: {msg}", flush=True)
        except Exception:  # noqa: BLE001 -- its output is the log, on a disk that may be full
            pass

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


def detach(argv: list[str], prefix: str, dry_run: bool, log_path: str | None) -> int:
    """Start the supervisor (`--supervise`, which runs the guard and restarts it) in a new session, output to the log."""
    p = paths(prefix, dry_run, log_path)
    other = recorded_pid(p["pid"])
    if other and _alive(other):
        print(f"research pods guard: guard pid {other} for prefix {prefix!r} is alive; `research pods guard stop --prefix {prefix}` first", file=sys.stderr)
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
    print(f"research pods guard: supervisor pid {proc.pid} prefix={prefix!r} log={p['log']} (`research pods guard status --prefix {prefix}`)")
    return 0


def status(prefix: str, dry_run: bool = False, as_json: bool = False) -> int:
    p = paths(prefix, dry_run)
    if not p["state"].exists():
        print(f"research pods guard: no guard state for prefix {prefix!r} ({p['state']})", file=sys.stderr)
        return 2
    state = json.loads(p["state"].read_text())
    pid = recorded_pid(p["pid"])
    out = {"prefix": prefix, "pid": pid, "alive": bool(pid and _alive(pid)), "started": iso(state.get("started")),
           "last_poll": iso(state.get("last_poll")), "spent": round(state.get("spent", 0.0), 2), "cap": state.get("cap"),
           "rate_now": state.get("rate_now"), "rate_kept": state.get("rate_kept"), "tripped": state.get("tripped"), "config": state.get("config"),
           "faults": state.get("faults") or {}, "cap_file_error": state.get("cap_file_error"), "tally_restarted": state.get("tally_restarted"),
           "pods": {k: v for k, v in state.get("pods", {}).items() if state.get("last_poll") and v.get("last_seen") == state["last_poll"]},
           "terminated": state.get("terminated", [])[-20:], "state_file": str(p["state"]), "log": str(p["log"])}
    if as_json:
        print(json.dumps(out, indent=1, sort_keys=True))
        return 0
    cfg = out["config"] or {}
    print(f"guard {prefix!r}: {'alive' if out['alive'] else 'NOT RUNNING'} (pid {pid}), last poll {out['last_poll']}"
          + (" DRY-RUN" if cfg.get("dry_run") else ""))
    print(f"  spent ${out['spent']:.2f}" + (f" of ${out['cap']:g}" if out["cap"] is not None else "")
          + f", rate ${out['rate_kept'] or 0:.2f}/h kept (${out['rate_now'] or 0:.2f}/h listed)"
          + (f" (max ${cfg['rate_max']:g}/h)" if cfg.get("rate_max") is not None else "")
          + (f", deadline {cfg['deadline']}" if cfg.get("deadline") else "") + (f", pod-max {cfg['pod_max_hours']:g} h" if cfg.get("pod_max_hours") else ""))
    if out["tripped"]:
        print(f"  TRIPPED {out['tripped']['at']}: {out['tripped']['why']}")
    for what, f in (state.get("faults") or {}).items():
        print(f"  FAULT: {what} writes failing since {f.get('since')} ({f.get('count')}x, last {f.get('error')}); enforcing from memory")
    if state.get("cap_file_error"):
        print(f"  cap file unreadable ({state['cap_file_error']}): the cap in force is kept")
    if state.get("tally_restarted"):
        t = state["tally_restarted"]
        print(f"  tally restarted {t.get('at')}: the state file {t.get('why')} (kept as {t.get('kept')})")
    for pid_, e in sorted(out["pods"].items(), key=lambda kv: kv[1].get("created") or 0):
        print(f"  {pid_} {e.get('name')} ${e.get('rate', 0):g}/h spent ${e.get('spent', 0):.2f} created {iso(e.get('created'))}")
    for t in out["terminated"][-10:]:
        print(f"  {'would-terminate' if t.get('dry_run') else 'terminated'} {t['at']} {t['id']} {t.get('name')} -- {t['why']}" + (f" [FAILED {t['error']}]" if t.get("error") else ""))
    return 0


def stop(prefix: str, dry_run: bool = False) -> int:
    p = paths(prefix, dry_run)
    pid = recorded_pid(p["pid"])
    if not pid or not _alive(pid):
        print(f"research pods guard: no live guard for prefix {prefix!r} ({p['pid']})", file=sys.stderr)
        return 1
    os.kill(pid, signal.SIGTERM)
    print(f"research pods guard: sent SIGTERM to guard pid {pid} (prefix {prefix!r})")
    return 0


def reset(prefix: str, by: str, dry_run: bool = False, clock: Callable[[], float] = time.time) -> int:
    """Clear the sticky trip for ``prefix`` (the vLLM coordinator hand-edited ``state.tripped`` after a lapsed deadline, 2026-09-26):
    the spend tally, pods and terminations stay; the trip moves to ``resets`` with who cleared it, and the log records it."""
    p = paths(prefix, dry_run)
    if not p["state"].exists():
        print(f"research pods guard: no guard state for prefix {prefix!r} ({p['state']})", file=sys.stderr)
        return 2
    pid = recorded_pid(p["pid"])
    if pid and _alive(pid):
        print(f"research pods guard: guard pid {pid} is running for prefix {prefix!r}; it rewrites the state each poll. "
              f"`research pods guard stop --prefix {prefix}` first, then reset, then start it again", file=sys.stderr)
        return 3
    state = json.loads(p["state"].read_text())
    trip = state.get("tripped")
    if not trip:
        print(f"research pods guard: prefix {prefix!r} is not tripped; nothing to reset")
        return 0
    now = iso(clock())
    state.setdefault("resets", []).append({"at": now, "by": by, "cleared": trip})
    state["tripped"] = None
    save_state(p["state"], state)
    with p["log"].open("a") as f:
        f.write(f"{now} RESET by {by}: cleared the trip of {trip.get('at')} ({trip.get('why')}); spent ${state.get('spent', 0.0):.2f} kept\n")
    print(f"research pods guard: cleared the trip for prefix {prefix!r} ({trip.get('why')}); spent ${state.get('spent', 0.0):.2f} kept")
    return 0


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="research pods guard", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", required=True, help="only pods whose name starts with this are counted or terminated")
    ap.add_argument("--deadline", help="ISO 8601 with a zone (2026-09-24T03:00Z): terminate every scoped pod then")
    ap.add_argument("--cap-usd", type=float, help="terminate every scoped pod once their integrated spend reaches this")
    ap.add_argument("--baseline", type=float, default=0.0, help="USD already spent, seeding a new tally (ignored when the state file exists)")
    ap.add_argument("--cap-file", help="read the cap from this file each poll when readable (phase gates)")
    ap.add_argument("--rate-max", type=float, help="USD/h: shed the newest scoped pods until the scoped rate is at or under this")
    ap.add_argument("--balance-floor", type=float, help="terminate every scoped pod when the account balance falls to this")
    ap.add_argument("--pod-max-hours", type=float, help="terminate a scoped pod running longer than this (never the guard's host)")
    ap.add_argument("--poll", type=float, default=60.0, help="seconds between polls (default 60)")
    ap.add_argument("--dry-run", action="store_true", help="log WOULD-TERMINATE; call nothing; separate state/log/pid files")
    ap.add_argument("--log", help="log file (default $RESEARCH_GUARD_DIR/guard-<prefix>.log)")
    ap.add_argument("--self-pod", help="the guard's own pod id (default: RUNPOD_POD_ID, /proc/1/environ, /root/dm/POD_ID)")
    ap.add_argument("--detach", action="store_true", help="start the supervisor in a new session with output to the log, print its pid, return")
    ap.add_argument("--supervise", action="store_true", help=argparse.SUPPRESS)    # what --detach starts: run the guard, restart it after a crash
    return ap


def main(argv: list[str]) -> int:
    """`argv` after `guard`."""
    if argv[:1] in (["status"], ["stop"]):
        ap = argparse.ArgumentParser(prog=f"research pods guard {argv[0]}")
        ap.add_argument("--prefix", required=True)
        ap.add_argument("--dry-run", action="store_true")
        if argv[0] == "status":
            ap.add_argument("--json", action="store_true")
        ns = ap.parse_args(argv[1:])
        return status(ns.prefix, ns.dry_run, ns.json) if argv[0] == "status" else stop(ns.prefix, ns.dry_run)
    if argv[:1] == ["reset"]:
        ap = argparse.ArgumentParser(prog="research pods guard reset")
        ap.add_argument("--prefix", required=True)
        ap.add_argument("--by", default=os.environ.get("RESEARCH_LANE") or f"{os.environ.get('USER', '?')}@{socket.gethostname()}",
                        help="who clears the trip, for the log (default $RESEARCH_LANE, else user@host)")
        ap.add_argument("--dry-run", action="store_true")
        ns = ap.parse_args(argv[1:])
        return reset(ns.prefix, ns.by, ns.dry_run)
    ns = _parser().parse_args(argv)
    if not ns.prefix.strip():
        print("research pods guard: --prefix must be non-empty", file=sys.stderr)
        return 2
    deadline = None
    if ns.deadline:
        deadline = parse_time(ns.deadline)
        if deadline is None:
            print(f"research pods guard: --deadline {ns.deadline!r} is not ISO 8601 with a zone (e.g. 2026-09-24T03:00Z)", file=sys.stderr)
            return 2
    cfg = Config(prefix=ns.prefix, deadline=deadline, cap_usd=ns.cap_usd, baseline=ns.baseline, cap_file=ns.cap_file, rate_max=ns.rate_max,
                 balance_floor=ns.balance_floor, pod_max_hours=ns.pod_max_hours, dry_run=ns.dry_run, self_pod=ns.self_pod or self_pod_id())
    if not cfg.limits() and not cfg.cap_file:
        print("research pods guard: no limit given (--deadline / --cap-usd / --rate-max / --balance-floor / --pod-max-hours); it would guard nothing",
              file=sys.stderr)
        return 2
    if cfg.cap_file and cfg.cap_usd is None and not ns.supervise:
        probe: dict[str, Any] = {}
        try:
            probe = json.loads(paths(cfg.prefix, cfg.dry_run)["state"].read_text())
        except (OSError, ValueError):
            pass
        if cap_now(cfg, probe if isinstance(probe, dict) else {}, lambda m: None) is None:
            print(f"research pods guard: --cap-file {cfg.cap_file} is unreadable, no --cap-usd, and no cap in force from an earlier run: "
                  "there is no cap to enforce", file=sys.stderr)
            return 2
    if ns.detach:
        return detach(argv, ns.prefix, ns.dry_run, ns.log)
    if ns.supervise:
        return supervise([sys.executable, "-m", "research", "pods", "guard", *[a for a in argv if a != "--supervise"]], ns.poll)

    def on_term(signum, frame):  # noqa: ARG001
        Recorder(paths(cfg.prefix, cfg.dry_run, ns.log), time.time).log(f"guard pid {os.getpid()} stopped (signal {signum})")
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, on_term)
    return run(cfg, poll_s=ns.poll, log_path=ns.log)
