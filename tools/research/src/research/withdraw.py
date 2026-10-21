"""`research cancel` of a run that hasn't started: withdraw it from its machine's hold.

A queued run (`research run --queue`, `--on auto`) waits on its machine before its runner starts, in `research disk-hold` and then
`research cpu-hold` (research.disk, research.cores): one process, the launch's runner pid, whose argv names the run's ack FIFO
(`--ack <root>/requests/<run>/.launch-ack`) and which execs the next stage in place when it is released, so the pid becomes
the runner's.  No status.json is written until then, and the running-run cancel has no workload group to signal.

    withdraw   under the request's launch lock (a launch still being answered is `busy`, so its own refusal can't race this
               one), each process whose argv is a hold naming the run's ack FIFO is stopped (SIGSTOP, and its stop seen in
               /proc), and its argv read again: a stopped process can't exec, so one still a hold is ended (SIGTERM, then
               SIGCONT) and one that exec'd on is continued and the run reported `started`, for the running-run cancel.  The
               intent and the result go in the run's events.jsonl as remote.cancel's manual-cancel record does
               (`cancel_intent` / `cancel_result`, actor manual).  Once every hold has exited, a run that still has no
               status.json or refused.json gets refused.json {run_id, reason, fault: Withdrawn, at_utc, by}, written
               atomically, so custody and the watches read it as refused before it ran, not a crash, and a re-delivered
               launch finds the claim refused.  The staged custody and store-read keys and the sent credentials go, and an
               exclusive lock it holds; its memory reservation ends with refused.json (research.mem).

Self-contained (stdlib, no package imports): remote.cancel sends this file to the run's machine as `python3 - ARGS_JSON`
({"run_dir", "reason", "by"} and the launcher's names of the credentials to drop), and it prints one JSON line,
`{"outcome": withdrawn|started|refused|busy|no-hold|unstoppable|survived|absent, ...}`."""
from __future__ import annotations

import fcntl
import json
import os
import select
import signal
import socket
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

HOLDS = ("disk-hold", "cpu-hold")
LAUNCH_ACK = ".launch-ack"                 # remote.LAUNCH_ACK
LAUNCH_LOCK = ".launch.lock"               # remote.LAUNCH_LOCK
FAULT = "Withdrawn"
EVENTS_SCHEMA = "telemetry/events/v0.1"    # research.telemetry.EVENTS_SCHEMA
STOP_POLLS, STOP_POLL_S = 400, 0.005       # a hold sleeps in its poll, so it stops at once; one in D state for 2 s is left alone
EXIT_WAIT_S = 30.0


def utc_now_ms() -> str:
    t = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{int((t % 1) * 1000):03d}Z"


class Procs:
    """The machine's process table and signals: /proc, os.kill and a pidfd to wait on an exit.  Tests give their own."""

    def __init__(self, root: Path = Path("/proc"), kill: Callable[[int, int], None] = os.kill,
                 sleep: Callable[[float], None] = time.sleep):
        self.root, self.kill, self.sleep = root, kill, sleep

    def pids(self) -> list[int]:
        try:
            return sorted(int(e) for e in os.listdir(self.root) if e.isdigit())
        except OSError:
            return []

    def argv(self, pid: int) -> list[str] | None:
        try:
            raw = (self.root / str(pid) / "cmdline").read_bytes()
        except OSError:
            return None
        return raw.decode(errors="replace").split("\0")[:-1] if raw else None

    def stat(self, pid: int) -> tuple[str, str] | None:
        """(state, starttime ticks), None once it is gone."""
        try:
            f = (self.root / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
            return f[0], f[19]
        except (OSError, IndexError):
            return None

    def stopped(self, pid: int) -> bool:
        """Whether `pid` has stopped (after a SIGSTOP), within STOP_POLLS reads; False when it is gone or a zombie."""
        for _ in range(STOP_POLLS):
            st = self.stat(pid)
            if st is None or st[0] in ("Z", "X"):
                return False
            if st[0] in ("T", "t"):
                return True
            self.sleep(STOP_POLL_S)
        return False

    def pidfd(self, pid: int) -> int | None:
        try:
            return os.pidfd_open(pid)
        except (OSError, AttributeError):
            return None

    def gone(self, pid: int, fd: int | None, wait_s: float) -> bool:
        """Whether `pid` (its pidfd `fd`, opened before the signal) exits within `wait_s`."""
        if fd is not None:
            try:
                return bool(select.select([fd], [], [], wait_s)[0])
            finally:
                os.close(fd)
        for _ in range(int(wait_s / 0.05)):
            if (st := self.stat(pid)) is None or st[0] in ("Z", "X"):
                return True
            self.sleep(0.05)
        return False


def hold_of(argv: list[str] | None, acks: set[str]) -> str | None:
    """The hold `argv` is (`... research disk-hold|cpu-hold OPTIONS -- CMD`) when its own options name one of `acks` with
    --ack, else None.  CMD is not read: a disk-hold's CMD is the cpu-hold it execs into, whose options are its own."""
    if not argv:
        return None
    for k, a in enumerate(argv[1:], 1):
        if a in HOLDS and (argv[k - 1] == "research" or argv[k - 1].endswith("/research")):
            opts = argv[k + 1:]
            opts = opts[:opts.index("--")] if "--" in opts else opts
            named = {opts[i + 1] for i in range(len(opts) - 1) if opts[i] == "--ack"} | {o[6:] for o in opts if o.startswith("--ack=")}
            return a if named & acks else None
    return None


def _read(p: Path) -> dict[str, Any] | None:
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def _write_atomic(p: Path, doc: dict[str, Any]) -> None:
    tmp = p.with_name(p.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    os.replace(tmp, p)


_SEQ = 0


def _event(run_dir: Path, rec: dict[str, Any]) -> None:
    """One events.jsonl record as research.telemetry.events.append writes it, in one write."""
    global _SEQ
    _SEQ += 1
    rec = {"schema": EVENTS_SCHEMA, "t_utc": utc_now_ms(), "t_mono": time.monotonic(), "host": socket.gethostname(), "pid": os.getpid(),
           "stream": f"{socket.gethostname()}:{os.getpid()}:withdraw", "seq": _SEQ, **rec}
    fd = os.open(run_dir / "events.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, (json.dumps(rec, separators=(",", ":"), default=str) + "\n").encode())
    finally:
        os.close(fd)


def _started(run_dir: Path) -> dict[str, Any] | None:
    """The outcome for a run that has records of its own by now: it started, or it was refused."""
    if (st := _read(run_dir / "status.json")) is not None:
        return {"outcome": "started", "state": st.get("state")}
    if (rf := _read(run_dir / "refused.json")) is not None:
        return {"outcome": "refused", "reason": rf.get("reason"), "fault": rf.get("fault")}
    return None


def _release(run_dir: Path, req_dir: Path, args: dict[str, Any]) -> list[str]:
    """What a refused launch gives back (remote._release_launch), where a held run keeps it: its staged keys in the request dir,
    the sent credentials in its run dir's inputs/ (moved there at launch) or the request's, its exclusive lock, and the memory
    reservation its runner's `--remote-json` names, when the file is still this run's (research.mem.release)."""
    gone = []
    argv = (_read(run_dir / "launch.json") or {}).get("argv") or []
    spec = next((argv[i + 1] for i in range(len(argv) - 1) if argv[i] == "--remote-json"), None)
    try:
        mem = Path((json.loads(spec) or {})["mem"]["reservation"]) if spec else None
    except (ValueError, KeyError, TypeError):
        mem = None
    if mem and (_read(mem) or {}).get("run_id") == run_dir.name:
        mem.unlink(missing_ok=True)
        gone.append(str(mem))
    for sub in args.get("cred_dirs") or ():
        p = req_dir / sub / str(args.get("cred_file") or "cred.json")
        if p.is_file():
            p.unlink()
            gone.append(str(p))
    for d in (run_dir, req_dir):
        for pattern in args.get("credentials") or ():
            parent, name = pattern.rsplit("/", 1)
            where = d / parent
            if where.is_symlink() or not where.is_dir():
                continue
            for p in where.glob(name):
                if not p.is_dir() or p.is_symlink():
                    p.unlink(missing_ok=True)
                    gone.append(str(p))
    lock = run_dir.parent / ".exclusive.lock"
    try:
        if lock.read_text().strip() == run_dir.name:
            lock.unlink()
            gone.append(str(lock))
    except OSError:
        pass
    return gone


def withdraw(args: dict[str, Any], procs: Procs | None = None) -> dict[str, Any]:
    procs = procs or Procs()
    run_dir = Path(args["run_dir"])
    if not run_dir.is_dir():
        return {"outcome": "absent", "why": f"no {run_dir} on the machine: its launch never claimed it"}
    if out := _started(run_dir):
        return out
    launch = _read(run_dir / "launch.json") or {}
    req_dir = Path(launch["request"]).parent if launch.get("request") else run_dir.parent.parent / "requests" / run_dir.name
    if not req_dir.is_dir():
        return _no_hold(procs, launch)
    with open(req_dir / LAUNCH_LOCK, "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"outcome": "busy", "why": "its launch-request is still answering the launch"}
        return _withdraw_locked(args, procs, run_dir, req_dir)


def _no_hold(procs: Procs, launch: dict[str, Any]) -> dict[str, Any]:
    runner = launch.get("runner_pid")
    return {"outcome": "no-hold", "runner_pid": runner, "runner_alive": isinstance(runner, int) and procs.stat(runner) is not None}


def _withdraw_locked(args: dict[str, Any], procs: Procs, run_dir: Path, req_dir: Path) -> dict[str, Any]:
    if out := _started(run_dir):
        return out
    launch = _read(run_dir / "launch.json") or {}
    acks = {str(req_dir / LAUNCH_ACK), os.path.realpath(req_dir / LAUNCH_ACK)}
    found = [(pid, h) for pid in procs.pids() if (h := hold_of(procs.argv(pid), acks))]
    if not found:
        return _no_hold(procs, launch)
    held: list[dict[str, Any]] = []
    moved: list[int] = []

    def resume(pids: list[int]) -> None:
        for p in pids:
            try:
                procs.kill(p, signal.SIGCONT)
            except ProcessLookupError:
                pass

    for pid, _ in found:
        try:
            procs.kill(pid, signal.SIGSTOP)
        except ProcessLookupError:
            continue
        except PermissionError:
            resume([h["pid"] for h in held])
            return {"outcome": "unstoppable", "pid": pid, "why": f"this user may not signal pid {pid}; nothing was ended"}
        if not procs.stopped(pid):
            if procs.stat(pid) is None:
                continue
            resume([pid, *(h["pid"] for h in held)])
            return {"outcome": "unstoppable", "pid": pid, "why": f"pid {pid} did not stop on SIGSTOP; nothing was ended"}
        if (h := hold_of(procs.argv(pid), acks)) and (st := procs.stat(pid)):
            held.append({"pid": pid, "hold": h, "starttime_ticks": st[1]})
        else:
            moved.append(pid)
    out = _started(run_dir)
    if moved or out:
        resume([*moved, *(h["pid"] for h in held)])
        return {**(out or {"outcome": "started", "state": None}), "moved_on": moved}
    if not held:
        return _no_hold(procs, launch)
    reason = f"{args['reason']} (research cancel by {args['by']})"
    target = {"pids": [h["pid"] for h in held], "holds": held, "ack": str(req_dir / LAUNCH_ACK)}
    _event(run_dir, {"kind": "cancel_intent", "actor": "manual", "reason": reason, "signal": int(signal.SIGTERM), "target": target,
                     "verified": True, "requested_utc": utc_now_ms(), "writer_pid": os.getpid(), "controller": "research.withdraw",
                     "fallback_record": False})
    fds = {h["pid"]: procs.pidfd(h["pid"]) for h in held}
    for h in held:
        try:
            procs.kill(h["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
    resume([h["pid"] for h in held])
    left = [h["pid"] for h in held if not procs.gone(h["pid"], fds[h["pid"]], EXIT_WAIT_S)]
    _event(run_dir, {"kind": "cancel_result", "actor": "manual", "signal": int(signal.SIGTERM), "target": target, "delivered": True,
                     "errno": 0, "delivered_utc": utc_now_ms(), "target_still_present": left})
    if left:
        return {"outcome": "survived", "pids": left, "holds": held}
    if out := _started(run_dir):
        return {**out, "holds": held}
    doc = {"run_id": run_dir.name, "reason": args["reason"], "fault": FAULT, "at_utc": utc_now_ms(), "by": args["by"], "holds": held}
    _write_atomic(run_dir / "refused.json", doc)
    released = _release(run_dir, req_dir, args)
    try:
        with open(run_dir / "launcher.log", "a") as log:
            log.write(f"research cancel: withdrawn from its {', '.join(h['hold'] for h in held)} (pid "
                      f"{', '.join(str(h['pid']) for h in held)}): {reason}; released {len(released)} file(s)\n")
    except OSError:
        pass
    return {"outcome": "withdrawn", "holds": held, "refused": doc, "released": released}


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python3 withdraw.py ARGS_JSON", file=sys.stderr)
        return 2
    print(json.dumps(withdraw(json.loads(argv[0]))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
