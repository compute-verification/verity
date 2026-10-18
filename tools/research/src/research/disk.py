"""Disk admission for a queued run (`research run --queue --disk-gb N`), on the machine, ahead of the run's allocation.

`research disk-hold --gb N --path P -- CMD...` admits CMD at once when the filesystem holding P would stay at or under HOLD_PCT
(80, node 1's `vy-disk-guard` level) of its size with N more GB used and no run is held for it; otherwise it holds CMD in a FIFO
queue, then execs CMD in its own process, so the run's pid is unchanged. The launch doesn't wait: the hold happens in the detached
process the machine starts, and its lines go to the run's launcher.log. With `--ack FIFO` (the launch's, remote.LAUNCH_ACK) a hold
acknowledges the launch as it starts, since the runner's own acknowledgement comes only after it. Requests other runs have declared
but not yet used aren't counted (a follow-up). A machine whose guard holds at another level sets RESEARCH_DISK_HOLD_PCT in the run's
environment.

The queue is P/.disk-hold/ (mode 1777, best effort: runs may be different users): one ticket per held run, a file named
`<ns>.<pid>` (when it was taken, and its pid) holding its GB and the process's start in /proc, and every decision taken under flock
on its `lock`. A held run is released when its ticket is the oldest live one, its N GB fit under HOLD_PCT, and no run was admitted
on the filesystem in the last 2 * POLL_S (the time in `last-release`): so held runs go in one at a time, in order, each starting to
write before the next is judged. A ticket whose pid is dead, or now names another process, is ignored and removed, so a killed
hold leaves nothing to wait on. A held run is released at HOLD_PCT itself, with no lower release mark: retention gc (retention.py) acts
only over HOLD_PCT and stops at RELEASE_PCT, so on a disk it can't take under RELEASE_PCT a release mark there starved every held
run while new runs were admitted at once (vy-nebius-2, 18 Oct, 54 runs held at 76-80%). The pacing does what the hysteresis did.
"""
from __future__ import annotations

import argparse
import fcntl
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Callable

HOLD_PCT = 80
RELEASE_PCT = 75   # retention gc's low-water mark (retention.py); not a release mark here (module docstring)
POLL_S = 30
GB = 1e9
QUEUE_DIR = ".disk-hold"
TICKET = re.compile(r"(\d+)\.(\d+)")


def started(pid: int) -> str:
    """`pid`'s start time in clock ticks since boot (/proc/PID/stat), "" where unreadable."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return ""


def alive(pid: int, start: str) -> bool:
    """Whether `pid` runs and, when its ticket recorded a `start`, is still that process (not its pid reused)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return not start or started(pid) in ("", start)


def _queue(q: Path) -> int:
    """The queue directory made, and its lock's fd."""
    q.mkdir(parents=True, exist_ok=True)
    try:
        q.chmod(0o1777)
    except OSError:
        pass
    fd = os.open(q / "lock", os.O_RDONLY | os.O_CREAT, 0o666)
    try:
        os.fchmod(fd, 0o666)
    except OSError:
        pass
    return fd


def live_tickets(q: Path, *, is_alive: Callable[[int, str], bool] = alive) -> list[Path]:
    """The queue's live tickets, oldest first; dead ones are removed (when this user may)."""
    live = []
    for p in q.iterdir():
        if not (m := TICKET.fullmatch(p.name)):
            continue
        try:
            start = (p.read_text().split()[1:] or [""])[0]
        except OSError:
            continue
        if is_alive(int(m[2]), start):
            live.append((int(m[1]), int(m[2]), p))
        else:
            try:
                p.unlink()
            except OSError:
                pass
    return [p for *_, p in sorted(live)]


def _last_release(q: Path) -> int | None:
    try:
        return int((q / "last-release").read_text())
    except (OSError, ValueError):
        return None


def _note_release(q: Path, now: int) -> None:
    try:
        fd = os.open(q / "last-release", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o666)
    except OSError:
        return
    try:
        os.fchmod(fd, 0o666)
    except OSError:
        pass
    try:
        os.write(fd, str(now).encode())
    finally:
        os.close(fd)


def hold(gb: float, path: Path, *, hold_pct: float = HOLD_PCT, poll_s: float = POLL_S,
         usage: Callable[[Path], tuple[int, int, int]] = shutil.disk_usage, sleep: Callable[[float], None] = time.sleep,
         log: Callable[[str], None] = lambda s: print(s, file=sys.stderr, flush=True),
         on_hold: Callable[[], None] = lambda: None, clock: Callable[[], int] = time.time_ns,
         is_alive: Callable[[int, str], bool] = alive, pid: int | None = None, queue: Path | None = None) -> int:
    """Return once `gb` more GB may go on `path`'s filesystem by the rule above; the number of polls it was held for.  `on_hold`
    runs once, when the hold starts; `clock` is in ns; `queue` defaults to path/QUEUE_DIR."""
    q = queue or path / QUEUE_DIR
    lock = _queue(q)

    def after() -> float:
        total, used, _ = usage(path)
        return 100 * (used + gb * GB) / total

    ticket = None
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            pct, ahead = after(), live_tickets(q, is_alive=is_alive)
            if pct <= hold_pct and not ahead:
                _note_release(q, clock())
                return 0
            pid = os.getpid() if pid is None else pid
            ticket = q / f"{clock()}.{pid}"
            fd = os.open(ticket, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            try:
                os.write(fd, f"{gb:g} {started(pid)}\n".encode())
            finally:
                os.close(fd)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
        why = f"{len(ahead)} held run(s) ahead" if ahead else f"{gb:g} GB more would fill {path}'s filesystem to {pct:.1f}% (over {hold_pct:g}%)"
        log(f"research disk-hold: holding: {why}; released in turn once {gb:g} GB more fits under {hold_pct:g}%, one run per "
            f"{2 * poll_s:g} s, polled every {poll_s:g} s")
        on_hold()
        held = 0
        while True:
            held += 1
            sleep(poll_s)
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                if live_tickets(q, is_alive=is_alive)[:1] != [ticket]:
                    continue
                now, last = clock(), _last_release(q)
                if last is not None and now - last < 2 * poll_s * 1e9:
                    continue
                if (pct := after()) > hold_pct:
                    continue
                _note_release(q, now)
                ticket.unlink(missing_ok=True)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
            log(f"research disk-hold: released after {held} poll(s): {pct:.1f}% with {gb:g} GB more")
            return held
    finally:
        if ticket is not None:
            ticket.unlink(missing_ok=True)
        os.close(lock)


def main(argv: list[str], *, usage: Callable[[Path], tuple[int, int, int]] = shutil.disk_usage,
         sleep: Callable[[float], None] = time.sleep, execvp: Callable[[str, list[str]], None] = os.execvp) -> int:
    ap = argparse.ArgumentParser(prog="research disk-hold")
    ap.add_argument("--gb", type=float, required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--ack", default=None, metavar="FIFO", help="the launch's acknowledgement FIFO, told when a hold starts")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    ns = ap.parse_args(argv)
    cmd = ns.cmd[1:] if ns.cmd[:1] == ["--"] else ns.cmd
    if not cmd:
        print("research disk-hold: give the command after --", file=sys.stderr)
        return 2

    def on_hold() -> None:
        if ns.ack:
            from .remote import ack_launch

            ack_launch(ns.ack, "disk-hold")

    pct = float(os.environ.get("RESEARCH_DISK_HOLD_PCT", HOLD_PCT))
    hold(ns.gb, Path(ns.path), hold_pct=pct, usage=usage, sleep=sleep, on_hold=on_hold)
    execvp(cmd[0], cmd)
    return 127


__all__ = ["HOLD_PCT", "POLL_S", "QUEUE_DIR", "RELEASE_PCT", "alive", "hold", "live_tickets", "main", "started"]
