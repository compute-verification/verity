#!/usr/bin/env python3
"""`python3 tools/check/slot.py -- CMD...`: CMD in a free merge-check slot of this machine, the way the lander's train checks run.

`check.py --record --on MACHINE` starts its check through this, so a lane's own check queues behind the train checks for a slot
instead of spreading over every core the checks share (vy-nebius-1, 1 Oct: two unslotted checks on 0-127 at load 137).

A machine's slots are SLOTS (`/workspace/research/locks/slots`; CHECK_SLOTS overrides), one per line:

    NAME CPUS [windows=FILE] [lead=MINUTES]      # e.g. `a 32-63`, `d 0-47 windows=/workspace/pouw/fill/windows lead=30`

The first slot whose lock (`check-NAME.lock` beside SLOTS, the lock the lander's `flock` takes) is free and no window blocks is
taken. Its lock stays held by CMD (an inherited descriptor), and CMD runs pinned to CPUS with that slot's test cache
(`~/.cache/verity/tests-slot-NAME`), without LEASE_DIR and with LEASE_DEADLINE_FILE pointing nowhere. A slot with windows= is
skipped while one of FILE's windows (`START_UTC MINUTES`, the node-2 fill runner's format) is open or starts within lead (30)
minutes: a check there would run into a whole-node timed measurement. With every slot busy it retries every POLL_S; with no
slots file, CMD runs as it is. Stdlib only: it runs before uv.

Waiters take slots in the order they came: each holds a locked ticket in LINE beside SLOTS while it waits, and only the oldest
live one may take a slot (a ticket nobody holds is a waiter that died, and is removed). The lander's own `flock` of a slot's
lock is not in the line: it takes that slot whenever it frees.
"""
from __future__ import annotations

import fcntl
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

SLOTS = Path(os.environ.get("CHECK_SLOTS", "/workspace/research/locks/slots"))
LINE = "check-line"
POLL_S = 20


def cpulist(s: str) -> set[int]:
    out: set[int] = set()
    for part in s.split(","):
        a, _, b = part.partition("-")
        out |= set(range(int(a), int(b or a) + 1))
    return out


def parse(text: str) -> list[dict]:
    slots = []
    for line in text.splitlines():
        f = line.split("#")[0].split()
        if len(f) < 2:
            continue
        opts = dict(x.split("=", 1) for x in f[2:])
        slots.append({"name": f[0], "cpus": f[1], "windows": opts.get("windows"), "lead": int(opts.get("lead", 30))})
    return slots


def blocked(slot: dict, now: datetime) -> str | None:
    """Why `slot` can't start a check at `now` (a window open or within its lead), or None."""
    if not slot["windows"]:
        return None
    for line in Path(slot["windows"]).read_text().splitlines():
        f = line.split("#")[0].split()
        if len(f) < 2:
            continue
        start = datetime.fromisoformat(f[0].replace("Z", "+00:00"))
        if start - timedelta(minutes=slot["lead"]) < now < start + timedelta(minutes=int(f[1])):
            return f"window {f[0]} ({f[1]} min) is open or starts within {slot['lead']} min"
    return None


def take(slots: list[dict], lock_dir: Path, now: datetime) -> tuple[dict, int] | None:
    """The first free, unblocked slot and the descriptor holding its lock, or None."""
    for slot in slots:
        if blocked(slot, now):
            continue
        fd = os.open(lock_dir / f"check-{slot['name']}.lock", os.O_RDWR | os.O_CREAT, 0o664)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            continue
        return slot, fd
    return None


def join(lock_dir: Path) -> tuple[Path, int]:
    """This waiter's ticket in the line and the descriptor holding its lock; named by arrival, and locked before it is visible."""
    d = lock_dir / LINE
    d.mkdir(exist_ok=True)
    name = f"{time.time_ns():020d}-{os.getpid()}"
    fd = os.open(d / f".{name}", os.O_RDWR | os.O_CREAT, 0o664)
    fcntl.flock(fd, fcntl.LOCK_EX)
    os.rename(d / f".{name}", d / name)
    return d / name, fd


def ahead(ticket: Path) -> int:
    """How many live waiters came before `ticket`'s; a dead waiter's ticket is removed."""
    n = 0
    for p in sorted(ticket.parent.iterdir()):
        if p.name >= ticket.name:
            break
        if p.name.startswith("."):
            continue
        try:
            fd = os.open(p, os.O_RDWR)
        except FileNotFoundError:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            p.unlink(missing_ok=True)
        except BlockingIOError:
            n += 1
        finally:
            os.close(fd)
    return n


def main(argv: list[str], now=lambda: datetime.now(timezone.utc), sleep=time.sleep) -> int:
    cmd = argv[argv.index("--") + 1:] if "--" in argv else argv
    if not cmd:
        sys.exit("usage: slot.py -- CMD...")
    if not SLOTS.exists():
        print(f"slot: no check slots on this machine ({SLOTS}); running unpinned", file=sys.stderr, flush=True)
        os.execvp(cmd[0], cmd)
    slots = parse(SLOTS.read_text())
    if not slots:
        sys.exit(f"slot: {SLOTS} lists no slots")
    ticket, tfd = join(SLOTS.parent)
    told = None
    while True:
        n = ahead(ticket)
        got = None if n else take(slots, SLOTS.parent, now())
        if got:
            break
        why = f"{n} check(s) ahead in line" if n else \
            "every check slot is taken (" + "; ".join(f"{s['name']}: {blocked(s, now()) or 'busy'}" for s in slots) + ")"
        if why != told:
            print(f"slot: {why}; retrying every {POLL_S} s", file=sys.stderr, flush=True)
            told = why
        sleep(POLL_S)
    ticket.unlink()
    os.close(tfd)
    slot, fd = got
    os.set_inheritable(fd, True)
    os.sched_setaffinity(0, cpulist(slot["cpus"]))
    env = {k: v for k, v in os.environ.items() if k != "LEASE_DIR"}
    env.update(LEASE_DEADLINE_FILE="/nonexistent/research-deadline", CHECK_SLOT=slot["name"],
               VERITY_TEST_CACHE=str(Path.home() / ".cache" / "verity" / f"tests-slot-{slot['name']}"))
    print(f"slot: check slot {slot['name']} (cores {slot['cpus']})", file=sys.stderr, flush=True)
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
