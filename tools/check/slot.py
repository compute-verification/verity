#!/usr/bin/env python3
"""`python3 tools/check/slot.py [--priority] [--by NAME] -- CMD...`: CMD in a free merge-check slot of this machine, the way the lander's train checks run.

`check.py --record --on MACHINE` starts its check through this, so a lane's own check queues behind the train checks for a slot
instead of spreading over every core the checks share (vy-nebius-1, 1 Oct: two unslotted checks on 0-127 at load 137).

A machine's slots are SLOTS (`/workspace/research/locks/slots`; CHECK_SLOTS overrides), one per line:

    NAME CPUS [windows=FILE] [lead=MINUTES]      # e.g. `a 32-63`, `d 0-47 windows=/workspace/pouw/fill/windows lead=30`

The first slot whose lock (`check-NAME.lock` beside SLOTS, the lock the lander's `flock` takes) is free and no window blocks is
taken. Its lock stays held by CMD (an inherited descriptor), and CMD runs pinned to CPUS with that slot's test cache
(`~/.cache/verity/tests-slot-NAME`), without LEASE_DIR and with LEASE_DEADLINE_FILE pointing nowhere. A slot with windows= is
skipped while one of FILE's windows (`START_UTC MINUTES`, the node-2 fill runner's format) is open or starts within lead (30)
minutes: a check there would run into a whole-node timed measurement. With every slot busy it retries every POLL_S, reading
SLOTS again each time, so a slot added while checks wait is taken by them; with no slots file, CMD runs as it is. A slot
with a core outside this process's own (os.sched_getaffinity) is skipped, since pinning onto it fails. Stdlib only: it runs before uv.

Waiters take slots in the order they came: each holds a locked ticket in LINE beside SLOTS while it waits, and only the oldest
live one may take a slot (a ticket nobody holds is a waiter that died, and is removed). The lander's own `flock` of a slot's
lock is not in the line: it takes that slot whenever it frees.

`python3 tools/check/slot.py --priority -- CMD...`: the lander's train checks (`train.sh`). A priority waiter's ticket sorts
before every lane waiter's, so it takes the next slot that frees ahead of the lanes' checks already waiting; it never stops a
running check, and priority waiters keep their own arrival order. Merges unblock everyone (root, 2 Oct).

`python3 tools/check/slot.py --status`: one line per slot, with who holds its lock (the pid in /proc/locks, when that process
started, its command) or `free`, and the window that blocks it, if any; then one line per live waiter, a priority one marked
so. A check that came through here also names its run (RESEARCH_RUN_ID), the tree it checks (its working directory, which
`research run` names by commit) and who asked (`--by NAME`, else RESEARCH_AGENT_URL): a waiter in its ticket, the holder in
`check-NAME.who` beside the lock, shown only while its pid holds the lock, so a hand-run `flock` is never shown as the check that
came before it. A waiting check prints the same lines, the waiters before it only, each time its reason to wait changes, so its
log says whom it waits behind.
"""
from __future__ import annotations

import fcntl
import json
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


def holder(lock: Path, proc: Path = Path("/proc")) -> int | None:
    """The pid whose flock holds `lock`, from /proc/locks (a waiter's line has `->` before FLOCK), or None while it's free."""
    try:
        st = lock.stat()
    except FileNotFoundError:
        return None
    want = (os.major(st.st_dev), os.minor(st.st_dev), st.st_ino)
    for line in (proc / "locks").read_text().splitlines():
        f = line.split()
        if len(f) > 5 and f[1] == "FLOCK" and f[5].count(":") == 2:
            maj, mn, ino = f[5].split(":")
            if (int(maj, 16), int(mn, 16), int(ino)) == want:
                return int(f[4])
    return None


def started(pid: int, proc: Path = Path("/proc")) -> datetime | None:
    """When `pid` started: its stat's starttime (clock ticks after boot) plus the boot time."""
    try:
        ticks = int((proc / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19])
        btime = next(int(ln.split()[1]) for ln in (proc / "stat").read_text().splitlines() if ln.startswith("btime "))
    except (OSError, ValueError, IndexError, StopIteration):
        return None
    return datetime.fromtimestamp(btime + ticks / os.sysconf("SC_CLK_TCK"), timezone.utc)


def asker(by: str | None = None) -> dict:
    """Who asked for a slot: this process, the run it belongs to, the tree it checks and who (--by, else RESEARCH_AGENT_URL)."""
    return {"pid": os.getpid(), "run": os.environ.get("RESEARCH_RUN_ID"), "tree": os.getcwd(),
            "by": by or os.environ.get("RESEARCH_AGENT_URL")}


def describe(who: dict) -> str:
    parts = [f"run {who['run']}" if who.get("run") else "", f"tree {Path(who['tree']).name}" if who.get("tree") else "",
             f"by {who['by']}" if who.get("by") else ""]
    return ", ".join(p for p in parts if p)


def _read(p: Path) -> dict:
    try:
        return json.loads(p.read_text() or "{}")
    except (OSError, ValueError):
        return {}


def status(slots: list[dict], lock_dir: Path, now: datetime, proc: Path = Path("/proc"), before: str | None = None) -> list[str]:
    """`--status`'s lines; with `before` (a ticket's name), only the waiters that came before it."""
    out = []
    for slot in slots:
        pid, held = holder(lock_dir / f"check-{slot['name']}.lock", proc), "free"
        if pid is not None:
            t = started(pid, proc)
            try:
                cmd = (proc / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
            except OSError:
                cmd = "(gone)"
            since = f" since {t:%Y-%m-%dT%H:%MZ} ({(now - t).total_seconds() / 60:.0f} min)" if t else ""
            who = _read(lock_dir / f"check-{slot['name']}.who")
            held = f"held by pid {pid}{since}" + (f" ({describe(who)})" if who.get("pid") == pid and describe(who) else "") + f": {cmd[:200]}"
        why = blocked(slot, now)
        out.append(f"{slot['name']}  cores {slot['cpus']}  {held}" + (f"; blocked: {why}" if why else ""))
    line = lock_dir / LINE
    for p in sorted(line.iterdir()) if line.is_dir() else []:
        if before is not None and p.name >= before:
            break
        if p.name.startswith("."):
            continue
        try:
            fd = os.open(p, os.O_RDONLY)
        except FileNotFoundError:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            continue                                                    # nobody holds it: a waiter that died
        except BlockingIOError:
            pass
        finally:
            os.close(fd)
        t = datetime.fromtimestamp(int(p.name.lstrip(PRIORITY).split("-")[0]) / 1e9, timezone.utc)
        who = _read(p)
        out.append(f"waiting since {t:%Y-%m-%dT%H:%MZ} ({(now - t).total_seconds() / 60:.0f} min)"
                   + (" with priority" if p.name.startswith(PRIORITY) else "") + f": pid {who.get('pid', p.name.split('-')[-1])}"
                   + (f" ({describe(who)})" if describe(who) else ""))
    return out


#: a priority waiter's ticket starts with this, which sorts before every digit (and before "." of a ticket being made)
PRIORITY = "!"


def join(lock_dir: Path, priority: bool = False, who: dict | None = None) -> tuple[Path, int]:
    """This waiter's ticket in the line and the descriptor holding its lock; named by arrival (after every priority ticket
    unless it is one), and locked before it is visible."""
    d = lock_dir / LINE
    d.mkdir(exist_ok=True)
    name = f"{PRIORITY if priority else ''}{time.time_ns():020d}-{os.getpid()}"
    fd = os.open(d / f".{name}", os.O_RDWR | os.O_CREAT, 0o664)
    fcntl.flock(fd, fcntl.LOCK_EX)
    if who:
        os.write(fd, (json.dumps(who) + "\n").encode())
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


def allowed() -> set[int]:
    return os.sched_getaffinity(0)


def usable(slots: list[dict]) -> list[dict]:
    """The slots whose every core this process may run on. Pinning onto any other fails (EINVAL), so a check never takes one:
    on vy-nebius-1 a `research run` session is held to cores 0-127, and slot d on 128-159 killed every check that took it
    (2 Oct 07:20Z)."""
    mine = allowed()
    return [s for s in slots if cpulist(s["cpus"]) <= mine]


def reread(slots: list[dict]) -> list[dict]:
    """The slots file as it is now, so a slot added while a check waits is one it can take; the last list it read when the file
    can't be read or lists none (mid-rewrite)."""
    try:
        return parse(SLOTS.read_text()) or slots
    except OSError:
        return slots


def main(argv: list[str], now=lambda: datetime.now(timezone.utc), sleep=time.sleep) -> int:
    if argv[:1] == ["--status"]:
        if not SLOTS.exists():
            print(f"no check slots on this machine ({SLOTS})")
            return 0
        print("\n".join(status(parse(SLOTS.read_text()), SLOTS.parent, now())))
        return 0
    opts, cmd = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else ([], argv)
    by = opts[opts.index("--by") + 1] if "--by" in opts[:-1] else None
    if not cmd or set(opts) - {"--priority", "--by", by}:
        sys.exit("usage: slot.py [--priority] [--by NAME] -- CMD...")
    if not SLOTS.exists():
        print(f"slot: no check slots on this machine ({SLOTS}); running unpinned", file=sys.stderr, flush=True)
        os.execvp(cmd[0], cmd)
    listed = parse(SLOTS.read_text())
    if not listed:
        sys.exit(f"slot: {SLOTS} lists no slots")
    slots = usable(listed)
    if len(slots) < len(listed):
        out = ", ".join(f"{s['name']} ({s['cpus']})" for s in listed if s not in slots)
        print(f"slot: skipping {out}: outside this process's cores", file=sys.stderr, flush=True)
    if not slots:
        sys.exit(f"slot: none of {SLOTS}'s slots is within this process's cores")
    ticket, tfd = join(SLOTS.parent, priority="--priority" in opts, who=asker(by))
    told = None
    while True:
        slots = usable(reread(slots)) or slots
        n = ahead(ticket)
        got = None if n else take(slots, SLOTS.parent, now())
        if got:
            break
        why = f"{n} check(s) ahead in line" if n else \
            "every check slot is taken (" + "; ".join(f"{s['name']}: {blocked(s, now()) or 'busy'}" for s in slots) + ")"
        if why != told:
            print(f"slot: {why}; retrying every {POLL_S} s", file=sys.stderr, flush=True)
            for ln in status(slots, SLOTS.parent, now(), before=ticket.name):
                print(f"slot:   {ln}", file=sys.stderr, flush=True)
            told = why
        sleep(POLL_S)
    ticket.unlink()
    os.close(tfd)
    slot, fd = got
    who = SLOTS.parent / f"check-{slot['name']}.who"
    tmp = who.with_name(f".{who.name}.{os.getpid()}")
    tmp.write_text(json.dumps(asker(by)) + "\n")
    tmp.chmod(0o664)
    os.replace(tmp, who)
    os.set_inheritable(fd, True)
    os.sched_setaffinity(0, cpulist(slot["cpus"]))
    env = {k: v for k, v in os.environ.items() if k != "LEASE_DIR"}
    env.update(LEASE_DEADLINE_FILE="/nonexistent/research-deadline", CHECK_SLOT=slot["name"],
               VERITY_TEST_CACHE=str(Path.home() / ".cache" / "verity" / f"tests-slot-{slot['name']}"))
    print(f"slot: check slot {slot['name']} (cores {slot['cpus']})", file=sys.stderr, flush=True)
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
