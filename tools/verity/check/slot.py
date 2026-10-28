#!/usr/bin/env python3
"""`python3 tools/verity/check/slot.py [--priority] [--by NAME] [--twin PAIR[@HOME]] [--wait-max S] [--run-max S] -- CMD...`: CMD in a free merge-check slot of this machine, the way the lander's train checks run.

`check.py --record --on MACHINE` starts its check through this, so a lane's own check queues behind the train checks for a slot
instead of spreading over every core the checks share (vy-nebius-1, 1 Oct: two unslotted checks on 0-127 at load 137).

A machine's slots are SLOTS (`/workspace/research/locks/slots`; CHECK_SLOTS overrides), one per line:

    NAME CPUS [windows=FILE] [lead=MINUTES] [only=trains] [mem=GB] [scope=1]   # e.g. `a 32-63 mem=560`, `d 0-47 windows=/workspace/pouw/fill/windows lead=30`

The first slot whose lock (`check-NAME.lock` beside SLOTS, the lock the lander's `flock` takes) is free and no window blocks is
taken. Its lock stays held by CMD (an inherited descriptor), and CMD runs pinned to CPUS with that slot's test cache
(`~/.cache/verity/tests-slot-NAME`), without LEASE_DIR and with LEASE_DEADLINE_FILE pointing nowhere. A slot with windows= is
skipped while one of FILE's windows (`START_UTC MINUTES`, the node-2 fill runner's format) is open or starts within lead (30)
minutes: a check there would run into a whole-node timed measurement. A slot with only=trains is taken only by a `--priority` waiter
(the lander's train checks), so a lane's long check or quick tier never holds the slot a train needs (a slot.py from before it
reads the key as unknown and the slot as any other). A slot with mem=GB is skipped while less than GB is free to this process
(`memory_gb`: the least of MemAvailable and what each cgroup v2 memory limit over it leaves, the user slice's on a VM, less what
live research.mem reservations hold but don't use yet, `reserved_gb`: for a `--priority` waiter, or on an only=trains slot, only
other priority checks' reservations, so a train never waits behind a research job's unused reservation), so a
check starts only when its measured peak fits and queues instead of being killed with the others under the slice (node 1,
8 Oct 02:00Z: two OOM kills at 1,605 of 1,716 GB across ~8 runs), and CMD gets GB as SLOT_MEM_ENV, which check.py makes
circuit-check's budget, so two checks' circuit-checks never each budget all the memory free when they began (node 2, 10 Oct
20:06Z: two cold ones, each at its six 125-151 GiB targets, filled the 1600 GiB slice); while the check runs, the slot holds a
research.mem reservation of GB (`reserve`), so `research run --mem-gb` admits nothing that needs what the check may still take.
With every slot busy it retries every POLL_S, reading
SLOTS again each time, so a slot added while checks wait is taken by them; with no slots file, CMD runs as it is, unless
CHECK_SLOTS named that file: then it exits NOT_STARTED, naming the pools the machine has (a run sent to a pool the node lacks
would run unpinned beside its GPU jobs). A slot
with a core outside this process's own (os.sched_getaffinity) is skipped, since pinning onto it fails. Stdlib only: it runs before uv.

Waiters take slots in the order they came: each holds a locked ticket in LINE beside SLOTS while it waits, and only the oldest
live one may take a slot (a ticket nobody holds is a waiter that died, and is removed). The lander's own `flock` of a slot's
lock is not in the line: it takes that slot whenever it frees.

`python3 tools/verity/check/slot.py --priority -- CMD...`: the lander's train checks (`train.sh`). A priority waiter's ticket sorts
before every lane waiter's, so it takes the next slot that frees ahead of the lanes' checks already waiting; it never stops a
running check, and priority waiters keep their own arrival order. Merges unblock everyone. With `--rank N`
(drive's train checks: the train's first-parent depth, so the lowest train's is the least), priority
waiters go by rank, then arrival: the train that lands next takes the next slot, ahead of trains above it that came first. A priority waiter without a rank (drive's post-landing run) goes after the ranked.

`--wait-max SECONDS`: a waiter that has no slot after SECONDS leaves the line and exits NOT_STARTED (75) without running CMD, so
its caller can tell a check that never started from one that failed. `--run-max SECONDS`: CMD's time limit, counted from the slot
being taken: inside a `research run` (RESEARCH_RUN_DIR), the run's timeout, which counted the wait too, becomes the time it has
run so far plus SECONDS, through its harness's control file (research.telemetry.run's CONTROL_TIMEOUT). `research queue ready
--on` sends both (#1344, #1282, #1314 timed out in node 1's line and were reported failed).

`--twin PAIR` / `--twin PAIR@HOME`: one of a check's twins, the same check launched on each machine it may run on when none had a
free slot (`check.py --record --on auto`), so it takes the first slot that frees on any of them (9 Oct: four checks waited up to
54 min on node 1 while a node 2 slot was free). Each waits in its own machine's line. The one holding a slot runs only if it
claims PAIR first: `mkdir TWINS/PAIR` beside SLOTS on this machine (`--twin PAIR`) or on HOME, over the ssh PEERS names for it.
Its twins, finding the claim (each poll here, every TWIN_PEEK_S from another machine, and whenever one holds a slot), leave
their line and exit NOT_STARTED. A claim that can't be made (HOME unreachable or not in PEERS) also exits NOT_STARTED, so a
pair never runs twice: the twin on HOME still runs. It fails closed: when HOME's own claim can't be written, or HOME is gone,
no twin runs, and every one exits NOT_STARTED, which `check.py --record` returns as "rerun".

`python3 tools/verity/check/slot.py --status [--json]`: one line per slot, with who holds its lock (the pid in /proc/locks, when that
process started, its command) or `free`, and the window that blocks it, if any, then a line for each of its windows open now or
starting within AHEAD_H (6) hours (start, minutes, its comment's first words); then one line per live waiter, a priority one
marked so. --json prints the same as one JSON object (`status_doc`), which `check.py --slots` reads from every machine. A
check that came through here also names its run (RESEARCH_RUN_ID), the tree it checks (its working directory, which
`research run` names by commit) and who asked (`--by NAME`, else RESEARCH_AGENT_URL): a waiter in its ticket, the holder in
`check-NAME.who` beside the lock, shown only while its pid holds the lock, so a hand-run `flock` is never shown as the check that
came before it. A waiting check prints the same lines, the waiters before it only, each time its reason to wait changes, so its
log says whom it waits behind. Inside a `research run` (RESEARCH_RUN_DIR) it also keeps the run's RUN_SLOT: waiting (since when and
why), held (which slot, since when) or left (why), so `research status` shows a run whose check waits here as `waiting`, not running.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

SLOTS = Path(os.environ.get("CHECK_SLOTS", "/workspace/research/locks/slots"))
#: a lane's own test run, which no landing needs (a PR lands on an eyeball review)
LANE_RUN = ('`research run --on vy-nebius-cpu-K --cwd source -- bash -lc "uv sync -q --all-packages --extra torch-cpu && '
            'uv run tools/verity/check/suites.py --changed $(git rev-parse origin/main) --quick"`')
LINE = "check-line"
POLL_S = 20
#: the exit of a waiter that gave up (`--wait-max`): CMD never ran
NOT_STARTED = 75
#: the run's timeout control file, relative to RESEARCH_RUN_DIR (research.telemetry.run.CONTROL_TIMEOUT)
RUN_TIMEOUT = Path("control") / "timeout_s"
#: the run's slot record, relative to RESEARCH_RUN_DIR (research.runs.SLOT_FILE): whether CMD waits in the line, holds a slot or left
RUN_SLOT = "slot.json"
#: a slot's mem= as CMD sees it: the GiB its check may take, which check.py gives circuit-check as its --mem-gb
SLOT_MEM_ENV = "CHECK_SLOT_MEM_GB"
#: the check's own scope unit (a `scope=1` slot), which spans.py reads its memory from
SCOPE_ENV = "CHECK_SCOPE_UNIT"
#: the GB (10^9 bytes) a run asks of its slot (`research run --pool --mem-gb`): a slot whose mem= is smaller waits for and reserves
#: this instead (24 Oct: a 450 GB run in proofs' 193 GB slot p5 reserved 193, and the ledger under-reported it)
RUN_MEM_ENV = "CHECK_RUN_MEM_GB"
USAGE = "usage: slot.py [--priority [--rank N]] [--by NAME] [--twin PAIR[@HOME]] [--wait-max SECONDS] [--run-max SECONDS] -- CMD..."
#: the claims of twins (`--twin`), one directory per pair, beside SLOTS on the machine that holds them; a week old, removed
TWINS = "twins"
#: machine name, then the ssh destination and options with which this machine's research user reaches it (`--twin PAIR@HOME`)
PEERS = Path(os.environ.get("CHECK_PEERS", "/etc/vy/peers"))
#: research.mem's reservations (its RAMLOCK_DIR, which `research mem` reads from the same variable)
RAMLOCK = Path(os.environ.get("RESEARCH_RAMLOCK_DIR", "/workspace/ramlock"))
#: how often a waiting twin whose claims are on another machine looks there for its twin's claim
TWIN_PEEK_S = 120
TWIN_SSH_S = 30


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
        slots.append({"name": f[0], "cpus": f[1], "windows": opts.get("windows"), "lead": int(opts.get("lead", 30)),
                      "trains": opts.get("only") == "trains", "mem": int(opts["mem"]) if "mem" in opts else None,
                      "scope": opts.get("scope") == "1"})
    return slots


def lane_pools(slots_file: Path) -> list[Path]:
    """The pools beside `slots_file` (`NAME/slots` in its directory, not a `*.bak-*` copy) with a slot a lane check may take: n2's
    `gpu` and cpu-1's `research` beside their all-trains default pools."""
    out = []
    for p in sorted(slots_file.parent.glob("*/slots")):
        if ".bak-" in p.parent.name:
            continue
        try:
            listed = parse(p.read_text())
        except (OSError, ValueError):
            continue
        if any(not s["trains"] for s in listed):
            out.append(p)
    return out


#: where a Nebius node caps each login session's memory (deploy.toml's 50-vy-job-memory.conf; research.telemetry.cgroup reads the
#: same files)
SESSION_DROPINS = Path("/etc/systemd/system/session-.scope.d")


def session_cap(dropins: Path | None = None) -> str | None:
    """The MemoryMax= every login session scope gets here (the last one set, as systemd reads the drop-ins), or None."""
    cap = None
    try:
        files = sorted((dropins or SESSION_DROPINS).glob("*.conf"))
    except OSError:
        return None
    for f in files:
        try:
            lines = f.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            k, _, v = line.strip().partition("=")
            if k.strip() == "MemoryMax":
                cap = v.strip() or None
    return None if cap in (None, "infinity") else cap


def scoped(cmd: list[str], slot: str, run_id: str) -> tuple[list[str], str]:
    """CMD in a transient systemd user scope of its own (`scope=1` on its slot), and the scope's unit name. `--scope` registers
    this process in it and execs CMD, so the pid and the inherited slot lock stay; the check's whole tree is then one cgroup,
    whose memory.peak spans.py records at each step's end and the check's (a session scope it shares with other runs has no
    per-check peak: 23 Oct, train-slot sizing rested on summed RSS, which counts shared pages over and over). On a node that
    caps each login session's memory the scope gets the same MemoryMax, as the session it leaves had: `research run` refuses
    to start a workload from an uncapped cgroup, which failed every test of the research suite that runs one locally inside
    a scoped check (exit 75, 26 Oct)."""
    unit = f"check-{slot}-{run_id}"
    cap = session_cap()
    props = [f"--property=MemoryMax={cap}"] if cap else []
    return ["systemd-run", "--user", "--scope", "--quiet", "--collect", f"--unit={unit}", *props, "--", *cmd], unit


def scope_refusal(run=subprocess.run) -> str | None:
    """Why `systemd-run --user --scope` can't run here (no user manager or bus: a run started over ssh without
    XDG_RUNTIME_DIR, no lingering), or None when it can. A `scope=1` slot fails open on it: its check runs unscoped rather than
    failing every train check that slot takes."""
    try:
        r = run(["systemd-run", "--user", "--scope", "--quiet", "--collect", "--", "true"], capture_output=True, text=True,
                timeout=10)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"{type(e).__name__}: {e}"
    return None if r.returncode == 0 else f"exit {r.returncode}: {(r.stderr or '').strip()[-200:] or 'no message'}"


def memory_gb(root: Path = Path("/")) -> int:
    """The memory this process may use, in GiB: the least of MemAvailable and what each cgroup v2 limit over it leaves (a run's
    session scope and the user slice on a VM; a pod's /proc/meminfo is the host's). A cgroup's page cache (active_file and
    inactive_file in its memory.stat) is in its memory.current but the kernel reclaims it at the limit, so it counts as left:
    on vy-nebius-cpu-1 273 GB of cache held slot a's 560 GB line shut with 677 GB available (9 Oct 20:50Z)."""
    def read(p: Path) -> str:
        try:
            return p.read_text().strip()
        except OSError:
            return ""

    left = [int(ln.split()[1]) << 10 for ln in read(root / "proc/meminfo").splitlines() if ln.startswith("MemAvailable:")]
    top = root / "sys/fs/cgroup"
    for line in read(root / "proc/self/cgroup").splitlines():
        if line.startswith("0::"):
            d = top / line[3:].strip("/")
            for c in (d, *[p for p in d.parents if p.is_relative_to(top)]):
                if (m := read(c / "memory.max")).isdigit():
                    stat = dict(ln.split()[:2] for ln in read(c / "memory.stat").splitlines() if len(ln.split()) >= 2)
                    cache = sum(int(stat.get(k, 0)) for k in ("active_file", "inactive_file") if stat.get(k, "").isdigit())
                    left.append(int(m) - int(read(c / "memory.current") or 0) + cache)
    return max(0, min(left)) >> 30 if left else 0


#: the run states that end a research.mem reservation (research.mem's TERMINAL)
ENDED = ("done", "failed", "cancelled")


def _rss_tree(pid: int, proc: Path) -> int | None:
    """Resident bytes of ``pid`` and its descendants, or None when it isn't in ``proc`` (research.mem's `_rss_tree`)."""
    parent, rss = {}, {}
    for e in (x for x in os.listdir(proc) if x.isdigit()) if proc.is_dir() else ():
        try:
            parent[int(e)] = int((proc / e / "stat").read_text().rsplit(")", 1)[1].split()[1])
            rss[int(e)] = next((int(ln.split()[1]) * 1024 for ln in (proc / e / "status").read_text().splitlines()
                                if ln.startswith("VmRSS:")), 0)
        except (OSError, IndexError, ValueError):
            continue
    if pid not in parent:
        return None
    tree, todo = {pid}, [pid]
    while todo:
        p = todo.pop()
        kids = [c for c, pp in parent.items() if pp == p and c not in tree]
        tree.update(kids)
        todo += kids
    return sum(rss.get(p, 0) for p in tree)


def reserved_gb(proc: Path = Path("/proc"), priority: bool = False) -> int:
    """GiB that live research.mem reservations in RAMLOCK hold but don't use yet, each run's largest once, as research.mem's
    `reserved_bytes` counts them: a reserved run lowering toward its peak isn't in MemAvailable yet (23 Oct 23:16Z, top: when a
    check ended on cpu-4, proofs' waiting 345 GB shares and a train's twins would race for the freed memory, and a twin admitted
    beside a lowering share reads well over 560 GB free). With `priority` (a train check's gate), only priority waiters'
    reservations (`reserve`'s "priority") count: any other job counts at the memory it holds, which `memory_gb` already left out,
    so trains come first (24 Oct, cpu-1: train slot b free at -91 GB net behind proofs' 279 GB pool reservations using 25 and 5)."""
    by: dict[str, float] = {}
    for f in sorted(RAMLOCK.glob("*.json")) if RAMLOCK.is_dir() else ():
        try:
            doc = json.loads(f.read_text())
            gb = float(doc["gb"]) * 1e9
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if priority and doc.get("priority") is not True:
            continue
        try:
            if doc.get("run_dir") and json.loads((Path(doc["run_dir"]) / "status.json").read_text()).get("state") in ENDED:
                continue
        except (OSError, ValueError, AttributeError):
            pass
        pid = doc.get("pid") if isinstance(doc.get("pid"), int) else None
        if pid is not None and not (proc / str(pid)).exists():
            continue
        used = _rss_tree(pid, proc) if pid is not None else None
        key = doc.get("run_id") or f.name
        by[key] = max(by.get(key, 0.0), max(0.0, gb - min(used or 0, gb)))
    return int(sum(by.values())) >> 30


def reserve(slot: dict, now: datetime, priority: bool = False) -> str | None:
    """A slot with mem=GB holds a research.mem reservation of that size while its check runs: `check-NAME.json` in RAMLOCK, in
    research.mem's format and under its admission lock, naming this process (CMD, after the exec) and its run, and marked
    "priority" for a `--priority` waiter's check (a train's, or drive's post-landing run), which other train checks' gates count in
    full (`reserved_gb`). research.mem counts
    the GB less what the check's process tree holds, and stops counting it when that pid is gone or the run ended. Without it,
    research.mem admitted work beside two full checks until node 1 ran out of memory (8:21Z, 10 Oct). The line to print, or None
    for a slot without mem=."""
    if not slot.get("mem"):
        return None
    gb = round(slot["mem"] * 2 ** 30 / 1e9, 1)                  # mem= is in GiB (`memory_gb`), research.mem's GB are 10^9 bytes
    path = RAMLOCK / f"check-{slot['name']}.json"
    run_dir = os.environ.get("RESEARCH_RUN_DIR")
    doc = {"owner": "check", "gb": gb, "started": f"{now:%Y-%m-%dT%H:%M:%SZ}", "pid": os.getpid(),
           "note": f"check slot {slot['name']}: {os.environ.get('RESEARCH_RUN_ID') or os.getcwd()}",
           **({"priority": True} if priority else {}), **({"run_id": os.environ.get("RESEARCH_RUN_ID"), "run_dir": run_dir} if run_dir else {})}
    try:
        RAMLOCK.mkdir(parents=True, exist_ok=True)
        with open(RAMLOCK / ".admission.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            tmp = path.with_name(f".{path.name}.{os.getpid()}")
            tmp.write_text(json.dumps(doc) + "\n")
            tmp.chmod(0o664)
            os.replace(tmp, path)
    except OSError as e:
        return f"slot: no research.mem reservation for slot {slot['name']} ({e}): research.mem won't count this check's peak"
    return f"slot: holds a {gb:g} GB research.mem reservation ({path}) while the check runs"


def _windows(slot: dict) -> list[tuple[str, datetime, int, str]]:
    """`slot`'s windows as its FILE writes them: (start as written, start, minutes, comment)."""
    out = []
    for line in Path(slot["windows"]).read_text().splitlines() if slot["windows"] else []:
        body, _, note = line.partition("#")
        f = body.split()
        if len(f) >= 2:
            out.append((f[0], datetime.fromisoformat(f[0].replace("Z", "+00:00")), int(f[1]), note.strip()))
    return out


def blocked(slot: dict, now: datetime, priority: bool = False) -> str | None:
    """Why `slot` can't start a check at `now` (a window open or within its lead, or less memory free than its mem=), or None. A
    `priority` waiter's check, or any check of an only=trains slot (which only those take), counts only other priority checks'
    reservations (`reserved_gb`)."""
    for raw, start, minutes, _note in _windows(slot):
        if start - timedelta(minutes=slot["lead"]) < now < start + timedelta(minutes=minutes):
            return f"window {raw} ({minutes} min) is open or starts within {slot['lead']} min"
    train = priority or bool(slot.get("trains"))
    if slot.get("mem") and (free := memory_gb() - (held := reserved_gb(priority=train))) < slot["mem"]:
        whose = "live train checks" if train else "live runs"
        return (f"{free} GB of memory free" + (f" ({held} GB more is reserved by {whose} that haven't reached it)" if held else "")
                + f", under the {slot['mem']} GB a check here may peak at")
    return None


#: how far ahead `--status` lists a windowed slot's windows, and how many of a window's comment words it shows
AHEAD_H = 6
NOTE_WORDS = 6


def upcoming(slot: dict, now: datetime, hours: int = AHEAD_H) -> list[dict]:
    """`slot`'s windows open at `now` or starting within `hours`, by start: when, how long, minutes until it starts (negative
    while open) and its comment's first NOTE_WORDS words."""
    out = []
    for raw, start, minutes, note in sorted(_windows(slot), key=lambda w: w[1]):
        if now < start + timedelta(minutes=minutes) and start < now + timedelta(hours=hours):
            words = note.split()
            out.append({"start": raw, "minutes": minutes, "starts_in_min": int((start - now).total_seconds() // 60),
                        "note": " ".join(words[:NOTE_WORDS]) + (" ..." if len(words) > NOTE_WORDS else "")})
    return out


def window_text(w: dict) -> str:
    when = "open now" if w["starts_in_min"] < 0 else f"starts in {w['starts_in_min']} min"
    return f"window {w['start']} ({w['minutes']} min), {when}" + (f": {w['note']}" if w["note"] else "")


def asking(slots: list[dict]) -> list[dict]:
    """``slots`` with each mem= raised to what the run asks (RUN_MEM_ENV, GB), in GiB as mem= is; unchanged without it."""
    try:
        gb = float(os.environ.get(RUN_MEM_ENV) or 0)
    except ValueError:
        gb = 0.0
    want = -(-int(gb * 1e9) // 2 ** 30) if gb > 0 else 0
    return [{**s, "mem": max(s.get("mem") or 0, want)} if want else s for s in slots]


def take(slots: list[dict], lock_dir: Path, now: datetime, priority: bool = False) -> tuple[dict, int] | None:
    """The first free, unblocked slot (`blocked` for a `priority` waiter or not) and the descriptor holding its lock, or None."""
    for slot in slots:
        if blocked(slot, now, priority):
            continue
        fd = os.open(lock_dir / f"check-{slot['name']}.lock", os.O_RDWR | os.O_CREAT, 0o664)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            continue
        return slot, fd
    return None


def locked(lock: Path) -> bool:
    """Whether some descriptor, this process's inherited one included, holds `lock`'s flock."""
    try:
        fd = os.open(lock, os.O_RDONLY)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(fd)
    return False


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


def asker(by: str | None = None, twin: str | None = None) -> dict:
    """Who asked for a slot: this process, the run it belongs to, the tree it checks and its commit (RESEARCH_SOURCE_SHA, the
    shipped tree's: a run's clone of it is <root>/scratch/<run id>/tree, which names none), who (--by, else RESEARCH_AGENT_URL),
    and its pair when it is a twin."""
    return {"pid": os.getpid(), "run": os.environ.get("RESEARCH_RUN_ID"), "tree": os.getcwd(),
            **({"commit": c} if (c := os.environ.get("RESEARCH_SOURCE_SHA")) else {}),
            "by": by or os.environ.get("RESEARCH_AGENT_URL"), **({"twin": twin} if twin else {})}


def describe(who: dict) -> str:
    tree = who.get("commit") or (Path(who["tree"]).name if who.get("tree") else "")
    parts = [f"run {who['run']}" if who.get("run") else "", f"tree {tree}" if tree else "",
             f"by {who['by']}" if who.get("by") else "", f"twin {who['twin']}" if who.get("twin") else "",
             f"slot taken {who['took']}" if who.get("took") else ""]
    return ", ".join(p for p in parts if p)


def _read(p: Path) -> dict:
    try:
        return json.loads(p.read_text() or "{}")
    except (OSError, ValueError):
        return {}


def options(opts: list[str]) -> tuple[bool, str | None, float | None, float | None, str | None, int | None]:
    """--priority, --by NAME, --wait-max SECONDS and --run-max SECONDS (each > 0), --twin PAIR[@HOME] and --rank N (with
    --priority) from the arguments before `--`; SystemExit with the usage on anything else."""
    priority, vals, i = False, {}, 0
    while i < len(opts):
        if opts[i] == "--priority":
            priority, i = True, i + 1
        elif opts[i] in ("--by", "--wait-max", "--run-max", "--twin", "--rank") and i + 1 < len(opts):
            vals[opts[i]], i = opts[i + 1], i + 2
        else:
            sys.exit(USAGE)
    secs = {}
    for k in ("--wait-max", "--run-max"):
        try:
            secs[k] = float(vals[k]) if k in vals else None
        except ValueError:
            sys.exit(f"slot: {k} {vals[k]!r} is not a number of seconds; {USAGE}")
        if secs[k] is not None and not secs[k] > 0:
            sys.exit(f"slot: {k} must be more than 0 seconds; {USAGE}")
    twin = vals.get("--twin")
    if twin is not None and not re.fullmatch(r"[A-Za-z0-9._-]+(@[A-Za-z0-9._-]+)?", twin):
        sys.exit(f"slot: --twin {twin!r} is not PAIR or PAIR@HOME (letters, digits, '.', '_', '-'); {USAGE}")
    rank = vals.get("--rank")
    if rank is not None and not (priority and re.fullmatch(r"\d{1,6}", rank)):
        sys.exit(f"slot: --rank {rank!r} needs --priority and a whole number below 1,000,000; {USAGE}")
    return priority, vals.get("--by"), secs["--wait-max"], secs["--run-max"], twin, None if rank is None else int(rank)


def limit_run(run_max: float, slot: str, run_dir: str | None, mono=time.monotonic) -> str:
    """Make the `research run` in ``run_dir`` end ``run_max`` seconds from now: its timeout (from the run's start, its
    status.json's `started_mono` on this machine's monotonic clock) becomes the time run so far plus ``run_max``. What was done,
    in a line; outside a run, or without its start, nothing is changed and the run keeps the timeout it had."""
    if not run_dir:
        return f"slot: --run-max {run_max:g} s not applied: not in a `research run` (no RESEARCH_RUN_DIR)"
    started = _read(Path(run_dir) / "status.json").get("started_mono")
    if not isinstance(started, (int, float)):
        return f"slot: --run-max {run_max:g} s not applied: {run_dir}/status.json has no started_mono; the run keeps its timeout"
    t = round(mono() - started + run_max, 3)
    ctl = Path(run_dir) / RUN_TIMEOUT
    ctl.parent.mkdir(parents=True, exist_ok=True)
    tmp = ctl.with_name(f".{ctl.name}.{os.getpid()}")
    tmp.write_text(json.dumps({"timeout_s": t, "actor": "timeout",
                               "reason": f"slot.py --run-max: {run_max:g} s from taking check slot {slot}"}) + "\n")
    os.replace(tmp, ctl)
    return f"slot: the run's timeout is now {t:g} s from its start: {run_max:g} s from taking slot {slot}"


def record_slot(run_dir: str | None, doc: dict) -> None:
    """``doc`` as the run's RUN_SLOT, replaced whole, so `research status` shows a check waiting in the line as waiting and not as
    running since its run began (10 Oct: af2d waited 95 min on node 1 and was reported running all along). Nothing outside a
    `research run`; a write that fails never stops the check."""
    if not run_dir:
        return
    p = Path(run_dir) / RUN_SLOT
    tmp = p.with_name(f".{p.name}.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(doc) + "\n")
        os.replace(tmp, p)
    except OSError as e:
        print(f"slot: the run's {RUN_SLOT} not written: {e}", file=sys.stderr, flush=True)


def slot_docs(slots: list[dict], lock_dir: Path, now: datetime, proc: Path = Path("/proc")) -> list[dict]:
    """Each slot as `--status --json` gives it: its name, cores, `holding`, what blocks it now (`blocked`) and its `upcoming`
    windows."""
    return [{"name": s["name"], "cpus": s["cpus"], "windows": s["windows"], "lead": s["lead"], "trains": s.get("trains", False),
             "mem": s.get("mem"), "holder": holding(lock_dir / f"check-{s['name']}.lock", lock_dir / f"check-{s['name']}.who", now, proc),
             "blocked": blocked(s, now), "upcoming": upcoming(s, now)} for s in slots]


def status(slots: list[dict], lock_dir: Path, now: datetime, proc: Path = Path("/proc"), before: str | None = None) -> list[str]:
    """`--status`'s lines (a windowed slot's `upcoming` windows below it); with `before` (a ticket's name), only the waiters
    that came before it."""
    out = []
    for s in slot_docs(slots, lock_dir, now, proc):
        out.append(f"{s['name']}  cores {s['cpus']}" + ("  trains only" if s["trains"] else "")
                   + (f"  mem {s['mem']} GB" if s["mem"] else "") + f"  {held_text(s['holder'])}" + (f"; blocked: {s['blocked']}" if s["blocked"] else ""))
        out += [f"{s['name']}  {window_text(w)}" for w in s["upcoming"]]
    return out + waiters(lock_dir / LINE, now, before)


def status_doc(now: datetime, proc: Path = Path("/proc")) -> dict:
    """`--status --json`: SLOTS, each slot (`slot_docs`; null without SLOTS) and the line's live waiters (`waiting`)."""
    if not SLOTS.exists():
        return {"file": str(SLOTS), "now": f"{now:%Y-%m-%dT%H:%MZ}", "slots": None, "waiters": []}
    return {"file": str(SLOTS), "now": f"{now:%Y-%m-%dT%H:%MZ}", "slots": slot_docs(parse(SLOTS.read_text()), SLOTS.parent, now, proc),
            "waiters": waiting(SLOTS.parent / LINE, now)}


def holding(lock: Path, who_file: Path, now: datetime, proc: Path = Path("/proc")) -> dict | None:
    """Who holds ``lock`` (None while it's free): its pid, since when and for how many minutes, its command, and the run, tree,
    asker and time taken its holder recorded in ``who_file``, kept only while that pid holds the lock."""
    pid = holder(lock, proc)
    if pid is None:
        return None
    t = started(pid, proc)
    try:
        cmd = (proc / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        cmd = "(gone)"
    out = {"pid": pid, "since": f"{t:%Y-%m-%dT%H:%MZ}" if t else None,
           "minutes": round((now - t).total_seconds() / 60) if t else None, "cmd": cmd[:200]}
    who = _read(who_file)
    if who.get("pid") == pid:
        out.update({k: who[k] for k in ("run", "tree", "commit", "by", "twin", "took") if who.get(k)})
    return out


def held_text(h: dict | None) -> str:
    if h is None:
        return "free"
    since = f" since {h['since']} ({h['minutes']} min)" if h["since"] else ""
    return f"held by pid {h['pid']}{since}" + (f" ({describe(h)})" if describe(h) else "") + f": {h['cmd']}"


def waiters(line: Path, now: datetime, before: str | None = None) -> list[str]:
    """One line per live waiter in ``line`` (`waiting`)."""
    return [waiter_text(w) for w in waiting(line, now, before)]


def waiter_text(w: dict) -> str:
    return (f"waiting since {w['since']} ({w['minutes']} min)" + (" with priority" if w["priority"] else "") + f": pid {w['pid']}"
            + (f" ({describe(w)})" if describe(w) else ""))


def waiting(line: Path, now: datetime, before: str | None = None) -> list[dict]:
    """Each live waiter in ``line``, oldest first (priority ones first, as they take slots): since when and for how many
    minutes, whether with priority, and the pid, run, tree and asker its ticket names; with `before` (a ticket's name), only
    those that came before it."""
    out = []
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
        t = datetime.fromtimestamp(int(p.name.lstrip(PRIORITY).split("-")[0].split(RANKED)[-1]) / 1e9, timezone.utc)
        who = _read(p)
        out.append({"since": f"{t:%Y-%m-%dT%H:%MZ}", "minutes": round((now - t).total_seconds() / 60),
                    "priority": p.name.startswith(PRIORITY), "pid": who.get("pid", int(p.name.split("-")[-1])),
                    **{k: who[k] for k in ("run", "tree", "commit", "by", "twin") if who.get(k)}})
    return out


#: a priority waiter's ticket starts with this, which sorts before every digit (and before "." of a ticket being made)
PRIORITY = "!"
#: a ranked priority ticket is `!RRRRRR~TIME-PID`: its zero-padded rank sorts before an unranked `!TIME-PID` (times begin 1...)
RANKED = "~"


def join(lock_dir: Path, priority: bool = False, who: dict | None = None, line: str = LINE,
         rank: int | None = None) -> tuple[Path, int]:
    """This waiter's ticket in the line and the descriptor holding its lock; named by arrival (after every priority ticket
    unless it is one; a ranked priority one by rank, then arrival), and locked before it is visible."""
    d = lock_dir / line
    d.mkdir(exist_ok=True)
    order = f"{rank:06d}{RANKED}" if priority and rank is not None else ""
    name = f"{PRIORITY if priority else ''}{order}{time.time_ns():020d}-{os.getpid()}"
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


DIRECT_CPUS_FIX = "a `research run` starts on /etc/vy/direct-cpus, so add the slot's cores there (`0-159` on node 1 since 6 Oct)"


def usable(slots: list[dict]) -> list[dict]:
    """The slots whose every core this process may run on. Pinning onto any other fails (EINVAL), so a check never takes one:
    on vy-nebius-1 a `research run` session was held to cores 0-127, and slot d on 128-159 killed every check that took it
    (2 Oct 07:20Z). A `research run` starts on the host's /etc/vy/direct-cpus, so a slot's cores must be listed there too."""
    mine = allowed()
    return [s for s in slots if cpulist(s["cpus"]) <= mine]


def reread(slots: list[dict]) -> list[dict]:
    """The slots file as it is now, so a slot added while a check waits is one it can take; the last list it read when the file
    can't be read or lists none (mid-rewrite)."""
    try:
        return parse(SLOTS.read_text()) or slots
    except OSError:
        return slots


def peers(path: Path | None = None) -> dict[str, list[str]]:
    """PEERS, one line per machine: its name, then ssh's options and destination for reaching it (`vy-nebius-1 -i KEY research@IP`)."""
    try:
        text = (path or PEERS).read_text()
    except OSError:
        return {}
    return {f[0]: f[1:] for f in (ln.split("#")[0].split() for ln in text.splitlines()) if len(f) >= 2}


def twin_here(op: str, lock_dir: Path, pair: str, who: dict) -> tuple[str, dict]:
    """`claim` (mkdir TWINS/PAIR under ``lock_dir``, first removing claims a week old) or `peek`: ("claimed", who), ("taken", the
    claimer's who), ("free", {}) or ("unreachable", why) when the claim can't be written."""
    t, d = lock_dir / TWINS, lock_dir / TWINS / pair
    try:
        if op == "claim":
            t.mkdir(exist_ok=True)
            for old in t.iterdir():
                try:
                    if time.time() - old.stat().st_mtime > 7 * 86400:
                        (old / "who").unlink(missing_ok=True)
                        old.rmdir()
                except OSError:
                    pass                                            # another claimer removed it first
            try:
                d.mkdir()
            except FileExistsError:
                pass
            else:
                (d / "who").write_text(json.dumps(who) + "\n")
                return "claimed", who
    except OSError as e:
        return "unreachable", {"why": f"{type(e).__name__}: {e}"}
    return ("taken", _read(d / "who")) if d.is_dir() else ("free", {})


def twin_there(op: str, home: str, lock_dir: Path, pair: str, who: dict) -> tuple[str, dict]:
    """`twin_here` on machine ``home`` (its claims under the same ``lock_dir``), over the ssh PEERS names; ("unreachable", why)
    when it can't be asked."""
    ssh = peers().get(home)
    if not ssh:
        return "unreachable", {"why": f"{home} isn't in {PEERS}"}
    t, d = shlex.quote(str(lock_dir / TWINS)), shlex.quote(str(lock_dir / TWINS / pair))
    look = f'if [ -d {d} ]; then echo taken; cat {d}/who 2>/dev/null; else echo free; fi'
    script = look if op == "peek" else (
        f'mkdir -p {t} && find {t} -mindepth 1 -maxdepth 1 -mtime +7 -exec rm -rf {{}} + ; '
        f'if mkdir {d} 2>/dev/null; then printf "%s\\n" {shlex.quote(json.dumps(who))} > {d}/who; echo claimed; else {look}; fi')
    try:
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", *ssh, script], capture_output=True, text=True,
                timeout=TWIN_SSH_S)
    except (OSError, subprocess.SubprocessError) as e:
        return "unreachable", {"why": f"ssh to {home}: {type(e).__name__}: {e}"}
    state, _, rest = r.stdout.partition("\n")
    if r.returncode or state not in ("claimed", "taken", "free"):
        return "unreachable", {"why": f"ssh to {home} (exit {r.returncode}): {(r.stderr or r.stdout).strip()[-200:]}"}
    try:
        info = json.loads(rest) if rest.strip() else {}
    except ValueError:
        info = {}
    return state, who if state == "claimed" else info if isinstance(info, dict) else {}


def twin(op: str, spec: str, who: dict) -> tuple[str, dict]:
    """`--twin PAIR` (its claims here) or `--twin PAIR@HOME`: `twin_here` or `twin_there`."""
    pair, _, home = spec.partition("@")
    return twin_there(op, home, SLOTS.parent, pair, who) if home else twin_here(op, SLOTS.parent, pair, who)


def leave(ticket: Path, tfd: int, why: str) -> int:
    """Leave the line without running CMD: NOT_STARTED."""
    ticket.unlink(missing_ok=True)
    os.close(tfd)
    print(f"slot: not started: {why}; exit {NOT_STARTED}", file=sys.stderr, flush=True)
    return NOT_STARTED


def main(argv: list[str], now=lambda: datetime.now(timezone.utc), sleep=time.sleep) -> int:
    if argv[:1] == ["--status"]:
        if argv[1:] == ["--json"]:
            print(json.dumps(status_doc(now())))
        elif argv[1:]:
            sys.exit("usage: slot.py --status [--json]")
        elif not SLOTS.exists():
            print(f"no check slots on this machine ({SLOTS})")
        else:
            print("\n".join(status(parse(SLOTS.read_text()), SLOTS.parent, now())))
        return 0
    opts, cmd = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else ([], argv)
    priority, by, wait_max, run_max, pair, rank = options(opts)
    if not cmd:
        sys.exit(USAGE)
    if not SLOTS.exists() and os.environ.get("CHECK_SLOTS"):
        pools = ", ".join(p.parent.name for p in lane_pools(SLOTS.parent.parent / "slots")) or "none"
        print(f"slot: not started: CHECK_SLOTS names {SLOTS}, which this machine doesn't have, and CMD would run on any core "
              f". Name one of its pools instead (CHECK_SLOTS={SLOTS.parent.parent}/NAME/slots, or `research run --pool "
              f"NAME`): {pools}; exit {NOT_STARTED}", file=sys.stderr, flush=True)
        return NOT_STARTED
    if not SLOTS.exists():
        print(f"slot: no check slots on this machine ({SLOTS}); running unpinned", file=sys.stderr, flush=True)
        os.execvp(cmd[0], cmd)
    listed = parse(SLOTS.read_text())
    if not listed:
        sys.exit(f"slot: {SLOTS} lists no slots")
    # already inside a slot this machine's lock still holds (a `research run --pool` job that wraps its own command in slot.py,
    # 19 Oct: the inner waiter queued behind its own outer lock and stopped cpu-1's research pool): run CMD in that slot
    if (held := os.environ.get("CHECK_SLOT")) and any(s["name"] == held for s in listed) and locked(SLOTS.parent / f"check-{held}.lock"):
        print(f"slot: already in check slot {held}; running CMD in it", file=sys.stderr, flush=True)
        os.execvp(cmd[0], cmd)
    slots = usable(listed)
    if len(slots) < len(listed):
        out = ", ".join(f"{s['name']} ({s['cpus']})" for s in listed if s not in slots)
        print(f"slot: skipping {out}: outside this process's cores; {DIRECT_CPUS_FIX}", file=sys.stderr, flush=True)
    if not slots:
        sys.exit(f"slot: none of {SLOTS}'s slots is within this process's cores; {DIRECT_CPUS_FIX}")
    if not priority and all(s["trains"] for s in slots):
        pools = lane_pools(SLOTS)
        where = ("a lane check takes a lane pool here: " + ", or ".join(
            f"`CHECK_SLOTS={p} python3 tools/verity/check/slot.py --by LANE -- CMD...`" for p in pools)
            + " (`research run --pool NAME` sends one by its directory's name, and `check.py --record --on MACHINE --pool NAME` a "
              "recorded check, for the pools `research run --help` names)"
            if pools else "a lane check runs on another machine")
        sys.exit(f"slot: every slot in {SLOTS} is for train checks (only=trains); {where}. A PR lands "
                 f"on an eyeball review, and no lane check or quick tier is required; to run its tests anyway: {LANE_RUN}")
    me = asker(by, pair)
    ticket, tfd = join(SLOTS.parent, priority=priority, who=me, rank=rank)
    told, joined, peeked = None, now(), None
    run_dir = os.environ.get("RESEARCH_RUN_DIR")
    since = {"since": f"{joined:%Y-%m-%dT%H:%M:%SZ}", **({"priority": True} if priority else {}), **({"twin": pair} if pair else {}),
             **({"rank": rank} if rank is not None else {})}
    record_slot(run_dir, {"state": "waiting", **since})

    def gone(why: str) -> int:
        record_slot(run_dir, {"state": "left", **since, "left": f"{now():%Y-%m-%dT%H:%M:%SZ}", "why": why})
        return leave(ticket, tfd, why)

    if pair:
        print(f"slot: twin {pair}: runs only if it takes a slot before its twins do", file=sys.stderr, flush=True)
    while True:
        slots = asking(usable(reread(slots)) or slots)
        if pair and ("@" not in pair or peeked is None or (now() - peeked).total_seconds() >= TWIN_PEEK_S):
            state, info = twin("peek", pair, me)
            peeked = now()
            if state == "taken":
                return gone(f"its twin took a slot first ({describe(info) or 'claimed'})")
        n = ahead(ticket)
        got = None if n else take(slots if priority else [s for s in slots if not s["trains"]], SLOTS.parent, now(), priority)
        if got and pair:
            state, info = twin("claim", pair, {**me, "took": f"{now():%Y-%m-%dT%H:%MZ}"})
            if state != "claimed":
                os.close(got[1])
                return gone(f"its twin took a slot first ({describe(info) or 'claimed'})" if state == "taken" else
                            f"twin {pair} can't be claimed ({info.get('why')}), so its twin there runs instead" if "@" in pair else
                            f"twin {pair}'s claim can't be written here ({info.get('why')}), and no twin runs without it: "
                            "rerun the check")
        if got:
            break
        why = f"{n} check(s) ahead in line" if n else \
            "every check slot is taken (" + "; ".join(f"{s['name']}: {blocked(s, now(), priority) or ('for train checks' if s['trains'] and not priority else 'busy')}"
                                                   for s in slots) + ")"
        if why != told:
            print(f"slot: {why}; retrying every {POLL_S} s", file=sys.stderr, flush=True)
            for ln in status(slots, SLOTS.parent, now(), before=ticket.name):
                print(f"slot:   {ln}", file=sys.stderr, flush=True)
            record_slot(run_dir, {"state": "waiting", **since, "why": why})
            told = why
        if wait_max is not None and (now() - joined).total_seconds() >= wait_max:
            return gone(f"no check slot within --wait-max {wait_max:g} s ({why})")
        sleep(POLL_S)
    ticket.unlink()
    os.close(tfd)
    slot, fd = got
    who = SLOTS.parent / f"check-{slot['name']}.who"
    tmp = who.with_name(f".{who.name}.{os.getpid()}")
    tmp.write_text(json.dumps(me) + "\n")
    tmp.chmod(0o664)
    os.replace(tmp, who)
    os.set_inheritable(fd, True)
    os.sched_setaffinity(0, cpulist(slot["cpus"]))
    env = {k: v for k, v in os.environ.items() if k not in ("LEASE_DIR", SLOT_MEM_ENV, SCOPE_ENV)}
    env.update(LEASE_DEADLINE_FILE="/nonexistent/research-deadline", CHECK_SLOT=slot["name"],
               VERITY_TEST_CACHE=str(Path.home() / ".cache" / "verity" / f"tests-slot-{slot['name']}"),
               **({SLOT_MEM_ENV: str(slot["mem"])} if slot.get("mem") else {}))
    record_slot(run_dir, {"state": "held", **since, "slot": slot["name"], "cores": slot["cpus"], "took": f"{now():%Y-%m-%dT%H:%M:%SZ}"})
    print(f"slot: check slot {slot['name']} (cores {slot['cpus']})", file=sys.stderr, flush=True)
    if said := reserve(slot, now(), priority):
        print(said, file=sys.stderr, flush=True)
    if run_max is not None:
        print(limit_run(run_max, slot["name"], run_dir), file=sys.stderr, flush=True)
    if slot.get("scope") and (why := scope_refusal()):
        print(f"slot: scope=1 but systemd-run --user doesn't run here ({why}); running unscoped", file=sys.stderr, flush=True)
    elif slot.get("scope"):
        cmd, unit = scoped(cmd, slot["name"], os.environ.get("RESEARCH_RUN_ID") or str(os.getpid()))
        env[SCOPE_ENV] = unit
        print(f"slot: check in its own scope {unit}.scope", file=sys.stderr, flush=True)
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
