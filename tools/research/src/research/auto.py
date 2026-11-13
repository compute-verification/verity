"""`research run --on auto`'s placement: the CPU node's pool (`spill`) with the most places for the job first, then node 2's
Verity share (the queue, `AUTO_NODE`) when it can start the job now, then the shortest pool line, and node 2's line for a job
no pool fits.

Node 2's share is 48 cores that queued runs claim whole (research.cores) in one first-come line, so two large runs (a 16-job
quick tier and a 32-core Lean check) can hold all 48 for hours while dozens of small runs wait behind them, with cpu-1's idle
cores unused. So before queueing, `read` asks each machine at once, over the ssh `research run
--on` uses, as `python3 -` with research.cores and the source tree's tools/verity/check/slot.py inlined (both stdlib only):
node 2 for its share's live claims and waiters, and each CPU node (`spill`) for its pool's `slot.py --status --json` and the
memory slot.py finds free there (`free_gb`, read once a pool). `choose` then takes, in order:

  - the pool with the most places for the job, the first by number among equals. A place is a slot that fits the job (not
    trains-only, at least its CPUs), idle and unblocked, with the machine's free memory covering what the job would reserve
    there (`_need`: the larger of the slot's mem=, or its book=, and the job's ask, which no cgroup enforces) once the
    waiters ahead have taken their own slots and memory, each by the ask its ticket names, as slot.py admits them (`_pool`).
    Idle slots are places only as far as the free memory goes, since every run that starts reserves its own. The most
    places, not the first pool that has one: a launch's read and its run's joining the line are half a minute apart, so the
    launches of one half minute read the same lines, and sent to one place all but one of them wait;
  - else node 2's share when it can start the job now, with no one waiting there (a run once queued 100 min behind cpu-6's 8
    held slots and 9 waiters while node 2's 48 CPUs sat free);
  - else the pool with the shortest wait, as the waiters that stay in its line per slot that runs, leaving out a pool whose
    line is stalled on memory (`STALL_MIN`): a job placed there waits with it, however short the line;
  - node 2 only when no pool fits the job or none can be read, and node 2 itself can be. Node 2 is the only GPU node,
    and CPU work there has held its GPUs at 0% under a load of 235 on 120 cores while cpu-5 ran at 8 of 192. A node 2
    that carries remote.RUNS_VIA_SLOT_FILE counts as unread: its launcher refuses a run that doesn't come through a check
    slot.

A run that takes a Lean slot (`lean_pool`) waits for it inside its CPU slot, holding that slot's cores and memory idle, so for
it `read` also asks each machine for its Lean slots (lean_slot.py's `--status --json`) and each step reads them (`_lean`): a
machine whose Lean slots give its pool none is left out; in the first two steps the job starts now only where a slot of that
Lean pool is free too, with nobody waiting for one; and the third takes the shortest line for a Lean slot, as waiters ahead
per slot of the pool (none where one is free), before the shortest line for a CPU slot. A machine whose Lean slots can't be
read comes last there, never first. The placement's reason names the Lean pool that decided it.

A GPU job, a run naming its own node, or one asking more CPUs than a pool slot has stays on node 2's queue. A machine that can't
be read is skipped; with node 2 unread and no pool fitting, the job is refused (`Unplaceable`) with where to run it. A pool
run waits up to slot_only.POOL_WAIT_S (4 h) in its line and then exits 75 without running; its --timeout counts from taking
the slot, and a wait inside it for a Lean audit slot (lean_slot.py) moves it later by as long.

The launch is told when the job should start where `choose` sent it (`expect`, from the documents `read` already has): now, in
about so many minutes behind the line it joins, or that nothing read gives an estimate. A job expected to wait past
START_WITHIN_MIN is late. Its launch says so in one line; under REFUSE it is refused at once instead, with the ways on:
`research run --accept-wait` queues it anyway, and `smaller` names the smaller request that starts now.

Node 2 takes no `--on auto` job while it is held: until the time in its HOLD_FILE, which infra writes with the reason when one
lane has the whole node. `choose` then treats it as a node 2 that is down, and the launcher is told until when and why (`off`).
A stopped node 2 costs a launch NODE_CONNECT_S: its probe's ssh gives up connecting after that long, beside the pools' probes,
and the placement says once that node 2 is down. A GPU or quiet job, which the queue alone books and no pool runs, asks node 2
alone and is refused while it is held or down (`closed`).
"""
from __future__ import annotations

import json
import math
import re
import shlex
import subprocess
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, NamedTuple

AUTO_NODE, AUTO_PROJECT = "vy-nebius-2", "verity"
#: the pool `--on auto` spills to on a CPU node that has one: any lane's research runs (slot_only.POOLS)
SPILL_POOL = "research"
#: a CPU node's name in the launcher's registry of machines; `spill` takes them by number
CPU_NODE = re.compile(r"vy-nebius-cpu-([0-9]+)")
DESCRIPTION = Path("infra/cluster/descriptions/nebius.toml")
SLOT_PY = Path("tools/verity/check/slot.py")
READ_TIMEOUT_S = 30
#: ssh's ConnectTimeout for node 2's probe: a stopped node 2 costs a launch this long, not remote.ssh_argv's default
NODE_CONNECT_S = 5
#: how `read` words a machine whose ssh connection fails (it is stopped, or off the network); `off` reads it back for node 2
NO_SSH = "its ssh doesn't answer"
#: the hold, a file on node 2 that infra writes when one lane has the whole node: its first line is `UNTIL REASON`, UNTIL a UTC
#: time as 2030-01-31T17:00Z (with no zone it is UTC, whatever the node's; an offset is honoured). Until then `--on auto` places
#: no job there and tells the launcher UNTIL and REASON; from UNTIL on the file means nothing, so it needs no removing. It
#: fails closed: a file that is there but can't be read (a wrong mode, a directory, bytes that are no text), or whose first
#: line names no time (empty, a typo, a date alone), holds until it is removed, and the launcher is shown what it says or why
#: it can't be read. Only a file that isn't there, or a time that has passed, opens the node
HOLD_FILE = "/etc/vy/auto-hold"
#: a machine whose DISK_PATH has fewer free inodes or GB than these takes no `--on auto` job: a machine out of inodes with half
#: its bytes free fails every job placed there, shipping its source or mid-build, with ENOSPC. One run of
#: a large suite (tools/research's) holds some 250k pytest temp files at once, so a floor below a few of those still fails jobs
DISK_PATH, MIN_FREE_INODES, MIN_FREE_GB = "/workspace", 1_000_000, 50
#: a pool whose line has waited this long beside a free slot is stalled on memory and takes no job (`_pool`), nor does one whose
#: every fitting slot is idle and blocked with slot.py's memory reason, which contains MEMORY_BLOCK
STALL_MIN, MEMORY_BLOCK = 5, " GB of memory free"
#: slot.py's BACKFILL_MAX_GB: a run asking at most this many GB passes the lane waiters ahead that only memory keeps from starting
BACKFILL_MAX_GB = 32
#: a launch is on time when its run starts within this many minutes; one `expect` finds late is told so, or refused (REFUSE)
START_WITHIN_MIN = 10
#: how long a pool run holds its slot, in minutes, as `expect` takes it: the mean over the pools' runs, since a line of n runs
#: on k slots drains in n means over k. A parameter of the estimate, set from the runs' own slot records, not read at launch
HOLD_MIN = 25
#: whether a late launch is refused, with the ways on (`research run --accept-wait` queues it anyway; a smaller request that
#: starts now, `smaller`), or only warned in one line. Off until infra turns it on
REFUSE = False
#: a machine's Lean pools (lean_slot.py's file, which the probe's copy of LEAN_SLOT_PY reads): a run that takes a Lean slot of a
#: pool goes only where that pool has one, since lean_changed.py refuses it at once elsewhere, or builds cold in its checkout
LEAN_SLOTS = "/workspace/research/locks/lean-slots"
LEAN_SLOT_PY = Path("tools/verity/check/lean_slot.py")


def lean_pool(command: list[str], env: list[str] = ()) -> str | None:
    """The Lean pool (lean_slot.py) ``command`` waits for a slot of, None when it takes none. lean_changed.py's `--records`,
    `--build` and `--try`, and `verity lean try`, take $LEAN_RECORDS_POOL's, else `build`; tools/verity/lean/check.py `--build`,
    which `--records --dev-cache` runs too, takes $LEAN_SLOT_POOL's, else `audit` (pool `any`, which stands in where a machine
    lists no `audit`, isn't counted). ``env`` is the run's --env (`K=V`), and a `K=V` word of the command counts as one; a word
    with spaces (`bash -lc "..."`) is read as the words a shell makes of it."""
    words: list[str] = []
    for w in map(str, command):
        try:
            words += shlex.split(w) if " " in w else [w]
        except ValueError:
            words.append(w)
    named = dict(kv.split("=", 1) for kv in (*env, *words) if "=" in kv and not kv.startswith("-"))
    opts = {w.split("=", 1)[0] for w in words}
    changed = any(Path(w).name == "lean_changed.py" for w in words) and opts & {"--records", "--build", "--try"}
    if changed and "--dev-cache" not in opts or any(words[i:i + 3] == ["verity", "lean", "try"] for i in range(len(words))):
        return named.get("LEAN_RECORDS_POOL") or "build"
    if changed or (any(w.endswith("lean/check.py") for w in words) and "--build" in opts):
        return named.get("LEAN_SLOT_POOL") or "audit"
    return None


class Unplaceable(Exception):
    """No pool fits the job and node 2 takes none (`off`): its queue would place it on a node that is held for one lane, or
    isn't up (a stopped node 2 refuses an `--on auto` job too big for a pool slot with a stale admission window)."""

PROBE = '''
import json, os, sys, types
from datetime import datetime, timezone
from pathlib import Path


def load(name, src, env):
    os.environ.update(env)
    m = types.ModuleType(name)
    sys.modules[name] = m
    exec(compile(src, name + ".py", "exec"), m.__dict__)
    return m


def hold():
    try:
        text = Path(HOLD).read_text()
    except (FileNotFoundError, NotADirectoryError):
        return None
    except (OSError, UnicodeDecodeError) as e:
        return {"until": None, "why": f"it can't be read ({type(e).__name__})"}
    line = (text.splitlines() or [""])[0].strip()
    until, _, why = line.partition(" ")
    try:
        if "T" not in until:
            raise ValueError("a date alone is no time")
        end = datetime.fromisoformat(until.replace("Z", "+00:00"))
    except ValueError:
        return {"until": None, "why": line}
    end = end if end.tzinfo else end.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc) if NOW is None else datetime.fromtimestamp(NOW, timezone.utc)
    return {"until": f"{end.astimezone(timezone.utc):%Y-%m-%dT%H:%MZ}", "why": why.strip()} if now < end else None


def share(cpus, d, root):
    h = hold()
    if h is not None:
        return {"hold": h}
    if os.path.exists(VIA_SLOT):
        raise RuntimeError(f"{VIA_SLOT}: its launcher takes a run only through a check slot")
    c = load("auto_cores", CORES_SRC, {})
    want, d = c.parse(cpus), Path(d)
    if not d.is_dir():
        return {"share": len(want), "claims": 0, "waiters": 0, "free": len(want)}
    held = [c.parse(f.read_text()) for f in d.glob("*.cpus") if c.held(f)]
    free = want - set().union(*held)
    got = c.carve(free, JOB_CPUS, Path(root))
    return {"share": len(want), "claims": len(held), "waiters": len([f for f in d.glob("wait.*") if c.held(f)]),
            "free": len(free), "fits": got is not None}


def doc(f):
    try:
        return f()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def lean_slots():
    load("check_slot_for_lean_slot", SLOT_SRC, {})                   # lean_slot.py takes the slot.py loaded under its SLOT_MODULE
    return load("auto_lean_slot", LEAN_SRC, {"LEAN_SLOTS": LEAN_SLOTS}).status_doc(datetime.now(timezone.utc))


def once(f):
    seen = {}

    def first(*a, **kw):
        key = (a, tuple(sorted(kw.items())))
        if key not in seen:
            seen[key] = f(*a, **kw)
        return seen[key]
    return first


def pool_doc(pool, slots):
    m = load("auto_slot_" + pool, SLOT_SRC, {"CHECK_SLOTS": slots})
    if not (hasattr(m, "memory_gb") and hasattr(m, "reserved_gb")):
        return m.status_doc(datetime.now(timezone.utc))
    # slot.py reads the machine's memory and every live reservation's process tree for each slot with mem=; once is the same answer
    m.memory_gb, m.reserved_gb = once(m.memory_gb), once(m.reserved_gb)
    d = m.status_doc(datetime.now(timezone.utc))
    d["free_gb"] = m.memory_gb() - m.reserved_gb(priority=False)
    return d


def disk_short():
    if not os.path.isdir(DISK):
        return None
    s = os.statvfs(DISK)
    gb = s.f_bavail * s.f_frsize / 1e9
    if s.f_files and s.f_favail < MIN_INODES:
        return f"{DISK} has {s.f_favail} inodes free, under {MIN_INODES}"
    if gb < MIN_GB:
        return f"{DISK} has {gb:.0f} GB free, under {MIN_GB}"
    return None


out = {}
if SHARE is not None:
    out["share"] = doc(lambda: share(SHARE, CLAIMS, CPU_ROOT))
short = disk_short()
if LEAN_SRC:
    out["lean"] = doc(lean_slots)
for pool, slots in POOLS.items():
    out[pool] = {"error": f"disk: {short}"} if short else doc(lambda: pool_doc(pool, slots))
print(json.dumps(out))
'''


def cpulist(text: str) -> int:
    n = 0
    for part in str(text).split(","):
        lo, _, hi = part.partition("-")
        n += int(hi or lo) - int(lo) + 1
    return n


def spill(machines: dict[str, dict] | None = None) -> tuple[tuple[str, str], ...]:
    """(machine, pool) where `--on auto` may spill: SPILL_POOL on every CPU node (CPU_NODE) of AUTO_PROJECT in the launcher's
    registry of machines (remote.load_machines; ``machines`` stands in for it), by number. No node is listed here, so one
    registered today is asked today, and a pool left out can't sit idle while the others queue. Which nodes keep the pool
    is the probe's to say: one with no slots file for it, or a file that lists no slot, is skipped (`_pool`) at the cost of
    its probe, which runs beside the others'."""
    if machines is None:
        from . import remote

        machines = remote.load_machines()
    nodes = sorted((int(m.group(1)), name) for name, entry in machines.items()
                   if (m := CPU_NODE.fullmatch(name)) and AUTO_PROJECT in (entry.get("projects") or [entry.get("project")]))
    return tuple((name, SPILL_POOL) for _, name in nodes)


def _pools(docs: dict[str, dict | str]) -> list[tuple[str, str]]:
    """(machine, pool) for each machine `read` asked for its pool: ``docs``' machines but node 2, in `spill`'s order."""
    def number(machine: str) -> tuple[float, str]:
        m = CPU_NODE.fullmatch(machine)
        return (int(m.group(1)) if m else math.inf, machine)

    return [(machine, SPILL_POOL) for machine in sorted((m for m in docs if m != AUTO_NODE), key=number)]


def share_cpus(source: Path) -> str | None:
    """The CPUs of AUTO_PROJECT's share on AUTO_NODE in the source tree's cluster description, None when it says none."""
    try:
        desc = tomllib.loads((Path(source) / DESCRIPTION).read_text())
    except (OSError, ValueError):
        return None
    return next((s["cpus"] for s in desc.get("shares", []) if s.get("project") == AUTO_PROJECT and s.get("node") == AUTO_NODE
                 and isinstance(s.get("cpus"), str)), None)


def payload(source: Path, cpus: int, share: str | None, pools: list[str], now: float | None = None, lean: bool = False) -> bytes:
    """The probe a machine runs as `python3 -`; `now` (an epoch) stands in for the machine's clock when HOLD_FILE is read. With
    `lean` it reads the machine's Lean slots too, with the source tree's lean_slot.py."""
    from . import cores, remote
    from .pods.nebius import slot_only

    head = {"CORES_SRC": Path(cores.__file__).read_text(), "SLOT_SRC": (Path(source) / SLOT_PY).read_text() if pools or lean else "",
            "LEAN_SRC": (Path(source) / LEAN_SLOT_PY).read_text() if lean else "",
            "JOB_CPUS": max(cpus, 1), "SHARE": share, "CLAIMS": str(cores.CLAIMS), "CPU_ROOT": str(cores.CPU_ROOT),
            "HOLD": HOLD_FILE, "NOW": now,
            "POOLS": {p: slot_only.POOLS[p] for p in pools}, "DISK": DISK_PATH, "MIN_INODES": MIN_FREE_INODES, "MIN_GB": MIN_FREE_GB,
            "VIA_SLOT": remote.RUNS_VIA_SLOT_FILE, "LEAN_SLOTS": LEAN_SLOTS}
    return ("".join(f"{k} = {v!r}\n" for k, v in head.items()) + PROBE).encode()


def _argv(name: str, target: dict) -> list[str]:
    """The probe's ssh to machine `name` at `target` (remote.pod_target's); node 2's gives up connecting after NODE_CONNECT_S."""
    from . import remote

    quick = {"connect_timeout": NODE_CONNECT_S} if name == AUTO_NODE else {}
    return [*remote.ssh_argv(target["host"], target["port"], user=target["user"], **quick), "python3 -"]


def _reach(name: str) -> list[str]:
    from . import remote

    return _argv(name, remote.pod_target(remote.resolve_machine(name, AUTO_PROJECT)))


def _ssh(argv: list[str], data: bytes) -> subprocess.CompletedProcess:
    return subprocess.run(argv, input=data, capture_output=True, timeout=READ_TIMEOUT_S)


def closed(n2: Any) -> str | None:
    """Why node 2 is closed to every `--on auto` job, a GPU or quiet one included, from its answer to `read`: it is down, or
    HOLD_FILE holds it. None otherwise."""
    if isinstance(n2, str) and n2.startswith(NO_SSH):
        return f"{AUTO_NODE} is down: {n2}"
    share = n2.get("share") if isinstance(n2, dict) else None
    if isinstance(share, dict) and isinstance(hold := share.get("hold"), dict):
        until = f"until {hold['until']}" if hold.get("until") else f"while it carries {HOLD_FILE}"
        return f"{AUTO_NODE} takes no `--on auto` job {until}" + (f": {hold['why']}" if hold.get("why") else "")
    return None


def off(n2: Any) -> str | None:
    """Why node 2 takes no job `choose` places, from its answer to `read`: it is `closed`, or its share wasn't read. None when
    its share was read."""
    if (why := closed(n2)) is not None:
        return why
    share = n2.get("share") if isinstance(n2, dict) else None
    if not isinstance(share, dict) or "error" in share:
        why = share.get("error") if isinstance(share, dict) else n2 if isinstance(n2, str) else "not read"
        return f"{AUTO_NODE}'s share wasn't read ({why})"
    return None


def read(source: Path, cpus: int, *, reach: Callable[[str], list[str]] | None = None, run: Callable | None = None,
         now: float | None = None, node_only: bool = False, lean: bool = False) -> dict[str, dict | str]:
    """Each machine's documents (node 2's under "share", a pool's under its name), read at once, or why they couldn't be. Node
    2's share is `{"hold": {until, why}}` while HOLD_FILE holds it (`now`: `payload`). With `node_only`, node 2 alone is asked;
    with `lean`, each machine's Lean slots too, under "lean" (lean_slot.py's status document)."""
    reach, run = reach or _reach, run or _ssh
    share = share_cpus(source)
    asks: dict[str, list[str]] = {AUTO_NODE: []}
    for machine, pool in () if node_only else spill():
        asks.setdefault(machine, []).append(pool)

    def one(item: tuple[str, list[str]]) -> dict | str:
        name, pools = item
        try:
            r = run(reach(name), payload(source, cpus, share if name == AUTO_NODE else None, pools, now, lean))
        except subprocess.TimeoutExpired:
            return f"no answer in {READ_TIMEOUT_S} s"
        except Exception as e:                                       # not listed, not this project's, no source: never chosen
            return str(e) or type(e).__name__
        if r.returncode:
            err = r.stderr.decode(errors="replace").strip().splitlines()
            if r.returncode == 255 and err and err[-1].startswith("ssh: connect to host "):
                return f"{NO_SSH} ({err[-1].rsplit(': ', 1)[-1]})"
            return f"exit {r.returncode}: {err[-1] if err else 'no output'}"
        try:
            doc = json.loads(r.stdout)
        except ValueError:
            return f"its answer is not JSON: {r.stdout[:120]!r}"
        return doc if isinstance(doc, dict) else f"its answer is not a JSON object: {r.stdout[:120]!r}"

    with ThreadPoolExecutor(max_workers=len(asks)) as ex:
        return dict(zip(asks, ex.map(one, asks.items())))


class Stalled(ValueError):
    """A pool whose line is stalled on memory (`_pool`): skipped while anything else can take the job, its last resort else."""

    def __init__(self, why: str, waiting: int):
        super().__init__(why)
        self.waiting = waiting


def _need(slot: dict, mem_gb: float) -> int:
    """The GiB a run asking ``mem_gb`` GB reserves in ``slot``: slot.py raises a slot's mem= to the run's ask (its `asking`: GB
    as GiB, rounded up) and one without mem= to that ask. A slot with book= reserves the larger of that and the ask, in place
    of its mem= (slot.py's `booked`)."""
    return max(slot.get("book") or slot.get("mem") or 0, -(-int(mem_gb * 1e9) // 2 ** 30) if mem_gb > 0 else 0)


def _blocked(slot: dict, free_gb: Any, mem_gb: float) -> str | None:
    """What keeps a job asking ``mem_gb`` GB from taking ``slot`` now, as slot.py decides it at admission: what blocks the slot
    for any check (its `blocked`), or less memory free than the job would reserve there (`_need`). So the machine's free
    memory (the pool's `free_gb`, GiB less what live reservations haven't reached) decides, not the slot's mem=. A document
    without it (no pool's since the probe reads it) leaves the slot's own `blocked`."""
    if slot.get("blocked"):
        return str(slot["blocked"])
    need = _need(slot, mem_gb)
    if need and isinstance(free_gb, (int, float)) and free_gb < need:
        return f"{free_gb:g}{MEMORY_BLOCK}, under the {need} GB this job would reserve there"
    return None


def _behind(ahead: list[dict], mem_gb: float, free_gb: float) -> list[dict]:
    """The waiters of ``ahead`` a run asking ``mem_gb`` GB waits behind with ``free_gb`` GiB free, as slot.py counts them (its
    `ahead`): all of them, but that a run asking at most BACKFILL_MAX_GB passes a lane waiter whose own ask is over what is
    free. A priority waiter, or one whose ticket names no ask, is never passed."""
    if mem_gb > BACKFILL_MAX_GB:
        return ahead
    return [w for w in ahead if w.get("priority") or not w.get("mem_gb") or w["mem_gb"] * 1e9 / 2 ** 30 <= free_gb]


def _pool(doc: Any, cpus: int, mem_gb: float) -> tuple[int, int, int, str]:
    """A pool as a job joining its line now finds it: the slots that run, the places it has for this job, the waiters it waits
    behind where it has none, and how to say it; ValueError when no slot of it fits the job. A slot fits by its CPUs: slot.py
    gives a job any slot and raises its memory to the job's ask, so a slot's mem= rules none out (`_blocked`).

    A place is an idle slot with the memory free for what this job reserves there, after the line ahead has taken its own,
    as slot.py admits it: each waiter in turn takes the first idle slot whose reservation (`_need`, by the ask its ticket
    names) fits the memory left, which its reservation then leaves smaller; one that can't stays, and so does every waiter
    behind it but those that pass it (`_behind`). The job has a place for each idle slot it could then take, the memory left
    counted down with each, and none while a waiter it can't pass stays. So ten idle slots beside memory for two runs are
    two places, and none once two waiters have taken them.

    A job with no place waits behind every waiter that stays, those it would pass now too: it passes a waiter only while
    memory is short for that one, and the memory or the slot the job itself waits for goes first to the earlier waiters it
    fits. Counted without them, a pool out of memory with a line of small runs read as the shortest line there was."""
    if not isinstance(doc, dict) or "error" in doc:
        raise ValueError(doc.get("error") if isinstance(doc, dict) else doc or "not read")
    if not doc.get("slots"):
        raise ValueError(f"{doc.get('file')} lists no slot")
    free_gb = doc.get("free_gb")
    slots = [s for s in doc["slots"] if not s.get("trains") and cpulist(s["cpus"]) >= cpus]
    if not slots:
        raise ValueError(f"no slot of {doc.get('file')} gives {cpus} CPUs")
    fit = [{**s, "blocked": _blocked(s, free_gb, mem_gb)} for s in slots]
    # a window (slot.py's `window ...` reason) closes a slot for hours or days, unlike a memory or disk block
    open_ = [s for s in fit if not str(s.get("blocked") or "").startswith("window ")]
    if not open_:
        raise ValueError(f"every slot of {doc.get('file')} that fits is closed by a window ({fit[0]['blocked']})")
    free = [s for s in open_ if s.get("holder") is None and not s.get("blocked")]
    waiters = doc.get("waiters") or []
    # what a run may take now, whatever it asks: the slots nothing holds and nothing blocks for every check
    idle = [s for s in slots if s.get("holder") is None and not s.get("blocked")]
    left = free_gb if isinstance(free_gb, (int, float)) else math.inf

    def take(ask: float) -> bool:
        nonlocal left
        for s in idle:
            if not (need := _need(s, ask)) or need <= left:
                idle.remove(s)
                left -= need
                return True
        return False

    stay, started = [], 0
    for w in waiters:
        if _behind(stay, ask := w.get("mem_gb") or 0, left) or not take(ask):
            stay.append(w)
        else:
            started += 1
    ahead = _behind(stay, mem_gb, left)
    # a waiter retries every 20 s (slot.py's POLL_S), so one that stays this long beside an idle slot waits on memory, and so
    # would this job behind it (cpu-3's head asked 256 GB for hours while its other slots sat free)
    if idle and ahead and (head := max(w.get("minutes") or 0 for w in ahead)) >= STALL_MIN:
        raise Stalled(f"its line is stalled: a run has waited {head} min while slot {idle[0]['name']} is free, for more "
                      f"memory than the {left:g} GB free there", len(waiters))
    if not started and all(s.get("holder") is None and MEMORY_BLOCK in str(s.get("blocked")) for s in open_):
        raise Stalled(f"its line is stalled: every slot of it that fits waits on memory and none runs ({open_[0]['blocked']})",
                      len(waiters))
    places = 0
    while not ahead and take(mem_gb):
        places += 1
    why = (f"slots {', '.join(s['name'] + (' free' if s in free else ' held' if s.get('holder') else ' blocked') for s in fit)}; "
           f"{len(waiters)} waiting" + (f", {places} place{'' if places == 1 else 's'} for this job" if places != len(free) else ""))
    # the line drains through the slots that run: one left idle, or blocked on memory or disk, adds no throughput until it clears
    running = sum(s.get("holder") is not None for s in open_) + started
    return max(running, 1), places, 0 if places else len(stay), why


def _ahead(fit: int, free: int, waiting: int) -> float:
    """A pool's line as a run joining it sees it (`_pool`'s numbers): 0 while it has a place for this run, else the waiters it
    waits behind and this run, per slot that runs. `choose` ranks the pools by it and `expect` turns it into minutes."""
    return 0 if free else (waiting + 1) / fit


def _lean(doc: Any, pool: str | None) -> tuple[float | None, str]:
    """The line a run joins for a slot of Lean ``pool`` on the machine whose `read` document is ``doc``, and how to say it:
    0 with no ``pool`` to take, where the machine caps no Lean, or while a slot is free for every waiter and this run; else the
    waiters ahead and this run, per slot of the pool; infinite when its Lean slots weren't read; None, with why, when they give
    the pool none, so the run would never get one there."""
    if not pool:
        return 0, ""
    lean = doc.get("lean") if isinstance(doc, dict) else None
    if not isinstance(lean, dict) or "error" in lean:
        return math.inf, f"; its Lean slots weren't read ({lean.get('error') if isinstance(lean, dict) else 'no answer'})"
    if (pools := lean.get("pools")) is None:
        return 0, "; no Lean slots to wait for there"
    if not (n := (pools.get(pool) or {}).get("n")):
        return None, f"its Lean slots give `{pool}` none"
    free = sum(s.get("holder") is None for s in pools[pool].get("slots") or [])      # a slot past N is listed only while held
    waiting = len(pools[pool].get("waiters") or [])
    return (0 if free > waiting else (waiting + 1) / n), f"; Lean `{pool}`: {free} of {n} free, {waiting} waiting"


def choose(docs: dict[str, dict | str], cpus: int, mem_gb: float, lean_pool: str | None = None
           ) -> tuple[tuple[str, str] | None, str]:
    """The pool to spill to ((machine, pool)), or None for node 2's queue, and why (module doc); `Unplaceable` when neither can
    take it. ``lean_pool``: the Lean pool the job takes a slot of (`lean_pool`), which `read` was asked for (its `lean`)."""
    cpus = max(cpus, 1)
    n2 = docs.get(AUTO_NODE)
    share = n2.get("share") if isinstance(n2, dict) else None
    if (said := off(n2)) is not None:
        share = None
    else:
        said = (f"{AUTO_NODE}'s share: {share['free']} of {share['share']} CPUs free, {share['claims']} claim(s), "
                f"{share['waiters']} waiting")
    of = f" of Lean pool `{lean_pool}`" if lean_pool else ""
    ranked, stalled, notes, most = [], [], [said], None
    for i, (machine, pool) in enumerate(_pools(docs)):
        doc = docs.get(machine)
        try:
            line, lean = _lean(doc, lean_pool)
            if line is None:
                notes.append(f"{machine}'s {pool} pool skipped: {lean}")
                continue
            fit, free, waiting, why = _pool(doc.get(pool) if isinstance(doc, dict) else doc, cpus, mem_gb)
        except Stalled as e:
            notes.append(f"{machine}'s {pool} pool skipped: {e}")
            stalled.append((e.waiting, i, (machine, pool)))
            continue
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            notes.append(f"{machine}'s {pool} pool skipped: {e}")
            continue
        notes.append(f"{machine}'s {pool} pool: {why}{lean}")
        if free and not line and (most is None or free > most[0]):
            most = (free, (machine, pool))
        ranked.append((line, _ahead(fit, free, waiting), i, (machine, pool)))
    if most is not None:
        places, target = most
        return target, "; ".join(notes) + (f": {target[0]}'s {target[1]} pool has the most places for it ({places}): a slot is "
                                           f"free there now") + (f", and one{of} with it" if of else "")
    starts = share is not None and share.get("fits") and not share["waiters"] and _lean(n2, lean_pool)[0] == 0
    if ranked and starts:
        return None, "; ".join(notes) + (f": no pool has a slot and one{of} free" if of else ": no pool slot is free") + \
            f", and it starts on {AUTO_NODE} now"
    if ranked:
        (line, _, _, target) = min(ranked)
        at = f"{target[0]}'s {target[1]} pool"
        if not of:
            return target, "; ".join(notes) + f": the shortest line per run that fits is {at}"
        if not line:
            return target, "; ".join(notes) + f": {at}, the shortest line per run that fits where a slot{of} is free"
        if line == math.inf:
            return target, "; ".join(notes) + f": no machine's Lean slots were read, so the shortest line per run that fits, {at}"
        return target, "; ".join(notes) + f": no machine has a slot{of} free, so the shortest line for one, per slot: {at}"
    if stalled and share is None:
        (_, _, target) = min(stalled)
        return target, "; ".join(notes) + (f": every pool that fits is stalled and {AUTO_NODE} can't be reached, so the shortest "
                                           f"stalled line, {target[0]}'s {target[1]} pool, where it waits until memory frees")
    if share is None:
        raise Unplaceable("; ".join(notes) + f": no pool fits and {AUTO_NODE} takes no job, so nothing would run it. "
                          f"Give it a node yourself: `research run --on MACHINE --mem-gb N`, a CPU node whose free memory fits "
                          f"it (`research pods list`), or shrink --cpus/--mem-gb to a pool slot's")
    if starts:
        return None, "; ".join(notes) + f": no pool fits, and it starts on {AUTO_NODE} now"
    return None, "; ".join(notes) + f": no pool fits, so {AUTO_NODE}'s line"


class Wait(NamedTuple):
    """What a launch is told of its wait where `choose` sent it (`expect`): the minutes until it starts (0: it starts now; None:
    no estimate), what it waits behind, and whether it is late: what is known of its wait is already past START_WITHIN_MIN."""

    minutes: float | None
    behind: str
    late: bool

    def told(self, place: str) -> str:
        if self.minutes is None:
            return f"no estimate of when it starts on {place}: {self.behind}"
        if not self.minutes:
            return f"it starts now on {place}"
        return f"it should start in about {_whole(self.minutes)} min on {place}: {self.behind}"


def _whole(minutes: float) -> int:
    return max(int(minutes + 0.5), 1)


def expect(docs: dict[str, dict | str], target: tuple[str, str] | None, cpus: int, mem_gb: float, lean_pool: str | None = None
           ) -> Wait:
    """The wait of a job `choose` sent to ``target`` (None: node 2's queue), from the same documents, so it costs no second
    probe. In a pool it is `_ahead` times HOLD_MIN: the runs ahead and this one, per slot that runs, each holding its slot for
    HOLD_MIN. That is wrong by as much as the runs ahead differ from HOLD_MIN, and it sees neither a waiter that gives up (it
    then says too long) nor the launches that read the same line at once (too short). No estimate, and never late for it,
    where nothing read says how long: node 2's line, whose claims have no known end, and a Lean slot's (the job takes its CPU
    slot and waits for the Lean slot inside it; it is late only when the CPU slot alone is). A stalled pool, `choose`'s last
    resort, has no estimate and is late: its line waits on memory, however short it is."""
    cpus = max(cpus, 1)
    doc = docs.get(AUTO_NODE if target is None else target[0])
    line, lean = _lean(doc, lean_pool)
    if line is None:                                           # its Lean slots give the pool none: `choose` sends no job there
        return Wait(None, lean, False)
    then = f", then a wait in that slot for one of Lean pool `{lean_pool}`, which has no estimate ({lean[2:]})" if line else ""
    if target is None:
        if (why := off(doc)) is not None:
            return Wait(None, why, False)
        share = doc["share"]
        if share.get("fits") and not share["waiters"] and not line:
            return Wait(0, "", False)
        return Wait(None, f"{share['free']} of its share's {share['share']} CPUs are free under {share['claims']} claim(s), with "
                          f"{share['waiters']} waiting, and nothing read says when a claim ends{then}", False)
    try:
        fit, free, waiting, _ = _pool(doc.get(target[1]) if isinstance(doc, dict) else doc, cpus, mem_gb)
    except Stalled as e:
        return Wait(None, str(e), True)
    except (ValueError, TypeError, AttributeError) as e:
        return Wait(None, str(e), False)
    minutes = _ahead(fit, free, waiting) * HOLD_MIN
    if not minutes and not line:
        return Wait(0, "", False)
    oldest = max((w.get("minutes") or 0 for w in doc[target[1]].get("waiters") or []), default=0)
    behind = (f"{waiting} waiting ahead" + (f" (the oldest for {oldest:g} min)" if oldest and waiting else "")
              + f" for the {fit} slot{'s that run' if fit > 1 else ' that runs'} there")
    if not line:
        return Wait(minutes, behind, minutes > START_WITHIN_MIN)
    cpu = f"about {_whole(minutes)} min for a CPU slot ({behind})" if minutes else "a CPU slot is free there now"
    return Wait(None, cpu + then, minutes > START_WITHIN_MIN)


def smaller(docs: dict[str, dict | str], cpus: int, mem_gb: float, lean_pool: str | None = None
            ) -> tuple[int, float, tuple[str, str]] | None:
    """The smallest change of a job's request that starts it now: (CPUs, GB, a pool that takes it), the request cut to a
    slot that is free in some pool and to the memory free on its machine (`_blocked`), where `expect` then gives it no wait
    (the first such pool; `choose` takes the one with the most places for it). Of several, the one that keeps the most of
    what was asked, as its share of the CPUs plus its share of the memory. None when no smaller request starts now."""
    cpus = max(cpus, 1)
    shapes = set()
    for machine, pool in _pools(docs):
        doc = docs.get(machine)
        doc = doc.get(pool) if isinstance(doc, dict) else None
        if not isinstance(doc, dict):
            continue
        free_gb = doc.get("free_gb")
        gb = min(mem_gb, int(free_gb * 2 ** 30 / 1e9)) if isinstance(free_gb, (int, float)) else mem_gb
        for s in doc.get("slots") or []:
            if not s.get("trains") and s.get("holder") is None and not s.get("blocked") and (gb > 0 or not mem_gb):
                shapes.add((min(cpus, cpulist(s["cpus"])), gb))
    shapes.discard((cpus, mem_gb))
    for c, m in sorted(shapes, key=lambda cm: (-(cm[0] / cpus + (cm[1] / mem_gb if mem_gb else 1)), cm)):
        for target in _pools(docs):
            if expect(docs, target, c, m, lean_pool).minutes == 0:
                return c, m, target
    return None
