"""`research run --on auto`'s placement: a free slot of a CPU node's pool (`SPILL`) first, then node 2's Verity share (the queue,
`AUTO_NODE`) when it can start the job now, then the shortest pool line, and node 2's line for a job no pool fits.

Node 2's share is 48 cores that queued runs claim whole (research.cores) in one first-come line, so two large runs (a 16-job
quick tier and a 32-core Lean check) can hold all 48 for hours while dozens of small runs wait behind them, with cpu-1's idle
cores unused. So before queueing, `read` asks each machine at once, over the ssh `research run
--on` uses, as `python3 -` with research.cores and the source tree's tools/verity/check/slot.py inlined (both stdlib only):
node 2 for its share's live claims and waiters, and each SPILL machine for its pool's `slot.py --status --json`. `choose`
then takes, in order:

  - a pool where a slot that fits the job (not trains-only, at least its CPUs and its memory: the slot's mem= reserves the
    run, with no cgroup cap) is free and unblocked for every waiter ahead and this job;
  - else node 2's share when it can start the job now, with no one waiting there (a run once queued 100 min behind cpu-6's 8
    held slots and 9 waiters while node 2's 48 CPUs sat free);
  - else the pool with the shortest wait, as waiters ahead per fitting slot;
  - node 2 only when no pool fits the job or none can be read, and node 2 itself can be. Node 2 is the only GPU node,
    and CPU work there has held its GPUs at 0% under a load of 235 on 120 cores while cpu-5 ran at 8 of 192. A node 2
    that carries remote.RUNS_VIA_SLOT_FILE counts as unread: its launcher refuses a run that doesn't come through a check
    slot.

A GPU job, a run naming its own node, or one asking more than a pool slot gives stays on node 2's queue. A machine that can't
be read is skipped; with node 2 unread and no pool fitting, the job is refused (`Unplaceable`) with where to run it. A pool
run waits up to slot_only.POOL_WAIT_S (4 h) in its line and then exits 75 without running; its --timeout counts from taking
the slot, and a wait inside it for a Lean audit slot (lean_slot.py) moves it later by as long.
"""
from __future__ import annotations

import json
import subprocess
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

AUTO_NODE, AUTO_PROJECT = "vy-nebius-2", "verity"
#: (machine, pool) where `--on auto` may spill: the `research` pools of cpu-1, 2, 3, 5, 6, 7, 8, 9 and 11 are any lane's
#: (slot_only.POOLS on cpu-1). A pool left out sits idle while the listed ones queue: cpu-7, 8, 9 and 11 sat near load 0 while
#: cpu-1, 5 and 6 queued 20 or more runs each. cpu-4's cores are the train slot's, proofs' and soundness's.
SPILL = (("vy-nebius-cpu-1", "research"), ("vy-nebius-cpu-5", "research"), ("vy-nebius-cpu-3", "research"),
         ("vy-nebius-cpu-2", "research"), ("vy-nebius-cpu-6", "research"), ("vy-nebius-cpu-7", "research"),
         ("vy-nebius-cpu-8", "research"), ("vy-nebius-cpu-9", "research"), ("vy-nebius-cpu-11", "research"))
DESCRIPTION = Path("infra/cluster/descriptions/nebius.toml")
SLOT_PY = Path("tools/verity/check/slot.py")
READ_TIMEOUT_S = 30
#: a machine whose DISK_PATH has fewer free inodes or GB than these takes no `--on auto` job: a machine out of inodes with half
#: its bytes free fails every job placed there, shipping its source or mid-build, with ENOSPC. One run of
#: a large suite (tools/research's) holds some 250k pytest temp files at once, so a floor below a few of those still fails jobs
DISK_PATH, MIN_FREE_INODES, MIN_FREE_GB = "/workspace", 1_000_000, 50


class Unplaceable(Exception):
    """No pool fits the job and node 2 can't be read: its queue would place it on a node that isn't up (a stopped node 2
    refuses an `--on auto` job too big for a pool slot with a stale admission window)."""

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


def share(cpus, d, root):
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
for pool, slots in POOLS.items():
    out[pool] = {"error": f"disk: {short}"} if short else \
        doc(lambda: load("auto_slot_" + pool, SLOT_SRC, {"CHECK_SLOTS": slots}).status_doc(datetime.now(timezone.utc)))
print(json.dumps(out))
'''


def cpulist(text: str) -> int:
    n = 0
    for part in str(text).split(","):
        lo, _, hi = part.partition("-")
        n += int(hi or lo) - int(lo) + 1
    return n


def share_cpus(source: Path) -> str | None:
    """The CPUs of AUTO_PROJECT's share on AUTO_NODE in the source tree's cluster description, None when it says none."""
    try:
        desc = tomllib.loads((Path(source) / DESCRIPTION).read_text())
    except (OSError, ValueError):
        return None
    return next((s["cpus"] for s in desc.get("shares", []) if s.get("project") == AUTO_PROJECT and s.get("node") == AUTO_NODE
                 and isinstance(s.get("cpus"), str)), None)


def payload(source: Path, cpus: int, share: str | None, pools: list[str]) -> bytes:
    from . import cores, remote
    from .pods.nebius import slot_only

    head = {"CORES_SRC": Path(cores.__file__).read_text(), "SLOT_SRC": (Path(source) / SLOT_PY).read_text() if pools else "",
            "JOB_CPUS": max(cpus, 1), "SHARE": share, "CLAIMS": str(cores.CLAIMS), "CPU_ROOT": str(cores.CPU_ROOT),
            "POOLS": {p: slot_only.POOLS[p] for p in pools}, "DISK": DISK_PATH, "MIN_INODES": MIN_FREE_INODES, "MIN_GB": MIN_FREE_GB,
            "VIA_SLOT": remote.RUNS_VIA_SLOT_FILE}
    return ("".join(f"{k} = {v!r}\n" for k, v in head.items()) + PROBE).encode()


def _reach(name: str) -> list[str]:
    from . import remote

    t = remote.pod_target(remote.resolve_machine(name, AUTO_PROJECT))
    return [*remote.ssh_argv(t["host"], t["port"], user=t["user"]), "python3 -"]


def _ssh(argv: list[str], data: bytes) -> subprocess.CompletedProcess:
    return subprocess.run(argv, input=data, capture_output=True, timeout=READ_TIMEOUT_S)


def read(source: Path, cpus: int, *, reach: Callable[[str], list[str]] | None = None, run: Callable | None = None) -> dict[str, dict | str]:
    """Each machine's documents (node 2's under "share", a pool's under its name), read at once, or why they couldn't be."""
    reach, run = reach or _reach, run or _ssh
    share = share_cpus(source)
    asks: dict[str, list[str]] = {AUTO_NODE: []}
    for machine, pool in SPILL:
        asks.setdefault(machine, []).append(pool)

    def one(item: tuple[str, list[str]]) -> dict | str:
        name, pools = item
        try:
            r = run(reach(name), payload(source, cpus, share if name == AUTO_NODE else None, pools))
        except subprocess.TimeoutExpired:
            return f"no answer in {READ_TIMEOUT_S} s"
        except Exception as e:                                       # not listed, not this project's, no source: never chosen
            return str(e) or type(e).__name__
        if r.returncode:
            err = r.stderr.decode(errors="replace").strip().splitlines()
            return f"exit {r.returncode}: {err[-1] if err else 'no output'}"
        try:
            doc = json.loads(r.stdout)
        except ValueError:
            return f"its answer is not JSON: {r.stdout[:120]!r}"
        return doc if isinstance(doc, dict) else f"its answer is not a JSON object: {r.stdout[:120]!r}"

    with ThreadPoolExecutor(max_workers=len(asks)) as ex:
        return dict(zip(asks, ex.map(one, asks.items())))


def _pool(doc: Any, cpus: int, mem_gb: float) -> tuple[int, int, int, str]:
    """A pool's fitting slots, free ones among them and waiters; ValueError when no slot of it fits the job."""
    if not isinstance(doc, dict) or "error" in doc:
        raise ValueError(doc.get("error") if isinstance(doc, dict) else doc or "not read")
    fit = [s for s in doc.get("slots") or [] if not s.get("trains") and cpulist(s["cpus"]) >= cpus
           and (s.get("mem") is None or s["mem"] >= mem_gb)]  # slot.py raises a slot without mem= to what the run asks
    if not fit:
        raise ValueError(f"no slot of {doc.get('file')} gives {cpus} CPUs and {mem_gb:g} GB")
    # a window (slot.py's `window ...` reason) closes a slot for hours or days, unlike a memory or disk block
    open_ = [s for s in fit if not str(s.get("blocked") or "").startswith("window ")]
    if not open_:
        raise ValueError(f"every slot of {doc.get('file')} that fits is closed by a window ({fit[0]['blocked']})")
    free = [s for s in open_ if s.get("holder") is None and not s.get("blocked")]
    waiting = len(doc.get("waiters") or [])
    why = (f"slots {', '.join(s['name'] + (' free' if s in free else ' held' if s.get('holder') else ' blocked') for s in fit)}; "
           f"{waiting} waiting")
    # the line drains through the slots that run: a slot blocked on memory or disk adds no throughput until it clears, and a
    # pool whose idle slots all wait on memory reservations would otherwise rank as a short line
    running = [s for s in open_ if s.get("holder") is not None or not s.get("blocked")]
    return max(len(running), 1), len(free), waiting, why


def choose(docs: dict[str, dict | str], cpus: int, mem_gb: float) -> tuple[tuple[str, str] | None, str]:
    """The pool to spill to ((machine, pool)), or None for node 2's queue, and why (module doc); `Unplaceable` when neither can
    take it."""
    cpus = max(cpus, 1)
    n2 = docs.get(AUTO_NODE)
    share = n2.get("share") if isinstance(n2, dict) else None
    if not isinstance(share, dict) or "error" in share:
        why = share.get("error") if isinstance(share, dict) else n2 if isinstance(n2, str) else "not read"
        share, said = None, f"{AUTO_NODE}'s share wasn't read ({why})"
    else:
        said = (f"{AUTO_NODE}'s share: {share['free']} of {share['share']} CPUs free, {share['claims']} claim(s), "
                f"{share['waiters']} waiting")
    ranked, notes = [], [said]
    for i, (machine, pool) in enumerate(SPILL):
        doc = docs.get(machine)
        try:
            fit, free, waiting, why = _pool(doc.get(pool) if isinstance(doc, dict) else doc, cpus, mem_gb)
        except (ValueError, KeyError, TypeError) as e:
            notes.append(f"{machine}'s {pool} pool skipped: {e}")
            continue
        notes.append(f"{machine}'s {pool} pool: {why}")
        if free > waiting:
            return (machine, pool), "; ".join(notes) + ": a slot is free there now"
        ranked.append(((waiting + 1) / fit, i, (machine, pool)))
    if ranked and share is not None and share.get("fits") and not share["waiters"]:
        return None, "; ".join(notes) + f": no pool slot is free, and it starts on {AUTO_NODE} now"
    if ranked:
        (_, _, target) = min(ranked)
        return target, "; ".join(notes) + f": the shortest line per run that fits is {target[0]}'s {target[1]} pool"
    if share is None:
        raise Unplaceable("; ".join(notes) + f": no pool fits and {AUTO_NODE} can't be reached, so nothing would run it. "
                          f"Give it a node yourself: `research run --on MACHINE --mem-gb N`, a CPU node whose free memory fits "
                          f"it (`research pods list`), or shrink --cpus/--mem-gb to a pool slot's")
    if share.get("fits") and not share["waiters"]:
        return None, "; ".join(notes) + f": no pool fits, and it starts on {AUTO_NODE} now"
    return None, "; ".join(notes) + f": no pool fits, so {AUTO_NODE}'s line"
