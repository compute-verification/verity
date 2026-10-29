"""`research run --on auto`'s placement: a CPU node's pool (`SPILL`) first, node 2's Verity share (the queue, `AUTO_NODE`) only
for a job no pool fits.

Node 2's share is 48 cores that queued runs claim whole (research.cores) in one first-come line; on 19 Oct two runs held all 48
from 17:40Z for hours (a 16-job quick tier and a 32-core Lean check) while 56 runs waited, the oldest since 17:42Z, a 4-CPU suite
36th in line, with cpu-1's idle cores unused. So before queueing, `read` asks each machine at once, over the ssh `research run
--on` uses, as `python3 -` with research.cores and the source tree's tools/verity/check/slot.py inlined (both stdlib only):
node 2 for its share's live claims and waiters, and each SPILL machine for its pool's `slot.py --status --json`. `choose`
then takes, in order:

  - a pool where a slot that fits the job (not trains-only, at least its CPUs and its memory: the slot's mem= reserves the
    run, with no cgroup cap) is free and unblocked for every waiter ahead and this job;
  - else the pool with the shortest wait, as waiters ahead per fitting slot;
  - node 2 only when no pool fits the job or none can be read, and node 2 itself can be. Node 2 is the only GPU node, and on
    24 Oct 02:10Z its GPUs sat at 0% under a load of 235 on 120 cores from CPU work while cpu-5 ran at 8 of 192 (top).

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
#: (machine, pool) where `--on auto` may spill: every CPU node's `research` pool, which is any lane's (slot_only.POOLS on cpu-1).
#: With only cpu-1, 2, 3, 5 and 6 listed, 89% of 28 Oct's 122,705 waiting run-minutes passed while a slot of the same pool was
#: free on another node. A node whose registry deadline has passed is skipped (`up`), so a stopped one costs no read.
SPILL = tuple((f"vy-nebius-cpu-{i}", "research") for i in (1, 5, 3, 2, 6, 4, 7, 8, 9, 10, 11, 12))
DESCRIPTION = Path("infra/cluster/descriptions/nebius.toml")
SLOT_PY = Path("tools/verity/check/slot.py")
READ_TIMEOUT_S = 30
#: a machine whose DISK_PATH has fewer free inodes or GB than these takes no `--on auto` job: on 27 Oct cpu-5 and cpu-6 ran out of
#: inodes with half their bytes free, and every job placed there failed shipping its source or mid-build with ENOSPC
DISK_PATH, MIN_FREE_INODES, MIN_FREE_GB = "/workspace", 400_000, 50
#: a node's Lean slot caps (tools/verity/check/lean_slot.py): a records run (`lean_changed.py --records`) waits there for a build
#: slot after its pool slot, so `choose` ranks such a run by both lines
LEAN_SLOTS = "/workspace/research/locks/lean-slots"


class Unplaceable(Exception):
    """No pool fits the job and node 2 can't be read, or takes runs only through its pools (remote.RUNS_VIA_SLOT_FILE): its
    queue would place it where nothing runs it (26 Oct: node 2 stopped, and an `--on auto` job too big for a pool slot was
    refused there with a stale admission window). A slot-only node 2 offered its share to a 33-CPU job and then refused it."""

PROBE = '''
import fcntl, json, os, sys, types
from datetime import datetime, timezone
from pathlib import Path


def load(name, src, env):
    os.environ.update(env)
    m = types.ModuleType(name)
    sys.modules[name] = m
    exec(compile(src, name + ".py", "exec"), m.__dict__)
    return m


def share(cpus, d, root):
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


def lean_build():
    f = Path(LEAN)
    if not f.is_file():
        return None
    n = next((int(w[1]) for w in (l.split("#")[0].split() for l in f.read_text().splitlines()) if len(w) == 2 and w[0] == "build"
              and w[1].isdigit()), 0)
    free = 0
    for i in range(n):
        p = f.parent / f"lean-build-{i}.lock"
        if not p.exists():
            free += 1
            continue
        fd = os.open(p, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            free += 1
        except BlockingIOError:
            pass
        finally:
            os.close(fd)
    line = f.parent / "lean-line-build"
    return {"n": n, "free": free, "waiters": len(list(line.iterdir())) if line.is_dir() else 0}


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
    if os.path.exists(SLOT_ONLY):
        out["slot_only"] = SLOT_ONLY
short = disk_short()
if POOLS:
    out["lean"] = doc(lean_build)
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
            "SLOT_ONLY": remote.RUNS_VIA_SLOT_FILE, "LEAN": LEAN_SLOTS}
    return ("".join(f"{k} = {v!r}\n" for k, v in head.items()) + PROBE).encode()


def _reach(name: str) -> list[str]:
    from . import remote

    t = remote.pod_target(remote.resolve_machine(name, AUTO_PROJECT))
    return [*remote.ssh_argv(t["host"], t["port"], user=t["user"]), "python3 -"]


def _ssh(argv: list[str], data: bytes) -> subprocess.CompletedProcess:
    return subprocess.run(argv, input=data, capture_output=True, timeout=READ_TIMEOUT_S)


def up(machine: str, now: float | None = None) -> bool:
    """Whether the registry gives ``machine`` no deadline or one still ahead (a node past it is stopped or about to be). An
    unlisted machine (remote.Refused) or an unreadable deadline (ValueError) raises, and `read` records it as the machine's."""
    import datetime as dt

    from . import remote

    dl = remote.resolve_machine(machine, AUTO_PROJECT).get("deadline")
    if not dl:
        return True
    end = dt.datetime.fromisoformat(str(dl).replace("Z", "+00:00")).timestamp()
    return end > (dt.datetime.now(dt.UTC).timestamp() if now is None else now)


def read(source: Path, cpus: int, *, reach: Callable[[str], list[str]] | None = None, run: Callable | None = None,
         alive: Callable[[str], bool] | None = None) -> dict[str, dict | str]:
    """Each machine's documents (node 2's under "share", a pool's under its name, its Lean build slots under "lean"), read at
    once, or why they couldn't be; a machine past its registry deadline (`up`) isn't asked."""
    reach, run, alive = reach or _reach, run or _ssh, alive or up
    share = share_cpus(source)
    asks: dict[str, list[str]] = {AUTO_NODE: []}
    for machine, pool in SPILL:
        asks.setdefault(machine, []).append(pool)

    def one(item: tuple[str, list[str]]) -> dict | str:
        name, pools = item
        try:
            if name != AUTO_NODE and not alive(name):
                return "past its registry deadline"
            r = run(reach(name), payload(source, cpus, share if name == AUTO_NODE else None, pools))
        except subprocess.TimeoutExpired:
            return f"no answer in {READ_TIMEOUT_S} s"
        except Exception as e:                    # not listed, not this project's, no source, bad deadline: never chosen
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
    free = [s for s in fit if s.get("holder") is None and not s.get("blocked")]
    waiting = len(doc.get("waiters") or [])
    why = (f"slots {', '.join(s['name'] + (' free' if s in free else ' held' if s.get('holder') else ' blocked') for s in fit)}; "
           f"{waiting} waiting")
    return len(fit), len(free), waiting, why


def _lean(doc: Any) -> tuple[int, int, int] | None:
    """A node's Lean build slots: (N, free, waiters), or None when it wasn't read or has none."""
    lean = doc.get("lean") if isinstance(doc, dict) else None
    if not isinstance(lean, dict) or "error" in lean or not lean.get("n"):
        return None
    return int(lean["n"]), int(lean.get("free") or 0), int(lean.get("waiters") or 0)


def choose(docs: dict[str, dict | str], cpus: int, mem_gb: float, lean: bool = False) -> tuple[tuple[str, str] | None, str]:
    """The pool to spill to ((machine, pool)), or None for node 2's queue, and why (module doc); `Unplaceable` when neither can
    take it. With ``lean`` (a records run), a node counts only with a Lean build slot, and its line is the longer of the two."""
    cpus = max(cpus, 1)
    n2 = docs.get(AUTO_NODE)
    share = n2.get("share") if isinstance(n2, dict) else None
    if isinstance(n2, dict) and n2.get("slot_only"):
        share, said = None, f"{AUTO_NODE} takes runs only through its pools ({n2['slot_only']}), so not from its share"
    elif not isinstance(share, dict) or "error" in share:
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
        rank = (waiting + 1) / fit
        if lean:
            builds = _lean(doc)
            if builds is None:
                notes.append(f"{machine}'s {pool} pool skipped: no Lean build slots read there")
                continue
            n, bfree, bwait = builds
            why += f"; Lean builds {bfree} of {n} free, {bwait} waiting"
            if free > waiting and bfree <= bwait:
                free = 0
            rank = max(rank, (bwait + 1) / n)
        notes.append(f"{machine}'s {pool} pool: {why}")
        if free > waiting:
            return (machine, pool), "; ".join(notes) + ": a slot is free there now"
        ranked.append((rank, i, (machine, pool)))
    if ranked:
        (_, _, target) = min(ranked)
        return target, "; ".join(notes) + f": the shortest line per run that fits is {target[0]}'s {target[1]} pool"
    if share is None:
        cant = "takes no run from its share" if isinstance(n2, dict) and n2.get("slot_only") else "can't be reached"
        raise Unplaceable("; ".join(notes) + f": no pool fits and {AUTO_NODE} {cant}, so nothing would run it. "
                          f"Give it a node yourself: `research run --on MACHINE --mem-gb N`, a CPU node whose free memory fits "
                          f"it (`research pods list`), or `--on {AUTO_NODE} --pool POOL` where a slot of that pool gives its "
                          f"CPUs, or shrink --cpus/--mem-gb to a pool slot's")
    if share.get("fits") and not share["waiters"]:
        return None, "; ".join(notes) + f": no pool fits, and it starts on {AUTO_NODE} now"
    return None, "; ".join(notes) + f": no pool fits, so {AUTO_NODE}'s line"
