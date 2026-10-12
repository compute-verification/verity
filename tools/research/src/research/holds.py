"""What a live run got where it executes, and why it waits, from research.scopemem's `node` reading of its machine.

    cpus     the workload's affinities, and its cgroups' AllowedCPUs (cpuset.cpus) where one sets them, else the cpuset
             they allow (cpuset.cpus.effective): a check slot pins with `taskset`, a queued job's scope with AllowedCPUs
    gpus     every live gpu-lease lease whose owner line names the run (`run=`, gpu-lease's RESEARCH_RUN_ID)
    slot     the check slot (slot.py) or Lean slot (lean_slot.py) whose holder names the run
    waits    a live gpu-lease wait file naming the run: its place among the directory's waiters and how many GPUs are
             unlocked there; on node 1, the lease pool's record and the cap that binds it (`n1_cap`).  A check-line or
             Lean-line ticket naming the run: its place in line, and the slots it may take (a lane's check never takes a
             trains-only slot) or its pool's N
"""
from __future__ import annotations

import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import scopemem

#: the cluster's description beside this package (infra/cluster), for node 1's lease-pool quotas
DESCRIPTION = Path(__file__).resolve().parents[4] / "infra" / "cluster" / "descriptions" / "nebius.toml"
NODE1 = "vy-nebius-1"
DEFAULT_LEASE_DIR = "/run/gpu-lease"


def _t(s: Any) -> str:
    """A UTC timestamp to the minute."""
    s = str(s or "-")
    return s[:16] + "Z" if len(s) >= 16 and s[10:11] == "T" else s


def cpus(run: dict[str, Any]) -> str | None:
    mem = [m for m in run.get("memory") or () if m.get("error") != "gone"]
    parts = [" / ".join(run["affinity"]) + " (affinity)"] if run.get("affinity") else []
    own = [f"AllowedCPUs {m['cpuset']} at {m['cgroup']}" for m in mem if m.get("cpuset")]
    eff = list(dict.fromkeys(m["cpus_effective"] for m in mem if m.get("cpus_effective")))
    parts += own or ([f"its cgroup allows {', '.join(eff)}"] if eff else [])
    return "; ".join(parts) or None


def gpus(run_id: str, node: dict[str, Any]) -> list[str]:
    """One line per lease (a pid's GPUs in one directory) that names the run."""
    out = []
    for d, ld in (node.get("leases") or {}).items():
        mine: dict[str, list[dict]] = {}
        for o in ld.get("owners") or ():
            if o.get("run") == run_id:
                mine.setdefault(o.get("pid", "?"), []).append(o)
        for pid, os_ in mine.items():
            o = os_[0]
            out.append(f"GPU {','.join(str(x['index']) for x in os_)}" + ("" if d == DEFAULT_LEASE_DIR else f" of {d}")
                       + f"  pid {pid} since {_t(o.get('since'))}" + (f" until {_t(o['until'])}" if o.get("until") else "")
                       + (f" mem {o['mem']}" if o.get("mem") else "") + (" preemptible" if o.get("preempt") == "1" else "")
                       + (" timed" if o.get("timed") == "1" else ""))
    return out


def n1_cap(lease_dir: str, n: int, waits: list[dict], limits: str | None, now: datetime, desc: Path = DESCRIPTION) -> str | None:
    """The cap node 1's lease pool in `lease_dir` holds a request for `n` GPUs to (n1_lease.py): at most its queue's nominal quota,
    less what the queue's other admitted Workloads reserve, and `borrow` GPUs more only while every waiter is --preemptible
    and before `borrow-until`, never more than `max`.  The quota is the description's; max, borrow and borrow-until are the
    pool's `n1-lease.limits` when it writes one (under a policy command), else unknown here.  None without the description."""
    try:
        doc = tomllib.loads(desc.read_text())
        pools = doc["policy"][NODE1]["lease-pool"]
        name, pool = next((k, p) for k, p in pools.items() if p.get("dir", DEFAULT_LEASE_DIR) == lease_dir)
        queue = pool.get("queue", name)
        quota = next(int(w["gpus"]) for w in doc["workstreams"] if w["name"] == queue and w["node"] == NODE1)
    except (OSError, KeyError, StopIteration, ValueError, tomllib.TOMLDecodeError):
        return None
    every = bool(waits) and all(w.get("preempt") == "1" for w in waits)
    lim = dict(tok.split("=", 1) for tok in (limits or "").split() if "=" in tok)
    try:
        mx, borrow = int(lim["max"]), int(lim["borrow"])
    except (KeyError, ValueError):
        cap = None if every else quota
        why = (f"node 1's {name} pool grants at most the {queue} queue's quota, {quota} GPU(s), less its other Workloads'"
               + (", and borrows more only while every waiter is --preemptible (its limits are unrecorded: no n1-lease.limits)" if every
                  else ", while a waiter isn't --preemptible"))
    else:
        until = lim.get("borrow-until") or ""
        lending = every and (not until or now.strftime("%Y-%m-%dT%H:%M:%SZ") < until)
        cap = min(mx, quota + (borrow if lending else 0))
        why = (f"node 1's {name} pool grants at most {cap} GPU(s): the {queue} queue's quota {quota}" + (f" + borrow {borrow}" if lending else "")
               + f", max {mx}, less its other Workloads' ({limits.strip()})")
    return why + (f"; this asks for {n}, more than that" if cap is not None and n > cap else "")


def gpu_waits(run_id: str, machine: str | None, node: dict[str, Any], now: datetime) -> list[str]:
    out = []
    for d, ld in (node.get("leases") or {}).items():
        waits = ld.get("waits") or []
        for i, w in enumerate(waits):
            if w.get("run") != run_id:
                continue
            n = int(w["n"]) if str(w.get("n", "")).isdigit() else 1
            pool = ld.get("pool")
            line = (f"gpu-lease {d}: {n} GPU(s)" + (f" on {w['on']}" if w.get("on") else "") + (" preemptible" if w.get("preempt") == "1" else "")
                    + f" since {_t(w.get('since'))}; {i} waiter(s) ahead, {ld['gpus'] - ld['locked']} of {ld['gpus']} GPUs unlocked")
            if pool:
                line += f"; pool {pool.get('pool')} GPU(s), {pool.get('pending_holders')} holder(s) pending" + \
                        (f", BLOCKED: {pool['blocked']}" if pool.get("blocked") else "")
            if machine == NODE1 and (cap := n1_cap(d, n, waits, ld.get("limits"), now)):
                line += f"; {cap}"
            out.append(line)
    return out


def _state(s: dict[str, Any]) -> str:
    return "held" if s.get("holder") else f"blocked ({s['blocked']})" if s.get("blocked") else "free"


def slots(run_id: str, node: dict[str, Any]) -> tuple[list[str], list[str]]:
    """The check and Lean slots the run holds, and its places in their lines."""
    held: list[str] = []
    waits: list[str] = []
    sl = node.get("slots") or {}
    check = sl.get("check") or {}
    for s in check.get("slots") or ():
        if (s.get("holder") or {}).get("run") == run_id:
            held.append(f"check slot {s['name']} (cores {s['cpus']}) since {s['holder'].get('since')}")
    for i, w in enumerate(check.get("waiters") or ()):
        if w.get("run") == run_id:
            mine = [s for s in check.get("slots") or () if w.get("priority") or not s.get("trains")]
            trains = [s["name"] for s in check.get("slots") or () if s not in mine]
            waits.append(f"check slot since {w['since']}" + (" with priority" if w.get("priority") else "") + f": {i} ahead in line; "
                         + ", ".join(f"{s['name']} {_state(s)}" for s in mine) + (f" ({', '.join(trains)} trains only)" if trains else ""))
    for pool, p in ((sl.get("lean") or {}).get("pools") or {}).items():
        for s in p.get("slots") or ():
            if (s.get("holder") or {}).get("run") == run_id:
                held.append(f"Lean slot {pool}/{s['index']} since {s['holder'].get('took') or s['holder'].get('since')}")
        for i, w in enumerate(p.get("waiters") or ()):
            if w.get("run") == run_id:
                busy = sum(1 for s in p.get("slots") or () if s.get("holder") and not s.get("past_n"))
                waits.append(f"Lean slot of pool {pool} since {w['since']}: {i} ahead in line; {busy} of its {p['n']} slot(s) held")
    return held, waits


def lines(run_id: str, machine: str | None, reading: dict[str, Any], now: datetime | None = None) -> list[tuple[str, str]]:
    """(label, text) for `research status` and `inspect`: mem, cpus, gpus, slot and waits, from `remote.node_readings`' entry
    (or a local `scopemem.node` reading's)."""
    now = now or datetime.now(UTC)
    run, node = reading.get("run") or {}, reading.get("node") or {}
    out = [("mem", scopemem.render(m)) for m in run.get("memory") or ()]
    if c := cpus(run):
        out.append(("cpus", c))
    out += [("gpus", g) for g in gpus(run_id, node)]
    held, waits = slots(run_id, node)
    out += [("slot", h) for h in held]
    out += [("waits", w) for w in gpu_waits(run_id, machine, node, now) + waits]
    for k, doc in (node.get("slots") or {}).items():
        if isinstance(doc, dict) and doc.get("error"):
            out.append(("slots", f"{k} slots unread: {doc['error']}"))
    return out
