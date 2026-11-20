"""What a live run got where it executes, and why it waits, from research.scopemem's `node` reading of its machine.

    cpus     the workload's affinities, and its cgroups' AllowedCPUs (cpuset.cpus) where one sets them, else the cpuset
             they allow (cpuset.cpus.effective): a check slot pins with `taskset`, a queued job's scope with AllowedCPUs
    gpus     every live gpu-lease lease whose owner line names the run (`run=`, gpu-lease's RESEARCH_RUN_ID)
    slot     the check slot (slot.py) or Lean slot (lean_slot.py) whose holder names the run
    waits    a live gpu-lease wait file naming the run: its place among the directory's waiters and how many GPUs are
             unlocked there.  A check-line or Lean-line ticket naming the run: its place in line, and the slots it may take
             (a lane's check never takes an `only=trains` slot) or its pool's N
"""
from __future__ import annotations

from typing import Any

from . import scopemem

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


def gpu_waits(run_id: str, node: dict[str, Any]) -> list[str]:
    out = []
    for d, ld in (node.get("leases") or {}).items():
        waits = ld.get("waits") or []
        for i, w in enumerate(waits):
            if w.get("run") != run_id:
                continue
            n = int(w["n"]) if str(w.get("n", "")).isdigit() else 1
            line = (f"gpu-lease {d}: {n} GPU(s)" + (f" on {w['on']}" if w.get("on") else "") + (" preemptible" if w.get("preempt") == "1" else "")
                    + f" since {_t(w.get('since'))}; {i} waiter(s) ahead, {ld['gpus'] - ld['locked']} of {ld['gpus']} GPUs unlocked")
            out.append(line)
    return out


def _state(s: dict[str, Any]) -> str:
    return "held" if s.get("holder") else f"blocked ({s['blocked']})" if s.get("blocked") else "free"


def _held(h: dict[str, Any]) -> str:
    """Since when a slot's holder has held it, where its record says (slot.py's `took`); else when its process started, as that:
    a check is one process from joining the line on, so that start is not when it took the slot."""
    return f"since {h['took']}" if h.get("took") else f"for a time not recorded (its process started {h.get('since')})"


def slots(run_id: str, node: dict[str, Any]) -> tuple[list[str], list[str]]:
    """The check and Lean slots the run holds, and its places in their lines."""
    held: list[str] = []
    waits: list[str] = []
    sl = node.get("slots") or {}
    check = sl.get("check") or {}
    for s in check.get("slots") or ():
        if (s.get("holder") or {}).get("run") == run_id:
            held.append(f"check slot {s['name']} (cores {s['cpus']}) {_held(s['holder'])}")
    for i, w in enumerate(check.get("waiters") or ()):
        if w.get("run") == run_id:
            mine = [s for s in check.get("slots") or () if w.get("priority") or not s.get("trains")]
            trains = [s["name"] for s in check.get("slots") or () if s not in mine]
            waits.append(f"check slot since {w['since']}" + (" with priority" if w.get("priority") else "") + f": {i} ahead in line; "
                         + ", ".join(f"{s['name']} {_state(s)}" for s in mine) + (f" ({', '.join(trains)}: candidates only)" if trains else ""))
    for pool, p in ((sl.get("lean") or {}).get("pools") or {}).items():
        for s in p.get("slots") or ():
            if (s.get("holder") or {}).get("run") == run_id:
                held.append(f"Lean slot {pool}/{s['index']} {_held(s['holder'])}")
        for i, w in enumerate(p.get("waiters") or ()):
            if w.get("run") == run_id:
                busy = sum(1 for s in p.get("slots") or () if s.get("holder") and not s.get("past_n"))
                waits.append(f"Lean slot of pool {pool} since {w['since']}: {i} ahead in line; {busy} of its {p['n']} slot(s) held")
    return held, waits


def lines(run_id: str, reading: dict[str, Any]) -> list[tuple[str, str]]:
    """(label, text) for `research status` and `inspect`: mem, cpus, gpus, slot and waits, from `remote.node_readings`' entry
    (or a local `scopemem.node` reading's)."""
    run, node = reading.get("run") or {}, reading.get("node") or {}
    out = [("mem", scopemem.render(m)) for m in run.get("memory") or ()]
    if c := cpus(run):
        out.append(("cpus", c))
    out += [("gpus", g) for g in gpus(run_id, node)]
    held, waits = slots(run_id, node)
    out += [("slot", h) for h in held]
    out += [("waits", w) for w in gpu_waits(run_id, node) + waits]
    for k, doc in (node.get("slots") or {}).items():
        if isinstance(doc, dict) and doc.get("error"):
            out.append(("slots", f"{k} slots unread: {doc['error']}"))
    return out
