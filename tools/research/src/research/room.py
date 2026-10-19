"""`research room [--ramlock DIR] [--json]`: this machine's room for a queued job beside its check slots, which `research run --on
auto` reads from each node its kind may run on (infra/cluster's `cluster submit --room`).

    available_gb   research.mem's: the limit less what is in use and what live reservations still hold
    idle_slots_gb  what every check slot that isn't running a check is promised (its mem=, GiB as GB): a slot's check reserves its
                   mem= only once it runs (slot.py `reserve`), and a job that took it meanwhile would hold the slot shut
    headroom_gb    available_gb - idle_slots_gb: what a job may take without making a check wait
    pools          each slot pool's (`LOCKS/<pool>/slots`, `research run --pool`) slots, free slots and the least mem= of them
    deadline_in_s  seconds to the machine's stop (/etc/research/deadline), or None

A slot with mem= is running a check while its `check-NAME.json` reservation is live; one without mem= reserves nothing and is
counted free. Read-only: it takes no lock a slot's taker takes."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from . import mem as M

LOCKS = Path(os.environ.get("RESEARCH_LOCKS_DIR", "/workspace/research/locks"))
DEADLINE = Path(os.environ.get("RESEARCH_DEADLINE_FILE", "/etc/research/deadline"))


def parse_slots(text: str) -> list[dict[str, Any]]:
    """slot.py's slots file: `NAME CPUS [k=v ...]  # comment` per line, with mem= (GiB) as GB."""
    out = []
    for line in text.splitlines():
        f = line.split("#")[0].split()
        if len(f) < 2:
            continue
        opts = dict(x.split("=", 1) for x in f[2:] if "=" in x)
        mem = opts.get("mem")
        out.append({"name": f[0], "cpus": f[1], "mem_gb": round(int(mem) * 2 ** 30 / 1e9, 1) if mem and mem.isdigit() else None})
    return out


def _deadline_in_s(path: Path, now: float) -> int | None:
    try:
        return int(int(path.read_text().split()[0]) - now)
    except (OSError, ValueError, IndexError):
        return None


def snapshot(ramlock: Path, locks: Path = LOCKS, deadline: Path = DEADLINE, now: float | None = None) -> dict[str, Any]:
    s = M.snapshot(ramlock)
    live = {x["file"] for x in s["reservations"] if x["live"]}
    files = {None: locks / "slots", **{d.name: d / "slots" for d in sorted(locks.iterdir()) if (d / "slots").is_file()}} \
        if locks.is_dir() else {}
    slots = []
    for pool, f in files.items():
        try:
            text = f.read_text()
        except OSError:
            continue
        for sl in parse_slots(text):
            slots.append({**sl, "pool": pool, "running": f"check-{sl['name']}.json" in live if sl["mem_gb"] else None})
    idle = round(sum(sl["mem_gb"] for sl in slots if sl["running"] is False), 1)
    pools: dict[str, dict[str, Any]] = {}
    for sl in slots:
        if sl["pool"] is not None:
            p = pools.setdefault(sl["pool"], {"slots": 0, "free": 0, "mem_gb": None})
            p["slots"] += 1
            p["free"] += sl["running"] is not True
            if sl["mem_gb"] is not None:
                p["mem_gb"] = sl["mem_gb"] if p["mem_gb"] is None else min(p["mem_gb"], sl["mem_gb"])
    return {"available_gb": s["available_gb"], "idle_slots_gb": idle, "headroom_gb": round(s["available_gb"] - idle, 1),
            "slots": slots, "pools": pools, "deadline_in_s": _deadline_in_s(deadline, time.time() if now is None else now)}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="research room", description=__doc__.split("\n\n")[0])
    ap.add_argument("--ramlock", default=os.environ.get("RESEARCH_RAMLOCK_DIR", M.RAMLOCK_DIR), metavar="DIR")
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args(argv)
    s = snapshot(Path(ns.ramlock), LOCKS, DEADLINE)
    if ns.json:
        print(json.dumps(s, default=str))
        return 0
    print(f"headroom {s['headroom_gb']:g} GB = available {s['available_gb']:g} - idle check slots {s['idle_slots_gb']:g}")
    for sl in s["slots"]:
        state = "running" if sl["running"] else "idle" if sl["running"] is False else "no mem="
        print(f"  {sl['pool'] + '/' if sl['pool'] else ''}{sl['name']} {sl['cpus']} mem {sl['mem_gb']} GB: {state}")
    return 0


__all__ = ["parse_slots", "snapshot", "main"]
