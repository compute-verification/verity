#!/usr/bin/env python3
"""`python3 slot_only.py [--runs DIR] [--marker FILE] [--log FILE] [--dry-run]`: on a machine whose MARKER exists (deploy.toml
installs it on vy-nebius-cpu-1), stop every live `research run` whose workload did not come through a check slot (`through_slot`).

A run outside slot.py lands beside the slots' checks without their lock or cores, and on cpu-1 three did in one night (8 Oct),
taking the slice from 586G to 647G beside a merge check. research.remote's launcher refuses such a run before it starts; this
stops one launched from a shipped tree that predates that refusal: SIGTERM to the workload's process group (status.json's pgid,
when a member's environment names the run's RESEARCH_RUN_DIR), and to every other process of the run but its runner, which
then records the run failed. Each stop is a line of LOG. vy-slot-only.timer runs it every minute. Stdlib only: deploy.toml
installs this file alone.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

MARKER = Path("/etc/vy/runs-via-slot")
RUNS = Path("/workspace/research/runs")
LOG = Path("/workspace/monitoring/slot-only.jsonl")
#: slot.py from a shipped tree
SLOT_PATHS = ("tools/verity/check/slot.py",)
#: `env CHECK_SLOTS=...` before slot.py: a pool of its own, with its own line, in one of these files (vy-nebius-cpu-1's
#: `proofs` pool, 13 Oct), so a lane's runs never queue behind the merge checks
POOL = re.compile(r"CHECK_SLOTS=/workspace/research/locks/[A-Za-z0-9_-]+/slots")
#: `research run --pool NAME`'s pools: `research` is any lane's research runs on cores the train slot holds, at nice 19 (INF-27,
#: cpu-1's /workspace/research/locks/research/slots), `proofs` is proofs' Lean pool
POOLS = {"research": "/workspace/research/locks/research/slots", "proofs": "/workspace/research/locks/proofs/slots"}


def in_pool(pool: str, lane: str, command: list[str]) -> list[str]:
    """CMD as a run in `pool`, the one way a lane's run takes it: `env CHECK_SLOTS=POOL python3 tools/verity/check/slot.py --by
    LANE -- nice -n 19 CMD...`, with slot.py from the run's own shipped tree (so the run's cwd is its source)."""
    if pool not in POOLS:
        raise ValueError(f"no pool {pool!r}: the pools are {', '.join(sorted(POOLS))}")
    return ["env", f"CHECK_SLOTS={POOLS[pool]}", "python3", SLOT_PATHS[0], "--by", lane, "--", "nice", "-n", "19", *command]


def through_slot(command: object) -> bool:
    """Whether a workload command is `[env CHECK_SLOTS=POOL] python3 <tree>/check/slot.py [OPTS] -- CMD...`, the way check.py's
    SLOT_WRAP, train.sh and `research queue ready --on` send a check: CMD then runs under a slot's lock and cores. `slot.py
    --status` is not a run."""
    if not (isinstance(command, list) and all(isinstance(c, str) for c in command)):
        return False
    if command[:1] == ["env"] and len(command) > 1 and POOL.fullmatch(command[1]):
        command = command[2:]
    if len(command) < 4 or command[0] != "python3" or command[1] not in SLOT_PATHS or "--" not in command[2:]:
        return False
    return command.index("--", 2) < len(command) - 1


def refusal(machine_marker: Path = MARKER) -> str:
    return (f"this machine takes a `research run` only through its check slots ({machine_marker}): send it as "
            f"`python3 tools/verity/check/slot.py [--by NAME] -- CMD...`, which `check.py --record --on` does")


def _json(p: Path) -> dict:
    try:
        d = json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def _proc_env(pid: int, proc: Path) -> str | None:
    try:
        env = (proc / str(pid) / "environ").read_bytes()
    except OSError:
        return None
    for kv in env.split(b"\0"):
        if kv.startswith(b"RESEARCH_RUN_DIR="):
            return kv.split(b"=", 1)[1].decode(errors="replace")
    return None


def _pgid(pid: int, proc: Path) -> int | None:
    try:
        stat = (proc / str(pid) / "stat").read_text()
    except OSError:
        return None
    return int(stat[stat.rindex(")") + 2:].split()[2])


def members(run_dir: Path, proc: Path = Path("/proc")) -> list[int]:
    """Every pid whose environment names ``run_dir`` as its RESEARCH_RUN_DIR."""
    want = str(run_dir.resolve())
    return sorted(int(x) for x in os.listdir(proc) if x.isdigit() and _proc_env(int(x), proc) == want)


def offenders(runs: Path) -> list[dict]:
    """Each live run under ``runs`` (status.json `running`) whose job.json command is not `through_slot`."""
    out = []
    for st_path in sorted(runs.glob("*/status.json")):
        d = st_path.parent
        st = _json(st_path)
        if st.get("state") != "running":
            continue
        command = (_json(d / "job.json").get("execution") or {}).get("command")
        if command is None or through_slot(command):
            continue
        out.append({"run": d.name, "dir": d, "pgid": st.get("pgid"), "command": command,
                    "runner_pid": _json(d / "launch.json").get("runner_pid")})
    return out


def stop(o: dict, *, proc: Path = Path("/proc"), kill=os.kill, dry_run: bool = False) -> dict:
    """SIGTERM the run's processes but its runner; the record of what was sent."""
    pids = [p for p in members(o["dir"], proc) if p != o["runner_pid"]]
    groups = {_pgid(p, proc) for p in pids}
    sent = []
    for p in pids:
        if dry_run:
            sent.append(p)
            continue
        try:
            kill(p, signal.SIGTERM)
            sent.append(p)
        except ProcessLookupError:
            pass
    return {"at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "run": o["run"], "command": o["command"][:12],
            "pgid": o["pgid"], "workload_group_seen": o["pgid"] in groups, "sigterm": sent, "dry_run": dry_run,
            "why": "not through a check slot (slot.py)"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="slot_only.py", description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--marker", type=Path, default=MARKER)
    ap.add_argument("--log", type=Path, default=LOG)
    ap.add_argument("--dry-run", action="store_true", help="print what it would stop, signal nothing")
    a = ap.parse_args(argv)
    if not a.marker.exists():
        return 0
    for o in offenders(a.runs):
        rec = stop(o, dry_run=a.dry_run)
        if not rec["sigterm"]:
            continue
        line = json.dumps(rec, sort_keys=True)
        print(line, flush=True)
        if not a.dry_run:
            a.log.parent.mkdir(parents=True, exist_ok=True)
            with a.log.open("a") as f:
                f.write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
