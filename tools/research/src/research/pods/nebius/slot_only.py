#!/usr/bin/env python3
"""`python3 slot_only.py [--runs DIR] [--marker FILE] [--log FILE] [--dry-run] [--renice N | --auditor]`: on a machine whose MARKER
exists (deploy.toml installs it on vy-nebius-cpu-1), stop every live `research run` whose workload did not come through a check slot
(`through_slot`).

With `--renice N` (vy-nebius-1's vy-direct-nice.timer, marker /etc/vy/direct-runs-nice) it stops nothing: it renices each such
run's processes to N, so the train checks' slots win the cores (top, 18 Oct: node 1 at load 410-468 on 192 cores, most of it lanes'
direct runs), and logs each run once, when it is first reniced.

With `--auditor` (an auditor host's vy-auditor-only.timer, marker AUDITOR), it stops every live run whose command is not one of the
marker's lines (`auditor_allows`): the host runs the auditor's own service and nothing else (AGENTS.md's auditor-trusted devices;
vy-challenger-1's marker lists no command, since its challenger runs as `auditor` through benchmarks/live_challenger/auditor.py).

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
import shlex
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

MARKER = Path("/etc/vy/runs-via-slot")
#: an auditor host's marker: each line but a comment is a command a `research run` there may be, as shlex splits it
AUDITOR = Path("/etc/vy/auditor")
RUNS = Path("/workspace/research/runs")
LOG = Path("/workspace/monitoring/slot-only.jsonl")
#: slot.py from a shipped tree
SLOT_PATHS = ("tools/verity/check/slot.py",)
#: `env CHECK_SLOTS=...` before slot.py: a pool of its own, with its own line, in one of these files (vy-nebius-cpu-1's
#: `proofs` pool, 13 Oct), so a lane's runs never queue behind the merge checks
POOL = re.compile(r"CHECK_SLOTS=/workspace/research/locks/[A-Za-z0-9_-]+/slots")
#: `research run --pool NAME`'s pools: `research` is any lane's research runs, where `research run --on auto` spills (INF-27,
#: cpu-1's /workspace/research/locks/research/slots: 8-53 since 19 Oct, cores no check slot holds), `proofs` is proofs' Lean pool,
#: `network` is network-accounting's proving pool (cpu-1, cpu-2, cpu-4, cpu-5 and node 1 hold one), `soundness` is soundness's
#: Lean pool (cpu-4, s0-s11, since 26 Oct)
POOLS = {"research": "/workspace/research/locks/research/slots", "proofs": "/workspace/research/locks/proofs/slots",
         "network": "/workspace/research/locks/network/slots", "soundness": "/workspace/research/locks/soundness/slots"}
#: how long a timed `research run --pool` waits in its pool's line for a slot by default (slot.py --wait-max; `--pool-wait`): an
#: hour lost proofs' runs their place in a saturated pool each hour (INF-153)
POOL_WAIT_S = 14400


def in_pool(pool: str, lane: str, command: list[str], run_max: float | None = None, wait_s: float = POOL_WAIT_S) -> list[str]:
    """CMD as a run in `pool`, the one way a lane's run takes it: `env CHECK_SLOTS=POOL python3 tools/verity/check/slot.py --by
    LANE [--wait-max POOL_WAIT_S --run-max S] -- nice -n 19 CMD...`, with slot.py from the run's own shipped tree (so the run's
    cwd is its source).  `run_max` is the run's --timeout: slot.py counts it from taking the slot, so the wait in the pool's line
    isn't spent from it (INF-106), and a run with no slot in `wait_s` exits NOT_STARTED (75) without running CMD.  The
    runner's own timeout must then cover both (`research run --pool` makes it wait_s + S)."""
    if pool not in POOLS:
        raise ValueError(f"no pool {pool!r}: the pools are {', '.join(sorted(POOLS))}")
    limit = ["--wait-max", f"{wait_s:g}", "--run-max", f"{run_max:g}"] if run_max else []
    return ["env", f"CHECK_SLOTS={POOLS[pool]}", "python3", SLOT_PATHS[0], "--by", lane, *limit, "--", "nice", "-n", "19", *command]


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


def refusal(machine_marker: Path = MARKER, pools: dict[str, str] | None = None) -> str:
    """Why the launcher refuses a run outside slot.py here, naming the `research run --pool` pools this machine has (27 Oct: lanes
    sent `verity lean try --on vy-nebius-1` without `--pool`, and one wrapped slot.py by hand)."""
    have = sorted(p for p, f in (POOLS if pools is None else pools).items() if Path(f).is_file())
    send = (f"add `--pool {'|'.join(have)}` to `research run` or `verity lean try`, which sends CMD through that pool's slots"
            if have else "send it as `python3 tools/verity/check/slot.py [--by NAME] -- CMD...`, which `check.py --record --on` does")
    return f"this machine takes a `research run` only through its check slots ({machine_marker}): {send}"


def auditor_commands(marker: Path = AUDITOR) -> list[list[str]]:
    """The commands an auditor host runs, one a line of its marker; none when the marker can't be read, so nothing runs."""
    try:
        lines = marker.read_text().splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    return [shlex.split(ln) for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def auditor_allows(command: object, marker: Path = AUDITOR) -> bool:
    """Whether a workload command is exactly one of the auditor host's commands: no wrapper, pool or extra argument."""
    return isinstance(command, list) and command in auditor_commands(marker)


def auditor_refusal(marker: Path = AUDITOR) -> str:
    allowed = " or ".join(f"`{shlex.join(c)}`" for c in auditor_commands(marker)) or "nothing (its list is empty or unreadable)"
    return (f"this machine is an auditor host ({marker}): it runs only its own service, so a `research run` here is {allowed}; "
            f"take other work to another machine")


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


def offenders(runs: Path, allowed=through_slot) -> list[dict]:
    """Each live run under ``runs`` (status.json `running`) whose job.json command is not `allowed`."""
    out = []
    for st_path in sorted(runs.glob("*/status.json")):
        d = st_path.parent
        st = _json(st_path)
        if st.get("state") != "running":
            continue
        command = (_json(d / "job.json").get("execution") or {}).get("command")
        if command is None or allowed(command):
            continue
        out.append({"run": d.name, "dir": d, "pgid": st.get("pgid"), "command": command,
                    "runner_pid": _json(d / "launch.json").get("runner_pid")})
    return out


def stop(o: dict, *, proc: Path = Path("/proc"), kill=os.kill, dry_run: bool = False,
         why: str = "not through a check slot (slot.py)") -> dict:
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
            "pgid": o["pgid"], "workload_group_seen": o["pgid"] in groups, "sigterm": sent, "dry_run": dry_run, "why": why}


def renice(o: dict, nice: int, *, proc: Path = Path("/proc"), dry_run: bool = False) -> dict:
    """Renice every process of the run to `nice` (its runner too); the record of the pids it moved."""
    moved = []
    for p in members(o["dir"], proc):
        try:
            if os.getpriority(os.PRIO_PROCESS, p) < nice:
                if not dry_run:
                    os.setpriority(os.PRIO_PROCESS, p, nice)
                moved.append(p)
        except (ProcessLookupError, PermissionError):
            pass
    return {"at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "run": o["run"], "command": o["command"][:12],
            "reniced": moved, "nice": nice, "dry_run": dry_run, "why": "a direct run, not through a check slot (slot.py)"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="slot_only.py", description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--marker", type=Path, default=MARKER)
    ap.add_argument("--log", type=Path, default=LOG)
    ap.add_argument("--dry-run", action="store_true", help="print what it would stop, signal nothing")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--renice", type=int, default=None, metavar="N", help="renice such runs to N instead of stopping them")
    mode.add_argument("--auditor", action="store_true", help="stop every run whose command is not one of the marker's lines")
    a = ap.parse_args(argv)
    if not a.marker.exists():
        return 0
    allowed = (lambda c: auditor_allows(c, a.marker)) if a.auditor else through_slot
    for o in offenders(a.runs, allowed):
        if a.renice is not None:
            rec = renice(o, a.renice, dry_run=a.dry_run)
        elif a.auditor:
            rec = stop(o, dry_run=a.dry_run, why=f"not a command of the auditor host's {a.marker}")
        else:
            rec = stop(o, dry_run=a.dry_run)
        if not rec.get("sigterm" if a.renice is None else "reniced"):
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
