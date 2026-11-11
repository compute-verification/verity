#!/usr/bin/env python3
"""Coordinators whose run has stopped mid-turn: a tool result is the last entry, with no step for --quiet-min minutes.

A coordinator that ended its turn and waits on background workers ends on its own message instead, and still takes follow-ups.

Usage: liveness.py RUN_DIR [--quiet-min 60] [--stale-min 120]
RUN_DIR is a batch-fetch-details directory fetched with includeTranscripts (index.json plus <bcId>/transcript.json).
Prints one line per held coordinator: `HELD <bcId> <name> <status> last step <UTC> (<minutes> min) last entry <role>`.
A held run takes no follow-ups (Slack, Linear, timers, Daniel's messages): Cursor starts no new run on the agent while
its wait is open. infra's agent watch frees one once a doorbell has rung with nothing run since (cancels the wait run
through the API, starts a new run), but misses one whose workers still stream progress; then ask infra to do the same
by hand (24 Oct: memory-accounting freed by the watch 13:26:55Z, network-accounting by infra), or Daniel presses Stop.
Also prints `STALE ...` for a run not IDLE that took no step for --stale-min minutes, whatever its last entry: a VM
suspended while WAITING_FOR_BACKGROUND_WORK looks alive by status and by its handle's Slack posts (compute-accounting's
research lane, 23 Oct 05:26-23:40Z), but its workers' results go unrelayed.
Prints `HUNG ...` for a RUNNING run with no step for --quiet-min minutes, whatever its last entry: Cursor's agent runtime
can stall mid-turn while status and activity stay fresh (memory-accounting, 26 Oct 01:47-04:39Z, TOP-90 row 37).
Exits 1 with `NOT CLEAN` on any flagged line, an unread transcript included: status and activity times passed that hang\ntwice.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    run = Path(sys.argv[1])
    quiet = float(sys.argv[sys.argv.index("--quiet-min") + 1]) if "--quiet-min" in sys.argv else 60
    stale = float(sys.argv[sys.argv.index("--stale-min") + 1]) if "--stale-min" in sys.argv else 120
    now = datetime.now(timezone.utc).timestamp() * 1000
    unknown, flagged = [], []

    def flag(line: str) -> None:
        flagged.append(line)
        print(line)

    for a in json.loads((run / "index.json").read_text())["agents"]:
        if a.get("isArchived"):
            continue
        if a["status"] in ("ERROR", "EXPIRED"):
            when = datetime.fromtimestamp(a.get("lastMessageActivityAtMs", 0) / 1000, timezone.utc).strftime("%m-%d %H:%MZ")
            flag(f"DOWN {a['bcId']} {a['name']} {a['status']} since about {when}")
            continue
        if a["status"] in ("IDLE", "FINISHED"):
            continue
        t = run / a["bcId"] / "transcript.json"
        if not t.exists():
            act = a.get("lastMessageActivityAtMs") or 0
            mins = (now - act) / 60000
            # Activity alone can't show HELD, and a follow-up queued to a suspended VM may move it, so a recent one stays UNKNOWN.
            kind = "STALE" if act and mins > stale else "UNKNOWN"
            flag(f"{kind} {a['bcId']} {a['name']} {a['status']} (no transcript; last activity {mins:.0f} min ago)")
            unknown.append(a["bcId"])
            continue
        msgs = json.loads(t.read_text())["messages"]
        if not msgs:                                    # an agent still in its first turn exports nothing
            flag(f"UNKNOWN {a['bcId']} {a['name']} {a['status']} (empty transcript: first turn not finished)")
            continue
        last = max((tc.get("completed_at_ms") or tc.get("started_at_ms") or 0
                    for m in msgs for tc in (m.get("tool_calls") or [])), default=0)
        mins = (now - last) / 60000
        act = a.get("lastMessageActivityAtMs") or 0
        # A resumed run's steps aren't exported until its turn ends: memory-accounting's four workers, resumed 4 Nov
        # 06:05Z, exported nothing past 3 Nov 19:11Z while they opened #7224. Their transcripts can't show HELD.
        if last and act - last > quiet * 60000 and a["status"] != "IDLE":
            when = datetime.fromtimestamp(act / 1000, timezone.utc).strftime("%m-%d %H:%MZ")
            flag(f"TURN-UNREAD {a['bcId']} {a['name']} {a['status']} (turn from {when} not in its transcript: judge by its"
                 f" commits, PRs and notes, or ask its launcher; never ring on this alone)")
            unknown.append(a["bcId"])
            continue
        if last and mins > quiet and msgs[-1].get("role") == "tool":     # a tool result with no next step: the run stopped mid-turn
            when = datetime.fromtimestamp(last / 1000, timezone.utc).strftime("%m-%d %H:%MZ")
            flag(f"HELD {a['bcId']} {a['name']} {a['status']} last step {when} ({mins:.0f} min) last entry {msgs[-1].get('role')}")
        elif last and mins > quiet and a["status"] == "RUNNING":
            when = datetime.fromtimestamp(last / 1000, timezone.utc).strftime("%m-%d %H:%MZ")
            flag(f"HUNG {a['bcId']} {a['name']} {a['status']} last step {when} ({mins:.0f} min) last entry {msgs[-1].get('role')}")
        elif last and mins > stale:
            when = datetime.fromtimestamp(last / 1000, timezone.utc).strftime("%m-%d %H:%MZ")
            flag(f"STALE {a['bcId']} {a['name']} {a['status']} last step {when} ({mins:.0f} min) last entry {msgs[-1].get('role')}")
    if unknown:
        # compute-accounting's research coordinator was held 06:52-11:45Z on 24 Oct while every hourly fetch missed its
        # transcript and its activity stamp stayed fresh: an UNKNOWN is not a pass until its own transcript is read.
        print(f"REFETCH {len(unknown)} (batch-fetch-details with includeTranscripts, a few at a time, then rerun): {' '.join(unknown)}")
    if flagged:
        print(f"NOT CLEAN: {len(flagged)} flagged ({len(unknown)} transcripts unread); liveness is clean only on exit 0")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
