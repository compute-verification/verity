"""Cancellation markers: intent before the signal, result after (brief §6 "intentional killers write evidence").

    tele cancel-intent --actor deadman|timeout|guardian|manual (--target-pid P | --target-pgid G | --target-pod ID) --signal 9 --reason "..."
                       [--attempt-dir D | --fallback FILE] [--expect-starttime TICKS] [--expect-comm NAME] [--expect-nonempty]
                       [--expect-run-dir D] [--deliver] [--job J --attempt N --project P]

The target's identity is verified from /proc (starttime ticks + comm) and recorded; with --deliver, a pid whose starttime does not match
--expect-starttime is NOT signalled (a recycled pid is not the job), nor a group that fails --expect-nonempty or --expect-run-dir (no member
whose environment names RESEARCH_RUN_DIR=D, which the runner exports to its workload: a pgid a stale status.json names may be another job's
group by now).  Logging never blocks the signal: a marker write failure is reported on stderr and the signal is still delivered; the fallback
file is the safety controller's own scoped record when no attempt dir is reachable.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

from . import CANCEL_ACTORS
from .events import append, events_path, utc_now_ms
from .procs import stat_fields

#: the group members a marker names: a whole group's list can pass the record cap (events.MAX_RECORD_BYTES), and the writer then drops
#: `target`, pgid included, so the classifier can't match the marker to the job it stopped
MEMBERS_KEPT = 16
DEFAULT_FALLBACK = os.environ.get("TELEMETRY_CANCEL_FALLBACK") or (
    "/root/dm/cancel.jsonl" if os.path.isdir("/root/dm") else str(Path.home() / ".veritor" / "telemetry" / "cancel.jsonl"))


def identify(pid: int) -> dict[str, Any]:
    st = stat_fields(pid)
    if st is None:
        return {"pid": pid, "exists": False}
    return {"pid": pid, "exists": True, "comm": st["comm"], "ppid": st["ppid"], "pgrp": st["pgrp"], "starttime_ticks": st["starttime_ticks"], "state": st["state"]}


def pgid_members(pgid: int) -> list[dict[str, Any]]:
    out = []
    try:
        for x in os.listdir("/proc"):
            if x.isdigit():
                st = stat_fields(int(x))
                if st and st["pgrp"] == pgid:
                    out.append({"pid": st["pid"], "comm": st["comm"], "starttime_ticks": st["starttime_ticks"]})
    except OSError:
        pass
    return out[:256]


def run_dir_of(pid: int) -> str | None:
    """RESEARCH_RUN_DIR in the environment `pid` was exec'd with, or None (absent, gone, or not ours to read)."""
    try:
        env = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return None
    for kv in env.split(b"\0"):
        if kv.startswith(b"RESEARCH_RUN_DIR="):
            return kv.split(b"=", 1)[1].decode(errors="replace")
    return None


def write_marker(path: Path, rec: dict[str, Any]) -> tuple[bool, str | None]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        append(path, rec)
        return True, None
    except OSError as e:
        return False, f"{e.__class__.__name__}: {e}"


def cancel(a: argparse.Namespace) -> int:
    if a.actor not in CANCEL_ACTORS:
        print(f"actor must be one of {CANCEL_ACTORS}", file=sys.stderr)
        return 2
    path = events_path(a.attempt_dir) if a.attempt_dir else Path(a.fallback or DEFAULT_FALLBACK)
    ident: dict[str, Any] = {}
    verified: bool | None = None
    if a.target_pid is not None:
        ident = identify(a.target_pid)
        if a.expect_starttime is not None:
            verified = ident.get("exists", False) and ident.get("starttime_ticks") == a.expect_starttime
        elif a.expect_comm is not None:
            verified = ident.get("exists", False) and ident.get("comm") == a.expect_comm
        else:
            verified = None   # no expectation supplied: identity recorded, not verified
    elif a.target_pgid is not None:
        members = pgid_members(a.target_pgid)
        ident = {"pgid": a.target_pgid, "members": members[:MEMBERS_KEPT], "members_total": len(members)}
        if a.expect_run_dir is not None:
            want = str(Path(a.expect_run_dir).resolve())
            ours = [m["pid"] for m in members if run_dir_of(m["pid"]) == want]
            ident["run_dir_members"] = ours[:MEMBERS_KEPT]
            verified = bool(ours)
        else:
            verified = bool(members) if a.expect_nonempty else None
    elif a.target_pod:
        ident = {"pod": a.target_pod}
    intent = {"kind": "cancel_intent", "actor": a.actor, "reason": a.reason, "signal": a.signal, "target": ident, "verified": verified,
              "project": a.project, "job": a.job, "attempt": a.attempt, "requested_utc": utc_now_ms(), "writer_pid": os.getpid(),
              "controller": a.controller, "fallback_record": a.attempt_dir is None}
    ok, why = write_marker(path, intent)
    if not ok:
        print(f"[tele cancel-intent] marker not written ({why}); proceeding", file=sys.stderr)
    if not a.deliver:
        print(f"intent recorded ({path}); no signal delivered (--deliver absent)", file=sys.stderr)
        return 0
    result: dict[str, Any] = {"kind": "cancel_result", "actor": a.actor, "signal": a.signal, "target": ident, "job": a.job, "attempt": a.attempt}
    if a.target_pod:
        result.update(delivered=None, reason="pod termination is performed by the caller (runpod API); result recorded by the caller")
        write_marker(path, result)
        return 0
    if verified is False:
        result.update(delivered=False, reason="refused: target identity does not match the expectation (recycled pid?)" if a.target_pid is not None
                      else "refused: no member of the group is the expected workload (a stale pgid, or one another job holds now)")
        write_marker(path, result)
        print("[tele cancel-intent] REFUSED: identity mismatch", file=sys.stderr)
        return 3
    try:
        if a.target_pgid is not None:
            os.killpg(a.target_pgid, a.signal)
        else:
            os.kill(a.target_pid, a.signal)
        result.update(delivered=True, errno=0, delivered_utc=utc_now_ms())
        rc = 0
    except ProcessLookupError as e:
        result.update(delivered=False, errno=e.errno, reason="no such process/group")
        rc = 0
    except PermissionError as e:
        result.update(delivered=False, errno=e.errno, reason="permission denied")
        rc = 4
    except OSError as e:
        result.update(delivered=False, errno=e.errno, reason=e.strerror)
        rc = 4
    if a.confirm_exit_s and a.target_pid is not None and result.get("delivered"):
        t_end = time.monotonic() + a.confirm_exit_s
        gone = False
        while time.monotonic() < t_end:
            st = stat_fields(a.target_pid)
            if st is None or st["starttime_ticks"] != ident.get("starttime_ticks") or st["state"] == "Z":
                gone = True
                break
            time.sleep(0.05)
        result["target_gone_within_s"] = a.confirm_exit_s if gone else None
        result["target_still_present"] = not gone
    write_marker(path, result)
    return rc


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--actor", required=True, choices=CANCEL_ACTORS)
    ap.add_argument("--reason", required=True)
    ap.add_argument("--signal", type=int, default=signal.SIGTERM)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--target-pid", type=int)
    g.add_argument("--target-pgid", type=int)
    g.add_argument("--target-pod")
    ap.add_argument("--attempt-dir", default=None)
    ap.add_argument("--fallback", default=None, help=f"marker file when no attempt dir is reachable (default {DEFAULT_FALLBACK})")
    ap.add_argument("--expect-starttime", type=int, default=None, help="/proc/<pid>/stat starttime ticks the caller observed when it chose the target")
    ap.add_argument("--expect-comm", default=None)
    ap.add_argument("--expect-nonempty", action="store_true", help="for --target-pgid: verified only if the group has members")
    ap.add_argument("--expect-run-dir", default=None, metavar="D",
                    help="for --target-pgid: verified only if a member's environment has RESEARCH_RUN_DIR=D (resolved), the run's workload")
    ap.add_argument("--deliver", action="store_true", help="send the signal after writing the intent, then record the result")
    ap.add_argument("--confirm-exit-s", type=float, default=0.0, help="after delivery, watch the target for this long and record whether it went")
    ap.add_argument("--project", default=None)
    ap.add_argument("--job", default=None)
    ap.add_argument("--attempt", type=int, default=None)
    ap.add_argument("--controller", default=None, help="name/version of the calling safety controller")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tele cancel-intent", description=__doc__.splitlines()[0])
    add_args(ap)
    return cancel(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
