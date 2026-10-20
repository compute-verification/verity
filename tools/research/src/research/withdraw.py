"""`research cancel` of a run its machine still holds ahead of its runner (`research disk-hold`, `research cpu-hold`): no runner, no
status.json, so no workload's group to signal.  The hold is the launch's own process (launch.json `runner_pid`), which execs the
runner in place once it releases, so the same pid is the run's runner a moment later and a cancel must never signal that.

`remote.cancel` sends this file's source to the machine (`python3 - RUN_DIR REQUEST_DIR REASON`, with the run's shipped tool on
PYTHONPATH), so it needs nothing from the machine's tool but the launch's release.  It ends the hold only while it is the run's:
launch.json's argv is `PY -m research disk-hold|cpu-hold ... --ack <REQUEST_DIR>/.launch-ack -- ...` and /proc/<pid>/cmdline is
still that argv.  The process is stopped (SIGSTOP) before the second look, so it can't exec the runner between the look and the
SIGTERM.  Then refused.json records `{fault: Withdrawn, reason}`, which `research status` reads as refused, and what the launch took
(memory reservation, exclusive lock, custody key) goes, as for a launch the machine refused.  Prints one JSON object."""
from __future__ import annotations

import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

HOLDS = ("disk-hold", "cpu-hold")
FAULT = "Withdrawn"
LAUNCH_ACK = ".launch-ack"
POLL_S, POLLS = 0.01, 500


class NotHeld(Exception):
    pass


def hold_of(argv: list[str], ack: str) -> str | None:
    """The hold's name when `argv` is a hold acknowledging the launch FIFO `ack`, else None."""
    if len(argv) < 4 or argv[1:3] != ["-m", "research"] or argv[3] not in HOLDS:
        return None
    head = argv[:argv.index("--")] if "--" in argv else argv
    return argv[3] if any(head[i] == "--ack" and head[i + 1] == ack for i in range(4, len(head) - 1)) else None


def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _cmdline(proc: Path, pid: int) -> list[str] | None:
    try:
        raw = (proc / str(pid) / "cmdline").read_bytes()
    except OSError:
        return None
    return raw.decode(errors="replace").split("\0")[:-1] if raw else None


def _state(proc: Path, pid: int) -> str | None:
    """/proc/<pid>/stat's state letter (R, S, T, Z, ...), None once the pid is gone."""
    try:
        return (proc / str(pid) / "stat").read_text().rpartition(")")[2].split()[0]
    except (OSError, IndexError):
        return None


def _until(pred: Callable[[], bool], sleep: Callable[[float], None]) -> bool:
    for _ in range(POLLS):
        if pred():
            return True
        sleep(POLL_S)
    return pred()


def _release(run_dir: Path, request_dir: Path, argv: list[str]) -> str:
    """What the launch took goes, through the shipped tool's own release (remote._release_launch); what went, or why not."""
    req = _read_json(request_dir / "request.json")
    if not isinstance(req, dict):
        return f"not released: no request.json in {request_dir}"
    runner = argv[argv.index("--") + 1:] if "--" in argv else []
    remote = next((json.loads(runner[i + 1]) for i in range(len(runner) - 1) if runner[i] == "--remote-json"), {}) if runner else {}
    try:
        from research.remote import _release_launch  # noqa: PLC0415  (the run's shipped tool, on PYTHONPATH)

        _release_launch(req, request_dir / "request.json", remote.get("mem"))
    except Exception as e:  # noqa: BLE001  (the hold is gone and refused.json written: say what stayed)
        return f"not released: {type(e).__name__}: {e}"
    return "released" + (f" (memory reservation {remote['mem'].get('reservation')})" if remote.get("mem") else "")


def withdraw(run_dir: Path, request_dir: Path, reason: str, *, proc: Path = Path("/proc"), kill: Callable[[int, int], None] = os.kill,
             sleep: Callable[[float], None] = time.sleep, release: Callable[[Path, Path, list[str]], str] = _release) -> dict[str, Any]:
    """End the run's hold and record it withdrawn; NotHeld (nothing signalled, nothing written) when it isn't held."""
    name = run_dir.name
    if (run_dir / "status.json").exists():
        raise NotHeld(f"{name}'s runner has started (status.json): cancel it as a running run")
    if (refused := _read_json(run_dir / "refused.json")) is not None:
        raise NotHeld(f"{name} was refused already ({(refused or {}).get('fault')}: {(refused or {}).get('reason')})")
    launch = _read_json(run_dir / "launch.json") or {}
    pid, argv, ack = launch.get("runner_pid"), launch.get("argv") or [], str(request_dir / LAUNCH_ACK)
    if not isinstance(pid, int) or pid <= 1:
        raise NotHeld(f"{name}'s launch.json names no process (runner_pid {pid!r})")
    if not (hold := hold_of(argv, ack)):
        raise NotHeld(f"{name}'s launch is not held: its argv is no `research disk-hold|cpu-hold --ack {ack}`")
    if _cmdline(proc, pid) != argv:
        raise NotHeld(f"pid {pid} is no longer {name}'s {hold} (it released, or exited): nothing was signalled")
    kill(pid, signal.SIGSTOP)
    try:
        stopped = _until(lambda: _state(proc, pid) in ("T", "t", "Z", None), sleep)
        if not stopped or _cmdline(proc, pid) != argv or (run_dir / "status.json").exists():
            raise NotHeld(f"{name}'s {hold} (pid {pid}) released as the cancel came: nothing was signalled; cancel it as a running run")
        kill(pid, signal.SIGTERM)
    finally:
        try:
            kill(pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
    if not _until(lambda: _state(proc, pid) in ("Z", None), sleep):
        raise NotHeld(f"{name}'s {hold} (pid {pid}) took SIGTERM and is still there; nothing was recorded")
    at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    rec = {"run_id": name, "reason": reason, "fault": FAULT, "at_utc": at, "withdrawn": {"hold": hold, "pid": pid}}
    tmp = run_dir / "refused.json.tmp"
    tmp.write_text(json.dumps(rec, indent=2) + "\n")
    os.replace(tmp, run_dir / "refused.json")
    released = release(run_dir, request_dir, argv)
    try:
        with open(run_dir / "launcher.log", "a") as log:
            log.write(f"research cancel: withdrawn from its {hold} (pid {pid}): {reason}; {released}\n")
    except OSError:
        pass
    return {"withdrawn": True, "pid": pid, "hold": hold, "released": released}


def main(argv: list[str]) -> int:
    run_dir, request_dir, reason = argv
    try:
        out = withdraw(Path(run_dir), Path(request_dir), reason)
    except NotHeld as e:
        out = {"withdrawn": False, "why": str(e)}
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
