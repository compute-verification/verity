"""One stage of the Match driver (`verity_vllm.pipeline.match`) run as a command.  `run_stage` starts the command, tees its
stdout and stderr to the stage's log, kills it when it passes the memory or wall limit the driver gives, and marks the stage on
the row's timeline.  The record it returns is the stage's entry in the run directory's `stages.json`."""

from __future__ import annotations

import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from verity_vllm import settings
from verity_vllm.pipeline.timeline import Timeline

__all__ = ["run_stage"]


def _rss_gb(pid: int) -> float | None:
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=5).stdout.strip()
        return int(out) / (1024 ** 2) if out else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def run_stage(stage: str, cmd: list[str], log_path: Path, *, max_rss_gb: float | None, timeout_s: float | None, cwd: Path | None = None) -> dict:
    """Run one command, teeing stdout+stderr to `log_path`; watch RSS / wall when limits are given."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rec: dict[str, Any] = {"stage": stage, "cmd": shlex.join(cmd), "log": str(log_path), "status": "ok", "returncode": None, "reason": None}
    peak = 0.0
    # the Match phase's sub-steps on the row timeline (VERITY_TIMELINE, else no-op): capture / control = one engine
    # process each (`work.*`), the CPU phases (fold, resolve, ...) are validation
    try:
        _tl = Timeline.from_env("run_config", hot=False)
    except OSError:   # the telemetry file's directory can't be made: the stage runs without it
        _tl = None
    _tl_stage = f"match.{stage}" if stage not in ("capture", "control") else f"work.match_{stage}"
    if _tl is not None:
        _tl.start(_tl_stage)
    with open(log_path, "w") as lf:
        lf.write(f"$ {shlex.join(cmd)}\n")
        lf.flush()
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(cwd) if cwd else None, env=dict(settings.environ(), PYTHONUNBUFFERED="1"))
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            if max_rss_gb is not None:
                rss = _rss_gb(proc.pid)
                if rss is not None:
                    peak = max(peak, rss)
                    if rss > max_rss_gb:
                        proc.kill()
                        rec.update(status="killed_memory", reason=f"RSS {rss:.2f} GB > {max_rss_gb} GB")
                        proc.wait()
                        break
            if timeout_s is not None and time.time() - t0 > timeout_s:
                proc.kill()
                rec.update(status="timeout", reason=f"wall > {timeout_s:.0f} s")
                proc.wait()
                break
            time.sleep(0.25)
        rc = proc.returncode
    rec["returncode"] = rc
    rec["seconds"] = round(time.time() - t0, 3)
    if max_rss_gb is not None:
        rec["peak_rss_gb"] = round(peak, 3)
    if rec["status"] == "ok" and rc != 0:
        rec.update(status="failed", reason=f"exit {rc}; see {log_path}")
    if _tl is not None:
        _tl.end(_tl_stage, rc=rc, status=rec["status"])
    return rec
