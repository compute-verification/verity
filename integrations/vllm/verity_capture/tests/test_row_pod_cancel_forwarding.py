"""[r19-int-19, F-r19-int-6] a cancelled row_pod.sh takes its running stage (and the stage's forked workers) with it.

The research runner cancels a row with killpg(pgid, SIGTERM) on the shell's session.  GNU `timeout` moves itself and the stage into
a new process group so that its own expiry can kill the stage's workers; that group never saw the runner's killpg, so the shell died and
the stage lived on with ppid 1 under timeout's own clock (#23's record GM-01: 54 min / 46 GB past its cancelled run; #11 the same).
row_pod.sh now runs the stage in the background under `wait`, traps TERM/INT/HUP, forwards the signal to timeout's pid (timeout forwards
to the stage and to its whole group), waits for the stage to be gone, records the cancel and exits 128+signal.

These tests run the real `run_to` / trap block extracted from row_pod.sh (between the `row_pod.signal-forwarding` markers) in a harness
with stub `log` / `tl`, under a stage that forks a grandchild, and require GNU timeout (Linux pods; skipped where absent).
"""
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

TREE = Path(__file__).resolve().parents[2]
ROW_POD = TREE / "verity_capture/sweep/row_pod.sh"

_GNU_TIMEOUT = shutil.which("timeout") is not None and b"GNU" in subprocess.run(["timeout", "--version"], capture_output=True).stdout
pytestmark = pytest.mark.skipif(not _GNU_TIMEOUT, reason="GNU timeout (coreutils) required -- the pods have it, macOS does not")


def _block() -> str:
    src = ROW_POD.read_text()
    m = re.search(r"# --- row_pod\.signal-forwarding begin.*?\n(.*?)\n# --- row_pod\.signal-forwarding end", src, re.DOTALL)
    assert m, "signal-forwarding block not found in row_pod.sh"
    return m.group(1)


# the stage: a python parent that forks a grandchild; both write their pids and sleep.  A row's fold / commit_delta look like this.
STAGE = r'''
import os, sys, time
d = sys.argv[1]
open(os.path.join(d, "stage.pid"), "w").write(str(os.getpid()))
pid = os.fork()
if pid == 0:
    open(os.path.join(d, "grandchild.pid"), "w").write(str(os.getpid()))
    time.sleep(600)
    sys.exit(0)
time.sleep(600)
'''


def _harness(tmp: Path, timeout_s: int, stage_exit: str | None = None) -> str:
    stage = (tmp / "stage.py")
    stage.write_text(STAGE if stage_exit is None else f"import sys; sys.exit({stage_exit})")
    return f"""#!/bin/bash
set -uo pipefail
D={tmp}
log() {{ echo "$(date -u +%FT%TZ) $*" >> "$D/row.log"; }}
tl() {{ echo "$*" >> "$D/timeline.txt"; }}
{_block()}
run_to {timeout_s} "$D/stage.log" python3 "{stage}" "$D"; rc=$?
echo "run_to rc=$rc" >> "$D/row.log"
exit $rc
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # a zombie is not alive for our purpose
    try:
        state = open(f"/proc/{pid}/stat").read().split(")")[-1].split()[0]
        return state != "Z"
    except OSError:
        return True


def _wait_for(pred, seconds: float) -> bool:
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        if pred():
            return True
        time.sleep(0.05)
    return pred()


def _start(tmp: Path, script: str) -> subprocess.Popen:
    h = tmp / "harness.sh"
    h.write_text(script)
    # the research runner starts the row as a session leader and later killpg()s that group -- same here
    p = subprocess.Popen(["bash", str(h)], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert _wait_for(lambda: (tmp / "grandchild.pid").exists() or (tmp / "row.log").exists() and "run_to rc" in (tmp / "row.log").read_text(), 10)
    return p


def test_killpg_on_the_row_takes_the_stage_and_its_grandchild(tmp_path):
    p = _start(tmp_path, _harness(tmp_path, 600))
    stage = int((tmp_path / "stage.pid").read_text()); grand = int((tmp_path / "grandchild.pid").read_text())
    assert _alive(stage) and _alive(grand)
    assert os.getpgid(stage) != p.pid, "GNU timeout puts the stage in its own process group -- the premise of the finding"
    os.killpg(p.pid, signal.SIGTERM)                     # what tele run does on cancel / timeout
    rc = p.wait(timeout=15)
    assert rc == 143, f"row exits 128+SIGTERM, got {rc}"
    assert _wait_for(lambda: not _alive(stage) and not _alive(grand), 10), "stage or its forked worker survived the cancel (the orphan class)"
    assert "CANCELLED by SIGTERM" in (tmp_path / "stage.log").read_text()
    row_log = (tmp_path / "row.log").read_text()
    assert "SIGTERM received: forwarding to the running stage" in row_log and "row cancelled by SIGTERM" in row_log
    tl = (tmp_path / "timeline.txt").read_text()
    assert "row cancelled signal=TERM" in tl and "row end ok=false stopped_at=cancel" in tl


def test_sigint_is_forwarded_the_same_way(tmp_path):
    p = _start(tmp_path, _harness(tmp_path, 600))
    stage = int((tmp_path / "stage.pid").read_text()); grand = int((tmp_path / "grandchild.pid").read_text())
    os.killpg(p.pid, signal.SIGINT)
    assert p.wait(timeout=15) == 130
    assert _wait_for(lambda: not _alive(stage) and not _alive(grand), 10)
    assert "CANCELLED by SIGINT" in (tmp_path / "stage.log").read_text()


def test_the_expiry_path_still_kills_the_forked_worker(tmp_path):
    """--foreground would have broken this: timeout's own expiry must still reach the stage's group (fold workers, commit legs)."""
    p = _start(tmp_path, _harness(tmp_path, 1))
    stage = int((tmp_path / "stage.pid").read_text()); grand = int((tmp_path / "grandchild.pid").read_text())
    rc = p.wait(timeout=15)
    assert rc == 124, f"timeout's rc is the stage's rc, got {rc}"
    assert _wait_for(lambda: not _alive(stage) and not _alive(grand), 10), "timeout expiry left the forked worker alive"
    assert "TIMEOUT after 1s" in (tmp_path / "stage.log").read_text()
    assert "CANCELLED" not in (tmp_path / "stage.log").read_text()      # an expiry is not a cancel


def test_a_normal_exit_returns_the_stage_rc_unchanged(tmp_path):
    p = _start(tmp_path, _harness(tmp_path, 600, stage_exit="7"))
    assert p.wait(timeout=15) == 7
    row_log = (tmp_path / "row.log").read_text()
    assert "run_to rc=7" in row_log and "cancelled" not in row_log
    assert not (tmp_path / "timeline.txt").exists()
