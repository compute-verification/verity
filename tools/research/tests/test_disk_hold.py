"""`research disk-hold`: a queued run's disk admission on the machine, ahead of its allocation (research.disk)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from research import disk, remote

SRC = str(Path(remote.__file__).resolve().parents[1])
TB = 1000 * disk.GB


def _disk(used_tb: list[float]):
    """A 10 TB disk whose use is read from the front of `used_tb` at each poll."""
    def usage(path):
        return int(10 * TB), int(used_tb.pop(0) * TB) if len(used_tb) > 1 else int(used_tb[0] * TB), 0
    return usage


def test_it_admits_at_once_up_to_80_percent_and_holds_past_it_until_75():
    sleeps, lines = [], []
    assert disk.hold(1000, Path("/x"), usage=_disk([6.9]), sleep=sleeps.append, log=lines.append) == 0
    assert sleeps == [] and lines == []
    held = disk.hold(1000, Path("/x"), usage=_disk([7.5, 7.0, 6.6, 6.4]), sleep=sleeps.append, log=lines.append)
    assert held == 3 and sleeps == [disk.POLL_S] * 3, "8.0 TB is over 80%; 7.6 is not yet 75%; 7.4 is"
    assert lines[0].startswith("research disk-hold: holding: 1000 GB more would fill /x's filesystem to 85.0%")
    assert lines[-1].startswith("research disk-hold: released after 3 poll(s): 74.0%")


def test_the_command_runs_in_its_place_once_admitted(tmp_path):
    events = []
    rc = disk.main(["--gb", "1000", "--path", str(tmp_path), "--", "echo", "ran"], usage=_disk([7.5, 7.0, 6.6, 6.4]),
                   sleep=lambda s: events.append("poll"), execvp=lambda f, a: events.append(("exec", f, a)))
    assert events == ["poll"] * 3 + [("exec", "echo", ["echo", "ran"])], "held for 3 polls, then exec'd in place"
    assert rc == 127, "execvp returns only when it fails"
    assert subprocess.run([sys.executable, "-m", "research", "disk-hold", "--gb", "0", "--path", str(tmp_path)],
                          capture_output=True, text=True, env={"PYTHONPATH": SRC}).returncode == 2
