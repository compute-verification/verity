"""The package replays with this directory's `replay.py` even when another `replay` comes first on the path, as
network_traces' does under ``demo_endpoint.py --engine e2e_h200.engine:frozen``; where the path's `replay` is this one, the
package's is that same module."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import replay
from e2e_h200 import plan, units

HERE = Path(__file__).resolve().parents[1]


def test_the_package_reads_the_paths_replay_when_it_is_this_directorys():
    assert plan.RP is units.RP is replay


def test_a_shadowing_replay_first_on_the_path_is_left_to_its_importers(tmp_path):
    (tmp_path / "replay.py").write_text("SHADOW = True\n")
    code = ("import replay, e2e_h200.plan as P, e2e_h200.units as U; "
            "print(replay.SHADOW, P.RP is U.RP, U.RP.__file__, hasattr(U.RP, 'key_of'))")
    path = os.pathsep.join([str(tmp_path), str(HERE)])
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=dict(os.environ, PYTHONPATH=path),
                         capture_output=True, text=True, check=True).stdout.split()
    assert out[:2] == ["True", "True"] and Path(out[2]).resolve() == HERE / "replay.py" and out[3] == "True"
