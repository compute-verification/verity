"""fp4_tile_gpu.py's port of Pearl-C4's tile check against #580's reference, on a small gate: the pinned NVFP4 vectors, two
random tiles of three families (one census family and two planted ones that reach R1 and the cap), and long chains word
by word at k = 3584.  Needs #580's tree in PEARL_C4_RULE_TREE and g++; the gate runs in a subprocess, so the tree's
verity is the only one it imports.  The full gate is `fp4_tile_gpu.py gate` (1,000 tiles per family)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "fp4_tile_gpu.py"
TREE = os.environ.get("PEARL_C4_RULE_TREE", "")

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not (TREE and shutil.which("g++")), reason="needs PEARL_C4_RULE_TREE (#580's tree) and g++")]


def test_the_port_matches_the_reference_on_a_small_gate(tmp_path):
    subprocess.run([sys.executable, str(SCRIPT), "gate", "--rule-tree", TREE, "--out", str(tmp_path),
                    "--families", "sparse-2of4,planted-screened,planted-dead", "--tiles-per-family", "2",
                    "--big-per-family", "0", "--chunk", "2", "--workers", "2", "--no-batches"],
                   check=True, env=os.environ | {"FP4_REPLAY_CACHE": str(tmp_path / "lib")})
    s = json.loads((tmp_path / "gate/summary.json").read_text())
    assert [c["case"] for c in s["pinned"]] == ["test-16-nvfp4", "prod-64-nvfp4"]
    assert s["pinned_mismatches"] == 0
    assert s["tiles"] == 6 and s["tile_mismatches"] == 0
    assert s["words"] == 3 * (16 + 4) and s["word_mismatches"] == 0
