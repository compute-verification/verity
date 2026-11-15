"""dc/probes/sm90_jbreak.inc, the probe list `dc/run.sh sm90jb` builds, is dc/gen_jbreak.py's output. The probes
themselves need an H200."""
from __future__ import annotations

from pathlib import Path

import gen_jbreak

DC = Path(__file__).resolve().parents[1] / "dc"


def test_probe_list_is_current():
    inc = DC / "probes" / "sm90_jbreak.inc"
    assert inc.read_text() == gen_jbreak.render(), "run python benchmarks/pouw/dc/gen_jbreak.py"
