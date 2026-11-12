"""dc/probes/sm90_jbreak.inc, the probe list `dc/run.sh sm90jb` builds, is dc/gen_jbreak.py's output. The probes
themselves need an H200."""
from __future__ import annotations

import importlib.util
from pathlib import Path

DC = Path(__file__).resolve().parents[1] / "dc"


def test_probe_list_is_current():
    spec = importlib.util.spec_from_file_location("dc_gen_jbreak_under_test", DC / "gen_jbreak.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    inc = DC / "probes" / "sm90_jbreak.inc"
    assert inc.read_text() == gen.render(), "run python benchmarks/pouw/dc/gen_jbreak.py"
