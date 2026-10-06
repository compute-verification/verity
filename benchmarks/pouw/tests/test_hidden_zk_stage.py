"""hidden_zk's fast staging (stage.py under HZ_STAGE=fast: fast_stage.py's patches of C-Flock's staging) writes the bytes the
library writes: the circuit, its instances, public file and pin, and the same stage.json but for its timings and its label."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

STAGE = Path(__file__).resolve().parents[1] / "hidden_zk" / "stage.py"
OWN = ("seconds", "maxrss_gb", "staging")


@pytest.mark.slow
def test_fast_staging_writes_the_library_bytes(tmp_path):
    """About 20 s: the (1,1) tile at K=64 staged both ways, each in its own process, since fast_stage patches the library in
    place. Nothing cheaper catches a library change under one of those patches that makes fast staging write other bytes."""
    runs = {mode: subprocess.Popen([sys.executable, str(STAGE), "tile", "64", "2", str(tmp_path / mode)], text=True,
                                   env={**os.environ, "HZ_STAGE": mode}, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for mode in ("fast", "library")}
    for mode, p in runs.items():
        _, err = p.communicate()
        assert p.returncode == 0, f"{mode}: {err[-2000:]}"
    fast, lib = tmp_path / "fast", tmp_path / "library"
    files = sorted(f.name for f in lib.iterdir())
    assert sorted(f.name for f in fast.iterdir()) == files
    for name in files:
        if name != "stage.json":
            assert (fast / name).read_bytes() == (lib / name).read_bytes(), name
    fs, ls = (json.loads((d / "stage.json").read_text()) for d in (fast, lib))
    assert (fs["staging"], ls["staging"]) == ("fast", "library")
    assert {k: v for k, v in fs.items() if k not in OWN} == {k: v for k, v in ls.items() if k not in OWN}
