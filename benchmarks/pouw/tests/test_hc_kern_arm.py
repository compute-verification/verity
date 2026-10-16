"""HC-KERN's filler arms (benchmarks/pouw/filler/hc_kern.py) meet the scorer's arm contract, select their dcap.cu map
through DFILL_MAP, and place the filler exactly where `fill-v1` does, so the kernel is their only change."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

FILLER = Path(__file__).resolve().parents[1] / "filler"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, FILLER / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


S = _load("scoring")


@pytest.mark.parametrize("attr, name, dmap", [("KernM2", "kern-m2", "2"), ("KernM2C18", "kern-m2-c18", "2"),
                                              ("KernM1", "kern-m1", "1")])
def test_hc_kern_arms(monkeypatch, attr, name, dmap):
    # the arm imports whichever `arms` module is registered when it loads, as under score.py
    A = sys.modules.get("arms") or _load("arms")
    monkeypatch.setenv("DFILL_MAP", "0")
    arm = A.load(f"benchmarks/pouw/filler/hc_kern.py:{attr}")
    assert arm.name == name and A.check_arm(arm) == []
    assert os.environ["DFILL_MAP"] == dmap
    v1 = A.load("fill-v1")
    assert all(A.placed(arm, *pt) == A.placed(v1, *pt) for pt in S.points())
    f = arm.filler
    assert (1 << f.chunk_log2) >= f.W * f.NC * f.S, "a chunk is at least one tile a warp"
