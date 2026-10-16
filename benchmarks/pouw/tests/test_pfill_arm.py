"""HC-PROVED's scorer arms (benchmarks/pouw/filler/pfill_arm.py) meet the filler arm contract, and place the filler as
FILL v1 does at every suite point."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

FILLER = Path(__file__).resolve().parents[1] / "filler"


def _module(name):
    """The scorer's module as test_filler_score.py registered it, else loaded here: an arm file imports `arms` from
    sys.modules, so one copy must serve every test in the process."""
    mod = sys.modules.get(name)
    if mod is not None and Path(getattr(mod, "__file__", "")).resolve() == FILLER / f"{name}.py":
        return mod
    spec = importlib.util.spec_from_file_location(name, FILLER / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_pfill_arms_meet_the_contract_and_place_as_fill_v1():
    A, S = _module("arms"), _module("scoring")
    v1 = A.load("fill-v1")
    for attr, nc in (("PFillRam", 4), ("PFillDuo", 2)):
        arm = A.load(f"benchmarks/pouw/filler/pfill_arm.py:{attr}")
        assert A.check_arm(arm) == []
        assert arm.filler == A.Filler(W=8, NC=nc, S=256, chunk_log2=19)
        assert arm.filler.S % 32 == 0  # pfill's dfill_make refuses any other S
        assert "benchmarks/pouw/decode_capacity/pfill.cuh" in arm.sources
        for ph, b, l in S.points():
            assert A.placed(arm, ph, b, l) == A.placed(v1, ph, b, l)
