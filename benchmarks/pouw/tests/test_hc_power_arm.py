"""HC-POWER's filler arms (benchmarks/pouw/filler/hc_power.py) meet the scorer's arm contract, run kern-m2's kernel
(DFILL_MAP 2, its chunk shape), and place the filler exactly where kern-m2 does at every decode point, so the prefill
split is their only change."""

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


@pytest.mark.parametrize("attr, name, pre", [("PowerPre4", "power-pre4", 4)])
def test_hc_power_arms(monkeypatch, attr, name, pre):
    A = sys.modules.get("arms") or _load("arms")
    monkeypatch.setenv("DFILL_MAP", "0")
    arm = A.load(f"benchmarks/pouw/filler/hc_power.py:{attr}")
    assert arm.name == name and arm.pre_sms == pre and A.check_arm(arm) == []
    assert os.environ["DFILL_MAP"] == "2"
    kern = A.load("benchmarks/pouw/filler/hc_kern.py:KernM2")
    assert arm.filler == kern.filler
    for ph, b, l in S.points():
        if ph == "prefill":
            assert A.placed(arm, ph, b, l) == A.Placement("split", arm.pre_sms)
        else:
            assert A.placed(arm, ph, b, l) == A.placed(kern, ph, b, l)


def test_power_split_names_every_scored_point(monkeypatch):
    A = sys.modules.get("arms") or _load("arms")
    monkeypatch.setenv("DFILL_MAP", "0")
    arm = A.load("benchmarks/pouw/filler/hc_power.py:PowerSplit")
    assert arm.name == "power-split" and A.check_arm(arm) == [] and os.environ["DFILL_MAP"] == "2"
    table = type(arm).placement.__globals__["POWER_SPLIT"]  # arms.load doesn't register the module
    assert {(b, l) for e in S.suite() for ph, b, l in e["points"] if ph == "decode"} == set(table)
    for ph, b, l in S.points():
        want = A.Placement("split", arm.pre_sms if ph == "prefill" else A.point_lookup(table, b, l))
        assert A.placed(arm, ph, b, l) == want
