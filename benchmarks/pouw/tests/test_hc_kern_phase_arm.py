"""HC-KERN's phase arm (benchmarks/pouw/filler/hc_kern_phase.py) meets the arm contract, selects map 1 through DFILL_MAP,
and differs from HC-SCHED's PhaseSchedCo2 only in its split filler and its phase fillers' shapes, never in where it
places a filler."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "filler"))
import psched_score as P

A, S = P.A, P.S
ARM = "benchmarks/pouw/filler/hc_kern_phase.py:KernPhase"


@pytest.fixture(autouse=True)
def _one_arms_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "arms", A)
    monkeypatch.setitem(sys.modules, "scoring", S)


def test_kern_phase_arm(monkeypatch):
    monkeypatch.setenv("DFILL_MAP", "0")
    arm = A.load(ARM)
    base = A.load("benchmarks/pouw/filler/psched_arm.py:PhaseSchedCo2")
    assert not A.check_arm(arm) and arm.name == "kern-m1-phase-co2"
    assert P.os.environ["DFILL_MAP"] == "1"
    assert arm.filler == A.Filler(W=16, NC=4, S=8192, chunk_log2=20)
    assert "benchmarks/pouw/filler/hc_kern_phase.py" in arm.sources
    assert P.filler_of(arm, "prefill", 8192, 0) == P.filler_of(base, "prefill", 8192, 0)
    for ph, b, l in S.points():
        assert A.placed(arm, ph, b, l) == A.placed(base, ph, b, l)
        q, q0 = P.phase_of(arm, ph, b, l), P.phase_of(base, ph, b, l)
        assert (q is None) == (q0 is None)
        if q is not None:
            assert q.S in (128, 256, 512) and q.NC in (2, 4) and q.tiles >= 1
    g = type(arm)._row.__globals__  # arms.load runs the module without registering it
    K, V3 = g["TABLE_K"], g["PS"].TABLE_V3
    assert K.keys() == V3.keys() and all(K[k][0] == V3[k][0] for k in K)
    assert sum(K[k] != V3[k] for k in K) == 12
    assert S.honesty(arm, {}, S.points())["ok"]


@pytest.mark.parametrize("b, l", [(32, 1024), (32, 24576), (48, 16384), (132, 5120), (144, 5120), (256, 1024)])
def test_kern_phase_covers_the_mixes_and_decode_points(b, l):
    arm = A.load(ARM)
    q = arm.phase("decode", b, l)
    assert q is not None and (q.S, q.NC) != (128, 4)
