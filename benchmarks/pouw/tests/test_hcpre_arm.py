"""HC-PREFILL's arm (benchmarks/pouw/filler/hcpre_arm.py) meets the arm contract, selects map 1 through DFILL_MAP, and
differs from HC-KERN's KernPhase only in its prefill co shape, never in where it places a filler."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "filler"))
import psched_score as P

A, S = P.A, P.S
ARM = "benchmarks/pouw/filler/hcpre_arm.py:PrefillCo"


@pytest.fixture(autouse=True)
def _one_arms_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "arms", A)
    monkeypatch.setitem(sys.modules, "scoring", S)


def test_prefill_co_arm(monkeypatch):
    monkeypatch.setenv("DFILL_MAP", "0")
    arm = A.load(ARM)
    base = A.load("benchmarks/pouw/filler/hc_kern_phase.py:KernPhase")
    assert not A.check_arm(arm) and arm.name == "hcpre-m1-co-s512"
    assert P.os.environ["DFILL_MAP"] == "1"
    assert arm.filler == base.filler
    assert "benchmarks/pouw/filler/hcpre_arm.py" in arm.sources
    assert P.filler_of(arm, "prefill", 8192, 0) == A.Filler(W=4, NC=4, S=512, chunk_log2=18)
    assert A.placed(arm, "prefill", 8192, 0) == A.Placement("co")
    for ph, b, l in S.points():
        assert A.placed(arm, ph, b, l) == A.placed(base, ph, b, l)
        assert P.phase_of(arm, ph, b, l) == P.phase_of(base, ph, b, l)
        if ph == "decode":
            assert P.filler_of(arm, ph, b, l) == P.filler_of(base, ph, b, l)
    assert S.honesty(arm, {}, S.points())["ok"]


def test_prefill_co_one_arm(monkeypatch):
    """PrefillCoOne is PrefillCo with DFILL_ONE set for every process and 2^20-MMA prefill chunks; it places every filler
    and shapes every decode filler the same."""
    monkeypatch.setenv("DFILL_MAP", "0")
    monkeypatch.delenv("DFILL_ONE", raising=False)
    arm = A.load("benchmarks/pouw/filler/hcpre_arm.py:PrefillCoOne")
    base = A.load(ARM)
    assert not A.check_arm(arm) and arm.name == "hcpre-m1-co-one"
    assert P.os.environ["DFILL_MAP"] == "1" and P.os.environ["DFILL_ONE"] == "4"
    assert arm.lineage["parent"] == base.name and "kernel:one-block-an-sm" in arm.lineage["mechanism"]
    assert P.filler_of(arm, "prefill", 8192, 0) == A.Filler(W=4, NC=4, S=512, chunk_log2=20)
    for ph, b, l in S.points():
        assert A.placed(arm, ph, b, l) == A.placed(base, ph, b, l)
        assert P.phase_of(arm, ph, b, l) == P.phase_of(base, ph, b, l)
        if ph == "decode":
            assert P.filler_of(arm, ph, b, l) == P.filler_of(base, ph, b, l)
    assert S.honesty(arm, {}, S.points())["ok"]
