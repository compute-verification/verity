"""The A100 (sm_80): honest's egemm80 profiles and the designs priced on AMPERE-80."""
import pytest

from verity_machine_model import eps as E
from verity_machine_model.model import DEVICES

SM80 = DEVICES / "sm_80"


def test_sm80_the_fold_binds_honest_at_l16_and_issue_at_l32():
    """At L = 16 duo's 8 ALU-pipe instructions a IMMA.16816 hold the floor program on row A, above JIM, so no route saves
    time at any q* and the core ε is ε_skip. At L = 32 the floor program is on JIM, IMMA.16832's 1.0156 a mainloop MMA
    (the peel's tail included) at 4.370 and every other instruction at 0.971, 1.2% above row A: lane words save time
    until q* reaches 1.149, and at q* = 1 the cheater reaches row A's floor of the fold, one warp instruction a
    SMSP-cycle on issue keeping a share of its words on the tensor route."""
    pi = E.machine("", SM80)
    assert E.UNI not in pi.c and pi.vector == "AMPERE-80"   # no uniform-datapath line: the uniform route is closed
    assert {"issue", "I"} <= set(pi.rows) and all(c["issue"] == 1 for c in pi.c.values())
    assert E.designs(SM80) == [f"dither-{n}-l{L}" for n in ("1s", "2s", "c") for L in (16, 32, 64)]
    alu = 4 / (1.96965 * 1.02)                               # an ALU instruction's SMSP-cycles on A, its rate + 2%
    d16 = E.design("dither-1s-l16", SM80)
    s16 = E.setup(d16, pi, SM80, "floor", "floor")
    assert s16.as_built.T * 128 == pytest.approx(8 * alu, rel=1e-6) and s16.as_built.binding == ("A",)
    for q in (1, 2, 4):
        e = E.eps(d16, pi, q, SM80, operands="floor", credit="floor")
        assert e.machine_eps == 0 and e.core() == pytest.approx(e.skip, abs=1e-12)
    assert E.threshold(s16) == 0
    assert e.T_honest * 128 == pytest.approx(23.6241) and e.T_honest_source == "measured"
    assert e.gamma_clock == pytest.approx(e.gamma * 1410 / 1380, rel=1e-5)
    s32 = E.setup(E.design("dither-1s-l32", SM80), pi, SM80, "floor", "floor")
    flat = 1.0156 * 4.37036 + (12 + 0.025641) * 0.970885
    assert s32.as_built.T * 128 == pytest.approx(flat, rel=1e-6) and s32.as_built.binding == ("JIM",)
    c1 = s32.cheat(1.0)
    assert c1.T * 128 == pytest.approx(8 * alu, rel=1e-6) and "A" in c1.binding and 0 < c1.words["lanes"] < 1
    assert s32.machine_eps(1.0) == pytest.approx(1 - 8 * alu / flat, rel=1e-6)
    assert E.threshold(s32) == pytest.approx(1.149, abs=1e-3) and s32.machine_eps(1.15) == pytest.approx(0, abs=1e-9)


def test_sm80_honest_is_measured_at_each_spacing():
    """Each design runs its own L's measured egemm80 arm, so honest's time is measured, and under credit H ε_sched is
    the hold's time above the as-built model."""
    pi = E.machine("", SM80)
    measured = {16: 23.6241, 32: 27.6427, 64: 21.0421}
    for L, t in measured.items():
        d = E.design(f"dither-2s-l{L}", SM80)
        e = E.eps(d, pi, 2.0, SM80)
        words = e.honest.words * e.honest.depth / L
        assert e.T_honest_source == "measured" and e.T_honest * words == pytest.approx(t)
        assert e.sched == pytest.approx(1 - e.as_built.T / e.T_honest, rel=1e-9) and e.sched > 0.2
