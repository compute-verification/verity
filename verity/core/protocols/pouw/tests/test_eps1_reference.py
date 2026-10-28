"""The EPS1 sm_120 H1 + fx1 design on words (`circuit.eps1_reference`): a non-registered reference, so the
served scheme (`circuit.h1tile`) can be tested against it (`test_h1tile_matches_eps1`).  These pin the reference's own
surface: it is not in `circuit.SCHEMES`, its arm is the s8 family, its work table is the served MACs a dot and its closure
is empty.  Pure Python, fast (no Boolean lowering): the arm's and the spot check's lowerings are circuit-check's."""
from __future__ import annotations

import pytest

from verity.ml.protocols.pouw import circuit
from verity.ml.protocols.pouw.circuit import eps1_reference as EPS
from verity.core.protocols.pouw.protocol import DomainError

SHAPE = EPS.Shape(16, 128, 16)       # KS = 4


def test_the_reference_is_not_registered():
    """Decision 102 is one design across schemes, so `eps1-sm120-h1fx1` has one registration, `h1tile`'s compile-pass
    scheme; this reference is not in `circuit.SCHEMES` and `circuit.traced_as` does not reach it."""
    assert EPS.SCHEME not in circuit.SCHEMES
    assert set(circuit.SCHEMES) == {"ncp-v2", "ncp-v2-shift24"}
    assert circuit.traced_as(EPS.SCHEME, SHAPE.m, SHAPE.k, SHAPE.n) is None


def test_the_reference_traces_its_arm_and_spot_check():
    """`eps1_reference.traced_as` builds the design the verifier would: the served arm, the spot check that is the drawn
    unit, the work table and the empty closure, each from the shape alone."""
    t = EPS.traced_as(SHAPE.m, SHAPE.k, SHAPE.n)
    assert isinstance(t, EPS.Eps1Traced) and t.scheme == EPS.SCHEME and t.kind == EPS.KIND and t.shape == SHAPE
    assert t.check == EPS.check(SHAPE.ks) and t.arm == EPS.arm(SHAPE.k)


def test_the_work_table_is_the_served_macs_per_dot_and_the_closure_is_empty():
    t = EPS.traced_as(SHAPE.m, SHAPE.k, SHAPE.n)
    assert t.work == EPS.work_ref(SHAPE)
    assert t.work["per_dot"] == SHAPE.k                               # K served MACs a output coordinate
    assert t.work["per_call"] == SHAPE.k * 32                         # 32 accumulator words a thread
    assert t.work["per_window"] == SHAPE.k * SHAPE.m * SHAPE.n
    assert t.closure == () and EPS.closure(SHAPE) == ()              # no unit's soundness rests on another's


def test_the_shape_domain_is_32_k_and_at_least_two_segments():
    with pytest.raises(DomainError, match="32 . k"):
        EPS.traced_as(16, 48, 16)
    with pytest.raises(DomainError, match="16 . m"):
        EPS.traced_as(15, 128, 16)
    with pytest.raises(DomainError, match="KS >= 2"):
        EPS.traced_as(16, 32, 16)                                     # KS = 1: no digit word


def test_the_arm_is_the_s8_family_at_its_noised_layout():
    from verity.ml.circuits.ops import s8 as OS8
    from verity.ml.circuits.defs import bind
    # the arm at K = 64 is exactly s8's edited-arm smoke binding (NOISE weight, LIMBS byteclamp, COMMIT prefix), which the
    # s8 sidecar already circuit-checks; so the arm needs no new circuit-check target, only the fx1 + chains spot check does.
    assert EPS.arm(64) == bind(OS8.S8Dot, **OS8.SMOKE_EDITED)
