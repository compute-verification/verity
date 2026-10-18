"""The EPS1 sm_120 H1 + fx1 scheme's registration (`circuit.eps1`, decision 102): it is a compile-pass scheme beside
ncp-v2, `circuit.traced_as` dispatches to it, and its work table and closure are the ones the draw law and the audit read.
Pure Python, fast (no Boolean lowering): the arm's and the spot check's lowerings are circuit-check's."""
from __future__ import annotations

import pytest

from verity.core.protocols.pouw import circuit
from verity.core.protocols.pouw.circuit import eps1 as EPS
from verity.core.protocols.pouw.protocol import DomainError

SHAPE = EPS.Shape(16, 128, 16)       # KS = 4


def test_the_scheme_is_registered_as_a_compile_pass_beside_ncp_v2():
    assert circuit.SCHEMES[EPS.SCHEME] == EPS.KIND
    assert EPS.SCHEME in circuit.COMPILE_PASS and "ncp-v2" not in circuit.COMPILE_PASS
    assert set(circuit.SCHEMES) == {"ncp-v2", "ncp-v2-shift24", EPS.SCHEME}


def test_traced_as_dispatches_to_eps1_and_leaves_ncp_v2_alone():
    t = circuit.traced_as(EPS.SCHEME, SHAPE.m, SHAPE.k, SHAPE.n)
    assert isinstance(t, EPS.Eps1Traced) and t.scheme == EPS.SCHEME and t.kind == EPS.KIND and t.shape == SHAPE
    assert t.check == EPS.check(SHAPE.ks) and t.arm == EPS.arm(SHAPE.k)
    # the native-executor path is untouched: ncp-v2 still traces its call Program, and an unknown scheme is None
    assert circuit.traced_as("ncp-v2", 32, 64, 32).definition.name == "NcpLinear"
    assert circuit.traced_as("ncp-v1", 16, 128, 16) is None


def test_the_work_table_is_the_served_macs_per_dot_and_the_closure_is_empty():
    t = circuit.traced_as(EPS.SCHEME, SHAPE.m, SHAPE.k, SHAPE.n)
    assert t.work == EPS.work_ref(SHAPE)
    assert t.work["per_dot"] == SHAPE.k                               # K served MACs a output coordinate
    assert t.work["per_call"] == SHAPE.k * 32                         # 32 accumulator words a thread
    assert t.work["per_window"] == SHAPE.k * SHAPE.m * SHAPE.n
    assert t.closure == () and EPS.closure(SHAPE) == ()              # no unit's soundness rests on another's


def test_the_shape_domain_is_32_k_and_at_least_two_segments():
    with pytest.raises(DomainError, match="32 . k"):
        circuit.traced_as(EPS.SCHEME, 16, 48, 16)
    with pytest.raises(DomainError, match="16 . m"):
        circuit.traced_as(EPS.SCHEME, 15, 128, 16)
    with pytest.raises(DomainError, match="KS >= 2"):
        circuit.traced_as(EPS.SCHEME, 16, 32, 16)                    # KS = 1: no digit word


def test_the_arm_is_the_s8_family_at_its_noised_layout():
    from verity.ml.circuits.ops import s8 as OS8
    from verity.ml.circuits.defs import bind
    # the arm at K = 64 is exactly s8's edited-arm smoke binding (NOISE weight, LIMBS byteclamp, COMMIT prefix), which the
    # s8 sidecar already circuit-checks; so the arm needs no new circuit-check target, only the fx1 + chains spot check does.
    assert EPS.arm(64) == bind(OS8.S8Dot, **OS8.SMOKE_EDITED)
