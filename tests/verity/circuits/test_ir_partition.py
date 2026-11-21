"""`verity.circuits.partition`: readiness on small Programs from test-local primitives, `gemm_program()`
(`TGemm{K=32,N=6}`: Input batches, a composite batch, nested calls) and `wide_program()` (64-bit gates)."""
from __future__ import annotations

import pytest

from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.partition import ReadinessReport, gate_width_readiness, readiness_32bit
from verity.circuits.program import Program
from verity.circuits.refs import tuple_of
from verity.circuits.types import Tuple, Value

from test_ir_liveness import V32, Add32, Gemm, Zero32


# ---------------------------------------------------------------------------------------------------------
# fixture programs
# ---------------------------------------------------------------------------------------------------------


def gemm_program() -> Program:
    """Root: Input16 x (32), Input16 w (192), batch TGemmCoordinate_v1{K=32} x 6 -> 248 gates.
    TGemmCoordinate body: call TDot (3 gates: TZero32 + 2 TDot16 steps), TCvt16."""
    return Program(bind(Gemm, K=32, N=6))


_Wide64 = PrimitiveDefinition("TWide64", 1, (), Value(64), lambda: 0, register=False)


def wide_program() -> Program:
    """A Program with a 64-bit Input and a 64-bit primitive gate (readiness violators of both kinds)."""
    W = CompositeDefinition("TWide", 1, (), lambda S: ((("seed", Value(64)),), Tuple(Value(64), V32)),
                            lambda B, S, seed: tuple_of(B.call(_Wide64), B.call(Add32, B.call(Zero32), B.call(Zero32))),
                            register=False)
    return Program(bind(W))


@pytest.fixture
def gemm():
    return gemm_program()


# -- readiness ----------------------------------------------------------------------------------------------


def test_readiness_ok_on_gemm(gemm):
    r = readiness_32bit(gemm)
    assert isinstance(r, ReadinessReport) and r.ok and r.violators == () and r.max_w == 32 and r.specs == 3


def test_readiness_lists_64bit_input_and_primitive():
    P_ = wide_program()
    r = readiness_32bit(P_)
    assert not r.ok and r.max_w == 64 and r.gates_over == 2
    prims = sorted((v.prim, v.form, v.members, v.instances) for v in r.violators)
    assert prims == [("Input64_v1", "batch", 1, 1), ("TWide64_v1", "primitive", 1, 1)]
    assert all(v.spec == P_.root.id for v in r.violators)
    # 64 is a parameter: at w_max = 64 the same Program is ready
    assert gate_width_readiness(P_, 64).ok


def test_readiness_counts_instances_by_multiplicity(gemm):
    # the 32-bit gates sit in TDot, instantiated 6 times: at w_max = 16 each is counted with multiplicity 6
    r = gate_width_readiness(gemm, 16)
    over = {(v.prim.split("_")[0], v.instances) for v in r.violators}
    assert ("TZero32", 6) in over and ("TDot16", 6) in over
    assert r.gates_over == 6 * 3   # TZero32 + 2 TDot16 steps per coordinate; TCvt16 (16) and Input16 are fine
