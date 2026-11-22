"""Readiness on Programs with 64-bit gates."""
from __future__ import annotations

import pytest
from verity.circuits.defs import bind
from verity.circuits.partition import ReadinessReport, gate_width_readiness, readiness_32bit
from verity.circuits.program import Program

from tests.query.test_partition_stubs import gemm_program, wide_program


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
    assert prims == [("Input64_v1", "batch", 1, 1), ("QpartWide64_v1", "primitive", 1, 1)]
    assert all(v.spec == P_.root.id for v in r.violators)
    # 64 is a parameter: at w_max = 64 the same Program is ready
    assert gate_width_readiness(P_, 64).ok


def test_readiness_gumbel_const64():
    G = pytest.importorskip("verity_vllm.definitions.registry.sampling")
    P_ = Program(bind(G.GumbelArgmaxBf16, V=8, SEED=12345, POS=3))
    r = readiness_32bit(P_)
    assert not r.ok
    assert sorted(v.prim for v in r.violators) == ["Const64[0x0000000000000003]_v1", "Const64[0x0000000000003039]_v1"]
    assert r.gates_over == 2


def test_readiness_counts_instances_by_multiplicity(gemm):
    # every AmpereBF16TcDot16 (32 bits) sits in DotBf16, instantiated 6 times: at w_max = 16 the count is 6 * 2 + 6 (Const32) + 6 (F2fpBf16? no: 16 bits)
    r = gate_width_readiness(gemm, 16)
    over = {(v.prim.split("_")[0], v.instances) for v in r.violators}
    assert ("Const32[0x00000000]", 6) in over and ("AmpereBF16TcDot16", 6) in over
    assert r.gates_over == 6 * 3   # Const32 + 2 Ampere steps per coordinate; F2fpBf16 (16) and Input16 are fine
