"""The tables vLLM's primitives read by index are declared (`PrimitiveDefinition.tables`): the SiLU, RN32 tanh and gelu-tanh
activations (`silu_bf16`, `tanh_rn`, `gelu_tanh_bf16`) and the int8 LUT softmax's exp2 table (`exp2_lut127`) are the program's,
registered inputs the primitives still hold, so the guard (`verity.ml.circuits.operations.program_data`) lists them; FA3's
MUFU.RCP table is the library's, and on bits DeepSeek-V4's MUFU.LG2 table is an operation."""
from __future__ import annotations

import pytest

from verity.ml.circuits import constants as K
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits import operations as O
from verity.ml.definitions.registry import b1, boolean_dsv4_moe as BD, dense, dsv4_moe as D, hopper
from verity.ml.definitions.registry import int8_exact as X
from verity.ml.definitions.registry import prims as P


def test_the_program_tables_are_declared_and_the_library_table_is_not_data():
    assert P.SiluMulBf16.tables == (("silu_bf16", "registered"),)
    assert dense.TanhF32Rn.tables == (("tanh_rn", "registered"),)
    assert dense.GeluTanhMulBf16.tables == (("gelu_tanh_bf16", "registered"),)
    assert X.Exp2Lut127.tables == (("exp2_lut127", "registered"),)
    assert hopper.Fa3InvSum.tables == (("rcp", "library"),) and K.held_tables(hopper.Fa3InvSum, O.is_library_table) == []
    assert D.MufuLg2Mantissa.tables == (("lg2", "library"),) and K.held_tables(D.MufuLg2Mantissa, O.is_library_table) == []


def test_the_guard_lists_a_program_table_the_primitive_still_holds():
    assert O.program_data(Program(b1.SiluMul.bind(I=2)).root) == {P.SiluMulBf16.id: [(0, "silu_bf16")]}
    assert O.program_data(Program(dense.GeluTanhMul.bind(I=2)).root) == {dense.GeluTanhMulBf16.id: [(0, "gelu_tanh_bf16")]}
    assert O.program_data(Program(dense.Bf16Tanh.bind(N=2)).root) == {dense.TanhF32Rn.id: [(0, "tanh_rn")]}
    assert O.program_data(Program(X.Int8LutSoftmax.bind(S=2)).root)[X.Exp2Lut127.id] == [(0, "exp2_lut127")]


@pytest.mark.slow
def test_the_hash_routers_boolean_programs_hold_no_data():
    """On bits, `MufuLg2Mantissa_v2` (the library's `lg2` folded into its gates, with its +0 sign bit) is an operation, so the
    hash router's Boolean Programs hold no data."""
    assert O.is_operation(bind(BD.MufuLg2Mantissa))
    for fn in (bind(BD.Dsv4HashWeight), bind(BD.MoeRouterHash, E=8, SCALE=2.5)):
        assert O.program_data(Program(fn).root) == {}, fn.id


def test_a_program_reading_mufu_lg2_holds_no_data():
    """DeepSeek-V4's MUFU.LG2 table is the library's `lg2`: a Program calling `MufuLg2Mantissa_v1` holds no program data."""
    reader = CompositeDefinition("Lg2Reader", 1, (), lambda S: ((("x", P.F32),), P.F32),
                                 lambda B, S, x: B.call(D.MufuLg2Mantissa, x), register=False)
    assert O.program_data(Program(bind(reader)).root) == {}
