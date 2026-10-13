"""The tables vLLM's primitives read by index are declared (`PrimitiveDefinition.tables`): the SiLU, RN32 tanh and gelu-tanh
activations (`silu_bf16`, `tanh_rn`, `gelu_tanh_bf16`) are the program's, registered inputs the primitives still hold, so the
guard (`verity.ml.circuits.operations.program_data`) lists them; FA3's MUFU.RCP table and DeepSeek-V4's MUFU.LG2 table are the
library's."""
from __future__ import annotations

from verity.ml.circuits import constants as K
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits import operations as O
from verity_vllm.program.registry import b1, boolean_dsv4_moe as BD, dense, dsv4_moe as D, hopper
from verity_vllm.program.registry import prims as P


def test_the_program_tables_are_declared_and_the_library_table_is_not_data():
    assert P.SiluMulBf16.tables == (("silu_bf16", "registered"),)
    assert dense.TanhF32Rn.tables == (("tanh_rn", "registered"),)
    assert dense.GeluTanhMulBf16.tables == (("gelu_tanh_bf16", "registered"),)
    assert hopper.Fa3InvSum.tables == (("rcp", "library"),) and K.held_tables(hopper.Fa3InvSum, O.is_library_table) == []
    assert D.MufuLg2Mantissa.tables == (("lg2", "library"),) and K.held_tables(D.MufuLg2Mantissa, O.is_library_table) == []


def test_the_guard_lists_a_program_table_the_primitive_still_holds():
    assert O.program_data(Program(b1.SiluMul.bind(I=2)).root) == {P.SiluMulBf16.id: [(0, "silu_bf16")]}
    assert O.program_data(Program(dense.GeluTanhMul.bind(I=2)).root) == {dense.GeluTanhMulBf16.id: [(0, "gelu_tanh_bf16")]}
    assert O.program_data(Program(dense.Bf16Tanh.bind(N=2)).root) == {dense.TanhF32Rn.id: [(0, "tanh_rn")]}


def test_a_program_reading_mufu_lg2_holds_no_data():
    """`MufuLg2Mantissa_v1`'s table is the library's `lg2`, so the guard lists nothing for it; on bits, `MufuLg2Mantissa_v2` is an
    operation, so the hash router's Boolean Programs hold no data either."""
    lg2 = bind(CompositeDefinition("KLg2", 1, (), lambda S: ((("x", P.F32),), P.F32),
                                   lambda B, S, x: B.call(D.MufuLg2Mantissa, x), register=False))
    root = Program(lg2).root
    assert K.program_data(root) == {D.MufuLg2Mantissa.id: [(0, "lg2")]}
    assert O.program_data(root) == {}
    assert O.is_operation(bind(BD.MufuLg2Mantissa))
    for fn in (bind(BD.Dsv4HashWeight), bind(BD.MoeRouterHash, E=8, SCALE=2.5)):
        assert O.program_data(Program(fn).root) == {}, fn.id
