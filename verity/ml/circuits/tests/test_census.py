"""`verity.ml.circuits.census` and `verity.ml.circuits.stacking` on small synthetic Programs: exact int8 Programs pass, and a
clamping, wrapping, FP-smuggling, rounding-heavy or input-holding one fails with its reason."""
from __future__ import annotations

import json

import pytest

from verity.ml.circuits import census as C
from verity.ml.circuits import stacking as P
from verity.ml.circuits.codec import canonical_json
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition
from verity.ml.circuits.ops import prims
from verity.ml.circuits.ops import fp32
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import array_of, tuple_of
from verity.ml.circuits.silicon.integer import BLACKWELL_SM120_S8_M16N8K32, s32_value, s32_word
from verity.ml.circuits.types import Array, Tuple as TupleT, Value

S8, S32 = Value(8), Value(32)
IMMA = prims.BlackwellS8ImmaDot32
QMMA = prims.BlackwellE4m3QmmaDot32

#: a wrapping s8 step (the IMMA without .satfinite, which has no registry entry): unclassified
WRAP = PrimitiveDefinition("TestS8ImmaWrap", 1, (("acc", S32), ("a", Array(32, S8)), ("b", Array(32, S8))), S32,
                           lambda *v: 0, register=False)


def _step(K: int, mma=IMMA) -> CompositeDefinition:
    """One step: acc + x . w over K s8 lanes, as K/32 chained steps of `mma`."""
    def body(B, _S, acc, x, w):
        for j in range(K // 32):
            acc = B.call(mma, acc, x[32 * j:32 * (j + 1)], w[32 * j:32 * (j + 1)])
        return acc
    return CompositeDefinition(f"TestStep{K}{mma.name}", 1, (), lambda _S: ((("acc", S32), ("x", Array(K, S8)), ("w", Array(K, S8))), S32),
                               body, register=False)


def _fma_epilogue(B, _S, acc):
    """Two FP32 roundings on an accumulator word, one of them the add of a run x run product."""
    return B.call(fp32.F32Add, B.call(fp32.F32Fma, acc, acc, acc), acc)


EPILOGUE = CompositeDefinition("TestEpilogue", 1, (), lambda _S: ((("acc", S32),), S32), _fma_epilogue, register=False)


def int8_program(S: int = 3, K: int = 64, *, mmas=None, acc0="zero", epilogue=False, reread=None) -> tuple[Program, list[int]]:
    """S steps: step s reads its run input x[s] and the registered w, and starts from step s-1's accumulator (dynamic memory);
    step 0 from zero, or from the run input `c` when acc0 = "input". `epilogue`: each step also outputs `EPILOGUE` of its
    accumulator. `reread`: (step, earlier step) whose x that step reads again. Returns the Program and its steps' first Calls."""
    steps = [_step(K, m) for m in (mmas or [IMMA] * S)]
    params = [("x", Array(S, Array(K, S8))), ("w", Array(K, S8))] + ([("c", S32)] if acc0 == "input" else [])
    starts: list[int] = []

    def body(B, _S, x, w, *c):
        inputs = len(B.nodes)
        starts[:] = [0]
        acc = c[0] if acc0 == "input" else B.call(prims.ZERO32)
        outs = []
        for s, st in enumerate(steps):
            if s:
                starts.append(len(B.nodes) - inputs)
            acc = B.call(st, acc, x[s], w)
            if reread and reread[0] == s:
                acc = B.call(_step(K), acc, x[reread[1]], w)
            outs.append(B.call(EPILOGUE, acc) if epilogue else acc)
        return array_of(outs)

    fn = CompositeDefinition("TestInt8Program", 1, (), lambda _S: (tuple(params), Array(S, S32)), body, register=False)
    return Program(fn, registered=("w",)), list(starts)


def test_the_s8_step_is_the_silicon_model():
    a, b = [127] * 32, [0x80] * 32  # 127 x -128
    assert IMMA.evaluate(s32_word(5), *a, *b) == BLACKWELL_SM120_S8_M16N8K32.step(s32_word(5), a, b)
    assert s32_value(IMMA.evaluate(s32_word(5), *a, *b)) == 5 - 32 * 127 * 128
    assert s32_value(IMMA.evaluate(s32_word(C.S32[0]), *a, *b)) == C.S32[0]  # .satfinite clamps once, at D


def test_an_exact_int8_program_passes():
    program, starts = int8_program()
    cen = C.census(program, starts)
    t = cen.total
    assert t["mma"] == {"BlackwellS8ImmaDot32_v1": {"kind": "s8-imma", "k": 32, "steps": 6, "exact": 6}}
    assert t["macs"] == {"exact_s8": 192, "other": 0}
    assert t["products"] == {"rr": 0, "rsel": 0, "rs": 192} and t["mma_acc"] == {"rr": 0, "rsel": 0, "rs": 6}
    assert P.roundings(t) == 0 and not t["unclassified"] and not cen.may_overflow
    mem = [s["memory"] for s in cen.steps]
    assert [m["inputs"] for m in mem] == [{"run_bits": 512, "registered_bits": 512}] + [{"run_bits": 512, "registered_bits": 0}] * 2
    assert [m["lifetimes"] for m in mem] == [{"1": 32}, {"1": 32}, {}]
    assert [m["output_bits"] for m in mem] == [32, 32, 32]
    assert cen.memory["max_lifetime"] == 1
    v = P.check(cen, c=0, R_in=64, L=1, windows=[P.Window(range(s, s + 1), 64) for s in range(3)])
    assert v.ok and v.reasons == (), v.reasons
    assert v.numbers["windows"][0] == {"steps": [0, 1], "reads": 64, "roundings": 0, "products": 0, "input_bytes": "64"}


def test_the_census_json_is_canonical_and_names_its_table():
    program, starts = int8_program()
    j = C.census(program, starts).to_json()
    assert j["schema"] == "verity/census/v1" and j["classes"] == C.digest()
    assert json.loads(canonical_json(j)) == j
    assert C.census(program, starts).to_json() == j
    assert set(j["definitions"]) == {"Const32[0x00000000]_v1", "TestStep64BlackwellS8ImmaDot32_v1"}
    assert j["definitions"]["TestStep64BlackwellS8ImmaDot32_v1"]["calls"] == 3


@pytest.mark.parametrize("unrolled", [True, False])
def test_an_accumulation_that_can_clamp_fails_and_one_step_shorter_is_exact(unrolled):
    """32 n products of s8 codes reach (-128)^2 * 32 n: exact in s32 up to n = 4095 chained steps, past it at 4096."""
    for n, exact in ((4095, True), (4096, False)):
        if unrolled:
            program, starts = int8_program(S=1, K=32 * n)
        else:
            program, starts = _scan_program(n)
        cen = C.census(program, starts)
        m = cen.total["mma"]["BlackwellS8ImmaDot32_v1"]
        assert m["steps"] == n
        if exact:
            assert m["exact"] == n and not cen.may_overflow
            continue
        assert m["exact"] < n and cen.may_overflow
        site = cen.may_overflow[0]
        assert site["sum"][1] > C.S32[1]
        v = P.check(cen, c=0, R_in=10 ** 9, L=1, windows=[P.Window(range(1), 10 ** 9)])
        assert [r["code"] for r in v.reasons] == ["s8-may-overflow"]


def _scan_program(n: int) -> tuple[Program, list[int]]:
    """One step: the accumulator carried through a scan over n blocks of 32 lanes."""
    def step_body(B, _S, acc, x, w):
        d = B.call(IMMA, acc, x, w)
        return tuple_of(d, d)

    step = CompositeDefinition("TestScanStep", 1, (), lambda _S: ((("acc", S32), ("x", Array(32, S8)), ("w", Array(32, S8))),
                                                                 TupleT((S32, S32))), step_body, register=False)

    def body(B, _S, x, w):
        out = B.scan(step, B.call(prims.ZERO32), xs=[x], shared=[w])
        return out[0]

    fn = CompositeDefinition("TestScanProgram", 1, (), lambda _S: ((("x", Array(n, Array(32, S8))), ("w", Array(32, S8))), S32), body,
                             register=False)
    return Program(fn, registered=("w",)), [0]


def test_an_accumulator_from_a_run_input_can_clamp():
    program, starts = int8_program(acc0="input")
    cen = C.census(program, starts)
    assert cen.total["mma"]["BlackwellS8ImmaDot32_v1"]["exact"] < 6
    assert cen.may_overflow[0]["acc"] == list(C.S32)
    v = P.check(cen, c=0, R_in=10 ** 9, L=1, windows=[P.Window(range(3), 192)])
    assert "s8-may-overflow" in [r["code"] for r in v.reasons]


def test_a_wrapping_step_is_unclassified_and_fails():
    program, starts = int8_program(mmas=[IMMA, WRAP, IMMA])
    cen = C.census(program, starts)
    assert cen.total["unclassified"] == {"TestS8ImmaWrap_v1": 2}
    v = P.check(cen, c=0, R_in=10 ** 9, L=1, windows=[P.Window(range(3), 192)])
    assert [r["code"] for r in v.reasons] == ["unclassified", "s8-may-overflow"]
    assert v.reasons[0]["primitives"] == {"TestS8ImmaWrap_v1": 2}
    assert v.reasons[1]["steps"] == {"BlackwellS8ImmaDot32_v1": 2}  # the next step starts from any s32 word


def test_a_program_smuggling_fp_mmas_fails():
    program, starts = int8_program(mmas=[IMMA, QMMA, IMMA])
    cen = C.census(program, starts)
    t = cen.total
    assert t["mma"]["BlackwellE4m3QmmaDot32_v1"] == {"kind": "fp", "k": 32, "steps": 2, "exact": 0}
    assert t["macs"] == {"exact_s8": 64, "other": 128}  # step 2 starts from an FP word: any s32 word
    v = P.check(cen, c=0, R_in=10 ** 9, L=1, windows=[P.Window(range(3), 192)])
    assert [r["code"] for r in v.reasons] == ["fp-mma", "s8-may-overflow"]
    assert v.reasons[0]["steps"] == {"BlackwellE4m3QmmaDot32_v1": 2}
    assert cen.steps[1]["macs"] == {"exact_s8": 0, "other": 64}


def test_roundings_are_budgeted_per_certified_byte_read():
    """An FMA epilogue on every step: 2 FP32 roundings and 1 run x run product a step. Over c (or c_rr) per certified byte
    read it fails; within both it passes."""
    program, starts = int8_program(epilogue=True)
    cen = C.census(program, starts)
    assert cen.total["roundings"]["arith"] == 6 and cen.total["products"]["rr"] == 3
    assert cen.steps[0]["roundings"]["arith"] == 2
    windows = [P.Window(range(3), 192)]
    assert P.check(cen, c=0, R_in=192, L=1, windows=windows).reasons[0]["code"] == "roundings"
    assert P.check(cen, c="1/32", R_in=192, L=1, windows=windows).ok
    bad = P.check(cen, c="1/32", c_rr="1/128", R_in=192, L=1, windows=windows)
    assert [r["code"] for r in bad.reasons] == ["products"] and bad.reasons[0]["products"] == 3
    assert P.check(cen, c="1/32", c_rr="1/64", R_in=192, L=1, windows=windows).ok


def test_inputs_lifetimes_and_coverage():
    program, starts = int8_program(reread=(2, 0))
    cen = C.census(program, starts)
    assert cen.steps[0]["memory"]["input_lifetimes"] == {"2": 512}
    assert cen.memory["max_lifetime"] == 2
    one = [P.Window(range(s, s + 1), 64) for s in range(3)]
    v = P.check(cen, c=0, R_in=64, L=1, windows=one)
    assert [r["code"] for r in v.reasons] == ["lifetime"] and v.reasons[0] == {"code": "lifetime", "max_lifetime": 2, "L": 1}
    assert P.check(cen, c=0, R_in=64, L=2, windows=one).ok
    v = P.check(cen, c=0, R_in=63, L=2, windows=one[:2])
    assert [r["code"] for r in v.reasons] == ["uncovered", "inputs", "inputs"]
    assert v.reasons[0]["steps"] == [2] and v.reasons[1]["input_bytes"] == "64"


def test_check_refuses_another_table_and_bad_windows():
    program, starts = int8_program()
    j = C.census(program, starts).to_json()
    with pytest.raises(ValueError, match="census"):
        P.check(dict(j, classes="00"), c=0, R_in=0, L=0, windows=[])
    for w in (P.Window(range(0, 4), 1), P.Window(range(2, 2), 1), P.Window(range(0, 3, 2), 1), P.Window(range(3), -1)):
        with pytest.raises(ValueError, match="window"):
            P.check(j, c=0, R_in=0, L=0, windows=[w])
    with pytest.raises(ValueError, match="steps"):
        C.census(program, [1])


def test_a_gather_selects_and_its_products_are_rsel():
    """A run index into a registered column outputs a selected word; a product of it with a run value is ``rsel``."""
    gather = PrimitiveDefinition("GatherBf16x4", 1, (("token", S32), ("column", Array(4, Value(16)))), Value(16),
                                 lambda t, *c: c[t], register=False)
    assert C.kind(gather.id) == C.Kind(gather=4)
    raw, t = C._Provenance().call(gather, (C.RUN, C.REGISTERED))
    assert t == C.SELECTED
    counts = C.classify({**raw, ("Bf16MulBf16_v1", (C.SELECTED, C.RUN)): 3, ("F32Mul_v1", (C.RUN, C.REGISTERED)): 2,
                         ("F32Mul_v1", (C.REGISTERED, C.REGISTERED)): 7})
    assert counts["gathers"] == 1 and counts["gather_fanin"] == 4
    assert counts["products"] == {"rr": 0, "rsel": 3, "rs": 2}
    assert counts["roundings"] == {"cast": 3, "special": 0, "arith": 2, "integer": 0}


def test_product_classes_and_charges():
    assert [C.product_class(a, b) for a, b in ((C.RUN, C.RUN), (C.RUN, C.SELECTED), (C.SELECTED, C.SELECTED), (C.RUN, C.REGISTERED),
                                               (C.SELECTED, C.CONSTANT))] == ["rr", "rsel", "rr", "rs", "rs"]
    assert C.kind("F32Fabs_v1") == C.Kind(compare=1) and C.kind("F32BitsShl23_v1") == C.Kind(integer=1)
    assert C.kind("Bf16ToF32_v1") == C.Kind() and C.kind("And2_v1") is None
    assert C.kind("BlackwellS8ImmaDot32_v1").mma == C.Mma("s8-imma", 32, C.S8, C.S32)
    div = C.classify({("F32Div_v2", (C.RUN, C.RUN)): 1, ("F32Div_v2", (C.RUN, C.REGISTERED)): 1})
    assert div["roundings"]["special"] == 1 and div["roundings"]["arith"] == 1 and div["products"] == {"rr": 1, "rsel": 0, "rs": 1}


def test_count_reads_a_definition_with_given_ranges():
    st = _step(64)
    assert C.count(st, (C.RUN, C.RUN, C.REGISTERED))["mma"]["BlackwellS8ImmaDot32_v1"]["exact"] == 0
    assert C.count(st, (C.RUN, C.RUN, C.REGISTERED), ranges=((0, 0), None, None))["mma"]["BlackwellS8ImmaDot32_v1"]["exact"] == 2


def test_the_table_is_pinned():
    assert C.spec()["kinds"]["BlackwellS8ImmaDot32_v1"] == {"mma": {"kind": "s8-imma", "k": 32, "operand": [-128, 127],
                                                                    "acc": [-(1 << 31), (1 << 31) - 1]}}
    assert len(C.digest()) == 128
