"""`verity.ml.circuits.stacking` (property/v1) on small synthetic Programs with their census/v1: an exact int8 Program passes,
and a clamping, wrapping, FP-smuggling, reinterpreting, carry-computing, gather-heavy, too wide or input-holding one fails with
its reason, or has its carries budgeted. The red team's attacks (benchmarks/property_redteam) are
cases here."""
from __future__ import annotations

import hashlib
import json
import random
import re
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits import stacking as P
from verity.ml.circuits.census import census
from verity.ml.circuits.codec import canonical_json
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, _Widths
from verity.ml.circuits.ops import fp32, prims, scalar
from verity.ml.circuits.ops.s8 import S32LowByte
from verity.ml.circuits.ports import In, Leaves, Out, Ports
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Coll, Strided, array_of, concat, tuple_of
from verity.ml.circuits.silicon.integer import BLACKWELL_SM120_S8_M16N8K32, s32_value, s32_word
from verity.ml.circuits.types import Array, Tuple as TupleT, Value

S8, S16, S32, BIT = Value(8), Value(16), Value(32), Value(1)
IMMA = prims.BlackwellS8ImmaDot32
QMMA = prims.BlackwellE4m3QmmaDot32
FREE = {"roundings": 0, "compare_select": 0, "gathers": 0, "folds": 0}


def stand_in(name: str, params, ret=S32, ports=None) -> PrimitiveDefinition:
    """A registry primitive's id with a dummy evaluator, for one this unit can't import (vLLM's, PoUW's): the table keys on ids."""
    return PrimitiveDefinition(name, 1, tuple(params), ret, lambda *v: 0, register=False, ports=ports)


#: the IMMA without .satfinite, which has no registry entry
WRAP = stand_in("TestS8ImmaWrap", (("acc", S32), ("a", Array(32, S8)), ("b", Array(32, S8))))
I32DIV = stand_in("I32Div", (("a", S32), ("b", S32)))
I32REM = stand_in("I32Rem", (("a", S32), ("b", S32)))
I32MUL = stand_in("I32Mul", (("a", S32), ("b", S32)))
U32AND = stand_in("U32And", (("a", S32), ("b", S32)))
I32SUB = stand_in("I32Sub", (("a", S32), ("b", S32)))
U32GE = stand_in("U32Ge", (("a", S32), ("b", S32)), BIT)
GATHER = stand_in("GatherBf16x16", (("i", S32), ("col", Array(16, S16))), S16)
GATHER128 = stand_in("GatherF32x128", (("i", S32), ("col", Array(128, S32))), S32)
BITAT = stand_in("BitAtx32", (("word", S32), ("i", S32)), BIT)
TABLE = stand_in("TableRead8x4", (("index", S16), ("table", Array(16, Array(64, S16)))), Value(64))
NCPDOT = stand_in("NcpDot16I7", (("acc", S32), ("a", Array(16, S8)), ("b", Array(16, S8))))
SQACC = stand_in("PouwSqAcc", (("acc", Value(576)), ("x", S32)), Value(576))
U8S8 = stand_in("BlackwellU8S8ImmaDot32", (("acc", S32), ("a", Array(32, S8)), ("b", Array(32, S8))))
F64MUL = stand_in("F64Mul", (("a", Value(64)), ("b", Value(64))), Value(64))
EXP2LUT = stand_in("Exp2Lut127", (("t", S8),))
#: the sha256 of vLLM's LUT softmax table (`engine.int8_attention.LUT`) as 256 little-endian u32 words
EXP2_LUT127_SHA256 = "6eb8e9b3478db58b4e5a711f31968394f7e93fa4e1df2e0d801eab92a5f66f84"


def comp(name: str, params, ret, body, ports=None) -> CompositeDefinition:
    return CompositeDefinition(name, 1, (), lambda _S: (tuple(params), ret), body, register=False, ports=ports)


def _step(K: int, mma=IMMA) -> CompositeDefinition:
    """acc + x . w over K s8 lanes, as K/32 chained steps of `mma`."""
    def body(B, _S, acc, x, w):
        for j in range(K // 32):
            acc = B.call(mma, acc, x[32 * j:32 * (j + 1)], w[32 * j:32 * (j + 1)])
        return acc
    return comp(f"TestStep{K}{mma.name}", (("acc", S32), ("x", Array(K, S8)), ("w", Array(K, S8))), S32, body)


EPILOGUE = comp("TestEpilogue", (("acc", S32),), S32, lambda B, _S, acc: B.call(fp32.F32Add, B.call(fp32.F32Fma, acc, acc, acc), acc))


def int8_program(S: int = 3, K: int = 64, *, mmas=None, acc0="zero", epilogue=False, reread=None,
                 ports=None) -> tuple[Program, list[int]]:
    """S steps: step s reads its run input x[s] and the registered w, and starts from step s-1's accumulator (dynamic memory);
    step 0 from zero, or from the run input `c` when acc0 = "input". `epilogue`: each step also outputs `EPILOGUE` of its
    accumulator. `reread`: (step, earlier step) whose x that step reads again. `ports`: the program function's. Returns the
    Program and its step ends."""
    steps = [_step(K, m) for m in (mmas or [IMMA] * S)]
    params = [("x", Array(S, Array(K, S8))), ("w", Array(K, S8))] + ([("c", S32)] if acc0 == "input" else [])
    ends: list[int] = []

    def body(B, _S, x, w, *c):
        ends.clear()
        acc = c[0] if acc0 == "input" else B.call(prims.ZERO32)
        outs = []
        for s, st in enumerate(steps):
            acc = B.call(st, acc, x[s], w)
            if reread and reread[0] == s:
                acc = B.call(_step(K), acc, x[reread[1]], w)
            outs.append(B.call(EPILOGUE, acc) if epilogue else acc)
            ends.append(len(B.nodes))
        return array_of(outs)

    fn = comp("TestInt8Program", params, Array(S, S32), body, ports=ports)
    return Program(fn, registered=("w",)), ends


def judge(program, ends, *, c=FREE, R_in=10 ** 9, L=None, windows=None, **fw):
    cen = census(program, ends)
    ws = windows if windows is not None else [P.Window(range(len(ends)), 10 ** 9, 1)]
    return P.check(program, cen, c=c, windows=ws, R_in=R_in, L=L, **fw)


def _one_gate_program(prim, registered=()) -> tuple[Program, list[int]]:
    """One gate of `prim` on the Program's inputs, one a parameter (run, or registered when `registered` names it)."""
    fn = comp(f"TestOne{prim.name}", prim.params, prim.ret, lambda B, _S, *xs: B.call(prim, *xs))
    program = Program(fn, registered=registered)
    return program, [len(program.root.body.nodes)]


def one_step_windows(S: int = 3, reads: int = 64) -> list[P.Window]:
    return [P.Window(range(s, s + 1), reads, 1) for s in range(S)]


def codes(v: P.Verdict) -> list[str]:
    return [r["code"] for r in v.reasons]


def test_the_s8_step_is_the_silicon_model():
    a, b = [127] * 32, [0x80] * 32  # 127 x -128
    assert IMMA.evaluate(s32_word(5), *a, *b) == BLACKWELL_SM120_S8_M16N8K32.step(s32_word(5), a, b)
    assert s32_value(IMMA.evaluate(s32_word(5), *a, *b)) == 5 - 32 * 127 * 128
    assert s32_value(IMMA.evaluate(s32_word(P.S32[0]), *a, *b)) == P.S32[0]  # .satfinite clamps once, at D


def test_an_exact_int8_program_passes():
    program, ends = int8_program()
    a = P.analyze(program, census(program, ends))
    assert (a.family, a.macs, a.affine, a.refused, a.unproven, a.sites) == (6, 192, 0, {}, {}, ())
    assert [s["units"] for s in a.steps] == [FREE] * 3
    assert [s["input_bits"] for s in a.steps] == [512] * 3  # x[s]: 64 s8 lanes a step; w is registered
    assert a.max_lifetime == 1  # each accumulator, and each x[s] from its arrival
    v = judge(program, ends, R_in=64, L=1, windows=one_step_windows())
    assert v.ok and v.reasons == (), v.reasons
    assert v.numbers["windows"][0] == {"steps": [0, 1], "reads": 64, "seconds": "1", "units": FREE, "input_bytes": "64"}
    assert (v.numbers["family_steps"], v.numbers["macs"], v.numbers["B"]) == (6, 192, 128)
    assert a.gates["BlackwellS8ImmaDot32_v1"] == {"class": "family", "instances": 6, "operand_bits": 544}
    assert judge(program, ends, B=1).ok  # a free step reads any operand bits


def test_the_verdict_json_is_canonical_and_names_its_table():
    program, ends = int8_program()
    j = judge(program, ends).to_json()
    assert j["schema"] == "verity/stacking-property/v1" and j["classes"] == P.digest()
    assert json.loads(canonical_json(j)) == j
    assert P.analyze(program, census(program, ends)).to_json()["classes"] == P.digest()


@pytest.mark.parametrize("unrolled", [True, False])
def test_an_accumulation_that_can_clamp_fails_and_one_step_shorter_is_exact(unrolled):
    """32 n products of s8 codes reach (-128)^2 * 32 n: within s32 up to n = 4095 chained steps, past it at 4096."""
    for n, exact in ((4095, True), (4096, False)):
        program, ends = int8_program(S=1, K=32 * n) if unrolled else _scan_program(n)
        v = judge(program, ends)
        if exact:
            assert v.ok and v.numbers["family_steps"] == n, v.reasons
            continue
        assert codes(v) == ["s8-window"]
        r = v.reasons[0]
        assert sum(r["steps"].values()) >= 1 and r["sites"][0]["sum"][1] > P.S32[1]


def _scan_program(n: int) -> tuple[Program, list[int]]:
    """One step: the accumulator carried through a scan over n blocks of 32 lanes."""
    def step_body(B, _S, acc, x, w):
        d = B.call(IMMA, acc, x, w)
        return tuple_of(d, d)

    step = comp("TestScanStep", (("acc", S32), ("x", Array(32, S8)), ("w", Array(32, S8))), TupleT((S32, S32)), step_body)
    ends: list[int] = []

    def body(B, _S, x, w):
        out = B.scan(step, B.call(prims.ZERO32), xs=[x], shared=[w])
        ends[:] = [len(B.nodes)]
        return out[0]

    fn = comp("TestScanProgram", (("x", Array(n, Array(32, S8))), ("w", Array(32, S8))), S32, body)
    return Program(fn, registered=("w",)), ends


def test_an_accumulator_from_a_run_input_is_refused():
    """Red team attack 2: a runtime C near the s32 edge makes the clamp an exact comparison, so its window is never proven."""
    program, ends = int8_program(acc0="input")
    v = judge(program, ends)
    assert codes(v) == ["s8-window"] and v.reasons[0]["sites"][0]["acc"] == list(P.S32)


BOUND = (-(1 << 20), 1 << 20)


def test_a_run_inputs_declared_bound_is_refused():
    """No declared `In(bound)` is taken on trust: evaluation doesn't enforce a run input's bound, so a run input whose
    port bounds it far from the s32 edge, whole or on some of its leaves, is refused when its Program is built (`E_PORT_BOUND`),
    and never judged as bounded."""
    with pytest.raises(ValueError, match=r"E_PORT_BOUND: program function TestInt8Program_v1: parameter 'c' leaves \[0, 1\)"):
        int8_program(acc0="input", ports=Ports(params={"c": In(bound=BOUND)}))
    with pytest.raises(ValueError, match=r"E_PORT_BOUND: program function TestFromInputLeaf_v1: parameter 'c' leaves \[0, 1\)"):
        _from("TestFromInputLeaf", lambda B, c: c[0], C0)


def _rem(a, b):
    """A remainder's magnitude is below its divisor's."""
    if b is None:
        return None
    m = max(abs(b[0]), abs(b[1]))
    return (1 - m, m - 1) if m else None


DIV = stand_in("I32Div", (("a", S32), ("b", S32)), ports=Ports(ret=Out(interval=BOUND)))
REM = stand_in("I32Rem", (("a", S32), ("b", S32)), ports=Ports(transfer=_rem))
FIRST = comp("TestFirst", (("c", Array(2, S32)),), S32, lambda B, _S, c: c[0], ports=Ports(ret=Out(interval=BOUND)))
SILENT = comp("TestFirstSilent", (("c", Array(2, S32)),), S32, lambda B, _S, c: c[0])
C0 = Ports(params={"c": Leaves(((0, 1, In(bound=BOUND)),))})
RANGE = stand_in("TestRangeS32", (("a", S32),), ports=Ports(ret=Out(interval=BOUND)))


def _from(name: str, start, ports=None) -> P.Analysis:
    """Two s8 steps from the accumulator `start` makes of the run input c (two s32 words)."""
    fn = comp(name, (("c", Array(2, S32)), ("x", Array(64, S8)), ("w", Array(64, S8))), S32,
              lambda B, _S, c, x, w: B.call(_step(64), start(B, c), x, w), ports=ports)
    program = Program(fn, registered=("w",))
    return P.analyze(program, census(program, [len(program.root.body.nodes)]))


@pytest.mark.parametrize("label,start,proven", [
    ("declared", lambda B, c: B.call(DIV, c[0], c[1]), True),
    ("undeclared", lambda B, c: B.call(I32DIV, c[0], c[1]), False),
    ("transfer", lambda B, c: B.call(REM, c[0], B.call(prims.const(32, 1000))), True),
    ("transfer-of-any-word", lambda B, c: B.call(REM, c[0], c[1]), False),
    ("input-leaf", lambda B, c: c[0], False),
    ("composite", lambda B, c: B.call(FIRST, B.batch(RANGE, c, axes=(0,))), True),
    ("silent-composite", lambda B, c: B.call(SILENT, c), False)])
def test_a_producers_declared_interval_narrows_its_value(label, start, proven):
    """A primitive's declared interval, its transfer on its operands' ranges, and a Definition's declared return (held to its
    body, here a range check's) narrow its value; a run input's leaf spans its width, and so does a composite that declares
    nothing over it."""
    a = _from(f"TestFrom{label.title().replace('-', '')}", start)
    assert a.unproven == ({} if proven else {IMMA.id: 2}), label


def test_a_declared_return_over_a_run_input_is_refused():
    """A Definition's declared return holds by its body: over a run input's leaf, which spans its width, it is refused when
    the Program is built, so a composite's declaration never stands in for a bound on the run input it reads."""
    with pytest.raises(ValueError, match="E_PORT_INTERVAL: TestFirst_v1 declares Out"):
        _from("TestFromCompositeUnchecked", lambda B, c: B.call(FIRST, c))


def test_port_intervals_read_as_signed_ranges():
    assert P._port_range([-5, 5], 32) == (-5, 5) and P._port_range([0, 100], 8) == (0, 100)
    assert P._port_range([0, (1 << 32) - 1], 32) == P.S32  # every encoding: any s32 value
    assert P._port_range([0, 200], 8) is None  # 128..200 read as signed is -128..-56: no one signed interval
    assert P._port_range([-1, 200], 8) is None and P._port_range(None, 8) is None  # no interval of a Value<8>; none
    assert P._meet((0, 10), (5, 20)) == (5, 10) and P._meet(None, (1, 2)) == (1, 2) and P._meet((0, 1), (2, 3)) is None


def test_a_wrapping_step_and_fp_mma_smuggling_are_refused():
    """A step the table doesn't list is refused, and so is an FP tensor-core step: a rounding of 544 operand bits, past B,
    and budgeted besides; the step after either starts from an accumulator of any word, so it isn't proven either."""
    for mma, code, listed in ((WRAP, "unlisted", {WRAP.id: 2}), (QMMA, "operand-bits", {QMMA.id: 544})):
        program, ends = int8_program(mmas=[IMMA, mma, IMMA])
        v = judge(program, ends)
        assert codes(v) == [code, "s8-window"] + (["roundings"] if mma is QMMA else []), v.reasons
        assert v.reasons[0]["primitives"] == listed
        assert v.reasons[1]["steps"] == {"BlackwellS8ImmaDot32_v1": 2}  # the third step's two: the second starts from the first
        assert v.numbers["family_steps"] == 2
    assert v.numbers["units"]["roundings"] == 2 and codes(judge(program, ends, B=544)) == ["s8-window", "roundings"]


#: an s32 word's view as 32 s8 lanes: one Definition returns it, and so passes the reinterpretation to its callers
NARROW = comp("TestNarrow", (("d", S32),), Array(32, S8),
              lambda B, _S, d: Coll(Array(32, S8), Strided(*d.refs[0], 1, 0, 0, 32)))


def _reinterpreting_program(how: str) -> tuple[Program, list[int]]:
    """An IMMA chain whose second step reads an s32 word back as s8 lanes, a free limb split: through a broadcast of the word
    ("broadcast"), the words of a batch of 32 steps ("affine"), or a Definition that returns that view ("ret"). "mixed": the
    second step reads the s8 field of a (s32, s8 x 32) tuple, at its own width."""
    def body(B, _S, x, w):
        d = B.call(IMMA, B.call(prims.ZERO32), x, w)
        if how == "broadcast":
            lanes = Coll(Array(32, S8), Strided(*d.refs[0], 1, 0, 0, 32))
        elif how == "affine":
            ds = B.batch(IMMA, B.call(prims.ZERO32), x, w, axes=(None, None, None), n=32)
            lanes = Coll(Array(32, S8), ds.refs)
        elif how == "ret":
            lanes = B.call(NARROW, d)
        else:
            pair = Coll(TupleT((S32, Array(32, S8))), concat([d.refs, x.refs]))
            pair = B.call(comp("TestPair", (("p", pair.type),), pair.type, lambda B, _S, p: p), pair)
            lanes = Coll(Array(32, S8), pair.refs.slice(1, 33))
        return B.call(IMMA, d, lanes, w)

    fn = comp(f"TestReinterpret{how.title()}", (("x", Array(32, S8)), ("w", Array(32, S8))), S32, body)
    program = Program(fn, registered=("w",))
    return program, [len(program.root.body.nodes)]


_IMMA_A = r"node \d+ \(primitive BlackwellS8ImmaDot32_v1\) argument 'a': leaf 0 is typed Value<8> but names leaf 0 of node \d+"


@pytest.mark.parametrize("how,where,built", [
    ("broadcast", "Root[TestReinterpretBroadcast_v1]_v1", _IMMA_A), ("affine", "Root[TestReinterpretAffine_v1]_v1", _IMMA_A),
    ("ret", "TestNarrow_v1", r"the body's return: leaf 0 is typed Value<8> but names leaf 0 of parameter 'd'"),
    ("mixed", None, None)])
def test_a_word_read_at_another_width_is_refused(how, where, built, monkeypatch):
    """A word read back at another width is refused when its Program is built (`E_REINTERPRET`, as when decoded); past that
    check (patched out here), the property refuses it too. The s8 field of a mixed tuple, read at its own width, passes both."""
    if where is None:
        v = judge(*_reinterpreting_program(how))
        assert v.ok and v.numbers["family_steps"] == 2, v.reasons
        return
    with pytest.raises(TypeError, match=f"E_REINTERPRET: {built}"):
        _reinterpreting_program(how)
    monkeypatch.setattr(_Widths, "first", lambda self, c: None)
    v = judge(*_reinterpreting_program(how))
    assert codes(v) == ["reinterpretation"] and v.reasons[0]["definitions"] == {where: 1}, v.reasons


def _carry_program() -> tuple[Program, list[int]]:
    """Red team attack 1 in miniature: a column sum by an IMMA chain from zero (in window), then its carry t -> (t mod 256,
    t div 256) by I32Rem and I32Div, a comparison carry by I32Le and SelectI32, a bit carry by U32And, a split-K combine of two
    in-window partials by I32Add (free), the same add on a run word (it may wrap: budgeted), and products by a constant (free)
    and of two run words (a rounding: reduced mod 2^32)."""
    def body(B, _S, x, w, r):
        p1 = B.call(_step(64), B.call(prims.ZERO32), x[:64], w[:64])
        p2 = B.call(_step(64), B.call(prims.ZERO32), x[64:], w[64:])
        k256 = B.call(prims.const(32, 256))
        outs = [B.call(I32REM, p1, k256), B.call(I32DIV, p1, k256),
                B.call(scalar.SelectI32, B.call(scalar.I32Le, p1, k256), p1, k256), B.call(U32AND, p1, k256),
                B.call(scalar.I32Add, p1, p2), B.call(scalar.I32Add, r, p1),
                B.call(I32MUL, p1, k256), B.call(I32MUL, p1, p2)]
        return array_of(outs)

    fn = comp("TestCarryProgram", (("x", Array(128, S8)), ("w", Array(128, S8)), ("r", S32)), Array(8, S32), body)
    program = Program(fn, registered=("w",))
    return program, [len(program.root.body.nodes)]


def test_carries_are_budgeted_and_only_proven_affine_maps_are_free():
    program, ends = _carry_program()
    a = P.analyze(program, census(program, ends))
    assert (a.family, a.affine, a.refused, a.unproven) == (4, 2, {}, {})  # the combine and the product by 256
    # I32Rem, I32Div, U32And, the add on r and the product of two run words; I32Le and SelectI32
    assert a.steps[0]["units"] == {**FREE, "roundings": 5, "compare_select": 2}
    v = judge(program, ends, windows=[P.Window(range(1), 7, 1)])
    assert codes(v) == ["roundings", "compare_select"]
    c = {**FREE, "roundings": "5/7", "compare_select": "2/7"}
    assert judge(program, ends, c=c, windows=[P.Window(range(1), 7, 1)]).ok


def _compare_split_program() -> tuple[Program, list[int]]:
    """Red team attack 1b: the carry t -> (t mod 256, t div 256) by compares and selects alone, 16 rounds of U32Ge against
    T = 256 * 2^b, a SelectI32 between r - T (I32Sub) and r, and one between q + 2^b (I32Add) and q."""
    def body(B, _S, x, w):
        r, q = B.call(_step(64), B.call(prims.ZERO32), x, w), B.call(prims.ZERO32)
        for b in range(15, -1, -1):
            T = B.call(prims.const(32, 256 << b))
            ge = B.call(U32GE, r, T)
            r = B.call(scalar.SelectI32, ge, B.call(I32SUB, r, T), r)
            q = B.call(scalar.SelectI32, ge, B.call(scalar.I32Add, q, B.call(prims.const(32, 1 << b))), q)
        return tuple_of(r, q)

    fn = comp("TestCompareSplit", (("x", Array(64, S8)), ("w", Array(64, S8))), TupleT((S32, S32)), body)
    program = Program(fn, registered=("w",))
    return program, [len(program.root.body.nodes)]


def test_a_carry_by_compares_alone_is_fully_budgeted():
    """Every round's compare and both selects are budgeted. The first round's r - T and 0 + 2^15 are proven affine maps (r
    is the in-window column sum), and every later one reads a select's output, any word, so it may wrap and is budgeted."""
    program, ends = _compare_split_program()
    a = P.analyze(program, census(program, ends))
    assert (a.family, a.affine, a.refused, a.unproven) == (2, 2, {}, {})
    assert a.steps[0]["units"] == {**FREE, "roundings": 30, "compare_select": 48}


def test_a_gather_from_a_constant_table_counts_its_index_bits():
    """Red team: a runtime-indexed gather over constant words is a lookup table of any function of its index, one of 128
    entries or the out-of-range constant: 8 bits."""
    def body(B, _S, i):
        table = array_of([B.call(prims.const(32, (k * k + 7 * k) % 251)) for k in range(128)])
        return B.call(GATHER128, i, table)

    program = Program(comp("TestTable", (("i", S32),), S32, body))
    ends = [len(program.root.body.nodes)]
    assert P.analyze(program, census(program, ends)).steps[0]["units"] == {**FREE, "gathers": 8}


def test_a_one_entry_gather_pays_for_its_out_of_range_test():
    """`GatherBf16x1` of a run index is i == 0 ? col[0] : 0x7FFF, a comparison: its out-of-range outcome is an index bit."""
    program, ends = _one_gate_program(stand_in("GatherBf16x1", (("i", S32), ("col", Array(1, S16))), S16), ("col",))
    assert P.analyze(program, census(program, ends)).steps[0]["units"] == {**FREE, "gathers": 1}


def test_the_red_teams_primitives_are_never_free():
    """Each of the 16 primitives the red team found computing an exact floor, mod, carry, comparison, selection or table read
    is budgeted; I32Mul is affine only with a constant operand (the carry test)."""
    found = ("F32BitsShl23 F32Fabs I32Mul I32Div I32Rem U32And U32Xor F32ToU32RzByte NcpRneClampI7 Ue8m0CeilByte I32Le U32Ge "
             "SelectI32 F32Sat PouwSqGe GatherF32x128").split()
    classes = {f: P.class_of(f + "_v1") for f in found}
    assert set(classes.values()) <= set(P.BUDGETED), classes
    assert P.class_of("I32Add_v1") == P.class_of("I32Sub_v1") == "affine"
    assert P.class_of("BlackwellS8ImmaDot32_v1") == "family" and P.class_of("NoSuchPrimitive_v1") is None


LEAN_REGISTRY = Path(__file__).resolve().parents[4] / "Security/Definitions/Accounting/Gates/Registry.lean"
#: the ids the table lists that Lean's registry doesn't yet: the exact-int8 profile's casts, f64 products and LUT index
NOT_IN_LEAN = {"F32MulWide_v1", "F32Rint_v1", "F32ToS8Rzi_v1", "F64Mul_v1", "F64ToU32RniSat_v1", "LutIndex_v1", "S32ToF32Rn_v1",
               "S8ToF32_v1"}


def lean_class_of() -> tuple[dict[str, str], dict[str, str]]:
    """Lean's `registry` and `families` (`Registry.lean`), each id or prefix with its class in this module's names; the
    table reads are the family `TableRead` whether Lean lists them as one or classes them by their bits (`tableReadBits`)."""
    src = LEAN_REGISTRY.read_text()
    entry = re.compile(r'\("([^"]+)",\s*\.(\w+)(?:\s+\.(\w+))?')
    budgeted = {"rounding": "roundings", "compareSelect": "compare_select", "fold": "folds"}
    cut = src.index("def families")
    ids, families = ({i: budgeted[sub] if k == "budgeted" else {"gather": "gathers"}.get(k, k) for i, k, sub in entry.findall(part)}
                     for part in (src[:cut], src[cut:]))
    if "def tableReadBits" in src:
        families.setdefault("TableRead", "gathers")
    return ids, families


def test_the_table_is_leans_class_of():
    """Lean's `Accounting.Gates.classOf` id for id: each of the registry's ids is in its class here, a family's instances in
    the family's (run and registered inputs are read by their node, not the table); and each id the table lists that Lean
    doesn't is in `NOT_IN_LEAN`. Then the FP dots are roundings (refused for their operand bits, not their class),
    and so are the integer products reduced mod 2^w and the field extractions; the clamps and the 1-bit logic ops are
    compare_selects; PoUW's wrapping dots are family steps (H1's segment dots among them), and the receipt folds' steps
    folds."""
    ids, families = lean_class_of()
    assert len(ids) > 200 and {i: (P.class_of(i), c) for i, c in ids.items() if P.class_of(i) != c} == {}
    assert set(P.CLASSES) - set(ids) <= NOT_IN_LEAN, set(P.CLASSES) - set(ids) - NOT_IN_LEAN
    instance = {"GatherBf16x": "GatherBf16x16_v1", "GatherF32x": "GatherF32x128_v1", "GatherI32x": "GatherI32x64_v1",
                "GatherE4m3x": "GatherE4m3x16_v1", "GatherE2m1x": "GatherE2m1x16_v1", "GatherE8m0x": "GatherE8m0x16_v1",
                "TableRead": "TableRead16x4_v1", "BitAtx": "BitAtx32_v1", "TopPMaskWordx": "TopPMaskWordx151936_v1"}
    listed = families.keys() & instance
    assert {f: P.class_of(instance[f]) for f in listed} == {f: families[f] for f in listed}
    assert set(families) - set(instance) == {"Input", "RegisteredInput"}
    assert {P.class_of(f) for f in ("HopperBF16WgmmaDot16_v1", "BlackwellNvf4OmmaDot64_v1", "DivFullScaleA_v2", "F32Fabs_v1",
                                    "I64ToI32_v1", "BitAtx32_v1", "PouwWindowWord_v1", "I32Mul_v1", "NcpDot16I7_v1",
                                    "PouwSqAcc_v1", "U32And_v1", "S32ClampRest_v1")} == {"roundings"}
    assert {P.class_of(f) for f in ("F32Sat_v1", "S32ClampS8_v1", "S32ClampS8_v2", "And2_v1", "Xor2_v1", "BitNot_v1", "Not_v1",
                                    "PouwSel32_v1")} == {"compare_select"}
    assert P.class_of("Bf16ToF32_v1") == "affine" and P.class_of("Lifted[Bf16ToF32_v1]_v2") == "roundings"
    assert {P.class_of(f) for f in ("PouwSegDot_v1", "PouwSegDot_v2", "PouwSegDot0_v1", "PouwSegDot0_v2", "PouwDot16U8S8_v1",
                                    "BlackwellS8ImmaDot32_v2")} == {"family"}
    assert {P.class_of(f) for f in ("PouwDuoAbsorb_v1", "PouwFx1Absorb_v1", "PouwFx1Absorb_v2")} == {"folds"}
    family = ("GatherBf16x128256_v1", "TableRead16x4_v1", "TopPMaskWordx151936_v1", "Const8[0x7f]_v1",
              "BlackwellU8S8ImmaDot32_v1", "Zero_v1", "GatherBf16x16_v2")
    assert [P.class_of(f) for f in family] == ["gathers"] * 2 + ["compare_select", "const", "family", None, None]
    assert [P.gather(f) for f in family[:3]] == [(0, 17, "range", False), (0, 16, "low", False), None]


def test_the_families_built_on_demand_and_lifts_are_budgeted():
    """The red team's round-2 table for the families served circuits build that Lean doesn't list yet: the sampler's and
    the lifting's gates are budgeted, ⊥ is a constant, and a lift takes its primitive's class, a free one's budgeted."""
    cases = {
        "GatherI32x64_v1": "gathers", "LGatherRow16x5_v1": "gathers", "LGatherRow32x8o7fc00000_v1": "gathers",
        "And3_v1": "compare_select", "Xor17_v1": "compare_select", "LiveCount8_v1": "roundings", "SplitsForSMS4_v1": "roundings",
        "LSelect32_v1": "compare_select", "LFixedSelect3x32_v1": "compare_select", "LIsEOS2_v1": "compare_select",
        "LEnable_v1": "compare_select", "LEmitNext[H=4]_v1": "compare_select", "LBot32_v1": "const",
        "Lifted[F32Add_v1]_v2": "roundings", "Lifted[I32Add_v1]_v2": "roundings", "Lifted[Const32[0x00000000]_v1]_v2":
        "compare_select", "Lifted[BlackwellS8ImmaDot32_v1]_v2": "roundings", "Lifted[LEmitNext[H=4]_v1]_v2{ORD=1}":
        "compare_select", "Lifted[LGatherRow16x5_v1]_v2": "gathers", "S32LowByte_v1": "roundings", "S32ClampS8_v1":
        "compare_select", "Lifted[NoSuchPrimitive_v1]_v2": None, "And1_v1": None, "LSelect_v1": None,
    }
    assert {f: P.class_of(f) for f in cases} == cases
    assert P.gather("LGatherRow16x5_v1") == P.gather("Lifted[LGatherRow16x5_v1]_v2") == (0, 3, "lifted", True)
    assert P.gather("Lifted[GatherBf16x16_v1]_v2") == (0, 5, "range", True)


def test_a_lifted_gather_scans_every_operands_tag():
    """`LGatherRow` is INV when any row is invalid: beside its index bits it pays a compare_select unit, at a constant index
    too, and its rows are operand bits (an unlifted gather's column isn't)."""
    lg = stand_in("LGatherRow16x3", (("idx", Value(33)), ("rows", Array(3, Value(17)))), Value(17))
    program, ends = _one_gate_program(lg, ("idx",))
    v = judge(program, ends, c={**FREE, "compare_select": 1}, windows=[P.Window(range(1), 1, 1)])
    assert v.ok and v.numbers["units"] == {**FREE, "compare_select": 1}
    program, ends = _one_gate_program(lg, ("rows",))
    v = judge(program, ends, c={**FREE, "compare_select": 1, "gathers": 3}, windows=[P.Window(range(1), 1, 1)])
    assert v.ok and v.numbers["units"] == {**FREE, "compare_select": 1, "gathers": 3}
    wide = stand_in("LGatherRow16x6", (("idx", Value(33)), ("rows", Array(6, Value(17)))), Value(17))
    program, ends = _one_gate_program(wide, ("rows",))
    v = judge(program, ends, c={**FREE, "compare_select": 1, "gathers": 4}, windows=[P.Window(range(1), 1, 1)])
    assert [(r["code"], r["primitives"]) for r in v.reasons] == [("operand-bits", {"LGatherRow16x6_v1": 33 + 6 * 17})]


def _gather_program(indices: str) -> tuple[Program, list[int]]:
    """An embedding: two batches of 4 gathers from a registered table of 4 x 16 bf16 words; "same": both by the run index i,
    "two": the second by j, "mapped": one batch whose members each have their own index."""
    def body(B, _S, i, j, ix, table):
        if indices == "mapped":
            return B.batch(GATHER, ix, table, axes=(0, 0))
        first = B.batch(GATHER, i, table, axes=(None, 0))
        second = B.batch(GATHER, i if indices == "same" else j, table, axes=(None, 0))
        return tuple_of(first, second)

    ret = Array(4, S16) if indices == "mapped" else TupleT((Array(4, S16), Array(4, S16)))
    fn = comp("TestGatherProgram", (("i", S32), ("j", S32), ("ix", Array(4, S32)), ("table", Array(4, Array(16, S16)))), ret, body)
    program = Program(fn, registered=("table",))
    return program, [len(program.root.body.nodes)]


@pytest.mark.parametrize("indices,bits", [("same", 5), ("two", 10), ("mapped", 20)])
def test_gathers_count_fresh_index_bits(indices, bits):
    """A gather of one of 16 words or the out-of-range constant reads 5 index bits; one index read by many gathers counts
    once."""
    program, ends = _gather_program(indices)
    assert P.analyze(program, census(program, ends)).steps[0]["units"] == {**FREE, "gathers": bits}


def test_index_bits_of_two_kinds_from_one_index_add():
    """The low 4 bits and the range test of one index tell apart more than either: a TableRead and a gather of it pay both."""
    tr = stand_in("TableRead4x1", (("index", S32), ("table", Array(1, Array(64, S16)))), S16)
    fn = comp("TestTwoKinds", (("i", S32), ("t", Array(1, Array(64, S16))), ("col", Array(16, S16))),
              TupleT((S16, S16, S16)), lambda B, _S, i, t, col: tuple_of(B.call(tr, i, t), B.call(GATHER, i, col),
                                                                         B.call(GATHER, i, col)))
    program = Program(fn, registered=("t", "col"))
    assert P.analyze(program, census(program, [len(program.root.body.nodes)])).steps[0]["units"] == {**FREE, "gathers": 9}


@pytest.mark.parametrize("registered,units", [((), 6), (("col",), 5), (("i",), 0)])
def test_a_gather_among_run_time_values_pays_one_unit_more(registered, units):
    """Lean's `Gates.units`: a run index pays its index bits, and one more when the column holds run-time values too (a
    16-way select among them); at a registered index the gather is wiring, whatever its column holds."""
    program, ends = _one_gate_program(GATHER, registered)
    assert P.analyze(program, census(program, ends)).steps[0]["units"] == {**FREE, "gathers": units}


def test_a_tree_of_gathers_pays_for_each_select_among_run_time_values():
    """Sixteen gathers by i over constant tables, then one by j over their outputs: a table of any function of (i, j). Without
    the selection unit it would cost j's and i's bits alone; the second level selects among run-time words."""
    def body(B, _S, i, j):
        leaves = [B.call(GATHER, i, array_of([B.call(prims.const(16, (k * 31 + r * 7) % 65521)) for r in range(16)]))
                  for k in range(16)]
        return B.call(GATHER, j, array_of(leaves))

    program = Program(comp("TestGatherTree", (("i", S32), ("j", S32)), S16, body))
    assert P.analyze(program, census(program, [len(program.root.body.nodes)])).steps[0]["units"] == {**FREE, "gathers": 11}


def test_a_field_extraction_is_a_rounding_at_any_position():
    """`BitAtx32` reads bit i of a word: a rounding whenever either operand is a run input, so a bit at a constant position
    (red team round 2's soundness item) is budgeted, not free wiring."""
    for registered in (("word",), ("i",)):
        program, ends = _one_gate_program(BITAT, registered)
        assert P.analyze(program, census(program, ends)).steps[0]["units"] == {**FREE, "roundings": 1}
    fn = comp("TestBitAt5", (("word", S32),), BIT, lambda B, _S, word: B.call(BITAT, word, B.call(prims.const(32, 5))))
    program = Program(fn)
    assert P.analyze(program, census(program, [len(program.root.body.nodes)])).steps[0]["units"] == {**FREE, "roundings": 1}


def test_the_bf16_widening_is_a_free_affine_map():
    """`Bf16ToF32` is its bf16 encoding times 2^16 read as signed, on its evaluator, so it never wraps and
    is free (an affine map), its value the operand's range times 2^16; a lift of it is budgeted."""
    w = prims.Bf16ToF32
    for x in [0, 1, 0x3F80, 0x7F80, 0x7FFF, 0x8000, 0xBF80, 0xFFFF] + random.Random(0).sample(range(1 << 16), 64):
        assert s32_value(w.evaluate(x)) == ((x ^ 0x8000) - 0x8000) << 16, hex(x)
    t = P.count(w, (True,))
    assert (t.units, t.affine) == ({}, 1)
    segs, _dyn, _t = P._Pass().call(w, (True,), ((-3, 0x3F80),), ("TestWiden", 0))
    assert segs == ((0, 1, (-3 << 16, 0x3F800000)),)
    segs, _dyn, _t = P._Pass().call(w, (True,), (None,), ("TestWiden", 0))
    assert segs == ((0, 1, (P.S32[0], 0x7FFF0000)),)


@pytest.mark.parametrize("prim,ranges", [
    (prims.Bf16ToF32, [(0, 0xFFFF)]),  # an unsigned reading of the encoding: clipped to s16, half its values would be lost
    (scalar.I32Add, [(0, (1 << 32) - 1), (0, 0)]),  # clipped to s32, the add would look in range though it wraps
    (I32SUB, [(0, 0), (0, (1 << 32) - 1)]),
    (I32MUL, [(1, 1), (0, (1 << 32) - 1)]),
    (scalar.I32Add, [(P.S32[1], P.S32[1]), (1, 1)]),  # within s32, and the sum wraps
])
def test_a_wrapping_affine_map_is_charged_never_clipped(prim, ranges):
    """Red team round 5: an affine map whose operand's range leaves its width's signed reading, or whose result leaves s32,
    isn't proven, so it is budgeted (a rounding) rather than free over a range clipped to s16 or s32."""
    t = P.count(prim, (True,) * len(ranges), ranges)
    assert (t.units, t.affine) == ({"roundings": 1}, 0)


def test_the_lut_softmax_gather_reads_8_index_bits_and_lies_in_its_tables_hull():
    """`Exp2Lut127` (the exact-int8 profile's LUT softmax) is a gather of a constant 256-entry table by an 8-bit index: 8 fresh
    index bits and no selection unit (its column is constant), and its value lies in the hull of rint(127 * 2^(-t/16)) and 0
    (Lean's `hull0`), so the softmax's sum of them by I32Add is a proven affine chain. The table is its evaluator's: its 256
    little-endian u32 words hash to vLLM's `engine.int8_attention.LUT`'s sha256."""
    table = np.rint(127.0 * np.exp2(-np.arange(256) / 16.0)).astype(int)
    assert P.EXP2_LUT127_TABLE == tuple(table.tolist())
    assert hashlib.sha256(np.asarray(P.EXP2_LUT127_TABLE, dtype="<u4").tobytes()).hexdigest() == EXP2_LUT127_SHA256
    assert P.EXP2_LUT127 == (min(0, int(table.min())), max(0, int(table.max()))) == (0, 127)
    assert P.gather("Exp2Lut127_v1") == (0, 8, "low", False) and P.class_of("Exp2Lut127_v1") == "gathers"

    def body(B, _S, t):
        e = [B.call(EXP2LUT, t[j]) for j in range(4)]
        total = e[0]
        for j in range(1, 4):
            total = B.call(scalar.I32Add, total, e[j])
        return total

    program = Program(comp("TestLutSum", (("t", Array(4, S8)),), S32, body))
    a = P.analyze(program, census(program, [len(program.root.body.nodes)]))
    assert (a.steps[0]["units"], a.affine, a.refused) == ({**FREE, "gathers": 32}, 3, {})


def test_the_exact_int8_profiles_primitives_are_listed():
    """The exact-int8 profile's primitives (vLLM's `int8_exact`), which Lean's registry doesn't list yet: its casts, f64
    products and `LutIndex` (min(floor(max(d, 0) a / 2^32), 255)) are roundings, its LUT a gather."""
    ids = "F32Rint F32ToS8Rzi S8ToF32 F32MulWide F64Mul F64ToU32RniSat LutIndex".split()
    assert {P.class_of(f + "_v1") for f in ids} == {"roundings"}


@pytest.mark.parametrize("prim,registered,bits,units", [
    (NCPDOT, ("b",), 32 + 2 * 16 * 8, {"roundings": 1}),
    (SQACC, (), 576 + 32, {"roundings": 1}),
    (F64MUL, (), 128, {"roundings": 1}),
    (TABLE, ("table",), None, {"gathers": 8}),
])
def test_a_gate_past_B_operand_bits_is_refused(prim, registered, bits, units):
    """Restricted B (Lean's `OperandBits`): an integer dot outside the family (288 operand bits) and PoUW's 576-bit square
    accumulator are refused at B = 128; the exact-int8 profile's f64 product reads 128 bits, within it; a table read of
    64-bit entries is within any B (a gather's column isn't operand bits: its value is one of them). At a B they fit, each is
    charged its class's unit (a gather its index bits)."""
    program, ends = _one_gate_program(prim, registered)
    one = [P.Window(range(1), 1, 1)]
    v = judge(program, ends, c={**FREE, **units}, windows=one)
    listed = [("operand-bits", {prim.id: bits})] if bits is not None and bits > P.OPERAND_BITS else []
    assert [(r["code"], r["primitives"]) for r in v.reasons] == listed, v.reasons
    assert v.numbers["units"] == {**FREE, **units}
    if bits is not None:
        assert judge(program, ends, c={**FREE, **units}, windows=one, B=bits).ok
        assert codes(judge(program, ends, c={**FREE, **units}, windows=one, B=bits - 1)) == ["operand-bits"]
    else:
        assert judge(program, ends, c={**FREE, **units}, windows=one, B=1).ok


def test_the_u8_s8_step_is_proven_over_its_unsigned_operand():
    """The u8 x s8 sibling's 32 products lie in 32 * [255 * -128, 255 * 127]: from C >= -2^31 + 32 * 32640 its D stays in s32.
    A constant lane 0xff is 255 to it, not -1."""
    lo = P.S32[0] + 32 * 255 * 128
    assert P.count(U8S8, (True, True, False), [(lo, 0), None, None]).family == 1
    assert P.count(U8S8, (True, True, False), [(lo - 1, 0), None, None]).unproven == {U8S8.id: 1}
    assert P.count(U8S8, (True, True, False), [(P.S32[0] + 32 * 128, 0), (-1, -1), None]).unproven == {U8S8.id: 1}


RZBYTE = stand_in("F32ToU32RzByte", (("x", S32),), S8)


def _quantizer(steps: str) -> CompositeDefinition:
    """P's quantizer (the red team's PQUANT): the low byte of trunc(min(max(p, 0.0), 127.0)), or with only one of the clamps."""
    def body(B, _S, p):
        if "max" in steps:
            p = B.call(scalar.F32Fmaxf, p, B.call(prims.const(32, 0)))
        if "min" in steps:
            p = B.call(scalar.F32Fminf, p, B.call(prims.const(32, 0x42FE0000)))
        return B.call(RZBYTE, p)
    return comp(f"TestPQuant{steps}", (("p", S32),), S8, body)


def _pv(K: int, steps: str = "maxmin") -> CompositeDefinition:
    """P . V over K tokens: P quantized from run f32 probabilities, V any s8, as K/32 chained s8 steps from zero."""
    def body(B, _S, p, v):
        q = B.batch(_quantizer(steps), p, axes=(0,))
        acc = B.call(prims.ZERO32)
        for j in range(K // 32):
            acc = B.call(IMMA, acc, q[32 * j:32 * (j + 1)], v[32 * j:32 * (j + 1)])
        return acc
    return comp(f"TestPV{K}{steps}", (("p", Array(K, S32)), ("v", Array(K, S8))), S32, body)


@pytest.mark.parametrize("K,steps,unproven", [
    (132_096, "maxmin", 0), (132_128, "maxmin", 1), (132_096, "min", 0), (131_040, "max", 0), (131_072, "max", 1)])
def test_ps_quantizer_bounds_it_to_0_127_so_pv_stays_in_window_to_132104_tokens(K, steps, unproven):
    """Red team round 2's over-strict case: with the range rules P lies in [0, 127] (fminf alone suffices: a negative or NaN
    truncates to 0), so 32 * 127 * 128 a step reaches -2^31 past 132,104 tokens; P from fmaxf alone is any s8 word, whose
    128 * 128 reaches 2^31 at 131,072."""
    t = P.count(_pv(K, steps), (True, True))
    assert (t.family, t.unproven) == (K // 32 - unproven, {IMMA.id: unproven} if unproven else {})
    assert t.units == {"roundings": K, "compare_select": K * (("max" in steps) + ("min" in steps))}


LUT_P = comp("TestLutP", (("t", S8),), S8, lambda B, _S, t: B.call(S32LowByte, B.call(EXP2LUT, t)))


def _lut_pv(K: int) -> CompositeDefinition:
    """Lean's `lutPv`: P . V' over K keys, P = S32LowByte(Exp2Lut127(t)) of a run index t, V' any s8, as K/32 chained s8 steps
    from zero."""
    def body(B, _S, t, v):
        p = B.batch(LUT_P, t, axes=(0,))
        acc = B.call(prims.ZERO32)
        for j in range(K // 32):
            acc = B.call(IMMA, acc, p[32 * j:32 * (j + 1)], v[32 * j:32 * (j + 1)])
        return acc
    return comp(f"TestLutPV{K}", (("t", Array(K, S8)), ("v", Array(K, S8))), S32, body)


@pytest.mark.parametrize("K,entry_max,unproven", [(132_096, None, 0), (132_128, None, 1), (131_040, 128, 0), (131_072, 128, 1)])
def test_lut_ps_low_byte_keeps_the_tables_hull_so_pv_stays_in_window_to_132096_keys(K, entry_max, unproven, monkeypatch):
    """Red team round 5: P = S32LowByte(Exp2Lut127(t)) lies in the table's hull, [0, 127], since a word in s8 is its own low
    byte, so 32 * 128 * 127 a step reaches -2^31 only past K = 132,096 (Lean's `LutPvWindow`). Read at its type, [-128, 127],
    as with one table entry of 128 (whose low byte is -128), 128 * 128 reaches 2^31 at 131,072."""
    if entry_max is not None:
        monkeypatch.setitem(P.RANGES, "Exp2Lut127_v1", lambda _t: (0, entry_max))
    t = P.count(_lut_pv(K), (True, True))
    assert (t.family, t.unproven) == (K // 32 - unproven, {IMMA.id: unproven} if unproven else {})
    assert t.units == {"gathers": 8 * K, "roundings": K}


def test_the_low_byte_rule_holds_on_the_evaluator():
    """`S32LowByte` of a word in s8 is the word (its s8 code read signed), and 128's is -128; a range past s8 is any s8 code."""
    assert all(P._signed(8, S32LowByte.evaluate(s32_word(y))) == y for y in range(-128, 128))
    assert P._signed(8, S32LowByte.evaluate(128)) == -128
    rule = P.RANGES["S32LowByte_v1"]
    assert P.RANGES["S32LowByte_v2"] is rule
    assert (rule((0, 127)), rule((-128, 127)), rule((-128, 128)), rule(None)) == ((0, 127), (-128, 127), None, None)


def test_the_range_rules_hold_on_the_evaluator():
    """`fmaxf` and `fminf` over sampled operand ranges (+-0, subnormals, +-inf, NaNs of both signs) give values inside the
    rule's range, as signed bit patterns."""
    rng = random.Random(0)
    special = [0, 0x80000000, 1, 0x007FFFFF, 0x3F800000, 0x42FE0000, 0x7F7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000,
               0x7F800001, 0xBF800000]
    ranges = [None, (0, 0), (0, 0x42FE0000), (0x3F800000, 0x7F800000), (0x7F800000, 0x7F800000), (P.S32[0], -1),
              (0x7F800001, 0x7FFFFFFF), (-5, 5), (P.S32[0], 0x42FE0000)]

    def draws(r):
        if r is None:
            return [v - (1 << 32) if v >> 31 else v for v in special] + [rng.randrange(*P.S32) for _ in range(40)]
        return [r[0], r[1]] + [s for s in (v - (1 << 32) if v >> 31 else v for v in special) if r[0] <= s <= r[1]] + [
            rng.randint(*r) for _ in range(40)]

    for prim in (scalar.F32Fmaxf, scalar.F32Fminf):
        rule = P.RANGES[prim.id]
        for a in ranges:
            for b in ranges:
                out = rule(a, b)
                if out is None:
                    continue
                for x in draws(a):
                    for y in draws(b):
                        z = prim.evaluate(x & 0xFFFFFFFF, y & 0xFFFFFFFF)
                        assert out[0] <= s32_value(z) <= out[1], (prim.id, a, b, hex(x), hex(y), hex(z), out)


def test_roundings_are_budgeted_per_certified_byte_read():
    program, ends = int8_program(epilogue=True)
    a = P.analyze(program, census(program, ends))
    assert [s["units"]["roundings"] for s in a.steps] == [2, 2, 2]  # the epilogue's FMA and add
    ws = one_step_windows()
    v = judge(program, ends, windows=ws)
    assert codes(v) == ["roundings"] * 3 and v.reasons[0]["c"] == "0"
    assert judge(program, ends, c={**FREE, "roundings": "1/32"}, windows=ws).ok
    assert codes(judge(program, ends, c={**FREE, "roundings": "1/33"}, windows=ws)) == ["roundings"] * 3


def test_inputs_lifetimes_and_coverage():
    """R_in is a rate: a window admits R_in times its duration of first-read run-input bytes (64 a step here)."""
    program, ends = int8_program()
    ws = one_step_windows()
    assert codes(judge(program, ends, R_in=63, windows=ws)) == ["inputs"] * 3
    assert judge(program, ends, R_in=32, windows=[P.Window(range(s, s + 1), 64, 2) for s in range(3)]).ok
    assert judge(program, ends, R_in=P.R_IN, windows=[P.Window(range(3), 64, "1/500")]).ok  # 192 bytes <= 200 at 100 KB/s
    assert codes(judge(program, ends, R_in=P.R_IN, windows=[P.Window(range(3), 64, "1/1000")])) == ["inputs"]
    assert codes(judge(program, ends, windows=ws[:2])) == ["uncovered"]
    program, ends = int8_program(reread=(2, 0))
    v = judge(program, ends, L=1, windows=ws)
    assert codes(v) == ["lifetime"] and v.reasons[0]["max_lifetime"] == 3  # x[0] arrives before step 0, read again in step 2
    v = judge(program, ends, L=None, windows=ws)
    assert v.ok and v.numbers["max_lifetime"] == 3


def test_check_refuses_bad_windows_c_and_another_programs_census():
    program, ends = int8_program()
    cen = census(program, ends)
    for ws in ([P.Window(range(0), 1, 1)], [P.Window(range(0, 4), 1, 1)], [P.Window(range(0, 3, 2), 1, 1)],
               [P.Window(range(3), -1, 1)], [P.Window(range(3), 1, 0)]):
        with pytest.raises(ValueError):
            P.check(program, cen, c=FREE, windows=ws)
    with pytest.raises(ValueError):
        P.check(program, cen, c={"roundings": 0}, windows=[P.Window(range(3), 1, 1)])
    for B in (0, -1, True, 128.0):
        with pytest.raises(ValueError):
            P.check(program, cen, c=FREE, windows=[P.Window(range(3), 1, 1)], B=B)
    other, other_ends = int8_program(S=2)
    with pytest.raises(ValueError):
        P.check(program, census(other, other_ends), c=FREE, windows=[P.Window(range(2), 1, 1)])


def test_caps_rates_and_durations_are_exact_and_a_float_is_refused():
    """Red team round 5: `Fraction(0.1)` is 0.1's binary value, a little more than 1/10, in the attacker's favour. c, R_in and
    a window's seconds are taken exactly (an int, a Fraction or a str), a float is refused with its exact spelling in the
    message, and a window's reads is an integer."""
    program, ends = int8_program(epilogue=True)
    a = P.analyze(program, census(program, ends))
    one = [P.Window(range(3), 192, 1)]
    for cap in (Fraction(1, 32), "1/32", "0.03125"):
        assert P.judge(a, c={**FREE, "roundings": cap}, windows=one).ok
    assert codes(P.judge(a, c={**FREE, "roundings": Fraction(1, 33)}, windows=one)) == ["roundings"]
    bad = [dict(c={**FREE, "roundings": 0.1}, windows=one), dict(c={**FREE, "gathers": np.float64(0.5)}, windows=one),
           dict(c=FREE, windows=one, R_in=1e5), dict(c=FREE, windows=[P.Window(range(3), 192, 0.5)])]
    for kw in bad:
        with pytest.raises(ValueError, match=r"the float .*: give it exactly, as Fraction\('"):
            P.judge(a, **kw)
    for kw in (dict(c={**FREE, "folds": None}, windows=one), dict(c=FREE, windows=[P.Window(range(3), 192.0, 1)])):
        with pytest.raises(ValueError):
            P.judge(a, **kw)


def test_count_reads_a_definition_with_given_ranges():
    st = _step(64)
    assert P.count(st, (True, True, False)).unproven == {"BlackwellS8ImmaDot32_v1": 2}  # an accumulator of any word
    t = P.count(st, (True, True, False), [(0, 0), None, None])
    assert (t.family, t.macs, t.unproven) == (2, 64, {})
    assert P.count(EPILOGUE, (False,)).units == {} and P.count(EPILOGUE, (True,)).units == {"roundings": 2}


def test_the_table_is_pinned():
    """Changing the table is a new VERSION and digest: a verdict names the one it was judged under."""
    assert json.loads(canonical_json(P.spec())) == P.spec()
    assert P.digest() == PINNED, P.digest()


PINNED = ("9d2ce91d019e29a91a9f180cdfa204a9bfeda9c71a755d60c795564330c4963d"
          "a872af5a8c69326f1e319fcc9b8d7cc2b4fea1c7f60c714f23ec1571e1173f8f")
