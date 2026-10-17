"""`verity.ml.circuits.stacking` (property/v1) on small synthetic Programs with their census/v1: an exact int8 Program passes,
and a clamping, wrapping, FP-smuggling, reinterpreting, carry-computing, gather-heavy or input-holding one fails with its
reason, or has its carries budgeted. The red team's attacks (benchmarks/property_redteam on cursor/property-redteam-6cd3) are
cases here."""
from __future__ import annotations

import json

import pytest

from verity.ml.circuits import stacking as P
from verity.ml.circuits.census import census
from verity.ml.circuits.codec import canonical_json
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition
from verity.ml.circuits.ops import fp32, prims, scalar
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Coll, Strided, array_of, concat, tuple_of
from verity.ml.circuits.silicon.integer import BLACKWELL_SM120_S8_M16N8K32, s32_value, s32_word
from verity.ml.circuits.types import Array, Tuple as TupleT, Value

S8, S16, S32, BIT = Value(8), Value(16), Value(32), Value(1)
IMMA = prims.BlackwellS8ImmaDot32
QMMA = prims.BlackwellE4m3QmmaDot32
FREE = {"roundings": 0, "compare_select": 0, "gathers": 0}


def stand_in(name: str, params, ret=S32) -> PrimitiveDefinition:
    """A registry primitive's id with a dummy evaluator, for one this unit can't import (vLLM's, PoUW's): the table keys on ids."""
    return PrimitiveDefinition(name, 1, tuple(params), ret, lambda *v: 0, register=False)


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


def comp(name: str, params, ret, body) -> CompositeDefinition:
    return CompositeDefinition(name, 1, (), lambda _S: (tuple(params), ret), body, register=False)


def _step(K: int, mma=IMMA) -> CompositeDefinition:
    """acc + x . w over K s8 lanes, as K/32 chained steps of `mma`."""
    def body(B, _S, acc, x, w):
        for j in range(K // 32):
            acc = B.call(mma, acc, x[32 * j:32 * (j + 1)], w[32 * j:32 * (j + 1)])
        return acc
    return comp(f"TestStep{K}{mma.name}", (("acc", S32), ("x", Array(K, S8)), ("w", Array(K, S8))), S32, body)


EPILOGUE = comp("TestEpilogue", (("acc", S32),), S32, lambda B, _S, acc: B.call(fp32.F32Add, B.call(fp32.F32Fma, acc, acc, acc), acc))


def int8_program(S: int = 3, K: int = 64, *, mmas=None, acc0="zero", epilogue=False, reread=None) -> tuple[Program, list[int]]:
    """S steps: step s reads its run input x[s] and the registered w, and starts from step s-1's accumulator (dynamic memory);
    step 0 from zero, or from the run input `c` when acc0 = "input". `epilogue`: each step also outputs `EPILOGUE` of its
    accumulator. `reread`: (step, earlier step) whose x that step reads again. Returns the Program and its step ends."""
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

    fn = comp("TestInt8Program", params, Array(S, S32), body)
    return Program(fn, registered=("w",)), ends


def judge(program, ends, *, c=FREE, R_in=10 ** 9, L=None, windows=None):
    cen = census(program, ends)
    ws = windows if windows is not None else [P.Window(range(len(ends)), 10 ** 9, 1)]
    return P.check(program, cen, c=c, windows=ws, R_in=R_in, L=L)


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
    assert (v.numbers["family_steps"], v.numbers["macs"]) == (6, 192)


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


def test_a_wrapping_step_and_fp_mma_smuggling_are_refused():
    """A step the table doesn't list is refused, and so is an FP tensor-core step; the step after either starts from an
    accumulator of any word, so it isn't proven either."""
    for mma, code in ((WRAP, "unlisted"), (QMMA, "fp-mma")):
        program, ends = int8_program(mmas=[IMMA, mma, IMMA])
        v = judge(program, ends)
        assert codes(v) == [code, "s8-window"], v.reasons
        assert v.reasons[0]["primitives"] == {mma.id: 2}
        assert v.reasons[1]["steps"] == {"BlackwellS8ImmaDot32_v1": 2}  # the third step's two: the second starts from the first
        assert v.numbers["family_steps"] == 2


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


@pytest.mark.parametrize("how,where", [("broadcast", "Root[TestReinterpretBroadcast_v1]_v1"),
                                       ("affine", "Root[TestReinterpretAffine_v1]_v1"), ("ret", "TestNarrow_v1"), ("mixed", None)])
def test_a_word_read_at_another_width_is_refused(how, where):
    program, ends = _reinterpreting_program(how)
    v = judge(program, ends)
    if where is None:
        assert v.ok and v.numbers["family_steps"] == 2, v.reasons
        return
    assert codes(v) == ["reinterpretation"] and v.reasons[0]["definitions"] == {where: 1}, v.reasons


def _carry_program() -> tuple[Program, list[int]]:
    """Red team attack 1 in miniature: a column sum by an IMMA chain from zero (in window), then its carry t -> (t mod 256,
    t div 256) by I32Rem and I32Div, a comparison carry by I32Le and SelectI32, a bit carry by U32And, a split-K combine of two
    in-window partials by I32Add (free), the same add on a run word (it may wrap: budgeted), and products by a constant (free)
    and of two run words (budgeted)."""
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
    assert a.steps[0]["units"] == {"roundings": 5, "compare_select": 2, "gathers": 0}
    v = judge(program, ends, windows=[P.Window(range(1), 7, 1)])
    assert codes(v) == ["roundings", "compare_select"]
    assert judge(program, ends, c={"roundings": "5/7", "compare_select": "2/7", "gathers": 0}, windows=[P.Window(range(1), 7, 1)]).ok


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
    assert a.steps[0]["units"] == {"roundings": 30, "compare_select": 48, "gathers": 0}


def test_a_gather_from_a_constant_table_counts_its_index_bits():
    """Red team: a runtime-indexed gather over constant words is a lookup table of any function of its 7-bit index."""
    def body(B, _S, i):
        table = array_of([B.call(prims.const(32, (k * k + 7 * k) % 251)) for k in range(128)])
        return B.call(GATHER128, i, table)

    program = Program(comp("TestTable", (("i", S32),), S32, body))
    ends = [len(program.root.body.nodes)]
    assert P.analyze(program, census(program, ends)).steps[0]["units"] == {"roundings": 0, "compare_select": 0, "gathers": 7}


def test_the_red_teams_primitives_are_never_free():
    """Each of the 16 primitives the red team found computing an exact floor, mod, carry, comparison, selection or table read
    is budgeted or refused; I32Mul is affine only with a constant operand (the test above)."""
    found = ("F32BitsShl23 F32Fabs I32Mul I32Div I32Rem U32And U32Xor F32ToU32RzByte NcpRneClampI7 Ue8m0CeilByte I32Le U32Ge "
             "SelectI32 F32Sat PouwSqGe GatherF32x128").split()
    classes = {f: P.class_of(f + "_v1") for f in found}
    assert classes.pop("I32Mul") == "affine"
    assert set(classes.values()) <= set(P.BUDGETED), classes
    assert P.class_of("I32Add_v1") == P.class_of("I32Sub_v1") == "affine"
    assert P.class_of("BlackwellS8ImmaDot32_v1") == "family" and P.class_of("NoSuchPrimitive_v1") is None


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


@pytest.mark.parametrize("indices,bits", [("same", 4), ("two", 8), ("mapped", 16)])
def test_gathers_count_fresh_index_bits(indices, bits):
    """A gather of one of 16 words reads 4 index bits; one index read by many gathers counts once."""
    program, ends = _gather_program(indices)
    assert P.analyze(program, census(program, ends)).steps[0]["units"] == {"roundings": 0, "compare_select": 0, "gathers": bits}


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
    other, other_ends = int8_program(S=2)
    with pytest.raises(ValueError):
        P.check(program, census(other, other_ends), c=FREE, windows=[P.Window(range(2), 1, 1)])


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


PINNED = ("a1037fbadfb9b1d643bd16dd4b68fc9098017a6962accc750a42b405337ebe6f"
          "ff5077210aae7989a52a062e6f8eb2c7cfae6d09fb05f1b680c8088f9bb73f9b")
