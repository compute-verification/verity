"""The circuit model (`Accounting.Gates`: `intervals`, `census`, `classOf`, `units`) held to the Python it states.

Security/Proofs/Accounting/Gates/Rows.lean carries rows the kernel evaluates the model on (`s8Rows`, `censusRows`,
`classRows`, `familyRows`, `tableReadRows`, `unitsRows`, `tableUnitsRows`), and each test here recomputes its rows from the
Python: `s8.no_wrap`'s `T_t` and `D_g`, census/v1 of the same Program (with property/v1's units of it), property/v1's
`class_of` and `gather` with circuits' input ids, and property/v1's units of a Program built gate for gate.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.ml.circuits import stacking as P
from verity.ml.circuits.census import census
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.ops import fp32, prims, scalar
from verity.ml.circuits.ops import s8 as S8
from verity.ml.circuits.program import Program, input_id
from verity.ml.circuits.refs import array_of, tuple_of
from verity.ml.circuits.types import Array, Tuple as TupleT, Value

ROWS = Path(__file__).resolve().parents[4] / "Security" / "Proofs" / "Accounting" / "Gates" / "Rows.lean"
S8T, S16, S32, BIT = Value(8), Value(16), Value(32), Value(1)
#: the s8 family's word steps, s8·s8 and u8·s8
DOT = {False: prims.BlackwellS8ImmaDot32, True: prims.BlackwellU8S8ImmaDot32}
#: ids whose family prefix Lean's `classOf` classes and property/v1's full patterns refuse: a `_v2` of a gather, an input,
#: a field extraction and a table read, and a gather with no column size. Lean is the looser side here.
LEAN_LOOSER = {"GatherBf16x16_v2", "Input8_v2", "BitAtx32_v2", "GatherF32xQ_v1", "TableRead16x4_v2"}


def lean_rows(name: str) -> list:
    """The rows of `def name` in Rows.lean as Python values: `none` as None, `some r` as r, `true`/`false` as booleans."""
    block = ROWS.read_text().split(f"\ndef {name} ", 1)[1]
    body = block[block.index(":= [") + 3:block.index("]\n\n") + 1]
    for lean, py in ((r"\bsome ", ""), (r"\bnone\b", "None"), (r"\btrue\b", "True"), (r"\bfalse\b", "False")):
        body = re.sub(lean, py, body)
    rows = ast.literal_eval(body)
    assert rows, f"no rows read from {name}"
    return rows


def comp(name: str, params, ret, body) -> CompositeDefinition:
    return CompositeDefinition(name, 1, (), lambda _S: (tuple(params), ret), body, register=False)


# -- intervals against s8.no_wrap -----------------------------------------------------------------------------------------


def binding(K, L, C, u8, A, B, tu8, tail) -> SimpleNamespace:
    """`S8Dot_v1`'s statics for a row: an edited arm with byte limbs when it has a tail, else the plain arm."""
    return SimpleNamespace(K=K, L=L, T0="zero" if C is None else "input", C=C, DOT=DOT[u8], A=A, B=B,
                           TAIL=tuple((n, "chain" if chain else "fresh", ga, gb) for n, chain, ga, gb in tail),
                           TDOT=DOT[tu8] if tail else None, NOISE="weight" if tail else "none",
                           LIMBS="bytes" if tail else "none", COMMIT="final" if L == K else "prefix")


def py_words(*row) -> list[tuple[int, int]] | None:
    try:
        return list(S8.no_wrap(binding(*row)).values())
    except ValueError:
        return None


def test_intervals_is_no_wrap():
    """`s8Interval_s8Rows`: Lean's derived intervals of an `S8Dot_v1`'s words, or its refusal, are `no_wrap`'s."""
    rows = lean_rows("s8Rows")
    assert len(rows) >= 13 and any(r[-1] is None for r in rows) and any(r[-1] is not None for r in rows)
    assert [(r[:-1], py_words(*r[:-1])) for r in rows] == [(r[:-1], r[-1]) for r in rows]


# -- census against census/v1 ---------------------------------------------------------------------------------------------

#: a code's interval by its format, and a word's from its producer
FMT = {False: (-8, 8), True: (0, 8)}
C_IN = (-(1 << 20), 1 << 20)


def census_program(matmul, N, K, L, acc, u8, tu8, tail, regB, limbs) -> tuple[Program, list[int]]:
    """Rows.lean's `censusDag` as a Program: `N` `S8Dot_v1` coordinates (each its own step) or one `S8Matmul_v1` of `N` rows,
    then a `LimbSplit_v1` (FORM "byteclamp" when clamp, else "bytes") of each `ys` word, in one more step. Returns the
    Program and its step ends."""
    m = 32 * sum(n for n, _ in tail)
    S = binding(K, L, C_IN if acc else None, u8, FMT[u8], FMT[False], tu8,
                [(n, chain, FMT[tu8], FMT[False]) for n, chain in tail])
    statics = vars(S)
    dot = bind(S8.S8Matmul, N=N, **statics) if matmul else bind(S8.S8Dot, **statics)
    splits = [bind(S8.LimbSplit, FORM="byteclamp" if clamp else "bytes", N=n,
                   Y=(-60000, 60000) if clamp else (-(1 << (8 * n - 2)), 1 << (8 * n - 2))) for clamp, n in limbs]
    params = ([("acc", Array(N, S32))] if acc else []) + [("a", Array(K, S8T)), ("b", Array(N, Array(K, S8T)))]
    params += [("ta", Array(m, S8T)), ("tb", Array(N, Array(m, S8T)))] if m else []
    params += [("ys", Array(len(limbs), S32))] if limbs else []
    names = [n for n, _ in params]
    rets = [dot.signature()[1]] * (1 if matmul else N) + [s.signature()[1] for s in splits]
    ends: list[int] = []

    def body(B, _S, *xs):
        x = dict(zip(names, xs))
        ends.clear()
        outs = []
        for n in range(1 if matmul else N):
            row = (lambda v: v) if matmul else (lambda v: v[n])
            args = ([row(x["acc"])] if acc else []) + [x["a"], row(x["b"])] + ([x["ta"], row(x["tb"])] if m else [])
            outs.append(B.call(dot, *args))
            ends.append(len(B.nodes))
        for i, s in enumerate(splits):
            outs.append(B.call(s, x["ys"][i]))
        if splits:
            ends.append(len(B.nodes))
        return tuple_of(*outs)

    fn = comp(f"TestCensus{'Matmul' if matmul else 'Dot'}", params, TupleT(tuple(rets)), body)
    program = Program(fn, registered=tuple(n for n in ("b", "tb") if regB and n in names))
    return program, ends


def census_counts(program: Program, ends: list[int]) -> tuple:
    """census/v1's counts as Rows.lean's `censusOf` reads them, with property/v1's units summed over the steps."""
    doc = census(program, ends)
    cls, tot = doc["classes"].values(), doc["totals"]
    inst = lambda p: sum(c["instances"] for c in cls if p(c))  # noqa: E731
    units = [s["units"] for s in P.analyze(program, doc).steps]
    return (inst(lambda c: c["family"] in P.FAMILY),
            tot["served_macs"] + tot["tail_atoms"], tot["served_macs"], tot["tail_atoms"],
            inst(lambda c: P.CONST_ID.fullmatch(c["family"]) or c["family"] in ("I32Add_v1", "I32Sub_v1", "Bf16ToF32_v1")),
            sum(c["instances"] * S8.LimbSplit.carries(c["statics"]) for c in cls if c["family"] == "LimbSplit_v1"),
            sum(u["roundings"] for u in units), sum(u["compare_select"] for u in units),
            tot["inputs"]["run"]["bits"], tot["inputs"]["registration"]["bits"],
            list(zip(doc["steps"]["served_macs"], doc["steps"]["tail_atoms"])))


def test_census_is_census_v1():
    """`censusOf_censusRows`: Lean's census of the s8 family's coordinates and limb splits is census/v1's of the same Program
    (served MACs and tail MACs apart, in all and per step) and its units property/v1's."""
    rows = lean_rows("censusRows")
    assert len(rows) >= 4
    assert [(r[:-1], census_counts(*census_program(*r[:-1]))) for r in rows] == [
        (r[:-1], tuple(r[-1][:-1]) + (list(r[-1][-1]),)) for r in rows]


# -- classOf against property/v1's table ----------------------------------------------------------------------------------


def py_class(fid: str) -> tuple[str, int]:
    """property/v1's class of `fid` as Rows.lean's `classTag` names it: an input's binding (circuits' `input_id`), an
    s8-family step's formats and k, a gather's index-bit kind (with n for the `low` kind), or the class (`unlisted`)."""
    if (inp := input_id(fid)) is not None:
        return inp[0], 0
    c = P.class_of(fid)
    if c == "family":
        st = P.FAMILY[fid]
        return ("family u8" if st.a == P.U8 else "family s8"), st.k
    if c == "gathers" and (g := P.gather(fid)) is not None and not g[3]:
        return g[2], g[1] if g[2] == "low" else 0
    return c or "unlisted", 0


@pytest.mark.parametrize("rows", ["classRows", "familyRows", "tableReadRows"])
def test_class_of_is_property_v1s(rows):
    """`classOf_classRows`, `classOf_familyRows`, `classOf_tableReadRows`: Lean's class of each id is property/v1's, but
    for the ids in `LEAN_LOOSER`, which property/v1 refuses and Lean's family prefixes class."""
    lean = {fid: (tag, k) for fid, tag, k in lean_rows(rows)}
    py = {fid: py_class(fid) for fid in lean}
    looser = lean.keys() & LEAN_LOOSER
    assert {f: py[f] for f in looser} == dict.fromkeys(looser, ("unlisted", 0))
    assert all(lean[f][0] != "unlisted" for f in looser)
    assert {f: py[f] for f in lean.keys() - looser} == {f: lean[f] for f in lean.keys() - looser}


# -- units against property/v1 --------------------------------------------------------------------------------------------


def stand_in(name: str, params, ret=S32) -> PrimitiveDefinition:
    return PrimitiveDefinition(name, 1, tuple(params), ret, lambda *v: 0, register=False)


#: the primitives the units rows call, by id: circuits' own, and stand-ins for ids this unit can't import
PRIMS = {d.id: d for d in (fp32.F32Add, scalar.I32Le, scalar.SelectI32, scalar.I32Add, prims.Bf16ToF32,
                           prims.BlackwellS8ImmaDot32, prims.ZERO32,
                           stand_in("PouwDuoAbsorb", (("s", S32), ("w", S32))),
                           stand_in("GatherBf16x16", (("i", S32), ("col", Array(16, S16))), S16),
                           stand_in("GatherBf16x1", (("i", S32), ("col", Array(1, S16))), S16),
                           stand_in("TableRead4x16", (("index", S32), ("table", Array(16, S16))), S16),
                           stand_in("BitAtx32", (("word", S32), ("i", S32)), BIT),
                           stand_in("TopPMaskWordx2", (("a", S32), ("b", S32))),
                           stand_in("Zero", (("a", S32),)))}


def units_program(gates: list[tuple[str, list[int]]]) -> tuple[Program, list[int]]:
    """Rows.lean's `dagOf` as a Program: each input gate a scalar parameter (registered for `RegisteredInput`), and each
    other gate a call of its primitive on the gates it names, an array parameter taking as many of them as it holds."""
    params = [(f"x{i}", Value(input_id(fid)[1])) for i, (fid, _) in enumerate(gates) if input_id(fid)]
    registered = tuple(f"x{i}" for i, (fid, _) in enumerate(gates) if (input_id(fid) or ("",))[0] == "registration")

    def body(B, _S, *xs):
        vals, ins = [], iter(xs)
        for fid, args in gates:
            if input_id(fid):
                vals.append(next(ins))
                continue
            fn, at, call = PRIMS[fid], 0, []
            for _n, t in fn.params:
                k = t.n if isinstance(t, Array) else 1
                part = [vals[j] for j in args[at:at + k]]
                call.append(array_of(part) if isinstance(t, Array) else part[0])
                at += k
            assert at == len(args), (fid, args)
            vals.append(B.call(fn, *call))
        return vals[-1]

    ret = PRIMS[gates[-1][0]].ret if not input_id(gates[-1][0]) else Value(input_id(gates[-1][0])[1])
    program = Program(comp(f"TestUnits{len(gates)}", params, ret, body), registered=registered)
    return program, [len(program.root.body.nodes)]


def py_units(gates) -> tuple[int, int, int, int] | None:
    program, ends = units_program(gates)
    a = P.analyze(program, census(program, ends))
    if "unlisted" in a.refused:
        return None
    return tuple(a.steps[0]["units"][c] for c in P.BUDGETED)


@pytest.mark.parametrize("rows", ["unitsRows", "tableUnitsRows"])
def test_units_is_property_v1s(rows):
    """`unitsOf_unitsRows`, `unitsOf_tableUnitsRows`: Lean's units by class of each circuit, or its refusal of an unlisted
    primitive, are property/v1's of the same Program."""
    lean = lean_rows(rows)
    assert [py_units(gs) for gs, _ in lean] == [out for _, out in lean]
