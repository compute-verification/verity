"""A read of a table the program brings (a word read as `verity.ml.definitions.registry.tables.table_read` makes one) lowers by its
Boolean form's builder (`verity.ml.circuits.boolean.table`, through `ir_lower._boolean_build`): the entry at the index's low bits
at every index and on random tables, in (2^IB - 1) * 16 EW ANDs over the unit's own input bits and no lookup; a read of a table
past `BT.MAX_WORDS` has no piece; and a Program whose reads share one registered table lowers under `Q_word` with every unit
lowered and each unit equal to the reference on synthetic draws."""
from __future__ import annotations

import random

import pytest

from verity.ml.circuits.boolean import table as BT
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array, Value
from verity.ml.circuits.ops.scalar import I32Add

from verity_flock import gf2
from verity_flock import ir_lower as IL
from verity_flock import partition_units as PU

#: (IB, EW): `exp2_lut127`'s 2^8 entries of two words, one-word and two-word entries, a 16-row table
SHAPES = [(8, 2), (6, 1), (5, 2), (10, 1)]
_WORDS: dict[tuple[int, int], PrimitiveDefinition] = {}


def _word(ib: int, ew: int) -> PrimitiveDefinition:
    """`FlockTableRead{IB}x{EW}_v1`, a word read as vllm's `table_read` makes one, and its Boolean form when it has one."""
    key = (ib, ew)
    if key not in _WORDS:
        mask, rows, iw = (1 << ib) - 1, (ew << ib) // BT.ROW_WORDS, 8 if ib <= 8 else 16 if ib <= 16 else 32

        def ev(idx: int, *words: int) -> int:
            e = (idx & mask) * ew
            return sum((words[e + k] & 0xFFFF) << (16 * k) for k in range(ew))

        def ev_lazy(get) -> int:
            e = 1 + (get(0) & mask) * ew
            return sum((get(e + k) & 0xFFFF) << (16 * k) for k in range(ew))

        p = PrimitiveDefinition(f"FlockTableRead{ib}x{ew}", 1, (("index", Value(iw)), ("table", Array(rows, Array(64, Value(16))))),
                                Value(16 * ew), ev, register=False)
        p.evaluate_lazy = ev_lazy  # type: ignore[attr-defined]
        if ew << ib <= BT.MAX_WORDS:
            BT.table_read(p, ib)
        _WORDS[key] = p
    return _WORDS[key]


@pytest.mark.parametrize("ib,ew", SHAPES)
def test_a_table_read_is_its_boolean_form_on_every_index(ib, ew):
    p = _word(ib, ew)
    assert p.id in IL.PIECES
    iw, n = p.params[0][1].w, (1 << ib) * ew
    rng = random.Random(ib * 10 + ew)
    lanes = [(i, [rng.getrandbits(16) for _ in range(n)]) for i in range(1 << ib)]
    lanes += [(rng.getrandbits(iw), [rng.getrandbits(16) for _ in range(n)]) for _ in range(64)]
    C = gf2.Circuit()
    idx, words = C.input("idx", iw), [C.input(f"w{j}", 16) for j in range(n)]
    C.output("o", IL.PIECES[p.id](C, idx, *words))
    assert (C.counts()["ands"], C.counts()["lookups"]) == (((1 << ib) - 1) * 16 * ew, 0)
    ins = {"idx": IL.slice_words([i for i, _ in lanes], iw)} | {f"w{j}": IL.slice_words([w[j] for _, w in lanes], 16) for j in range(n)}
    outs, _ = C.evaluate(ins, len(lanes))
    got = [int(x) for x in IL.unslice_words(outs["o"], len(lanes))]
    assert got == [p.evaluate(i, *w) for i, w in lanes]


def test_a_read_of_a_table_past_max_words_has_no_piece():
    p = _word(13, 1)
    assert BT.MAX_WORDS < 1 << 13 and p.id not in IL.PIECES


def test_reads_of_one_registered_table_lower_unit_by_unit():
    """Four reads of one registered `exp2_lut127`-shaped table (2^8 entries of two words), each entry added to the next: under
    `Q_word` every unit lowers, the reads' units are the read's ANDs, and every unit equals the reference on synthetic draws."""
    read = _word(8, 2)
    idx_t, tab_t, I32 = read.params[0][1], read.params[1][1], Value(32)
    K = 4

    def body(B, S, t, table):
        e = [B.call(read, t[k], table) for k in range(K)]
        return Coll(Array(K, I32), concat([B.call(I32Add, e[k], e[(k + 1) % K]).refs for k in range(K)]))

    top = CompositeDefinition("FlockTestTableReads", 1, (), lambda S: ((("t", Array(K, idx_t)), ("table", tab_t)), Array(K, I32)),
                              body, register=False)
    units = PU.partition_units(Program(top, registered=("table",)), "q-word")
    r = PU.check(units, T=8)
    assert r["units_lowered"] == r["units"] == 2 * K and r["shapes_lowered"] == r["shapes"] and r["mismatches"] == 0
    assert [(c["units"], c["ands"]) for c in r["classes"] if c["definition"] == read.id] == [(K, 255 * 32)]


@pytest.fixture(autouse=True)
def _table_reads_unwrapped(monkeypatch):
    """`boolean_export`'s trace wraps the pieces of `IL.PIECES`, and each one resolved after it (`PIECES.wrap`), once a test in the
    worker exports; these tests compare their own table reads' pieces, so those run unwrapped whichever test ran first."""
    monkeypatch.setattr(IL.PIECES, "wrap", None)
    for k in [k for k in dict.keys(IL.PIECES) if isinstance(k, str) and k.startswith("FlockTableRead")]:
        monkeypatch.setitem(IL.PIECES, k, getattr(IL.PIECES[k], "__wrapped__", IL.PIECES[k]))
