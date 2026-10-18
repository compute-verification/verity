"""A read of a table the program brings (`verity_vllm.program.registry.tables`' `TableRead{IB}x{EW}_v1`) lowers by its Boolean form's
builder (`verity.ml.circuits.boolean.table`, through `ir_lower._boolean_build`): the entry at the index's low bits at every index and
on random tables, in (2^IB - 1) * 16 EW ANDs over the unit's own input bits and no lookup; and a Program whose reads share one
registered table lowers under `Q_word` with every unit lowered and each unit equal to the reference on synthetic draws."""
from __future__ import annotations

import random

import pytest

T = pytest.importorskip("verity_vllm.program.registry.tables")
from verity.ml.circuits.boolean import table as BT  # noqa: E402
from verity.ml.circuits.defs import CompositeDefinition  # noqa: E402
from verity.ml.circuits.program import Program  # noqa: E402
from verity.ml.circuits.refs import Coll, concat  # noqa: E402
from verity.ml.circuits.types import Array, Value  # noqa: E402
from verity.ml.circuits.ops.scalar import I32Add  # noqa: E402

from verity_flock import gf2  # noqa: E402
from verity_flock import ir_lower as IL  # noqa: E402
from verity_flock import partition_units as PU  # noqa: E402

#: (IB, EW): `exp2_lut127`'s 2^8 entries of two words, one-word and two-word entries, a 16-row table
SHAPES = [(8, 2), (6, 1), (5, 2), (10, 1)]


@pytest.mark.parametrize("ib,ew", SHAPES)
def test_a_table_read_is_its_boolean_form_on_every_index(ib, ew):
    p = T.table_read(ib, ew)
    assert IL.PIECES[p.id] is BT.table_read(p, ib).build
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


def test_reads_of_one_registered_table_lower_unit_by_unit():
    """Four reads of one registered `exp2_lut127`-shaped table (2^8 entries of two words), each entry added to the next: under
    `Q_word` every unit lowers, the reads' units are the read's ANDs, and every unit equals the reference on synthetic draws."""
    read = T.table_read(8, 2)
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
