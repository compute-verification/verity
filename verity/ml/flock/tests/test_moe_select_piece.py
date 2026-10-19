"""A static MoE pair unit's select of its expert's word (`MoeSelectBf16x{E}_v1`, circuits' `verity.ml.circuits.ops.moe`) and a
routed FP8 expert's gather (`GatherE4m3x{E}_v1`, vLLM's `fp8_moe_experts`) lower by their `tail_pieces.FAMILIES` pieces in a
process that never made their Boolean forms: the word at every id in [0, E) and the fill at the ids outside it (E, the negative
ids, the widest), on random columns, with no lookup, the select in the ANDs of circuits' `MoeSelectBf16x{E}_v2` (its pins at
E = 3, 128 and 256); and a unit that selects its expert's words inside itself lowers under `Q_word` with every unit equal to
the reference."""
from __future__ import annotations

import inspect
import random

import pytest

from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Array, Value

from verity_flock import gf2
from verity_flock import ir_lower as IL
from verity_flock import partition_units as PU
from verity_flock import tail_pieces as TP

I32 = Value(32)
#: family -> (word bits, the word an id outside [0, E) selects, its piece)
FAMILY = {"MoeSelectBf16": (16, 0x7FFF, TP.moe_select_bf16), "GatherE4m3": (8, 0x7F, TP.gather_e4m3)}
#: the ANDs of circuits' `MoeSelectBf16x{E}_v2` (its circuit-check pins), whose construction the piece is
PINNED_ANDS = {3: 93, 128: 2072, 256: 4119}
_WORDS: dict[tuple[str, int], PrimitiveDefinition] = {}


def _word(family: str, E: int) -> PrimitiveDefinition:
    """`{family}x{E}_v1`, ``column[e]`` for the signed i32 ``e`` in [0, E), else the family's fill, unregistered."""
    if (family, E) not in _WORDS:
        bits, fill, _ = FAMILY[family]

        def ev(e: int, *column: int) -> int:
            return column[e] if e < E else fill

        def ev_lazy(get) -> int:
            e = get(0)
            return get(1 + e) if e < E else fill

        p = PrimitiveDefinition(f"{family}x{E}", 1, (("e", I32), ("column", Array(E, Value(bits)))), Value(bits), ev, register=False)
        p.evaluate_lazy = ev_lazy  # type: ignore[attr-defined]
        _WORDS[(family, E)] = p
    return _WORDS[(family, E)]


@pytest.mark.parametrize("family,E", [(f, E) for f in FAMILY for E in (1, 3, 5, 128, 256)])
def test_a_select_is_the_gather_on_every_id(family, E):
    p = _word(family, E)
    bits, _, piece = FAMILY[family]
    assert inspect.unwrap(IL.PIECES[p.id]) is inspect.unwrap(piece)  # the Boolean export's tracer wraps every piece
    rng = random.Random(E)
    ids = list(range(E)) + [E, E + 1, 1 << 31, (1 << 31) - 1, 0xFFFFFFFF, 0xFFFFFFFF - E] + [rng.getrandbits(32) for _ in range(32)]
    lanes = [(e, [rng.getrandbits(bits) for _ in range(E)]) for e in ids]
    C = gf2.Circuit()
    e, col = C.input("e", 32), [C.input(f"w{j}", bits) for j in range(E)]
    C.output("o", IL.PIECES[p.id](C, e, *col))
    assert C.counts()["lookups"] == 0
    if family == "MoeSelectBf16" and E in PINNED_ANDS:
        assert C.counts()["ands"] == PINNED_ANDS[E]
    ins = {"e": IL.slice_words(ids, 32)} | {f"w{j}": IL.slice_words([w[j] for _, w in lanes], bits) for j in range(E)}
    outs, _ = C.evaluate(ins, len(lanes))
    assert [int(x) for x in IL.unslice_words(outs["o"], len(lanes))] == [p.evaluate(i, *w) for i, w in lanes]


def test_a_unit_that_selects_its_experts_words_lowers():
    """K words of one routed expert among E = 128, each selected from its slab column: under `Q_word` every unit lowers, each is
    the select's ANDs, and every unit equals the reference on synthetic draws."""
    E, K = 128, 4
    sel = _word("MoeSelectBf16", E)
    sig = lambda S: ((("e", I32), ("cols", Array(K, Array(E, Value(16))))), Array(K, Value(16)))  # noqa: E731
    row = bind(CompositeDefinition("FlockTestMoeSelectRow", 1, (), sig, lambda B, S, e, cols: B.batch(sel, e, cols, axes=(None, 0)),
                                   register=False))
    top = CompositeDefinition("FlockTestMoeSelectPair", 1, (), sig, lambda B, S, e, cols: B.call(row, e, cols), register=False)
    units = PU.partition_units(Program(top), "q-word")
    r = PU.check(units, T=8)
    assert r["units_lowered"] == r["units"] and r["shapes_lowered"] == r["shapes"] and r["mismatches"] == 0
    assert {c["ands"] for c in r["classes"] if c["definition"] == sel.id} == {PINNED_ANDS[E]}
