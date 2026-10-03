"""The hiding tile's digest (`hashes.KeccakP1600R12`, TurboSHAKE128's permutation) lowers in C-Flock through PoUW's registered
Boolean Definitions of its word operations (`verity_pouw.circuit.boolean`, found by `ir_lower`'s registry fallback) with χ's
output committed: 38,400 rows (12 rounds x 1,600 χ ANDs and 1,600 commits), 652,800 nonzeros, the IR reference's outputs.
Without the commits the forms grow round over round and one permutation runs out of memory."""

from __future__ import annotations

import random

from verity.evaluation.reference import evaluate
from verity.ml.boolean.forms import Circuit
from verity_flock import ir_lower as IL
from verity_pouw.circuit import hashes, pc4  # noqa: F401  (pc4, the hiding tile's builder, registers the Definitions)


def test_keccak_p_lowers_through_the_registered_definitions_with_chi_committed():
    ops = ("PouwXor32_v1", "PouwChi32_v1", "PouwRotl64Lo_v1", "PouwRotl64Hi_v1")
    assert all(IL._boolean_build(p) is IL.PIECES[p] for p in ops)
    C = Circuit()
    ins = [C.input(f"s{i}", 32) for i in range(50)]
    outs = IL.lower(C, hashes.KeccakP1600R12, ins)
    C.output("o", [b for w in outs for b in w])
    n = C.counts()
    assert (n["ands"], n["nnz_AB"]) == (38_400, 652_800)
    assert all(len(b.s) == 1 for w in outs for b in w)
    R = random.Random(20261003)
    states = [[R.getrandbits(32) for _ in range(50)] for _ in range(8)]
    vals, _ = C.evaluate({f"s{i}": IL.slice_words([s[i] for s in states], 32) for i in range(50)}, len(states))
    got = [[int(v) for v in IL.unslice_words(vals["o"][32 * j:32 * (j + 1)], len(states))] for j in range(50)]
    assert [[got[j][t] for j in range(50)] for t in range(len(states))] == [list(evaluate(hashes.KeccakP1600R12, s)) for s in states]
