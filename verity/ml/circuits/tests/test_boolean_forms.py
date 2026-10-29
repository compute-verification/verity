"""`forms.Circuit.commit`: a word as committed bits, one row for each bit that is not one committed bit already or a
constant, with every value unchanged; a Definition traced from a builder that commits (`trace.TracingCircuit`) has no gate for
it, so its And gates are the builder's ANDs without the commits. A `trace.part`'s ``build``, called as C-Flock's lowering
calls it (statics first only when it has some), traces to the part's own gates."""
from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np

from verity.ml.circuits.evaluation.bits import evaluate_words
from verity.ml.circuits import boolean as BL
from verity.ml.circuits.types import Array
from verity.ml.circuits.boolean.forms import ONE, ZERO, Circuit
from verity.ml.circuits.boolean.trace import definition, part, trace

U4 = Array(4, BL.BIT)


def test_commit_gives_each_form_a_row_and_keeps_bits_and_constants() -> None:
    C = Circuit()
    a, b = C.input("a", 4), C.input("b", 4)
    word = [a[0], a[1].inv(), a[2] ^ b[2], ZERO, ONE, a[3] ^ b[3] ^ b[0]]
    got = C.commit(word)
    assert got[0] is word[0] and got[3] is ZERO and got[4] is ONE
    assert all(len(w.s) == 1 and w.c == 0 for i, w in enumerate(got) if i in (1, 2, 5))
    assert [C.data[next(iter(got[i].s))] for i in (1, 2, 5)] == [(word[i], ONE) for i in (1, 2, 5)]
    C.output("word", word)
    C.output("got", got)
    assert C.counts()["ands"] == 3
    R = random.Random(20261003)
    vals, _ = C.evaluate({"a": [R.getrandbits(64) for _ in range(4)], "b": [R.getrandbits(64) for _ in range(4)]}, 64)
    assert vals["got"] == vals["word"]


def _xor_and(C, a, b):
    return C.commit([x ^ C.AND(x, y) for x, y in zip(a, b)])


def test_a_traced_commit_is_no_gate() -> None:
    d = definition("TestCommitXorAnd", 1, _xor_and, (("a", U4), ("b", U4)), U4, register=False)
    assert d.traced().counts == {"and": 4, "xor": 4, "not": 0}
    C = Circuit()
    C.output("o", d.build(C, C.input("a", 4), C.input("b", 4)))
    assert C.counts()["ands"] == 8
    a, b = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
    got = evaluate_words(d.bind(), a.reshape(-1, 1), b.reshape(-1, 1)).reshape(-1)
    assert (got == (a ^ (a & b)).reshape(-1)).all()


def _sig_part(S):
    return (("a", U4), ("b", U4)), Array(getattr(S, "K", 4), BL.BIT)


def _build_part(C, S, a, b):
    return _xor_and(C, a, b)[:getattr(S, "K", 4)]


def test_a_parts_build_is_called_as_the_lowering_calls_it() -> None:
    for statics, key in (((), ()), (("K",), (3,))):
        d = part("TestPartXorAnd" + "".join(statics), statics, _sig_part, _build_part, "", register=False)
        S = SimpleNamespace(**dict(zip(statics, key)))
        build = (lambda C, *e, d=d, S=S: d.build(C, S, *e)) if statics else d.build
        params, ret = _sig_part(S)
        got, want = trace(build, [t for _, t in params], ret), d.traced(*key)
        assert (got.gates, got.outputs) == (want.gates, want.outputs) and want.counts["and"] == ret.leaves, statics
        assert d.build.__module__ == __name__
