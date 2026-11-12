"""`TokenSelect_v1{V}` (greedy sampling) cut into units of at most a Flock block each (`registry.greedy_stages`), and their twin,
from which the serving reader commits a row's units:

* every unit's Definition on the twin's inputs for it (its lanes of the row, and the combine's chunks' pairs) computes the twin's
  outputs, and the twin's token is `TokenSelect_v1`'s as the circuit evaluator computes it, on rows of ties (within a chunk and
  across chunks), NaN, -inf, +inf and signed zeros, at small V and at GLM-5.3's;
* at the served statics the combine computes the token's two words on rows whose winner sits past 2^16 lanes and ties across
  chunks;
* `plan` wires each combine parameter to its chunk's output of its type, every output is one 16-bit word, and `cut` counts its
  units, at small and served statics.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate
from verity.circuits.types import Tuple
from verity.definitions.registry import b1
from verity.definitions.registry import greedy_stages as G

SERVED_V = 154880
NAN, NEG0, POS0, NINF, PINF, ONE = 0x7FC1, 0x8000, 0x0000, 0xFF80, 0x7F80, 0x3F80


def bf16_row(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x, np.float32).view(np.uint32) >> 16).astype(np.uint16)


def rows(V: int, seed: int) -> dict[str, np.ndarray]:
    """Edge rows of V bf16 words: plain, ties within a chunk and across chunks, NaN at logits[0] and elsewhere, every lane -inf,
    every lane NaN, signed zeros, +inf twice, one finite lane at the end, and a uniform row."""
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(V) * 3).astype(np.float32)
    w = G.width(V)
    out = {"plain": bf16_row(x), "ties": bf16_row(np.round(x)), "ties4": bf16_row(np.round(x / 4))}
    a = bf16_row(x)
    a[0] = NAN
    out["nan0"] = a
    a = bf16_row(x)
    a[rng.choice(V, min(3, V - 1), replace=False)] = NAN
    out["nans"] = a
    out["allninf"] = np.full(V, NINF, np.uint16)
    a = np.full(V, NINF, np.uint16)
    a[0] = NAN
    out["nan0ninf"] = a
    a = np.full(V, NAN, np.uint16)
    a[V // 2] = NINF
    out["allnan"] = a
    a = np.full(V, NAN, np.uint16)
    a[0], a[V - 1] = NINF, ONE
    out["nan_tail"] = a
    a = np.full(V, NEG0, np.uint16)
    a[V - 1] = POS0
    out["zeros"] = a
    a = bf16_row(x)
    a[V // 3] = a[V - 1] = PINF
    out["pinf"] = a
    a = np.full(V, NINF, np.uint16)
    a[V - 1] = 0xFF7F
    out["last"] = a
    out["uniform"] = np.full(V, ONE, np.uint16)
    a = np.full(V, NINF, np.uint16)
    a[0] = a[min(w, V - 1)] = 0xBF80
    out["tie_across"] = a
    a = np.full(V, NINF, np.uint16)
    a[min(w + 1, V - 1)] = a[min(3 * w, V - 1)] = ONE
    out["tie_later"] = a
    return out


def via_units(V: int, x: np.ndarray) -> dict[tuple, list[int]]:
    """Every unit of `plan(V)` evaluated on its lanes of ``x`` and on its writers' outputs."""
    outs: dict[tuple, list[int]] = {}
    for u in G.plan(V):
        wires = {p: (k, j) for p, k, j in u.wires}
        args = [[int(v) for v in x[u.lanes[0]:u.lanes[1]]] if p == "logits" else outs[wires[p][0]][wires[p][1]]
                for p, _t in u.defn.signature()[0]]
        outs[u.key] = [int(v) for v in evaluate(u.defn, *args)]
    return outs


def check(V: int, x: np.ndarray, want: int, name: str) -> None:
    tok, words = G.twin(V, x)
    outs = via_units(V, x)
    assert outs == {k: [int(a[0]) for a in f] for k, f in words.items()}, (V, name)
    assert tok == want and outs[("combine",)] == [want & 0xFFFF, want >> 16], (V, name, tok, want)


@pytest.mark.parametrize("V", [2, 3, 16, 17, 200])
def test_the_units_compute_the_twin_s_words_whose_token_is_the_call_s(V):
    for name, x in rows(V, seed=V).items():
        want = int(evaluate(bind(b1.TokenSelect, V=V), [int(v) for v in x])[0])
        check(V, x, want, name)


def _planted(lanes: dict[int, int], x0: int = NINF) -> np.ndarray:
    x = np.full(SERVED_V, NINF, np.uint16)
    x[0] = x0
    for i, w in lanes.items():
        x[i] = w
    return x


#: rows of GLM-5.3's vocabulary, each -inf but the lanes it names, and the token: past 2^16 lanes (the token's high word 1 and 2),
#: the row's last lane, a tie across chunks (the earlier one wins), a NaN that never wins, and a NaN logits[0] that is never left
PLANTED = [({70000: ONE}, NINF, 70000), ({131072: ONE, 154879: ONE}, NINF, 131072), ({154879: 0xFF7F}, NINF, 154879),
           ({3 * 16384 + 5: ONE, 7 * 16384: ONE, 20: 0x3F00}, NINF, 3 * 16384 + 5), ({65535: NAN, 65536: 0x3F00}, NINF, 65536),
           ({90000: PINF}, NAN, 0), ({16384: NEG0}, POS0, 0), ({16384: POS0, 100000: 0x4000}, NEG0, 100000)]


def test_the_served_combine_computes_the_token_s_two_words():
    """The units at GLM-5.3's statics on rows whose winner each `PLANTED` names: the combine evaluated on the row's last chunk
    and its other chunks' pairs off the twin (whose chunk Definitions the slow test holds)."""
    combine = G.plan(SERVED_V)[-1]
    lo, hi = combine.lanes
    for lanes, x0, want in PLANTED:
        x = _planted(lanes, x0)
        tok, words = G.twin(SERVED_V, x)
        got = evaluate(combine.defn, [int(v) for v in x[lo:hi]], *(int(words[k][j][0]) for _p, k, j in combine.wires))
        assert tok == want and [int(v) for v in got] == [want & 0xFFFF, want >> 16], (lanes, x0)


@pytest.mark.slow
def test_the_served_units_compute_the_call_s_token():
    """Rows of GLM-5.3's vocabulary through every unit's Definition and through `TokenSelect_v1{V=154880}`.  It catches what
    the small rows can't: a chunk of 16,384 lanes, the combine's last chunk of 7,424, and its high word; its seconds are the
    evaluator's, about six a row."""
    xs = rows(SERVED_V, seed=16)
    for name in ("plain", "ties", "nan0", "tie_later"):
        want = int(evaluate(bind(b1.TokenSelect, V=SERVED_V), [int(v) for v in xs[name]])[0])
        check(SERVED_V, xs[name], want, name)
    check(SERVED_V, _planted({131072: ONE, 154879: ONE}), 131072, "planted")


def _fields(ret) -> list:
    return [f[1] if isinstance(f, tuple) else f for f in ret.fields] if isinstance(ret, Tuple) else [ret]


@pytest.mark.parametrize("V", [2, 16, 200, 65537, SERVED_V])
def test_the_plan_wires_each_pair_to_its_chunk_and_every_output_is_a_16_bit_word(V):
    units = G.plan(V)
    chunks = units[:-1]
    assert [u.lanes for u in units] == b1.chunks(V, G.width(V)) and units[-1].key == ("combine",)
    assert [u.defn.bindings["FIRST"] for u in chunks] == [True] + [False] * (len(chunks) - 1)
    seen = {u.key: u.defn for u in chunks}
    params = dict(units[-1].defn.signature()[0])
    assert params["logits"].n == units[-1].lanes[1] - units[-1].lanes[0]
    assert [(p, k, j) for p, k, j in units[-1].wires] == [w for c in range(len(chunks))
                                                          for w in ((f"b{c}", ("chunk", c), 0), (f"i{c}", ("chunk", c), 1))]
    for p, k, j in units[-1].wires:
        assert _fields(seen[k].signature()[1])[j] == params[p], (p, k, j)
    for u in units:
        assert [list(f.leaf_widths()) for f in _fields(u.defn.signature()[1])] == [[16], [16]], u.key
    assert sum(n for _d, n in G.cut(bind(b1.TokenSelect, V=V))) == len(units)


def test_the_served_cut_s_units():
    """At GLM-5.3's vocabulary the Call is 10 units of 3 Definitions: nine chunks of 16,384 lanes (the first FIRST), and the
    combine, which reads the last chunk of 7,424."""
    cut = G.cut(bind(b1.TokenSelect, V=SERVED_V))
    assert [(descriptor_id(d), n) for d, n in cut] == [("GreedyChunk_v1{N=16384,FIRST=true}", 1),
                                                       ("GreedyChunk_v1{N=16384,FIRST=false}", 8),
                                                       ("GreedyCombine_v1{V=154880,N=16384}", 1)]
    assert Counter(u.key[0] for u in G.plan(SERVED_V)) == {"chunk": 9, "combine": 1}
    assert G.plan(SERVED_V)[-1].lanes == (147456, 154880)
    assert b1.CUTS["TokenSelect_v1"] == G.__name__


@pytest.mark.parametrize("V,N", [(16, 3), (16, 32), (2**32 + 1, 2**16), (1 << 17, 1 << 17)])
def test_a_combine_of_chunks_whose_index_is_no_16_bit_word_is_refused(V, N):
    """A chunk width that is no power of two (its index would carry into the chunk's offset), wider than V or than 2^16 lanes,
    and a row past 2^32 lanes (its token is no two words)."""
    with pytest.raises(ValueError):
        bind(G.GreedyCombine, V=V, N=N).signature()


@pytest.mark.parametrize("first", ["false", 0, None])
def test_a_chunk_whose_first_is_no_bool_is_refused(first):
    """An id's ``FIRST=false`` parsed as a Python literal is the string 'false', which is true: refused, not the first chunk."""
    with pytest.raises(ValueError, match="FIRST"):
        bind(G.GreedyChunk, N=8, FIRST=first).signature()
