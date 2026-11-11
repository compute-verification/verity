"""`TokenSelect_v1{V}` (greedy sampling) cut into units of at most a Flock block each (`registry.greedy_stages`), and their twin,
from which the serving reader commits a row's units:

* every unit's Definition on the twin's inputs for it (its lanes of the row, and the combine's chunks' pairs) computes the twin's
  outputs, and the twin's token is `TokenSelect_v1`'s as the circuit evaluator computes it, on rows of ties (within a chunk and
  across chunks), NaN, -inf, +inf and signed zeros, at small V and at GLM-5.3's;
* at the served statics the combine computes the token's two words on rows whose winner sits past 2^16 lanes and ties across
  chunks, and at every index on each side of the two lanes where a chunk's indices pass a multiple of 2^16 (`carries`);
* `GreedySelect_v1`, whose body is the plan, computes the same two words;
* `plan` wires each combine parameter to its chunk's output of its type, every chunk is whole, every output is 16-bit words,
  and `cut` counts its units, at small and served statics; a K that does not divide V is refused.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate
from verity.circuits.types import Array, Tuple
from verity.definitions.registry import b1
from verity.definitions.registry import greedy_stages as G

SERVED_V = 154880
SERVED_N = 7744
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


def via_units(V: int, x: np.ndarray, K: int | None = None) -> dict[tuple, list[int]]:
    """Every unit of `plan(V, K)` evaluated on its lanes of ``x`` and on its writers' outputs."""
    outs: dict[tuple, list[int]] = {}
    for u in G.plan(V, K):
        wires = {p: (k, j) for p, k, j in u.wires}
        args = [[int(v) for v in x[u.lanes[0]:u.lanes[1]]] if p == "logits" else outs[wires[p][0]][wires[p][1]]
                for p, _t in u.defn.signature()[0]]
        outs[u.key] = [int(v) for v in evaluate(u.defn, *args)]
    return outs


def check(V: int, x: np.ndarray, want: int, name: str, K: int | None = None) -> None:
    tok, words = G.twin(V, x, K)
    outs = via_units(V, x, K)
    assert outs == {k: [int(w) for a in f for w in a] for k, f in words.items()}, (V, name)
    assert tok == want and outs[("combine",)] == [want & 0xFFFF, want >> 16], (V, name, tok, want)


@pytest.mark.parametrize("V", [2, 3, 16, 17, 23, 200, 640])
def test_the_units_compute_the_twin_s_words_whose_token_is_the_call_s(V):
    """Rows of one lane a chunk (V up to 20), of one chunk (23, a prime), and of 20 chunks; and the select whose body is the
    plan, the same words."""
    for name, x in rows(V, seed=V).items():
        want = int(evaluate(bind(b1.TokenSelect, V=V), [int(v) for v in x])[0])
        check(V, x, want, name)
        assert [int(w) for w in evaluate(G.select(V), [int(v) for v in x])] == [want & 0xFFFF, want >> 16], (V, name)


@pytest.mark.parametrize("K", [1, 2, 4, 5, 8, 10, 40])
def test_any_k_that_divides_v_is_a_plan_of_the_same_function(K):
    """K is a number: every divisor of V = 40 plans the same Definitions at N = 40 / K, and each computes the Call's token."""
    V = 40
    assert len(G.plan(V, K)) == K and {u.defn.bindings["N"] for u in G.plan(V, K)} == {V // K}
    for name, x in rows(V, seed=K).items():
        want = int(evaluate(bind(b1.TokenSelect, V=V), [int(v) for v in x])[0])
        check(V, x, want, name, K)
        assert [int(w) for w in evaluate(G.select(V, K), [int(v) for v in x])] == [want & 0xFFFF, want >> 16], (K, name)


def _planted(lanes: dict[int, int], x0: int = NINF) -> np.ndarray:
    x = np.full(SERVED_V, NINF, np.uint16)
    x[0] = x0
    for i, w in lanes.items():
        x[i] = w
    return x


#: rows of GLM-5.3's vocabulary, each -inf but the lanes it names, and the token: past 2^16 lanes (the token's high word 1 and 2),
#: the row's last lane, a tie across chunks (the earlier one wins), a NaN that never wins, a NaN logits[0] that is never left,
#: and a maximum on each side of the two lanes where a chunk's indices pass a multiple of 2^16 (chunk 8's lane 3,584 is the
#: row's 65,536 and chunk 16's lane 7,168 its 131,072), alone, tied across the lane, and tied with a later chunk
PLANTED = [({70000: ONE}, NINF, 70000), ({131072: ONE, 154879: ONE}, NINF, 131072), ({154879: 0xFF7F}, NINF, 154879),
           ({3 * SERVED_N + 5: ONE, 7 * SERVED_N: ONE, 20: 0x3F00}, NINF, 3 * SERVED_N + 5),
           ({65535: NAN, 65536: 0x3F00}, NINF, 65536),
           ({90000: PINF}, NAN, 0), ({SERVED_N: NEG0}, POS0, 0), ({SERVED_N: POS0, 100000: 0x4000}, NEG0, 100000),
           ({65535: ONE}, NINF, 65535), ({65536: ONE}, NINF, 65536), ({65535: ONE, 65536: ONE}, NINF, 65535),
           ({65536: ONE, 65537: ONE, 100000: ONE}, NINF, 65536), ({65535: 0x3F00, 65536: ONE}, NINF, 65536),
           ({131071: ONE}, NINF, 131071), ({131072: ONE}, NINF, 131072), ({131071: ONE, 131072: ONE}, NINF, 131071),
           ({131072: ONE, 131073: ONE, 150000: ONE}, NINF, 131072), ({131071: 0x3F00, 131072: ONE}, NINF, 131072),
           ({8 * SERVED_N: ONE}, NINF, 8 * SERVED_N), ({9 * SERVED_N - 1: ONE}, NINF, 9 * SERVED_N - 1),
           ({16 * SERVED_N: ONE}, NINF, 16 * SERVED_N), ({17 * SERVED_N - 1: ONE, 17 * SERVED_N: ONE}, NINF, 17 * SERVED_N - 1)]
#: the equality test's rows of `PLANTED`: on each side of both lanes, a tie across the lane (the earlier side wins) and a tie
#: of the later side with a later chunk
CARRIED = [({65535: ONE, 65536: ONE}, 65535), ({65536: ONE, 65537: ONE, 100000: ONE}, 65536),
           ({131071: ONE, 131072: ONE}, 131071), ({131072: ONE, 131073: ONE, 150000: ONE}, 131072)]


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


def test_the_served_chunks_that_pass_a_multiple_of_2_16_are_two():
    """At N = 7,744 of V = 154,880 a multiple of 2^16 falls inside chunk 8, at its lane 3,584, and chunk 16, at its lane
    7,168, and inside no other; at N = 9,680 (K = 16) inside chunks 6 and 13; at a power of two inside none."""
    assert G.width(SERVED_V) == SERVED_N and G.carries(SERVED_V, SERVED_N) == {8: 3584, 16: 7168}
    assert G.carries(SERVED_V, SERVED_V // 16) == {6: 7456, 13: 5232}
    assert all(c * n + t == (c * n // G.SPAN + 1) * G.SPAN for n in (SERVED_N, SERVED_V // 16)
               for c, t in G.carries(SERVED_V, n).items())
    assert G.carries(1 << 20, 1 << 14) == G.carries(1 << 17, 1 << 16) == {}


def test_every_index_of_a_served_chunk_is_the_token_s_two_words():
    """What holds every output to one word at a width that is no power of two: the combine at GLM-5.3's statics on each
    chunk's pair alone above -inf, at its first and last index and, in the two chunks a multiple of 2^16 falls inside, at
    the indices around that lane.  The token cN + i is its low word, wrapped, and its high word, ``cN >> 16`` and one more
    from that lane on."""
    combine = G.plan(SERVED_V)[-1].defn
    K = SERVED_V // SERVED_N
    for c in range(K - 1):
        t = G.carries(SERVED_V, SERVED_N).get(c)
        for i in (0, SERVED_N - 1, *((t - 2, t - 1, t, t + 1) if t else ())):
            pairs = [w for k in range(K - 1) for w in ((ONE, i) if k == c else (NINF, 0))]
            tok = c * SERVED_N + i
            got = evaluate(combine, [NINF] * SERVED_N, *pairs)
            assert [int(v) for v in got] == [tok % 65536, tok // 65536], (c, i)
    assert (8 * SERVED_N + 3584, 16 * SERVED_N + 7168) == (65536, 131072)


def test_the_widest_chunk_with_a_carry_compares_its_last_index():
    """A chunk a multiple of 2^16 falls inside is at most 0x7F81 lanes: its last index, 0x7F80, is the largest word bf16
    orders as the integers do (+inf's).  At that width the row's last lane is the token's two words, and one lane wider
    is refused; a power of two has no such chunk and is taken up to 2^16 lanes."""
    N = G.ORDERED + 1
    V = 3 * N
    assert G.carries(V, N) == {2: 65536 - 2 * N}
    x = [NINF] * N
    x[-1] = ONE
    got = evaluate(bind(G.GreedyCombine, V=V, N=N), x, NINF, 0, NINF, 0)
    assert [int(v) for v in got] == [(V - 1) % 65536, (V - 1) // 65536]
    with pytest.raises(ValueError, match="a multiple of 2\\^16 falls inside a chunk"):
        bind(G.GreedyCombine, V=3 * (N + 1), N=N + 1).signature()
    bind(G.GreedyCombine, V=1 << 17, N=1 << 16).signature()
    bind(G.GreedyCombine, V=1 << 18, N=1 << 15).signature()


@pytest.mark.slow
def test_the_served_units_compute_the_call_s_token():
    """Rows of GLM-5.3's vocabulary through every unit's Definition and through `TokenSelect_v1{V=154880}`: edge rows, and
    `CARRIED`'s, a maximum on each side of the two lanes where a chunk's indices pass a multiple of 2^16, with a tie across
    each.  It catches what the small rows can't: a chunk of 7,744 lanes, the combine's high word and its carry against the
    Call itself; its seconds are the evaluator's, about six a row, twelve with its units."""
    xs = rows(SERVED_V, seed=16)
    for name in ("plain", "ties", "nan0", "tie_later"):
        want = int(evaluate(bind(b1.TokenSelect, V=SERVED_V), [int(v) for v in xs[name]])[0])
        check(SERVED_V, xs[name], want, name)
    for lanes, token in CARRIED:
        x = _planted(lanes)
        want = int(evaluate(bind(b1.TokenSelect, V=SERVED_V), [int(v) for v in x])[0])
        assert want == token, lanes
        check(SERVED_V, x, want, str(lanes))


def _fields(ret) -> list:
    return [f[1] if isinstance(f, tuple) else f for f in ret.fields] if isinstance(ret, Tuple) else [ret]


@pytest.mark.parametrize("V", [2, 16, 23, 200, 640, SERVED_V])
def test_the_plan_wires_each_pair_to_its_chunk_and_every_output_is_16_bit_words(V):
    units = G.plan(V)
    chunks = units[:-1]
    K, N = G.units(V), G.width(V)
    assert K * N == V and [u.lanes for u in units] == [(c * N, (c + 1) * N) for c in range(K)] and units[-1].key == ("combine",)
    assert [u.defn.bindings["FIRST"] for u in chunks] == ([True] + [False] * (len(chunks) - 1))[:len(chunks)]
    seen = {u.key: u.defn for u in chunks}
    params = dict(units[-1].defn.signature()[0])
    assert params["logits"].n == N
    assert [(p, k, j) for p, k, j in units[-1].wires] == [w for c in range(len(chunks))
                                                          for w in ((f"b{c}", ("chunk", c), 0), (f"i{c}", ("chunk", c), 1))]
    for p, k, j in units[-1].wires:
        assert _fields(seen[k].signature()[1])[j] == params[p], (p, k, j)
    for u in chunks:
        assert [list(f.leaf_widths()) for f in _fields(u.defn.signature()[1])] == [[16], [16]], u.key
    assert units[-1].defn.signature()[1] == G.select(V).signature()[1] == Array(2, G.U16)
    assert sum(n for _d, n in G.cut(bind(b1.TokenSelect, V=V))) == len(units)


def test_the_served_cut_s_units():
    """At GLM-5.3's vocabulary the Call is `UNITS` = 20 units of 3 Definitions: nineteen chunks of 7,744 lanes (the first
    FIRST), and the combine, which reads the last chunk, 7,744 lanes too."""
    cut = G.cut(bind(b1.TokenSelect, V=SERVED_V))
    assert [(descriptor_id(d), n) for d, n in cut] == [("GreedyChunk_v1{N=7744,FIRST=true}", 1),
                                                       ("GreedyChunk_v1{N=7744,FIRST=false}", 18),
                                                       ("GreedyCombine_v2{V=154880,N=7744}", 1)]
    assert G.units(SERVED_V) == G.UNITS == 20
    assert Counter(u.key[0] for u in G.plan(SERVED_V)) == {"chunk": 19, "combine": 1}
    assert G.plan(SERVED_V)[-1].lanes == (147136, 154880)
    assert descriptor_id(G.select(SERVED_V)) == "GreedySelect_v1{V=154880,N=7744}"
    assert b1.CUTS["TokenSelect_v1"] == G.__name__


@pytest.mark.parametrize("V,K", [(SERVED_V, 18), (SERVED_V, 3), (16, 3), (200, 0), (65537, None), (65537, 1), (1 << 18, 2)])
def test_a_plan_whose_chunks_are_not_whole_or_past_2_16_lanes_is_refused(V, K):
    """A K that does not divide V (a narrower last chunk would start off the chunks' rows), and one whose chunks are past
    2^16 lanes: a prime past 2^16 has no plan."""
    with pytest.raises(ValueError):
        G.plan(V, K)


@pytest.mark.parametrize("V,N", [(16, 3), (16, 32), (2**32 + 2**16, 2**16), (1 << 17, 1 << 17), (120000, 40000)])
def test_a_combine_whose_chunks_are_not_whole_or_whose_index_is_no_16_bit_word_is_refused(V, N):
    """A chunk width that does not divide V, wider than V or than 2^16 lanes, a row past 2^32 lanes (its token is no two
    words), and a width past 0x7F81 lanes with a multiple of 2^16 inside a chunk (its carry's compare); the select whose
    body is the plan refuses the same."""
    for d in (G.GreedyCombine, G.GreedySelect):
        with pytest.raises(ValueError):
            bind(d, V=V, N=N).signature()


@pytest.mark.parametrize("first", ["false", 0, None])
def test_a_chunk_whose_first_is_no_bool_is_refused(first):
    """An id's ``FIRST=false`` parsed as a Python literal is the string 'false', which is true: refused, not the first chunk."""
    with pytest.raises(ValueError, match="FIRST"):
        bind(G.GreedyChunk, N=8, FIRST=first).signature()
