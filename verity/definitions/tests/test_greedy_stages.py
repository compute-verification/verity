"""`TokenSelect_v1{V}` (greedy sampling) cut into units of at most a Flock block each (`registry.greedy_stages`), and their twin,
from which the serving reader commits a row's units:

* every unit's Definition on the twin's inputs for it (its lanes of the row, and the combine's chunks' results) computes the
  twin's outputs, and the twin's token is `TokenSelect_v1`'s as the circuit evaluator computes it, on rows of ties (within a
  chunk and across chunks), NaN, -inf, +inf and signed zeros, at small V and at GLM-5.3's;
* at the served statics the combine computes the token's two words on rows whose winner sits past 2^16 lanes and ties across
  chunks, its own among them, and each chunk its token index's two words at its first and last index and on each side of the
  two lanes where a chunk's indices pass a multiple of 2^16 (`carries`);
* a chunk whose carry is at a lane up to 128, where its compare reads +0 or a subnormal word, computes them on each side of it;
* `GreedySelect_v1`, whose body is the plan, computes the same two words;
* `plan` wires each combine parameter to its chunk's one output, every chunk is whole and its own Definition, every output is
  16-bit words, and `cut` counts its units, at small and served statics; a K that does not divide V, and a chunk off its
  row's chunks, are refused.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate
from verity.circuits.types import Array
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
        wires = {p: k for p, k, _field in u.wires}
        args = [[int(v) for v in x[u.lanes[0]:u.lanes[1]]] if p == "logits" else outs[wires[p]]
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
#: row's 65,536 and chunk 16's lane 7,168 its 131,072), alone, tied across the lane, and tied with a later chunk; and the
#: combine's own chunk, the last, against a wired one (`OWN`)
PLANTED = [({70000: ONE}, NINF, 70000), ({131072: ONE, 154879: ONE}, NINF, 131072), ({154879: 0xFF7F}, NINF, 154879),
           ({3 * SERVED_N + 5: ONE, 7 * SERVED_N: ONE, 20: 0x3F00}, NINF, 3 * SERVED_N + 5),
           ({65535: NAN, 65536: 0x3F00}, NINF, 65536),
           ({90000: PINF}, NAN, 0), ({SERVED_N: NEG0}, POS0, 0), ({SERVED_N: POS0, 100000: 0x4000}, NEG0, 100000),
           ({65535: ONE}, NINF, 65535), ({65536: ONE}, NINF, 65536), ({65535: ONE, 65536: ONE}, NINF, 65535),
           ({65536: ONE, 65537: ONE, 100000: ONE}, NINF, 65536), ({65535: 0x3F00, 65536: ONE}, NINF, 65536),
           ({131071: ONE}, NINF, 131071), ({131072: ONE}, NINF, 131072), ({131071: ONE, 131072: ONE}, NINF, 131071),
           ({131072: ONE, 131073: ONE, 150000: ONE}, NINF, 131072), ({131071: 0x3F00, 131072: ONE}, NINF, 131072),
           ({8 * SERVED_N: ONE}, NINF, 8 * SERVED_N), ({9 * SERVED_N - 1: ONE}, NINF, 9 * SERVED_N - 1),
           ({16 * SERVED_N: ONE}, NINF, 16 * SERVED_N), ({17 * SERVED_N - 1: ONE, 17 * SERVED_N: ONE}, NINF, 17 * SERVED_N - 1),
           ({19 * SERVED_N: ONE}, NINF, 19 * SERVED_N), ({19 * SERVED_N - 1: ONE, 19 * SERVED_N: ONE}, NINF, 19 * SERVED_N - 1)]
#: the equality test's rows of `PLANTED`: on each side of both lanes, a tie across the lane (the earlier side wins) and a tie
#: of the later side with a later chunk
CARRIED = [({65535: ONE, 65536: ONE}, 65535), ({65536: ONE, 65537: ONE, 100000: ONE}, 65536),
           ({131071: ONE, 131072: ONE}, 131071), ({131072: ONE, 131073: ONE, 150000: ONE}, 131072)]
#: the equality test's rows on the combine's own chunk, lanes 147,136 on, against a wired chunk: a tie, taken by the smaller
#: token index (the wired chunk's), with a chunk before the first carry and with the lane of the second; and the own chunk
#: above a wired one, taken
OWN = [({5 * SERVED_N + 7: ONE, 19 * SERVED_N + 3: ONE}, 5 * SERVED_N + 7), ({131072: ONE, 154879: ONE}, 131072),
       ({5 * SERVED_N + 7: 0x3F00, 19 * SERVED_N + 3: ONE}, 19 * SERVED_N + 3)]


def test_the_served_combine_computes_the_token_s_two_words():
    """The units at GLM-5.3's statics on rows whose winner each `PLANTED` and `OWN` names: the combine evaluated on the row's
    last chunk and its other chunks' results off the twin (whose chunk Definitions the slow test holds)."""
    combine = G.plan(SERVED_V)[-1]
    lo, hi = combine.lanes
    for lanes, x0, want in (*PLANTED, *((lanes, NINF, token) for lanes, token in OWN)):
        x = _planted(lanes, x0)
        tok, words = G.twin(SERVED_V, x)
        got = evaluate(combine.defn, [int(v) for v in x[lo:hi]], *([int(w) for w in words[k][0]] for _p, k, _f in combine.wires))
        assert tok == want and [int(v) for v in got] == [want & 0xFFFF, want >> 16], (lanes, x0)


def test_the_served_chunks_that_pass_a_multiple_of_2_16_are_two():
    """At N = 7,744 of V = 154,880 a multiple of 2^16 falls inside chunk 8, at its lane 3,584, and chunk 16, at its lane
    7,168, and inside no other; at N = 9,680 (K = 16) inside chunks 6 and 13; at a power of two inside none."""
    assert G.width(SERVED_V) == SERVED_N and G.carries(SERVED_V, SERVED_N) == {8: 3584, 16: 7168}
    assert (G.carry(SERVED_N, 8 * SERVED_N), G.carry(SERVED_N, 9 * SERVED_N), G.carry(SERVED_N, 0)) == (3584, None, None)
    assert G.carries(SERVED_V, SERVED_V // 16) == {6: 7456, 13: 5232}
    assert all(c * n + t == (c * n // G.SPAN + 1) * G.SPAN for n in (SERVED_N, SERVED_V // 16)
               for c, t in G.carries(SERVED_V, n).items())
    assert G.carries(1 << 20, 1 << 14) == G.carries(1 << 17, 1 << 16) == {}
    assert (8 * SERVED_N + 3584, 16 * SERVED_N + 7168) == (65536, 131072)


def _alone(n: int, i: int) -> list[int]:
    """A chunk of ``n`` lanes, each -inf but lane ``i``."""
    x = [NINF] * n
    x[i] = ONE
    return x


@pytest.mark.parametrize("c", range(SERVED_V // SERVED_N - 1))
def test_every_served_chunk_returns_its_token_index_s_two_words(c):
    """What holds every output to one word at a width that is no power of two: each chunk Definition at GLM-5.3's statics on
    one lane alone above -inf, at its first and last index and, in the two chunks a multiple of 2^16 falls inside, at the
    indices around that lane.  The token index cN + i is its low word, wrapped, and its high word, ``cN >> 16`` and one more
    from that lane on; and a chunk with no lane above -inf returns (-inf, cN)."""
    chunk = G.plan(SERVED_V)[c]
    assert chunk.defn.bindings == {"N": SERVED_N, "OFF": c * SERVED_N}
    t = G.carries(SERVED_V, SERVED_N).get(c)
    assert (t is not None) == (c in (8, 16))
    for i in (0, SERVED_N - 1, *((t - 2, t - 1, t, t + 1) if t else ())):
        tok = c * SERVED_N + i
        assert [int(v) for v in evaluate(chunk.defn, _alone(SERVED_N, i))] == [ONE, tok % 65536, tok // 65536], (c, i)
    if c in (1, 8):
        got = evaluate(chunk.defn, [NINF] * SERVED_N)
        assert [int(v) for v in got] == [NINF, c * SERVED_N % 65536, c * SERVED_N // 65536]


def test_the_served_combine_takes_each_chunk_s_words_and_its_own_chunk_s():
    """The combine at GLM-5.3's statics with one chunk alone above -inf takes that chunk's two words as they are, each wired
    chunk's in turn; with none above, its own chunk's lane, at its first and last index; and on a tie between a wired chunk
    and its own, the wired chunk's, the smaller token index."""
    combine = G.plan(SERVED_V)[-1].defn
    K = SERVED_V // SERVED_N
    none = [[NINF, c * SERVED_N % 65536, c * SERVED_N // 65536] for c in range(K - 1)]
    for c in range(K - 1):
        results = [[ONE, 0x1234 + c, 0x0055 + c] if k == c else r for k, r in enumerate(none)]
        assert [int(v) for v in evaluate(combine, [NINF] * SERVED_N, *results)] == [0x1234 + c, 0x0055 + c], c
    for i in (0, SERVED_N - 1):
        tok = (K - 1) * SERVED_N + i
        assert [int(v) for v in evaluate(combine, _alone(SERVED_N, i), *none)] == [tok % 65536, tok // 65536], i
        tied = [[ONE, 7, 1] if k == 12 else r for k, r in enumerate(none)]
        assert [int(v) for v in evaluate(combine, _alone(SERVED_N, i), *tied)] == [7, 1], i


def test_the_widest_chunk_with_a_carry_compares_its_last_index():
    """A chunk a multiple of 2^16 falls inside is at most 0x7F81 lanes: its last index, 0x7F80, is the largest word bf16
    orders as the integers do (+inf's).  At that width the chunk's last lane is its token index's two words, in the chunk
    and in the combine's own; one lane wider is refused by both, by the plan and by the select; a power of two has no such
    chunk and is taken up to 2^16 lanes."""
    N = G.ORDERED + 1
    V = 3 * N
    assert G.carries(V, N) == {2: 65536 - 2 * N}
    want = [(V - 1) % 65536, (V - 1) // 65536]
    assert [int(v) for v in evaluate(bind(G.GreedyChunk, N=N, OFF=2 * N), _alone(N, N - 1))] == [ONE, *want]
    none = [[NINF, c * N % 65536, c * N // 65536] for c in range(2)]
    assert [int(v) for v in evaluate(bind(G.GreedyCombine, V=V, N=N), _alone(N, N - 1), *none)] == want
    for d, statics in ((G.GreedyChunk, {"N": N + 1, "OFF": 2 * (N + 1)}), (G.GreedyCombine, {"V": 3 * (N + 1), "N": N + 1}),
                       (G.GreedySelect, {"V": 3 * (N + 1), "N": N + 1}), (G.GreedySelect, {"V": 4 * (N + 1), "N": N + 1})):
        with pytest.raises(ValueError, match="a multiple of 2\\^16 falls inside it"):
            bind(d, **statics).signature()
    with pytest.raises(ValueError, match="a multiple of 2\\^16 falls inside it"):
        G.plan(3 * (N + 1), 3)
    bind(G.GreedyChunk, N=N + 1, OFF=N + 1).signature()
    bind(G.GreedyCombine, V=4 * (N + 1), N=N + 1).signature()
    for V2, N2 in ((1 << 17, 1 << 16), (1 << 18, 1 << 15)):
        bind(G.GreedyCombine, V=V2, N=N2).signature()
        bind(G.GreedyChunk, N=N2, OFF=V2 - 2 * N2).signature()


@pytest.mark.parametrize("N,OFF,t", [(3, 65535, 1), (24, 65520, 16), (146, 65408, 128)])
def test_a_carry_at_a_lane_among_the_subnormal_words_is_taken_from_that_lane_on(N, OFF, t):
    """A chunk whose carry is at a lane t <= 128 compares its index with t - 1, +0 or a subnormal word (1 .. 0x7F), where
    the served chunks' carries compare normal numbers: at t = 1 with +0, at t = 16 a subnormal with a subnormal, at t = 128
    the smallest normal number with the largest subnormal.  One lane alone above -inf at index t - 1 is the token 65,535,
    words (65,535, 0), and at index t the token 65,536, words (0, 1); a lowering compared with the reference does not say
    which the high word is."""
    chunk = bind(G.GreedyChunk, N=N, OFF=OFF)
    assert G.carry(N, OFF) == t and OFF + t == G.SPAN
    assert [int(v) for v in evaluate(chunk, _alone(N, t - 1))] == [ONE, 65535, 0]
    assert [int(v) for v in evaluate(chunk, _alone(N, t))] == [ONE, 0, 1]
    for i in sorted({0, t - 2, t + 1, N - 1} & set(range(N))):
        assert [int(v) for v in evaluate(chunk, _alone(N, i))] == [ONE, (OFF + i) % 65536, (OFF + i) // 65536], i
    assert [int(v) for v in evaluate(chunk, [NINF] * N)] == [NINF, OFF, 0]


@pytest.mark.slow
def test_the_served_units_compute_the_call_s_token():
    """Rows of GLM-5.3's vocabulary through every unit's Definition and through `TokenSelect_v1{V=154880}`: edge rows;
    `CARRIED`'s, a maximum on each side of the two lanes where a chunk's indices pass a multiple of 2^16, with a tie across
    each; and `OWN`'s, the combine's own chunk tied with a wired chunk, taken by the smaller token index, and above one.  It
    catches what the small rows can't: a chunk of 7,744 lanes, a chunk's high word and its carry, and the combine's choice
    between a wired result and its own chunk, against the Call itself; its seconds are the evaluator's, about six a row,
    twelve with its units."""
    xs = rows(SERVED_V, seed=16)
    for name in ("plain", "ties", "nan0", "tie_later"):
        want = int(evaluate(bind(b1.TokenSelect, V=SERVED_V), [int(v) for v in xs[name]])[0])
        check(SERVED_V, xs[name], want, name)
    for lanes, token in (*CARRIED, *OWN):
        x = _planted(lanes)
        want = int(evaluate(bind(b1.TokenSelect, V=SERVED_V), [int(v) for v in x])[0])
        assert want == token, lanes
        check(SERVED_V, x, want, str(lanes))


@pytest.mark.parametrize("V", [2, 16, 23, 200, 640, SERVED_V])
def test_the_plan_wires_each_result_to_its_chunk_and_every_output_is_16_bit_words(V):
    units = G.plan(V)
    chunks = units[:-1]
    K, N = G.units(V), G.width(V)
    assert K * N == V and [u.lanes for u in units] == [(c * N, (c + 1) * N) for c in range(K)] and units[-1].key == ("combine",)
    assert [u.defn.bindings for u in chunks] == [{"N": N, "OFF": c * N} for c in range(K - 1)]
    assert len({descriptor_id(u.defn) for u in units}) == K
    seen = {u.key: u.defn for u in chunks}
    params = dict(units[-1].defn.signature()[0])
    assert params["logits"].n == N and list(params) == ["logits", *(f"r{c}" for c in range(K - 1))]
    assert list(units[-1].wires) == [(f"r{c}", ("chunk", c), 0) for c in range(K - 1)]
    for p, k, _field in units[-1].wires:
        assert seen[k].signature()[1] == params[p] == Array(3, G.U16), (p, k)
    assert units[-1].defn.signature()[1] == G.select(V).signature()[1] == Array(2, G.U16)
    assert G.cut(bind(b1.TokenSelect, V=V)) == [(u.defn, 1) for u in units]


def test_the_served_cut_s_units():
    """At GLM-5.3's vocabulary the Call is `UNITS` = 20 units of 20 Definitions: nineteen chunks of 7,744 lanes, each at its
    own OFF, and the combine, which reads the last chunk, 7,744 lanes too."""
    cut = G.cut(bind(b1.TokenSelect, V=SERVED_V))
    assert [(descriptor_id(d), n) for d, n in cut] == [
        *((f"GreedyChunk_v2{{N=7744,OFF={c * 7744}}}", 1) for c in range(19)), ("GreedyCombine_v2{V=154880,N=7744}", 1)]
    assert G.units(SERVED_V) == G.UNITS == 20
    assert Counter(u.key[0] for u in G.plan(SERVED_V)) == {"chunk": 19, "combine": 1}
    assert G.plan(SERVED_V)[-1].lanes == (147136, 154880)
    assert descriptor_id(G.select(SERVED_V)) == "GreedySelect_v1{V=154880,N=7744}"
    assert b1.CUTS["TokenSelect_v1"] == G.__name__
    assert G.CUTS["GreedySelect_v1"](G.select(SERVED_V)) == cut
    assert [(descriptor_id(d), n) for d, n in G.cut(G.select(SERVED_V, 16))] == [
        *((f"GreedyChunk_v2{{N=9680,OFF={c * 9680}}}", 1) for c in range(15)), ("GreedyCombine_v2{V=154880,N=9680}", 1)]


@pytest.mark.parametrize("V,K", [(SERVED_V, 18), (SERVED_V, 3), (16, 3), (200, 0), (65537, None), (65537, 1), (1 << 18, 2),
                                 (120000, 3)])
def test_a_plan_whose_chunks_are_not_whole_or_past_2_16_lanes_is_refused(V, K):
    """A K that does not divide V (a narrower last chunk would start off the chunks' rows), one whose chunks are past
    2^16 lanes (a prime past 2^16 has no plan), and one with a chunk past 0x7F81 lanes that a multiple of 2^16 falls inside."""
    with pytest.raises(ValueError):
        G.plan(V, K)


@pytest.mark.parametrize("V,N", [(16, 3), (16, 32), (2**32 + 2**16, 2**16), (1 << 17, 1 << 17), (80000, 40000)])
def test_a_combine_whose_chunks_are_not_whole_or_whose_index_is_no_16_bit_word_is_refused(V, N):
    """A chunk width that does not divide V, wider than V or than 2^16 lanes, a row past 2^32 lanes (its token is no two
    words), and a width past 0x7F81 lanes with a multiple of 2^16 inside the combine's own chunk (its carry's compare); the
    select whose body is the plan refuses the same, and a row with such a chunk anywhere."""
    for d in (G.GreedyCombine, G.GreedySelect):
        with pytest.raises(ValueError):
            bind(d, V=V, N=N).signature()
    bind(G.GreedyCombine, V=120000, N=40000).signature()
    with pytest.raises(ValueError):
        bind(G.GreedySelect, V=120000, N=40000).signature()


@pytest.mark.parametrize("N,OFF", [(8, 3), (8, 12), (8, -8), (8, True), (8, "8"), (8, None), (8, 8.0), (1 << 16, 1 << 32),
                                   (0, 0), (1 << 17, 0), (40000, 40000)])
def test_a_chunk_off_its_row_s_chunks_is_refused(N, OFF):
    """OFF is a multiple of N (a chunk that starts off the chunks' rows is no unit of a plan), an int (an id's ``OFF=8``),
    with OFF + N at most 2^32; N is 1 to 2^16 lanes; and a chunk past 0x7F81 lanes that a multiple of 2^16 falls inside."""
    with pytest.raises(ValueError):
        bind(G.GreedyChunk, N=N, OFF=OFF).signature()
