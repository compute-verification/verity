"""`registry.sampled_accept`: each Definition computes its numpy reference's words, on the hand vectors and on random rows.

The reference (`sampled_accept.tile_stats`, `row_lse`, `accept_uniform`, `sampled_accept`) is written over the kernels'
layouts with `topp_split`'s reduction trees and `sampling_rows`' Gumbel row; the Definitions' bodies are written gate by
gate.  The Call runs at 17 tiles of 512 logits, the smallest layout it takes (the last tile 16 logits), with two draft
tokens, and at tiles of 1,024 where a tile and the row end are two parts; its units alone at the pin's 8,192 logits a tile
and at two and four vectors a thread, where a thread's own sum has an order.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from tests.program import sampled_accept_vectors as SV
from verity.circuits.defs import bind
from verity.circuits.evaluation.reference import evaluate
from verity.circuits.liveness import dead_gates
from verity.circuits.program import Program
from verity.definitions.registry import sampled_accept as A, sampling as SA, spec
from verity.definitions.registry.sampling_rows import gumbel_select_row

F = np.float32
M32 = 0xFFFFFFFF
TILE = 512
K, V = 2, 16 * TILE + 16
NEG_INF = SA.NEG_INF_BITS
VECTORS = {v["name"]: v for v in SV.vectors(K, V, TILE)}


def _words(x) -> list[int]:
    return [int(w) for w in np.ascontiguousarray(x, dtype=F).view(np.uint32).reshape(-1)]


def _whole(logits, draft, seed, pos, temp, tile=TILE) -> tuple:
    k, v = len(draft), np.asarray(logits).shape[1]
    return evaluate(bind(A.SampledAccept, K=k, V=v, TILE=tile), _words(logits), [d & M32 for d in draft], seed, pos, temp)


def _outputs(ref: dict) -> tuple:
    return (*ref["verdict"], ref["accepted"], *(t & M32 for t in ref["appended"]))


def _rows(rng, n: int, v: int, keep: float, spread: float) -> np.ndarray:
    rows = rng.normal(0, spread, (n, v)).astype(F)
    rows[rng.random((n, v)) >= keep] = -np.inf
    rows[:, rng.integers(v)] = 0.0
    return rows


def test_the_hand_vectors_are_the_nine_edges_and_the_split():
    assert list(VECTORS) == ["accepted-everywhere", "refused-everywhere", "draft-outside-top-p", "draft-placeholder",
                             "logprob-equals-log-u", "u-at-floor-accepted", "logprob-at-log-u", "logprob-above-log-u",
                             "refused-draft-masked", "halves-split"]
    assert SA.stream_key(SV.FLOOR_SEED, SV.FLOOR_POS) == 0
    floor = VECTORS["u-at-floor-accepted"]["ref"]
    assert floor["u"][K - 1] == SA.TL_RAND_MIN_BITS and floor["logprob"][K - 1] == floor["log_u"][K - 1] - 1
    assert [name for name, v in VECTORS.items() if v["ref"]["halves_split"] is not None] == ["halves-split"]


@pytest.mark.parametrize("name", list(VECTORS))
def test_sampled_accept_equals_its_reference_on_a_hand_vector(name):
    v = VECTORS[name]
    assert _whole(v["logits"], v["draft"], v["seed"], v["pos"], v["temp"]) == _outputs(v["ref"])


def test_sampled_accept_equals_its_reference_on_random_rows():
    """Rows of a few hundred finite logits over 19 tiles, three draft tokens, each the row's maximum or a random lane."""
    rng = np.random.default_rng(20261112)
    v = 18 * TILE + 48
    for case in range(2):
        logits = _rows(rng, 4, v, 0.03, 2.5)
        draft = [int(np.argmax(logits[i])) if rng.random() < 0.7 else int(rng.integers(v)) for i in range(3)]
        seed, pos, temp = int(rng.integers(1 << 62)), [int(p) for p in 40 + case + np.arange(4)], int(F(0.8).view(np.uint32))
        assert _whole(logits, draft, seed, pos, temp) == _outputs(A.sampled_accept(logits, draft, seed, pos, temp, TILE))


@pytest.mark.parametrize("name", ["accepted-everywhere", "logprob-equals-log-u", "refused-draft-masked"])
def test_sampled_accept_equals_its_reference_where_a_tile_is_two_parts(name):
    """Tiles of 1,024 logits, each two parts of 512, and a last tile of 528: the row end reads its second part's 16 lanes and
    the thread sums the first part hands on."""
    tile, v = 1024, 16 * 1024 + 528
    assert [u.defn.id for u in A.plan(K, v, tile) if u.key[0] == "row"] == ["AcceptRowEnd_v1{T=17,N=16,AT=512}"] * K
    vec = {x["name"]: x for x in SV.vectors(K, v, tile)}[name]
    assert _whole(vec["logits"], vec["draft"], vec["seed"], vec["pos"], vec["temp"], tile) == _outputs(vec["ref"])


def test_the_outputs_are_accept_v1s():
    for k in (1, 2, 5):
        assert bind(A.SampledAccept, K=k, V=V, TILE=TILE).signature()[1] == bind(spec.Accept, K=k).signature()[1]


def _tile(row, tile: int) -> tuple[int, int]:
    """A tile's maximum and sum by its units: the head, then its parts of `part` lanes, each handing its thread sums on."""
    w, n, step = _words(row), len(row), A.part(tile)
    m = evaluate(bind(A.AcceptTileHead, N=n, FIRST=True, LAST=False), w, 0)[0]
    acc: tuple = ()
    for at in range(0, n, step):
        pn = min(step, n - at)
        acc = evaluate(bind(A.AcceptTileSum, N=pn, AT=at, LAST=at + pn == n), w[at:at + pn], m, *([list(acc)] if at else []))
    return m, acc[0]


@pytest.mark.parametrize("n, tile", [(512, 512), (16, 512), (496, 512), (1024, 1024), (2048, 2048), (1536, 2048), (1552, 2048),
                                     (8192, 8192), (7424, 8192)])
def test_a_tiles_head_and_parts_are_the_kernels_tile(n, tile):
    """A tile's maximum and summed exponentials at one, two, four and sixteen vectors a thread, whole and cut short, in one
    part, two and three (a last part of whole vectors, and one of 16 lanes that leaves most threads' carried sums as they
    are): dense rows, a row of one logit, a row of none (its sum the select's 0) and a row whose maximum is 30 above the rest."""
    rng = np.random.default_rng(n + tile)
    rows = [rng.normal(0, 3, n).astype(F), np.full(n, -np.inf, F), np.full(n, -np.inf, F), (rng.normal(0, 1, n) - 30).astype(F)]
    rows[1][n // 2] = 1.5
    rows[3][n - 1] = 2.0
    for row in rows:
        padded = np.concatenate([row, np.full(tile - n, -np.inf, F)])
        m, s = A.tile_stats(padded, tile)
        assert _tile(row, tile) == (_words(m)[0], _words(s)[0])
    assert _tile(rows[2], tile) == (NEG_INF, 0)


def _row_end(logits, d: int, seed: int, pos: int, tile: int = TILE) -> int:
    """`AcceptRowEnd_v1` on a row whose last tile is one part: its other tiles' maxima and sums from the reference."""
    m, s = A.tile_stats(logits, tile)
    t, v = len(m), len(logits)
    n = v - (t - 1) * tile
    assert n <= A.part(tile)
    others = [w for pair in zip(_words(m[:-1]), _words(s[:-1])) for w in pair]
    at = _words(logits[d])[0] if 0 <= d < v else NEG_INF
    return evaluate(bind(A.AcceptRowEnd, T=t, N=n, AT=0), _words(logits[v - n:]), _words(m[-1])[0], at, seed, pos, *others)[0]


@pytest.mark.parametrize("name", list(VECTORS))
def test_the_row_end_is_lane_0s_ratio_test_on_a_hand_vector(name):
    """Every draft row of every hand vector: the rows whose log-probability is log(u), one place below and one above it,
    the uniform at its floor, and the row lanes 0 and 16 disagree on."""
    v = VECTORS[name]
    for i, d in enumerate(v["draft"]):
        assert _row_end(v["logits"][i], d, v["seed"], v["pos"][i]) == v["ref"]["test"][i]


@pytest.mark.parametrize("v", [16 * TILE + 16, 18 * TILE + 48, 32 * TILE])
def test_the_row_end_is_lane_0s_ratio_test_at_17_19_and_32_tiles(v):
    rng = np.random.default_rng(v)
    for case, row in enumerate((rng.normal(0, 3, v).astype(F), _rows(rng, 1, v, 0.01, 2.0)[0])):
        for d in (int(np.argmax(row)), int(rng.integers(v))):
            ref = A.sampled_accept(np.stack([row, row]), [d], 77 + case, [9, 10], int(F(1.0).view(np.uint32)), TILE)
            assert _row_end(row, d, 77 + case, 9) == ref["test"][0]


def test_the_row_end_is_lane_0s_where_the_two_halves_differ():
    v = VECTORS["halves-split"]
    low, high = A.row_lse(*A.tile_stats(v["logits"][0], TILE))
    assert high == low + 1 and v["ref"]["halves_split"] == 0
    assert _row_end(v["logits"][0], v["draft"][0], v["seed"], v["pos"][0]) == 1


def test_accept_uniform_is_tl_rand_with_its_floor():
    rng = np.random.default_rng(5)
    keys = [0, 1, 2, 3, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFD, 0xFFFFFFFE, 0xFFFFFFFF, 0x7FFFFF80, 0x7FFFFFBF, 0x7FFFFFC0]
    keys += [int(k) for k in rng.integers(0, 1 << 32, 64)]
    for key in keys:
        u = A.accept_uniform(key)
        assert evaluate(A.AcceptUniform, key) == (u,)
        assert SA.TL_RAND_MIN_BITS <= u < 0x3F800000
    assert {A.accept_uniform(k) for k in (0, 1, 0xFFFFFFFE, 0xFFFFFFFF)} == {SA.TL_RAND_MIN_BITS}
    assert A.accept_uniform(2) > SA.TL_RAND_MIN_BITS
    # the uniform is the Gumbel noise's own: the same word, sign fold, conversion, scale and floor
    assert all(A.accept_uniform(SA.philox4x32_10_word0(v, 0, 0, 0, key, 0)) == SA._rand_lane(key, v) for key in keys[:8] for v in (0, 7))


def test_exp_lane_is_tl_exp_of_the_difference():
    xs = np.array([0.0, -0.0, 1.5, -1.5, -87.0, -88.0, -103.0, -104.5, -150.0, -1e30, -np.inf, 3.0e-39], F)
    for m in (F(0.0), F(2.5), F(-7.25)):
        want = A.tl_exp(xs - m)
        for x, e in zip(xs, want):
            assert evaluate(A.AcceptExpLane, _words(x)[0], _words(m)[0]) == (_words(e)[0],)
    assert evaluate(A.AcceptExpLane, _words(2.5)[0], _words(2.5)[0]) == (0x3F800000,)
    assert evaluate(A.AcceptExpLane, NEG_INF, _words(2.5)[0]) == (0,)


@pytest.mark.parametrize("sizes", [(1,), (2, 1), (16, 16, 1), (5, 7, 4)])
def test_the_heads_carry_the_logit_at_the_draft_token(sizes):
    """A row cut into tiles of the given lanes: the last head's logit is the row's at d, -inf for a d that is no index (the
    placeholder -1 among them), and each head's maximum is its tile's."""
    v = sum(sizes)
    rng = np.random.default_rng(v)
    row = _words(rng.normal(0, 3, v))
    for d in (0, v - 1, v // 2, v, v + 5, 0x7FFFFFFF, 0x80000000, M32):
        carry, lo = (), 0
        for t, n in enumerate(sizes):
            last = t == len(sizes) - 1
            out = evaluate(bind(A.AcceptTileHead, N=n, FIRST=t == 0, LAST=last), row[lo:lo + n], d, *carry)
            assert _f(out[0]) == max(_f(w) for w in row[lo:lo + n])
            carry, lo = out[1:], lo + n
            assert last or carry[0] == lo
        assert carry == (row[d] if d < v else NEG_INF,)


def _f(word: int) -> float:
    return float(np.uint32(word).view(F))


def _chunks(rows, tests, draft, seed, pos, temp, n: int) -> tuple:
    """`AcceptResample_v1` over rows [K+1, V] in chunks of n lanes, each handing its carry on: the last chunk's outputs."""
    k, v = len(draft), rows.shape[1]
    out: tuple = ()
    for lo in range(0, v, n):
        hi = min(lo + n, v)
        fn = bind(A.AcceptResample, K=k, N=hi - lo, FIRST=lo == 0, LAST=hi == v)
        out = evaluate(fn, *(_words(r[lo:hi]) for r in rows), *tests, [d & M32 for d in draft], seed, pos, temp,
                       *([list(out)] if lo else []))
    return out


@pytest.mark.parametrize("n", [4, 3, 11])
def test_the_resample_chunks_select_the_row_mask_it_and_run_the_gumbel_select(n):
    """Rows of 11 lanes in chunks of 4, 4, 3; of 3, 3, 3, 2; and whole: every pattern of two tests, draft tokens that are the
    refused row's best lane, a placeholder and a token past the row, at a temperature and at zero.  The token is the Gumbel
    select of the first refused row with its draft token's lane at -inf, the bonus row's when none is refused."""
    rng = np.random.default_rng(n)
    v, seed, pos = 11, 123456789, [70, 71, 72]
    for case in range(3):
        rows = rng.normal(0, 2, (3, v)).astype(F)
        rows[rng.random((3, v)) < 0.3] = -np.inf
        rows[:, case] = 1.0
        for draft in ([int(np.argmax(rows[0])), int(np.argmax(rows[1]))], [-1, 4], [v + 3, 0]):
            for tests in ((0, 0), (1, 0), (0, 1), (1, 1)):
                for temp in (int(F(0.7).view(np.uint32)), 0):
                    r = 0 if not tests[0] else 1 if not tests[1] else 2
                    residual = rows[r].copy()
                    if r < 2 and 0 <= draft[r] < v:
                        residual[draft[r]] = -np.inf
                    token, _ = gumbel_select_row(np.ascontiguousarray(residual).view(np.uint32), seed, pos[r], temp)
                    verdict = [int(r > 0), int(r > 1)]
                    appended = [draft[j] & M32 if j < r else token if j == r else M32 for j in range(3)]
                    assert _chunks(rows, tests, draft, seed, pos, temp, n) == (*verdict, r, *appended)


def test_the_resample_is_the_gumbel_select_of_the_masked_row():
    v = VECTORS["refused-draft-masked"]
    at = K - 1
    row = v["logits"][at].copy()
    whole, _ = gumbel_select_row(np.ascontiguousarray(row).view(np.uint32), v["seed"], v["pos"][at], v["temp"])
    assert whole == v["draft"][at] != v["ref"]["token"]
    row[v["draft"][at]] = -np.inf
    assert gumbel_select_row(np.ascontiguousarray(row).view(np.uint32), v["seed"], v["pos"][at], v["temp"])[0] == v["ref"]["token"]


@pytest.mark.parametrize("statics, why", [
    (dict(K=2, V=16 * TILE, TILE=TILE), "17 to 32 tiles"),
    (dict(K=2, V=32 * TILE + 16, TILE=TILE), "17 to 32 tiles"),
    (dict(K=2, V=16 * TILE + 8, TILE=TILE), "multiple of 16"),
    (dict(K=2, V=17 * 500, TILE=500), "multiple of 512"),
    (dict(K=0, V=V, TILE=TILE), "positive number of draft tokens"),
])
def test_sampled_accept_refuses_a_layout_the_kernels_were_not_read_at(statics, why):
    with pytest.raises(ValueError, match=why):
        bind(A.SampledAccept, **statics)


def test_sampled_accept_has_no_dead_gate():
    units = [A.AcceptUniform, bind(A.AcceptTileHead, N=1, FIRST=True, LAST=True), bind(A.AcceptTileHead, N=3, FIRST=False, LAST=False),
             bind(A.AcceptTileHead, N=3, FIRST=False, LAST=True), bind(A.AcceptTileSum, N=16, AT=0, LAST=True),
             bind(A.AcceptTileSum, N=16, AT=512, LAST=True), bind(A.AcceptRowEnd, T=17, N=16, AT=0),
             bind(A.AcceptResample, K=2, N=2, FIRST=True, LAST=False), bind(A.AcceptResample, K=2, N=1, FIRST=False, LAST=False),
             bind(A.AcceptResample, K=2, N=2, FIRST=False, LAST=True), bind(A.AcceptResample, K=1, N=1, FIRST=False, LAST=True)]
    for fn in (bind(A.SampledAccept, K=K, V=V, TILE=TILE), *units):
        assert dead_gates(Program(fn), transitive=True).computed == 0, fn.id


def test_the_plan_reads_every_lane_once_and_writes_each_unit_before_its_readers():
    """GLM-5.3's layout with five draft tokens: nine Definitions; each draft row's lanes once in a head and once in a part
    or the row end, every row's lanes once in the resample; a wire names a unit already written and one of its outputs."""
    k, v = 5, 154880
    plan = A.plan(k, v)
    n = Counter(u.defn.id for u in plan)
    assert n == {
        "AcceptTileHead_v1{N=8192,FIRST=True,LAST=False}": k, "AcceptTileHead_v1{N=8192,FIRST=False,LAST=False}": 17 * k,
        "AcceptTileHead_v1{N=7424,FIRST=False,LAST=True}": k, "AcceptTileSum_v1{N=4096,AT=0,LAST=False}": 19 * k,
        "AcceptTileSum_v1{N=4096,AT=4096,LAST=True}": 18 * k, "AcceptRowEnd_v1{T=19,N=3328,AT=4096}": k,
        "AcceptResample_v1{K=5,N=32,FIRST=True,LAST=False}": 1, "AcceptResample_v1{K=5,N=32,FIRST=False,LAST=False}": 4838,
        "AcceptResample_v1{K=5,N=32,FIRST=False,LAST=True}": 1}
    assert dict(A.cut(bind(A.SampledAccept, K=k, V=v, TILE=A.STATS_TILE))) == dict(Counter(u.defn for u in plan))
    seen: dict = {}
    read = {kind: np.zeros((k + 1, v), np.int64) for kind in ("head", "sum", "resample")}
    for u in plan:
        params = [name for name, _ in u.defn.signature()[0]]
        assert sorted(params) == sorted([name for name, _ in u.reads] + [name for name, _, _ in u.wires]), u.key
        for _, (name, *at) in u.reads:
            if name == "logits":
                read["sum" if u.key[0] == "row" else u.key[0]][at[0], at[1]:at[2]] += 1
        for _, key, i in u.wires:
            ret = seen[key].signature()[1]
            assert i < (len(ret.fields) if hasattr(ret, "fields") else 1), (u.key, key, i)
        seen[u.key] = u.defn
    assert (read["head"][:k] == 1).all() and (read["sum"][:k] == 1).all() and (read["resample"] == 1).all()
    assert not read["head"][k].any() and not read["sum"][k].any()
    assert plan[-1].key == ("resample", v // A.RESAMPLE_N - 1)


@pytest.mark.parametrize("fn, statics, why", [
    (A.AcceptTileSum, dict(N=100, AT=0, LAST=False), "whole vectors of every thread"),
    (A.AcceptTileSum, dict(N=16, AT=100, LAST=True), "multiple of 512"),
    (A.AcceptTileHead, dict(N=0, FIRST=True, LAST=False), "positive int"),
    (A.AcceptTileHead, dict(N=4, FIRST=1, LAST=False), "FIRST is a bool"),
    (A.AcceptRowEnd, dict(T=16, N=16, AT=0), "17 to 32"),
    (A.AcceptResample, dict(K=0, N=4, FIRST=True, LAST=False), "positive number of draft tokens"),
    (A.AcceptResample, dict(K=1, N=1, FIRST=True, LAST=True), "one lane"),
])
def test_a_unit_refuses_statics_it_has_no_layout_for(fn, statics, why):
    with pytest.raises(ValueError, match=why):
        bind(fn, **statics)
