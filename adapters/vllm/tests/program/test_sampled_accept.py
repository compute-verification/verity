"""`registry.sampled_accept`: each Definition computes its numpy reference's words, on the hand vectors and on random rows.

The reference (`sampled_accept.tile_stats`, `row_lse`, `accept_uniform`, `sampled_accept`) is written over the kernels'
layouts with `topp_split`'s reduction trees and `sampling_rows`' Gumbel row; the Definitions' bodies are written gate by
gate.  The whole Definition runs at 17 tiles of 512 logits, the smallest layout it takes (the last tile 16 logits), with two
draft tokens; a tile at the pin's 8192 logits and at two and four vectors a thread, where a thread's own sum has an order.
"""
from __future__ import annotations

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


def test_the_outputs_are_accept_v1s():
    for k in (1, 2, 5):
        assert bind(A.SampledAccept, K=k, V=V, TILE=TILE).signature()[1] == bind(spec.Accept, K=k).signature()[1]


@pytest.mark.parametrize("n, tile", [(512, 512), (16, 512), (496, 512), (1024, 1024), (2048, 2048), (1536, 2048), (8192, 8192)])
def test_tile_stats_is_the_kernels_tile(n, tile):
    """A tile's maximum and summed exponentials at one, two, four and sixteen vectors a thread, whole and cut short: dense
    rows, a row of one logit, a row of none (its sum the select's 0) and a row whose maximum is 30 above the rest."""
    rng = np.random.default_rng(n + tile)
    rows = [rng.normal(0, 3, n).astype(F), np.full(n, -np.inf, F), np.full(n, -np.inf, F), (rng.normal(0, 1, n) - 30).astype(F)]
    rows[1][n // 2] = 1.5
    rows[3][n - 1] = 2.0
    fn = bind(A.AcceptTileStats, N=n, TILE=tile)
    for row in rows:
        padded = np.concatenate([row, np.full(tile - n, -np.inf, F)])
        m, s = A.tile_stats(padded, tile)
        assert evaluate(fn, _words(row)) == (_words(m)[0], _words(s)[0])
    assert evaluate(fn, _words(rows[2])) == (NEG_INF, 0)


@pytest.mark.parametrize("v", [16 * TILE + 16, 18 * TILE + 48, 32 * TILE])
def test_row_lse_is_lane_0s(v):
    rng = np.random.default_rng(v)
    fn = bind(A.AcceptRowLse, V=v, TILE=TILE)
    for row in (rng.normal(0, 3, v).astype(F), _rows(rng, 1, v, 0.01, 2.0)[0]):
        assert evaluate(fn, _words(row)) == (A.row_lse(*A.tile_stats(row, TILE))[0],)


def test_row_lse_is_lane_0s_where_the_two_halves_differ():
    row = VECTORS["halves-split"]["logits"][0]
    low, high = A.row_lse(*A.tile_stats(row, TILE))
    assert high == low + 1
    assert evaluate(bind(A.AcceptRowLse, V=V, TILE=TILE), _words(row)) == (low,)


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


@pytest.mark.parametrize("v", [1, 2, 16, 33])
def test_logit_at_and_the_residual_index_their_row(v):
    rng = np.random.default_rng(v)
    row = _words(rng.normal(0, 3, v))
    for d in (0, v - 1, v // 2, v, v + 5, 0x7FFFFFFF, 0x80000000, M32):
        inside = d < v
        assert evaluate(bind(A.LogitAt, V=v), row, d) == (row[d] if inside else NEG_INF,)
        assert evaluate(bind(A.AcceptResidual, V=v), row, d) == tuple(NEG_INF if j == d else w for j, w in enumerate(row))


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
    for fn in (bind(A.SampledAccept, K=K, V=V, TILE=TILE), bind(A.AcceptTileStats, N=16, TILE=TILE), A.AcceptUniform,
               bind(A.LogitAt, V=1), bind(A.AcceptResidual, V=1)):
        assert dead_gates(Program(fn), transitive=True).computed == 0, fn.id
