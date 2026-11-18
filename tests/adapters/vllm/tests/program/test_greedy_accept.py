"""`GreedyAccept_v1{K,V}` held to `numerics.greedy_accept_twin`, the numpy twin of vLLM d9105ea80's greedy rejection kernels:
ties inside a tile and across tiles, +0 against -0, -inf rows, the short last tile, the -1 placeholder, every accept count, and
the kernels' tile sizes at GLM-5.3's statics. The twin's NaN handling: a NaN tile maximum is -inf across tiles, and a row with a
NaN inside a tile is declined."""
from __future__ import annotations

import numpy as np
import pytest
from verity.circuits import bind
from verity.circuits.evaluation import evaluate_call
from verity_vllm.definitions.numerics import greedy_accept_twin as GT
from verity_vllm.definitions.registry import spec

from tests.program.crosscheck import _standalone

POS0, NEG0, NEG_INF, NAN = 0x0000, 0x8000, 0xFF80, 0x7FC0


def bf16(x: float) -> int:
    return int(np.array([x], dtype=np.float32).view(np.uint32)[0] >> 16)


def _signed(w: int) -> int:
    return w - (1 << 32) if w & 0x80000000 else w


def definition(K: int, V: int):
    prog, call = _standalone(bind(spec.GreedyAccept, K=K, V=V))

    def run(draft, logits) -> tuple[list[int], int, list[int]]:
        out = evaluate_call(prog.circuit, call, [d & 0xFFFFFFFF for d in draft] + [int(w) for w in np.asarray(logits).ravel()])
        return list(out[:K]), out[K], [_signed(w) for w in out[K + 1:]]
    return run


def _row(V: int, rng: np.random.Generator, at: dict[int, int] | None = None) -> np.ndarray:
    """Negative finite bf16 words, with ``at``'s words placed."""
    row = (0x8000 | rng.integers(0x3F80, 0x4100, V)).astype(np.uint16)
    for i, w in (at or {}).items():
        row[i] = w
    return row


V, TILES = 200, (64, 16)
ONE, TWO = bf16(1.0), bf16(2.0)
#: a row each, (name, {index: word}) over a row of negative values: where its maximum is and what it ties with
ROWS = [
    ("max-in-first-tile", {3: TWO}),
    ("max-in-short-last-tile", {199: TWO}),
    ("tie-in-one-tile", {10: TWO, 40: TWO}),
    ("tie-across-tiles", {130: TWO, 70: TWO, 199: TWO}),
    ("pos0-before-neg0", {5: POS0, 150: NEG0}),
    ("neg0-before-pos0", {5: NEG0, 150: POS0}),
    ("neg0-at-0-pos0-in-last-tile", {0: NEG0, 192: POS0}),
    ("inf-and-max", {0: 0x7F80, 100: 0x7F80}),
]


def _case(rng: np.random.Generator, K: int, rows: list[dict[int, int]], accepted: int, placeholder: bool = False):
    """K drafts against K + 1 rows: the first ``accepted`` drafts are their rows' argmaxes, the next one is not (or the -1
    placeholder)."""
    logits = np.stack([_row(V, rng, r) for r in rows])
    target = [GT.row_argmax(logits[j], TILES[0]) for j in range(K)]
    draft = [target[j] if j < accepted else (-1 if placeholder and j == accepted else (target[j] + 1) % V) for j in range(K)]
    return draft, logits


@pytest.mark.parametrize("name,at", ROWS, ids=[n for n, _ in ROWS])
def test_a_rows_argmax_is_the_first_index_of_its_maximum_in_the_definition_and_the_twin(name, at):
    """Each row's argmax is the first index of its maximum, -0 equal to +0, in the Definition and in the twin at either tile
    size, the bonus row's included."""
    rng = np.random.default_rng(len(name))
    K = 2
    rows = [at, {}, at]
    logits = np.stack([_row(V, rng, r) for r in rows])
    want = int(np.argmax(GT.f32_of_bf16(logits[0]) == GT.f32_of_bf16(logits[0]).max()))
    for tile in (*TILES, GT.STATS_TILE, GT.RESAMPLE_TILE):
        assert GT.row_argmax(logits[0], tile) == want
    draft = [want, GT.row_argmax(logits[1], TILES[0])]
    got = definition(K, V)(draft, logits)
    assert got == GT.greedy_accept(draft, logits, TILES) == ([1, 1], 2, [want, draft[1], want])


def test_every_row_negative_infinity_selects_token_0():
    rows = np.full((3, V), NEG_INF, dtype=np.uint16)
    assert GT.row_argmax(rows[0], TILES[0]) == GT.row_argmax(rows[0], GT.STATS_TILE) == 0
    assert definition(2, V)([0, 0], rows) == GT.greedy_accept([0, 0], rows, TILES) == ([1, 1], 2, [0, 0, 0])


@pytest.mark.parametrize("accepted", range(6))
@pytest.mark.parametrize("placeholder", [False, True])
def test_the_definition_is_the_twin_at_every_accept_count(accepted, placeholder):
    """Five drafts: the step keeps the argmaxes of rows 0..accepted, the bonus row's when all five are accepted, and -1 past
    them; the -1 placeholder is rejected as a wrong draft is."""
    if placeholder and accepted == 5:
        pytest.skip("no draft left to be the placeholder")
    rng = np.random.default_rng(accepted + 10 * placeholder)
    K = 5
    rows = [ROWS[j % len(ROWS)][1] for j in range(K + 1)]
    draft, logits = _case(rng, K, rows, accepted, placeholder)
    want = GT.greedy_accept(draft, logits, TILES)
    assert want[1] == accepted
    assert definition(K, V)(draft, logits) == want


def test_a_nan_tile_maximum_is_negative_infinity_across_tiles():
    """`_compute_global_target_argmax` and `_insert_resampled_kernel` map a NaN tile maximum to -inf before `tl.argmax`."""
    best = np.array([np.nan, 1.0, 3.0, np.nan], dtype=np.float32)
    assert GT.across_tiles(best, np.array([5, 70, 130, 199])) == 130
    assert GT.across_tiles(np.array([np.nan, -np.inf], dtype=np.float32), np.array([7, 70])) == 7


def test_a_row_with_a_nan_inside_a_tile_is_declined():
    """In a tile with a NaN, `_argmax_combine`'s result depends on the reduction's order, which the twin does not state."""
    row = _row(V, np.random.default_rng(1), {3: TWO, 90: NAN})
    with pytest.raises(GT.Declined, match=r"tiles \[1\] of 64 logits hold a NaN"):
        GT.row_argmax(row, TILES[0])
    with pytest.raises(GT.Declined):
        GT.greedy_accept([3], np.stack([row, row]), TILES)


@pytest.mark.slow
def test_at_glm53s_statics_the_definition_is_the_twin_at_the_kernels_tiles():
    """GLM-5.3's five drafts over 154,880 logits, where the twin's tiles are the kernels' 8,192 and 1,024: ties across a tile
    boundary of each, +0 against -0, and the maximum in the short last tile, at two accept counts. About 45 s; the only test
    that evaluates the Definition at the served V, where a row is 19 tiles of the draft rows' size and 152 of the bonus row's."""
    K, Vg = 5, 154880
    run = definition(K, Vg)
    rng = np.random.default_rng(53)
    at = [{8191: TWO, 8192: TWO}, {1023: ONE, 1024: TWO, 16384: TWO}, {100: NEG0, 9000: POS0}, {154879: TWO},
          {0: ONE, 154000: ONE}, {1023: TWO, 1024: TWO, 2047: TWO}]
    logits = np.stack([(0x8000 | rng.integers(0x3F80, 0x4100, Vg)).astype(np.uint16) for _ in range(K + 1)])
    for j, a in enumerate(at):
        for i, w in a.items():
            logits[j, i] = w
    target = [GT.row_argmax(logits[j], GT.STATS_TILE) for j in range(K + 1)]
    assert target == [8191, 1024, 100, 154879, 0, 1023]
    assert GT.row_argmax(logits[K], GT.RESAMPLE_TILE) == 1023
    for accepted in (5, 2):
        draft = [target[j] if j < accepted else target[j] + 1 for j in range(K)]
        want = GT.greedy_accept(draft, logits)
        assert want[1] == accepted and run(draft, logits) == want
