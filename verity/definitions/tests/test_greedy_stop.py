"""`GreedyStopCheck_v1` (`verity.definitions.registry.greedy_stop`) against its reference on rows of a small V, on a token whose
high word is not 0, with a last id that differs from the token in one bit of either word, and where the length rule fails."""
from __future__ import annotations

import numpy as np
import pytest

from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate
from verity.definitions.registry import greedy_stages as G
from verity.definitions.registry import greedy_stop as X

V, N = 320, 16
NINF, ONE = 0xFF80, 0x3F80
EOS = (154820, 154827, 154829)


def _check(row: np.ndarray, results: list[list[int]], ids: list[int], **statics) -> int:
    fn = bind(X.GreedyStopCheck, V=V, N=N, T=len(ids), **statics)
    return int(evaluate(fn, [int(v) for v in row[V - N:]], *results, [i & 0xFFFFFFFF for i in ids])[0])


def _results(row: np.ndarray) -> tuple[int, list[list[int]]]:
    token, words = G.twin(V, row)
    return token, [[int(w) for w in words[u.key][0]] for u in G.plan(V)[:-1]]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_check_is_its_reference_on_rows(seed):
    """On a row of normal logits, the last id the row's token passes and that id with its low bit, its bit 16 or its bit 31
    flipped is refused, each as `reference` computes; the length rule's own refusal (FINISH 0 at max_tokens) refuses too."""
    rng = np.random.default_rng(seed)
    row = (np.asarray(rng.standard_normal(V) * 3, np.float32).view(np.uint32) >> 16).astype(np.uint16)
    token, results = _results(row)
    st = dict(STOP=EOS, MAX_TOKENS=2, PROMPT_LEN=8, MAX_MODEL_LEN=64)
    for last, finish, want in [(token, 2, 1), (token ^ 1, 2, 0), (token ^ 1 << 16, 2, 0), (token ^ 1 << 31, 2, 0), (token, 0, 0)]:
        ids = [5, last]
        assert _check(row, results, ids, FINISH=finish, **st) == want, (last, finish)
        assert X.reference(row, ids, V, N, FINISH=finish, **st) == want, (last, finish)


def test_a_token_past_2_16_compares_both_words():
    """A wired chunk's result alone above -inf with token words (0x1234, 0x0055): the id 0x00551234 passes, and one with either
    word changed is refused."""
    row = np.full(V, NINF, np.uint16)
    _token, results = _results(row)
    results[3] = [ONE, 0x1234, 0x0055]
    st = dict(STOP=(), MAX_TOKENS=1, PROMPT_LEN=8, MAX_MODEL_LEN=64, FINISH=2)
    assert _check(row, results, [0x00551234], **st) == 1
    assert _check(row, results, [0x00561234], **st) == 0
    assert _check(row, results, [0x00551235], **st) == 0
    assert _check(row, results, [0x12340055], **st) == 0


def test_the_check_refuses_the_length_rule_s_statics():
    with pytest.raises(ValueError, match="refuses this binding"):
        bind(X.GreedyStopCheck, V=V, N=N, T=2, STOP=(), MAX_TOKENS=2, PROMPT_LEN=8, MAX_MODEL_LEN=64, FINISH=1)
