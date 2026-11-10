"""`StopRule_v1` (`verity.definitions.registry.stop_rule`) evaluated as words, and `StopRule_v2` on gates, each equal to
`stop_rule.reference` on hand-picked edges and seeded vectors that include words past 2^31 (the function is total on every
word), at a single stop id, two, and GLM-5.3's three at 32 outputs; and `StopRule_v2` is `StopRule_v1`'s Boolean version."""
from __future__ import annotations

import random

import numpy as np
import pytest

from verity.circuits.boolean.trace import word_of
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate_batch
from verity.circuits.evaluation.bits import evaluate_words
from verity.definitions import boolean as SW
from verity.definitions.registry import stop_rule as X
from verity.definitions.registry.boolean_stop_rule import StopRuleV2

M32 = 0xFFFFFFFF
GLM_EOS = (154820, 154827, 154829)


def _edges(T: int, stops: tuple[int, ...]) -> list[tuple]:
    eos, other = stops[-1], 5
    plain = [other] * T
    return [
        ([eos] + plain[1:], 100, stops, 10, 131072),             # a stop id at the first position
        (plain[:-1] + [eos], 100, stops, 10, 131072),            # at the last
        (plain, 100, stops, 10, 131072),                         # no stop within T
        (plain, T, stops, 10, 131072),                           # max_tokens = T
        (plain, max(T - 1, 1), stops, 10, 131072),               # max_tokens inside T
        (plain[:-1] + [eos], T, stops, 10, 131072),              # a stop id at max_tokens: the stop wins
        (plain, 100, stops, 131072 - T, 131072),                 # the model's length at the last position
        (plain, 100, stops, 131072 - 1, 131072),                 # at the first
        (plain, 100, stops, 131072, 131072),                     # a prompt already at the model's length
        (plain, 0, stops, 10, 131072),                           # max_tokens 0 never stops
        (plain, 1, stops, 131072, 131072),                       # max_tokens and the model's length at once
        ([stops[0]] * T, 100, stops, 10, 131072),                # every position a stop id
        (plain, 0x80000000, stops, 10, 131072),                  # max_tokens past 2^31: negative, never stops
        (plain, 100, stops, 0xFFFFFFFF, 3),                      # prompt_len −1 as a signed word: 3 is reached at 4
    ]


def _seeded(T: int, K: int, n: int, seed: int) -> list[tuple]:
    rng = random.Random(seed)
    pool = [0, 1, 2, 7, 154820, 0x7FFFFFFF, 0x80000000, M32]
    out = []
    for _ in range(n):
        stops = tuple(rng.choice(pool) for _ in range(K))
        ids = [rng.choice(pool + list(stops)) for _ in range(T)]
        mt = rng.choice([0, 1, 2, T - 1, T, T + 1, 0x7FFFFFFF, 0x80000000, M32])
        p = rng.choice([0, 1, 9, 131071, 0x7FFFFFF0, M32])
        mml = rng.choice([0, 1, 10, 11, 12, 131072, 0x7FFFFFFF, 0x80000001])
        out.append((ids, mt, stops, p, mml))
    return out


def _args(cases: list[tuple]) -> list[np.ndarray]:
    return [np.array([c[k] for c in cases], dtype=np.uint32) for k in range(5)]


@pytest.mark.parametrize("T,stops", [(1, (2,)), (5, (2, 3)), (32, GLM_EOS)])
def test_word_and_gates_are_the_reference(T: int, stops: tuple[int, ...]) -> None:
    K = len(stops)
    cases = _edges(T, stops) + _seeded(T, K, 300, seed=T * 16 + K)
    want = [X.reference(*c) for c in cases]
    word = np.asarray(evaluate_batch(bind(X.StopRule, T=T, NSTOP=K), *_args(cases), kernel="reference"))
    gates = np.asarray(evaluate_words(bind(StopRuleV2, T=T, NSTOP=K), *_args(cases)))
    assert [tuple(int(v) for v in r) for r in word] == want
    assert [tuple(int(v) for v in r) for r in gates] == want
    assert {f for _, f in want} == {0, 1, 2, 3}


def test_reference_on_the_edges() -> None:
    T, stops = 4, GLM_EOS
    got = [X.reference(*c) for c in _edges(T, stops)]
    assert got == [(1, 1), (4, 1), (4, 0), (4, 2), (3, 2), (4, 1), (4, 3), (1, 3), (1, 3), (4, 0), (1, 2), (1, 1), (4, 0),
                   (4, 3)]


def test_the_boolean_form_is_the_word_view() -> None:
    v2 = bind(StopRuleV2, T=32, NSTOP=3)
    assert word_of(v2).id == "StopRule_v1{T=32,NSTOP=3}"
    assert SW.boolean_version(word_of(v2)).id == v2.id


@pytest.mark.parametrize("statics", [dict(T=0, NSTOP=1), dict(T=4, NSTOP=0), dict(T=True, NSTOP=1)])
def test_refuses_a_binding_with_no_output_or_no_stop_id(statics: dict) -> None:
    for d in (X.StopRule, StopRuleV2):
        with pytest.raises(ValueError, match="refuses this binding"):
            bind(d, **statics)
