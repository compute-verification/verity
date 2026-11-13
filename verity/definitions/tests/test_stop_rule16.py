"""`StopRule16_v1` (`verity.definitions.registry.stop_rule`) evaluated as words, and `StopRule16_v2` on gates: each is
`stop_rule.reference16`, the word n + finish * 8192 of `stop_rule.reference`'s (n, finish), on `test_stop_rule`'s edges and
seeded vectors; the word on each finish code and at both ends of n; the reference alone at 1 and at 8,191 ids, the most the
word holds; the refusal past them; and `StopRule16_v2` is `StopRule16_v1`'s Boolean version, one output of 16 bits."""
from __future__ import annotations

import numpy as np
import pytest
from test_stop_rule import GLM_EOS, _args, _edges, _seeded

from verity.circuits import boolean as BL
from verity.circuits.boolean.trace import word_of
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate_batch
from verity.circuits.evaluation.bits import evaluate_words
from verity.circuits.types import Array, Value
from verity.definitions import boolean as SW
from verity.definitions.registry import stop_rule as X
from verity.definitions.registry.boolean_stop_rule import StopRule16V2

T_MAX = 8191


@pytest.mark.parametrize("T,stops", [(1, (2,)), (5, (2, 3)), (32, GLM_EOS), (T_MAX, GLM_EOS)])
def test_the_reference_is_n_plus_finish_times_8192(T: int, stops: tuple[int, ...]) -> None:
    cases = _edges(T, stops) + _seeded(T, len(stops), 40, seed=T * 16 + len(stops))
    for c in cases:
        n, finish = X.reference(*c)
        w = X.reference16(*c)
        assert w == n + finish * 8192 and (w & 8191, w >> 13) == (n, finish) and w < 1 << 15
    assert {X.reference16(*c) >> 13 for c in cases} == {0, 1, 2, 3}


def test_the_word_on_each_finish_and_at_both_ends_of_n() -> None:
    T, stops, far = 4, GLM_EOS, 131072
    eos, plain = stops[-1], [5] * 4
    cases = {
        "runs on: n = T, finish 0": ((plain, 100, stops, 10, far), 4),
        "a stop id at the first position: n = 1, finish 1": (([eos] + plain[1:], 100, stops, 10, far), 1 + 8192),
        "a stop id at the last position: n = T, finish 1": ((plain[:-1] + [eos], 100, stops, 10, far), 4 + 8192),
        "max_tokens 1: n = 1, finish 2": ((plain, 1, stops, 10, far), 1 + 2 * 8192),
        "max_tokens T: n = T, finish 2": ((plain, T, stops, 10, far), 4 + 2 * 8192),
        "the model's length at the first position: n = 1, finish 3": ((plain, 100, stops, far - 1, far), 1 + 3 * 8192),
        "the model's length at the last position: n = T, finish 3": ((plain, 100, stops, far - T, far), 4 + 3 * 8192),
        "max_tokens 0 never stops": ((plain, 0, stops, 10, far), 4),
        "max_tokens -1 never stops, a stop id or not": (([eos] + plain[1:], 0xFFFFFFFF, stops, 10, far), 4),
    }
    assert {name: X.reference16(*c) for name, (c, _) in cases.items()} == {name: w for name, (_, w) in cases.items()}
    word = evaluate_batch(bind(X.StopRule16, T=T, NSTOP=3), *_args([c for c, _ in cases.values()]), kernel="reference")
    gates = evaluate_words(bind(StopRule16V2, T=T, NSTOP=3), *_args([c for c, _ in cases.values()]))
    assert [int(v) for v in np.asarray(word).reshape(-1)] == [w for _, w in cases.values()]
    assert [int(v) for v in np.asarray(gates).reshape(-1)] == [w for _, w in cases.values()]


def test_the_reference_at_one_id_and_at_the_most_the_word_holds() -> None:
    stops, far = (7,), 131072
    assert X.reference16([7], 100, stops, 10, far) == 1 + 8192
    assert X.reference16([5], 1, stops, 10, far) == 1 + 2 * 8192
    assert X.reference16([5], 100, stops, far - 1, far) == 1 + 3 * 8192
    assert X.reference16([5], 100, stops, 10, far) == 1
    plain = [5] * T_MAX
    assert X.reference16(plain[:-1] + [7], 100000, stops, 10, far) == T_MAX + 8192 == 0x3FFF
    assert X.reference16(plain, T_MAX, stops, 10, far) == T_MAX + 2 * 8192 == 0x5FFF
    assert X.reference16(plain, 100000, stops, far - T_MAX, far) == T_MAX + 3 * 8192 == 0x7FFF
    assert X.reference16(plain, 100000, stops, 10, far) == T_MAX
    with pytest.raises(ValueError, match="at most 8191 ids"):
        X.reference16(plain + [5], 100000, stops, 10, far)


@pytest.mark.parametrize("T,stops", [(1, (2,)), (5, (2, 3)), (32, GLM_EOS)])
def test_word_and_gates_are_the_reference(T: int, stops: tuple[int, ...]) -> None:
    K = len(stops)
    cases = _edges(T, stops) + _seeded(T, K, 300, seed=T * 16 + K)
    want = [X.reference16(*c) for c in cases]
    word = np.asarray(evaluate_batch(bind(X.StopRule16, T=T, NSTOP=K), *_args(cases), kernel="reference"))
    gates = np.asarray(evaluate_words(bind(StopRule16V2, T=T, NSTOP=K), *_args(cases)))
    assert word.shape == gates.shape == (len(cases),)
    assert [int(v) for v in word] == want
    assert [int(v) for v in gates] == want
    assert {w >> 13 for w in want} == {0, 1, 2, 3}


def test_the_boolean_form_is_the_word_view_with_one_output_of_16_bits() -> None:
    v1, v2 = bind(X.StopRule16, T=5, NSTOP=2), bind(StopRule16V2, T=5, NSTOP=2)
    assert word_of(v2).id == v1.id == "StopRule16_v1{T=5,NSTOP=2}"
    assert SW.boolean_version(v1).id == v2.id == "StopRule16_v2{T=5,NSTOP=2}"
    assert v1.signature()[1] == Value(16) and v2.signature()[1] == Array(16, BL.BIT)
    assert [n for n, _ in v1.signature()[0]] == [n for n, _ in bind(X.StopRule, T=5, NSTOP=2).signature()[0]]
    assert [t for _, t in v1.signature()[0]] == [t for _, t in bind(X.StopRule, T=5, NSTOP=2).signature()[0]]


def test_refuses_more_ids_than_the_word_holds_and_what_the_stop_rule_refuses() -> None:
    for d in (X.StopRule16, StopRule16V2):
        with pytest.raises(ValueError, match="refuses this binding: T must be at most 8191"):
            bind(d, T=T_MAX + 1, NSTOP=1)
        assert bind(d, T=T_MAX, NSTOP=1).id == f"{d.name}_v{d.version}{{T=8191,NSTOP=1}}"
        for statics in (dict(T=0, NSTOP=1), dict(T=4, NSTOP=0), dict(T=True, NSTOP=1)):
            with pytest.raises(ValueError, match=r"refuses this binding: T must be in \[1, 2\^31 - 1\)"):
                bind(d, **statics)
