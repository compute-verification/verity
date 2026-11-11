"""`StopRule_v1` (its words), `StopRule_v2` (its gates) and their Python function (`stop_rule.reference`) held to the derived
rule they compute, `encodings/derived.py`'s ``stop_rule``, over every prefix of a request's ids: no prefix shorter than n
ends, the prefix of n ends with the finish reason's name (1 "eos", 2 and 3 "length"), and with finish 0 no prefix up to T
ends.  On the cases the rule's callers meet (a stop id first and last, ``max_tokens`` exactly, ``max_model_len`` reached,
GLM-5.3's three stop ids, no stop within T) and on seeded ones over the served domain."""
from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import pytest

from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate_batch
from verity.circuits.evaluation.bits import evaluate_words
from verity.definitions.registry import stop_rule as X
from verity.definitions.registry.boolean_stop_rule import StopRuleV2


def _derived():
    spec = importlib.util.spec_from_file_location("enc_derived", Path(__file__).resolve().parents[1] / "encodings" / "derived.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


D = _derived()
GLM_EOS = (154820, 154827, 154829)
GLM_MAX_MODEL_LEN = 131072
NAME = {1: "eos", 2: "length", 3: "length"}


def _cfg(stops: tuple[int, ...], max_model_len: int):
    return D.Config(block_size=16, num_gpu_blocks=4096, max_num_batched_tokens=8192, max_model_len=max_model_len,
                    eos_token_id=stops[0], stop_token_ids=tuple(stops[1:]))


def _holds(ids: list[int], max_tokens: int, stops: tuple[int, ...], prompt_len: int, max_model_len: int,
           n: int, finish: int) -> None:
    cfg = _cfg(stops, max_model_len)
    ends = [D.stop_rule(ids[:L], prompt_len, max_tokens, cfg) for L in range(1, len(ids) + 1)]
    if finish == 0:
        assert n == len(ids) and ends == [None] * len(ids)
    else:
        assert ends[:n - 1] == [None] * (n - 1) and ends[n - 1] == NAME[finish]
        assert finish != 2 or n == max_tokens
        assert finish != 3 or prompt_len + n >= max_model_len


def _evaluate(T: int, stops: tuple[int, ...], cases: list[tuple]) -> list[tuple[int, int]]:
    """Each case's (n, finish) from the word Definition, the Boolean Definition and the reference, which must agree."""
    args = [np.array([c[k] for c in cases], dtype=np.uint32) for k in range(5)]
    word = np.asarray(evaluate_batch(bind(X.StopRule, T=T, NSTOP=len(stops)), *args, kernel="reference"))
    gates = np.asarray(evaluate_words(bind(StopRuleV2, T=T, NSTOP=len(stops)), *args))
    out = [tuple(int(v) for v in r) for r in word]
    assert out == [tuple(int(v) for v in r) for r in gates] == [X.reference(*c) for c in cases]
    return out


T = 32


def _named() -> list[tuple[str, tuple, tuple[int, int]]]:
    """(what, (ids, max_tokens, stops, prompt_len, max_model_len), (n, finish)) at GLM-5.3's stop ids and length."""
    s, mml, plain = GLM_EOS, GLM_MAX_MODEL_LEN, [1000 + i for i in range(T)]

    def at(i: int, tok: int) -> list[int]:
        return plain[:i] + [tok] + plain[i + 1:]

    return [
        ("eos first", (at(0, s[0]), 256, s, 100, mml), (1, 1)),
        ("eos last", (at(T - 1, s[0]), 256, s, 100, mml), (T, 1)),
        ("second stop id", (at(6, s[1]), 256, s, 100, mml), (7, 1)),
        ("third stop id", (at(9, s[2]), 256, s, 100, mml), (10, 1)),
        ("earliest of several", (at(20, s[0])[:11] + [s[2]] + at(20, s[0])[12:], 256, s, 100, mml), (12, 1)),
        ("max_tokens exactly", (plain, 17, s, 100, mml), (17, 2)),
        ("max_tokens = T", (plain, T, s, 100, mml), (T, 2)),
        ("max_tokens 1", (plain, 1, s, 100, mml), (1, 2)),
        ("eos at max_tokens", (at(16, s[2]), 17, s, 100, mml), (17, 1)),
        ("eos past max_tokens", (at(20, s[0]), 17, s, 100, mml), (17, 2)),
        ("max_model_len reached", (plain, 256, s, mml - 5, mml), (5, 3)),
        ("max_model_len at the last position", (plain, 256, s, mml - T, mml), (T, 3)),
        ("prompt at max_model_len", (plain, 256, s, mml, mml), (1, 3)),
        ("max_model_len and max_tokens at once", (plain, 5, s, mml - 5, mml), (5, 2)),
        ("eos at max_model_len", (at(4, s[1]), 256, s, mml - 5, mml), (5, 1)),
        ("no stop within T", (plain, 256, s, 100, mml), (T, 0)),
        ("max_model_len past T", (plain, 256, s, mml - T - 1, mml), (T, 0)),
        ("max_tokens 0", (at(0, s[0]), 0, s, 100, mml), (T, 0)),
    ]


@pytest.mark.parametrize("what,case,want", _named(), ids=[w for w, _, _ in _named()])
def test_named_cases_are_the_derived_rule(what: str, case: tuple, want: tuple[int, int]) -> None:
    (got,) = _evaluate(T, GLM_EOS, [case])
    assert got == want
    _holds(*case, *got)


@pytest.mark.parametrize("T_,stops", [(1, (2,)), (6, (2, 3)), (T, GLM_EOS)])
def test_seeded_cases_are_the_derived_rule(T_: int, stops: tuple[int, ...]) -> None:
    rng = random.Random(T_ * 7 + len(stops))
    cases = []
    for _ in range(200):
        ids = [rng.choice([*stops, 5, 6, 154879]) if rng.random() < 0.15 else rng.randrange(154880) for _ in range(T_)]
        mml = rng.choice([T_, 64, GLM_MAX_MODEL_LEN, (1 << 31) - 1])
        cases.append((ids, rng.choice([0, 1, 2, T_ - 1, T_, T_ + 1, 4096, (1 << 31) - 1]), stops,
                      rng.choice([0, 1, max(mml - T_, 0), mml - 1, mml, mml + 3, rng.randrange(mml)]), mml))
    got = _evaluate(T_, stops, cases)
    for case, (n, finish) in zip(cases, got):
        _holds(*case, n, finish)
    assert {f for _, f in got} == {0, 1, 2, 3}
