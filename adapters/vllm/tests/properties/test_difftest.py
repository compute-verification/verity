"""`evaluate_spec`, the replay's one-call evaluator: it agrees with a numpy reference on BiasAdd (bf16 add, RNE), is
deterministic, and refuses operands that are not its parameters."""
from __future__ import annotations

import numpy as np
import pytest
from verity.circuits.defs import bind
from verity.definitions.registry import b1

from verity_vllm.check.replay.stoch_recompute import evaluate_spec


def _bf16_words(n: int, rng: np.random.Generator) -> np.ndarray:
    f = (rng.standard_normal(n) * float(rng.choice([0.05, 1.0, 8.0]))).astype(np.float32)
    return (f.view(np.uint32) >> 16).astype(np.uint16)


def _bf16_add_ref(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    fa = (a.astype(np.uint32) << 16).view(np.float32)
    fb = (b.astype(np.uint32) << 16).view(np.float32)
    s = (fa + fb).astype(np.float32)
    # round-to-nearest-even to bf16
    u = s.view(np.uint32).astype(np.uint64)
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16).astype(np.uint16)


def test_evaluate_spec_bias_add_matches_numpy_reference():
    rng = np.random.default_rng(0)
    N = 8
    y = _bf16_words(N, rng)
    b = _bf16_words(N, rng)
    spec = bind(b1.BiasAdd, N=N)
    got = evaluate_spec(spec, [y, b])
    exp = _bf16_add_ref(y, b)
    finite = ~np.isin(exp & 0x7F80, [0x7F80])
    assert np.array_equal(np.array(got, np.uint16)[finite], exp[finite])
    assert got == evaluate_spec(spec, [y, b])


def test_evaluate_spec_refuses_operands_that_are_not_its_parameters() -> None:
    spec = bind(b1.BiasAdd, N=4)
    y = _bf16_words(4, np.random.default_rng(0))
    for operands, why in (([y], "2 parameters, 1 operands"), ([y, y, y], "2 parameters, 3 operands"),
                          ([y, y[:3]], "parameter b has 4 leaves, operand has 3 words"),
                          ([np.append(y, y[:1]), y], "parameter y has 4 leaves, operand has 5 words")):
        with pytest.raises(ValueError) as refused:
            evaluate_spec(spec, operands)
        assert str(refused.value) == f"{spec.id}: {why}"
    assert evaluate_spec(spec, [y.reshape(2, 2), y]) == evaluate_spec(spec, [y, y])      # leaves are counted, not shapes
