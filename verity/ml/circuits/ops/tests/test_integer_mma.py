"""``verity.ml.circuits.silicon.integer``: IMMA m16n8k32 s8 x s8 -> s32, an exact integer step, against the PRO 6000 probe."""
from __future__ import annotations

import random

import pytest

from verity.ml.circuits.silicon.errors import InvalidArtifact
from verity.ml.circuits.silicon.instructions import INSTRUCTIONS
from verity.ml.circuits.silicon.integer import (
    BLACKWELL_SM120_S8_M16N8K32,
    BLACKWELL_SM120_S8_M16N8K32_WRAP,
    S32_MAX,
    S32_MIN,
    s32_value,
    s32_word,
)
from verity.ml.circuits.silicon.models import MODELS, tc_dot_chain
from verity.ml.circuits.silicon.term import term_value

SAT, WRAP = BLACKWELL_SM120_S8_M16N8K32, BLACKWELL_SM120_S8_M16N8K32_WRAP

# Run r20261015-071801-6d3c (benchmarks/fp8_instantiation/imma.py `edge_case`): every entry of A is 127 and every entry
# of B is sign * 127, zero incoming accumulator, the device word of one output: (k, sign) -> s32 value.
DEVICE_EDGE = {(133_120, 1): 2_147_092_480, (133_120, -1): -2_147_092_480, (133_152, 1): 2_147_483_647}


def code(v: int) -> int:
    return v & 0xFF


def test_step_is_the_exact_integer_dot() -> None:
    rng = random.Random(3)
    for model in (SAT, WRAP):
        for _ in range(500):
            a = [rng.randrange(-128, 128) for _ in range(32)]
            b = [rng.randrange(-128, 128) for _ in range(32)]
            c = rng.randrange(-(1 << 30), 1 << 30)
            d = model.step(s32_word(c), [code(x) for x in a], [code(x) for x in b])
            assert s32_value(d) == c + sum(x * y for x, y in zip(a, b))


def test_products_are_exact_terms() -> None:
    for a in (-128, -127, -1, 0, 1, 127):
        for b in (-128, -3, 0, 3, 127):
            n, e = term_value(SAT.product(code(a), code(b)))
            assert n * 2 ** e == a * b


def test_out_of_range_saturates_or_wraps() -> None:
    a, b = [code(127)] * 32, [code(127)] * 32
    top = s32_word(S32_MAX - 100)
    assert s32_value(SAT.step(top, a, b)) == S32_MAX
    assert s32_value(WRAP.step(top, a, b)) == S32_MAX - 100 + 32 * 127 * 127 - (1 << 32)
    bottom = s32_word(S32_MIN + 100)
    neg = [code(-127)] * 32
    assert s32_value(SAT.step(bottom, a, neg)) == S32_MIN
    assert s32_value(WRAP.step(bottom, a, neg)) == S32_MIN + 100 - 32 * 127 * 127 + (1 << 32)


def test_the_device_edge_words() -> None:
    """The saturating model reproduces the probe's three edge words as a chain of k/32 steps; the wrapping model
    reproduces the two in range and not the one past it, so the kernel the probe ran saturates."""
    for (k, sign), word in DEVICE_EDGE.items():
        a, b = [code(127)] * k, [code(sign * 127)] * k
        assert s32_value(tc_dot_chain(SAT, 0, a, b)) == word
        wrapped = s32_value(tc_dot_chain(WRAP, 0, a, b))
        assert (wrapped == word) == (abs(sign * k * 127 * 127) <= S32_MAX)


def test_registry_entry() -> None:
    ins = INSTRUCTIONS["sm120.mma.m16n8k32.s8"]
    assert ins.model is SAT and ins.status == "hypothesis" and ins.acc_dtype == "s32" and ins.shape == (16, 8, 32)
    assert "r20261015-071801-6d3c" in ins.evidence[0]
    assert MODELS[SAT.name] is SAT and MODELS[WRAP.name] is WRAP


def test_refuses_words_outside_the_formats() -> None:
    with pytest.raises(InvalidArtifact):
        SAT.step(1 << 32, [0] * 32, [0] * 32)
    with pytest.raises(InvalidArtifact):
        SAT.step(0, [256] + [0] * 31, [0] * 32)
    with pytest.raises(InvalidArtifact):
        SAT.step(0, [0] * 31, [0] * 31)
