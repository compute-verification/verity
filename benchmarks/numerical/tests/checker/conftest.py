"""Shared inputs for the checker tests: structured transition-unit cases, random finite inputs, the RTX 4090
golden records and the K=1536 conformance vector."""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from verity_numerical.checker import REAL

ROOT = Path(__file__).resolve().parents[4]
GOLDEN = ROOT / "verity/ml/circuits/ops/tests/fixtures/golden/ada_bf16_m16n8k16.json"
K1536 = ROOT / "fixtures/typed-obligation-v0/stage2--qkv-proj-decode-0-k1536.json"

K = 16


def bf16(sign: int, t: int, f: int) -> int:
    return (sign << 15) | (t << 7) | f


def f32(sign: int, t: int, f: int) -> int:
    return (sign << 31) | (t << 23) | f


ONE = bf16(0, 127, 0)
NEG_ONE = bf16(1, 127, 0)
MAX_FINITE = bf16(0, 254, 0x7F)
MIN_SUBNORMAL = bf16(0, 0, 1)
MAX_SUBNORMAL = bf16(0, 0, 0x7F)
ZERO = 0


def _fill(pairs: list[tuple[int, int]]) -> tuple[list[int], list[int]]:
    pairs = pairs + [(ZERO, ZERO)] * (K - len(pairs))
    return [a for a, _ in pairs], [b for _, b in pairs]


def structured_cases() -> list[tuple[str, int, list[int], list[int]]]:
    """(name, c, a, b): every family the handoff names, both groups exercised."""
    cases: list[tuple[str, int, list[int], list[int]]] = []

    def add(name: str, c: int, pairs: list[tuple[int, int]]) -> None:
        a, b = _fill(pairs)
        cases.append((name, c, a, b))

    def add_words(name: str, c: int, a: list[int], b: list[int]) -> None:
        cases.append((name, c, a, b))

    # max shifts: a huge state with tiny products (products flush), and tiny state with huge products (state flushes)
    add("max-shift/state-dominates", f32(0, 254, 0), [(MIN_SUBNORMAL, MIN_SUBNORMAL)] * 16)
    add("max-shift/products-dominate", f32(0, 1, 1), [(MAX_FINITE, ONE)] + [(MIN_SUBNORMAL, MIN_SUBNORMAL)] * 15)
    add("max-shift/group2-huge-after-tiny-group1", f32(0, 0, 1), [(MIN_SUBNORMAL, ONE)] * 8 + [(MAX_FINITE, MAX_FINITE)] * 8)
    # cancellation to zero: products cancel pairwise; state cancels a product; cancellation across the two groups
    add("cancel/pairs", 0, [(ONE, ONE), (ONE, NEG_ONE)] * 8)
    add("cancel/state-vs-product", f32(1, 127, 0), [(ONE, ONE)])
    add("cancel/across-groups", 0, [(ONE, ONE)] * 8 + [(ONE, NEG_ONE)] * 8)
    add("cancel/negative-zero-state", f32(1, 0, 0), [(ZERO, ONE)] * 16)
    add("cancel/all-zero", 0, [])
    # subnormal operands and accumulators
    add("subnormal/operands", 0, [(MIN_SUBNORMAL, bf16(0, 200, 0)), (MAX_SUBNORMAL, MAX_SUBNORMAL), (MIN_SUBNORMAL, MIN_SUBNORMAL)])
    add("subnormal/state-passes-through", f32(0, 0, 0x2A), [])
    add("subnormal/state-plus-tiny-products", f32(1, 0, 1), [(MIN_SUBNORMAL, bf16(0, 127, 0x40))] * 16)
    add("subnormal/result-subnormal", 0, [(MIN_SUBNORMAL, bf16(0, 120, 0))] * 16)
    # truncation to zero: a nonzero sum whose window is below the smallest subnormal
    add("zero/truncation", 0, [(MIN_SUBNORMAL, bf16(0, 110, 0))])
    add("zero/truncation-negative", 0, [(MIN_SUBNORMAL, bf16(1, 110, 0))])
    # near-overflow and saturation, in either group, either sign, sticky through group 2
    add("overflow/group1-positive", 0, [(MAX_FINITE, MAX_FINITE)] * 8)
    add("overflow/group1-negative", 0, [(bf16(1, 254, 0x7F), MAX_FINITE)] * 8)
    add("overflow/group2-only", 0, [(ONE, ONE)] * 8 + [(MAX_FINITE, MAX_FINITE)] * 8)
    add("overflow/group1-sticky-through-cancelling-group2", 0, [(MAX_FINITE, MAX_FINITE)] * 8 + [(bf16(1, 254, 0x7F), MAX_FINITE)] * 8)
    add("overflow/state-max-plus-one-product", f32(0, 254, 0x7FFFFF), [(MAX_FINITE, ONE)])
    add("overflow/state-max-minus-products", f32(0, 254, 0x7FFFFF), [(bf16(1, 254, 0), ONE)] * 16)
    add("near/just-below", 0, [(bf16(0, 254, 0), bf16(0, 127, 0x40))])
    add("near/exact-max", 0, [(bf16(0, 254, 0x7F), bf16(0, 127, 0x7F))])
    # nonzero accumulators with products in both groups; ties in the maximum; max window (l = 30)
    add("acc/nonzero-both-groups", f32(0, 130, 0x123456), [(bf16(0, 128, 3), bf16(1, 126, 5))] * 16)
    add("acc/ties-in-maximum", f32(0, 127, 0), [(ONE, ONE)] * 16)
    add("acc/max-window", f32(0, 254, 0x7FFFFF), [(bf16(0, 190, 0x7F), bf16(0, 191, 0x7F))] * 16)
    add("acc/negative-state-positive-products", f32(1, 130, 0), [(ONE, ONE)] * 16)
    # every product exponent equal to the state's; alternating signs at different magnitudes
    add_words("mixed/alternating", f32(0, 120, 7),
              [bf16(i & 1, 110 + i, i * 9 & 0x7F) for i in range(16)],
              [bf16((i >> 1) & 1, 130 - i, (i * 13) & 0x7F) for i in range(16)])
    return cases


def finite_bf16(rng: random.Random) -> int:
    while True:
        w = rng.getrandbits(16)
        if (w >> 7) & 0xFF != 0xFF:
            return w


def finite_f32(rng: random.Random) -> int:
    while True:
        w = rng.getrandbits(32)
        if (w >> 23) & 0xFF != 0xFF:
            return w


def random_cases(n: int, seed: int = 2026) -> list[tuple[int, list[int], list[int]]]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        c = finite_f32(rng) if i % 3 else 0
        out.append((c, [finite_bf16(rng) for _ in range(K)], [finite_bf16(rng) for _ in range(K)]))
    return out


def _words(hex_text: str, bits: int) -> list[int]:
    width = bits // 4
    return [int(hex_text[i:i + width], 16) for i in range(0, len(hex_text), width)]


def golden_records() -> list[tuple[str, int, list[int], list[int], int]]:
    """Every all-finite record of the RTX 4090 capture: (family, acc, a, b, d)."""
    data = json.loads(GOLDEN.read_text())
    assert data["pipeline"] == "ada_bf16_m16n8k16"
    out = []
    for r in data["records"]:
        c, a, b, d = int(r["acc"], 16), _words(r["a"], 16), _words(r["b"], 16), int(r["d"], 16)
        finite = (c >> 23) & 0xFF != 0xFF and all((w >> 7) & 0xFF != 0xFF for w in a + b)
        if finite:
            out.append((r["family"], c, a, b, d))
    return out


def k1536_vector() -> tuple[list[int], list[int], int]:
    rec = json.loads(K1536.read_text())
    [values] = rec["values"]
    k = (len(values) - 1) // 2
    assert k == 1536
    return values[:k], values[k:2 * k], values[2 * k]


@pytest.fixture(scope="session")
def structured():
    return structured_cases()


@pytest.fixture(scope="session")
def golden():
    return golden_records()


@pytest.fixture(scope="session")
def params():
    return REAL
