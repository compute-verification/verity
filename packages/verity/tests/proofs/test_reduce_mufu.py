"""``veritor.reduce-mufu@1``: the guest's manifest digest and its integer-exact FP32 semantics."""

from __future__ import annotations

import random
from fractions import Fraction

import numpy as np
import pytest
from verity.proofs import reduce_mufu as R
from verity.proofs.typed_obligation import family_gate_set


def _finite_words(rnd: random.Random, n: int) -> list[int]:
    out = []
    for _ in range(n):
        e = rnd.choice([0, rnd.randint(1, 254), rnd.randint(100, 160)])
        out.append((rnd.getrandbits(1) << 31) | (e << 23) | rnd.getrandbits(23))
    return out


def _f32(u: int) -> np.float32:
    return np.array([u], dtype=np.uint32).view(np.float32)[0]


def _bits(x: np.float32) -> int:
    return int(np.array([x], dtype=np.float32).view(np.uint32)[0])


def _value(u: int) -> Fraction:
    negative, magnitude, exponent = R._unpack(u)
    v = Fraction(magnitude) * Fraction(2) ** exponent
    return -v if negative else v


def test_digest_is_the_guests():
    gate_set = family_gate_set("veritor.reduce-mufu@1", 32)
    assert gate_set is not None and str(gate_set.digest) == R.REDUCE_MUFU_DIGEST
    assert family_gate_set("veritor.reduce-mufu@1", 64) is None


def test_add_mul_div_equal_ieee_on_finite_words():
    rnd = random.Random(20260926)
    words = _finite_words(rnd, 6000)
    with np.errstate(all="ignore"):
        for a, b in zip(words[::2], words[1::2]):
            fa, fb = _f32(a), _f32(b)
            assert R.f32_add(a, b) == _bits(fa + fb)
            assert R.f32_mul(a, b) == _bits(fa * fb)
            if fb != 0:
                assert R.f32_div(a, b) == _bits(fa / fb)


def test_fma_is_one_rounding_of_the_exact_value():
    rnd = random.Random(7)
    words = _finite_words(rnd, 3000)
    for a, b, c in zip(words[::3], words[1::3], words[2::3]):
        exact = _value(a) * _value(b) + _value(c)
        got = R.f32_fma(a, b, c)
        if R._exp(got) == 0xFF:
            continue
        # the result is the float nearest the exact value: no float lies strictly between them
        for neighbour in ((got & 0x7FFF_FFFF) - 1, (got & 0x7FFF_FFFF) + 1):
            if 0 <= neighbour < 0x7F80_0000:
                n = _value(neighbour | (got & 0x8000_0000))
                assert not (abs(n - exact) < abs(_value(got) - exact))


def test_table_gates_accept_only_the_row_the_operand_reads():
    gate_set = R.make_reduce_mufu_gate_set()
    x = 0x3F00_0000  # 0.5
    index = R.rsqrt_approx_index(x)
    word = 0x3F80_0000
    assert gate_set["rsqrt_approx"].evaluate((x, (index << 32) | word)) == R.rsqrt_approx(x, (index << 32) | word)
    with pytest.raises(RuntimeError):
        gate_set["rsqrt_approx"].evaluate((x, ((index ^ 1) << 32) | word))


def test_elem_bf16_is_the_guests():
    from verity.proofs.elem_bf16 import ELEM_BF16_DIGEST, silu_mul_bf16, silu_table

    assert str(family_gate_set("veritor.elem-bf16@1", 32).digest) == ELEM_BF16_DIGEST
    table = silu_table()  # raises unless it hashes to the guest's embedded table
    assert table[0x0000] == 0 and table[0x8000] == 0 and table[0x3F80] == 0x3F3B  # silu(1) = 0.7311 -> bf16 0x3F3B
    assert silu_mul_bf16(0x3F80, 0x4000) == 0x3FBB  # 2 * bf16(silu(1))
    assert silu_mul_bf16(0x7FC1, 0x3F80) == 0x7FC0
