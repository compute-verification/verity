"""``verity.ml.circuits.silicon.term``: the exact decoders and products agree with Python floats wherever floats represent the value.

Every finite BF16 / E4M3 / E5M2 / E2M1 / TF32 encoding is a dyadic rational far inside double range, so the decoders and the
products are checked against ``struct`` / ``Fraction`` on every code (every pair, for the 8- and 4-bit formats); the
FP32 codecs are checked on every boundary word and a 20k-word sample.
"""
from __future__ import annotations

import math
import random
import struct
from fractions import Fraction

import pytest

from verity.errors import InvalidArtifact
from verity.ml.circuits.silicon.term import (
    BF16,
    E2M1,
    E4M3,
    E5M2,
    FORMATS,
    TF32,
    PRODUCT_WIDTH,
    ZERO,
    Term,
    bf16_product,
    decode,
    decode_bf16,
    decode_e2m1,
    decode_e4m3,
    decode_e5m2,
    e2m1_product,
    e4m3_product,
    e5m2_product,
    fp32_term,
    pack_fp32,
    product,
    term_value,
    tf32_product,
)


def _f32(word: int) -> float:
    return struct.unpack("<f", struct.pack("<I", word & 0xFFFFFFFF))[0]


def _decoded_value(dec) -> Fraction:
    assert not isinstance(dec, str)
    sign, exp, sig = dec
    return (-1 if sign else 1) * Fraction(sig) * Fraction(2) ** exp


def _term_value(term: Term) -> Fraction:
    n, e = term_value(term)
    return Fraction(n) * Fraction(2) ** e


# --- decoders ------------------------------------------------------------------------------------------------------------


def test_bf16_decoder_matches_float_on_every_word() -> None:
    for word in range(1 << 16):
        dec = decode_bf16(word)
        ref = _f32(word << 16)
        if math.isnan(ref):
            assert dec == "nan"
        elif math.isinf(ref):
            assert dec == ("-inf" if ref < 0 else "+inf")
        else:
            assert _decoded_value(dec) == Fraction(ref), (hex(word), dec, ref)
            assert dec[0] == word >> 15  # signed zero keeps its sign


def test_e4m3_decoder_is_the_ocp_table() -> None:
    # OCP FP8 E4M3: bias 7, no infinities, NaN only at 0x7F/0xFF, max finite 448 (S.1111.110), min subnormal 2^-9
    finite = {}
    for word in range(256):
        dec = decode_e4m3(word)
        if word & 0x7F == 0x7F:
            assert dec == "nan"
            continue
        sign = -1 if word >> 7 else 1
        e, m = (word >> 3) & 0xF, word & 7
        expect = sign * (Fraction(m, 8) * Fraction(2) ** -6 if e == 0 else (1 + Fraction(m, 8)) * Fraction(2) ** (e - 7))
        assert _decoded_value(dec) == expect, hex(word)
        finite[word] = expect
    assert len(finite) == 254
    assert max(finite.values()) == 448
    assert min(v for v in finite.values() if v > 0) == Fraction(1, 512)


def test_e5m2_decoder_is_the_ocp_table() -> None:
    # OCP FP8 E5M2: bias 15, IEEE-style infinities at S.11111.00 and NaN at S.11111.{01,10,11}, max finite 57344
    finite = {}
    for word in range(256):
        dec = decode_e5m2(word)
        e, m = (word >> 2) & 0x1F, word & 3
        if e == 0x1F:
            assert dec == ("nan" if m else ("-inf" if word >> 7 else "+inf")), hex(word)
            continue
        sign = -1 if word >> 7 else 1
        expect = sign * (Fraction(m, 4) * Fraction(2) ** -14 if e == 0 else (1 + Fraction(m, 4)) * Fraction(2) ** (e - 15))
        assert _decoded_value(dec) == expect, hex(word)
        finite[word] = expect
    assert len(finite) == 248
    assert max(finite.values()) == 57344
    assert min(v for v in finite.values() if v > 0) == Fraction(2) ** -16


def test_e2m1_decoder_is_the_fp4_table() -> None:
    table = [0, Fraction(1, 2), 1, Fraction(3, 2), 2, 3, 4, 6]
    for code in range(16):
        dec = decode_e2m1(code)
        assert not isinstance(dec, str)  # no specials in E2M1
        assert _decoded_value(dec) == (-1 if code >> 3 else 1) * table[code & 7], code
        assert dec[0] == code >> 3


def test_generic_decode_dispatches_by_dtype_and_rejects_wide_words() -> None:
    assert decode("bf16", 0x3F80) == decode_bf16(0x3F80) and _decoded_value(decode_bf16(0x3F80)) == 1
    assert decode("e4m3", 0x38) == decode_e4m3(0x38) and _decoded_value(decode_e4m3(0x38)) == 1
    assert decode("e5m2", 0x3C) == decode_e5m2(0x3C) and _decoded_value(decode_e5m2(0x3C)) == 1
    assert decode("e2m1", 0x2) == decode_e2m1(0x2) and _decoded_value(decode_e2m1(0x2)) == 1
    assert set(FORMATS) == {"bf16", "e4m3", "e5m2", "e2m1", "tf32"}
    assert (BF16.bits, E4M3.bits, E5M2.bits, E2M1.bits, TF32.bits) == (16, 8, 8, 4, 19)
    assert (BF16.bias, E4M3.bias, E5M2.bias, E2M1.bias, TF32.bias) == (127, 7, 15, 1, 127)
    for fmt in FORMATS.values():
        with pytest.raises(InvalidArtifact):
            fmt.decode(1 << fmt.bits)
    with pytest.raises(InvalidArtifact):
        decode("fp6", 0)


# --- products ------------------------------------------------------------------------------------------------------------


def _check_products(fmt, prod, codes) -> int:
    checked = 0
    for a in codes:
        da = fmt.decode(a)
        for b in codes:
            db = fmt.decode(b)
            if isinstance(da, str) or isinstance(db, str):
                with pytest.raises(InvalidArtifact):
                    prod(a, b)
                continue
            t = prod(a, b)
            assert isinstance(t, Term)
            assert 0 <= t.magnitude < 1 << PRODUCT_WIDTH
            assert _term_value(t) == _decoded_value(da) * _decoded_value(db), (fmt.name, hex(a), hex(b))
            if t.magnitude == 0:
                assert t == ZERO  # zero products are the one unsigned zero, as in Hawkeye
            else:
                assert t.negative == bool((a ^ b) >> (fmt.bits - 1))
                assert (t.magnitude >> fmt.product_shift) << fmt.product_shift == t.magnitude, "carried at the top"
            assert product(fmt.name, a, b) == t  # the dtype-keyed entry point agrees
            checked += 1
    return checked


def test_bf16_product_is_exact_on_a_sample() -> None:
    rng = random.Random(0x5EED)
    codes = [rng.randrange(1 << 16) for _ in range(200)]
    codes += [0x0000, 0x8000, 0x0001, 0x007F, 0x0080, 0x7F7F, 0xFF7F, 0x3F80, 0xBF80, 0x7F80, 0x7FC0, 0xFF80]
    assert _check_products(BF16, bf16_product, codes) > 30_000
    assert BF16.product_shift == 9 and BF16.min_exponent == -133


def test_e4m3_product_is_exact_on_every_pair() -> None:
    assert _check_products(E4M3, e4m3_product, range(256)) == 254 * 254
    assert E4M3.product_shift == 17 and E4M3.min_exponent == -9
    assert _term_value(e4m3_product(0x7E, 0x7E)) == 448 * 448


def test_e5m2_product_is_exact_on_every_pair() -> None:
    assert _check_products(E5M2, e5m2_product, range(256)) == 248 * 248
    assert E5M2.product_shift == 19 and E5M2.min_exponent == -16
    assert _term_value(e5m2_product(0x7B, 0x7B)) == 57344 * 57344
    assert _term_value(e5m2_product(0x01, 0x01)) == Fraction(2) ** -32  # subnormal x subnormal, kept exactly


def test_e2m1_product_is_exact_on_every_pair() -> None:
    assert _check_products(E2M1, e2m1_product, range(16)) == 256
    assert E2M1.product_shift == 21
    assert _term_value(e2m1_product(0x7, 0x7)) == 36 and not e2m1_product(0x7, 0x7).negative
    assert e2m1_product(0xF, 0x7).negative
    assert e2m1_product(0x0, 0x7) == ZERO and e2m1_product(0x8, 0x7) == ZERO


def test_tf32_product_is_exact_on_a_sample() -> None:
    """TF32 is an FP32 word's top 19 bits: its value is that word's with the low 13 bits cleared."""
    rng = random.Random(0x7F32)
    codes = [rng.randrange(1 << 19) for _ in range(160)]
    codes += [0x00000, 0x40000, 0x00001, 0x003FF, 0x00400, 0x3FBFF, 0x7FBFF, 0x1FC00, 0x5FC00, 0x3FC00, 0x3FE00, 0x7FC00]
    assert _check_products(TF32, tf32_product, codes) > 20_000
    assert TF32.product_shift == 3 and TF32.min_exponent == -136
    for w in (0x3F800000, 0x00002000, 0xC0490FDB, 0x7F7FFFFF):
        assert _decoded_value(TF32.decode(w >> 13)) == Fraction(struct.unpack("<f", struct.pack("<I", w & ~0x1FFF))[0])


def test_generic_product_rejects_unknown_dtype() -> None:
    with pytest.raises(InvalidArtifact):
        product("fp6", 0, 0)


# --- FP32 codecs ---------------------------------------------------------------------------------------------------------


def test_fp32_term_round_trips_finite_words() -> None:
    rng = random.Random(1)
    words = [rng.randrange(1 << 32) for _ in range(20_000)]
    words += [0, 1, 0x007FFFFF, 0x00800000, 0x7F7FFFFF, 0xFF7FFFFF, 0x3F800000, 0xBF800000, 0x80000001]
    for w in words:
        if (w >> 23) & 0xFF == 0xFF:
            with pytest.raises(InvalidArtifact):
                fp32_term(w)
            continue
        t = fp32_term(w)
        assert _term_value(t) == Fraction(_f32(w)), hex(w)
        if w & 0x7FFFFFFF:
            assert pack_fp32(t) == w, hex(w)
        else:
            assert t == ZERO and pack_fp32(t) == 0  # the packer's zero is +0; -0 in is handled by the models


def test_pack_fp32_overflow_and_subnormals() -> None:
    assert pack_fp32(Term(False, 128, 1 << 23)) == 0x7F800000
    assert pack_fp32(Term(True, 128, 1 << 23)) == 0xFF800000
    # a 24-bit magnitude without bit 23 at exponent -126 is a subnormal word
    assert pack_fp32(Term(False, -126, 1)) == 0x00000001
    assert pack_fp32(Term(True, -126, 0x7FFFFF)) == 0x807FFFFF
    assert _term_value(fp32_term(0x00000001)) == Fraction(2) ** -149
