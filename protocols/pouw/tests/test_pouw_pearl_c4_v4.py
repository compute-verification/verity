"""pearl_c4's v4 candidate (rank-2 FFMA noise, research store ``docs/pouw/pearl-c4-v4.md``): the noise is two nearest-even
FMAs, it is never zero, identical rows draw distinct factors, the factors are exact in FP32 and in three BF16 pieces, and
the one-step BF16 peel removes the noise up to its BF16 roundings."""

from __future__ import annotations

from fractions import Fraction

import pearl_c4_vectors as V
from verity_pouw.schemes import pearl_c4 as F
from verity_pouw.schemes import pearl_kw as P

K = 128
SEED_A, SEED_B = b"a" * 32, b"b" * 32


def rn_f32(v: Fraction) -> int:
    """The FP32 word of v rounded to nearest, ties to even (normal range)."""
    if v == 0:
        return 0
    a = abs(v)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    q = a / Fraction(2) ** (e - 23)
    n = q.numerator // q.denominator
    rem = q - n
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and n % 2):
        n += 1
    if n == 1 << 24:
        n, e = 1 << 23, e + 1
    return (((e + 127) << 23) | (n - (1 << 23))) | (0x80000000 if v < 0 else 0)


def val(word: int) -> Fraction:
    return Fraction(P.f32_of_bits(word))


def test_rows_and_columns_are_nonzero_codes_and_admissible_jitters():
    for i in range(256):
        e1, e2, k1, k2 = F.v4_row(SEED_A, F.ROLE_EA, i)
        assert e1 & 7 and e2 & 7 and 0 <= k1 < F.V4_K and 0 <= k2 < F.V4_K and (F.V4_J_ONE + k1) % 5
    for t in range(256):
        f1, f2 = F.v4_column(SEED_B, F.ROLE_FA, t)
        assert f1 & 7 and f2 & 7
    assert F.v4_column(SEED_B, F.ROLE_FA, 3) != F.v4_column(SEED_B, F.ROLE_FB, 3)


def test_factors_are_exact_in_fp32_and_in_three_bf16_pieces():
    for i in range(64):
        row = F.v4_row(SEED_A, F.ROLE_EA, i)
        for a in (0x08, 0x30, 0x55, 0x7E):
            av = Fraction(P.fp8_to_f32(a))
            for w, e, kq in zip(F.v4_row_words(row, a), row[:2], row[2:]):
                assert val(w) == F.e2m1_value(e) * av * Fraction(F.V4_J_ONE + kq, F.V4_J_ONE)
                assert sum(Fraction(P.f32_of_bits(h << 16)) for h in F.v4_pieces(w)) == val(w)
    f = F.v4_column_words((0x9, 0x7))
    assert (val(f[0]), val(f[1])) == (Fraction(-1, 2), Fraction(15, 2))


def test_the_noise_is_two_nearest_even_fmas():
    x = F.parse_row(V.rows("atom", "A", 1, K)[0], K)
    _, _, alpha, _, a = F.v4_stats(F.NVFP4, x)
    ew = F.v4_row_words(F.v4_row(SEED_A, F.ROLE_EA, 0), a)
    for t in range(K):
        c = F.fp32.mul(alpha, x[t])
        fw = F.v4_column_words(F.v4_column(SEED_B, F.ROLE_FA, t))
        inner = rn_f32(val(c) + val(ew[0]) * val(fw[0]))
        assert F.v4_noise(c, ew, fw) == rn_f32(val(inner) + val(ew[1]) * val(fw[1]))


def test_no_noise_value_is_zero():
    """e1·f1·(2^14 + k1) + (5/4)·e2·f2·(2^14 + k2) = 0 needs a factor 5 on the left: no product of two nonzero E2M1
    values (times 16, an integer) has one, and no admissible 2^14 + k1 does.  Both sets are checked in full."""
    mags = [F.e2m1_value(c) for c in range(1, 8)]
    assert all((16 * p * q).denominator == 1 and (16 * p * q).numerator % 5 for p in mags for q in mags)
    assert all((F.V4_J_ONE + k) % 5 for k in range(F.V4_K) if (F.V4_J_ONE + k) % 5)
    n_zero = 0
    for i in range(256):
        row = F.v4_row(SEED_A, F.ROLE_EA, i)
        ew = F.v4_row_words(row, 0x30)
        for t in range(64):
            fw = F.v4_column_words(F.v4_column(SEED_B, F.ROLE_FA, t))
            n_zero += val(ew[0]) * val(fw[0]) + val(ew[1]) * val(fw[1]) == 0
    assert n_zero == 0


def test_identical_rows_draw_distinct_factors_and_form_distinct_operands():
    rows = [F.v4_row(SEED_A, F.ROLE_EA, i) for i in range(64)]
    assert len(set(rows)) == 64
    x = F.parse_row(V.rows("atom", "A", 1, K)[0], K)
    cols = F.v4_basis(SEED_B, F.ROLE_FA, K)
    side = F.v4_form(F.NVFP4, [x] * 16, rows[:16], cols)
    assert len({tuple(c) for c in side.codes}) == 16
    assert all(F.scale_bytes_ok(F.NVFP4, s) for s in side.scales) and all(0x08 <= a <= 0x7E for a in side.ae)


def test_the_one_step_peel_removes_the_noise_up_to_its_bf16_roundings():
    """U against the exact rational (A′ − N_A)(B̃ − N_B)ᵀ with N = e′·F′ᵀ: the peel's BF16 roundings of A′·F′_B and of
    e′_B·G − B̃·F′_A are the only difference; C̃ is far from it."""
    xa = [F.parse_row(r, K) for r in V.rows("clean-up", "A", 16, K)]
    xb = [F.parse_row(r, K) for r in V.rows("clean-up", "B", 16, K)]
    ra = [F.v4_row(SEED_A, F.ROLE_EA, i) for i in range(16)]
    rb = [F.v4_row(SEED_B, F.ROLE_EB, j) for j in range(16)]
    fa, fb = F.v4_basis(SEED_B, F.ROLE_FA, K), F.v4_basis(SEED_B, F.ROLE_FB, K)
    a, b = F.v4_form(F.NVFP4, xa, ra, fa), F.v4_form(F.NVFP4, xb, rb, fb)
    pa, pb = F.v4_peel_factors_a(F.NVFP4, a, ra, fb), F.v4_peel_factors_b(F.NVFP4, b, rb, fa, fb)

    def value(side, i):
        s = [Fraction(P.fp8_to_f32(c)) for c in side.scales[i]]
        return [F.e2m1_value(c) * s[t // 16] for t, c in enumerate(side.codes[i])]

    def noise(side, rows, basis, i):
        ew = [val(w) for w in F.v4_row_words(rows[i], side.ae[i])]
        return [ew[0] * val(F.v4_column_words(c)[0]) + ew[1] * val(F.v4_column_words(c)[1]) for c in basis]

    va, na = [value(a, i) for i in range(16)], [noise(a, ra, fa, i) for i in range(16)]
    vb, nb = [value(b, j) for j in range(16)], [noise(b, rb, fb, j) for j in range(16)]
    err_u = err_c = Fraction(0)
    for i in range(16):
        for j in range(16):
            c = F.chain(F.NVFP4, a.codes[i], b.codes[j], a.scales[i], b.scales[j])
            u = F.v4_peel(c, pa[i], pb[j])
            exact = sum((p - q) * (r - t) for p, q, r, t in zip(va[i], na[i], vb[j], nb[j]))
            size = sum(abs(p * r) for p, r in zip(va[i], vb[j]))
            du = abs(val(u) - exact)
            assert du <= size / 512
            err_u += du
            err_c += abs(val(c) - exact)
    assert err_c > 100 * err_u


def test_the_witness_counts_a_dominant_element_dead_and_ordinary_ones_live():
    x = F.parse_row(V.rows("atom", "A", 1, K)[0], K)
    _, _, alpha, _, a = F.v4_stats(F.NVFP4, x)
    dead = F.v4_salt_dead(F.NVFP4, x, alpha, a)
    assert sum(dead) < K // 2
    big = list(x)
    big[5] = P.f32_bits(1e4) & 0xFFFF0000
    _, _, alpha2, _, a2 = F.v4_stats(F.NVFP4, big)
    assert F.v4_salt_dead(F.NVFP4, big, alpha2, a2)[5]
