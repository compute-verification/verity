"""Exact tensor-core merge D = Z1 + Z2 by one int8 m16n8k32 MMA (block-diagonal, base-255 radix).

The running accumulator Z1 enters as the C input (free). Z2 enters as A·B: each merged word is a radix sum
Z2 = sum_g d_g * w_g of u8 digits d_g with u8 weights w = (1, 255, 255, ...). int8 caps every weight at 255, so
`{1, 255, ...}` is the most range-efficient gapless radix: digit 0 is Z2 mod 255 and the rest share the quotient.

With n digit-groups and s merged columns (n*s <= 32 k-slots, s <= 8), group g of column j sits at slot
kappa = g*s + j, with B[kappa][j'] = w_g if j' == j else 0. Then D[i][j] = Z1[i][j] + Z2[i][j] for j < s.
One MMA is 4,096 MAC slots and merges 16*s words, so a merge costs 4096 / (16*s) = 256/s units a word.

The FP16 construction below (`build_tile_f16`) has no 255 cap: two digits, 32 units a word for any word in [-2^22, 2^22).
"""

import struct

K, M, N = 32, 16, 8
UNITS_PER_MMA = M * N * K


def vmax(n: int) -> int:
    """The largest word every value up to which n digits represent exactly."""
    return 254 + (n - 1) * 255 * 255


def cols(n: int) -> int:
    return min(N, K // n)


def units_per_word(n: int) -> float:
    return UNITS_PER_MMA / (M * cols(n))


def digits(v: int, n: int) -> list[int]:
    if not 0 <= v <= vmax(n):
        raise ValueError(f"{v} outside [0, {vmax(n)}] for n={n}")
    d = [v % 255]
    q = v // 255
    for _ in range(1, n):
        take = min(q, 255)
        d.append(take)
        q -= take
    assert q == 0
    return d


def weights(n: int) -> list[int]:
    return [1] + [255] * (n - 1)


def build_tile(z2: list[list[int]], n: int):
    """A[16][32] and Bt[8][32] (u8) so that A·B equals z2 on the first cols(n) columns, zero elsewhere."""
    s = cols(n)
    a = [[0] * K for _ in range(M)]
    bt = [[0] * K for _ in range(N)]
    w = weights(n)
    for g in range(n):
        for j in range(s):
            bt[j][g * s + j] = w[g]
    for i in range(M):
        for j in range(s):
            for g, d in enumerate(digits(z2[i][j], n)):
                a[i][g * s + j] = d
    return a, bt


def mma(a, bt, c):
    """The exact u8·u8 m16n8k32 step with an s32 accumulator, wrapping modulo 2^32."""
    d = [[0] * N for _ in range(M)]
    for i in range(M):
        for j in range(N):
            acc = c[i][j] + sum(a[i][k] * bt[j][k] for k in range(K))
            d[i][j] = (acc + 2**31) % 2**32 - 2**31
    return d


# FP16 m16n8k16 with an FP32 accumulator. FP16 holds powers of two exactly up to 2^15, so the radix is (1, 2^11): a
# word u in [-2^22, 2^22) is lo = u mod 2^11 in [0, 2047] and hi = floor(u / 2^11) in [-2048, 2047], both exact FP16.
# Two digits fill k = 16 at s = 8, so one MMA (2,048 MACs at 2 units a MAC) merges all 128 words: 32 units a word.
K16, F16_BITS = 16, 11
UNITS_PER_MMA_F16 = 2 * M * N * K16


def fp16_exact(x: int) -> bool:
    try:
        return float(struct.unpack("<e", struct.pack("<e", float(x)))[0]) == x
    except OverflowError:
        return False


def fp32_exact(x: int) -> bool:
    return float(struct.unpack("<f", struct.pack("<f", float(x)))[0]) == x


def f16_digits(u: int) -> tuple[int, int]:
    if not -(1 << 2 * F16_BITS) <= u < 1 << 2 * F16_BITS:
        raise ValueError(f"{u} outside [-2^22, 2^22)")
    return u & ((1 << F16_BITS) - 1), u >> F16_BITS


def build_tile_f16(z2: list[list[int]]):
    """A[16][16] and Bt[8][16] (integer-valued FP16) so that A·B equals z2 on all 8 columns."""
    a = [[0] * K16 for _ in range(M)]
    bt = [[0] * K16 for _ in range(N)]
    for j in range(N):
        bt[j][j], bt[j][N + j] = 1, 1 << F16_BITS
    for i in range(M):
        for j in range(N):
            a[i][j], a[i][N + j] = f16_digits(z2[i][j])
    return a, bt


def mma_f16(a, bt, c):
    """The m16n8k16 step on integer operands, over the integers: what an exact FP32 accumulation returns. Every operand
    must be an exact FP16 and every result an exact FP32 (|D| <= 2^24); the GPU check is that silicon returns this."""
    assert all(fp16_exact(x) for row in a + bt for x in row)
    d = [[c[i][j] + sum(a[i][k] * bt[j][k] for k in range(K16)) for j in range(N)] for i in range(M)]
    assert all(fp32_exact(x) for row in c + d for x in row)
    return d
