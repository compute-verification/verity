"""`filler.cu`'s filler, word for word in Python (no numpy, so `gate.py` runs on a node's bare Python).

A filler thread `ftid` holds salted registers a[4], b[2][2] = gen(seed, ftid, ...). A warp's 32 lanes form one A (16×32 s8)
and two B (32×8 s8) by the m16n8k32 fragment layout (PTX ISA, `mma.m16n8k32` with .s8), and tile j accumulates
A·B[j & 1] once a round. After round n a lane's tile-j words are n·P[j & 1] at its four accumulator positions (mod 2³²),
and chain j absorbs them with duo at key positions 4j..4j+3, as `benchmarks/receipt_phi/twins.py` defines duo.
"""

from __future__ import annotations

from functools import lru_cache

M32 = 0xFFFFFFFF
NKEY = 32
FT = 4


def lowbias32(x: int) -> int:
    x &= M32
    x ^= x >> 16
    x = (x * 0x7FEB352D) & M32
    x ^= x >> 15
    x = (x * 0x846CA68B) & M32
    x ^= x >> 16
    return x


def gen(seed: int, a: int, b: int, c: int) -> int:
    inner = lowbias32((b * 0x85EBCA77 & M32) ^ lowbias32((c + 0x632BE5AB) & M32))
    return lowbias32(seed ^ lowbias32((a * 0x9E3779B1 & M32) ^ inner))


def rotl(x: int, r: int) -> int:
    r &= 31
    return ((x << r) | (x >> (32 - r))) & M32 if r else x


def rot_of(t: int) -> int:
    return ((t * 7 + 5) % 31) + 1


@lru_cache(maxsize=None)
def keys(kseed: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (tuple(gen(kseed, 1, i, 0) | 1 for i in range(NKEY)), tuple(gen(kseed, 3, i, 0) for i in range(NKEY)))


def duo(st: tuple[int, int], p: list[int], t0: int, kseed: int) -> tuple[int, int]:
    ka, kk = keys(kseed)
    y, z = st
    for i, w in enumerate(p):
        t = t0 + i
        u = y ^ w ^ kk[t]
        y = (u * ka[t] + z) & M32
        z = (rotl(u, rot_of(t)) + y) & M32
    return y, z


def finish(st: tuple[int, int]) -> int:
    return st[0] ^ rotl(st[1], 16)


def s8(reg: int, i: int) -> int:
    v = (reg >> (8 * i)) & 0xFF
    return v - 256 if v >= 128 else v


def a_pos(lane: int, i: int) -> tuple[int, int]:
    """Row and column of A element a_i (0 <= i < 16) in lane `lane`'s fragment."""
    g, t = lane >> 2, lane & 3
    row = g if (i < 4 or 8 <= i < 12) else g + 8
    col = 4 * t + (i & 3) + (16 if i >= 8 else 0)
    return row, col


def b_pos(lane: int, i: int) -> tuple[int, int]:
    """Row (k) and column (n) of B element b_i (0 <= i < 8)."""
    g, t = lane >> 2, lane & 3
    return 4 * t + (i & 3) + (16 if i >= 4 else 0), g


def c_pos(lane: int, r: int) -> tuple[int, int]:
    """Row and column of accumulator word c_r (0 <= r < 4)."""
    g, t = lane >> 2, lane & 3
    return g + (8 if r >= 2 else 0), 2 * t + (r & 1)


@lru_cache(maxsize=None)
def warp_products(seed: int, warp: int) -> tuple:
    """P[0], P[1] (16×8 int) of filler warp `warp` (its lanes are ftid 32·warp .. 32·warp + 31)."""
    A = [[0] * 32 for _ in range(16)]
    B = [[[0] * 8 for _ in range(32)] for _ in range(2)]
    for lane in range(32):
        ftid = 32 * warp + lane
        a = [gen(seed, ftid, 7, i) for i in range(4)]
        for i in range(16):
            r, c = a_pos(lane, i)
            A[r][c] = s8(a[i // 4], i % 4)
        for h in range(2):
            b = [gen(seed, ftid, 8 + h, k) for k in range(2)]
            for i in range(8):
                r, c = b_pos(lane, i)
                B[h][r][c] = s8(b[i // 4], i % 4)
    return tuple(tuple(tuple(sum(A[m][k] * B[h][k][n] for k in range(32)) for n in range(8)) for m in range(16))
                 for h in range(2))


def thread_words(seed: int, kseed: int, iters: int, ftid: int) -> list[int]:
    """Filler thread `ftid`'s 20 output words after `iters` rounds: 4 receipts, then tile j's 4 accumulator words."""
    P = warp_products(seed, ftid // 32)
    lane = ftid % 32
    per = [[P[j & 1][c_pos(lane, r)[0]][c_pos(lane, r)[1]] for r in range(4)] for j in range(FT)]
    st = [(gen(seed, ftid, j, 100), gen(seed, ftid, j, 101)) for j in range(FT)]
    for n in range(1, iters + 1):
        st = [duo(st[j], [(n * w) & M32 for w in per[j]], 4 * j, kseed) for j in range(FT)]
    return [finish(s) for s in st] + [(iters * w) & M32 for j in range(FT) for w in per[j]]
