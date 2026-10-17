"""CPU twins of the fast-decode prototypes (`tcsq.cu`, `nttsq.cu`): the same index arithmetic, so a test pins every address,
fragment and bound the kernels rely on before a GPU runs them.

- `Width`: the widths the prototypes take, w = 32 (32 PER + 2) bits (P2 v4's 16,448 at PER = 16), each with the smallest
  c0 = 1 mod 4 that makes 2^w - c0 prime (`WIDTHS`; 21,065 at 16,448 is v4's).
- `tc_square`: z^2's byte-convolution coefficients by int8 tensor-core tiles, emulated fragment by fragment
  (`mma.m16n8k32.row.col.s32.u8.u8.s32`): A is the 16 x 32 Toeplitz block G_m[v][c] = z[16m + v - c], built from a
  zero-gapped arena by funnel shifts and byte swaps; B is 8 consecutive 32-byte chunks; tile (pi, i) accumulates output
  columns u = pi + 16 i + 2 j (coefficient 16 u + v) over every m = pi mod 2.  The k slots are permuted so that a lane's
  two B words are one 8-byte load: slot 4 kq + e is chunk byte 8 kq + e, slot 16 + 4 kq + e is 8 kq + 4 + e.
- `karatsuba_plan`, `kara_square`: the same on top of d levels of Karatsuba, split at whole tiles (h a multiple of 8
  chunks), D = |x0 - x1| exact, every base square added into the output tiles at its offsets with its sign.
- `fold`: the kernels' reduction of the coefficients mod p (limb sums V, carry-save W, the fold by c0, the bias 2^30 p).
- `ntt_square`: the square by a 2048-point cyclic NTT over Goldilocks (2^64 - 2^32 + 1) on 16-bit limbs, the seven
  coefficients past 2048 recomputed and taken out of the wrap; `limb_sums16` its limb sums for `fold`.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

WIDTHS = {16: 21065, 8: 7305, 4: 1017, 2: 2397}
"""PER -> c0: the smallest c0 = 1 mod 4 with 2^w - c0 prime, w = 32 (32 PER + 2) (gmpy2.is_prime, 30 rounds)."""


@dataclass(frozen=True)
class Width:
    per: int

    @property
    def c0(self) -> int:
        return WIDTHS[self.per]

    @property
    def limbs(self) -> int:
        """32-bit limbs: 32 PER in the main part, two more (the top 8 bytes) on the last lane."""
        return 32 * self.per + 2

    @property
    def w(self) -> int:
        return 32 * self.limbs

    @property
    def n(self) -> int:
        return 4 * self.limbs

    @property
    def p(self) -> int:
        return (1 << self.w) - self.c0

    @property
    def q_dense(self) -> int:
        """32-byte chunks of the dense square: the top 8 bytes in a chunk of their own."""
        return 4 * self.per + 1

    @property
    def q_main(self) -> int:
        """32-byte chunks of the main part (all but the top 8 bytes), which Karatsuba splits."""
        return 4 * self.per

    def max_depth(self) -> int:
        """Karatsuba levels whose halves are whole tiles (8 chunks)."""
        d, h = 0, self.q_main
        while h % 16 == 0:
            d, h = d + 1, h // 2
        return d


DEPLOYMENT = Width(16)
GAP = 64
"""Zero bytes before and after every operand in the arena: A reads at most 31 bytes past either end, plus word alignment."""


def ntiles(q: int) -> int:
    return -(-q // 4)


def tile_valid(q: int, m: int, pi: int, i: int) -> bool:
    s0 = 8 * i - (m - pi) // 2
    return s0 < q and s0 + 8 > 0


def mma_count(q: int) -> int:
    return sum(tile_valid(q, m, m % 2, i) for m in range(2 * q + 2) for i in range(ntiles(q)))


def _words(arena: np.ndarray) -> np.ndarray:
    return arena.view("<u4").astype(np.uint64)


def a_fragment(arena: np.ndarray, base: int, m: int) -> np.ndarray:
    """Every lane's A registers for G_m (shape 32 x 4: a0 .. a3), as the kernel loads them from the arena."""
    words = _words(arena)
    lane = np.arange(32)
    g, kq = lane >> 2, lane & 3
    a = base + 16 * m + g - 8 * kq - 7
    wi, sh = a >> 2, (a & 3) * 8
    w = [((words[wi + k + 1] << 32 | words[wi + k]) >> sh.astype(np.uint64)) & 0xFFFFFFFF for k in range(4)]
    bs = [_bswap(x) for x in w]
    return np.stack([bs[1], bs[3], bs[0], bs[2]], axis=1)


def _bswap(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64)
    return ((x & 0xFF) << 24) | ((x >> 8 & 0xFF) << 16) | ((x >> 16 & 0xFF) << 8) | (x >> 24 & 0xFF)


def b_fragment(arena: np.ndarray, base: int, q: int, s0: int) -> np.ndarray:
    """Every lane's B registers (32 x 2) for chunks s0 .. s0 + 7, zero for chunks outside [0, q)."""
    words = _words(arena)
    lane = np.arange(32)
    g, kq = lane >> 2, lane & 3
    s = s0 + g
    ok = (s >= 0) & (s < q)
    a = base + 32 * np.where(ok, s, 0) + 8 * kq
    b0 = np.where(ok, words[a >> 2], 0)
    b1 = np.where(ok, words[(a >> 2) + 1], 0)
    return np.stack([b0, b1], axis=1)


def _unpack(reg: np.ndarray) -> np.ndarray:
    return np.stack([(reg >> (8 * e)) & 0xFF for e in range(4)], axis=-1).astype(np.int64)


def mma(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """m16n8k32 u8 x u8 -> s32 from per-lane fragments (PTX ISA's layouts); D as 32 x 4 (d0 .. d3)."""
    A = np.zeros((16, 32), dtype=np.int64)
    B = np.zeros((32, 8), dtype=np.int64)
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        A[g, 4 * t:4 * t + 4] = _unpack(a[lane, 0])
        A[g + 8, 4 * t:4 * t + 4] = _unpack(a[lane, 1])
        A[g, 16 + 4 * t:20 + 4 * t] = _unpack(a[lane, 2])
        A[g + 8, 16 + 4 * t:20 + 4 * t] = _unpack(a[lane, 3])
        B[4 * t:4 * t + 4, g] = _unpack(b[lane, 0])
        B[16 + 4 * t:20 + 4 * t, g] = _unpack(b[lane, 1])
    D = A @ B
    lane = np.arange(32)
    g, t = lane >> 2, lane & 3
    return np.stack([D[g, 2 * t], D[g, 2 * t + 1], D[g + 8, 2 * t], D[g + 8, 2 * t + 1]], axis=1)


def d_positions(pi: int, i: int) -> np.ndarray:
    """Coefficient index of every lane's d0 .. d3 in tile (pi, i)."""
    lane = np.arange(32)
    g, kq = lane >> 2, lane & 3
    k = 16 * pi + 256 * i + 64 * kq + g
    return np.stack([k, k + 32, k + 8, k + 40], axis=1)


def arena_of(operands: list[bytes], q: int) -> tuple[np.ndarray, list[int]]:
    """Operands (each at most 32 q bytes, little-endian) in zero-gapped slots of 32 q + GAP bytes; their data offsets."""
    slot = 32 * q + GAP
    arena = np.zeros(GAP + slot * len(operands) + 8, dtype=np.uint8)
    bases = []
    for j, x in enumerate(operands):
        base = GAP + slot * j
        arena[base:base + len(x)] = np.frombuffer(x, dtype=np.uint8)
        bases.append(base)
    return arena, bases


def tile_square(arena: np.ndarray, base: int, q: int) -> np.ndarray:
    """Base square of the q-chunk operand at `base`: tiles[pi][i] as 32 x 4 per-lane accumulators."""
    nt = ntiles(q)
    acc = np.zeros((2, nt, 32, 4), dtype=np.int64)
    for m in range(2 * q + 2):
        pi = m % 2
        a = a_fragment(arena, base, m)
        for i in range(nt):
            if tile_valid(q, m, pi, i):
                acc[pi, i] += mma(a, b_fragment(arena, base, q, 8 * i - (m - pi) // 2))
    return acc


def scatter(tiles: np.ndarray, length: int) -> np.ndarray:
    out = np.zeros(length, dtype=np.int64)
    for pi in range(2):
        for i in range(tiles.shape[1]):
            pos = d_positions(pi, i)
            keep = pos < length
            out[pos[keep]] = tiles[pi, i][keep]
    return out


def tc_square(z: int, q: int) -> np.ndarray:
    """z^2's byte-convolution coefficients (length 64 q) by the dense tiling, z < 2^(256 q)."""
    arena, (base,) = arena_of([z.to_bytes(32 * q, "little")], q)
    return scatter(tile_square(arena, base, q), 64 * q)


@lru_cache(maxsize=None)
def karatsuba_plan(q: int, depth: int) -> tuple[tuple[str, tuple[int, ...], int], ...]:
    """Base operands of d levels (digits 0: low half, 1: high half, D: |low - high|), each with the tile offsets its square
    is added at and its sign: z^2 = z0^2 + X^2 z1^2 + X (z0^2 + z1^2 - D^2), X = 2^(8 h) at every level."""
    plan: list[tuple[str, tuple[int, ...], int]] = [("", (0,), 1)]
    h = q
    for _ in range(depth):
        h //= 2
        if h % 8:
            raise ValueError("a Karatsuba half is whole tiles")
        t = h // 8
        nxt = []
        for digits, offs, sign in plan:
            nxt.append((digits + "0", tuple(o + s for o in offs for s in (0, t)), sign))
            nxt.append((digits + "1", tuple(o + s for o in offs for s in (2 * t, t)), sign))
            nxt.append((digits + "D", tuple(o + t for o in offs), -sign))
        plan = nxt
    return tuple(plan)


def kara_operands(z: int, q: int, depth: int) -> list[int]:
    """Every base operand's value, in `karatsuba_plan` order."""
    out = []
    for digits, _, _ in karatsuba_plan(q, depth):
        x, size = z, 32 * q
        for dgt in digits:
            size //= 2
            lo, hi = x & ((1 << 8 * size) - 1), x >> 8 * size
            x = lo if dgt == "0" else hi if dgt == "1" else abs(lo - hi)
        out.append(x)
    return out


def kara_square(z: int, q: int, depth: int) -> tuple[np.ndarray, int]:
    """z^2's coefficients (length 64 q, signed) by `depth` Karatsuba levels over the tiling; and the MMA count."""
    plan = karatsuba_plan(q, depth)
    qb = q >> depth
    ops = kara_operands(z, q, depth)
    arena, bases = arena_of([x.to_bytes(32 * qb, "little") for x in ops], qb)
    out = np.zeros((2, ntiles(q), 32, 4), dtype=np.int64)
    for (_, offs, sign), base in zip(plan, bases):
        t = tile_square(arena, base, qb)
        for o in offs:
            out[:, o:o + t.shape[1]] += sign * t
    return scatter(out, 64 * q), len(plan) * mma_count(qb)


def coeff_value(c: np.ndarray, radix_bits: int = 8) -> int:
    return sum(int(v) << (radix_bits * k) for k, v in enumerate(c.tolist()))


def limb_sums(c: np.ndarray, width: Width) -> list[int]:
    """V_x = sum_t c[4x + t] 2^(8t) for x < 2 L (the kernels' int64 limb sums)."""
    cc = np.zeros(8 * width.limbs, dtype=np.int64)
    cc[:min(len(c), len(cc))] = c[:len(cc)]
    return [int(cc[4 * x]) + (int(cc[4 * x + 1]) << 8) + (int(cc[4 * x + 2]) << 16) + (int(cc[4 * x + 3]) << 24)
            for x in range(2 * width.limbs)]


def top_terms(z: int, width: Width) -> list[int]:
    """E_x: the limb sums of 2 M T X + T^2 X^2 (z = M + X T, X = 2^(32 * 32 PER), T the top 8 bytes), which the Karatsuba
    kernel adds in its fold from the limbs it holds: 2 (lo(M_{x-Lm} T0) + hi(M_{x-Lm-1} T0) + lo(M_{x-Lm-1} T1) +
    hi(M_{x-Lm-2} T1)) and T^2's four limbs at 2 Lm."""
    lm = 32 * width.per
    limb = [(z >> 32 * y) & 0xFFFFFFFF for y in range(width.limbs)]
    t0, t1 = limb[lm], limb[lm + 1]
    m = lambda y: limb[y] if 0 <= y < lm else 0  # noqa: E731
    lo, hi = (lambda a, b: (a * b) & 0xFFFFFFFF), (lambda a, b: (a * b) >> 32)
    tsq = (t0 | t1 << 32) ** 2
    e = []
    for x in range(2 * width.limbs):
        j = x - lm
        v = 2 * (lo(m(j), t0) + hi(m(j - 1), t0) + lo(m(j - 1), t1) + hi(m(j - 2), t1))
        if 0 <= x - 2 * lm < 4:
            v += (tsq >> 32 * (x - 2 * lm)) & 0xFFFFFFFF
        e.append(v)
    return e


BIAS_SHIFT = 30


def fold(v: list[int], width: Width, bounds: dict | None = None) -> int:
    """The kernels' reduction of limb sums V (2 L of them, signed) to a value = sum V_x 2^(32x) mod p: W_x = lo32(V_x) +
    (V_{x-1} >> 32), U_L = W_L + c0 W_{L+L} (+ c0^2 W_{2L} at L = 0), x_L = U_L + b_L with b = 2^30 p spread over the limbs
    (b_L = 2^30 (2^32 - 1), b_0 = 2^30 (2^32 - c0)); returns sum x_L 2^(32 L), records the extremes in `bounds`."""
    L, c0 = width.limbs, width.c0
    hi = lambda x: v[x] >> 32 if 0 <= x < 2 * L else 0  # noqa: E731
    W = [(v[x] & 0xFFFFFFFF) + hi(x - 1) for x in range(2 * L)] + [hi(2 * L - 1)]
    total = 0
    for i in range(L):
        u = W[i] + c0 * W[i + L] + (c0 * c0 * W[2 * L] if i == 0 else 0)
        b = (1 << BIAS_SHIFT) * ((1 << 32) - (c0 if i == 0 else 1))
        x = u + b
        if bounds is not None:
            bounds["v"] = max(bounds.get("v", 0), max(abs(t) for t in v))
            bounds["u"] = max(bounds.get("u", 0), abs(u))
            bounds["x_min"] = min(bounds.get("x_min", 1 << 64), x)
            bounds["x_max"] = max(bounds.get("x_max", 0), x)
        total += x << 32 * i
    return total


def d_map(width: Width, z: int) -> int:
    s = z * z % width.p
    return s if z % 2 == 0 else (-s) % width.p


def decode(width: Width, c: int, k: int, t: int) -> int:
    """P2's D(c) = d(sigma_K(d(c))) - t mod p at this width (`p2_v1.reference.decode`'s maps)."""
    r = d_map(width, c)
    x = r ^ k
    return (d_map(width, x if x < width.p else r) - t) % width.p


GOLDILOCKS = (1 << 64) - (1 << 32) + 1
NTT_N = 2048


@lru_cache(maxsize=None)
def ntt_root(n: int = NTT_N) -> int:
    """The primitive n-th root of unity the NTT kernel uses: 7^((P - 1) / n) (7 generates Goldilocks' group)."""
    return pow(7, (GOLDILOCKS - 1) // n, GOLDILOCKS)


def brev(i: int, bits: int) -> int:
    return int(format(i, f"0{bits}b")[::-1], 2)


def _butterflies(n: int, s: int):
    """Stage s's butterflies as the kernel indexes them: b -> (i0, i1, twiddle index j << (log n - 1 - s))."""
    logn = n.bit_length() - 1
    for b in range(n // 2):
        j = b & ((1 << s) - 1)
        i0 = ((b >> s) << (s + 1)) + j
        yield i0, i0 + (1 << s), j << (logn - 1 - s)


def ntt_forward(x: list[int], root: int) -> list[int]:
    """Cyclic NTT by decimation in time: bit-reversed in, natural out (`nttsq.cu`'s `forward`)."""
    x, P, n = list(x), GOLDILOCKS, len(x)
    tw = [pow(root, j, P) for j in range(n // 2)]
    for s in range(n.bit_length() - 1):
        for i0, i1, k in _butterflies(n, s):
            u, v = x[i0], x[i1] * tw[k] % P
            x[i0], x[i1] = (u + v) % P, (u - v) % P
    return x


def ntt_inverse(x: list[int], root: int) -> list[int]:
    """The unnormalized transform at root^-1 by decimation in frequency: natural in, bit-reversed out (`inverse`)."""
    x, P, n = list(x), GOLDILOCKS, len(x)
    itw = [pow(root, -j, P) for j in range(n // 2)]
    for s in reversed(range(n.bit_length() - 1)):
        for i0, i1, k in _butterflies(n, s):
            u, v = x[i0], x[i1]
            x[i0], x[i1] = (u + v) % P, (u - v) * itw[k] % P
    return x


def ntt_square(z: int, nbytes: int) -> list[int]:
    """z^2's 16-bit-limb convolution coefficients (length 2 * nbytes / 2 - 1) by the 2048-point cyclic NTT as the kernel
    runs it (limbs placed bit-reversed, forward, pointwise square times N^-1, inverse, read bit-reversed), with the wrap of
    the coefficients past 2047 recomputed directly and taken out."""
    limbs = [(z >> 16 * i) & 0xFFFF for i in range(nbytes // 2)]
    if len(limbs) > NTT_N:
        raise ValueError("too wide for one 2048-point NTT")
    bits = NTT_N.bit_length() - 1
    x = [0] * NTT_N
    for i, v in enumerate(limbs):
        x[brev(i, bits)] = v
    w, ninv = ntt_root(), pow(NTT_N, -1, GOLDILOCKS)
    f = ntt_forward(x, w)
    y = ntt_inverse([v * v % GOLDILOCKS * ninv % GOLDILOCKS for v in f], w)
    out_len = 2 * len(limbs) - 1
    wrap = [sum(limbs[i] * limbs[k - i] for i in range(k - len(limbs) + 1, len(limbs))) for k in range(NTT_N, out_len)]
    out = [y[brev(k, bits)] for k in range(NTT_N)] + wrap
    for k, top in enumerate(wrap):
        out[k] -= top
    return out[:out_len]


def limb_sums16(c: list[int], width: Width) -> list[int]:
    """V_x = c[2x] + c[2x + 1] 2^16 for x < 2 L, from 16-bit-limb coefficients (the NTT kernel's limb sums)."""
    cc = list(c) + [0] * (4 * width.limbs - len(c))
    return [cc[2 * x] + (cc[2 * x + 1] << 16) for x in range(2 * width.limbs)]
