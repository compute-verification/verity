"""The receipt φ candidates of `phi.cu`, word for word in Python (no numpy, so `gate.py` runs on a node's bare Python).

Each candidate absorbs committed words p into a chain value, keyed by position t (KA/KB odd, KK/KC addends, all from
`keys(kseed)` exactly as `phi.cu`'s `set_keys`). `absorb(kind, state, words, t0, keys)` takes the state as (y, z): z is
used only by the wide kinds, mum (a 64-bit state, y = hi, z = lo) and duo (two words), which `finish` folds into the
32-bit receipt. duo2's state is (y, z, y2, z2): two duo lanes absorbing the same word, lane 2 with multiplier KB,
addend KC and its own rotations. The paired kinds (ram, xma) take words two at a time, keyed by the first word's
position.

lut's nonlinearity is three keyed tables in shared memory (`lut_tables(kseed)`): the word's bits 0-10, 11-21 and 22-31
index T0 (2048 entries), T1 (2048) and T2 (1024), and their XOR enters a 128-bit state (y, z, y2, z2) through one word,
mixed by one rotation and two XORs. The word reaches the state only through the tables. Its receipt runs `LUT_BLANK`
rounds that feed the state's own y ^ y2 through the tables, then folds the four words.

``OPS`` is the SASS each candidate should compile to per absorbed word; `sass_mix.py` counts what it does compile to.
"""

from __future__ import annotations

import functools

M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1
NKEY = 32
KINDS = ("mad", "ra", "ram", "xma", "ar", "mum", "xsm", "xsm4", "mxm", "chi", "xsm2", "duo", "duo2", "lut")
PAIRED = ("ram", "xma")
WIDE = ("mum", "duo", "duo2", "lut")
FOUR = ("duo2", "lut")
LUT_SIZES = (2048, 2048, 1024)
LUT_BLANK = 8
OPS = {
    "mad": "IMAD (y*a_t + p)",
    "ra": "LEA.HI or SHF.L.W+IADD (rotl(y, r_t) + p)",
    "ram": "per pair: rotl+add (LEA.HI), IMAD (*a_t + p1)",
    "xma": "per pair: LOP3 (y^p0^k_t), IMAD (*a_t + p1)",
    "ar": "IADD3 (y+p+k_t), SHF.L.W (rot r_t)",
    "mum": "LOP3 (hi^lo^p), IMAD.WIDE.U32 (*a_t + S, 64-bit)",
    "xsm": "SHF.R (y>>16), LOP3 (y^y>>16^p), IMAD (*a_t)",
    "xsm4": "LOP3 (y^p), SHF.R, LOP3 (^k_t), IMAD (*a_t)",
    "mxm": "IMAD (y*a_t + p), SHF.R, LOP3 (^k_t), IMAD (*b_t)",
    "chi": "LOP3 (y^p^k_t), 2 SHF.L.W (rot 1, rot 2), LOP3 (u ^ ~r1 & r2)",
    "xsm2": "SHF, LOP3, IMAD, SHF, LOP3, IMAD",
    "duo": "LOP3 (y^p^k_t), IMAD (u*a_t + z), LEA.HI or SHF.L.W+IADD (rotl(u, r_t) + y')",
    "duo2": "duo twice: lane 1 (a_t, k_t, R1), lane 2 (b_t, c_t, R2) on the same word",
    "lut": "3 field extracts (LOP3, SHF+LOP3, SHF), 3 LDS, LOP3 (T0^T1^T2), SHF.L.W (rot R1), LOP3 (z^x^u), LOP3 (y2^u)",
}

# duo2's rotations: lane 1 a permutation of 1..31 (period 31), lane 2 of 2..30 (period 29), so the two lanes repeat a
# rotation together only 899 words apart. In each lane no 2, 3 or 4 consecutive rotations sum to a multiple of 32, and
# the lanes differ at every position of a 256-word block.
R1 = (15, 18, 20, 5, 29, 25, 4, 9, 1, 6, 12, 28, 8, 7, 31, 19, 10, 17, 3, 11, 27, 14, 2, 26, 16, 30, 13, 23, 21, 24, 22)
R2 = (6, 23, 7, 19, 4, 14, 10, 12, 9, 28, 11, 20, 26, 2, 22, 17, 5, 16, 15, 3, 25, 8, 13, 29, 30, 27, 21, 24, 18)


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


class Keys:
    def __init__(self, ka, kb, kk, kc=None, lt=None):
        self.ka, self.kb, self.kk = list(ka), list(kb), list(kk)
        self.kc = list(kc) if kc is not None else [0] * len(self.ka)
        self.lt = list(lt) if lt is not None else [0] * sum(LUT_SIZES)


def lut_tables(kseed: int) -> list[int]:
    """lut's T0, T1, T2 back to back, as `phi.cu`'s `set_keys` fills them (a window's key would expand them by SHA-512)."""
    return [gen(kseed, 5 + j, i, 0) for j, n in enumerate(LUT_SIZES) for i in range(n)]


@functools.lru_cache(maxsize=8)
def keys(kseed: int) -> Keys:
    ka = [gen(kseed, 1, i, 0) | 1 for i in range(NKEY)]
    kb = [gen(kseed, 2, i, 0) | 1 for i in range(NKEY)]
    kk = [gen(kseed, 3, i, 0) for i in range(NKEY)]
    kc = [gen(kseed, 4, i, 0) for i in range(NKEY)]
    return Keys(ka, kb, kk, kc, lut_tables(kseed))


def lut_word(lt: list[int], w: int) -> int:
    return lt[w & 0x7FF] ^ lt[2048 + ((w >> 11) & 0x7FF)] ^ lt[4096 + (w >> 22)]


def absorb(kind: str, st: tuple[int, ...], p: list[int], t0: int, k: Keys) -> tuple[int, ...]:
    if kind == "lut":
        y, z, y2, z2 = st
        for i, w in enumerate(p):
            u = rotl(y, R1[(t0 + i) % 31])
            y, z, y2, z2 = z ^ lut_word(k.lt, w) ^ u, y2 ^ u, z2, y
        return y, z, y2, z2
    if kind == "duo2":
        y, z, y2, z2 = st
        for i, w in enumerate(p):
            t = t0 + i
            u = y ^ w ^ k.kk[t]
            y = (u * k.ka[t] + z) & M32
            z = (rotl(u, R1[t % 31]) + y) & M32
            u = y2 ^ w ^ k.kc[t]
            y2 = (u * k.kb[t] + z2) & M32
            z2 = (rotl(u, R2[t % 29]) + y2) & M32
        return y, z, y2, z2
    y, z = st
    n = len(p)
    step = 2 if kind in PAIRED else 1
    assert n % step == 0
    for i in range(0, n, step):
        t = t0 + i
        a, b, c = k.ka[t], k.kb[t], k.kk[t]
        if kind == "mad":
            y = (y * a + p[i]) & M32
        elif kind == "ra":
            y = (rotl(y, rot_of(t)) + p[i]) & M32
        elif kind == "ram":
            y = (rotl(y, rot_of(t)) + p[i]) & M32
            y = (y * a + p[i + 1]) & M32
        elif kind == "xma":
            y = ((y ^ p[i] ^ c) * a + p[i + 1]) & M32
        elif kind == "ar":
            y = rotl((y + p[i] + c) & M32, rot_of(t))
        elif kind == "mum":
            u = y ^ z ^ p[i]
            w = (u * a + ((y << 32) | z)) & M64
            y, z = w >> 32, w & M32
        elif kind == "xsm":
            y = ((y ^ (y >> 16) ^ p[i]) * a) & M32
        elif kind == "xsm4":
            v = y ^ p[i]
            y = ((v ^ (v >> 16) ^ c) * a) & M32
        elif kind == "mxm":
            u = (y * a + p[i]) & M32
            y = ((u ^ (u >> 16) ^ c) * b) & M32
        elif kind == "chi":
            u = y ^ p[i] ^ c
            y = u ^ ((~rotl(u, 1) & M32) & rotl(u, 2))
        elif kind == "xsm2":
            u = ((y ^ (y >> 16) ^ p[i]) * a) & M32
            y = ((u ^ (u >> 13) ^ c) * b) & M32
        elif kind == "duo":
            u = y ^ p[i] ^ c
            y = (u * a + z) & M32
            z = (rotl(u, rot_of(t)) + y) & M32
        else:
            raise ValueError(kind)
    return y, z


def finish(kind: str, st: tuple[int, ...], k: Keys | None = None) -> int:
    """The block's receipt: mum's hi ^ lo, duo's y ^ rotl(z, 16), duo2's lanes so folded with lane 2 rotated by 8,
    lut's four words so folded after `LUT_BLANK` rounds on y ^ y2 (which need the key's tables), every other kind's y."""
    if kind == "lut":
        assert k is not None, "lut's receipt reads the tables"
        for r in range(LUT_BLANK):
            st = absorb("lut", st, [st[0] ^ st[2]], r, k)
        y, z, y2, z2 = st
        return y ^ rotl(z, 8) ^ rotl(y2, 16) ^ rotl(z2, 24)
    if kind == "duo2":
        y, z, y2, z2 = st
        return y ^ rotl(z, 16) ^ rotl(y2 ^ rotl(z2, 16), 8)
    y, z = st
    return y ^ z if kind == "mum" else y ^ rotl(z, 16) if kind == "duo" else y


PU, PW = 4, 8


def pure_chains(kind: str, iters: int, seed: int, kseed: int, tid: int) -> list[int]:
    """`pure_k<kind>`'s PU output words for thread `tid`: chain u absorbs chain u+1's state (and u+2's, for a pair)."""
    k = keys(kseed)
    n = 4 if kind in FOUR else 2
    st = [tuple(gen(seed, tid, u, 100 + j) for j in range(n)) for u in range(PU)]
    step = 2 if kind in PAIRED else 1
    for _ in range(iters):
        for r in range(0, PW, step):
            w = [[st[(u + 1) % PU][0], st[(u + 2) % PU][0]][:step] for u in range(PU)]
            st = [absorb(kind, st[u], w[u], r, k) for u in range(PU)]
    return [finish(kind, s, k) for s in st]
