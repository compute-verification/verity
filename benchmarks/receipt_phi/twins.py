"""The receipt φ candidates of `phi.cu`, word for word in Python (no numpy, so `gate.py` runs on a node's bare Python).

Each candidate absorbs committed words p into a chain value, keyed by position t (KA/KB odd, KK addends, all from
`keys(kseed)` exactly as `phi.cu`'s `set_keys`). `absorb(kind, state, words, t0, keys)` takes the state as (y, z): z is
used only by the wide kinds, mum (a 64-bit state, y = hi, z = lo) and duo (two words), which `finish` folds into the
32-bit receipt. The paired kinds (ram, xma) take words two at a time, keyed by the first word's position.

``OPS`` is the SASS each candidate should compile to per absorbed word; `sass_mix.py` counts what it does compile to.
"""

from __future__ import annotations

M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1
NKEY = 32
KINDS = ("mad", "ra", "ram", "xma", "ar", "mum", "xsm", "xsm4", "mxm", "chi", "xsm2", "duo")
PAIRED = ("ram", "xma")
WIDE = ("mum", "duo")
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
}


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
    def __init__(self, ka, kb, kk):
        self.ka, self.kb, self.kk = list(ka), list(kb), list(kk)


def keys(kseed: int) -> Keys:
    ka = [gen(kseed, 1, i, 0) | 1 for i in range(NKEY)]
    kb = [gen(kseed, 2, i, 0) | 1 for i in range(NKEY)]
    kk = [gen(kseed, 3, i, 0) for i in range(NKEY)]
    return Keys(ka, kb, kk)


def absorb(kind: str, st: tuple[int, int], p: list[int], t0: int, k: Keys) -> tuple[int, int]:
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


def finish(kind: str, st: tuple[int, int]) -> int:
    """The block's receipt: mum's hi ^ lo, duo's y ^ rotl(z, 16), every other kind's y."""
    y, z = st
    return y ^ z if kind == "mum" else y ^ rotl(z, 16) if kind == "duo" else y


PU, PW = 4, 8


def pure_chains(kind: str, iters: int, seed: int, kseed: int, tid: int) -> list[int]:
    """`pure_k<kind>`'s PU output words for thread `tid`: chain u absorbs chain u+1's state (and u+2's, for a pair)."""
    k = keys(kseed)
    st = [(gen(seed, tid, u, 100), gen(seed, tid, u, 101)) for u in range(PU)]
    step = 2 if kind in PAIRED else 1
    for _ in range(iters):
        for r in range(0, PW, step):
            w = [[st[(u + 1) % PU][0], st[(u + 2) % PU][0]][:step] for u in range(PU)]
            st = [absorb(kind, st[u], w[u], r, k) for u in range(PU)]
    return [finish(kind, s) for s in st]
