"""SHA-512, SHAKE256 and TurboSHAKE128 in gates, over the 32-bit halves of `words`.

* `Sha512Compress` (a Definition): one SHA-512 compression, the state and the block as (hi, lo) halves of their 64-bit
  words, big-endian, as FIPS 180-4 has them.
* `KeccakF1600` (a Definition): the Keccak-f[1600] permutation, lane ``x + 5y`` as its (lo, hi) halves, little-endian;
  `KeccakP1600R12` its last 12 rounds, TurboSHAKE's Keccak-p[1600, 12].
* `sha512(e, words)`: SHA-512 of the message whose big-endian bytes are ``words`` (64-bit words, each half a gate or an int),
  padded as FIPS 180-4 pads it; the digest as 16 halves (hi, lo per word).
* `shake256(e, lanes, n)`, `turboshake128(e, lanes, n, domain)`: the first ``n`` bytes of SHAKE256 or TurboSHAKE128 of the
  message whose little-endian bytes are ``lanes`` (64-bit lanes, each half a gate or an int), as 4-byte half-lanes in byte
  order.

Constant halves are Python ints until a gate reads them; `Emit` turns them into one constant node per value and body.
Messages here are whole words, so a message's byte length is a multiple of 8.
"""
from __future__ import annotations

from typing import Sequence, Union

from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import Coll, array_of
from verity.ml.circuits.types import Array
from verity.ml.circuits.ops.prims import const

from . import words as W

Half = Union[Coll, int]
Pair = tuple[Half, Half]
M32 = W.M32


class Emit:
    """A body under construction: constants deduplicated, and the word operations as calls."""

    def __init__(self, B) -> None:
        self.B = B
        self._consts: dict[int, Coll] = {}

    def c(self, v: int) -> Coll:
        v &= M32
        got = self._consts.get(v)
        if got is None:
            got = self._consts[v] = self.B.call(const(32, v))
        return got

    def g(self, h: Half) -> Coll:
        return self.c(h) if isinstance(h, int) else h

    def call(self, fn, *args: Half) -> Half:
        if all(isinstance(a, int) for a in args):
            return fn.evaluate(*args)
        return self.B.call(fn, *(self.g(a) for a in args))

    # 64-bit words as (hi, lo) for SHA-512
    def add(self, p: Pair, q: Pair) -> Pair:
        return self.call(W.PouwAdd64Hi, p[0], p[1], q[0], q[1]), self.call(W.PouwAdd64Lo, p[1], q[1])

    def sigma(self, name: str, p: Pair) -> Pair:
        return (self.call(getattr(W, f"PouwSha512{name}Hi"), p[0], p[1]),
                self.call(getattr(W, f"PouwSha512{name}Lo"), p[0], p[1]))

    def ch(self, e: Pair, f: Pair, g: Pair) -> Pair:
        return self.call(W.PouwCh32, e[0], f[0], g[0]), self.call(W.PouwCh32, e[1], f[1], g[1])

    def maj(self, a: Pair, b: Pair, c: Pair) -> Pair:
        return self.call(W.PouwMaj32, a[0], b[0], c[0]), self.call(W.PouwMaj32, a[1], b[1], c[1])

    # 64-bit lanes as (lo, hi) for Keccak
    def xor(self, p: Pair, q: Pair) -> Pair:
        return self.call(W.PouwXor32, p[0], q[0]), self.call(W.PouwXor32, p[1], q[1])

    def rotl(self, p: Pair, c: int) -> Pair:
        if c % 64 == 0:
            return p
        return self.call(W.PouwRotl64Lo, p[0], p[1], c), self.call(W.PouwRotl64Hi, p[0], p[1], c)

    def chi(self, a: Pair, b: Pair, c: Pair) -> Pair:
        return self.call(W.PouwChi32, a[0], b[0], c[0]), self.call(W.PouwChi32, a[1], b[1], c[1])

    def array(self, halves: Sequence[Half]) -> Coll:
        return array_of([self.g(h) for h in halves])


# ---- SHA-512 -------------------------------------------------------------------------------------------------------------

SHA512_K = (
    0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc, 0x3956c25bf348b538, 0x59f111f1b605d019,
    0x923f82a4af194f9b, 0xab1c5ed5da6d8118, 0xd807aa98a3030242, 0x12835b0145706fbe, 0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235, 0xc19bf174cf692694, 0xe49b69c19ef14ad2, 0xefbe4786384f25e3,
    0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65, 0x2de92c6f592b0275, 0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5,
    0x983e5152ee66dfab, 0xa831c66d2db43210, 0xb00327c898fb213f, 0xbf597fc7beef0ee4, 0xc6e00bf33da88fc2, 0xd5a79147930aa725,
    0x06ca6351e003826f, 0x142929670a0e6e70, 0x27b70a8546d22ffc, 0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed, 0x53380d139d95b3df,
    0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b, 0xa2bfe8a14cf10364, 0xa81a664bbc423001,
    0xc24b8b70d0f89791, 0xc76c51a30654be30, 0xd192e819d6ef5218, 0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8, 0x1e376c085141ab53, 0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8, 0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb,
    0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3, 0x748f82ee5defb2fc, 0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec,
    0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915, 0xc67178f2e372532b, 0xca273eceea26619c, 0xd186b8c721c0c207,
    0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178, 0x06f067aa72176fba, 0x0a637dc5a2c898a6, 0x113f9804bef90dae, 0x1b710b35131c471b,
    0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc, 0x431d67c49c100d4c, 0x4cc5d4becb3e42b6, 0x597f299cfc657e2a,
    0x5fcb6fab3ad6faec, 0x6c44198c4a475817,
)
SHA512_IV = (0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1, 0x510e527fade682d1,
             0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179)


def _pair(v: int) -> Pair:
    return v >> 32, v & M32


def _compress_body(B, S, st, blk):
    e = Emit(B)
    w: list[Pair] = [(blk[2 * t], blk[2 * t + 1]) for t in range(16)]
    for t in range(16, 80):
        s = e.add(e.sigma("SmallSigma1", w[t - 2]), w[t - 7])
        w.append(e.add(e.add(s, e.sigma("SmallSigma0", w[t - 15])), w[t - 16]))
    init = [(st[2 * i], st[2 * i + 1]) for i in range(8)]
    a, b, c, d, f_e, f, g, h = init
    for t in range(80):
        t1 = e.add(e.add(e.add(e.add(h, e.sigma("BigSigma1", f_e)), e.ch(f_e, f, g)), _pair(SHA512_K[t])), w[t])
        t2 = e.add(e.sigma("BigSigma0", a), e.maj(a, b, c))
        h, g, f, f_e, d, c, b, a = g, f, f_e, e.add(d, t1), c, b, a, e.add(t1, t2)
    out = [e.add(x, y) for x, y in zip(init, (a, b, c, d, f_e, f, g, h))]
    return e.array([hv for p in out for hv in p])


Sha512Compress = bind(CompositeDefinition(
    "PouwSha512Compress", 1, (), lambda S: ((("state", Array(16, W.U32)), ("block", Array(32, W.U32))), Array(16, W.U32)),
    _compress_body, doc="One SHA-512 compression (FIPS 180-4), each 64-bit word as its (hi, lo) halves.",
    conformance="protocol-defined"))


def sha512(e: Emit, words: Sequence[Pair]) -> list[Half]:
    """SHA-512 of the big-endian bytes of ``words``: the 8-word digest as 16 halves (hi, lo per word)."""
    n = len(words)
    msg = list(words) + [_pair(1 << 63)]
    while len(msg) % 16 != 14:
        msg.append((0, 0))
    msg += [(0, 0), _pair(64 * n)]                          # the 128-bit message length in bits
    state: list[Half] = [h for v in SHA512_IV for h in _pair(v)]
    for blk in range(0, len(msg), 16):
        halves = [h for p in msg[blk:blk + 16] for h in p]
        state = list(e.B.call(Sha512Compress, e.array(state), e.array(halves)))
    return state


# ---- Keccak-f[1600] and SHAKE256 -------------------------------------------------------------------------------------------

KECCAK_RC = (
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000, 0x000000000000808B, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800A, 0x800000008000000A, 0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
)
#: ρ's rotation offsets, ``KECCAK_ROT[x][y]``
KECCAK_ROT = ((0, 36, 3, 41, 18), (1, 44, 10, 45, 2), (62, 6, 43, 15, 61), (28, 55, 25, 21, 56), (27, 20, 39, 8, 14))
RATE_LANES = 17                                              # SHAKE256: 136-byte rate


def _keccak_body(B, S, s, first: int = 0):
    e = Emit(B)
    a = [[(s[2 * (x + 5 * y)], s[2 * (x + 5 * y) + 1]) for y in range(5)] for x in range(5)]
    for r in range(first, 24):
        # θ: each column's parity folds in lane y = 0 first, so a message whose y = 0 lanes carry a unit's own index makes
        # every gate of the first round depend on that index (no gate a function of values two units share)
        col = []
        for x in range(5):
            p = a[x][0]
            for y in range(1, 5):
                p = e.xor(p, a[x][y])
            col.append(p)
        d = [e.xor(col[(x - 1) % 5], e.rotl(col[(x + 1) % 5], 1)) for x in range(5)]
        a = [[e.xor(a[x][y], d[x]) for y in range(5)] for x in range(5)]
        b: list[list[Pair]] = [[(0, 0)] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                b[y][(2 * x + 3 * y) % 5] = e.rotl(a[x][y], KECCAK_ROT[x][y])
        a = [[e.chi(b[x][y], b[(x + 1) % 5][y], b[(x + 2) % 5][y]) for y in range(5)] for x in range(5)]
        lo, hi = KECCAK_RC[r] & M32, KECCAK_RC[r] >> 32
        a[0][0] = (e.call(W.PouwXor32, a[0][0][0], lo) if lo else a[0][0][0],
                   e.call(W.PouwXor32, a[0][0][1], hi) if hi else a[0][0][1])
    return e.array([h for y in range(5) for x in range(5) for h in a[x][y]])


KeccakF1600 = bind(CompositeDefinition(
    "PouwKeccakF1600", 1, (), lambda S: ((("state", Array(50, W.U32)),), Array(50, W.U32)), _keccak_body,
    doc="The Keccak-f[1600] permutation (FIPS 202), lane x + 5y as its (lo, hi) halves.", conformance="protocol-defined"))

KeccakP1600R12 = bind(CompositeDefinition(
    "PouwKeccakP1600R12", 1, (), lambda S: ((("state", Array(50, W.U32)),), Array(50, W.U32)),
    lambda B, S, s: _keccak_body(B, S, s, 12),
    doc="Keccak-p[1600, 12], TurboSHAKE128's permutation (RFC 9861): Keccak-f[1600]'s last 12 rounds, lane x + 5y as its "
        "(lo, hi) halves.", conformance="protocol-defined"))
TS128_RATE_LANES = 21                                        # TurboSHAKE128: 168-byte rate


def _lane(v: int) -> Pair:
    return v & M32, v >> 32


def _sponge(e: Emit, lanes: Sequence[Pair], n: int, rate: int, pad_byte: int, perm) -> list[Half]:
    msg = list(lanes)
    at = len(msg) % rate
    pad = [(0, 0)] * (rate - at)
    first = pad[0]
    pad[0] = (first[0] ^ pad_byte, first[1])
    last = pad[-1]
    pad[-1] = (last[0], last[1] ^ 0x80000000)
    msg += pad
    state: list[Pair] = [(0, 0)] * 25
    for blk in range(0, len(msg), rate):
        block = msg[blk:blk + rate]
        if blk == 0:
            state = [block[i] if i < rate else (0, 0) for i in range(25)]
        else:
            state = [e.xor(state[i], block[i]) if i < rate else state[i] for i in range(25)]
        out = e.B.call(perm, e.array([h for p in state for h in p]))
        state = [(out[2 * i], out[2 * i + 1]) for i in range(25)]
    halves: list[Half] = []
    while True:
        for i in range(rate):
            halves += [state[i][0], state[i][1]]
            if 4 * len(halves) >= n:
                return halves[:(n + 3) // 4]
        out = e.B.call(perm, e.array([h for p in state for h in p]))
        state = [(out[2 * i], out[2 * i + 1]) for i in range(25)]


def shake256(e: Emit, lanes: Sequence[Pair], n: int) -> list[Half]:
    """The first ``n`` bytes of SHAKE256 of the little-endian bytes of ``lanes``, as ``ceil(n / 4)`` half-lanes."""
    return _sponge(e, lanes, n, RATE_LANES, 0x1F, KeccakF1600)


def turboshake128(e: Emit, lanes: Sequence[Pair], n: int, domain: int) -> list[Half]:
    """The first ``n`` bytes of TurboSHAKE128 (domain byte ``domain``) of the little-endian bytes of ``lanes``, as
    ``ceil(n / 4)`` half-lanes."""
    if not 0x01 <= domain <= 0x7F:
        raise ValueError("TurboSHAKE's domain byte is in 0x01 … 0x7F")
    return _sponge(e, lanes, n, TS128_RATE_LANES, domain, KeccakP1600R12)


def sha_words(data: bytes) -> list[Pair]:
    """Bytes (a multiple of 8) as SHA-512's big-endian word halves."""
    return [_pair(int.from_bytes(data[i:i + 8], "big")) for i in range(0, len(data), 8)]


def keccak_lanes(data: bytes) -> list[Pair]:
    """Bytes (a multiple of 8) as Keccak's little-endian lane halves."""
    return [_lane(int.from_bytes(data[i:i + 8], "little")) for i in range(0, len(data), 8)]


__all__ = ["Emit", "KeccakF1600", "KeccakP1600R12", "Sha512Compress", "keccak_lanes", "sha512", "sha_words", "shake256",
           "turboshake128"]
