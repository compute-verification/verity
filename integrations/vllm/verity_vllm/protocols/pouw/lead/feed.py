"""The lead's coin feed: the coins the network certifier samples for one call of the lead (layout
``blc4a_t64_e7_s1_rot_fold``) once it holds the call's root(A_j), packed as the served kernel reads them, and the
in-process certifier that stands in for the certifier's coin link (`CoinLink`).

* **A call** (`Call`): its pass, its index in the pass, its linear's index among the registered linears and its shape
  m × K × n, which the up-link carries beside root(A_j) so the certifier can size the call's feed.
* **Its sizes** (`Shape`): S = 1 + ⌈max(K − 21, 0)/28⌉ k32 steps (step 0 holds 21 data lanes, each later step 28),
  H = ⌈(S + 3)/32⌉ row words, R = 16⌈m/16⌉ committed rows and C = 8⌈n/8⌉ committed columns (m and n padded to whole
  16 × 8 MMA tiles) and T = ⌈R/64⌉ coin tiles.
* **The coins** (`CallCoins`, each a small unsigned integer held as a byte): b [R, S] step bits, a1 [R, 4] step-0 bits,
  g [4, C] in 0..6 (the tag weight G = g − 3) and u [S, T, 3, C] in 0..127 (the coin lane s = 2u − 127). b[:, 0]
  enters no word.
* **The feed** (`Feed`, one release a call, in the order the link sends it):

  ============  ==================================================================================================
  row words     u32 [R, H]: row i's bit e is a1[i, e] (e < 4) and bit t + 3 is b[i, t] (t = 1 .. S − 1)
  G words       u32 [C]: byte e is the int8 G[e, j]. The certifier samples a u32 u and keeps it below `G_KEEP`; G[e, j]
                is digit e of u mod 2401 in base 7, less 3, so G is uniform on [−3, 3]⁴
  lane words    u32 [T, S, C]: bytes 0-2 the int8 coin lanes s[t, rt, ℓ, j]; byte 3 public, C[3, j] at step 0, else 1
  col sources   i32 [T, 2, C]: 0x4B400000 − σ, σ[rt, c, j] = Σ_t Σ_ℓ ROT_W[c][ℓ]·s[t, rt, ℓ, j], row class c = i mod 2
  ============  ==================================================================================================

  The kernel reads lane words and column sources at the stride of its computed columns N (`Feed.lanes`,
  `Feed.sources`), with zeros past C; rows from R on read zero row words.
* **The in-process certifier** (`InProcessCertifier`): samples a call's coins only once it holds the call's root,
  releases them once, and refuses a second release of a call, a root that isn't 64 bytes and a call the lead doesn't
  admit. Its coins are the serving process's own bytes, so a pass under it is a measurement and credits nothing (source
  ``in-process``, as `pipeline.commit_coins` names it).
"""
from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

LAYOUT = "blc4a_t64_e7_s1_rot_fold"
STEP0_LANES, LATER_LANES = 21, 28
ROW_TILE, COL_TILE, COIN_TILE = 16, 8, 64
COIN_BITS = 7
#: ROT_W[c][ℓ], coin lane ℓ's weight in a row of class c = i mod 2: 127 on lane c, 1 on the other two
ROT_W = np.array([[127, 1, 1], [1, 127, 1]], dtype=np.int64)
#: a sampled u32 is G's when below this: 2401 ⌊2³² / 2401⌋ (872 of the 2³² are rejected)
G_KEEP = 2401 * (2**32 // 2401)
#: 1.5 · 2²³: the fold reads a word plus its column source as this float plus the integer it leaves
MAGIC = 0x4B400000
#: the most MMA tiles of words the lead admits in a call, (R/16)·(C/8)·S
MAX_TILES = 2**40
ROOT_BYTES = 64
IN_PROCESS = "in-process"

#: ``rand(n)``: n bytes from the certifier's source
Rand = Callable[[int], bytes]


class FeedError(ValueError):
    """A release the certifier refuses."""


def steps(k: int) -> int:
    """k32 steps for inner dimension ``k``: step 0 holds 21 data lanes and each later step 28."""
    return 1 + (max(k - STEP0_LANES, 0) + LATER_LANES - 1) // LATER_LANES


def codes(cols: int) -> np.ndarray:
    """C[e, j] for j < ``cols``, int64 [4, cols]: the base-13 digits of j + 14,281 less 6, which are the low four balanced
    base-13 digits of j + 1."""
    c = np.arange(cols, dtype=np.int64) + 14281
    return np.stack([(c // 13**e) % 13 - 6 for e in range(4)])


@dataclass(frozen=True)
class Shape:
    """A call of ``m`` rows, inner dimension ``k`` and ``n`` columns, and the sizes of its coins and feed."""

    m: int
    k: int
    n: int

    @property
    def S(self) -> int:
        return steps(self.k)

    @property
    def H(self) -> int:
        return (self.S + 3 + 31) // 32

    @property
    def R(self) -> int:
        return ROW_TILE * (-(-self.m // ROW_TILE))

    @property
    def C(self) -> int:
        return COL_TILE * (-(-self.n // COL_TILE))

    @property
    def T(self) -> int:
        return -(-self.R // COIN_TILE)

    def computed(self, bn: int) -> int:
        """N, the columns a kernel of ``bn``-column tiles computes."""
        return bn * (-(-self.C // bn))

    def refusal(self) -> str | None:
        """Why the lead doesn't take the call, or None: it admits a call of at least one row and one column, an inner
        dimension of at least 1 and at most `MAX_TILES` MMA tiles of words."""
        if self.m < 1 or self.n < 1:
            return f"lead: a call of {self.m} x {self.n} has no words"
        if self.k < 1:
            return f"lead: inner dimension {self.k}; the lead admits K >= 1"
        if (tiles := (self.R // ROW_TILE) * (self.C // COL_TILE) * self.S) > MAX_TILES:
            return f"lead: {tiles} MMA tiles of words, past the 2^40 the lead admits"
        return None

    def feed_words(self) -> int:
        """The release's leading words: R·H row words, then C G words."""
        return self.R * self.H + self.C

    def lane_words(self) -> int:
        return self.T * self.S * self.C

    def source_words(self) -> int:
        return 2 * self.T * self.C


@dataclass(frozen=True)
class Call:
    """Call ``index`` of pass ``pass_index``, on registered linear ``linear``, of ``m`` × ``k`` × ``n``."""

    pass_index: int
    index: int
    linear: int
    m: int
    k: int
    n: int

    @property
    def shape(self) -> Shape:
        return Shape(self.m, self.k, self.n)


def _bytes(rand: Rand, count: int) -> np.ndarray:
    got = rand(count)
    if len(got) != count:
        raise FeedError(f"the coin source gave {len(got)} bytes for {count}")
    return np.frombuffer(got, dtype=np.uint8)


def g_digits(u: int) -> tuple[int, int, int, int] | None:
    """g[0..3] for a sampled u32 ``u``: the base-7 digits of u mod 2401, or None when u is rejected (u ≥ `G_KEEP`)."""
    if u >= G_KEEP:
        return None
    v = u % 2401
    return v % 7, v // 7 % 7, v // 49 % 7, v // 343


@dataclass(frozen=True)
class CallCoins:
    """A call's coins: ``b`` [R, S] and ``a1`` [R, 4] in {0, 1}, ``g`` [4, C] in 0..6 and ``u`` [S, T, 3, C] in 0..127,
    all uint8."""

    shape: Shape
    b: np.ndarray
    a1: np.ndarray
    g: np.ndarray
    u: np.ndarray

    def __post_init__(self) -> None:
        sh = self.shape
        want = {"b": ((sh.R, sh.S), 1), "a1": ((sh.R, 4), 1), "g": ((4, sh.C), 6), "u": ((sh.S, sh.T, 3, sh.C), 127)}
        for name, (dims, top) in want.items():
            a = getattr(self, name)
            if a.shape != dims or a.dtype != np.uint8 or (a.size and int(a.max()) > top):
                raise FeedError(f"coins.{name}: {a.dtype} {a.shape}, want uint8 {dims} in 0..{top}")

    @classmethod
    def sample(cls, shape: Shape, rand: Rand) -> CallCoins:
        """The call's coins from ``rand``'s bytes: a byte for each bit and each 7-bit coin (its low bits), and G by
        rejection from little-endian u32s."""
        sh = shape
        b = (_bytes(rand, sh.R * sh.S) & 1).reshape(sh.R, sh.S)
        a1 = (_bytes(rand, sh.R * 4) & 1).reshape(sh.R, 4)
        g = np.empty((4, sh.C), dtype=np.uint8)
        j = 0
        while j < sh.C:
            for u in _bytes(rand, 4 * (sh.C - j)).view("<u4").tolist():
                if (d := g_digits(u)) is not None:
                    g[:, j] = d
                    j += 1
        u = (_bytes(rand, sh.S * sh.T * 3 * sh.C) & 0x7F).reshape(sh.S, sh.T, 3, sh.C)
        return cls(sh, b, a1, g, u)

    def s(self) -> np.ndarray:
        """The coin lanes s = 2u − 127, int64 [S, T, 3, C]."""
        return 2 * self.u.astype(np.int64) - (2**COIN_BITS - 1)

    def G(self) -> np.ndarray:
        """The tag weights G = g − 3, int64 [4, C]."""
        return self.g.astype(np.int64) - 3


def _signed_bytes(v: np.ndarray) -> np.ndarray:
    return (v.astype(np.int64) & 0xFF).astype(np.uint32)


@dataclass(frozen=True)
class Feed:
    """A call's release: its row words, G words, lane words (committed-column stride) and σ, packed from its coins."""

    call: Call
    rows: np.ndarray
    g: np.ndarray
    lanes_c: np.ndarray
    sigma: np.ndarray

    @classmethod
    def of(cls, call: Call, coins: CallCoins) -> Feed:
        sh = call.shape
        if coins.shape != sh:
            raise FeedError(f"coins of {coins.shape} for a call of {sh}")
        bits = np.zeros((sh.R, 32 * sh.H), dtype=np.uint8)
        bits[:, :4] = coins.a1
        bits[:, 4:sh.S + 3] = coins.b[:, 1:]
        rows = np.packbits(bits, axis=1, bitorder="little").view("<u4").astype(np.uint32)
        gw = np.zeros(sh.C, dtype=np.uint32)
        for e, ge in enumerate(coins.G()):
            gw |= _signed_bytes(ge) << np.uint32(8 * e)
        s = coins.s()
        lanes = np.zeros((sh.T, sh.S, sh.C), dtype=np.uint32)
        for ell in range(3):
            lanes |= _signed_bytes(s[:, :, ell, :].transpose(1, 0, 2)) << np.uint32(8 * ell)
        lanes[:, 0, :] |= _signed_bytes(codes(sh.C)[3]) << np.uint32(24)
        lanes[:, 1:, :] |= np.uint32(1 << 24)
        sigma = np.einsum("cl,stlj->tcj", ROT_W, s)
        return cls(call, rows, gw, lanes, sigma)

    def head(self) -> np.ndarray:
        """The release's leading words, u32 [R·H + C]: the row words, then the G words."""
        return np.concatenate([self.rows.reshape(-1), self.g])

    def lanes(self, n: int) -> np.ndarray:
        """The lane words at the kernel's stride ``n`` (its computed columns), u32 [T·S·n] at (rt·S + t)·n + j."""
        return self._strided(self.lanes_c, n, np.uint32)

    def sources(self, n: int) -> np.ndarray:
        """The column sources at stride ``n``, i32 [2·T·n] at (rt·2 + c)·n + j: 0x4B400000 − σ, and 0 past C."""
        return self._strided(MAGIC - self.sigma, n, np.int32)

    def _strided(self, a: np.ndarray, n: int, dtype: type) -> np.ndarray:
        c = self.call.shape.C
        if n < c:
            raise FeedError(f"stride {n} is below the call's {c} committed columns")
        out = np.zeros((*a.shape[:-1], n), dtype=dtype)
        out[..., :c] = a
        return out.reshape(-1)


class CoinLink(Protocol):
    """The serving side's coin link: ``release(call, root)`` sends the call and its root(A_j) up and returns its feed."""

    source: str

    def release(self, call: Call, root: bytes) -> Feed: ...


class InProcessCertifier:
    """`CoinLink` on the serving process's own bytes (``rand``): a measurement, whose coins credit nothing."""

    source = IN_PROCESS

    def __init__(self, rand: Rand = secrets.token_bytes) -> None:
        self.rand = rand
        #: (pass, call) -> the root each released call's coins followed
        self.roots: dict[tuple[int, int], bytes] = {}

    def release(self, call: Call, root: bytes) -> Feed:
        if len(root) != ROOT_BYTES:
            raise FeedError(f"call {call.index}: a root of {len(root)} bytes; root(A_j) is {ROOT_BYTES}")
        if (key := (call.pass_index, call.index)) in self.roots:
            raise FeedError(f"pass {key[0]} call {key[1]}: released already; a call's coins are released once")
        if (why := call.shape.refusal()) is not None:
            raise FeedError(why)
        self.roots[key] = bytes(root)
        return Feed.of(call, CallCoins.sample(call.shape, self.rand))
