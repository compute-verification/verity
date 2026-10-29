"""The lead's calls at the network certifier: it flips each call's coins once it holds the call's A root, and counts and
refuses the call's egress.

A **call** is one matmul of the lead (`Pouw.Lead.lead`). The prover registers its A root with its shape, `rows` rows of
A, `cols` output columns and inner dimension `K`, and later the root of its committed words; between the two the call is
**open**.

* **The coins** (`Pouw.Lead.Coins`). Once the certifier holds a call's A root, and not before, it flips the call's coins
  from its own entropy and releases them once, as its reply to that registration: the release follows the registration,
  never a clock. Over the call's committed shape (`CallShape`: m rows, n columns, S k32 steps, ⌈m/64⌉ coin tiles) they
  are
  - the tag bits b, m × S, and the step-1 bits a₁, m × 4, each a bit of entropy;
  - the tag weights G = g − 3 in [−3, 3], 4 × n: column j's four are the base-7 digits of the first 32-bit word (big
    endian) below `G_BOUND` = 7⁴ · ⌊2³² / 7⁴⌋, so they are exactly uniform;
  - the coin lanes s = 2u − 127, S × tiles × 3 × n, u the low 7 bits of a byte.
* **The count.** Every bucket a boundary link ticks while a call is open counts against it: ⌈log2(r + 1)⌉ timing bits at
  allocation r, and each real frame's content bits. A bucket inside several calls' spans counts against each. The
  control channel, which carries the registrations in and the coins out, ends at the certifier and is no boundary link.
  A call whose count passes its bits, cap · rows · n · K / fwdMacs (`Pouw.Lead.Layout.callBits`), is refused at once,
  compared in integers, and so is a call still open when the window closes. A refusal is the call's verdict in the
  record, never a dropped frame.

The record (`Calls.summary`) carries each call's shape, roots, coins' digest and verdict; the counts are witness.
"""

from __future__ import annotations

import enum
import os
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from verity.ml.commitments.identity import tagged_sha512

#: Rows a coin tile, and bits a coin lane's u (the lead's `tile` and `e`).
TILE, LANE_BITS = 64, 7
#: G's rejection bound: the largest multiple of 7⁴ = 2401 not above 2³², 2401 · 1,788,824. The 872 words at or above it
#: are rejected, so an accepted word's residue mod 2401, G's four digits, is uniform.
G_BOUND = 7**4 * (2**32 // 7**4)
#: The tag of the released stream's digest in the record.
STREAM_TAG = "verity-network-certifier/call-coins/v1"

#: The certifier's entropy: `entropy(k)` is the next k bytes.
Entropy = Callable[[int], bytes]

_BITS = tuple(bytes((x >> k) & 1 for k in range(8)) for x in range(256))
_LOW = bytes(x % 2**LANE_BITS for x in range(256))
_S8 = bytes((2 * u - (2**LANE_BITS - 1)) % 256 for u in range(2**LANE_BITS)) + bytes(256 - 2**LANE_BITS)
_G8 = bytes((g - 3) % 256 for g in range(7)) + bytes(249)


class CallRefusal(ValueError):
    """A registration or bucket the certifier refuses: malformed, out of order, or after the window closed."""


class Verdict(enum.Enum):
    """A call's verdict in the record."""

    ACCEPTED = "accepted"
    PAST_BITS = "past its bits"
    UNCLOSED = "open when the window closed"
    NO_COINS = "its coins were not flipped"


@dataclass(frozen=True, slots=True)
class CallShape:
    """A call's registered shape: `rows` rows of A, `cols` output columns, inner dimension `K`."""

    rows: int
    cols: int
    K: int

    def __post_init__(self) -> None:
        for name in ("rows", "cols", "K"):
            v = getattr(self, name)
            if type(v) is not int or v < 1:
                raise CallRefusal(f"a call's {name} is a positive integer, not {v!r}")

    @property
    def m(self) -> int:
        """Rows its words commit, over whole MMA tiles: 16⌈rows/16⌉ (`padRows`)."""
        return 16 * -(-self.rows // 16)

    @property
    def n(self) -> int:
        """Columns its words commit, over whole MMA tiles: 8⌈cols/8⌉ (`padCols`)."""
        return 8 * -(-self.cols // 8)

    @property
    def S(self) -> int:
        """k32 steps a word: step 0 holds 21 data lanes and each later step 28 (`stepsFor`)."""
        return 1 + (max(self.K - 21, 0) + 27) // 28

    @property
    def tiles(self) -> int:
        """Coin tiles: ⌈m/64⌉."""
        return -(-self.m // TILE)

    @property
    def coins(self) -> int:
        """Coins the call takes: m·S tag bits, 4m step-1 bits, 4n tag weights and S·tiles·3·n coin lanes."""
        m, n, S = self.m, self.n, self.S
        return m * S + 4 * m + 4 * n + S * self.tiles * 3 * n


@dataclass(frozen=True, slots=True)
class Coins:
    """A call's coins, one byte each in `Pouw.Lead.Coins`' index order: `b[i·S + t]` and `a1[i·4 + q]` in {0, 1},
    `g[q·n + j]` in [0, 7), `u[((t·tiles + k)·3 + ℓ)·n + j]` in [0, 128)."""

    shape: CallShape
    b: bytes
    a1: bytes
    g: bytes
    u: bytes

    def __post_init__(self) -> None:
        sh = self.shape
        for name, size, bound in (("b", sh.m * sh.S, 2), ("a1", sh.m * 4, 2), ("g", 4 * sh.n, 7),
                                  ("u", sh.S * sh.tiles * 3 * sh.n, 2**LANE_BITS)):
            v = getattr(self, name)
            if type(v) is not bytes or len(v) != size or max(v, default=0) >= bound:
                raise ValueError(f"{name} is {size} bytes below {bound} at {sh}")

    def G(self, q: int, j: int) -> int:
        """Tag weight G[q, j] = g − 3 (`Coins.G`)."""
        return self.g[q * self.shape.n + j] - 3

    def s(self, t: int, k: int, lane: int, j: int) -> int:
        """Coin lane s_t[k, ℓ, j] = 2u − 127 (`Coins.s`)."""
        sh = self.shape
        return 2 * self.u[((t * sh.tiles + k) * 3 + lane) * sh.n + j] - (2**LANE_BITS - 1)


def _read(entropy: Entropy, k: int) -> bytes:
    out = bytes(entropy(k))
    if len(out) != k:
        raise ValueError(f"the entropy gave {len(out)} bytes, not {k}")
    return out


def _bits(entropy: Entropy, count: int) -> bytes:
    """`count` bits of entropy, one a byte, each byte's least significant bit first."""
    return b"".join(map(_BITS.__getitem__, _read(entropy, -(-count // 8))))[:count]


def _weights(entropy: Entropy, n: int) -> bytes:
    """g for n columns: column j's digits are those of the j-th accepted 32-bit word, the words rejected at `G_BOUND`."""
    words: list[int] = []
    while len(words) < n:
        raw = _read(entropy, 4 * (n - len(words)))
        words += [w % 7**4 for w in struct.unpack(f">{len(raw) // 4}I", raw) if w < G_BOUND]
    return b"".join(bytes(w // 7**q % 7 for w in words) for q in range(4))


def flip(shape: CallShape, entropy: Entropy = os.urandom) -> Coins:
    """The call's coins from `entropy`, read in this order: b's bits, a₁'s bits, G's words, the lanes' bytes."""
    m, n, S = shape.m, shape.n, shape.S
    b = _bits(entropy, m * S)
    a1 = _bits(entropy, m * 4)
    g = _weights(entropy, n)
    u = _read(entropy, S * shape.tiles * 3 * n).translate(_LOW)
    return Coins(shape, b, a1, g, u)


def stream(coins: Coins) -> bytes:
    """The released stream: each coin one signed byte (two's complement), in `Pouw.Lead.CoinIdx`'s order: b and a₁ as
    0 or 1, then G in [−3, 3], then s, odd in [−127, 127]."""
    return coins.b + coins.a1 + coins.g.translate(_G8) + coins.u.translate(_S8)


@dataclass(frozen=True, slots=True)
class EgressCap:
    """The egress cap the window's workload registers: `cap` bits a token, apportioned among a forward's calls by their
    MACs out of `fwd_macs`, the forward's MACs a token (the lead's `cap` and `fwdMacs`)."""

    cap: int
    fwd_macs: int

    def __post_init__(self) -> None:
        if type(self.cap) is not int or self.cap < 0 or type(self.fwd_macs) is not int or self.fwd_macs < 1:
            raise ValueError(f"an egress cap is bits ≥ 0 over MACs ≥ 1, not {self.cap!r} over {self.fwd_macs!r}")

    def passed(self, shape: CallShape, count: int) -> bool:
        """Whether `count` bits pass the call's bits, cap · rows · n · K / fwdMacs (`Layout.callBits`), in integers."""
        return count * self.fwd_macs > self.cap * shape.rows * shape.n * shape.K


def timing_bits(allocation: int) -> int:
    """One link's bucket's timing bits at allocation r: ⌈log2(r + 1)⌉, as its real frames number 0 … r."""
    if type(allocation) is not int or allocation < 0:
        raise CallRefusal(f"an allocation is an integer ≥ 0, not {allocation!r}")
    return allocation.bit_length()


def _spent(k: int) -> bytes:
    raise ValueError("the certifier's entropy failed earlier in the window")


def _root(root: bytes, what: str) -> bytes:
    if not isinstance(root, bytes | bytearray) or len(root) != 64:
        raise CallRefusal(f"{what} is a 64-byte SHA-512 root")
    return bytes(root)


@dataclass(slots=True)
class _Call:
    shape: CallShape
    a_root: bytes
    verdict: Verdict | None
    coins: bytes = b""
    words_root: bytes | None = None
    count: int = 0


class Calls:
    """One window's calls on one node, as the network certifier keeps them: registered by the prover's call ids, each
    released its coins once, and counted while open."""

    def __init__(self, cap: EgressCap, entropy: Entropy = os.urandom) -> None:
        self.cap = cap
        self._entropy = entropy
        self._calls: dict[int, _Call] = {}
        self._open: set[int] = set()
        self._closed = False

    def _live(self) -> None:
        if self._closed:
            raise CallRefusal("the window is closed")

    def a_root(self, call: int, root: bytes, shape: CallShape) -> bytes:
        """Register call `call`'s A root and shape, open its count, and release its coins: the stream, once, as the only
        reply to this registration. A registration it refuses flips no coin. A call whose coins aren't flipped is
        refused (`NO_COINS`) and never opens, and so is every later call of the window: once the entropy fails, the
        certifier flips nothing more."""
        self._live()
        if type(call) is not int or call < 0:
            raise CallRefusal(f"a call id is an integer ≥ 0, not {call!r}")
        if call in self._calls:
            raise CallRefusal(f"call {call}'s A root is registered already, and its coins went out once")
        if not isinstance(shape, CallShape):
            raise CallRefusal(f"a call's shape is a CallShape, not {type(shape).__name__}")
        c = self._calls[call] = _Call(shape, _root(root, "an A root"), Verdict.NO_COINS)
        try:
            out = stream(flip(shape, self._entropy))
        except Exception:
            self._entropy = _spent
            raise
        c.coins, c.verdict = tagged_sha512(STREAM_TAG, out), None
        self._open.add(call)
        return out

    def bucket(self, allocation: int, frames: Sequence[int] = ()) -> list[int]:
        """Count one boundary link's bucket against every open call: ⌈log2(r + 1)⌉ timing bits at allocation r, and
        `frames`, each real frame's content bits (at most r of them). Returns the calls it refuses, in id order."""
        self._live()
        bits = timing_bits(allocation)
        if len(frames) > allocation or any(type(f) is not int or f < 0 for f in frames):
            raise CallRefusal(f"a bucket at allocation {allocation} holds at most {allocation} real frames, each its "
                              f"content bits ≥ 0, not {list(frames)!r}")
        bits += sum(frames)
        refused = []
        for call in sorted(self._open):
            c = self._calls[call]
            if c.verdict is None:
                c.count += bits
                if self.cap.passed(c.shape, c.count):
                    c.verdict = Verdict.PAST_BITS
                    refused.append(call)
        return refused

    def words_root(self, call: int, root: bytes) -> Verdict:
        """Register call `call`'s words root and close its count. Returns its verdict: accepted unless refused."""
        self._live()
        if call not in self._open:
            raise CallRefusal(f"call {call} is not open: its words root follows its A root's coins, once")
        c = self._calls[call]
        c.words_root = _root(root, "a words root")
        self._open.discard(call)
        if c.verdict is None:
            c.verdict = Verdict.ACCEPTED
        return c.verdict

    def close(self) -> dict[int, Verdict]:
        """Close the window: a call still open is refused (`UNCLOSED`) unless it was already. Returns every call's
        verdict, in id order."""
        self._live()
        for call in self._open:
            c = self._calls[call]
            if c.verdict is None:
                c.verdict = Verdict.UNCLOSED
        self._open.clear()
        self._closed = True
        return {call: c.verdict for call, c in sorted(self._calls.items())}

    def summary(self) -> list[dict]:
        """The record's calls, in id order: each one's shape, roots, coins' digest and verdict (None while it is open).
        The counts are witness and stay out."""
        return [{"call": call, "rows": c.shape.rows, "cols": c.shape.cols, "K": c.shape.K, "a_root": c.a_root.hex(),
                 "coins": c.coins.hex() or None, "words_root": c.words_root.hex() if c.words_root else None,
                 "verdict": c.verdict.value if c.verdict else None}
                for call, c in sorted(self._calls.items())]
