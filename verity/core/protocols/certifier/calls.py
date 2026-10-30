"""The lead's calls at the network certifier: it flips each call's coins once it holds the call's A root, releases them as
the coin feed the lead's kernel reads, and counts and refuses each forward's egress.

A **call** is one matmul of the lead (`Pouw.Lead.lead`), call j of forward f. The prover registers each call's A root
with its shape, `rows` rows of A, `cols` output columns, inner dimension `K` and the column tile `bn` (128 or 256) the
kernel runs it at, which orders its committed words, and once a forward the root of the forward's committed words. A
forward is **open** from its first call's A root to its words root.

* **The coins** (`Pouw.Lead.Coins`). Once the certifier holds a call's A root, and not before, it flips the call's coins
  from its own entropy and releases them once, as its reply to that registration: the release follows the registration,
  never a clock. The release is 32-bit words, little endian, every length fixed by the call's committed shape
  (`CallShape`: m rows, n columns, S k32 steps, H = ⌈(S + 3)/32⌉ row words a row, ⌈m/64⌉ coin tiles), in this order:
  - the row words, H a row: bit q of row i's is its step-1 bit a₁[i, q] (q < 4), bit t + 3 its tag bit b_t[i] at a step
    t from 1 to S − 1, and every other bit 0. Step 0 reads no tag bit (`Pouw.Lead.coinTerm`), so none is flipped;
  - the G words, one a column: byte q of column j's is the int8 G[q, j] = g − 3 in [−3, 3], digit q in base 7 of the
    first 32-bit word of entropy below `G_BOUND` = 7⁴ · ⌊2³² / 7⁴⌋, so G is exactly uniform;
  - then a coin tile at a time, the tile's lane-coin words, one a step and column: bytes 0–2 its coin lanes s = 2u − 127
    as int8, u the low 7 bits of a byte of entropy, and byte 3 public, the column code's top digit C[3, j]
    (`Pouw.Lead.colCode`) at step 0 and 1 at a later step; and the tile's column sources, one a row class (i mod 2) and
    column, β_j − σ mod 2³² with β_j the column's public bias (`bias`, from its code and S) and σ the class's weights
    (`Pouw.Lead.rotWeight`) against the tile's coin lanes summed over its steps. They are no coins but a public
    function of them, which the kernel's epilogue folds out; the fold's 0x4B400000 and each row's walk are in the row
    word the serving side builds, so the certifier needs neither.
* **The count, per forward.** Every bucket a boundary link ticks while a forward is open counts once against it:
  ⌈log2(r + 1)⌉ timing bits at allocation r, and each real frame's content bits. The forwards are serial: a forward's
  A root is refused while another forward is open, so no bucket is inside two spans. What the node's release at the
  boundary costs, `release_bits`, counts once, when the words root closes the forward. The control channel, which carries the registrations in and the releases out, ends at the certifier and
  is no boundary link. A forward whose count then passes its bits, the sum of its calls' cap · rows · n · K / fwdMacs
  (`Pouw.Lead.Layout.callBits`; 32 · rows bits at cap 32 for a whole forward), is refused, compared in integers, and so
  is a forward still open when the window closes. A refusal is the forward's verdict in the record, never a dropped
  frame.

The record (`Calls.summary`) carries each forward's calls, with their shapes (`bn` included, so the verifier rebuilds
the committed words' order), A roots and releases' digests, and its words root and verdict; the counts are witness.
"""

from __future__ import annotations

import enum
import os
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from verity.ml.commitments.identity import tagged_sha512

#: Rows a coin tile, and bits a coin lane's u (the lead's `tile` and `e`).
TILE, LANE_BITS = 64, 7
#: G's rejection bound: the largest multiple of 7⁴ = 2401 not above 2³², 2401 · 1,788,824. The 872 words at or above it
#: are rejected, so an accepted word's residue mod 2401, G's four digits, is uniform.
G_BOUND = 7**4 * (2**32 // 7**4)
#: The most MMA tiles a call's committed words may span over its steps, rowBlocks · colBlocks · S, for the verifier to
#: admit it (`Pouw.Lead.Layout.Admits`).
ADMITTED_TILES = 2**40
#: A row class's weights on the three coin lanes (`Pouw.Lead.rotWeight`): even rows (127, 1, 1), odd rows (1, 127, 1).
ROT_W = ((127, 1, 1), (1, 127, 1))
#: The column tiles the lead's kernel runs a call at.
BNS = (128, 256)
#: The tag of the release's digest in the record.
RELEASE_TAG = "verity-network-certifier/call-release/v1"

#: The certifier's entropy: `entropy(k)` is the next k bytes.
Entropy = Callable[[int], bytes]

_LOW = bytes(x % 2**LANE_BITS for x in range(256))
_S8 = bytes((2 * u - (2**LANE_BITS - 1)) % 256 for u in range(2**LANE_BITS)) + bytes(256 - 2**LANE_BITS)
_G_BYTES = frozenset((g - 3) % 256 for g in range(7))


class CallRefusal(ValueError):
    """A registration or bucket the certifier refuses: malformed, out of order, or after the window closed."""


class ForwardVerdict(enum.Enum):
    """A forward's verdict in the record."""

    ACCEPTED = "accepted"
    PAST_BITS = "past its bits"
    UNCLOSED = "open when the window closed"
    NO_COINS = "a call's coins were not flipped"


@dataclass(frozen=True, slots=True)
class CallShape:
    """A call's registered shape: `rows` rows of A, `cols` output columns, inner dimension `K`, one the verifier admits
    (`ADMITTED_TILES`), and the column tile `bn` the kernel runs it at, one of `BNS`."""

    rows: int
    cols: int
    K: int
    bn: int

    def __post_init__(self) -> None:
        for name in ("rows", "cols", "K"):
            v = getattr(self, name)
            if type(v) is not int or v < 1:
                raise CallRefusal(f"a call's {name} is a positive integer, not {v!r}")
        if type(self.bn) is not int or self.bn not in BNS:
            raise CallRefusal(f"a call's column tile bn is {' or '.join(map(str, BNS))}, not {self.bn!r}")
        if self.m // 16 * (self.n // 8) * self.S > ADMITTED_TILES:
            raise CallRefusal(f"a call of {self.rows} rows, {self.cols} columns and K {self.K} spans more than 2^40 MMA "
                              "tiles, which the verifier doesn't admit")

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
    def H(self) -> int:
        """Row words a row: ⌈(S + 3)/32⌉, for its four step-1 bits and S − 1 tag bits."""
        return (self.S + 3 + 31) // 32

    @property
    def tiles(self) -> int:
        """Coin tiles: ⌈m/64⌉."""
        return -(-self.m // TILE)

    @property
    def coins(self) -> int:
        """Coins the call takes: S + 3 bits a row, four tag weights a column and 3·S coin lanes a tile and column."""
        return self.m * (self.S + 3) + 4 * self.n + 3 * self.S * self.tiles * self.n

    @property
    def macs(self) -> int:
        """Its MACs a token, the share of the egress cap it brings its forward: rows · n · K (`Layout.callBits`)."""
        return self.rows * self.n * self.K

    @property
    def release_bytes(self) -> int:
        """The release's length: m·H row words, n G words, and S·n lane-coin words and 2n column sources a tile."""
        return 4 * (self.m * self.H + self.n + self.tiles * (self.S + 2) * self.n)


def col_code(j: int, q: int) -> int:
    """Digit q of column j's code C[:, j], in [−6, 6]: the balanced base-13 digits of j + 1 (`Pouw.Lead.colCode`)."""
    return (j + 14281) // 13**q % 13 - 6


def bias(S: int, j: int) -> int:
    """Column j's bias β_j at S steps, ⌊(Σ_q C[q, j] + 3(S − 1) + 1)/2⌋ (`Pouw.Lead.bias`): public, from the column and
    the call's steps alone. The call's committed words start from −β_j, which its column sources give back."""
    return (sum(col_code(j, q) for q in range(4)) + 3 * (S - 1) + 1) // 2


def _row_mask(shape: CallShape) -> int:
    """The row words' bits that hold coins, as one little-endian integer: each row's first S + 3."""
    return int.from_bytes(((1 << (shape.S + 3)) - 1).to_bytes(4 * shape.H, "little") * shape.m, "little")


@dataclass(frozen=True, slots=True)
class CallCoins:
    """A call's coins as the certifier flips them, in the release's words: `rows`, the m·H row words; `g`, the n G words;
    `u`, the coin lanes' u in [0, 128), `u[((k·S + t)·n + j)·3 + ℓ]` at tile k, step t, column j and lane ℓ, the
    lane-coin words' order."""

    shape: CallShape
    rows: bytes
    g: bytes
    u: bytes

    def __post_init__(self) -> None:
        sh = self.shape
        for name, size, ok in (("rows", 4 * sh.m * sh.H, lambda v: int.from_bytes(v, "little") & ~_row_mask(sh) == 0),
                               ("g", 4 * sh.n, lambda v: set(v) <= _G_BYTES),
                               ("u", 3 * sh.S * sh.tiles * sh.n, bytes.isascii)):
            v = getattr(self, name)
            if type(v) is not bytes or len(v) != size or not ok(v):
                raise ValueError(f"{name} isn't {size} bytes of the call's {name} at {sh}")

    def a1(self, i: int, q: int) -> int:
        """Step-1 bit a₁[i, q]: bit q of row i's first word."""
        return self.rows[4 * self.shape.H * i] >> q & 1

    def b(self, i: int, t: int) -> int:
        """Tag bit b_t[i] at a step t ≥ 1: bit t + 3 of row i's words."""
        if not 1 <= t < self.shape.S:
            raise IndexError(f"tag bits are at steps 1 to {self.shape.S - 1}, not {t}")
        return self.rows[4 * self.shape.H * i + (t + 3) // 8] >> (t + 3) % 8 & 1

    def G(self, q: int, j: int) -> int:
        """Tag weight G[q, j] = g − 3 (`Coins.G`)."""
        return (self.g[4 * j + q] + 128) % 256 - 128

    def s(self, t: int, k: int, lane: int, j: int) -> int:
        """Coin lane s_t[k, ℓ, j] = 2u − 127 (`Coins.s`)."""
        sh = self.shape
        return 2 * self.u[((k * sh.S + t) * sh.n + j) * 3 + lane] - (2**LANE_BITS - 1)


def _read(entropy: Entropy, k: int) -> bytes:
    out = bytes(entropy(k))
    if len(out) != k:
        raise ValueError(f"the entropy gave {len(out)} bytes, not {k}")
    return out


def _weights(entropy: Entropy, n: int) -> bytes:
    """The G words for n columns: column j's from the j-th accepted 32-bit word (little endian), the words at or above
    `G_BOUND` rejected, byte q the int8 of the residue mod 7⁴'s base-7 digit q, less 3."""
    words: list[int] = []
    while len(words) < n:
        raw = _read(entropy, 4 * (n - len(words)))
        words += [w % 7**4 for w in struct.unpack(f"<{len(raw) // 4}I", raw) if w < G_BOUND]
    return bytes((w // 7**q % 7 - 3) % 256 for w in words for q in range(4))


def flip(shape: CallShape, entropy: Entropy = os.urandom) -> CallCoins:
    """The call's coins from `entropy`, read in the release's order: the row words' bytes, each row's bits from S + 3 up
    cleared; G's 32-bit words; the coin lanes' bytes, each one's low 7 bits a u."""
    raw = _read(entropy, 4 * shape.m * shape.H)
    rows = (int.from_bytes(raw, "little") & _row_mask(shape)).to_bytes(len(raw), "little")
    g = _weights(entropy, shape.n)
    u = _read(entropy, 3 * shape.S * shape.tiles * shape.n).translate(_LOW)
    return CallCoins(shape, rows, g, u)


def _sources(u: bytes, S: int, betas: Sequence[int]) -> bytes:
    """A tile's column sources from its coin lanes' u: class c's word for column j is β_j − σ mod 2³², with
    σ = Σ_t Σ_ℓ ROT_W[c][ℓ] · s_t[ℓ, j] and β_j = `betas[j]`. The steps' u are added in 64-bit fields of one integer,
    which no admitted call's S values below 2⁷ carry out of."""
    n = len(betas)
    wide, total = bytearray(24 * n), 0
    for t in range(S):
        wide[0::8] = u[3 * n * t:3 * n * (t + 1)]
        total += int.from_bytes(wide, "little")
    U = struct.unpack(f"<{3 * n}Q", total.to_bytes(24 * n, "little"))
    out = []
    for w0, w1, w2 in ROT_W:
        lanes = 127 * S * (w0 + w1 + w2)
        out.append(struct.pack(f"<{n}I", *((beta + lanes - 2 * (w0 * a + w1 * b + w2 * c)) % 2**32
                                           for beta, a, b, c in zip(betas, U[0::3], U[1::3], U[2::3]))))
    return b"".join(out)


def release(coins: CallCoins) -> bytes:
    """The call's release: its row words and G words, then a coin tile at a time the tile's lane-coin words (word t·n + j
    of the tile: bytes 0–2 its lanes s_t[ℓ, k, j] as int8, byte 3 C[3, j] at step 0 and 1 later) and its column sources
    (word c·n + j for row class c, β_j − σ)."""
    sh = coins.shape
    S, n = sh.S, sh.n
    top = bytes(col_code(j, 3) % 256 for j in range(n)) + b"\x01" * ((S - 1) * n)
    betas = [bias(S, j) for j in range(n)]
    out: list[bytes | bytearray] = [coins.rows, coins.g]
    span = 3 * S * n
    for k in range(sh.tiles):
        u = coins.u[k * span:(k + 1) * span]
        s = u.translate(_S8)
        lanes = bytearray(4 * S * n)
        lanes[0::4], lanes[1::4], lanes[2::4], lanes[3::4] = s[0::3], s[1::3], s[2::3], top
        out += [lanes, _sources(u, S, betas)]
    return b"".join(out)


@dataclass(frozen=True, slots=True)
class EgressCap:
    """The egress cap the window's workload registers: `cap` bits a token, apportioned among a forward's calls by their
    MACs out of `fwd_macs`, the forward's MACs a token (the lead's `cap` and `fwdMacs`)."""

    cap: int
    fwd_macs: int

    def __post_init__(self) -> None:
        if type(self.cap) is not int or self.cap < 0 or type(self.fwd_macs) is not int or self.fwd_macs < 1:
            raise ValueError(f"an egress cap is bits ≥ 0 over MACs ≥ 1, not {self.cap!r} over {self.fwd_macs!r}")

    def passed(self, macs: int, count: int) -> bool:
        """Whether `count` bits pass the bits of calls of `macs` MACs (`CallShape.macs` summed), cap · macs / fwdMacs
        (the sum of their `Layout.callBits`), in integers."""
        return count * self.fwd_macs > self.cap * macs


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


def _id(v: object, what: str) -> int:
    if type(v) is not int or v < 0:
        raise CallRefusal(f"a {what} id is an integer ≥ 0, not {v!r}")
    return v


@dataclass(slots=True)
class _Call:
    shape: CallShape
    a_root: bytes
    release: bytes = b""


@dataclass(slots=True)
class _Forward:
    calls: dict[int, _Call] = field(default_factory=dict)
    count: int = 0
    words_root: bytes | None = None
    verdict: ForwardVerdict | None = None


class Calls:
    """One window's forwards on one node, as the network certifier keeps them: each forward's calls registered by the
    prover's ids, each call released its coins once, and the open forward counted. `release_bits` is what the node's
    release at a forward's end costs, counted once a forward."""

    def __init__(self, cap: EgressCap, entropy: Entropy = os.urandom, release_bits: int = 0) -> None:
        if type(release_bits) is not int or release_bits < 0:
            raise ValueError(f"a release's bits are an integer ≥ 0, not {release_bits!r}")
        self.cap, self.release_bits = cap, release_bits
        self._entropy = entropy
        self._forwards: dict[int, _Forward] = {}
        self._open: int | None = None
        self._closed = False

    def _live(self) -> None:
        if self._closed:
            raise CallRefusal("the window is closed")

    @property
    def open_forward(self) -> int | None:
        """The open forward, or None."""
        return self._open

    def a_root(self, forward: int, call: int, root: bytes, shape: CallShape) -> bytes:
        """Register call `call` of forward `forward`, its A root and shape, and release its coins (`release`), once, as
        the only reply to this registration. A registration it refuses flips no coin. A call whose coins aren't flipped
        refuses its forward (`NO_COINS`), which stays open to its words root, and so is every later forward of the
        window: once the entropy fails, the certifier flips nothing more. It is `open`, the flip from the entropy `open`
        returns, and `released` (or `unflipped`), so a caller can flip outside its lock."""
        entropy = self.open(forward, call, root, shape)
        try:
            out = release(flip(shape, entropy))
        except Exception:
            self.unflipped(forward, call)
            raise
        self.released(forward, call, out)
        return out

    def open(self, forward: int, call: int, root: bytes, shape: CallShape) -> Entropy:
        """Register call `call` of forward `forward`, its A root and shape, ahead of its coins: the forward's first call
        opens it, while no other forward is open. Returns the entropy the call's coins are flipped from."""
        self._live()
        _id(forward, "forward")
        _id(call, "call")
        if not isinstance(shape, CallShape):
            raise CallRefusal(f"a call's shape is a CallShape, not {type(shape).__name__}")
        root = _root(root, "an A root")
        if self._open is not None and self._open != forward:
            raise CallRefusal(f"forward {self._open} is open: the forwards are serial, and forward {forward}'s calls "
                              "follow its words root")
        f = self._forwards.get(forward)
        if f is None:
            f = self._forwards[forward] = _Forward()
            self._open = forward
        elif self._open != forward:
            raise CallRefusal(f"forward {forward} is closed")
        if call in f.calls:
            raise CallRefusal(f"call {call} of forward {forward} is registered already, and its coins went out once")
        f.calls[call] = _Call(shape, root)
        return self._entropy

    def released(self, forward: int, call: int, out: bytes) -> None:
        """Record the call's release, `out`, which goes out only if this returns: not once the window has closed."""
        self._live()
        self._forwards[forward].calls[call].release = tagged_sha512(RELEASE_TAG, out)

    def unflipped(self, forward: int, call: int) -> None:
        """The call's coins weren't flipped: its forward is refused (`NO_COINS`) unless it was already, and the window
        flips nothing more."""
        self._entropy = _spent
        f = self._forwards[forward]
        if f.verdict is None:
            f.verdict = ForwardVerdict.NO_COINS

    def bucket(self, allocation: int, frames: Sequence[int] = ()) -> None:
        """Count one boundary link's bucket against the open forward, if one is: ⌈log2(r + 1)⌉ timing bits at allocation
        r, and `frames`, each real frame's content bits (at most r of them)."""
        self._live()
        bits = timing_bits(allocation)
        if len(frames) > allocation or any(type(f) is not int or f < 0 for f in frames):
            raise CallRefusal(f"a bucket at allocation {allocation} holds at most {allocation} real frames, each its "
                              f"content bits ≥ 0, not {list(frames)!r}")
        if self._open is not None:
            self._forwards[self._open].count += bits + sum(frames)

    def words_root(self, forward: int, root: bytes) -> ForwardVerdict:
        """Register forward `forward`'s words root and close it, its release's bits counted. Returns its verdict:
        refused if its count passes its calls' bits (`EgressCap.passed`), else accepted, unless refused already."""
        self._live()
        if forward != self._open:
            raise CallRefusal(f"forward {forward} is not open: its words root follows its calls' A roots, once")
        f = self._forwards[forward]
        f.words_root = _root(root, "a words root")
        f.count += self.release_bits
        self._open = None
        if f.verdict is None:
            past = self.cap.passed(sum(c.shape.macs for c in f.calls.values()), f.count)
            f.verdict = ForwardVerdict.PAST_BITS if past else ForwardVerdict.ACCEPTED
        return f.verdict

    def close(self) -> dict[int, ForwardVerdict]:
        """Close the window: a forward still open is refused (`UNCLOSED`) unless it was already. Returns every forward's
        verdict, in id order."""
        self._live()
        if self._open is not None:
            f = self._forwards[self._open]
            if f.verdict is None:
                f.verdict = ForwardVerdict.UNCLOSED
        self._open = None
        self._closed = True
        return {k: f.verdict for k, f in sorted(self._forwards.items())}

    def summary(self) -> list[dict]:
        """The record's forwards, in id order: each one's calls (in id order, each its shape, A root and release's
        digest), its words root and its verdict (None while it is open). The counts are witness and stay out."""
        return [{"forward": k,
                 "calls": [{"call": j, "rows": c.shape.rows, "cols": c.shape.cols, "K": c.shape.K, "bn": c.shape.bn,
                            "a_root": c.a_root.hex(), "release": c.release.hex() or None}
                           for j, c in sorted(f.calls.items())],
                 "words_root": f.words_root.hex() if f.words_root else None,
                 "verdict": f.verdict.value if f.verdict else None}
                for k, f in sorted(self._forwards.items())]
