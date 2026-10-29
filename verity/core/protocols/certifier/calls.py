"""The lead's calls at the network certifier: it flips each call's coins once it holds the call's A root, releases them as
the coin feed the lead's kernel reads, and counts and refuses the call's egress.

A **call** is one matmul of the lead (`Pouw.Lead.lead`). The prover registers its A root with its shape, `rows` rows of
A, `cols` output columns and inner dimension `K`, and later the root of its committed words; between the two the call is
**open**.

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
    column, 0x4B400000 − σ with σ the class's weights (`Pouw.Lead.rotWeight`) against the tile's coin lanes summed over
    its steps. They are no coins but a public function of them, which the kernel's epilogue folds out.
* **The count.** Every bucket a boundary link ticks while a call is open counts against it: ⌈log2(r + 1)⌉ timing bits at
  allocation r, and each real frame's content bits. A bucket inside several calls' spans counts against each. The
  control channel, which carries the registrations in and the releases out, ends at the certifier and is no boundary
  link. A call whose count passes its bits, cap · rows · n · K / fwdMacs (`Pouw.Lead.Layout.callBits`), is refused at
  once, compared in integers, and so is a call still open when the window closes. A refusal is the call's verdict in the
  record, never a dropped frame.

The record (`Calls.summary`) carries each call's shape, roots, release's digest and verdict; the counts are witness.

**On the node** (`NodeCalls`), one window's calls are PoUW window p's, from p's first tick on any of the node's links,
and every link's proxy counts each of its buckets against them as it ticks (`active.Proxy(calls=…)`), so a call's count
is over all of the node's boundary links. The prover registers on the control channel (`serve_calls`, `CallsClient`):
an A root is answered with the call's release, and a words root with its verdict.
"""

from __future__ import annotations

import asyncio
import enum
import json
import os
import socket
import struct
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass

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
#: A column source's offset, 1.5 · 2²³ as an fp32's bits.
SOURCE_BIAS = 0x4B400000
#: The tag of the release's digest in the record.
RELEASE_TAG = "verity-network-certifier/call-release/v1"

#: The certifier's entropy: `entropy(k)` is the next k bytes.
Entropy = Callable[[int], bytes]

_LOW = bytes(x % 2**LANE_BITS for x in range(256))
_S8 = bytes((2 * u - (2**LANE_BITS - 1)) % 256 for u in range(2**LANE_BITS)) + bytes(256 - 2**LANE_BITS)
_G_BYTES = frozenset((g - 3) % 256 for g in range(7))


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
    """A call's registered shape: `rows` rows of A, `cols` output columns, inner dimension `K`, one the verifier admits
    (`ADMITTED_TILES`)."""

    rows: int
    cols: int
    K: int

    def __post_init__(self) -> None:
        for name in ("rows", "cols", "K"):
            v = getattr(self, name)
            if type(v) is not int or v < 1:
                raise CallRefusal(f"a call's {name} is a positive integer, not {v!r}")
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
    def release_bytes(self) -> int:
        """The release's length: m·H row words, n G words, and S·n lane-coin words and 2n column sources a tile."""
        return 4 * (self.m * self.H + self.n + self.tiles * (self.S + 2) * self.n)


def col_code(j: int, q: int) -> int:
    """Digit q of column j's code C[:, j], in [−6, 6]: the balanced base-13 digits of j + 1 (`Pouw.Lead.colCode`)."""
    return (j + 14281) // 13**q % 13 - 6


def _row_mask(shape: CallShape) -> int:
    """The row words' bits that hold coins, as one little-endian integer: each row's first S + 3."""
    return int.from_bytes(((1 << (shape.S + 3)) - 1).to_bytes(4 * shape.H, "little") * shape.m, "little")


@dataclass(frozen=True, slots=True)
class Coins:
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


def flip(shape: CallShape, entropy: Entropy = os.urandom) -> Coins:
    """The call's coins from `entropy`, read in the release's order: the row words' bytes, each row's bits from S + 3 up
    cleared; G's 32-bit words; the coin lanes' bytes, each one's low 7 bits a u."""
    raw = _read(entropy, 4 * shape.m * shape.H)
    rows = (int.from_bytes(raw, "little") & _row_mask(shape)).to_bytes(len(raw), "little")
    g = _weights(entropy, shape.n)
    u = _read(entropy, 3 * shape.S * shape.tiles * shape.n).translate(_LOW)
    return Coins(shape, rows, g, u)


def _sources(u: bytes, S: int, n: int) -> bytes:
    """A tile's column sources from its coin lanes' u: class c's word for column j is 0x4B400000 − σ mod 2³², with
    σ = Σ_t Σ_ℓ ROT_W[c][ℓ] · s_t[ℓ, j]. The steps' u are added in 64-bit fields of one integer, which no admitted
    call's S values below 2⁷ carry out of."""
    wide, total = bytearray(24 * n), 0
    for t in range(S):
        wide[0::8] = u[3 * n * t:3 * n * (t + 1)]
        total += int.from_bytes(wide, "little")
    U = struct.unpack(f"<{3 * n}Q", total.to_bytes(24 * n, "little"))
    out = []
    for w0, w1, w2 in ROT_W:
        base = SOURCE_BIAS + 127 * S * (w0 + w1 + w2)
        out.append(struct.pack(f"<{n}I", *((base - 2 * (w0 * a + w1 * b + w2 * c)) % 2**32
                                           for a, b, c in zip(U[0::3], U[1::3], U[2::3]))))
    return b"".join(out)


def release(coins: Coins) -> bytes:
    """The call's release: its row words and G words, then a coin tile at a time the tile's lane-coin words (word t·n + j
    of the tile: bytes 0–2 its lanes s_t[ℓ, k, j] as int8, byte 3 C[3, j] at step 0 and 1 later) and its column sources
    (word c·n + j for row class c)."""
    sh = coins.shape
    S, n = sh.S, sh.n
    top = bytes(col_code(j, 3) % 256 for j in range(n)) + b"\x01" * ((S - 1) * n)
    out: list[bytes | bytearray] = [coins.rows, coins.g]
    span = 3 * S * n
    for k in range(sh.tiles):
        u = coins.u[k * span:(k + 1) * span]
        s = u.translate(_S8)
        lanes = bytearray(4 * S * n)
        lanes[0::4], lanes[1::4], lanes[2::4], lanes[3::4] = s[0::3], s[1::3], s[2::3], top
        out += [lanes, _sources(u, S, n)]
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
    release: bytes = b""
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
        """Register call `call`'s A root and shape, open its count, and release its coins (`release`), once, as the only
        reply to this registration. A registration it refuses flips no coin. A call whose coins aren't flipped is
        refused (`NO_COINS`) and its count closes, and so is every later call of the window: once the entropy fails, the
        certifier flips nothing more. It is `open`, the flip from the entropy `open` returns, and `released` (or
        `unflipped`), which `NodeCalls` runs with only the first and last under its lock."""
        entropy = self.open(call, root, shape)
        try:
            out = release(flip(shape, entropy))
        except Exception:
            self.unflipped(call)
            raise
        self.released(call, out)
        return out

    def open(self, call: int, root: bytes, shape: CallShape) -> Entropy:
        """Register call `call`'s A root and shape and open its count, ahead of its coins. Returns the entropy its
        coins are flipped from."""
        self._live()
        if type(call) is not int or call < 0:
            raise CallRefusal(f"a call id is an integer ≥ 0, not {call!r}")
        if call in self._calls:
            raise CallRefusal(f"call {call}'s A root is registered already, and its coins went out once")
        if not isinstance(shape, CallShape):
            raise CallRefusal(f"a call's shape is a CallShape, not {type(shape).__name__}")
        self._calls[call] = _Call(shape, _root(root, "an A root"), None)
        self._open.add(call)
        return self._entropy

    def released(self, call: int, out: bytes) -> None:
        """Record call `call`'s release, `out`, which goes out only if this returns: not once the window has closed."""
        self._live()
        self._calls[call].release = tagged_sha512(RELEASE_TAG, out)

    def unflipped(self, call: int) -> None:
        """Call `call`'s coins weren't flipped: it is refused (`NO_COINS`) unless it was already, its count closes, and
        the window flips nothing more."""
        self._entropy = _spent
        c = self._calls[call]
        self._open.discard(call)
        if c.verdict is None:
            c.verdict = Verdict.NO_COINS

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
            raise CallRefusal(f"call {call} is not open: its words root follows its A root's release, once")
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
        """The record's calls, in id order: each one's shape, roots, release's digest and verdict (None while it is
        open). The counts are witness and stay out."""
        return [{"call": call, "rows": c.shape.rows, "cols": c.shape.cols, "K": c.shape.K, "a_root": c.a_root.hex(),
                 "release": c.release.hex() or None, "words_root": c.words_root.hex() if c.words_root else None,
                 "verdict": c.verdict.value if c.verdict else None}
                for call, c in sorted(self._calls.items())]


class NodeCalls:
    """A node's calls across its PoUW windows and its boundary links, as the network certifier keeps them. PoUW window
    p's calls (`Calls`) are the node's from p's first tick on any of the node's links (`tick`), which closes the window
    before it and gives that window's record to `publish(window, Calls.summary())` (by default kept in `records`).
    Every link's every bucket counts against the calls open when its tick runs, whatever window the tick is in: that
    is when its frames leave or enter the node. The control channel (`serve_calls`) and the links' ticks may run on
    other threads, and a lock orders them. A call's coins are flipped outside the lock, so no tick waits for a flip;
    the call is open from its A root on, so the buckets ticked meanwhile count against it. `publish` runs under the
    lock and doesn't call back."""

    def __init__(self, cap: EgressCap, entropy: Entropy = os.urandom,
                 publish: Callable[[int, list[dict]], None] | None = None) -> None:
        self.cap, self._entropy = cap, entropy
        self.records: dict[int, list[dict]] = {}
        self.publish = publish or self.records.__setitem__
        self.window: int | None = None
        self._calls: Calls | None = None
        self._closed = False
        self._lock = threading.Lock()

    def tick(self, window: int, allocation: int, frames: Sequence[int] = ()) -> list[int]:
        """One link's bucket at its tick in PoUW window `window`, counted against the open calls (`Calls.bucket`):
        from `window`'s first tick on, its calls are the node's. Returns the calls it refuses."""
        with self._lock:
            if self._closed:
                raise CallRefusal("the node's calls are closed")
            if type(window) is not int or window < 0:
                raise CallRefusal(f"a PoUW window is an integer ≥ 0, not {window!r}")
            if self.window is None or window > self.window:
                self._close_window()
                self.window, self._calls = window, Calls(self.cap, self._entropy)
            return self._calls.bucket(allocation, frames)

    def a_root(self, call: int, root: bytes, shape: CallShape) -> bytes:
        """`Calls.a_root` in the node's window, its coins flipped outside the lock: they go out only if the window is
        still open once they are flipped."""
        with self._lock:
            calls = self._live()
            entropy = calls.open(call, root, shape)
        try:
            out = release(flip(shape, entropy))
        except Exception:
            with self._lock:
                calls.unflipped(call)
            raise
        with self._lock:
            calls.released(call, out)
        return out

    def words_root(self, call: int, root: bytes) -> Verdict:
        """`Calls.words_root` in the node's window."""
        with self._lock:
            return self._live().words_root(call, root)

    def close(self) -> None:
        """Close the node's last window, its open calls refused, and publish its record. Nothing is taken after."""
        with self._lock:
            if not self._closed:
                self._close_window()
                self._closed = True

    def _live(self) -> Calls:
        if self._closed:
            raise CallRefusal("the node's calls are closed")
        if self._calls is None:
            raise CallRefusal("no PoUW window has begun: a window begins at its first tick")
        return self._calls

    def _close_window(self) -> None:
        if self._calls is not None:
            self._calls.close()
            self.publish(self.window, self._calls.summary())


#: The longest request or answer line on the control channel. A release's bytes follow its answer line.
CALLS_LINE_BYTES = 1024
_REQUEST = "a request is a JSON object of call, a_root (hex), rows, cols and K, or of call and words_root (hex)"


def _line(answer: dict) -> bytes:
    return json.dumps(answer).encode() + b"\n"


def _call_answer(node: NodeCalls, line: bytes) -> tuple[dict, bytes]:
    """`serve_calls`' answer to one request line, and the bytes that follow it."""
    try:
        req = json.loads(line)
        call = req["call"]
        if set(req) == {"call", "a_root", "rows", "cols", "K"}:
            root, shape = bytes.fromhex(req["a_root"]), CallShape(req["rows"], req["cols"], req["K"])
        elif set(req) == {"call", "words_root"}:
            root, shape = bytes.fromhex(req["words_root"]), None
        else:
            raise KeyError("call")
    except CallRefusal as e:
        return {"call": call, "refused": str(e)}, b""
    except (ValueError, KeyError, TypeError):
        return {"refused": _REQUEST}, b""
    try:
        if shape is None:
            return {"call": call, "verdict": node.words_root(call, root).value}, b""
        out = node.a_root(call, root, shape)
    except CallRefusal as e:
        return {"call": call, "refused": str(e)}, b""
    except (OSError, RuntimeError, ValueError) as e:
        return {"call": call, "refused": f"call {call}'s coins weren't flipped: {e}"}, b""
    return {"call": call, "release": len(out)}, out


async def serve_calls(node: NodeCalls, host: str, port: int) -> asyncio.Server:
    """The certifier's control channel for the node's calls, on the loop of its links' proxies (`active.Proxy.run`). It
    answers each JSON line a connection sends, in order, with one JSON line: `{"call", "a_root", "rows", "cols", "K"}`
    (the root in hex) with `{"call", "release": length}` and then the release's `length` bytes, `{"call",
    "words_root"}` with `{"call", "verdict"}`, and what `NodeCalls` refuses, or a line that isn't a request, with
    `{"call", "refused": reason}` (without a call for a line that names none). Each request runs off the loop, so no
    tick waits for a flip. The channel ends at the certifier: none of its bytes reaches a boundary link, and none
    counts. Returns the listening server: `close()` stops it."""

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                try:
                    line = await reader.readline()
                except ValueError:
                    while await reader.read(1 << 16):  # to the sender's end, so the refusal isn't lost to a reset
                        pass
                    writer.write(_line({"refused": f"a request is one line of at most {CALLS_LINE_BYTES} bytes"}))
                    break
                if not line:
                    break
                answer, out = await asyncio.to_thread(_call_answer, node, line)
                writer.write(_line(answer))
                if out:
                    writer.write(out)
                await writer.drain()
        except ConnectionError:
            pass
        finally:
            writer.close()

    return await asyncio.start_server(handle, host, port, limit=CALLS_LINE_BYTES)


class CallsClient:
    """The prover's handle on the certifier's control channel (`serve_calls`) at `host`:`port`, one connection that
    carries its requests in order: `a_root(call, root, shape)` returns the call's release and `words_root(call, root)`
    its verdict. A refusal raises `CallRefusal` with the certifier's reason, and an answer that isn't one `ValueError`.
    `timeout` bounds the socket's waits (None: no bound)."""

    def __init__(self, host: str, port: int, *, timeout: float | None = None) -> None:
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._stream = self._sock.makefile("rwb")

    def a_root(self, call: int, root: bytes, shape: CallShape) -> bytes:
        answer = self._ask({"call": call, "a_root": bytes(root).hex(), "rows": shape.rows, "cols": shape.cols,
                            "K": shape.K})
        if answer.get("release") != shape.release_bytes:
            raise ValueError(f"call {call}'s answer isn't a release of {shape.release_bytes} bytes: {answer}")
        out = self._stream.read(shape.release_bytes)
        if len(out) != shape.release_bytes:
            raise ValueError(f"the control channel ended {len(out)} bytes into call {call}'s release")
        return out

    def words_root(self, call: int, root: bytes) -> Verdict:
        answer = self._ask({"call": call, "words_root": bytes(root).hex()})
        try:
            return Verdict(answer.get("verdict"))
        except ValueError:
            raise ValueError(f"call {call}'s answer isn't a verdict: {answer}") from None

    def _ask(self, body: dict) -> dict:
        self._stream.write(_line(body))
        self._stream.flush()
        line = self._stream.readline(CALLS_LINE_BYTES)
        try:
            answer = json.loads(line)
        except ValueError:
            answer = None
        if isinstance(answer, dict) and isinstance(answer.get("refused"), str):
            raise CallRefusal(answer["refused"])
        if not (isinstance(answer, dict) and answer.get("call") == body["call"]):
            raise ValueError(f"the certifier's answer to call {body['call']!r} isn't one: {line[:200]!r}")
        return answer

    def close(self) -> None:
        self._stream.close()
        self._sock.close()

    def __enter__(self) -> CallsClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
