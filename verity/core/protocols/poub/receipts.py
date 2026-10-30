"""The receipts of one pass, the seed chain that orders passes, the plan of what a pass reads, and the tree that
commits them.

~~~text
Z(x)          x zero-padded to whole 128-byte blocks (one SHA-512 block here)
seed_(t,i)    SHA512(SEED_DOMAIN ‖ σ_t ‖ u32be i ‖ root_(t,i−1)),  SEED_DOMAIN = "verity/poub/seed/v0\0",
              σ_t window t's challenge (its live coin, `coins.py`), root_(t,−1) = root_(t−1,last) in a session and the
              registration's digest for its first window; in a session with appends, the root chained to them
              (`audit.chained`)
plan_(t,i)    R runs (first, count) in read order: count ≥ 1, inside the store, pairwise disjoint, and no run starting
              where the one before it ends (it would be the same run).  Position p is the p-th block of the runs
              concatenated, L_i positions in all.
              digest SHA512(PLAN_DOMAIN ‖ u32be R ‖ (u64be first ‖ u64be count) × R),  PLAN_DOMAIN = "…/plan/v0\0"
spread(z)     z laid over b = 4096 bytes: byte u of z XORed into byte u mod b (a z of b bytes or more folds), a shorter
              z repeated
e_(i,j)       E_j, the store's block j; under a registration that binds z (opt-in), E_j ⊕ spread(z_(i,j))
y_(i,p)       sketch-sha512, the default: SHA512(Z(SKETCH_RECEIPT_DOMAIN ‖ frame(seed_i)) ‖ u64be j ‖ σ_(i,j)),
              j = plan_i[p],  SKETCH_RECEIPT_DOMAIN = "verity/poub/sketch-receipt/v0\0"      64 bytes
              block-sha512, for comparison: SHA512(Z(RECEIPT_DOMAIN ‖ frame(seed_i)) ‖ u64be j ‖ e_(i,j)),
              RECEIPT_DOMAIN = "verity/poub/receipt/v0\0"                                    64 bytes
k_s           sketch-sha512's rows, the registration's: 16 (the default) or 64 (`ROWS`); 0 under block-sha512
S_i           sketch-sha512's k_s × 4096 signs of pass i: K = derive(seed_i, SIGN_DOMAIN, {"tag": "sketch-sha512",
              "k": k_s, "n": 4096}), row c the first 512 bytes of K.stream(("row", c)), bit u (byte u // 8, LSB
              first) 1 → −1, 0 → +1,  SIGN_DOMAIN = "verity/poub/sketch-signs/v0"; over a P2 store,
              k_s × 8224, n = 8224 and 1,028 bytes a row, a unit of four blocks a vector (`p2.py`)
σ_(i,j)       S_i · e_(i,j), its 4096 bytes as int8, over the integers (|σ[c]| ≤ 2^19): k_s int32 little-endian, 64
              bytes at 16 rows (y one compression after the seed's block) and 256 at 64 (three)
leaf c        v_c = y_(16c) ‖ … ‖ y_(16c+15), positions in plan order, the last leaf short; committed as an
              hm96-sha512/row/v2 leaf: x_c = sha512/row/v2(v_c), value hm96-sha512 tree_leaf(key, commit_string(key,
              x_c, salt_c)), salt_c 192 fresh bytes, key hm96-sha512's pinned key
root_(t,i)    frame-v3-sha512 over the leaves: CommitmentDomain(binding = SHA512(TREE_DOMAIN ‖ seed_i ‖ plan digest),
              owner 0, positions 0 … ⌈L_i/16⌉ − 1), schema "hm96-sha512/row/v2",  TREE_DOMAIN = "…/receipt-tree/v0\0"
~~~

`frame` is `verity.ml.randomness.frame`.  A receipt is seed-first: the seed fills the first SHA-512 block, so no state
over a block's bytes exists before the seed, and the seed exists only after the previous pass's root.  A receipt is of
the block alone unless the registration binds z, PoUW's opt-in join: a receipter takes `z` = None, or `z_(i,j)`,
PoUW's noised output of the tile before block j's read in pass i, which then enters elementwise, before any
compression, so a relay that meets the block and z over a link carries `min(|z|, b)` bytes per block, not a digest of
one side.

The registration names its receipt kind (`KINDS`): sketch-sha512 (`DEFAULT`, the stated receipt), or block-sha512, kept
for comparison.  A sketch-sha512 receipt hashes 72 bytes after the seed's block at 16 rows, not 4,104: one SHA-512
compression where block-sha512 takes 33 (three at 64 rows), and the signs, derived from the pass's seed, exist only
after the previous pass's root, as the seed does.  Under sketch-sha512 the registration also names k_s
(`sketch_rows`), which the signs' key binds.  Leaves, salts and the tree are the same under either kind.  What each
kind rests on is `PROTOCOL.md` §4.

The registration also names its block layout (`LAYOUTS`): how a kernel's sketch vectors map to the registered blocks,
each block j the encoded store's bytes 4096·j … 4096·j + 4095, whole codec words.  Under store-order/v0, the only one,
a vector is one registered block's bytes in store order.  `plan_of` takes a kernel's map from its vectors' bytes to
the store's and returns the plan its vectors are, refusing padding, a codec word split across vectors and a block
twice, so that only registered blocks' bytes are credited.  The fused GEMM's fragment cut is refused until the byte
permutation ra-codec-tensor proposes for it is a layout of its own here.

A pass's plan is public: the prover sends it with the root, and the root binds its digest.  Its runs are disjoint so
that one read of a block serves at most one receipt of the pass; a plan that repeats a block is refused.  Its order is
the tree's, not necessarily the reads': listed in store order, a plan is one run per contiguous range it reads.
`Plan.from_array` takes the runs as an (R, 2) array, as a store holds them, and checks and encodes them column by
column through the buffer protocol, in the standard library (a protocol imports no third-party package, and a verifier
no `verity.ml` code); the suite holds it to `Plan` on the same rows.
`Plan.changed` is the next pass's plan from its change, the blocks it adds and removes, in store order.
"""

from __future__ import annotations

import bisect
import hashlib
import itertools
import math
import operator
import struct
import sys
from array import array
from collections.abc import Iterator, Sequence

from verity.ml.commitments import Commitment, CommitmentDomain, MerkleTree, Opening, RangeIndexedDomain, VerificationLimits
from verity.ml.commitments import hm96, verify_opening
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2, sha512_row_v2_digest
from verity.ml.randomness import derive, frame

from .codec import BLOCK_BYTES

SEED_DOMAIN = b"verity/poub/seed/v0\0"
PLAN_DOMAIN = b"verity/poub/plan/v0\0"
RECEIPT_DOMAIN = b"verity/poub/receipt/v0\0"
SKETCH_RECEIPT_DOMAIN = b"verity/poub/sketch-receipt/v0\0"
SIGN_DOMAIN = "verity/poub/sketch-signs/v0"
TREE_DOMAIN = b"verity/poub/receipt-tree/v0\0"
BLOCK_SHA512 = "block-sha512"
SKETCH_SHA512 = "sketch-sha512"
SKETCH_ROWS = 16
ROWS = (16, 64)
SCHEMA = SCHEMA_HM96_SHA512_ROW_V2
CHALLENGE_BYTES = 32
DIGEST_BYTES = 64
PER_LEAF = 16
OWNER = 0
HM96 = hm96.SHA512
KEY = HM96.default_key
SALT_BYTES = HM96.salt_bytes
LIMITS = VerificationLimits()


def _digest(value: bytes, name: str) -> bytes:
    if type(value) is not bytes or len(value) != DIGEST_BYTES:
        raise ValueError(f"{name} is {DIGEST_BYTES} bytes")
    return value


def seed(challenge: bytes, i: int, previous: bytes) -> bytes:
    """`seed_(t,i)`: pass `i`'s seed under the window's challenge, after the root before it (`previous`, chained to
    the session's appends by `audit.chained`)."""
    if type(challenge) is not bytes or len(challenge) != CHALLENGE_BYTES:
        raise ValueError(f"a window's challenge is {CHALLENGE_BYTES} bytes")
    if type(i) is not int or not 0 <= i < 1 << 32:
        raise ValueError("a pass index is a u32")
    return hashlib.sha512(SEED_DOMAIN + challenge + i.to_bytes(4, "big") + _digest(previous, "a root")).digest()


def store_blocks(blocks: object) -> int:
    """`N`, a store's blocks: a u64 (the registration's digest encodes it), at least 1."""
    if type(blocks) is not int or not 1 <= blocks < 1 << 64:
        raise ValueError("a store is 1 to 2^64 − 1 blocks")
    return blocks


_RUNS = "a plan is 1 to 2^32 − 1 runs"
_RUN = "a run is (first, count) with count ≥ 1, inside the store"
_CONTINUED = "a run that starts where the one before it ends is part of it"
_DISJOINT = "a plan's runs are disjoint: no block is read twice in a pass"
#: The buffer formats `Plan.from_array` reads in place: one native integer (struct's codes; numpy's int8 … uint64).
_NATIVE_INTS = frozenset("bBhHiIlLqQ")


class Plan:
    """What one pass reads, in read order, in a store of `blocks` blocks: `runs` of `(first, count)` consecutive blocks.
    Construction refuses any plan but a canonical one of distinct blocks (the format above), so equal read sequences
    have equal digests and no block is read twice.  `Plan.from_array` is the same plan, or the same refusal, from an
    (R, 2) integer array of runs, checked column by column."""

    __slots__ = ("blocks", "_encoding", "_firsts", "_ends", "_runs", "_digest")

    def __init__(self, runs: Sequence[tuple[int, int]], blocks: int) -> None:
        store_blocks(blocks)
        runs = tuple(runs) if isinstance(runs, Sequence) else None
        if runs is None or not 1 <= len(runs) < 1 << 32:
            raise ValueError(_RUNS)
        for r in runs:
            if (type(r) is not tuple or len(r) != 2 or any(type(x) is not int for x in r) or r[0] < 0 or r[1] < 1
                    or r[0] + r[1] > blocks):
                raise ValueError(_RUN)
        if any(a[0] + a[1] == b[0] for a, b in zip(runs, runs[1:])):
            raise ValueError(_CONTINUED)
        ordered = sorted(runs)
        if any(a[0] + a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
            raise ValueError(_DISJOINT)
        encoding = len(runs).to_bytes(4, "big") + b"".join(f.to_bytes(8, "big") + c.to_bytes(8, "big") for f, c in runs)
        self._set(blocks, encoding, array("Q", (f for f, _ in runs)), array("Q", itertools.accumulate(c for _, c in runs)),
                  runs)

    def _set(self, blocks: int, encoding: bytes, firsts: array, ends: array, runs: tuple | None) -> None:
        for name, value in (("blocks", blocks), ("_encoding", encoding), ("_firsts", firsts), ("_ends", ends),
                            ("_runs", runs), ("_digest", None)):
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("a Plan is immutable")

    @classmethod
    def from_array(cls, runs, blocks: int) -> Plan:
        """`Plan` of the rows of `runs`, an (R, 2) array, as `(first, count)` tuples: the same plan with the same
        digest, or the same refusal.  An array that exports C-ordered native integers (a numpy array of int8 … uint64)
        is read through the buffer protocol and checked column by column, each rule over every run in the reference's
        order; anything else is `Plan` of its rows (`tolist`) as tuples.

        Disjointness is checked on the firsts and the ends sorted apart, the k-th end at most the (k+1)-th first for
        every k: runs of count ≥ 1 are disjoint exactly then.  (Sorted by first, disjoint runs are sorted by end too.
        Conversely, the k runs ending by the (k+1)-th first start before it, so no two runs share a first and those k
        are the first k by first: the k-th by first ends by the next one's first.)"""
        store_blocks(blocks)
        try:
            view = memoryview(runs)
        except (TypeError, ValueError):
            view = None
        if (view is None or view.ndim != 2 or view.shape[1] != 2 or view.format not in _NATIVE_INTS
                or not view.c_contiguous):
            rows = runs.tolist()
            return cls(tuple(tuple(r) if type(r) is list else r for r in rows) if type(rows) is list else rows, blocks)
        if not 1 <= len(view) < 1 << 32:
            raise ValueError(_RUNS)
        raw = view.cast("B")
        flat = raw.cast(view.format)
        firsts, counts = flat[0::2].tolist(), flat[1::2].tolist()
        ends = list(map(operator.add, firsts, counts))
        if min(firsts) < 0 or min(counts) < 1 or max(ends) > blocks:
            raise ValueError(_RUN)
        if any(map(operator.eq, ends, itertools.islice(firsts, 1, None))):
            raise ValueError(_CONTINUED)
        if any(map(operator.gt, sorted(ends), itertools.islice(sorted(firsts), 1, None))):
            raise ValueError(_DISJOINT)
        words = array("Q")
        if view.itemsize == words.itemsize:
            words.frombytes(raw)  # every word is nonnegative now, so a signed one's bytes are its u64's
        else:
            words.extend(flat.tolist())
        native_firsts = words[0::2]
        if sys.byteorder == "little":
            words.byteswap()
        plan = object.__new__(cls)
        plan._set(blocks, len(firsts).to_bytes(4, "big") + words.tobytes(), native_firsts,
                  array("Q", itertools.accumulate(counts)), None)
        return plan

    @property
    def runs(self) -> tuple[tuple[int, int], ...]:
        if self._runs is None:
            object.__setattr__(self, "_runs", tuple(struct.iter_unpack(">QQ", memoryview(self._encoding)[4:])))
        return self._runs

    @classmethod
    def whole(cls, blocks: int) -> Plan:
        """Every block, in store order."""
        return cls(((0, blocks),), blocks)

    @classmethod
    def of(cls, order: Sequence[int], blocks: int) -> Plan:
        """The plan that reads `order`, block indices in read order (the prover-side registry's form)."""
        runs: list[tuple[int, int]] = []
        for j in order:
            if runs and type(j) is int and runs[-1][0] + runs[-1][1] == j:
                runs[-1] = (runs[-1][0], runs[-1][1] + 1)
            else:
                runs.append((j, 1))
        return cls(tuple(runs), blocks)

    def changed(self, added: Sequence[tuple[int, int]], removed: Sequence[tuple[int, int]], blocks: int) -> Plan:
        """The next pass's plan from its change, the form a plan travels in (PROTOCOL.md §8): this plan's blocks less
        `removed`, each of which it holds, plus `added`, none of which it then holds, in store order, in a store of
        `blocks` blocks, at least this plan's.  `added` and `removed` are runs (first, count) in any order, disjoint,
        or empty.  Refused (ValueError) otherwise, or if no block is left."""
        if type(blocks) is not int or blocks < self.blocks:
            raise ValueError("a plan's store only grows")

        def spans(runs: Sequence[tuple[int, int]], name: str) -> list[tuple[int, int]]:
            if not isinstance(runs, Sequence) or any(
                    type(r) is not tuple or len(r) != 2 or any(type(x) is not int for x in r) or r[0] < 0 or r[1] < 1
                    or r[0] + r[1] > blocks for r in runs):
                raise ValueError(f"{name} is runs (first, count) with count ≥ 1, inside the store")
            out = sorted((f, f + c) for f, c in runs)
            if any(a[1] > b[0] for a, b in zip(out, out[1:])):
                raise ValueError(f"{name} is disjoint runs")
            return out

        kept, gone = [], iter(spans(removed, "a change's removed blocks"))
        r = next(gone, None)
        for a, b in sorted((f, f + c) for f, c in self.runs):
            while r is not None and r[0] < b:
                if r[0] < a or r[1] > b:
                    raise ValueError("a change removes a block its plan does not hold")
                if r[0] > a:
                    kept.append((a, r[0]))
                a, r = r[1], next(gone, None)
            if a < b:
                kept.append((a, b))
        if r is not None:
            raise ValueError("a change removes a block its plan does not hold")
        out: list[tuple[int, int]] = []
        for a, b in sorted(kept + spans(added, "a change's added blocks")):
            if out and a < out[-1][1]:
                raise ValueError("a change adds a block its plan holds")
            if out and a == out[-1][1]:
                out[-1] = (out[-1][0], b)
            else:
                out.append((a, b))
        return Plan(tuple((a, b - a) for a, b in out), blocks)

    @property
    def length(self) -> int:
        """L, the pass's positions."""
        return self._ends[-1]

    def block(self, p: int) -> int:
        """The block at position `p`."""
        if type(p) is not int or not 0 <= p < self.length:
            raise IndexError(f"a plan of {self.length} positions has no position {p!r}")
        r = bisect.bisect_right(self._ends, p)
        return self._firsts[r] + p - (self._ends[r - 1] if r else 0)

    def __iter__(self) -> Iterator[int]:
        for first, count in self.runs:
            yield from range(first, first + count)

    def encode(self) -> bytes:
        return self._encoding

    @property
    def digest(self) -> bytes:
        if self._digest is None:
            object.__setattr__(self, "_digest", hashlib.sha512(PLAN_DOMAIN + self._encoding).digest())
        return self._digest

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Plan):
            return NotImplemented
        return self.blocks == other.blocks and self._encoding == other._encoding

    def __hash__(self) -> int:
        return hash((self.blocks, self._encoding))

    def __repr__(self) -> str:
        return f"Plan(runs={self.runs!r}, blocks={self.blocks!r})"

    def __reduce__(self) -> tuple:
        return type(self), (self.runs, self.blocks)


def prefix(seed: bytes) -> bytes:
    """`Z(RECEIPT_DOMAIN ‖ frame(seed))`, one SHA-512 block."""
    body = RECEIPT_DOMAIN + frame(_digest(seed, "a pass seed"))
    return body + bytes(-len(body) % 128)


def spread(z: bytes, b: int = BLOCK_BYTES) -> bytes:
    """`z` laid over `b` bytes: folded by XOR when it is `b` bytes or longer, repeated when it is shorter."""
    if type(z) is not bytes or not z:
        raise ValueError("a tile's output is nonempty bytes")
    if len(z) < b:
        return (z * -(-b // len(z)))[:b]
    out = int.from_bytes(z[:b], "little")
    for u in range(b, len(z), b):
        out ^= int.from_bytes(z[u:u + b], "little")
    return out.to_bytes(b, "little")


def _xor(a: bytes, b: bytes) -> bytes:
    return (int.from_bytes(a, "little") ^ int.from_bytes(b, "little")).to_bytes(len(a), "little")


def _index(j: int) -> bytes:
    if type(j) is not int or not 0 <= j < 1 << 64:
        raise ValueError("a block index is a u64")
    return j.to_bytes(8, "big")


def _block(block: bytes) -> bytes:
    if type(block) is not bytes or len(block) != BLOCK_BYTES:
        raise ValueError(f"a block is {BLOCK_BYTES} bytes")
    return block


def _e(block: bytes, z: bytes | None) -> bytes:
    """`e_(i,j)`: the block, or the block ⊕ spread(z) when the receipt binds z."""
    return _block(block) if z is None else _xor(_block(block), spread(z))


class Receipter:
    """block-sha512 receipts under one pass's seed, the kind kept for comparison: `receipter(j, block)` is `y_(i,j)`,
    and `receipter(j, block, z)` the receipt that binds z."""

    kind = BLOCK_SHA512

    def __init__(self, seed: bytes) -> None:
        self.head = hashlib.sha512(prefix(seed))

    def __call__(self, j: int, block: bytes, z: bytes | None = None) -> bytes:
        h = self.head.copy()
        h.update(_index(j))
        h.update(_e(block, z))
        return h.digest()


def _width(width: object) -> int:
    if type(width) is not int or width < 8 or width % 8:
        raise ValueError("a sketch's vector is a positive whole number of 8-byte groups")
    return width


def sign_bits(seed: bytes, rows: int = SKETCH_ROWS, width: int = BLOCK_BYTES) -> tuple[bytes, ...]:
    """`S_i`'s sign bits, `rows` = k_s rows of `width`/8 bytes (512 at a registered block's 4,096 bytes, 1,028 at a P2
    unit's 8,224, `p2.UNIT_BYTES`): bit u of row c (byte u // 8, LSB first) is 1 where `S_i[c][u] = −1`.  The key binds
    k_s and the width, so 64 rows are not 16 rows and 48 more, and a unit's signs are not a block's."""
    rows = sketch_rows(SKETCH_SHA512, rows)
    key = derive(_digest(seed, "a pass seed"), SIGN_DOMAIN, {"tag": SKETCH_SHA512, "k": rows, "n": _width(width)})
    return tuple(key.stream(("row", c), width // 8) for c in range(rows))


def sketch_prefix(seed: bytes) -> bytes:
    """`Z(SKETCH_RECEIPT_DOMAIN ‖ frame(seed))`, one SHA-512 block."""
    body = SKETCH_RECEIPT_DOMAIN + frame(_digest(seed, "a pass seed"))
    return body + bytes(-len(body) % 128)


_BYTE_BITS = tuple(bytes((b >> u) & 1 for u in range(8)) for b in range(256))


def _int8_sum(planes: list[int], mask: int) -> int:
    """Σ e[u] over the bytes u whose bit 8u `mask` sets, e a block's bytes as int8 and `planes[k]` the block as an
    integer shifted right by k: bits 0 to 6 count 2^k each, bit 7 −2^7."""
    return sum((p & mask).bit_count() << k for k, p in enumerate(planes[:7])) - ((planes[7] & mask).bit_count() << 7)


class SketchReceipter:
    """sketch-sha512 receipts under one pass's seed at `rows` = k_s, the default kind: `receipter(j, block)` is
    `y_(i,j)`, and `sketch(block)` its `σ_(i,j)`; given `z`, the receipt and sketch that bind it.  The product is in
    pure Python, a row's ±1 dot product as T − 2·(the sum where its sign is −1), each sum over the block's bit planes:
    about 0.25 ms a block at 16 rows.  `width` is the vector's bytes: a registered block's 4,096, or a P2 unit's 8,224
    (`p2.py`, where z binds nothing), whose sketch is the sum of its pieces' (`partial`)."""

    kind = SKETCH_SHA512

    def __init__(self, seed: bytes, rows: int = SKETCH_ROWS, width: int = BLOCK_BYTES) -> None:
        self.rows, self.width = sketch_rows(SKETCH_SHA512, rows), _width(width)
        self._signs = tuple(int.from_bytes(b"".join(_BYTE_BITS[b] for b in row), "little")
                            for row in sign_bits(seed, self.rows, width))
        self._every = int.from_bytes(b"\1" * width, "little")
        self.head = hashlib.sha512(sketch_prefix(seed))

    def _vector(self, block: bytes, z: bytes | None) -> bytes:
        if self.width == BLOCK_BYTES:
            return _e(block, z)
        if z is not None or type(block) is not bytes or len(block) != self.width:
            raise ValueError(f"a {self.width}-byte vector's sketch is of its {self.width} bytes alone")
        return block

    def _product(self, x: int) -> bytes:
        planes = [x >> k for k in range(8)]
        total = _int8_sum(planes, self._every)
        return struct.pack(f"<{self.rows}i", *(total - 2 * _int8_sum(planes, row) for row in self._signs))

    def sketch(self, block: bytes, z: bytes | None = None) -> bytes:
        return self._product(int.from_bytes(self._vector(block, z), "little"))

    def partial(self, piece: bytes, offset: int) -> bytes:
        """The sketch of `piece` laid at byte `offset` of a vector, the rest of it zero: `S_i[c][offset + x]` against
        `int8(piece[x])`, k_s int32 little-endian.  The sketch is linear, so a vector's is its disjoint pieces'
        partials summed row by row."""
        if (type(piece) is not bytes or not piece or type(offset) is not int or offset < 0
                or offset + len(piece) > self.width):
            raise ValueError(f"a piece is nonempty bytes inside the {self.width}-byte vector")
        return self._product(int.from_bytes(piece, "little") << 8 * offset)

    def receipt(self, j: int, sigma: bytes) -> bytes:
        """`y` from vector j's sketch `sigma`, however it was computed: SHA512(prefix ‖ u64be j ‖ σ)."""
        if type(sigma) is not bytes or len(sigma) != 4 * self.rows:
            raise ValueError(f"a sketch is {self.rows} int32s")
        h = self.head.copy()
        h.update(_index(j))
        h.update(sigma)
        return h.digest()

    def __call__(self, j: int, block: bytes, z: bytes | None = None) -> bytes:
        return self.receipt(j, self.sketch(block, z))


RECEIPTERS = {SKETCH_SHA512: SketchReceipter, BLOCK_SHA512: Receipter}
KINDS = tuple(RECEIPTERS)
DEFAULT = SKETCH_SHA512


def receipt_kind(kind: object) -> str:
    """`kind`, if it is one of `KINDS`."""
    if type(kind) is not str or kind not in RECEIPTERS:
        raise ValueError(f"a receipt kind is one of {list(KINDS)}")
    return kind


def sketch_rows(kind: str, rows: object = None) -> int:
    """A registration's k_s under `kind`: sketch-sha512's, one of `ROWS` (`SKETCH_ROWS` if None), and 0 under
    block-sha512, which sketches nothing."""
    if receipt_kind(kind) != SKETCH_SHA512:
        if rows is not None and (type(rows) is not int or rows != 0):
            raise ValueError(f"{kind} sketches nothing: its rows are 0")
        return 0
    if rows is None:
        return SKETCH_ROWS
    if type(rows) is not int or rows not in ROWS:
        raise ValueError(f"sketch-sha512's rows are one of {list(ROWS)}")
    return rows


def receipter(kind: str, seed: bytes, rows: int | None = None) -> Receipter | SketchReceipter:
    """The receipter of `kind` under one pass's seed, at k_s = `rows` under sketch-sha512 (`sketch_rows`)."""
    rows = sketch_rows(kind, rows)
    return SketchReceipter(seed, rows) if rows else Receipter(seed)


STORE_ORDER = "store-order/v0"
LAYOUTS = (STORE_ORDER,)
LAYOUT = STORE_ORDER


def block_layout(name: object) -> str:
    """`name`, if it is one of `LAYOUTS`: how a kernel's sketch vectors map to the registered blocks."""
    if type(name) is not str or name not in LAYOUTS:
        raise ValueError(f"a block layout is one of {list(LAYOUTS)}")
    return name


def plan_of(vectors: object, blocks: int, layout: str = LAYOUT) -> Plan:
    """The plan a kernel's sketch vectors are under `layout`, the registered blocks it may credit, or ValueError.
    `vectors[v][x]` is the encoded store's byte that byte x of vector v holds (block j's are 4096·j …), or −1 where it
    holds none, padding; a numpy array is read through `tolist`.  Under store-order/v0 vector v is one registered
    block's 4,096 bytes in store order, so whole codec words (32 at p1024, 64 at p512): a vector with padding, a byte
    past the store, a codec word split with another vector, or a block another vector already is, is refused."""
    block_layout(layout)
    store_blocks(blocks)
    rows = vectors.tolist() if hasattr(vectors, "tolist") else vectors
    if not isinstance(rows, Sequence) or not rows:
        raise ValueError("a layout maps one or more vectors")
    order = []
    for v, row in enumerate(rows):
        if not isinstance(row, Sequence) or len(row) != BLOCK_BYTES:
            raise ValueError(f"vector {v} is not {BLOCK_BYTES} bytes")
        first = row[0]
        if type(first) is not int or first < 0 or first % BLOCK_BYTES or first >= blocks * BLOCK_BYTES:
            raise ValueError(f"vector {v} does not start at a registered block: its first byte is {first!r}")
        if list(row) != list(range(first, first + BLOCK_BYTES)):
            raise ValueError(f"vector {v} is not block {first // BLOCK_BYTES}'s {BLOCK_BYTES} bytes in store order: it "
                             "holds padding, another block's bytes or a codec word split with another vector")
        order.append(first // BLOCK_BYTES)
    try:
        return Plan.of(order, blocks)
    except ValueError as e:
        raise ValueError(f"the vectors repeat a registered block: {e}") from None


def leaves(positions: int) -> int:
    return math.ceil(positions / PER_LEAF)


def binding(seed: bytes, plan: Plan) -> bytes:
    """The receipt tree's binding: the pass's seed and its plan's digest."""
    return hashlib.sha512(TREE_DOMAIN + _digest(seed, "a pass seed") + plan.digest).digest()


def domain(seed: bytes, plan: Plan) -> CommitmentDomain:
    """The domain of the receipt tree of the pass with seed `seed` and plan `plan`."""
    return CommitmentDomain(binding(seed, plan), OWNER, RangeIndexedDomain(0, leaves(plan.length)), hash="sha512")


def hidden(row: bytes, salt: bytes) -> bytes:
    """`row`'s committed value as an hm96-sha512/row/v2 leaf: the hm96-sha512 leaf over its sha512/row/v2 digest under
    `salt` (a receipt leaf's receipts, or a payload block)."""
    return hidden_digest(sha512_row_v2_digest(row, 8 * len(row)), salt)


def hidden_digest(x: bytes, salt: bytes) -> bytes:
    """`hidden` of a row from its sha512/row/v2 digest `x` alone (a payload block's `d_j`)."""
    return HM96.tree_leaf(KEY, HM96.commit_string(KEY, x, salt))


def tree(seed: bytes, plan: Plan, ys: Sequence[bytes], salts: Sequence[bytes]) -> MerkleTree:
    """The receipt tree of one pass: `ys` in plan order, sixteen to a leaf, leaf c hidden under `salts[c]`."""
    if len(ys) != plan.length or len(salts) != leaves(plan.length):
        raise ValueError(f"the plan has {plan.length} positions in {leaves(plan.length)} leaves, not {len(ys)} in "
                         f"{len(salts)}")
    values = {c: hidden(b"".join(ys[PER_LEAF * c:PER_LEAF * (c + 1)]), salts[c]) for c in range(len(salts))}
    return MerkleTree(domain(seed, plan), values, lambda _c: SCHEMA)


def leaf_receipt(leaf: bytes, p: int, positions: int) -> bytes | None:
    """Position `p`'s receipt inside its leaf's receipts, or None if they are not that leaf's size."""
    c = p // PER_LEAF
    if len(leaf) != DIGEST_BYTES * min(PER_LEAF, positions - PER_LEAF * c):
        return None
    k = p - PER_LEAF * c
    return leaf[DIGEST_BYTES * k:DIGEST_BYTES * (k + 1)]


def verify_leaf(seed: bytes, plan: Plan, root: bytes, p: int, leaf: bytes, salt: bytes, path: tuple[bytes, ...]) -> bool:
    """Whether `leaf`, the receipts of position `p`'s leaf, under `salt` and with `path`, is that leaf of the tree whose
    root is `root`."""
    d = domain(seed, plan)
    return verify_opening(d, Commitment(root, d.count), Opening(p // PER_LEAF, hidden(leaf, salt), path), SCHEMA, LIMITS)
