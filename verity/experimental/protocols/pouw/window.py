"""PoUW's audit window as the proof service sees it: what the window makes public, the grid its tile count is padded to,
the size of the uniform draw, and the window's advice (its per-call entries and the map from a drawn index to a tile),
with the window statement's reference (`note:verity-root/20261006T0545Z-draft-daniel-one-recursive-architecture-v2`,
PoUW; the joint note's L2, L3 and L4).

**Public items** (`PUBLIC`), exactly the draft's list.  The service refuses a registration that makes anything else public;
the window's own root and W's root are salted roots, which the rule allows.

    item       kind         source     what it is
    verdict    output       outcome    every drawn tile proved and opened, or not
    work       output       window     the credited work, as the tile count N padded to a grid level (`bucket`)
    check      structure    parameter  the tile template every unit instantiates, its work per tile, the commitment scheme
    sampling   structure    parameter  ε, δ, the grid and the law subset:K′ they fix with the work level (`k_prime`)
    drawn      coins        outcome    the verifier's own draw, after the receipt
    root_W     commitment   parameter  the weights' salted root

**The grid** (`GRID_BITS` = s): the positive integers with at most s + 1 significant bits, so consecutive points above
2^(s+1) differ by a factor of at most 1 + 2^-s (9/8 at s = 3).  A window with N_real real tiles registers one level ℓ, the
largest point g_ℓ <= N_real: it is credited g_ℓ tiles' work and pads its tiles to N = g_(ℓ+1) > N_real with filler tiles,
at least one.  So N is `pad(N_real + 1)`, the least grid point at or above the real tiles and the one filler a window
always has: the private partition's unit-count bucket n_b (`note:circuits/20261006T0615Z-draft-private-partitions`,
G6) on the same grid, one bucket and not two.  Both outputs are fixed by ℓ, so the window's influenceable public output is one grid level, and over a
declared range [lo, hi] it takes `levels(lo, hi)` = q values: log2 q bits in the draft's leakage bound.  The honest cost is
under 1 − 2^-s of the credit (at most 11% at s = 3) and the factor N / g_ℓ <= 1 + 2^-s in K′.

**The draw** (L3): uniform subset:K′ over the N equal tiles, K′ the smallest K with (1 − ε·W/(N·w_max))^K <= δ, W the
credited work and w_max the most work a tile does (`k_prime`).  A wrong set holding a share ε of W has at least ε·W/w_max
tiles, so a draw of K′ without replacement misses it with probability at most δ.  Filler tiles are credited nothing.

**The advice** (L2): one entry per committed call, then one filler entry, then empty entries up to the public bound P, each
an `Entry` row committed in one window root registered before the draw.  A call's sides are dense: each side tree holds only
the rows its tiles keep (a row the scheme refused is no tile's), in order, so its tiles are the products of A's rows and
B's, row-major, and index u of its range is (u div n_b, u mod n_b); each side row carries its place in the call (I, J), and
the call's record, which the entry binds by SHA-512, lists them (``keep``).  A call's entry also holds the root of its
pass's -h3 tile tree, where each drawn tile's served leaf is read, and the call's place in that pass (tile0, cols, m), so a
unit computes the leaf and its word in it from the rows it reads (`tile_read`).  The filler entry holds zero rows on both
sides and the range [N_real, N).  `tile_at` maps a drawn index to its entry and rows, and `window_problem` is the window
statement (L4) in Python over the entry rows: the rows are entry rows, the ranges tile [0, N) in order, each call's range
is its tiles, the filler covers the rest, every call's -h3 root is the pass's, the calls' pass tiles are ordered and
disjoint, and the registered level is `bucket` of the real tiles; given the records and the side rows' indices it also
holds those to the entries.  It is not proved yet: no Program states it.

**The layout for gates** (`note:proofs/20261006T1940Z-draft-units-to-inner-layout`, §2) is specified in PoUW's PROTOCOL.md,
"The served audit window": the entry row (`Entry`, 336 bytes), the side rows (`side_row`), the dense side trees of 2^D
leaves (`DEPTHS`) padded with frame-v3's pad leaf under the window's side domains (`side_domain`, `side_levels`,
`side_path`), and the tile instances' public inputs (`PUBLIC_PORTS`, `sessions`, `session_cap`, `instance_ts`,
`public_inputs`); `tests/vectors/window.json` pins them.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Any, Mapping, Sequence

from verity.ml.circuits.codec import canonical_json
from verity.ml.commitments import hm96
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2, row_leaf, sha512_row_v2_digest

GRID_BITS = 3

#: (item, kind, source, reason): the window's public items, in the service's `Public` form
PUBLIC = (
    ("verdict", "output", "outcome", "what the credit rests on: every drawn tile was proved and opened, or not"),
    ("work", "output", "window", "the credited work, as the tile count padded to a grid level: the draw ranges over its N "
                                 "tiles and the auditor credits its work; one level, so at most log2 q bits"),
    ("check", "structure", "parameter", "the check's identity: the tile template, its work per tile and the commitment "
                                        "scheme fix the auditor's verifier"),
    ("sampling", "structure", "parameter", "ε, δ and the grid, and the law subset:K′ they fix with the level: the guarantee "
                                           "is stated in them"),
    ("drawn", "coins", "outcome", "the verifier's own draw, after the receipt"),
    ("root_W", "commitment", "parameter", "the weights' salted root: what the tiles' weight rows are committed to, hiding"),
)
#: the items a registration carries; the verdict and the draw come after it
REGISTERED = ("work", "check", "sampling", "root_W")
KINDS = ("call", "filler", "none")


# ---- the grid ---------------------------------------------------------------------------------------------------------------

def _shift(n: int, bits: int) -> int:
    return max(0, n.bit_length() - bits - 1)


def _int(n: Any) -> bool:
    return isinstance(n, int) and not isinstance(n, bool)


def _positive(n: Any) -> bool:
    return _int(n) and n >= 1


def on_grid(n: Any, bits: int = GRID_BITS) -> bool:
    return _positive(n) and n & ((1 << _shift(n, bits)) - 1) == 0


def floor_point(n: int, bits: int = GRID_BITS) -> int:
    """The largest grid point at most n >= 1."""
    if not _positive(n):
        raise ValueError(f"the grid is over positive integers, not {n!r}")
    s = _shift(n, bits)
    return n >> s << s


def pad(n: int, bits: int = GRID_BITS) -> int:
    """The least grid point at least n >= 1: a unit count's public bucket."""
    g = floor_point(n, bits)
    return g if g == n else point(level(g, bits) + 1, bits)


def level(g: int, bits: int = GRID_BITS) -> int:
    """Grid point g's index, from 1 at level 0."""
    if not on_grid(g, bits):
        raise ValueError(f"{g!r} is not a point of the {bits}-bit grid")
    small = 1 << (bits + 1)
    if g < small:
        return g - 1
    e = _shift(g, bits)
    return small - 1 + (e - 1) * (1 << bits) + (g >> e) - (1 << bits)


def point(ell: int, bits: int = GRID_BITS) -> int:
    """The grid point at level ell (`level`'s inverse)."""
    if not isinstance(ell, int) or isinstance(ell, bool) or ell < 0:
        raise ValueError(f"a level is a natural number, not {ell!r}")
    small = 1 << (bits + 1)
    if ell < small - 1:
        return ell + 1
    r = ell - (small - 1)
    return ((1 << bits) + r % (1 << bits)) << (r // (1 << bits) + 1)


def levels(lo: int, hi: int, bits: int = GRID_BITS) -> int:
    """q: the grid levels a window's tile count in [lo, hi] can register."""
    return level(floor_point(hi, bits), bits) - level(floor_point(lo, bits), bits) + 1


@dataclass(frozen=True)
class Work:
    """A window's work level: it is credited `credited` tiles, point(level), and pads to `n` tiles, point(level + 1)."""
    level: int
    credited: int
    n: int

    @classmethod
    def at(cls, ell: int, bits: int = GRID_BITS) -> "Work":
        return cls(ell, point(ell, bits), point(ell + 1, bits))

    def to_json(self, tile_work: int) -> dict[str, int]:
        return {"level": self.level, "n": self.n, "credited_tiles": self.credited, "credited": self.credited * tile_work}


def bucket(real: int, bits: int = GRID_BITS) -> Work:
    """The level of a window of `real` real tiles: credited floor_point(real), padded to the next point."""
    return Work.at(level(floor_point(real, bits), bits), bits)


# ---- the draw -----------------------------------------------------------------------------------------------------------

#: (a/b)^k is compared exactly in integers while k·bitlength(b) is at most this many bits
EXACT_BITS = 1 << 14


def miss_at_most(q: Fraction, k: int, delta: Fraction) -> bool:
    """(q)^k <= δ, exactly, for 0 <= q < 1, k >= 0 and 0 < δ.  In integers while they are small (and always when
    k < bitlength(den δ)); otherwise from bounds lo <= q^k·2^P <= hi, square-and-multiply rounding down and up, at a
    precision P that doubles until they decide.  They do: with q = a/b in lowest terms and b >= 2, q^k = δ needs b^k = den δ,
    so 2^k <= den δ and k < bitlength(den δ)."""
    a, b, dn, dd = q.numerator, q.denominator, delta.numerator, delta.denominator
    if k < dd.bit_length() or k * b.bit_length() <= EXACT_BITS:
        return a ** k * dd <= dn * b ** k
    prec = 64 + 2 * k.bit_length() + dd.bit_length()
    while True:
        lo, hi = _fixed_pow(a, b, k, prec)
        if hi * dd <= dn << prec:
            return True
        if lo * dd > dn << prec:
            return False
        prec *= 2


def _fixed_pow(a: int, b: int, k: int, prec: int) -> tuple[int, int]:
    """(lo, hi) with lo <= (a/b)^k·2^prec <= hi."""
    one = 1 << prec
    lo, hi, base_lo, base_hi = one, one, (a << prec) // b, -(-(a << prec) // b)
    while k:
        if k & 1:
            lo, hi = lo * base_lo >> prec, -(-(hi * base_hi) >> prec)
        k >>= 1
        if k:
            base_lo, base_hi = base_lo * base_lo >> prec, -(-(base_hi * base_hi) >> prec)
    return lo, hi


def k_prime(eps: Fraction, delta: Fraction, credited: int, n: int, w_max: int) -> int:
    """L3's K′: the smallest K >= 1 with (1 − p)^K <= δ in rationals, p = ε·credited/(n·w_max), and n if no K <= n has it
    (`credited` the credited work, `n` the tiles, each doing at most `w_max`); 1 when p = 1.  Exact: the binary64
    logarithms give only a starting K, which exact comparisons (`miss_at_most`) then step to the answer."""
    eps, delta = Fraction(eps), Fraction(delta)
    if not (0 < eps <= 1 and 0 < delta < 1 and _positive(n) and _positive(w_max) and _int(credited)
            and 0 < credited <= n * w_max):
        raise ValueError(f"k_prime needs 0 < ε <= 1, 0 < δ < 1 and 0 < credited <= n·w_max, not {eps}, {delta}, "
                         f"{credited}, {n}, {w_max}")
    p = eps * credited / (n * w_max)
    if p == 1:
        return 1
    step, target = math.log1p(-float(p)), math.log(float(delta))
    guess = target / step if step < 0 else math.inf
    k = n if guess >= n else max(1, math.ceil(guess))
    while k > 1 and miss_at_most(1 - p, k - 1, delta):
        k -= 1
    while k < n and not miss_at_most(1 - p, k, delta):
        k += 1
    return k


def sampling(eps: Fraction, delta: Fraction, work: Work, tile_work: int, bits: int = GRID_BITS) -> dict[str, Any]:
    """The `sampling` item: ε, δ, the grid and the law they fix with the level (every tile does `tile_work`)."""
    k = k_prime(eps, delta, work.credited * tile_work, work.n, tile_work)
    return {"eps": str(Fraction(eps)), "delta": str(Fraction(delta)), "grid_bits": bits, "law": {"law": "subset", "k": k}}


# ---- the entry row ----------------------------------------------------------------------------------------------------------

#: a tile index's bits: the declared range's tile counts are at most 2^40
T_BITS = 40
#: the side trees' public depths: A holds a call's kept activation rows (at most the served batch's 8,192 tokens), B its
#: kept weight rows
DEPTHS = {"a": 13, "b": 17}
#: the window root's port, and the schema its entry rows are committed under
PORT = "entries"
ENTRY_SCHEMA = SCHEMA_HM96_SHA512_ROW_V2
ENTRY_BYTES = 336
ENTRY_WORDS = ENTRY_BYTES // 2
ENTRY_BITS = 8 * ENTRY_BYTES
KIND_CODE = {"none": 0, "call": 1, "filler": 2}
_HEAD = struct.Struct("<6Q")
#: tile0, cols, m and a zero u64, after root_h3
_PASS = struct.Struct("<4Q")
_PASS_AT = 240
_DIGEST = 64
_ZERO = bytes(_DIGEST)
_HM = hm96.SHA512
#: the pass's tiles are 64 x 64 (`audit.tile_refs` over Pearl-C's tiles), and a short last tile row is whole 16-row
#: ticket subtiles (Pearl-C's domain: 16 | m)
TILE = 64
ROW_STEP = 16


@dataclass(frozen=True)
class Entry:
    """One entry of the window root, as its committed bit row holds it (PROTOCOL.md's layout): its kind, its tile range
    [lo, hi), each side's row count and root, its pass's -h3 tile-tree root, a call's record's SHA-512, and where the call
    sits in its pass: its first tile there `tile0`, its tile columns `cols` = ⌈n/64⌉ and its rows `m`.  `row` and
    `from_row` encode any values that fit the row's fields; `layout_problem` holds them to the layout."""

    kind: str
    lo: int = 0
    hi: int = 0
    n_a: int = 0
    n_b: int = 0
    root_a: bytes = _ZERO
    root_b: bytes = _ZERO
    root_h3: bytes = _ZERO
    record: bytes = _ZERO
    tile0: int = 0
    cols: int = 0
    m: int = 0

    def row(self) -> bytes:
        if self.kind not in KIND_CODE:
            raise ValueError(f"an entry's kind is one of {KINDS}, not {self.kind!r}")
        if any(len(d) != _DIGEST for d in (self.root_a, self.root_b, self.root_h3, self.record)):
            raise ValueError("an entry's roots and record are 64-byte digests")
        try:
            head = _HEAD.pack(KIND_CODE[self.kind], self.lo, self.hi, self.n_a, self.n_b, 0)
            at = _PASS.pack(self.tile0, self.cols, self.m, 0)
        except struct.error as e:
            raise ValueError(f"an entry's integers are u64s: {e}") from None
        return head + self.root_a + self.root_b + self.root_h3 + at + self.record

    def words(self) -> list[int]:
        """The row as its 168 u16 words, little-endian."""
        return list(struct.unpack(f"<{ENTRY_WORDS}H", self.row()))

    @classmethod
    def from_row(cls, row: bytes) -> "Entry":
        """The entry a row holds; raises ValueError for a row of another length, an unknown kind or a nonzero pad."""
        row = bytes(row)
        if len(row) != ENTRY_BYTES:
            raise ValueError(f"an entry row is {ENTRY_BYTES} bytes, not {len(row)}")
        code, lo, hi, n_a, n_b, pad = _HEAD.unpack_from(row)
        tile0, cols, m, pad2 = _PASS.unpack_from(row, _PASS_AT)
        kind = {v: k for k, v in KIND_CODE.items()}.get(code)
        if kind is None:
            raise ValueError(f"kind {code} is none of {KINDS} ({sorted(KIND_CODE.values())})")
        if pad:
            raise ValueError("its bytes 40..48 are not zero")
        if pad2:
            raise ValueError("its bytes 264..272 are not zero")
        return cls(kind, lo, hi, n_a, n_b, row[48:112], row[112:176], row[176:240], row[272:336], tile0, cols, m)

    @classmethod
    def from_words(cls, words: Sequence[int]) -> "Entry":
        if len(words) != ENTRY_WORDS or any(not 0 <= int(w) < 1 << 16 for w in words):
            raise ValueError(f"an entry row is {ENTRY_WORDS} u16 words")
        return cls.from_row(struct.pack(f"<{ENTRY_WORDS}H", *map(int, words)))

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "lo": self.lo, "hi": self.hi, "n_a": self.n_a, "n_b": self.n_b, "root_a": self.root_a.hex(),
                "root_b": self.root_b.hex(), "root_h3": self.root_h3.hex(), "tile0": self.tile0, "cols": self.cols,
                "m": self.m, "record": self.record.hex()}

    @classmethod
    def from_json(cls, x: Mapping[str, Any]) -> "Entry":
        return cls(x["kind"], x["lo"], x["hi"], x["n_a"], x["n_b"], bytes.fromhex(x["root_a"]), bytes.fromhex(x["root_b"]),
                   bytes.fromhex(x["root_h3"]), bytes.fromhex(x["record"]), x["tile0"], x["cols"], x["m"])


def record_digest(record: Mapping[str, Any]) -> bytes:
    """A call's record as its entry binds it: the SHA-512 of its canonical JSON (RFC 8785)."""
    return hashlib.sha512(canonical_json(record)).digest()


def pass_tiles(entry: Entry) -> int:
    """A call's tiles in its pass: ⌈m/64⌉ tile rows of `cols` tiles."""
    return -(-entry.m // TILE) * entry.cols


def layout_problem(entry: Entry) -> str | None:
    """Why an entry's values are not the layout's, or None: its range in [0, 2^40); an empty entry the zero row; a call's or
    the filler's sides 1..2^D rows; the filler's -h3 root, record, tile0, cols and m zero; a call's m in whole 16s up to
    2^D_A and at least n_a, its cols 1 to 2^(D_B − 6) with 64·cols at least n_b, and its pass tiles
    [tile0, tile0 + ⌈m/64⌉·cols) within 2^40."""
    if not (entry.lo < 1 << T_BITS and entry.hi < 1 << T_BITS):
        return f"its range [{entry.lo}, {entry.hi}) is not of {T_BITS}-bit indices"
    if entry.kind == "none":
        return None if entry == Entry("none") else "an empty entry is not the zero row"
    for side, n in (("a", entry.n_a), ("b", entry.n_b)):
        if not 1 <= n <= 1 << DEPTHS[side]:
            return f"side {side} holds {n} rows, not 1 to 2^{DEPTHS[side]}"
    if entry.kind == "filler":
        if any(entry.root_h3):
            return "the filler's -h3 root is not zero"
        if any(entry.record):
            return "the filler's record is not zero"
        if entry.tile0 or entry.cols or entry.m:
            return "the filler's tile0, cols and m are not zero"
        return None
    if not (1 <= entry.m <= 1 << DEPTHS["a"] and entry.m % ROW_STEP == 0 and entry.n_a <= entry.m):
        return f"its m {entry.m} is not a multiple of {ROW_STEP} from n_a = {entry.n_a} to 2^{DEPTHS['a']}"
    if not (1 <= entry.cols <= 1 << (DEPTHS["b"] - 6) and entry.n_b <= TILE * entry.cols):
        return f"its cols {entry.cols} is not 1 to 2^{DEPTHS['b'] - 6} with 64·cols >= n_b = {entry.n_b}"
    if entry.tile0 + pass_tiles(entry) > 1 << T_BITS:
        return f"its pass tiles [{entry.tile0}, {entry.tile0 + pass_tiles(entry)}) are not of {T_BITS}-bit indices"
    return None


def record_problem(entry: Entry, record: Mapping[str, Any]) -> str | None:
    """Why a call's record is not its entry's, or None: the entry binds its SHA-512; its ``shape`` [m, k, n] and ``tile0``
    give the entry's m, cols = ⌈n/64⌉ and tile0; and its ``keep``, each side's places in the call of the rows its side tree
    holds, is n_a strictly increasing indices of [0, m) and n_b of [0, n)."""
    try:
        if record_digest(record) != entry.record:
            return "its record is not the one the entry binds"
        m, _k, n = record["shape"]
        tile0, keep = record["tile0"], record["keep"]
        if (entry.tile0, entry.cols, entry.m) != (tile0, -(-n // TILE), m):
            return f"its tile0, cols and m are not its record's {tile0}, ⌈{n}/64⌉ and {m}"
        for side, count, top in (("a", entry.n_a, m), ("b", entry.n_b, n)):
            rows = list(keep[side])
            if len(rows) != count or any(not isinstance(i, int) or isinstance(i, bool) for i in rows) \
                    or rows != sorted(set(rows)) or (rows and not 0 <= rows[0] <= rows[-1] < top):
                return f"its record's side-{side} keep is not {count} increasing indices of [0, {top})"
    except (KeyError, TypeError, ValueError):
        return "its record has no shape [m, k, n], tile0 and keep {a, b}"
    return None


def entry_digest(entry: Entry) -> bytes:
    """x, the entry row's ``sha512/row/v2`` digest."""
    return sha512_row_v2_digest(entry.row(), ENTRY_BITS)


def entry_bc(entry: Entry, salt: bytes) -> bytes:
    """An entry's b ‖ c: hm96-sha512 over its row's digest, under `salt`."""
    return _HM.commit_string(_HM.default_key, entry_digest(entry), salt)


def entry_leaf(dom: CommitmentDomain, e: int, bc: bytes) -> bytes:
    """Entry e's frame-v3-sha512 leaf in the window root, its row an ``hm96-sha512/row/v2`` row."""
    return row_leaf(dom, e, _HM.tree_leaf(_HM.default_key, bc), ENTRY_SCHEMA)


# ---- the side trees -----------------------------------------------------------------------------------------------------

SIDE_DOMAIN = "verity/pouw/window/side-domain/v1"
#: the frame's owner, as C-Flock's statement domains have it
OWNER = -1
#: a side row's index field: the row's place in its call (I < 2^13 for A, J < 2^17 for B) as a u32, two u16 words low
#: first, then six zero words, so the field is one 128-bit word after the row's operand words
SIDE_INDEX_WORDS = 8


def side_row(words: Sequence[int], index: int) -> list[int]:
    """A side tree's row: the operand row's u16 words (fp8-p: k/2 code words, then P's 64), then its index field."""
    if len(words) % SIDE_INDEX_WORDS:
        raise ValueError(f"a side row's operand words are whole 128-bit words, not {len(words)} u16 words")
    if not (isinstance(index, int) and not isinstance(index, bool) and 0 <= index < 1 << 32):
        raise ValueError(f"a side row's index is a u32, not {index!r}")
    return [int(w) for w in words] + [index & 0xFFFF, index >> 16] + [0] * (SIDE_INDEX_WORDS - 2)


def side_index(row: Sequence[int]) -> int:
    """The index a side row holds; raises ValueError unless its index field is a u32 and zeros."""
    field = [int(w) for w in row[-SIDE_INDEX_WORDS:]]
    if len(row) <= SIDE_INDEX_WORDS or any(field[2:]) or any(not 0 <= w < 1 << 16 for w in field[:2]):
        raise ValueError("the row's last 128-bit word is not a u32 index and zeros")
    return field[0] | field[1] << 16


def side_domain(window_domain: bytes, side: str) -> CommitmentDomain:
    """The domain of every side-`side` tree of the window whose root's domain id is `window_domain`: 2^D positions, bound
    to that id, the side and D."""
    if side not in DEPTHS:
        raise ValueError(f"a side is one of {sorted(DEPTHS)}, not {side!r}")
    if len(window_domain) != _DIGEST:
        raise ValueError("the window's domain id is 64 bytes")
    d = DEPTHS[side]
    binding = identity_digest_sha512(SIDE_DOMAIN, {"window": window_domain.hex(), "side": side, "depth": d})
    return CommitmentDomain(binding, OWNER, RangeIndexedDomain(0, 1 << d), hash="sha512")


@lru_cache(maxsize=4)
def _pads(domain_id: bytes) -> dict[tuple[int, int], bytes]:
    return {}


def pad_node(dom: CommitmentDomain, level: int, index: int) -> bytes:
    """The node at (`level`, `index`) of a subtree of `dom` that holds pad leaves only (computed once per domain)."""
    memo = _pads(dom.domain_id)
    key = (level, index)
    if key not in memo:
        memo[key] = dom.pad(index) if level == 0 else dom.node(level - 1, index, pad_node(dom, level - 1, 2 * index),
                                                               pad_node(dom, level - 1, 2 * index + 1))
    return memo[key]


def side_levels(dom: CommitmentDomain, leaves: Sequence[bytes]) -> tuple[tuple[bytes, ...], ...]:
    """A side tree over `leaves` (n >= 1 of `dom`'s count), each later leaf frame-v3's pad: each level's nodes that hold a
    leaf of the n, the root last.  `side_path` reads the rest from `pad_node`."""
    width = dom.count
    if not 1 <= len(leaves) <= width or width & (width - 1):
        raise ValueError(f"a side tree holds 1 to {width} rows, not {len(leaves)}")
    level, levels, d = list(leaves), [], 0
    while True:
        levels.append(tuple(level))
        if len(level) == 1 and 1 << d == width:
            return tuple(levels)
        level = [dom.node(d, y, level[2 * y], level[2 * y + 1] if 2 * y + 1 < len(level) else pad_node(dom, d, 2 * y + 1))
                 for y in range((len(level) + 1) // 2)]
        d += 1


def side_path(dom: CommitmentDomain, levels: Sequence[Sequence[bytes]], index: int) -> list[bytes]:
    """Leaf `index`'s siblings, leaf level first: D of them."""
    out = []
    for d, level in enumerate(levels[:-1]):
        s = index ^ 1
        out.append(level[s] if s < len(level) else pad_node(dom, d, s))
        index >>= 1
    return out


# ---- the advice and the window statement --------------------------------------------------------------------------------

def tiles(entry: Entry) -> int:
    return entry.n_a * entry.n_b if entry.kind != "none" else 0


def tile_in(entry: Entry, t: int) -> tuple[int, int]:
    """(A row, B row) of index t in its entry's range: the side trees' rows' products, row-major."""
    if entry.kind == "none" or not entry.lo <= t < entry.hi or entry.n_b < 1:
        raise ValueError(f"index {t} is not in the entry's range")
    return divmod(t - entry.lo, entry.n_b)


def tile_at(entries: Sequence[Entry], t: int) -> tuple[int, int, int]:
    """(entry, A row, B row) of drawn index t."""
    for e, entry in enumerate(entries):
        if entry.kind != "none" and entry.lo <= t < entry.hi:
            return (e, *tile_in(entry, t))
    raise ValueError(f"index {t} is in no entry's range")


def tile_read(entry: Entry, i: int, j: int) -> tuple[int, int]:
    """(p, at): the -h3 leaf a call's tile reads and its word's place in that leaf, from the entry row and the two side
    rows' indices I and J alone.  p = tile0 + (I div 64)·cols + (J div 64), the pass's tile holding (I, J); at =
    (I mod 64) | (J mod 64) << 6 | (rows/16 − 1) << 12, rows = min(64, m − 64·(I div 64)) that tile's height."""
    if entry.kind != "call" or not (0 <= i < entry.m and 0 <= j < TILE * entry.cols):
        raise ValueError(f"({i}, {j}) is not a word of the call's m x 64·cols")
    rows = min(TILE, entry.m - TILE * (i // TILE))
    return entry.tile0 + (i // TILE) * entry.cols + j // TILE, i % TILE | (j % TILE) << 6 | (rows // ROW_STEP - 1) << 12


def window_problem(rows: Sequence[bytes], work: Work, bound: int, root_h3: bytes, bits: int = GRID_BITS,
                   records: Sequence[Mapping[str, Any] | None] | None = None,
                   indices: Sequence[Mapping[str, Sequence[int]] | None] | None = None) -> str | None:
    """The window statement (L4) on the committed entry rows, or None: `bound` rows, each an entry row of the layout
    (`Entry.from_row`, `layout_problem`); the calls, then one filler, then empty entries; the ranges tile [0, work.n) in
    order; a call's range is its tiles; the filler's holds the rest, within its rows; every call's -h3 root is `root_h3`,
    the pass's published tile commitment; the calls' pass tiles are in order and disjoint; and the level is `bucket` of
    the calls' tiles.  Credited tiles are only the calls'.

    The gates read the entry rows only.  Given the calls' `records` (per entry, None past the calls) it also holds each to
    its entry (`record_problem`), and given each entry's side rows' `indices` (`side_index`, per side) it holds a call's to
    its record's keep and the filler's to zero: so i -> I and j -> J are injective, and no two window tiles read one
    word."""
    if len(root_h3) != _DIGEST:
        raise ValueError("the pass's -h3 tile root is 64 bytes")
    if len(rows) != bound:
        return f"{len(rows)} entries, not the bound {bound}"
    entries = []
    for e, row in enumerate(rows):
        try:
            entry = Entry.from_row(row)
        except ValueError as x:
            return f"entry {e}: not an entry row: {x}"
        why = layout_problem(entry)
        if why:
            return f"entry {e}: not the layout: {why}"
        entries.append(entry)
    kinds = [x.kind for x in entries]
    calls = kinds.index("filler") if "filler" in kinds else -1
    if calls < 1 or kinds[:calls] != ["call"] * calls or kinds[calls + 1:] != ["none"] * (bound - calls - 1):
        return "the entries are the calls, then one filler, then empty entries"
    at = in_pass = 0
    for e, entry in enumerate(entries[:calls + 1]):
        if entry.lo != at or entry.hi <= at:
            return f"entry {e}: its range does not start where the last ended, or is empty"
        size = entry.hi - entry.lo
        if entry.kind == "call" and size != tiles(entry):
            return f"entry {e}: its range is {size} tiles, not its {tiles(entry)} tiles"
        if entry.kind == "call" and entry.root_h3 != root_h3:
            return f"entry {e}: its -h3 root is not the pass's tile commitment"
        if entry.kind == "call" and entry.tile0 < in_pass:
            return f"entry {e}: its pass tiles start at {entry.tile0}, before the last call's end {in_pass}"
        if entry.kind == "filler" and size > tiles(entry):
            return f"entry {e}: the filler's range is {size} tiles, beyond its {tiles(entry)}"
        at = entry.hi
        in_pass = entry.tile0 + pass_tiles(entry) if entry.kind == "call" else in_pass
    real = entries[calls].lo
    if at != work.n:
        return f"the ranges end at {at}, not the padded count {work.n}"
    if bucket(real, bits) != work:
        return f"{real} real tiles are level {bucket(real, bits).level}, not the registered level {work.level}"
    if records is not None:
        if len(records) != bound or any(r is not None for r in records[calls:]):
            return f"{len(records)} records, not one per call and None past the calls"
        for e in range(calls):
            why = record_problem(entries[e], records[e])
            if why:
                return f"entry {e}: {why}"
    if indices is not None:
        if records is None:
            raise ValueError("the side rows' indices are held to the calls' records, so give both")
        for e, entry in enumerate(entries[:calls + 1]):
            got = indices[e] if e < len(indices) else None
            for side, count in (("a", entry.n_a), ("b", entry.n_b)):
                want = list(records[e]["keep"][side]) if entry.kind == "call" else [0] * count
                if got is None or [int(i) for i in got[side]] != want:
                    return f"entry {e}: its side-{side} rows' indices are not " + \
                        ("its record's keep" if entry.kind == "call" else "zero")
    return None


# ---- the tile instances' public inputs ------------------------------------------------------------------------------------

#: C-Flock's format of the verifier's own ``--public-inputs`` file (its PROTOCOL.md §16.10, Lean's ``HmIn.checkOwn``)
PUBLIC_INPUTS_FORMAT = "verity/flock-public-inputs/v1"
#: a tile instance's public input ports, in order, as its circuit's META ``public_inputs`` must name them: each port's u16
#: words (whole 128-bit words, so no row leaf shares one), the public datum it is recomputed from and its rule.  t is a
#: u128 below 2^40; the root and the domain ids are 64 bytes, each u16 word two of them, little-endian.  domain_h3 is the
#: domain of the pass's -h3 tile tree, which binds its tile count and so its depth.
PUBLIC_PORTS = (
    {"port": "t", "words": 8, "seed": "drawn", "derivation": "verity/pouw/window/t/v1"},
    {"port": "root", "words": 32, "seed": "registration", "derivation": "verity/pouw/window/root/v1"},
    {"port": "domain", "words": 32, "seed": "registration", "derivation": "verity/pouw/window/domain/v1"},
    {"port": "domain_a", "words": 32, "seed": "registration", "derivation": "verity/pouw/window/side-domain-a/v1"},
    {"port": "domain_b", "words": 32, "seed": "registration", "derivation": "verity/pouw/window/side-domain-b/v1"},
    {"port": "domain_h3", "words": 32, "seed": "pass", "derivation": "verity/pouw/window/domain-h3/v1"},
)
#: C-Flock's largest session: m = k_log + nbl <= 35, nbl = max(3, ⌈log2 blocks⌉) (its PROTOCOL.md §16.5, `shape_in_range`)
M_MAX = 35


def session_cap(k_log: int, per_block: int = 1) -> int:
    """The most instances one session holds when each block of 2^`k_log` bits holds `per_block` of them (C-Flock's
    m <= `M_MAX`)."""
    if not (_positive(k_log) and k_log <= M_MAX - 3 and _positive(per_block)):
        raise ValueError(f"a block of 2^{k_log!r} bits does not fit a session of m <= {M_MAX}")
    return per_block << (M_MAX - k_log)


def sessions(k: int, cap: int) -> int:
    """S, the fixed-shape sessions of `cap` instances that hold a window's K′ = `k` drawn tiles.  Served windows draw
    K′ <= 31,178 (ε = 0.1%, δ = 2^-40 over levels 87 to 143) and K′ <= 9,937 (ε = 1%, δ = 2^-128)."""
    if not (_positive(k) and _positive(cap)):
        raise ValueError(f"K′ and the cap are positive, not {k!r} and {cap!r}")
    return -(-k // cap)


def _registered_root(registration: Mapping[str, Any]) -> tuple[bytes, bytes]:
    roots = [r for r in registration.get("roots") or () if isinstance(r, Mapping) and r.get("port") == PORT]
    if len(roots) != 1:
        raise ValueError(f"the registration names {len(roots)} roots of port {PORT!r}, not one")
    root, dom = bytes.fromhex(roots[0]["root"]), bytes.fromhex(roots[0]["domain"])
    if len(root) != _DIGEST or len(dom) != _DIGEST:
        raise ValueError("the window root and its domain id are 64 bytes")
    return root, dom


def instance_ts(registration: Mapping[str, Any], drawn: Sequence[int], sigma: int, cap: int) -> list[int]:
    """Session σ's tile indices: drawn[σ·cap + s] while that is drawn, N − 1 after.  Raises ValueError unless `drawn` is
    the registration's K′ distinct indices of [0, N), ascending, and σ one of its S sessions."""
    n, law = registration.get("population"), registration.get("law")
    if not _positive(n) or n > 1 << T_BITS:
        raise ValueError(f"the registration's population {n!r} is not a tile count of at most 2^{T_BITS}")
    k = law.get("k") if isinstance(law, Mapping) and law.get("law") == "subset" else None
    if not _positive(k):
        raise ValueError(f"the registration's law {law!r} is not subset:K′")
    drawn = list(drawn)
    if len(drawn) != k or any(not isinstance(t, int) or isinstance(t, bool) for t in drawn) or drawn != sorted(set(drawn)) \
            or not 0 <= drawn[0] <= drawn[-1] < n:
        raise ValueError(f"the draw is not {k} distinct ascending indices of [0, {n})")
    if not isinstance(sigma, int) or not 0 <= sigma < sessions(k, cap):
        raise ValueError(f"session {sigma!r} is not one of the window's {sessions(k, cap)}")
    return [drawn[sigma * cap + s] if sigma * cap + s < k else n - 1 for s in range(cap)]


def public_inputs(registration: Mapping[str, Any], drawn: Sequence[int], sigma: int, cap: int, domain_h3: bytes) -> bytes:
    """F_σ: the verifier's own ``--public-inputs`` file for session σ's `cap` tile instances, from the registration (the
    one-stage record: its population N, its law subset:K′ and its window root and domain at port `PORT`), its own draw,
    σ, the cap and `domain_h3`, the domain id of the pass's -h3 tile tree (the verifier's own, from the pass's epoch and
    tile count).  A header line ``{format, instances, ports}`` (`PUBLIC_PORTS`), then each instance's words
    (`instance_ts`'s t, the root, the window domain, side A's and side B's domains, domain_h3), u16 LE, instance-major."""
    root, dom = _registered_root(registration)
    if len(domain_h3) != _DIGEST:
        raise ValueError("the pass's -h3 domain id is 64 bytes")
    common = root + dom + side_domain(dom, "a").domain_id + side_domain(dom, "b").domain_id + bytes(domain_h3)
    body = b"".join(t.to_bytes(16, "little") + common for t in instance_ts(registration, drawn, sigma, cap))
    head = {"format": PUBLIC_INPUTS_FORMAT, "instances": cap, "ports": [dict(p) for p in PUBLIC_PORTS]}
    return json.dumps(head, sort_keys=True).encode() + b"\n" + body


__all__ = ["DEPTHS", "ENTRY_SCHEMA", "Entry", "GRID_BITS", "KINDS", "M_MAX", "PORT", "PUBLIC", "PUBLIC_INPUTS_FORMAT",
           "PUBLIC_PORTS", "REGISTERED", "T_BITS", "Work", "bucket", "entry_bc", "entry_digest", "entry_leaf", "floor_point",
           "instance_ts", "k_prime", "layout_problem", "level", "levels", "on_grid", "pad_node", "pass_tiles", "point",
           "public_inputs", "record_digest", "record_problem", "sampling", "session_cap", "sessions", "side_domain",
           "side_index", "side_levels", "side_path", "side_row", "tile_at", "tile_in", "tile_read", "tiles", "window_problem"]
