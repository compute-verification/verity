"""PoUW's audit window as the proof service sees it: what the window makes public, the grid its credit is rounded to, the
size of the uniform draw, and the window's advice (its per-call entries over a public shape, and the map from a drawn
index to a tile), with the window statement's reference
(`note:verity-root/20261006T0545Z-draft-daniel-one-recursive-architecture-v2`, PoUW; the joint note's L2, L3 and L4).

**The window is static** (Daniel, 18 Oct; `note:compute-accounting/20261018T2115Z-draft-static-window`): its structure is
a function of public parameters only, the window's **shape** (`Call`: each audited call's batch slots m, weight rows n and
first tile in its pass).  Every slot's row and every weight row is committed in place, so the tile count N, every entry's
range and rows, the side trees and `tile_at` follow from the shape and from no value.  What the values decide is a
selection, two annotation words in each side row: ``ok``, its forming's verdict (`Pc8RowA`/`Pc8RowB`'s ``ok``: 1 means the
noise floor ρ·α >= 1 holds), and ``admit``, whether it holds admitted work (a slot's request; every weight row).  Tile
(i, j) is credited iff s = admit_i ∧ ok_i ∧ admit_j ∧ ok_j, and the window credits the masked sum of s over its tiles,
rounded down to the grid.  A row that is not admitted is the canonical empty row, zero words and ok 0 (the emptiness
check), so an empty slot's ok can't be set, and an admitted zero row has ok 0.

**Public items** (`PUBLIC`), exactly the draft's list and the shape.  The service refuses a registration that makes
anything else public; the window's own root and W's root are salted roots, which the rule allows.

    item       kind         source     what it is
    verdict    output       outcome    every drawn tile proved and opened, or not
    work       output       window     the credited tiles, the masked sum rounded down to a grid level, of the N tiles
    check      structure    parameter  the tile template every unit instantiates, its work per tile, the commitment scheme
    sampling   structure    parameter  ε, δ, the grid and the law subset:K′ they fix with the work level (`k_prime`)
    shape      structure    parameter  the window's calls (slots, weight rows, first pass tile): N and every tile's place
    drawn      coins        outcome    the verifier's own draw, after the receipt
    root_W     commitment   parameter  the weights' salted root

**The grid** (`GRID_BITS` = s): the positive integers with at most s + 1 significant bits, so consecutive points above
2^(s+1) differ by a factor of at most 1 + 2^-s (9/8 at s = 3).  A window whose masked sum is N_real credited tiles
registers one level ℓ, the largest point g_ℓ <= N_real, and is credited g_ℓ tiles' work out of its N (`credit`).  So the
window's influenceable public output is one grid level, and over a declared range [lo, hi] it takes `levels(lo, hi)` = q
values: log2 q bits in the draft's leakage bound.  The honest cost is under 1 − 2^-s of the credit (at most 11% at
s = 3); the padding (empty slots, refused rows, the filler) costs the factor N / g_ℓ in K′ and no credit.

**The draw** (L3): uniform subset:K′ over the N equal tiles, K′ the smallest K with (1 − ε·W/(N·w_max))^K <= δ, W the
credited work and w_max the most work a tile does (`k_prime`).  A wrong set holding a share ε of W has at least ε·W/w_max
tiles, so a draw of K′ without replacement misses it with probability at most δ.

**The advice** (L2): one entry per call of the shape, then one filler entry, then empty entries up to the public bound P,
each an `Entry` row committed in one window root registered before the draw.  Call c's entry has n_a = m_c (every slot)
and n_b = n_c (every weight row), its range the m_c·n_c tiles after the calls before it; side A's leaf i is slot i's row,
side B's leaf j weight row j, so index u of its range is (u div n_b, u mod n_b).  Each side row carries its place in the
call (I, J), its ok word and its admit word (`side_row`).  A call's entry also holds the root of its pass's -h3 tile
tree, where each drawn tile's served leaf is read, and the call's place in that pass (tile0, cols, m), so a unit computes
the leaf and its word in it from the rows it reads (`tile_read`); and the SHA-512 of the call's record.  The filler entry
holds one canonical empty row on each side and the last index N − 1.  `tile_at` maps a drawn index to its entry and rows
from the shape alone, and `window_problem` is the window statement (L4) in Python over the entry rows and the side rows'
fields (`side_field`): the entries are the shape's, every call's -h3 root is the pass's, each side row's index is its
leaf, every row not admitted is the canonical empty row, the filler's are not admitted, and the registered level is
`credit` of the masked sum; given the records it also holds them to the entries.  It is not proved yet: no Program
states it, and the annotation words are the committer's, since the units that form them (the A rows' formings, the B
rows' formings each epoch, the request queue's admission, the tally's limbs) are not yet the always-proved set of
`note:compute-accounting/20261018T2200Z-draft-padded-credit-rule` (§8): until they are, a row with both words set wrongly
is credited.  The draw stays uniform over the N tiles, never weighted by an annotation.

**The layout for gates** (`note:proofs/20261006T1940Z-draft-units-to-inner-layout`, §2) is specified in PoUW's PROTOCOL.md,
"The served audit window": the entry row (`Entry`, 336 bytes), the side rows (`side_row`), the side trees of 2^D leaves
(`DEPTHS`) padded with frame-v3's pad leaf under the window's side domains (`side_domain`, `side_levels`, `side_path`), and
the tile instances' public inputs (`PUBLIC_PORTS`, `sessions`, `session_cap`, `instance_ts`, `public_inputs`);
`tests/vectors/window.json` pins them.

**Live windows** (decision 36).  The check item's optional ``coins`` is the window's coin mode, ``live`` or ``seeded``
(`mode`; an item that names none is seeded, as Pearl-C's served path registers).  A seeded window's noise is its scheme's
derivation from the salt and the prover's roots.  A live window's comes from the proof service's challenger
(`verity.core.protocols.sampled_proofs.challenger`), whose first client this is: the epoch's coins are the layout
`live_families(R)` under `LIVE_SCHEME`, its salt coin, its draw coin and R round coins (``pouw/round``, rounds numbered
from 0 as Compose's `Declared.round` numbers them), all committed before the epoch.  Round r's coin opens once, after the
verifier recorded round r's root, the root of the values round r commits, and round r's noise key is `round_key` of that
coin and that root.  `rounds_reason` is the rule on a window's round coins (Lean: `PouwWindow.Rounds`).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Any, Mapping, Sequence

from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.ml.circuits.codec import canonical_json
from verity.ml.commitments import hm96
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2, row_leaf, sha512_row_v2_digest
from verity.ml.randomness import Key, derive

GRID_BITS = 3

#: (item, kind, source, reason): the window's public items, in the service's `Public` form
PUBLIC = (
    ("verdict", "output", "outcome", "what the credit rests on: every drawn tile was proved and opened, or not"),
    ("work", "output", "window", "the credited work, the masked sum of credited tiles rounded down to a grid level, of the "
                                 "shape's N tiles the draw ranges over; one level, so at most log2 q bits"),
    ("check", "structure", "parameter", "the check's identity: the tile template, its work per tile and the commitment "
                                        "scheme fix the auditor's verifier"),
    ("sampling", "structure", "parameter", "ε, δ and the grid, and the law subset:K′ they fix with the level: the guarantee "
                                           "is stated in them"),
    ("shape", "structure", "parameter", "the window's calls, each its batch slots, weight rows and first tile in the pass: "
                                        "they fix N, every entry's range and rows, and the tile each index is"),
    ("drawn", "coins", "outcome", "the verifier's own draw, after the receipt"),
    ("root_W", "commitment", "parameter", "the weights' salted root: what the tiles' weight rows are committed to, hiding"),
)
#: the items a registration carries; the verdict and the draw come after it
REGISTERED = ("work", "check", "sampling", "shape", "root_W")
KINDS = ("call", "filler", "none")
#: the check item's coin mode (`mode`), Compose's `CoinMode`
COINS = "coins"
MODES = ("live", "seeded")
LIVE_SCHEME = "verity/pouw/epoch-coins/live/v1"
SALT, DRAW, ROUND = "pouw/salt", "pouw/draw", "pouw/round"
ROUND_NOISE_DOMAIN = "verity/pouw/round-noise/v1"


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
    """A window's work level: of its `n` tiles (the shape's, `shape_tiles`) it is credited `credited`, point(level)."""
    level: int
    credited: int
    n: int

    @classmethod
    def at(cls, ell: int, n: int, bits: int = GRID_BITS) -> "Work":
        return cls(ell, point(ell, bits), n)

    def to_json(self, tile_work: int) -> dict[str, int]:
        return {"level": self.level, "n": self.n, "credited_tiles": self.credited, "credited": self.credited * tile_work}


def credit(real: int, n: int, bits: int = GRID_BITS) -> Work:
    """The work of a window of `n` tiles whose masked sum is `real` credited tiles, 1 <= real < n (its filler is never
    credited): its level is floor_point(real)'s."""
    if not (_positive(real) and _int(n) and real < n):
        raise ValueError(f"a window of {n!r} tiles credits 1 to {n!r} − 1 of them, not {real!r}")
    return Work.at(level(floor_point(real, bits), bits), n, bits)


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
    step, target = math.log1p(-float(p)) if float(p) < 1 else -math.inf, math.log(float(delta))
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
#: the side trees' public depths: A holds a call's batch slots (at most the served batch's 8,192 tokens), B its weight rows
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
    """Why a call's record is not its entry's, or None: the entry binds its SHA-512, and its ``shape`` [m, k, n] and
    ``tile0`` give the entry's m, cols = ⌈n/64⌉ and tile0."""
    try:
        if record_digest(record) != entry.record:
            return "its record is not the one the entry binds"
        m, _k, n = record["shape"]
        tile0 = record["tile0"]
        if (entry.tile0, entry.cols, entry.m) != (tile0, -(-n // TILE), m):
            return f"its tile0, cols and m are not its record's {tile0}, ⌈{n}/64⌉ and {m}"
    except (KeyError, TypeError, ValueError):
        return "its record has no shape [m, k, n] and tile0"
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
#: first, then its ok word and its admit word, then four zero words, so the field is one 128-bit word after the row's
#: operand words
SIDE_INDEX_WORDS = 8


def _bit(x: Any) -> bool:
    return _int(x) and x in (0, 1)


def side_row(words: Sequence[int], index: int, ok: int, admit: int) -> list[int]:
    """A side tree's row: the operand row's u16 words (fp8-p: k/2 code words, then P's 64), then its index field, which
    holds its place in the call, `ok`, its forming's verdict (1 formed with the noise floor, 0 refused or a zero row), and
    `admit`, whether the row holds admitted work (a slot's request from the batch's queue; every weight row; 0 for an
    empty slot and the filler)."""
    if len(words) % SIDE_INDEX_WORDS:
        raise ValueError(f"a side row's operand words are whole 128-bit words, not {len(words)} u16 words")
    if not (_int(index) and 0 <= index < 1 << 32):
        raise ValueError(f"a side row's index is a u32, not {index!r}")
    if not (_bit(ok) and _bit(admit)):
        raise ValueError(f"a side row's ok and admit are 0 or 1, not {ok!r} and {admit!r}")
    return [int(w) for w in words] + [index & 0xFFFF, index >> 16, ok, admit] + [0] * (SIDE_INDEX_WORDS - 4)


def side_field(row: Sequence[int]) -> tuple[int, int, int, int]:
    """(index, ok, admit, blank) of a side row, blank 1 when its operand words are all zero; raises ValueError unless its
    index field is a u32 index, two bits and zeros."""
    field = [int(w) for w in row[-SIDE_INDEX_WORDS:]]
    if len(row) <= SIDE_INDEX_WORDS or any(field[4:]) or field[2] not in (0, 1) or field[3] not in (0, 1) \
            or any(not 0 <= w < 1 << 16 for w in field[:2]):
        raise ValueError("the row's last 128-bit word is not a u32 index, an ok bit, an admit bit and zeros")
    return field[0] | field[1] << 16, field[2], field[3], int(not any(int(w) for w in row[:-SIDE_INDEX_WORDS]))


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


# ---- the shape, the advice and the window statement ----------------------------------------------------------------------

@dataclass(frozen=True)
class Call:
    """One call of a window's shape: its batch slots `m` (side A's rows, every slot whether it holds a token or not), its
    weight rows `n` (side B's) and its first tile `tile0` in its pass."""
    m: int
    n: int
    tile0: int

    def to_json(self) -> dict[str, int]:
        return {"m": self.m, "n": self.n, "tile0": self.tile0}


def shape_of(x: Any) -> tuple[Call, ...]:
    """The shape a JSON value holds, a non-empty list of calls ``{m, n, tile0}`` of natural numbers; raises ValueError for
    any other value."""
    bad = ValueError("a window's shape is a non-empty list of calls {m, n, tile0} of natural numbers")
    if not isinstance(x, (list, tuple)) or not x:
        raise bad
    out = []
    for c in x:
        if not isinstance(c, Mapping) or set(c) != {"m", "n", "tile0"} \
                or not all(_int(c[k]) and c[k] >= 0 for k in ("m", "n", "tile0")):
            raise bad
        out.append(Call(c["m"], c["n"], c["tile0"]))
    return tuple(out)


def shape_json(shape: Sequence[Call]) -> list[dict[str, int]]:
    return [c.to_json() for c in shape]


def shape_tiles(shape: Sequence[Call]) -> int:
    """N: the calls' m·n tiles each, and the filler's one."""
    return sum(c.m * c.n for c in shape) + 1


def shape_problem(shape: Sequence[Call], bound: int) -> str | None:
    """Why `shape` is not the shape of a window of `bound` entries, or None: 1 to bound − 1 calls (the filler is one more
    entry); each call's m whole 16s from 16 to 2^D_A and its n 1 to 2^D_B; the calls' pass tiles in order and disjoint,
    within 2^40; and its N below 2^40, so every range is of 40-bit indices."""
    if not 1 <= len(shape) <= bound - 1:
        return f"{len(shape)} calls, not 1 to the bound {bound} less the filler"
    in_pass = 0
    for c, call in enumerate(shape):
        if not (ROW_STEP <= call.m <= 1 << DEPTHS["a"] and call.m % ROW_STEP == 0):
            return f"call {c}: its m {call.m} is not a multiple of {ROW_STEP} from {ROW_STEP} to 2^{DEPTHS['a']}"
        if not 1 <= call.n <= 1 << DEPTHS["b"]:
            return f"call {c}: its n {call.n} is not 1 to 2^{DEPTHS['b']}"
        if call.tile0 < in_pass:
            return f"call {c}: its pass tiles start at {call.tile0}, before the last call's end {in_pass}"
        in_pass = call.tile0 + -(-call.m // TILE) * -(-call.n // TILE)
        if in_pass > 1 << T_BITS:
            return f"call {c}: its pass tiles end at {in_pass}, beyond 2^{T_BITS}"
    if shape_tiles(shape) >= 1 << T_BITS:
        return f"its {shape_tiles(shape)} tiles are not of {T_BITS}-bit indices"
    return None


def shape_entries(shape: Sequence[Call], bound: int) -> list[Entry]:
    """The entries' public fields, each a function of the shape (their roots and records zero): call c's range is the
    m_c·n_c tiles after the calls before it, its sides m_c and n_c rows, its pass place (tile0, ⌈n/64⌉, m); the filler's
    range the last index N − 1, one row each side; then empty entries up to `bound`."""
    out, at = [], 0
    for c in shape:
        out.append(Entry("call", at, at + c.m * c.n, c.m, c.n, tile0=c.tile0, cols=-(-c.n // TILE), m=c.m))
        at += c.m * c.n
    out.append(Entry("filler", at, at + 1, 1, 1))
    return out + [Entry("none")] * (bound - len(out))


def public_fields(entry: Entry) -> tuple[str, int, int, int, int, int, int, int]:
    """An entry's fields that the shape fixes: kind, range, side rows, tile0, cols and m."""
    return entry.kind, entry.lo, entry.hi, entry.n_a, entry.n_b, entry.tile0, entry.cols, entry.m


def tile_at(shape: Sequence[Call], t: int) -> tuple[int, int, int]:
    """(entry, A row, B row) of drawn index t, from the shape alone: in call c's range, t − lo_c = i·n_c + j, slot i and
    weight row j; N − 1 is the filler's (C, 0, 0)."""
    if not _int(t) or t < 0:
        raise ValueError(f"index {t!r} is not one of the window's tiles")
    at = 0
    for c, call in enumerate(shape):
        if t < at + call.m * call.n:
            i, j = divmod(t - at, call.n)
            return c, i, j
        at += call.m * call.n
    if t == at:
        return len(shape), 0, 0
    raise ValueError(f"index {t} is not one of the window's {at + 1} tiles")


def tile_read(entry: Entry, i: int, j: int) -> tuple[int, int]:
    """(p, at): the -h3 leaf a call's tile reads and its word's place in that leaf, from the entry row and the two side
    rows' indices I and J alone.  p = tile0 + (I div 64)·cols + (J div 64), the pass's tile holding (I, J); at =
    (I mod 64) | (J mod 64) << 6 | (rows/16 − 1) << 12, rows = min(64, m − 64·(I div 64)) that tile's height."""
    if entry.kind != "call" or not (0 <= i < entry.m and 0 <= j < TILE * entry.cols):
        raise ValueError(f"({i}, {j}) is not a word of the call's m x 64·cols")
    rows = min(TILE, entry.m - TILE * (i // TILE))
    return entry.tile0 + (i // TILE) * entry.cols + j // TILE, i % TILE | (j % TILE) << 6 | (rows // ROW_STEP - 1) << 12


def _fields(fields: Any, count: int) -> list[Sequence[Mapping[str, Sequence[Sequence[int]]]]]:
    """`fields` as `window_problem` reads it: per entry up to the filler, each side's [index, ok, admit, blank] per row
    (`side_field`), integers."""
    bad = ValueError("the side rows' fields are, per call and then the filler, each side's [index, ok, admit, blank] per "
                     "row")
    if not isinstance(fields, (list, tuple)) or len(fields) != count:
        raise bad
    for f in fields:
        if not isinstance(f, Mapping) or any(not isinstance(f.get(s), (list, tuple)) for s in DEPTHS):
            raise bad
        for s in DEPTHS:
            if any(not isinstance(p, (list, tuple)) or len(p) != 4 or not all(_int(x) for x in p) for p in f[s]):
                raise bad
    return list(fields)


def selected(fields: Mapping[str, Sequence[Sequence[int]]]) -> dict[str, int]:
    """Each side's selected rows, Σ ok·admit over its rows' fields: a tile is credited iff both its rows are selected."""
    return {s: sum(p[1] * p[2] for p in fields[s]) for s in DEPTHS}


def window_problem(rows: Sequence[bytes], work: Work, bound: int, root_h3: bytes, shape: Sequence[Call],
                   fields: Sequence[Mapping[str, Sequence[Sequence[int]]]], bits: int = GRID_BITS,
                   records: Sequence[Mapping[str, Any] | None] | None = None) -> str | None:
    """The window statement (L4) on the committed entry rows and the side rows' index fields, or None: `bound` rows, each
    an entry row of the layout (`Entry.from_row`, `layout_problem`) whose public fields are the shape's (`shape_entries`);
    every call's -h3 root is `root_h3`, the pass's published tile commitment; in `fields` (per call and then the filler,
    each side's (index, ok, admit, blank) per row, `side_field`) each row's index is its leaf, its ok, admit and blank
    are bits, the filler's rows are not admitted, and a row that is not admitted is the canonical empty row (blank, ok
    0: the emptiness check, so an empty slot's ok can't be set); and the registered work is `credit` of the masked sum,
    Σ over the calls of (Σ ok_A·admit_A)·(Σ ok_B·admit_B) (`selected`): tile (i, j) is credited iff s = admit_i ∧ ok_i
    ∧ admit_j ∧ ok_j, so an empty slot, a refused row and the filler are never credited.

    Given the calls' `records` (per entry, None past the calls) it also holds each to its entry (`record_problem`).  Raises
    ValueError for a shape that is not one (`shape_problem`), a root not 64 bytes, or fields not of their form."""
    if len(root_h3) != _DIGEST:
        raise ValueError("the pass's -h3 tile root is 64 bytes")
    why = shape_problem(shape, bound)
    if why:
        raise ValueError(f"the shape is not a window's: {why}")
    fields = _fields(fields, len(shape) + 1)
    if len(rows) != bound:
        return f"{len(rows)} entries, not the bound {bound}"
    want = shape_entries(shape, bound)
    entries = []
    for e, row in enumerate(rows):
        try:
            entry = Entry.from_row(row)
        except ValueError as x:
            return f"entry {e}: not an entry row: {x}"
        why = layout_problem(entry)
        if why:
            return f"entry {e}: not the layout: {why}"
        if public_fields(entry) != public_fields(want[e]):
            return f"entry {e}: its kind, range, rows, tile0, cols and m are not the shape's"
        if entry.kind == "call" and entry.root_h3 != root_h3:
            return f"entry {e}: its -h3 root is not the pass's tile commitment"
        entries.append(entry)
    real = 0
    for e, entry in enumerate(entries[:len(shape) + 1]):
        for side, count in (("a", entry.n_a), ("b", entry.n_b)):
            got = fields[e][side]
            if [p[0] for p in got] != list(range(count)):
                return f"entry {e}: its side-{side} rows' indices are not their leaves 0 to {count - 1}"
            if any(x not in (0, 1) for p in got for x in p[1:]):
                return f"entry {e}: a side-{side} row's ok, admit or blank is not 0 or 1"
            if entry.kind == "filler" and any(p[2] for p in got):
                return f"entry {e}: the filler's side-{side} row is admitted"
            for i, p in enumerate(got):
                if not p[2] and (p[1] or not p[3]):
                    return f"entry {e}: side-{side} row {i} is not admitted and not the canonical empty row"
        sel = selected(fields[e])
        real += sel["a"] * sel["b"] if entry.kind == "call" else 0
    n = shape_tiles(shape)
    if real < 1:
        return "no tile is credited: no call has an admitted slot and weight row that both passed their forming"
    got_work = credit(real, n, bits)
    if got_work != work:
        return f"{real} credited tiles of {n} are level {got_work.level}, not the registered level {work.level} of {work.n}"
    if records is not None:
        if len(records) != bound or any(r is not None for r in records[len(shape):]):
            return f"{len(records)} records, not one per call and None past the calls"
        for e in range(len(shape)):
            why = record_problem(entries[e], records[e])
            if why:
                return f"entry {e}: {why}"
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


# ---- live windows -----------------------------------------------------------------------------------------------------------

def mode(check: Mapping[str, Any]) -> str:
    """A window's coin mode (`MODES`), its check item's ``coins``: seeded when the item names none."""
    return check.get(COINS, "seeded")


def live_families(rounds: int) -> tuple[tuple[str, int], ...]:
    """A live epoch's coin layout: its salt coin, its draw coin and `rounds` round coins, the most rounds a window takes."""
    return ((SALT, 1), (DRAW, 1), (ROUND, rounds))


def live_challenger(epoch: int, rounds: int, entropy: Callable[[int], bytes] = os.urandom) -> Ch.LiveChallenger:
    return Ch.LiveChallenger(LIVE_SCHEME, epoch, live_families(rounds), entropy)


def live_epoch_reason(commitment: Ch.LiveCommitment) -> str | None:
    """Why `commitment` is not a live epoch's coin commitment (`LIVE_SCHEME`, `live_families`); None when it is."""
    if commitment.scheme != LIVE_SCHEME:
        return f"the coin commitment's scheme is {commitment.scheme}, not {LIVE_SCHEME}"
    f = commitment.families
    if len(f) != 3 or f[0] != (SALT, 1) or f[1] != (DRAW, 1) or f[2][0] != ROUND:
        return "the coin commitment's coins are not a live epoch's: one salt, one draw, then the rounds'"
    return None


def rounds_reason(commitment: Ch.LiveCommitment, roots: Sequence[bytes], openings: Sequence[Ch.LiveOpening]) -> str | None:
    """Why `openings` are not a live window's round coins over the round roots the verifier recorded, `roots` in order:
    round r's coin opened once, after round r's root, and every recorded round's coin opened; None when they are."""
    return live_epoch_reason(commitment) or Ch.order_reason(commitment, ROUND, roots, openings)


def round_key(commitment: Ch.LiveCommitment, coin: bytes, r: int, root: bytes) -> Key:
    """Round r's noise key, from its opened coin and the round's root: no Fiat–Shamir over the prover's roots."""
    return derive(coin, ROUND_NOISE_DOMAIN, {"coins": commitment.root, "epoch": commitment.epoch, "root": root, "round": r})


__all__ = ["COINS", "Call", "DEPTHS", "DRAW", "ENTRY_SCHEMA", "Entry", "GRID_BITS", "KINDS", "LIVE_SCHEME", "MODES", "M_MAX",
           "PORT", "PUBLIC", "PUBLIC_INPUTS_FORMAT", "PUBLIC_PORTS", "REGISTERED", "ROUND", "ROUND_NOISE_DOMAIN", "SALT",
           "T_BITS", "Work", "credit", "entry_bc", "entry_digest", "entry_leaf", "floor_point", "instance_ts", "k_prime",
           "layout_problem", "level", "levels", "live_challenger", "live_epoch_reason", "live_families", "mode", "on_grid",
           "pad_node", "pass_tiles", "point", "public_fields", "public_inputs", "record_digest", "record_problem",
           "round_key", "rounds_reason", "sampling", "selected", "session_cap", "sessions", "shape_entries", "shape_json",
           "shape_of", "shape_problem", "shape_tiles", "side_domain", "side_field", "side_levels", "side_path", "side_row",
           "tile_at", "tile_read", "window_problem"]
