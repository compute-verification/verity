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
B's, row-major, and index u of its range is (u div n_b, u mod n_b); the rows' places in the call stay in its record, which
the entry binds by SHA-512.  The filler entry holds zero rows on both sides and the range [N_real, N).  `tile_at` maps a
drawn index to its entry and rows, and `window_problem` is the window statement (L4) in Python over the entry rows: the
rows are entry rows, the ranges tile [0, N) in order, each call's range is its tiles, the filler covers the rest, and the
registered level is `bucket` of the real tiles.  It is not proved yet: no Program states it.

**The layout for gates** (`note:proofs/20261006T1940Z-draft-units-to-inner-layout`, §2) is specified in PoUW's PROTOCOL.md,
"The served audit window": the entry row (`Entry`, 240 bytes), the dense side trees of 2^D leaves (`DEPTHS`) padded with
frame-v3's pad leaf under the window's side domains (`side_domain`, `side_levels`, `side_path`), and the tile instances'
public inputs (`PUBLIC_PORTS`, `sessions`, `instance_ts`, `public_inputs`); `tests/vectors/window.json` pins them.
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

from verity.primitives.circuits.codec import canonical_json
from verity.primitives.commitments import hm96
from verity.primitives.commitments.identity import identity_digest_sha512
from verity.primitives.commitments.indexed import RangeIndexedDomain
from verity.primitives.commitments.merkle import CommitmentDomain
from verity.primitives.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2, row_leaf, sha512_row_v2_digest

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


def _positive(n: Any) -> bool:
    return isinstance(n, int) and not isinstance(n, bool) and n >= 1


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

def k_prime(eps: Fraction, delta: Fraction, credited: int, n: int, w_max: int) -> int:
    """L3's K′: the smallest K with (1 − ε·credited/(n·w_max))^K <= δ, at most n (`credited` the credited work, `n` the tiles,
    each doing at most `w_max`).  Computed in binary64 logarithms, then K stepped until the inequality holds there."""
    eps, delta = Fraction(eps), Fraction(delta)
    if not (0 < eps <= 1 and 0 < delta < 1 and _positive(n) and _positive(w_max) and 0 < credited <= n * w_max):
        raise ValueError(f"k_prime needs 0 < ε <= 1, 0 < δ < 1 and 0 < credited <= n·w_max, not {eps}, {delta}, "
                         f"{credited}, {n}, {w_max}")
    p = eps * credited / (n * w_max)
    if p == 1:
        return 1
    step, target = math.log1p(-float(p)), math.log(float(delta))
    k = max(1, math.ceil(target / step))
    while k > 1 and (k - 1) * step <= target:
        k -= 1
    while k * step > target:
        k += 1
    return min(k, n)


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
ENTRY_BYTES = 240
ENTRY_WORDS = ENTRY_BYTES // 2
ENTRY_BITS = 8 * ENTRY_BYTES
KIND_CODE = {"none": 0, "call": 1, "filler": 2}
_HEAD = struct.Struct("<6Q")
_DIGEST = 64
_ZERO = bytes(_DIGEST)
_HM = hm96.SHA512


@dataclass(frozen=True)
class Entry:
    """One entry of the window root, as its committed bit row holds it (the module docstring's layout): its kind, its tile
    range [lo, hi), each side's row count and root, and a call's record's SHA-512.  `row` and `from_row` encode any values
    that fit the row's fields; `layout_problem` holds them to the layout."""

    kind: str
    lo: int = 0
    hi: int = 0
    n_a: int = 0
    n_b: int = 0
    root_a: bytes = _ZERO
    root_b: bytes = _ZERO
    record: bytes = _ZERO

    def row(self) -> bytes:
        if self.kind not in KIND_CODE:
            raise ValueError(f"an entry's kind is one of {KINDS}, not {self.kind!r}")
        if any(len(d) != _DIGEST for d in (self.root_a, self.root_b, self.record)):
            raise ValueError("an entry's roots and record are 64-byte digests")
        try:
            head = _HEAD.pack(KIND_CODE[self.kind], self.lo, self.hi, self.n_a, self.n_b, 0)
        except struct.error as e:
            raise ValueError(f"an entry's integers are u64s: {e}") from None
        return head + self.root_a + self.root_b + self.record

    def words(self) -> list[int]:
        """The row as its 120 u16 words, little-endian."""
        return list(struct.unpack(f"<{ENTRY_WORDS}H", self.row()))

    @classmethod
    def from_row(cls, row: bytes) -> "Entry":
        """The entry a row holds; raises ValueError for a row of another length, an unknown kind or a nonzero pad."""
        row = bytes(row)
        if len(row) != ENTRY_BYTES:
            raise ValueError(f"an entry row is {ENTRY_BYTES} bytes, not {len(row)}")
        code, lo, hi, n_a, n_b, pad = _HEAD.unpack_from(row)
        kind = {v: k for k, v in KIND_CODE.items()}.get(code)
        if kind is None:
            raise ValueError(f"kind {code} is none of {KINDS} ({sorted(KIND_CODE.values())})")
        if pad:
            raise ValueError("its bytes 40..48 are not zero")
        return cls(kind, lo, hi, n_a, n_b, row[48:112], row[112:176], row[176:240])

    @classmethod
    def from_words(cls, words: Sequence[int]) -> "Entry":
        if len(words) != ENTRY_WORDS or any(not 0 <= int(w) < 1 << 16 for w in words):
            raise ValueError(f"an entry row is {ENTRY_WORDS} u16 words")
        return cls.from_row(struct.pack(f"<{ENTRY_WORDS}H", *map(int, words)))

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "lo": self.lo, "hi": self.hi, "n_a": self.n_a, "n_b": self.n_b, "root_a": self.root_a.hex(),
                "root_b": self.root_b.hex(), "record": self.record.hex()}

    @classmethod
    def from_json(cls, x: Mapping[str, Any]) -> "Entry":
        return cls(x["kind"], x["lo"], x["hi"], x["n_a"], x["n_b"], bytes.fromhex(x["root_a"]), bytes.fromhex(x["root_b"]),
                   bytes.fromhex(x["record"]))


def record_digest(record: Mapping[str, Any]) -> bytes:
    """A call's record as its entry binds it: the SHA-512 of its canonical JSON (RFC 8785)."""
    return hashlib.sha512(canonical_json(record)).digest()


def layout_problem(entry: Entry) -> str | None:
    """Why an entry's values are not the layout's, or None: its range in [0, 2^40); an empty entry the zero row; a call's or
    the filler's sides 1..2^D rows; the filler's record zero."""
    if not (entry.lo < 1 << T_BITS and entry.hi < 1 << T_BITS):
        return f"its range [{entry.lo}, {entry.hi}) is not of {T_BITS}-bit indices"
    if entry.kind == "none":
        return None if entry == Entry("none") else "an empty entry is not the zero row"
    for side, n in (("a", entry.n_a), ("b", entry.n_b)):
        if not 1 <= n <= 1 << DEPTHS[side]:
            return f"side {side} holds {n} rows, not 1 to 2^{DEPTHS[side]}"
    if entry.kind == "filler" and entry.record != _ZERO:
        return "the filler's record is not zero"
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


def window_problem(rows: Sequence[bytes], work: Work, bound: int, bits: int = GRID_BITS) -> str | None:
    """The window statement (L4) on the committed entry rows, or None: `bound` rows, each an entry row of the layout
    (`Entry.from_row`, `layout_problem`); the calls, then one filler, then empty entries; the ranges tile [0, work.n) in
    order; a call's range is its tiles; the filler's holds the rest, within its rows; and the level is `bucket` of the
    calls' tiles.  Credited tiles are only the calls'."""
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
    at = 0
    for e, entry in enumerate(entries[:calls + 1]):
        if entry.lo != at or entry.hi <= at:
            return f"entry {e}: its range does not start where the last ended, or is empty"
        size = entry.hi - entry.lo
        if entry.kind == "call" and size != tiles(entry):
            return f"entry {e}: its range is {size} tiles, not its {tiles(entry)} tiles"
        if entry.kind == "filler" and size > tiles(entry):
            return f"entry {e}: the filler's range is {size} tiles, beyond its {tiles(entry)}"
        at = entry.hi
    real = entries[calls].lo
    if at != work.n:
        return f"the ranges end at {at}, not the padded count {work.n}"
    if bucket(real, bits) != work:
        return f"{real} real tiles are level {bucket(real, bits).level}, not the registered level {work.level}"
    return None


# ---- the tile instances' public inputs ------------------------------------------------------------------------------------

#: C-Flock's format of the verifier's own ``--public-inputs`` file (its PROTOCOL.md §16.10, Lean's ``HmIn.checkOwn``)
PUBLIC_INPUTS_FORMAT = "verity/flock-public-inputs/v1"
#: a tile instance's public input ports, in order, as its circuit's META ``public_inputs`` must name them: each port's u16
#: words (whole 128-bit words, so no row leaf shares one), the public datum it is recomputed from and its rule.  t is a
#: u128 below 2^40; the root and the domain ids are 64 bytes, each u16 word two of them, little-endian.
PUBLIC_PORTS = (
    {"port": "t", "words": 8, "seed": "drawn", "derivation": "verity/pouw/window/t/v1"},
    {"port": "root", "words": 32, "seed": "registration", "derivation": "verity/pouw/window/root/v1"},
    {"port": "domain", "words": 32, "seed": "registration", "derivation": "verity/pouw/window/domain/v1"},
    {"port": "domain_a", "words": 32, "seed": "registration", "derivation": "verity/pouw/window/side-domain-a/v1"},
    {"port": "domain_b", "words": 32, "seed": "registration", "derivation": "verity/pouw/window/side-domain-b/v1"},
)


def sessions(k: int, cap: int) -> int:
    """S, the fixed-shape sessions of `cap` instances that hold a window's K′ = `k` drawn tiles."""
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


def public_inputs(registration: Mapping[str, Any], drawn: Sequence[int], sigma: int, cap: int) -> bytes:
    """F_σ: the verifier's own ``--public-inputs`` file for session σ's `cap` tile instances, from the registration (the
    one-stage record: its population N, its law subset:K′ and its window root and domain at port `PORT`), its own draw,
    σ and the cap alone.  A header line ``{format, instances, ports}`` (`PUBLIC_PORTS`), then each instance's words
    (`instance_ts`'s t, the root, the window domain, side A's and side B's domains), u16 LE, instance-major."""
    root, dom = _registered_root(registration)
    common = root + dom + side_domain(dom, "a").domain_id + side_domain(dom, "b").domain_id
    body = b"".join(t.to_bytes(16, "little") + common for t in instance_ts(registration, drawn, sigma, cap))
    head = {"format": PUBLIC_INPUTS_FORMAT, "instances": cap, "ports": [dict(p) for p in PUBLIC_PORTS]}
    return json.dumps(head, sort_keys=True).encode() + b"\n" + body


__all__ = ["DEPTHS", "ENTRY_SCHEMA", "Entry", "GRID_BITS", "KINDS", "PORT", "PUBLIC", "PUBLIC_INPUTS_FORMAT", "PUBLIC_PORTS",
           "REGISTERED", "T_BITS", "Work", "bucket", "entry_bc", "entry_digest", "entry_leaf", "floor_point", "instance_ts",
           "k_prime", "layout_problem", "level", "levels", "on_grid", "pad_node", "point", "public_inputs", "record_digest",
           "sampling", "sessions", "side_domain", "side_levels", "side_path", "tile_at", "tile_in", "tiles", "window_problem"]
