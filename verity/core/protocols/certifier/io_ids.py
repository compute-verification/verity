"""The ingress-ids tie (`io_ids`, a prototype on a fake upstream Commit and the warden's own ingress): `IoTieRequest`
with the prompt ids' values held to the Commit's `c00.input/ids` rows, so the model computes on the prompt the user
sent (IoTieRequest counts each request's prompt fields and checks each is a u32, but ties the ids' values to
nothing). Like IoTieRequest it is one unit a bucket row of the warden's registered ingress
(`warden_rows.bucket_port`).

**The Commit side** (the fake upstream's layout, what the served path's Commit exposes): one `hm96-sha512/row/v2` row a
step under a fresh salt, the step's input ids u32be in the forward's token order (its batch rows in order, row r's n_r
tokens at offset Σ_{r' < r} n_{r'}), zero past to P, in a dense `frame-v3-sha512` tree of the session's steps whose
root and domain the Commit registers before the draw. A bucket reads its S rows at the map's steps as registered
reads (`port_reads`, the egress port's `io_tie.port_reads` over `INPUT_IDS`), so no gate of the unit hashes them.

**The map** (`bucket_map`, the verifier's, public in the named gap; at the target under the advice's hiding root).
The bucket's prompt fields in scan order are its lanes': lane by lane, connection q's positions from the lane's
carry-in fields to its carry-out's, q advice request i's where q = order[i] + 1; which cells hold them is the warden's
timing. Field f is position p of request i, and its class is that of the position whose computation i's KV holds
there, its root: (i, p), or for a prefix-cache hit (p below i's hit) its source's root at p, so a cached position
shares a class with the computation it reads. `keys[f]` is field f's class, numbered from 0 in order of first
appearance, the sink 2^CB − 1 past the bucket's fields. The opened steps are every step with a token whose root is a
class of the bucket, and `ekeys[e]` is opened entry e's class, or the sink. The verifier refuses an advice whose class
has no computation in any step.

**What 1 means** (`reference`): every read row opens at its step under the Commit's root (the verifier's registered
reads, outside the unit, as the bucket row's under the warden's; here `reads_open`), and the unit is 1
(`unit_reference`): IoTieRequest's output on the row and the lanes, and the bucket's fields (`scan`) and the read
entries hold one value a class, each class with a field. So every computation of a prompt position of the bucket, and
every KV read in its place, is the id the user sent there.

**The unit** (`IoTieRequestIds{B, L, R, P, S}`) is IoTieRequest's scan, lane by lane, plus:
- each request cell's slots (`IoTieIdsSlots{B}`): slot k is a field when the frame's count of fields is above k, its
  value the u32 its 10 characters spell, its ordinal nf + k, nf the bucket's fields before the cell;
- the classes routed to the fields (`io_tie.IoTieRoute` over M1 = 2^D1 ≥ NK slot positions, NK = N·K): record f is
  (keys[f], f), and `IoTieIdsKeys` holds each field to the record routed to it, with index its ordinal and a class
  not the sink, every other slot the sink;
- the members sorted by class (`IoTieRoute` over M2 = 2^D2 ≥ NK + E): each slot (class, 0, value), each opened entry
  (ekeys[e], 1, its id), the rest the sink; `IoTieIdsChain` checks the classes non-decreasing, the values equal within
  a class but the sink, and each class's first member a slot (`IoTieIdsFirst` on the first member and `IoTieIdsLink`
  on each member and its successor, one batch). A Beneš network is a permutation under any switches, so every
  member is in the sorted order and the switches only sort.
Public: IoTieRequest's lanes, the Commit's root and domain id (its registration), the steps (the port's read
positions), `keys` and `ekeys`. Witness: the row and the read rows (each with its salt and path, the registered
reads'), and both routings' switches.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple, Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import ZERO, add, and_reduce, const_bits
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.types import Array
from verity.ml.commitments.gates.frame_v3 import _part, opens
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as SCHEMA
from verity.core.protocols.certifier import head_check as HC
from verity.core.protocols.certifier import io_request as Q
from verity.core.protocols.certifier import io_tie as IT
from verity.core.protocols.certifier.active import HEADER, ID_DIGITS, ID_FIELD_BYTES

BIT, BYTE = HC.BIT, HC.BYTE
#: a slot's record out of `IoTieIdsSlots`: field, ordinal, value
SLOT = 1 + 32 + 32
#: the registered value the ingress port reads, the Commit's input ids
INPUT_IDS = "c00.input/ids"


@dataclass(frozen=True)
class Shape:
    """An ingress bucket row (`io_request.Shape`'s B, L, R) and its Commit port: S read rows of P input ids
    (E = S·P entries) from a tree of depth DS. Each of the N = L·R cells has K slots, NK in all; classes are CB-bit,
    below NK, the sink 2^CB − 1; the classes are routed over M1 = 2^D1 ≥ NK positions and the members sorted over
    M2 = 2^D2 ≥ NK + E."""

    B: int
    L: int
    R: int
    P: int
    S: int
    DS: int

    def __post_init__(self) -> None:
        Q.Shape(self.B, self.L, self.R)
        for name, least, most in (("P", 1, 1 << 28), ("S", 1, 1 << 32), ("DS", 1, 32)):
            if type(getattr(self, name)) is not int or not least <= getattr(self, name) <= most:
                raise ValueError(f"a Commit port has {least} <= {name} <= {most}, not {getattr(self, name)!r}")
        if self.NK + self.E > 1 << 32:
            raise ValueError(f"a bucket has at most 2^32 slots and entries, not {self.NK} and {self.E}")

    @property
    def request(self) -> Q.Shape:
        return Q.Shape(self.B, self.L, self.R)

    @property
    def K(self) -> int:
        return Q.layout(self.B).room // ID_FIELD_BYTES

    @property
    def NK(self) -> int:
        return self.L * self.R * self.K

    @property
    def E(self) -> int:
        return self.S * self.P

    @property
    def CB(self) -> int:
        return self.NK.bit_length()

    @property
    def sink(self) -> int:
        return (1 << self.CB) - 1

    @property
    def D1(self) -> int:
        return IT.route_depth(self.NK, 1)

    @property
    def D2(self) -> int:
        return IT.route_depth(self.NK + self.E, 1)

    @property
    def route(self) -> int:
        return (2 * self.D1 - 1) << (self.D1 - 1)

    @property
    def sort(self) -> int:
        return (2 * self.D2 - 1) << (self.D2 - 1)


# --- the fake upstream: the Commit's input-ids rows ------------------------------------------------------------------

class Row(NamedTuple):
    """A step's batch row (advice v0's `Row`, reduced): advice request q, its first position and its tokens this
    step."""

    q: int
    start: int
    n: int


def commit_rows(steps: Sequence[Sequence[Row]], tokens: Sequence[Sequence[int]], P: int) -> list[bytes]:
    """Each step's `c00.input/ids` row as the Commit holds it (`io_tie.port_row`): its rows' input ids in batch order,
    request q's position p being tokens[q][p] (its prompt, then its outputs)."""
    return [IT.port_row([tokens[r.q][r.start + t] for r in rows for t in range(r.n)], P) for rows in steps]


def commit(rows: Sequence[bytes], seed: bytes = b"commit") -> tuple[tuple[bytes, ...], list[list[bytes]], bytes]:
    """The fake Commit's tree of the rows under fake salts: (salts, levels, domain id)."""
    salts = tuple(IT.fake_salt(seed, s) for s in range(len(rows)))
    dom = IT.domain(hashlib.sha512(seed).digest(), len(rows)).domain_id
    return salts, IT.rows_levels(dom, rows, salts), dom


# --- the verifier's map ----------------------------------------------------------------------------------------------

class Map(NamedTuple):
    """A bucket's map: each field's class in scan order (the sink past them), the opened steps, and each opened
    entry's class (the sink for an entry no class of the bucket reads)."""

    keys: tuple[int, ...]
    steps: tuple[int, ...]
    ekeys: tuple[int, ...]


def root(q: int, p: int, sources: Mapping[int, tuple[int, int]]) -> tuple[int, int]:
    """The position whose computation request q's KV holds at position p: (q, p), or, below q's prefix-cache hit
    (`sources[q]` = (its source, the hit)), its source's root at p. Raises `ValueError` on a source that isn't an
    earlier request."""
    while q in sources and p < sources[q][1]:
        src = sources[q][0]
        if type(src) is not int or not 0 <= src < q:
            raise ValueError(f"request {q}'s prefix-cache source is an earlier request, not {src!r}")
        q = src
    return q, p


def fields_of(lanes: Sequence[Q.Lane], order: Sequence[int] | None = None) -> list[tuple[int, int]]:
    """The bucket's prompt fields in scan order, each (advice request, position), from its lanes: connection q is
    advice request i where order[i] + 1 = q (the identity by default)."""
    of = None if order is None else {o + 1: i for i, o in enumerate(order)}
    return [(ln.q - 1 if of is None else of[ln.q], p) for ln in lanes if ln.q for p in range(ln.fields, ln.fields_out)]


def bucket_map(shape: Shape, prompt_lens: Sequence[int], lanes: Sequence[Q.Lane], steps: Sequence[Sequence[Row]],
               sources: Mapping[int, tuple[int, int]], order: Sequence[int] | None = None) -> Map:
    """The verifier's map of an ingress bucket (the module docstring) from the advice's prompt lengths, its steps' rows,
    its prefix-cache sources and its order, and the bucket's lanes (`io_request.lanes`). Raises `ValueError` on more
    fields than NK, a step past P tokens or 2^DS steps, a class no step computes, or more opened steps than S."""
    sh, cls, keys = shape, {}, []
    for q, p in fields_of(lanes, order):
        keys.append(cls.setdefault(root(q, p, sources), len(cls)))
    if len(keys) > sh.NK:
        raise ValueError(f"the bucket has {len(keys)} prompt fields, more than NK = {sh.NK}")
    if len(steps) > 1 << sh.DS:
        raise ValueError(f"{len(steps)} steps are past a Commit of depth {sh.DS}")
    reads: dict[int, list[tuple[int, int]]] = {}
    for s, rows in enumerate(steps):
        off = 0
        for r in rows:
            for t in range(r.n):
                if r.start + t < prompt_lens[r.q] and root(r.q, r.start + t, sources) in cls:
                    reads.setdefault(s, []).append((off + t, cls[root(r.q, r.start + t, sources)]))
            off += r.n
        if off > sh.P:
            raise ValueError(f"step {s} runs {off} tokens, more than P = {sh.P}")
    missing = set(cls.values()) - {c for got in reads.values() for _, c in got}
    if missing:
        q, p = next(r for r, c in cls.items() if c == min(missing))
        raise ValueError(f"no step computes request {q}'s prompt position {p}")
    opened = sorted(reads) or [0]
    if len(opened) > sh.S:
        raise ValueError(f"the bucket's fields are computed in {len(opened)} steps, more than S = {sh.S}")
    ekeys = [sh.sink] * sh.E
    for k, s in enumerate(opened):
        for off, c in reads.get(s, ()):
            ekeys[k * sh.P + off] = c
    return Map(tuple(keys) + (sh.sink,) * (sh.NK - len(keys)),
               tuple(opened) + (opened[-1],) * (sh.S - len(opened)), tuple(ekeys))


# --- the reference ---------------------------------------------------------------------------------------------------

class Instance(NamedTuple):
    """One ingress bucket's ids tie: IoTieRequest's instance (the row and the lanes), the map, the opened rows with
    their salts and paths, and the Commit's root and domain id."""

    request: Q.Instance
    map: Map
    port: tuple[bytes, ...]
    port_salts: tuple[bytes, ...]
    paths: tuple[tuple[bytes, ...], ...]
    port_root: bytes
    port_domain: bytes


def scan(shape: Shape, request: Q.Instance) -> tuple[list[int], list[int]] | None:
    """The bucket's prompt fields' values in scan order and each one's slot position (cell·K + slot), or None where
    IoTieRequest refuses the row and the lanes."""
    rq, lay = shape.request, Q.layout(shape.B)
    if not Q.reference(rq, request):
        return None
    vals, at = [], []
    for k, ln in enumerate(request.lanes):
        left = ln.left
        for i in range(k * rq.R, (k + 1) * rq.R):
            c = request.cells[i]
            got = Q.read(lay, c, ln.q, left)
            if not isinstance(got, Q.Frame):
                continue
            base = HEADER.size + (lay.start if not left else 0)
            for j in range(got.fields):
                o = base + ID_FIELD_BYTES * j + 1
                vals.append(int(c[o:o + ID_DIGITS]))
                at.append(i * shape.K + j)
            left = (left or got.length) - got.used
    return vals, at


def matches(shape: Shape, vals: Sequence[int], mp: Map, port: Sequence[bytes]) -> bool:
    """The fields and the opened entries hold one value a class, each class with a field: field f's class is keys[f],
    not the sink, and each entry with a class other than the sink is its class's value (`IoTieIdsChain`'s reference)."""
    sh = shape
    if len(vals) > sh.NK or len(mp.keys) != sh.NK or len(mp.ekeys) != sh.E:
        return False
    held: dict[int, int] = {}
    for k, v in zip(mp.keys, vals):
        if k == sh.sink or held.setdefault(k, v) != v:
            return False
    return all(k == sh.sink or held.get(k) == entry(sh, port, e) for e, k in enumerate(mp.ekeys))


def entry(shape: Shape, port: Sequence[bytes], e: int) -> int:
    """Opened entry e: row ⌊e / P⌋'s (e mod P)-th input id."""
    s, r = divmod(e, shape.P)
    return int.from_bytes(port[s][4 * r:4 * r + 4], "big")


def _check_sizes(shape: Shape, inst: Instance) -> None:
    sh, mp = shape, inst.map
    if len(inst.port) != sh.S or len(inst.port_salts) != sh.S or len(inst.paths) != sh.S or len(mp.steps) != sh.S or \
            any(len(x) != 4 * sh.P for x in inst.port) or any(len(p) != sh.DS for p in inst.paths) or \
            any(not 0 <= s < 1 << sh.DS for s in mp.steps):
        raise ValueError(f"an instance has S = {sh.S} steps, rows of {4 * sh.P} bytes and paths of DS = {sh.DS}")


def reads_open(shape: Shape, inst: Instance) -> bool:
    """Every read row opens at its step under the Commit's root: the bucket's `port_reads`, in the clear."""
    _check_sizes(shape, inst)
    return all(opens(SCHEMA, inst.port_root, inst.port_domain, HC.hiding_value(x, s), st, list(p)) == 1
               for x, s, st, p in zip(inst.port, inst.port_salts, inst.map.steps, inst.paths))


def unit_reference(shape: Shape, inst: Instance) -> int:
    """`IoTieRequestIds`' output: IoTieRequest's on the row and the lanes, and the fields and the read entries held
    one value a class."""
    _check_sizes(shape, inst)
    got = scan(shape, inst.request)
    return int(got is not None and matches(shape, got[0], inst.map, inst.port))


def reference(shape: Shape, inst: Instance) -> int:
    """The bucket's ids tie (the module docstring, "What 1 means"): its registered reads and its unit."""
    return int(reads_open(shape, inst) and unit_reference(shape, inst) == 1)


def port_reads(maps: Sequence[Map], root: bytes, domain_id: bytes, count: int, P: int,
               reads_from: int = 0) -> dict[str, dict[str, Any]]:
    """The verifier's own `--registered` ports of the buckets' units over the Commit's `count` rows of P input ids
    under its registration's `root` and `domain_id`, never an instance's (`io_tie.port_reads`): unit `reads_from + u`
    reads the verifier's map u's steps."""
    return IT.port_reads([m.steps for m in maps], root, domain_id, count, P, INPUT_IDS, reads_from)


def instance(shape: Shape, request: Q.Instance, mp: Map, rows: Sequence[bytes], salts: Sequence[bytes],
             levels: Sequence[Sequence[bytes]], domain: bytes) -> Instance:
    """The prover's instance: the map's steps' rows opened in the Commit's tree."""
    port, port_salts, paths = IT.open_port(levels, rows, salts, mp.steps)
    return Instance(request, mp, port, port_salts, paths, levels[-1][0], domain)


# --- the prover's routings -------------------------------------------------------------------------------------------

def _fill(perm: list[int]) -> list[int]:
    taken = {p for p in perm if p >= 0}
    rest = iter(i for i in range(len(perm)) if i not in taken)
    return [p if p >= 0 else next(rest) for p in perm]


def route_switches(shape: Shape, inst: Instance) -> list[int]:
    """The classes' routing: record f to the slot position of the bucket's field f, the other positions in order."""
    got = scan(shape, inst.request)
    perm = [-1] * (1 << shape.D1)
    for f, s in enumerate(got[1] if got is not None else ()):
        perm[f] = s
    return IT.benes(_fill(perm))


def sort_switches(shape: Shape, inst: Instance) -> list[int]:
    """The members' sort as the unit forms them: each slot position its field's class (the sink but at a field), each
    entry its class, by (class, slot before entry, position)."""
    sh = shape
    got = scan(sh, inst.request)
    keys = [sh.sink] * sh.NK
    for f, s in enumerate(got[1] if got is not None else ()):
        keys[s] = inst.map.keys[f]
    members = [(k, 0) for k in keys] + [(k, 1) for k in inst.map.ekeys]
    members += [(sh.sink, 1)] * ((1 << sh.D2) - len(members))
    perm = [0] * len(members)
    for rank, m in enumerate(sorted(range(len(members)), key=lambda m: (*members[m], m))):
        perm[m] = rank
    return IT.benes(perm)


# --- the circuit -----------------------------------------------------------------------------------------------------

def _decimal(C, nibbles: Sequence[Sequence]) -> list:
    """The value of decimal digits (low nibbles, most significant first; a space's is 0), least significant bit first,
    each step v·10 + d as wide as its largest value."""
    v: list = []
    for k, d in enumerate(nibbles, 1):
        w = (10 ** k - 1).bit_length()
        pad = lambda x: (list(x) + [ZERO] * w)[:w]  # noqa: E731
        v = IT._add(C, IT._add(C, pad([ZERO] * 3 + v), pad([ZERO] + v)), pad(d))
    return v


def _sig_slots(S):
    Q.layout(S.B)
    return (("cell", Array(S.B, BYTE)), ("frame", Array(Q.FRAME_OUT, BIT)), ("nf", Array(32, BIT))), \
        Array(32 + (S.B - HEADER.size) // ID_FIELD_BYTES * SLOT, BIT)


def _build_slots(C, S, *e):
    lay = Q.layout(S.B)
    cell, fo, nf = e[:S.B], list(e[S.B]), list(e[S.B + 1])
    req, start, count = fo[1], fo[2], fo[36:68]
    out = add(C, nf, [C.AND(req, b) for b in count], width=32)
    for k in range(lay.room // ID_FIELD_BYTES):
        nib = []
        for d in range(ID_DIGITS):
            cont = cell[HEADER.size + ID_FIELD_BYTES * k + 1 + d][:4]
            o = HEADER.size + lay.start + ID_FIELD_BYTES * k + 1 + d
            first = cell[o][:4] if lay.start + ID_FIELD_BYTES * (k + 1) <= lay.room else [ZERO] * 4
            nib.append([C.AND(start, a ^ b) ^ b for a, b in zip(first, cont)])
        value = _decimal(C, nib)[:32]
        out += [C.AND(req, Q._gt_const(C, count, k)), *add(C, nf, const_bits(k, 32), width=32), *value]
    return [out]


IoTieIdsSlots = _part(
    "IoTieIdsSlots", ("B",), _sig_slots, _build_slots,
    "(nf', then each slot k's (field, ordinal, value)) of a B-byte ingress cell given its `IoTieRequestFrame` output "
    "and nf, the bucket's prompt fields before it: nf' is nf plus the frame's fields at a request cell; slot k is a "
    "field iff the cell is a request cell with more than k fields, its ordinal nf + k, its value the u32 its 10 "
    "characters spell at its place in a first cell's (start) or a continuation's reading; u32s least significant "
    "bit first")


def _check_keys(S) -> None:
    if not all(type(x) is int for x in (S.NK, S.CB, S.D1)) or not (1 <= S.NK <= 1 << S.D1 and 1 <= S.D1 <= 32 and
                                                                   S.NK < 1 << S.CB <= 1 << 32):
        raise ValueError(f"the keys have 1 <= NK <= 2^D1, NK < 2^CB <= 2^32 and 1 <= D1 <= 32, not NK={S.NK!r}, "
                         f"CB={S.CB!r}, D1={S.D1!r}")


def _sig_keys(S):
    _check_keys(S)
    return (("slots", Array(S.NK, Array(SLOT, BIT))), ("routed", Array(1 << S.D1, Array(S.CB + S.D1, BIT)))), \
        Array(1 + S.NK * S.CB, BIT)


def _build_keys(C, S, *e):
    slots, routed = e[:S.NK], e[S.NK:]
    oks, keys = [], []
    for s in range(S.NK):
        field, ordinal, r = slots[s][0], slots[s][1:33], routed[s]
        key, idx = r[:S.CB], r[S.CB:S.CB + S.D1]
        hit = and_reduce(C, [IT._eq(C, idx, ordinal[:S.D1]), *[b.inv() for b in ordinal[S.D1:]],
                             and_reduce(C, key).inv()])
        oks.append(C.AND(field, hit.inv()).inv())
        keys += [C.AND(b.inv(), field).inv() for b in key]
    return [[and_reduce(C, oks), *keys]]


IoTieIdsKeys = _part(
    "IoTieIdsKeys", ("NK", "CB", "D1"), _sig_keys, _build_keys,
    "(ok, each slot's class) of NK slots (field, ordinal, value) and the class records (class, index) routed to their "
    "positions: ok iff at each field the routed record's index is the field's ordinal and its class isn't the sink "
    "(2^CB − 1); a field's class is its record's, every other slot's the sink")


def _member(S, m):
    return m[:S.CB], m[S.CB], m[S.CB + 1:]


def _check_member(S):
    if type(S.CB) is not int or not 1 <= S.CB <= 32:
        raise ValueError(f"a chain's member has 1 <= CB <= 32, not CB={S.CB!r}")


def _sig_first(S):
    _check_member(S)
    return (("m", Array(S.CB + 33, BIT)),), BIT


def _build_first(C, S, m):
    key, kind, _ = _member(S, m)
    return [C.AND(kind, and_reduce(C, key).inv()).inv()]


IoTieIdsFirst = _part(
    "IoTieIdsFirst", ("CB",), _sig_first, _build_first,
    "1 iff the member (class, kind, value) is a slot (kind 0) or its class is the sink (2^CB − 1): the chain's first")


def _sig_link(S):
    _check_member(S)
    return (("a", Array(S.CB + 33, BIT)), ("b", Array(S.CB + 33, BIT))), Array(3, BIT)


def _build_link(C, S, a, b):
    (ka, _, va), (kb, kind, vb) = _member(S, a), _member(S, b)
    same, sink = IT._eq(C, ka, kb), and_reduce(C, kb)
    return [IT._lt(C, kb, ka).inv(), C.AND(C.AND(same, sink.inv()), IT._eq(C, va, vb).inv()).inv(),
            C.AND(C.AND(same.inv(), sink.inv()), kind).inv()]


IoTieIdsLink = _part(
    "IoTieIdsLink", ("CB",), _sig_link, _build_link,
    "(b's class isn't below a's, b has a's value if they share a class but the sink, b is a slot if it starts a class "
    "but the sink) of a member a and its successor b (class, kind, value), CB-bit classes, the sink 2^CB − 1")


def _sig_chain(S):
    if type(S.CB) is not int or not 1 <= S.CB <= 32 or type(S.D2) is not int or not 1 <= S.D2 <= 32:
        raise ValueError(f"a chain has 1 <= CB <= 32 and 1 <= D2 <= 32, not CB={S.CB!r}, D2={S.D2!r}")
    return (("members", Array(1 << S.D2, Array(S.CB + 33, BIT))),), BIT


def _body_chain(B, S, members):
    W, n, m = S.CB + 33, 1 << S.D2, HC._refs(members)
    pairs = Array(n - 1, Array(W, BIT))
    first = B.call(bind(IoTieIdsFirst, CB=S.CB), HC._coll(Array(W, BIT), m[:W]))
    links = B.batch(bind(IoTieIdsLink, CB=S.CB), HC._coll(pairs, m[:W * (n - 1)]), HC._coll(pairs, m[W:]), axes=(0, 0))
    # one And over the first member's bit and each link's three in order: the flat build's `and_reduce`, gate for gate
    bits = [*HC._refs(first), *HC._refs(links)]
    return B.call(BL.and_gate(len(bits)), HC._coll(Array(len(bits), BIT), bits))


IoTieIdsChain = CompositeDefinition(
    "IoTieIdsChain", 1, ("CB", "D2"), _sig_chain, _body_chain, conformance="exact-model-tested",
    doc="1 iff the 2^D2 members (class, kind, value) are sorted by class, CB bits, each member with its predecessor's "
        "class has its value unless the class is the sink (2^CB − 1), and each member that starts a class other than "
        "the sink is a slot (kind 0): `IoTieIdsFirst` on the first member and `IoTieIdsLink` on each member and its "
        "successor, batched, so a binding traces one link however many members")


def _unit_shape(S) -> Shape:
    """The unit's shape: the Commit's depth DS is its reads' (`port_reads`), not the unit's."""
    return Shape(S.B, S.L, S.R, S.P, S.S, 1)


def _sig_ids(S):
    sh = _unit_shape(S)
    request, _ = Q._sig_request(sh.request)
    return ((*request, ("keys", Array(sh.NK, Array(sh.CB, BIT))), ("ekeys", Array(sh.E, Array(sh.CB, BIT))),
             (IT.PORT, Array(S.S, Array(4 * S.P, BYTE))), ("route", Array(sh.route, BIT)),
             ("sort", Array(sh.sort, BIT))), BIT)


def _body_ids(B, S, row, lanes_, keys, ekeys, port, route, sort):
    sh = _unit_shape(S)
    CB = sh.CB
    K, cr, lr, pr = HC._Consts(B), HC._refs(row), HC._refs(lanes_), HC._refs(port)
    oks = []
    nf, slots = HC._coll(Array(32, BIT), K.bits(0, 32)), []
    for k in range(S.L):
        ln = lr[Q.LANE * k:Q.LANE * (k + 1)]
        q = HC._coll(Array(32, BIT), ln[0:32])
        state = HC._coll(Array(Q.STATE, BIT), ln[32:96] + K.bits(0, 2) + K.bits(1, 1))
        for j in range(S.R):
            i = S.R * k + j
            cell = HC._coll(Array(S.B, BYTE), cr[8 * S.B * i:8 * S.B * (i + 1)])
            fo = B.call(bind(Q.IoTieRequestFrame, B=S.B), cell, HC._coll(Array(32, BIT), HC._refs(state)[:32]), q)
            so = HC._refs(B.call(bind(IoTieIdsSlots, B=S.B), cell, fo, nf))
            nf, slots = HC._coll(Array(32, BIT), so[:32]), slots + so[32:]
            state = B.call(bind(Q.IoTieRequestStep), state, fo)
        oks.append(B.call(bind(Q.IoTieRequestEnd), state, HC._coll(Array(32, BIT), ln[96:128]),
                          HC._coll(Array(32, BIT), ln[128:160])))
    kr, xs1 = HC._refs(keys), []
    for f in range(1 << sh.D1):
        xs1 += (kr[CB * f:CB * (f + 1)] if f < sh.NK else K.bits(sh.sink, CB)) + K.bits(f, sh.D1)
    routed1 = B.call(bind(IT.IoTieRoute, W=CB + sh.D1, D=sh.D1),
                     HC._coll(Array(1 << sh.D1, Array(CB + sh.D1, BIT)), xs1), route)
    ko = HC._refs(B.call(bind(IoTieIdsKeys, NK=sh.NK, CB=CB, D1=sh.D1), HC._coll(Array(sh.NK, Array(SLOT, BIT)), slots),
                         routed1))
    ek, members = HC._refs(ekeys), []
    for s in range(sh.NK):
        members += ko[1 + CB * s:1 + CB * (s + 1)] + K.bits(0, 1) + slots[SLOT * s + 33:SLOT * (s + 1)]
    for e in range(sh.E):
        # entry e's id is u32be at bytes 4e ... 4e + 3 of the opened rows: its bit j is bit j % 8 of byte 4e + 3 - j // 8
        members += ek[CB * e:CB * (e + 1)] + K.bits(1, 1) + [pr[8 * (4 * e + 3 - j // 8) + j % 8] for j in range(32)]
    for _ in range(sh.NK + sh.E, 1 << sh.D2):
        members += K.bits(sh.sink, CB) + K.bits(1, 1) + K.bits(0, 32)
    ordered = B.call(bind(IT.IoTieRoute, W=CB + 33, D=sh.D2), HC._coll(Array(1 << sh.D2, Array(CB + 33, BIT)), members),
                     sort)
    oks.append(B.call(bind(IoTieIdsChain, CB=CB, D2=sh.D2), ordered))
    bits = [r for o in oks for r in HC._refs(o)] + [ko[0]]
    return B.call(BL.and_gate(len(bits)), HC._coll(Array(len(bits), BIT), bits))


IoTieRequestIds = CompositeDefinition(
    "IoTieRequestIds", 1, ("B", "L", "R", "P", "S"), _sig_ids, _body_ids,
    conformance="exact-model-tested",
    doc="1 iff IoTieRequest's checks hold on the bucket row and its lanes (each lane's scan from its carry-in on its "
        "ordinal to its carry-out), and the bucket's prompt fields (`IoTieIdsSlots`), each given its class by the "
        "routed records (`IoTieRoute`, `IoTieIdsKeys`), and the entries of the S Commit rows of P input ids, sorted by "
        "class by the switches (`IoTieRoute`), hold one value a class, each class starting with a field "
        "(`IoTieIdsChain`): the ingress-ids tie of one bucket row of the warden's registered root, every cell opened, "
        "the row and the Commit's rows at the bucket's steps each the proof system's registered read")


# --- arguments -------------------------------------------------------------------------------------------------------

def slots_arguments(cells: Sequence[bytes], frames: Sequence[Sequence[int]], nfs: Sequence[int]) -> list[list[int]]:
    """`IoTieIdsSlots`' arguments for a batch of (cell, `IoTieRequestFrame` output bits, nf)."""
    return [[HC._bytes_bits(c) for c in cells], [list(f) for f in frames], [HC._int_bits(x, 32) for x in nfs]]


def arguments(shape: Shape, inst: Instance, route: Sequence[int], sort: Sequence[int]) -> list[list[int]]:
    """`IoTieRequestIds`' arguments in its signature's order, each its leaves' bits: IoTieRequest's (the row and the
    lanes), then the map's classes, the read rows (as the registered reads give them) and both routings."""
    sh, mp = shape, inst.map
    return [*Q.arguments(sh.request, inst.request), [b for k in mp.keys for b in HC._int_bits(k, sh.CB)],
            [b for k in mp.ekeys for b in HC._int_bits(k, sh.CB)], [b for r in inst.port for b in HC._bytes_bits(r)],
            list(route), list(sort)]


__all__ = ["INPUT_IDS", "Instance", "IoTieIdsChain", "IoTieIdsFirst", "IoTieIdsKeys", "IoTieIdsLink", "IoTieIdsSlots",
           "IoTieRequestIds", "Map", "Row", "SLOT", "Shape", "arguments", "bucket_map", "commit", "commit_rows", "entry",
           "fields_of", "instance", "matches", "port_reads", "reads_open", "reference", "root", "route_switches", "scan",
           "slots_arguments", "sort_switches", "unit_reference"]
