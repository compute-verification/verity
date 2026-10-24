"""The ingress-ids tie (`io_ids`): `IoTieRequest` with the prompt ids' values held to the input ids of the Commit's
step-input port, so the model computes on the prompt the user sent (REDTEAM phase 9 #1: IoTieRequest counts each
request's prompt fields and checks each is a u32, and the ids' values were tied to nothing).

**The Commit side** is the step-input port the served rows' tie reads (`served_tie`, vllm's served Commit): each step's
rows-sha512 stream starts with its input ids, P little-endian elements of IB bytes in the forward's token order (its
batch rows in order, row r's n_r tokens at offset Σ_{r' < r} n_{r'}), zero past its tokens, P the engine's token budget,
then its positions. So the ids are the step tree's first KI = ⌈P·IB / 4096⌉ chunks (`served_tie.ids`, the shared read).
A window's unit takes the opened steps' KI chunks as rows ``ids_<s>_<j>``, and the verifier holds each row's b‖c to leaf
j of step ``steps[s]``'s tree, and that tree's root through its step root to the run root (`served_tie.chunk_ok`,
`leaves`), natively, as the served tie holds its positions chunks.

**The map** (`window_map`, the verifier's, public in the named gap; at the target under the advice's hiding root).
The window's prompt fields in scan order are the view's: request carry_in.q's from carry_in.fields, then each ended
request's, then carry_out.q's up to carry_out.fields; their cells are private (which cells hold request frames is the
warden's timing). Field f is position p of request q, and its class is that of the position whose computation q's KV
holds there, its root: (q, p), or for a prefix-cache hit (p below q's hit) its source's root at p, so a cached
position shares a class with the computation it reads. `keys[f]` is field f's class, numbered from 0 in order of
first appearance, the sink 2^CB − 1 past the window's fields. The opened steps are every step with a token whose
root is a class of the window, and `ekeys[e]` is opened entry e's class, or the sink. The verifier refuses an advice
whose class has no computation in any step.

**What 1 means** (`reference`): IoTieRequest's output on the cells and the view, and the window's fields (`scan`) and
the opened entries hold one value a class, each class with a field, an entry's value its whole IB-byte element. With
the native check of the opened chunks, every computation of a prompt position of the window, and every KV read in its
place, is the id the user sent there.

**The unit** (`IoTieRequestIds{B, T, R, OWN, P, S, IB}`) is IoTieRequest's scan, the cells hashed once, plus:
- each opened step's ids (`IoTieIdsPort{P, IB, CB}`): entry t's id is the low 32 bits of its element, and an entry
  whose class isn't the sink has the element's other bits zero;
- each request cell's slots (`IoTieIdsSlots{B}`): slot k is a field when the frame's count of fields is above k, its
  value the u32 its 10 characters spell, its ordinal nf + k, nf the window's fields before the cell;
- the classes routed to the fields (`io_tie.IoTieRoute` over M1 = 2^D1 ≥ NK slot positions, NK = (N − OWN)·K): record
  f is (keys[f], f), and `IoTieIdsKeys` holds each field to the record routed to it, with index its ordinal and a class
  not the sink, every other slot the sink;
- the members sorted by class (`IoTieRoute` over M2 = 2^D2 ≥ NK + E): each slot (class, 0, value), each opened entry
  (ekeys[e], 1, its id), the rest the sink; `IoTieIdsChain` checks the classes non-decreasing, the values equal within
  a class but the sink, and each class's first member a slot. A Beneš network is a permutation under any switches, so
  every member is in the sorted order and the switches only sort.
Public: IoTieRequest's, the steps, `keys` and `ekeys`. Witness: IoTieRequest's, the opened chunks and both routings'
switches.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Mapping, NamedTuple, Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import ZERO, add, and_reduce, const_bits, or_reduce
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.types import Array
from verity.ml.commitments.gates.frame_v3 import _part
from verity.core.protocols.certifier import head_check as HC
from verity.core.protocols.certifier import io_request as Q
from verity.core.protocols.certifier import io_tie as IT
from verity.core.protocols.certifier import served_tie as ST
from verity.core.protocols.certifier.active import HEADER, ID_DIGITS, ID_FIELD_BYTES

BIT, BYTE = HC.BIT, HC.BYTE
CHUNK = ST.CHUNK
#: a slot's record out of `IoTieIdsSlots`: field, ordinal, value
SLOT = 1 + 32 + 32


@dataclass(frozen=True)
class Shape:
    """An ingress link-window (`io_request.Shape`'s B, T, R, OWN) and its Commit port: S opened steps of P input ids of
    IB bytes (E = S·P entries), each step's KI chunks. Each of the N − OWN request cells has K slots, NK in all;
    classes are CB-bit, below NK, the sink 2^CB − 1; the classes are routed over M1 = 2^D1 ≥ NK positions and the
    members sorted over M2 = 2^D2 ≥ NK + E."""

    B: int
    T: int
    R: int
    OWN: int
    P: int
    S: int
    IB: int

    def __post_init__(self) -> None:
        Q.Shape(self.B, self.T, self.R, self.OWN)
        for name, least, most in (("P", 1, 1 << 28), ("S", 1, 1 << 32), ("IB", 4, 8)):
            if type(getattr(self, name)) is not int or not least <= getattr(self, name) <= most:
                raise ValueError(f"a Commit port has {least} <= {name} <= {most}, not {getattr(self, name)!r}")
        if self.NK + self.E > 1 << 32:
            raise ValueError(f"a window has at most 2^32 slots and entries, not {self.NK} and {self.E}")

    @property
    def request(self) -> Q.Shape:
        return Q.Shape(self.B, self.T, self.R, self.OWN)

    @property
    def K(self) -> int:
        return Q.layout(self.B).room // ID_FIELD_BYTES

    @property
    def NK(self) -> int:
        return (self.T * self.R - self.OWN) * self.K

    @property
    def E(self) -> int:
        return self.S * self.P

    @property
    def KI(self) -> int:
        return -(-self.P * self.IB // CHUNK)

    @property
    def port(self) -> SimpleNamespace:
        """The step-input port's statics as `served_tie` reads them (its T the token budget P)."""
        return SimpleNamespace(T=self.P, IB=self.IB, PB=8)

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


# --- the fake upstream: the steps' input ports -----------------------------------------------------------------------

class Row(NamedTuple):
    """A step's batch row (a served row, reduced): request q (its arrival ordinal), its first position and its tokens
    this step."""

    q: int
    start: int
    n: int


def ports(shape: Shape, steps: Sequence[Sequence[Row]], tokens: Sequence[Sequence[int]]) -> list[list[bytes]]:
    """Each step's step-input port as its stream starts (`served_tie.port_chunks`): its rows' input ids in batch order,
    request q's position p being tokens[q][p] (its prompt, then its outputs), then their positions."""
    return [ST.port_chunks(shape.port, [tokens[r.q][r.start + t] for r in rows for t in range(r.n)],
                           [r.start + t for r in rows for t in range(r.n)]) for rows in steps]


# --- the verifier's map ----------------------------------------------------------------------------------------------

class Map(NamedTuple):
    """A window's map: each field's class in scan order (the sink past them), the opened steps, and each opened
    entry's class (the sink for an entry no class of the window reads)."""

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


def fields_of(view: Q.RequestView) -> list[tuple[int, int]]:
    """The window's prompt fields in scan order, each (request, position), from the view."""
    q, p, out = view.carry_in.q, view.carry_in.fields, []
    for n in view.ended:
        out += [(q, i) for i in range(p, n)]
        q, p = q + 1, 0
    return out + [(q, i) for i in range(p, view.carry_out.fields)]


def window_map(shape: Shape, prompt_lens: Sequence[int], view: Q.RequestView, steps: Sequence[Sequence[Row]],
               sources: Mapping[int, tuple[int, int]]) -> Map:
    """The verifier's map of an ingress window (the module docstring) from the advice's prompt lengths (arrival order),
    its steps' rows and its prefix-cache sources, and the window's view. Raises `ValueError` on more fields than NK, a
    step past P tokens, a class no step computes, or more opened steps than S."""
    sh, cls, keys = shape, {}, []
    for q, p in fields_of(view):
        keys.append(cls.setdefault(root(q, p, sources), len(cls)))
    if len(keys) > sh.NK:
        raise ValueError(f"the window has {len(keys)} prompt fields, more than NK = {sh.NK}")
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
    opened = sorted(reads)
    if len(opened) > sh.S:
        raise ValueError(f"the window's fields are computed in {len(opened)} steps, more than S = {sh.S}")
    ekeys = [sh.sink] * sh.E
    for k, s in enumerate(opened):
        for off, c in reads[s]:
            ekeys[k * sh.P + off] = c
    return Map(tuple(keys) + (sh.sink,) * (sh.NK - len(keys)),
               tuple(opened) + (opened[-1],) * (sh.S - len(opened)), tuple(ekeys))


# --- the reference ---------------------------------------------------------------------------------------------------

class Instance(NamedTuple):
    """One ingress link-window's ids tie: IoTieRequest's instance, the map, and each opened step's KI ids chunks."""

    request: Q.Instance
    map: Map
    chunks: tuple[tuple[bytes, ...], ...]


def scan(shape: Shape, cells: Sequence[bytes], carry: Q.Carry, window: int = 0
         ) -> tuple[list[int], list[int]] | None:
    """The window's prompt fields' values in scan order and each one's slot position ((cell − OWN)·K + slot), or None
    where `io_request.cut` refuses the cells."""
    rq, lay, pad = shape.request, Q.layout(shape.B), IT.pad_cell(shape.B)
    if Q.cut(rq, cells, carry, window) is None:
        return None
    vals, at, left = [], [], carry.left
    for i, c in enumerate(cells):
        if (rq.OWN and i == 0) or c == pad:
            continue
        p = Q.parse_frame(lay, c[HEADER.size:], not left, left)
        base = HEADER.size + (lay.start if not left else 0)
        for k in range(p.fields):
            o = base + ID_FIELD_BYTES * k + 1
            vals.append(int(c[o:o + ID_DIGITS]))
            at.append((i - rq.OWN) * shape.K + k)
        left = (left if p.length is None else p.length) - p.used
    return vals, at


def matches(shape: Shape, vals: Sequence[int], mp: Map, chunks: Sequence[Sequence[bytes]]) -> bool:
    """The fields and the opened entries hold one value a class, each class with a field: field f's class is keys[f],
    not the sink, and each entry with a class other than the sink is its class's value (`IoTieIdsChain`'s reference),
    its whole element (`IoTieIdsPort`'s high bits)."""
    sh = shape
    if len(vals) > sh.NK or len(mp.keys) != sh.NK or len(mp.ekeys) != sh.E:
        return False
    held: dict[int, int] = {}
    for k, v in zip(mp.keys, vals):
        if k == sh.sink or held.setdefault(k, v) != v:
            return False
    return all(k == sh.sink or held.get(k) == entry(sh, chunks, e) for e, k in enumerate(mp.ekeys))


def entry(shape: Shape, chunks: Sequence[Sequence[bytes]], e: int) -> int:
    """Opened entry e: step ⌊e / P⌋'s (e mod P)-th input id, its whole IB-byte element."""
    s, t = divmod(e, shape.P)
    return int.from_bytes(b"".join(chunks[s])[shape.IB * t:shape.IB * (t + 1)], "little")


def reference(shape: Shape, inst: Instance) -> int:
    """`IoTieRequestIds`' output (the module docstring, "What 1 means")."""
    sh, r, mp = shape, inst.request, inst.map
    if len(inst.chunks) != sh.S or len(mp.steps) != sh.S or \
            any(len(c) != sh.KI or any(len(x) != CHUNK for x in c) for c in inst.chunks):
        raise ValueError(f"an instance has S = {sh.S} steps of KI = {sh.KI} {CHUNK}-byte chunks")
    if not Q.reference(sh.request, r):
        return 0
    got = scan(sh, r.cells, r.view.carry_in, r.window)
    return int(got is not None and matches(sh, got[0], mp, inst.chunks))


def instance(shape: Shape, request: Q.Instance, mp: Map, ports_: Sequence[Sequence[bytes]]) -> Instance:
    """The prover's instance: the map's steps' ids chunks, each step's port (`ports`) as its stream starts."""
    return Instance(request, mp, tuple(tuple(ports_[s][:shape.KI]) for s in mp.steps))


def leaves(shape: Shape, mp: Map) -> list[tuple[str, int, int]]:
    """The verifier's native check of the port: each unit row (its name), whose b‖c is to be leaf j of step s's tree
    (`served_tie.chunk_ok`), as (row, s, j)."""
    return [(f"ids_{k}_{j}", s, j) for k, s in enumerate(mp.steps) for j in range(shape.KI)]


# --- the prover's routings -------------------------------------------------------------------------------------------

def _fill(perm: list[int]) -> list[int]:
    taken = {p for p in perm if p >= 0}
    rest = iter(i for i in range(len(perm)) if i not in taken)
    return [p if p >= 0 else next(rest) for p in perm]


def route_switches(shape: Shape, inst: Instance) -> list[int]:
    """The classes' routing: record f to the slot position of the window's field f, the other positions in order."""
    r = inst.request
    got = scan(shape, r.cells, r.view.carry_in, r.window)
    perm = [-1] * (1 << shape.D1)
    for f, s in enumerate(got[1] if got is not None else ()):
        perm[f] = s
    return IT.benes(_fill(perm))


def sort_switches(shape: Shape, inst: Instance) -> list[int]:
    """The members' sort as the unit forms them: each slot position its field's class (the sink but at a field), each
    entry its class, by (class, slot before entry, position)."""
    sh, r = shape, inst.request
    got = scan(sh, r.cells, r.view.carry_in, r.window)
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
    "and nf, the window's prompt fields before it: nf' is nf plus the frame's fields at a request frame; slot k is a "
    "field iff the cell is a request frame with more than k fields, its ordinal nf + k, its value the u32 its 10 "
    "characters spell at its place in a first frame's (start) or a continuation's reading; u32s least significant "
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


def _sig_chain(S):
    if type(S.CB) is not int or not 1 <= S.CB <= 32 or type(S.D2) is not int or not 1 <= S.D2 <= 32:
        raise ValueError(f"a chain has 1 <= CB <= 32 and 1 <= D2 <= 32, not CB={S.CB!r}, D2={S.D2!r}")
    return (("members", Array(1 << S.D2, Array(S.CB + 33, BIT))),), BIT


def _build_chain(C, S, *ms):
    def parts(m):
        return m[:S.CB], m[S.CB], m[S.CB + 1:]

    key, kind, _ = parts(ms[0])
    oks = [C.AND(kind, and_reduce(C, key).inv()).inv()]
    for t in range(1, len(ms)):
        (ka, _, va), (kb, kind, vb) = parts(ms[t - 1]), parts(ms[t])
        same, sink = IT._eq(C, ka, kb), and_reduce(C, kb)
        oks += [IT._lt(C, kb, ka).inv(), C.AND(C.AND(same, sink.inv()), IT._eq(C, va, vb).inv()).inv(),
                C.AND(C.AND(same.inv(), sink.inv()), kind).inv()]
    return [and_reduce(C, oks)]


IoTieIdsChain = _part(
    "IoTieIdsChain", ("CB", "D2"), _sig_chain, _build_chain,
    "1 iff the 2^D2 members (class, kind, value) are sorted by class, CB bits, each member with its predecessor's "
    "class has its value unless the class is the sink (2^CB − 1), and each member that starts a class other than the "
    "sink is a slot (kind 0)")


def _port_shape(S) -> SimpleNamespace:
    if not all(type(x) is int for x in (S.P, S.IB, S.CB)) or not (1 <= S.P <= 1 << 28 and 4 <= S.IB <= 8 and
                                                                   1 <= S.CB <= 32):
        raise ValueError(f"a port has 1 <= P <= 2^28, 4 <= IB <= 8 and 1 <= CB <= 32, not P={S.P!r}, IB={S.IB!r}, "
                         f"CB={S.CB!r}")
    return SimpleNamespace(T=S.P, IB=S.IB, KI=-(-S.P * S.IB // CHUNK))


def _sig_port(S):
    pt = _port_shape(S)
    return (("ekeys", Array(S.P, Array(S.CB, BIT))),
            *((f"ids_{j}", Array(8 * CHUNK, BIT)) for j in range(pt.KI))), Array(1 + 32 * S.P, BIT)


def _build_port(C, S, *e):
    pt = _port_shape(S)
    ekeys, oks, out = e[:S.P], [], []
    for k, x in zip(ekeys, ST.ids(pt, e[S.P:])):
        oks.append(C.AND(and_reduce(C, k).inv(), or_reduce(C, x[32:])).inv())
        out += x[:32]
    return [[and_reduce(C, oks), *out]]


IoTieIdsPort = _part(
    "IoTieIdsPort", ("P", "IB", "CB"), _sig_port, _build_port,
    "(ok, each token's id) of a step's ids chunks, the step-input port's first ⌈P·IB / 4096⌉ (`served_tie.ids`) and "
    "the P tokens' classes: token t's id is the low 32 bits of its IB-byte little-endian element; ok iff every token "
    "whose class isn't the sink (2^CB − 1) has its element's other bits zero; ids least significant bit first")


def _sig_ids(S):
    sh = Shape(S.B, S.T, S.R, S.OWN, S.P, S.S, S.IB)
    request, _ = Q._sig_request(sh.request)
    ids = tuple((f"ids_{s}_{j}", Array(8 * CHUNK, BIT)) for s in range(S.S) for j in range(sh.KI))
    return ((*request, ("keys", Array(sh.NK, Array(sh.CB, BIT))), ("ekeys", Array(sh.E, Array(sh.CB, BIT))), *ids,
             ("route", Array(sh.route, BIT)), ("sort", Array(sh.sort, BIT))), BIT)


def _body_ids(B, S, root_, dom, cells, salts, *rest):
    sh = Shape(S.B, S.T, S.R, S.OWN, S.P, S.S, S.IB)
    rq, CB, n = sh.request, sh.CB, S.S * sh.KI
    head, (keys, ekeys), chunks, (route, sort) = rest[:-n - 4], rest[-n - 4:-n - 2], rest[-n - 2:-2], rest[-2:]
    window, (left, fields, ended, count, received, left_out, fields_out, switches_) = \
        (head[0], head[1:]) if S.OWN else (None, head)
    K, cr, er = HC._Consts(B), HC._refs(cells), HC._refs(ended)
    row = 8 * S.B * S.R
    oks = [B.call(bind(IT.NetworkRowsRoot, N_BITS=row, COUNT=S.T), cells.reshape(Array(S.T, Array(row, BIT))), salts,
                  root_, dom)]
    xs = []
    for k in range(rq.M):
        xs += (er[32 * k:32 * (k + 1)] if k < rq.N else K.bits(0, 32)) + K.bits(k, rq.D)
    routed = HC._refs(B.call(bind(IT.IoTieRoute, W=rq.W, D=rq.D), HC._coll(Array(rq.M, Array(rq.W, BIT)), xs),
                             switches_))

    def cell(i: int):
        return HC._coll(Array(S.B, BYTE), cr[8 * S.B * i:8 * S.B * (i + 1)])

    if S.OWN:
        oks.append(B.call(bind(Q.IoTieRequestOwn, B=S.B), cell(0), window))
    state = HC._coll(Array(Q.STATE, BIT), HC._refs(left) + HC._refs(fields) + K.bits(0, Q.STATE - 65) + K.bits(1, 1))
    nf, slots = HC._coll(Array(32, BIT), K.bits(0, 32)), []
    for i in range(S.OWN, rq.N):
        fo = B.call(bind(Q.IoTieRequestFrame, B=S.B), cell(i), HC._coll(Array(32, BIT), HC._refs(state)[:32]))
        so = HC._refs(B.call(bind(IoTieIdsSlots, B=S.B), cell(i), fo, nf))
        nf, slots = HC._coll(Array(32, BIT), so[:32]), slots + so[32:]
        state = B.call(bind(Q.IoTieRequestStep, D=rq.D, NEW=int(i > 0 and i % S.R == 0)), state, fo,
                       HC._coll(Array(rq.W, BIT), routed[rq.W * i:rq.W * (i + 1)]))
    oks.append(B.call(bind(Q.IoTieRequestEnd), state, count, received, left_out, fields_out))
    kr, xs1 = HC._refs(keys), []
    for f in range(1 << sh.D1):
        xs1 += (kr[CB * f:CB * (f + 1)] if f < sh.NK else K.bits(sh.sink, CB)) + K.bits(f, sh.D1)
    routed1 = B.call(bind(IT.IoTieRoute, W=CB + sh.D1, D=sh.D1),
                     HC._coll(Array(1 << sh.D1, Array(CB + sh.D1, BIT)), xs1), route)
    ko = HC._refs(B.call(bind(IoTieIdsKeys, NK=sh.NK, CB=CB, D1=sh.D1), HC._coll(Array(sh.NK, Array(SLOT, BIT)), slots),
                         routed1))
    ek, members, flags = HC._refs(ekeys), [], [ko[0]]
    for s in range(sh.NK):
        members += ko[1 + CB * s:1 + CB * (s + 1)] + K.bits(0, 1) + slots[SLOT * s + 33:SLOT * (s + 1)]
    for s in range(S.S):
        po = HC._refs(B.call(bind(IoTieIdsPort, P=S.P, IB=S.IB, CB=CB),
                             HC._coll(Array(S.P, Array(CB, BIT)), ek[CB * S.P * s:CB * S.P * (s + 1)]),
                             *chunks[sh.KI * s:sh.KI * (s + 1)]))
        flags.append(po[0])
        for t in range(S.P):
            e = S.P * s + t
            members += ek[CB * e:CB * (e + 1)] + K.bits(1, 1) + po[1 + 32 * t:1 + 32 * (t + 1)]
    for _ in range(sh.NK + sh.E, 1 << sh.D2):
        members += K.bits(sh.sink, CB) + K.bits(1, 1) + K.bits(0, 32)
    ordered = B.call(bind(IT.IoTieRoute, W=CB + 33, D=sh.D2), HC._coll(Array(1 << sh.D2, Array(CB + 33, BIT)), members),
                     sort)
    oks.append(B.call(bind(IoTieIdsChain, CB=CB, D2=sh.D2), ordered))
    bits = [r for o in oks for r in HC._refs(o)] + flags
    return B.call(BL.and_gate(len(bits)), HC._coll(Array(len(bits), BIT), bits))


IoTieRequestIds = CompositeDefinition(
    "IoTieRequestIds", 1, ("B", "T", "R", "OWN", "P", "S", "IB"), _sig_ids, _body_ids,
    conformance="exact-model-tested",
    doc="1 iff IoTieRequest's checks hold on the cells and the view (its rows, own frame, scan and totals), and the "
        "window's prompt fields (`IoTieIdsSlots`), each given its class by the routed records (`IoTieRoute`, "
        "`IoTieIdsKeys`), and the S opened steps' input ids (`IoTieIdsPort`, each step's P ids of IB bytes from its "
        "step-input port's chunks, an id of a class its whole element), sorted by class by the switches "
        "(`IoTieRoute`), hold one value a class, each class starting with a field (`IoTieIdsChain`): the ingress-ids "
        "tie of one link-window, every cell opened; the chunks' b‖c are the verifier's to hold to the Commit (`leaves`)")


# --- arguments -------------------------------------------------------------------------------------------------------

def slots_arguments(cells: Sequence[bytes], frames: Sequence[Sequence[int]], nfs: Sequence[int]) -> list[list[int]]:
    """`IoTieIdsSlots`' arguments for a batch of (cell, `IoTieRequestFrame` output bits, nf)."""
    return [[HC._bytes_bits(c) for c in cells], [list(f) for f in frames], [HC._int_bits(x, 32) for x in nfs]]


def arguments(shape: Shape, inst: Instance, bits: Sequence[int], route: Sequence[int], sort: Sequence[int]
              ) -> list[list[int]]:
    """`IoTieRequestIds`' arguments in its signature's order, each its leaves' bits: IoTieRequest's (with its switches
    `bits`), then the map's classes, the opened steps' ids chunks, and both routings."""
    sh, mp = shape, inst.map
    return [*Q.arguments(sh.request, inst.request, bits), [b for k in mp.keys for b in HC._int_bits(k, sh.CB)],
            [b for k in mp.ekeys for b in HC._int_bits(k, sh.CB)],
            *[HC._bytes_bits(c) for step in inst.chunks for c in step], list(route), list(sort)]


def port_arguments(ekeys: Sequence[Sequence[int]], chunks: Sequence[Sequence[bytes]], CB: int) -> list[list[int]]:
    """`IoTieIdsPort`'s arguments for a batch of (a step's P classes, its KI ids chunks)."""
    return [[[b for k in ks for b in HC._int_bits(k, CB)] for ks in ekeys],
            *[[HC._bytes_bits(cs[j]) for cs in chunks] for j in range(len(chunks[0]))]]


__all__ = ["CHUNK", "Instance", "IoTieIdsChain", "IoTieIdsKeys", "IoTieIdsPort", "IoTieIdsSlots", "IoTieRequestIds",
           "Map", "Row", "SLOT", "Shape", "arguments", "entry", "fields_of", "instance", "leaves", "matches",
           "port_arguments", "ports", "reference", "root", "route_switches", "scan", "slots_arguments", "sort_switches",
           "window_map"]
