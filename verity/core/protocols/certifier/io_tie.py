"""The I/O tie (`PROTOCOL.md`, "The I/O tie"), a prototype on a fake upstream: the reference of one egress
link-window's tie, a unit a bucket row that opens every cell of it, and the Boolean Definitions it is built from.
Nothing here is in a Spec or proved. The cells are the warden's (`warden`, its rows as `warden_rows` registers them,
registration v2); no Commit writes the port yet, so the fake upstream commits it as the design has the Commit commit
it, under `hm96-sha512/row/v2` rows in a `frame-v3-sha512` tree.

**The cells are the warden's.** A link-window's egress is T rows, row t bucket t's R = lanes·r slots of B bytes, so
cell i is slot i mod R of bucket ⌊i / R⌋. A bucket's live connections hold r slots each, ranked by ordinal (slots
ρ·r … ρ·r + r − 1 for rank ρ), and a slot no connection holds is B zero bytes (`pad_cell(B)`, a `PAD` on stream 0).
Connection q, admitted at bucket s, has a `PAD` on q in each of its slots before s + D (`pad_cell(B, q)`); then its
egress cell i is in bucket s + D + ⌊i / r⌋, its slot i mod r: its n events (`data_cell`) and its `END`
(`end_cell`), and the rest of that bucket a `PAD` on q; it holds no slot after it. A `DATA` cell is the warden's cut
of the server's bytes, P = B − 9 of them, and a response is its events, each `id_event` padded with spaces to P, then
`TRAILER`, shorter than P: so the warden's cut (`warden.Warden._egress`) is exactly n `DATA` cells, one event each,
and an `END` that holds `TRAILER`, whenever the bytes are there by their deadlines. q is the request's ingress ordinal
on the wire, from 1 as `active.Requests` numbers it, which the warden writes from the connection it terminated.

**The map is a function of the advice** (`window_map`): for each slot, the cell the warden must have released
there, as a record (q, e): a `DATA` on q whose id is entry e of its bucket's port, an `END` on q (`END_ENTRY`) or a
`PAD` on q (`EMPTY`), q = 0 a slot no connection holds. Advice request i is on stream order[i] + 1 (order[i] its
arrival order from 0: advice v1's order permutation, which carries and charges how the engine's numbering departs
from the warden's; v0's is the identity) and was admitted at bucket admitted[i], which the warden's FIFO makes
non-decreasing in arrival order; its output k must be released at admitted[i] + D + ⌊k / r⌋, so advice v0's release
is held to the schedule. No prover value says where a cell goes. The advice is private; here the verifier computes
the map in the clear from the opened advice and the admissions (which the warden's wiring gives, opened to the
verifier for now), the named gap ("the advice is opened to the verifier until the compile unit exists"). The target
reads the map inside the proof under the advice's hiding root.

**The port is the Commit's**: its sampled ids, one hiding row a step (`port_row`: the step's P ids, u32be in
batch-row order, zero past its batch) in a dense frame-v3-sha512 tree of the session's steps, whose root the Commit
registers (`verity.ml.commitments.advice`'s registration: `hm96-sha512/row/v2` leaves under `domain`). Bucket t's
unit reads S rows (`Shape.S`), at the steps its bucket's outputs were sampled in (`window_map`'s steps, public with the
map), and its entry e = σ·P + r is read row σ's r-th id. The real port is vLLM's Commit (its binding map's
`sampled_token_ids` under the run root), which this fake upstream stands in for.

**What 1 means** (`reference`): every bucket row is row t of the warden's registered root, and every port row bucket t
reads is its step's row of the Commit's registered root (the verifier's registered reads, outside the unit; here
`rows_root` and the port's openings), and each bucket's unit is 1 (`bucket_reference`): every cell is its map's
(`cells_match`): the record's kind on the record's stream, a `DATA` cell's id port entry e. So a duplicated, dropped,
reordered, moved or altered cell, a covert byte, a cell on another stream or in another connection's slots, a
fabricated output, an early, late or missing `END`, a `FAIL`, a wrong id and a forged port row each make it 0.

    IoTieCell{B}(cell: Array<B, Byte>) -> Array<67>        ok, data, end, stream (u32), id (u32), LSB first
    IoTieSwitch{W}(a: Array<W>, b: Array<W>, s) -> Array<2, Array<W>>    (a, b), swapped iff s: W ANDs
    IoTieRoute{W, D}(xs: Array<M, Array<W>>, switches: Array<S_W>) -> Array<M, Array<W>>
        a Beneš network of S_W = (2D − 1)·2^(D − 1) switches: a column of 2^(D − 1) `IoTieSwitch`es, two
        `IoTieRoute{W, D − 1}`, a column, so a binding is built once whatever its uses
    IoTieMatch{N, D, E}(cells: Array<N, Array<67>>, map: Array<N, Array<8, Byte>>, routed: Array<M, Array<32 + D>>)
        -> Value<1>
    IoTieContent{B, R, P, S}(cells: Array<R, Array<B, Byte>>, map: Array<R, Array<8, Byte>>,
        port: Array<S, Array<4P, Byte>>, switches: Array<S_W>) -> Value<1>
    IoTieBucket{B, R, P, S}(row: Array<8·R·B>, map, port: Array<S, Array<4P, Byte>>, switches) -> Value<1>
        E = S·P port entries a bucket, M = 2^D ≥ max(R, E)

`IoTieBucket` is the unit, one a bucket row. Its `row` is the warden's registered row t and its `port` S rows of the
Commit's registered port, which the proof system reads (the Lean verifier's registered reads, `flock-verify
--registered`: unit t reads row t by `warden_rows.bucket_port`, and its `port[s]` its bucket's s-th step by
`port_reads`), so no gate of it hashes. It holds every cell to the map (`IoTieContent`): the port's E entries, each its
id and its index e, routed to the slots by the prover's switches; any setting is a permutation, and `IoTieMatch` holds
each `DATA` cell's id to the entry the map names, by its index. A link-window's tie is its T units. Public: the
warden's root and domain id (its registration), the port's root and domain id (the Commit's), and, until the named gap
closes, the steps (the port's read positions) and the map. Witness: the rows and the read port rows (each with its salt
and path, the registered reads'), and the switches.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Any, NamedTuple, Sequence

from verity.ml.circuits.boolean.forms import ONE, ZERO, and_reduce
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.types import Array
from verity.ml.commitments import hm96
from verity.ml.commitments.gates.frame_v3 import _part, leaf_ref, node_ref, opens
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import _PAD, CommitmentDomain, frame_hash, _uint
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as SCHEMA
from verity.core.protocols.certifier import head_check as HC
from verity.core.protocols.certifier.active import DATA, EVENT_END, EVENT_HEAD, HEADER, PAD
from verity.core.protocols.sampled_proofs.one_stage import produced
from verity.core.protocols.sampled_proofs.one_stage import registered as G

BIT, BYTE = HC.BIT, HC.BYTE
ID_DIGITS = 10
ID_EVENT_HEAD = EVENT_HEAD[:EVENT_HEAD.index(b'"delta"')] + b'"token_id":'
ID_EVENT_TAIL = b',"finish_reason":null}]}' + EVENT_END
ID_EVENT_BYTES = len(ID_EVENT_HEAD) + ID_DIGITS + len(ID_EVENT_TAIL)
#: the id's first character in a cell: the header, then the event's head
ID_AT = HEADER.size + len(ID_EVENT_HEAD)
#: the least cell: the header and one event, which a `DATA` cell holds with no padding
CELL_USED = HEADER.size + ID_EVENT_BYTES
#: the warden's `END` kind (`warden.END`), beside `active`'s `PAD` and `DATA`
END = 0x20
#: a response's last bytes, after its events: the warden's `END` holds them
TRAILER = b"data: [DONE]" + EVENT_END
#: `IoTieCell`'s output: ok, data, end, the stream and the id
CELL_OUT = 3 + 32 + 32
#: a map record: the stream q and the port entry e, u32be each; e is `EMPTY` for a `PAD` on q and `END_ENTRY` for
#: its `END`, and q is 0 for a slot no connection holds
MAP = struct.Struct(">II")
EMPTY = 0xFFFF_FFFF
END_ENTRY = EMPTY - 1
#: the unit's port parameter: S whole rows of the Commit's port, row s a registered read at its own key `port[s]`
PORT = "port"
#: the registered value the egress port reads, the Commit's sampled ids
SAMPLED = "sampled_token_ids"


# --- the cells -------------------------------------------------------------------------------------------------------

def id_event(token_id: int, size: int = ID_EVENT_BYTES) -> bytes:
    """The canonical event of one token id in `size` bytes: `ID_EVENT_HEAD`, the id right-aligned in 10 decimal
    characters, then `ID_EVENT_TAIL` with spaces (JSON whitespace) ahead of its blank line to fill `size`."""
    if type(token_id) is not int or not 0 <= token_id < 1 << 32:
        raise ValueError(f"a token id is a u32, not {token_id!r}")
    if type(size) is not int or size < ID_EVENT_BYTES:
        raise ValueError(f"an id event is at least {ID_EVENT_BYTES} bytes, not {size!r}")
    tail = ID_EVENT_TAIL[:-len(EVENT_END)] + b" " * (size - ID_EVENT_BYTES) + EVENT_END
    return ID_EVENT_HEAD + b"%10d" % token_id + tail


def _check_cell_bytes(cell_bytes: object) -> None:
    if type(cell_bytes) is not int or cell_bytes < CELL_USED:
        raise ValueError(f"a cell has at least {CELL_USED} bytes, room for an id event, not {cell_bytes!r}")


def _check_stream(stream: object) -> None:
    if type(stream) is not int or not 0 <= stream < 1 << 32:
        raise ValueError(f"a stream is a u32, not {stream!r}")


def data_cell(token_id: int, stream: int, cell_bytes: int) -> bytes:
    """A `DATA` cell on the request's stream: the warden's P = B − 9 bytes of the response, its id's event."""
    _check_cell_bytes(cell_bytes)
    _check_stream(stream)
    p = cell_bytes - HEADER.size
    return HEADER.pack(DATA, stream, p) + id_event(token_id, p)


def pad_cell(cell_bytes: int, stream: int = 0) -> bytes:
    """A `PAD` on the stream, no payload, zero fill: B zero bytes on stream 0, a slot no connection holds."""
    _check_cell_bytes(cell_bytes)
    _check_stream(stream)
    return HEADER.pack(PAD, stream, 0).ljust(cell_bytes, b"\0")


def end_cell(stream: int, cell_bytes: int) -> bytes:
    """A response's `END` on its stream: the rest of its bytes, `TRAILER`, and zero fill."""
    _check_cell_bytes(cell_bytes)
    _check_stream(stream)
    return (HEADER.pack(END, stream, len(TRAILER)) + TRAILER).ljust(cell_bytes, b"\0")


class Parsed(NamedTuple):
    """A cell's reading: its kind (`PAD`, `DATA` or `END`), its stream, and a `DATA` cell's id (0 otherwise)."""

    kind: int
    stream: int
    id: int

    @property
    def data(self) -> bool:
        return self.kind == DATA


def parse_cell(cell: bytes) -> Parsed | None:
    """The cell's one reading, or None for a cell that is none of the three: `IoTieCell`'s reference."""
    if len(cell) < CELL_USED:
        return None
    kind, stream, _ = HEADER.unpack_from(cell)
    if kind in (PAD, END):
        return Parsed(kind, stream, 0) if cell == (pad_cell(len(cell), stream) if kind == PAD else
                                                   end_cell(stream, len(cell))) else None
    digits = cell[ID_AT:ID_AT + ID_DIGITS]
    if kind != DATA or not digits.lstrip(b" ").isdigit():
        return None
    token_id = int(digits)
    return Parsed(DATA, stream, token_id) if token_id < 1 << 32 and cell == data_cell(token_id, stream, len(cell)) else None


def expected_cell(record: tuple[int, int], cell_bytes: int, port_id: int | None = None) -> bytes:
    """The one cell a map record admits: a `PAD` or `END` on q, or a `DATA` on q with the port's id at its entry."""
    q, e = record
    if e == EMPTY:
        return pad_cell(cell_bytes, q)
    if e == END_ENTRY:
        return end_cell(q, cell_bytes)
    if port_id is None:
        raise ValueError("a data record's cell needs its port id")
    return data_cell(port_id, q, cell_bytes)


# --- the window, its map and its port --------------------------------------------------------------------------------

def route_depth(cells: int, entries: int) -> int:
    """D of the routing over M = 2^D ≥ max(cells, entries) positions."""
    return max(1, (max(cells, entries) - 1).bit_length())


@dataclass(frozen=True)
class Shape:
    """A link-window's egress, T ≥ 2 buckets of R slots of B-byte cells (N = T·R cells, the warden's T rows of R =
    lanes·r slots), and each bucket's port: S read rows of P ids each (E = S·P entries) from a port tree of depth
    DS. A bucket's routing is over M = 2^D ≥ max(R, E) positions."""

    B: int
    T: int
    R: int
    P: int
    S: int
    DS: int

    def __post_init__(self) -> None:
        _check_cell_bytes(self.B)
        for name, least in (("T", 2), ("R", 1), ("P", 1), ("S", 1), ("DS", 1)):
            if type(getattr(self, name)) is not int or getattr(self, name) < least:
                raise ValueError(f"{name} is an int of at least {least}, not {getattr(self, name)!r}")
        if self.N > 1 << 32 or self.E >= END_ENTRY or self.DS > 32:
            raise ValueError(f"a window has at most 2^32 cells, a bucket fewer than 2^32 − 2 port entries and a port of "
                             f"depth at most 32, not N = {self.N}, E = {self.E}, DS = {self.DS}")

    @property
    def N(self) -> int:
        return self.T * self.R

    @property
    def E(self) -> int:
        return self.S * self.P

    @property
    def D(self) -> int:
        return route_depth(self.R, self.E)

    @property
    def M(self) -> int:
        return 1 << self.D

    @property
    def W(self) -> int:
        """A routed entry's bits: its id and its index."""
        return 32 + self.D

    @property
    def switches(self) -> int:
        return (2 * self.D - 1) << (self.D - 1)

    def bucket(self, i: int) -> int:
        return i // self.R


class Output(NamedTuple):
    """Advice v0's kept output (`advice.Output`): step `step`'s batch row `row` sampled request `request`'s output
    `index`."""

    step: int
    row: int
    request: int
    index: int


class MapEntry(NamedTuple):
    """A slot's record in the map: its stream q, the request's ingress ordinal on the wire (order[i] + 1, or 0 for a
    slot no connection holds), and e: a `DATA` cell's port entry, `END_ENTRY` for the `END`, `EMPTY` for a `PAD`."""

    q: int
    e: int


#: a slot no connection holds: B zero bytes, a `PAD` on stream 0
UNHELD = MapEntry(0, EMPTY)


def _count(name: str, value: object, least: int) -> int:
    if type(value) is not int or value < least:
        raise ValueError(f"{name} is an int of at least {least}, not {value!r}")
    return value


def window_map(shape: Shape, outputs: Sequence[Output], release: Sequence[int], window: int, admitted: Sequence[int],
               delay: int, slots: int, order: Sequence[int] | None = None
               ) -> tuple[tuple[MapEntry, ...], tuple[tuple[int, ...], ...]]:
    """The verifier's map of egress window `window` (buckets window·T … window·T + T − 1), from the advice's outputs
    (`advice.outputs(a)`) and release buckets (`a.release`), each request's admission bucket `admitted[i]`, and the
    warden's D = `delay` and r = `slots`; with each bucket's S steps, whose port rows its unit opens. Advice request
    i is on stream order[i] + 1, its ingress ordinal on the wire (order[i] its arrival order from 0, advice v1's
    order permutation; None is v0's identity). Its output k must be released at admitted[i] + D + ⌊k / r⌋, and its
    `END` is its cell n_i, so it holds r slots of every bucket from admitted[i] to admitted[i] + D + ⌊n_i / r⌋, at
    rank ρ among that bucket's live requests by stream: slot ρ·r + j is a `PAD` before admitted[i] + D, then its cell
    (b − admitted[i] − D)·r + j: an output, with its entry σ·P + row (σ its step's place among the bucket's steps),
    the `END`, or a `PAD`. A slot past the live requests' is `UNHELD`. A bucket's steps are sorted and padded to S by
    repeating the last (step 0 for a bucket with no output)."""
    sh = shape
    D, r = _count("the delay", delay, 0), _count("a connection's slots", slots, 1)
    if sh.R % r:
        raise ValueError(f"a bucket's R = {sh.R} slots are lanes of r = {r}")
    if len(outputs) != len(release):
        raise ValueError("one release bucket an output")
    n_req = len(admitted)
    order = range(n_req) if order is None else order
    if sorted(order) != list(range(n_req)) or any(not 0 <= o.request < n_req for o in outputs):
        raise ValueError("an order is a permutation of the advice's requests")
    if n_req >= 1 << 32:
        raise ValueError("a stream is a u32: fewer than 2^32 − 1 requests")
    arrival = sorted(range(n_req), key=lambda i: order[i])
    for a, b in zip(arrival, arrival[1:]):
        if _count("an admission bucket", admitted[b], 0) < _count("an admission bucket", admitted[a], 0):
            raise ValueError(f"the warden admits in arrival order: request {b} (stream {order[b] + 1}) is admitted at "
                             f"bucket {admitted[b]}, before request {a} (stream {order[a] + 1}) at {admitted[a]}")
    outs: list[dict[int, Output]] = [{} for _ in range(n_req)]
    for o, b in zip(outputs, release):
        want = admitted[o.request] + D + o.index // r
        if b != want:
            raise ValueError(f"request {o.request}'s output {o.index} is released at bucket {b}, not the warden's {want}")
        if o.index in outs[o.request] or not 0 <= o.row < sh.P:
            raise ValueError(f"an output is its request's once, at a batch row below P = {sh.P}, not {o}")
        outs[o.request][o.index] = o
    if any(sorted(x) != list(range(len(x))) for x in outs):
        raise ValueError("a request's outputs are its indices 0 … n − 1")
    last = [admitted[i] + D + len(outs[i]) // r for i in range(n_req)]
    records, steps = [UNHELD] * sh.N, []
    for t in range(sh.T):
        b = window * sh.T + t
        live = [i for i in arrival if admitted[i] <= b <= last[i]]
        if len(live) * r > sh.R:
            raise ValueError(f"bucket {b} has {len(live)} live requests, more than R / r = {sh.R // r} lanes")
        cell = {i: [(b - admitted[i] - D) * r + j if b >= admitted[i] + D else -1 for j in range(r)] for i in live}
        mine = sorted({outs[i][k].step for i in live for k in cell[i] if 0 <= k < len(outs[i])})
        if len(mine) > sh.S:
            raise ValueError(f"bucket {b}'s outputs were sampled in {len(mine)} steps, more than S = {sh.S}")
        at = {s: k for k, s in enumerate(mine)}
        for rho, i in enumerate(live):
            q = order[i] + 1
            for j, k in enumerate(cell[i]):
                e = (at[outs[i][k].step] * sh.P + outs[i][k].row if 0 <= k < len(outs[i]) else
                     END_ENTRY if k == len(outs[i]) else EMPTY)
                records[t * sh.R + rho * r + j] = MapEntry(q, e)
        mine += [mine[-1] if mine else 0] * (sh.S - len(mine))
        if any(not 0 <= s < 1 << sh.DS for s in mine):
            raise ValueError(f"a step is a position in a port of depth {sh.DS}, not {mine}")
        steps.append(tuple(mine))
    return tuple(records), tuple(steps)


def map_bytes(cell_map: Sequence[MapEntry]) -> bytes:
    """The map as the unit reads it: one 8-byte record a slot, q and e u32be."""
    return b"".join(MAP.pack(*m) for m in cell_map)


def port_row(ids: Sequence[int], P: int) -> bytes:
    """A step's port row: its batch rows' sampled ids, u32be in batch order, zero past them to P."""
    if len(ids) > P or any(type(x) is not int or not 0 <= x < 1 << 32 for x in ids):
        raise ValueError(f"a port row holds at most {P} u32 ids, not {list(ids)!r}")
    return b"".join(x.to_bytes(4, "big") for x in ids).ljust(4 * P, b"\0")


def port_id(shape: Shape, port: Sequence[bytes], e: int) -> int:
    """Port entry e: opened row ⌊e / P⌋'s (e mod P)-th id."""
    s, r = divmod(e, shape.P)
    return int.from_bytes(port[s][4 * r:4 * r + 4], "big")


def bucket_rows(shape: Shape, cells: Sequence[bytes]) -> list[bytes]:
    """The warden's rows: each bucket's R cells, R·B bytes."""
    if len(cells) != shape.N:
        raise ValueError(f"a window has {shape.N} cells, not {len(cells)}")
    return [b"".join(cells[t * shape.R:(t + 1) * shape.R]) for t in range(shape.T)]


def domain(binding: bytes, count: int) -> CommitmentDomain:
    """A dense frame-v3-sha512 domain of `count` rows, as the warden's per-window cells and the port use."""
    return CommitmentDomain(binding, -1, RangeIndexedDomain(0, count), hash="sha512")


def rows_levels(domain_id: bytes, rows: Sequence[bytes], salts: Sequence[bytes]) -> list[list[bytes]]:
    """The dense frame-v3-sha512 tree, under `domain_id`, of the rows' `hm96-sha512/row/v2` leaves, each under its salt,
    and the domain's pads to a power of two, level by level from the leaves (`tree_levels` over `domain`'s leaves)."""
    if len(rows) != len(salts) or len(rows) < 2:
        raise ValueError(f"a rows tree is over at least 2 rows, each with its salt, not {len(rows)} rows and "
                         f"{len(salts)} salts")
    level = [leaf_ref(domain_id, SCHEMA, HC.hiding_value(r, s), i) for i, (r, s) in enumerate(zip(rows, salts))]
    width = 1 << (len(rows) - 1).bit_length()
    level += [frame_hash(_PAD, domain_id, _uint(i), hash="sha512") for i in range(len(rows), width)]
    levels = [level]
    while len(level) > 1:
        level = [node_ref(domain_id, len(levels) - 1, t, level[2 * t], level[2 * t + 1]) for t in range(len(level) // 2)]
        levels.append(level)
    return levels


def rows_root(domain_id: bytes, rows: Sequence[bytes], salts: Sequence[bytes]) -> bytes:
    """The rows tree's root: the warden's registered root, which the proof system's registered read holds a row to."""
    return rows_levels(domain_id, rows, salts)[-1][0]


def rows_path(levels: Sequence[Sequence[bytes]], pos: int) -> tuple[bytes, ...]:
    """Row `pos`'s siblings from the leaves up (`FrameV3Read`'s path)."""
    return tuple(levels[k][(pos >> k) ^ 1] for k in range(len(levels) - 1))


def fake_salt(seed: bytes, i: int) -> bytes:
    """A deterministic hiding salt for a fake upstream's tests (a prover draws fresh ones)."""
    out = b""
    while len(out) < hm96.SHA512.salt_bytes:
        out += hashlib.sha512(seed + i.to_bytes(8, "big") + len(out).to_bytes(4, "big")).digest()
    return out[:hm96.SHA512.salt_bytes]


@dataclass(frozen=True)
class Instance:
    """One egress link-window's tie: the shape; the warden's cells with one salt a bucket row, their root and domain
    id (its registration: unit t reads row t there); the verifier's map and each bucket's S steps; each bucket's
    opened port rows (its steps' rows of the Commit's tree), their salts and paths; and the Commit's root and domain
    id. Public: the roots, the domain ids, the map and the steps; the rest is witness."""

    shape: Shape
    cells: tuple[bytes, ...]
    cell_salts: tuple[bytes, ...]
    cells_root: bytes
    cells_domain: bytes
    map: tuple[MapEntry, ...]
    steps: tuple[tuple[int, ...], ...]
    port: tuple[tuple[bytes, ...], ...]
    port_salts: tuple[tuple[bytes, ...], ...]
    paths: tuple[tuple[tuple[bytes, ...], ...], ...]
    port_root: bytes
    port_domain: bytes

    def bucket(self, t: int) -> slice:
        """Bucket t's cells and records."""
        return slice(t * self.shape.R, (t + 1) * self.shape.R)


# --- the reference ---------------------------------------------------------------------------------------------------

def cells_match(shape: Shape, cells: Sequence[bytes], cell_map: Sequence[MapEntry], port: Sequence[bytes]) -> bool:
    """A bucket's cells against its records and its opened port rows (`IoTieContent`'s reference): each cell the one
    its record admits (`expected_cell`), a `DATA` record's entry below E."""
    for c, m in zip(cells, cell_map, strict=True):
        if m.e in (EMPTY, END_ENTRY):
            want = expected_cell(m, shape.B)
        elif 0 <= m.e < shape.E:
            want = expected_cell(m, shape.B, port_id(shape, port, m.e))
        else:
            return False
        if c != want:
            return False
    return True


def _check_sizes(inst: Instance) -> None:
    sh = inst.shape
    sizes = {"cells": (len(inst.cells), sh.N), "cell salts": (len(inst.cell_salts), sh.T), "map": (len(inst.map), sh.N),
             "buckets' steps": (len(inst.steps), sh.T), "buckets' port rows": (len(inst.port), sh.T),
             "buckets' port salts": (len(inst.port_salts), sh.T), "buckets' paths": (len(inst.paths), sh.T)}
    for name, (got, want) in sizes.items():
        if got != want:
            raise ValueError(f"the instance has {got} {name}, the shape {want}")
    for t in range(sh.T):
        if not len(inst.steps[t]) == len(inst.port[t]) == len(inst.port_salts[t]) == len(inst.paths[t]) == sh.S or any(
                len(r) != 4 * sh.P for r in inst.port[t]) or any(len(p) != sh.DS for p in inst.paths[t]) or any(
                not 0 <= s < 1 << sh.DS for s in inst.steps[t]):
            raise ValueError(f"bucket {t} opens S = {sh.S} port rows of {4 * sh.P} bytes, each with its salt, a path of "
                             f"{sh.DS} siblings and a step below 2^{sh.DS}")


def _port_opens(inst: Instance, t: int) -> bool:
    """Bucket t's port rows open at its steps under the Commit's root: its `port_reads`, in the clear."""
    return all(opens(SCHEMA, inst.port_root, inst.port_domain, HC.hiding_value(r, s), x, list(p)) == 1
               for r, s, x, p in zip(inst.port[t], inst.port_salts[t], inst.steps[t], inst.paths[t]))


def bucket_reference(inst: Instance, t: int) -> bool:
    """`IoTieBucket`'s output for bucket t: its cells are its records', with its port rows' ids."""
    _check_sizes(inst)
    b = inst.bucket(t)
    return cells_match(inst.shape, inst.cells[b], inst.map[b], inst.port[t])


def reference_parts(inst: Instance) -> dict[str, bool]:
    """Each part's verdict in the clear: the cells' root and every bucket's port openings (the registered reads of the
    rows and of the port), and every bucket's content (its unit)."""
    _check_sizes(inst)
    sh = inst.shape
    cells = rows_root(inst.cells_domain, bucket_rows(sh, inst.cells), inst.cell_salts) == inst.cells_root
    return {"cells_root": cells, "port": all(_port_opens(inst, t) for t in range(sh.T)),
            "content": all(cells_match(sh, inst.cells[inst.bucket(t)], inst.map[inst.bucket(t)], inst.port[t])
                           for t in range(sh.T))}


def reference(inst: Instance) -> bool:
    """The link-window's tie: its rows are the registered roots', and every bucket's unit is 1."""
    return all(reference_parts(inst).values())


# --- the verifier's registered reads of the port ---------------------------------------------------------------------

def port_reads(steps: Sequence[Sequence[int]], root: bytes, domain_id: bytes, count: int, P: int,
               value: str = SAMPLED, reads_from: int = 0) -> dict[str, dict[str, Any]]:
    """The verifier's own `--registered` ports (`warden_rows.reads_file`'s `ports`) of a population of units that each
    read S rows of the Commit's port, its `count` rows of P ids under `root` in the domain `domain_id`: `port[s]` (`port`
    when S = 1, as `registered.reads` keys a parameter of one row), the unit's s-th port row, has unit `reads_from + u`
    read row steps[u][s]. The steps are the verifier's map's, so every read is at a public position, and the Lean
    verifier holds each read row to its leaf of the Commit's root by its b ‖ c (`registered.verify_read`), as it holds
    a bucket row to the warden's (`warden_rows.bucket_port`)."""
    widths = {len(st) for st in steps}
    if len(widths) != 1 or 0 in widths:
        raise ValueError("every unit reads the same S >= 1 port rows")
    if any(type(x) is not int or not 0 <= x < count for st in steps for x in st):
        raise ValueError(f"a read is a step of the Commit's {count} rows")
    S = widths.pop()
    e = {"leaves": count, "words": 2 * P, "word_bits": G.WORD_BITS, "domain": bytes(domain_id).hex()}
    extra = {G.READS_FROM: int(reads_from)} if reads_from else {}
    return {(PORT if S == 1 else f"{PORT}[{s}]"): produced.port(value, e, bytes(root).hex(), [st[s] for st in steps],
                                                                SCHEMA) | extra for s in range(S)}


# --- the prover's openings and routing -------------------------------------------------------------------------------

def open_port(levels: Sequence[Sequence[bytes]], rows: Sequence[bytes], salts: Sequence[bytes],
              steps: Sequence[int]) -> tuple[tuple[bytes, ...], tuple[bytes, ...], tuple[tuple[bytes, ...], ...]]:
    """A bucket's port openings from the Commit's rows, salts and tree: each step's row, salt and path."""
    return (tuple(rows[s] for s in steps), tuple(salts[s] for s in steps), tuple(rows_path(levels, s) for s in steps))


def benes(perm: Sequence[int]) -> list[int]:
    """The switch bits of a Beneš network that takes input i to output perm[i] (the looping algorithm), in
    `IoTieRoute`'s order: the input column, the upper half's network, the lower half's, the output column."""
    n = len(perm)
    if n < 2 or n & (n - 1) or sorted(perm) != list(range(n)):
        raise ValueError("a Beneš network routes a permutation of a power of two at least 2")
    if n == 2:
        return [perm[0]]
    inv = [0] * n
    for i, d in enumerate(perm):
        inv[d] = i
    color: list[int | None] = [None] * n
    for start in range(0, n, 2):
        i = start
        # i goes up and its switch partner down; the input sharing that partner's output switch goes up next
        while color[i] is None:
            color[i], color[i ^ 1] = 0, 1
            i = inv[perm[i ^ 1] ^ 1]
    upper, lower, out = [0] * (n // 2), [0] * (n // 2), [0] * (n // 2)
    for i in range(n):
        (upper if color[i] == 0 else lower)[i // 2] = perm[i] // 2
        if color[i] == 0:
            out[perm[i] // 2] = perm[i] & 1
    return [color[2 * k] for k in range(n // 2)] + benes(upper) + benes(lower) + out  # type: ignore[list-item]


def benes_apply(xs: Sequence, bits: Sequence[int]) -> list:
    """`IoTieRoute` in the clear: the network's outputs for its inputs under the switch bits."""
    it = iter(bits)

    def run(v: list) -> list:
        if len(v) == 2:
            return v[::-1] if next(it) else v
        up, down = [], []
        for k in range(len(v) // 2):
            a, b = (v[2 * k + 1], v[2 * k]) if next(it) else (v[2 * k], v[2 * k + 1])
            up.append(a)
            down.append(b)
        up, down = run(up), run(down)
        res = []
        for a, b in zip(up, down):
            res += [b, a] if next(it) else [a, b]
        return res

    out = run(list(xs))
    if next(it, None) is not None:
        raise ValueError("more switch bits than the network has")
    return out


def entry_perm(shape: Shape, cell_map: Sequence[MapEntry]) -> list[int]:
    """A bucket's routing, which the prover sets: port entry e to the slot whose record names it, the other positions
    in order."""
    perm, taken = [-1] * shape.M, set()
    for i, m in enumerate(cell_map):
        if 0 <= m.e < shape.M and perm[m.e] < 0:
            perm[m.e] = i
            taken.add(i)
    rest = iter(i for i in range(shape.M) if i not in taken)
    return [p if p >= 0 else next(rest) for p in perm]


def switches(inst: Instance, t: int) -> list[int]:
    """The prover's switch bits for bucket t (`entry_perm`'s)."""
    return benes(entry_perm(inst.shape, inst.map[inst.bucket(t)]))


# --- the circuit -----------------------------------------------------------------------------------------------------

def _add(C, x: Sequence, y: Sequence) -> list:
    """x + y, both as wide as x, the carry out dropped: a ripple adder, 2 ANDs a bit."""
    out, c = [], ZERO
    for a, b in zip(x, y, strict=True):
        t = a ^ b
        out.append(t ^ c)
        c = C.AND(a, b) ^ C.AND(c, t)
    return out


def _digits(C, chars: Sequence[Sequence]) -> tuple[list, list]:
    """The id field's checks (spaces then digits, a nonzero first digit unless it is the last, below 2^32) and its
    value's 32 bits, least significant first."""
    space = [and_reduce(C, [HC._same(c[b], 0x20 >> b & 1) for b in range(8)]) for c in chars]
    # a digit is 0x30 ... 0x39: high nibble 0011, low nibble at most 9 (not b3 with b2 or b1)
    digit = [C.AND(and_reduce(C, [c[7].inv(), c[6].inv(), c[5], c[4]]), C.AND(c[3], C.AND(c[2].inv(), c[1].inv()).inv()).inv())
             for c in chars]
    zero = [and_reduce(C, [c[b].inv() for b in range(4)]) for c in chars]
    oks = []
    for i in range(ID_DIGITS):
        oks.append(space[i] ^ digit[i])
        if i:
            oks.append(C.AND(digit[i - 1], space[i]).inv())
        if i < ID_DIGITS - 1:
            lead = digit[i] if i == 0 else C.AND(digit[i], space[i - 1])
            oks.append(C.AND(lead, zero[i]).inv())
    oks.append(digit[-1])
    width = 34
    v = [ZERO] * width
    for c in chars:
        # v·10 + the character's low nibble (a space's is 0)
        v = _add(C, _add(C, [ZERO] * 3 + v[:-3], [ZERO] + v[:-1]), [*c[:4], *[ZERO] * (width - 4)])
    return oks + [v[32].inv(), v[33].inv()], v[:32]


def _sig_cell(S):
    _check_cell_bytes(S.B)
    return (("cell", Array(S.B, BYTE)),), Array(CELL_OUT, BIT)


def _build_cell(C, S, *cell):
    data_t, end_t = data_cell(0, 0, S.B), end_cell(0, S.B)
    free, head = {*range(1, 5), *range(ID_AT, ID_AT + ID_DIGITS)}, HEADER.size + len(TRAILER)
    lead = [i for i in range(head) if i not in free]
    # a PAD and an END are zero past the END's payload, and before it each is its kind's but for the stream
    fill = and_reduce(C, [cell[i][b].inv() for i in range(head, S.B) for b in range(8)])
    pad = C.AND(fill, and_reduce(C, [cell[i][b].inv() for i in lead for b in range(8)]))
    end = C.AND(fill, and_reduce(C, [HC._same(cell[i][b], end_t[i] >> b & 1) for i in lead for b in range(8)]))
    digits_ok, token = _digits(C, cell[ID_AT:ID_AT + ID_DIGITS])
    data = and_reduce(C, [HC._same(cell[i][b], data_t[i] >> b & 1) for i in range(S.B) if i not in free
                          for b in range(8)] + digits_ok)
    # the stream is u32 big-endian at bytes 1 ... 4: its bit j is bit j % 8 of byte 4 - j // 8
    return [[C.AND(data.inv(), C.AND(pad.inv(), end.inv())).inv(), data, end,
             *(cell[4 - j // 8][j % 8] for j in range(32)), *token]]


IoTieCell = _part(
    "IoTieCell", ("B",), _sig_cell, _build_cell,
    "(ok, data, end, stream, id) of a B-byte cell: ok iff it is `pad_cell(B, stream)`, `end_cell(stream, B)` or "
    "`data_cell(id, stream, B)`, its bytes but the stream and the id field the template's, the field spaces then "
    "digits, the first digit nonzero unless it is the last, below 2^32; data and end iff the latter two; stream and id "
    "u32, least significant bit first")


def _check_route(S) -> None:
    if type(S.D) is not int or not 1 <= S.D <= 32 or type(S.W) is not int or S.W < 1:
        raise ValueError(f"a Beneš network has 1 <= D <= 32 and W >= 1, not D={S.D!r}, W={S.W!r}")


def _sig_route(S):
    _check_route(S)
    return ((("xs", Array(1 << S.D, Array(S.W, BIT))), ("switches", Array((2 * S.D - 1) << (S.D - 1), BIT))),
            Array(1 << S.D, Array(S.W, BIT)))


def _sig_switch(S):
    if type(S.W) is not int or S.W < 1:
        raise ValueError(f"a switch has W >= 1, not W={S.W!r}")
    return (("a", Array(S.W, BIT)), ("b", Array(S.W, BIT)), ("s", BIT)), Array(2, Array(S.W, BIT))


def _build_switch(C, S, a, b, s):
    t = [C.AND(s[0], x ^ y) for x, y in zip(a, b)]
    return [[x ^ u for x, u in zip(a, t)], [y ^ u for y, u in zip(b, t)]]


IoTieSwitch = _part(
    "IoTieSwitch", ("W",), _sig_switch, _build_switch,
    "(a, b), swapped iff s: W ANDs")


def _body_route(B, S, xs, switches_):
    W, h = S.W, 1 << (S.D - 1)
    sw, half = HC._refs(switches_), Array(h, Array(W, BIT))

    def words(refs: list) -> list:
        return [refs[W * k:W * (k + 1)] for k in range(len(refs) // W)]

    def column(pairs: list, bits: list) -> list:
        """The h switches' (a, b) pairs, swapped by their bits: one batch of `IoTieSwitch`."""
        out = words(HC._refs(B.batch(bind(IoTieSwitch, W=W), HC._coll(half, [r for a, _ in pairs for r in a]),
                                     HC._coll(half, [r for _, b in pairs for r in b]), HC._coll(Array(h, BIT), bits),
                                     axes=(0, 0, 0))))
        return list(zip(out[0::2], out[1::2]))

    x = words(HC._refs(xs))
    if S.D == 1:
        last = column([(x[0], x[1])], sw)
    else:
        # `benes`' bit order: the input column, the upper half's, the lower half's, the output column
        sub = (2 * S.D - 3) << (S.D - 2)
        first = column([(x[2 * k], x[2 * k + 1]) for k in range(h)], sw[:h])
        net = bind(IoTieRoute, W=W, D=S.D - 1)
        up, down = (words(HC._refs(B.call(net, HC._coll(half, [r for p in first for r in p[i]]),
                                          HC._coll(Array(sub, BIT), sw[h + i * sub:h + (i + 1) * sub]))))
                    for i in (0, 1))
        last = column(list(zip(up, down)), sw[h + 2 * sub:])
    return HC._coll(Array(2 * h, Array(W, BIT)), [r for p in last for w in p for r in w])


IoTieRoute = CompositeDefinition(
    "IoTieRoute", 1, ("W", "D"), _sig_route, _body_route, conformance="exact-model-tested",
    doc="a Beneš network on 2^D W-bit inputs: the input column's 2^(D − 1) switches (`IoTieSwitch`: s swaps "
        "inputs 2k and 2k + 1 into the halves), the upper half's `IoTieRoute{W, D − 1}`, the lower half's, the "
        "output column's switches; W ANDs a switch")


def _check_match(S) -> None:
    if type(S.N) is not int or S.N < 1 or type(S.D) is not int or not 1 <= S.D <= 32 or type(S.E) is not int or not (
            1 <= S.E <= 1 << S.D and S.N <= 1 << S.D and S.E < END_ENTRY):
        raise ValueError(f"a match has N, E >= 1 with N, E <= 2^D, E < 2^32 − 2 and 1 <= D <= 32, not N={S.N!r}, "
                         f"D={S.D!r}, E={S.E!r}")


def _sig_match(S):
    _check_match(S)
    return ((("cells", Array(S.N, Array(CELL_OUT, BIT))), ("map", Array(S.N, Array(MAP.size, BYTE))),
             ("routed", Array(1 << S.D, Array(32 + S.D, BIT)))), BIT)


def _eq(C, x: Sequence, y: Sequence):
    return and_reduce(C, [(a ^ b).inv() for a, b in zip(x, y, strict=True)])


def _lt(C, x: Sequence, y: Sequence):
    """x < y, unsigned, least significant bit first: 1 AND a bit."""
    lt = ZERO
    for a, b in zip(x, y, strict=True):
        lt = lt ^ C.AND(a ^ b, lt ^ b)
    return lt


def _build_match(C, S, *e):
    N = S.N
    cells, mp, routed = e[:N], e[N:N + MAP.size * N], e[N + MAP.size * N:]
    below = [ONE if S.E >> j & 1 else ZERO for j in range(32)]

    def field(i: int, f: int) -> list:
        at = MAP.size * i + 4 * f
        return [mp[at + 3 - j // 8][j % 8] for j in range(32)]

    oks = []
    for i in range(N):
        c, r = cells[i], routed[i]
        q, x = field(i, 0), field(i, 1)
        # e ≥ 2^32 − 2: a PAD's record (e odd) or an END's; any other is a DATA cell's
        marker = and_reduce(C, x[1:])
        end = C.AND(marker, x[0].inv())
        hit = and_reduce(C, [_eq(C, c[35:67], r[:32]), _eq(C, r[32:], x[:S.D]), _lt(C, x, below)])
        oks += [c[0], _eq(C, c[3:35], q), c[1] ^ marker, (c[2] ^ end).inv(), C.AND(marker.inv(), hit.inv()).inv()]
    return [and_reduce(C, oks)]


IoTieMatch = _part(
    "IoTieMatch", ("N", "D", "E"), _sig_match, _build_match,
    "1 iff each of the N cells (ok, data, end, stream, id) is ok, on its map record (q, e)'s stream q, and of its "
    "kind: a PAD where e = 2^32 − 1, an END where e = 2^32 − 2, else a DATA cell with e < E whose position's routed "
    "entry (id, index) has index e and its id")


def _bucket_shape(S) -> Shape:
    return Shape(S.B, 2, S.R, S.P, S.S, getattr(S, "DS", 1))


def _sig_content(S):
    sh = _bucket_shape(S)
    return ((("cells", Array(S.R, Array(S.B, BYTE))), ("map", Array(S.R, Array(MAP.size, BYTE))),
             ("port", Array(S.S, Array(4 * S.P, BYTE))), ("switches", Array(sh.switches, BIT))), BIT)


def _body_content(B, S, cells, mp, port, switches_):
    sh = _bucket_shape(S)
    K, cr, pr = HC._Consts(B), HC._refs(cells), HC._refs(port)
    outs = []
    for i in range(S.R):
        outs += HC._refs(B.call(bind(IoTieCell, B=S.B), HC._coll(Array(S.B, BYTE), cr[8 * S.B * i:8 * S.B * (i + 1)])))
    xs = []
    for e in range(sh.M):
        if e < sh.E:
            # entry e's id is u32be at bytes 4e ... 4e + 3 of the opened rows: its bit j is bit j % 8 of byte 4e + 3 - j // 8
            xs += [pr[8 * (4 * e + 3 - j // 8) + j % 8] for j in range(32)]
        else:
            xs += K.bits(0, 32)
        xs += K.bits(e, sh.D)
    routed = B.call(bind(IoTieRoute, W=sh.W, D=sh.D), HC._coll(Array(sh.M, Array(sh.W, BIT)), xs), switches_)
    return B.call(bind(IoTieMatch, N=S.R, D=sh.D, E=sh.E), HC._coll(Array(S.R, Array(CELL_OUT, BIT)), outs), mp, routed)


IoTieContent = CompositeDefinition(
    "IoTieContent", 1, ("B", "R", "P", "S"), _sig_content, _body_content, conformance="exact-model-tested",
    doc="1 iff every one of a bucket's R B-byte cells is a PAD, an END or a DATA cell (`IoTieCell`) as its map record "
        "says, on the record's stream, and each DATA cell's id is the port entry the record names: the bucket's "
        "port's E = S·P entries (each its id and index) and M − E zero ones routed by the switches (`IoTieRoute`) to "
        "the slots and matched there (`IoTieMatch`)")


def _sig_bucket(S):
    sh = _bucket_shape(S)
    return ((("row", Array(8 * S.R * S.B, BIT)), ("map", Array(S.R, Array(MAP.size, BYTE))),
             (PORT, Array(S.S, Array(4 * S.P, BYTE))), ("switches", Array(sh.switches, BIT))), BIT)


def _body_bucket(B, S, row, mp, port, switches_):
    return B.call(bind(IoTieContent, B=S.B, R=S.R, P=S.P, S=S.S), row.reshape(Array(S.R, Array(S.B, BYTE))), mp, port,
                  switches_)


IoTieBucket = CompositeDefinition(
    "IoTieBucket", 1, ("B", "R", "P", "S"), _sig_bucket, _body_bucket, conformance="exact-model-tested",
    doc="1 iff every cell of the bucket row (the warden's registered row, read by the proof system: bit 8k + i is bit i "
        "of byte k) is its map record's, with a DATA cell's id the entry the record names of the S port rows (the "
        "Commit's registered rows at the bucket's steps, read by the proof system) (`IoTieContent`): the I/O tie of "
        "one egress bucket row, every cell opened; a link-window's tie is its T rows'")


# --- arguments -------------------------------------------------------------------------------------------------------

def content_arguments(inst: Instance, t: int, bits: Sequence[int]) -> list[list[int]]:
    """Bucket t's `IoTieContent` arguments in its signature's order, each its leaves' bits."""
    b = inst.bucket(t)
    return [[x for c in inst.cells[b] for x in HC._bytes_bits(c)], HC._bytes_bits(map_bytes(inst.map[b])),
            [x for r in inst.port[t] for x in HC._bytes_bits(r)], list(bits)]


def bucket_arguments(inst: Instance, t: int, bits: Sequence[int]) -> list[list[int]]:
    """Bucket t's `IoTieBucket` arguments in its signature's order, each its leaves' bits: the row and the port rows as
    the registered reads give them, then the map and the switches as `IoTieContent` takes them."""
    return content_arguments(inst, t, bits)


__all__ = ["CELL_OUT", "CELL_USED", "EMPTY", "END", "END_ENTRY", "ID_AT", "ID_EVENT_BYTES", "Instance",
           "IoTieBucket", "IoTieCell", "IoTieContent", "IoTieMatch", "IoTieRoute", "IoTieSwitch", "MAP", "MapEntry",
           "Output", "PORT", "Parsed", "SAMPLED", "Shape", "TRAILER", "UNHELD", "benes", "benes_apply",
           "bucket_arguments", "bucket_reference", "bucket_rows", "cells_match", "content_arguments", "data_cell",
           "domain", "end_cell", "entry_perm", "expected_cell", "fake_salt", "id_event", "map_bytes", "open_port",
           "pad_cell", "parse_cell", "port_id", "port_reads", "port_row", "reference", "reference_parts",
           "route_depth", "rows_levels", "rows_path", "rows_root", "switches", "window_map"]
