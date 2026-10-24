"""The I/O tie (`PROTOCOL.md`, "The I/O tie"), a prototype on a fake upstream: the reference of one egress
link-window's tie, a unit at selection `All` that opens every cell, and the Boolean Definitions it is built from.
Nothing here is in a Spec or proved, and no warden or Commit writes these values yet: the fake upstream commits them as
the design has the warden and the Commit commit them, under `hm96-sha512/row/v2` rows in `frame-v3-sha512` trees.

**The cells.** A link-window's egress is N = T·R cells of B bytes in the warden's schedule order, R a bucket, so cell
i is in bucket ⌊i / R⌋ of the window. A cell is a pad (`pad_cell`, B zero bytes) or a data cell (`data_cell`): a
`DATA` header whose stream is the request's ingress ordinal, which the warden writes from the connection it
terminated, then one token id's event (`id_event`) and zero fill. The warden commits one row a bucket (`bucket_rows`).

**The map is a function of the advice** (`window_map`): for each cell, the output the warden must have released there,
its request q and its entry e in the window's port, or a pad. Bucket t's outputs (release bucket w·T + t) fill its
first cells sorted by (request, index), the order the warden releases a bucket in, and pads fill the rest. No prover
value says where an output goes. The advice is private; here the verifier computes the map in the clear from the
opened advice v0, which is the named gap ("the advice is opened to the verifier until the compile unit exists"). The
target reads the map inside the proof under the advice's hiding root.

**The port is the Commit's**: its sampled ids, one hiding row a step (`port_row`: the step's P ids, u32be in batch-row
order, zero past its batch) in a dense frame-v3-sha512 tree of the session's steps, whose root the Commit registers.
The window opens S rows (`Shape.S`), at the steps its outputs were sampled in (`window_map`'s steps, public with the map), and its
entry e = σ·P + r is opened row σ's r-th id. The real port is vllm's Commit (its binding map's `sampled_token_ids`
under the run root), which this fake upstream stands in for.

**What 1 means** (`reference`): the cells' root is their bucket rows', every opened port row opens at its step under
the port's root, and every cell is its map's (`cells_match`): a pad where the map has a pad, else a data cell on
stream q whose id is port entry e. So a duplicated, dropped, reordered, moved or altered cell, a covert byte, a cell
on another stream, a fabricated output, a wrong id and a forged port row each make it 0.

    IoTieCell{B}(cell: Array<B, Byte>) -> Array<66>                  ok, data, stream (u32), id (u32), LSB first
    IoTieRoute{W, D}(xs: Array<M, Array<W>>, switches: Array<S_W>) -> Array<M, Array<W>>
        a Beneš network of S_W = (2D − 1)·2^(D − 1) switches, W ANDs each
    IoTieMatch{N, D, E}(cells: Array<N, Array<66>>, map: Array<N, Array<8, Byte>>, routed: Array<M, Array<32 + D>>)
        -> Value<1>
    IoTieContent{B, T, R, P, S}(cells: Array<N, Array<B, Byte>>, map: Array<N, Array<8, Byte>>,
        port: Array<S, Array<4P, Byte>>, switches: Array<S_W>) -> Value<1>
    IoTiePort{P, S, DS}(root, domain, port: Array<S, Array<4P, Byte>>, salts: Array<S, Salt>,
        steps: Array<S, Array<DS>>, paths: Array<S, Array<DS, Digest>>) -> Value<1>
    NetworkRowsRoot{N_BITS, COUNT}(rows: Array<COUNT, Array<N_BITS>>, salts: Array<COUNT, Salt>, root, domain) -> Value<1>
    IoTieWindow{B, T, R, P, S, DS}(cells_root, cells_domain, cells, cell_salts, port_root, port_domain, port, port_salts,
        steps, paths, map, switches) -> Value<1>
        E = S·P port entries, M = 2^D ≥ max(N, E)

`IoTieWindow` is the unit: it recomputes the cells' root from every bucket row and salt, opens each port row at its
step (`FrameV3Read`), and holds every cell to the map (`IoTieContent`). The content routes the port's E entries, each
its id and its index e, to the cells by the prover's switches (`switches`); any setting is a permutation, and
`IoTieMatch` holds each data cell's id to the entry the map names, by its index. Public: both roots and domain ids, and,
until the named gap closes, the steps and the map. Witness: the cells and their salts, the opened port rows, their salts
and paths, and the switches.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import NamedTuple, Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import ONE, ZERO, and_reduce
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.types import Array
from verity.ml.commitments import hm96
from verity.ml.commitments.gates.frame_v3 import (LEVEL, FrameV3Leaf, FrameV3Node, _bytes as _fv_bytes, _hash_uint, _part,
                                                  index_bytes, leaf_ref, node_ref, opens, read)
from verity.ml.commitments.gates.hm96 import Hm96Sha512Leaf
from verity.ml.commitments.gates.lms import LmsEqual
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import _PAD, CommitmentDomain, _hash, _uint
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as SCHEMA
from verity.core.protocols.certifier import head_check as HC
from verity.core.protocols.certifier.active import DATA, EVENT_END, EVENT_HEAD, HEADER, PAD

BIT, BYTE, DIGEST, SALT = HC.BIT, HC.BYTE, HC.DIGEST, HC.SALT
ID_DIGITS = 10
ID_EVENT_HEAD = EVENT_HEAD[:EVENT_HEAD.index(b'"delta"')] + b'"token_id":'
ID_EVENT_TAIL = b',"finish_reason":null}]}' + EVENT_END
ID_EVENT_BYTES = len(ID_EVENT_HEAD) + ID_DIGITS + len(ID_EVENT_TAIL)
#: the id's first character in a cell: the header, then the event's head
ID_AT = HEADER.size + len(ID_EVENT_HEAD)
#: a data cell's bytes before its zero fill
CELL_USED = HEADER.size + ID_EVENT_BYTES
#: `IoTieCell`'s output: ok, data, the stream and the id
CELL_OUT = 2 + 32 + 32
#: a map record: the request q and the port entry e, u32be each; a pad's is all 0xFF
MAP = struct.Struct(">II")
EMPTY = 0xFFFF_FFFF


# --- the cells -------------------------------------------------------------------------------------------------------

def id_event(token_id: int) -> bytes:
    """The canonical event of one token id: `ID_EVENT_HEAD`, the id right-aligned in 10 decimal characters,
    `ID_EVENT_TAIL`; `ID_EVENT_BYTES` long whatever the id."""
    if type(token_id) is not int or not 0 <= token_id < 1 << 32:
        raise ValueError(f"a token id is a u32, not {token_id!r}")
    return ID_EVENT_HEAD + b"%10d" % token_id + ID_EVENT_TAIL


def _check_cell_bytes(cell_bytes: object) -> None:
    if type(cell_bytes) is not int or cell_bytes <= CELL_USED:
        raise ValueError(f"a cell has more than {CELL_USED} bytes, room for an id event and its fill, not {cell_bytes!r}")


def data_cell(token_id: int, stream: int, cell_bytes: int) -> bytes:
    """A data cell: a `DATA` header on the request's stream, its id's event, zero fill."""
    _check_cell_bytes(cell_bytes)
    if type(stream) is not int or not 0 <= stream < 1 << 32:
        raise ValueError(f"a stream is a u32, not {stream!r}")
    return (HEADER.pack(DATA, stream, ID_EVENT_BYTES) + id_event(token_id)).ljust(cell_bytes, b"\0")


def pad_cell(cell_bytes: int) -> bytes:
    """A pad cell: a `PAD` header on stream 0 with no payload, zero fill, so B zero bytes."""
    _check_cell_bytes(cell_bytes)
    return HEADER.pack(PAD, 0, 0).ljust(cell_bytes, b"\0")


class Parsed(NamedTuple):
    """A cell's reading: a data cell's stream and id, or a pad's (False, 0, 0)."""

    data: bool
    stream: int
    id: int


def parse_cell(cell: bytes) -> Parsed | None:
    """The cell's one reading, or None for a cell that is neither: `IoTieCell`'s reference."""
    if len(cell) <= CELL_USED:
        return None
    if cell == pad_cell(len(cell)):
        return Parsed(False, 0, 0)
    digits = cell[ID_AT:ID_AT + ID_DIGITS]
    if not digits.lstrip(b" ").isdigit():
        return None
    token_id, stream = int(digits), HEADER.unpack_from(cell)[1]
    return Parsed(True, stream, token_id) if token_id < 1 << 32 and cell == data_cell(token_id, stream, len(cell)) else None


# --- the window, its map and its port --------------------------------------------------------------------------------

def route_depth(cells: int, entries: int) -> int:
    """D of the routing over M = 2^D ≥ max(cells, entries) positions."""
    return max(1, (max(cells, entries) - 1).bit_length())


@dataclass(frozen=True)
class Shape:
    """A link-window's egress, T ≥ 2 buckets of R cells of B bytes (N = T·R cells, the warden's T rows), and its port:
    S opened rows of P ids each (E = S·P entries) from a port tree of depth DS. The routing is over M = 2^D ≥
    max(N, E) positions."""

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
        if self.N > 1 << 32 or self.E >= 1 << 32 or self.DS > 32:
            raise ValueError(f"a window has at most 2^32 cells, fewer than 2^32 port entries and a port of depth at most "
                             f"32, not N = {self.N}, E = {self.E}, DS = {self.DS}")

    @property
    def N(self) -> int:
        return self.T * self.R

    @property
    def E(self) -> int:
        return self.S * self.P

    @property
    def D(self) -> int:
        return route_depth(self.N, self.E)

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
    """A data cell's place in the map: its request q and its port entry e."""

    q: int
    e: int


def window_map(shape: Shape, outputs: Sequence[Output], release: Sequence[int],
               window: int) -> tuple[tuple[MapEntry | None, ...], tuple[int, ...]]:
    """The verifier's map of egress window `window` from advice v0 (`advice.outputs(a)` and `a.release`), and the steps
    whose port rows the window opens. The outputs released in the window's bucket t fill cells t·R, t·R + 1, … sorted by
    (request, index), each with its request and its entry σ·P + row, σ its step's place among the window's steps; the
    rest are pads (None). The steps are sorted and padded to S by repeating the last (step 0 for an empty window)."""
    sh = shape
    if len(outputs) != len(release):
        raise ValueError("one release bucket an output")
    mine = [(b - window * sh.T, o) for o, b in zip(outputs, release) if b // sh.T == window]
    steps = sorted({o.step for _, o in mine})
    if len(steps) > sh.S:
        raise ValueError(f"the window's outputs were sampled in {len(steps)} steps, more than S = {sh.S}")
    at = {s: k for k, s in enumerate(steps)}
    cells: list[MapEntry | None] = [None] * sh.N
    for t in range(sh.T):
        here = sorted((o for b, o in mine if b == t), key=lambda o: (o.request, o.index))
        if len(here) > sh.R:
            raise ValueError(f"bucket {t} releases {len(here)} outputs, more than R = {sh.R}")
        for k, o in enumerate(here):
            if not 0 <= o.row < sh.P or not 0 <= o.request < EMPTY:
                raise ValueError(f"an output's row is below P = {sh.P} and its request below 2^32 − 1, not {o}")
            cells[t * sh.R + k] = MapEntry(o.request, at[o.step] * sh.P + o.row)
    steps += [steps[-1] if steps else 0] * (sh.S - len(steps))
    if any(not 0 <= s < 1 << sh.DS for s in steps):
        raise ValueError(f"a step is a position in a port of depth {sh.DS}, not {steps}")
    return tuple(cells), tuple(steps)


def map_bytes(cell_map: Sequence[MapEntry | None]) -> bytes:
    """The map as the unit reads it: one 8-byte record a cell, q and e u32be, a pad's all 0xFF."""
    return b"".join(b"\xff" * MAP.size if m is None else MAP.pack(*m) for m in cell_map)


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
    level += [_hash(_PAD, domain_id, _uint(i), hash="sha512") for i in range(len(rows), width)]
    levels = [level]
    while len(level) > 1:
        level = [node_ref(domain_id, len(levels) - 1, t, level[2 * t], level[2 * t + 1]) for t in range(len(level) // 2)]
        levels.append(level)
    return levels


def rows_root(domain_id: bytes, rows: Sequence[bytes], salts: Sequence[bytes]) -> bytes:
    """The rows tree's root (`NetworkRowsRoot`'s reference)."""
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
    """One egress link-window's tie: the shape; the warden's cells with one salt a bucket row, their root and domain id;
    the verifier's map and steps; and the port's opened rows (the steps' rows of the Commit's tree), their salts and
    paths, and the Commit's root and domain id. Public: the roots, the domain ids, the map and the steps; the rest is
    witness."""

    shape: Shape
    cells: tuple[bytes, ...]
    cell_salts: tuple[bytes, ...]
    cells_root: bytes
    cells_domain: bytes
    map: tuple[MapEntry | None, ...]
    steps: tuple[int, ...]
    port: tuple[bytes, ...]
    port_salts: tuple[bytes, ...]
    paths: tuple[tuple[bytes, ...], ...]
    port_root: bytes
    port_domain: bytes


# --- the reference ---------------------------------------------------------------------------------------------------

def cells_match(shape: Shape, cells: Sequence[bytes], cell_map: Sequence[MapEntry | None], port: Sequence[bytes]) -> bool:
    """Every cell its map's: a pad where the map has a pad; else a data cell on the map's stream q whose id is port
    entry e, e below E (`IoTieContent`'s reference)."""
    for c, m in zip(cells, cell_map, strict=True):
        p = parse_cell(c) if len(c) == shape.B else None
        if p is None or (m is None and p.data):
            return False
        if m is not None and not (p.data and p.stream == m.q and 0 <= m.e < shape.E and p.id == port_id(shape, port, m.e)):
            return False
    return True


def reference_parts(inst: Instance) -> dict[str, bool]:
    """Each part's verdict in the clear: the cells' root, the port's openings and the content."""
    sh = inst.shape
    sizes = {"cells": (len(inst.cells), sh.N), "cell salts": (len(inst.cell_salts), sh.T), "map": (len(inst.map), sh.N),
             "steps": (len(inst.steps), sh.S), "port rows": (len(inst.port), sh.S),
             "port salts": (len(inst.port_salts), sh.S), "paths": (len(inst.paths), sh.S)}
    for name, (got, want) in sizes.items():
        if got != want:
            raise ValueError(f"the instance has {got} {name}, the shape {want}")
    if any(len(r) != 4 * sh.P for r in inst.port) or any(len(p) != sh.DS for p in inst.paths) or any(
            not 0 <= s < 1 << sh.DS for s in inst.steps):
        raise ValueError(f"a port row is {4 * sh.P} bytes, a path {sh.DS} siblings and a step below 2^{sh.DS}")
    cells = rows_root(inst.cells_domain, bucket_rows(sh, inst.cells), inst.cell_salts) == inst.cells_root
    port = all(opens(SCHEMA, inst.port_root, inst.port_domain, HC.hiding_value(r, s), x, list(p)) == 1
               for r, s, x, p in zip(inst.port, inst.port_salts, inst.steps, inst.paths))
    return {"cells_root": cells, "port": port, "content": cells_match(sh, inst.cells, inst.map, inst.port)}


def reference(inst: Instance) -> bool:
    """`IoTieWindow`'s output: every part's."""
    return all(reference_parts(inst).values())


# --- the prover's openings and routing -------------------------------------------------------------------------------

def open_port(levels: Sequence[Sequence[bytes]], rows: Sequence[bytes], salts: Sequence[bytes],
              steps: Sequence[int]) -> tuple[tuple[bytes, ...], tuple[bytes, ...], tuple[tuple[bytes, ...], ...]]:
    """The window's port openings from the Commit's rows, salts and tree: each step's row, salt and path."""
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


def entry_perm(shape: Shape, cell_map: Sequence[MapEntry | None]) -> list[int]:
    """The routing the prover sets: port entry e to the cell the map names it at, the other positions in order."""
    perm, taken = [-1] * shape.M, set()
    for i, m in enumerate(cell_map):
        if m is not None and 0 <= m.e < shape.M and perm[m.e] < 0:
            perm[m.e] = i
            taken.add(i)
    rest = iter(i for i in range(shape.M) if i not in taken)
    return [p if p >= 0 else next(rest) for p in perm]


def switches(inst: Instance) -> list[int]:
    """The prover's switch bits (`entry_perm`'s)."""
    return benes(entry_perm(inst.shape, inst.map))


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
    template, free = data_cell(0, 0, S.B), {*range(1, 5), *range(ID_AT, ID_AT + ID_DIGITS)}
    fill = and_reduce(C, [cell[i][b].inv() for i in range(CELL_USED, S.B) for b in range(8)])
    digits_ok, token = _digits(C, cell[ID_AT:ID_AT + ID_DIGITS])
    data = and_reduce(C, [HC._same(cell[i][b], template[i] >> b & 1) for i in range(CELL_USED) if i not in free
                          for b in range(8)] + digits_ok)
    pad = and_reduce(C, [cell[i][b].inv() for i in range(CELL_USED) for b in range(8)])
    # the stream is u32 big-endian at bytes 1 ... 4: its bit j is bit j % 8 of byte 4 - j // 8
    return [[C.AND(fill, C.AND(data.inv(), pad.inv()).inv()), data, *(cell[4 - j // 8][j % 8] for j in range(32)), *token]]


IoTieCell = _part(
    "IoTieCell", ("B",), _sig_cell, _build_cell,
    "(ok, data, stream, id) of a B-byte cell: ok iff it is the pad (B zero bytes) or `data_cell(id, stream, B)`, its "
    "bytes but the stream and the id field the template's, the field spaces then digits, the first digit nonzero "
    "unless it is the last, below 2^32; data iff the latter; stream and id u32, least significant bit first")


def _check_route(S) -> None:
    if type(S.D) is not int or not 1 <= S.D <= 32 or type(S.W) is not int or S.W < 1:
        raise ValueError(f"a Beneš network has 1 <= D <= 32 and W >= 1, not D={S.D!r}, W={S.W!r}")


def _sig_route(S):
    _check_route(S)
    return ((("xs", Array(1 << S.D, Array(S.W, BIT))), ("switches", Array((2 * S.D - 1) << (S.D - 1), BIT))),
            Array(1 << S.D, Array(S.W, BIT)))


def _swap(C, s, a: Sequence, b: Sequence) -> tuple[list, list]:
    t = [C.AND(s, x ^ y) for x, y in zip(a, b)]
    return [x ^ u for x, u in zip(a, t)], [y ^ u for y, u in zip(b, t)]


def _build_route(C, S, *e):
    it = iter(e[-1])

    def run(v: list) -> list:
        if len(v) == 2:
            return list(_swap(C, next(it), v[0], v[1]))
        up, down = [], []
        for k in range(len(v) // 2):
            a, b = _swap(C, next(it), v[2 * k], v[2 * k + 1])
            up.append(a)
            down.append(b)
        up, down = run(up), run(down)
        res = []
        for a, b in zip(up, down):
            res += _swap(C, next(it), a, b)
        return res

    return run([list(x) for x in e[:-1]])


IoTieRoute = _part(
    "IoTieRoute", ("W", "D"), _sig_route, _build_route,
    "a Beneš network on 2^D W-bit inputs: the input column's 2^(D − 1) switches (s swaps inputs 2k and 2k + 1 "
    "into the halves), the upper half's network, the lower half's, the output column's switches; W ANDs a switch")


def _check_match(S) -> None:
    if type(S.N) is not int or S.N < 1 or type(S.D) is not int or not 1 <= S.D <= 32 or type(S.E) is not int or not (
            1 <= S.E <= 1 << S.D and S.N <= 1 << S.D and S.E < 1 << 32):
        raise ValueError(f"a match has N, E >= 1 with N, E <= 2^D, E < 2^32 and 1 <= D <= 32, not N={S.N!r}, D={S.D!r}, "
                         f"E={S.E!r}")


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
        full = and_reduce(C, q).inv()
        hit = and_reduce(C, [_eq(C, c[2:34], q), _eq(C, c[34:66], r[:32]), _eq(C, r[32:], x[:S.D]), _lt(C, x, below)])
        oks += [c[0], (c[1] ^ full).inv(), C.AND(full, hit.inv()).inv()]
    return [and_reduce(C, oks)]


IoTieMatch = _part(
    "IoTieMatch", ("N", "D", "E"), _sig_match, _build_match,
    "1 iff each of the N cells (ok, data, stream, id) is ok and a data cell exactly where its map record (q, e) is "
    "full (q ≠ 2^32 − 1), and at a full record its stream is q, e < E, and the entry routed to its position (id, "
    "index) has index e and its id")


def _sig_content(S):
    sh = Shape(S.B, S.T, S.R, S.P, S.S, 1)
    return ((("cells", Array(sh.N, Array(S.B, BYTE))), ("map", Array(sh.N, Array(MAP.size, BYTE))),
             ("port", Array(S.S, Array(4 * S.P, BYTE))), ("switches", Array(sh.switches, BIT))), BIT)


def _body_content(B, S, cells, mp, port, switches_):
    sh = Shape(S.B, S.T, S.R, S.P, S.S, 1)
    K, cr, pr = HC._Consts(B), HC._refs(cells), HC._refs(port)
    outs = []
    for i in range(sh.N):
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
    return B.call(bind(IoTieMatch, N=sh.N, D=sh.D, E=sh.E), HC._coll(Array(sh.N, Array(CELL_OUT, BIT)), outs), mp, routed)


IoTieContent = CompositeDefinition(
    "IoTieContent", 1, ("B", "T", "R", "P", "S"), _sig_content, _body_content, conformance="exact-model-tested",
    doc="1 iff every one of the window's N = T·R B-byte cells is a pad or a data cell (`IoTieCell`) as its map record "
        "says, and each data cell's stream is the record's request and its id the port entry the record names: the "
        "port's E = S·P entries (each its id and index) and M − E zero ones routed by the switches (`IoTieRoute`) to "
        "the cells' positions and matched there (`IoTieMatch`)")


def _sig_pad(S):
    return (("domain", DIGEST), ("x", Array(8 * S.B, BIT))), DIGEST


FrameV3Pad = _part(
    "FrameV3Pad", ("B",), _sig_pad, lambda C, S, *e: _hash_uint(C, _PAD, [_fv_bytes(e[0:8])], e[8], S.B, lambda u: [u]),
    "the frame-v3-sha512 pad at rank x under the domain id domain (`CommitmentDomain.pad`); x LSB first, its uint "
    "1 ... B bytes")


def _sig_rows(S):
    if type(S.N_BITS) is not int or S.N_BITS < 1 or type(S.COUNT) is not int or not 2 <= S.COUNT <= 1 << 32:
        raise ValueError(f"a rows root has N_BITS >= 1 and 2 <= COUNT <= 2^32, not N_BITS={S.N_BITS!r}, COUNT={S.COUNT!r}")
    return ((("rows", Array(S.COUNT, Array(S.N_BITS, BIT))), ("salts", Array(S.COUNT, SALT)), ("root", DIGEST),
             ("domain", DIGEST)), BIT)


def _body_rows(B, S, rows, salts, root, dom):
    K, rr, sr, sb = HC._Consts(B), HC._refs(rows), HC._refs(salts), SALT.leaves
    D = (S.COUNT - 1).bit_length()
    nb = index_bytes(D)
    level = []
    for i in range(S.COUNT):
        x = B.call(bind(HC.NetworkRowDigest, N_BITS=S.N_BITS), HC._coll(Array(S.N_BITS, BIT), rr[S.N_BITS * i:S.N_BITS * (i + 1)]))
        v = B.call(Hm96Sha512Leaf, x, HC._coll(SALT, sr[sb * i:sb * (i + 1)]))[2]
        level.append(B.call(bind(FrameV3Leaf, B=nb, SCHEMA=SCHEMA), dom, v, HC._coll(Array(8 * nb, BIT), K.bits(i, 8 * nb))))
    for i in range(S.COUNT, 1 << D):
        level.append(B.call(bind(FrameV3Pad, B=nb), dom, HC._coll(Array(8 * nb, BIT), K.bits(i, 8 * nb))))
    for k in range(D):
        nb = index_bytes(D - k - 1)
        level = [B.call(bind(FrameV3Node, B=nb), dom, HC._coll(LEVEL, K.bits(k, 8)), HC._coll(Array(8 * nb, BIT), K.bits(t, 8 * nb)),
                        HC._coll(BIT, K.bits(0, 1)), level[2 * t], level[2 * t + 1]) for t in range(len(level) // 2)]
    return B.call(bind(LmsEqual, N=64), level[0], root)


NetworkRowsRoot = CompositeDefinition(
    "NetworkRowsRoot", 1, ("N_BITS", "COUNT"), _sig_rows, _body_rows, conformance="exact-model-tested",
    doc="1 iff root is the dense frame-v3-sha512 root, under the domain id domain, of the COUNT N_BITS-bit rows' "
        "hm96-sha512/row/v2 leaves, each under its salt, and the domain's pads to a power of two: every row's digest, "
        "hm96 leaf and frame-v3 leaf, every pad and every node")


def _sig_port(S):
    for name, least, most in (("P", 1, 1 << 28), ("S", 1, 1 << 32), ("DS", 1, 32)):
        if type(getattr(S, name)) is not int or not least <= getattr(S, name) <= most:
            raise ValueError(f"a port read has {least} <= {name} <= {most}, not {getattr(S, name)!r}")
    return ((("root", DIGEST), ("domain", DIGEST), ("port", Array(S.S, Array(4 * S.P, BYTE))), ("salts", Array(S.S, SALT)),
             ("steps", Array(S.S, Array(S.DS, BIT))), ("paths", Array(S.S, Array(S.DS, DIGEST)))), BIT)


def _body_port(B, S, root, dom, port, salts, steps, paths):
    rr, sr, st, pa = HC._refs(port), HC._refs(salts), HC._refs(steps), HC._refs(paths)
    nb, sb, db = 32 * S.P, SALT.leaves, DIGEST.leaves * S.DS
    oks = []
    for s in range(S.S):
        x = B.call(bind(HC.NetworkRowDigest, N_BITS=nb), HC._coll(Array(nb, BIT), rr[nb * s:nb * (s + 1)]))
        v = B.call(Hm96Sha512Leaf, x, HC._coll(SALT, sr[sb * s:sb * (s + 1)]))[2]
        oks += HC._refs(B.call(read(S.DS, SCHEMA), root, dom, v, HC._coll(Array(S.DS, BIT), st[S.DS * s:S.DS * (s + 1)]),
                               HC._coll(Array(S.DS, DIGEST), pa[db * s:db * (s + 1)])))
    if S.S == 1:
        return HC._coll(BIT, oks)
    return B.call(BL.and_gate(S.S), HC._coll(Array(S.S, BIT), oks))


IoTiePort = CompositeDefinition(
    "IoTiePort", 1, ("P", "S", "DS"), _sig_port, _body_port, conformance="exact-model-tested",
    doc="1 iff each of the S port rows of P u32 ids, its hm96-sha512/row/v2 leaf under its salt, opens at its step under "
        "root in the depth-DS dense frame-v3-sha512 tree of domain id domain (`FrameV3Read`)")


def _sig_window(S):
    sh = Shape(S.B, S.T, S.R, S.P, S.S, S.DS)
    return ((("cells_root", DIGEST), ("cells_domain", DIGEST), ("cells", Array(sh.N, Array(S.B, BYTE))),
             ("cell_salts", Array(sh.T, SALT)), ("port_root", DIGEST), ("port_domain", DIGEST),
             ("port", Array(S.S, Array(4 * S.P, BYTE))), ("port_salts", Array(S.S, SALT)),
             ("steps", Array(S.S, Array(S.DS, BIT))), ("paths", Array(S.S, Array(S.DS, DIGEST))),
             ("map", Array(sh.N, Array(MAP.size, BYTE))), ("switches", Array(sh.switches, BIT))), BIT)


def _body_window(B, S, cells_root, cells_domain, cells, cell_salts, port_root, port_domain, port, port_salts, steps, paths,
                 mp, switches_):
    cell_row = 8 * S.B * S.R
    oks = [B.call(bind(NetworkRowsRoot, N_BITS=cell_row, COUNT=S.T), cells.reshape(Array(S.T, Array(cell_row, BIT))),
                  cell_salts, cells_root, cells_domain),
           B.call(bind(IoTiePort, P=S.P, S=S.S, DS=S.DS), port_root, port_domain, port, port_salts, steps, paths),
           B.call(bind(IoTieContent, B=S.B, T=S.T, R=S.R, P=S.P, S=S.S), cells, mp, port, switches_)]
    bits = [r for o in oks for r in HC._refs(o)]
    return B.call(BL.and_gate(3), HC._coll(Array(3, BIT), bits))


IoTieWindow = CompositeDefinition(
    "IoTieWindow", 1, ("B", "T", "R", "P", "S", "DS"), _sig_window, _body_window, conformance="exact-model-tested",
    doc="1 iff the cells' bucket rows are the cells root's (`NetworkRowsRoot`), each port row opens at its step under "
        "the port root (`IoTiePort`), and every cell is its map's with its id the port entry the map names "
        "(`IoTieContent`): the I/O tie of one egress link-window, every cell opened")


# --- arguments -------------------------------------------------------------------------------------------------------

def content_arguments(inst: Instance, bits: Sequence[int]) -> list[list[int]]:
    """`IoTieContent`'s arguments in its signature's order, each its leaves' bits."""
    return [[b for c in inst.cells for b in HC._bytes_bits(c)], HC._bytes_bits(map_bytes(inst.map)),
            [b for r in inst.port for b in HC._bytes_bits(r)], list(bits)]


def port_arguments(inst: Instance) -> list[list[int]]:
    """`IoTiePort`'s arguments in its signature's order, each its leaves' bits."""
    d, ds = HC._digest_bits, inst.shape.DS
    return [d(inst.port_root), d(inst.port_domain), [b for r in inst.port for b in HC._bytes_bits(r)],
            [b for s in inst.port_salts for b in d(s)], [b for s in inst.steps for b in HC._int_bits(s, ds)],
            [b for p in inst.paths for x in p for b in d(x)]]


def window_arguments(inst: Instance, bits: Sequence[int]) -> list[list[int]]:
    """`IoTieWindow`'s arguments in its signature's order, each its leaves' bits."""
    d = HC._digest_bits
    cells, mp, _, sw = content_arguments(inst, bits)
    root, dom, port, salts, steps, paths = port_arguments(inst)
    return [d(inst.cells_root), d(inst.cells_domain), cells, [b for s in inst.cell_salts for b in d(s)], root, dom, port,
            salts, steps, paths, mp, sw]


__all__ = ["CELL_OUT", "CELL_USED", "EMPTY", "FrameV3Pad", "ID_AT", "ID_EVENT_BYTES", "Instance", "IoTieCell", "IoTieContent",
           "IoTieMatch", "IoTiePort", "IoTieRoute", "IoTieWindow", "MAP", "MapEntry", "NetworkRowsRoot", "Output", "Parsed",
           "Shape", "benes", "benes_apply", "bucket_rows", "cells_match", "content_arguments", "data_cell", "domain",
           "entry_perm", "fake_salt", "id_event", "map_bytes", "open_port", "pad_cell", "parse_cell", "port_arguments",
           "port_id", "port_row", "reference", "reference_parts", "route_depth", "rows_levels", "rows_path", "rows_root",
           "switches", "window_arguments", "window_map"]
