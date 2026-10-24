"""The I/O tie (`PROTOCOL.md`, "The I/O tie"), a prototype on a fake upstream: the reference of one egress
link-window's tie, a unit at selection `All` that opens every cell, and the Boolean Definitions it is built from.
Nothing here is in a Spec or proved, and no warden writes these cells yet: the fake upstream commits them as the design
has the warden commit them, one `hm96-sha512/row/v2` row a bucket (its R cells) under a per-window `frame-v3-sha512`
root.

**The cells.** A link-window's egress is N = T·R cells of B bytes in the warden's schedule order, R a bucket, so cell
i is in bucket ⌊i / R⌋ of the window. A cell is a pad (`pad_cell`, B zero bytes) or a data cell (`data_cell`): a
`DATA` header whose stream is the request's ingress ordinal, which the warden writes from the connection it
terminated, then one token id's event (`id_event`) and zero fill. The warden releases a bucket's frames left-packed
and sorted by stream, each stream's in its order, so the order within a bucket is a function of which frames it holds.
The circuit routes M = 2^D ≥ N slots (`Shape`), the N cells and M − N pads of its own.

**The window's table** is one value per link-window, committed before the tie runs: the window's outputs in (q, j)
order, one 16-byte slot each (`Slot`: the request's ordinal q, the output's index j, its session-global egress bucket b
and its token id, u32be each), then empty slots, all 0xFF, to M slots, TS a row (`Shape.TS`: 256, a 4,096-byte row,
or M / 2 below M = 512). q, j and b are advice v0's outputs and release buckets; the id is the token port's, the
server's sampled id (`window_slots` says what the table must equal).

**What 1 means** (`reference`): both roots are their rows' (the cells' and the table's), every cell is a pad or a data
cell, every bucket is in that order (its data cells first, their streams non-decreasing), the table's full slots come
first, sorted by (q, j) with each request's j consecutive, and the data cells are in one-to-one correspondence with
the full slots, each stream's data cells in schedule order to its request's slots in order, every slot's q, id and b
its cell's stream, id and bucket (the window's first bucket plus ⌊i / R⌋). So a duplicated, dropped, reordered or
altered cell, a covert byte in a pad, and a cell on another request's stream each make it 0.

    IoTieCell{B}(cell: Array<B, Byte>) -> Array<66>                  ok, data, stream (u32), id (u32), LSB first
    IoTieOrder{T, R}(cells: Array<N, Array<33>>) -> Value<1>         each cell's data bit and stream
    IoTieRoute{W, D}(xs: Array<M, Array<W>>, switches: Array<S>) -> Array<M, Array<W>>
        a Beneš network of S = (2D − 1)·2^(D − 1) switches, W ANDs each
    IoTieMatch{D, DB}(routed: Array<M, Array<W>>, table: Array<M, Array<16, Byte>>, base: Array<32>) -> Value<1>
    IoTieContent{B, T, R}(cells: Array<N, Array<B, Byte>>, table: Array<M, Array<16, Byte>>, base: Array<32>,
        switches: Array<S>) -> Value<1>
    NetworkRowsRoot{N_BITS, COUNT}(rows: Array<COUNT, Array<N_BITS>>, salts: Array<COUNT, Salt>, root, domain) -> Value<1>
    IoTieWindow{B, T, R}(cells_root, cells_domain, cells, cell_salts, table_root, table_domain, table, table_salts,
        base, switches) -> Value<1>
        W = 65 + DB + D: a cell's data bit, stream and id, its bucket (DB = bitlen(T − 1) bits) and its index

`IoTieWindow` is the unit: it recomputes both roots from every row and salt, the cells' from T rows of R·B bytes and
the table's from M / TS rows of 16·TS. A row's digest is most of its cost and grows with its bytes, its hm96 leaf and
frame-v3 leaf and nodes don't, so rows this long hash every byte about once. `IoTieContent` is the tie without the
roots. The switches are the prover's (`switches`): any setting is a permutation of the slots, and `IoTieMatch` holds
the routed cells to the table.
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
                                                  index_bytes, leaf_ref, node_ref)
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
SLOT = struct.Struct(">IIII")
#: a table row's slots at M ≥ 512: 4,096 bytes
TABLE_ROW_SLOTS = 256
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


# --- the window and its table ----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Shape:
    """A link-window's egress: T ≥ 2 buckets of R cells of B bytes, N = T·R cells, routed and matched over M = 2^D ≥ N
    slots (D ≥ 4); the cells N … M − 1 are pads the circuit makes, not the warden. The warden's rows are the T
    buckets, the table's M / TS rows of TS slots."""

    B: int
    T: int
    R: int

    def __post_init__(self) -> None:
        _check_cell_bytes(self.B)
        for name, least in (("T", 2), ("R", 1)):
            if type(getattr(self, name)) is not int or getattr(self, name) < least:
                raise ValueError(f"{name} is an int of at least {least}, not {getattr(self, name)!r}")
        if self.N > 1 << 32:
            raise ValueError(f"a window of {self.N} cells has more than 2^32")

    @property
    def N(self) -> int:
        return self.T * self.R

    @property
    def D(self) -> int:
        return max(4, (self.N - 1).bit_length())

    @property
    def M(self) -> int:
        return 1 << self.D

    @property
    def DB(self) -> int:
        """A bucket's bits in a routed tuple."""
        return max(1, (self.T - 1).bit_length())

    @property
    def W(self) -> int:
        return 65 + self.DB + self.D

    @property
    def switches(self) -> int:
        return (2 * self.D - 1) << (self.D - 1)

    @property
    def TS(self) -> int:
        """A table row's slots: two rows at least."""
        return min(TABLE_ROW_SLOTS, self.M // 2)

    @property
    def table_rows(self) -> int:
        return self.M // self.TS

    def bucket(self, i: int) -> int:
        return i // self.R


class Slot(NamedTuple):
    """A table slot: request q's output j, released in egress bucket b, of token id `id`."""

    q: int
    j: int
    b: int
    id: int


def window_slots(outputs: Sequence[tuple[int, int]], release: Sequence[int], ids: Sequence[int], window: int,
                 T: int) -> list[Slot]:
    """What the table's full slots must be for egress window `window`: the outputs (each (q, j), q the request's
    ingress ordinal, j its output index) whose release bucket is in the window (release // T == window), sorted by
    (q, j), each with its bucket and its id. With advice v0 these are `advice.outputs(a)` (request, index) and
    `a.release`; the ids are the Commit's sampled ids of those outputs (`runner.sampler/sampled_token_ids`)."""
    if not len(outputs) == len(release) == len(ids):
        raise ValueError("one release bucket and one id an output")
    return sorted(Slot(q, j, b, x) for (q, j), b, x in zip(outputs, release, ids) if b // T == window)


def table_rows(slots: Sequence[Slot], shape: Shape) -> list[bytes]:
    """The table's rows: the slots, then empty slots (all 0xFF) to M, TS a row."""
    if len(slots) > shape.M:
        raise ValueError(f"{len(slots)} slots don't fit a table of {shape.M}")
    data = b"".join(SLOT.pack(*s) for s in slots) + b"\xff" * (SLOT.size * (shape.M - len(slots)))
    n = SLOT.size * shape.TS
    return [data[k:k + n] for k in range(0, len(data), n)]


def table_slots(rows: Sequence[bytes]) -> list[Slot]:
    """Every slot of the table's rows, empty ones included."""
    data = b"".join(rows)
    return [Slot(*SLOT.unpack_from(data, k)) for k in range(0, len(data), SLOT.size)]


def bucket_rows(shape: Shape, cells: Sequence[bytes]) -> list[bytes]:
    """The warden's rows: each bucket's R cells, R·B bytes."""
    if len(cells) != shape.N:
        raise ValueError(f"a window has {shape.N} cells, not {len(cells)}")
    return [b"".join(cells[t * shape.R:(t + 1) * shape.R]) for t in range(shape.T)]


def domain(binding: bytes, count: int) -> CommitmentDomain:
    """A dense frame-v3-sha512 domain of `count` rows, as the warden's per-window cells and the table use."""
    return CommitmentDomain(binding, -1, RangeIndexedDomain(0, count), hash="sha512")


def rows_root(domain_id: bytes, rows: Sequence[bytes], salts: Sequence[bytes]) -> bytes:
    """The dense frame-v3-sha512 root, under `domain_id`, of the rows' `hm96-sha512/row/v2` leaves, each under its
    salt, and the domain's pads to a power of two (`NetworkRowsRoot`'s reference; `tree_levels` over `domain`'s leaves
    gives the same root)."""
    if len(rows) != len(salts) or len(rows) < 2:
        raise ValueError(f"a rows root is over at least 2 rows, each with its salt, not {len(rows)} rows and "
                         f"{len(salts)} salts")
    level = [leaf_ref(domain_id, SCHEMA, HC.hiding_value(r, s), i) for i, (r, s) in enumerate(zip(rows, salts))]
    width = 1 << (len(rows) - 1).bit_length()
    level += [_hash(_PAD, domain_id, _uint(i), hash="sha512") for i in range(len(rows), width)]
    k = 0
    while len(level) > 1:
        level = [node_ref(domain_id, k, t, level[2 * t], level[2 * t + 1]) for t in range(len(level) // 2)]
        k += 1
    return level[0]


def fake_salt(seed: bytes, i: int) -> bytes:
    """A deterministic hiding salt for a fake upstream's tests (a prover draws fresh ones)."""
    out = b""
    while len(out) < hm96.SHA512.salt_bytes:
        out += hashlib.sha512(seed + i.to_bytes(8, "big") + len(out).to_bytes(4, "big")).digest()
    return out[:hm96.SHA512.salt_bytes]


@dataclass(frozen=True)
class Instance:
    """One egress link-window's tie: the shape, the window's first bucket `base` (w·T) and the two committed values,
    the cells (with one salt a bucket row, `bucket_rows`) and the table's rows, each with its salts, root and domain id
    (the roots and domain ids public, the rest witness)."""

    shape: Shape
    base: int
    cells: tuple[bytes, ...]
    cell_salts: tuple[bytes, ...]
    cells_root: bytes
    cells_domain: bytes
    rows: tuple[bytes, ...]
    row_salts: tuple[bytes, ...]
    table_root: bytes
    table_domain: bytes


# --- the reference ---------------------------------------------------------------------------------------------------

def _sorted_table(slots: Sequence[Slot], rows: Sequence[bytes]) -> int | None:
    """The full slots' count when they come first, sorted by (q, j) with each request's j consecutive, and every
    empty slot is all 0xFF; else None."""
    full = [s.q != EMPTY for s in slots]
    n = sum(full)
    if any(full[n:]) or any(b"".join(rows)[SLOT.size * m:SLOT.size * (m + 1)] != b"\xff" * SLOT.size
                            for m in range(n, len(slots))):
        return None
    for a, b in zip(slots[:n], slots[1:n]):
        if (b.j != a.j + 1) if a.q == b.q else a.q > b.q:
            return None
    return n


def _bucket_order(shape: Shape, parsed: Sequence[Parsed]) -> bool:
    """Each bucket's cells in the canonical order the warden releases them in: its data cells first (left-packed),
    their streams non-decreasing (`IoTieOrder`'s reference)."""
    for i in range(shape.N - 1):
        a, b = parsed[i], parsed[i + 1]
        if shape.bucket(i) == shape.bucket(i + 1) and (b.data and not a.data or a.data and b.data and b.stream < a.stream):
            return False
    return True


def canonical_map(shape: Shape, base: int, cells: Sequence[bytes], rows: Sequence[bytes]) -> list[int] | None:
    """The map the tie admits, cell i to slot perm[i] over the M cells (the N cells, then the circuit's pads): each
    stream's data cells in schedule order to its request's slots in order, the pads to the empty slots (in order here;
    any order of the pads is admitted too). None when there is none: a cell that is neither, a bucket out of the
    canonical order, an unsorted table, a stream and a request with other counts, or a slot whose id or bucket isn't
    its cell's."""
    parsed, slots = [parse_cell(c) for c in cells], table_slots(rows)
    n = _sorted_table(slots, rows)
    if n is None or any(p is None for p in parsed) or len(parsed) != shape.N or len(slots) != shape.M:
        return None
    if not _bucket_order(shape, parsed):
        return None
    parsed += [Parsed(False, 0, 0)] * (shape.M - shape.N)
    by_q: dict[int, list[int]] = {}
    for m in range(n):
        by_q.setdefault(slots[m].q, []).append(m)
    by_stream: dict[int, list[int]] = {}
    pads = []
    for i, p in enumerate(parsed):
        (by_stream.setdefault(p.stream, []) if p.data else pads).append(i)
    if by_q.keys() != by_stream.keys() or any(len(by_q[q]) != len(by_stream[q]) for q in by_q):
        return None
    perm = [0] * len(parsed)
    for q, ms in by_q.items():
        for i, m in zip(by_stream[q], ms):
            if slots[m].id != parsed[i].id or slots[m].b != base + shape.bucket(i):
                return None
            perm[i] = m
    for i, m in zip(pads, range(n, len(slots))):
        perm[i] = m
    return perm


def reference_parts(inst: Instance) -> dict[str, bool]:
    """Each part's verdict in the clear: the cells' root, the table's root, and the content (`IoTieContent`'s)."""
    sh = inst.shape
    if len(inst.cells) != sh.N or len(inst.rows) != sh.table_rows:
        raise ValueError(f"the instance has {len(inst.cells)} cells and {len(inst.rows)} table rows, the shape "
                         f"{sh.N} and {sh.table_rows}")
    if type(inst.base) is not int or not 0 <= inst.base <= (1 << 32) - sh.T:
        raise ValueError(f"a window's first bucket is a u32 with T buckets after it, not {inst.base!r}")
    cells = rows_root(inst.cells_domain, bucket_rows(sh, inst.cells), inst.cell_salts)
    table = rows_root(inst.table_domain, inst.rows, inst.row_salts)
    return {"cells_root": cells == inst.cells_root, "table_root": table == inst.table_root,
            "content": all(len(c) == sh.B for c in inst.cells) and canonical_map(sh, inst.base, inst.cells, inst.rows) is not None}


def reference(inst: Instance) -> bool:
    """`IoTieWindow`'s output: every part's."""
    return all(reference_parts(inst).values())


# --- the prover's routing --------------------------------------------------------------------------------------------

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


def switches(inst: Instance) -> list[int]:
    """The prover's switch bits: the canonical map's, or, where none exists, a permutation that pairs each stream's
    cells with its request's slots as far as they go, so the circuit's refusal is the checks', not the routing's."""
    sh = inst.shape
    perm = canonical_map(sh, inst.base, inst.cells, inst.rows)
    if perm is None:
        parsed = [parse_cell(c) for c in inst.cells] + [Parsed(False, 0, 0)] * (sh.M - sh.N)
        slots, free, perm = table_slots(inst.rows), set(range(sh.M)), [-1] * sh.M
        for i, p in enumerate(parsed):
            want = [m for m in sorted(free) if p is not None and (slots[m].q == p.stream if p.data else slots[m].q == EMPTY)]
            if want:
                perm[i] = want[0]
                free.discard(want[0])
        rest = sorted(free)
        for i in range(sh.M):
            if perm[i] < 0:
                perm[i] = rest.pop(0)
    return benes(perm)


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


def _sig_match(S):
    if type(S.D) is not int or not 4 <= S.D <= 32 or type(S.DB) is not int or not 1 <= S.DB <= 32:
        raise ValueError(f"a match has 4 <= D <= 32 and 1 <= DB <= 32, not D={S.D!r}, DB={S.DB!r}")
    return ((("routed", Array(1 << S.D, Array(65 + S.DB + S.D, BIT))), ("table", Array(1 << S.D, Array(SLOT.size, BYTE))),
             ("base", Array(32, BIT))), BIT)


def _eq(C, x: Sequence, y: Sequence):
    return and_reduce(C, [(a ^ b).inv() for a, b in zip(x, y, strict=True)])


def _lt(C, x: Sequence, y: Sequence):
    """x < y, unsigned, least significant bit first: 1 AND a bit."""
    lt = ZERO
    for a, b in zip(x, y, strict=True):
        lt = lt ^ C.AND(a ^ b, lt ^ b)
    return lt


def _build_match(C, S, *e):
    M = 1 << S.D
    routed, table, base = e[:M], e[M:M + SLOT.size * M], e[-1]

    def field(m: int, f: int) -> list:
        at = SLOT.size * m + 4 * f
        return [table[at + 3 - j // 8][j % 8] for j in range(32)]

    oks, prev = [], None
    for m in range(M):
        r = routed[m]
        d, rq, rid, rb, ri = r[0], r[1:33], r[33:65], r[65:65 + S.DB], r[65 + S.DB:]
        q, j, b, x = (field(m, f) for f in range(4))
        f = and_reduce(C, q).inv()
        ones = and_reduce(C, [table[SLOT.size * m + k][t] for k in range(SLOT.size) for t in range(8)])
        hit = _eq(C, rq + rid + _add(C, list(base), [*rb, *[ZERO] * (32 - S.DB)]), q + x + b)
        oks += [(d ^ f).inv(), C.AND(f, hit.inv()).inv(), C.AND(f.inv(), ones.inv()).inv()]
        if prev is not None:
            pf, pq, pj, pi = prev
            inc, c = [], ONE
            for t in pj:
                inc.append(t ^ c)
                c = C.AND(t, c)
            same = _eq(C, pq, q)
            then = C.AND(C.AND(_eq(C, inc, j), c.inv()), _lt(C, pi, ri))
            q_lt = _lt(C, pq, q)
            pair = q_lt ^ C.AND(same, then ^ q_lt)
            oks += [C.AND(f, pf.inv()).inv(), C.AND(C.AND(pf, f), pair.inv()).inv()]
        prev = (f, q, j, ri)
    return [and_reduce(C, oks)]


IoTieMatch = _part(
    "IoTieMatch", ("D", "DB"), _sig_match, _build_match,
    "1 iff each of the 2^D routed cells (data, stream, id, bucket, index) is at a full table slot exactly when it is a "
    "data cell, a full slot's q, id and b its cell's stream, id and base + bucket, an empty slot (q = 2^32 − 1) all "
    "ones, the full slots first, and each next full slot of the same request at j + 1 with a later cell, of another "
    "request at a larger q")


def _sig_order(S):
    for name in ("T", "R"):
        if type(getattr(S, name)) is not int or getattr(S, name) < 1:
            raise ValueError(f"{name} is a positive int, not {getattr(S, name)!r}")
    return (("cells", Array(S.T * S.R, Array(33, BIT))),), BIT


def _build_order(C, S, *cells):
    oks = []
    for i in range(S.T * S.R - 1):
        if i // S.R == (i + 1) // S.R:
            (da, qa), (db, qb) = (cells[i][0], cells[i][1:]), (cells[i + 1][0], cells[i + 1][1:])
            oks += [C.AND(da.inv(), db).inv(), C.AND(C.AND(da, db), _lt(C, qb, qa)).inv()]
    return [and_reduce(C, oks)]


IoTieOrder = _part(
    "IoTieOrder", ("T", "R"), _sig_order, _build_order,
    "1 iff each bucket of R of the T·R cells (each its data bit and stream, LSB first) holds its data cells first and "
    "their streams non-decreasing: the canonical order the warden releases a bucket in")


def _sig_content(S):
    sh = Shape(S.B, S.T, S.R)
    return ((("cells", Array(sh.N, Array(S.B, BYTE))), ("table", Array(sh.M, Array(SLOT.size, BYTE))),
             ("base", Array(32, BIT)), ("switches", Array(sh.switches, BIT))), BIT)


def _body_content(B, S, cells, table, base, switches_):
    sh = Shape(S.B, S.T, S.R)
    K, cr = HC._Consts(B), HC._refs(cells)
    oks, xs, order = [], [], []
    for i in range(sh.N):
        out = HC._refs(B.call(bind(IoTieCell, B=S.B), HC._coll(Array(S.B, BYTE), cr[8 * S.B * i:8 * S.B * (i + 1)])))
        oks.append(out[0])
        order += out[1:34]
        xs += out[1:] + K.bits(sh.bucket(i), sh.DB) + K.bits(i, sh.D)
    for i in range(sh.N, sh.M):
        xs += K.bits(0, CELL_OUT - 1 + sh.DB) + K.bits(i, sh.D)
    oks += HC._refs(B.call(bind(IoTieOrder, T=S.T, R=S.R), HC._coll(Array(sh.N, Array(33, BIT)), order)))
    routed = B.call(bind(IoTieRoute, W=sh.W, D=sh.D), HC._coll(Array(sh.M, Array(sh.W, BIT)), xs), switches_)
    oks += HC._refs(B.call(bind(IoTieMatch, D=sh.D, DB=sh.DB), routed, table, base))
    return B.call(BL.and_gate(len(oks)), HC._coll(Array(len(oks), BIT), oks))


IoTieContent = CompositeDefinition(
    "IoTieContent", 1, ("B", "T", "R"), _sig_content, _body_content, conformance="exact-model-tested",
    doc="1 iff every one of the window's N = T·R B-byte cells is a pad or a data cell (`IoTieCell`), each bucket is in "
        "the canonical order (`IoTieOrder`), and the cells, each with its bucket ⌊i / R⌋ and index i, and M − N pads "
        "after them, routed by the switches (`IoTieRoute`), match the table (`IoTieMatch`)")


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


def _sig_window(S):
    sh = Shape(S.B, S.T, S.R)
    return ((("cells_root", DIGEST), ("cells_domain", DIGEST), ("cells", Array(sh.N, Array(S.B, BYTE))),
             ("cell_salts", Array(sh.T, SALT)), ("table_root", DIGEST), ("table_domain", DIGEST),
             ("table", Array(sh.table_rows, Array(SLOT.size * sh.TS, BYTE))), ("table_salts", Array(sh.table_rows, SALT)),
             ("base", Array(32, BIT)), ("switches", Array(sh.switches, BIT))), BIT)


def _body_window(B, S, cells_root, cells_domain, cells, cell_salts, table_root, table_domain, table, table_salts, base,
                 switches_):
    sh = Shape(S.B, S.T, S.R)
    cell_row, table_row = 8 * S.B * S.R, 8 * SLOT.size * sh.TS
    oks = [B.call(bind(NetworkRowsRoot, N_BITS=cell_row, COUNT=sh.T), cells.reshape(Array(sh.T, Array(cell_row, BIT))),
                  cell_salts, cells_root, cells_domain),
           B.call(bind(NetworkRowsRoot, N_BITS=table_row, COUNT=sh.table_rows),
                  table.reshape(Array(sh.table_rows, Array(table_row, BIT))), table_salts, table_root, table_domain),
           B.call(bind(IoTieContent, B=S.B, T=S.T, R=S.R), cells, table.reshape(Array(sh.M, Array(SLOT.size, BYTE))),
                  base, switches_)]
    bits = [r for o in oks for r in HC._refs(o)]
    return B.call(BL.and_gate(3), HC._coll(Array(3, BIT), bits))


IoTieWindow = CompositeDefinition(
    "IoTieWindow", 1, ("B", "T", "R"), _sig_window, _body_window, conformance="exact-model-tested",
    doc="1 iff the cells' bucket rows are the cells root's (`NetworkRowsRoot`), the table's rows the table root's, and "
        "the cells match the table (`IoTieContent`): the I/O tie of one egress link-window, every cell opened")


# --- arguments -------------------------------------------------------------------------------------------------------

def content_arguments(inst: Instance, bits: Sequence[int]) -> list[list[int]]:
    """`IoTieContent`'s arguments in its signature's order, each its leaves' bits."""
    return [[b for c in inst.cells for b in HC._bytes_bits(c)], [b for r in inst.rows for b in HC._bytes_bits(r)],
            HC._int_bits(inst.base, 32), list(bits)]


def window_arguments(inst: Instance, bits: Sequence[int]) -> list[list[int]]:
    """`IoTieWindow`'s arguments in its signature's order, each its leaves' bits."""
    d = HC._digest_bits
    cells, rows, base, sw = content_arguments(inst, bits)
    return [d(inst.cells_root), d(inst.cells_domain), cells, [b for s in inst.cell_salts for b in d(s)],
            d(inst.table_root), d(inst.table_domain), rows, [b for s in inst.row_salts for b in d(s)], base, sw]


__all__ = ["CELL_OUT", "CELL_USED", "EMPTY", "FrameV3Pad", "ID_AT", "ID_EVENT_BYTES", "Instance", "IoTieCell", "IoTieContent",
           "IoTieMatch", "IoTieOrder", "IoTieRoute", "IoTieWindow", "NetworkRowsRoot", "Parsed", "SLOT", "Shape", "Slot",
           "TABLE_ROW_SLOTS", "benes", "benes_apply", "bucket_rows", "canonical_map", "content_arguments", "data_cell",
           "domain", "fake_salt", "id_event", "pad_cell", "parse_cell", "reference", "reference_parts", "rows_root",
           "switches", "table_rows", "table_slots", "window_arguments", "window_slots"]
