"""IoTieRequest, a prototype on the warden's own ingress: the unit a
bucket row that binds each request's prompt_len and arrival ordinal to the warden's certified ingress cells, its
reference in the clear and the Boolean Definitions it is built from. Nothing here is in a Spec or proved.

**The cells are the warden's** (`warden.Warden._ingress`, its rows as `warden_rows` registers them, a row a bucket).
A link-window's ingress is T rows, row t bucket t's L·R slots of B bytes, R the warden's r. A bucket's live
connections, admitted and not past their ingress `END`, hold R slots each, ranked by ordinal (lane ρ: slots ρ·R …
ρ·R + R − 1), and a slot no connection holds is B zero bytes. Connection q's cells are its client's bytes cut as
`warden.cells` cuts them: byte k in its request cell ⌊k / P⌋, a `DATA` of exactly P = B − 9 bytes while more follow,
then an `END` with the last 1 … P; and a `PAD` on q (no payload) where fewer than P of its bytes wait and its client
hasn't ended, and after its `END`. So a bucket's R cells on q are `DATA` cells, then its `END` or none, then `PAD`s.
The bytes are a canonical id request (`active.id_request` at offset 0: the cells carry the bytes alone, no LENGTH),
whose every field is at a place fixed by the prompt's length and B alone, and whose length is its content-length plus
`BODY_AT`. Each cell is parsed without its request's prompt length (`parse_frame`): the head and the fixed fields at
their places, 11-byte prompt fields (`[` or `,`, then a u32 in 10 characters), the gap spaces only where the next
piece would straddle a cell, and `]}` then zero fill at the request's last byte. So each request's prompt_len is its
count of prompt fields, read from the certified bytes. A name (`model`, `user`) is printable ASCII but `"` and `\\`
in quotes, so a request whose name needs a JSON escape is refused.

**The lanes** (`Lane`, public in the named gap): each lane's connection, its ordinal q (0 for a lane no connection
holds), and its request's carry-in and carry-out, (bytes left, prompt fields so far) before and after the bucket's
cells: (0, 0) before its first cell and (0, n) after its `END`, n its prompt_len. The verifier computes them (`lanes`)
from the advice's prompt lengths, advice request i on ordinal order[i] + 1 (advice v1's order permutation; v0's is
the identity), and the warden's ingress release (`Release`: each connection's first bucket and its request cells a
bucket), which it takes in the clear for now as it takes the ingress record's wiring (the named gap). So connection
q's lanes chain by construction: (0, 0) into its first bucket, each bucket's carry-out the next one's carry-in, and
(0, n_q) out of its last, and the link's bucket verdicts hold every request's prompt_len to its cells, whatever
buckets it spans (a carry checked alone would accept any added count).

**What 1 means** (`reference`): in each lane, the R cells read from the carry-in are pads and request cells on its
ordinal, no request cell after a pad or after the request's `END`, each request cell that cell of a canonical id
request with its header's length (P for a `DATA`, the bytes left for the `END`), leaving the carry-out with no count
past 2^32; a lane of ordinal 0 is B zero bytes a slot. That the row is row t of the warden's registered root is the
proof system's registered read, outside the unit (`warden_rows.bucket_port`).

    IoTieRequestFrame{B}(cell: Array<B, Byte>, left, q: Array<32>) -> Array<68>
        pad, req, start, ended, the bytes left after it (u32) and its prompt fields (u32), LSB first
    IoTieRequestStep(state: Array<67>, frame: Array<68>) -> Array<67>
        state: left, fields (u32 each), over, padded, ok
    IoTieRequestEnd(state: Array<67>, left_out, fields_out: Array<32>) -> Value<1>
    IoTieRequest{B, L, R}(row: Array<8·L·R·B>, lanes: Array<L, Array<160>>) -> Value<1>
        a lane: q, left, fields, left_out, fields_out, u32 each, LSB first

`IoTieRequest` is the unit, one a bucket row: in each lane, one `IoTieRequestFrame` and one `IoTieRequestStep` a cell
from the carry-in, then `IoTieRequestEnd` against the carry-out. The frame reads its cell's header (kind, ordinal
against q, length) and its payload against the bytes its request has left (0: the request starts here) both ways, a
first cell's reading and a continuation's, each with its slots at fixed places and `]}` where the bytes left put it.
The step keeps (left, fields, over, padded, ok), and a request cell after a pad or the `END` drops ok. Public: the
lanes. Witness: the row, which the registered read holds to the warden's root.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import OR, ONE, ZERO, add, and_reduce, const_bits, mux, or_reduce, sub
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.types import Array
from verity.ml.commitments.gates.frame_v3 import _part
from verity.core.protocols.certifier import head_check as HC
from verity.core.protocols.certifier import io_tie as IT
from verity.core.protocols.certifier import warden as W
from verity.core.protocols.certifier.active import (DATA, HEADER, ID_DIGITS, ID_FIELD_BYTES, ID_NAME_BYTES,
                                                   ID_REQUEST_HEAD, PAD, id_request)
from verity.core.protocols.certifier.grid import Certifier

BIT, BYTE = HC.BIT, HC.BYTE
END = b"]}"
#: the variable fields of an id request's head: (key, width, kind)
FIELDS = ((b'{"max_tokens":', ID_DIGITS, "u32"), (b',"temperature":', 8, "ratio"), (b',"top_p":', 8, "ratio"),
          (b',"seed":', ID_DIGITS, "u32"), (b',"model":', ID_NAME_BYTES, "name"),
          (b',"user":', ID_NAME_BYTES, "name"))
#: the body's offset in an id request: `ID_REQUEST_HEAD`, the content-length in `ID_DIGITS`, then CRLF CRLF
BODY_AT = len(ID_REQUEST_HEAD) + ID_DIGITS + 4
#: `IoTieRequestFrame`'s output: pad, req, start, ended, left and fields
FRAME_OUT = 4 + 32 + 32
#: `IoTieRequestStep`'s state: left, fields, over, padded, ok
STATE = 32 + 32 + 1 + 1 + 1
#: a lane's bits: q, left, fields, left_out, fields_out
LANE = 5 * 32


class Layout(NamedTuple):
    """An id request's first-cell layout at cell size B: `room` = P payload bytes a cell, the head's bytes up to the
    first prompt field (`head`, the zero request's), its variable spans (the content-length first), and `start`, the
    first prompt field's offset in the payload."""

    room: int
    head: bytes
    spans: tuple[tuple[int, int, str], ...]
    start: int


@lru_cache(maxsize=None)
def layout(B: int) -> Layout:
    """`Layout` at cell size B. Raises `ValueError` where the head and one prompt field don't fit the first cell
    unmoved."""
    room = B - HEADER.size
    try:
        zero = id_request(model="", user="", prompt=[0], max_tokens=0, frame_bytes=B, offset=0)
    except ValueError as e:
        raise ValueError(f"a {B}-byte cell has no room for an id request: {e}") from None
    spans = [(len(ID_REQUEST_HEAD), ID_DIGITS, "content-length")]
    at = BODY_AT
    for key, width, kind in FIELDS:
        if zero[at:at + len(key)] != key:
            raise ValueError(f"at {B}-byte cells an id request's head doesn't fit its first cell unmoved")
        spans.append((at + len(key), width, kind))
        at += len(key) + width
        if key == b',"seed":':
            at += len(b',"stop":[],"stream":true')
    start = zero.index(b',"prompt":') + len(b',"prompt":')
    if start + ID_FIELD_BYTES > room or zero[start:start + 1] != b"[":
        raise ValueError(f"at {B}-byte cells an id request's first prompt field isn't in its first cell")
    return Layout(room, zero[:start], tuple(spans), start)


def _u32(field: bytes) -> bool:
    s = field.lstrip(b" ")
    return len(field) == ID_DIGITS and 0 < len(s) and s.isdigit() and (s == b"0" or s[:1] != b"0") and \
        int(s) < 1 << 32


def _ratio(field: bytes) -> bool:
    """`%8.6f` of a number in [0, 10): a digit, `.`, six digits."""
    return len(field) == 8 and field[1:2] == b"." and (field[:1] + field[2:]).isdigit()


def _name(field: bytes) -> bool:
    """A JSON string of printable ASCII but `"` and `\\`, padded with spaces."""
    raw = field.rstrip(b" ")
    return len(field) == ID_NAME_BYTES and len(raw) >= 2 and raw[:1] == raw[-1:] == b'"' and \
        all(0x20 <= c < 0x7F and c not in b'"\\' for c in raw[1:-1])


CHECK = {"u32": _u32, "ratio": _ratio, "name": _name}


class Frame(NamedTuple):
    """A request cell's parse: its request's length (a first cell's, from its content-length, else None), its prompt
    fields and the request bytes it holds."""

    length: int | None
    fields: int
    used: int


def parse_frame(lay: Layout, payload: bytes, first: bool, left: int) -> Frame | None:
    """The parse of one request cell's payload (P bytes, zero fill past the request's last byte), `left` the request's
    bytes still to come before it (ignored on a first cell, whose content-length says), or None for a payload that
    isn't that cell of a canonical id request."""
    if len(payload) != lay.room:
        return None
    if first:
        for i, (a, b) in enumerate(zip(payload[:lay.start], lay.head)):
            if a != b and not any(o <= i < o + w for o, w, _ in lay.spans):
                return None
        for o, w, kind in lay.spans[1:]:
            if not CHECK[kind](payload[o:o + w]):
                return None
        cl = payload[lay.spans[0][0]:lay.spans[0][0] + ID_DIGITS]
        if not _u32(cl) or int(cl) + BODY_AT >= 1 << 32:
            return None
        length = left = int(cl) + BODY_AT
        at = lay.start
    else:
        length, at = None, 0
    if left <= 0:
        return None
    end = min(left, lay.room)
    fields = 0
    while at + ID_FIELD_BYTES <= end and payload[at:at + 1] == (b"[" if first and at == lay.start else b","):
        if not _u32(payload[at + 1:at + ID_FIELD_BYTES]):
            return None
        at, fields = at + ID_FIELD_BYTES, fields + 1
    if first and not fields:
        return None
    rest = left - end
    if rest:
        gap = payload[at:]
        if gap.strip(b" ") or len(gap) >= (len(END) if rest == len(END) else ID_FIELD_BYTES):
            return None
    elif payload[at:end] != END or payload[end:].strip(b"\0"):
        return None
    return Frame(length, fields, end)


#: `read`'s reading of a `PAD` on the lane's ordinal
PADDED = "pad"


def read(lay: Layout, cell: bytes, q: int, left: int) -> Frame | str | None:
    """A cell in a lane of ordinal q read against its request's bytes left: `PADDED` for a `PAD` on q (B zero bytes for
    q = 0), its `Frame` for a request cell on q (a `DATA` of P bytes that doesn't end the request or its `END` of the
    bytes left), else None."""
    B = lay.room + HEADER.size
    if len(cell) != B:
        return None
    kind, ordinal, n = HEADER.unpack_from(cell)
    if ordinal != q:
        return None
    if cell == W.cell(PAD, q, b"", B):
        return PADDED
    if not q or kind not in (DATA, W.END):
        return None
    p = parse_frame(lay, cell[HEADER.size:], not left, left)
    if p is None:
        return None
    before = left or p.length
    return p if (kind, n) == ((W.END, before) if before <= lay.room else (DATA, lay.room)) else None


def lane_cut(lay: Layout, cells: Sequence[bytes], q: int, left: int, fields: int) -> tuple[int, int] | None:
    """A lane's cells read from the carry (left, fields): the carry after them, or None where a cell isn't a pad or a
    request cell on q, or a request cell follows a pad or its request's `END`."""
    padded = False
    for c in cells:
        got = read(lay, c, q, left)
        if got is None:
            return None
        if got == PADDED:
            padded = True
            continue
        if padded or (not left and fields):
            return None
        left = (left or got.length) - got.used
        fields += got.fields
    return left, fields


class Lane(NamedTuple):
    """A lane of a bucket's view (public in the named gap): its connection's ordinal (0: no connection's) and its
    request's carry-in and carry-out, (bytes left, prompt fields so far)."""

    q: int
    left: int
    fields: int
    left_out: int
    fields_out: int


#: a lane no connection holds
UNHELD = Lane(0, 0, 0, 0, 0)


@dataclass(frozen=True)
class Shape:
    """An ingress bucket row: L lanes of R cells of B bytes (N = L·R slots), R the warden's r."""

    B: int
    L: int
    R: int

    def __post_init__(self) -> None:
        if not all(type(x) is int for x in (self.B, self.L, self.R)) or self.L < 1 or self.R < 1 or \
                self.N >= 1 << 32:
            raise ValueError(f"an ingress row has L ≥ 1 lanes of R ≥ 1 cells, fewer than 2^32, not {self!r}")
        layout(self.B)

    @property
    def N(self) -> int:
        return self.L * self.R


class Instance(NamedTuple):
    """One ingress bucket row's tie: its cells (witness, the registered row's slots) and its lanes (public)."""

    cells: tuple[bytes, ...]
    lanes: tuple[Lane, ...]


def reference(shape: Shape, inst: Instance) -> int:
    """IoTieRequest's output (the module docstring, "What 1 means")."""
    if len(inst.cells) != shape.N or len(inst.lanes) != shape.L:
        return 0
    lay, R = layout(shape.B), shape.R
    for k, ln in enumerate(inst.lanes):
        got = lane_cut(lay, inst.cells[k * R:(k + 1) * R], ln.q, ln.left, ln.fields)
        if got is None or got[1] >= 1 << 32 or got != (ln.left_out, ln.fields_out):
            return 0
    return 1


# --- the verifier's lanes --------------------------------------------------------------------------------------------

def request_bytes(prompt_len: int, B: int) -> int:
    """The bytes of an id request of a `prompt_len`-token prompt in the warden's cells."""
    return len(id_request(model="", user="", prompt=[0] * prompt_len, max_tokens=0, frame_bytes=B, offset=0))


@lru_cache(maxsize=4096)
def carries(prompt_len: int, B: int) -> tuple[tuple[int, int], ...]:
    """The cut's (bytes left, prompt fields) after each request cell 0, 1, … of an id request of a `prompt_len`-token
    prompt, from (0, 0) before its first: one more than its request cells, the last (0, prompt_len)."""
    lay = layout(B)
    raw = W.cells(id_request(model="", user="", prompt=[0] * prompt_len, max_tokens=0, frame_bytes=B, offset=0), B, 1)
    out, left, fields = [(0, 0)], 0, 0
    for k in range(0, len(raw), B):
        p = parse_frame(lay, raw[k + HEADER.size:k + B], not left, left)
        left, fields = (left or p.length) - p.used, fields + p.fields
        out.append((left, fields))
    return tuple(out)


class Release(NamedTuple):
    """Connection q's ingress in the warden's record: admitted at bucket `first` (the session's), its request cells
    (`DATA` and its `END`) in each bucket from it on, its `END` in the last."""

    q: int
    first: int
    cells: tuple[int, ...]


def lanes(shape: Shape, prompt_lens: Sequence[int], releases: Sequence[Release], bucket: int,
          order: Sequence[int] | None = None) -> tuple[Lane, ...]:
    """The verifier's lanes of session bucket `bucket` (the module docstring, "The lanes"): its live connections by
    ordinal, connection q = order[i] + 1 carrying advice request i's prompt_lens[i], then unheld lanes. Raises
    `ValueError` where the releases aren't the warden's schedule (ordinals 1, 2, … admitted in order, at most L live, R
    request cells a bucket), the connections aren't the advice's requests one to one, or a connection's cells aren't
    its request's, ending in its last bucket."""
    order = list(range(len(prompt_lens))) if order is None else list(order)
    if sorted(order) != list(range(len(prompt_lens))):
        raise ValueError("the order is a permutation of the advice's requests")
    if len(releases) != len(prompt_lens):
        raise ValueError(f"{len(releases)} connections for the advice's {len(prompt_lens)} requests: each request is "
                         "one connection's")
    n_of = {o + 1: prompt_lens[i] for i, o in enumerate(order)}
    last_first, live = -1, []
    for k, rel in enumerate(releases):
        if rel.q != k + 1 or rel.first < last_first:
            raise ValueError(f"connection {rel.q} where {k + 1} is next, admitted at {rel.first}: the ordinals are "
                             "1, 2, … admitted in order")
        last_first = rel.first
        if rel.q not in n_of:
            raise ValueError(f"connection {rel.q} carries no advice request")
        steps = carries(n_of[rel.q], shape.B)
        if not rel.cells or sum(rel.cells) != len(steps) - 1 or not rel.cells[-1] or \
                any(not 0 <= c <= shape.R for c in rel.cells):
            raise ValueError(f"connection {rel.q}'s cells {rel.cells} aren't its request's {len(steps) - 1}, at most "
                             f"R = {shape.R} a bucket and its END in its last")
        k_ = bucket - rel.first
        if 0 <= k_ < len(rel.cells):
            j = sum(rel.cells[:k_])
            live.append(Lane(rel.q, *steps[j], *steps[j + rel.cells[k_]]))
    if len(live) > shape.L:
        raise ValueError(f"bucket {bucket} has {len(live)} live connections, past L = {shape.L}")
    return tuple(live) + (UNHELD,) * (shape.L - len(live))


# --- the fake upstream: the warden's own ingress ---------------------------------------------------------------------

class Link(NamedTuple):
    """A link's ingress served through the warden: its rows, a bucket each (a slot no connection holds B zero bytes),
    each connection's release, and the ingress records its grid closed, a window each."""

    rows: list[list[bytes]]
    releases: list[Release]
    records: tuple = ()


def serve(params: W.WardenParams, requests: Sequence[bytes], arrivals: Sequence[int], rate: Sequence[int],
          buckets: int, hold: int = 0) -> Link:
    """Connection k, ordinal k + 1, arrives at bucket arrivals[k] with request requests[k], and the warden's own ingress
    (`warden.Warden._ingress`) serves it: admitted in arrival order at the first bucket with a free lane, which it
    holds until `hold` buckets after its ingress `END` (standing in for its egress), its client's bytes arriving
    rate[k] a bucket from its admission (its `END` with the last)."""
    B = params.cell_bytes
    warden = SimpleNamespace(params=params, requests_released=0, run_id=None)
    conns = [W._Connection(ordinal=k + 1, reader=None, writer=None, done=None, arrived=a)
             for k, a in enumerate(arrivals)]
    queue, active, grid, rows, records = list(conns), [], Certifier("requests", params.grid), [], []
    got: dict[int, list[int]] = {}
    for t in range(buckets):
        while queue and queue[0].arrived <= t and len(active) < params.lanes:
            c = queue.pop(0)
            c.admitted, c.pad = t, W.cell(PAD, c.ordinal, b"", B)
            active.append(c)
        for c in active:
            k = c.ordinal - 1
            sent = min(len(requests[k]), rate[k] * (t - c.admitted))
            c.ingress += requests[k][c.bytes_in:sent]
            c.bytes_in, c.ingress_ended = sent, sent == len(requests[k])
            if c.ingress_end is None:
                frames = W.Warden._ingress(warden, c, t)
                got.setdefault(c.ordinal, []).append(sum(f[0] in (DATA, W.END) for f in frames))
                for f in frames:
                    grid.enqueue(f)
        active = [c for c in active if c.ingress_end is None or t < c.ingress_end + hold]
        rows.append([bytes(B) if f is None else f for f in grid.tick()])
        if (t + 1) % params.window == 0:
            records.append(grid.close_window())
    if queue or any(c.ingress_end is None for c in conns):
        raise ValueError("every connection ends its ingress within the link's buckets")
    return Link(rows, [Release(c.ordinal, c.admitted, tuple(got[c.ordinal])) for c in conns], tuple(records))


# --- the circuit -----------------------------------------------------------------------------------------------------

#: the largest u32 in an id request's 10 characters
U32_MAX = b"%10d" % ((1 << 32) - 1)


class _Chars:
    """Predicates of a cell's payload bytes (8 bits each, least significant first), each built once."""

    def __init__(self, C, data: Sequence[Sequence]) -> None:
        self.C, self.data, self.memo = C, data, {}

    def eq(self, i: int, v: int):
        if (i, v) not in self.memo:
            self.memo[i, v] = and_reduce(self.C, [HC._same(self.data[i][b], v >> b & 1) for b in range(8)])
        return self.memo[i, v]

    def digit(self, i: int):
        if (i, "d") not in self.memo:
            # 0x30 ... 0x39: high nibble 0011, low nibble at most 9 (not b3 with b2 or b1)
            C, c = self.C, self.data[i]
            self.memo[i, "d"] = C.AND(and_reduce(C, [c[7].inv(), c[6].inv(), c[5], c[4]]),
                                      C.AND(c[3], C.AND(c[2].inv(), c[1].inv()).inv()).inv())
        return self.memo[i, "d"]

    def nibble0(self, i: int):
        if (i, "0") not in self.memo:
            self.memo[i, "0"] = and_reduce(self.C, [self.data[i][b].inv() for b in range(4)])
        return self.memo[i, "0"]


def _eq(C, x: Sequence, y: Sequence):
    return and_reduce(C, [(a ^ b).inv() for a, b in zip(x, y, strict=True)])


def _eq_const(C, x: Sequence, k: int):
    return and_reduce(C, [HC._same(b, k >> j & 1) for j, b in enumerate(x)]) if k < 1 << len(x) else ZERO


def _gt_const(C, x: Sequence, k: int):
    """x > k, x least significant bit first: one AND a bit."""
    gt = ZERO
    for j, b in enumerate(x):
        gt = C.AND(b, gt) if k >> j & 1 else OR(C, b, gt)
    return gt


def _u32_ok(C, ch: _Chars, at: int) -> list:
    """The checks that the 10 characters at `at` are a u32 as `_u32` reads one: spaces then digits, at least one, the
    first nonzero unless it is the last, and, all ten digits, at most `U32_MAX` (compared digit by digit)."""
    sp = [ch.eq(at + i, 0x20) for i in range(ID_DIGITS)]
    dg = [ch.digit(at + i) for i in range(ID_DIGITS)]
    oks = []
    for i in range(ID_DIGITS):
        oks.append(sp[i] ^ dg[i])
        if i:
            oks.append(C.AND(dg[i - 1], sp[i]).inv())
        if i < ID_DIGITS - 1:
            lead = dg[i] if i == 0 else C.AND(dg[i], sp[i - 1])
            oks.append(C.AND(lead, ch.nibble0(at + i)).inv())
    oks.append(dg[-1])
    gt, same = ZERO, ONE
    for i in range(ID_DIGITS):
        nib, k = ch.data[at + i][:4], U32_MAX[i] & 0xF
        gt = gt ^ C.AND(same, _gt_const(C, nib, k))
        same = C.AND(same, _eq_const(C, nib, k))
    return oks + [C.AND(dg[0], gt).inv()]


def _ratio_ok(C, ch: _Chars, at: int) -> list:
    return [ch.digit(at), ch.eq(at + 1, 0x2E)] + [ch.digit(at + i) for i in range(2, 8)]


def _name_ok(C, ch: _Chars, at: int) -> list:
    """`_name`: a quote, printable ASCII but `"` and `\\` up to the closing quote, then spaces."""
    oks, open_ = [ch.eq(at, 0x22)], ONE
    for i in range(at + 1, at + ID_NAME_BYTES):
        c, quote = ch.data[i], ch.eq(i, 0x22)
        # 0x20 ... 0x7e is b7 = 0 with b6 or b5, but 0x7f; the three exceptions exclude each other
        plain = C.AND(C.AND(c[7].inv(), OR(C, c[6], c[5])), (ch.eq(i, 0x7F) ^ quote ^ ch.eq(i, 0x5C)).inv())
        oks.append(mux(C, open_, [plain ^ quote], [ch.eq(i, 0x20)])[0])
        open_ = C.AND(open_, quote.inv())
    return oks + [open_.inv()]


_SPANS = {"u32": _u32_ok, "ratio": _ratio_ok, "name": _name_ok}


def _reading(C, ch: _Chars, zero: Sequence, lay: Layout, first: bool, L: Sequence | None) -> tuple:
    """One reading of a request cell's payload: a first cell's (`first`, L its content-length plus `BODY_AT`) or a
    continuation's (L the bytes left): (ok, ended, the bytes left after it, its prompt fields, L)."""
    room = lay.room
    s0 = lay.start if first else 0
    oks = []
    if first:
        free = {i for o, w, _ in lay.spans for i in range(o, o + w)}
        oks += [ch.eq(i, lay.head[i]) for i in range(lay.start) if i not in free]
        o = lay.spans[0][0]
        cl_ok, cl = IT._digits(C, ch.data[o:o + ID_DIGITS])
        body = add(C, cl, const_bits(BODY_AT, 32), width=32, carry_out=True)
        oks += cl_ok + [body[32].inv()]
        L = body[:32]
        for o, _, kind in lay.spans[1:]:
            oks += _SPANS[kind](C, ch, o)
    K, g = divmod(room - s0, ID_FIELD_BYTES)
    at = [s0 + ID_FIELD_BYTES * k for k in range(K + 1)]
    fields = [and_reduce(C, [ch.eq(at[k], 0x5B if first and k == 0 else 0x2C)] + _u32_ok(C, ch, at[k] + 1))
              for k in range(K)]
    last = _gt_const(C, L, room).inv()
    # not its last cell: every slot a field, then the gap's spaces, fewer than 2 if the request's `]}` is next
    whole = fields + [ch.eq(i, 0x20) for i in range(at[K], room)]
    if g >= len(END):
        whole.append(_eq_const(C, L, room + len(END)).inv())
    # its last: `]}` at the place e[k] of slot k, which ends the request; every slot before it a field, zero past it
    e = {k: _eq_const(C, L, at[k] + len(END)) for k in range(1 if first else 0, K + 1) if at[k] + len(END) <= room}
    lo, after = [ZERO], ZERO
    for k in sorted(e):
        lo[0] = lo[0] ^ e[k]
    for k in range(K, -1, -1):
        span = range(at[k], min(at[k] + ID_FIELD_BYTES, room))
        ek = e.get(k, ZERO)
        if k < K:
            lo.append(C.AND(after, fields[k].inv()).inv())
        if k in e:
            lo.append(C.AND(ek, and_reduce(C, [ch.eq(at[k], END[0]), ch.eq(at[k] + 1, END[1])] +
                                       [zero[i] for i in span[2:]]).inv()).inv())
        lo.append(C.AND((after ^ ek).inv(), and_reduce(C, [zero[i] for i in span]).inv()).inv())
        after = after ^ ek
    ok = and_reduce(C, oks + [and_reduce(C, lo) ^ C.AND(last.inv(), and_reduce(C, whole))])
    rest = [C.AND(last.inv(), b) for b in sub(C, L, const_bits(room, 32), 32)[0]]
    count = []
    for j in range(32):
        bit = ZERO
        for k in e:
            if k >> j & 1:
                bit = bit ^ e[k]
        count.append(bit ^ last.inv() if K >> j & 1 else bit)
    return ok, last, rest, count, list(L)


def _u32be(cell: Sequence, at: int) -> list:
    """The u32be at bytes at … at + 3 of a cell, least significant bit first."""
    return [cell[at + 3 - j // 8][j % 8] for j in range(32)]


def _sig_frame(S):
    layout(S.B)
    return (("cell", Array(S.B, BYTE)), ("left", Array(32, BIT)), ("q", Array(32, BIT))), Array(FRAME_OUT, BIT)


def _build_frame(C, S, *e):
    lay = layout(S.B)
    H, room = HEADER.size, lay.room
    cell, left, q = e[:S.B], list(e[S.B]), list(e[S.B + 1])
    ch = _Chars(C, cell[H:])
    zero = [ch.eq(i, 0) for i in range(room)]
    kind, length, on_q = list(cell[0]), _u32be(cell, 5), _eq(C, _u32be(cell, 1), q)
    pad = and_reduce(C, [on_q] + [b.inv() for b in kind + length] + zero)
    f_ok, f_end, f_rest, f_count, f_len = _reading(C, ch, zero, lay, True, None)
    c_ok, c_end, c_rest, c_count, _ = _reading(C, ch, zero, lay, False, left)
    start = and_reduce(C, [b.inv() for b in left])
    ok, ended, *L = mux(C, start, [f_ok, f_end, *f_len], [c_ok, c_end, *left])
    header = mux(C, ended, [C.AND(_eq_const(C, kind, W.END), _eq(C, length, L))],
                 [C.AND(_eq_const(C, kind, DATA), _eq_const(C, length, room))])[0]
    req = and_reduce(C, [or_reduce(C, q), on_q, ok, header])
    return [[pad, req, start, ended, *mux(C, start, f_rest, c_rest), *mux(C, start, f_count, c_count)]]


IoTieRequestFrame = _part(
    "IoTieRequestFrame", ("B",), _sig_frame, _build_frame,
    "(pad, req, start, ended, left, fields) of a B-byte ingress cell in a lane of ordinal q, read against its "
    "request's bytes left before it (left; 0: a request starts here, start): pad iff the cell is a PAD on q (B zero "
    "bytes for q = 0); req iff q isn't 0 and the cell is a DATA on q of B − 9 bytes, or an END on q of the bytes left "
    "where they end the request, whose payload `io_request.parse_frame` accepts as that cell of a canonical id "
    "request; ended iff the request's last byte is in it; left the bytes the request has after it and fields its "
    "prompt fields, u32 each, least significant bit first")


def _sig_step(S):
    return (("state", Array(STATE, BIT)), ("frame", Array(FRAME_OUT, BIT))), Array(STATE, BIT)


def _build_step(C, S, state, fo):
    left, fields, over, padded, ok = state[0:32], state[32:64], state[64], state[65], state[66]
    pad, req, start, rest, count = fo[0], fo[1], fo[2], fo[4:36], fo[36:68]
    done = C.AND(start, or_reduce(C, fields))
    total = add(C, fields, count, width=32, carry_out=True)
    valid = OR(C, pad, and_reduce(C, [req, padded.inv(), done.inv()]))
    return [[*mux(C, req, rest, left), *mux(C, req, total[:32], fields), OR(C, over, C.AND(req, total[32])),
             OR(C, padded, pad), C.AND(ok, valid)]]


IoTieRequestStep = _part(
    "IoTieRequestStep", (), _sig_step, _build_step,
    "a lane's scan state after one cell (`IoTieRequestFrame`'s output): its request's bytes left and prompt fields "
    "so far (over: 2^32 or more), whether a pad has been seen in the lane, and ok, which drops unless the cell is a "
    "pad, or a request cell before the lane's pads while its request hasn't ended (left 0 with fields)")
# C-Flock's lowering calls a static-free Definition's builder without S, as `block_check._part` builds them
IoTieRequestStep.build = lambda C, *e: _build_step(C, SimpleNamespace(), *e)  # type: ignore[attr-defined]


def _sig_end(S):
    return (("state", Array(STATE, BIT)), ("left_out", Array(32, BIT)), ("fields_out", Array(32, BIT))), BIT


def _build_end(C, S, state, left_out, fields_out):
    return [and_reduce(C, [state[66], state[64].inv(), _eq(C, state[0:32], left_out), _eq(C, state[32:64], fields_out)])]


IoTieRequestEnd = _part(
    "IoTieRequestEnd", (), _sig_end, _build_end,
    "1 iff a lane's scan state is ok and its request has left_out bytes left and fields_out prompt fields so far, "
    "fewer than 2^32 (over clear)")
IoTieRequestEnd.build = lambda C, *e: _build_end(C, SimpleNamespace(), *e)  # type: ignore[attr-defined]


def _sig_request(S):
    sh = Shape(S.B, S.L, S.R)
    return (("row", Array(8 * S.B * sh.N, BIT)), ("lanes", Array(S.L, Array(LANE, BIT)))), BIT


def _body_request(B, S, row, lanes_):
    K, cr, lr = HC._Consts(B), HC._refs(row), HC._refs(lanes_)
    oks = []
    for k in range(S.L):
        ln = lr[LANE * k:LANE * (k + 1)]
        q = HC._coll(Array(32, BIT), ln[0:32])
        state = HC._coll(Array(STATE, BIT), ln[32:96] + K.bits(0, 2) + K.bits(1, 1))
        for j in range(S.R):
            i = S.R * k + j
            cell = HC._coll(Array(S.B, BYTE), cr[8 * S.B * i:8 * S.B * (i + 1)])
            fo = B.call(bind(IoTieRequestFrame, B=S.B), cell, HC._coll(Array(32, BIT), HC._refs(state)[:32]), q)
            state = B.call(bind(IoTieRequestStep), state, fo)
        oks.append(B.call(bind(IoTieRequestEnd), state, HC._coll(Array(32, BIT), ln[96:128]),
                          HC._coll(Array(32, BIT), ln[128:160])))
    bits = [r for o in oks for r in HC._refs(o)]
    bits += K.bits(1, 2 - len(bits)) if len(bits) < 2 else []
    return B.call(BL.and_gate(len(bits)), HC._coll(Array(len(bits), BIT), bits))


IoTieRequest = CompositeDefinition(
    "IoTieRequest", 1, ("B", "L", "R"), _sig_request, _body_request, conformance="exact-model-tested",
    doc="1 iff each of the bucket row's L lanes, its R cells scanned from the lane's carry-in (left, fields) on its "
        "ordinal q, one `IoTieRequestFrame` and `IoTieRequestStep` a cell, is pads and canonical id-request cells on q, "
        "none after a pad or the request's END, leaving its carry-out (left_out, fields_out) with no count past 2^32 "
        "(`IoTieRequestEnd`): the ingress tie of one bucket row of the warden's registered root (the row the proof "
        "system's registered read, bit 8k + i bit i of byte k), every cell opened; a link-window's tie is its T rows'")
IoTieRequest.gated = True  # type: ignore[attr-defined]  # circuit-check holds its digest pins (`tie_service.gate`)
#: the circuit digests circuit-check pins (`head_check.pinned_digest`)
PINS = Path(__file__).with_name("io_request.circuit-check.json")


# --- arguments -------------------------------------------------------------------------------------------------------

def frame_arguments(cells: Sequence[bytes], lefts: Sequence[int], qs: Sequence[int]) -> list[list[int]]:
    """`IoTieRequestFrame`'s arguments for a batch of (cell, left, q), each its leaves' bits a row."""
    return [[HC._bytes_bits(c) for c in cells], [HC._int_bits(x, 32) for x in lefts], [HC._int_bits(x, 32) for x in qs]]


def lane_bits(lanes_: Sequence[Lane]) -> list[int]:
    """The lanes' bits as the unit reads them, u32s least significant bit first. Raises `ValueError` past a u32."""
    if any(type(x) is not int or not 0 <= x < 1 << 32 for ln in lanes_ for x in ln):
        raise ValueError(f"a lane is five u32s, not {lanes_!r:.200}")
    return [b for ln in lanes_ for x in ln for b in HC._int_bits(x, 32)]


def arguments(shape: Shape, inst: Instance) -> list[list[int]]:
    """`IoTieRequest`'s arguments in its signature's order, each its leaves' bits: the row's, then the lanes'."""
    if len(inst.cells) != shape.N or len(inst.lanes) != shape.L:
        raise ValueError(f"a row is N = {shape.N} cells and L = {shape.L} lanes")
    return [[b for c in inst.cells for b in HC._bytes_bits(c)], lane_bits(inst.lanes)]


__all__ = ["BODY_AT", "END", "FRAME_OUT", "Frame", "Instance", "IoTieRequest", "IoTieRequestEnd", "IoTieRequestFrame",
           "IoTieRequestStep", "LANE", "Lane", "Layout", "Link", "PADDED", "PINS", "Release", "STATE", "Shape",
           "UNHELD", "arguments", "carries", "frame_arguments", "lane_bits", "lane_cut", "lanes", "layout",
           "parse_frame", "read", "reference", "request_bytes", "serve"]
