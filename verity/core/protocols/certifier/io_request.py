"""IoTieRequest (`PROTOCOL.md`, "The I/O tie", "The ingress tie"), a prototype on a fake upstream: one ingress
link-window's tie, the unit at selection `All` that binds each request's prompt_len and arrival ordinal to the warden's
certified ingress cells, its reference in the clear and the Boolean Definitions it is built from. Nothing here is in a
Spec or proved, and no warden commits these cells yet.

**The cells.** A link-window's ingress is N = T·R cells of B bytes, R a bucket (the ingress allocation), committed as
the egress cells are (`io_tie.bucket_rows`, `io_tie.rows_root`). Bucket 0's first cell is the certifier's own frame
when OWN = 1 (`active.salt_frame` of the window and `PLACEHOLDER`, as the grid holds it); then each bucket holds its
request frames, then pads. A request frame is `active.request_frames`' (`DATA` on `CANONICAL_STREAM`, B − 9 bytes of
payload), and the requests are CERT-VARLEN's id requests (`active.id_request`), whose every field is at a place fixed
by the prompt's length and B alone.

**The cut** (`cut`) runs as `active.Requests` does: a request starts a frame with its LENGTH and ends at the frame
that holds its last byte. The link's k-th request (from 0) is advice request k, its ordinal; `active.Requests` streams
it as k + 1. Each frame is parsed without its request's prompt length (`parse_frame`): the head and the fixed fields at
their places, 11-byte prompt fields (`[` or `,`, then a u32 in 10 characters), the gap spaces only where the next piece
would straddle a frame, and `]}` then zero fill at the request's last byte. So each request's prompt_len is its count
of prompt fields, read from the certified bytes. A name (`model`, `user`) is printable ASCII but `"` and `\\` in
quotes, so a request whose name needs a JSON escape is refused.

**The view** (`request_view`, public in the named gap, computed by the verifier from the opened advice as
`advised.view` is): the carry-in, the request in progress at the window's first request frame (its ordinal, its bytes
left and its prompt fields so far); `ended`, the prompt lengths of the requests carry_in.q, carry_in.q + 1, … whose
last frame falls in the window; `received`, the window's request frames; and the carry-out, the request in progress
after the window's last request frame, which is the next window's carry-in. All four come from the advice's prompt
lengths and `received`. At the target the unit reads them under the advice's hiding root.

**What 1 means** (`reference`): the cells' root is their bucket rows', every cell is a pad, the own frame where it
goes or a canonical request frame, each bucket's request frames come first, the cut from the carry-in ends exactly as
many requests as `ended` names, in order, each with its prompt_len, in `received` request frames, and leaves the
carry-out's bytes left and prompt fields.

**The statements** (`statements`): window w's `RequestStatement` is the warden's registered root and domain id, w,
and `request_view` of w, so a link's statements chain (`chained`): window 0's carry-in is the link's start, (0, 0, 0),
and each carry-out is the next window's carry-in. The verifier computes them from its own advice and holds each
IoTieRequest proof to its window's; `advised.IoTieVerdict` is built from those proofs (`tie_service`), one verdict a
window, under the advice's digest. With the carry-out held, a window's verdict says the cells take its request in
progress from the carry-in to the carry-out, so a link's verdicts hold every request's prompt_len, spanning windows
or not, by the chain from window 0 (REDTEAM phase 9 (b): checked alone, a carried prompt-field count accepts any
added amount). The ordinals are the verifier's: the cells carry none, and the j-th request ending in window w is
carry_in.q + j by the same chain.

**The link and its registrations** (REDTEAM phase 10 (4)): `registered[w]` is the warden's `Registration` of window
w as the verifier read it, the link's name, the registration's digest and the root and domain id it names, and each
statement carries the link and the digest beside the root and domain id the circuit reads. `statements` takes one
link's registrations, each window its own, and `chained` holds the statements to one link and distinct
registrations, so a verdict built from them names the link and the registrations its proofs were verified on, and
`advised.view` can check both against its link's. The domain id, which the warden derives from the session, link,
direction and window, keeps a proof on one link-window's root from opening another's.

    IoTieRequestFrame{B}(cell: Array<B, Byte>, left: Array<32>) -> Array<68>
        pad, req, start, ended, the bytes left after it (u32) and its prompt fields (u32), LSB first
    IoTieRequestStep{D, NEW}(state: Array<131>, frame: Array<68>, routed: Array<32 + D>) -> Array<131>
        state: left, fields (u32 each), over, the requests ended and the request frames so far (u32 each), padded, ok
    IoTieRequestOwn{B}(cell: Array<B, Byte>, window: Array<64>) -> Value<1>
    IoTieRequestEnd(state: Array<131>, count, received, left_out, fields_out: Array<32>) -> Value<1>
    IoTieRequest{B, T, R, OWN}(root, domain, cells: Array<N, Array<B, Byte>>, salts: Array<T, Salt>,
        [window: Array<64>], left, fields: Array<32>, ended: Array<N, Array<32>>, count, received, left_out,
        fields_out: Array<32>, switches: Array<S_W>) -> Value<1>
        N = T·R, M = 2^D ≥ N, S_W = (2D − 1)·2^(D − 1); `window` only when OWN = 1

`IoTieRequest` is the unit: it recomputes the cells' root from every bucket row and salt (`io_tie.NetworkRowsRoot`),
holds the own frame to its window, and scans the cells from the carry-in, one `IoTieRequestFrame` and one
`IoTieRequestStep` a cell. The frame reads its cell against the bytes its request has left (0: a request starts here):
both readings, a first frame's and a continuation's, each with its slots at fixed places and `]}` where the bytes left
put it. The view's prompt lengths, each with its index j, are routed by the prover's switches (`io_tie.IoTieRoute`), and
the step holds the frame that ends the j-th request to entry j: its index and its count of fields. `IoTieRequestEnd`
holds the count of requests ended to `count`, of request frames to `received`, and the scan's last bytes left and
prompt fields to the carry-out's, with no count past 2^32. Public: the root and domain id, the window, and, until the
named gap closes, the carry-in's left and fields, `ended`, `count`, `received` and the carry-out's left and fields.
Witness: the cells, their salts and the switches.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple, Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import OR, ONE, ZERO, add, and_reduce, const_bits, mux, sub
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.types import Array
from verity.ml.commitments.gates.frame_v3 import _part
from verity.core.protocols.certifier import head_check as HC
from verity.core.protocols.certifier import io_tie as IT
from verity.core.protocols.certifier.active import (CANONICAL_STREAM, DATA, HEADER, ID_DIGITS, ID_FIELD_BYTES,
                                                   ID_NAME_BYTES, ID_REQUEST_HEAD, LENGTH, PLACEHOLDER,
                                                   SALT_FRAME_BYTES, id_request, id_request_frames, request_frames,
                                                   salt_frame)

BIT, BYTE, DIGEST, SALT = HC.BIT, HC.BYTE, HC.DIGEST, HC.SALT
END = b"]}"
#: the variable fields of an id request's head: (key, width, kind)
FIELDS = ((b'{"max_tokens":', ID_DIGITS, "u32"), (b',"temperature":', 8, "ratio"), (b',"top_p":', 8, "ratio"),
          (b',"seed":', ID_DIGITS, "u32"), (b',"model":', ID_NAME_BYTES, "name"),
          (b',"user":', ID_NAME_BYTES, "name"))
#: the content-length's offset from the head: `ID_REQUEST_HEAD`, the length in `ID_DIGITS`, then CRLF CRLF
BODY_AT = len(ID_REQUEST_HEAD) + ID_DIGITS + 4
#: `IoTieRequestFrame`'s output: pad, req, start, ended, left and fields
FRAME_OUT = 4 + 32 + 32
#: `IoTieRequestStep`'s state: left, fields, over, ended, frames, padded, ok
STATE = 32 + 32 + 1 + 32 + 32 + 1 + 1


class Layout(NamedTuple):
    """An id request's frame-0 layout at frame size B: `room` payload bytes a frame, the head's bytes up to the first
    prompt field (`head`, the zero request's), the variable spans in it, and `start`, the first prompt field's
    offset. Payload offsets count from the frame's payload, whose first `lead` bytes in frame 0 are LENGTH
    (`request_frames`'), or none on the warden's cells (lead 0), where the content-length gives the request's
    length."""

    room: int
    head: bytes
    spans: tuple[tuple[int, int, str], ...]
    start: int
    lead: int = LENGTH.size


def layout(B: int, lead: int = LENGTH.size) -> Layout:
    """`Layout` at frame size B, the request `lead` bytes into frame 0's payload (`active.id_request`'s `lead`: 4 or
    0). Raises `ValueError` where the head and one prompt field don't fit frame 0 unmoved."""
    if lead not in (0, LENGTH.size):
        raise ValueError(f"an id request is 0 or {LENGTH.size} bytes into its first frame, not {lead!r}")
    room = B - HEADER.size
    try:
        zero = id_request(model="", user="", prompt=[0], max_tokens=0, frame_bytes=B, lead=lead)
    except ValueError as e:
        raise ValueError(f"a {B}-byte frame has no room for an id request: {e}") from None
    payload = (LENGTH.pack(len(zero)) if lead else b"") + zero
    spans = [(0, lead, "length"), (lead + len(ID_REQUEST_HEAD), ID_DIGITS, "content-length")]
    at = lead + BODY_AT
    for key, width, kind in FIELDS:
        if payload[at:at + len(key)] != key:
            raise ValueError(f"at {B}-byte frames an id request's head doesn't fit frame 0 unmoved")
        spans.append((at + len(key), width, kind))
        at += len(key) + width
        if key == b',"seed":':
            at += len(b',"stop":[],"stream":true')
    start = payload.index(b',"prompt":[') + len(b',"prompt":')
    if start + ID_FIELD_BYTES > room or payload[start:start + 1] != b"[":
        raise ValueError(f"at {B}-byte frames an id request's first prompt field isn't in frame 0")
    return Layout(room, payload[:start], tuple(spans), start, lead)


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
    """A request frame's parse: its request's LENGTH (frame 0's, else None), its prompt fields and the request bytes
    it holds."""

    length: int | None
    fields: int
    used: int


def parse_frame(lay: Layout, payload: bytes, first: bool, left: int) -> Frame | None:
    """The parse of one request frame's payload, `left` the request's bytes still to come before it (ignored on a
    first frame, whose LENGTH says), or None for a payload that isn't that frame of a canonical id request."""
    if len(payload) != lay.room:
        return None
    if first:
        for i, (a, b) in enumerate(zip(payload[:lay.start], lay.head)):
            if a != b and not any(o <= i < o + w for o, w, _ in lay.spans):
                return None
        for o, w, kind in lay.spans[2:]:
            if not CHECK[kind](payload[o:o + w]):
                return None
        cl = payload[lay.spans[1][0]:lay.spans[1][0] + ID_DIGITS]
        if not _u32(cl) or int(cl) + BODY_AT >= 1 << 32:
            return None
        length = LENGTH.unpack(payload[:LENGTH.size])[0] if lay.lead else int(cl) + BODY_AT
        if int(cl) != length - BODY_AT:
            return None
        left, at, base = length, lay.start, lay.lead
    else:
        length, at, base = None, 0, 0
    if left <= 0:
        return None
    end = base + min(left, lay.room - base)
    fields = 0
    while at + ID_FIELD_BYTES <= end and payload[at:at + 1] == (b"[" if first and at == lay.start else b","):
        if not _u32(payload[at + 1:at + ID_FIELD_BYTES]):
            return None
        at, fields = at + ID_FIELD_BYTES, fields + 1
    if first and not fields:
        return None
    rest = left - (end - base)
    if rest:
        gap = payload[at:]
        if gap.strip(b" ") or len(gap) >= (len(END) if rest == len(END) else ID_FIELD_BYTES):
            return None
    elif payload[at:end] != END or payload[end:].strip(b"\0"):
        return None
    return Frame(length, fields, end - base)


class Carry(NamedTuple):
    """The cut's state at a frame boundary: the request in progress (its ordinal, its bytes left, its prompt fields so
    far), or, with no bytes left, the next request's ordinal."""

    q: int
    left: int
    fields: int


class RequestView(NamedTuple):
    """IoTieRequest's public inputs for one window from the advice, in the named gap: the carry-in, the prompt lengths
    of the requests carry_in.q, carry_in.q + 1, … that end in the window, the window's request frames and the
    carry-out."""

    carry_in: Carry
    ended: tuple[int, ...]
    received: int
    carry_out: Carry


class Registration(NamedTuple):
    """The warden's registration of an ingress link-window as the verifier read it (the warden's `check_registration`):
    the link's name, the registration's digest (the cells digest its certificate signs, which binds the link,
    direction, window, schedule, wiring, domain and root), and the root and domain id it names."""

    link: str
    digest: bytes
    root: bytes
    domain: bytes


class RequestStatement(NamedTuple):
    """IoTieRequest's public statement for an ingress link's window `window`: the link, the digest of the warden's
    registration of the link-window, the root and domain id it names (the circuit's), and the view (`statements`)."""

    link: str
    window: int
    registration: bytes
    root: bytes
    domain: bytes
    view: RequestView


@dataclass(frozen=True)
class Shape:
    """An ingress link-window: T buckets of R cells of B bytes, OWN the certifier's own frame at bucket 0's head (0 or
    1). At most N = T·R requests end in it, and the view's prompt lengths are routed over M = 2^D ≥ N positions."""

    B: int
    T: int
    R: int
    OWN: int = 0

    def __post_init__(self) -> None:
        if not all(type(x) is int for x in (self.B, self.T, self.R, self.OWN)) or self.T < 2 or self.R < 1 or \
                self.OWN not in (0, 1) or self.N >= 1 << 32:
            raise ValueError(f"an ingress window has T ≥ 2 buckets of R ≥ 1 cells, fewer than 2^32, and OWN 0 or 1, "
                             f"not {self!r}")
        if self.B < SALT_FRAME_BYTES:
            raise ValueError(f"a {self.B}-byte frame has no room for an id request or the own frame")
        layout(self.B)

    @property
    def N(self) -> int:
        return self.T * self.R

    @property
    def D(self) -> int:
        return IT.route_depth(self.N, self.N)

    @property
    def M(self) -> int:
        return 1 << self.D

    @property
    def W(self) -> int:
        """A routed entry's bits: a prompt length and its index."""
        return 32 + self.D

    @property
    def switches(self) -> int:
        return (2 * self.D - 1) << (self.D - 1)


def ends(prompt_lens: Sequence[int], B: int) -> tuple[int, ...]:
    """Each request's end on the ingress link, its request frames up to and including its last (`advised.ends` over
    `active.id_request_frames` at B)."""
    out, end = [], 0
    for n in prompt_lens:
        end += id_request_frames(n, B)
        out.append(end)
    return tuple(out)


def _carry_at(prompt_lens: Sequence[int], B: int, frame: int) -> Carry:
    """The cut's state before the link's request frame `frame` (counted from 0), from the prompt lengths."""
    lay, before = layout(B), 0
    for q, (n, e) in enumerate(zip(prompt_lens, ends(prompt_lens, B))):
        if frame < e:
            f = frame - before
            if not f:
                return Carry(q, 0, 0)
            raw = request_frames(id_request(model="", user="", prompt=[0] * n, max_tokens=0, frame_bytes=B), B)
            left, fields = 0, 0
            for k in range(f):
                p = parse_frame(lay, raw[k * B + HEADER.size:(k + 1) * B], k == 0, left)
                left = (p.length if k == 0 else left) - p.used
                fields += p.fields
            return Carry(q, left, fields)
        before = e
    return Carry(len(prompt_lens), 0, 0)


def request_view(prompt_lens: Sequence[int], received: Sequence[int], window: int, B: int) -> RequestView:
    """Window `window`'s view from the advice's prompt lengths, in arrival order, and each window's request frames.
    Raises `ValueError` where the requests' frames aren't the windows' total."""
    e = ends(prompt_lens, B)
    if (e[-1] if e else 0) != sum(received):
        raise ValueError(f"the requests take {e[-1] if e else 0} request frames and the windows receive "
                         f"{sum(received)}")
    lo = sum(received[:window])
    hi = lo + received[window]
    ended = tuple(n for n, x in zip(prompt_lens, e) if lo < x <= hi)
    return RequestView(_carry_at(prompt_lens, B, lo), ended, received[window], _carry_at(prompt_lens, B, hi))


def statements(shape: Shape, prompt_lens: Sequence[int], received: Sequence[int],
               registered: Sequence[Registration]) -> tuple[RequestStatement, ...]:
    """An ingress link's statements, one a window: `registered[w]`, the warden's registration of window w, and
    `request_view` of w from the advice's prompt lengths, in arrival order, and `received`. Raises `ValueError` unless
    there is one registration a window, at least one, all of one link and no two the same, and the requests' frames are
    the windows' total."""
    prompt_lens, received = _u32s(prompt_lens, "the advice's prompt lengths"), _u32s(received, "received")
    regs = tuple(registered)
    if not received or len(received) != len(regs):
        raise ValueError(f"{len(received)} windows received and {len(regs)} registered: a link's statements "
                         f"cover every window, at least one")
    if not all(isinstance(r, Registration) and type(r.link) is str and r.link and
               all(type(x) is bytes and len(x) == 64 for x in (r.digest, r.root, r.domain)) for r in regs):
        raise ValueError("each window's registration is a Registration: a link's name and 64-byte digest, root and "
                         "domain id")
    if len({r.link for r in regs}) != 1 or len({r.digest for r in regs}) != len(regs):
        raise ValueError(f"a link's statements are on one link's registrations, each window its own, not "
                         f"{sorted({r.link for r in regs})} with {len({r.digest for r in regs})} digests for "
                         f"{len(regs)} windows")
    return tuple(RequestStatement(r.link, w, r.digest, r.root, r.domain, request_view(prompt_lens, received, w, shape.B))
                 for w, r in enumerate(regs))


def chained(stmts: Sequence[RequestStatement]) -> bool:
    """Whether a link's statements chain: one link, each window its own registration, windows 0, 1, … in order,
    window 0's carry-in the link's start (0, 0, 0), each carry-out the next window's carry-in and the last one between
    requests, each carry-out's ordinal its carry-in's plus the requests that end in the window."""
    start, seen = Carry(0, 0, 0), set()
    for w, s in enumerate(stmts):
        v = s.view
        if s.link != stmts[0].link or s.registration in seen or s.window != w or v.carry_in != start or \
                v.carry_out.q != v.carry_in.q + len(v.ended):
            return False
        seen.add(s.registration)
        start = v.carry_out
    return bool(stmts) and start.left == 0 and start.fields == 0


def _u32s(value: Sequence[int], what: str) -> tuple[int, ...]:
    out = tuple(value)
    if not all(type(x) is int and 0 <= x < 1 << 32 for x in out):
        raise ValueError(f"{what} is a sequence of u32s, not {out!r:.200}")
    return out


def own_frame(window: int, B: int) -> bytes:
    """The certifier's own frame in window `window`'s grid: its salt frame carrying `PLACEHOLDER`."""
    return salt_frame(window, PLACEHOLDER, B)


class Cut(NamedTuple):
    """A window's cut: the prompt fields of each request that ends in it, its request frames, the carry after it, and
    the cells that end the requests."""

    ended: tuple[int, ...]
    frames: int
    carry: Carry
    at: tuple[int, ...]


def cut(shape: Shape, cells: Sequence[bytes], carry: Carry, window: int = 0) -> Cut | None:
    """The cut of the window's cells from `carry`, or None where a cell isn't a pad, the own frame where it goes or a
    canonical request frame, or a request frame follows a pad in its bucket."""
    lay, pad = layout(shape.B), IT.pad_cell(shape.B)
    q, left, fields = carry
    ended, at, frames, padded = [], [], 0, False
    for i, c in enumerate(cells):
        if shape.OWN and i == 0:
            if c != own_frame(window, shape.B):
                return None
            continue
        if i % shape.R == 0:
            padded = False
        if len(c) != shape.B:
            return None
        if c == pad:
            padded = True
            continue
        if padded or HEADER.unpack(c[:HEADER.size]) != (DATA, CANONICAL_STREAM, lay.room):
            return None
        p = parse_frame(lay, c[HEADER.size:], not left, left)
        if p is None:
            return None
        if p.length is not None:
            left, fields = p.length, 0
        left, fields, frames = left - p.used, fields + p.fields, frames + 1
        if not left:
            ended.append(fields)
            at.append(i)
            q, fields = q + 1, 0
    return Cut(tuple(ended), frames, Carry(q, left, fields), tuple(at))


def rows(shape: Shape, cells: Sequence[bytes]) -> list[bytes]:
    """The warden's rows: each bucket's R cells, R·B bytes (`io_tie.bucket_rows`)."""
    return [b"".join(cells[t * shape.R:(t + 1) * shape.R]) for t in range(shape.T)]


class Instance(NamedTuple):
    """One ingress link-window's tie: the cells and their salts (witness), the warden's root and domain id, the view
    and the window."""

    cells: tuple[bytes, ...]
    salts: tuple[bytes, ...]
    root: bytes
    domain: bytes
    view: RequestView
    window: int = 0


def reference(shape: Shape, inst: Instance) -> int:
    """IoTieRequest's output (the module docstring, "What 1 means")."""
    if len(inst.cells) != shape.N or len(inst.salts) != shape.T:
        return 0
    if IT.rows_root(inst.domain, rows(shape, inst.cells), inst.salts) != inst.root:
        return 0
    v = inst.view
    got = cut(shape, inst.cells, v.carry_in, inst.window)
    return int(got is not None and got.ended == v.ended and got.frames == v.received and
               got.carry[1:] == v.carry_out[1:])


# --- the fake upstream: a warden's ingress cells ---------------------------------------------------------------------

def windows_of(shape: Shape, requests: Sequence[bytes], counts: Sequence[int]) -> list[list[bytes]]:
    """Each window's N cells: the requests' frames (`active.request_frames`) released FIFO, counts[b] in session
    bucket b, after the own frame of each window's bucket 0 when OWN = 1, then pads."""
    B = shape.B
    stream = b"".join(request_frames(r, B) for r in requests)
    frames = [stream[i:i + B] for i in range(0, len(stream), B)]
    if len(counts) % shape.T or sum(counts) != len(frames):
        raise ValueError(f"{len(counts)} buckets releasing {sum(counts)} frames aren't whole windows of the "
                         f"requests' {len(frames)}")
    out, k = [], 0
    for w in range(len(counts) // shape.T):
        cells = []
        for t in range(shape.T):
            own = [own_frame(w, B)] if t == 0 and shape.OWN else []
            n = counts[w * shape.T + t]
            if len(own) + n > shape.R:
                raise ValueError(f"bucket {w * shape.T + t} releases {n} frames past the allocation {shape.R}")
            cells += own + frames[k:k + n] + [IT.pad_cell(B)] * (shape.R - len(own) - n)
            k += n
        out.append(cells)
    return out


def instance(shape: Shape, cells: Sequence[bytes], view: RequestView, window: int = 0, seed: bytes = b"in"
             ) -> Instance:
    """The warden's commitment to `cells` under fake salts, with `view`."""
    salts = tuple(IT.fake_salt(seed + window.to_bytes(4, "big"), t) for t in range(shape.T))
    dom = IT.domain(hashlib.sha512(b"ingress" + window.to_bytes(4, "big")).digest(), shape.T).domain_id
    return Instance(tuple(cells), salts, IT.rows_root(dom, rows(shape, cells), salts), dom, view, window)


def switches(shape: Shape, inst: Instance) -> list[int]:
    """The prover's routing: entry j to the cell that ends the window's j-th request, the other positions in order
    (the identity where the cut fails)."""
    got = cut(shape, inst.cells, inst.view.carry_in, inst.window)
    at = got.at if got is not None else ()
    perm, taken = list(at) + [-1] * (shape.M - len(at)), set(at)
    rest = iter(i for i in range(shape.M) if i not in taken)
    return IT.benes([p if p >= 0 else next(rest) for p in perm])


# --- the circuit -----------------------------------------------------------------------------------------------------

#: the largest u32 in an id request's 10 characters
U32_MAX = b"%10d" % ((1 << 32) - 1)


class _Chars:
    """Predicates of a frame's payload bytes (8 bits each, least significant first), each built once."""

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


def _reading(C, ch: _Chars, zero: Sequence, lay: Layout, first: bool, L: Sequence) -> tuple:
    """One reading of a request frame's payload: a first frame's (`first`, L its LENGTH, ignored at lead 0, where the
    content-length gives it) or a continuation's (L the bytes left): (ok, ended, the bytes left after it, its prompt
    fields, the request's bytes left before it: L, or at lead 0 a first frame's content-length plus the head)."""
    room = lay.room
    s0, base = (lay.start, lay.lead) if first else (0, 0)
    avail = room - base
    oks = []
    if first:
        free = {i for o, w, _ in lay.spans for i in range(o, o + w)}
        oks += [ch.eq(i, lay.head[i]) for i in range(lay.lead, lay.start) if i not in free]
        o = lay.spans[1][0]
        cl_ok, cl = IT._digits(C, ch.data[o:o + ID_DIGITS])
        body = add(C, cl, const_bits(BODY_AT, 32), width=32, carry_out=True)
        oks += cl_ok + [body[32].inv()]
        if lay.lead:
            oks.append(_eq(C, body[:32], L))
        else:
            L = body[:32]
        for o, _, kind in lay.spans[2:]:
            oks += _SPANS[kind](C, ch, o)
    K, g = divmod(room - s0, ID_FIELD_BYTES)
    at = [s0 + ID_FIELD_BYTES * k for k in range(K + 1)]
    fields = [and_reduce(C, [ch.eq(at[k], 0x5B if first and k == 0 else 0x2C)] + _u32_ok(C, ch, at[k] + 1))
              for k in range(K)]
    last = _gt_const(C, L, avail).inv()
    # not its last frame: every slot a field, then the gap's spaces, fewer than 2 if the request's `]}` is next
    whole = fields + [ch.eq(i, 0x20) for i in range(at[K], room)]
    if g >= len(END):
        whole.append(_eq_const(C, L, avail + len(END)).inv())
    # its last: `]}` at the place e[k] of slot k, which ends the request; every slot before it a field, zero past it
    e = {k: _eq_const(C, L, at[k] + len(END) - base) for k in range(1 if first else 0, K + 1)
         if at[k] + len(END) <= room}
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
    rest = [C.AND(last.inv(), b) for b in sub(C, L, const_bits(avail, 32), 32)[0]]
    count = []
    for j in range(32):
        bit = ZERO
        for k in e:
            if k >> j & 1:
                bit = bit ^ e[k]
        count.append(bit ^ last.inv() if K >> j & 1 else bit)
    return ok, last, rest, count, L


def _sig_frame(S):
    layout(S.B)
    return (("cell", Array(S.B, BYTE)), ("left", Array(32, BIT))), Array(FRAME_OUT, BIT)


def _build_frame(C, S, *e):
    lay = layout(S.B)
    H, room = HEADER.size, lay.room
    cell, left = e[:S.B], list(e[S.B])
    ch = _Chars(C, cell[H:])
    zero = [ch.eq(i, 0) for i in range(room)]
    pad = and_reduce(C, [cell[i][b].inv() for i in range(H) for b in range(8)] + zero)
    header = HEADER.pack(DATA, CANONICAL_STREAM, room)
    head = and_reduce(C, [HC._same(cell[i][b], header[i] >> b & 1) for i in range(H) for b in range(8)])
    length = [cell[H + 3 - j // 8][j % 8] for j in range(32)]
    f_ok, f_end, f_rest, f_count, _ = _reading(C, ch, zero, lay, True, length)
    c_ok, c_end, c_rest, c_count, _ = _reading(C, ch, zero, lay, False, left)
    start = and_reduce(C, [b.inv() for b in left])
    ok, ended = mux(C, start, [f_ok, f_end], [c_ok, c_end])
    return [[pad, C.AND(head, ok), start, ended, *mux(C, start, f_rest, c_rest), *mux(C, start, f_count, c_count)]]


IoTieRequestFrame = _part(
    "IoTieRequestFrame", ("B",), _sig_frame, _build_frame,
    "(pad, req, start, ended, left, fields) of a B-byte ingress cell read against its request's bytes left before it "
    "(left; 0: a request starts here, start): pad iff the cell is B zero bytes; req iff it is a DATA frame on stream 0 "
    "of B − 9 bytes whose payload `io_request.parse_frame` accepts as that frame of a canonical id request; ended iff "
    "the request's last byte is in it; left the bytes the request has after it and fields its prompt fields, u32 "
    "each, least significant bit first")


def _sig_step(S):
    if type(S.D) is not int or not 1 <= S.D <= 32 or S.NEW not in (0, 1):
        raise ValueError(f"a step has 1 <= D <= 32 and NEW 0 or 1, not D={S.D!r}, NEW={S.NEW!r}")
    return (("state", Array(STATE, BIT)), ("frame", Array(FRAME_OUT, BIT)), ("routed", Array(32 + S.D, BIT))), \
        Array(STATE, BIT)


def _build_step(C, S, state, fo, routed):
    left, fields, over, j, frames = state[0:32], state[32:64], state[64], state[65:97], state[97:129]
    padded, ok = ZERO if S.NEW else state[129], state[130]
    pad, req, start, ended, rest, count = fo[0], fo[1], fo[2], fo[3], fo[4:36], fo[36:68]
    keep, stay = start.inv(), ended.inv()
    total = add(C, [C.AND(keep, b) for b in fields], count, width=32, carry_out=True)
    big = OR(C, C.AND(keep, over), total[32])
    end = C.AND(req, ended)
    match = and_reduce(C, [_eq(C, routed[:32], total[:32]), _eq(C, routed[32:], j[:S.D]), big.inv()])
    ok = and_reduce(C, [ok, pad ^ C.AND(req, padded.inv()), C.AND(end, match.inv()).inv()])
    return [[*mux(C, req, rest, left), *mux(C, req, [C.AND(stay, b) for b in total[:32]] + [C.AND(stay, big)],
                                            [*fields, over]),
             *add(C, j, [end], width=32), *add(C, frames, [req], width=32), OR(C, padded, pad), ok]]


IoTieRequestStep = _part(
    "IoTieRequestStep", ("D", "NEW"), _sig_step, _build_step,
    "the ingress scan's state after one cell (`IoTieRequestFrame`'s output): its request's bytes left and prompt "
    "fields so far (over: 2^32 or more), the requests ended and request frames so far, whether a pad has been seen "
    "in the bucket (NEW: the cell starts one) and ok, which drops unless the cell is a pad or a request frame before "
    "the bucket's pads, and, where it ends a request, the routed entry is (fields, the requests ended before it)")


def _sig_own(S):
    if type(S.B) is not int or S.B < SALT_FRAME_BYTES:
        raise ValueError(f"an own frame has B >= {SALT_FRAME_BYTES}, not {S.B!r}")
    return (("cell", Array(S.B, BYTE)), ("window", Array(64, BIT))), BIT


def _build_own(C, S, *e):
    cell, window, tmpl = e[:S.B], e[S.B], own_frame(0, S.B)
    w = range(HEADER.size, HEADER.size + 8)
    oks = [HC._same(cell[i][b], tmpl[i] >> b & 1) for i in range(S.B) if i not in w for b in range(8)]
    # the window is u64 big-endian at bytes 9 ... 16: its bit j is bit j % 8 of byte 16 - j // 8
    oks += [HC._same(cell[HEADER.size + 7 - j // 8][j % 8], window[j]) for j in range(64)]
    return [and_reduce(C, oks)]


IoTieRequestOwn = _part(
    "IoTieRequestOwn", ("B",), _sig_own, _build_own,
    "1 iff the B-byte cell is the certifier's own frame of the window (u64, least significant bit first) as the grid "
    "holds it: `active.salt_frame(window, PLACEHOLDER, B)`")


def _sig_end(S):
    return (("state", Array(STATE, BIT)), ("count", Array(32, BIT)), ("received", Array(32, BIT)),
            ("left_out", Array(32, BIT)), ("fields_out", Array(32, BIT))), BIT


def _build_end(C, S, state, count, received, left_out, fields_out):
    return [and_reduce(C, [state[130], state[64].inv(), _eq(C, state[65:97], count), _eq(C, state[97:129], received),
                           _eq(C, state[0:32], left_out), _eq(C, state[32:64], fields_out)])]


IoTieRequestEnd = _part(
    "IoTieRequestEnd", (), _sig_end, _build_end,
    "1 iff the scan's state is ok, its requests ended are count, its request frames received, and its request in "
    "progress has left_out bytes left and fields_out prompt fields so far, fewer than 2^32 (over clear)")
# C-Flock's lowering calls a static-free Definition's builder without S, as `block_check._part` builds them
IoTieRequestEnd.build = lambda C, *e: _build_end(C, SimpleNamespace(), *e)  # type: ignore[attr-defined]


def _sig_request(S):
    sh = Shape(S.B, S.T, S.R, S.OWN)
    own = (("window", Array(64, BIT)),) if S.OWN else ()
    return ((("root", DIGEST), ("domain", DIGEST), ("cells", Array(sh.N, Array(S.B, BYTE))),
             ("salts", Array(S.T, SALT)), *own, ("left", Array(32, BIT)), ("fields", Array(32, BIT)),
             ("ended", Array(sh.N, Array(32, BIT))),
             ("count", Array(32, BIT)), ("received", Array(32, BIT)), ("left_out", Array(32, BIT)),
             ("fields_out", Array(32, BIT)), ("switches", Array(sh.switches, BIT))), BIT)


def _body_request(B, S, root, dom, cells, salts, *rest):
    sh = Shape(S.B, S.T, S.R, S.OWN)
    window, (left, fields, ended, count, received, left_out, fields_out, switches_) = \
        (rest[0], rest[1:]) if S.OWN else (None, rest)
    K, cr, er = HC._Consts(B), HC._refs(cells), HC._refs(ended)
    row = 8 * S.B * S.R
    oks = [B.call(bind(IT.NetworkRowsRoot, N_BITS=row, COUNT=S.T), cells.reshape(Array(S.T, Array(row, BIT))), salts,
                  root, dom)]
    xs = []
    for k in range(sh.M):
        xs += (er[32 * k:32 * (k + 1)] if k < sh.N else K.bits(0, 32)) + K.bits(k, sh.D)
    routed = HC._refs(B.call(bind(IT.IoTieRoute, W=sh.W, D=sh.D), HC._coll(Array(sh.M, Array(sh.W, BIT)), xs),
                             switches_))

    def cell(i: int):
        return HC._coll(Array(S.B, BYTE), cr[8 * S.B * i:8 * S.B * (i + 1)])

    if S.OWN:
        oks.append(B.call(bind(IoTieRequestOwn, B=S.B), cell(0), window))
    state = HC._coll(Array(STATE, BIT), HC._refs(left) + HC._refs(fields) + K.bits(0, STATE - 65) + K.bits(1, 1))
    for i in range(S.OWN, sh.N):
        fo = B.call(bind(IoTieRequestFrame, B=S.B), cell(i), HC._coll(Array(32, BIT), HC._refs(state)[:32]))
        state = B.call(bind(IoTieRequestStep, D=sh.D, NEW=int(i > 0 and i % S.R == 0)), state, fo,
                       HC._coll(Array(sh.W, BIT), routed[sh.W * i:sh.W * (i + 1)]))
    oks.append(B.call(bind(IoTieRequestEnd), state, count, received, left_out, fields_out))
    bits = [r for o in oks for r in HC._refs(o)]
    return B.call(BL.and_gate(len(bits)), HC._coll(Array(len(bits), BIT), bits))


IoTieRequest = CompositeDefinition(
    "IoTieRequest", 1, ("B", "T", "R", "OWN"), _sig_request, _body_request, conformance="exact-model-tested",
    doc="1 iff the cells' bucket rows are the root's (`NetworkRowsRoot`), the own frame is where it goes when OWN "
        "(`IoTieRequestOwn`), and the cells scanned from the carry (left, fields), one `IoTieRequestFrame` and "
        "`IoTieRequestStep` a cell, are pads and canonical id-request frames, each bucket's frames before its pads, "
        "ending count requests whose prompt fields are ended's, in order (routed by the switches, `IoTieRoute`), in "
        "received request frames, and leaving the carry-out (left_out, fields_out) (`IoTieRequestEnd`): the ingress "
        "tie of one link-window, every cell opened")
IoTieRequest.gated = True  # type: ignore[attr-defined]  # circuit-check holds its digest pins (`tie_service.gate`)
#: the circuit digests circuit-check pins (`head_check.pinned_digest`)
PINS = Path(__file__).with_name("io_request.circuit-check.json")


# --- arguments -------------------------------------------------------------------------------------------------------

def frame_arguments(cells: Sequence[bytes], lefts: Sequence[int]) -> list[list[int]]:
    """`IoTieRequestFrame`'s arguments for a batch of (cell, left), each its leaves' bits a row."""
    return [[HC._bytes_bits(c) for c in cells], [HC._int_bits(x, 32) for x in lefts]]


def arguments(shape: Shape, inst: Instance, bits: Sequence[int]) -> list[list[int]]:
    """`IoTieRequest`'s arguments in its signature's order, each its leaves' bits."""
    v, d, ci, co = inst.view, HC._digest_bits, inst.view.carry_in, inst.view.carry_out
    if len(v.ended) > shape.N or any(not 0 <= x < 1 << 32 for x in (*v.ended, ci.left, ci.fields, v.received,
                                                                       co.left, co.fields)):
        raise ValueError(f"a view has at most N = {shape.N} prompt lengths and u32 values, not {v!r}")
    ended = list(v.ended) + [0] * (shape.N - len(v.ended))
    own = [HC._int_bits(inst.window, 64)] if shape.OWN else []
    return [d(inst.root), d(inst.domain), [b for c in inst.cells for b in HC._bytes_bits(c)],
            [b for s in inst.salts for b in d(s)], *own, HC._int_bits(ci.left, 32), HC._int_bits(ci.fields, 32),
            [b for n in ended for b in HC._int_bits(n, 32)], HC._int_bits(len(v.ended), 32),
            HC._int_bits(v.received, 32), HC._int_bits(co.left, 32), HC._int_bits(co.fields, 32), list(bits)]


__all__ = ["BODY_AT", "Carry", "Cut", "END", "FRAME_OUT", "Frame", "Instance", "IoTieRequest", "IoTieRequestEnd",
           "IoTieRequestFrame", "IoTieRequestOwn", "IoTieRequestStep", "Layout", "PINS", "RequestStatement",
           "RequestView", "STATE", "Shape", "arguments", "chained", "cut", "ends", "frame_arguments", "instance",
           "layout", "own_frame", "parse_frame", "reference", "request_view", "rows", "statements", "switches",
           "windows_of"]
