"""IoTieRequest (`PROTOCOL.md`, "The I/O tie", "The ingress tie"), a prototype on a fake upstream: the reference, in
the clear, of one ingress link-window's tie, the unit at selection `All` that binds each request's prompt_len and
arrival ordinal to the warden's certified ingress cells. Nothing here is in a Spec or proved, no warden commits these
cells yet, and the Boolean Definition isn't built.

**The cells.** A link-window's ingress is N = T·R cells of B bytes, R a bucket (the ingress allocation), committed as
the egress cells are (`io_tie.bucket_rows`, `io_tie.rows_root`). Bucket 0's first cell is the certifier's own frame
when OWN = 1 (`active.salt_frame`); then each bucket holds its request frames, then pads. A request frame is
`active.request_frames`' (`DATA` on `CANONICAL_STREAM`, B − 9 bytes of payload), and the requests are CERT-VARLEN's
id requests (`active.id_request`), whose every field is at a place fixed by the prompt's length and B alone.

**The cut** (`cut`) runs as `active.Requests` does: a request starts a frame with its LENGTH and ends at the frame
that holds its last byte; the k-th request is ordinal k. Each frame is parsed without its request's prompt length
(`parse_frame`): the head and the fixed fields at their places, 11-byte prompt fields (`[` or `,`, then a u32 in 10
characters), the gap spaces only where the next piece would straddle a frame, and `]}` then zero fill at the
request's last byte. So each request's prompt_len is its count of prompt fields, read from the certified bytes.

**The view** (`request_view`, public in the named gap, computed by the verifier from the opened advice as
`advised.view` is): the carry, the request in progress at the window's first request frame (its ordinal, its bytes
left and its prompt fields so far), and `ended`, the (ordinal, prompt_len) of each request whose last frame falls in
the window, from the advice's prompt lengths and `received`. At the target the unit reads them under the advice's
hiding root.

**What 1 means** (`reference`): the cells' root is their bucket rows', every cell is a pad, the own frame where it
goes or a canonical request frame, each bucket's request frames come first, the cut from the carry ends exactly the
requests `ended` names, in order, each with its prompt_len, and the window's request frames number `received`.

No carry-out is checked: on a link whose every ingress window is accepted, each window's carry is the true one by
induction from window 0, since every earlier request's prompt_len, and so its frames, is held to the cells, and a
carry that disagrees with the cells misplaces the `]}` or a request's head.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import NamedTuple, Sequence

from verity.core.protocols.certifier import io_tie as IT
from verity.core.protocols.certifier.active import (CANONICAL_STREAM, DATA, HEADER, ID_DIGITS, ID_FIELD_BYTES,
                                                   ID_NAME_BYTES, ID_REQUEST_HEAD, LENGTH, SALT, SALT_STREAM,
                                                   id_request, id_request_frames, request_frames)

END = b"]}"
#: the variable fields of an id request's head: (key, width, kind)
FIELDS = ((b'{"max_tokens":', ID_DIGITS, "u32"), (b',"temperature":', 8, "ratio"), (b',"top_p":', 8, "ratio"),
          (b',"seed":', ID_DIGITS, "u32"), (b',"model":', ID_NAME_BYTES, "name"),
          (b',"user":', ID_NAME_BYTES, "name"))


class Layout(NamedTuple):
    """An id request's frame-0 layout at frame size B: `room` payload bytes a frame, the head's bytes up to the first
    prompt field (`head`, the zero request's), the variable spans in it, and `start`, the first prompt field's
    offset. Payload offsets count from the frame's payload, whose first 4 bytes in frame 0 are LENGTH."""

    room: int
    head: bytes
    spans: tuple[tuple[int, int, str], ...]
    start: int


def layout(B: int) -> Layout:
    """`Layout` at frame size B. Raises `ValueError` where the head and one prompt field don't fit frame 0 unmoved."""
    room = B - HEADER.size
    try:
        zero = id_request(model="", user="", prompt=[0], max_tokens=0, frame_bytes=B)
    except ValueError as e:
        raise ValueError(f"a {B}-byte frame has no room for an id request: {e}") from None
    payload = LENGTH.pack(len(zero)) + zero
    spans = [(0, LENGTH.size, "length"), (LENGTH.size + len(ID_REQUEST_HEAD), ID_DIGITS, "content-length")]
    at = LENGTH.size + len(ID_REQUEST_HEAD) + ID_DIGITS + 4
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
    return Layout(room, payload[:start], tuple(spans), start)


def _u32(field: bytes) -> bool:
    s = field.lstrip(b" ")
    return len(field) == ID_DIGITS and 0 < len(s) and s.isdigit() and (s == b"0" or s[:1] != b"0") and \
        int(s) < 1 << 32


def _ratio(field: bytes) -> bool:
    try:
        return len(field) == 8 and b"%8.6f" % float(field) == field and 0 <= float(field) < 10
    except ValueError:
        return False


def _name(field: bytes) -> bool:
    raw = field.rstrip(b" ")
    try:
        s = json.loads(raw)
    except ValueError:
        return False
    return isinstance(s, str) and json.dumps(s).encode() == raw and len(field) == ID_NAME_BYTES


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
        length = LENGTH.unpack(payload[:LENGTH.size])[0]
        for i, (a, b) in enumerate(zip(payload[:lay.start], lay.head)):
            if a != b and not any(o <= i < o + w for o, w, _ in lay.spans):
                return None
        for o, w, kind in lay.spans[2:]:
            if not CHECK[kind](payload[o:o + w]):
                return None
        cl = payload[lay.spans[1][0]:lay.spans[1][0] + ID_DIGITS]
        if not _u32(cl) or int(cl) != length - len(ID_REQUEST_HEAD) - ID_DIGITS - 4:
            return None
        left, at, base = length, lay.start, LENGTH.size
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
    """IoTieRequest's public inputs for one window, in the named gap: the carry, the requests that end in the window
    with their prompt lengths, and the window's request frames."""

    carry: Carry
    ended: tuple[tuple[int, int], ...]
    received: int


@dataclass(frozen=True)
class Shape:
    """An ingress link-window: T buckets of R cells of B bytes, OWN the certifier's own frame at bucket 0's head (0 or
    1), and Q the most requests that can end in it (T·R)."""

    B: int
    T: int
    R: int
    OWN: int = 0

    def __post_init__(self) -> None:
        if not all(type(x) is int for x in (self.B, self.T, self.R, self.OWN)) or self.T < 2 or self.R < 1 or \
                self.OWN not in (0, 1):
            raise ValueError(f"an ingress window has T ≥ 2 buckets of R ≥ 1 cells and OWN 0 or 1, not {self!r}")
        layout(self.B)

    @property
    def N(self) -> int:
        return self.T * self.R

    @property
    def Q(self) -> int:
        return self.N


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
    ended = tuple((q, n) for q, (n, x) in enumerate(zip(prompt_lens, e)) if lo < x <= hi)
    return RequestView(_carry_at(prompt_lens, B, lo), ended, received[window])


def is_own(cell: bytes, window: int) -> bool:
    kind, stream, n = HEADER.unpack(cell[:HEADER.size])
    return kind == SALT and stream == SALT_STREAM and cell[HEADER.size:HEADER.size + 8] == window.to_bytes(8, "big")


def cut(shape: Shape, cells: Sequence[bytes], carry: Carry, window: int = 0
        ) -> tuple[tuple[tuple[int, int], ...], int, Carry] | None:
    """The cut of the window's cells from `carry`: the (ordinal, prompt fields) of each request that ends in it, its
    request frames and the carry after it; None where a cell isn't a pad, the own frame where it goes or a canonical
    request frame, or a request frame follows a pad in its bucket."""
    lay, pad = layout(shape.B), IT.pad_cell(shape.B)
    q, left, fields = carry
    ended, frames = [], 0
    for t in range(shape.T):
        bucket = cells[t * shape.R:(t + 1) * shape.R]
        if t == 0 and shape.OWN:
            if not is_own(bucket[0], window):
                return None
            bucket = bucket[1:]
        padded = False
        for c in bucket:
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
                ended.append((q, fields))
                q, fields = q + 1, 0
    return tuple(ended), frames, Carry(q, left, fields)


def rows(shape: Shape, cells: Sequence[bytes]) -> list[bytes]:
    """The warden's rows: each bucket's R cells, R·B bytes (`io_tie.bucket_rows`)."""
    return [b"".join(cells[t * shape.R:(t + 1) * shape.R]) for t in range(shape.T)]


class Instance(NamedTuple):
    """One ingress link-window's tie: the cells and their salts (witness), the warden's root and domain id, and the
    view."""

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
    got = cut(shape, inst.cells, inst.view.carry, inst.window)
    return int(got is not None and got[0] == inst.view.ended and got[1] == inst.view.received)


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
            own = [HEADER.pack(SALT, SALT_STREAM, 72) + w.to_bytes(8, "big") + bytes(B - HEADER.size - 8)] \
                if t == 0 and shape.OWN else []
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


__all__ = ["Carry", "Frame", "Instance", "Layout", "RequestView", "Shape", "cut", "ends", "instance", "is_own", "rows",
           "layout", "parse_frame", "reference", "request_view", "windows_of"]
