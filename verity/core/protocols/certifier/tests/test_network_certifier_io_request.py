"""IoTieRequest's reference on a fake upstream (`io_request`): each request's prompt_len and arrival ordinal held to
the warden's ingress cells. The first negative is REDTEAM's phase-6 split (`cursor/redteam-phase6-advised-061a`,
target 1): two requests' prompt lengths swapped with their total kept, which N2-ADVICE's advised floor accepts."""
from __future__ import annotations

import pytest

from verity.core.protocols.certifier import io_request as Q
from verity.core.protocols.certifier import io_tie as IT
from verity.core.protocols.certifier.active import (HEADER, ID_DIGITS, ID_REQUEST_HEAD, LENGTH, id_request,
                                                   id_request_frames, request_frames)

PL = (5, 60, 12, 100, 1)
COUNTS = (1, 1, 1, 0, 1, 2, 1, 1, 1, 1, 0, 0)


def _requests(prompt_lens, B):
    return [id_request(model="glm", user=f"u{i}", prompt=[(7919 * i + 104729 * k) % 151329 for k in range(n)],
                       max_tokens=16, temperature=0.7, top_p=0.95, seed=i, frame_bytes=B)
            for i, n in enumerate(prompt_lens)]


def _received(shape, counts):
    return tuple(sum(counts[w * shape.T:(w + 1) * shape.T]) for w in range(len(counts) // shape.T))


def _link(shape, prompt_lens=PL, counts=COUNTS):
    windows = Q.windows_of(shape, _requests(prompt_lens, shape.B), counts)
    return windows, _received(shape, counts)


def _verdicts(shape, windows, received, prompt_lens):
    return [Q.reference(shape, Q.instance(shape, cells, Q.request_view(prompt_lens, received, w, shape.B), w))
            for w, cells in enumerate(windows)]


@pytest.mark.parametrize("B", [512, 4096])
def test_a_request_parses_to_its_prompt_length(B):
    """Every frame of an id request parses, and the request's fields and bytes are its prompt length and its own."""
    lay = Q.layout(B)
    for n in (1, 2, 11, 12, 13, 56, 57, 58, 336, 337, 338, 708, 709):
        raw = request_frames(_requests([n], B)[0], B)
        frames = [raw[k:k + B] for k in range(0, len(raw), B)]
        assert len(frames) == id_request_frames(n, B)
        left = fields = 0
        for k, f in enumerate(frames):
            p = Q.parse_frame(lay, f[HEADER.size:], k == 0, left)
            assert p is not None, (n, k)
            left = (p.length if k == 0 else left) - p.used
            fields += p.fields
        assert (left, fields) == (0, n)


def test_redteam_phase6_split_is_refused():
    """REDTEAM phase 6, target 1, as IoTieRequest's first negative. At B = 4,096 a 338-token prompt takes 2 frames and a
    16-token one 1, released one a bucket ([1, 1, 1, 0]). The forged advice swaps the two prompt lengths, keeping the
    total, so its ends move from (2, 3) to (1, 3) and the advised floor accepts an acausal release (ONE-LINE FIX: bind
    each request's prompt_len to the ingress record's content). The tie reads 338 fields from request 0's cells."""
    shape = Q.Shape(4096, 4, 9)
    SHORT, LONG = 16, 338
    [cells], received = _link(shape, (LONG, SHORT), (1, 1, 1, 0))
    assert Q.ends((LONG, SHORT), shape.B) == (2, 3) and Q.ends((SHORT, LONG), shape.B) == (1, 3)
    assert _verdicts(shape, [cells], received, (LONG, SHORT)) == [1]
    assert _verdicts(shape, [cells], received, (SHORT, LONG)) == [0]


@pytest.mark.parametrize("own", [0, 1])
def test_requests_across_windows_are_their_own(own):
    """Requests that straddle windows carry into the next (`request_view`'s carry), with and without the own frame.
    Each forgery is refused in the window where its request ends: a swap that keeps the total, and lengths one off
    either way. Each keeps every request's frames, so `advised.ends` doesn't move and only the tie sees it."""
    shape = Q.Shape(512, 4, 3, OWN=own)
    windows, received = _link(shape)
    assert [v.carry.left > 0 for v in (Q.request_view(PL, received, w, 512) for w in range(3))] == [False, True, True]
    assert _verdicts(shape, windows, received, PL) == [1, 1, 1]
    for forged in ((5, 100, 12, 60, 1), (5, 61, 12, 99, 1), (5, 59, 12, 101, 1)):
        assert Q.ends(forged, 512) == Q.ends(PL, 512)
        assert _verdicts(shape, windows, received, forged) == [1, 0, 0]


def _tamper(cell, at, new):
    return cell[:at] + new + cell[at + len(new):]


def _cell_tampers(shape, cells):
    """(what, cells) for each way a window's cells can differ from canonical id requests, recommitted by the warden."""
    lay = Q.layout(shape.B)
    pad = IT.pad_cell(shape.B)
    h = HEADER.size
    i = next(k for k, c in enumerate(cells) if c[h + LENGTH.size:h + LENGTH.size + 4] == b"POST")
    first = cells[i]
    length = LENGTH.unpack(first[h:h + LENGTH.size])[0]
    cl = lay.spans[1][0]
    model, user = lay.spans[6][0], lay.spans[7][0]
    field = h + lay.start
    body = length - len(ID_REQUEST_HEAD) - ID_DIGITS - 4
    assert first[h + cl:h + cl + ID_DIGITS] == b"%10d" % body
    out = [
        ("the head's method", _tamper(first, h + LENGTH.size, b"GET ")),
        ("a fixed key", _tamper(first, h + lay.spans[2][0] - 2, b"x")),
        ("LENGTH one more", _tamper(first, h, LENGTH.pack(length + 1))),
        ("content-length one less", _tamper(first, h + cl, b"%10d" % (body - 1))),
        ("a prompt field's digit", _tamper(first, field + 10, b"x")),
        ("a prompt field's leading zero", _tamper(first, field + 1, b"0000000001")),
        ("a prompt id of 2^32", _tamper(first, field + 1, b"4294967296")),
        ("the first field's bracket", _tamper(first, field, b",")),
        ("a name's unescaped quote", _tamper(first, h + model + 1, b'"')),
        ("a user past its quotes", _tamper(first, h + user + 6, b"x")),
        ("a ratio's form", _tamper(first, h + lay.spans[3][0], b"0.7     ")),
        ("the stream", HEADER.pack(1, 1, lay.room) + first[h:]),
        ("the header's length", HEADER.pack(1, 0, lay.room - 1) + first[h:]),
    ]
    out = [(w, cells[:i] + [c] + cells[i + 1:]) for w, c in out]
    last = max(k for k, c in enumerate(cells) if c.rstrip(b"\0").endswith(Q.END))
    end = cells[last].rstrip(b"\0")
    out += [("a byte in the fill", cells[:last] + [_tamper(cells[last], len(end) + 1, b"\1")] + cells[last + 1:]),
            ("the end", cells[:last] + [_tamper(cells[last], len(end) - 2, b"] ")] + cells[last + 1:])]
    t = i // shape.R
    row = cells[t * shape.R:(t + 1) * shape.R]
    if pad in row:
        j = t * shape.R + row.index(pad)
        moved = cells[:]
        moved[i], moved[j] = moved[j], moved[i]
        if j > i:
            out.append(("a request frame after a pad", moved))
    k = next((k for k in range(len(cells) - 1) if cells[k] != pad and cells[k + 1] != pad and
              not Q.is_own(cells[k], 0) and cells[k] != cells[k + 1]), None)
    if k is not None:
        swapped = cells[:]
        swapped[k], swapped[k + 1] = swapped[k + 1], swapped[k]
        out.append(("two frames swapped", swapped))
    return out


def test_each_way_a_cell_can_differ_is_refused():
    shape = Q.Shape(512, 4, 3)
    windows, received = _link(shape)
    cells = windows[1]
    view = Q.request_view(PL, received, 1, 512)
    assert Q.reference(shape, Q.instance(shape, cells, view, 1)) == 1
    cases = _cell_tampers(shape, cells)
    assert len(cases) >= 16
    for what, bad in cases:
        assert Q.reference(shape, Q.instance(shape, bad, view, 1)) == 0, what


def test_a_cell_or_salt_other_than_the_wardens_is_refused():
    shape = Q.Shape(512, 4, 3)
    windows, received = _link(shape)
    inst = Q.instance(shape, windows[1], Q.request_view(PL, received, 1, 512), 1)
    other = Q.instance(shape, windows[0], Q.request_view(PL, received, 0, 512), 1)
    assert Q.reference(shape, inst._replace(cells=other.cells)) == 0
    assert Q.reference(shape, inst._replace(salts=(inst.salts[1],) + inst.salts[1:])) == 0
    assert Q.reference(shape, inst._replace(root=other.root)) == 0


def test_a_view_other_than_the_cells_is_refused():
    """A frame count, a carry or an ended list other than the cells' refuses the window."""
    shape = Q.Shape(512, 4, 3)
    windows, received = _link(shape)
    v = Q.request_view(PL, received, 1, 512)
    inst = Q.instance(shape, windows[1], v, 1)
    q, left, fields = v.carry
    for bad in (v._replace(received=v.received + 1), v._replace(received=v.received - 1),
                v._replace(carry=Q.Carry(q + 1, left, fields)), v._replace(carry=Q.Carry(q, left, fields + 1)),
                v._replace(carry=Q.Carry(q, left + 11, fields)), v._replace(carry=Q.Carry(q, 0, 0)),
                v._replace(ended=v.ended[:-1]), v._replace(ended=v.ended + ((9, 1),)),
                v._replace(ended=tuple(reversed(v.ended)))):
        assert Q.reference(shape, inst._replace(view=bad)) == 0, bad


def test_the_own_frame_is_where_it_goes():
    shape = Q.Shape(512, 4, 3, OWN=1)
    windows, received = _link(shape)
    v = Q.request_view(PL, received, 0, 512)
    cells = windows[0]
    assert Q.reference(shape, Q.instance(shape, cells, v, 0)) == 1
    assert Q.reference(shape, Q.instance(shape, cells, v, 1)) == 0
    assert Q.reference(shape, Q.instance(shape, [IT.pad_cell(512)] + cells[1:], v, 0)) == 0


def test_a_frame_too_small_for_the_head_is_refused():
    with pytest.raises(ValueError, match="no room|frame 0"):
        Q.Shape(256, 4, 3)
