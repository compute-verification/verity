"""IoTieRequest on a fake upstream (`io_request`): each request's prompt_len and arrival ordinal held to the warden's
ingress cells, by the reference and by the Boolean Definitions. The first negative is REDTEAM's phase-6 split
(`cursor/redteam-phase6-advised-061a`, target 1): two requests' prompt lengths swapped with their total kept, which
N2-ADVICE's advised floor accepts."""
from __future__ import annotations

import random

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.core.protocols.certifier import head_check as HC
from verity.core.protocols.certifier import io_request as Q
from verity.core.protocols.certifier import io_tie as IT
from verity.core.protocols.certifier.active import (DATA, HEADER, ID_DIGITS, ID_REQUEST_HEAD, LENGTH, id_request,
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


def _instances(shape, windows, received, prompt_lens):
    return [Q.instance(shape, cells, Q.request_view(prompt_lens, received, w, shape.B), w)
            for w, cells in enumerate(windows)]


def _verdicts(shape, windows, received, prompt_lens):
    return [Q.reference(shape, i) for i in _instances(shape, windows, received, prompt_lens)]


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
    for forged in FORGED:
        assert Q.ends(forged, 512) == Q.ends(PL, 512)
        assert _verdicts(shape, windows, received, forged) == [1, 0, 0]


FORGED = ((5, 100, 12, 60, 1), (5, 61, 12, 99, 1), (5, 59, 12, 101, 1))


def _tamper(cell, at, new):
    return cell[:at] + new + cell[at + len(new):]


def _first(cells):
    h = HEADER.size
    return next(k for k, c in enumerate(cells) if c[h + LENGTH.size:h + LENGTH.size + 4] == b"POST")


def _frame_tampers(B, first):
    """(what, cell) for each way a request's first frame can differ from a canonical id request's."""
    lay, h = Q.layout(B), HEADER.size
    length = LENGTH.unpack(first[h:h + LENGTH.size])[0]
    cl = lay.spans[1][0]
    model, user = lay.spans[6][0], lay.spans[7][0]
    field = h + lay.start
    body = length - len(ID_REQUEST_HEAD) - ID_DIGITS - 4
    assert first[h + cl:h + cl + ID_DIGITS] == b"%10d" % body
    return [
        ("the head's method", _tamper(first, h + LENGTH.size, b"GET ")),
        ("a fixed key", _tamper(first, h + lay.spans[2][0] - 2, b"x")),
        ("LENGTH one more", _tamper(first, h, LENGTH.pack(length + 1))),
        ("content-length one less", _tamper(first, h + cl, b"%10d" % (body - 1))),
        ("a prompt field's digit", _tamper(first, field + 10, b"x")),
        ("a prompt field's leading zero", _tamper(first, field + 1, b"0000000001")),
        ("a prompt id of 2^32", _tamper(first, field + 1, b"4294967296")),
        ("the first field's bracket", _tamper(first, field, b",")),
        ("a name's unescaped quote", _tamper(first, h + model + 1, b'"')),
        ("a name's escape", _tamper(first, h + model + 1, b"\\")),
        ("a user past its quotes", _tamper(first, h + user + 6, b"x")),
        ("a ratio's form", _tamper(first, h + lay.spans[3][0], b"0.7     ")),
        ("the stream", HEADER.pack(DATA, 1, lay.room) + first[h:]),
        ("the header's length", HEADER.pack(DATA, 0, lay.room - 1) + first[h:]),
    ]


def _cell_tampers(shape, cells):
    """(what, cells) for each way a window's cells can differ from canonical id requests, recommitted by the warden."""
    pad = IT.pad_cell(shape.B)
    i = _first(cells)
    out = [(w, cells[:i] + [c] + cells[i + 1:]) for w, c in _frame_tampers(shape.B, cells[i])]
    last = max(k for k, c in enumerate(cells) if c.rstrip(b"\0").endswith(Q.END))
    end = cells[last].rstrip(b"\0")
    out += [("a byte in the fill", cells[:last] + [_tamper(cells[last], len(end) + 1, b"\1")] + cells[last + 1:]),
            ("the end", cells[:last] + [_tamper(cells[last], len(end) - 2, b"] ")] + cells[last + 1:])]
    k = next(k for k in range(len(cells) - 1) if (k + 1) % shape.R and cells[k][0] == DATA and cells[k + 1] == pad)
    moved = cells[:]
    moved[k], moved[k + 1] = moved[k + 1], moved[k]
    out.append(("a request frame after a pad", moved))
    k = next(k for k in range(len(cells) - 1) if cells[k][0] == cells[k + 1][0] == DATA and cells[k] != cells[k + 1])
    swapped = cells[:]
    swapped[k], swapped[k + 1] = swapped[k + 1], swapped[k]
    return out + [("two frames swapped", swapped)]


def test_each_way_a_cell_can_differ_is_refused():
    shape = Q.Shape(512, 4, 3)
    windows, received = _link(shape)
    cells = windows[1]
    view = Q.request_view(PL, received, 1, 512)
    assert Q.reference(shape, Q.instance(shape, cells, view, 1)) == 1
    cases = _cell_tampers(shape, cells)
    assert len(cases) >= 18
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


def _view_tampers(v):
    """(what, view) for each way a view can differ from the cells'. The carry's ordinal isn't one: the cells carry
    none, and the verifier computes it."""
    q, left, fields = v.carry
    wrap = (1 << 32) - 1 - fields
    return [("a frame more", v._replace(received=v.received + 1)),
            ("a frame less", v._replace(received=v.received - 1)),
            ("a field more carried", v._replace(carry=Q.Carry(q, left, fields + 1))),
            ("a slot more left", v._replace(carry=Q.Carry(q, left + 11, fields))),
            ("no carry", v._replace(carry=Q.Carry(q, 0, 0))),
            ("carried fields whose sum wraps 2^32", v._replace(carry=Q.Carry(q, left, fields + wrap),
                                                               ended=((v.ended[0] + wrap) % (1 << 32),) + v.ended[1:])),
            ("a request fewer", v._replace(ended=v.ended[:-1])), ("a request more", v._replace(ended=v.ended + (1,))),
            ("two requests swapped", v._replace(ended=tuple(reversed(v.ended))))]


def test_a_view_other_than_the_cells_is_refused():
    shape = Q.Shape(512, 4, 3)
    windows, received = _link(shape)
    v = Q.request_view(PL, received, 1, 512)
    inst = Q.instance(shape, windows[1], v, 1)
    assert Q.reference(shape, inst) == 1 and len(set(v.ended)) == len(v.ended) == 2
    for what, bad in _view_tampers(v):
        assert Q.reference(shape, inst._replace(view=bad)) == 0, what


def test_the_own_frame_is_where_it_goes():
    shape = Q.Shape(512, 4, 3, OWN=1)
    windows, received = _link(shape)
    v = Q.request_view(PL, received, 0, 512)
    cells = windows[0]
    assert Q.reference(shape, Q.instance(shape, cells, v, 0)) == 1
    assert Q.reference(shape, Q.instance(shape, cells, v, 1)) == 0
    assert Q.reference(shape, Q.instance(shape, [IT.pad_cell(512)] + cells[1:], v, 0)) == 0
    assert Q.reference(shape, Q.instance(shape, [_tamper(cells[0], 30, b"\1")] + cells[1:], v, 0)) == 0


def test_a_frame_too_small_for_the_head_is_refused():
    with pytest.raises(ValueError, match="no room|frame 0"):
        Q.Shape(256, 4, 3)


# --- the circuit -----------------------------------------------------------------------------------------------------

def _bits(rows):
    return np.asarray(rows, dtype=np.uint8)


def _int(bits) -> int:
    return sum(int(b) << j for j, b in enumerate(bits))


def _frame_reference(B, cell, left):
    """`IoTieRequestFrame`'s output in the clear: (pad, req, start), and, for a request frame, (ended, left, fields)."""
    lay = Q.layout(B)
    header = HEADER.unpack(cell[:HEADER.size]) == (DATA, 0, lay.room)
    p = Q.parse_frame(lay, cell[HEADER.size:], not left, left) if header else None
    head = (int(cell == IT.pad_cell(B)), int(p is not None), int(left == 0))
    if p is None:
        return head
    rest = (left if p.length is None else p.length) - p.used
    return head + (int(rest == 0), rest, p.fields)


def _frames(B, cases):
    args = Q.frame_arguments(*zip(*cases))
    out = np.asarray(evaluate_bits(bind(Q.IoTieRequestFrame, B=B), *[_bits(a) for a in args]))
    got = []
    for o in out.reshape(len(cases), Q.FRAME_OUT):
        head = (int(o[0]), int(o[1]), int(o[2]))
        got.append(head + ((int(o[3]), _int(o[4:36]), _int(o[36:68])) if o[1] else ()))
    return got


def test_the_frame_circuit_is_its_reference():
    """`IoTieRequestFrame` against `parse_frame` on every frame of requests of several prompt lengths, each read against
    its request's true bytes left and against lefts one off, one slot off and 2 off, on each way a first frame can
    differ, and on a pad and the own frame."""
    B = 512
    cases = []
    for raw in (request_frames(r, B) for r in _requests((1, 2, 11, 12, 13, 40, 56, 57, 58, 100), B)):
        left = 0
        for k in range(0, len(raw), B):
            c = raw[k:k + B]
            cases += [(c, x) for x in sorted({left, left + 1, left - 1, left + 2, left + 11, 1}) if x >= 0]
            left = _frame_reference(B, c, left)[4]
    first = request_frames(_requests((40,), B)[0], B)[:B]
    cases += [(c, 0) for _, c in _frame_tampers(B, first)]
    cases += [(IT.pad_cell(B), 0), (IT.pad_cell(B), 7), (Q.own_frame(3, B), 0)]
    want = [_frame_reference(B, c, x) for c, x in cases]
    assert sum(w[1] for w in want) >= 19 and len(cases) - sum(w[1] for w in want) >= 50
    assert _frames(B, cases) == want
    assert HC.ands(bind(Q.IoTieRequestFrame, B=B)) == 20_924


def test_redteam_phase6_frames_read_their_prompt_lengths():
    """REDTEAM phase 6's cells at B = 4,096 through `IoTieRequestFrame`: the 338-token request's first frame doesn't end
    it and holds 337 prompt fields, its second ends it with 1 more, and the 16-token request's one frame ends it with
    16. So the forged split's first request, 16 fields ending in one frame, is not in these cells."""
    shape = Q.Shape(4096, 4, 9)
    [cells], _ = _link(shape, (338, 16), (1, 1, 1, 0))
    frames = [cells[0], cells[9], cells[18]]
    lefts = [0, _frame_reference(4096, frames[0], 0)[4], 0]
    assert [g[1:] for g in _frames(4096, list(zip(frames, lefts)))] == [(1, 1, 0, lefts[1], 337), (1, 0, 1, 0, 1),
                                                                         (1, 1, 1, 0, 16)]
    assert HC.ands(bind(Q.IoTieRequestFrame, B=4096)) == 170_994


def _unit(shape, cases):
    fn = bind(Q.IoTieRequest, B=shape.B, T=shape.T, R=shape.R, OWN=shape.OWN)
    args = [Q.arguments(shape, i, bits) for i, bits in cases]
    out = evaluate_bits(fn, *[_bits([a[k] for a in args]) for k in range(len(args[0]))])
    return [int(b) for b in np.asarray(out).reshape(-1)]


def _swapped_routing(shape, inst):
    """The view's two prompt lengths swapped, with the switches that route each ending cell the entry carrying its
    count of fields."""
    at = Q.cut(shape, inst.cells, inst.view.carry, inst.window).at
    perm, taken = [at[1], at[0]], set(at)
    perm += [i for i in range(shape.M) if i not in taken]
    return inst._replace(view=inst.view._replace(ended=inst.view.ended[::-1])), IT.benes(perm)


@pytest.mark.slow
@pytest.mark.parametrize("own", [0, 1])
def test_the_unit_is_its_reference(own):
    """`IoTieRequest` against `reference` on the link's three windows, honest and under each forged split, REDTEAM's
    phase-6 split at B = 512, each way a cell, a salt, the root or the view can differ, random switches, and a
    two-sided tamper: the view's two prompt lengths swapped with switches that route each ending cell the entry with
    its count, refused by the entry's index. What the frame's test can't reach is the wiring of each cell into its
    bucket's row and its frame, the scan's carry from cell to cell and across bucket boundaries, the routing of the
    view's prompt lengths to the cells that end requests, the own frame, and the totals. Building the rows' SHA-512
    bodies takes most of its time."""
    shape = Q.Shape(512, 4, 2, OWN=own)
    windows, received = _link(shape)
    insts = [i for pl in (PL, *FORGED) for i in _instances(shape, windows, received, pl)]
    [split], rs = _link(shape, (40, 5), (1, 1, 1, 0))
    assert Q.ends((40, 5), 512) == (2, 3) and Q.ends((5, 40), 512) == (1, 3)
    insts += _instances(shape, [split], rs, (40, 5)) + _instances(shape, [split], rs, (5, 40))
    honest = Q.instance(shape, windows[1], Q.request_view(PL, received, 1, 512), 1)
    insts += [Q.instance(shape, c, honest.view, 1) for _, c in _cell_tampers(shape, windows[1])]
    insts += [honest._replace(view=v) for _, v in _view_tampers(honest.view)]
    other = insts[0]
    insts += [honest._replace(salts=(honest.salts[1],) + honest.salts[1:]), honest._replace(root=other.root)]
    want = [Q.reference(shape, i) for i in insts]
    assert want[:14] == [1, 1, 1] + [1, 0, 0] * 3 + [1, 0] and sum(want) == 7 and len(insts) >= 40
    swapped, bits = _swapped_routing(shape, honest)
    rng = random.Random(9)
    cases = [(i, Q.switches(shape, i)) for i in insts]
    cases += [(honest, [rng.randrange(2) for _ in range(shape.switches)]), (swapped, bits),
              (swapped, Q.switches(shape, swapped))]
    assert _unit(shape, cases) == want + [0, 0, 0]
    memo = {}
    a = lambda fn: HC.ands(fn, memo)  # noqa: E731
    steps = sum(a(bind(Q.IoTieRequestStep, D=shape.D, NEW=int(i > 0 and i % shape.R == 0)))
                for i in range(own, shape.N))
    parts = (a(bind(IT.NetworkRowsRoot, N_BITS=8 * shape.B * shape.R, COUNT=shape.T)) + (shape.N - own) *
             a(bind(Q.IoTieRequestFrame, B=shape.B)) + steps + a(bind(IT.IoTieRoute, W=shape.W, D=shape.D)) +
             a(bind(Q.IoTieRequestEnd)) + own * a(bind(Q.IoTieRequestOwn, B=shape.B)) + own + 1)
    assert a(bind(Q.IoTieRequest, B=shape.B, T=shape.T, R=shape.R, OWN=own)) == parts
