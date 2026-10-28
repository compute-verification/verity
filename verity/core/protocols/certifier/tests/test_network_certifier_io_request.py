"""IoTieRequest on the warden's own ingress (`io_request`): each request's prompt_len and arrival ordinal held to the
warden's ingress cells, a bucket row at a time, by the reference and by the Boolean Definitions. The first negative
is the swapped split: two requests' prompt lengths swapped with their total kept, which the advised floor alone
accepts."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.core.protocols.certifier import head_check as HC
from verity.core.protocols.certifier import io_request as Q
from verity.core.protocols.certifier import warden as W
from verity.core.protocols.certifier import warden_rows as WR
from verity.core.protocols.certifier.active import DATA, HEADER, PAD, id_request
from verity.core.protocols.certifier.certificate import Direction

B = 512
#: two lanes of r = 2 slots, 8-bucket windows
PARAMS = W.WardenParams(cell_bytes=B, slots=2, lanes=2, window=8, delay=1)
SHAPE = Q.Shape(B, PARAMS.lanes, PARAMS.slots)
#: request 1 (100 tokens, 3 cells) and request 3 (60, 3) trickle in over buckets, the others arrive whole, and
#: request 4 arrives alone at the last bucket of window 0 and ends in window 1
PL, ARRIVALS, RATES = (5, 100, 12, 60, 1), (0, 0, 1, 2, 7), (10 ** 6, 1100, 10 ** 6, 600, 10 ** 6)


def _ids(i, k):
    return (7919 * i + 104729 * k) % 151329


def _requests(prompt_lens, B=B, idfn=_ids):
    return [id_request(model="glm", user=f"u{i}", prompt=[idfn(i, k) for k in range(n)], max_tokens=16,
                       temperature=0.7, top_p=0.95, seed=i, frame_bytes=B, offset=0)
            for i, n in enumerate(prompt_lens)]


@pytest.fixture(scope="module")
def link():
    return Q.serve(PARAMS, _requests(PL), ARRIVALS, RATES, buckets=2 * PARAMS.window)


def _verdicts(link, prompt_lens, shape=SHAPE, order=None):
    """Each bucket's verdict on the verifier's lanes from `prompt_lens`, or None where the verifier's lanes refuse the
    advice against the warden's release."""
    try:
        return [Q.reference(shape, Q.Instance(tuple(row), Q.lanes(shape, prompt_lens, link.releases, t, order)))
                for t, row in enumerate(link.rows)]
    except ValueError:
        return None


def _accepted(verdicts):
    return verdicts is not None and all(verdicts)


@pytest.mark.parametrize("B", [512, 4096])
def test_a_request_parses_to_its_prompt_length(B):
    """Every cell of an id request as the warden cuts it (`warden.cells`) parses, and its cells' fields and bytes are
    its prompt length and its own; `carries` is that cut's carry after each cell."""
    lay = Q.layout(B)
    for n in (1, 2, 11, 12, 13, 56, 57, 58, 336, 337, 338, 339, 708, 709):
        raw = W.cells(_requests([n], B)[0], B, 1)
        cells = [raw[k:k + B] for k in range(0, len(raw), B)]
        assert len(cells) == len(Q.carries(n, B)) - 1
        left = fields = 0
        for k, c in enumerate(cells):
            got = Q.read(lay, c, 1, left)
            assert isinstance(got, Q.Frame), (n, k)
            left, fields = (left or got.length) - got.used, fields + got.fields
            assert (left, fields) == Q.carries(n, B)[k + 1]
        assert (left, fields) == (0, n)


def test_every_bucket_of_the_wardens_ingress_is_accepted(link):
    """The link served through the warden's own ingress (`warden.Warden._ingress`): every bucket row is accepted on the
    verifier's lanes, which chain each connection from (0, 0) to (0, n) over its buckets. Requests trickle in over
    three buckets, two connections share buckets, and a connection moves to lane 0 when the one before it ends."""
    assert _verdicts(link, PL) == [1] * len(link.rows)
    held = {}
    for t in range(len(link.rows)):
        for rank, ln in enumerate(Q.lanes(SHAPE, PL, link.releases, t)):
            if ln.q:
                held.setdefault(ln.q, []).append((rank, ln))
    for q, got in held.items():
        lanes_ = [ln for _, ln in got]
        assert (lanes_[0].left, lanes_[0].fields) == (0, 0) and (lanes_[-1].left_out, lanes_[-1].fields_out) == \
            (0, PL[q - 1])
        assert all((a.left_out, a.fields_out) == (b.left, b.fields) for a, b in zip(lanes_, lanes_[1:]))
    assert max(len(g) for g in held.values()) >= 3 and {r for g in held.values() for r, _ in g} == {0, 1}
    assert any(len({r for r, _ in g}) == 2 for g in held.values())
    kinds = {HEADER.unpack_from(c)[0] for row in link.rows for c in row}
    assert kinds == {PAD, DATA, W.END}


FORGED = ((5, 101, 12, 59, 1), (5, 99, 12, 61, 1), (5, 60, 12, 100, 1))


def test_a_forged_advice_is_refused(link):
    """Lengths one off either way that keep every request's cells, so the release can't see them, are refused by a
    bucket's verdict; a swap that keeps the total moves a request's cells, so the verifier's lanes refuse it."""
    for forged in FORGED[:2]:
        assert [len(Q.carries(n, B)) for n in forged] == [len(Q.carries(n, B)) for n in PL]
        v = _verdicts(link, forged)
        assert v is not None and not all(v), forged
    assert _verdicts(link, FORGED[2]) is None or not all(_verdicts(link, FORGED[2]))
    assert not _accepted(_verdicts(link, PL[:-1])) and not _accepted(_verdicts(link, PL + (3,)))


def test_redteam_phase6_split_is_refused():
    """The swapped split, as IoTieRequest's first negative. At B = 4,096 a 400-token prompt takes 2 cells and a
    16-token one 1, released one a bucket (r = 1). The forged advice swaps the two prompt lengths, keeping the total,
    so its ends move and the advised floor alone accepts an acausal release, unless each request's prompt_len is bound
    to the ingress record's content. The tie reads 400 fields from connection 1's cells, so the swap's lanes are
    refused by the release and, built anyway, by the unit."""
    params = W.WardenParams(cell_bytes=4096, slots=1, lanes=2, window=4, delay=1)
    shape = Q.Shape(4096, 2, 1)
    LONG, SHORT = 400, 16
    link = Q.serve(params, _requests((LONG, SHORT), 4096), (0, 0), (10 ** 6, 10 ** 6), buckets=4)
    assert [r.cells for r in link.releases] == [(0, 1, 1), (0, 1)]
    assert _verdicts(link, (LONG, SHORT), shape) == [1, 1, 1, 1]
    assert _verdicts(link, (SHORT, LONG), shape) is None
    forged = [Q.Lane(1, *Q.carries(SHORT, 4096)[0], *Q.carries(SHORT, 4096)[1]),
              Q.Lane(2, *Q.carries(LONG, 4096)[0], *Q.carries(LONG, 4096)[1])]
    assert Q.reference(shape, Q.Instance(tuple(link.rows[1]), tuple(forged))) == 0


def test_the_advice_order_names_each_requests_connection():
    """Advice request i is on connection order[i] + 1 (advice v1's order permutation): advice listing the requests in
    another order than they arrived is accepted with its order and refused with the identity, or another order."""
    arrived = (60, 5, 12)
    link = Q.serve(PARAMS, _requests(arrived), (0, 0, 1), (600, 10 ** 6, 10 ** 6), buckets=PARAMS.window)
    advice, order = (5, 12, 60), (1, 2, 0)
    assert _accepted(_verdicts(link, advice, order=order))
    for bad in (None, (0, 1, 2), (2, 1, 0), (1, 0, 2)):
        assert not _accepted(_verdicts(link, advice, order=bad)), bad
    with pytest.raises(ValueError, match="permutation"):
        Q.lanes(SHAPE, advice, link.releases, 0, (1, 1, 0))


def test_the_lanes_refuse_a_release_other_than_the_wardens(link):
    """`lanes` raises where the release isn't the warden's schedule: ordinals out of order or admitted out of order, a
    connection with no advice request, cells that aren't its request's or past R a bucket, no `END` in its last
    bucket, or more live connections than lanes."""
    rel = list(link.releases)
    bad = {"renumbered": [r._replace(q=r.q + 1) for r in rel],
           "admitted out of order": [rel[0]._replace(first=3)] + rel[1:],
           "a connection more": rel + [Q.Release(len(rel) + 1, 9, (1,))],
           "a cell more": [rel[0]._replace(cells=rel[0].cells + (1,))] + rel[1:],
           "a cell less": [rel[0], rel[1]._replace(cells=rel[1].cells[:-2] + (rel[1].cells[-2] - 1,
                                                                             rel[1].cells[-1]))] + rel[2:],
           "past R": [rel[0], rel[1]._replace(cells=(0, 3))] + rel[2:],
           "no END in its last": [rel[0]._replace(cells=rel[0].cells + (0,))] + rel[1:]}
    for what, r in bad.items():
        with pytest.raises(ValueError):
            [Q.lanes(SHAPE, PL[:len(r)], r, t) for t in range(len(link.rows))]
    with pytest.raises(ValueError, match="past L"):
        Q.lanes(Q.Shape(B, 1, 2), PL, rel, 2)


def _tamper(cell, at, new):
    return cell[:at] + new + cell[at + len(new):]


def _first_cell_tampers(first):
    """(what, cell) for each way a request's first cell (a `DATA`) can differ from the warden's cell of a canonical id
    request."""
    lay, h = Q.layout(len(first)), HEADER.size
    kind, q, n = HEADER.unpack_from(first)
    assert kind == DATA and n == lay.room
    cl, ratio = lay.spans[0][0], lay.spans[2][0]
    model, user = lay.spans[5][0], lay.spans[6][0]
    field = h + lay.start
    body = int(first[h + cl:h + cl + 10])
    return [
        ("the head's method", _tamper(first, h, b"GET ")),
        ("a fixed key", _tamper(first, h + ratio - 2, b"x")),
        ("content-length one less", _tamper(first, h + cl, b"%10d" % (body - 1))),
        ("content-length one more", _tamper(first, h + cl, b"%10d" % (body + 1))),
        ("a prompt field's digit", _tamper(first, field + 10, b"x")),
        ("a prompt field's leading zero", _tamper(first, field + 1, b"0000000001")),
        ("a prompt id of 2^32", _tamper(first, field + 1, b"4294967296")),
        ("the first field's bracket", _tamper(first, field, b",")),
        ("a name's unescaped quote", _tamper(first, h + model + 1, b'"')),
        ("a name's escape", _tamper(first, h + model + 1, b"\\")),
        ("a user past its quotes", _tamper(first, h + user + 6, b"x")),
        ("a ratio's form", _tamper(first, h + ratio, b"0.7     ")),
        ("another connection's ordinal", HEADER.pack(kind, q + 1, n) + first[h:]),
        ("ordinal 0", HEADER.pack(kind, 0, n) + first[h:]),
        ("the header's length", HEADER.pack(kind, q, n - 1) + first[h:]),
        ("an END's kind", HEADER.pack(W.END, q, n) + first[h:]),
        ("a PAD's kind", HEADER.pack(PAD, q, n) + first[h:]),
        ("a FAIL's kind", HEADER.pack(W.FAIL, q, n) + first[h:]),
    ]


def _row_tampers(row, lanes_):
    """(what, row) for each way a bucket row can differ from the warden's, as a function of the row and its lanes."""
    R, out = SHAPE.R, []
    lane_of = {i: lanes_[i // R] for i in range(len(row))}

    def at(i, cell):
        return [*row[:i], cell, *row[i + 1:]]

    def kind(i):
        return HEADER.unpack_from(row[i])[0]

    first = next(i for i in range(len(row)) if kind(i) == DATA and lane_of[i].q and not lane_of[i].left and
                 i % R == 0)
    out += [(w, at(first, c)) for w, c in _first_cell_tampers(row[first])]
    end = next(i for i in range(len(row)) if kind(i) == W.END and (i + 1) % R and kind(i + 1) == PAD)
    filled = row[end].rstrip(b"\0")
    swapped = list(row)
    swapped[end], swapped[end + 1] = swapped[end + 1], swapped[end]
    lanes_swapped = list(row[R:2 * R]) + list(row[:R]) + list(row[2 * R:])
    out += [("a byte in the fill", at(end, _tamper(row[end], len(filled) + 1, b"\1"))),
            ("the end", at(end, _tamper(row[end], len(filled) - 2, b"] "))),
            ("a request cell after a pad", swapped),
            ("a PAD on another ordinal", at(end + 1, W.cell(PAD, lane_of[end].q + 1, b"", B))),
            ("a PAD with a payload", at(end + 1, _tamper(row[end + 1], HEADER.size, b"\1"))),
            ("the END as a DATA", at(end, _tamper(row[end], 0, bytes([DATA])))),
            ("the lanes' cells swapped", lanes_swapped)]
    return out


def _row_with(link, pred):
    return next(t for t, row in enumerate(link.rows) if pred(row, Q.lanes(SHAPE, PL, link.releases, t)))


def test_each_way_a_cell_can_differ_is_refused(link):
    def rich(row, lanes_):
        kinds = [HEADER.unpack_from(c)[0] for c in row]
        return all(ln.q for ln in lanes_) and DATA in kinds and W.END in kinds and \
            any(k == W.END and (i + 1) % SHAPE.R and kinds[i + 1] == PAD for i, k in enumerate(kinds)) and \
            any(k == DATA and i % SHAPE.R == 0 and not lanes_[i // SHAPE.R].left for i, k in enumerate(kinds))

    t = _row_with(link, rich)
    lanes_ = Q.lanes(SHAPE, PL, link.releases, t)
    row = link.rows[t]
    assert Q.reference(SHAPE, Q.Instance(tuple(row), lanes_)) == 1
    cases = _row_tampers(row, lanes_)
    assert len(cases) >= 25
    for what, bad in cases:
        assert Q.reference(SHAPE, Q.Instance(tuple(bad), lanes_)) == 0, what
    idle = _row_with(link, lambda r, ls: ls[0].q and not ls[1].q)
    lanes_ = Q.lanes(SHAPE, PL, link.releases, idle)
    row = list(link.rows[idle])
    assert Q.reference(SHAPE, Q.Instance(tuple(row), lanes_)) == 1 and row[-1] == bytes(B)
    for bad in (W.cell(PAD, lanes_[0].q, b"", B), _tamper(bytes(B), B - 1, b"\1")):
        assert Q.reference(SHAPE, Q.Instance(tuple(row[:-1] + [bad]), lanes_)) == 0


def _lane_tampers(lanes_):
    """(what, lanes) for each way the lanes can differ from the release's."""
    out = []
    for k, ln in enumerate(lanes_):
        if not ln.q:
            continue
        for f in Q.Lane._fields:
            for d in (1, -1):
                if getattr(ln, f) + d >= 0:
                    out.append((f"lane {k}'s {f} {d:+d}", lanes_[:k] + (ln._replace(**{f: getattr(ln, f) + d}),) +
                                lanes_[k + 1:]))
        out.append((f"lane {k} unheld", lanes_[:k] + (Q.UNHELD,) + lanes_[k + 1:]))
    if all(ln.q for ln in lanes_):
        out.append(("the lanes swapped", lanes_[::-1]))
    return out


def test_lanes_other_than_the_releases_are_refused(link):
    t = _row_with(link, lambda r, ls: all(ln.q for ln in ls) and any(ln.left or ln.left_out for ln in ls))
    lanes_ = Q.lanes(SHAPE, PL, link.releases, t)
    inst = Q.Instance(tuple(link.rows[t]), lanes_)
    assert Q.reference(SHAPE, inst) == 1
    for what, bad in _lane_tampers(lanes_):
        assert Q.reference(SHAPE, inst._replace(lanes=bad)) == 0, what


def test_a_client_that_ends_after_its_last_byte_is_refused():
    """A non-canonical client: its request's bytes fill a whole `DATA` (338 tokens at B = 4,096, exactly P bytes) and
    it sends its `END` empty after it, so the warden releases a `DATA` then an empty `END`. The tie reads the
    request's last byte in the `DATA`, where only an `END` may hold it, and refuses. The client shim (`warden.cells`)
    sends those bytes as one `END`, which is accepted."""
    shape = Q.Shape(4096, 1, 2)
    raw = _requests([338], 4096)[0]
    assert len(raw) == shape.B - HEADER.size
    lane = Q.Lane(1, *Q.carries(338, 4096)[0], *Q.carries(338, 4096)[-1])
    honest = (W.cell(W.END, 1, raw, 4096), W.cell(PAD, 1, b"", 4096))
    late = (W.cell(DATA, 1, raw, 4096), W.cell(W.END, 1, b"", 4096))
    assert Q.reference(shape, Q.Instance(honest, (lane,))) == 1
    assert Q.reference(shape, Q.Instance(late, (lane,))) == 0


def test_each_buckets_row_is_the_wardens_registered_row(link):
    """The unit's row is the warden's registration v2 (`warden_rows`): bucket t's row is row t of the ingress record the
    grid closed, the rows open the warden's root, the verifier's port (`bucket_port`) has unit t read row t, and the
    host's registered read refuses rows out of place or edited."""
    assert len(link.records) == 2
    for w, record in enumerate(link.records):
        cells = WR.Cells.of("requests", Direction.INGRESS, record, PARAMS, hashlib.sha512(b"session").digest())
        T = PARAMS.window
        assert cells.rows == tuple(b"".join(r) for r in link.rows[w * T:(w + 1) * T])
        port = WR.bucket_port(cells.head, PARAMS)
        assert (port["reads"], port["leaves"], port["words"]) == (list(range(T)), T, SHAPE.N * B // 2)
        _, _, entry = cells.statement(list(range(T)))
        assert WR.refusal(port, cells.domain, entry, list(cells.bcs)) is None
        swapped = [1, 0, *range(2, T)]
        assert "unit 0 reads position 1" in \
            WR.refusal(port, cells.domain, cells.statement(swapped)[2], [cells.bcs[t] for t in swapped])
        edited = [WR.bc(_tamper(cells.rows[0], HEADER.size, b"\1"), cells.salts[0]), *cells.bcs[1:]]
        assert "does not open" in WR.refusal(port, cells.domain, entry, edited)


def test_a_cell_too_small_for_the_head_is_refused():
    for B in (128, 256):
        with pytest.raises(ValueError, match="no room|doesn't fit"):
            Q.Shape(B, 4, 3)


# --- the circuit -----------------------------------------------------------------------------------------------------

def _bits(rows):
    return np.asarray(rows, dtype=np.uint8)


def _int(bits) -> int:
    return sum(int(b) << j for j, b in enumerate(bits))


def _frame_reference(B, cell, left, q):
    """`IoTieRequestFrame`'s output in the clear: (pad, req, start), and, for a request cell, (ended, left, fields)."""
    got = Q.read(Q.layout(B), cell, q, left)
    head = (int(got == Q.PADDED), int(isinstance(got, Q.Frame)), int(left == 0))
    if not isinstance(got, Q.Frame):
        return head
    rest = (left or got.length) - got.used
    return head + (int(rest == 0), rest, got.fields)


def _frames(B, cases):
    args = Q.frame_arguments(*zip(*cases))
    out = np.asarray(evaluate_bits(bind(Q.IoTieRequestFrame, B=B), *[_bits(a) for a in args]))
    got = []
    for o in out.reshape(len(cases), Q.FRAME_OUT):
        head = (int(o[0]), int(o[1]), int(o[2]))
        got.append(head + ((int(o[3]), _int(o[4:36]), _int(o[36:68])) if o[1] else ()))
    return got


def test_the_frame_circuit_is_its_reference():
    """`IoTieRequestFrame` against `read` on every cell of requests of several prompt lengths, each read against its
    request's true bytes left and against lefts one off, one slot off and 2 off, on its own ordinal, another and 0,
    on each way a first cell can differ, and on pads."""
    cases = []
    for n, raw in zip((1, 2, 11, 12, 13, 40, 56, 57, 58, 100), (W.cells(r, B, 3) for r in _requests(
            (1, 2, 11, 12, 13, 40, 56, 57, 58, 100)))):
        left = 0
        for k in range(0, len(raw), B):
            c = raw[k:k + B]
            cases += [(c, x, q) for x in sorted({left, left + 1, left - 1, left + 2, left + 11, 1}) if x >= 0
                      for q in (3, 4, 0)]
            left = _frame_reference(B, c, left, 3)[4]
    first = W.cells(_requests((40,))[0], B, 3)[:B]
    cases += [(c, 0, 3) for _, c in _first_cell_tampers(first)]
    pad = W.cell(PAD, 3, b"", B)
    cases += [(pad, 0, 3), (pad, 7, 3), (pad, 0, 2), (bytes(B), 0, 0), (bytes(B), 0, 1), (_tamper(pad, 20, b"\1"), 0, 3)]
    want = [_frame_reference(B, *c) for c in cases]
    assert sum(w[1] for w in want) >= 26 and sum(w[0] for w in want) == 3 and len(cases) >= 280
    assert _frames(B, cases) == want
    assert HC.ands(bind(Q.IoTieRequestFrame, B=B)) == 20_876


@pytest.mark.slow
def test_redteam_phase6_cells_read_their_prompt_lengths():
    """The swapped split's cells at B = 4,096 through `IoTieRequestFrame`: the 400-token request's first cell doesn't
    end it and holds 338 prompt fields, its second ends it with 62 more, and the 16-token request's one cell ends it
    with 16. So the forged split's first request, 16 fields ending in one cell, is not in these cells.

    Slow (about 7 s, the served cell's ANDs traced and evaluated): the one test of the circuit on these cells
    at the served cell size, which the reference's quick test reads in the clear and the unit's test reads at
    B = 512."""
    long_, short = (W.cells(r, 4096, q) for r, q in zip(_requests((400, 16), 4096), (1, 2)))
    cases = [(long_[:4096], 0, 1), (long_[4096:], Q.carries(400, 4096)[1][0], 1), (short, 0, 2)]
    assert [g[1:] for g in _frames(4096, cases)] == [(1, 1, 0, cases[1][1], 338), (1, 0, 1, 0, 62),
                                                     (1, 1, 1, 0, 16)]
    assert HC.ands(bind(Q.IoTieRequestFrame, B=4096)) == FRAME_4096


#: `IoTieRequestFrame{B = 4096}`'s ANDs, the served cell's
FRAME_4096 = 166_615


def _state(left, fields, over, padded, ok):
    return HC._int_bits(left, 32) + HC._int_bits(fields, 32) + [over, padded, ok]


def _frame_out(pad, req, start, ended, left, fields):
    return [pad, req, start, ended] + HC._int_bits(left, 32) + HC._int_bits(fields, 32)


def test_the_step_keeps_the_lanes_scan():
    """`IoTieRequestStep` on a scan's state and a frame: a request cell takes its bytes left and adds its fields
    (carrying past 2^32 into over), a pad sets padded, and ok drops for a cell that's neither, a request cell after a
    pad, or one after the request's `END` (left 0 with fields)."""
    cases = [
        (_state(0, 0, 0, 0, 1), _frame_out(0, 1, 1, 0, 400, 30), _state(400, 30, 0, 0, 1)),
        (_state(400, 30, 0, 0, 1), _frame_out(0, 1, 0, 1, 0, 7), _state(0, 37, 0, 0, 1)),
        (_state(0, 37, 0, 0, 1), _frame_out(1, 0, 1, 0, 0, 0), _state(0, 37, 0, 1, 1)),
        (_state(0, 37, 0, 1, 1), _frame_out(1, 0, 1, 0, 0, 0), _state(0, 37, 0, 1, 1)),
        (_state(0, 37, 0, 0, 1), _frame_out(0, 1, 1, 1, 0, 5), _state(0, 42, 0, 0, 0)),
        (_state(9, 3, 0, 1, 1), _frame_out(0, 1, 0, 1, 0, 1), _state(0, 4, 0, 1, 0)),
        (_state(9, 3, 0, 0, 1), _frame_out(0, 0, 0, 0, 0, 0), _state(9, 3, 0, 0, 0)),
        (_state(9, (1 << 32) - 1, 0, 0, 1), _frame_out(0, 1, 0, 1, 0, 2), _state(0, 1, 1, 0, 1)),
        (_state(0, 0, 0, 0, 0), _frame_out(1, 0, 1, 0, 0, 0), _state(0, 0, 0, 1, 0)),
    ]
    out = evaluate_bits(bind(Q.IoTieRequestStep), _bits([s for s, _, _ in cases]), _bits([f for _, f, _ in cases]))
    assert np.asarray(out).reshape(len(cases), Q.STATE).tolist() == [w for _, _, w in cases]


def test_the_end_holds_the_carry_out():
    """`IoTieRequestEnd` on a scan's state: 1 iff ok, over clear, and the bytes left and prompt fields are left_out and
    fields_out, each one bit off in turn; padded is free."""
    honest = (310, 12, 0, 1, 1)
    cases = [(honest, (310, 12), 1), ((310, 12, 0, 0, 1), (310, 12), 1), ((310, 12, 1, 0, 1), (310, 12), 0),
             ((310, 12, 0, 0, 0), (310, 12), 0)]
    for k in range(2):
        for b in (0, 31):
            pub = [310, 12]
            pub[k] ^= 1 << b
            cases.append((honest, tuple(pub), 0))
    args = [[_state(*s) for s, _, _ in cases]] + [[HC._int_bits(p[k], 32) for _, p, _ in cases] for k in range(2)]
    out = evaluate_bits(bind(Q.IoTieRequestEnd), *[_bits(a) for a in args])
    assert [int(b) for b in np.asarray(out).reshape(-1)] == [w for _, _, w in cases]


def test_the_unit_is_its_reference(link):
    """`IoTieRequest` against `reference` on every bucket row of the link, honest and under each forged advice's lanes,
    each way a cell or a lane can differ, and the late `END`'s lane at B = 512. What the frame's test can't reach is
    the wiring of each slot into its lane and its frame, the scan's carry from cell to cell from the lane's carry-in,
    the ordinal each lane's cells are read on, and the carry-out."""
    insts = []
    for prompt_lens in (PL, *FORGED[:2]):
        insts += [Q.Instance(tuple(row), Q.lanes(SHAPE, prompt_lens, link.releases, t))
                  for t, row in enumerate(link.rows)]
    t = next(t for t in range(len(link.rows)) if Q.reference(SHAPE, insts[t]) and len(
        {HEADER.unpack_from(c)[0] for c in link.rows[t]}) == 3 and all(ln.q for ln in insts[t].lanes))
    honest = insts[t]
    insts += [honest._replace(cells=tuple(bad)) for _, bad in _row_tampers(link.rows[t], honest.lanes)]
    insts += [honest._replace(lanes=bad) for _, bad in _lane_tampers(honest.lanes)]
    want = [Q.reference(SHAPE, i) for i in insts]
    n = len(link.rows)
    assert want[:n] == [1] * n and not all(want[n:2 * n]) and not all(want[2 * n:3 * n])
    assert sum(want[3 * n:]) == 0 and len(insts) >= 90
    fn = bind(Q.IoTieRequest, B=B, L=SHAPE.L, R=SHAPE.R)
    args = [Q.arguments(SHAPE, i) for i in insts]
    out = evaluate_bits(fn, *[_bits([a[k] for a in args]) for k in range(2)])
    assert [int(b) for b in np.asarray(out).reshape(-1)] == want
    a = HC.ands
    parts = SHAPE.N * (a(bind(Q.IoTieRequestFrame, B=B)) + a(bind(Q.IoTieRequestStep))) + \
        SHAPE.L * a(bind(Q.IoTieRequestEnd)) + a(BL.and_gate(SHAPE.L))
    assert a(fn) == parts == 84_175
