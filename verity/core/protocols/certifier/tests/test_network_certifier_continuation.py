"""The continuation rule (`continuation`) on the network certifier's own records: nine network-certifier-form id
requests and their responses of id events, ticked through the network certifier's ingress and egress
(`scan_traffic.serve`: built traffic, nothing served) over four execution phases of T = 8 buckets, r = 2 cells a
bucket, D = 3. At each close b the stamps are the open responses' (q, N, e) in release order, e each response's id
cells in buckets before b, which is the schedule's count from its admission and `warden.audit`'s release order, two
requests released in one bucket numbered by ordinal. The cases at a close:

- a response still inside its delay at b is stamped e = 0, and its continuation is its request;
- one cut at three closes in a row has its whole count at each, the difference its outputs between them, and the same
  continuation whether built from the request or from the close before (`resumed`);
- one whose `END` is in bucket b − 1 has no stamp; nor has one whose last id, in b − 1, is its `max_tokens`-th or an
  end-of-sequence id, its `END` falling in bucket b: `max_tokens` − e = 0 is never a continuation;
- one with an id cell in bucket b itself is stamped with that id still to come, `max_tokens` − e = 1;
- a request whose cells straddle b, and a response the certifier failed before b, have none.

A continuation is the certified request with the released ids after its prompt and `max_tokens` − e left, the pair
(q, P + e); rendered as an id request and released as an arrival of another session it reads back as itself. No cell
at or after b moves a stamp: the records before b give the same edge alone, and beside any later records. Each
disagreement between a record's stamps and the records is refused by its name (a stamp dropped, added for a finished
or unreleased response, doubled, out of release order, of another N, of another e), as are a response past its
`max_tokens` and records the rule can't read."""
from __future__ import annotations

from dataclasses import replace

import pytest
import scan_traffic as ST

from verity.core.protocols.certifier import continuation as C
from verity.core.protocols.certifier import io_request as Q
from verity.core.protocols.certifier import io_tie as IT
from verity.core.protocols.certifier import warden as W
from verity.core.protocols.certifier.active import PAD, id_request
from verity.core.protocols.certifier.grid import Reason, Status

B = 512
PARAMS = W.WardenParams(cell_bytes=B, slots=2, delay=3, lanes=4, window=8, egress_cells=64)
T, R, D = PARAMS.window, PARAMS.slots, PARAMS.delay
EOS = (2,)
FAST = 1 << 20


def _ids(q: int, n: int, last: int | None = None) -> tuple[int, ...]:
    out = [1000 * q + k for k in range(n)]
    return tuple(out[:-1] + [last]) if last is not None else tuple(out)


#: (admission bucket, prompt ids, max_tokens, response ids) of connections 1 … 9; connection 9's server never answers
TRAFFIC = (
    (0, 3, 4, _ids(1, 4)),             # ended in the first execution phase
    (1, 100, 20, _ids(2, 20)),         # four request cells, its END in bucket 3; its ids in buckets 4 … 13
    (2, 5, 6, _ids(3, 6)),             # END in bucket 3 too; its sixth id in bucket 7, its response's END in bucket 8
    (5, 7, 40, _ids(4, 40)),           # released at 6, ids in buckets 8 … 27: open at 8, 16 and 24
    (7, 4, 10, _ids(5, 3, EOS[0])),    # its request's END in bucket 8: it straddles the first close
    (11, 6, 9, _ids(6, 4, EOS[0])),    # its end-of-sequence id in bucket 15, its response's END in bucket 16
    (12, 2, 1, _ids(7, 1)),            # its one id and its END in bucket 15
    (18, 9, 7, _ids(8, 7)),            # ids in buckets 21 … 24: one id in bucket 24
    (19, 3, 5, ()),                    # failed late at its first egress cell, bucket 22
)
LATE = (8,)
#: each close's stamps
STAMPS = {0: (), 8: ((2, 1, 8), (4, 3, 0)), 16: ((4, 3, 16),), 24: ((4, 3, 32), (8, 7, 6)), 32: ()}


def _request(q: int, prompt: int, max_tokens: int) -> C.Request:
    return C.Request(model="unsloth/Llama-3.2-1B", user=f"user-{q}", prompt=tuple(50 * q + k for k in range(prompt)),
                     max_tokens=max_tokens, temperature=0.7, top_p=0.95, seed=11 * q)


REQUESTS = tuple(_request(q, p, m) for q, (_, p, m, _) in enumerate(TRAFFIC, 1))


def _bytes(r: C.Request) -> bytes:
    return id_request(**r._asdict(), frame_bytes=B, offset=0)


def _serve(traffic=TRAFFIC, requests=REQUESTS, late=LATE, windows=4) -> ST.Served:
    return ST.serve(PARAMS, [_bytes(r) for r in requests], [a for a, *_ in traffic], [FAST] * len(traffic),
                    [xs for *_, xs in traffic], windows, late)


@pytest.fixture(scope="module")
def run() -> ST.Served:
    return _serve()


def _edge(run: ST.Served, b: int, eos=EOS) -> C.Edge:
    return C.edge(PARAMS, run.ingress, run.egress, b, eos)


def _with_cell(records, t: int, j: int, cell: bytes | None):
    """The records with bucket t's slot j replaced."""
    out = list(records)
    rows = [list(row) for row in out[t // T].grid]
    rows[t % T][j] = cell
    out[t // T] = replace(out[t // T], grid=tuple(tuple(row) for row in rows))
    return out


def _slot(records, t: int, q: int, k: int = 0) -> int:
    """The slot of connection q's k-th cell in bucket t."""
    row = records[t // T].grid[t % T]
    return [j for j, c in enumerate(row) if c is not None and W.read_cell(c, B)[1] == q][k]


def test_each_closes_stamps_are_the_open_responses_released_counts_in_release_order(run):
    assert run.admitted == tuple(a for a, *_ in TRAFFIC)
    audit = W.audit(PARAMS, run.ingress, run.egress, run.failed)
    assert [w["accepted"] for w in audit["windows"]] == [True, True, False, True]
    conns = {c["ordinal"]: c for c in audit["connections"]}
    assert [conns[q]["ingress_end"] for q in range(1, 10)] == [1, 3, 3, 6, 8, 12, 13, 19, 20]
    assert conns[9]["failed"] == "late" and conns[9]["ended"] == 22
    for b, want in STAMPS.items():
        edge = _edge(run, b)
        assert edge.stamps == want and [(c.q, c.n, c.e) for c in edge.continuations] == list(want)
        responses = C.read(PARAMS, run.ingress, run.egress, b)
        by_release = sorted((c for c in conns.values() if c["ingress_end"] < b),
                            key=lambda c: (c["ingress_end"], c["ordinal"]))
        assert [(r.q, r.n, r.released) for r in responses] == \
            [(c["ordinal"], n, c["ingress_end"]) for n, c in enumerate(by_release)]
        for r in responses:
            s, ids = run.admitted[r.q - 1], run.outputs[r.q]
            scheduled = min(len(ids), max(0, (b - s - D) * R))
            assert C.released(r, b) == ids[:scheduled]
            assert r.cells == tuple((s + D + i // R, x) for i, x in enumerate(ids[:scheduled]))
            assert r.request == REQUESTS[r.q - 1]
            assert r.ended == (conns[r.q]["ended"] if conns[r.q]["ended"] < b else None)


def test_a_continuation_is_its_certified_request_with_the_released_ids_after_its_prompt(run):
    two, four = _edge(run, 8).continuations
    assert (two.q, two.n, two.e) == (2, 1, 8) and len(two.request.prompt) == 100 + 8
    assert two.request == REQUESTS[1]._replace(prompt=REQUESTS[1].prompt + _ids(2, 8), max_tokens=20 - 8)
    assert four.e == 0 and four.request == REQUESTS[3]
    for b in STAMPS:
        for c in _edge(run, b).continuations:
            was = REQUESTS[c.q - 1]
            assert c.request.prompt == was.prompt + run.outputs[c.q][:c.e]
            assert 0 < c.request.max_tokens == was.max_tokens - c.e <= was.max_tokens
            assert c.request._replace(prompt=was.prompt, max_tokens=was.max_tokens) == was
    again = ST.serve(PARAMS, [_bytes(two.request), _bytes(four.request)], [0, 0], [FAST] * 2,
                     [run.outputs[2][8:], run.outputs[4]], 4)
    assert [(r.q, r.n, r.request) for r in C.read(PARAMS, again.ingress, again.egress, 32)] == \
        [(2, 0, four.request), (1, 1, two.request)]


def test_a_response_cut_at_closes_in_a_row_has_one_continuation_however_it_is_built(run):
    def four(b: int) -> C.Response:
        return next(r for r in C.read(PARAMS, run.ingress, run.egress, b) if r.q == 4)

    whole = four(32)
    assert [len(C.released(whole, b)) for b in (8, 16, 24, 32)] == [0, 16, 32, 40]
    assert [C.is_open(whole, b, EOS) for b in (8, 16, 24, 32)] == [True, True, True, False]
    at8 = C.resumed(four(8), 8, EOS)
    assert at8.request == REQUESTS[3] and at8.released == 8 and at8.cells == ()
    at16 = C.resumed(C.resumed(whole, 8, EOS), 16, EOS)
    assert at16 == C.resumed(whole, 16, EOS) and len(at16.request.prompt) == 7 + 16
    assert at16.request.max_tokens == 40 - 16 and at16.cells == whole.cells[16:]
    (stamp,), (cont,) = C.stamps([at16], 24, EOS), C.continuations([at16], 24, EOS)
    assert stamp == (4, 3, 32 - 16)
    direct = next(c for c in _edge(run, 24).continuations if c.q == 4)
    assert cont.request == direct.request and direct.e == 32 and direct.request.max_tokens == 8
    with pytest.raises(ValueError, match="isn't open"):
        C.resumed(whole, 32, EOS)


def test_a_response_that_ends_at_the_close_isnt_continued_and_one_with_an_id_in_its_bucket_is(run):
    at = {b: {r.q: r for r in C.read(PARAMS, run.ingress, run.egress, b)} for b in (8, 16, 24, 32)}
    three = at[8][3]
    assert three.ended is None and len(C.released(three, 8)) == 6 == three.request.max_tokens
    assert at[16][3].ended == 8 and not C.is_open(three, 8, EOS) and not C.is_open(three, 8, ())
    seven, six = at[16][7], at[16][6]
    assert seven.ended == 15 and not C.is_open(seven, 16, EOS)
    assert six.ended is None and at[24][6].ended == 16 and [t for t, _ in six.cells] == [14, 14, 15, 15]
    assert six.cells[-1][1] in EOS and len(six.cells) < six.request.max_tokens
    assert not C.is_open(six, 16, EOS)
    assert C.stamps(list(at[16].values()), 16, ()) == ((4, 3, 16), (6, 5, 4))
    eight = at[32][8]
    assert [t for t, _ in eight.cells] == [21, 21, 22, 22, 23, 23, 24] and eight.ended == 24
    cont = next(c for c in _edge(run, 24).continuations if c.q == 8)
    assert cont.e == 6 and cont.request.max_tokens == 1 and cont.request.prompt[-6:] == _ids(8, 6)
    assert 5 not in at[8] and at[16][5].released == 8 and not C.is_open(at[16][5], 8, EOS)
    for b, want in STAMPS.items():
        assert C.stamps(list(at[32].values()), b, EOS) == want
    assert at[24][9].ended == 22 and at[24][9].cells == () and not C.is_open(at[24][9], 24, EOS)


def test_no_cell_at_or_after_the_close_moves_a_stamp(run):
    other = _serve(traffic=tuple((a, p, m, _ids(q + 20, len(xs))) for q, (a, p, m, xs) in enumerate(TRAFFIC, 1)))
    for b in (8, 16, 24):
        w = b // T
        alone = C.edge(PARAMS, run.ingress[:w], run.egress[:w], b, EOS)
        beside = C.edge(PARAMS, run.ingress[:w] + other.ingress[w:], run.egress[:w] + other.egress[w:], b, EOS)
        assert alone == beside == _edge(run, b)
    for b in (16, 24):
        assert C.edge(PARAMS, other.ingress, other.egress, b, EOS).continuations != _edge(run, b).continuations


@pytest.mark.parametrize("name, written", [
    ("missing", [(4, 3, 32)]),
    ("missing", []),
    ("extra", [(4, 3, 32), (7, 6, 1), (8, 7, 6)]),
    ("extra", [(4, 3, 32), (8, 7, 6), (9, 8, 0)]),
    ("extra", [(4, 3, 32), (8, 7, 6), (10, 9, 0)]),
    ("duplicate", [(4, 3, 32), (4, 3, 32), (8, 7, 6)]),
    ("order", [(8, 7, 6), (4, 3, 32)]),
    ("release", [(4, 2, 32), (8, 7, 6)]),
    ("release", [(4, 7, 32), (8, 3, 6)]),
    ("count", [(4, 3, 33), (8, 7, 6)]),
    ("count", [(4, 3, 32), (8, 7, 7)]),
    ("count", [(4, 3, 16), (8, 7, 6)]),
    ("count", [(4, 3, 32), (8, 7, 0)]),
])
def test_each_disagreement_between_the_stamps_and_the_records_is_refused_by_name(run, name, written):
    responses = C.read(PARAMS, run.ingress, run.egress, 24)
    C.check(STAMPS[24], responses, 24, EOS)
    C.check([C.Stamp(*s) for s in STAMPS[24]], responses, 24, EOS)
    with pytest.raises(C.Refused, match=f"^{name}: ") as e:
        C.check(written, responses, 24, EOS)
    assert e.value.name == name and name in C.REFUSALS
    for b, want in STAMPS.items():
        if b != 24 and want != STAMPS[24]:
            with pytest.raises(C.Refused):
                C.check(want, responses, 24, EOS)


def test_a_response_past_its_max_tokens_is_refused():
    traffic = ((0, 3, 4, _ids(1, 4)), (1, 3, 5, _ids(2, 12)))
    run = _serve(traffic, tuple(_request(q, p, m) for q, (_, p, m, _) in enumerate(traffic, 1)), (), 2)
    assert _edge(run, 0).stamps == ()
    for b in (8, 16):
        with pytest.raises(C.Refused, match="^over: response 2 has (8|12) id cells") as e:
            _edge(run, b)
        assert e.value.name == "over"
    r = C.Response(1, 0, 2, _request(1, 3, 2), ((4, 7), (4, 8), (9, 9)))
    assert C.stamps([r], 4, EOS) == ((1, 0, 0),) and C.stamps([r], 8, EOS) == ()
    with pytest.raises(C.Refused, match="^over: "):
        C.stamps([r], 16, EOS)


def test_records_the_rule_cant_read_are_refused(run):
    def refused(match: str, ingress=run.ingress, egress=run.egress, b: int = 24, params=PARAMS) -> None:
        with pytest.raises(C.Refused, match=f"^records: .*{match}") as e:
            C.edge(params, ingress, egress, b, EOS)
        assert e.value.name == "records"

    for b in (-8, 4, 12, 8.0):
        refused("whole number of execution phases", b=b)
    refused("no complete ingress record of execution phase 2", ingress=run.ingress[:2])
    refused("no complete egress record of execution phase 1", egress=run.egress[:1] + run.egress[2:])
    refused("no complete ingress record of execution phase 4", b=40)
    refused("two egress records of one execution phase", egress=run.egress + run.egress[:1])
    faulted = [replace(run.egress[0], status=Status(Reason.OVERFLOW)), *run.egress[1:]]
    refused("no complete egress record of execution phase 0", egress=faulted, b=8)
    refused("id egress form", params=replace(PARAMS, egress="bytes"))

    j = _slot(run.egress, 9, 4)
    cell = run.egress[1].grid[1][j]
    assert IT.parse_cell(cell).id == 4002
    for bad, why in ((cell[:IT.ID_AT] + b"0" + cell[IT.ID_AT + 1:], "isn't an id cell"),
                     (cell[:-1] + b"\1", "egress bucket 9"),
                     (W.cell(W.REFUSED, 4, b"", B), "isn't an id cell"),
                     (IT.end_cell(4, B)[:-1] + b"\1", "egress bucket 9")):
        refused(why, egress=_with_cell(run.egress, 9, j, bad))
    ended = _with_cell(run.egress, 9, j, IT.end_cell(4, B))
    refused("egress bucket 9: connection 4's cell isn't an id cell", egress=ended)
    refused("egress bucket 10: connection 4's cell isn't an id cell",
            egress=_with_cell(ended, 9, _slot(run.egress, 9, 4, 1), IT.pad_cell(B, 4)))
    twice = _with_cell(run.egress, 15, _slot(run.egress, 15, 7, 0), IT.end_cell(7, B))
    refused("connection 7's end cell isn't its one end", egress=twice, b=16)
    refused("connection 10's cell isn't an id cell", egress=_with_cell(run.egress, 9, j, IT.data_cell(5, 10, B)))
    early = _with_cell(run.egress, 5, _slot(run.egress, 5, 4), IT.data_cell(5, 4, B))
    refused("egress bucket 5: connection 4's cell isn't an id cell", egress=early, b=8)

    i = _slot(run.ingress, 2, 2)
    data = run.ingress[0].grid[2][i]
    assert W.read_cell(data, B)[0] == W.DATA
    lay = Q.layout(B)
    moved, zero, cut = bytearray(data), bytearray(data), bytearray(data)
    at = W.HEADER.size + lay.start + 1
    assert data[at:at + 10] == b"       100"
    moved[at:at + 10], zero[at + 6:at + 7], cut[at - 1:at] = b"100       ", b"0", b"}"
    for bad, why in ((zero, "isn't a canonical id request \\("), (cut, "isn't a canonical id request \\("),
                     (moved, "isn't a canonical id request: its bytes aren't the ones its fields give")):
        refused(f"connection 2's request {why}", ingress=_with_cell(run.ingress, 2, i, bytes(bad)), b=8)
    assert [W.read_cell(c, B)[0] for c in run.ingress[0].grid[1][:2]] == [W.END, PAD]
    refused("connection 1's cell isn't its request's next",
            ingress=_with_cell(run.ingress, 1, _slot(run.ingress, 1, 1, 1), W.cell(W.DATA, 1, b"x" * PARAMS.payload, B)),
            b=8)
    refused("connection 2's cell isn't its request's next",
            ingress=_with_cell(run.ingress, 2, i, W.cell(W.FAIL, 2, b"", B)), b=8)
    assert _edge(run, 0) == C.Edge((), ()) == C.edge(PARAMS, (), (), 0, EOS)


def test_responses_out_of_their_records_order_are_refused():
    one = C.Response(1, 0, 2, _request(1, 3, 9), ((5, 7), (6, 8)))
    two = C.Response(2, 1, 3, _request(2, 3, 9), ((6, 9),))
    assert C.stamps([two, one], 8, EOS) == ((1, 0, 2), (2, 1, 1))
    for bad in ([one, two._replace(q=1)], [one, two._replace(n=0)], [one, two._replace(released=1, cells=())],
                [one._replace(cells=((6, 7), (5, 8))), two], [one._replace(cells=((1, 7),)), two]):
        with pytest.raises(C.Refused, match="^records: "):
            C.stamps(bad, 8, EOS)
        with pytest.raises(C.Refused, match="^records: "):
            C.check([], bad, 8, EOS)
