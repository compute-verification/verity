"""The I/O tie's prototype (`io_tie`) on the warden's own egress: variable-length responses on a link served through
the warden's schedule (`warden.Warden._egress` cuts every cell, the certifier's grid lays them out a row a bucket, as
`warden_rows` registers them); advice-v0-shaped outputs, from which the verifier computes each window's map (through
advice v1's order permutation where the engine's numbering isn't the arrival order, as on TRACE-CORPUS's heavy trace);
and the Commit's sampled ids, one hiding row a step, which each bucket's unit opens at its outputs' steps. The
reference accepts every honest window and refuses every way the cells can differ from the map and the port, every
port that isn't the Commit's and a connection the warden failed; the Boolean parts agree with it."""

import hashlib
import json
import random
from collections import deque
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.commitments.gates.frame_v3 import opens
from verity.ml.commitments.merkle import tree_levels
from verity.ml.commitments.rowleaf import row_leaf
from verity.core.protocols.certifier import head_check as HC, io_tie as IT, warden as W, warden_rows as WR
from verity.core.protocols.certifier.active import DATA, HEADER, PAD
from verity.core.protocols.certifier.certificate import Direction
from verity.core.protocols.certifier.grid import Certifier

#: the circuits' shape: three bucket rows of 4 lanes of r = 2 slots, each bucket's 4 port rows of 4 ids from a session
#: of 8 steps, M = 16 routed positions
SMALL = IT.Shape(256, 3, 8, P=4, S=4, DS=3)
#: the reference's shape: windows of six bucket rows of 6 lanes of r = 2 slots, each bucket's 4 port rows of 8 ids
#: from a session of 128 steps
BIG = IT.Shape(256, 6, 12, P=8, S=4, DS=7)
V = 151_329


def _h(label: bytes) -> bytes:
    return hashlib.sha512(label).digest()


def wardens(shape: IT.Shape, slots: int, delay: int) -> W.WardenParams:
    """The warden's schedule a shape reads: B, T and lanes·r = R."""
    return W.WardenParams(cell_bytes=shape.B, slots=slots, delay=delay, lanes=shape.R // slots, window=shape.T)


class Link(NamedTuple):
    """A fake upstream's link served through the warden: its schedule, its egress rows (bucket b's R cells, a slot no
    connection holds B zero bytes), advice v0's outputs and release buckets, each advice request's admission bucket
    and arrival order (`order`), the Commit's rows of sampled ids (one a step of the session), the warden's
    `fails` frames, and the egress records its grid closed, a window each."""

    params: W.WardenParams
    rows: list[list[bytes]]
    outputs: list[IT.Output]
    release: list[int]
    admitted: list[int]
    order: list[int]
    port: list[bytes]
    fails: list[bytes]
    records: tuple = ()


def serve(shape: IT.Shape, params: W.WardenParams, arrivals: list[int], responses: list[list[int]], buckets: int,
          engine: list[int] | None = None, late: frozenset[int] = frozenset()) -> Link:
    """Connection k, ordinal k + 1, arrives at bucket arrivals[k] with response responses[k] (its ids), and the warden's
    own egress (`warden.Warden._egress`) serves it for `buckets` buckets: admitted at the first bucket with a free lane
    in arrival order, its server's bytes all there (its events of P bytes each, then `TRAILER`), but for a connection
    in `late`, whose last event never comes, so the warden fails it there. Advice request i is arrival engine[i] (the
    identity by default), its output j sampled at step r·s + (i mod 2) + j, in the batch row of its rank by i among
    the step's requests."""
    B, r, D, P = params.cell_bytes, params.slots, params.delay, params.payload
    fails: list[bytes] = []
    warden = SimpleNamespace(params=params, fails=SimpleNamespace(enqueue=fails.append))
    conns = []
    for k, (a, xs) in enumerate(zip(arrivals, responses, strict=True)):
        c = W._Connection(ordinal=k + 1, reader=None, writer=None, done=None, arrived=a)
        c.egress = bytearray(b"".join(IT.id_event(x, P) for x in xs[:len(xs) - (k in late)]))
        if k not in late:
            c.egress += IT.TRAILER
        c.egress_ended = k not in late
        conns.append(c)
    queue, active, grid, rows, records = deque(conns), [], Certifier("responses", params.grid), [], []
    for t in range(buckets):
        while queue and queue[0].arrived <= t and len(active) < params.lanes:
            c = queue.popleft()
            c.admitted, c.pad = t, W.cell(PAD, c.ordinal, b"", B)
            active.append(c)
        for c in active:
            for frame in W.Warden._egress(warden, c, t):
                grid.enqueue(frame)
        active = [c for c in active if c.ended is None]
        rows.append([bytes(B) if f is None else f for f in grid.tick()])
        if (t + 1) % params.window == 0:
            records.append(grid.close_window())
    assert not queue and not active, "every connection ends within the link's buckets"
    engine = list(range(len(conns))) if engine is None else engine
    computed: dict[int, list[tuple[int, int, int]]] = {}
    for i, k in enumerate(engine):
        for j in range(len(responses[k])):
            computed.setdefault(r * conns[k].admitted + i % 2 + j, []).append((i, j, responses[k][j]))
    outputs, release, ids = [], [], {}
    for step, batch in sorted(computed.items()):
        for row, (i, j, x) in enumerate(sorted(batch)):
            outputs.append(IT.Output(step, row, i, j))
            release.append(conns[engine[i]].admitted + D + j // r)
            ids[step, row] = x
    assert max(computed, default=0) < 1 << shape.DS and all(len(b) <= shape.P for b in computed.values())
    port = [IT.port_row([ids.get((s, row), 0) for row in range(shape.P)], shape.P) for s in range(1 << shape.DS)]
    return Link(params, rows, outputs, release, [conns[k].admitted for k in engine], engine, port, fails, tuple(records))


def commit(shape: IT.Shape, link: Link, window: int, seed: bytes = b"w", order: list[int] | None = None) -> IT.Instance:
    """Window `window`'s tie: its rows committed as the warden registers them (a row a bucket), the map and each
    bucket's steps the verifier computes from the advice (with `order`, the link's by default), and each bucket's port
    rows opened from the Commit's rows at its steps."""
    p = link.params
    cmap, steps = IT.window_map(shape, link.outputs, link.release, window, link.admitted, p.delay, p.slots,
                                link.order if order is None else order)
    cells = [c for t in range(shape.T) for c in link.rows[window * shape.T + t]]
    cd, pd = IT.domain(_h(b"cells" + seed), shape.T), IT.domain(_h(b"port" + seed), len(link.port))
    cs = [IT.fake_salt(seed + b"c", t) for t in range(shape.T)]
    ps = [IT.fake_salt(seed + b"p", s) for s in range(len(link.port))]
    levels = IT.rows_levels(pd.domain_id, link.port, ps)
    opened = [IT.open_port(levels, link.port, ps, st) for st in steps]
    return IT.Instance(shape, tuple(cells), tuple(cs), IT.rows_root(cd.domain_id, IT.bucket_rows(shape, cells), cs),
                       cd.domain_id, cmap, steps, tuple(o[0] for o in opened), tuple(o[1] for o in opened),
                       tuple(o[2] for o in opened), levels[-1][0], pd.domain_id)


def with_cells(inst: IT.Instance, cells) -> IT.Instance:
    """The window with the warden's cells replaced, committed as the warden commits what it saw."""
    return replace(inst, cells=tuple(cells),
                   cells_root=IT.rows_root(inst.cells_domain, IT.bucket_rows(inst.shape, cells), inst.cell_salts))


def busy(seed: int, shape: IT.Shape = BIG, slots: int = 2, delay: int = 2, count: int = 14, longest: int = 15,
         **kw) -> Link:
    """`count` connections arriving over the link's first buckets, responses of 0 … `longest` ids."""
    rng = random.Random(seed)
    arrivals = sorted(rng.randrange(3 * shape.T) for _ in range(count))
    responses = [[rng.randrange(V) for _ in range(rng.randrange(longest + 1))] for _ in range(count)]
    return serve(shape, wardens(shape, slots, delay), arrivals, responses, 6 * shape.T, **kw)


def small(**kw) -> Link:
    """Window 0 of SMALL's link: two connections admitted at once, one with an empty response that frees its lane, a
    third admitted into it, and a fourth after."""
    return serve(SMALL, wardens(SMALL, 2, 1), [0, 0, 0, 0, 1], [[11, 12, 13], [], [14], [15, 16, 17, 18, 19], [20, 21]],
                 6, **kw)


def busiest(link: Link, shape: IT.Shape = BIG) -> int:
    """The window of the link with the most data cells."""
    return max(range(len(link.rows) // shape.T), key=lambda w: sum(not _marker(m) for m in commit(shape, link, w).map))


@pytest.fixture(scope="module")
def link():
    return busy(1)


@pytest.fixture(scope="module")
def big(link):
    return commit(BIG, link, busiest(link))


def _marker(m: IT.MapEntry) -> bool:
    return m.e in (IT.EMPTY, IT.END_ENTRY)


def test_a_cell_parses_to_its_one_reading_and_is_the_wardens():
    P = 256 - HEADER.size
    for x, s in ((0, 0), (7, 3), (151_328, 999), ((1 << 32) - 1, (1 << 32) - 1)):
        cell = IT.data_cell(x, s, 256)
        assert IT.parse_cell(cell) == (DATA, s, x) and W.read_cell(cell, 256) == (DATA, s, IT.id_event(x, P))
        assert IT.parse_cell(IT.pad_cell(256, s)) == (PAD, s, 0) and W.read_cell(IT.pad_cell(256, s), 256) == (PAD, s, b"")
        assert IT.parse_cell(IT.end_cell(s, 256)) == (IT.END, s, 0)
        assert W.read_cell(IT.end_cell(s, 256), 256) == (W.END, s, IT.TRAILER)
    assert IT.END == W.END and IT.pad_cell(256) == bytes(256) and len(IT.TRAILER) < IT.ID_EVENT_BYTES
    assert IT.data_cell(5, 1, IT.CELL_USED) == HEADER.pack(DATA, 1, IT.ID_EVENT_BYTES) + IT.id_event(5)
    good, end = IT.data_cell(12, 3, 256), IT.end_cell(3, 256)
    for bad in (good[:IT.ID_AT] + b"0000000012" + good[IT.ID_AT + 10:], good[:IT.ID_AT] + b" " * 10 + good[IT.ID_AT + 10:],
                good[:IT.ID_AT] + b"4294967296" + good[IT.ID_AT + 10:], good[:IT.ID_AT] + b"      1 2 " + good[IT.ID_AT + 10:],
                good[:-3] + b"!\n\n", good[:-12] + b"%10d" % 7 + b"\n\n", bytes([DATA ^ 1]) + good[1:],
                IT.pad_cell(256, 9)[:-1] + b"\x01", end[:-1] + b"\x01", W.cell(W.FAIL, 3, b"", 256),
                W.cell(W.END, 3, b"", 256), W.cell(W.END, 3, IT.id_event(12, P), 256), W.cell(W.REFUSED, 0, b"", 256),
                good[:-1]):
        assert IT.parse_cell(bad) is None
    with pytest.raises(ValueError, match="u32"):
        IT.data_cell(1 << 32, 0, 256)
    with pytest.raises(ValueError, match="at least"):
        IT.pad_cell(IT.CELL_USED - 1)


def test_the_map_is_the_wardens_slots():
    """Window 0 of T = 3, two lanes of r = 2, D = 1: request 0 (stream 1) and request 1 (stream 2) are admitted at
    bucket 0, so each has two `PAD`s there; request 1's response is empty, so its `END` is its cell 0, at bucket 1,
    and its lane is free at bucket 2, where request 2 (stream 3) is admitted. Request 0's outputs 0 and 1 are at
    bucket 1, its output 2 and its `END` at bucket 2."""
    sh = IT.Shape(256, 3, 4, P=4, S=3, DS=3)
    outputs = [IT.Output(4, 0, 0, 0), IT.Output(5, 0, 0, 1), IT.Output(6, 1, 0, 2), IT.Output(7, 0, 2, 0)]
    release, admitted = [1, 1, 2, 3], [0, 0, 2]
    cmap, steps = IT.window_map(sh, outputs, release, 0, admitted, 1, 2)
    E, P, U = IT.END_ENTRY, IT.EMPTY, IT.UNHELD
    assert cmap == tuple(IT.MapEntry(*m) for m in ((1, P), (1, P), (2, P), (2, P), (1, 0), (1, 4), (2, E), (2, P),
                                                    (1, 1), (1, E), (3, P), (3, P)))
    assert steps == ((0, 0, 0), (4, 5, 5), (6, 6, 6))
    assert IT.map_bytes(cmap[:2]) == IT.MAP.pack(1, P) * 2 and U == (0, P)
    assert IT.window_map(sh, outputs, release, 1, admitted, 1, 2)[0][:4] == ((3, 0), (3, E), U, U)
    assert IT.window_map(sh, [], [], 5, [], 1, 2) == ((U,) * 12, ((0,) * 3,) * 3)
    order = [1, 0, 2]
    assert IT.window_map(sh, outputs, release, 0, admitted, 1, 2, order)[0][:8] == tuple(
        IT.MapEntry(*m) for m in ((1, P), (1, P), (2, P), (2, P), (1, E), (1, P), (2, 0), (2, 4)))
    refused = {
        "permutation": dict(order=[0, 0, 2]),
        "not the warden's": dict(release=[1, 1, 3, 3]),
        "arrival order": dict(admitted=[2, 0, 2]),
        "more than R / r": dict(admitted=[0, 0, 0], release=[1, 1, 2, 1]),
        "more than S": dict(shape=IT.Shape(256, 3, 4, P=4, S=1, DS=3)),
        "below P": dict(shape=IT.Shape(256, 3, 4, P=1, S=3, DS=3)),
        "indices 0": dict(outputs=outputs[1:], release=release[1:]),
        "lanes of r": dict(slots=3),
        "depth": dict(shape=IT.Shape(256, 3, 4, P=4, S=3, DS=2)),
    }
    for match, kw in refused.items():
        args = dict(shape=sh, outputs=outputs, release=release, window=0, admitted=admitted, delay=1, slots=2) | kw
        with pytest.raises(ValueError, match=match):
            IT.window_map(**args)
    with pytest.raises(ValueError, match="once"):
        IT.window_map(sh, outputs + outputs[:1], release + release[:1], 0, admitted, 1, 2)
    row = IT.port_row([7, 1 << 31], 4)
    assert row == (7).to_bytes(4, "big") + (1 << 31).to_bytes(4, "big") + bytes(8)
    assert IT.port_id(sh, [bytes(16), row], 5) == 1 << 31
    with pytest.raises(ValueError, match="at most 4"):
        IT.port_row([1] * 5, 4)
    served = IT.Shape(256, 600, 64, P=256, S=8, DS=23)
    assert (served.N, served.E, served.D, served.W) == (38_400, 2_048, 11, 43)
    assert [len(r) for r in IT.bucket_rows(SMALL, [IT.pad_cell(256)] * SMALL.N)] == [2_048] * 3
    with pytest.raises(ValueError, match="at least 2"):
        IT.Shape(256, 1, 8, P=1, S=1, DS=1)


def test_a_rows_root_is_its_domains_tree_and_a_path_opens_its_row():
    for count in (2, 3, 5, 8):
        dom = IT.domain(_h(b"d"), count)
        rows, salts = [bytes([i]) * 40 for i in range(count)], [IT.fake_salt(b"s", i) for i in range(count)]
        leaves = [row_leaf(dom, i, HC.hiding_value(r, s), IT.SCHEMA) for i, (r, s) in enumerate(zip(rows, salts))]
        root = IT.rows_root(dom.domain_id, rows, salts)
        assert root == tree_levels(dom, leaves)[-1][0]
        levels = IT.rows_levels(dom.domain_id, rows, salts)
        for i in range(count):
            path = IT.rows_path(levels, i)
            assert opens(IT.SCHEMA, root, dom.domain_id, HC.hiding_value(rows[i], salts[i]), i, path) == 1
            assert opens(IT.SCHEMA, root, dom.domain_id, HC.hiding_value(rows[i - 1], salts[i - 1]), i, path) == 0


def test_every_window_of_the_wardens_egress_is_accepted(link, big):
    """Every window of the busy link, each cell the warden's own, is accepted, the idle ones too; so is SMALL's."""
    kinds = set()
    for w in range(6):
        inst = commit(BIG, link, w)
        assert IT.reference_parts(inst) == {"cells_root": True, "port": True, "content": True}, w
        assert all(IT.bucket_reference(inst, t) for t in range(BIG.T))
        kinds |= {W.read_cell(c, BIG.B)[0] for c in inst.cells}
    assert kinds == {PAD, DATA, W.END} and not link.fails
    assert all(m == IT.UNHELD for m in commit(BIG, link, 5).map)
    assert sum(not _marker(m) for m in big.map) >= 30 and len({m.q for m in big.map if not _marker(m)}) >= 4
    assert IT.reference(commit(SMALL, small(), 0)) and IT.reference(commit(SMALL, small(), 1))


def test_each_buckets_row_is_the_wardens_registered_row(link):
    """The units read the warden's registration v2 (`warden_rows`): bucket t's row is row t of the record the grid
    closed (`window_rows`, its wiring read off it), it opens the warden's root at t, and the verifier's port
    (`bucket_port`) has unit t read row t, an idle window's too."""
    p = link.params
    assert len(link.records) == 6
    for w, record in enumerate(link.records):
        cells = WR.Cells.of("responses", Direction.EGRESS, record, p, _h(b"session"))
        assert cells.rows == tuple(IT.bucket_rows(BIG, commit(BIG, link, w).cells))
        assert all(WR.opens(cells.domain, cells.root, t, cells.bcs[t], cells.path(t)) for t in range(BIG.T))
        port = WR.bucket_port(cells.head, p)
        assert (port["reads"], port["leaves"], port["words"]) == (list(range(BIG.T)), BIG.T, BIG.R * BIG.B // 2)
    assert WR.wiring(link.records[5], p).connections == ()


def _set(cells, i, cell):
    return [*cells[:i], cell, *cells[i + 1:]]


def _swap(cells, a, b):
    out = list(cells)
    out[a], out[b] = out[b], out[a]
    return out


def _cell_tampers(inst):
    """Each way a server or a prover can make the window's cells differ from what the warden was certified to
    release, as a function of the cells. None where the window has no cells to do it with."""
    sh, m, B, R = inst.shape, inst.map, inst.shape.B, inst.shape.R
    data = [i for i in range(sh.N) if not _marker(m[i])]
    pads = [i for i in range(sh.N) if m[i].e == IT.EMPTY and m[i].q]
    ends = [i for i in range(sh.N) if m[i].e == IT.END_ENTRY]
    unheld = [i for i in range(sh.N) if m[i] == IT.UNHELD]
    parsed = {i: IT.parse_cell(inst.cells[i]) for i in data}

    def first(gen):
        return next(gen, None)

    same = first((a, b) for a in data for b in data if a < b and m[a].q == m[b].q and parsed[a].id != parsed[b].id)
    other = first((i, m[j].q) for i in data for j in range(i // R * R, i // R * R + R) if m[j].q not in (0, m[i].q))
    dup = first((i, j) for i in data for j in [*pads, *ends, *data]
                if m[j].q == m[i].q and inst.cells[j] != inst.cells[i])
    early = first((i - 1, i) for i in ends if i % R and not _marker(m[i - 1]) and m[i - 1].q == m[i].q)
    later = first((i, i + 1) for i in ends if (i + 1) % R and m[i + 1] == (m[i].q, IT.EMPTY))
    r = 2
    groups = first(t for t in range(sh.T) if m[t * R].q and m[t * R + r].q and m[t * R + r].q != m[t * R].q)

    def swapped(t):
        cells = list(inst.cells)
        cells[t * R:t * R + 2 * r] = cells[t * R + r:t * R + 2 * r] + cells[t * R:t * R + r]
        return cells

    def early_end(i, j):
        return _set(_set(inst.cells, i, IT.end_cell(m[i].q, B)), j, IT.pad_cell(B, m[j].q))

    d = data[-1]
    return {
        "a duplicated cell": None if dup is None else _set(inst.cells, dup[1], inst.cells[dup[0]]),
        "a dropped output": _set(inst.cells, d, IT.pad_cell(B, m[d].q)),
        "a reorder within a response": None if same is None else _swap(inst.cells, *same),
        "a cell on another request's stream": None if other is None else _set(
            inst.cells, other[0], IT.data_cell(parsed[other[0]].id, other[1], B)),
        "a cell on a stream with no request": _set(inst.cells, d, IT.data_cell(parsed[d].id, 77_777, B)),
        "a wrong id": _set(inst.cells, data[1], IT.data_cell((parsed[data[1]].id + 1) % V, m[data[1]].q, B)),
        "a covert byte in a pad": None if not pads else _set(inst.cells, pads[-1], inst.cells[pads[-1]][:-1] + b"\x01"),
        "a covert byte in a data cell's padding": _set(inst.cells, data[-2], inst.cells[data[-2]][:-3] + b"!\n\n"),
        "a pad on another stream": None if not pads else _set(inst.cells, pads[0], IT.pad_cell(B, m[pads[0]].q + 1)),
        "a stream in an unheld slot": None if not unheld else _set(inst.cells, unheld[0], IT.pad_cell(B, m[data[0]].q)),
        "a dropped END": None if not ends else _set(inst.cells, ends[0], IT.pad_cell(B, m[ends[0]].q)),
        "an early END": None if early is None else early_end(*early),
        "a late END": None if later is None else _swap(inst.cells, *later),
        "a FAIL for an output": _set(inst.cells, d, W.cell(W.FAIL, m[d].q, b"", B)),
        "two connections' slots swapped": None if groups is None else swapped(groups),
    }


def _map_tampers(inst):
    """Maps other than the verifier's (it computes the map, so these are the circuit's semantics, not a prover's
    move), each refusing the honest cells."""
    m = list(inst.map)
    data = [i for i in range(len(m)) if not _marker(m[i])]
    pad = next(i for i in range(len(m)) if m[i].e == IT.EMPTY and m[i].q)
    end = next(i for i in range(len(m)) if m[i].e == IT.END_ENTRY)

    def at(i, entry):
        return tuple(_set(m, i, IT.MapEntry(*entry)))

    return {
        "another request": at(data[0], (m[data[0]].q + 1, m[data[0]].e)),
        "another entry": at(data[0], (m[data[0]].q, m[data[1]].e)),
        "an entry past the port's": at(data[0], (m[data[0]].q, inst.shape.E)),
        "a pad where a data cell is": at(data[0], (m[data[0]].q, IT.EMPTY)),
        "an END where a data cell is": at(data[0], (m[data[0]].q, IT.END_ENTRY)),
        "an output where a pad is": at(pad, (m[pad].q, m[data[0]].e)),
        "an END where a pad is": at(pad, (m[pad].q, IT.END_ENTRY)),
        "a pad where the END is": at(end, (m[end].q, IT.EMPTY)),
        "a pad on another stream": at(pad, (m[pad].q + 1, IT.EMPTY)),
        "an unheld slot where a pad is": at(pad, IT.UNHELD),
    }


def _content_ok(inst, cells=None, cmap=None) -> bool:
    cells, cmap = cells or inst.cells, cmap or inst.map
    return all(IT.cells_match(inst.shape, cells[inst.bucket(t)], cmap[inst.bucket(t)], inst.port[t])
               for t in range(inst.shape.T))


def test_each_way_the_cells_differ_from_the_map_and_the_port_is_refused(big):
    for name, cells in _cell_tampers(big).items():
        assert cells is not None and not _content_ok(big, cells), name
        parts = IT.reference_parts(with_cells(big, cells))
        assert parts == {"cells_root": True, "port": True, "content": False}, name
    for name, cmap in _map_tampers(big).items():
        assert not _content_ok(big, cmap=cmap), name
    other = commit(BIG, busy(2), 2)
    assert IT.reference(other) and other.cells != big.cells
    assert IT.reference_parts(with_cells(big, other.cells))["content"] is False


def test_a_connection_the_warden_failed_is_refused():
    """A server whose last event isn't there by its deadline: the warden sends its `FAIL` in that slot (and its reason
    to the `fails` record). The advice that claims the whole response is refused at that cell, and one that claims
    the cells the warden sent maps an `END` where the `FAIL` is."""
    honest, failed = busy(3), busy(3, late=frozenset({4}))
    assert not honest.fails and len(failed.fails) == 1 and W.read_reason(failed.fails[0], BIG.B) == (5, "late")
    windows = [w for w in range(6) if failed.rows[w * BIG.T:(w + 1) * BIG.T] != honest.rows[w * BIG.T:(w + 1) * BIG.T]]
    assert windows
    for w in windows:
        assert IT.reference(commit(BIG, honest, w))
        failed_cells = [c for t in range(BIG.T) for c in failed.rows[w * BIG.T + t]]
        assert IT.reference_parts(with_cells(commit(BIG, honest, w), failed_cells)) == {
            "cells_root": True, "port": True, "content": False}
    k = failed.order.index(4)
    short = [o for o in failed.outputs if o.request != k or o.index < max(p.index for p in failed.outputs if p.request == k)]
    keep = [b for o, b in zip(failed.outputs, failed.release) if o in short]
    assert not all(IT.reference(commit(BIG, failed._replace(outputs=short, release=keep), w)) for w in windows)


def test_a_port_that_isnt_the_commits_is_refused(big):
    """The port's openings are the prover's witness and the port's root is the Commit's: a row, a salt, a sibling or a
    step other than the opening's, or the root of another commitment, each refuse the bucket."""
    t = next(t for t in range(BIG.T) if any(not _marker(m) for m in big.map[big.bucket(t)]))
    row = bytearray(big.port[t][1])
    row[2] ^= 1
    path = list(big.paths[t][1])
    path[1] = _h(b"x")
    other = commit(BIG, busy(1), 0, seed=b"another commit")
    steps = list(big.steps[t])
    steps[0] ^= 1

    def at(field, value):
        return replace(big, **{field: _set(getattr(big, field), t, tuple(value))})

    for name, inst in {"a row": at("port", _set(big.port[t], 1, bytes(row))),
                       "a salt": at("port_salts", _set(big.port_salts[t], 1, IT.fake_salt(b"x", 0))),
                       "a sibling": at("paths", _set(big.paths[t], 1, tuple(path))),
                       "a step": at("steps", steps),
                       "another commitment's root": replace(big, port_root=other.port_root)}.items():
        assert IT.reference_parts(inst)["port"] is False and not IT.bucket_reference(inst, t), name
    with pytest.raises(ValueError, match="instance has"):
        IT.reference(replace(big, steps=big.steps[1:]))


def test_a_request_admitted_ahead_of_an_earlier_arrival_is_refused():
    """Advice v0 numbers requests in order of first row, the warden streams in order of arrival, and the map holds
    every slot's stream to its request's: so a request whose number isn't its arrival order puts its cells on another
    stream than a v0 map names. Advice v1's order permutation names the arrival, so the same cells are accepted
    through it, and only through the permutation that is the warden's."""
    rng = random.Random(21)
    arrivals = [0, 0, 1, 1, 2, 2]
    responses = [[rng.randrange(V) for _ in range(n)] for n in (5, 7, 3, 9, 4, 6)]
    params = wardens(BIG, 2, 2)
    swap = [1, 0, 2, 3, 4, 5]
    straight, served = (serve(BIG, params, arrivals, responses, 24, engine) for engine in (None, swap))
    assert IT.reference(commit(BIG, straight, 0)) and IT.reference(commit(BIG, served, 0))
    assert IT.reference_parts(commit(BIG, served, 0, order=list(range(6))))["content"] is False
    assert IT.reference_parts(commit(BIG, served, 0, order=[0, 1, 3, 2, 4, 5]))["content"] is False


def serve_trace(shape: IT.Shape, trace: dict) -> tuple[Link, list[int]]:
    """A recorded link's egress through the warden's own egress: its admission buckets, delay, lanes and output
    lengths, r = 1. The trace's `order` is each request's rank in the warden's release order (by released bucket,
    then ordinal), so its arrival order, the ordinal's, is its rank by (admitted, order), which the warden's FIFO
    admission keeps. Request i is connection order[i] (`serve`'s engine), its output k released at bucket
    admitted[i] + delay + k; ids are synthetic. Returns the link and the arrival order."""
    admitted, lens, delay = trace["admitted"], trace["output_len"], trace["delay"]
    arrival = sorted(range(len(admitted)), key=lambda i: (admitted[i], trace["order"][i]))
    order = [0] * len(arrival)
    for k, i in enumerate(arrival):
        order[i] = k

    def token(i: int, k: int) -> int:
        return (1_000_003 * i + 7_919 * k) % V

    params = W.WardenParams(cell_bytes=shape.B, slots=1, delay=delay, lanes=trace["lanes"], window=shape.T,
                            egress_cells=1 << 20)
    responses = [[token(i, k) for k in range(lens[i])] for i in arrival]
    end = max(a + delay + n for a, n in zip(admitted, lens)) + 1
    link = serve(shape, params, [admitted[i] for i in arrival], responses, -(-end // shape.T) * shape.T, order)
    assert link.admitted == admitted and link.order == order
    return link, order


def test_the_heavy_traces_honest_egress_is_accepted_through_its_order():
    """TRACE-CORPUS's heavy trace (`fixtures/warden-heavy-order.json`, from
    art:d538087522581c66e103caa5e5929dcc10ed8602aa485d8538cff4d5d3943b16): 1,000 requests, 126,502 outputs over buckets
    301 … 589, whose engine admits in the API server's body-read order, out of the warden's arrival order. Served
    through the warden's own egress, every egress window of T = 16 buckets of 1,024 lanes is accepted through advice
    v1's order permutation, each request on its stream from admission. Without it (v0's identity) the map can't be
    built: the advice's admissions aren't in its numbering's order, which the warden's FIFO would need."""
    trace = json.loads((Path(__file__).parent / "fixtures" / "warden-heavy-order.json").read_text())
    sh = IT.Shape(256, 16, 1_024, P=1_024, S=2, DS=10)
    link, order = serve_trace(sh, trace)
    assert len(link.outputs) == 126_502 and order != list(range(1_000))
    windows = len(link.rows) // sh.T
    accepted = [IT.reference(commit(sh, link, w)) for w in range(windows)]
    assert all(accepted) and windows == 37
    with pytest.raises(ValueError, match="arrival order"):
        commit(sh, link._replace(order=list(range(1_000))), 20)


def test_the_benes_network_routes_every_permutation():
    rng = random.Random(4)
    for n in (2, 4, 8, 16, 128):
        for _ in range(20):
            perm = list(range(n))
            rng.shuffle(perm)
            bits, D = IT.benes(perm), n.bit_length() - 1
            assert len(bits) == (2 * D - 1) << (D - 1)
            out = IT.benes_apply(list(range(n)), bits)
            assert all(out[perm[i]] == i for i in range(n))
    for bad in ([0], [0, 0, 1, 2], [0, 1, 2]):
        with pytest.raises(ValueError, match="permutation"):
            IT.benes(bad)


def _bits(rows):
    return np.asarray(rows, dtype=np.uint8)


def _eval(fn, *args):
    return [int(b) for b in np.asarray(evaluate_bits(fn, *[_bits(a) for a in args])).reshape(-1)]


def _int(bits) -> int:
    return sum(int(b) << j for j, b in enumerate(bits))


def test_the_cell_circuit_is_its_reference():
    good, end = IT.data_cell(4_000_000_000, 70_000, 256), IT.end_cell(5, 256)
    cells = [good, IT.pad_cell(256), IT.pad_cell(256, 9), IT.pad_cell(256, (1 << 32) - 1), end, IT.end_cell(0, 256),
             IT.data_cell(0, 0, 256), IT.data_cell(151_328, 2, 256)]
    for digits in (b"0000000012", b"          ", b"      1 2 ", b"4294967296", b"9999999999", b"         0", b"        -1"):
        cells.append(good[:IT.ID_AT] + digits + good[IT.ID_AT + 10:])
    cells += [good[:-1] + b"\x01", good[:-3] + b"!\n\n", IT.pad_cell(256, 9)[:-1] + b"\x01", end[:-1] + b"\x01",
              end[:12] + b"x" + end[13:], bytes([DATA ^ 1]) + good[1:], bytes([W.FAIL]) + end[1:],
              HEADER.pack(DATA, 1, 247 + 1) + good[HEADER.size:], HEADER.pack(PAD, 3, 1) + bytes(247),
              HEADER.pack(W.END, 3, len(IT.TRAILER) + 1) + end[HEADER.size:], b"\x00" * 9 + good[9:]]
    out = np.asarray(evaluate_bits(bind(IT.IoTieCell, B=256), _bits([HC._bytes_bits(c) for c in cells])))
    out = out.reshape(len(cells), IT.CELL_OUT)
    for c, o in zip(cells, out):
        p = IT.parse_cell(c)
        assert int(o[0]) == (p is not None), c
        if p is not None:
            assert (int(o[1]), int(o[2]), _int(o[3:35]), _int(o[35:67])) == (p.kind == DATA, p.kind == IT.END, p.stream,
                                                                               p.id)
    assert sum(IT.parse_cell(c) is not None for c in cells) == 9


def test_the_route_circuit_is_its_reference():
    rng = random.Random(5)
    W_, D = 9, 4
    xs = [[rng.randrange(2) for _ in range(W_)] for _ in range(1 << D)]
    cases = []
    for _ in range(3):
        perm = list(range(1 << D))
        rng.shuffle(perm)
        cases.append(IT.benes(perm))
    cases.append([rng.randrange(2) for _ in range(len(cases[0]))])
    fn = bind(IT.IoTieRoute, W=W_, D=D)
    out = np.asarray(evaluate_bits(fn, _bits([[b for x in xs for b in x]] * len(cases)), _bits(cases)))
    for bits, o in zip(cases, out.reshape(len(cases), 1 << D, W_)):
        assert o.tolist() == IT.benes_apply(xs, bits)


def _changed(inst, cells, cmap):
    """The buckets where a tamper differs from the honest window."""
    return [t for t in range(inst.shape.T) if tuple(cells[inst.bucket(t)]) != inst.cells[inst.bucket(t)]
            or tuple(cmap[inst.bucket(t)]) != inst.map[inst.bucket(t)]]


def _content(cases):
    fn = bind(IT.IoTieContent, B=SMALL.B, R=SMALL.R, P=SMALL.P, S=SMALL.S)
    return _eval(fn, *[list(col) for col in zip(*[IT.content_arguments(i, t, bits) for i, t, bits in cases])])


def test_the_content_circuit_is_its_reference():
    """`IoTieContent` against `cells_match` on each honest bucket of SMALL's window, and on every cell tamper and map
    tamper at the bucket it changes, each with the prover's routing; and random switches refuse an honest bucket."""
    inst = commit(SMALL, small(), 0)
    cases = [(inst, t) for t in range(SMALL.T)]
    for cells in _cell_tampers(inst).values():
        if cells is not None:
            cases += [(with_cells(inst, cells), t) for t in _changed(inst, cells, inst.map)]
    for cmap in _map_tampers(inst).values():
        cases += [(replace(inst, map=cmap), t) for t in _changed(inst, inst.cells, cmap)]
    want = [int(IT.cells_match(SMALL, i.cells[i.bucket(t)], i.map[i.bucket(t)], i.port[t])) for i, t in cases]
    assert want[:3] == [1, 1, 1] and sum(want) == 3 and len(cases) >= 20
    rng = random.Random(7)
    noise = (inst, 1, [rng.randrange(2) for _ in range(SMALL.switches)])
    assert _content([*((i, t, IT.switches(i, t)) for i, t in cases), noise]) == [*want, 0]


def test_two_ids_swapped_with_the_routing_that_matches_them_are_refused():
    """A two-sided tamper of the content's witness: two data cells' ids swapped, and the switches set to route each
    cell the entry carrying its new id, so every routed id matches its cell. The routed entry's index isn't the one the
    map names, and that refuses it; with the honest switches the ids refuse it."""
    inst = commit(SMALL, small(), 0)
    t = 1
    a, b = [i for i in range(SMALL.R * t, SMALL.R * (t + 1)) if not _marker(inst.map[i])][:2]
    pa, pb = IT.parse_cell(inst.cells[a]), IT.parse_cell(inst.cells[b])
    assert pa.id != pb.id
    swapped = with_cells(inst, _set(_set(inst.cells, a, IT.data_cell(pb.id, pa.stream, SMALL.B)), b,
                                    IT.data_cell(pa.id, pb.stream, SMALL.B)))
    perm = IT.entry_perm(SMALL, inst.map[inst.bucket(t)])
    ea, eb = inst.map[a].e, inst.map[b].e
    perm[ea], perm[eb] = perm[eb], perm[ea]
    assert not IT.bucket_reference(swapped, t)
    assert _content([(inst, t, IT.switches(inst, t)), (swapped, t, IT.benes(perm)),
                     (swapped, t, IT.switches(swapped, t))]) == [1, 0, 0]


def _tz(x: int) -> int:
    return (x & -x).bit_length() - 1


def test_the_parts_ands():
    """Each composite but the hashing ones is its parts: what the cost table in PROTOCOL.md extrapolates from."""
    sh, memo = SMALL, {}
    a = lambda fn: HC.ands(fn, memo)  # noqa: E731
    route, cell = a(bind(IT.IoTieRoute, W=sh.W, D=sh.D)), a(bind(IT.IoTieCell, B=sh.B))
    assert route == sh.W * sh.switches
    for N, D, E in ((sh.R, sh.D, sh.E), (12, 5, 20), (100, 7, 96)):
        assert a(bind(IT.IoTieMatch, N=N, D=D, E=E)) == N * (131 + D - _tz(E)) - 1
    match = a(bind(IT.IoTieMatch, N=sh.R, D=sh.D, E=sh.E))
    assert a(bind(IT.IoTieContent, B=sh.B, R=sh.R, P=sh.P, S=sh.S)) == sh.R * cell + route + match
    assert (cell, a(bind(IT.IoTieCell, B=4_096))) == (CELL_256, CELL_4096)


CELL_256, CELL_4096 = 4_139, 50_224


def _read(D: int) -> int:
    """`FrameV3Read{D}`'s ANDs: its leaf, D nodes and `LmsEqual`."""
    leaf = {1: 110_125, 2: 110_904, 3: 111_683}
    node = {1: 166_475, 2: 167_630, 3: 168_774}
    return leaf[IT.index_bytes(D)] + sum(node[IT.index_bytes(D - k - 1)] for k in range(D)) + 511


def _digest(n_bits: int) -> int:
    return 57_947 * -(-(n_bits // 8 + 17) // 128)


@pytest.mark.slow
def test_the_port_circuit_is_its_reference():
    """`IoTiePort` on a bucket's honest openings and on each way they can differ from the Commit's (a row, a salt, a
    sibling, a step, another root), and its ANDs: S × (the row's digest, its hm96 leaf, `FrameV3Read{DS}`) and the
    join. What the reference's tests can't reach is the wiring of each row's bytes, salt, step and path into its read.
    Building the hm96 and frame-v3 gadgets' Boolean bodies takes most of its time."""
    inst, t = commit(SMALL, small(), 0), 1
    other = commit(SMALL, small(), 0, seed=b"other")
    row = bytearray(inst.port[t][1])
    row[5] ^= 4
    steps = list(inst.steps[t])
    steps[2] ^= 1

    def at(field, value):
        return replace(inst, **{field: _set(getattr(inst, field), t, tuple(value))})

    cases = [inst, at("port", _set(inst.port[t], 1, bytes(row))),
             at("port_salts", _set(inst.port_salts[t], 0, IT.fake_salt(b"x", 0))),
             at("paths", _set(inst.paths[t], 3, (inst.paths[t][3][0], _h(b"x"), inst.paths[t][3][2]))),
             at("steps", steps), replace(inst, port_root=other.port_root)]
    want = [int(IT.bucket_reference(i, t)) for i in cases]
    assert want == [1, 0, 0, 0, 0, 0]
    fn = bind(IT.IoTiePort, P=SMALL.P, S=SMALL.S, DS=SMALL.DS)
    assert _eval(fn, *[list(col) for col in zip(*[IT.port_arguments(i, t) for i in cases])]) == want
    assert HC.ands(fn) == SMALL.S * (_digest(32 * SMALL.P) + 215_597 + _read(SMALL.DS)) + SMALL.S - 1


@pytest.mark.slow
def test_the_bucket_circuit_is_its_reference():
    """`IoTieBucket` on each honest bucket row and on one wrong in its port's root or its content: what the parts'
    tests can't reach is the wiring of the registered row's bits into the cells, and of the opened rows into both the
    port's reads and the content's entries. Building the hm96 and frame-v3 gadgets' Boolean bodies takes most of its
    time."""
    inst = commit(SMALL, small(), 0)
    other = commit(SMALL, small(), 0, seed=b"other")
    wrong = with_cells(inst, _cell_tampers(inst)["a wrong id"])
    t_wrong = _changed(inst, wrong.cells, inst.map)[0]
    cases = [(inst, t) for t in range(SMALL.T)] + [(replace(inst, port_root=other.port_root), 1), (wrong, t_wrong)]
    want = [int(IT.bucket_reference(i, t)) for i, t in cases]
    assert want == [1, 1, 1, 0, 0]
    args = [IT.bucket_arguments(i, t, IT.switches(i, t)) for i, t in cases]
    fn = bind(IT.IoTieBucket, B=SMALL.B, R=SMALL.R, P=SMALL.P, S=SMALL.S, DS=SMALL.DS)
    assert _eval(fn, *[list(col) for col in zip(*args)]) == want
