"""The I/O tie's prototype (`io_tie`) on a fake upstream: requests' variable-length responses interleaved over a
link-window's cells as a warden would schedule them, each output's release bucket its cell's; advice-v0-shaped outputs,
from which the verifier computes the window's map; and the Commit's sampled ids, one hiding row a step, which the
window opens at its outputs' steps. The reference accepts the honest window and refuses every way the cells can differ
from the map and the port, and every port that isn't the Commit's; the Boolean parts agree with it."""

import hashlib
import random
from dataclasses import replace
from typing import NamedTuple

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.commitments.gates.frame_v3 import opens
from verity.ml.commitments.merkle import tree_levels
from verity.ml.commitments.rowleaf import row_leaf
from verity.core.protocols.certifier import head_check as HC, io_tie as IT
from verity.core.protocols.certifier.active import DATA, HEADER

#: the circuits' shape: N = 12 cells in three bucket rows (and a pad in their tree), 4 port rows of 4 ids from a
#: session of 8 steps, M = 16 routed positions
SMALL = IT.Shape(256, 3, 4, P=4, S=4, DS=3)
#: the reference's shape: N = 120 cells, 128 port rows of 8 ids from a session of 256 steps
BIG = IT.Shape(256, 6, 20, P=8, S=128, DS=8)
WINDOW = 7
V = 151_329


def _h(label: bytes) -> bytes:
    return hashlib.sha512(label).digest()


class Served(NamedTuple):
    """What a fake upstream served: the warden's cells of window `WINDOW`, advice v0's outputs and release buckets,
    and the Commit's rows of sampled ids, one a step of the session."""

    cells: list[bytes]
    outputs: list[IT.Output]
    release: list[int]
    rows: list[bytes]


def serve(shape: IT.Shape, responses: dict[int, tuple[int, list[int]]], seed: int, occupancy: list[int] | None = None,
          arrival: dict[int, int] | None = None, spread: int = 3) -> Served:
    """Request q's outputs j0, j0 + 1, … (`responses[q] = (j0, ids)`) released in window `WINDOW`: interleaved at
    random in each request's order, cut into buckets of random occupancy (or `occupancy`), each bucket released
    left-packed and sorted by stream, request q's cells on stream `arrival[q]` (its ingress ordinal, q by default).
    Output j of request q is sampled at step 1 + start_q + j − j0 (start_q below `spread`) in batch row q's rank, and
    each request has an output in the windows before (when j0 > 0) and after; the Commit's other ids are random."""
    rng = random.Random(seed)
    order = [q for q, (_, xs) in responses.items() for _ in xs]
    rng.shuffle(order)
    if occupancy is None:
        occupancy = [0] * shape.T
        for _ in order:
            occupancy[rng.choice([t for t in range(shape.T) if occupancy[t] < shape.R])] += 1
    nxt, placed, k = dict.fromkeys(responses, 0), [], 0
    for t, n in enumerate(occupancy):
        bucket = []
        for q in order[k:k + n]:
            bucket.append((q, nxt[q]))
            nxt[q] += 1
        placed += [(t * shape.R + s, q, j) for s, (q, j) in enumerate(sorted(bucket))]
        k += n
    start = {q: rng.randrange(spread) for q in responses}
    row = {q: r for r, q in enumerate(sorted(responses))}
    arrival = arrival or {}

    def output(q: int, j: int) -> IT.Output:
        return IT.Output(1 + start[q] + j - responses[q][0], row[q], q, j)

    cells = [IT.pad_cell(shape.B)] * shape.N
    outputs, release, ids = [], [], {}
    for i, q, k in placed:
        j0, xs = responses[q]
        cells[i] = IT.data_cell(xs[k], arrival.get(q, q), shape.B)
        o = output(q, j0 + k)
        outputs.append(o)
        release.append(WINDOW * shape.T + shape.bucket(i))
        ids[o.step, o.row] = xs[k]
    for q, (j0, xs) in responses.items():
        for j, b in ((j0 - 1, WINDOW * shape.T - 1), (j0 + len(xs), (WINDOW + 1) * shape.T)):
            if j >= 0:
                o = output(q, j)
                outputs.append(o)
                release.append(b)
                ids[o.step, o.row] = rng.randrange(V)
    rows = [IT.port_row([ids.get((s, r), rng.randrange(V)) for r in range(shape.P)], shape.P) for s in range(1 << shape.DS)]
    return Served(cells, outputs, release, rows)


def commit(shape: IT.Shape, served: Served, seed: bytes = b"w") -> IT.Instance:
    """The window's tie: the cells committed as the warden commits them (a row a bucket), the map and steps the
    verifier computes from the advice, and the port opened from the Commit's rows at those steps."""
    cmap, steps = IT.window_map(shape, served.outputs, served.release, WINDOW)
    cd, pd = IT.domain(_h(b"cells" + seed), shape.T), IT.domain(_h(b"port" + seed), len(served.rows))
    cs = [IT.fake_salt(seed + b"c", t) for t in range(shape.T)]
    ps = [IT.fake_salt(seed + b"p", s) for s in range(len(served.rows))]
    levels = IT.rows_levels(pd.domain_id, served.rows, ps)
    port, salts, paths = IT.open_port(levels, served.rows, ps, steps)
    return IT.Instance(shape, tuple(served.cells), tuple(cs), IT.rows_root(cd.domain_id, IT.bucket_rows(shape, served.cells), cs),
                       cd.domain_id, cmap, steps, port, salts, paths, levels[-1][0], pd.domain_id)


def with_cells(inst: IT.Instance, cells) -> IT.Instance:
    """The window with the warden's cells replaced, committed as the warden commits what it saw."""
    return replace(inst, cells=tuple(cells),
                   cells_root=IT.rows_root(inst.cells_domain, IT.bucket_rows(inst.shape, cells), inst.cell_salts))


def responses(seed: int, shape: IT.Shape, count: int = 4, longest: int | None = None) -> dict[int, tuple[int, list[int]]]:
    """`count` requests of variable lengths, each at most `longest`, filling at most two thirds of the window, some
    begun in earlier windows."""
    rng = random.Random(seed)
    qs = sorted(rng.sample(range(1, 1_000), count))
    total = 2 * shape.N // 3
    while True:
        cuts = sorted(rng.sample(range(1, total), count - 1))
        lengths = [b - a for a, b in zip([0, *cuts], [*cuts, total])]
        if longest is None or max(lengths) <= longest:
            break
    return {q: (rng.choice([0, 0, rng.randrange(1, 5_000)]), [rng.randrange(V) for _ in range(n)])
            for q, n in zip(qs, lengths)}


def served_honestly(shape: IT.Shape, seed: int = 1) -> Served:
    if shape.S < 2 * shape.N // 3:
        return serve(shape, responses(seed, shape, longest=shape.S), seed, spread=1)
    return serve(shape, responses(seed, shape), seed)


def honest(shape: IT.Shape, seed: int = 1) -> IT.Instance:
    return commit(shape, served_honestly(shape, seed))


@pytest.fixture(scope="module")
def big():
    return honest(BIG)


def test_a_cell_parses_to_its_one_reading():
    for x, s in ((0, 0), (7, 3), (151_328, 999), ((1 << 32) - 1, (1 << 32) - 1)):
        assert IT.parse_cell(IT.data_cell(x, s, 256)) == (True, s, x)
    assert IT.parse_cell(IT.pad_cell(256)) == (False, 0, 0) and IT.pad_cell(256) == bytes(256)
    good = IT.data_cell(12, 3, 256)
    for bad in (good[:IT.ID_AT] + b"0000000012" + good[IT.ID_AT + 10:], good[:IT.ID_AT] + b" " * 10 + good[IT.ID_AT + 10:],
                good[:IT.ID_AT] + b"4294967296" + good[IT.ID_AT + 10:], good[:IT.ID_AT] + b"      1 2 " + good[IT.ID_AT + 10:],
                good[:-1] + b"\x01", b"\x01" + bytes(255), bytes([DATA ^ 1]) + good[1:]):
        assert IT.parse_cell(bad) is None
    with pytest.raises(ValueError, match="u32"):
        IT.data_cell(1 << 32, 0, 256)
    with pytest.raises(ValueError, match="more than"):
        IT.pad_cell(IT.CELL_USED)


def test_the_map_is_the_windows_outputs_in_the_wardens_order():
    """Window 7 of T = 3 is buckets 21 … 23: bucket 21 releases request 5's output 0 and request 2's output 3, which
    the warden sorts by stream; bucket 22 request 5's output 1; the outputs at buckets 20 and 24 are other windows'."""
    sh = IT.Shape(256, 3, 2, P=4, S=3, DS=3)
    outputs = [IT.Output(4, 1, 5, 0), IT.Output(4, 0, 2, 3), IT.Output(5, 1, 5, 1), IT.Output(5, 0, 2, 4), IT.Output(3, 2, 9, 0)]
    cmap, steps = IT.window_map(sh, outputs, [21, 21, 22, 24, 20], 7)
    assert steps == (4, 5, 5)
    assert cmap == (IT.MapEntry(2, 0), IT.MapEntry(5, 1), IT.MapEntry(5, 5), None, None, None)
    assert IT.map_bytes(cmap) == IT.MAP.pack(2, 0) + IT.MAP.pack(5, 1) + IT.MAP.pack(5, 5) + b"\xff" * 24
    assert IT.window_map(sh, outputs, [0] * 5, 7) == ((None,) * 6, (0, 0, 0))
    with pytest.raises(ValueError, match="more than R"):
        IT.window_map(sh, outputs, [21, 21, 21, 24, 20], 7)
    with pytest.raises(ValueError, match="more than S"):
        IT.window_map(IT.Shape(256, 3, 2, P=4, S=1, DS=3), outputs, [21, 21, 22, 24, 20], 7)
    with pytest.raises(ValueError, match="below P"):
        IT.window_map(IT.Shape(256, 3, 2, P=1, S=3, DS=3), outputs, [21, 21, 22, 24, 20], 7)
    row = IT.port_row([7, 1 << 31], 4)
    assert row == (7).to_bytes(4, "big") + (1 << 31).to_bytes(4, "big") + bytes(8)
    assert IT.port_id(sh, [bytes(16), row], 5) == 1 << 31
    with pytest.raises(ValueError, match="at most 4"):
        IT.port_row([1] * 5, 4)
    served = IT.Shape(256, 600, 1_686, P=256, S=4_096, DS=23)
    assert (served.N, served.E, served.D, served.W) == (1_011_600, 1 << 20, 20, 52)
    assert [len(r) for r in IT.bucket_rows(SMALL, [IT.pad_cell(256)] * SMALL.N)] == [1_024] * 3
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


def test_the_honest_window_is_accepted_and_so_are_an_empty_and_a_full_one(big):
    assert IT.reference_parts(big) == {"cells_root": True, "port": True, "content": True}
    assert len({IT.parse_cell(c).stream for c in big.cells if IT.parse_cell(c).data}) == 4
    empty = serve(BIG, {}, 2)
    assert IT.reference(commit(BIG, empty)) and all(m is None for m in commit(BIG, empty).map)
    assert IT.reference(commit(BIG, serve(BIG, {4: (0, [random.Random(2).randrange(V) for _ in range(BIG.N)])}, 2)))
    assert IT.reference(honest(SMALL, 6))


def _data(inst):
    return [i for i, c in enumerate(inst.cells) if IT.parse_cell(c).data]


def _set(cells, i, cell):
    return [*cells[:i], cell, *cells[i + 1:]]


def _swap(cells, a, b):
    out = list(cells)
    out[a], out[b] = out[b], out[a]
    return out


def _cell_tampers(inst):
    """Each way a server can make the window's cells differ from what it was certified to release, as a function of
    the cells. None where the window has no cells to do it with."""
    sh, d = inst.shape, _data(inst)
    B, R = sh.B, sh.R
    p = [i for i in range(sh.N) if i not in d]
    parsed = {i: IT.parse_cell(inst.cells[i]) for i in d}
    streams = sorted({x.stream for x in parsed.values()})
    occ = [sum(1 for i in d if sh.bucket(i) == t) for t in range(sh.T)]
    last = {t: t * R + occ[t] - 1 for t in range(sh.T) if occ[t]}

    def first(gen):
        return next(gen, None)

    def neighbours(i):
        t = sh.bucket(i)
        lo = parsed[i - 1].stream if i > t * R else -1
        hi = parsed[i + 1].stream if i + 1 in parsed and sh.bucket(i + 1) == t else 1 << 32
        return lo, hi

    same = first((a, b) for a in d for b in d if a < b and parsed[a].stream == parsed[b].stream and parsed[a].id != parsed[b].id)
    other = first((i, s) for i in d for s in streams if s != parsed[i].stream and neighbours(i)[0] <= s <= neighbours(i)[1])
    roomy = first(t for t in last if occ[t] < R)
    mover = first(t for t in last if t + 1 < sh.T and occ[t + 1] < R)
    mixed = first(i for i in d if i + 1 in parsed and sh.bucket(i + 1) == sh.bucket(i) and parsed[i + 1].stream != parsed[i].stream)

    def moved(t):
        """bucket t's last data cell moved into bucket t + 1 at its place by stream, ahead of its own stream's cells"""
        cells, x = list(inst.cells), inst.cells[last[t]]
        cells[last[t]] = IT.pad_cell(B)
        head = (t + 1) * R
        rest = list(inst.cells[head:head + occ[t + 1]])
        at = sum(IT.parse_cell(c).stream < parsed[last[t]].stream for c in rest)
        cells[head:head + occ[t + 1] + 1] = [*rest[:at], x, *rest[at:]]
        return cells

    return {
        "a duplicated cell": None if roomy is None else _set(inst.cells, last[roomy] + 1, inst.cells[last[roomy]]),
        "a dropped output": _set(inst.cells, last[sh.bucket(d[-1])], IT.pad_cell(B)),
        "a reorder within a response": None if same is None else _swap(inst.cells, *same),
        "a cell on another request's stream": None if other is None else _set(
            inst.cells, other[0], IT.data_cell(parsed[other[0]].id, other[1], B)),
        "a cell on a stream with no request": _set(inst.cells, d[-1], IT.data_cell(parsed[d[-1]].id, 77_777, B)),
        "a wrong id": _set(inst.cells, d[1], IT.data_cell((parsed[d[1]].id + 1) % V, parsed[d[1]].stream, B)),
        "a covert byte in a pad": _set(inst.cells, p[1], IT.pad_cell(B)[:-1] + b"\x01"),
        "a covert byte in a data cell's fill": _set(inst.cells, d[2], inst.cells[d[2]][:-1] + b"\x01"),
        "an output moved to the next bucket": None if mover is None else moved(mover),
        "a data cell after a pad in its bucket": None if roomy is None else _swap(inst.cells, last[roomy], last[roomy] + 1),
        "two streams out of order in a bucket": None if mixed is None else _swap(inst.cells, mixed, mixed + 1),
    }


def _map_tampers(inst):
    """Maps other than the verifier's (it computes the map, so these are the circuit's semantics, not a prover's
    move), each refusing the honest cells."""
    d, m = _data(inst), list(inst.map)
    pad = next(i for i in range(inst.shape.N) if m[i] is None)

    def at(i, entry):
        return tuple(_set(m, i, entry))

    return {
        "another request": at(d[0], m[d[0]]._replace(q=m[d[0]].q + 1)),
        "another entry": at(d[0], m[d[0]]._replace(e=m[d[1]].e)),
        "an entry past the port's": at(d[0], m[d[0]]._replace(e=inst.shape.E)),
        "a pad where a data cell is": at(d[0], None),
        "an output where a pad is": at(pad, IT.MapEntry(m[d[0]].q, m[d[0]].e)),
    }


def test_each_way_the_cells_differ_from_the_map_and_the_port_is_refused(big):
    for name, cells in _cell_tampers(big).items():
        assert cells is not None and not IT.cells_match(BIG, cells, big.map, big.port), name
        parts = IT.reference_parts(with_cells(big, cells))
        assert parts == {"cells_root": True, "port": True, "content": False}, name
    for name, cmap in _map_tampers(big).items():
        assert not IT.cells_match(BIG, big.cells, cmap, big.port), name
    other = honest(BIG, 2)
    assert IT.reference(other) and other.cells != big.cells
    assert IT.reference_parts(with_cells(big, other.cells))["content"] is False


def test_a_port_that_isnt_the_commits_is_refused(big):
    """The port's openings are the prover's witness and the port's root is the Commit's: a row, a salt, a sibling or a
    step other than the opening's, or the root of another commitment, each refuse the window."""
    row = bytearray(big.port[3])
    row[2] ^= 1
    path = list(big.paths[3])
    path[1] = _h(b"x")
    other = commit(BIG, served_honestly(BIG, 1), seed=b"another commit")
    steps = list(big.steps)
    steps[0] = steps[0] ^ 1
    for name, inst in {"a row": replace(big, port=_set(big.port, 3, bytes(row))),
                       "a salt": replace(big, port_salts=_set(big.port_salts, 3, IT.fake_salt(b"x", 0))),
                       "a sibling": replace(big, paths=_set(big.paths, 3, tuple(path))),
                       "a step": replace(big, steps=tuple(steps)),
                       "another commitment's root": replace(big, port_root=other.port_root)}.items():
        assert IT.reference_parts(inst)["port"] is False, name
    with pytest.raises(ValueError, match="instance has"):
        IT.reference(replace(big, steps=big.steps[1:]))


def test_a_request_admitted_ahead_of_an_earlier_arrival_is_refused():
    """Advice v0 numbers requests in order of first row, the warden streams in order of arrival, and the map holds
    every data cell's stream to its request: so a request whose first row isn't its arrival order (the server admitted
    a later arrival first) puts its outputs on another stream than the map names, and every request has an output."""
    rs = responses(21, BIG, count=4)
    qs = sorted(rs)
    assert IT.reference(commit(BIG, serve(BIG, rs, 21)))
    assert IT.reference(commit(BIG, serve(BIG, rs, 21, arrival={q: q for q in qs})))
    swapped = {qs[0]: qs[1], qs[1]: qs[0]}
    inst = commit(BIG, serve(BIG, rs, 21, arrival=swapped))
    assert IT.reference_parts(inst) == {"cells_root": True, "port": True, "content": False}


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
    good = IT.data_cell(4_000_000_000, 70_000, 256)
    cells = [good, IT.pad_cell(256), IT.data_cell(0, 0, 256), IT.data_cell(151_328, 2, 256)]
    for digits in (b"0000000012", b"          ", b"      1 2 ", b"4294967296", b"9999999999", b"         0", b"        -1"):
        cells.append(good[:IT.ID_AT] + digits + good[IT.ID_AT + 10:])
    cells += [good[:-1] + b"\x01", IT.pad_cell(256)[:-1] + b"\x01", bytes([DATA ^ 1]) + good[1:],
              HEADER.pack(DATA, 1, IT.ID_EVENT_BYTES + 1) + good[HEADER.size:], b"\x00" * 9 + good[9:]]
    out = np.asarray(evaluate_bits(bind(IT.IoTieCell, B=256), _bits([HC._bytes_bits(c) for c in cells])))
    out = out.reshape(len(cells), IT.CELL_OUT)
    for c, o in zip(cells, out):
        p = IT.parse_cell(c)
        assert int(o[0]) == (p is not None), c
        if p is not None:
            assert (bool(o[1]), _int(o[2:34]), _int(o[34:66])) == tuple(p)


def test_the_route_circuit_is_its_reference():
    rng = random.Random(5)
    W, D = 9, 4
    xs = [[rng.randrange(2) for _ in range(W)] for _ in range(1 << D)]
    cases = []
    for _ in range(3):
        perm = list(range(1 << D))
        rng.shuffle(perm)
        cases.append(IT.benes(perm))
    cases.append([rng.randrange(2) for _ in range(len(cases[0]))])
    fn = bind(IT.IoTieRoute, W=W, D=D)
    out = np.asarray(evaluate_bits(fn, _bits([[b for x in xs for b in x]] * len(cases)), _bits(cases)))
    for bits, o in zip(cases, out.reshape(len(cases), 1 << D, W)):
        assert o.tolist() == IT.benes_apply(xs, bits)


def _content(cases):
    fn = bind(IT.IoTieContent, B=SMALL.B, T=SMALL.T, R=SMALL.R, P=SMALL.P, S=SMALL.S)
    return _eval(fn, *[list(col) for col in zip(*[IT.content_arguments(i, bits) for i, bits in cases])])


def test_the_content_circuit_is_its_reference():
    """`IoTieContent` against `cells_match` on the honest window, every cell tamper and every map tamper, each with the
    prover's routing; and random switches refuse the honest window."""
    inst = honest(SMALL, 6)
    cases = [inst] + [with_cells(inst, c) for c in _cell_tampers(inst).values() if c is not None]
    cases += [replace(inst, map=m) for m in _map_tampers(inst).values()]
    want = [int(IT.cells_match(SMALL, i.cells, i.map, i.port)) for i in cases]
    assert want[0] == 1 and sum(want) == 1 and len(cases) >= 15
    rng = random.Random(7)
    noise = (inst, [rng.randrange(2) for _ in range(SMALL.switches)])
    assert _content([*((i, IT.switches(i)) for i in cases), noise]) == [*want, 0]


def test_two_ids_swapped_with_the_routing_that_matches_them_are_refused():
    """A two-sided tamper of the content's witness: two data cells' ids swapped, and the switches set to route each
    cell the entry carrying its new id, so every routed id matches its cell. The routed entry's index isn't the one the
    map names, and that refuses it; with the honest switches the ids refuse it."""
    inst = honest(SMALL, 6)
    a, b = _data(inst)[:2]
    pa, pb = IT.parse_cell(inst.cells[a]), IT.parse_cell(inst.cells[b])
    assert pa.id != pb.id
    swapped = with_cells(inst, _set(_set(inst.cells, a, IT.data_cell(pb.id, pa.stream, SMALL.B)), b,
                                    IT.data_cell(pa.id, pb.stream, SMALL.B)))
    perm = IT.entry_perm(SMALL, inst.map)
    ea, eb = inst.map[a].e, inst.map[b].e
    perm[ea], perm[eb] = perm[eb], perm[ea]
    assert not IT.cells_match(SMALL, swapped.cells, swapped.map, swapped.port)
    assert _content([(inst, IT.switches(inst)), (swapped, IT.benes(perm)), (swapped, IT.switches(swapped))]) == [1, 0, 0]


def _tz(x: int) -> int:
    return (x & -x).bit_length() - 1


def test_the_parts_ands():
    """Each composite but the hashing ones is its parts: what the cost table in PROTOCOL.md extrapolates from."""
    sh, memo = SMALL, {}
    a = lambda fn: HC.ands(fn, memo)  # noqa: E731
    route, cell = a(bind(IT.IoTieRoute, W=sh.W, D=sh.D)), a(bind(IT.IoTieCell, B=sh.B))
    assert route == sh.W * sh.switches
    for N, D, E in ((sh.N, sh.D, sh.E), (12, 5, 20), (100, 7, 96)):
        assert a(bind(IT.IoTieMatch, N=N, D=D, E=E)) == N * (130 + D - _tz(E)) - 1
    match = a(bind(IT.IoTieMatch, N=sh.N, D=sh.D, E=sh.E))
    assert a(bind(IT.IoTieContent, B=sh.B, T=sh.T, R=sh.R, P=sh.P, S=sh.S)) == sh.N * cell + route + match
    assert cell == 3_678


def _read(D: int) -> int:
    """`FrameV3Read{D}`'s ANDs: its leaf, D nodes and `LmsEqual`."""
    leaf = {1: 110_125, 2: 110_904, 3: 111_683}
    node = {1: 166_475, 2: 167_630, 3: 168_774}
    return leaf[IT.index_bytes(D)] + sum(node[IT.index_bytes(D - k - 1)] for k in range(D)) + 511


def _digest(n_bits: int) -> int:
    return 57_947 * -(-(n_bits // 8 + 17) // 128)


@pytest.mark.slow
def test_the_port_circuit_is_its_reference():
    """`IoTiePort` on the honest window's openings and on each way they can differ from the Commit's (a row, a salt, a
    sibling, a step, another root), and its ANDs: S × (the row's digest, its hm96 leaf, `FrameV3Read{DS}`) and the
    join. What the reference's tests can't reach is the wiring of each row's bytes, salt, step and path into its read.
    Building the hm96 and frame-v3 gadgets' Boolean bodies takes most of its time."""
    inst = honest(SMALL, 6)
    other = honest(SMALL, 8)
    row = bytearray(inst.port[1])
    row[5] ^= 4
    steps = list(inst.steps)
    steps[2] ^= 1
    cases = [inst, replace(inst, port=_set(inst.port, 1, bytes(row))),
             replace(inst, port_salts=_set(inst.port_salts, 0, IT.fake_salt(b"x", 0))),
             replace(inst, paths=_set(inst.paths, 3, (inst.paths[3][0], _h(b"x"), inst.paths[3][2]))),
             replace(inst, steps=tuple(steps)), replace(inst, port_root=other.port_root)]
    want = [int(IT.reference_parts(i)["port"]) for i in cases]
    assert want == [1, 0, 0, 0, 0, 0]
    fn = bind(IT.IoTiePort, P=SMALL.P, S=SMALL.S, DS=SMALL.DS)
    assert _eval(fn, *[list(col) for col in zip(*[IT.port_arguments(i) for i in cases])]) == want
    assert HC.ands(fn) == SMALL.S * (_digest(32 * SMALL.P) + 215_597 + _read(SMALL.DS)) + SMALL.S - 1


@pytest.mark.slow
def test_the_window_circuit_is_its_reference():
    """`IoTieWindow` on an honest window and on one wrong in a cell's salt, the port's root or the content: what the
    parts' tests can't reach is the wiring of every cell into both its bucket's row and the content, of the opened
    rows into both the port's reads and the content's entries, and the pad past the T rows in the cells' tree
    (`FrameV3Pad`). Building the hm96 and frame-v3 gadgets' Boolean bodies takes most of its time."""
    inst = honest(SMALL, 6)
    other = honest(SMALL, 8)
    cases = [inst, replace(inst, cell_salts=(IT.fake_salt(b"x", 0), *inst.cell_salts[1:])),
             replace(inst, port_root=other.port_root), with_cells(inst, _cell_tampers(inst)["a wrong id"])]
    want = [int(IT.reference(i)) for i in cases]
    assert want == [1, 0, 0, 0]
    rows = [bytes([i]) * 128 for i in range(3)]
    salts = [IT.fake_salt(b"r", i) for i in range(3)]
    dom = IT.domain(_h(b"r"), 3)
    root = IT.rows_root(dom.domain_id, rows, salts)
    d = HC._digest_bits
    rargs = [[b for r in rows for b in HC._bytes_bits(r)], [b for s in salts for b in d(s)], d(root), d(dom.domain_id)]
    assert _eval(bind(IT.NetworkRowsRoot, N_BITS=1024, COUNT=3),
                 *[[x, y] for x, y in zip(rargs, [*rargs[:2], d(other.port_root), rargs[3]])]) == [1, 0]
    args = [IT.window_arguments(i, IT.switches(i)) for i in cases]
    fn = bind(IT.IoTieWindow, B=SMALL.B, T=SMALL.T, R=SMALL.R, P=SMALL.P, S=SMALL.S, DS=SMALL.DS)
    assert _eval(fn, *[list(col) for col in zip(*args)]) == want
