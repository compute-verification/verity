"""The I/O tie's prototype (`io_tie`) on a fake upstream: requests' variable-length responses interleaved over a
link-window's cells as a warden would schedule them, each output's release bucket its cell's, the window's table from
advice-v0-shaped outputs and the port's ids, both committed under hiding rows. The reference accepts the honest window
and refuses every way the cells can differ from the table; the Boolean parts agree with it."""

import hashlib
import random
from dataclasses import replace

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.commitments.merkle import tree_levels
from verity.ml.commitments.rowleaf import row_leaf
from verity.core.protocols.certifier import head_check as HC, io_tie as IT
from verity.core.protocols.certifier.active import DATA, HEADER

#: the circuits' shape: N = 12 cells, M = 16 slots
SMALL = IT.Shape(256, 4, 3)
#: the reference's shape: N = 120 cells, M = 128 slots
BIG = IT.Shape(256, 6, 20)
WINDOW = 7
V = 151_329


def _h(label: bytes) -> bytes:
    return hashlib.sha512(label).digest()


def serve(shape: IT.Shape, responses: dict[int, tuple[int, list[int]]], seed: int):
    """The warden's cells for request q's outputs j0, j0 + 1, … (`responses[q] = (j0, ids)`) released in window
    `WINDOW`, interleaved at random cells in each request's order, pads elsewhere; and the outputs, release buckets
    and ids advice v0 and the port list, with each request's outputs in the windows before and after."""
    rng = random.Random(seed)
    order = [q for q, (_, xs) in responses.items() for _ in xs]
    rng.shuffle(order)
    at = sorted(rng.sample(range(shape.N), len(order)))
    cells = [IT.pad_cell(shape.B)] * shape.N
    outputs, release, ids, nxt = [], [], [], dict.fromkeys(responses, 0)
    for i, q in zip(at, order):
        j0, xs = responses[q]
        k, nxt[q] = nxt[q], nxt[q] + 1
        cells[i] = IT.data_cell(xs[k], q, shape.B)
        outputs.append((q, j0 + k))
        release.append(WINDOW * shape.T + shape.bucket(i))
        ids.append(xs[k])
    for q, (j0, xs) in responses.items():
        if j0:
            outputs.append((q, j0 - 1))
            release.append(WINDOW * shape.T - 1)
            ids.append(1)
        outputs.append((q, j0 + len(xs)))
        release.append((WINDOW + 1) * shape.T)
        ids.append(2)
    return cells, outputs, release, ids


def commit(shape: IT.Shape, cells, rows, seed: bytes = b"w") -> IT.Instance:
    """Both values committed as the warden (the cells) and the prover (the table) commit them."""
    cd, td = IT.domain(_h(b"cells" + seed), shape.N), IT.domain(_h(b"table" + seed), shape.M // IT.SLOTS_A_ROW)
    cs = [IT.fake_salt(seed + b"c", i) for i in range(len(cells))]
    ts = [IT.fake_salt(seed + b"t", i) for i in range(len(rows))]
    return IT.Instance(shape, WINDOW * shape.T, tuple(cells), tuple(cs), IT.rows_root(cd.domain_id, cells, cs),
                       cd.domain_id, tuple(rows), tuple(ts), IT.rows_root(td.domain_id, rows, ts), td.domain_id)


def responses(seed: int, shape: IT.Shape, count: int = 4) -> dict[int, tuple[int, list[int]]]:
    """`count` requests of variable lengths filling at most two thirds of the window, some begun in earlier windows."""
    rng = random.Random(seed)
    qs = sorted(rng.sample(range(1, 1_000), count))
    cuts = sorted(rng.sample(range(1, 2 * shape.N // 3), count - 1))
    lengths = [b - a for a, b in zip([0, *cuts], [*cuts, 2 * shape.N // 3])]
    return {q: (rng.choice([0, 0, rng.randrange(1, 5_000)]), [rng.randrange(V) for _ in range(n)])
            for q, n in zip(qs, lengths)}


def honest(shape: IT.Shape, seed: int = 1) -> IT.Instance:
    cells, outputs, release, ids = serve(shape, responses(seed, shape), seed)
    return commit(shape, cells, IT.table_rows(IT.window_slots(outputs, release, ids, WINDOW, shape.T), shape.M))


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


def test_the_table_is_the_windows_outputs_in_request_order():
    outputs = [(5, 0), (2, 3), (5, 1), (2, 4), (9, 0)]
    slots = IT.window_slots(outputs, [70, 71, 75, 84, 69], [10, 20, 30, 40, 50], 7, 10)
    assert slots == [IT.Slot(2, 3, 71, 20), IT.Slot(5, 0, 70, 10), IT.Slot(5, 1, 75, 30)]
    rows = IT.table_rows(slots, 16)
    assert len(rows) == 2 and all(len(r) == IT.ROW_BYTES for r in rows)
    assert IT.table_slots(rows)[:3] == slots and set(b"".join(rows)[3 * IT.SLOT.size:]) == {0xFF}
    with pytest.raises(ValueError, match="don't fit"):
        IT.table_rows(slots * 6, 16)


def test_a_rows_root_is_its_domains_tree():
    for count in (2, 3, 5, 8):
        dom = IT.domain(_h(b"d"), count)
        rows, salts = [bytes([i]) * 40 for i in range(count)], [IT.fake_salt(b"s", i) for i in range(count)]
        leaves = [row_leaf(dom, i, HC.hiding_value(r, s), IT.SCHEMA) for i, (r, s) in enumerate(zip(rows, salts))]
        assert IT.rows_root(dom.domain_id, rows, salts) == tree_levels(dom, leaves)[-1][0]


def test_the_honest_window_is_accepted_and_so_are_an_empty_and_a_full_one(big):
    assert IT.reference_parts(big) == {"cells_root": True, "table_root": True, "content": True}
    assert len({IT.parse_cell(c).stream for c in big.cells if IT.parse_cell(c).data}) == 4
    empty = commit(BIG, [IT.pad_cell(BIG.B)] * BIG.N, IT.table_rows([], BIG.M))
    assert IT.reference(empty)
    cells, outputs, release, ids = serve(BIG, {4: (0, list(range(BIG.N)))}, 2)
    assert IT.reference(commit(BIG, cells, IT.table_rows(IT.window_slots(outputs, release, ids, WINDOW, BIG.T), BIG.M)))


def _data(inst):
    return [i for i, c in enumerate(inst.cells) if IT.parse_cell(c).data]


def _set(cells, i, cell):
    return [*cells[:i], cell, *cells[i + 1:]]


def _cell_tampers(inst):
    """Each way a prover can make the window's cells differ from the table, as a function of the cells."""
    d, B, R = _data(inst), inst.shape.B, inst.shape.R
    p = [i for i in range(inst.shape.N) if i not in d]
    parsed = {i: IT.parse_cell(inst.cells[i]) for i in d}
    by_stream: dict[int, list[int]] = {}
    for i in d:
        by_stream.setdefault(parsed[i].stream, []).append(i)
    same = next(ix[:2] for ix in by_stream.values() if len(ix) > 1 and parsed[ix[0]].id != parsed[ix[1]].id)
    other = next(b for b in d if parsed[b].stream != parsed[d[0]].stream)
    swapped = list(inst.cells)
    swapped[same[0]], swapped[same[1]] = swapped[same[1]], swapped[same[0]]
    # a data cell moved R cells later, past no cell of its own stream, so only its bucket changes
    move = next(i for i in d if i + R < inst.shape.N and i + R in p
                and not any(i < k <= i + R for k in by_stream[parsed[i].stream] if k != i))
    moved = _set(_set(inst.cells, move, IT.pad_cell(B)), move + R, inst.cells[move])
    return {
        "a duplicated cell": _set(inst.cells, p[0], inst.cells[d[0]]),
        "a dropped output": _set(inst.cells, d[3], IT.pad_cell(B)),
        "a reorder within a response": swapped,
        "a cell on another request's stream": _set(inst.cells, d[0], IT.data_cell(parsed[d[0]].id, parsed[other].stream, B)),
        "a cell on a stream with no request": _set(inst.cells, d[0], IT.data_cell(parsed[d[0]].id, 77_777, B)),
        "a wrong id": _set(inst.cells, d[1], IT.data_cell((parsed[d[1]].id + 1) % V, parsed[d[1]].stream, B)),
        "a covert byte in a pad": _set(inst.cells, p[1], IT.pad_cell(B)[:-1] + b"\x01"),
        "a covert byte in a data cell's fill": _set(inst.cells, d[2], inst.cells[d[2]][:-1] + b"\x01"),
        "an output moved to the next bucket": moved,
    }


def _table_tampers(inst):
    """Each way the table can be malformed or differ from the cells, as a function of its slots."""
    slots = IT.table_slots(inst.rows)
    n = sum(s.q != IT.EMPTY for s in slots)
    first = next(m for m in range(1, n) if slots[m].q == slots[m - 1].q)
    nxt = next(m for m in range(1, n) if slots[m].q != slots[m - 1].q)

    def rows(ss, pad=b""):
        data = b"".join(IT.SLOT.pack(*s) for s in ss)
        data = (data + pad + b"\xff" * (IT.SLOT.size * inst.shape.M))[:IT.SLOT.size * inst.shape.M]
        return [data[k:k + IT.ROW_BYTES] for k in range(0, len(data), IT.ROW_BYTES)]

    full = slots[:n]
    return {
        "a bucket off by one": rows([*full[:2], full[2]._replace(b=full[2].b + 1), *full[3:]]),
        "a j skipped": rows([*full[:first], *(s._replace(j=s.j + 1) for s in full[first:first + 1]), *full[first + 1:]]),
        "two requests out of order": rows([*full[nxt:], *full[:nxt]]),
        "an empty slot not all 0xFF": rows(full, IT.SLOT.pack(IT.EMPTY, 0, 0, 0)),
        "an empty slot before a full one": rows([*full[:-1], IT.Slot(IT.EMPTY, IT.EMPTY, IT.EMPTY, IT.EMPTY), full[-1]]),
        "an output missing": rows(full[:-1]),
        "an extra output": rows([*full, IT.Slot(full[-1].q, full[-1].j + 1, full[-1].b, 5)]),
    }


def test_each_way_the_cells_differ_from_the_table_is_refused(big):
    for name, cells in _cell_tampers(big).items():
        assert IT.canonical_map(BIG, big.base, cells, big.rows) is None, name
    for name, rows in _table_tampers(big).items():
        assert IT.canonical_map(BIG, big.base, big.cells, rows) is None, name
    assert IT.canonical_map(BIG, big.base + 1, big.cells, big.rows) is None


def test_a_root_that_isnt_its_rows_is_refused(big):
    other = honest(BIG, 3)
    assert IT.reference_parts(replace(big, cells_root=other.cells_root))["cells_root"] is False
    assert IT.reference_parts(replace(big, table_root=other.table_root))["table_root"] is False
    assert not IT.reference(replace(big, cell_salts=(IT.fake_salt(b"x", 0), *big.cell_salts[1:])))
    with pytest.raises(ValueError, match="first bucket"):
        IT.reference(replace(big, base=(1 << 32) - 1))


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


def test_the_content_circuit_is_its_reference():
    inst = honest(SMALL, 6)
    cases = [(inst.cells, inst.rows)]
    cases += [(c, inst.rows) for c in _cell_tampers(inst).values()]
    cases += [(inst.cells, r) for r in _table_tampers(inst).values()]
    insts = [commit(SMALL, c, r) for c, r in cases]
    want = [int(IT.reference_parts(i)["content"]) for i in insts]
    assert want[0] == 1 and sum(want) == 1
    args = [IT.content_arguments(i, IT.switches(i)) for i in insts]
    rng = random.Random(7)
    args.append(IT.content_arguments(inst, [rng.randrange(2) for _ in range(SMALL.switches)]))
    want.append(0)
    fn = bind(IT.IoTieContent, B=SMALL.B, T=SMALL.T, R=SMALL.R)
    assert _eval(fn, *[list(col) for col in zip(*args)]) == want


def test_the_parts_ands():
    """Each composite is its parts: what the cost table in PROTOCOL.md extrapolates from."""
    sh, memo = SMALL, {}
    a = lambda fn: HC.ands(fn, memo)  # noqa: E731
    route, cell = a(bind(IT.IoTieRoute, W=sh.W, D=sh.D)), a(bind(IT.IoTieCell, B=sh.B))
    assert route == sh.W * sh.switches
    match = a(bind(IT.IoTieMatch, D=sh.D, DB=sh.DB))
    assert a(bind(IT.IoTieContent, B=sh.B, T=sh.T, R=sh.R)) == sh.N * cell + route + match + sh.N
    assert 3_600 < cell < 3_800 and 380 * sh.M < match < 400 * sh.M


@pytest.mark.slow
def test_the_window_circuit_is_its_reference():
    """`IoTieWindow` on an honest window and on one wrong in a cell's salt, the table root or the content: what the
    parts' tests can't reach is the wiring of every cell into both its root and the content, and the pads past N in
    the cells' tree (`FrameV3Pad`). Building the hm96 and frame-v3 gadgets' Boolean bodies takes most of its time."""
    inst = honest(SMALL, 6)
    other = honest(SMALL, 8)
    cases = [inst, replace(inst, cell_salts=(IT.fake_salt(b"x", 0), *inst.cell_salts[1:])),
             replace(inst, table_root=other.table_root), commit(SMALL, _cell_tampers(inst)["a wrong id"], inst.rows)]
    want = [int(IT.reference(i)) for i in cases]
    assert want == [1, 0, 0, 0]
    rows = [bytes([i]) * IT.ROW_BYTES for i in range(3)]
    salts = [IT.fake_salt(b"r", i) for i in range(3)]
    dom = IT.domain(_h(b"r"), 3)
    root = IT.rows_root(dom.domain_id, rows, salts)
    d = HC._digest_bits
    rargs = [[b for r in rows for b in HC._bytes_bits(r)], [b for s in salts for b in d(s)], d(root), d(dom.domain_id)]
    assert _eval(bind(IT.NetworkRowsRoot, N_BITS=8 * IT.ROW_BYTES, COUNT=3),
                 *[[x, y] for x, y in zip(rargs, [*rargs[:2], d(other.table_root), rargs[3]])]) == [1, 0]
    args = [IT.window_arguments(i, IT.switches(i)) for i in cases]
    fn = bind(IT.IoTieWindow, B=SMALL.B, T=SMALL.T, R=SMALL.R)
    assert _eval(fn, *[list(col) for col in zip(*args)]) == want
