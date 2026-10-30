"""Adversarial cases against the I/O tie's prototype (`io_tie`; PROTOCOL.md, "Advice for
arbitrary serving: what it can carry"), as the tie that opens every cell answers them:

- No field of a cell but the id is the server's: another frame kind, payload length, a finish reason or a second id
  in the padding is refused by the reference and by `IoTieCell`. The stream parses, because the warden writes it from
  the connection it terminated; a cell on another request's stream is refused by the map, which names every slot's
  stream.
- A variable-length response has its map: the verifier computes each slot's record from the advice's outputs and
  release buckets, so interleaved responses of any lengths are accepted, and a count one off either way is refused.
- The order within a bucket: the warden's slots are its connections' by ordinal, r each in cell order, and the map
  holds every slot to its record, so of every order of a bucket's cells the tie accepts the warden's alone, and the
  order says nothing the admissions and release buckets don't.
- Attack 8, a covert payload in m cells: every cell is opened, so a payload of m = 1, 3 or 17 cells is refused. The
  K-draw tie this replaced passed it with probability (1 − m/N)^K, its price kept as the comparison.
"""

import itertools
import math
import random

import numpy as np

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.core.protocols.certifier import head_check as HC, io_tie as IT
from verity.core.protocols.certifier.active import DATA, HEADER

from test_network_certifier_io_tie import (BIG, V, _cell_tampers, _content_ok, busiest, busy, commit, serve, wardens,
                                           with_cells)

B = BIG.B
P = B - HEADER.size


def _cell_circuit(cells):
    out = np.asarray(evaluate_bits(bind(IT.IoTieCell, B=B), np.asarray([HC._bytes_bits(c) for c in cells], dtype=np.uint8)))
    return [tuple(int(x) for x in o[:2]) for o in out.reshape(len(cells), IT.CELL_OUT)]


def test_no_cell_field_but_the_id_is_the_servers():
    x, stream = 151_000, 12
    cell = IT.data_cell(x, stream, B)
    event = IT.id_event(x, P)
    stop = event.replace(b'"finish_reason":null', b'"finish_reason":"stop"').replace(b"  \n\n", b"\n\n")
    refused = {
        "another kind": bytes([DATA ^ 1]) + cell[1:],
        "a shorter payload": HEADER.pack(DATA, stream, P - 1) + cell[HEADER.size:],
        "a finish reason": HEADER.pack(DATA, stream, P) + stop,
        "a second id in the padding": cell[:-12] + b"%10d" % 7 + b"\n\n",
    }
    assert IT.parse_cell(cell) == (DATA, stream, x)
    for name, v in refused.items():
        assert len(v) == B and IT.parse_cell(v) is None, name
    assert _cell_circuit([cell, *refused.values()]) == [(1, 1), (0, 0), (0, 0), (0, 0), (0, 0)]

    link = busy(11)
    inst = commit(BIG, link, busiest(link))
    cells = _cell_tampers(inst)["a cell on another request's stream"]
    (i,) = [k for k, (a, b) in enumerate(zip(cells, inst.cells)) if a != b]
    assert IT.parse_cell(cells[i]).stream != IT.parse_cell(inst.cells[i]).stream and _cell_circuit([cells[i]]) == [(1, 1)]
    assert not _content_ok(inst, cells)


def _claimed(link, k: int, n: int):
    """The link's advice with request k's response claimed as n outputs, one fewer or one more than it served (the
    extra one at its last output's step and row)."""
    mine = sorted((o for o in link.outputs if o.request == k), key=lambda o: o.index)
    keep = [(o, b) for o, b in zip(link.outputs, link.release) if o.request != k or o.index < n]
    if n > len(mine):
        last = mine[-1]
        p = link.params
        keep.append((IT.Output(last.step, last.row, k, n - 1), link.admitted[k] + p.delay + (n - 1) // p.slots))
    return link._replace(outputs=[o for o, _ in keep], release=[b for _, b in keep])


def test_a_variable_length_response_has_its_map():
    lengths = [1, 2, 17, 40, 9, 0]
    rng = random.Random(12)
    responses = [[rng.randrange(V) for _ in range(n)] for n in lengths]
    link = serve(BIG, wardens(BIG, 2, 2), [0, 0, 1, 2, 2, 3], responses, 5 * BIG.T)
    windows = [commit(BIG, link, w) for w in range(5)]
    assert all(IT.reference(inst) for inst in windows)
    counts = {}
    for c in (c for inst in windows for c in inst.cells):
        p = IT.parse_cell(c)
        if p.data:
            counts[p.stream] = counts.get(p.stream, 0) + 1
    assert counts == {k + 1: n for k, n in enumerate(lengths) if n}
    inst = commit(BIG, link, busiest(link))
    tampers = _cell_tampers(inst)
    for bad in (tampers["a dropped output"], tampers["a duplicated cell"]):
        assert bad is not None and not _content_ok(inst, bad)
    for k, n in ((2, 16), (2, 18), (3, 39), (3, 41)):
        claimed = _claimed(link, k, n)
        assert not all(IT.reference(commit(BIG, claimed, w)) for w in range(5)), (k, n)


def test_a_buckets_order_carries_nothing():
    """Of every order of a bucket's cells, the tie accepts the warden's bytes alone: with r = 1 each slot is its own
    connection's, so a bucket whose slots are all held has distinct cells and exactly one order accepted, and the only
    other orders accepted swap byte-identical cells (the zero cells of unheld slots)."""
    one = IT.Shape(B, 2, 4, P=4, S=4, DS=4)
    link = serve(one, wardens(one, 1, 1), [0, 0, 0, 0], [[10, 11], [12], [], [13, 14, 15]], 3 * one.T)
    full = 0
    for w in range(3):
        inst = commit(one, link, w)
        assert IT.reference(inst)
        for t in range(one.T):
            honest = inst.cells[inst.bucket(t)]
            accepted = [order for order in itertools.permutations(range(one.R)) if IT.cells_match(
                one, [honest[k] for k in order], inst.map[inst.bucket(t)], inst.port[t])]
            assert {tuple(honest[k] for k in order) for order in accepted} == {honest}, (w, t)
            assert len(accepted) == math.prod(math.factorial(honest.count(c)) for c in set(honest)), (w, t)
            full += len(set(honest)) == one.R
    assert full >= 3
    assert {IT.parse_cell(c).kind for c in commit(one, link, 0).cells[one.R:]} == {DATA, IT.END}


def test_a_payload_in_m_cells_is_refused_and_the_samples_price_is_the_comparison():
    """Every cell is opened, so a payload in any m cells is refused. The K-draw tie measured the changed share: what
    passed K draws at δ was m < N·ln(1/δ)/K cells, each free to carry log2 V bits; at the served shape's busiest window
    (N = 35,386) and K = 20 a window, δ = 2^−40 let about 49,000 cells over a link (Σ_w ε_w·N_w), 844 kbit at
    V = 151,329."""
    link = busy(13)
    inst = commit(BIG, link, busiest(link))
    data = [i for i, c in enumerate(inst.cells) if IT.parse_cell(c).data]
    rng = random.Random(8)
    assert len(data) >= 17
    for m in (1, 3, 17):
        cells = list(inst.cells)
        for i in rng.sample(data, m):
            p = IT.parse_cell(cells[i])
            cells[i] = IT.data_cell((p.id + 1 + rng.randrange(1000)) % V, p.stream, B)
        assert not _content_ok(inst, cells)
        assert not IT.reference(with_cells(inst, cells))
    K, delta, N = 20, 2.0 ** -40, 35_386
    passing = N * math.log(1 / delta) / K
    assert 49_000 < passing < 49_100
    assert 843_000 < passing * math.log2(V) < 845_000
