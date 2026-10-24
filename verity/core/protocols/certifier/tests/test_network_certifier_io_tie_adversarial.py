"""Adversarial cases against the I/O tie's prototype (`io_tie`) from the advice red team (PROTOCOL.md, "Advice for
arbitrary serving: what it can carry"), as the tie that opens every cell answers them:

- No field of a cell but the id is the server's: another frame kind, payload length, a finish reason or a second id
  in the fill is refused by the reference and by `IoTieCell`. The stream parses, because the warden writes it from
  the connection it terminated; a cell on another request's stream is refused by the map, which names every data
  cell's request.
- A variable-length response has its map: the verifier computes each cell's output from the advice's outputs and
  release buckets, so interleaved responses of any lengths are accepted, and a count one off either way is refused.
- The order within a bucket, which advice v0 leaves to the tie: the map puts a bucket's outputs in (request, index)
  order, as the warden releases them, and the tie accepts that order alone, so it says nothing the release buckets
  don't.
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

from test_network_certifier_io_tie import BIG, V, _cell_tampers, commit, honest, serve, with_cells

B = BIG.B


def _cell_circuit(cells):
    out = np.asarray(evaluate_bits(bind(IT.IoTieCell, B=B), np.asarray([HC._bytes_bits(c) for c in cells], dtype=np.uint8)))
    return [tuple(int(x) for x in o[:2]) for o in out.reshape(len(cells), IT.CELL_OUT)]


def test_no_cell_field_but_the_id_is_the_servers():
    x, stream = 151_000, 12
    cell = IT.data_cell(x, stream, B)
    event = IT.id_event(x)
    refused = {
        "another kind": bytes([DATA ^ 1]) + cell[1:],
        "a longer payload": HEADER.pack(DATA, stream, IT.ID_EVENT_BYTES + 1) + cell[HEADER.size:],
        "a finish reason": (HEADER.pack(DATA, stream, len(event) + 2)
                            + event.replace(b'"finish_reason":null', b'"finish_reason":"stop"')).ljust(B, b"\0"),
        "a second id in the fill": (cell[:IT.CELL_USED] + b"%10d" % 7).ljust(B, b"\0"),
    }
    assert IT.parse_cell(cell) == (True, stream, x)
    for name, v in refused.items():
        assert len(v) == B and IT.parse_cell(v) is None, name
    # (ok, data): a changed header or event is no data cell; a second id leaves the data cell and fails its fill
    assert _cell_circuit([cell, *refused.values()]) == [(1, 1), (0, 0), (0, 0), (0, 0), (0, 1)]

    inst = honest(BIG, 11)
    cells = _cell_tampers(inst)["a cell on another request's stream"]
    (i,) = [k for k, (a, b) in enumerate(zip(cells, inst.cells)) if a != b]
    assert IT.parse_cell(cells[i]).stream != IT.parse_cell(inst.cells[i]).stream and _cell_circuit([cells[i]]) == [(1, 1)]
    assert not IT.cells_match(BIG, cells, inst.map, inst.port)


def test_a_variable_length_response_has_its_map():
    lengths = {3: 1, 41: 2, 42: 17, 500: 40, 977: 9}
    rng = random.Random(12)
    responses = {q: (rng.randrange(3), [rng.randrange(V) for _ in range(n)]) for q, n in lengths.items()}
    served = serve(BIG, responses, 12)
    inst = commit(BIG, served)
    assert IT.reference(inst)
    counts = {}
    for c in served.cells:
        p = IT.parse_cell(c)
        if p.data:
            counts[p.stream] = counts.get(p.stream, 0) + 1
    assert counts == lengths
    tampers = _cell_tampers(inst)
    for bad in (tampers["a dropped output"], tampers["a duplicated cell"]):
        assert not IT.cells_match(BIG, bad, inst.map, inst.port)


def test_a_buckets_order_carries_nothing():
    """Advice v0's open item, the order within a bucket: of every order of a bucket's frames, the tie accepts one, so
    the order is a function of which frames the bucket holds, which advice v0's release buckets say and charge."""
    one = IT.Shape(B, 2, 8, P=8, S=8, DS=4)
    inst = commit(one, serve(one, {3: (0, [10, 11]), 8: (4, [12, 13]), 20: (0, [14])}, 14, [5, 0]))
    assert IT.reference(inst) and [IT.parse_cell(c).stream for c in inst.cells[:5]] == [3, 3, 8, 8, 20]
    accepted = 0
    for order in itertools.permutations(range(5)):
        cells = [*(inst.cells[k] for k in order), *inst.cells[5:]]
        accepted += IT.cells_match(one, cells, inst.map, inst.port)
    assert accepted == 1


def test_a_payload_in_m_cells_is_refused_and_the_draws_price_is_the_comparison():
    """Every cell is opened, so a payload in any m cells is refused. The K-draw tie measured the changed share: what
    passed K draws at δ was m < N·ln(1/δ)/K cells, each free to carry log2 V bits; at the served shape's busiest window
    (N = 35,386) and K = 20 a window, δ = 2^−40 let about 49,000 cells over a link (Σ_w ε_w·N_w), 844 kbit at
    V = 151,329."""
    inst = honest(BIG, 13)
    data = [i for i, c in enumerate(inst.cells) if IT.parse_cell(c).data]
    rng = random.Random(8)
    for m in (1, 3, 17):
        cells = list(inst.cells)
        for i in rng.sample(data, m):
            p = IT.parse_cell(cells[i])
            cells[i] = IT.data_cell((p.id + 1 + rng.randrange(1000)) % V, p.stream, B)
        assert not IT.cells_match(BIG, cells, inst.map, inst.port)
        assert not IT.reference(with_cells(inst, cells))
    K, delta, N = 20, 2.0 ** -40, 35_386
    passing = N * math.log(1 / delta) / K
    assert 49_000 < passing < 49_100
    assert 843_000 < passing * math.log2(V) < 845_000
