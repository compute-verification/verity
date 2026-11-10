"""Each connection's egress cell count held to its request Program's output count, for any prompt length and any
ending, on the network certifier's path: the scan's egress unit (`egress.EgressBucket`) renders each bucket's row from
the slot state, whose n is the request Program's (`SlotEntry_v1`'s entry, the scan's Programs' row), and its proof opens
only on the network certifier's own row. At the scan class w63k4 (the window circuit's 8,32,4,4,16), five
network-certifier-form id requests of mixed prompt lengths (1 to KI = 32 ids) and `max_tokens`, on four lanes (the fifth
queued until a lane frees), are answered with what `SlotEntry_v1` at its circuit-check row `SlotEntry_v1:4` keeps of the
model's samples: the outputs up to `max_tokens` and the first stop id (128001), that one included. Three end on the stop
id and two on `max_tokens`, one of them with a stop id past it. Served through the network certifier's own ingress and
egress (`scan_traffic.serve`), each connection's egress is n + 1 cells (n id events, then the `END` with the trailer);
the scan of the ingress rows accepts every bucket, its Programs' row at each request's anchor `SlotEntry_v1`'s words bit
for bit, and every egress bucket's row, by the reference and by the gates, is the network certifier's. A server that
sends one output past the stop id, or one short, or nothing (the network certifier fails it late) leaves the ingress rows
and so the scan's state as they were, and the first egress bucket whose row isn't the network certifier's is the bucket
of the first cell at which the lanes differ, every differing cell of it on that connection, by the reference and by the
gates: that bucket's proof doesn't open."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scan_traffic as ST

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate_batch
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.definitions.registry import io_shell as X
from verity.ml.protocols.certifier import egress as EG
from verity.ml.protocols.certifier import ingress_scan as IS
from verity.core.protocols.certifier.active import HEADER, id_request
from verity.core.protocols.certifier.warden import DATA, END, FAIL, WardenParams

B = 512
#: w63k4 and the network certifier's schedule at it
CLS = IS.Class(B=B, r=4, lanes=4, C=47, IDB=17, EVERY=16, WITHIN=12, D=27, KO=4, KI=32)
PARAMS = WardenParams(cell_bytes=B, slots=4, bucket_ms=100, delay=27, lanes=4, queue=4, window=32, ingress_cells=47,
                      ingress_buckets=12, egress_cells=5, every=16)
SHAPE = CLS.egress
EGRESS = bind(EG.EgressBucket, B=B, R=SHAPE.R, r=SHAPE.r, NR=SHAPE.NR, K=SHAPE.K, IB=SHAPE.IB, D=SHAPE.D)
EOS = 128001
#: each request's prompt length, its max_tokens and the model's KO samples (the window circuit's last position's)
REQUESTS = ((1, 4, (EOS, 7, 8, 9)), (32, 4, (500, 131071, EOS, 3)), (5, 2, (40, 41, EOS, 43)), (17, 4, (1, 2, 3, 4)),
            (9, 3, (0, EOS, 77, 78)))
#: what vLLM serves of each: up to max_tokens and the first stop id, that one included
SERVED = ((EOS,), (500, 131071, EOS), (40, 41), (1, 2, 3, 4), (0, EOS))
ARRIVALS = (0, 2, 2, 12, 15)
WINDOWS = 2


def _slot_entry():
    """`SlotEntry_v1` at its circuit-check row `SlotEntry_v1:4` (`io_shell.circuit-check.py`)."""
    spec = importlib.util.spec_from_file_location("io_shell_sidecar",
                                                  Path(X.__file__).with_name("io_shell.circuit-check.py"))
    sidecar = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sidecar)
    return dict(sidecar._model_roots())["SlotEntry_v1:4"]()


def _bits(words) -> list[int]:
    return [w >> i & 1 for w in words for i in range(16)]


def _kept(entry: list[int]) -> tuple[int, ...]:
    """An entry's kept ids (`ingress_scan.entry_layout`: read, n, then the ids)."""
    n = sum(entry[1 + i] << i for i in range(32))
    return tuple(sum(entry[33 + CLS.IDB * k + i] << i for i in range(CLS.IDB)) for k in range(n))


def _serve(outputs, late=()):
    reqs = [id_request(model="unsloth/Llama-3.2-1B", user=f"user-{q}", prompt=[(7 * k + 11 * q) % 128000
                                                                              for k in range(lp)],
                       max_tokens=m, seed=q, frame_bytes=B, offset=0) for q, (lp, m, _) in enumerate(REQUESTS, 1)]
    return ST.serve(PARAMS, reqs, ARRIVALS, [1 << 20] * len(reqs), outputs, WINDOWS, late)


def _cells(row: bytes) -> list[tuple[int, int]]:
    """A row's cells' (kind, ordinal)."""
    return [HEADER.unpack_from(row, i * B)[:2] for i in range(SHAPE.R)]


@pytest.fixture(scope="module")
def run():
    entry = _slot_entry()
    S = entry.bindings
    states = [[(lp - S["NP"] - S["KO"]) & X.M32, m, *xs] for lp, m, xs in REQUESTS]
    out = np.asarray(evaluate_batch(entry, np.ones((len(states), 1), np.uint32), np.asarray(states, np.uint32),
                                    kernel="reference")).reshape(len(states), -1).astype(np.uint64)
    words = (out.reshape(len(states), -1, 16) << np.arange(16, dtype=np.uint64)).sum(axis=2).tolist()
    entries = [_bits(w) for w in words]
    outputs = {q: _kept(e) for q, e in enumerate(entries, 1)}
    CLS.serves(PARAMS)
    served = _serve([outputs[q] for q in sorted(outputs)])
    return SimpleNamespace(bindings=S, states=states, words=words, entries=entries, outputs=outputs, served=served,
                           chain=ST.chain(CLS, *served.rows(), outputs))


def test_slot_entry_keeps_each_requests_outputs_to_its_max_tokens_and_first_stop_id(run):
    S = run.bindings
    assert (S["KO"], S["IDB"], S["NP"], S["STOP"]) == (CLS.KO, CLS.IDB, CLS.KI, (EOS,))
    assert run.words == [X.entry_reference(1, s, S["KO"], S["IDB"], S["NP"], S["STOP"]) for s in run.states]
    assert tuple(run.outputs.values()) == SERVED
    assert [EOS in xs for xs in SERVED] == [True, True, False, False, True]
    assert [len(xs) for xs in SERVED] == [1, 3, 2, 4, 2] and len({lp for lp, _, _ in REQUESTS}) == len(REQUESTS)


def test_each_connections_egress_is_its_programs_n_plus_one_cells_and_every_bucket_renders_it(run):
    ch, s = run.chain, run.served
    assert s.admitted == (0, 2, 2, 12, 28) and not s.fails
    cells = [c for row in ch.egress for c in _cells(row)]
    for q, xs in run.outputs.items():
        assert (cells.count((DATA, q)), cells.count((END, q)), cells.count((FAIL, q))) == (len(xs), 1, 0), q
    E, read = IS.entry_bits(CLS), []
    for t, prog in enumerate(ch.progs):
        for j in range(CLS.NR):
            if prog[j * E]:
                q = ch.scan[t].slots[j].q
                read.append(q)
                assert prog[j * E:(j + 1) * E] == run.entries[q - 1], (t, j)
                assert ch.scan[t].slots[j].fields == REQUESTS[q - 1][0]
    assert sorted(read) == list(run.outputs)
    for t in range(len(ch.egress)):
        assert EG.outputs(SHAPE, ch.view(t)) == (EG.ACCEPT, ch.egress[t]), t
    views = [ch.view(t) for t in range(len(ch.egress))]
    got = np.asarray(evaluate_bits(EGRESS, *[np.asarray(list(col), dtype=np.uint8)
                                             for col in zip(*(EG.bucket_arguments(SHAPE, v) for v in views))]))
    assert got.reshape(len(views), -1).tolist() == [EG.output_bits(SHAPE, v) for v in views]


@pytest.mark.parametrize("q, sent, theirs, ours", [
    (2, +1, DATA, END),    # one output past the stop id
    (4, -1, END, DATA),    # one output short of max_tokens
    (3, None, FAIL, DATA),  # nothing: the network certifier fails it late at its first egress cell
], ids=["past-the-stop", "short", "late"])
def test_a_lane_off_its_programs_n_is_refused_at_the_bucket_of_its_first_differing_cell(run, q, sent, theirs, ours):
    n = len(run.outputs[q])
    outputs = [run.outputs[k] for k in sorted(run.outputs)]
    if sent is None:
        got, first = _serve(outputs, late=(q - 1,)), 0
        assert len(got.fails) == 1
    else:
        outputs[q - 1] = REQUESTS[q - 1][2][:n + sent]
        got, first = _serve(outputs), min(n, n + sent)
    ingress, egress = got.rows()
    ch = ST.chain(CLS, ingress, egress, run.outputs)
    assert ingress == run.chain.ingress and ch.states == run.chain.states and got.admitted == run.served.admitted
    refused = [t for t in range(len(egress)) if EG.outputs(SHAPE, ch.view(t)) != (EG.ACCEPT, egress[t])]
    t = got.admitted[q - 1] + CLS.D + first // CLS.r
    assert refused and refused[0] == t
    rendered = EG.outputs(SHAPE, ch.view(t))[1]
    differ = [i for i in range(SHAPE.R) if rendered[i * B:(i + 1) * B] != egress[t][i * B:(i + 1) * B]]
    assert differ and {_cells(egress[t])[i][1] for i in differ} == {_cells(rendered)[i][1] for i in differ} == {q}
    i = differ[0]
    assert i % CLS.r == first % CLS.r and (_cells(egress[t])[i][0], _cells(rendered)[i][0]) == (theirs, ours)
    gates = evaluate_bits(EGRESS, *[np.asarray([a], dtype=np.uint8) for a in EG.bucket_arguments(SHAPE, ch.view(t))])
    assert np.asarray(gates).reshape(-1).tolist() == EG.output_bits(SHAPE, ch.view(t))
