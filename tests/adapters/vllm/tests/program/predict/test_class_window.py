"""verity_vllm.program.predict.class_window: GLM-5.3's Program at the class (43, 4) at TP8 rank 0, cut to its first decoder
layer, and its class window. Two requests of the class run one Program and one window; the first bit and count are
constants inside the first select, every later one a committed output wired from the step before; one boundary read of
`b.request` feeds every select and the check; only the decode steps' key blocks are masked, each by the bit of the step
before; and the window passes its wiring checks with only the other ranks' partials free. On a small V, the selects and
the check, wired as the window wires them, commit zeros past a request's stop and accept only its honest ids and count.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np
import pytest
from verity.circuits.codec import descriptor_id
from verity.circuits.evaluation.batch import evaluate_batch
from verity.commitments.periods import interface as IF

from tests.program.predict.test_node_window import ENGINE, GLM53, TARGET
from verity_vllm.definitions.registry import greedy_stages as G
from verity_vllm.engine.profiles.dense_generic import pinned_hf_config
from verity_vllm.engine.served_seed import DEFAULT_SEED
from verity_vllm.pipeline.boundary_span import Span
from verity_vllm.program.predict import class_window as CW
from verity_vllm.program.predict import node_window as NW
from verity_vllm.program.predict import node_window_dense as ND
from verity_vllm.program.predict import node_window_embed as NE
from verity_vllm.program.predict import node_window_multimem as NM
from verity_vllm.program.predict import node_window_sparse as NS
from verity_vllm.program.predict.phi import Phi
from verity_vllm.program.predict.steps import Registered

FAMILIES = {**NW.FAMILIES, **ND.FAMILIES, **NS.FAMILIES, **NM.FAMILIES, **NE.FAMILIES}
P, O = 43, 4


def registered(layers: int, rank: int = 0) -> Registered:
    """GLM-5.3 at TP8 on H200 cut to its first ``layers`` decoder layers, class G's engine, at rank ``rank``."""
    hf = json.loads(pinned_hf_config(GLM53["repo"], GLM53["revision"]).read_text())
    dense = min(layers, hf["first_k_dense_replace"])
    hf = {**hf, "num_hidden_layers": layers, "first_k_dense_replace": dense, "indexer_types": hf["indexer_types"][:layers],
          "mlp_layer_types": ["dense"] * dense + ["sparse"] * (layers - dense)}
    phi = Phi(model={**GLM53, "hf_config": hf, "dtype": "float8_e4m3fn"}, target=TARGET, execution="eager",
              tp={"world": 8, "rank": rank}, engine=dict(ENGINE))
    return Registered.of(phi, default_seed=DEFAULT_SEED)


def _digest(circuit: IF.Circuit) -> str:
    return hashlib.sha512(IF.circuit_bytes(circuit.description())).hexdigest()


@pytest.fixture(scope="module")
def program():
    return CW.class_request(registered(1), P, O)


@pytest.fixture(scope="module")
def rank0(program):
    return CW.rank_window(0, program.program.fn, FAMILIES)


def _templates(circuit: IF.Circuit, base: str) -> list[str]:
    return [t for t in circuit.units if t.split("{")[0] == base]


def _selects(circuit: IF.Circuit) -> tuple[str, str, str]:
    """The first select's, the later selects' and the check's templates."""
    (first,) = _templates(circuit, "GreedyCombineFirst_v1")
    (live,) = _templates(circuit, "GreedyCombineLive_v1")
    (check,) = _templates(circuit, "GreedyStopCheckClass_v1")
    return first, live, check


def _into(circuit: IF.Circuit, reader: str, port: str) -> list[tuple[int, str, int, str]]:
    """Each reader unit of ``reader``'s ``port`` with its writer: (reader unit, writer, writer unit, writer's output)."""
    out = []
    for w in circuit.wires:
        if (w.reader, w.port) == (reader, port):
            ws = w.source(np.arange(w.reader_units * w.cut[1])).reshape(w.reader_units, w.cut[1])[:, 0]
            out += [(w.at + i, w.writer, int(s), w.out) for i, s in enumerate(ws)]
    return sorted(out)


def test_two_requests_of_a_class_run_one_program_and_one_window(program, rank0):
    """Two requests of P prompt ids each, with different prompts, ``max_tokens``, ``ignore_eos``, stops and finishes, have
    the class (P, O): their prompt's length and the served ids' class (`Span.shape`'s second). The class's Program built
    for each is one digest, as is its window: nothing of either request is in either."""
    a = Span("a", tuple(range(1, P + 1)), {"max_tokens": 3}, (5, 6, 7), "length")
    b = Span("b", tuple(range(40, 40 + P)), {"max_tokens": 9, "ignore_eos": True}, (9, 8, 1, 4), "stop")
    assert a.prompt != b.prompt and (len(a.prompt), a.shape[1]) == (len(b.prompt), b.shape[1]) == (P, O)
    other = CW.class_request(registered(1), len(b.prompt), b.shape[1])
    assert other.digest == program.digest and len(program.digest) == 128
    assert _digest(CW.rank_window(0, other.program.fn, FAMILIES).circuit) == _digest(rank0.circuit)


def test_the_first_bit_and_count_are_constants_not_boundary_rows(rank0):
    """The first select has neither a bit nor a count port, and no boundary read or free port is a bit or a count."""
    c = rank0.circuit
    first, _live, _check = _selects(c)
    assert {CW.COUNT, CW.LIVE}.isdisjoint(p for p, _w in c.layouts[first][0].rows)
    assert not [r for r in c.boundary_reads if r.port in (CW.COUNT, CW.LIVE)]
    assert not [f for f in rank0.free if f.port in (CW.COUNT, CW.LIVE)]


def test_every_later_bit_and_count_is_wired_from_the_step_before(rank0):
    """Decode step s's select reads its bit and count from step s - 1's select, the check from the last one's, each a
    committed output by a step-to-step wire; each decode step's embedding read takes the step before's key."""
    c = rank0.circuit
    first, live, check = _selects(c)
    assert c.units[first] == 1 and c.units[live] == O - 1 and c.units[check] == 1
    before = [(first, 0)] + [(live, s) for s in range(O - 2)]
    for port, out in ((CW.LIVE, CW.NEXT), (CW.COUNT, CW.AHEAD)):
        assert _into(c, live, port) == [(s, wt, wu, out) for s, (wt, wu) in enumerate(before)]
        assert _into(c, check, port) == [(0, live, O - 2, out)]
    keys = [(wt, wu) for w in c.wires if w.port == "tok" and w.out == CW.KEY for _u, wt, wu, _o in _into(c, w.reader, w.port)
            if wt == w.writer]
    assert sorted(set(keys)) == sorted(before)
    assert c.order is not None


def test_the_limits_reach_every_select_and_the_check_by_one_read_of_one_row(rank0):
    """Every select and the check read ``request`` from `b.request`'s first 8 leaves, one boundary read a template,
    ``step [0]``; nothing else reads `b.request` and no ``request`` port is wired or free."""
    c = rank0.circuit
    reads = [r for r in c.boundary_reads if r.param == CW.LIMITS_ROW]
    assert sorted((r.reader, r.port, r.at, r.reader_units, r.base, r.step) for r in reads) == sorted(
        (t, CW.REQUEST, 0, c.units[t], CW.LIMITS[0], (0,)) for t in _selects(c))
    assert not [w for w in c.wires if w.port == CW.REQUEST]
    assert not [f for f in rank0.free if f.port == CW.REQUEST]


def test_only_the_decode_steps_key_blocks_are_masked_each_by_the_step_before(program):
    """The prefill's key blocks are `SparseMlaHeadBlock_v1`, unmasked; each decode step's are the masked block, every
    unit of decode step s reading select s - 1's bit."""
    c, w = CW.class_window(program.program.fn, FAMILIES)
    first, live, _check = _selects(c)
    before = [(first, 0)] + [(live, s) for s in range(O - 2)]
    blocks = [t for t in c.units if t.split("{")[0] == CW.KEY_BLOCK]
    assert blocks and w.masked
    assert {w._step_of[w.node_of[(t, a)]] for t in blocks for a, _n, _f in w.allocs[t]} == {0}
    seen = set()
    for tid, at, n in w.masked:
        step = w._step_of[w.node_of[(tid, at)]]
        got = Counter((wt, wu, o) for u, wt, wu, o in _into(c, tid, CW.LIVE) if at <= u < at + n)
        assert got == Counter({(*before[step - 1], CW.NEXT): n}), (tid, at, step)
        seen.add(step)
    assert seen == set(range(1, O))


def test_the_rank_window_passes_its_checks_with_only_peers_free(rank0):
    """The circuit is well formed (its wires checked at construction, `interface.check_wires`, and ordered), every
    template a proof block, the prompt, the limits and the served ids its boundary reads, and only the other ranks'
    partials free."""
    c = rank0.circuit
    for fn, _units in c.templates:
        IF.check_proof_block(fn, c.cuts.get(descriptor_id(fn)))
    assert {r.param for r in c.boundary_reads} == {"b.prompt", CW.LIMITS_ROW, "o0"}
    assert {f.kind for f in rank0.free} <= {"peer"}, sorted({f"{f.tid} {f.port} {f.kind} {f.family}" for f in rank0.free})
    served = [r for r in c.boundary_reads if r.param == "o0"]
    assert sum(r.reader_units for r in served) == O


def _word16(v: int, n: int) -> np.ndarray:
    return np.array([[(v >> i) & 1 for i in range(16)]] * n, np.uint8)


@pytest.mark.parametrize("stop_at", [1, 3, None])
def test_a_request_that_stops_early_commits_zeros_past_its_stop_and_the_check_accepts_it_alone(stop_at):
    """On rows of a small V, O = 4 steps of made-up logits, the first select and the later ones wired as the window wires
    them (each step's bit and count the step before's): a request stopping at step 1, at step 3, or running to its
    max_tokens of O serves its ids to its stop and zeros past it, its keys zero from its stop on, its count n; the check
    accepts it, and refuses it with any id past the stop made nonzero or the count moved."""
    Vs, mml = 200, 4096
    rng = np.random.default_rng(stop_at or 0)
    rows = rng.integers(0x3F80, 0x4100, (O, Vs)).astype(np.uint16)
    tokens = [G.twin(Vs, r)[0] for r in rows]
    stops = (tokens[stop_at - 1],) if stop_at else (Vs + 1,)
    n = next(i + 1 for i, t in enumerate(tokens) if t in stops) if stop_at else O
    plan = G.plan(Vs)
    N = plan[-1].lanes[1] - plan[-1].lanes[0]
    request = np.array([[P, O, 0, 0]], np.uint32)

    def chunks(row):
        return [np.asarray(evaluate_batch(u.defn, row[None, u.lanes[0]:u.lanes[1]], kernel="reference")).reshape(1, -1)
                for u in plan[:-1]]

    k = O.bit_length()

    def split(out):
        """(o0, o1, o2, o3): the ids as integers, the bit and the count as integers."""
        flat = [int(x) for x in np.asarray(out).reshape(-1)]
        return (flat[0] | flat[1] << 16, flat[2], sum(b << i for i, b in enumerate(flat[3:3 + k])),
                flat[3 + k] | flat[4 + k] << 16)

    outs = []
    state: tuple = ()
    for s in range(O):
        fn = (CW.first_select if s == 0 else CW.live_select)(Vs, N, O, stops, mml)
        got = split(evaluate_batch(fn, rows[None, s, -N:], *chunks(rows[s]), request, *state, kernel="reference"))
        outs.append(got)
        state = (_word16(got[2], 1), _word16(got[1], 1))
    served = [int(o[0]) for o in outs]
    assert served == [*tokens[:n], *[0] * (O - n)]
    assert [int(o[3]) for o in outs] == [*tokens[:n - 1], *[0] * (O - n + 1)]
    assert [o[2] for o in outs] == [min(s + 1, n) for s in range(O)]
    assert [o[1] for o in outs] == [int(s + 1 < n) for s in range(O)]
    check = CW.class_check(Vs, N, P, O, stops, mml)

    def verdict(ids, cnt):
        return int(np.asarray(evaluate_batch(check, rows[None, -1, -N:], *chunks(rows[-1]), np.array([ids], np.uint32), request,
                                             _word16(cnt, 1), _word16(outs[-1][1], 1), kernel="reference")).reshape(-1)[0])

    assert verdict(served, n) == 1
    if n < O:
        assert verdict([*served[:-1], 1], n) == 0
    assert verdict(served, n - 1 if n > 1 else n + 1) == 0
