"""`requirement`: a toy decoder's live and read bits per step, held to `boundary.py`'s exact in(S) and out(S), and the link
to the audit's store and plans (`StoreMatches`, `PlanMatches`) with the transfer bounds of
`Security/Models/Accounting/Guarantees/Requirement.lean` checked on every held set."""

from __future__ import annotations

from itertools import chain, combinations

import numpy as np
import pytest

from verity.ml.circuits import boundary
from verity.ml.circuits import requirement as R
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition
from verity.ml.circuits.ops import fp32
from verity.ml.circuits.parts import Part, n_members
from verity.ml.circuits.program import Program, is_input
from verity.ml.circuits.refs import array_of
from verity.ml.circuits.types import Array, Value

S, D, V = 3, 2, 4
F32 = I32 = Value(32)


def _gather_lazy(get) -> int:
    t = get(0)
    return get(1 + t) if 0 <= t < V else 0


#: an embedding column's gather, (token, column) -> column[token], whose evaluator fetches the token and the one word
GATHER = PrimitiveDefinition(f"TestGatherF32x{V}", 1, (("token", I32), ("column", Array(V, F32))), F32,
                             lambda t, *col: col[t] if 0 <= t < V else 0, register=False)
GATHER.evaluate_lazy = _gather_lazy  # type: ignore[attr-defined]
#: the next token from an attention sum
PICK = PrimitiveDefinition("TestPick", 1, (("a", F32),), I32, lambda a: (a >> 20) % V, register=False)


def decoder() -> tuple[Program, list[int]]:
    """S decode steps over registered weights: step s embeds its token (a gather per hidden coordinate, D of them), projects
    it to a key k_s (the KV cache's entry), sums every key so far (attention reads the whole cache) and picks the next
    token. Returns the Program and its steps' first Calls."""
    starts: list[int] = []

    def body(B, _S, tok0, emb, wk):
        inputs = len(B.nodes)
        tok, ks, outs = tok0, [], []
        for _ in range(S):
            starts.append(len(B.nodes) - inputs)
            h = [B.call(GATHER, tok, emb[d]) for d in range(D)]
            ks.append(B.call(fp32.F32Fma, h[1], wk[1], B.call(fp32.F32Mul, h[0], wk[0])))
            a = ks[0]
            for k in ks[1:]:
                a = B.call(fp32.F32Add, a, k)
            tok = B.call(PICK, a)
            outs.append(tok)
        return array_of(outs)

    sig = lambda _S: ((("tok0", I32), ("emb", Array(D, Array(V, F32))), ("wk", Array(D, F32))), Array(S, I32))
    fn = CompositeDefinition("TestDecoder", 1, (), sig, body, register=False)
    return Program(fn, registered=("emb", "wk")), starts


def _inputs() -> dict[int, int]:
    """tok0 = 1 (gate 0), the embedding's D·V words (gates 1-8) and the key projection's D weights (gates 9-10)."""
    f = lambda x: fp32.bits_of(np.float32(x))
    return {0: 1, **{1 + i: f(0.5 + i) for i in range(D * V)}, 9: f(1.5), 10: f(-0.25)}


@pytest.fixture(scope="module")
def run():
    program, starts = decoder()
    schedule = R.Schedule.by_calls(program, starts)
    return program, starts, schedule, R.requirement(program, schedule, _inputs())


def test_the_decoders_requirement(run):
    """Registered: the embedding's 8 words and the 2 key weights, 320 bits, live at every step. Each step adds its key to
    the live KV and hands its token to the next step; a step reads the 2 embedding words its gathers select (structurally
    all 8), both weights, its token and every key so far."""
    program, _, schedule, req = run
    assert req.to_json() == {"steps": 3, "live_bits": [320, 384, 416], "read_bits": [128, 192, 224], "registered_bits": 320}
    assert sorted(req.live(2) - req.registered) == [14, 19, 21]  # k_0, k_1 and tok_2
    assert sorted(req.reads(1)) == [3, 7, 9, 10, 14, 15]  # emb[0][2], emb[1][2], wk, k_0, tok_1
    assert R.requirement(program, schedule).to_json()["read_bits"] == [320, 384, 416]
    assert (req.held(0, 3), req.read(0, 3)) == (1120, 544)


def _part(program: Program, nodes) -> Part:
    body = program.root.body
    return Part(program.circuit.root_scope, frozenset((k, m) for k in nodes for m in range(n_members(body.nodes[k]))))


def test_live_and_reads_are_the_boundary(run):
    """live(t) = out(S_<t) ∩ in(S_≥t) with the inputs there before t, and the structural reads(t) = in(S_t) less the run
    inputs that arrive at t, by `boundary.out_gates` and `boundary.in_gates` over the steps' root Calls."""
    program, starts, schedule, _ = run
    req = R.requirement(program, schedule)
    calls = [k for k, nd in enumerate(program.root.body.nodes) if not is_input(nd.fn)]
    by_step = [calls[a:b] for a, b in zip(starts, starts[1:] + [len(calls)])]
    inputs = range(program.input_gates)
    for t in range(req.steps):
        later = boundary.in_gates(_part(program, chain(*by_step[t:])))
        out = boundary.out_gates(_part(program, chain(*by_step[:t]))) if t else frozenset()
        assert req.live(t) == (out & later) | {v for v in later if v in inputs and req.born[v] < t}
        assert req.reads(t) == {v for v in boundary.in_gates(_part(program, by_step[t])) if req.born[v] < t}


def _subsets(xs):
    xs = sorted(xs)
    return chain.from_iterable(combinations(xs, r) for r in range(len(xs) + 1))


@pytest.mark.parametrize("B", [64, 128, R.BLOCK_BITS])
def test_the_honest_store_matches_and_the_transfer_bounds_hold(run, B):
    """`pack` gives a store and plans that are W's at every step, and on them RequirementHeld, RequirementDense and
    RequirementMoved hold for every held and moved set of blocks."""
    _, _, _, req = run
    layout, store = R.pack(req, B)
    for t in range(req.steps):
        live, reads, resident = req.live(t), req.reads(t), store.resident(t)
        plan = R.read_blocks(req, layout, t)
        assert R.store_matches(req, layout, store, B, t) and R.plan_matches(req, layout, plan, B, t)
        for H in _subsets(resident):
            assert req.live_bits(t) <= layout.bits_at(req, live, set(H)) + len(resident - set(H)) * B
        dense = min(layout.bits_in(req, live, j) for j in resident)
        assert len(resident) * dense <= req.live_bits(t)
        for M in _subsets(plan):
            assert req.read_bits(t) <= layout.bits_at(req, reads, set(M)) + len(plan - set(M)) * B


def test_the_honest_store_at_64_bit_blocks(run):
    """Five registered blocks; k_0 and tok_1 share step 0's append, k_1 and tok_2 step 1's; every block is freed at step 2,
    its values' last read. At step 2 step 0's append holds 32 live bits of 64 (tok_1 is dead): the store's density
    there is 1/2, so RequirementDense's D is 32."""
    _, _, _, req = run
    layout, store = R.pack(req, 64)
    assert (store.registered, dict(store.written)) == (5, {0: frozenset({5}), 1: frozenset({6}), 2: frozenset()})
    assert [sorted(store.resident(t)) for t in range(3)] == [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5], list(range(7))]
    assert [layout.bits_in(req, req.live(2), j) for j in range(7)] == [64, 64, 64, 64, 64, 32, 64]
    assert [sorted(R.read_blocks(req, layout, t)) for t in range(3)] == [[0, 2, 4], [1, 3, 4, 5], [0, 2, 4, 5, 6]]


def test_a_block_or_a_read_w_does_not_have_fails_the_link(run):
    """A registered block W doesn't keep live, a live block freed early, a planned block W doesn't read at that step and a
    payload that isn't W's values each break the link."""
    _, _, _, req = run
    layout, store = R.pack(req, 64)
    extra = R.Store(store.registered + 1, store.written, store.freed, {**store.payload, 5: 0})
    assert not R.store_matches(req, layout, extra, 64, 0)
    early = R.Store(store.registered, store.written, {**store.freed, 1: frozenset({5})}, store.payload)
    assert R.store_matches(req, layout, early, 64, 1) and not R.store_matches(req, layout, early, 64, 2)
    assert not R.plan_matches(req, layout, R.read_blocks(req, layout, 1) | {6}, 64, 1)
    assert not R.plan_matches(req, layout, R.read_blocks(req, layout, 1) - {5}, 64, 1)
    forged = R.Store(store.registered, store.written, store.freed, {**store.payload, 6: store.payload[6] ^ 1})
    assert not R.store_matches(req, layout, forged, 64, 2)


def test_past_the_executed_prefix_nothing_runs(run):
    """Without step 2's gates (the request ended after step 1), k_1 has only step 1's reader and tok_2 none: neither is
    ever live."""
    program, _, schedule, _ = run
    prefix = R.Schedule(tuple(r for r in schedule.ranges if r[2] < 2))
    req = R.requirement(program, prefix, _inputs())
    assert req.to_json() == {"steps": 2, "live_bits": [320, 384], "read_bits": [128, 192], "registered_bits": 320}
    assert req.last[19] == req.born[19] == 1 and 21 not in req.last
    assert not {19, 21} & (req.live(1) | req.live(2))
