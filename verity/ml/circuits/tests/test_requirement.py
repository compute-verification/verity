"""`requirement`: a toy decoder's live and read bits per step, held to `boundary.py`'s exact in(S) and out(S), and the link
to the audit's store and plans (`StoreMatches`, `PlanMatches`) with the transfer bounds of
`Security/Definitions/Accounting/Guarantees/Requirement.lean` checked on every held set."""

from __future__ import annotations

from itertools import chain, combinations

import numpy as np
import pytest

from test_census import programs, request
from verity.ml.circuits import boundary
from verity.ml.circuits import requirement as R
from verity.ml.circuits.census import census, step_ends
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

    def sig(_S):
        return ((("tok0", I32), ("emb", Array(D, Array(V, F32))), ("wk", Array(D, F32))), Array(S, I32))

    fn = CompositeDefinition("TestDecoder", 1, (), sig, body, register=False)
    return Program(fn, registered=("emb", "wk")), starts


def _inputs() -> dict[int, int]:
    """tok0 = 1 (gate 0), the embedding's D·V words (gates 1-8) and the key projection's D weights (gates 9-10)."""
    def f(x):
        return fp32.bits_of(np.float32(x))

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
    assert R.upper_bound(program, schedule).to_json()["read_bits"] == [320, 384, 416]
    assert (req.held(0, 3), req.read(0, 3)) == (1120, 544)


def _part(program: Program, nodes) -> Part:
    body = program.root.body
    return Part(program.circuit.root_scope, frozenset((k, m) for k in nodes for m in range(n_members(body.nodes[k]))))


def test_live_and_reads_are_the_boundary(run):
    """live(t) = out(S_<t) ∩ in(S_≥t) with the inputs there before t, and the upper bound's reads(t) = in(S_t) less the run
    inputs that arrive at t, by `boundary.out_gates` and `boundary.in_gates` over the steps' root Calls."""
    program, starts, schedule, _ = run
    req = R.upper_bound(program, schedule)
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
    """`pack` gives a store and plans that are W's at every step, crediting every live value, and on them
    RequirementHeld, RequirementHeldLive, RequirementDense and RequirementMoved hold for every held and moved set of
    blocks."""
    _, _, _, req = run
    layout, store = R.pack(req, B)
    for t in range(req.steps):
        live, reads, resident = req.live(t), req.reads(t), store.resident(t)
        plan = R.read_blocks(req, layout, t)
        assert R.credited(req, layout, t) == live and R.credited_reads(req, layout, t) == reads
        assert R.store_matches(req, layout, store, B, t) and R.plan_matches(req, layout, plan, B, t)
        for H in _subsets(resident):
            assert req.live_bits(t) <= layout.bits_at(req, live, set(H)) + len(resident - set(H)) * B
            missing = sum(layout.bits_in(req, live, j) for j in resident - set(H))
            assert req.live_bits(t) == layout.bits_at(req, live, set(H)) + missing
        dense = min(layout.bits_in(req, live, j) for j in resident)
        assert len(resident) * dense <= req.live_bits(t)
        for M in _subsets(plan):
            assert req.read_bits(t) <= layout.bits_at(req, reads, set(M)) + len(plan - set(M)) * B


def test_the_honest_store_at_64_bit_blocks(run):
    """Five registered blocks; k_0 and tok_1 share step 0's append, k_1 and tok_2 step 1's; every block is freed at step 2,
    its values' last structural reader. At step 2 step 0's append holds 32 live bits of 64 (tok_1 is dead): the store's density
    there is 1/2, so RequirementDense's D is 32."""
    _, _, _, req = run
    layout, store = R.pack(req, 64)
    assert (store.registered, dict(store.written)) == (5, {0: frozenset({5}), 1: frozenset({6}), 2: frozenset()})
    assert [sorted(store.resident(t)) for t in range(3)] == [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5], list(range(7))]
    assert [layout.bits_in(req, req.live(2), j) for j in range(7)] == [64, 64, 64, 64, 64, 32, 64]
    assert [sorted(R.read_blocks(req, layout, t)) for t in range(3)] == [[0, 2, 4], [1, 3, 4, 5], [0, 2, 4, 5, 6]]


def test_a_credited_block_missing_or_forged_fails_the_link(run):
    """A live block freed early, a block W reads left out of the plan and a payload that isn't W's values each break the
    link. Filler keeps it: a registered block W doesn't keep live (index 5, before step 0's append writes it there) and a
    planned block W doesn't read at that step hold none of W's credited values."""
    _, _, _, req = run
    layout, store = R.pack(req, 64)
    extra = R.Store(store.registered + 1, store.written, store.freed, store.payload)
    assert all(R.store_matches(req, layout, extra, 64, t) for t in range(req.steps))
    assert 5 in extra.resident(0) and layout.bits_in(req, R.credited(req, layout, 0), 5) == 0
    early = R.Store(store.registered, store.written, {**store.freed, 1: frozenset({5})}, store.payload)
    assert R.store_matches(req, layout, early, 64, 1) and not R.store_matches(req, layout, early, 64, 2)
    assert R.plan_matches(req, layout, R.read_blocks(req, layout, 1) | {6}, 64, 1)
    assert not R.plan_matches(req, layout, R.read_blocks(req, layout, 1) - {5}, 64, 1)
    forged = R.Store(store.registered, store.written, store.freed, {**store.payload, 6: store.payload[6] ^ 1})
    assert not R.store_matches(req, layout, forged, 64, 2)


def test_a_block_a_later_step_fills_is_credited_once_complete(run):
    """With k_1 (born at step 1) beside k_0 and tok_1 in step 0's append, as a KV pool fills a page, that block is
    incomplete at step 1: its live k_0 and tok_1 aren't credited then, 64 of the live bits, and the store is W's with
    the block written in step 1's record, the one that completes it. Written in step 0's record, without k_1, it is
    resident at step 1 crediting nothing, but at step 2 its payload lacks k_1."""
    _, _, _, req = run
    layout, store = R.pack(req, 128)
    assert (store.registered, layout.block[14], layout.block[15], layout.block[19], layout.offset[15]) == (3, 3, 3, 4, 32)
    paged = R.Layout({**layout.block, 19: 3}, {**layout.offset, 19: 64})
    filled = R.Store(3, {0: frozenset(), 1: frozenset({3, 4})}, store.freed,
                     {**store.payload, 3: store.payload[3] | req.values[19] << 64})
    assert R.incomplete(req, paged, 1) == {3, 4} and not R.incomplete(req, paged, 2)
    assert R.credited(req, paged, 1) == req.live(1) - {14, 15} and R.credited(req, paged, 2) == req.live(2)
    assert R.credited_reads(req, paged, 1) == req.reads(1) - {14, 15}
    assert req.live_bits(1) - req.bits(R.credited(req, paged, 1)) == 64
    assert all(R.store_matches(req, paged, filled, 128, t) for t in range(req.steps))
    assert R.store_matches(req, paged, store, 128, 1) and not R.store_matches(req, paged, store, 128, 2)


def test_past_the_executed_prefix_nothing_runs(run):
    """Without step 2's gates (the request ended after step 1), k_1 has only step 1's reader and tok_2 none: neither is
    ever live."""
    program, _, schedule, _ = run
    prefix = R.Schedule(tuple(r for r in schedule.ranges if r[2] < 2))
    req = R.requirement(program, prefix, _inputs())
    assert req.to_json() == {"steps": 2, "live_bits": [320, 384], "read_bits": [128, 192], "registered_bits": 320}
    assert req.last[19] == req.born[19] == 1 and 21 not in req.last
    assert not {19, 21} & (req.live(1) | req.live(2))


def test_a_placement_credits_the_gates_the_execution_computes(run):
    """The run's input, the token, is an input gate wherever it is placed: all on one device over the whole run, the
    credits at each value's width are the computed gates', 576 bits, not 608 with the token's. The wiring is topological,
    as `RequirementHelp` needs. A placement that leaves a value without a device, or a gate without a tick at its step,
    is refused, not read as device 0 or tick 0."""
    program, _, schedule, req = run
    ex = R.execution(program, schedule, _inputs())
    assert ex.topological() and ex.inputs == {0} and ex.computed == ex.ran - ex.registered - {0}
    one, price, T = R.Placement({v: 0 for v in ex.ran}, {0: list(range(req.steps + 1))}), dict(ex.width), req.steps
    assert one.region(ex, {0}, 0, T) == one.step_region(ex, {0}, 0, T) == ex.computed
    assert one.credits(ex, price, {0}, (), 0, T) == 576 and one.ingress(ex, {0}, 0, T) == 0
    for bad in (R.Placement({v: 0 for v in ex.ran - {0}}, one.ticks), R.Placement(one.leaf, {0: [0]})):
        assert not bad.places(ex)
        with pytest.raises(ValueError, match="doesn't place"):
            bad.credits(ex, price, {0}, (), 0, T)


def _starts(program: Program, ends: list[int]) -> list[int]:
    """A census's step ends (root nodes) as `Schedule.by_calls`'s starts (Calls)."""
    return [0] + [e - sum(1 for nd in program.root.body.nodes if is_input(nd.fn)) for e in ends[:-1]]


def _span(req: R.Requirement, v: int) -> tuple:
    """A value's birth and its last structural reader after it (a read in its own step, such as a scan's carry, keeps nothing live)."""
    b, last = req.born.get(v), req.last.get(v)
    return b, last if last is not None and b is not None and last > b else None


def _census_path(program: Program, ends: list[int]) -> R.Requirement:
    """`from_census` is `upper_bound` at the census's grain: each census value's gates have its birth and last structural reader, and
    every step's live gates and bounded reads are its values'."""
    c = census(program, ends)
    schedule = R.Schedule.by_calls(program, _starts(program, ends))
    by_census, by_gates = R.from_census(c, schedule), R.upper_bound(program, schedule)
    gates = lambda vs: R.census_gates(program, c, vs)  # noqa: E731
    assert R.census_offsets(c) == list(program.root.body.offsets) and R.from_census(c) == by_census
    assert gates(by_census.registered) == by_gates.registered
    for x in range(len(c["values"]["producer"])):
        assert {_span(by_gates, g) for g in gates([x])} <= {_span(by_census, x)}
    for t in range(by_gates.steps):
        assert gates(by_census.live(t)) == by_gates.live(t) and gates(by_census.reads_upper(t)) == by_gates.reads(t)
    want = by_gates.to_json()
    if not all(by_census.resolved(t) for t in range(by_census.steps)):
        want["read_bits_upper"] = want.pop("read_bits")
    assert by_census.to_json() == want
    return by_census


@pytest.mark.parametrize("label,P", [("decoder", decoder()[0]), *programs()], ids=lambda x: x if isinstance(x, str) else "")
def test_the_census_path_is_the_expansion(label, P):
    """On every test Program, with every Call its own step and with one step for all."""
    calls = [k + 1 for k, nd in enumerate(P.root.body.nodes) if not is_input(nd.fn)]
    for ends in (calls, calls[-1:]):
        _census_path(P, ends)


def test_the_decoders_census_path(run):
    """At the decoder's steps: the requirement of `test_the_decoders_requirement`, with its gathers' reads bounded (the
    whole embedding, 256 bits) and refused, since a census holds no run values."""
    program, starts, _, req = run
    nodes = program.root.body.nodes
    calls = [k for k, nd in enumerate(nodes) if not is_input(nd.fn)]
    got = _census_path(program, [calls[s] for s in starts[1:]] + [len(nodes)])
    assert got.to_json() == {"steps": 3, "live_bits": [320, 384, 416], "read_bits_upper": [320, 384, 416], "registered_bits": 320}
    assert all(got.read_bits_upper(t) - req.read_bits(t) == 6 * 32 for t in range(3))
    with pytest.raises(ValueError, match="reads_upper"):
        got.read(0, 1)


def test_the_request_census_path():
    """`test_census.request` at its sampling steps: weights live throughout, a KV row from the step after its write to its
    last attention, a sampled token into the step that embeds it, the prompt token never (it arrives with step 0); the
    table's selection leaves every step's reads bounded."""
    P = request()
    got = _census_path(P, step_ends(P, {"TSelect_v1"}))
    assert got.to_json() == {"steps": 3, "live_bits": [1088, 1184, 1248], "read_bits_upper": [1088, 1184, 1248],
                             "registered_bits": 1088}
    assert not any(got.resolved(t) for t in range(3))


def test_a_split_call_another_schema_and_an_approximate_census_are_refused():
    program = request()
    c = census(program)
    schedule = R.census_schedule(c)
    lo, hi, s = next(r for r in schedule.ranges if r[1] - r[0] > 1)
    split = R.Schedule(tuple(r for r in schedule.ranges if r[0] != lo) + ((lo, lo + 1, s), (lo + 1, hi, s)))
    with pytest.raises(ValueError, match="splits root gates"):
        R.from_census(c, split)
    with pytest.raises(ValueError, match="verity/census/v1"):
        R.from_census(c | {"schema": "verity/census/v0"})
    with pytest.raises(ValueError, match="approximate"):
        R.from_census(c | {"approximate": True})


def test_a_gather_needs_the_runs_inputs(run):
    """Without the run's inputs the gathers' selected words are unknown: `requirement` refuses them, and `upper_bound`
    counts each gather's whole column, more reads than any run makes."""
    program, _, schedule, req = run
    with pytest.raises(ValueError, match="gather"):
        R.requirement(program, schedule)
    bound = R.upper_bound(program, schedule)
    assert bound.live(1) == req.live(1) and req.reads(1) < bound.reads(1)


def test_an_execution_that_does_not_follow_its_wiring_is_refused(run):
    """`Requirement.of` takes only an execution that follows its wiring (`Execution.Follows`): a gate whose operand is
    produced at a later step, or that uses a value that isn't its operand, is refused."""
    program, _, schedule, _ = run
    ex = R.execution(program, schedule, _inputs())
    assert ex.follows() and R.Requirement.of(ex) == R.requirement(program, schedule, _inputs())
    late = R.Execution(ex.width, ex.registered, {**ex.step, 14: 2}, ex.ops, ex.uses, ex.values)
    foreign = R.Execution(ex.width, ex.registered, ex.step, ex.ops, {**ex.uses, 14: ex.uses[14] | {21}}, ex.values)
    for bad in (late, foreign):
        assert not bad.follows()
        with pytest.raises(ValueError, match="Follows"):
            R.Requirement.of(bad)
