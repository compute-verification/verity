"""The ingress-ids tie on a fake upstream Commit and the warden's own ingress (`io_ids`): each prompt id the warden
certified on ingress held to every computation of its position in the Commit's `c00.input/ids` rows, and every
prefix-cache read in place of one, a bucket row at a time, by the reference and by the Boolean Definitions. The first
negative is a prompt substitution: IoTieRequest alone leaves the prompt ids' values bound to nothing, so the model
could compute on another prompt of the same length."""
from __future__ import annotations

import hashlib
import os
import random

import numpy as np
import pytest

from verity.ml.circuits.boolean.forms import and_reduce
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.commitments import advice as ADV
from verity.ml.commitments.gates.frame_v3 import _part
from verity.core.protocols.certifier import head_check as HC
from verity.ml.protocols.certifier import io_ids as I
from verity.core.protocols.certifier import io_request as Q
from verity.core.protocols.certifier import io_tie as IT
from verity.core.protocols.certifier import warden as W
from verity.core.protocols.certifier import warden_rows as WR
from verity.core.protocols.certifier.active import id_request

PL = (5, 60, 12, 100, 1)
B, P = 512, 16
PARAMS = W.WardenParams(cell_bytes=B, slots=2, lanes=2, window=8, delay=1)
#: request 2 trickles in so that its `END` shares bucket 4 with request 3's first two cells
ARRIVALS, RATES = (0, 0, 2, 2, 7), (10 ** 6, 10 ** 6, 300, 10 ** 6, 10 ** 6)
#: the buckets that hold prompt fields: request 0's and 1's first, 1's last, 2's and 3's first, 3's last, 4's
BUCKETS = 9
#: request 3 reads request 1's KV for its first 32 positions (another bucket's), or request 2's for its first 8 (its own)
ACROSS, WITHIN = {3: (1, 32)}, {3: (2, 8)}
SHAPE = I.Shape(B, PARAMS.lanes, PARAMS.slots, P, 8, 4)


def _prompts(sources):
    ps = [[(7919 * i + 104729 * k) % 151329 for k in range(n)] for i, n in enumerate(PL)]
    for q, (src, hit) in sources.items():
        ps[q][:hit] = ps[src][:hit]
    return ps


def _steps(sources):
    """Chunked prefill in arrival order, P tokens a step, each request's first output in the step after its prefill,
    and a recompute of request 1's first 16 positions (a preemption) last."""
    pos, todo, steps = {q: sources.get(q, (0, 0))[1] for q in range(len(PL))}, list(range(len(PL))), []
    while todo:
        rows, room = [], P
        for q in list(todo):
            if not room:
                break
            n = min(room, PL[q] - pos[q])
            rows.append(I.Row(q, pos[q], n or 1))
            room -= n or 1
            if n:
                pos[q] += n
            else:
                todo.remove(q)
        steps.append(rows)
    return steps + [[I.Row(1, 0, 16)]]


def _tokens(ps):
    return [p + [900_000 + 100 * q + k for k in range(4)] for q, p in enumerate(ps)]


def _link(sources, ps=None, tokens=None, shape=SHAPE):
    """Each bucket's instance: the warden's cells of the requests with prompts `ps` (the sources' by default), the
    Commit of `tokens` (the prompts' by default) on the schedule, and the verifier's lanes and map."""
    ps = _prompts(sources) if ps is None else ps
    reqs = [id_request(model="glm", user=f"u{i}", prompt=p, max_tokens=16, temperature=0.7, top_p=0.95, seed=i,
                       frame_bytes=B, offset=0) for i, p in enumerate(ps)]
    link = Q.serve(PARAMS, reqs, ARRIVALS, RATES, buckets=BUCKETS)
    steps = _steps(sources)
    rows = I.commit_rows(steps, _tokens(ps) if tokens is None else tokens, P)
    salts, levels, dom = I.commit(rows)
    out = []
    for t, row in enumerate(link.rows):
        lanes_ = Q.lanes(shape.request, PL, link.releases, t)
        mp = I.bucket_map(shape, PL, lanes_, steps, sources)
        out.append(I.instance(shape, Q.Instance(tuple(row), lanes_), mp, rows, salts, levels, dom))
    return out


def _verdicts(insts):
    return [I.reference(SHAPE, i) for i in insts]


def _holding(insts, q, p):
    """The bucket whose fields hold request q's position p."""
    return next(t for t, i in enumerate(insts) if (q, p) in I.fields_of(i.request.lanes))


def test_an_honest_link_is_tied_in_every_bucket():
    """Chunked prefill across buckets, a recompute after a preemption, and a prefix-cache hit read from another
    bucket's request or from one in the same bucket: every bucket accepted. The fields are the lanes', in order, and
    a cached position shares its source's class."""
    assert _verdicts(_link(ACROSS)) == [1] * BUCKETS
    within = _link(WITHIN)
    assert _verdicts(within) == [1] * BUCKETS
    t = _holding(within, 3, 0)
    assert t == _holding(within, 2, 0) == 4
    keys, fields = within[t].map.keys, I.fields_of(within[t].request.lanes)
    assert len(fields) == 69 and fields[:2] == [(2, 0), (2, 1)]
    shared = [keys[fields.index((3, p))] == keys[fields.index((2, p))] for p in range(12)]
    assert shared == [True] * 8 + [False] * 4
    assert sum(len(I.fields_of(i.request.lanes)) for i in within) == sum(PL)


def test_redteam_phase9_prompt_substitution_is_refused():
    """A prompt substitution: the model computes on a prompt of the same length with one id changed, here request 2's
    position 5. IoTieRequest accepts every bucket, since the fields' count and form are the user's; the ids tie
    refuses the bucket whose cells hold that field, since its class holds the user's id and the Commit's."""
    tokens = _tokens(_prompts(ACROSS))
    tokens[2][5] ^= 1
    insts = _link(ACROSS, tokens=tokens)
    assert [Q.reference(SHAPE.request, i.request) for i in insts] == [1] * BUCKETS
    t = _holding(insts, 2, 5)
    assert _verdicts(insts) == [int(k != t) for k in range(BUCKETS)]


@pytest.mark.parametrize("sources", [ACROSS, WITHIN])
def test_a_cache_hit_on_another_prefix_is_refused(sources):
    """A prefix-cache hit claimed on another prompt: request 3's user sent another id at a cached position than its
    source's. Read from another bucket, the class holds request 3's field and the source's computations; from the
    same bucket, also the source's field. Either way the bucket holding request 3's field refuses."""
    ps = _prompts(sources)
    ps[3][4] ^= 1
    insts = _link(sources, ps=ps)
    t = _holding(insts, 3, 4)
    assert _verdicts(insts) == [int(k != t) for k in range(BUCKETS)]


def test_a_commit_that_differs_at_a_source_is_refused_in_both_buckets():
    """The Commit computes request 1's position 3 on another id: request 1's own bucket refuses, and so does the
    bucket of request 3, which reads that position from the cache."""
    tokens = _tokens(_prompts(ACROSS))
    tokens[1][3] ^= 1
    insts = _link(ACROSS, tokens=tokens)
    bad = {_holding(insts, 1, 3), _holding(insts, 3, 3)}
    assert len(bad) == 2 and _verdicts(insts) == [int(k not in bad) for k in range(BUCKETS)]


def test_the_read_rows_are_the_commits_at_the_maps_steps():
    """Each bucket's read rows are the Commit's registered rows (`verity.ml.commitments.advice`'s registration) at its
    map's steps: the verifier's own ports (`port_reads`, `INPUT_IDS`) have unit u read bucket u's steps, and the Lean
    verifier's registered reads (`warden_rows.refusal`) accept every bucket's rows and refuse a row, a salt or a step
    other than the Commit's, as `reads_open` does in the clear."""
    insts, count = _link(ACROSS), len(_steps(ACROSS))
    seed = hashlib.sha512(b"commit").digest()
    rows = I.commit_rows(_steps(ACROSS), _tokens(_prompts(ACROSS)), P)
    salts = [IT.fake_salt(b"commit", s) for s in range(count)]
    root = ADV.commit(rows, 32 * P, seed, salts).root
    assert root == insts[0].port_root
    dom = IT.domain(seed, count)
    ports = I.port_reads([i.map for i in insts], root, dom.domain_id, count, P)
    assert [p["reads"] for p in ports.values()] == [[i.map.steps[s] for i in insts] for s in range(SHAPE.S)]

    def refusals(insts_):
        return [WR.refusal(ports[f"{IT.PORT}[{s}]"], dom, {
            "value": I.INPUT_IDS, "positions": [i.map.steps[s] for i in insts_],
            "paths": [b"".join(i.paths[s]).hex() for i in insts_]}, [WR.bc(i.port[s], i.port_salts[s]) for i in insts_])
            for s in range(SHAPE.S)]

    assert refusals(insts) == [None] * SHAPE.S and all(I.reads_open(SHAPE, i) for i in insts)
    u, b = 4, insts[4]
    row = bytearray(b.port[1])
    row[3] ^= 1
    steps = list(b.map.steps)
    steps[0] = (steps[0] + 1) % count
    for name, bad in {"a row": b._replace(port=(b.port[0], bytes(row), *b.port[2:])),
                      "a salt": b._replace(port_salts=(IT.fake_salt(b"x", 0), *b.port_salts[1:])),
                      "a step": b._replace(map=b.map._replace(steps=tuple(steps)))}.items():
        assert not I.reads_open(SHAPE, bad) and not I.reference(SHAPE, bad), name
        assert any(refusals([*insts[:u], bad, *insts[u + 1:]])), name


def test_a_second_tree_of_the_users_ids_is_refused_under_the_commits_registration():
    """The forward computes request 2's position 5 on another id, and the Commit registers that
    tree. The prover hands the verifier instances opened from a second tree of the user's ids under the same public
    domain, so every unit is 1. The verifier's own ports name the registration's root and its own maps' steps, never
    an instance's, so the second tree's rows open none of them."""
    tokens = _tokens(_prompts(ACROSS))
    tokens[2][5] ^= 1
    rows = I.commit_rows(_steps(ACROSS), tokens, P)
    salts, levels, dom_id = I.commit(rows)
    registered = ADV.commit(rows, 32 * P, hashlib.sha512(b"commit").digest(), salts)
    forged = _link(ACROSS)
    assert registered.root == levels[-1][0] != forged[0].port_root and forged[0].port_domain == dom_id
    assert [I.unit_reference(SHAPE, i) for i in forged] == [1] * BUCKETS
    ports = I.port_reads([i.map for i in forged], registered.root, dom_id, len(rows), P)
    assert {p["root"] for p in ports.values()} == {registered.root.hex()}
    dom = IT.domain(hashlib.sha512(b"commit").digest(), len(rows))
    assert all(WR.refusal(ports[f"{IT.PORT}[{s}]"], dom, {
        "value": I.INPUT_IDS, "positions": [i.map.steps[s] for i in forged],
        "paths": [b"".join(i.paths[s]).hex() for i in forged]}, [WR.bc(i.port[s], i.port_salts[s]) for i in forged])
        for s in range(SHAPE.S))


def test_the_map_refuses_an_advice_it_cant_tie():
    """The verifier refuses an advice whose class no step computes (a cached position whose source never computed
    it), a source that isn't an earlier request, a step past P, and more opened steps than S."""
    insts = _link(ACROSS)
    lanes_ = insts[4].request.lanes
    steps = _steps(ACROSS)
    with pytest.raises(ValueError, match="no step computes request 1's prompt position"):
        I.bucket_map(SHAPE, PL, lanes_, [[r for r in s if r.q != 1] for s in steps], ACROSS)
    with pytest.raises(ValueError, match="an earlier request"):
        I.bucket_map(SHAPE, PL, lanes_, steps, {3: (4, 32)})
    with pytest.raises(ValueError, match="more than P"):
        I.bucket_map(SHAPE, PL, lanes_, steps + [[I.Row(0, 0, 5), I.Row(2, 0, 12)]], ACROSS)
    with pytest.raises(ValueError, match="more than S"):
        I.bucket_map(I.Shape(B, 2, 2, P, 4, 4), PL, lanes_, steps, ACROSS)


def test_the_order_names_each_lanes_request():
    """With advice v1's order, connection q's fields are advice request i's where order[i] + 1 = q."""
    lanes_ = (Q.Lane(2, 0, 0, 0, 2), Q.Lane(1, 0, 0, 0, 1))
    assert I.fields_of(lanes_) == [(1, 0), (1, 1), (0, 0)]
    assert I.fields_of(lanes_, order=(1, 0)) == [(0, 0), (0, 1), (1, 0)]


def test_the_map_and_its_routings_are_the_cuts():
    """Bucket 4's map: its 69 fields' classes then the sink, every opened step's entries a class of the bucket or the
    sink, every class with an entry; the routing takes record f to field f's slot, and the sort orders the members.
    An idle bucket's map has no field and opens step 0."""
    insts = _link(ACROSS)
    inst = insts[4]
    sh, mp = SHAPE, inst.map
    vals, at = I.scan(sh, inst.request)
    assert len(vals) == len(at) == 69 and mp.keys[69:] == (sh.sink,) * (sh.NK - 69)
    classes = set(mp.keys[:69])
    assert set(mp.ekeys) - {sh.sink} == classes
    assert all(I.entry(sh, inst.port, e) == vals[mp.keys.index(k)] for e, k in enumerate(mp.ekeys) if k != sh.sink)
    assert len(I.route_switches(sh, inst)) == sh.route and len(I.sort_switches(sh, inst)) == sh.sort
    idle = insts[0].map
    assert idle.keys == (sh.sink,) * sh.NK and idle.steps == (0,) * sh.S and set(idle.ekeys) == {sh.sink}


# --- the circuit -----------------------------------------------------------------------------------------------------

def _bits(rows):
    return np.asarray(rows, dtype=np.uint8)


def _int(bits) -> int:
    return sum(int(b) << j for j, b in enumerate(bits))


def test_the_slots_circuit_is_its_reference():
    """`IoTieIdsSlots` on every cell of the link (first cells, continuations, pads, unheld slots), given the cell's
    `IoTieRequestFrame` output on its lane's ordinal and the bucket's fields before it: nf', and each slot's field bit,
    ordinal and, at a field, the value `scan` reads."""
    sh, cases, want = SHAPE, [], []
    frame = bind(Q.IoTieRequestFrame, B=B)
    for inst in _link(WITHIN):
        r = inst.request
        vals, at = I.scan(sh, r)
        nf = 0
        for k, ln in enumerate(r.lanes):
            left = ln.left
            for i in range(k * sh.R, (k + 1) * sh.R):
                fo = [int(b) for b in np.asarray(evaluate_bits(frame, *[_bits(a) for a in Q.frame_arguments(
                    [r.cells[i]], [left], [ln.q])])).reshape(-1)]
                mine = {s % sh.K: vals[f] for f, s in enumerate(at) if s // sh.K == i}
                cases.append((r.cells[i], fo, nf))
                want.append((nf + len(mine), [(int(k in mine), nf + k, mine.get(k)) for k in range(sh.K)]))
                left, nf = (_int(fo[4:36]) if fo[1] else left), nf + len(mine)
    out = np.asarray(evaluate_bits(bind(I.IoTieIdsSlots, B=B), *[_bits(a) for a in I.slots_arguments(*zip(*cases))]))
    got = []
    for o in out.reshape(len(cases), -1):
        slots = [o[32 + I.SLOT * k:32 + I.SLOT * (k + 1)] for k in range(sh.K)]
        got.append((_int(o[:32]), [(int(s[0]), _int(s[1:33]), _int(s[33:]) if s[0] else None) for s in slots]))
    assert sum(f for _, slots in want for f, _, _ in slots) == sum(PL) and got == want
    assert HC.ands(bind(I.IoTieIdsSlots, B=B)) == 24_670


def test_the_keys_circuit_is_its_reference():
    """`IoTieIdsKeys` at NK = 5 on random slots and routed records: ok iff every field's record has its ordinal and a
    class other than the sink; each field's class its record's, every other slot's the sink."""
    NK, CB, D1 = 5, 3, 3
    rng = random.Random(5)
    cases, want = [], []
    for _ in range(200):
        slots = [(rng.randrange(2), rng.choice([rng.randrange(8), rng.randrange(1 << 32)]), rng.randrange(1 << 32))
                 for _ in range(NK)]
        routed = [(rng.choice([7, rng.randrange(8)]), rng.randrange(8)) for _ in range(8)]
        for s, (fld, o, _) in enumerate(slots):
            if fld and rng.random() < 0.8:
                routed[s] = (rng.randrange(7), o % 8)
                slots[s] = (fld, o % 8, slots[s][2])
        cases.append((slots, routed))
        ok = all(not f or (routed[s][1] == o and routed[s][0] != 7) for s, (f, o, _) in enumerate(slots))
        want.append((int(ok), [routed[s][0] if f else 7 for s, (f, _, _) in enumerate(slots)]))
    args = [[[b for f, o, v in sl for b in [f] + HC._int_bits(o, 32) + HC._int_bits(v, 32)] for sl, _ in cases],
            [[b for k, x in ro for b in HC._int_bits(k, CB) + HC._int_bits(x, D1)] for _, ro in cases]]
    out = np.asarray(evaluate_bits(bind(I.IoTieIdsKeys, NK=NK, CB=CB, D1=D1), *[_bits(a) for a in args]))
    got = [(int(o[0]), [_int(o[1 + CB * s:1 + CB * (s + 1)]) for s in range(NK)]) for o in out.reshape(len(cases), -1)]
    assert got == want and 40 < sum(w[0] for w in want) < 160


def _chain_reference(ms, sink: int) -> int:
    """1 iff the members (class, kind, value) have classes that don't decrease, one value a class but the sink, and each
    class but the sink starting with a slot (kind 0)."""
    if ms[0][0] != sink and ms[0][1]:
        return 0
    for (ka, _, va), (kb, kind, vb) in zip(ms, ms[1:]):
        if kb < ka or (kb == ka != sink and va != vb) or (kb != ka and kb != sink and kind):
            return 0
    return 1


def test_the_chain_circuit_is_its_reference():
    """`IoTieIdsChain` at 8 members on sorted and unsorted runs: 1 iff the classes don't decrease, a class but the sink
    holds one value, and each class but the sink starts with a slot."""
    CB, D2, sink = 3, 3, 7
    ref = lambda ms: _chain_reference(ms, sink)  # noqa: E731
    rng = random.Random(7)
    cases = []
    for _ in range(300):
        keys = sorted(rng.choice([0, 1, 2, 5, sink]) for _ in range(8)) if rng.random() < 0.8 else \
            [rng.randrange(8) for _ in range(8)]
        vals = {k: rng.randrange(1 << 32) for k in range(8)}
        ms, prev = [], None
        for k in keys:
            kind = 0 if k != prev else rng.choice([0, 1, 1])
            v = vals[k] if rng.random() < 0.9 else rng.randrange(1 << 32)
            ms.append((k, kind if rng.random() < 0.9 else 1 - kind, v))
            prev = k
        cases.append(ms)
    want = [ref(ms) for ms in cases]
    args = [[[b for k, kind, v in ms for b in HC._int_bits(k, CB) + [kind] + HC._int_bits(v, 32)] for ms in cases]]
    out = np.asarray(evaluate_bits(bind(I.IoTieIdsChain, CB=CB, D2=D2), *[_bits(a) for a in args]))
    assert [int(b) for b in out.reshape(-1)] == want and 60 < sum(want) < 240


def _build_flat_chain(C, S, *ms):
    """`IoTieIdsChain` as one flat trace, the build the batched one replaced."""
    def parts(m):
        return m[:S.CB], m[S.CB], m[S.CB + 1:]

    key, kind, _ = parts(ms[0])
    oks = [C.AND(kind, and_reduce(C, key).inv()).inv()]
    for t in range(1, len(ms)):
        (ka, _, va), (kb, kind, vb) = parts(ms[t - 1]), parts(ms[t])
        same, sink = IT._eq(C, ka, kb), and_reduce(C, kb)
        oks += [IT._lt(C, kb, ka).inv(), C.AND(C.AND(same, sink.inv()), IT._eq(C, va, vb).inv()).inv(),
                C.AND(C.AND(same.inv(), sink.inv()), kind).inv()]
    return [and_reduce(C, oks)]


FLAT_CHAIN = _part("IoTieIdsChainFlat", ("CB", "D2"), I._sig_chain, _build_flat_chain, "IoTieIdsChain's flat trace")


def _chain_ands(CB: int, D2: int) -> int:
    """The first member's CB ANDs, each link's 3CB + 33 and the And's one a link."""
    return CB + ((1 << D2) - 1) * (3 * CB + 36)


def _chains(rng, CB: int, D2: int, count: int) -> list[list[list[int]]]:
    """Runs of 2^D2 members (class, kind, value): the even ones valid chains (sorted classes, each class but the sink one
    value and a slot first), the odd ones a valid chain with one member's class, kind or value redrawn."""
    n, sink = 1 << D2, (1 << CB) - 1
    out = []
    for c in range(count):
        ks = sorted(rng.choice([rng.randrange(sink), sink]) for _ in range(n))
        vals = {k: rng.randrange(1 << 32) for k in ks}
        ms = [[k, rng.randrange(2) if k == sink or (t and ks[t - 1] == k) else 0,
               rng.randrange(1 << 32) if k == sink else vals[k]] for t, k in enumerate(ks)]
        if c % 2:
            m = ms[rng.randrange(n)]
            f = rng.randrange(3)
            m[f] = rng.randrange(1 << CB) if f == 0 else 1 - m[1] if f == 1 else rng.randrange(1 << 32)
        out.append(ms)
    return out


def _chain_out(fn, CB: int, chains) -> list[int]:
    rows = [[b for k, kind, v in ms for b in HC._int_bits(k, CB) + [kind] + HC._int_bits(v, 32)] for ms in chains]
    return [int(b) for b in np.asarray(evaluate_bits(fn, _bits(rows))).reshape(-1)]


def test_the_batched_chain_is_the_flat_one():
    """`IoTieIdsChain` (`IoTieIdsFirst`, one batch of `IoTieIdsLink` over each member and its successor, one And) against
    the flat trace it replaced: the same ANDs, and the same outputs as each other and the reference on random runs, at
    D2 = 1 to 8. At IoTieRequestIds{4096, 16, 1, 512, 6}'s sort (CB = 13, D2 = 14) a binding's body is its three nodes,
    so a link is traced once, with the flat trace's ANDs, and its outputs are the reference's; the flat trace there is
    `test_the_batched_chain_is_the_flat_one_at_the_served_sort`'s."""
    rng, refused = random.Random(11), 0
    for CB, D2 in ((1, 1), (3, 1), (2, 2), (3, 3), (6, 4), (1, 5), (9, 6), (13, 8)):
        new, flat = bind(I.IoTieIdsChain, CB=CB, D2=D2), bind(FLAT_CHAIN, CB=CB, D2=D2)
        assert HC.ands(new) == HC.ands(flat) == _chain_ands(CB, D2)
        chains = _chains(rng, CB, D2, 16)
        want = [_chain_reference(ms, (1 << CB) - 1) for ms in chains]
        assert _chain_out(new, CB, chains) == _chain_out(flat, CB, chains) == want and all(want[0::2])
        refused += want.count(0)
    assert refused >= 20
    deep = bind(I.IoTieIdsChain, CB=13, D2=14)
    assert [nd.form for nd in deep.body.nodes] == ["call", "batch", "primitive"]
    assert HC.ands(deep) == _chain_ands(13, 14) == 1_228_738
    chains = _chains(rng, 13, 14, 4)
    assert _chain_out(deep, 13, chains) == [_chain_reference(ms, (1 << 13) - 1) for ms in chains]


@pytest.mark.skipif(not os.environ.get("VERITY_CERTIFIER_HEAVY"),
                    reason="the flat trace of 1.2M ANDs, a few GB and minutes: VERITY_CERTIFIER_HEAVY=1")
def test_the_batched_chain_is_the_flat_one_at_the_served_sort():
    """`IoTieIdsChain{13, 14}`, IoTieRequestIds{4096, 16, 1, 512, 6}'s, against its flat trace on random runs, valid and
    with a member redrawn: what the default test's D2 <= 8 can't show is the batch's wiring at 16,383 links."""
    rng = random.Random(14)
    chains = _chains(rng, 13, 14, 8)
    want = [_chain_reference(ms, (1 << 13) - 1) for ms in chains]
    assert _chain_out(bind(I.IoTieIdsChain, CB=13, D2=14), 13, chains) == want and all(want[0::2])
    assert _chain_out(bind(FLAT_CHAIN, CB=13, D2=14), 13, chains) == want


@pytest.mark.slow
def test_the_unit_is_its_reference():
    """`IoTieRequestIds{512, 2, 2, 16, 8}` against `unit_reference` on the link's buckets, honest, under the prompt
    substitution, the Commit differing at a cache source, and a cache hit on another prefix in the same bucket; a
    field's class the sink, an entry's class another class, two fields' classes swapped, and a lane's carry-out one
    field short; and random switches for each routing. What the parts' tests can't reach is the wiring: each lane's
    cells into its frames on its ordinal, the slots of each cell from its frame and the bucket's running count of
    fields, the classes' routing to the fields, the members (slots, the read rows' entries, the sink) into the sort,
    and IoTieRequest's checks beside them. No gate of it hashes: the read rows are the registered reads'.

    Slow (the unit's ANDs traced and evaluated on 19 cases, about 18 s)."""
    sh = SHAPE
    ps = _prompts(ACROSS)
    sub, src = _tokens(ps), _tokens(ps)
    sub[2][5] ^= 1
    src[1][3] ^= 1
    within = _prompts(WITHIN)
    within[3][4] ^= 1
    honest = _link(ACROSS)
    h4, mp = honest[4], honest[4].map
    insts = honest + _link(ACROSS, tokens=sub)[4:6] + _link(ACROSS, tokens=src)[1:2] + _link(WITHIN, ps=within)[4:5]
    e = next(k for k, x in enumerate(mp.ekeys) if x != sh.sink)
    ek, ks = list(mp.ekeys), list(mp.keys)
    ek[e] = (ek[e] + 1) % 69
    ks[0], ks[1] = ks[1], ks[0]
    lanes_ = h4.request.lanes
    short = (lanes_[0]._replace(fields_out=lanes_[0].fields_out - 1),) + lanes_[1:]
    insts += [h4._replace(map=mp._replace(keys=(sh.sink,) + mp.keys[1:])), h4._replace(map=mp._replace(ekeys=tuple(ek))),
              h4._replace(map=mp._replace(keys=tuple(ks))), h4._replace(request=h4.request._replace(lanes=short))]
    want = [I.unit_reference(sh, i) for i in insts]
    assert want == [I.reference(sh, i) for i in insts] == [1] * BUCKETS + [0, 1, 0, 0] + [0, 0, 0, 0]
    rng = random.Random(3)
    cases = [(i, I.route_switches(sh, i), I.sort_switches(sh, i)) for i in insts]
    cases += [(h4, [rng.randrange(2) for _ in range(sh.route)], I.sort_switches(sh, h4)),
              (h4, I.route_switches(sh, h4), [rng.randrange(2) for _ in range(sh.sort)])]
    fn = bind(I.IoTieRequestIds, B=sh.B, L=sh.L, R=sh.R, P=sh.P, S=sh.S)
    args = [I.arguments(sh, i, r, s) for i, r, s in cases]
    out = evaluate_bits(fn, *[_bits([a[k] for a in args]) for k in range(len(args[0]))])
    assert [int(b) for b in np.asarray(out).reshape(-1)] == want + [0, 0]
    a = HC.ands
    rq = sh.request
    parts = (rq.N * (a(bind(Q.IoTieRequestFrame, B=sh.B)) + a(bind(Q.IoTieRequestStep)) + a(bind(I.IoTieIdsSlots, B=B)))
             + rq.L * a(bind(Q.IoTieRequestEnd))
             + a(bind(IT.IoTieRoute, W=sh.CB + sh.D1, D=sh.D1)) + a(bind(I.IoTieIdsKeys, NK=sh.NK, CB=sh.CB, D1=sh.D1))
             + a(bind(IT.IoTieRoute, W=sh.CB + 33, D=sh.D2)) + a(bind(I.IoTieIdsChain, CB=sh.CB, D2=sh.D2)) + rq.L + 1)
    assert a(fn) == parts
