"""The ingress-ids tie on a fake upstream (`io_ids`): each prompt id the warden certified on ingress held to every
computation of its position in the Commit's `c00.input/ids` rows, and every prefix-cache read in place of one, by the
reference and by the Boolean Definitions. The first negative is REDTEAM phase 9 #1 (`cursor/redteam-phase9-iotiereq-061a`):
IoTieRequest left the prompt ids' values bound to nothing, so the model could compute on another prompt of the same
length."""
from __future__ import annotations

import random

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.core.protocols.certifier import head_check as HC
from verity.core.protocols.certifier import io_ids as I
from verity.core.protocols.certifier import io_request as Q
from verity.core.protocols.certifier import io_tie as IT
from verity.core.protocols.certifier.active import id_request

PL = (5, 60, 12, 100, 1)
COUNTS = (1, 1, 1, 0, 1, 2, 1, 1, 1, 1, 0, 0)
B, P = 512, 16
#: request 3 reads request 1's KV for its first 32 positions (another window's), or request 2's for its first 8 (its own)
ACROSS, WITHIN = {3: (1, 32)}, {3: (2, 8)}


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


SHAPE = I.Shape(B, 4, 2, 1, P, 8, 4)


def _link(sources, ps=None, tokens=None, shape=SHAPE):
    """Each window's instance: the warden's cells of the requests with prompts `ps` (the sources' by default), the
    Commit of `tokens` (the prompts' by default) on the schedule, and the verifier's map."""
    ps = _prompts(sources) if ps is None else ps
    reqs = [id_request(model="glm", user=f"u{i}", prompt=p, max_tokens=16, temperature=0.7, top_p=0.95, seed=i,
                       frame_bytes=B) for i, p in enumerate(ps)]
    windows = Q.windows_of(shape.request, reqs, COUNTS)
    received = tuple(sum(COUNTS[w * shape.T:(w + 1) * shape.T]) for w in range(len(windows)))
    steps = _steps(sources)
    rows = I.commit_rows(steps, _tokens(ps) if tokens is None else tokens, P)
    salts, levels, dom = I.commit(rows)
    out = []
    for w, cells in enumerate(windows):
        view = Q.request_view(PL, received, w, B)
        mp = I.window_map(shape, PL, view, steps, sources)
        out.append(I.instance(shape, Q.instance(shape.request, cells, view, w), mp, rows, salts, levels, dom))
    return out


def _verdicts(insts):
    return [I.reference(SHAPE, i) for i in insts]


def test_an_honest_link_is_tied_in_every_window():
    """Chunked prefill across windows, a recompute after a preemption, and a prefix-cache hit read from another
    window's request or from one in the same window: every window accepted. The fields are the view's, in order, and
    a cached position shares its source's class."""
    assert _verdicts(_link(ACROSS)) == [1, 1, 1]
    within = _link(WITHIN)
    assert _verdicts(within) == [1, 1, 1]
    v, keys = within[1].request.view, within[1].map.keys
    fields = I.fields_of(v)
    assert len(fields) == 72 and fields[:3] == [(1, 57), (1, 58), (1, 59)]
    shared = [keys[fields.index((3, p))] == keys[fields.index((2, p))] for p in range(12)]
    assert shared == [True] * 8 + [False] * 4


def test_redteam_phase9_prompt_substitution_is_refused():
    """REDTEAM phase 9 #1, live under v3: the model computes on a prompt of the same length with one id changed, here
    request 2's position 5. IoTieRequest accepts every window, since the fields' count and form are the user's; the
    ids tie refuses the window whose cells hold that field, since its class holds the user's id and the Commit's."""
    ps = _prompts(ACROSS)
    tokens = _tokens(ps)
    tokens[2][5] ^= 1
    insts = _link(ACROSS, tokens=tokens)
    assert [Q.reference(SHAPE.request, i.request) for i in insts] == [1, 1, 1]
    assert _verdicts(insts) == [1, 0, 1]


@pytest.mark.parametrize("sources", [ACROSS, WITHIN])
def test_a_cache_hit_on_another_prefix_is_refused(sources):
    """A prefix-cache hit claimed on another prompt: request 3's user sent another id at a cached position than its
    source's. Read from another window, the class holds request 3's field and the source's computations; from the same
    window, also the source's field. Either way the window holding request 3's field refuses."""
    ps = _prompts(sources)
    ps[3][4] ^= 1
    assert _verdicts(_link(sources, ps=ps)) == [1, 0, 1]


def test_a_commit_that_differs_at_a_source_is_refused_in_both_windows():
    """The Commit computes request 1's position 3 on another id: request 1's own window refuses, and so does the
    window of request 3, which reads that position from the cache."""
    ps = _prompts(ACROSS)
    tokens = _tokens(ps)
    tokens[1][3] ^= 1
    assert _verdicts(_link(ACROSS, tokens=tokens)) == [0, 0, 1]


def test_the_map_refuses_an_advice_it_cant_tie():
    """The verifier refuses an advice whose class no step computes (a cached position whose source never computed
    it), a source that isn't an earlier request, a step past P, and more opened steps than S."""
    shape = SHAPE
    view = Q.request_view(PL, (3, 5, 2), 1, B)
    steps = _steps(ACROSS)
    with pytest.raises(ValueError, match="no step computes request 1's prompt position"):
        I.window_map(shape, PL, view, [s for s in steps if not any(r.q == 1 for r in s)], ACROSS)
    with pytest.raises(ValueError, match="an earlier request"):
        I.window_map(shape, PL, view, steps, {3: (4, 32)})
    with pytest.raises(ValueError, match="more than P"):
        I.window_map(shape, PL, view, steps + [[I.Row(0, 0, 5), I.Row(2, 0, 12)]], ACROSS)
    with pytest.raises(ValueError, match="more than S"):
        I.window_map(I.Shape(B, 4, 2, 1, P, 4, 4), PL, view, steps, ACROSS)


def test_the_map_and_its_routings_are_the_cuts():
    """Window 1's map: its 72 fields' classes then the sink, every opened step's entries a class of the window or the
    sink, every class with an entry; the routing takes record f to field f's slot, and the sort orders the members."""
    inst = _link(ACROSS)[1]
    sh, mp, r = SHAPE, inst.map, inst.request
    vals, at = I.scan(sh, r.cells, r.view.carry_in, r.window)
    assert len(vals) == len(at) == 72 and mp.keys[72:] == (sh.sink,) * (sh.NK - 72)
    classes = set(mp.keys[:72])
    assert set(mp.ekeys) - {sh.sink} == classes
    assert all(I.entry(sh, inst.port, e) == vals[mp.keys.index(k)] for e, k in enumerate(mp.ekeys) if k != sh.sink)
    assert len(I.route_switches(sh, inst)) == sh.route and len(I.sort_switches(sh, inst)) == sh.sort


# --- the circuit -----------------------------------------------------------------------------------------------------

def _bits(rows):
    return np.asarray(rows, dtype=np.uint8)


def _int(bits) -> int:
    return sum(int(b) << j for j, b in enumerate(bits))


def test_the_slots_circuit_is_its_reference():
    """`IoTieIdsSlots` on every cell of the link (first frames, continuations, pads), given the cell's
    `IoTieRequestFrame` output and the window's fields before it: nf', and each slot's field bit, ordinal and, at a
    field, the value `scan` reads."""
    sh, cases, want = SHAPE, [], []
    frame = bind(Q.IoTieRequestFrame, B=B)
    for inst in _link(WITHIN):
        r = inst.request
        vals, at = I.scan(sh, r.cells, r.view.carry_in, r.window)
        left, nf = r.view.carry_in.left, 0
        for i in range(sh.OWN, sh.request.N):
            fo = [int(b) for b in np.asarray(evaluate_bits(frame, *[_bits(a) for a in Q.frame_arguments(
                [r.cells[i]], [left])])).reshape(-1)]
            mine = {s % sh.K: vals[f] for f, s in enumerate(at) if s // sh.K == i - sh.OWN}
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


def test_the_chain_circuit_is_its_reference():
    """`IoTieIdsChain` at 8 members on sorted and unsorted runs: 1 iff the classes don't decrease, a class but the sink
    holds one value, and each class but the sink starts with a slot."""
    CB, D2, sink = 3, 3, 7

    def ref(ms):
        if ms[0][0] != sink and ms[0][1]:
            return 0
        for (ka, _, va), (kb, kind, vb) in zip(ms, ms[1:]):
            if kb < ka or (kb == ka != sink and va != vb) or (kb != ka and kb != sink and kind):
                return 0
        return 1

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


@pytest.mark.slow
def test_the_unit_is_its_reference():
    """`IoTieRequestIds{512, 4, 2, 1, 16, 8, 4}` against `reference` on the link's three windows, honest, under the
    phase-9 substitution, the Commit differing at a cache source, and a cache hit on another prefix in the same window;
    a field's class the sink, an entry's class another class and two fields' classes swapped; and random switches for
    each routing. What the parts' tests can't reach is the wiring: the slots of each cell from the scan's frame and
    its running count of fields, the classes' routing to the fields, the members (slots, the opened rows' entries,
    the sink) into the sort, and IoTieRequest's and the port's checks beside them.

    Slow (about 140 s, the unit's 12.9 million ANDs traced and evaluated, most of it the rows' and the port's SHA-512
    bodies)."""
    sh = SHAPE
    ps = _prompts(ACROSS)
    sub, src = _tokens(ps), _tokens(ps)
    sub[2][5] ^= 1
    src[1][3] ^= 1
    within = _prompts(WITHIN)
    within[3][4] ^= 1
    honest = _link(ACROSS)
    h1, mp = honest[1], honest[1].map
    insts = honest + _link(ACROSS, tokens=sub) + _link(ACROSS, tokens=src) + _link(WITHIN, ps=within)[1:2]
    e = next(k for k, x in enumerate(mp.ekeys) if x != sh.sink)
    ek, ks = list(mp.ekeys), list(mp.keys)
    ek[e] = (ek[e] + 1) % 72
    ks[0], ks[1] = ks[1], ks[0]
    insts += [h1._replace(map=mp._replace(keys=(sh.sink,) + mp.keys[1:])), h1._replace(map=mp._replace(ekeys=tuple(ek))),
              h1._replace(map=mp._replace(keys=tuple(ks)))]
    want = [I.reference(sh, i) for i in insts]
    assert want == [1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0]
    rng = random.Random(3)
    cases = [(i, I.route_switches(sh, i), I.sort_switches(sh, i)) for i in insts]
    cases += [(h1, [rng.randrange(2) for _ in range(sh.route)], I.sort_switches(sh, h1)),
              (h1, I.route_switches(sh, h1), [rng.randrange(2) for _ in range(sh.sort)])]
    fn = bind(I.IoTieRequestIds, B=sh.B, T=sh.T, R=sh.R, OWN=sh.OWN, P=sh.P, S=sh.S, DS=sh.DS)
    args = [I.arguments(sh, i, Q.switches(sh.request, i.request), r, s) for i, r, s in cases]
    out = evaluate_bits(fn, *[_bits([a[k] for a in args]) for k in range(len(args[0]))])
    assert [int(b) for b in np.asarray(out).reshape(-1)] == want + [0, 0]
    memo = {}
    a = lambda f: HC.ands(f, memo)  # noqa: E731
    rq = sh.request
    parts = (a(bind(Q.IoTieRequest, B=sh.B, T=sh.T, R=sh.R, OWN=sh.OWN)) + (rq.N - sh.OWN) * a(bind(I.IoTieIdsSlots, B=B))
             + a(bind(IT.IoTiePort, P=sh.P, S=sh.S, DS=sh.DS)) + a(bind(IT.IoTieRoute, W=sh.CB + sh.D1, D=sh.D1))
             + a(bind(I.IoTieIdsKeys, NK=sh.NK, CB=sh.CB, D1=sh.D1)) + a(bind(IT.IoTieRoute, W=sh.CB + 33, D=sh.D2))
             + a(bind(I.IoTieIdsChain, CB=sh.CB, D2=sh.D2)) + 3)
    assert a(fn) == parts == 12_924_899
