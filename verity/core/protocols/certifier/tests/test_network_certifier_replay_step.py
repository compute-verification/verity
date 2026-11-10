"""replay_step: one vLLM v1 scheduler step as a Boolean Definition, held to the replay (the fixture's steps, as
`benchmarks/network_e2e/encodings/replay_vectors.py --fixture 40` cut them from the replay of a CPU trace, four
corpus traces and two async CPU traces), to its reference on random walks over configurations the corpus doesn't
serve (async scheduling and the walk entries among them), and refusing what its size class or the prefix layer's counts
can't express."""
from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.core.protocols.certifier import replay_step as RS
from verity.core.protocols.certifier.head_check import ands

FIXTURE = Path(__file__).parent / "fixtures" / "replay_step_vectors.json"


def _fixture():
    return [(t["trace"], t["statics"], [RS.from_json(t["statics"], v) for v in t["vectors"]]) for t in json.loads(FIXTURE.read_text())]


def _engine(**kw):
    e = dict(max_num_seqs=256, max_num_batched_tokens=64, max_model_len=128, num_gpu_blocks=24, block_size=16,
             enable_chunked_prefill=True, enable_prefix_caching=False, long_prefill_token_threshold=0, watermark=0.0,
             scheduler_reserve_full_isl=True)
    e.update(kw)
    return e


def test_the_reference_is_the_replay():
    seen = 0
    for name, st, cases in _fixture():
        S = SimpleNamespace(**st)
        for k, (state, inp, want) in enumerate(cases):
            assert RS.reference(S, state, inp) == want, f"{name} vector {k}"
            seen += 1
    assert seen >= 150


def test_the_fixture_has_preemptions_hits_and_admissions():
    kinds = set()
    for _name, st, cases in _fixture():
        for state, inp, want in cases:
            if any(r.id in {x.id for x in state.running} for r in want.state.waiting):
                kinds.add("async preempt" if st["AS"] else "preempt")
            if any(inp.hit):
                kinds.add("hit")
            if want.used > want.taken:
                kinds.add("declined")
            if any(n > 1 and start > 0 for _p, _i, _k, start, n in want.rows):
                kinds.add("chunk")
            if {r.id for r in state.running if r.fin} & {i for _p, i, *_ in want.rows}:
                kinds.add("lag")
    assert kinds == {"preempt", "async preempt", "hit", "declined", "chunk", "lag"}


@pytest.mark.slow
def test_the_circuit_is_the_replay():
    """Four bindings at R=16, W=64, A=16 (about 95k ANDs each), compiled and evaluated once each: about 40 s."""
    by: dict[str, tuple[dict, list]] = {}
    for name, st, cases in _fixture():
        by.setdefault(json.dumps(st, sort_keys=True), (st, []))[1].extend((name, k, c) for k, c in enumerate(cases))
    for st, cases in by.values():
        got = RS.evaluate(st, [(s, i) for _n, _k, (s, i, _w) in cases])
        bad = [(n, k) for g, (n, k, (_s, _i, w)) in zip(got, cases) if g != w]
        assert not bad, f"vectors {bad[:5]}"


def _walk(st: dict, steps: int, rng: random.Random):
    """(state, inputs) pairs along the reference's own runs from the empty engine, with random arrivals, decisions and
    prefix counts (some of them out of range, which the step refuses), restarting after a refusal."""
    S = SimpleNamespace(**st)
    out, state, nxt = [], RS.empty(S), 0
    for _ in range(steps):
        arrivals = tuple(RS.Req(nxt + a, rng.randint(1, 3 * S.B), rng.randint(1, 5), cap=rng.randint(0, 1) if S.AS else 0)
                         for a in range(rng.randint(0, S.A)))
        W, A = S.W, S.A
        hit = tuple(rng.choice((0, 0, 1, 2)) for _ in range(W + A))
        evict = tuple(min(h, rng.choice((0, 1, 3))) if rng.random() < 0.95 else h + 1 for h in hit)
        ret = tuple(rng.choice((0, 1, 2, 4)) for _ in range(S.R))
        pfx = tuple(rng.randrange(1 << S.EW) for _ in range(S.R)) if S.EW else ()
        inp = RS.Inputs(arrivals, tuple(rng.choice((1, 1, 0)) for _ in range(A)), hit, evict, ret, pfx)
        out.append((state, inp))
        got = RS.reference(S, state, inp)
        if got.ok:
            state, nxt = got.state, nxt + got.taken
        else:
            state = RS.empty(S)
            nxt += len(arrivals)
    return out


CONFIGS = [
    dict(R=3, W=4, A=3, engine=_engine()),
    dict(R=3, W=4, A=3, engine=_engine(enable_prefix_caching=True)),
    dict(R=4, W=3, A=2, engine=_engine(max_num_seqs=3, enable_chunked_prefill=False, max_num_batched_tokens=96)),
    dict(R=3, W=4, A=3, engine=_engine(long_prefill_token_threshold=20, watermark=0.1, scheduler_reserve_full_isl=False)),
    dict(R=2, W=2, A=3, engine=_engine(num_gpu_blocks=10, enable_prefix_caching=True, watermark=0.2)),
    dict(R=3, W=4, A=3, engine=_engine(async_scheduling=True)),
    dict(R=3, W=4, A=3, engine=_engine(async_scheduling=True, enable_prefix_caching=True), EW=4),
    dict(R=2, W=2, A=3, engine=_engine(async_scheduling=True, num_gpu_blocks=10, enable_prefix_caching=True, watermark=0.2)),
    dict(R=3, W=4, A=3, engine=_engine(enable_prefix_caching=True), EW=3),
]


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: "R{R}W{W}A{A}".format(**c) + ("async" if c["engine"].get("async_scheduling") else "")
                         + ("EW{EW}".format(**c) if c.get("EW") else ""))
def test_the_circuit_is_the_reference_on_random_walks(cfg):
    st = RS.statics(cfg["R"], cfg["W"], cfg["A"], cfg["engine"], 11, cfg.get("EW", 0))
    S = SimpleNamespace(**st)
    cases = _walk(st, 400, random.Random(repr(cfg)))
    want = [RS.reference(S, s, i) for s, i in cases]
    assert sum(w.ok for w in want) > 200 and any(not w.ok for w in want)
    assert any(w.ok and w.rows for w in want)
    if S.AS:
        lags = [(r, w) for (s, _i), w in zip(cases, want) if w.ok for r in s.running if r.fin]
        assert any(r.cap for r, _w in lags) and any(not r.cap and r.id in {i for _p, i, *_ in w.rows} for r, w in lags)
    if S.EW:
        assert any(r.pfx for (s, _i), w in zip(cases, want) if w.ok for r in w.state.running)
    got = RS.evaluate(st, cases)
    bad = [k for k, (g, w) in enumerate(zip(got, want)) if g != w]
    assert not bad, f"steps {bad[:5]}: {got[bad[0]] if bad else ''} vs {want[bad[0]] if bad else ''}"


def _victims(S, st, inp):
    """The reference's running loop alone: how many requests it preempts, and whether one preempts itself."""
    run, free, budget, i, n_victims = list(st.running), st.free, S.T, 0, 0
    while i < len(run) and budget > 0:
        r = run[i]
        n = min(r.P + r.out - r.comp, S.H or budget, budget, S.L - r.comp - 1)
        if r.fin and r.cap:
            n = 0
        if n <= 0:
            i += 1
            continue
        need = RS.cdiv(r.comp + n, S.B) - RS.cdiv(r.comp, S.B)
        while need > free:
            j = len(run) - 1
            v = run.pop()
            n_victims += 1
            free += inp.ret[j] if S.PC else RS.cdiv(v.comp, S.B)
            if j == i:
                return n_victims, True
        free, budget, i = free - need, budget - n, i + 1
    return n_victims, False


@pytest.mark.parametrize("pc,asy", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_the_circuit_is_the_reference_under_kv_pressure(pc, asy):
    """Running lists a few tokens from their blocks' ends and 0-3 free blocks: the circuit's one comparison a slot (the
    victims a suffix) against the reference's pops, over one, two and three victims and requests preempting themselves;
    under async scheduling, requests that stopped the step before among them (a victim of those leaves, not waits)."""
    st = RS.statics(5, 6, 2, _engine(num_gpu_blocks=14, enable_prefix_caching=bool(pc), async_scheduling=bool(asy)), 7)
    S, rng, cases = SimpleNamespace(**st), random.Random(pc + 2 * asy), []
    for _ in range(1500):
        run = []
        for i in range(rng.randint(0, S.R)):
            O, P = rng.randint(1, 8), rng.randint(1, 60)
            if asy and rng.random() < 0.3:
                run.append(RS.Req(i, P, O, O, P + O - 1, rng.randint(0, 1), 1))
                continue
            out = rng.randint(0, O - 1)
            comp = min(rng.choice((P + out - 1, rng.randint(0, P + out - 1), 16 * rng.randint(1, 4) - rng.randint(0, 1))),
                       P + out - 1)
            run.append(RS.Req(i, P, O, out, comp, rng.randint(0, 1) if asy else 0))
        arr = tuple(RS.Req(80 + a, rng.randint(1, 40), rng.randint(1, 5)) for a in range(rng.randint(0, S.A)))
        cases.append((RS.State(tuple(run), (), rng.randint(0, 3)),
                      RS.Inputs(arr, (1, 0), ret=tuple(rng.choice((0, 1, 2, 3, 5)) for _ in range(S.R)))))
    want = [RS.reference(S, s, i) for s, i in cases]
    shapes = {_victims(S, s, i) for (s, i), w in zip(cases, want) if w.ok}
    assert {(1, False), (2, False), (3, False), (1, True), (2, True)} <= shapes
    assert any(not w.ok for w in want)
    if asy:
        assert any(r.fin and r.id not in {x.id for x in w.state.waiting} and len(w.state.running) < len(s.running)
                   and any(x.id in {y.id for y in s.running} for x in w.state.waiting)
                   for (s, _i), w in zip(cases, want) if w.ok for r in s.running[len(w.state.running):])
    got = RS.evaluate(st, cases)
    bad = [k for k, (g, w) in enumerate(zip(got, want)) if g != w]
    assert not bad, f"cases {bad[:5]}"


def _both(st, state, inp):
    S = SimpleNamespace(**st)
    ref = RS.reference(S, state, inp)
    (got,) = RS.evaluate(st, [(state, inp)])
    assert got == ref
    return ref


def test_refusals():
    st = RS.statics(2, 1, 2, _engine(num_gpu_blocks=12), 7)
    decoding = (RS.Req(0, 16, 9, 1, 16), RS.Req(1, 16, 9, 1, 16))
    # an admission into a full running list (2 slots, max_num_seqs 256)
    assert not _both(st, RS.State(decoding, (), 7), RS.Inputs((RS.Req(2, 8, 2),), (1, 0))).ok
    # declined, the same step is fine: the arrival waits outside
    assert _both(st, RS.State(decoding, (), 7), RS.Inputs((RS.Req(2, 8, 2),), (0, 0))).used == 1
    # still admitting after the A arrivals looked ahead (the next might have arrived)
    wide = RS.statics(4, 1, 2, _engine(num_gpu_blocks=12), 7)
    one = (RS.Req(0, 16, 9, 1, 16),)
    arrivals = (RS.Req(2, 1, 2), RS.Req(3, 1, 2))
    assert not _both(wide, RS.State(one, (), 7), RS.Inputs(arrivals, (1, 1))).ok
    assert _both(wide, RS.State(one, (), 7), RS.Inputs(arrivals, (1, 0))).taken == 1
    # an arrival with no prompt, or longer than max_model_len
    assert not _both(st, RS.empty(SimpleNamespace(**st)), RS.Inputs((RS.Req(0, 0, 2),))).ok
    assert not _both(st, RS.empty(SimpleNamespace(**st)), RS.Inputs((RS.Req(0, 120, 9),))).ok
    # three requests a token past a block each and no free block: the first preempts the last, and the waiting list
    # (one request already) passes W = 1
    at_edge = tuple(RS.Req(i, 32, 9, 1, 32) for i in range(3))
    narrow = RS.statics(3, 1, 1, _engine(num_gpu_blocks=12), 7)
    assert not _both(narrow, RS.State(at_edge, (RS.Req(3, 30, 4),), 0), RS.Inputs(())).ok
    fine = _both(narrow, RS.State(at_edge, (), 0), RS.Inputs(()))
    assert [r.id for r in fine.state.waiting] == [2] and [p for p, *_ in fine.rows] == [0, 1]


def test_prefix_counts_are_bounded():
    st = RS.statics(2, 2, 2, _engine(enable_prefix_caching=True), 7)
    empty = RS.empty(SimpleNamespace(**st))
    a = (RS.Req(0, 40, 3),)
    assert _both(st, empty, RS.Inputs(a, hit=(2, 0, 0, 0), evict=(1, 0, 0, 0))).rows == ((0, 0, 0, 32, 8),)
    assert not _both(st, empty, RS.Inputs(a, hit=(1, 0, 0, 0), evict=(2, 0, 0, 0))).ok      # more evictable than hit
    assert not _both(st, empty, RS.Inputs((RS.Req(0, 32, 3),), hit=(2, 0, 0, 0))).ok        # the hit reaches the last token
    done = RS.State((RS.Req(0, 40, 1, 0, 39),), (), 18)
    assert _both(st, done, RS.Inputs((), ret=(3, 0))).state.free == 21                       # stops, returns 3 of 3
    assert not _both(st, done, RS.Inputs((), ret=(4, 0))).ok                                  # returns more than it holds


def test_async_lag_rows():
    """A request whose last output the step before sampled runs one more decode row, which samples nothing, and leaves
    at the update with the blocks it holds after it; unless its output length is its max_tokens, when it gets no row."""
    st = RS.statics(2, 1, 1, _engine(num_gpu_blocks=12, async_scheduling=True), 7)
    stopping = RS.State((RS.Req(0, 16, 2, 1, 16),), (), 9)
    first = _both(st, stopping, RS.Inputs(()))
    assert first.rows == ((0, 0, 1, 16, 1),) and first.state.running == (RS.Req(0, 16, 2, 2, 17, 0, 1),)
    lag = _both(st, first.state, RS.Inputs(()))
    assert lag.rows == ((0, 0, 1, 17, 1),) and lag.state.running == () and lag.state.free == 10
    capped = _both(st, RS.State((RS.Req(0, 16, 2, 2, 17, 1, 1),), (), 9), RS.Inputs(()))
    assert capped.rows == () and capped.state.running == () and capped.state.free == 11
    sync = RS.statics(2, 1, 1, _engine(num_gpu_blocks=12), 7)
    assert _both(sync, stopping, RS.Inputs(())).state.running == ()


def test_walk_entries_move_with_their_slots():
    """A row's slot takes its call's entry, a running slot without a row keeps its own, and both move with the slot."""
    st = RS.statics(3, 2, 2, _engine(enable_prefix_caching=True), 7, 4)
    # 0 stops, 1 takes the rest of the budget, 2 gets no row
    run = (RS.Req(0, 16, 1, 0, 15, pfx=3), RS.Req(1, 100, 5, 0, 37, pfx=5), RS.Req(2, 40, 5, 2, 41, pfx=6))
    got = _both(st, RS.State(run, (), 20), RS.Inputs((RS.Req(3, 8, 2),), (0, 0), row_pfx=(7, 8, 9)))
    assert [p for p, *_ in got.rows] == [0, 1]
    assert [(r.id, r.pfx) for r in got.state.running] == [(1, 8), (2, 6)]
    # an admitted row's slot takes its second call's entry
    got = _both(st, RS.State((RS.Req(1, 40, 5, 2, 41, pfx=5),), (), 20), RS.Inputs((RS.Req(3, 8, 2),), (1, 0), row_pfx=(7, 9)))
    assert [(r.id, r.pfx) for r in got.state.running] == [(1, 7), (3, 9)]


def test_looked_is_the_lookups_the_scheduler_reads():
    """`looked` marks each waiting slot the loop reaches, the one whose allocation ends it included, and each arrival
    `would_admit` checks, asked or not; none past the loop's end, once the budget is spent or in a step that preempted."""
    st = RS.statics(3, 2, 2, _engine(enable_prefix_caching=True, num_gpu_blocks=12), 7, 4)
    running = (RS.Req(0, 40, 5, 2, 41),)
    waiting = (RS.Req(1, 20, 2), RS.Req(2, 90, 2))
    # waiting 1 is admitted, waiting 2 doesn't fit the 4 free blocks: both read, the arrival not reached
    got = _both(st, RS.State(running, waiting, 4), RS.Inputs((RS.Req(3, 8, 2),), (1, 0)))
    assert got.looked == (1, 1, 0, 0) and got.used == 0
    # an empty waiting list: the first arrival is read and declined, the second isn't reached
    got = _both(st, RS.State(running, (), 9), RS.Inputs((RS.Req(3, 8, 2), RS.Req(4, 8, 2)), (0, 1)))
    assert got.looked == (0, 0, 1, 0) and got.used == 1 and got.taken == 0
    # the running row takes the whole budget: nothing is read
    got = _both(st, RS.State((RS.Req(0, 100, 5, 0, 30),), waiting, 9), RS.Inputs((RS.Req(3, 8, 2),), (1, 0)))
    assert got.looked == (0, 0, 0, 0)
    # a step that preempted runs no waiting loop
    edge = tuple(RS.Req(i, 32, 9, 1, 32) for i in range(2))
    got = _both(st, RS.State(edge, (RS.Req(5, 8, 2),), 0), RS.Inputs((), ret=(2, 2, 0)))
    assert [r.id for r in got.state.waiting][:1] == [1] and got.looked == (0, 0, 0, 0)
    assert RS.statics(3, 2, 2, _engine(), 7)["EW"] == 0 and _both(RS.statics(3, 2, 2, _engine(), 7), RS.State(running, (), 9),
                                                                   RS.Inputs((RS.Req(3, 8, 2),), (0, 0))).looked == ()


def test_size_counts_the_gates():
    for cfg in CONFIGS[:2] + CONFIGS[6:7]:
        st = RS.statics(cfg["R"], cfg["W"], cfg["A"], cfg["engine"], 7, cfg.get("EW", 0))
        assert RS.size(**st)["ands"] == ands(RS.unit(**st))
