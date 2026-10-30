"""The compile slice with vLLM's block pool (`compile_slice_v2.CompileSlice_v2`): CacheStep_v2's calls wired to
ReplayStep_v1 by equality. On the demonstration slices (`compile_slice_v2.demo`, `demo_async`), where computes evict,
a leaving request frees a block another slot still holds, a request is preempted (AS = 0) and late `cache_blocks`
cache a block (AS = 1), the honest unit is accepted with the pool's own hits, returns and stamps, and each choice the
pool leaves a prover is refused by ``cache`` alone: an overstated ``held``, a made-up eviction, a threshold stamp off,
returns off and a pool row
that isn't one pool's encoding. The circuit is its reference on every case, at G = P and G = 1, and on random windows
of a six-block engine under eviction pressure. The registered book is the arith fixture's async-light stream's."""
from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.core.protocols.certifier import compile_slice as CS, replay_chain as RC
from verity.ml.protocols.certifier import compile_slice_v2 as V2
from verity.core.protocols.certifier import replay_step as RS

FIXTURES = Path(__file__).parent / "fixtures"


def book() -> dict:
    streams = json.loads((FIXTURES / "arith_encode_vectors.json").read_text())["streams"]
    return next(s for s in streams if s["source"].startswith("async-light"))["decisions"]["book"]


BOOK = book()


@pytest.fixture(scope="module")
def synth():
    st, sl = V2.demo()
    return SimpleNamespace(st=st, S=SimpleNamespace(**st), sl=sl)


@pytest.fixture(scope="module")
def asynth():
    st, sl = V2.demo_async()
    return SimpleNamespace(st=st, S=SimpleNamespace(**st), sl=sl)


def _refused(c: CS.Case) -> set[str]:
    return {f for f, ok in c.flags.items() if not ok}


def _calls(x) -> list[list[tuple]]:
    """The honest run's calls by step: (kind, slot, wh, nins, new, tau, held, ret)."""
    sc = V2.scan(x.S, x.sl)
    Sc, St = CS.chain(x.S), CS.step_statics(x.S)
    names = RC.arrival_names(Sc)
    out = []
    for u, (arr, bits) in enumerate(sc.fronts):
        h, e, r, t = sc.wit[u]
        st = CS._state(Sc, sc.states[u])
        inp = RS.Inputs(tuple(RS.Req(**dict(zip(names, a))) for a in arr), tuple(bits), h, e, r)
        got = V2.pool_step(x.S, sc.pools[u], st, inp, RS.reference(St, st, inp), sc.tokens, t)
        assert got.pool == sc.pools[u + 1]
        out.append([(k, o.slot, o.wh, o.nins, o.new, o.tau, o.held, o.ret) for k, o in got.calls])
    return out


def _offset(fields, name: str) -> int:
    return sum(n * w for f, n, w in fields[:[f[0] for f in fields].index(name)])


def _noncanonical(S, c: CS.Case, u: int) -> CS.Case:
    """`c` with pool row u's first dead copy's stamp 1: a pool row that isn't one pool's encoding."""
    at = _offset(V2.fields(S)["wit"], "pool") + u * V2.pool_width(S)
    p = V2.decode_pool(S, c.rows["wit"][at:at + V2.pool_width(S)])
    k = next(x * S.CD + len(e.copies if e else ()) for x, e in enumerate(p.pool.table) if not e or len(e.copies) < S.CD)
    wit = list(c.rows["wit"])
    wit[at + _offset(V2.pool_fields(S), "stamp") + k * S.SW] = 1
    rows = {**c.rows, "wit": wit}
    flags = V2.check(S, rows)
    return replace(c, rows=rows, flags=flags, verdict=CS.verdict(flags))


def _cases(x, overstate, made_up, taus, rets, stale):
    honest = V2.case(x.S, x.sl, BOOK)
    cases = {"honest": honest, "overstate": V2.case(x.S, x.sl, BOOK, overstate=overstate),
             "made_up": V2.case(x.S, x.sl, BOOK, made_up=made_up)}
    for k, v in taus.items():
        cases[f"tau_{k}"] = V2.case(x.S, x.sl, BOOK, taus=v)
    for k, v in rets.items():
        cases[f"ret_{k}"] = V2.case(x.S, x.sl, BOOK, rets=v)
    cases["stale"] = _noncanonical(x.S, honest, stale)
    return cases


def _sync_cases(x):
    return _cases(x, {1: 1}, {4: {0: 1}}, {"later": {2: {0: 2}}, "none": {2: {0: 0}}}, {"less": {3: {1: 0}}}, 2)


def _async_cases(x):
    return _cases(x, {0: 1}, {3: {0: 1}}, {"later": {4: {1: 2}}, "none": {4: {1: 0}}},
                  {"less": {4: {0: 1}}, "more": {4: {0: 3}}}, 4)


def test_the_demo_runs_the_pool(synth):
    """The honest unit is accepted, with every call the docstring's demo names: request 1 looked up at step 0 with a
    1-unit hit, taken at step 1, and freed at once holding 3 blocks of which it returns 2 (slot 0 holds the third);
    both computes of step 2 evict (stamps 1 and 2); request 2 preempted at step 3 (1 block returned) as request 0
    leaves (4); request 2 readmitted at step 4 with a hit on its own queued first block."""
    x = synth
    c = V2.case(x.S, x.sl, BOOK)
    assert c.verdict == V2.ACCEPT == 0x3F
    assert c.steps == (((0, 0, 0, 0, 11),), ((0, 0, 1, 11, 1), (1, 1, 0, 4, 8)), ((0, 0, 1, 12, 1), (1, 2, 0, 0, 4)),
                       ((0, 0, 1, 13, 1),), ((0, 2, 1, 4, 1),))
    assert _calls(x) == [
        [("take", 0, 0, 0, 0, 0, 0, 0), ("compute", 0, 0, 2, 3, 0, 0, 0), ("lookup", 0, 1, 0, 0, 0, 0, 0)],
        [("compute", 0, 0, 1, 0, 0, 0, 0), ("take", 1, 1, 0, 0, 0, 0, 0), ("compute", 1, 0, 2, 2, 0, 0, 0),
         ("free", 1, 0, 0, 0, 0, 3, 2)],
        [("compute", 0, 0, 0, 1, 1, 0, 0), ("take", 1, 0, 0, 0, 0, 0, 0), ("compute", 1, 0, 1, 1, 2, 0, 0)],
        [("compute", 0, 0, 0, 0, 0, 0, 0), ("preempt", 1, 0, 0, 0, 0, 1, 1), ("free", 0, 0, 0, 0, 0, 4, 4)],
        [("take", 0, 1, 0, 0, 0, 0, 0), ("compute", 0, 0, 0, 1, 0, 0, 0), ("free", 0, 0, 0, 0, 0, 2, 2)]]


def test_the_async_demo_runs_the_pool(asynth):
    """At AS = 1 a row's compute caches only its known tokens and the slot's next step completes the block
    (``late``): slot 0's late call at step 4 caches one. Request 2 is looked up at step 3 with a 1-unit hit on request
    0's first block and taken at step 4, its compute evicting (stamp 3), and request 0 leaves holding 3 blocks of
    which it returns 2."""
    x = asynth
    c = V2.case(x.S, x.sl, BOOK)
    assert c.verdict == V2.ACCEPT
    calls = _calls(x)
    assert [k for k, *_ in calls[2]] == ["compute", "compute", "late", "late", "free"]
    assert calls[3] == [("compute", 0, 0, 0, 0, 0, 0, 0), ("lookup", 0, 1, 0, 0, 0, 0, 0), ("late", 0, 0, 0, 0, 0, 0, 0)]
    assert calls[4] == [("compute", 0, 0, 0, 0, 0, 0, 0), ("take", 1, 1, 0, 0, 0, 0, 0), ("compute", 1, 0, 2, 2, 3, 0, 0),
                        ("late", 0, 0, 1, 0, 0, 0, 0), ("free", 0, 0, 0, 0, 0, 3, 2)]


@pytest.mark.parametrize("which", ["synth", "asynth"])
def test_held_and_new_are_replay_steps(request, which):
    """A free of a slot that shares a block, with ``held`` one past ⌈comp / B⌉, returns one block
    more and passes ReplayStep_v1's ``ret`` ≤ held(r) and its free count, so the next allocation skips its eviction.
    The prover's run with that free (its own rows served) is refused by ``cache`` alone, since every free's ``held`` is
    ReplayStep_v1's ⌈comp / B⌉. Two past it returns more than held(r), which ReplayStep_v1 refuses."""
    x = request.getfixturevalue(which)
    rid = 0 if x.S.AS else 1
    c = V2.case(x.S, x.sl, BOOK, overstate={rid: 1})
    assert c.verdict == 0x1F and _refused(c) == {"cache"}
    with pytest.raises(ValueError, match="refuses"):
        V2.case(x.S, x.sl, BOOK, overstate={rid: 2})


@pytest.mark.parametrize("which", ["synth", "asynth"])
def test_the_circuit_is_its_reference(request, which):
    """Each prover choice is refused by ``cache`` alone, and the unit's verdict word is the reference's on every case:
    the honest run, the overstated ``held``, a compute that evicts the oldest queued copy with no eviction due, the
    evicting compute's stamp one later or 0, a victim's or leaver's returns one less (or one more, up to held(r)) and
    a pool row with a dead copy's stamp set."""
    x = request.getfixturevalue(which)
    cases = (_async_cases if x.S.AS else _sync_cases)(x)
    assert {k: _refused(c) for k, c in cases.items()} == {k: set() if k == "honest" else {"cache"} for k in cases}
    assert V2.evaluate(x.st, list(cases.values())) == [c.verdict for c in cases.values()]


def test_the_served_rows_in_blocks(synth):
    """At G = 1 ``srv`` is five rows, one a step; the unit accepts the honest run, refuses the overstated ``held``, and
    costs what it does at G = P."""
    st = {**synth.st, "G": 1}
    S = SimpleNamespace(**st)
    assert V2.row_names(S) == ("pub", "adv", "cert", "srv0", "srv1", "srv2", "srv3", "srv4", "cin", "cout", "wit")
    assert V2.unit(**st).id.endswith(",G=1,CD=2,SW=6}") and V2.unit(**synth.st).id.endswith(",G=5,CD=2,SW=6}")
    cases = [V2.case(S, synth.sl, BOOK), V2.case(S, synth.sl, BOOK, overstate={1: 1})]
    assert [c.verdict for c in cases] == [0x3F, 0x1F]
    assert V2.evaluate(st, cases) == [0x3F, 0x1F]
    assert V2.size(**st)["ands"] == V2.size(**synth.st)["ands"]


def _window(rng: random.Random, AS: int) -> CS.Slice:
    """Three arrivals of 4 to 16 tokens, most sharing a first block, and five random decisions."""
    base = [rng.randrange(1, 15) for _ in range(4)]
    arrivals, tokens = [], []
    for i in range(3):
        P, O = rng.randrange(4, 13), rng.randrange(1, 5)
        pre = base if rng.random() < 0.6 else [rng.randrange(1, 15) for _ in range(4)]
        tokens.append(tuple((pre + [rng.randrange(1, 15) for _ in range(P + O)])[:P + O]))
        arrivals.append((i, P, O, rng.randrange(2)) if AS else (i, P, O))
    return CS.Slice(tuple(arrivals), tuple(rng.randrange(2) for _ in range(5)), 5, tuple(tokens))


@pytest.mark.parametrize("which", ["synth", "asynth"])
def test_random_windows_under_eviction(request, which):
    """On 40 random windows of the demo's class (six blocks: 26 of the synchronous windows and 13 of the asynchronous
    evict), the honest unit is accepted wherever the class holds the window, and each request's free overstated by
    one is refused by ``cache`` alone wherever it changes the rows (16 and 15); the circuit is its reference on all of
    them."""
    x = request.getfixturevalue(which)
    rng = random.Random(18 + x.S.AS)
    cases, refused, outside, evicting = [], 0, 0, 0
    for _ in range(40):
        sl = _window(rng, x.S.AS)
        try:
            honest = V2.case(x.S, sl, BOOK)
        except ValueError:
            outside += 1
            continue
        assert honest.verdict == V2.ACCEPT
        evicting += any(any(w[3]) for w in V2.scan(x.S, sl).wit)
        cases.append(honest)
        for rid in range(3):
            try:
                c = V2.case(x.S, sl, BOOK, overstate={rid: 1})
            except ValueError:
                continue
            if c.rows != honest.rows:
                assert _refused(c) == {"cache"}
                cases.append(c)
                refused += 1
    assert (outside, evicting, refused) == ((0, 26, 16) if not x.S.AS else (0, 13, 15))
    assert V2.evaluate(x.st, cases) == [c.verdict for c in cases]


def test_size(synth, asynth):
    assert V2.size(**synth.st) == {"ands": 83092, "pub_bits": 256, "adv_bits": 16, "cert_bits": 240, "srv_bits": 160,
                                   "cin_bits": 96, "cout_bits": 96, "wit_bits": 2256}
    assert V2.size(**asynth.st)["ands"] == 91785
