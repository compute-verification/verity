"""`cache_step_v3`'s CacheStep_v3: the circuit against its specification (`Stream`) on random calls at a small class
(prefixes shared, so hits, takes of held and of free blocks, evictions and both kinds of returns), on windows of the
encodings' traces (`benchmarks/mempool/stream.py`, which holds `Stream` to the replay's pool after every call), on
tampered witnesses, and on each refusal a call can meet, built by the non-strict `Stream` so the refused call's
accesses are otherwise consistent."""
from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.protocols.certifier.prover import cache_step_v3 as V3

SMALL = V3.statics(NS=192, NE=96, NB=6, RW=4, KW=3, REFW=3, TW=8)
S = SimpleNamespace(**SMALL)
T = V3.TAGS
VECTORS = Path(__file__).parent / "fixtures" / "cache_step_v3_vectors.json"
GLM = {"NB": 5000, "RW": 20, "KW": 11, "REFW": 9, "TW": 28}


def _scenario(seed: int, S=S, cls=V3.Stream) -> V3.Stream:
    """Requests of 1 to 3 blocks over a few prompt families on a 5-block pool: each looks up its prefix, takes its hit if
    the free queue covers what it needs, allocates the rest (evicting from the queue's head), caches all or all but its
    last block; held requests are freed at random (and before a block could pass REFW holders), returning hashless
    blocks to the front and cached ones to the back. A request that doesn't fit stays at the head, as vLLM's does: it
    looks up again with nothing changed (the memo's case), then after a free (which may change what it read), and
    gives up after a few tries."""
    rng = random.Random(seed)
    fam: dict = {}
    keys, reqs = {}, {}
    for r in range((1 << S.RW) - 1):
        n, f = rng.randint(1, 3), rng.randrange(3)
        share = rng.randint(n // 2, n)
        for k in range(n):
            keys[V3.rk(S, r, k)] = fam.setdefault((f, k), rng.getrandbits(128) | 1) if k < share else \
                rng.getrandbits(128) | 1
        reqs[r] = n
    st = cls(S, V3.empty_pool(S, keys), 1, S.NB - 1)
    nf, active = S.NB - 1, {}
    order = list(reqs)
    rng.shuffle(order)
    for r in order:
        while active and (rng.random() < 0.4 or len(active) >= (1 << S.REFW) - 2):
            q = rng.choice(sorted(active))
            nf += st.call(V3.Call("free", q, active.pop(q)))[0]
        n = reqs[r]
        lim = n - 1 if n > 1 and rng.random() < 0.5 else n
        for tries in range(4):
            h, ev = st.call(V3.Call("hit", r, lim))
            if n - h + ev <= nf or not active:
                break
            if tries % 2:
                q = rng.choice(sorted(active))
                nf += st.call(V3.Call("free", q, active.pop(q)))[0]
        if n - h + ev > nf:
            continue
        if h:
            st.call(V3.Call("take", r, 0, h))
        nf -= ev
        if n - h:
            st.call(V3.Call("alloc", r, h, n - h))
        nf -= n - h
        nc = n - rng.randint(0, 1)
        if nc > h:
            st.call(V3.Call("cache", r, h, nc - h))
        active[r] = n
        if len(st.slots) > 80:
            break
    assert len(st.slots) <= S.NS and len(st.entries()) <= S.NE
    return st


def _blocked(seed: int, cls=V3.Stream) -> V3.Stream:
    """Head-of-line blocking on a 5-block pool. Each step a running request may finish (its free) or grow by a hashless
    block (an alloc, evicting from the queue's head); then the waiting queue's head looks up its prefix and is admitted
    (take, alloc, cache) only if the free queue covers it, else it stays at the head and looks up again next step, as
    vLLM's scheduler does: the memo's case, and where the step changed what its walk read, the dirty bit's."""
    rng = random.Random(seed)
    fam, keys, need = {}, {}, {}
    for r in range((1 << S.RW) - 1):
        need[r], f = rng.randint(2, 4), rng.randrange(2)
        for k in range(need[r]):
            keys[V3.rk(S, r, k)] = fam.setdefault((f, k), rng.getrandbits(128) | 1)
    st = cls(S, V3.empty_pool(S, keys), 1, S.NB - 1)
    nf, held, waiting = S.NB - 1, {}, list(need)
    while waiting and len(st.slots) < 110:
        u = rng.random()
        if held and u < 0.3:
            q = rng.choice(sorted(held))
            nf += st.call(V3.Call("free", q, held.pop(q)))[0]
        elif held and nf and u < 0.6:
            q = rng.choice(sorted(held))
            if held[q] < 1 << S.KW:
                st.call(V3.Call("alloc", q, held[q], 1))
                held[q], nf = held[q] + 1, nf - 1
        r = waiting[0]
        n = need[r]
        h, ev = st.call(V3.Call("hit", r, n - 1))
        if n - h + ev > nf or len(held) >= 4:
            continue
        if h:
            st.call(V3.Call("take", r, 0, h))
        st.call(V3.Call("alloc", r, h, n - h))
        if n - 1 > h:
            st.call(V3.Call("cache", r, h, n - 1 - h))
        nf -= n - h + ev
        held[r] = n
        waiting.pop(0)
    assert len(st.slots) <= S.NS and len(st.entries()) <= S.NE
    return st


class _Blind(V3.Stream):
    """A memo blind to what changed since its walk: no access dirties it."""

    def _dirties(self, tag: int, v: int, w: int) -> int:
        return 0


def _coins(seed: int) -> tuple[int, int]:
    rng = random.Random(100 + seed)
    return rng.getrandbits(128), rng.getrandbits(128)


def _expected(st: V3.Stream, alpha: int, S=S) -> tuple:
    return (1, st.head, st.tail, len(st.records), V3.bus(S, st.records, alpha))


def _args(st, alpha, gamma, slots=None, entries=None, head=1, tail=None, S=S):
    return V3.arguments(S, alpha, gamma, head, S.NB - 1 if tail is None else tail,
                        st.slots if slots is None else slots, st.entries() if entries is None else entries)


def test_the_circuit_is_its_reference_on_random_calls() -> None:
    streams = [_scenario(s) for s in range(8)]
    cases, want = [], []
    for s, st in enumerate(streams):
        al, ga = _coins(s)
        assert st.fingerprints(al, ga)[0] == st.fingerprints(al, ga)[1]
        cases.append(_args(st, al, ga))
        want.append(_expected(st, al))
    assert [tuple(o) for o in V3.evaluate(SMALL, cases)] == want
    kinds = {rec[1] for st in streams for rec in st.records}
    assert kinds == set(V3.KINDS.values())
    evictions = sum(1 for st in streams for x in st.slots if x.tag == T["K"] and x.v and not x.w)
    returns = sum(rec[5] for st in streams for rec in st.records if rec[1] == V3.KINDS["free"])
    held_takes = sum(1 for st in streams for x in st.slots if x.tag == T["B"] and V3.bsplit(S, x.v)["ref"]
                     and V3.bsplit(S, x.w)["ref"] > V3.bsplit(S, x.v)["ref"])
    assert evictions > 5 and returns > 10 and held_takes > 2, (evictions, returns, held_takes)


def test_the_memo_is_its_reference() -> None:
    """Blocked heads that look up again: the circuit is its reference, whose every memoized hit is checked against a
    walk of the memories as they are, and both the memo and its dirty bit's re-walks are reached."""
    streams = [_blocked(s) for s in range(16)]
    cases, want = [], []
    for s, st in enumerate(streams):
        al, ga = _coins(s)
        cases.append(_args(st, al, ga))
        want.append(_expected(st, al))
    assert [tuple(o) for o in V3.evaluate(SMALL, cases)] == want
    memos, rewalks = sum(st.memos for st in streams), sum(st.rewalks for st in streams)
    assert memos > 30 and rewalks >= 2, (memos, rewalks)


def test_the_memo_needs_its_dirty_bit() -> None:
    """Blind to what changed since its walk, the memo returns a stale (hit, evictable) on some of those streams: an alloc
    evicted a block the walk hit, or a free or take changed whether one is held. So the re-walks above are ones the
    circuit's dirty bit must force."""
    stale = 0
    for s in range(16):
        try:
            _blocked(s, cls=_Blind)
        except AssertionError:
            stale += 1
    assert stale >= 2, stale


@pytest.mark.slow
def test_the_circuit_runs_the_traces_windows() -> None:
    """Windows of the encodings' traces, each with evictions, the pool the trace left carried in: the circuit accepts
    each, at its ladder's class, with the stream's queue ends, calls and bus product."""
    data = json.loads(VECTORS.read_text())
    assert len(data["runs"]) == 3
    for run in data["runs"]:
        ws = run["windows"]
        assert len(ws) == 2
        st_ = dict(run["statics"], NS=V3.size_class(max(len(w["slots"]) for w in ws)),
                   NE=V3.size_class(max(len(w["entries"]) for w in ws)))
        S_ = SimpleNamespace(**st_)
        cases, want = [], []
        for j, w in enumerate(ws):
            al, ga = _coins(j)
            slots = [V3.Slot(*x[:5], 0, 0, *x[5:], 0, 0) for x in w["slots"]]
            cases.append(V3.arguments(S_, al, ga, w["head"], w["tail"], slots, [tuple(e) for e in w["entries"]]))
            want.append((1, *w["end"], len(w["records"]), V3.bus(S_, [tuple(r) for r in w["records"]], al)))
        assert [tuple(o) for o in V3.evaluate(st_, cases)] == want, run["run"]


def _tampers(st: V3.Stream) -> dict[str, tuple[list, list]]:
    """The stream's slots and entries with one thing changed."""
    sl, en = list(st.slots), st.entries()
    rng = random.Random(5)
    j = rng.randrange(len(sl))
    starts = [i for i, x in enumerate(sl) if x.start]
    hit = next(i for i in starts if sl[i].kind == V3.KINDS["hit"])
    mid = next(i for i in range(1, len(sl)) if not sl[i].start)
    memo = next(i for i, x in enumerate(sl) if not x.use)
    idle = V3.Slot(*[0] * 11)
    e = rng.randrange(len(en))
    return {
        "a memoized hit's read": (sl[:memo] + [sl[memo]._replace(v=1)] + sl[memo + 1:], en),
        "a read's value": (sl[:j] + [sl[j]._replace(v=sl[j].v ^ 1 << rng.randrange(128))] + sl[j + 1:], en),
        "a read's time": (sl[:j] + [sl[j]._replace(t=sl[j].t ^ 1)] + sl[j + 1:], en),
        "a read at its own now": (sl[:j] + [sl[j]._replace(t=j + 1)] + sl[j + 1:], en),
        "a slot dropped": (sl[:j] + sl[j + 1:], en),
        "the last access cut": (sl[:-1], en),
        "an entry's initial value": (sl, en[:e] + [(*en[e][:2], en[e][2] ^ 1, *en[e][3:])] + en[e + 1:]),
        "an entry dropped": (sl, en[:e] + en[e + 1:]),
        "a call's request at its start": (sl[:hit] + [sl[hit]._replace(r=sl[hit].r ^ 1)] + sl[hit + 1:], en),
        "a field inside a call": (sl[:mid] + [sl[mid]._replace(r=1)] + sl[mid + 1:], en),
        "b given to a hit": (sl[:hit] + [sl[hit]._replace(b=1)] + sl[hit + 1:], en),
        "an idle slot's read": (sl + [idle._replace(v=1)], en),
        "a field of a slot between calls": (sl + [idle._replace(r=2)], en),
    }


def test_a_tampered_witness_is_refused() -> None:
    st = _scenario(3)
    al, ga = _coins(0)
    honest = _expected(st, al)
    tampers = _tampers(st)
    out = V3.evaluate(SMALL, [_args(st, al, ga)] + [_args(st, al, ga, s, e) for s, e in tampers.values()])
    assert tuple(out[0]) == honest
    took = {name: tuple(o) for name, o in zip(tampers, out[1:]) if o.ok}
    assert not took, took


def _run(calls, keys=None, init=None, strict=False) -> V3.Stream:
    base = V3.empty_pool(S, keys or {})
    init = init or {}
    st = V3.Stream(S, lambda tag, a: init.get((tag, a), base(tag, a)), 1, S.NB - 1, strict=strict)
    for c in calls:
        st.call(c)
    return st


K1 = 0x1F2E3D4C5B6A79881F2E3D4C5B6A7988
C = V3.Call
REFUSALS = {
    "take of an uncached key": ([C("take", 0, 0, 1)], {V3.rk(S, 0, 0): K1}, None),
    "take past REFW holders": ([C("alloc", 0, 0, 1), C("cache", 0, 0, 1)] + [C("take", r, 0, 1) for r in range(1, 8)],
                               {V3.rk(S, r, 0): K1 for r in range(8)}, None),
    "alloc for r = 2^RW - 1": ([C("alloc", (1 << S.RW) - 1, 0, 1)], None, None),
    "alloc from an empty queue": ([C("alloc", 0, 0, S.NB - 1), C("alloc", 1, 0, 1)], None, None),
    "alloc of a held head": ([C("alloc", 0, 0, 1)], None, {(T["B"], 1): V3.bput(S, {"next": 2, "ref": 1})}),
    "alloc into a held request slot": ([C("alloc", 0, 0, 1), C("alloc", 0, 0, 1)], None, None),
    "cache of an empty request slot": ([C("cache", 0, 0, 1)], {V3.rk(S, 0, 0): K1}, None),
    "cache of a block with a key": ([C("alloc", 0, 0, 1), C("cache", 0, 0, 1), C("cache", 0, 0, 1)],
                                    {V3.rk(S, 0, 0): K1}, None),
    "take into a held request slot": ([C("alloc", 0, 0, 1), C("cache", 0, 0, 1), C("take", 0, 0, 1)],
                                      {V3.rk(S, 0, 0): K1}, None),
    "free of an empty request slot": ([C("free", 0, 1)], None, None),
    "free of an unheld block": ([C("free", 0, 1)], None, {(T["RB"], V3.rk(S, 0, 0)): 2}),
}


def test_each_refusal_is_refused() -> None:
    """Each refusal's stream is consistent (R = W, every call ended) and runs the circuit's path, so ok = 0 is the
    refusal's check alone (with the slot's checks off, the circuit accepts every one); the same stream without its
    refused call is accepted. The strict `Stream` raises on each."""
    al, ga = _coins(1)
    cases, prefixes = [], []
    for name, (calls, keys, init) in REFUSALS.items():
        st = _run(calls, keys, init)
        assert st.refusals, name
        R, W = st.fingerprints(al, ga)
        assert R == W, name
        cases.append(V3.arguments(S, al, ga, 1, S.NB - 1, st.slots, st.entries()))
        pre = _run(calls[:-1], keys, init)
        assert not pre.refusals, name
        prefixes.append(V3.arguments(S, al, ga, 1, S.NB - 1, pre.slots, pre.entries()))
        with pytest.raises(V3.Refused):
            _run(calls, keys, init, strict=True)
    out = V3.evaluate(SMALL, cases + prefixes)
    assert [o.ok for o in out[:len(cases)]] == [0] * len(cases), dict(zip(REFUSALS, (o.ok for o in out)))
    assert [o.ok for o in out[len(cases):]] == [1] * len(cases)


def test_a_call_out_of_its_ranges_raises() -> None:
    st = V3.Stream(S, V3.empty_pool(S, {}), 1, S.NB - 1, strict=False)
    for c in (C("hit", 0, 1, 1), C("free", 0, 1, 1), C("alloc", 1 << S.RW, 0, 1), C("alloc", 0, 6, 3),
              C("hit", 0, (1 << S.KW) + 1), C("evict", 0, 1)):
        with pytest.raises(V3.Refused):
            st.call(c)
    assert st.call(C("alloc", 0, 0, 0)) == (0, 0) and not st.slots and not st.records


def test_the_ladder() -> None:
    assert [V3.size_class(n, 1) for n in (1, 2, 3, 4, 5, 7, 9, 13, 17, 100, 129, 193)] == \
        [2, 2, 3, 4, 6, 8, 12, 16, 24, 128, 192, 256]
    assert V3.size_class(10) == 64
    assert all(V3.size_class(n) >= n for n in range(1, 5000))


def test_the_slot_price_at_glm_widths() -> None:
    """The measured price `benchmarks/mempool/cost.py` charges a stream slot and a list entry (before the witness
    hash), at GLM-5.3's widths: NB 5,000 blocks, 20-bit requests, 11-bit block indices, 9-bit holders, 28-bit times."""
    assert V3.part_ands(tuple(GLM[k] for k in V3.SLOT_STATICS)) == {"PoolSlot": 16058, "PoolEntry": 10777, "PoolEnd": 129}
