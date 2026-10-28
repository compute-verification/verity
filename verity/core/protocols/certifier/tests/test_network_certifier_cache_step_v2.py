"""`cache_step_v2`'s CacheStep_v2: the circuit against its specification (`reference`) on random calls at a small size
class (two copies an entry, so duplicates too), the witnesses it derives (the hit, the evictable count, the threshold
stamp, the returns), its refusals, and the difftest vectors: slices of two async traces, every call vLLM's pool made
(`benchmarks/network_e2e/encodings/cache_pool_vectors.py`, which holds the unit to the replay's `_KV`), each with its
refused variants."""
from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.ml.protocols.certifier import cache_step_v2 as C2

SMALL = C2.statics(8, 4, 3, 4, 24, 3, 2, 7, 6)
S = SimpleNamespace(**SMALL)
Z = C2.widths(S)
VECTORS = Path(__file__).parent / "fixtures" / "cache_step_v2_vectors.json"


def _queued(pool):
    return sorted(c.stamp for x in pool.table if x for c in x.copies if not c.mask)


def _threshold(pool, new):
    e = max(new - pool.hf, 0)
    q = _queued(pool)
    return q[e - 1] if 0 < e <= len(q) else 0


def _derived(pool, op):
    """The (wh, ev) pairs the reference accepts for a lookup."""
    return [(wh, ev) for wh in range(Z.KU + 1) for ev in range(Z.KU + 1)
            if C2.reference(S, pool, replace(op, wh=wh, ev=ev)).ok]


def _honest(rng, n, fams):
    """n calls of requests over a few prompt families, so prefixes repeat and the pool fills and evicts: lookups (most
    taking their hit), computes of a slot's next units with the blocks they need, and frees with their returns."""
    pool, calls, held = C2.empty(S), [], {}
    for _ in range(n):
        idle = [s for s in range(S.R) if s not in held]
        k = rng.random()
        if idle and (k < .4 or not held):
            s, r = idle[0], rng.randrange(1 << S.RW)
            cut = rng.randrange(Z.KU + 1) * S.U
            toks = fams[rng.randrange(len(fams))][:cut] + [rng.randrange(2) for _ in range(S.L - cut)]
            K = rng.randrange(1, S.L + 1)
            op = C2.CallInputs(slot=s, r=r, units=tuple(tuple(toks[i * S.U:(i + 1) * S.U]) for i in range(Z.KU)), admit=True,
                       take=rng.random() < .8, K=K)
            (wh, ev), = _derived(pool, op)
            op = replace(op, wh=wh, ev=ev)
            calls.append((pool, op))
            pool = C2.reference(S, pool, op).pool
            if op.take:
                held[s] = [r, toks, K, wh, wh]
        elif held and k < .75:
            s = rng.choice(sorted(held))
            r, toks, K, done, blocks = held[s]
            full = min(K // S.U, Z.KU)
            want = rng.randrange(done, full + 1) if full >= done else done
            need = max(-(-min(K, want * S.U + rng.randrange(S.U)) // S.U) - blocks, 0)
            new = min(need, pool.hf + len(_queued(pool)))
            op = C2.CallInputs(slot=s, r=r, units=tuple(tuple(toks[i * S.U:(i + 1) * S.U]) for i in range(done, Z.KU)),
                       nins=want - done, new=new, tau=_threshold(pool, new))
            out = C2.reference(S, pool, op)
            calls.append((pool, op))
            if out.ok:
                pool, held[s] = out.pool, [r, toks, K, want, blocks + new]
        elif held:
            s = rng.choice(sorted(held))
            _r, _toks, _K, done, blocks = held[s]
            only = sum(1 for x in pool.table if x for c in x.copies if c.mask == 1 << s)
            op = C2.CallInputs(slot=s, free=True, held=max(blocks, done), ret=only + max(blocks, done) - done)
            out = C2.reference(S, pool, op)
            calls.append((pool, op))
            if out.ok:
                pool = out.pool
                del held[s]
    return calls


def _perturbed(rng, op):
    """`op` with one field moved: the hit or the evictable count a step either side, the threshold anywhere, the
    returns or the blocks held a step either side, the blocks taken, the slot or the call's kind."""
    k = rng.randrange(8)
    step = rng.choice((-1, 1))
    if k == 0:
        return replace(op, wh=max(op.wh + step, 0))
    if k == 1:
        return replace(op, ev=max(op.ev + step, 0))
    if k == 2:
        return replace(op, tau=rng.randrange(1 << S.SW))
    if k == 3:
        return replace(op, ret=max(op.ret + step, 0))
    if k == 4:
        return replace(op, held=max(op.held + step, 0))
    if k == 5:
        return replace(op, new=rng.randrange(1 << Z.HW))
    if k == 6:
        return replace(op, slot=rng.randrange(1 << Z.SL))
    return replace(op, admit=rng.random() < .5, take=rng.random() < .5, free=rng.random() < .5)


def _fams(rng, n=3):
    return [[rng.randrange(2) for _ in range(S.L)] for _ in range(n)]


def _calls(seed, n_cases):
    rng = random.Random(seed)
    fams, calls = _fams(rng), []
    while len(calls) < n_cases:
        calls += _honest(rng, rng.randrange(4, 30), fams)
    return rng, fams, calls


def test_the_circuit_is_its_reference_on_random_calls() -> None:
    rng, _, calls = _calls(7, 1000)
    cases = [c for pool, op in calls for c in ((pool, op), (pool, _perturbed(rng, op)))]
    want = [C2.reference(S, p, o) for p, o in cases]
    assert C2.evaluate(SMALL, cases) == want
    ok = [(p, o, w) for (p, o), w in zip(cases, want) if w.ok]
    assert 0.5 < len(ok) / len(cases) < 0.8
    assert sum(1 for p, o, _ in ok if o.new > p.hf) > 40
    assert sum(1 for p, o, _ in ok if o.free and o.ret) > 80
    assert sum(1 for _, o, _ in ok if o.admit and o.take and o.wh) > 60
    assert sum(1 for _, o, _ in ok if o.admit and not o.take and o.wh) > 10
    assert sum(1 for _, _, w in ok if any(x and len(x.copies) == S.D for x in w.pool.table)) > 60
    assert sum(1 for p, _, w in ok if sum(map(bool, w.pool.table)) < sum(map(bool, p.table))) > 15


def test_a_lookup_derives_its_hit_and_evictable_count() -> None:
    """At every pool a random run reaches, a lookup accepts one (wh, ev): the longest cached prefix within its cap,
    and the number of its units' oldest copies that no slot holds."""
    rng, fams, calls = _calls(3, 600)
    hits = seen = 0
    for pool, _ in calls:
        toks = fams[rng.randrange(3)][:rng.randrange(Z.KU + 1) * S.U]
        toks += [rng.randrange(2) for _ in range(S.L - len(toks))]
        op = C2.CallInputs(slot=rng.randrange(S.R), r=1, units=tuple(tuple(toks[i * S.U:(i + 1) * S.U]) for i in range(Z.KU)),
                   admit=True, K=rng.randrange(1, S.L + 1))
        index = {(x.parent, x.tokens): (i, x) for i, x in enumerate(pool.table) if x}
        pfx, wh, ev = S.N, 0, 0
        while (wh + 1) * S.U < op.K and (pfx, op.units[wh]) in index:
            pfx, x = index[(pfx, op.units[wh])]
            wh, ev = wh + 1, ev + (not x.copies[0].mask)
        assert _derived(pool, op) == [(wh, ev)]
        hits, seen = hits + (wh > 0), seen + (ev > 0)
    assert hits > 100 and seen > 30


def test_the_threshold_hint_leaves_no_freedom() -> None:
    """A compute that evicts accepts one threshold stamp, the e-th oldest queued one, and one that doesn't evict
    accepts only 0."""
    _, _, calls = _calls(5, 1000)
    evicting = 0
    for pool, op in calls:
        if op.admit or op.free or not C2.reference(S, pool, op).ok:
            continue
        good = [t for t in range(1 << S.SW) if C2.reference(S, pool, replace(op, tau=t)).ok]
        assert good == [_threshold(pool, op.new)]
        evicting += op.new > pool.hf
    assert evicting > 20


def test_a_free_returns_its_only_held_copies_and_its_hashless_blocks() -> None:
    _, _, calls = _calls(9, 300)
    frees = 0
    for pool, op in calls:
        if not op.free or not C2.reference(S, pool, op).ok:
            continue
        has = sum(1 for x in pool.table if x for c in x.copies if c.mask >> op.slot & 1)
        assert [r for r in range(1 << Z.HW) if C2.reference(S, pool, replace(op, ret=r)).ok] == [op.ret]
        assert not any(C2.reference(S, pool, replace(op, held=h, ret=op.ret - op.held + h)).ok for h in range(has))
        frees += has > 0
    assert frees > 20


def _entry(parent, tokens, level, *copies):
    return C2.Entry(parent, tokens, 3, level, tuple(C2.Copy(m, s) for m, s in copies))


def test_refusals() -> None:
    """Each refusal the docstring lists, beside the same call with the fault removed, which the unit accepts."""
    N, KU, U = S.N, Z.KU, S.U
    u = tuple((k % 3, 1, 2, k % 2) for k in range(KU))
    w = ((7,) * U,) * KU
    e0 = C2.empty(S)

    def chain(n, mask=1, slot_at=None):
        table = tuple(_entry(N if e == 0 else e - 1, u[e], e + 1, (mask, 0)) for e in range(n)) + (None,) * (N - n)
        spfx = (n - 1 if slot_at is None else slot_at,) + (N,) * (S.R - 1) if mask else (N,) * S.R
        return C2.Pool(table, spfx, 3, 0)

    two_q = C2.Pool((_entry(N, u[0], 1, (0, 1)), _entry(0, u[1], 2, (0, 0))) + (None,) * (N - 2), (N,) * S.R, 4, 2)
    full = C2.Pool(chain(KU).table[:KU] + (_entry(N, (5,) * U, 1, (1, 0)), _entry(N, (6,) * U, 1, (1, 0))),
                   (KU - 1,) + (N,) * (S.R - 1), 1, 0)
    holed = replace(full, table=full.table[:-1] + (None,))
    orphan = C2.Pool((_entry(N, u[0], 1, (0, 1)), _entry(0, u[1], 2, (0, 2))) + (None,) * (N - 2), (N,) * S.R, 0, 3)
    look = C2.CallInputs(slot=1, r=1, units=u, admit=True, K=S.L)
    pairs = [  # (what, refused (pool, op), accepted (pool, op))
        ("a take without admit", (e0, C2.CallInputs(take=True)), (e0, C2.CallInputs())),
        ("an admit that frees", (e0, C2.CallInputs(admit=True, free=True, K=1)), (e0, C2.CallInputs(admit=True, K=1))),
        ("new blocks on a lookup", (e0, replace(look, new=1)), (e0, look)),
        ("blocks held off a free", (e0, replace(look, held=1)), (e0, look)),
        ("returns off a free", (e0, replace(look, ret=1)), (e0, look)),
        ("a hit past the window", (e0, replace(look, wh=KU + 1)), (e0, look)),
        ("units cached past the window", (e0, C2.CallInputs(units=u, nins=KU + 1)), (e0, C2.CallInputs(units=u, nins=KU))),
        ("a slot past R", (e0, replace(look, slot=S.R)), (e0, look)),
        ("a take on a slot that holds blocks", (chain(1), replace(look, slot=0, take=True, wh=1)),
         (chain(1), replace(look, take=True, wh=1))),
        ("an entry's copy past D", (C2.Pool((_entry(N, u[0], 1, (1, 0), (2, 0)),) + (None,) * (N - 1), (N,) * S.R, 3, 0),
                                    C2.CallInputs(slot=2, units=u, nins=1, new=1)),
         (chain(1), C2.CallInputs(slot=2, units=u, nins=1, new=1))),
        ("a new entry in a full table", (full, C2.CallInputs(slot=2, units=w, nins=1, new=1)),
         (holed, C2.CallInputs(slot=2, units=w, nins=1, new=1))),
        ("a new entry past level KU", (holed, C2.CallInputs(slot=0, units=w, nins=1, new=1)),
         (replace(holed, spfx=(KU - 2,) + full.spfx[1:]), C2.CallInputs(slot=0, units=w, nins=1, new=1))),
        ("hf past NB − 1", (e0, C2.CallInputs(free=True, held=1, ret=1)),
         (replace(e0, hf=S.NB - 2), C2.CallInputs(free=True, held=1, ret=1))),
        ("a clock past its width", (replace(e0, clock=(1 << S.SW) - KU), C2.CallInputs(free=True)),
         (replace(e0, clock=(1 << S.SW) - KU - 1), C2.CallInputs(free=True))),
        ("an eviction that orphans an entry", (orphan, C2.CallInputs(slot=2, new=1, tau=1)),
         (replace(orphan, table=(_entry(N, u[0], 1, (0, 2)), _entry(0, u[1], 2, (0, 1))) + orphan.table[2:]),
          C2.CallInputs(slot=2, new=1, tau=1))),
        ("more evicted than queued", (two_q, C2.CallInputs(slot=2, new=7, tau=1)),
         (two_q, C2.CallInputs(slot=2, new=6, tau=1))),
        ("a threshold with nothing to evict", (two_q, C2.CallInputs(slot=2, new=4, tau=1)),
         (two_q, C2.CallInputs(slot=2, new=4))),
        ("a hit unit past the lookup's cap", (two_q, replace(look, K=2 * U, wh=2, ev=2)),
         (two_q, replace(look, K=2 * U + 1, wh=2, ev=2))),
        ("a hit shorter than the cached prefix", (two_q, replace(look, wh=1, ev=1)), (two_q, replace(look, wh=2, ev=2))),
        ("a wrong evictable count", (two_q, replace(look, wh=2, ev=1)), (two_q, replace(look, wh=2, ev=2))),
        ("a free holding fewer blocks than its copies", (chain(2), C2.CallInputs(slot=0, free=True, held=1, ret=1)),
         (chain(2), C2.CallInputs(slot=0, free=True, held=2, ret=2))),
    ]
    for what, bad, good in pairs:
        assert not C2.reference(S, *bad).ok, what
        assert C2.reference(S, *good).ok, what
    cases = [c for _, bad, good in pairs for c in (bad, good)]
    assert C2.evaluate(SMALL, cases) == [C2.reference(S, *c) for c in cases]


def test_a_free_stamps_its_copies_deepest_first_after_every_queued_one() -> None:
    """vLLM frees a request's blocks in reverse, to the free queue's back: the stamps of a free's copies are past every
    queued stamp, deepest lowest, and the clock moves by KU."""
    N = S.N
    u = tuple((k % 3, 1, 2, k % 2) for k in range(Z.KU))
    pool = C2.Pool(tuple(_entry(N if e == 0 else e - 1, u[e], e + 1, (1, 0)) for e in range(3)) + (None,) * (N - 3),
                   (2,) + (N,) * (S.R - 1), 3, 10)
    out = C2.reference(S, pool, C2.CallInputs(slot=0, free=True, held=4, ret=4))
    assert [x.copies[0] for x in out.pool.table[:3]] == [C2.Copy(0, 10 + Z.KU - lv) for lv in (1, 2, 3)]
    assert (out.pool.clock, out.pool.hf, out.pool.spfx[0]) == (10 + Z.KU, 4, N)
    evict = C2.reference(S, out.pool, C2.CallInputs(slot=1, new=5, tau=10 + Z.KU - 3))
    assert [x is None for x in evict.pool.table[:3]] == [False, False, True]


def _slices():
    return json.loads(VECTORS.read_text())["slices"]


_FIELDS = ("slot", "r", "admit", "take", "free", "K", "wh", "ev", "nins", "new", "tau", "held", "ret")


@pytest.mark.slow
@pytest.mark.parametrize("name", ["async-prefix-cache.jsonl", "async-crush.json.gz"])
def test_the_difftest_vectors(name) -> None:
    """Each slice's calls, chained from the empty pool, are accepted, each refused variant beside them (a made-up
    eviction, a threshold a copy either side, returns a block either side, a hit or an evictable count a step either
    side, the hit of a cache that never evicts) is refused, and the circuit gives the reference's output on every one."""
    sl = _slices()[name]
    st = sl["statics"]
    T = SimpleNamespace(**st)
    KU = C2.widths(T).KU

    def op(row, u0, known):
        f = {k: (bool(v) if k in ("admit", "take", "free") else v) for k, v in zip(_FIELDS, row)}
        ts = list(sl["tokens"][str(f["r"])][:known][u0 * T.U:(u0 + KU) * T.U]) if known else []
        ts += [0] * (KU * T.U - len(ts))
        return C2.CallInputs(**f, units=tuple(tuple(ts[k * T.U:(k + 1) * T.U]) for k in range(KU)) if known else ())

    pool, cases, refused = C2.empty(T), [], set()
    for k, (kind, row, u0, known, bad) in enumerate(sl["calls"]):
        honest = op(row, u0, known)
        out = C2.reference(T, pool, honest)
        assert out.ok, (k, kind)
        cases.append((pool, honest, out))
        for what, b in bad:
            no = C2.reference(T, pool, op(b, u0, known))
            assert not no.ok, (k, what)
            cases.append((pool, op(b, u0, known), no))
            refused.add(what)
        pool = out.pool
    assert {"a made-up eviction", "the threshold one copy newer", "the threshold past every queued copy",
            "the returns plus one", "the returns less one", "a hit one unit longer"} <= refused
    if name == "async-crush.json.gz":
        assert {"the threshold one copy older", "the hit of a cache that never evicts"} <= refused
    assert sum(1 for p, o, x in cases if x.ok and o.new > p.hf) >= 5
    assert C2.evaluate(st, [(p, o) for p, o, _ in cases]) == [x for _, _, x in cases]


def test_size() -> None:
    """Each pinned class's ANDs (`cache_step_v2.circuit-check.json`, its C-Flock lowering's) within 3% of the
    Definition's, and a window unit's growth in N: a comparison against each entry and a masked write to it,
    2·(U·TW + EW) + 2·RW + HW + D·R ANDs and a few more."""
    pins = json.loads((Path(C2.__file__).with_name("cache_step_v2.circuit-check.json")).read_text())["definitions"]
    assert "CacheStep_v2{" + ",".join(f"{k}={SMALL[k]}" for k in C2.STATICS) + "}" in pins
    for key, pin in pins.items():
        st = {k: int(v) for k, v in (kv.split("=") for kv in key[key.index("{") + 1:-1].split(","))}
        assert pin["and"] <= C2.size(**st)["ands"] <= 1.03 * pin["and"], key
    a, b = (C2.size(**{**SMALL, "N": n})["unit"] for n in (16, 24))
    per = 2 * (S.U * S.TW + C2.widths(SimpleNamespace(**{**SMALL, "N": 16})).EW) + 2 * S.RW + Z.HW + S.D * S.R
    assert per <= (b - a) / 8 <= per + 16
