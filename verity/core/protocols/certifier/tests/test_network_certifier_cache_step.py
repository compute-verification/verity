"""`cache_step`'s CacheStep_v1: the circuit against its specification (`reference`) on random tables and rows, at a small
size class, and on the difftest vectors: two slices of real traces through the derived encoding's `Prefixes`
(`benchmarks/network_e2e/encodings/cache_vectors.py`), whose every call the reference must reproduce."""
from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.core.protocols.certifier import cache_step as CS

SMALL = CS.statics(8, 4, 8, 4, 32)
S = SimpleNamespace(**SMALL)
VECTORS = Path(__file__).parent / "fixtures" / "cache_step_vectors.json"


def _units(rng, n_units, alphabet=3):
    return tuple(tuple(rng.randrange(alphabet) for _ in range(S.U)) for _ in range(n_units))


def _longest(table, row):
    """The derived hit of an admitting row: the wh the reference accepts with `sent` = 0, or None."""
    for wh in range(CS.widths(S).KU + 1):
        if CS.reference(S, table, replace(row, wh=wh, sent=False, nins=0)).ok:
            return wh
    return None


def _honest(rng, table, n):
    """n calls of random requests over a small token alphabet, so prefixes repeat: each admitting row's hit derived or,
    sent, a cached prefix, and the units it computes after; the calls and the table after them."""
    KU = CS.widths(S).KU
    calls, walk = [], {}
    for _ in range(n):
        r = rng.randrange(1 << S.RW)
        toks = [t for u in _units(rng, KU, 2) for t in u]
        K = rng.randrange(1, S.L + 1)
        units = tuple(tuple(toks[k * S.U:(k + 1) * S.U]) for k in range(KU))
        if r not in walk or rng.random() < .5:
            look = CS.Row(r=r, units=units, admit=True, K=K)
            wh = _longest(table, look)
            sent = rng.random() < .3
            if sent:
                wh = rng.randrange((wh + 1) // 2, wh + 1)
            nins = rng.randrange(wh, K // S.U + 1) if K // S.U >= wh else wh
            row = replace(look, sent=sent, wh=wh, nins=nins)
        else:
            row = CS.Row(r=r, units=units, pfx=walk[r], nins=rng.randrange(KU + 1))
        out = CS.reference(S, table, row)
        calls.append((table, row))
        if out.ok:
            table, walk[r] = out.table, out.pfx
    return calls, table


def _perturbed(rng, table, row):
    """`row` with one field moved: the hit a unit either side, sent flipped, K or nins moved, or the start entry."""
    KU = CS.widths(S).KU
    k = rng.randrange(5)
    if k == 0:
        return replace(row, wh=max(row.wh + rng.choice((-1, 1)), 0))
    if k == 1:
        return replace(row, sent=not row.sent)
    if k == 2:
        return replace(row, K=rng.randrange(S.L + 1))
    if k == 3:
        return replace(row, nins=rng.randrange(KU + 2) if KU + 1 < 1 << CS.widths(S).HW else rng.randrange(KU + 1))
    return replace(row, pfx=rng.randrange(len(table) + 1) if table else S.N, admit=False)


def test_the_circuit_is_its_reference_on_random_calls() -> None:
    rng = random.Random(7)
    cases = []
    while len(cases) < 1200:
        calls, _ = _honest(rng, (), rng.randrange(1, 9))
        for table, row in calls:
            cases.append((table, row))
            cases.append((table, _perturbed(rng, table, row)))
    want = [CS.reference(S, t, r) for t, r in cases]
    got = CS.evaluate(SMALL, cases)
    assert got == want
    ok = [w for w, (_, r) in zip(want, cases) if w.ok]
    assert 0.3 < len(ok) / len(cases) < 0.9
    assert sum(1 for w, (_, r) in zip(want, cases) if w.ok and r.admit and r.wh and not r.sent) > 20
    assert sum(1 for w, (_, r) in zip(want, cases) if w.ok and r.admit and r.wh and r.sent) > 10


def test_a_derived_hit_is_exactly_the_longest_cached_prefix() -> None:
    """While the bound holds, a hit a unit shorter or longer than the cached prefix is refused; past it, every cached
    prefix is accepted, and none longer."""
    rng = random.Random(3)
    calls, table = _honest(rng, (), 12)
    seen = 0
    for _ in range(200):
        row = CS.Row(r=1, units=_units(rng, CS.widths(S).KU, 2), admit=True, K=rng.randrange(1, S.L + 1))
        wh = _longest(table, row)
        for x in range(CS.widths(S).KU + 1):
            assert CS.reference(S, table, replace(row, wh=x)).ok == (x == wh)
            assert CS.reference(S, table, replace(row, wh=x, sent=True)).ok == (x <= wh)
        seen += wh > 0
    assert seen > 20


def test_refusals() -> None:
    KU = CS.widths(S).KU
    u = _units(random.Random(1), KU)
    full = tuple(CS.Entry(S.N if e == 0 else e - 1, u[e], 3) for e in range(S.N))
    cases = [
        (full, CS.Row(r=1, units=tuple((9,) * S.U for _ in range(KU)), admit=True, K=S.L, nins=1)),
        ((), CS.Row(r=1, units=u, admit=True, K=S.L, wh=KU + 1, sent=True)),
        ((), CS.Row(r=1, units=u, nins=KU + 1, pfx=S.N)),
        (full[:2], CS.Row(r=1, units=u, admit=True, K=2 * S.U, wh=2, sent=True)),
    ]
    assert not any(CS.reference(S, t, r).ok for t, r in cases)
    assert CS.reference(S, full[:2], replace(cases[3][1], K=2 * S.U + 1)).ok
    assert CS.evaluate(SMALL, cases) == [CS.Out(False)] * len(cases)


def test_a_lower_numbered_request_takes_a_unit_over() -> None:
    """A request that computes a cached unit (past the bound, a hit shorter than the cache) owns it if its number is
    lower, and a later hit's sources name it."""
    u = _units(random.Random(2), CS.widths(S).KU)
    table = (CS.Entry(S.N, u[0], 9), CS.Entry(0, u[1], 9))
    rows = [CS.Row(r=4, units=u, admit=True, sent=True, K=S.L, nins=3),
            CS.Row(r=12, units=u, admit=True, sent=True, K=S.L, nins=3)]
    want = [CS.reference(S, table, r) for r in rows]
    assert [x.owner for x in want[0].table] == [4, 4, 4] and [x.owner for x in want[1].table] == [9, 9, 12]
    look = CS.Row(r=14, units=u, admit=True, K=S.L, wh=3)
    hit = CS.reference(S, want[0].table, look)
    assert hit.ok and CS.sources(S, want[0].table, hit.trail, 3) == ((4, 3 * S.U),)
    cases = [(table, r) for r in rows] + [(want[0].table, look)]
    assert CS.evaluate(SMALL, cases) == want + [hit]


def _slices():
    return json.loads(VECTORS.read_text())["slices"]


@pytest.mark.slow
@pytest.mark.parametrize("name", ["corpus-smallkv", "sync-crush.json.gz"])
def test_the_difftest_vectors(name) -> None:
    """Each slice's calls, chained from the empty table, are what `Prefixes` derived when the vectors were made, and the
    circuit gives the reference's outputs on every lookup, every refusal and every eighth other call."""
    sl = _slices()[name]
    st = sl["statics"]
    T = SimpleNamespace(**st)
    KU = CS.widths(T).KU

    def window(r, u0):
        ts = list(sl["tokens"][str(r)][u0 * T.U:(u0 + KU) * T.U])
        ts += [0] * (KU * T.U - len(ts))
        return tuple(tuple(ts[k * T.U:(k + 1) * T.U]) for k in range(KU))

    table, cases = (), []
    for k, (r, u0, admit, sent, K, wh, nins, pfx, ok, opfx, trail, n) in enumerate(sl["calls"]):
        row = CS.Row(r=r, units=window(r, u0), admit=bool(admit), sent=bool(sent), K=K, wh=wh, nins=nins, pfx=pfx)
        out = CS.reference(T, table, row)
        assert out.ok == bool(ok), k
        if out.ok:
            assert (out.pfx, list(out.trail[:len(trail)]), len(out.table)) == (opfx, trail, n), k
        if admit or not ok or k % 8 == 0:
            cases.append((table, row, out))
        if out.ok:
            table = out.table
    assert sum(1 for _, r, o in cases if r.admit and r.wh and o.ok) >= 1
    assert sum(1 for _, _, o in cases if not o.ok) >= 6
    got = CS.evaluate(st, [(t, r) for t, r, _ in cases])
    assert got == [o for _, _, o in cases]


def test_size() -> None:
    """Each pinned class's ANDs (`cache_step.circuit-check.json`, its Flock lowering's, which merges the ANDs a unit
    repeats on identical operands) within 3% of the Definition's, and a unit's growth in N: a comparison against each
    entry and a masked write to it, 2·(U·TW + EW) + 2·RW ANDs and a few more."""
    pins = json.loads((Path(CS.__file__).with_name("cache_step.circuit-check.json")).read_text())["definitions"]
    assert "CacheStep_v1{" + ",".join(f"{k}={SMALL[k]}" for k in CS.STATICS) + "}" in pins
    for key, pin in pins.items():
        st = {k: int(v) for k, v in (kv.split("=") for kv in key[key.index("{") + 1:-1].split(","))}
        assert pin["and"] <= CS.size(**st)["ands"] <= 1.03 * pin["and"], key
    a, b = (CS.size(**{**SMALL, "N": n})["unit"] for n in (16, 24))
    per = 2 * (S.U * S.TW + CS.widths(SimpleNamespace(**{**SMALL, "N": 16})).EW) + 2 * S.RW
    assert per <= (b - a) / 8 <= per + 8
