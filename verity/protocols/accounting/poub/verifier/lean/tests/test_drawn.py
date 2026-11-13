"""`audit.AuditRun` held to the remote audit's verifier in Lean (the definition) on sampled sessions: honest ones of a
sampled shape (windows, passes, appends, releases, plans sent whole or as changes, steps, samples), and recorded ones
(`ra_cases.SCENARIOS`) with one or two edits, each to an event (dropped, repeated, swapped with the next, or one of its
fields changed: a clock reading, an index, a step, a run, a root, an opening's byte or path) or to the case (a
parameter, a coin, the registration). An edit may make an honest session dishonest, a dishonest one fail elsewhere, or the case one that both
sides refuse to decode."""

from __future__ import annotations

import copy
import shutil

import pytest
import ra_cases as M
from hypothesis import strategies as st

import verity_difftest as D
from verity.protocols.accounting.poub.verifier.receipts import Plan

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

server = D.server_fixture(D.Port(M.EXE, "session", M.session, st.nothing(), package=M.PACKAGE, of_record=M.OF_RECORD))
HEX = frozenset("0123456789abcdef")


def paths(x, prefix=()):
    """Every place in a JSON value, with what is there."""
    yield prefix, x
    if isinstance(x, dict):
        for k in sorted(x):
            yield from paths(x[k], prefix + (k,))
    elif isinstance(x, list):
        for n, v in enumerate(x):
            yield from paths(v, prefix + (n,))


def changed(take, v):
    """`v` changed by one sampled edit of its kind."""
    if isinstance(v, bool):
        return not v
    if isinstance(v, int):
        return take(st.sampled_from([v + 1, v - 1, 0, -1, v + 2 ** 64, 2 * v + 1]))
    if isinstance(v, str) and v and len(v) % 2 == 0 and HEX.issuperset(v):
        op = take(st.integers(0, 9))
        if op <= 5:
            i = take(st.integers(0, len(v) - 1))
            return v[:i] + "0123456789abcdef"[(int(v[i], 16) + take(st.integers(1, 15))) % 16] + v[i + 1:]
        return {6: v[:-2], 7: v + "00", 8: "", 9: take(st.sampled_from([v + "0", v.upper()]))}[op]
    if isinstance(v, str):
        return take(st.sampled_from(["", "x", "sketch-sha512", "block-sha512", "p512", "p1024", "verify", "receive"]))
    if v is None:
        return take(st.sampled_from([0, [], "00", [[0, 1]], [[], []]]))
    if isinstance(v, list):
        if not v:
            return take(st.sampled_from([[0], [[0, 1]], ["00"], [[], []]]))
        k, op = take(st.integers(0, len(v) - 1)), take(st.integers(0, 3))
        if op == 0:
            return v[:k] + v[k + 1:]
        if op == 1:
            return v[:k + 1] + [copy.deepcopy(v[k])] + v[k + 1:]
        if op == 2 and k + 1 < len(v):
            return v[:k] + [v[k + 1], v[k]] + v[k + 2:]
        return []
    k = take(st.sampled_from(sorted(v)))
    if take(st.integers(0, 3)):
        return v | {k: changed(take, v[k])}
    return {x: y for x, y in v.items() if x != k}


def edit(take, case: dict) -> None:
    """One edit, in place: four times in five to one event (or the event list), else to the rest of the case."""
    events = case.get("events")
    if isinstance(events, list) and events and take(st.integers(0, 4)):
        k = take(st.integers(0, len(events) - 1))
        places = [(("events",), events)] + [(("events", k) + p, v) for p, v in paths(events[k])]
    else:
        places = [(p, v) for p, v in paths(case) if p and p[0] != "events"]
    path, v = places[take(st.integers(0, len(places) - 1))]
    x = case
    for p in path[:-1]:
        x = x[p]
    x[path[-1]] = changed(take, v)


@st.composite
def edited(take, base: st.SearchStrategy) -> dict:
    case = copy.deepcopy(take(base))
    for _ in range(take(st.integers(1, 2))):
        edit(take, case)
    return case


def runs(blocks) -> tuple:
    return Plan.of(sorted(blocks), max(blocks) + 1).runs if blocks else ()


def removed(plan: Plan, gone: set) -> tuple:
    """`gone` as runs a change removes: each inside one of the plan's runs, as `Plan.changed` takes them."""
    return tuple(r for f, c in sorted(plan.runs) for r in runs(gone & set(range(f, f + c))))


@st.composite
def shaped(take, shared: bool = False) -> dict:
    """An honest session of a sampled shape over the default world's six chunks: at most four pages appended, a page no
    plan held yet released either way or named withdrawn. `shared`: two to four windows, sampled two to four a coin."""
    windows = take(st.integers(2, 4) if shared else st.integers(1, 2))
    per_coin = take(st.integers(2, 4)) if shared else 1
    r = M.Run(windows=windows, samples=take(st.integers(1, 40)), groups_per_coin=per_coin)
    N = r.store.registration.blocks
    live, held, budget = set(range(N)), set(), 4
    r.start()
    for _t in range(windows):
        for _k in range(take(st.integers(1, 3))):
            if budget and take(st.booleans()):
                n = take(st.integers(1, min(2, budget)))
                budget -= n
                a = r.append(n)
                live |= set(range(a.first, a.first + n))
            if len(live) > 1 and take(st.integers(0, 3)) == 0:
                j = take(st.sampled_from(sorted(live)))
                r.release(((j, 1),), ((j, 1),) if j >= N and j not in held and take(st.booleans()) else ())
                live.discard(j)
            blocks = take(st.sets(st.sampled_from(sorted(live)), min_size=1, max_size=8))
            held |= blocks
            step = r.step + take(st.integers(1, 3))
            how = take(st.integers(0, 2))
            if how == 0 and live >= set(range(N)) and blocks == set(range(N)):
                r.run(step=step)
            elif how == 1:
                r.run(change=(runs(blocks - set(r.plan)), removed(r.plan, set(r.plan) - blocks)), step=step)
            else:
                r.run(Plan.of(take(st.permutations(sorted(blocks))), r.store.extent), step=step)
        r.boundary()
    r.finish()
    return r.case()


BASES = ("honest", "honest-appends", "honest-by-root", "honest-rekey", "honest-subset", "honest-block-sha512",
         "drop-appended", "stale-append-coin", "lifetime", "refusals", "release-adjacent", "plan-change-removes-missing",
         "honest-withdraw", "honest-deferred-rekey", "withdraw-planned", "honest-paced", "honest-shared-4",
         "honest-shared-paced-uneven", "honest-paced-one-flight", "after-boundary")


def test_drawn_honest_sessions_agree(server) -> None:
    stats = D.run(D.Port(M.EXE, "session", M.session, shaped(), package=M.PACKAGE, of_record=M.OF_RECORD,
                         need_rejects=False), examples=25, server=server)
    assert stats.outputs == 25, stats


def test_drawn_honest_sessions_drawn_several_windows_a_coin_agree(server) -> None:
    stats = D.run(D.Port(M.EXE, "session", M.session, shaped(shared=True), package=M.PACKAGE, of_record=M.OF_RECORD,
                         need_rejects=False), examples=15, server=server)
    assert stats.outputs == 15, stats


def test_drawn_edits_of_recorded_sessions_agree(server) -> None:
    base = st.sampled_from(BASES).map(lambda name: M.recorded(name)[0])
    stats = D.run(D.Port(M.EXE, "session", M.session, edited(base), package=M.PACKAGE, of_record=M.OF_RECORD),
                  examples=400, server=server)
    assert stats.outputs > 150 and stats.rejects > 40, stats


def test_drawn_edits_of_drawn_sessions_agree(server) -> None:
    stats = D.run(D.Port(M.EXE, "session", M.session, edited(shaped()), package=M.PACKAGE, of_record=M.OF_RECORD),
                  examples=20, server=server)
    assert stats.outputs > 5, stats
