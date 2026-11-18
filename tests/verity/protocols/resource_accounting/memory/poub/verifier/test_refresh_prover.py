"""The reference prover under refresh, against the verifier (`AuditRun` under `Params.refresh`): every credited chunk,
registered and appended, re-keyed on the rota with its origin, planned only under a key that joined fewer than L
windows before and after its probation, and opened with its origin's position, salt and path; and with refresh off,
the prover's bytes are as before."""

from __future__ import annotations

import dataclasses
import hashlib

import pytest
from poub_fakes import Clock, Entropy
from test_appends import N, honest_session, pages, salts, store

from verity.protocols.resource_accounting.memory.poub.prover import audit_prover
from verity.protocols.resource_accounting.memory.poub.prover.rota import Rota
from verity.protocols.resource_accounting.memory.poub.verifier import audit, codec, coins, receipts

LIFETIME, WINDOWS, PASSES = 2, 7, 2


def runs_of(indices) -> tuple[tuple[int, int], ...]:
    out: list[list[int]] = []
    for j in sorted(indices):
        if out and out[-1][0] + out[-1][1] == j:
            out[-1][1] += 1
        else:
            out.append([j, 1])
    return tuple((f, c) for f, c in out)


@dataclasses.dataclass
class Session:
    store: audit_prover.Store
    run: audit.AuditRun
    plans: list[tuple[int, receipts.Plan]]
    sent: list[tuple[tuple | None, int, int]]
    verdicts: list[audit.Verdict]


def session(freed: dict[int, list[int]] | None = None, *, tamper=None) -> Session:
    """W windows of L = 2 over the test store, each opened by its due re-keys, a fresh append of pages in windows 0
    and 1, and `PASSES` passes a window over the refresh plan; `freed` names the chunks freed at the start of a window.
    Every append goes to the verifier with its origin and is encoded under the coin it replies with, every release,
    plan and root is received as it is made, and every receipt is sampled and opened, `tamper` rewriting each opening
    before the verifier reads it; the windows are verified until one is rejected."""
    s, rota = store(), Rota(N, LIFETIME, WINDOWS)
    reg = s.registration
    p = audit.Params(blocks=N, deadline_ns=10**6, windows=WINDOWS, samples=10**6, delta=1e-6, window_ns=10,
                     codec=reg.codec, lifetime=LIFETIME, refresh=True)
    clock = Clock()
    run = audit.AuditRun(p, reg, None, coins.local(reg.digest, 0, WINDOWS, Entropy(b"coins")), not_before_ns=0,
                         clock=clock)
    refresh, prover = audit_prover.Refresh(s, rota), audit_prover.Prover(s, None, entropy=Entropy(b"salts"))
    challenges, previous, head, step, passes = [run.start()], reg.digest, None, -1, {}
    out = Session(s, run, [], [], [])

    def send(a: audit_prover.Append, t: int) -> None:
        nonlocal head
        first, coin = run.append(a.root, a.count, a.origin)
        assert first == a.first
        head = audit.append_head(head, first, a.count, a.root)
        if a.origin is not None:
            s.encode(first, coin)
        out.sent.append((a.origin, a.first, a.count))
        if old := refresh.appended(a, t):
            assert run.release(runs_of(old)) is None

    for t in range(WINDOWS):
        for q in (freed or {}).get(t, []):
            assert run.release(runs_of([refresh.freed(q)])) is None
        for origin, count in refresh.due(t):
            send(s.rekey(origin, count), t)
        if t < 2:
            count = 3 if t == 0 else 2
            send(s.append(pages(s.extent, count), salts(b"page salts %d" % s.extent, count)), t)
        for i in range(PASSES):
            ps = prover.run(challenges[t], i, previous, refresh.plan(t), head=head)
            step += 1
            assert run.receive(t, i, ps.root, ps.plan.runs, step=step) is None, (t, i)
            passes[t, i], previous = ps, ps.root
            out.plans.append((t, ps.plan))
        clock.now += p.window_ns
        if (coin := run.boundary()) is not None:
            challenges.append(coin)
    clock.now = run.deadline(WINDOWS - 1) + 1
    for t in range(WINDOWS):
        opened = [(passes[t, i].plan.block(q), prover.open(passes[t, i], q)) for i, q in run.close(t)]
        out.verdicts.append(run.verify(t, [tamper(s, j, o) if tamper else o for j, o in opened]))
        if not out.verdicts[-1].accepted:
            break
    return out


ACCEPTED = [audit.Verdict(t, True) for t in range(WINDOWS)]


def test_a_refresh_session_rekeys_every_chunk_on_the_rota_and_plans_it_only_under_a_live_key():
    """Registered class c (blocks 4c … 4c + 3 at N = 8, L = 2) is re-keyed in the windows ≡ c mod 2 from 1; the pages
    of window 0 (append 0, fresh ordinal 0) in windows 2, 4, 6 and those of window 1 (append 2, after window 1's
    re-key; fresh ordinal 1) in 3, 5.  A page is first planned at its born + L, under its first re-key, and never
    under its own append's key; the verifier accepts every window and records the pages' probation."""
    r = session()
    assert r.verdicts == ACCEPTED
    rekeys = [(o, c) for o, _, c in r.sent if o is not None]
    assert rekeys == [(("registered", 4), 4),
                      (("registered", 0), 4), (("append", 0, 0), 3), (("registered", 4), 4), (("append", 2, 0), 2),
                      (("registered", 0), 4), (("append", 0, 0), 3), (("registered", 4), 4), (("append", 2, 0), 2),
                      (("registered", 0), 4), (("append", 0, 0), 3)]
    assert r.plans[0][1].runs == ((0, N),)
    pages0, pages1 = range(N, N + 3), range(N + 7, N + 9)
    s = r.store
    for t, plan in r.plans:
        origins = {p if (p := s.origin_position(j)) is not None else j for j in plan}
        assert set(range(N)) <= origins, t
        assert (set(pages0) <= origins) == (t >= 2) and (set(pages1) <= origins) == (t >= 3), t
        assert not set(pages0) & set(plan) and not set(pages1) & set(plan)
    assert all(s.blocks[j] is None for j in [*pages0, *pages1])
    record = r.run.record()
    assert record.refresh and sum(sum(w.probation) for w in record.windows) > 0


def test_a_chunk_freed_is_not_rekeyed_and_a_freed_run_cuts_its_piece():
    """Registered blocks 5 and 6 freed before window 3 leave class 1's re-key at 3 as the runs 4 and 7; a page freed
    before window 2 leaves its append's re-key as the other two pages."""
    r = session({2: [N + 1], 3: [5, 6]})
    assert r.verdicts == ACCEPTED
    rekeys = [(o, c) for o, _, c in r.sent if o is not None]
    assert rekeys[2:7] == [(("append", 0, 0), 1), (("append", 0, 2), 1), (("registered", 4), 1),
                           (("registered", 7), 1), (("append", 2, 0), 2)]
    for t, plan in r.plans:
        origins = {p if (p := r.store.origin_position(j)) is not None else j for j in plan}
        assert N + 1 not in origins and (t < 3 or not {5, 6} & origins)


def test_a_rekey_rejects_an_origin_outside_the_store_or_of_a_rekey():
    s = session().store
    with pytest.raises(ValueError, match="never of a re-key"):
        s.rekey(("append", 4, 0), 1)
    with pytest.raises(ValueError, match="never of a re-key"):
        s.rekey(("registered", N - 1), 2)
    with pytest.raises(ValueError, match="1 … 512"):
        s.rekey(("registered", 0), 0)


@pytest.mark.parametrize("tamper", ["origin", "coin"])
def test_the_verifier_refuses_a_rekey_s_opening_at_another_origin_or_under_another_coin(tamper):
    """What this file's sessions are accepted on is not vacuous: the first window with a re-key's block is rejected
    when its openings name another origin position, or carry the block encoded under another coin."""
    def rewrite(s: audit_prover.Store, j: int, o: audit.Opening) -> audit.Opening:
        if o.origin_position is None:
            return o
        if tamper == "origin":
            return dataclasses.replace(o, origin_position=o.origin_position ^ 1)
        return dataclasses.replace(o, block=codec.encode(s.payload(j), j, s.registration.key, bound=True,
                                                         coin=bytes(32)))
    r = session(tamper=rewrite)
    assert [v.accepted for v in r.verdicts] == [True, False]


def test_with_refresh_off_the_reference_prover_s_bytes_are_as_before():
    """`test_appends.honest_session`, which never re-keys under refresh, opened whole: its pass roots and openings
    hash to the digest they had before refresh existed, and no opening names an origin."""
    r = honest_session()
    h = hashlib.sha512()
    for (t, i), ps in sorted(r.passes.items()):
        h.update(ps.root)
        for q in range(ps.plan.length):
            o = r.prover.open(ps, q)
            assert o.origin_position is None
            h.update(repr(tuple(getattr(o, f.name) for f in dataclasses.fields(o) if f.name != "origin_position"))
                     .encode())
    assert all(a.origin is None and a.root == a.tree.commitment.root for a in r.store.appends)
    assert h.hexdigest()[:32] == OFF_DIGEST


OFF_DIGEST = "44f48586406025d6efb522a0729da090"
