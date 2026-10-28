"""The schedule check (`schedule.check`, `Session.scheduled`): a pass's plan is the published schedule's at its step, so
the layout a credited pass reads is the verifier's function of the schedule and moves only with it, and the ledger
credits no appended block without the session's own verdict.  The check refuses the collapse that makes
`SteeredLayout`'s `a` empty (every layout steered through the coins themselves, `a` = their bits): at a fixed
schedule every layout but one is refused, whatever the coins.  And it holds the schedule to the cap on its bits the
session committed before its coins, which is what the ledger credits: a schedule's own bits can follow the coins."""

from __future__ import annotations

import itertools

import pytest

from test_appends import CAP, HONEST_SCHEDULE, N, P512, Run, Verdict, honest_session, salts, store
from test_deferred_map import STEPS, deferred, drive, history
from test_memory_map import page, serving
from verity.core.protocols.poub import audit, memory_map, schedule
from verity.core.protocols.poub.memory_map import Header, Replay
from verity.core.protocols.poub.receipts import Plan
from verity.core.protocols.poub.schedule import Refused, Schedule

COMMON = dict(capacity_bytes_per_s=4096 * 10**9 / 10, chip_bytes=0, rebuild=audit.Rebuild(P512, 0.0))


@pytest.fixture(scope="module")
def honest() -> tuple[Run, list[audit.Verdict]]:
    r = honest_session()
    return r, r.finish()


def passes_of(run: Run) -> list[tuple[int, tuple[tuple[int, int], ...]]]:
    return [(s, p.runs) for w in run.s.windows for s, p in zip(w.steps, w.plans, strict=True)]


def runs(blocks: list[int]) -> tuple[tuple[int, int], ...]:
    """Sorted blocks as a plan's runs, adjacent ones joined."""
    out: list[list[int]] = []
    for j in blocks:
        if out and out[-1][0] + out[-1][1] == j:
            out[-1][1] += 1
        else:
            out.append([j, 1])
    return tuple((f, c) for f, c in out)


def test_the_honest_session_s_plans_are_its_schedule_s_and_the_verdict_carries_the_cap(honest) -> None:
    """Each pass's plan is `HONEST_SCHEDULE.plan` at its step and the appends its `append_runs`; the verdict names the
    session and carries the cap its parameters committed, within which the encoding's 25 bytes are."""
    r, _ = honest
    assert [p for _, p in passes_of(r)] == [HONEST_SCHEDULE.plan(N, s) for s in HONEST_SCHEDULE.steps]
    assert r.s.appended == HONEST_SCHEDULE.append_runs(N) == ((N, 4), (N + 4, 2))
    v = r.s.scheduled(HONEST_SCHEDULE)
    rec = r.s.record()
    assert v.of(rec) and v.cap == r.p.schedule_cap == CAP
    assert HONEST_SCHEDULE.bits == 8 * len(HONEST_SCHEDULE.encode()) == 200 <= CAP


def test_the_cap_committed_before_the_coins_is_credited_and_a_schedule_over_it_is_refused(honest) -> None:
    """The red team's RD-18: the schedule is published after the coins, so its length can
    follow them.  `padded` gives every pass the honest plan, as `HONEST_SCHEDULE` does, at 264 bits (registered block
    7 freed at step 2^40, after every pass), so a prover may publish either on the same plans, as the coins say.
    Credited at its own schedule's bits, its 200-bit sessions would read a `SteeredLayout` bound that holds only at
    the longest schedule's.  So the ledger credits the cap the session committed before its coins, 256, whatever the
    schedule it published within it, and refuses a schedule over it, as the Lean does; a session that committed 264
    takes `padded` and is credited 264."""
    r, _ = honest
    rec = r.s.record()
    padded = Schedule(HONEST_SCHEDULE.registered + ((7, 1, 2**40),), HONEST_SCHEDULE.appends, HONEST_SCHEDULE.steps)
    assert [padded.plan(N, s) for s in padded.steps] == [HONEST_SCHEDULE.plan(N, s) for s in HONEST_SCHEDULE.steps]
    assert (HONEST_SCHEDULE.bits, padded.bits, CAP) == (200, 264, 256)
    v = r.s.scheduled(HONEST_SCHEDULE)
    assert audit.ledger(rec, **COMMON, credit_post_seed=Verdict(True), schedule=v).steering == CAP
    with pytest.raises(Refused) as e:
        r.s.scheduled(padded)
    assert str(e.value) == "the schedule is 264 bits, over the cap of 256 committed before the coins"
    wide = honest_session(schedule_cap=264)
    assert [x.accepted for x in wide.finish()] == [True, True]
    for sc in (HONEST_SCHEDULE, padded):
        assert audit.ledger(wide.s.record(), **COMMON, credit_post_seed=Verdict(True),
                            schedule=wide.s.scheduled(sc)).steering == 264
    assert schedule.check(padded, N, 264, passes_of(wide), wide.s.appended) == 264
    with pytest.raises(Refused, match="the schedule is 200 bits, over the cap of 199 committed before the coins"):
        schedule.check(HONEST_SCHEDULE, N, 199, passes_of(wide), wide.s.appended)


def test_a_session_that_commits_no_cap_credits_no_appended_block(honest) -> None:
    """The default commits no cap (`Params.schedule_cap` None): the session gives no schedule verdict, so the ledger
    credits no appended block, and without a property verdict its ledger is the capped session's, the default
    unchanged."""
    r = honest_session(schedule_cap=None)
    assert [x.accepted for x in r.finish()] == [True, True]
    with pytest.raises(audit.SessionError, match="commit no cap on its schedule's bits"):
        r.s.scheduled(HONEST_SCHEDULE)
    rec = r.s.record()
    with pytest.raises(ValueError, match="schedule check"):
        audit.ledger(rec, **COMMON, credit_post_seed=Verdict(True))
    none = audit.ledger(rec, **COMMON)
    assert none.steering is None and none.to_json() == audit.ledger(honest[0].s.record(), **COMMON).to_json()


def test_at_a_fixed_schedule_every_coin_steered_layout_but_the_schedule_s_is_refused(honest) -> None:
    """The collapse: a prover whose pass at step 3 holds or drops each of pages 10 … 13 as four bits `y` of the coins
    say satisfies `SteeredLayout` at `a` = 4, the bits of `y` itself, as every layout does at `a` = |Y|.  At the
    published schedule exactly one `y` is accepted, the schedule's own plan, and the other 15 are refused: the layout
    the check passes is a function of the schedule alone, so `y` steers it only by changing the schedule, whose bits
    the verdict counts."""
    r, _ = honest
    passes, appends = passes_of(r), list(r.s.appended)
    accepted = []
    for y in itertools.product((False, True), repeat=4):
        held = [0, 1, *range(3, 9), *(j for j, keep in zip(range(10, 14), y) if keep)]
        steered = passes[:3] + [(3, runs(held))] + passes[4:]
        try:
            accepted.append((y, schedule.check(HONEST_SCHEDULE, N, CAP, steered, appends)))
        except Refused as e:
            assert str(e) == "pass 3's plan is not the schedule's at step 3"
    assert accepted == [((True, True, True, True), 200)]


def steered_session(drop: int | None) -> Run:
    """`honest_session` with its pass at step 3 dropping block `drop` from the plan (None: dropping none): a layout the
    session accepts either way, every plan's blocks appended and unreleased."""
    r = Run(store())
    assert r.run() is None
    r.append(4)
    assert r.run(change=(((N, 4),), ())) is None
    assert r.s.release(((2, 1),)) is None and r.s.release(((9, 1),)) is None
    assert r.run(change=((), ((2, 1), (9, 1)))) is None
    r.boundary()
    r.append(2)
    added = tuple((j, 1) for j in (N + 4, N + 5) if j != drop)
    assert r.run(change=(added, ())) is None
    assert r.run(Plan(((0, 2), (3, 6), (10, 4)), r.store.extent)) is None
    r.boundary()
    return r


def test_a_layout_the_session_accepts_is_refused_credit_unless_the_schedule_gives_it() -> None:
    """A prover that drops page 12 from its pass at step 3 is accepted by the session, every window verified, but its
    plan is not the honest schedule's, so the check refuses it and the ledger credits no appended block.  The layout
    moves only with the schedule: the schedule that writes page 12 at step 3, not 2, gives it, at its own bits."""
    r = steered_session(N + 4)
    assert [v.accepted for v in r.finish()] == [True, True]
    with pytest.raises(Refused, match="pass 3's plan is not the schedule's at step 3"):
        r.s.scheduled(HONEST_SCHEDULE)
    rec = r.s.record()
    with pytest.raises(ValueError, match="schedule check"):
        audit.ledger(rec, **COMMON, credit_post_seed=Verdict(True))
    late = Schedule(HONEST_SCHEDULE.registered, (HONEST_SCHEDULE.appends[0], ((3, None), (2, None))),
                    HONEST_SCHEDULE.steps)
    v = r.s.scheduled(late)
    assert late.bits == 200 and audit.ledger(rec, **COMMON, credit_post_seed=Verdict(True), schedule=v).steering == CAP


def test_the_ledger_refuses_a_verdict_on_another_session_or_not_the_check_s(honest) -> None:
    """A verdict names its session by its registration and its coin commitment: another session's (the steered one's
    coins are the same draw, so its registration is changed here), or a look-alike not made by the check, credits
    nothing.  Without a property verdict no schedule is needed."""
    r, _ = honest
    rec = r.s.record()
    v = r.s.scheduled(HONEST_SCHEDULE)
    for bad in (audit.Scheduled(bytes(64), v.coins, v.cap), audit.Scheduled(v.registration, bytes(64), v.cap),
                type("Scheduled", (), {"cap": 0, "of": lambda self, record: True})()):
        with pytest.raises(ValueError, match="schedule check"):
            audit.ledger(rec, **COMMON, credit_post_seed=Verdict(True), schedule=bad)
    assert audit.ledger(rec, **COMMON, schedule=v).steering is None


@pytest.mark.parametrize("sched, why", [
    (Schedule(((2, 1, 1), (2, 1, 3)), HONEST_SCHEDULE.appends, HONEST_SCHEDULE.steps),
     "the registered frees are runs of the registered store, in store order and disjoint, each at a step below 2^64"),
    (Schedule((), ((),), (0,)), "append 0 has no block"),
    (Schedule((), (((4, 4),),), (0,)), "append 0's block 0 is freed at step 4, not after its write at step 4"),
    (Schedule((), (), (0, 0)), "the passes' steps strictly increase below 2^64"),
    (Schedule(HONEST_SCHEDULE.registered, HONEST_SCHEDULE.appends[:1], HONEST_SCHEDULE.steps),
     "the session made 2 appends, the schedule 1"),
    (Schedule(HONEST_SCHEDULE.registered, (HONEST_SCHEDULE.appends[0] + ((0, None),), HONEST_SCHEDULE.appends[1]),
              HONEST_SCHEDULE.steps), "append 0 is (8, 4), the schedule's (8, 5)"),
    (Schedule(HONEST_SCHEDULE.registered, HONEST_SCHEDULE.appends, HONEST_SCHEDULE.steps[:4]),
     "the session has 5 passes, the schedule 4"),
    (Schedule(HONEST_SCHEDULE.registered, HONEST_SCHEDULE.appends, (0, 1, 2, 3, 5)),
     "pass 4 is at step 4, the schedule's at 5"),
], ids=["registered", "empty-append", "free-before-write", "steps", "appends", "append-extent", "passes", "step"])
def test_a_schedule_that_is_not_one_or_not_the_session_s_is_refused(honest, sched, why) -> None:
    with pytest.raises(Refused) as e:
        honest[0].s.scheduled(sched)
    assert str(e.value) == why


def test_the_schedule_of_a_memory_map_gives_the_plans_replay_takes() -> None:
    """`memory_map.schedule` on a map with a registered slot overwritten, a block written over and a freed one, driven
    through a session a pass a step: its plans are the schedule's and the session's verdict holds."""
    st = store()
    head = Header(st.registration.digest, N)
    run, replay, steps = Run(st), Replay(head, st.registration), serving(head)
    assert run.run(change=replay.take(0), step=0) is None
    for k, s in enumerate(steps):
        change = replay.apply(s)
        if change.count:
            a = st.append([page(w.slot, s.step) for w in s.written], salts(b"page salts %d" % s.step, change.count))
            assert run.commit(a.root, a.count) == change.first
        if change.released:
            assert run.s.release(change.released) is None
        assert run.run(change=replay.take(s.step + 1), step=s.step + 1) is None
        if k == 1:
            run.boundary()
    run.boundary()
    passes = [s for w in run.s.windows for s in w.steps]
    sched = memory_map.schedule(head, steps, run.s.appended, passes)
    assert sched.registered == ((4, 2, 0), (6, 1, 3))
    assert sched.appends == (((0, 3), (0, 5), (0, 3)), ((3, None), (3, None)), ((5, 6),))
    assert sched.bits <= CAP and run.s.scheduled(sched).cap == CAP


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_schedule_of_a_deferred_map_with_withdrawals_and_re_keys_is_accepted(seed) -> None:
    """A seeded history's map under deferred residency, withdrawn blocks and re-keys among its lines, drives a session;
    the schedule `memory_map.schedule` makes of it passes the check, each withdrawn block never resident."""
    oracle, encodes, appends, _ = history(seed)
    dm = deferred(oracle, encodes, 0, N)
    run = Run(store(), samples=40, deferred=True, schedule_cap=8 * 1024)
    assert drive(run, dm, appends, STEPS, boundaries=(STEPS // 2,)) is None
    sched = memory_map.schedule(Header(run.store.registration.digest, N), dm, run.s.appended,
                                [s for w in run.s.windows for s in w.steps])
    blocks = {j: b for (first, _), a in zip(run.s.appended, sched.appends) for j, b in enumerate(a, start=first)}
    assert all(blocks[j] is None for s in dm for j in s.withdrawn)
    assert sched.bits <= 8 * 1024 and run.s.scheduled(sched).cap == 8 * 1024
