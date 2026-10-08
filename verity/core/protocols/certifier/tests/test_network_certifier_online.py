"""The online prover (`online`): each window's declaration reads only the frames ready by its first bucket less the
lag, a snapshot of what it reads chooses the same anywhere, its summary of a past window is the least δ per ρ that
would have met what it saw, and the rules declare what they say."""

import pickle
import random
from fractions import Fraction

import pytest
from verity.core.protocols.certifier import online
from verity.core.protocols.certifier.calibration import choose_syncs
from verity.core.protocols.certifier.schedule import ClockSync, LogicalFrame, SyncSet, handing_buckets

T = 10
SIGMA = SyncSet(tuple(range(6)), (Fraction(1, 2), Fraction(1), Fraction(2)))
ENVELOPE = ClockSync(5, Fraction(1, 2))
RULES = ("last", "cover-3-1", "cover-2-0-fallback", "cover-2-1-rho1", "sticky-2-1-3")


def session_frames(rng: random.Random, anchor: int, horizon: int) -> list[tuple[int, int, int]]:
    """One session's (anchor, index, ready): its first frame 1 to 4 buckets after the anchor, then a frame every 0 to
    2 buckets."""
    out, ready = [], anchor + rng.randint(1, 4)
    for i in range(rng.randint(1, 6)):
        out.append((anchor, i, ready))
        ready += rng.randint(0, 2)
    return [f for f in out if f[2] < horizon]


def instance(rng: random.Random, windows: int = 6, sessions: int = 12) -> list[tuple[int, int, int]]:
    """Sessions anchored at random over the windows, in canonical order (ready, then anchor)."""
    frames = [f for _ in range(sessions) for f in session_frames(rng, rng.randrange(windows * T), (windows + 2) * T)]
    return sorted(frames, key=lambda f: (f[2], f[0], f[1]))


def split(frames: list[tuple[int, int, int]]) -> tuple[list[LogicalFrame], list[int]]:
    return [LogicalFrame(b"", a, i) for a, i, _ in frames], [r for _, _, r in frames]


def declared(frames, rule: str, windows: int = 6, lag: int = 0) -> list[ClockSync]:
    syncs, _, _ = online.declare([split(frames)], SIGMA, T, online.rule(rule), windows, ENVELOPE, lag=lag)
    return syncs


@pytest.mark.parametrize("lag", [0, 3])
@pytest.mark.parametrize("rule", RULES)
def test_no_declaration_reads_a_frame_ready_after_it(rule, lag):
    """Window w's sync is the same however the frames ready after w·T − lag are redrawn: their ready buckets, their
    anchors, how many there are. So no rule looks past its declaration, less the lag."""
    rng = random.Random(1)
    for _ in range(40):
        frames = instance(rng)
        want = declared(frames, rule, lag=lag)
        for w in range(6):
            seen = [f for f in frames if f[2] <= w * T - lag]
            future = sorted((f for _ in range(rng.randint(0, 12))
                             for f in session_frames(rng, rng.randrange(max(0, w * T - lag), 6 * T), 8 * T)),
                            key=lambda f: (f[2], f[0], f[1]))
            future = [(a, i, max(r, w * T - lag + 1)) for a, i, r in future]
            assert declared(seen + future, rule, lag=lag)[: w + 1] == want[: w + 1]


@pytest.mark.parametrize("rule", RULES)
def test_a_snapshot_chooses_what_the_prover_does(rule):
    """At each window, `snapshot()` holds only the frames ready by w·T − lag, and its choice, made after a pickle
    round trip (as in the proxy's worker process), is the prover's own; `declare(sync)` declares a choice made so. The
    lag is from 0 to T − 1 buckets."""
    rng = random.Random(5)
    for lag in (0, 1, 4, T - 1):
        for _ in range(15):
            frames = instance(rng)
            prover = online.Prover(SIGMA, T, online.rule(rule), ENVELOPE, lag=lag)
            for k, (f, r) in enumerate(zip(*split(frames), strict=True)):
                prover.arrive(0, k, f, r)
            for w in range(6):
                snap = pickle.loads(pickle.dumps(prover.snapshot()))
                assert snap.links[0].ready == [r for _, _, r in frames[: len(snap.links[0].ready)]]
                assert all(r <= w * T - lag for r in snap.links[0].ready)
                assert len(snap.links[0].ready) == len(frames) or frames[len(snap.links[0].ready)][2] > w * T - lag
                assert prover.declare(snap.choose()) == declared(frames, rule, lag=lag)[w]
    with pytest.raises(ValueError, match="lag"):
        online.Prover(SIGMA, T, online.rule("last"), ENVELOPE, lag=T)


class Recording:
    """A rule that records what each declaration is given and declares a random value of Σ_sync, so that the windows
    around a summarized one are at values other than its own."""

    def __init__(self, depth: int, seed: int) -> None:
        self.rng, self.given = random.Random(seed), []
        self.rule = online.Rule("recording", depth, self.choose, lambda *_: len(list(SIGMA)))
        self.declared = []

    def choose(self, past, sync_set, envelope, declared=()):
        self.given.append(list(past))
        self.declared.append(tuple(declared))
        return self.rng.choice(list(sync_set))


def test_a_past_window_is_summarized_by_brute_force():
    """Each `Past` is, by brute force over the frames ready at the declaration, the least δ per ρ meeting the window's
    frames with the other windows as declared, their total hand time, and whether its own declaration was late."""
    rng = random.Random(2)
    for seed in range(60):
        frames = instance(rng)
        rec = Recording(3, seed)
        syncs, _, _ = online.declare([split(frames)], SIGMA, T, rec.rule, 6, ENVELOPE)
        for w, past in enumerate(rec.given):
            seen_frames, seen_ready = split([f for f in frames if f[2] <= w * T])
            assert [p.window for p in past] == [u for u in range(max(0, w - 3), w)
                                                if any(f.anchor // T == u for f in seen_frames)]
            for p in past:
                mine = [k for k, f in enumerate(seen_frames) if f.anchor // T == p.window]

                def cost(sync, u=p.window, mine=mine):
                    h = handing_buckets(seen_frames, [sync if v == u else s for v, s in enumerate(syncs[:w])], T)
                    return None if any(seen_ready[k] > h[k] for k in mine) else sum(h[k] for k in mine)

                assert p.sync == syncs[p.window] and p.frames == len(mine)
                assert p.late == (cost(syncs[p.window]) is None)
                for rho, x in zip(sorted(SIGMA.rhos), p.frontier, strict=True):
                    met = [d for d in SIGMA.deltas if cost(ClockSync(d, rho)) is not None]
                    assert x == ((min(met), cost(ClockSync(min(met), rho))) if met else None)


def test_cover_meets_the_recent_windows_with_its_margin():
    """`cover-K-M` declares a value at least M buckets of δ past every one of the last K windows' frontiers at its ρ,
    the envelope only when no value is, and `last` the previous window's least hand time."""
    rng = random.Random(3)
    for _ in range(60):
        frames = instance(rng)
        for name, k, margin in (("cover-3-1", 3, 1), ("cover-2-2", 2, 2), ("last", 1, 0)):
            rec = Recording(k, 0)
            rule = online.rule(name)
            spy = online.Rule(name, k, rec.choose, rule.choices)
            syncs, _, _ = online.declare([split(frames)], SIGMA, T, spy, 6, ENVELOPE)
            for past in rec.given:
                sync = rule.choose(past, SIGMA, ENVELOPE)
                ok = [s for s in SIGMA if past and all(
                    (x := p.frontier[sorted(SIGMA.rhos).index(s.rho)]) is not None and s.delta >= x[0] + margin
                    for p in past)]
                assert (sync in ok) if ok else sync == ENVELOPE
                if name == "last" and ok:
                    met = [i for i, x in enumerate(past[0].frontier) if x is not None]
                    i = min(met, key=lambda i: (past[0].frontier[i][1], past[0].frontier[i][0], -i))
                    assert sync == ClockSync(past[0].frontier[i][0], sorted(SIGMA.rhos)[i])


def test_last_is_the_hindsight_choice_of_a_window_seen_whole():
    """A window whose every frame is ready by the next one's start: `last` declares for the next window what
    `choose_syncs` chooses for it."""
    rng = random.Random(4)
    for _ in range(100):
        frames = [f for _ in range(4) for f in session_frames(rng, rng.randrange(T // 2), T)]
        frames.sort(key=lambda f: (f[2], f[0], f[1]))
        if not frames:
            continue
        want, _, misses = choose_syncs([split(frames)], SIGMA, T)
        if not misses:
            assert declared(frames, "last", 2)[1] == want[0]


def test_the_envelope_without_history_and_after_a_miss():
    """Window 0 has no history, so it declares the envelope. Window 1's one frame is 3 buckets after its anchor, late
    under the 1 bucket window 0 showed: `cover-1-0` then takes δ = 3 for window 2, and `-fallback` the envelope. A
    prover given an earlier run's history starts from it instead."""
    frames = [(0, 0, 1), (10, 0, 13), (20, 0, 21)]
    plain, fallback = declared(frames, "cover-1-0", 3), declared(frames, "cover-1-0-fallback", 3)
    assert plain[:2] == fallback[:2] == [ENVELOPE, ClockSync(1, 2)]
    assert plain[2] == ClockSync(3, 2) and fallback[2] == ENVELOPE
    _, history, _ = online.declare([split(frames)], SIGMA, T, online.rule("cover-1-0"), 3, ENVELOPE)
    syncs, _, _ = online.declare([split([(0, 0, 1)])], SIGMA, T, online.rule("cover-1-0"), 1, ENVELOPE, history)
    assert syncs == [ClockSync(1, 2)]


def test_one_clock_over_several_links():
    """Over two links, a past window is met only where it is met on both, so the declaration covers the slower."""
    fast, slow = [(0, 0, 1), (0, 1, 2)], [(0, 0, 4)]
    syncs, _, _ = online.declare([split(fast), split(slow)], SIGMA, T, online.rule("last"), 2, ENVELOPE)
    assert syncs[1].delta == 4


def test_a_rho_margin_declares_a_slower_rho():
    """`cover-K-M-rhoR` declares cover-K-M's δ at the ρ R steps of Σ_sync slower, or the slowest, and only those ρ
    count among its choices."""
    rng = random.Random(5)
    rhos = sorted(SIGMA.rhos)
    for _ in range(40):
        frames = instance(rng)
        rec = Recording(2, 0)
        online.declare([split(frames)], SIGMA, T, rec.rule, 6, ENVELOPE)
        for past in rec.given:
            plain = online.rule("cover-2-1").choose(past, SIGMA, ENVELOPE)
            slow = online.rule("cover-2-1-rho1").choose(past, SIGMA, ENVELOPE)
            if plain == ENVELOPE:
                assert slow == ENVELOPE
            else:
                assert slow == ClockSync(plain.delta, rhos[max(0, rhos.index(plain.rho) - 1)])
    past = [online.Past(0, ENVELOPE, 1, False, ((0, 1), (0, 1), (0, 1)))]
    assert online.rule("cover-1-1").choices(past, SIGMA, ENVELOPE) == 5 * 3
    assert online.rule("cover-1-1-rho1").choices(past, SIGMA, ENVELOPE) == 5 * 2
    assert online.rule("cover-1-0").choices([], SIGMA, ENVELOPE) == 1


@pytest.mark.parametrize("name, free_rule", [("sticky-2-1-3", "cover-2-1"), ("sticky-2-1-3-rho1", "cover-2-1-rho1")])
def test_sticky_keeps_its_declaration_between_free_windows(name, free_rule):
    """`sticky-K-M-N[-rhoR]` declares cover-K-M[-rhoR] in windows 0, 1, 1+N, ...; in the others it keeps the
    previous value while that has M buckets of δ to spare over the last K windows' frontier at its ρ, and otherwise
    escalates to a value fixed by the previous one alone: the next δ up, or past the largest the next slower ρ. So a
    sender chooses between at most two values there."""
    rng = random.Random(6)
    k, margin, every = 2, 1, 3
    sticky, base = online.rule(name), online.rule(free_rule)
    rhos, top = sorted(SIGMA.rhos), max(SIGMA.deltas)
    kept = escalated = 0
    for _ in range(60):
        frames = instance(rng, windows=9)
        given = []

        def spy(past, sync_set, envelope, declared=(), given=given):
            given.append(list(past))
            return sticky.choose(past, sync_set, envelope, declared)

        syncs, _, choices = online.declare([split(frames)], SIGMA, T, online.Rule("spy", k, spy, sticky.choices), 9,
                                           ENVELOPE)
        assert syncs == online.declare([split(frames)], SIGMA, T, sticky, 9, ENVELOPE)[0]
        for w, (past, sync) in enumerate(zip(given, syncs, strict=True)):
            if w in (0, 1, 4, 7):
                assert sync == base.choose(past, SIGMA, ENVELOPE)
                assert choices[w] == base.choices(past, SIGMA, ENVELOPE)
                continue
            p = syncs[w - 1]
            i = rhos.index(p.rho)
            up = (ClockSync(p.delta + 1, p.rho) if p.delta < top
                  else ClockSync(top, rhos[i - 1]) if i > 0 else p)
            assert choices[w] == len({p, up})
            if all(q.frontier[i] is not None and q.frontier[i][0] + margin <= p.delta for q in past):
                kept += 1
                assert sync == p
            else:
                escalated += 1
                assert sync == up
    assert kept and escalated


def test_rule_names():
    assert online.rule("last").depth == 1 and online.rule("cover-4-1-fallback").name == "cover-4-1-fallback"
    assert online.rule("cover-3-3-rho1").name == "cover-3-3-rho1" and online.rule("sticky-3-2-8").depth == 3
    assert online.rule("sticky-3-2-8-rho1").name == "sticky-3-2-8-rho1"
    for bad in ("cover", "cover-0-1", "cover-2-x", "cover-2-1-later", "first", "sticky-3-2", "sticky-3-2-0",
                "cover-3-2-rho", "cover-3-2-fallback-rho1", "sticky-3-2-8-fallback", "sticky-3-2-rho1",
                "sticky-3-2-8-rho1-fallback"):
        with pytest.raises(ValueError):
            online.rule(bad)
    with pytest.raises(ValueError, match="after its anchor"):
        declared([(5, 0, 5)], "last", 1)
