"""The certified tick's step (`advice_code.tick_step.reference`) against the replay that runs the rule
(`advice_code.replay.replay(tick=...)`).

Scanned from the empty engine at T = 0 over the certified arrivals and their release buckets, the step gives the
replay's rows, its scheduler's state before every schedule and its clock after every step with rows
(`Replay.clock`), under sync and async scheduling, on a plan that idles, saturates, preempts and hits the prefix
cache. The iterations of the replay's loop that only free finished requests change nothing and are no step of the
scan, and the replay's queue of skipped requests is empty after every schedule. Small engines hold what the plan
doesn't reach: a step saturated by its waiting list alone, one saturated by a preempted request that neither of its
states shows, and a request that takes no token and so preempts nothing. A window of the scan that opens on a drained
engine needs the clock the window before it left. And the step refuses a class that can't see the first arrival it
doesn't take.
"""
from __future__ import annotations

import dataclasses
import random
from types import SimpleNamespace

import pytest
from verity.protocols.certifier.verifier import replay_step as RS

from verity_vllm.pipeline.advice_code import pool as PL
from verity_vllm.pipeline.advice_code import replay as RP
from verity_vllm.pipeline.advice_code import schedule as SC
from verity_vllm.pipeline.advice_code import tick_step as TS

TICK = SC.Tick(bucket_us=100_000, sat_us=30_000, free_us=40_000)
EOS = 2
N = 36
BOTH = pytest.mark.parametrize("asy", [False, True], ids=["sync", "async"])


def engine(asy: bool) -> SC.Engine:
    """A small engine that chunks prefills, preempts and caches prefixes."""
    return SC.Engine(max_num_seqs=4, max_num_batched_tokens=64, max_model_len=256, num_gpu_blocks=8, block_size=16,
                     enable_prefix_caching=True, async_scheduling=asy)


def plan(seed: int = 7) -> tuple[list[SC.RequestIO], list[int]]:
    """The release sequence `benchmarks/network_e2e/encodings/tests/test_tick.py` serves behind the gate: N requests
    released in bursts with idle gaps between, each prompt a 32-token prefix shared with every third request then its
    own tail, each output ending on EOS or running to max_tokens; and each one's release bucket."""
    rng = random.Random(seed)
    t, ios, released = 0.05, [], []
    for i in range(N):
        t += rng.choice((0.0, 0.0, 0.004, 0.02, 0.06, 0.4, 1.2))
        released.append(int(round(t, 4) * 1e6) // TICK.bucket_us)
        prompt = (100 + i % 3,) * 32 + tuple(1000 + 64 * i + j for j in range(rng.randint(1, 30)))
        n = rng.randint(1, 10)
        eos = rng.random() < 0.7
        ios.append(SC.RequestIO(prompt_len=len(prompt), max_tokens=n + 3 if eos else n, output_len=n, eos=eos,
                                tokens=prompt + (3,) * (n - 1) + ((EOS,) if eos else (3,))))
    return ios, released


def statics(e: SC.Engine, n: int, R: int, W: int, A: int) -> SimpleNamespace:
    return SimpleNamespace(**RS.statics(R, W, A, dataclasses.asdict(e), max(n - 1, 1).bit_length()))


def arrival(i: int, io: SC.RequestIO, asy: bool) -> RS.Req:
    return RS.Req(i, io.prompt_len, io.output_len, cap=int(asy and io.output_len >= io.max_tokens))


@dataclasses.dataclass
class Unit:
    """One `schedule()` of the replay with the update that follows it (sync: its own step's; async: the step
    before's), as the step sees it."""

    idle: bool                              # nothing running or waiting, and the update before finished a request
    running: tuple[RS.Req, ...]             # before the schedule
    queue: tuple[RS.Req, ...]               # vLLM's waiting queue before the schedule, the arrivals added included
    free: int
    added: int                              # arrivals added so far
    slots: list[int] = dataclasses.field(default_factory=list)                  # the running list after the schedule
    rows: list[SC.Row] = dataclasses.field(default_factory=list)
    preempted: bool = False
    hits: dict[int, tuple[int, int]] = dataclasses.field(default_factory=dict)  # request -> hit blocks, evictable
    frees: dict[int, int] = dataclasses.field(default_factory=dict)             # request -> blocks returned


def record(monkeypatch: pytest.MonkeyPatch, e: SC.Engine, ios: list[SC.RequestIO],
           released: list[int]) -> tuple[RP.Replay, list[Unit]]:
    """The replay under the tick, and what each of its schedules saw. Asserts the skipped queue is empty after each."""
    asy = e.async_scheduling
    units: list[Unit] = []
    added = [0]
    add, schedule, hit, free = RP.Scheduler.add, RP.Scheduler.schedule, PL.BlockPool.hit, PL.BlockPool.free_request

    def cap(r: SC.RequestState) -> int:
        return int(asy and r.io.output_len >= r.io.max_tokens)

    def counted_add(self: RP.Scheduler, i: int) -> None:
        added[0] += 1
        add(self, i)

    def recorded_schedule(self: RP.Scheduler, on_open: None = None) -> tuple[SC.Step, dict[int, bool]]:
        u = Unit(self.quiet() and bool(self.finished_ids),
                 tuple(RS.Req(r.i, r.P, r.io.output_len, r.n_out + r.placeholders, r.num_computed, cap(r),
                              int(asy and r.n_out + r.placeholders == r.io.output_len)) for r in self.running),
                 tuple(RS.Req(r.i, r.P, r.io.output_len, r.n_out, 0, cap(r)) for r in self.waiting),
                 len(self.kv.free), added[0])
        units.append(u)
        step, samples = schedule(self, on_open)
        assert not self.skipped
        u.slots, u.rows, u.preempted = [r.i for r in self.running], list(step.rows), bool(step.preempted)
        return step, samples

    def recorded_hit(self: PL.BlockPool, r: SC.RequestState) -> list[int]:
        out = hit(self, r)
        units[-1].hits[r.i] = (len(out), sum(1 for b in out if self.ref[b] == 0))
        return out

    def recorded_free(self: PL.BlockPool, r: SC.RequestState) -> None:
        back = sum(1 for b in self.req_blocks.get(r.i, ()) if self.ref[b] == 1)
        units[-1].frees[r.i] = units[-1].frees.get(r.i, 0) + back
        free(self, r)

    with monkeypatch.context() as m:
        m.setattr(RP.Scheduler, "add", counted_add)
        m.setattr(RP.Scheduler, "schedule", recorded_schedule)
        m.setattr(PL.BlockPool, "hit", recorded_hit)
        m.setattr(PL.BlockPool, "free_request", recorded_free)
        rp = RP.replay(e, ios, tick=TICK, released=released)
    return rp, units


def inputs(S: SimpleNamespace, st: RS.State, cursor: int, ios: list[SC.RequestIO], released: list[int],
           u: Unit | None = None) -> TS.Inputs:
    """The step's inputs at `cursor`: the next A arrivals with their buckets and, from the replay's unit, the prefix
    counts by `ReplayStep_v1`'s candidates (the waiting slots, or the arrival a quiet engine takes; then the arrivals)
    and slots (a leaving request's in the running list after the schedule, a preempted one's before it)."""
    view = range(cursor, min(cursor + S.A, len(ios)))
    arrivals = tuple(arrival(i, ios[i], bool(S.AS)) for i in view)
    buckets = tuple(released[i] for i in view)
    if u is None:
        return TS.Inputs(arrivals, buckets)
    forced = int(not st.running and not st.waiting and cursor < len(ios))
    waiting = [r.id for r in st.waiting] or [cursor] * forced
    hit, evict, ret = [0] * (S.W + S.A), [0] * (S.W + S.A), [0] * S.R
    for i, (h, ev) in u.hits.items():
        c = waiting.index(i) if i in waiting else S.W + i - cursor - forced
        assert 0 <= c < S.W + S.A
        hit[c], evict[c] = h, ev
    before = [r.id for r in st.running]
    for i, back in u.frees.items():
        ret[u.slots.index(i) if i in u.slots else before.index(i)] = back
    return TS.Inputs(arrivals, buckets, tuple(hit), tuple(evict), tuple(ret))


Step = tuple[TS.State, int, TS.Out]         # a step's state and cursor before it, and its outputs


def scan(S: SimpleNamespace, ios: list[SC.RequestIO], released: list[int], st: TS.State, cursor: int = 0) -> list[Step]:
    """The step from `st` and `cursor` until the arrivals are taken and the engine is drained, on an engine with no
    prefix cache."""
    assert not S.PC
    out: list[Step] = []
    while cursor < len(ios) or st.step.running or st.step.waiting:
        assert len(out) < 1000
        got = TS.reference(S, TICK, st, inputs(S, st.step, cursor, ios, released))
        assert got.ok and got.state is not None
        out.append((st, cursor, got))
        st, cursor = got.state, cursor + got.taken
    return out


def advance(step: Step) -> int:
    st, _, out = step
    assert out.state is not None
    return out.state.clock - st.clock


def small(asy: bool, blocks: int, requests: list[tuple[int, int, bool]],
          released: list[int]) -> tuple[SimpleNamespace, list[SC.RequestIO], list[Step]]:
    """Requests (prompt length, output length, whether it ends on EOS, else at max_tokens) on a two-slot engine of
    `blocks` 16-token blocks with no prefix cache: the class, the requests and the scan from the empty engine, held
    to the replay's rows and its clock."""
    e = SC.Engine(max_num_seqs=2, max_num_batched_tokens=64, max_model_len=64, num_gpu_blocks=blocks + 1, block_size=16,
                  async_scheduling=asy)
    ios = [SC.RequestIO(prompt_len=p, max_tokens=o + 4 if eos else o, output_len=o, eos=eos) for p, o, eos in requests]
    S = statics(e, len(ios), R=2, W=2, A=3)
    rp = RP.replay(e, ios, tick=TICK, released=released)
    steps = scan(S, ios, released, TS.empty(S))
    with_rows = [o for _, _, o in steps if o.rows]
    assert [[(i, start, n) for _, i, _, start, n in o.rows] for o in with_rows] == [s.rows for s in rp.steps]
    assert [o.state.clock for o in with_rows if o.state is not None] == rp.clock
    return S, ios, steps


@BOTH
def test_the_scan_is_the_replay(asy: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    e = engine(asy)
    ios, released = plan()
    rp, units = record(monkeypatch, e, ios, released)
    S = statics(e, N, R=4, W=4, A=5)
    for k, u in enumerate(units):
        if u.idle:
            assert not (u.running or u.queue or u.rows or u.hits or u.frees)
            assert k + 1 == len(units) or (units[k + 1].running, units[k + 1].free) == (u.running, u.free)
    st, cursor = TS.empty(S), 0
    clocks: list[int] = []
    seen = {"frees only": any(u.idle for u in units)}
    for u in (u for u in units if not u.idle):
        quiet = not st.step.running and not st.step.waiting
        assert (st.step.running, st.step.free) == (u.running, u.free)
        assert u.queue[:len(st.step.waiting)] == st.step.waiting
        assert [r.id for r in u.queue[len(st.step.waiting):]] == list(range(cursor, u.added))
        assert TS._preempts(S, st.step) == u.preempted
        out = TS.reference(S, TICK, st, inputs(S, st.step, cursor, ios, released, u))
        assert out.ok and out.state is not None
        assert [(i, start, n) for _, i, _, start, n in out.rows] == u.rows
        assert [(slot, kind) for slot, _, kind, _, _ in out.rows] == [
            (u.slots.index(i), int(n == 1 and start >= ios[i].prompt_len)) for i, start, n in u.rows]
        if out.rows:
            clocks.append(out.state.clock)
        took = 0 if quiet else out.state.clock - st.clock
        for name, here in (("idles", quiet and cursor > 0), ("preempts", u.preempted), ("no row", not out.rows),
                           ("hits", any(h for h, _ in u.hits.values())),
                           ("evicts", any(ev for _, ev in u.hits.values())),
                           ("saturates", took == TICK.sat_us), ("runs free", took == TICK.free_us)):
            seen[name] = seen.get(name, False) or here
        st, cursor = out.state, cursor + out.taken
    assert (st.step, cursor) == (RS.empty(S), N)
    assert clocks == rp.clock and len(clocks) == len(rp.steps)
    assert seen == {"frees only": True, "idles": True, "preempts": True, "no row": asy, "hits": True, "evicts": True,
                    "saturates": True, "runs free": True}


def test_a_request_left_waiting_saturates_the_step() -> None:
    """Sync. Request 0's 17th token needs the block request 1 holds, so step 2 preempts request 1. It then waits while
    request 0 runs, with no arrival left: steps 3 to 9 preempt nothing and take nothing, and the waiting list alone
    saturates them. Step 10 takes it back and runs free."""
    S, _, steps = small(False, 2, [(15, 10, True), (10, 8, True)], [0, 0])
    assert [TS._preempts(S, st.step) for st, _, _ in steps[:4]] == [False, False, True, False]
    for step in steps[3:10]:
        st, cursor, out = step
        assert [r.id for r in st.step.waiting] == [1] and cursor == 2 and not out.taken
        assert not TS._preempts(S, st.step) and advance(step) == TICK.sat_us
    assert [i for _, i, _, _, _ in steps[10][2].rows] == [1] and advance(steps[10]) == TICK.free_us


def test_a_preempted_request_about_to_leave_saturates_the_step() -> None:
    """Async. Request 1's last output is in flight when request 0 needs the block it holds. Step 2 preempts request 1,
    which vLLM's waiting queue then holds, so T advances by τ_sat. The step's update finishes it: the next state's
    waiting list is empty and its running list has lost request 1 as it loses one that finished."""
    S, _, steps = small(True, 2, [(15, 4, True), (10, 2, True)], [0, 0])
    st, cursor, out = steps[2]
    assert out.state is not None and cursor == 2 and not out.taken
    assert [(r.id, r.fin) for r in st.step.running] == [(0, 0), (1, 1)] and not st.step.waiting
    assert [r.id for r in out.state.step.running] == [0] and not out.state.step.waiting
    assert TS._preempts(S, st.step) and advance(steps[2]) == TICK.sat_us


def test_a_request_at_its_max_tokens_preempts_nothing() -> None:
    """Async. Request 0 runs to its max_tokens. At step 7 its last output is in flight, it has computed 16 tokens and
    no block is free: the scheduler gives it no token, so it preempts nothing and the step runs free."""
    S, _, steps = small(True, 2, [(10, 7, False), (5, 9, True)], [0, 0])
    st = steps[7][0]
    assert [(r.id, r.comp, r.fin, r.cap) for r in st.step.running[:1]] == [(0, 16, 1, 1)] and not st.step.free
    assert not TS._preempts(S, st.step) and advance(steps[7]) == TICK.free_us


def test_the_preemption_test_is_the_references() -> None:
    """`tick_step._preempts` against `replay_step.reference` on random running lists, budgets, block sizes, length
    bounds and prefill thresholds. The reference says whether it preempted only through the next waiting list: it
    requeues the last running request first, unless that request's last output is in flight (`fin`). `fin` changes
    a token count only with `cap`, so with it cleared elsewhere the list shows every preemption, except where the
    last running request is `fin` and `cap`, and there a requeued request still shows one."""
    rng = random.Random(0)
    seen = {False: 0, True: 0}
    for _ in range(3000):
        asy = rng.random() < 0.5
        e = SC.Engine(max_num_seqs=4, max_num_batched_tokens=rng.choice((6, 64)), max_model_len=rng.choice((40, 256)),
                      num_gpu_blocks=64, block_size=rng.choice((4, 16)), async_scheduling=asy,
                      long_prefill_token_threshold=rng.choice((0, 3)))
        S = statics(e, 4, R=4, W=4, A=5)
        running = []
        for i in range(rng.randint(1, 4)):
            P, out = rng.randint(1, 44), rng.randint(0, 3)
            running.append(RS.Req(i, P, out + rng.randint(1, 3), out, rng.randint(0, P + out - 1),
                                  cap=int(asy and rng.random() < 0.5), fin=int(asy and rng.random() < 0.3)))
        st = RS.State(tuple(running), (), rng.randint(0, 2))
        shown = RS.State(tuple(dataclasses.replace(r, fin=r.fin & r.cap) for r in running), (), st.free)
        out = RS.reference(S, shown, RS.Inputs(()))
        assert out.ok and out.state is not None
        requeued = bool(out.state.waiting)
        if requeued or not (running[-1].fin and running[-1].cap):
            assert TS._preempts(S, st) == requeued
            seen[requeued] += 1
    assert min(seen.values()) > 500


@BOTH
def test_a_window_carries_the_clock(asy: bool) -> None:
    """Request 0's last step takes T from 180,000 to 220,000, past bucket 1's close at 200,000, and leaves the engine
    drained. The window that opens there with T = 220,000 gives the replay's steps. One that restarts T moves it to
    200,000, the close of request 1's bucket, so T reaches bucket 2's close a step later and request 2 is admitted a
    step late."""
    released = [0, 1, 2]
    S, ios, steps = small(asy, 15, [(4, 2 if asy else 3, True), (4, 6, True), (4, 2, True)], released)
    k = next(k for k, (st, cursor, _) in enumerate(steps) if cursor == 1 and st.step == RS.empty(S))
    opening = steps[k][0]
    assert opening.clock == 220_000 > TICK.close(released[1]) == 200_000
    assert scan(S, ios, released, opening, 1) == steps[k:]
    restarted = scan(S, ios, released, TS.empty(S), 1)
    assert advance(restarted[0]) == TICK.close(released[1]) + TICK.free_us

    def admitted(window: list[Step]) -> int:
        return next(j for j, (_, _, o) in enumerate(window) if any(i == 2 for _, i, _, _, _ in o.rows))

    assert admitted(restarted) == admitted(steps[k:]) + 1


def test_the_step_refuses_a_class_that_cannot_see_the_first_arrival_it_leaves() -> None:
    e = engine(False)
    ios, released = plan()
    S = statics(e, N, R=4, W=4, A=4)
    with pytest.raises(ValueError, match=r"A ≥ min\(M, R\) \+ 1 = 5"):
        TS.reference(S, TICK, TS.empty(S), inputs(S, RS.empty(S), 0, ios, released))
    S = statics(e, N, R=4, W=4, A=5)
    with pytest.raises(ValueError, match="2 release buckets for 3 arrivals"):
        TS.reference(S, TICK, TS.empty(S), TS.Inputs(tuple(arrival(i, ios[i], False) for i in range(3)), (0, 0)))
