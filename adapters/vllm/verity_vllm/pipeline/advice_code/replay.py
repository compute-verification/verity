"""The scheduler replay: the advice's rows as vLLM V1's scheduler, replayed from what the certified I/O already shows.

The advice (I1) carries every batch row serving chose. vLLM V1's scheduler (`vllm/v1/core/sched/scheduler.py` at the pods'
pin, d9105ea80) decides those rows deterministically from:

- the registered engine configuration (`Engine`): `max_num_seqs`, `max_num_batched_tokens`, `max_model_len`,
  `long_prefill_token_threshold`, chunked prefill, the KV block size, `watermark`, `scheduler_reserve_full_isl`, prefix
  caching, async scheduling, the draft settings, and `num_gpu_blocks` (fixed at startup by memory profiling, and
  registered before the run: a prover that chose it could buy derived hits with a bigger pool);
- each request's prompt length and `max_tokens` (the certified request), where it stopped (the committed sampled tokens),
  with prefix caching on its tokens (their block hashes), and with a draft the drafts each of its rows kept;
- the order requests reached the engine (the certified ingress order);
- hidden state that is a function of the history above: the waiting and running queues, each request's computed tokens,
  in-flight tokens and output placeholders, and the KV block pool (`pool.BlockPool`: its free count, and with caching its LRU
  free queue and hash map);
- one timing input: the engine step each request is first visible at (the engine core adds every request that arrived
  before it calls `schedule`), and under async scheduling how many of that step's `schedule` calls came first: with a
  step in flight the engine core can call `schedule` again at the same step count and schedule nothing (when every
  running request's last token is in flight, say), and a request added before that call meets blocks the step in
  flight still holds.

`replay` is that model: the served subset of `Scheduler.schedule` and `update_from_output`, `AsyncScheduler`, and the
engine core's two loops (`EngineCore.step`, `step_with_batch_queue`), over the block pool. It refuses what it doesn't
model (priority policy, a draft other than one MTP module's, KV connectors, multimodal, LoRA, structured output, aborts
and stop strings) rather than guess. It takes the timing input as each request's admission step and calls passed (`admit`, `passed`), or as
`decide`, asked only at an *open* step, one whose waiting loop reached the queue's end with room to admit the next
request, whether that request had arrived: at every other step it would be blocked whether it had arrived or not, and an
arrival into a quiet engine starts the next step. `compare` matches two schedules step by step.

**The certified tick** (``--admit tick``, `verity_vllm.pipeline.serve_admit.TickGate`) removes the timing input: the engine
adds a request when a logical clock T has passed the close of its network-certifier release bucket, and T is a function of
the steps (`Tick`, registered with the engine class). `replay(tick=..., released=...)` runs that rule from the certified
release buckets alone, so it asks nothing.

**With a draft** (`Engine.num_speculative_tokens` K > 0: speculative decoding by the model's own MTP module) a decode row
is its token and then the drafts the request holds, K unless a budget or the context limit cuts the row. A request
holds K drafts after a row that samples: under async scheduling from the moment the row is scheduled, as placeholders;
otherwise once the step has executed. A waiting request whose first row would be one token is given K placeholder drafts
too, in a step with no prefill chunk of a running request, beside a running row or after a prefix hit. Every allocation
takes slots for K tokens past its row, and blocks taken for drafts that are rejected stay with the request; a prefix hit
gives up its last block; only tokens the request has are cached. A row's output is its token and the drafts it kept:
the rest roll its computed tokens (and placeholders) back. How many it kept is the model's choice, so it is an input,
`RequestIO.kept`, as where the request stopped is; `Step.drafts` says how many each row carried.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from verity_vllm.pipeline.advice_code.pool import BlockPool
from verity_vllm.pipeline.advice_code.schedule import (
    FINISHED,
    PREEMPTED,
    RUNNING,
    WAITING,
    BlockHashes,
    Engine,
    RequestIO,
    RequestState,
    Step,
    Tick,
)

__all__ = ["compare", "Decide", "drive", "Emitted", "Match", "Open", "Queue", "Replay", "replay", "Scheduler"]


@dataclass
class _Pass:
    """One `schedule()` in progress: its budgets left, its step so far, its rows (request -> tokens, first position), the
    requests its waiting loop set aside, and whether its running loop scheduled a row, and one of a request whose last
    row was a prefill chunk."""

    token_budget: int
    input_budget: int
    step: Step
    sched: dict[int, int]
    starts: dict[int, int]
    skipped_now: deque[RequestState]
    running_rows: bool = False
    prefill_scheduled: bool = False

    def fork(self, reqs: Sequence[RequestState]) -> _Pass:
        st = self.step
        return _Pass(self.token_budget, self.input_budget,
                     Step(rows=list(st.rows), new=list(st.new), resumed=list(st.resumed), preempted=list(st.preempted),
                          cache_hit=dict(st.cache_hit), sources={k: list(v) for k, v in st.sources.items()},
                          drafts=dict(st.drafts)),
                     dict(self.sched), dict(self.starts), deque(reqs[r.i] for r in self.skipped_now),
                     self.running_rows, self.prefill_scheduled)


class Scheduler:
    def __init__(self, engine: Engine, requests: Sequence[RequestIO]) -> None:
        self.e = engine
        self.kv = BlockPool(engine)
        hashes = BlockHashes() if engine.enable_prefix_caching else None
        if engine.num_speculative_tokens:
            for i, io in enumerate(requests):
                if io.kept is None or len(io.kept) + sum(io.kept) != io.output_len or any(
                        type(k) is not int or not 0 <= k <= engine.num_speculative_tokens for k in io.kept):
                    raise ValueError(f"request {i}: the engine drafts {engine.num_speculative_tokens} tokens a row, so the "
                                     f"replay needs the drafts each of its rows kept (each 0 to "
                                     f"{engine.num_speculative_tokens}, and with one token a row its {io.output_len} outputs)")
        self.whatif = False         # a fork's: it runs a schedule that did not happen
        self.reqs = [RequestState(i, io, hashes, engine.block_size) for i, io in enumerate(requests)]
        self.waiting: deque[RequestState] = deque()
        self.skipped: deque[RequestState] = deque()
        self.running: list[RequestState] = []
        self.finished_ids: set[int] = set()

    def fork(self, touched: Iterable[int] = ()) -> Scheduler:
        """A copy to run on while this one waits, which may change only the requests in the engine and `touched` (the
        rest, and the block-hash interning, which only names contents, are shared)."""
        c = object.__new__(Scheduler)
        c.e, c.kv, c.whatif = self.e, self.kv.fork(), True
        c.reqs = list(self.reqs)
        for i in {*touched, *(r.i for r in (*self.running, *self.waiting, *self.skipped))}:
            c.reqs[i] = self.reqs[i].fork()
        c.waiting = deque(c.reqs[r.i] for r in self.waiting)
        c.skipped = deque(c.reqs[r.i] for r in self.skipped)
        c.running = [c.reqs[r.i] for r in self.running]
        c.finished_ids = set(self.finished_ids)
        return c

    def add(self, i: int) -> None:
        self.waiting.append(self.reqs[i])

    def has_requests(self) -> bool:
        return bool(self.waiting or self.skipped or self.running or self.finished_ids)

    def quiet(self) -> bool:
        """Nothing running or waiting: the next arrival is what starts the next executed step."""
        return not (self.waiting or self.skipped or self.running)

    def _preempt(self, r: RequestState) -> None:
        assert r.status == RUNNING
        self.kv.free_request(r)
        r.status = PREEMPTED
        r.num_computed = 0
        r.drop_stale = r.drop_stale and r.stale > 0
        r.stale = r.in_flight
        r.placeholders = 0
        r.spec = 0
        r.preemptions += 1
        self.waiting.appendleft(r)

    def _padded(self, p: _Pass, n: int, num_computed: int) -> bool:
        """Whether a waiting request's row of n tokens from `num_computed` is given K placeholder drafts in pass p: a
        one-token row, in a step with no prefill chunk of a running request, beside a running row or after a prefix
        hit, where the drafts and a sampled token fit the context."""
        k = self.e.num_speculative_tokens
        return (k > 0 and n == 1 and not p.prefill_scheduled and (p.running_rows or num_computed > 0)
                and num_computed + 1 + k + 1 <= self.e.max_model_len)

    def would_admit(self, i: int, token_budget: int, input_budget: int, p: _Pass | None = None) -> bool:
        """Whether request i, appended to the waiting queue, would be scheduled where the waiting loop stands (changes
        nothing). `p` is the pass the loop is in, which a draft's padded row reads."""
        e = self.e
        if token_budget <= 0 or input_budget <= 0 or len(self.running) >= e.max_num_seqs:
            return False
        r = self.reqs[i]
        assert r.status == WAITING and r.num_computed == 0
        hit = self.kv.hit(r)
        budget = min(token_budget, input_budget)
        n = r.num_tokens - len(hit) * e.block_size
        if p is not None and self._padded(p, n, len(hit) * e.block_size):
            if 1 + e.num_speculative_tokens > budget:
                return False
            n = 1 + e.num_speculative_tokens
        if 0 < e.long_prefill_token_threshold < n:
            n = e.long_prefill_token_threshold
        if not e.enable_chunked_prefill and n > budget:
            return False
        n = min(n, budget)
        return self.kv.allocate_slots(r, n, hit, full_sequence_must_fit=e.scheduler_reserve_full_isl,
                                      has_scheduled_reqs=bool(self.running), dry=True)

    def schedule(self, on_open: Callable[[_Pass], bool] | None = None) -> tuple[Step, dict[int, bool]]:
        """One `schedule()`. When the waiting loop reaches the queue's end with budget left, `on_open(pass)` may append one
        more request to the waiting queue (returning True), and the loop goes on."""
        p = self._running()
        if not p.step.preempted:
            self._waiting(p, on_open)
        return self._close(p)

    def _running(self) -> _Pass:
        e = self.e
        thr = e.long_prefill_token_threshold
        L = e.max_model_len
        p = _Pass(e.max_num_batched_tokens, e.max_num_batched_tokens, Step(rows=[]), {}, {}, deque())
        step, sched, starts = p.step, p.sched, p.starts

        idx = 0
        while idx < len(self.running) and p.token_budget > 0:
            r = self.running[idx]
            if p.input_budget <= 0:
                break
            if r.placeholders > 0 and r.num_computed + 2 - r.placeholders >= r.P + r.io.max_tokens:
                idx += 1
                continue
            n = r.num_tokens + r.spec + r.placeholders - r.num_computed
            if 0 < thr < n:
                n = thr
            n = min(n, p.token_budget, p.input_budget)
            n = min(n, L - r.num_computed - 1)
            if n == 0:
                idx += 1
                continue
            ok = False
            while True:
                ok = self.kv.allocate_slots(r, n)
                if ok:
                    break
                victim = self.running.pop()
                self._preempt(victim)
                step.preempted.append(victim.i)
                if victim is r:
                    break
            if not ok:
                break
            sched[r.i] = n
            starts[r.i] = r.num_computed
            p.running_rows = True
            p.prefill_scheduled |= r.is_prefill_chunk
            p.token_budget -= n
            p.input_budget -= n
            idx += 1
            if r.spec:
                d = n + r.num_computed - r.num_tokens - r.placeholders
                if d > 0:
                    step.drafts[r.i] = min(d, r.spec)
                r.spec = 0
        return p

    def _waiting(self, p: _Pass, on_open: Callable[[_Pass], bool] | None) -> None:
        e = self.e
        thr = e.long_prefill_token_threshold
        step, sched, starts = p.step, p.sched, p.starts
        while p.token_budget > 0:
            if not (self.waiting or self.skipped):
                if on_open is None or not on_open(p):
                    break
            if p.input_budget <= 0:
                break
            if len(self.running) >= e.max_num_seqs:
                break
            q = self.skipped if self.skipped else self.waiting
            r = q[0]
            if r.stale > 0 and not r.drop_stale:
                q.popleft()
                p.skipped_now.appendleft(r)
                continue
            hit = self.kv.hit(r) if r.num_computed == 0 else []
            num_computed = r.num_computed if r.num_computed else len(hit) * e.block_size
            budget = min(p.token_budget, p.input_budget)
            n = r.num_tokens - num_computed
            pad = self._padded(p, n, num_computed)
            if pad:
                if 1 + e.num_speculative_tokens > budget:
                    break
                n = 1 + e.num_speculative_tokens
            if 0 < thr < n:
                n = thr
            if not e.enable_chunked_prefill and n > budget:
                break
            n = min(n, budget)
            assert n > 0
            if not self.kv.allocate_slots(r, n, hit, full_sequence_must_fit=e.scheduler_reserve_full_isl,
                                          has_scheduled_reqs=bool(self.running)):
                break
            if pad:
                step.drafts[r.i] = e.num_speculative_tokens
            q.popleft()
            self.running.append(r)
            (step.new if r.status == WAITING else step.resumed).append(r.i)
            if hit:
                step.cache_hit[r.i] = len(hit) * e.block_size
                runs: list[tuple[int, int]] = []
                for b in hit:
                    o = self.kv.owner[b]
                    assert o is not None
                    if runs and runs[-1][0] == o:
                        runs[-1] = (o, runs[-1][1] + e.block_size)
                    else:
                        runs.append((o, e.block_size))
                step.sources[r.i] = runs
            sched[r.i] = n
            starts[r.i] = num_computed
            p.token_budget -= n
            p.input_budget -= n
            r.status = RUNNING
            r.num_computed = num_computed
        if p.skipped_now:
            self.skipped.extendleft(p.skipped_now)
            p.skipped_now = deque()

    def _close(self, p: _Pass) -> tuple[Step, dict[int, bool]]:
        e, step, sched, starts = self.e, p.step, p.sched, p.starts
        samples: dict[int, bool] = {}
        for i, n in sched.items():
            r = self.reqs[i]
            r.num_computed += n
            r.in_flight += n
            r.is_prefill_chunk = r.num_computed < r.num_tokens + r.placeholders
            samples[i] = not r.is_prefill_chunk
        if e.async_scheduling:
            for i in sched:
                r = self.reqs[i]
                if not r.is_prefill_chunk:
                    r.placeholders += 1 + step.drafts.get(i, 0)
                    r.spec = e.num_speculative_tokens
        self.finished_ids = set()
        step.rows = [(i, starts[i], n) for i, n in sched.items()]
        return step, samples

    def _stop(self, r: RequestState) -> bool:
        if r.io.eos and r.n_out == r.io.output_len:
            r.finish = "stop"
            return True
        if r.num_tokens >= self.e.max_model_len or r.n_out >= r.io.max_tokens:
            r.finish = "length"
            return True
        return False

    def _kept(self, r: RequestState, d: int) -> int:
        """The drafts request r's row, which carries d and emits, kept: its next entry of `io.kept`."""
        kept = r.io.kept
        if not self.e.num_speculative_tokens or kept is None or r.emitted >= len(kept):
            return 0
        k = kept[r.emitted]
        if k > d:
            if not self.whatif:
                raise ValueError(f"request {r.i}: emitting row {r.emitted} kept {k} drafts, but the row the replay "
                                 f"schedules there carries {d}")
            k = d
        return k

    def update(self, step: Step, samples: dict[int, bool]) -> list[int]:
        """`update_from_output` for one executed step, its executor a fake whose row samples iff it ends at the request's
        frontier, and with a draft keeps the drafts `RequestIO.kept` says. Returns the requests that finished."""
        K = self.e.num_speculative_tokens
        stopped_running: set[int] = set()
        stopped_preempted: set[int] = set()
        done = []
        for i, _, n in step.rows:
            r = self.reqs[i]
            r.in_flight -= n
            stale = False
            if r.stale > 0:
                stale = True
                r.stale -= n
                assert r.stale >= 0
            if r.status == FINISHED or (stale and r.drop_stale) or not samples[i]:
                continue
            before = r.status
            d = step.drafts.get(i, 0)
            kept = self._kept(r, d)
            if d and not stale:
                if r.num_computed > 0:
                    r.num_computed -= d - kept
                if r.placeholders > 0:
                    r.placeholders -= d - kept
            r.emitted += 1
            took = 0
            stopped = False
            while took < 1 + kept and not stopped:
                r.append_output()
                took += 1
                stopped = self._stop(r)
            if took < 1 + kept and not self.whatif:
                raise ValueError(f"request {i}: emitting row {r.emitted - 1} kept {kept} drafts, but the request stops "
                                 f"after {took} of its {1 + kept} tokens")
            if self.e.async_scheduling:
                if not stale:
                    r.placeholders -= took
                    assert r.placeholders >= 0
                if before == RUNNING and self.kv.caching:
                    self.kv.cache_blocks(r, r.num_computed - r.placeholders)
            if stopped:
                r.status = FINISHED
                self.finished_ids.add(i)
                self.kv.free_request(r)
                (stopped_running if before == RUNNING else stopped_preempted).add(i)
                done.append(i)
        if K and not self.e.async_scheduling and step.rows:
            # the engine core hands the scheduler the runner's drafts after the step: K for each request of its batch,
            # none for one whose row was a prefill chunk
            for i, _, _ in step.rows:
                r = self.reqs[i]
                if r.status != FINISHED:
                    r.spec = 0 if r.is_prefill_chunk else K
        if stopped_running:
            self.running = [r for r in self.running if r.i not in stopped_running]
        if stopped_preempted:
            self.waiting = deque(r for r in self.waiting if r.i not in stopped_preempted)
            self.skipped = deque(r for r in self.skipped if r.i not in stopped_preempted)
        return done


# --- the engine core's loop -------------------------------------------------------------------------------------------------


@dataclass
class Replay:
    steps: list[Step]
    finish: list[str | None]
    admitted: list[int | None]      # the executed step each request was first visible at
    passed: list[int]               # the `schedule` calls already made at that step when it became visible
    forced: list[bool]              # it arrived into a quiet engine
    asked: list[list[int]]          # the open steps it was asked about, its admission last
    clock: list[int]                # under the certified tick, T after each executed step (µs); else empty


Emitted = list[dict[int, int]]     # per executed step, in order: request -> the index of the output it emitted there


@dataclass(frozen=True)
class Open:
    """The scheduler where an open step's waiting loop stands: its state, the step's token budget left and its rows so
    far (request -> tokens), all read only; and `whatif(now, want)`, the engine run on from here on a copy, the request
    asked about arriving now (or before the next step) and nothing after it, until it has emitted `want` outputs (or
    finished, or `_WHATIF_STEPS` steps): what each executed step emitted."""

    sched: Scheduler
    budget: int
    scheduled: dict[int, int]
    whatif: Callable[[bool, int], Emitted]


Decide = Callable[[int, int, Open], bool]
Queue = deque[tuple[Step, dict[int, bool]]]
_WHATIF_STEPS = 256


def drive(s: Scheduler, queue: Queue, *, top: Callable[[], bool], schedule: Callable[[], tuple[Step, dict[int, bool]]],
          executed: Callable[[Step, dict[int, bool]], None], resume: tuple[Step, dict[int, bool]] | None = None) -> None:
    """The engine core's loop (`EngineCore.step`; `step_with_batch_queue` under async scheduling, two steps in flight):
    `top()` before each schedule (False ends it), `executed` after each step's update; `resume`, a step this iteration
    has already scheduled."""
    while True:
        if resume is None and not top():
            return
        if not s.e.async_scheduling:
            step, samples = resume or schedule()
            resume = None
            executed(step, samples)
            continue
        if resume is not None or s.has_requests():
            step, samples = resume or schedule()
            resume = None
            queue.appendleft((step, samples))
            if len(queue) < 2 and (step.rows or s.has_requests()):
                continue
        elif not queue:
            continue
        step, samples = queue.pop()
        executed(step, samples)


def _whatif(s: Scheduler, queue: Queue, p: _Pass, i: int, now: bool, want: int) -> Emitted:
    c, q = s.fork({i, *(r.i for r in p.skipped_now), *(j for st, _ in queue for j, _, _ in st.rows)}), deque(queue)
    p, r, later = p.fork(c.reqs), c.reqs[i], not now
    if now:
        c.add(i)
    c._waiting(p, None)
    out: Emitted = []
    count = 0

    def top() -> bool:
        nonlocal later
        if later:
            c.add(i)
            later = False
        return (c.has_requests() or bool(q)) and r.n_out < want and r.status != FINISHED and count < _WHATIF_STEPS

    def schedule() -> tuple[Step, dict[int, bool]]:
        nonlocal count
        count += 1
        return c.schedule(None)

    def executed(step: Step, samples: dict[int, bool]) -> None:
        before = {j: c.reqs[j].n_out for j, _, _ in step.rows}
        c.update(step, samples)
        if step.rows:
            out.append({j: m for j, m in before.items() if c.reqs[j].n_out > m})

    drive(c, q, top=top, schedule=schedule, executed=executed, resume=c._close(p))
    return out


def replay(engine: Engine, requests: Sequence[RequestIO], *, admit: Sequence[int] | None = None,
           passed: Sequence[int] | None = None, decide: Decide | None = None, tick: Tick | None = None,
           released: Sequence[int] | None = None, max_steps: int | None = None) -> Replay:
    """Run the scheduler over `requests` (in the order they reached the engine) and return the executed steps.

    Exactly one of: `admit`, each request's `before_step` (it is added before the first `schedule` after `admit[i]`
    executed steps once `passed[i]` calls have been made at that step count, as the engine core adds it; `passed`
    defaults to all 0, and `replay(admit=r.admitted, passed=r.passed)` reproduces any run `r`, which `admitted` alone
    can't under async scheduling); `decide(i, k, open)`, asked at each open step k whether the next request i has
    arrived (an arrival into a quiet engine is added without asking); or `tick` with `released`, each request's
    certified release bucket (the requests in release order, so nondecreasing), added by the certified tick.
    Under the tick, T starts at 0; before each iteration of the engine core's loop, when nothing is running or waiting T
    jumps to the next request's bucket close, and every request whose bucket has closed by T ((a + 1)·τ_b ≤ T) is added,
    except at an iteration that only frees finished requests (nothing running or waiting), which adds none; after each
    step that schedules a token, T advances by τ_sat if the waiting queue is nonempty and by τ_free if it is empty
    (`Replay.clock` keeps T after each such step)."""
    n = len(requests)
    if (admit is not None) + (decide is not None) + (tick is not None) != 1:
        raise ValueError("give exactly one of admit, decide and tick")
    if (tick is None) != (released is None):
        raise ValueError("give the release buckets with the tick, and only with it")
    if passed is not None and admit is None:
        raise ValueError("give passed only with admit")
    if admit is not None:
        if len(admit) != n:
            raise ValueError(f"{len(admit)} admission steps for {n} requests")
        passed = [0] * n if passed is None else passed
        if len(passed) != n or any(type(c) is not int or c < 0 for c in passed):
            raise ValueError(f"give one int ≥ 0 of passed per request ({len(passed)} for {n} requests)")
        if any((admit[k], passed[k]) > (admit[k + 1], passed[k + 1]) for k in range(n - 1)):
            raise ValueError("admission steps, then passed, must be nondecreasing in arrival order (FCFS)")
    if released is not None:
        if len(released) != n:
            raise ValueError(f"{len(released)} release buckets for {n} requests")
        if any(type(a) is not int or a < 0 for a in released) or any(released[k] > released[k + 1] for k in range(n - 1)):
            raise ValueError("release buckets are ints ≥ 0, nondecreasing in release order")
    s = Scheduler(engine, requests)
    out = Replay(steps=[], finish=[None] * n, admitted=[None] * n, forced=[False] * n, asked=[[] for _ in range(n)],
                 clock=[], passed=[0] * n)
    nxt = 0
    T = 0
    here = 0                        # the `schedule` calls made since the last step with rows
    queue: Queue = deque()
    limit = max_steps if max_steps is not None else 4 * (sum(r.prompt_len + r.output_len for r in requests) + n) + 64

    def add(forced: bool) -> None:
        nonlocal nxt
        s.add(nxt)
        out.admitted[nxt] = len(out.steps)
        out.passed[nxt] = here
        out.forced[nxt] = forced
        nxt += 1

    def on_open(p: _Pass) -> bool:
        if decide is None or nxt >= n or not s.would_admit(nxt, p.token_budget, p.input_budget, p):
            return False
        out.asked[nxt].append(len(out.steps))
        i = nxt
        o = Open(s, min(p.token_budget, p.input_budget), p.sched, lambda now, want: _whatif(s, queue, p, i, now, want))
        if not decide(i, len(out.steps), o):
            return False
        add(False)
        return True

    def top() -> bool:
        nonlocal T
        if admit is not None and passed is not None:
            while nxt < n and (admit[nxt], passed[nxt]) <= (len(out.steps), here):
                add(s.quiet())
        elif tick is not None and released is not None:
            if not (s.quiet() and s.finished_ids):
                if s.quiet() and nxt < n:
                    T = max(T, tick.close(released[nxt]))
                while nxt < n and tick.close(released[nxt]) <= T:
                    add(s.quiet())
        elif nxt < n and s.quiet():
            add(True)
        if not s.has_requests() and not queue:
            if nxt >= n:
                return False
            raise ValueError(f"request {nxt}: admitted at step {admit[nxt] if admit else None} but the engine is idle before "
                             f"step {len(out.steps)}, which only its arrival starts")
        if len(out.steps) > limit:
            raise RuntimeError(f"no progress after {limit} steps")
        return True

    def schedule() -> tuple[Step, dict[int, bool]]:
        nonlocal T, here
        step, samples = s.schedule(on_open if decide is not None else None)
        here += 1
        if step.rows:
            out.steps.append(step)
            here = 0
            if tick is not None:
                T += tick.sat_us if (s.waiting or s.skipped) else tick.free_us
                out.clock.append(T)
        return step, samples

    def executed(step: Step, samples: dict[int, bool]) -> None:
        for i in s.update(step, samples):
            out.finish[i] = s.reqs[i].finish

    drive(s, queue, top=top, schedule=schedule, executed=executed)
    return out


@dataclass
class Match:
    steps: int                      # steps compared (the longer of the two)
    matched: int                    # steps whose rows agree (as a set of (request, start, n))
    first_mismatch: int | None
    ordered: int                    # steps whose rows also agree in order


def compare(a: Sequence[Step], b: Sequence[Step]) -> Match:
    matched = ordered = 0
    first = None
    for k in range(max(len(a), len(b))):
        if k < len(a) and k < len(b) and a[k].key() == b[k].key():
            matched += 1
            ordered += a[k].rows == b[k].rows
        elif first is None:
            first = k
    return Match(steps=max(len(a), len(b)), matched=matched, first_mismatch=first, ordered=ordered)
