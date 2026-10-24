"""ENC-REPLAY: the advice as vLLM V1's scheduler, replayed from what the certified I/O already shows.

The advice (I1) carries every batch row serving chose. vLLM V1's scheduler (`vllm/v1/core/sched/scheduler.py` at the pods'
pin, d9105ea80) decides those rows deterministically from:

- the registered engine configuration: `max_num_seqs`, `max_num_batched_tokens`, `max_model_len`, `long_prefill_token_threshold`,
  chunked prefill, the KV block size, `watermark`, `scheduler_reserve_full_isl`, prefix caching, async scheduling, and
  `num_gpu_blocks` (fixed at startup by memory profiling, so a per-session field unless the deployment pins it);
- each request's prompt length and `max_tokens` (the certified request), where it stopped (the committed sampled tokens),
  and, with prefix caching on, its tokens (their block hashes);
- the order requests reached the engine (the certified ingress order);
- hidden state that is a function of the history above: the waiting and running queues, each request's computed tokens,
  in-flight tokens and output placeholders, and the KV block pool (its free count, and with caching its LRU free queue and
  hash map);
- one timing input: the engine step each request is first visible at (the `before_step` of its `arrive` record: the engine
  core adds every request that arrived before it calls `schedule`).

`replay` is that model: the served subset of `Scheduler.schedule` and `update_from_output`, `AsyncScheduler`,
`KVCacheManager.allocate_slots`, the full-attention manager and `BlockPool`, line for line, and the engine core's two loops
(`EngineCore.step`, `step_with_batch_queue`). It refuses what it doesn't model (priority policy, spec decode, KV
connectors, multimodal, LoRA, structured output, aborts and stop strings) rather than guess.

**The residual advice** is the timing input, coded online inside the replay. A request's visibility matters only at an
*open* step: one whose waiting loop reached the queue's end with room to admit it. At every other step it would be blocked
whether it had arrived or not, and when the engine is quiet (nothing running or waiting) the next arrival is what starts
the next step, so neither costs a bit. At its open steps the request was passed over until it was admitted, so its
advice is one count: Elias gamma of the open steps it was passed over (`encode(..., clocked=False)`), or, with the
certifier's buckets as public context (`clocked=True`), that count relative to the first open step whose preceding output
was released no earlier than the request's ingress bucket (`_Clock`). `decode` replays from the advice, and `table`
counts the bits per certifier window against I1 v0's fields (`v0_bits`).
"""

from __future__ import annotations

import json
import math
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STEP_TRACE = "verity-vllm/step-trace/v1"


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class Engine:
    """The configuration vLLM's scheduler reads, as `verity_vllm.engine.step_trace.scheduler_config` records it."""

    max_num_seqs: int
    max_num_batched_tokens: int
    max_model_len: int
    num_gpu_blocks: int
    block_size: int = 16
    long_prefill_token_threshold: int = 0
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = False
    async_scheduling: bool = False
    watermark: float = 0.0
    scheduler_reserve_full_isl: bool = True
    policy: str = "fcfs"

    def __post_init__(self) -> None:
        if self.policy != "fcfs":
            raise ValueError(f"policy {self.policy!r}: the replay models FCFS only")

    @classmethod
    def from_header(cls, config: dict[str, Any]) -> Engine:
        asy = config.get("scheduler") == "AsyncScheduler" if config.get("scheduler") else config.get("async_scheduling")
        return cls(max_num_seqs=int(config["max_num_seqs"]), max_num_batched_tokens=int(config["max_num_batched_tokens"]),
                   max_model_len=int(config["max_model_len"]), num_gpu_blocks=int(config["num_gpu_blocks"]),
                   block_size=int(config.get("block_size", 16)),
                   long_prefill_token_threshold=int(config.get("long_prefill_token_threshold") or 0),
                   enable_chunked_prefill=bool(config.get("enable_chunked_prefill", True)),
                   enable_prefix_caching=bool(config.get("enable_prefix_caching", False)), async_scheduling=bool(asy),
                   watermark=float(config.get("watermark", 0.0) or 0.0),
                   scheduler_reserve_full_isl=bool(config.get("scheduler_reserve_full_isl", True)),
                   policy=str(config.get("policy", "fcfs")))


@dataclass(frozen=True)
class RequestIO:
    """One request as the certified I/O shows it. `output_len` counts every token the request emitted, its stop token
    included; `eos` is whether it stopped on a stop token there (else it ran to `max_tokens` or `max_model_len`). `tokens`,
    the prompt then the output, is read only with prefix caching on."""

    prompt_len: int
    max_tokens: int
    output_len: int
    eos: bool
    tokens: tuple[int, ...] | None = None


Row = tuple[int, int, int]          # (request index, first position, tokens scheduled)


@dataclass
class Step:
    """One executed step: its rows in the scheduler's order (running, then new and resumed), and the rest of what
    `SchedulerOutput` says of it."""

    rows: list[Row]
    new: list[int] = field(default_factory=list)
    resumed: list[int] = field(default_factory=list)
    preempted: list[int] = field(default_factory=list)
    cache_hit: dict[int, int] = field(default_factory=dict)

    def key(self) -> tuple[Row, ...]:
        return tuple(sorted(self.rows))


# --- the KV cache: one full-attention group (KVCacheManager, FullAttentionManager, BlockPool) -----------------------------


class _Hashes:
    """Block hashes as interned ids: equal ids iff equal chained (parent, tokens), so there are no collisions."""

    def __init__(self) -> None:
        self._ids: dict[tuple[int, tuple[int, ...]], int] = {}

    def __call__(self, parent: int, block: tuple[int, ...]) -> int:
        return self._ids.setdefault((parent, block), len(self._ids) + 1)


class _KV:
    def __init__(self, engine: Engine) -> None:
        self.B = engine.block_size
        self.L = engine.max_model_len
        self.caching = engine.enable_prefix_caching
        self.watermark_blocks = int(engine.watermark * engine.num_gpu_blocks)
        self.free: OrderedDict[int, None] = OrderedDict((b, None) for b in range(1, engine.num_gpu_blocks))  # block 0 is null
        self.ref = [0] * engine.num_gpu_blocks
        self.block_hash: list[int | None] = [None] * engine.num_gpu_blocks
        self.by_hash: dict[int, OrderedDict[int, None]] = {}
        self.req_blocks: dict[int, list[int]] = {}
        self.num_cached: dict[int, int] = {}

    def _pop_hash(self, b: int) -> None:
        h = self.block_hash[b]
        if h is None:
            return
        blocks = self.by_hash[h]
        del blocks[b]
        if not blocks:
            del self.by_hash[h]
        self.block_hash[b] = None

    def _new_blocks(self, k: int) -> list[int]:
        assert k <= len(self.free)
        out = []
        for _ in range(k):
            b, _ = self.free.popitem(last=False)
            if self.caching:
                self._pop_hash(b)
            self.ref[b] = 1
            out.append(b)
        return out

    def _free_blocks(self, ordered: Iterable[int]) -> None:
        first, last = [], []
        for b in ordered:
            self.ref[b] -= 1
            if self.ref[b] == 0:
                (first if self.block_hash[b] is None or not self.caching else last).append(b)
        for b in reversed(first):
            self.free[b] = None
            self.free.move_to_end(b, last=False)
        for b in last:
            self.free[b] = None

    def hit(self, r: _R) -> list[int]:
        """`find_longest_cache_hit` over the request's full-block hashes, at most `num_tokens - 1` tokens."""
        if not self.caching:
            return []
        out = []
        for h in r.block_hashes[: (r.num_tokens - 1) // self.B]:
            blocks = self.by_hash.get(h)
            if not blocks:
                break
            out.append(next(iter(blocks)))
        return out

    def blocks_to_allocate(self, r: _R, num_tokens: int, hit: Sequence[int]) -> int:
        held = len(self.req_blocks.get(r.i, ()))
        evictable = sum(1 for b in hit if self.ref[b] == 0)
        return max(cdiv(num_tokens, self.B) - len(hit) - held, 0) + evictable

    def allocate_slots(self, r: _R, n: int, hit: Sequence[int] = (), *, full_sequence_must_fit: bool = False,
                       has_scheduled_reqs: bool = True, dry: bool = False) -> bool:
        num_local = r.num_computed + len(hit) * self.B
        total = min(num_local, self.L)
        wm = self.watermark_blocks if has_scheduled_reqs and r.status in (_W, _P) else 0
        if full_sequence_must_fit and self.blocks_to_allocate(r, min(r.num_tokens, self.L), hit) + wm > len(self.free):
            return False
        need_slot = min(total + n, self.L)
        if self.blocks_to_allocate(r, need_slot, hit) + wm > len(self.free):
            return False
        if dry:
            return True
        blocks = self.req_blocks.setdefault(r.i, [])
        if hit:
            for b in hit:
                if self.ref[b] == 0:
                    del self.free[b]
                self.ref[b] += 1
            blocks.extend(hit)
            self.num_cached[r.i] = len(blocks)
        k = cdiv(need_slot, self.B) - len(blocks)
        if k > 0:
            blocks.extend(self._new_blocks(k))
        if self.caching:
            self.cache_blocks(r, min(total + n, r.num_tokens))
        return True

    def cache_blocks(self, r: _R, num_tokens: int) -> None:
        done = self.num_cached.get(r.i, 0)
        full = num_tokens // self.B
        if done >= full:
            return
        blocks = self.req_blocks[r.i]
        for k in range(done, full):
            b = blocks[k]
            if self.block_hash[b] is not None:
                self._pop_hash(b)
            h = r.block_hashes[k]
            self.by_hash.setdefault(h, OrderedDict())[b] = None
            self.block_hash[b] = h
        self.num_cached[r.i] = full

    def free_request(self, r: _R) -> None:
        blocks = self.req_blocks.pop(r.i, [])
        self.num_cached.pop(r.i, None)
        self._free_blocks(reversed(blocks))


# --- the scheduler -------------------------------------------------------------------------------------------------------

_W, _RUN, _P, _F = "waiting", "running", "preempted", "finished"


class _R:
    """A request's scheduler state (`vllm.v1.request.Request`, the fields the served subset reads)."""

    def __init__(self, i: int, io: RequestIO, hashes: _Hashes | None, block_size: int) -> None:
        self.i = i
        self.io = io
        self.P = io.prompt_len
        self.n_out = 0
        self.num_computed = 0
        self.placeholders = 0
        self.status = _W
        self.is_prefill_chunk = False
        self.in_flight = 0
        self.stale = 0
        self.drop_stale = False
        self.preemptions = 0
        self.finish: str | None = None
        self._hashes = hashes
        self._B = block_size
        self.block_hashes: list[int] = []
        if hashes is not None:
            if io.tokens is None or len(io.tokens) < io.prompt_len + io.output_len:
                raise ValueError(f"request {i}: prefix caching is on, so the replay needs its tokens (prompt then output)")
            self._extend_hashes()

    @property
    def num_tokens(self) -> int:
        return self.P + self.n_out

    def _extend_hashes(self) -> None:
        toks = self.io.tokens
        assert toks is not None and self._hashes is not None
        while (len(self.block_hashes) + 1) * self._B <= self.num_tokens:
            k = len(self.block_hashes)
            parent = self.block_hashes[-1] if self.block_hashes else 0
            self.block_hashes.append(self._hashes(parent, tuple(toks[k * self._B:(k + 1) * self._B])))

    def append_output(self) -> None:
        self.n_out += 1
        if self._hashes is not None:
            self._extend_hashes()


class _Sched:
    def __init__(self, engine: Engine, requests: Sequence[RequestIO]) -> None:
        self.e = engine
        self.kv = _KV(engine)
        hashes = _Hashes() if engine.enable_prefix_caching else None
        self.reqs = [_R(i, io, hashes, engine.block_size) for i, io in enumerate(requests)]
        self.waiting: deque[_R] = deque()
        self.skipped: deque[_R] = deque()
        self.running: list[_R] = []
        self.finished_ids: set[int] = set()

    def add(self, i: int) -> None:
        self.waiting.append(self.reqs[i])

    def has_requests(self) -> bool:
        return bool(self.waiting or self.skipped or self.running or self.finished_ids)

    def quiet(self) -> bool:
        """Nothing running or waiting: the next arrival is what starts the next executed step."""
        return not (self.waiting or self.skipped or self.running)

    def _preempt(self, r: _R) -> None:
        assert r.status == _RUN
        self.kv.free_request(r)
        r.status = _P
        r.num_computed = 0
        r.drop_stale = r.drop_stale and r.stale > 0
        r.stale = r.in_flight
        r.placeholders = 0
        r.preemptions += 1
        self.waiting.appendleft(r)

    def would_admit(self, i: int, token_budget: int, input_budget: int) -> bool:
        """Whether request i, appended to the waiting queue, would be scheduled where the waiting loop stands (changes
        nothing)."""
        e = self.e
        if token_budget <= 0 or input_budget <= 0 or len(self.running) >= e.max_num_seqs:
            return False
        r = self.reqs[i]
        assert r.status == _W and r.num_computed == 0
        hit = self.kv.hit(r)
        budget = min(token_budget, input_budget)
        n = r.num_tokens - len(hit) * e.block_size
        if 0 < e.long_prefill_token_threshold < n:
            n = e.long_prefill_token_threshold
        if not e.enable_chunked_prefill and n > budget:
            return False
        n = min(n, budget)
        return self.kv.allocate_slots(r, n, hit, full_sequence_must_fit=e.scheduler_reserve_full_isl,
                                      has_scheduled_reqs=bool(self.running), dry=True)

    def schedule(self, on_open: Callable[[int, int], bool] | None = None) -> tuple[Step, dict[int, bool]]:
        """One `schedule()`. When the waiting loop reaches the queue's end with budget left, `on_open(token_budget,
        input_budget)` may append one more request to the waiting queue (returning True), and the loop goes on."""
        e = self.e
        token_budget = e.max_num_batched_tokens
        input_budget = e.max_num_batched_tokens
        thr = e.long_prefill_token_threshold
        L = e.max_model_len
        step = Step(rows=[])
        sched: dict[int, int] = {}
        starts: dict[int, int] = {}

        idx = 0
        while idx < len(self.running) and token_budget > 0:
            r = self.running[idx]
            if input_budget <= 0:
                break
            if r.placeholders > 0 and r.num_computed + 2 - r.placeholders >= r.P + r.io.max_tokens:
                idx += 1
                continue
            n = r.num_tokens + r.placeholders - r.num_computed
            if 0 < thr < n:
                n = thr
            n = min(n, token_budget, input_budget)
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
            token_budget -= n
            input_budget -= n
            idx += 1

        if not step.preempted:
            skipped_now: deque[_R] = deque()
            while token_budget > 0:
                if not (self.waiting or self.skipped):
                    if on_open is None or not on_open(token_budget, input_budget):
                        break
                if input_budget <= 0:
                    break
                if len(self.running) >= e.max_num_seqs:
                    break
                q = self.skipped if self.skipped else self.waiting
                r = q[0]
                if r.stale > 0 and not r.drop_stale:
                    q.popleft()
                    skipped_now.appendleft(r)
                    continue
                hit = self.kv.hit(r) if r.num_computed == 0 else []
                num_computed = r.num_computed if r.num_computed else len(hit) * e.block_size
                budget = min(token_budget, input_budget)
                n = r.num_tokens - num_computed
                if 0 < thr < n:
                    n = thr
                if not e.enable_chunked_prefill and n > budget:
                    break
                n = min(n, budget)
                assert n > 0
                if not self.kv.allocate_slots(r, n, hit, full_sequence_must_fit=e.scheduler_reserve_full_isl,
                                              has_scheduled_reqs=bool(self.running)):
                    break
                q.popleft()
                self.running.append(r)
                (step.new if r.status == _W else step.resumed).append(r.i)
                if hit:
                    step.cache_hit[r.i] = len(hit) * e.block_size
                sched[r.i] = n
                starts[r.i] = num_computed
                token_budget -= n
                input_budget -= n
                r.status = _RUN
                r.num_computed = num_computed
            if skipped_now:
                self.skipped.extendleft(skipped_now)

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
                    r.placeholders += 1
        self.finished_ids = set()
        step.rows = [(i, starts[i], n) for i, n in sched.items()]
        return step, samples

    def _stop(self, r: _R) -> bool:
        if r.io.eos and r.n_out == r.io.output_len:
            r.finish = "stop"
            return True
        if r.num_tokens >= self.e.max_model_len or r.n_out >= r.io.max_tokens:
            r.finish = "length"
            return True
        return False

    def update(self, step: Step, samples: dict[int, bool]) -> list[int]:
        """`update_from_output` for one executed step, its executor a fake whose row samples iff it ends at the request's
        frontier. Returns the requests that finished."""
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
            if r.status == _F or (stale and r.drop_stale) or not samples[i]:
                continue
            before = r.status
            r.append_output()
            stopped = self._stop(r)
            if self.e.async_scheduling:
                if not stale:
                    r.placeholders -= 1
                    assert r.placeholders >= 0
                if before == _RUN and self.kv.caching:
                    self.kv.cache_blocks(r, r.num_computed - r.placeholders)
            if stopped:
                r.status = _F
                self.finished_ids.add(i)
                self.kv.free_request(r)
                (stopped_running if before == _RUN else stopped_preempted).add(i)
                done.append(i)
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
    forced: list[bool]              # it arrived into a quiet engine
    asked: list[list[int]]          # the open steps it was asked about, its admission last


Decide = Callable[[int, int], bool]


def replay(engine: Engine, requests: Sequence[RequestIO], *, admit: Sequence[int] | None = None,
           decide: Decide | None = None, max_steps: int | None = None) -> Replay:
    """Run the scheduler over `requests` (in the order they reached the engine) and return the executed steps.

    Exactly one of: `admit`, each request's `before_step` (it is added before the first `schedule` after `admit[i]`
    executed steps, as the engine core adds it), or `decide(i, k)`, asked at each open step k whether the next request i
    has arrived (an arrival into a quiet engine is added without asking)."""
    n = len(requests)
    if (admit is None) == (decide is None):
        raise ValueError("give exactly one of admit and decide")
    if admit is not None:
        if len(admit) != n:
            raise ValueError(f"{len(admit)} admission steps for {n} requests")
        if any(admit[k] > admit[k + 1] for k in range(n - 1)):
            raise ValueError("admission steps must be nondecreasing in arrival order (FCFS)")
    s = _Sched(engine, requests)
    out = Replay(steps=[], finish=[None] * n, admitted=[None] * n, forced=[False] * n, asked=[[] for _ in range(n)])
    nxt = 0
    queue: deque[tuple[Step, dict[int, bool]]] = deque()
    limit = max_steps if max_steps is not None else 4 * (sum(r.prompt_len + r.output_len for r in requests) + n) + 64

    def add(forced: bool) -> None:
        nonlocal nxt
        s.add(nxt)
        out.admitted[nxt] = len(out.steps)
        out.forced[nxt] = forced
        nxt += 1

    def on_open(token_budget: int, input_budget: int) -> bool:
        if decide is None or nxt >= n or not s.would_admit(nxt, token_budget, input_budget):
            return False
        out.asked[nxt].append(len(out.steps))
        if not decide(nxt, len(out.steps)):
            return False
        add(False)
        return True

    def schedule() -> tuple[Step, dict[int, bool]]:
        step, samples = s.schedule(on_open if decide is not None else None)
        if step.rows:
            out.steps.append(step)
        return step, samples

    def finished(ids: list[int]) -> None:
        for i in ids:
            out.finish[i] = s.reqs[i].finish

    while True:
        if admit is not None:
            while nxt < n and admit[nxt] <= len(out.steps):
                add(s.quiet())
        elif nxt < n and s.quiet():
            add(True)
        if not s.has_requests() and not queue:
            if nxt >= n:
                break
            raise ValueError(f"request {nxt}: admitted at step {admit[nxt] if admit else None} but the engine is idle before "
                             f"step {len(out.steps)}, which only its arrival starts")
        if len(out.steps) > limit:
            raise RuntimeError(f"no progress after {limit} steps")
        if not engine.async_scheduling:
            step, samples = schedule()
            finished(s.update(step, samples))
            continue
        if s.has_requests():
            step, samples = schedule()
            queue.appendleft((step, samples))
            if len(queue) < 2 and (step.rows or s.has_requests()):
                continue
        elif not queue:
            continue
        step, samples = queue.pop()
        finished(s.update(step, samples))
    return out


# --- traces ------------------------------------------------------------------------------------------------------------------


@dataclass
class Trace:
    """A served run's step trace (`verity-vllm/step-trace/v1`), its requests in the order they reached the engine, with
    the certifier's view of its timing when the trace has a clock: each request's ingress time and each step's release
    time (its output's tokens leaving through the egress certifier)."""

    engine: Engine
    ids: list[str]
    requests: list[RequestIO]
    before_step: list[int]          # each request's admission step, as the engine core added it
    steps: list[Step]
    emits: list[bool]               # whether each step emitted a token
    ingress_t: list[float | None]
    release_t: list[float | None]
    header: dict[str, Any] = field(default_factory=dict)

    def first_scheduled(self) -> list[int]:
        first = [-1] * len(self.requests)
        for k, st in enumerate(self.steps):
            for i, _, _ in st.rows:
                if first[i] < 0:
                    first[i] = k
        return first

    def clocked(self) -> bool:
        return all(t is not None for t in self.ingress_t) and all(t is not None for t in self.release_t)


def load_step_trace(records: Iterable[dict[str, Any]], *, tokens: dict[str, Sequence[int]] | None = None,
                    ingress_lag_s: float = 0.0, release_lag_s: float = 0.0) -> Trace:
    """A `Trace` from `verity_vllm.engine.step_trace` records. Output lengths and stops come from the trace's `emitted` and
    `finished` (in a served run the committed sampled tokens and the certified egress say the same). An `arrive` or
    `step` record's `t` is the engine's clock; a record's `ingress_t` / `release_t`, where the recorder has the
    certifier's, wins, and otherwise the engine's clock less `ingress_lag_s` (plus `release_lag_s`) stands in for it."""
    recs = list(records)
    if not recs or recs[0].get("kind") != "header" or recs[0].get("schema") != STEP_TRACE:
        raise ValueError(f"not a {STEP_TRACE} trace")
    engine = Engine.from_header(recs[0]["config"])
    arrivals = [r for r in recs if r["kind"] == "arrive"]
    ids = [r["request"] for r in arrivals]
    index = {rid: k for k, rid in enumerate(ids)}
    if len(index) != len(ids):
        raise ValueError("a request arrives twice")
    steps_recs = sorted((r for r in recs if r["kind"] == "step"), key=lambda r: r["step"])
    if [r["step"] for r in steps_recs] != list(range(len(steps_recs))):
        raise ValueError("the trace's steps are not 0..S-1 each once")
    emitted = [0] * len(ids)
    finish: list[str | None] = [None] * len(ids)
    steps = []
    for r in steps_recs:
        steps.append(Step(rows=[(index[a], int(b), int(c)) for a, b, c in r["rows"]], new=[index[a] for a in r.get("new", ())],
                          resumed=[index[a] for a in r.get("resumed", ())], preempted=[index[a] for a in r.get("preempted", ())],
                          cache_hit={index[a]: int(k) for a, k in (r.get("cache_hit") or {}).items()}))
        for a, k in (r.get("emitted") or {}).items():
            emitted[index[a]] += int(k)
        for a, f in (r.get("finished") or {}).items():
            finish[index[a]] = f["reason"]
    for r in recs:
        if r["kind"] == "finish":
            finish[index[r["request"]]] = r["reason"]
    requests = []
    for k, a in enumerate(arrivals):
        sp = a.get("sampling") or {}
        if finish[k] not in ("stop", "length"):
            raise ValueError(f"request {ids[k]}: finish {finish[k]!r}; the replay models stop and length only")
        if sp.get("stop"):
            raise ValueError(f"request {ids[k]}: stop strings finish from the frontend, outside the scheduler")
        max_tokens = sp.get("max_tokens")
        toks = tuple(tokens[ids[k]]) if tokens is not None and ids[k] in tokens else None
        requests.append(RequestIO(prompt_len=int(a["prompt_len"]), max_tokens=int(max_tokens) if max_tokens is not None
                                  else engine.max_model_len - int(a["prompt_len"]), output_len=emitted[k],
                                  eos=finish[k] == "stop", tokens=toks))

    def at(r: dict[str, Any], key: str, lag: float) -> float | None:
        if r.get(key) is not None:
            return float(r[key])
        return None if r.get("t") is None else float(r["t"]) + lag

    return Trace(engine=engine, ids=ids, requests=requests, before_step=[int(a["before_step"]) for a in arrivals], steps=steps,
                 emits=[bool(r.get("emitted")) for r in steps_recs],
                 ingress_t=[at(a, "ingress_t", -ingress_lag_s) for a in arrivals],
                 release_t=[at(r, "release_t", release_lag_s) for r in steps_recs], header=recs[0])


def read_step_trace(path: str | Path, **kw: Any) -> Trace:
    return load_step_trace((json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()), **kw)


# --- the residual advice ----------------------------------------------------------------------------------------------------


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


def bucket(t: float, bucket_s: float) -> int:
    return math.floor(t / bucket_s + 1e-9)


@dataclass(frozen=True)
class Decision:
    """One open step's question for the next request: had it arrived (`arrived`)? `passed` counts the request's earlier
    open steps, at which it hadn't."""

    request: int
    step: int
    arrived: bool
    passed: int


def decisions(trace: Trace) -> tuple[list[Decision], Replay]:
    """The trace's decisions in the order the replay asks them (by step, then request), from each request's first
    scheduled step. Refuses a trace the replay doesn't reproduce."""
    first = trace.first_scheduled()
    if any(f < 0 for f in first):
        raise ValueError(f"requests never scheduled: {[trace.ids[i] for i, f in enumerate(first) if f < 0][:8]}")
    rp = replay(trace.engine, trace.requests, decide=lambda i, k: k >= first[i])
    m = compare(rp.steps, trace.steps)
    if m.first_mismatch is not None:
        raise ValueError(f"the replay diverges from the trace at step {m.first_mismatch} ({m.matched}/{m.steps} steps match)")
    out = [Decision(i, k, not forced and p == len(asked) - 1, p)
           for i, (asked, forced) in enumerate(zip(rp.asked, rp.forced)) for p, k in enumerate(asked)]
    return sorted(out, key=lambda d: (d.step, d.request)), rp


def gamma_bits(v: int) -> int:
    """Elias gamma's length for v >= 1."""
    return 2 * v.bit_length() - 1


def gamma_values(rp: Replay) -> list[int | None]:
    """The plainest residual: per request, the open steps it was passed over, sent as Elias gamma of v + 1. A request the
    engine took because it went quiet was passed over at every open step it was asked at (v = their count), and None
    when there were none (0 bits)."""
    return [None if f and not a else len(a) - (0 if f else 1) for a, f in zip(rp.asked, rp.forced)]


class _Clock:
    """The certified timing a decision is coded in: request i's ingress bucket against the release buckets around step k.
    The engine core scheduled step k after processing step k - lag's output (lag 1, or 2 under async scheduling) and
    before any step from k on had emitted, so the newest step at or before k - lag that emitted a token left at about
    when k was scheduled or before (`prev`), and the first step from k on that emitted left after it (`next`)."""

    def __init__(self, trace: Trace, bucket_s: float) -> None:
        if not trace.clocked():
            raise ValueError("the trace has no certifier clock (ingress_t and release_t)")
        self.lag = 2 if trace.engine.async_scheduling else 1
        self.b_in = [bucket(t, bucket_s) for t in trace.ingress_t]                          # type: ignore[arg-type]
        rel = [bucket(t, bucket_s) if e else None for t, e in zip(trace.release_t, trace.emits)]  # type: ignore[arg-type]
        self.prev: list[int | None] = []
        last = None
        for b in rel:
            last = b if b is not None else last
            self.prev.append(last)
        self.next: list[int | None] = [None] * len(rel)
        nxt = None
        for k in range(len(rel) - 1, -1, -1):
            nxt = rel[k] if rel[k] is not None else nxt
            self.next[k] = nxt

    def context(self, i: int, k: int) -> tuple[int, int]:
        j = k - self.lag
        p = self.prev[j] if 0 <= j < len(self.prev) else None
        n = self.next[k] if k < len(self.next) else None
        return (2 if p is None else max(-1, min(1, self.b_in[i] - p)), 2 if n is None else int(self.b_in[i] > n))


# --- an adaptive binary arithmetic code for the decisions (Witten, Neal and Cleary's coder, KT estimates per context) --

_FULL, _HALF, _QTR = 1 << 32, 1 << 31, 1 << 30


class _KT:
    def __init__(self) -> None:
        self.n: dict[Any, tuple[int, int]] = {}

    def freqs(self, ctx: Any) -> tuple[int, int]:
        n0, n1 = self.n.get(ctx, (0, 0))
        return 2 * n0 + 1, 2 * n1 + 1

    def update(self, ctx: Any, bit: bool) -> None:
        n0, n1 = self.n.get(ctx, (0, 0))
        self.n[ctx] = (n0 + (not bit), n1 + bool(bit))


class _Encoder:
    def __init__(self) -> None:
        self.lo, self.hi, self.pending, self.count = 0, _FULL - 1, 0, 0
        self.bits: list[int] = []

    def put(self, bit: bool, f0: int, f1: int) -> None:
        split = self.lo + (self.hi - self.lo + 1) * f0 // (f0 + f1)
        if bit:
            self.lo = split
        else:
            self.hi = split - 1
        self.count += 1
        while True:
            if self.hi < _HALF:
                self._emit(0)
            elif self.lo >= _HALF:
                self._emit(1)
                self.lo -= _HALF
                self.hi -= _HALF
            elif self.lo >= _QTR and self.hi < 3 * _QTR:
                self.pending += 1
                self.lo -= _QTR
                self.hi -= _QTR
            else:
                break
            self.lo, self.hi = 2 * self.lo, 2 * self.hi + 1

    def _emit(self, b: int) -> None:
        self.bits.append(b)
        self.bits.extend([1 - b] * self.pending)
        self.pending = 0

    def finish(self) -> list[int]:
        if self.count:
            self.pending += 1
            self._emit(0 if self.lo < _QTR else 1)
        return self.bits


class _Decoder:
    def __init__(self, bits: Sequence[int]) -> None:
        self.bits, self.pos, self.lo, self.hi, self.code = bits, 0, 0, _FULL - 1, 0
        for _ in range(32):
            self.code = 2 * self.code + self._next()

    def _next(self) -> int:
        b = self.bits[self.pos] if self.pos < len(self.bits) else 0
        self.pos += 1
        return b

    def get(self, f0: int, f1: int) -> bool:
        split = self.lo + (self.hi - self.lo + 1) * f0 // (f0 + f1)
        bit = self.code >= split
        if bit:
            self.lo = split
        else:
            self.hi = split - 1
        while True:
            if self.hi < _HALF:
                pass
            elif self.lo >= _HALF:
                self.lo -= _HALF
                self.hi -= _HALF
                self.code -= _HALF
            elif self.lo >= _QTR and self.hi < 3 * _QTR:
                self.lo -= _QTR
                self.hi -= _QTR
                self.code -= _QTR
            else:
                break
            self.lo, self.hi, self.code = 2 * self.lo, 2 * self.hi + 1, 2 * self.code + self._next()
        return bit


def windows(trace: Trace, *, bucket_s: float, window_buckets: int) -> tuple[list[int], list[int]]:
    """Each request's certifier window (its ingress bucket's) and each step's (its release bucket's); one window without a
    clock."""
    if not trace.clocked():
        return [0] * len(trace.requests), [0] * len(trace.steps)
    return ([bucket(t, bucket_s) // window_buckets for t in trace.ingress_t],         # type: ignore[arg-type]
            [bucket(t, bucket_s) // window_buckets for t in trace.release_t])         # type: ignore[arg-type]


@dataclass
class Code:
    """The residual advice under the adaptive code: one bit string per certifier window (a request's decisions go in its
    ingress window's), and what an ideal coder would spend (`ideal`, bits)."""

    clocked: bool
    bucket_s: float
    window_buckets: int
    bits: dict[int, list[int]]
    ideal: dict[int, float]


def _context(clock: _Clock | None, i: int, k: int, passed: int) -> tuple[int, ...]:
    return (min(passed, 2),) + (clock.context(i, k) if clock is not None else ())


def encode(trace: Trace, *, clocked: bool, bucket_s: float = 0.1, window_buckets: int = 600) -> Code:
    """Code each decision with the adaptive binary code, its context the request's passes so far (`min(passed, 2)`) and,
    clocked, `_Clock.context`."""
    ds, _ = decisions(trace)
    clock = _Clock(trace, bucket_s) if clocked else None
    rw, _ = windows(trace, bucket_s=bucket_s, window_buckets=window_buckets)
    model, enc, ideal = _KT(), {}, {}
    for d in ds:
        ctx = _context(clock, d.request, d.step, d.passed)
        f0, f1 = model.freqs(ctx)
        w = rw[d.request]
        enc.setdefault(w, _Encoder()).put(d.arrived, f0, f1)
        ideal[w] = ideal.get(w, 0.0) - math.log2((f1 if d.arrived else f0) / (f0 + f1))
        model.update(ctx, d.arrived)
    return Code(clocked, bucket_s, window_buckets, {w: e.finish() for w, e in enc.items()}, ideal)


def decode(trace: Trace, code: Code) -> Replay:
    """Replay from the residual advice: the trace's certified I/O and clock only (its steps are not read)."""
    clock = _Clock(trace, code.bucket_s) if code.clocked else None
    rw, _ = windows(trace, bucket_s=code.bucket_s, window_buckets=code.window_buckets)
    model, dec = _KT(), {}
    passed = [0] * len(trace.requests)

    def decide(i: int, k: int) -> bool:
        ctx = _context(clock, i, k, passed[i])
        f0, f1 = model.freqs(ctx)
        w = rw[i]
        if w not in dec:
            dec[w] = _Decoder(code.bits.get(w, []))
        bit = dec[w].get(f0, f1)
        model.update(ctx, bit)
        passed[i] += 1
        return bit

    return replay(trace.engine, trace.requests, decide=decide)


# --- bits per certifier window ----------------------------------------------------------------------------------------


def width(n: int) -> int:
    """Bits for a value in [0, n)."""
    return math.ceil(math.log2(n)) if n > 1 else 0


def v0_widths(trace: Trace, *, window_buckets: int) -> dict[str, int]:
    """I1 v0's fields at fixed width (a stand-in until ADV-SPEC's `bits`): per request its id (an index), prompt_len,
    output_len and finish; per step its row count and release bucket; per row its request, kind, start, n, cache_hit and
    preempted."""
    e = trace.engine
    R = max(len(trace.requests), 1)
    req = width(R) + width(e.max_model_len + 1) + width(max((r.max_tokens for r in trace.requests), default=1) + 1) + 1
    row = width(R) + 1 + width(e.max_model_len) + width(e.max_num_batched_tokens + 1) + width(e.max_model_len) + 1
    step = width(e.max_num_seqs + 1) + width(window_buckets)
    return {"request": req, "row": row, "step": step}


def table(trace: Trace, *, bucket_s: float = 0.1, window_buckets: int = 600) -> dict[str, Any]:
    """Per certifier window: I1 v0's bits, the steps the replay matched, and the residual advice's bits under each code
    (Elias gamma; the adaptive code, unclocked and clocked), each code checked by decoding it back to the trace's steps."""
    truth = replay(trace.engine, trace.requests, admit=trace.before_step)
    ds, rp = decisions(trace)
    g = gamma_values(rp)
    codes = {"adaptive": encode(trace, clocked=False, bucket_s=bucket_s, window_buckets=window_buckets)}
    if trace.clocked():
        codes["adaptive_clocked"] = encode(trace, clocked=True, bucket_s=bucket_s, window_buckets=window_buckets)
    decoded = {name: compare(decode(trace, c).steps, trace.steps).__dict__ for name, c in codes.items()}
    rw, sw = windows(trace, bucket_s=bucket_s, window_buckets=window_buckets)
    v0 = v0_widths(trace, window_buckets=window_buckets)
    rows = []
    for w in sorted(set(rw) | set(sw)):
        ks = [k for k, x in enumerate(sw) if x == w]
        rs = [i for i, x in enumerate(rw) if x == w]
        row = {"window": w, "requests": len(rs), "steps": len(ks), "rows": sum(len(trace.steps[k].rows) for k in ks),
               "matched": sum(1 for k in ks if k < len(truth.steps) and truth.steps[k].key() == trace.steps[k].key()),
               "forced": sum(1 for i in rs if rp.forced[i]), "decisions": sum(1 for d in ds if rw[d.request] == w),
               "v0_bits": len(rs) * v0["request"] + sum(v0["step"] + len(trace.steps[k].rows) * v0["row"] for k in ks),
               "gamma_bits": sum(gamma_bits(g[i] + 1) for i in rs if g[i] is not None)}
        for name, c in codes.items():
            row[f"{name}_bits"] = len(c.bits.get(w, []))
            row[f"{name}_ideal"] = round(c.ideal.get(w, 0.0), 2)
        rows.append(row)
    return {"engine": dict(trace.engine.__dict__), "requests": len(trace.requests), "steps": len(trace.steps),
            "match_before_step": compare(truth.steps, trace.steps).__dict__, "decoded": decoded, "v0_widths": v0,
            "bucket_s": bucket_s, "window_buckets": window_buckets, "windows": rows,
            "passed": [x for x in g if x is not None]}

