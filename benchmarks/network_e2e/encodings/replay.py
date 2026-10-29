"""The scheduler replay: the advice as vLLM V1's scheduler, replayed from what the certified I/O already shows.

The advice (I1) carries every batch row serving chose. vLLM V1's scheduler (`vllm/v1/core/sched/scheduler.py` at the pods'
pin, d9105ea80) decides those rows deterministically from:

- the registered engine configuration: `max_num_seqs`, `max_num_batched_tokens`, `max_model_len`, `long_prefill_token_threshold`,
  chunked prefill, the KV block size, `watermark`, `scheduler_reserve_full_isl`, prefix caching, async scheduling, and
  `num_gpu_blocks` (fixed at startup by memory profiling, and registered before the run: a prover that chose it could
  buy derived hits with a bigger pool);
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
the next step, so neither costs a bit. So the advice is one binary decision per open step, whether the next request had
arrived (`decisions`): `gamma_values` sends each request's count of open steps passed over as Elias gamma, and `encode`
codes the decisions with an adaptive binary arithmetic code inside the replay, one bit string per certifier window. Its
contexts (`CONTEXTS`) are the request's passes so far, its certified ingress bucket against the release buckets around
the step (`_Clock`), and its certified egress: the replay, forked at the open step and run on with the request arriving
now or a step later and nothing after it, says which of its steps emit its outputs, and an earlier request's output from
each of those steps says which bucket it left in (`_Egress`). `decode` replays from that advice and the certified I/O
alone, and
`table` counts the bits per certifier window against I1 v0's (`advice.bits_by_window`, else `v0_widths`).
"""

from __future__ import annotations

import gzip
import json
import math
import re
import copy
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
    sources: dict[int, list[tuple[int, int]]] = field(default_factory=dict)   # a hit's runs (request, tokens) by owner

    def key(self) -> tuple[Row, ...]:
        return tuple(sorted(self.rows))


# --- the KV cache: one full-attention group (KVCacheManager, FullAttentionManager, BlockPool) -----------------------------


class _Hashes:
    """Block hashes as interned ids: equal ids iff equal chained (parent, tokens), so there are no collisions."""

    def __init__(self) -> None:
        self._ids: dict[tuple[int, tuple[int, ...]], int] = {}

    def __call__(self, parent: int, block: tuple[int, ...]) -> int:
        return self._ids.setdefault((parent, block), len(self._ids) + 1)


class _Free:
    """`BlockPool`'s free-block queue: blocks are taken from the front, freed to either end, and a cached one is taken
    from wherever it is on a hit."""

    def __init__(self, blocks: Iterable[int]) -> None:
        self.d: OrderedDict[int, None] = OrderedDict((b, None) for b in blocks)

    def __len__(self) -> int:
        return len(self.d)

    def pop(self) -> int:
        return self.d.popitem(last=False)[0]

    def take(self, b: int) -> None:
        del self.d[b]

    def push(self, b: int, *, front: bool) -> None:
        self.d[b] = None
        if front:
            self.d.move_to_end(b, last=False)


class _FreeFork:
    """A fork's free-block queue over its parent's, which stays as it is while the fork runs: the blocks the fork freed to
    the front, then the parent's it hasn't taken, then those it freed to the back."""

    def __init__(self, base: _Free) -> None:
        self.base, self.it, self.gone = base.d, iter(base.d), set()
        self.front: OrderedDict[int, None] = OrderedDict()
        self.back: OrderedDict[int, None] = OrderedDict()
        self.n = len(base.d)

    def __len__(self) -> int:
        return self.n

    def pop(self) -> int:
        self.n -= 1
        if self.front:
            return self.front.popitem(last=False)[0]
        for b in self.it:
            if b not in self.gone:
                return b
        return self.back.popitem(last=False)[0]

    def take(self, b: int) -> None:
        self.n -= 1
        if b in self.front:
            del self.front[b]
        elif b in self.back:
            del self.back[b]
        else:
            self.gone.add(b)

    def push(self, b: int, *, front: bool) -> None:
        self.n += 1
        if front:
            self.front[b] = None
            self.front.move_to_end(b, last=False)
        else:
            self.back[b] = None


class _KV:
    def __init__(self, engine: Engine) -> None:
        self.B = engine.block_size
        self.L = engine.max_model_len
        self.caching = engine.enable_prefix_caching
        self.watermark_blocks = int(engine.watermark * engine.num_gpu_blocks)
        self.free: _Free | _FreeFork = _Free(range(1, engine.num_gpu_blocks))  # block 0 is the null block
        self.ref = [0] * engine.num_gpu_blocks
        self.block_hash: list[int | None] = [None] * engine.num_gpu_blocks
        self.by_hash: dict[int, OrderedDict[int, None]] = {}
        self.mine: set[int] | None = None       # a fork's: the hashes whose blocks it has copied from its parent's
        self.req_blocks: dict[int, list[int]] = {}
        self.num_cached: dict[int, int] = {}
        self.owner: list[int | None] = [None] * engine.num_gpu_blocks     # the request that computes a block's KV

    def fork(self) -> _KV:
        """A copy to run on while this one waits (its free queue and hash map are overlays on this one's)."""
        assert isinstance(self.free, _Free)
        k = object.__new__(_KV)
        k.B, k.L, k.caching, k.watermark_blocks = self.B, self.L, self.caching, self.watermark_blocks
        k.free, k.ref, k.block_hash, k.owner = _FreeFork(self.free), list(self.ref), list(self.block_hash), list(self.owner)
        k.by_hash, k.mine = dict(self.by_hash), set()
        k.req_blocks = {i: list(bs) for i, bs in self.req_blocks.items()}
        k.num_cached = dict(self.num_cached)
        return k

    def _blocks(self, h: int) -> OrderedDict[int, None]:
        """The blocks cached under h, to change (a fork's own copy)."""
        bs = self.by_hash.get(h)
        if bs is None or (self.mine is not None and h not in self.mine):
            bs = self.by_hash[h] = OrderedDict() if bs is None else OrderedDict(bs)
            if self.mine is not None:
                self.mine.add(h)
        return bs

    def _pop_hash(self, b: int) -> None:
        h = self.block_hash[b]
        if h is None:
            return
        blocks = self._blocks(h)
        del blocks[b]
        if not blocks:
            del self.by_hash[h]
        self.block_hash[b] = None

    def _new_blocks(self, k: int) -> list[int]:
        assert k <= len(self.free)
        out = []
        for _ in range(k):
            b = self.free.pop()
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
            self.free.push(b, front=True)
        for b in last:
            self.free.push(b, front=False)

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
                    self.free.take(b)
                self.ref[b] += 1
            blocks.extend(hit)
            self.num_cached[r.i] = len(blocks)
        k = cdiv(need_slot, self.B) - len(blocks)
        if k > 0:
            new = self._new_blocks(k)
            for b in new:
                self.owner[b] = r.i
            blocks.extend(new)
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
            self._blocks(h)[b] = None
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

    def fork(self) -> _R:
        r = copy.copy(self)
        r.block_hashes = list(self.block_hashes)
        return r


@dataclass
class _Pass:
    """One `schedule()` in progress: its budgets left, its step so far, its rows (request -> tokens, first position) and
    the requests its waiting loop set aside."""

    token_budget: int
    input_budget: int
    step: Step
    sched: dict[int, int]
    starts: dict[int, int]
    skipped_now: deque[_R]

    def fork(self, reqs: Sequence[_R]) -> _Pass:
        st = self.step
        return _Pass(self.token_budget, self.input_budget,
                     Step(rows=list(st.rows), new=list(st.new), resumed=list(st.resumed), preempted=list(st.preempted),
                          cache_hit=dict(st.cache_hit), sources={k: list(v) for k, v in st.sources.items()}),
                     dict(self.sched), dict(self.starts), deque(reqs[r.i] for r in self.skipped_now))


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

    def fork(self, touched: Iterable[int] = ()) -> _Sched:
        """A copy to run on while this one waits, which may change only the requests in the engine and `touched` (the
        rest, and the block-hash interning, which only names contents, are shared)."""
        c = object.__new__(_Sched)
        c.e, c.kv = self.e, self.kv.fork()
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
            n = r.num_tokens + r.placeholders - r.num_computed
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
            p.token_budget -= n
            p.input_budget -= n
            idx += 1
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
            r.status = _RUN
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


Emitted = list[dict[int, int]]     # per executed step, in order: request -> the index of the output it emitted there


@dataclass(frozen=True)
class Open:
    """The scheduler where an open step's waiting loop stands: its state, the step's token budget left and its rows so
    far (request -> tokens), all read only; and `whatif(now, want)`, the engine run on from here on a copy, the request
    asked about arriving now (or before the next step) and nothing after it, until it has emitted `want` outputs (or
    finished, or `_WHATIF_STEPS` steps): what each executed step emitted."""

    sched: _Sched
    budget: int
    scheduled: dict[int, int]
    whatif: Callable[[bool, int], Emitted]


Decide = Callable[[int, int, Open], bool]
Queue = deque[tuple[Step, dict[int, bool]]]
_WHATIF_STEPS = 256


def _drive(s: _Sched, queue: Queue, *, top: Callable[[], bool], schedule: Callable[[], tuple[Step, dict[int, bool]]],
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


def _whatif(s: _Sched, queue: Queue, p: _Pass, i: int, now: bool, want: int) -> Emitted:
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
        return (c.has_requests() or bool(q)) and r.n_out < want and r.status != _F and count < _WHATIF_STEPS

    def schedule() -> tuple[Step, dict[int, bool]]:
        nonlocal count
        count += 1
        return c.schedule(None)

    def executed(step: Step, samples: dict[int, bool]) -> None:
        before = {j: c.reqs[j].n_out for j, _, _ in step.rows}
        c.update(step, samples)
        if step.rows:
            out.append({j: m for j, m in before.items() if c.reqs[j].n_out > m})

    _drive(c, q, top=top, schedule=schedule, executed=executed, resume=c._close(p))
    return out


def replay(engine: Engine, requests: Sequence[RequestIO], *, admit: Sequence[int] | None = None,
           decide: Decide | None = None, max_steps: int | None = None) -> Replay:
    """Run the scheduler over `requests` (in the order they reached the engine) and return the executed steps.

    Exactly one of: `admit`, each request's `before_step` (it is added before the first `schedule` after `admit[i]`
    executed steps, as the engine core adds it), or `decide(i, k, open)`, asked at each open step k whether the next
    request i has arrived (an arrival into a quiet engine is added without asking)."""
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
    queue: Queue = deque()
    limit = max_steps if max_steps is not None else 4 * (sum(r.prompt_len + r.output_len for r in requests) + n) + 64

    def add(forced: bool) -> None:
        nonlocal nxt
        s.add(nxt)
        out.admitted[nxt] = len(out.steps)
        out.forced[nxt] = forced
        nxt += 1

    def on_open(p: _Pass) -> bool:
        if decide is None or nxt >= n or not s.would_admit(nxt, p.token_budget, p.input_budget):
            return False
        out.asked[nxt].append(len(out.steps))
        i = nxt
        o = Open(s, min(p.token_budget, p.input_budget), p.sched, lambda now, want: _whatif(s, queue, p, i, now, want))
        if not decide(i, len(out.steps), o):
            return False
        add(False)
        return True

    def top() -> bool:
        if admit is not None:
            while nxt < n and admit[nxt] <= len(out.steps):
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
        step, samples = s.schedule(on_open if decide is not None else None)
        if step.rows:
            out.steps.append(step)
        return step, samples

    def executed(step: Step, samples: dict[int, bool]) -> None:
        for i in s.update(step, samples):
            out.finish[i] = s.reqs[i].finish

    _drive(s, queue, top=top, schedule=schedule, executed=executed)
    return out


# --- traces ------------------------------------------------------------------------------------------------------------------


@dataclass
class Trace:
    """A served run's step trace (`verity-vllm/step-trace/v1`), its requests in the order they reached the engine, with
    the certifier's view of its timing when the trace has a clock: each request's ingress time and each step's release
    time (its output's tokens leaving through the egress certifier), or each kept output's own (`output_t`) where the
    recorder saw them leave one by one."""

    engine: Engine
    ids: list[str]
    requests: list[RequestIO]
    before_step: list[int]          # each request's admission step, as the engine core added it
    steps: list[Step]
    emits: list[bool]               # whether each step emitted a token
    ingress_t: list[float | None]
    release_t: list[float | None]
    header: dict[str, Any] = field(default_factory=dict)
    output_t: list[list[float]] | None = None

    def first_scheduled(self) -> list[int]:
        first = [-1] * len(self.requests)
        for k, st in enumerate(self.steps):
            for i, _, _ in st.rows:
                if first[i] < 0:
                    first[i] = k
        return first

    def clocked(self) -> bool:
        return all(t is not None for t in self.ingress_t) and all(t is not None for t in self.release_t)

    def kept_steps(self) -> list[list[int]]:
        """Each request's kept outputs as the steps that emitted them: a row emits when it reaches the request's frontier
        (its prompt and the outputs kept so far), up to its output length."""
        kept: list[list[int]] = [[] for _ in self.requests]
        for s, st in enumerate(self.steps):
            for i, start, n in st.rows:
                if start + n == self.requests[i].prompt_len + len(kept[i]) and len(kept[i]) < self.requests[i].output_len:
                    kept[i].append(s)
        return kept

    def output_times(self) -> list[list[float]]:
        """When each kept output left: its own time where the trace has one, else its step's release."""
        if self.output_t is not None:
            return self.output_t
        return [[self.release_t[s] for s in ks] for ks in self.kept_steps()]  # type: ignore[misc]

    def origin(self, bucket_s: float) -> int:
        """The session's first bucket, its earliest ingress's, which the certified record's buckets count from (the
        advice module's session-global buckets); 0 without a clock."""
        return min(bucket(t, bucket_s) for t in self.ingress_t) if self.clocked() else 0  # type: ignore[arg-type]

    def certified(self, bucket_s: float = 0.1) -> Certified:
        """What the verifier holds of this run without advice (`Certified`)."""
        if not self.clocked():
            return Certified(self.engine, list(self.requests), bucket_s, None, None)
        o = self.origin(bucket_s)
        return Certified(self.engine, list(self.requests), bucket_s,
                         [bucket(t, bucket_s) - o for t in self.ingress_t],                              # type: ignore[arg-type]
                         [[bucket(t, bucket_s) - o for t in ts] for ts in self.output_times()])


@dataclass(frozen=True)
class Certified:
    """What the verifier holds without advice: the registered engine, each request's certified I/O in the order it reached
    the engine, and, with a clock, each request's ingress bucket and the bucket each of its kept outputs left in, counted
    from the session's first (`Trace.origin`). The decoder reads nothing else."""

    engine: Engine
    requests: list[RequestIO]
    bucket_s: float
    ingress: list[int] | None
    release: list[list[int]] | None

    def windows(self, window_buckets: int) -> list[int]:
        """Each request's certifier window (its ingress bucket's); one window without a clock."""
        return [0] * len(self.requests) if self.ingress is None else [b // window_buckets for b in self.ingress]


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


def records(path: str | Path) -> list[dict[str, Any]]:
    """A step trace's records, from JSON lines or gzipped JSON lines (`.gz`)."""
    p = Path(path)
    text = gzip.decompress(p.read_bytes()).decode() if p.suffix == ".gz" else p.read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def read_step_trace(path: str | Path, **kw: Any) -> Trace:
    return load_step_trace(records(path), **kw)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").split("\n")
    return [json.loads(line) for line in lines[:-1] if line.strip()]


def read_corpus_trace(d: str | Path, *, num_gpu_blocks: int | None = None, block_size: int = 16,
                      egress: str = "engine", ingress: str = "client") -> Trace:
    """One trace of the trace corpus (`benchmarks/network_e2e/traces`, store kind `vllm-trace-corpus/v1`): its
    directory's `steps.jsonl`, `requests.jsonl` and `meta.json` (the server's argv), and `server.log` for the KV pool's
    size unless `num_gpu_blocks` is given. The hook records no arrivals, so a request's engine order and admission step
    are its first scheduling (FCFS schedules new requests in arrival order), and a step's release is the time its output
    was back. With `egress="client"` each kept output left when the client got the streamed chunk that carried it (its
    `chunks`), which is what an egress certifier on the link sees.

    A request's certified ingress is the client's send time (`ingress="client"`, a passive certifier on today's link),
    or with `ingress="warden"` the server's own arrival stamp (`arrival`, its body read), the order vLLM schedules in.
    That is not what design v3's warden certifies: behind it the engine still follows the body read, and the warden's
    release order (a, ordinal) leaves order exceptions to charge (the corpus's `<trace>.released` traces carry it as
    their ingress). `ingress="warden"` is the floor a server admitting in release order at the next bucket would give."""
    d = Path(d)
    argv = json.loads((d / "meta.json").read_text())["server_argv"]

    def opt(name: str) -> str | None:
        return argv[argv.index(name) + 1] if name in argv else None

    if num_gpu_blocks is None:
        m = re.search(r"GPU KV cache size: ([\d,]+) tokens", (d / "server.log").read_text(errors="replace"))
        if m is None:
            raise ValueError(f"{d / 'server.log'}: no 'GPU KV cache size' line; give num_gpu_blocks")
        L = int(opt("--max-model-len") or 0)
        # vLLM logs int(num_blocks / cdiv(L, block_size) * L) (`get_kv_cache_capacity`), which this inverts exactly
        num_gpu_blocks = cdiv(int(m.group(1).replace(",", "")) * cdiv(L, block_size), L)
    engine = Engine(max_num_seqs=int(opt("--max-num-seqs") or 0), max_num_batched_tokens=int(opt("--max-num-batched-tokens") or 0),
                    max_model_len=int(opt("--max-model-len") or 0), num_gpu_blocks=num_gpu_blocks, block_size=block_size,
                    enable_chunked_prefill="--no-enable-chunked-prefill" not in argv,
                    enable_prefix_caching="--enable-prefix-caching" in argv,
                    async_scheduling="--no-async-scheduling" not in argv,
                    long_prefill_token_threshold=int(opt("--long-prefill-token-threshold") or 0))
    raw = [s for s in _jsonl(d / "steps.jsonl") if s["total_num_scheduled_tokens"]]
    reqs = {r["id"]: r for r in _jsonl(d / "requests.jsonl")}
    order, seen = [], set()
    for s in raw:
        for r in s["new"]:
            if r["id"] not in seen and not r.get("resumed"):
                seen.add(r["id"])
                order.append(r["id"])
    if missing := [rid for rid in order if rid not in reqs]:
        raise ValueError(f"{len(missing)} scheduled requests the engine never freed (unfinished at the drain?): {missing[:4]}")
    index = {rid: k for k, rid in enumerate(order)}
    requests = []
    for rid in order:
        r = reqs[rid]
        if r["finish_reason"] not in ("stop", "length") or (r["finish_reason"] == "stop") != bool(r["eos"]):
            raise ValueError(f"request {rid}: finish {r['finish_reason']!r} (eos {r['eos']}); the replay models EOS and length")
        requests.append(RequestIO(prompt_len=int(r["prompt_len"]), max_tokens=int(r["max_tokens"]), output_len=int(r["output_len"]),
                                  eos=bool(r["eos"]), tokens=tuple(r["prompt_ids"]) + tuple(r["output_ids"])))
    steps, emits, release = [], [], []
    for s in raw:
        rows = {r["id"]: (index[r["id"]], int(r["num_computed_tokens"]), int(r["num_scheduled_tokens"]))
                for r in s["new"] + s["cached"] if r["num_scheduled_tokens"]}
        batch = [rid for rid in s.get("batch") or () if rid in rows]
        ids = batch if sorted(batch) == sorted(rows) else list(rows)
        steps.append(Step(rows=[rows[rid] for rid in ids],
                          new=[index[r["id"]] for r in s["new"] if not r.get("resumed")],
                          resumed=[index[r["id"]] for r in s["new"] + s["cached"] if r.get("resumed")],
                          preempted=[index[rid] for rid in s.get("preempted") or () if rid in index],
                          cache_hit={index[r["id"]]: int(r["num_computed_tokens"]) for r in s["new"]
                                     if not r.get("resumed") and r["num_computed_tokens"]}))
        emits.append(any(s.get("sampled") or ()))
        release.append(s.get("t_output"))
    first = [-1] * len(order)
    for k, st in enumerate(steps):
        for i, _, _ in st.rows:
            if first[i] < 0:
                first[i] = k
    if ingress == "client":
        ingress_t = [reqs[rid].get("send_unix") or reqs[rid].get("arrival") for rid in order]
    elif ingress == "warden":
        ingress_t = [reqs[rid].get("arrival") for rid in order]
    else:
        raise ValueError(f"ingress {ingress!r}: client or warden")
    output_t = None
    if egress == "client":
        output_t = []
        for rid, io in zip(order, requests):
            r = reqs[rid]
            ts = [r["send_unix"] + float(s) for s, n in r.get("chunks") or () for _ in range(int(n))]
            if len(ts) != io.output_len:
                raise ValueError(f"request {rid}: its chunks stream {len(ts)} tokens of {io.output_len}")
            output_t.append(ts)
    elif egress != "engine":
        raise ValueError(f"egress {egress!r}: engine or client")
    return Trace(engine=engine, ids=order, requests=requests, before_step=first, steps=steps, emits=emits,
                 ingress_t=ingress_t, release_t=release, output_t=output_t,
                 header={"kind": "header", "schema": "vllm-trace-corpus/v1", "config": dict(engine.__dict__), "source": str(d),
                         "egress": egress, "ingress": ingress})


def order_bits(trace: Trace) -> float:
    """What the engine's arrival order costs beyond the certified ingress order: per request in engine order, an adaptive
    bit for whether it is the earliest certified ingress not yet admitted, and Elias gamma of its rank among those when
    not (ideal code length, bits)."""
    if not trace.clocked():
        return 0.0
    left = sorted(range(len(trace.requests)), key=lambda i: (trace.ingress_t[i], i))
    model, cost = _KT(), 0.0
    for i in range(len(trace.requests)):
        rank = left.index(i)
        f0, f1 = model.freqs(())
        cost -= math.log2((f1 if rank == 0 else f0) / (f0 + f1))
        model.update((), rank == 0)
        if rank:
            cost += gamma_bits(rank)
        left.pop(rank)
    return cost


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
    rp = replay(trace.engine, trace.requests, decide=lambda i, k, _: k >= first[i])
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


def _clocked(cert: Certified) -> tuple[list[int], list[list[int]]]:
    if cert.ingress is None or cert.release is None:
        raise ValueError("the trace has no certifier clock (ingress_t and release_t)")
    return cert.ingress, cert.release


class _Clock:
    """The certified timing a decision is coded in: request i's ingress bucket against the release buckets around the
    step. The engine core scheduled it after processing the outputs it has (those of every step but the one in flight
    under async scheduling) and before any later output left, so the newest output it has processed left at about when it
    scheduled or before (`prev`), and the first output it hasn't, of a request in the engine or of i, after (`next`)."""

    def __init__(self, cert: Certified) -> None:
        self.b_in, self.out = _clocked(cert)

    def context(self, i: int, o: Open) -> tuple[int, int]:
        s = o.sched
        p = max((self.out[r.i][min(r.n_out, len(self.out[r.i])) - 1] for r in s.reqs if r.n_out and self.out[r.i]),
                default=None)
        ahead = [self.out[i][0]] if self.out[i] else []
        for r in (*s.running, *s.waiting, *s.skipped):
            m = r.n_out + r.placeholders
            if m < len(self.out[r.i]):
                ahead.append(self.out[r.i][m])
        n = min(ahead, default=None)
        return (2 if p is None else max(-1, min(1, self.b_in[i] - p)), 2 if n is None else int(self.b_in[i] > n))


class _Egress:
    """Request i's certified egress against the engine run on from the open step (`Open.whatif`), i arriving now or
    before the next step and nothing after it. Under FCFS that run is the served one for i and every request ahead of
    it: a later arrival goes behind them in the running order, so takes none of their budget, and is preempted before
    them. So under the true hypothesis each of i's outputs leaves in the bucket of an earlier request's output from the
    same step, which the certified egress shows; of any of them, since a frontend streams one step's outputs over some
    milliseconds. They are compared up to i's first change of bucket, where some earlier request's must change too (2: no
    earlier request's output to compare with)."""

    def __init__(self, cert: Certified) -> None:
        _, self.out = _clocked(cert)

    def lines_up(self, i: int, em: Emitted, edge: int | None) -> int:
        a, checked = self.out[i], 0
        for e in em:
            t = e.get(i)
            if t is None:
                continue
            bs = {self.out[j][m] for j, m in e.items() if j < i and m < len(self.out[j])}
            if bs:
                if (a[t] not in bs) if edge is None or t < edge else (bs == {a[0]}):
                    return 0
                checked += 1
        return 1 if checked else 2

    def context(self, i: int, o: Open) -> tuple[int, int]:
        a = self.out[i]
        if not a:
            return (2, 2)
        edge = next((t for t in range(1, len(a)) if a[t] != a[t - 1]), None)
        want = len(a) if edge is None else edge + 1
        return (self.lines_up(i, o.whatif(True, want), edge), self.lines_up(i, o.whatif(False, want), edge))


# --- an adaptive binary arithmetic code for the decisions (Witten, Neal and Cleary's coder, KT estimates per context) --

_FULL, _HALF, _QTR = 1 << 32, 1 << 31, 1 << 30


class _KT:
    """Krichevsky-Trofimov counts per context, from `prior`'s where given (pseudo-observations of 0s and 1s)."""

    def __init__(self, prior: dict[Any, tuple[int, int]] | None = None) -> None:
        self.n: dict[Any, tuple[int, int]] = dict(prior or {})

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
    o = trace.origin(bucket_s)
    return ([(bucket(t, bucket_s) - o) // window_buckets for t in trace.ingress_t],   # type: ignore[arg-type]
            [(bucket(t, bucket_s) - o) // window_buckets for t in trace.release_t])   # type: ignore[arg-type]


@dataclass
class Code:
    """The residual advice under the adaptive code: one bit string per certifier window (a request's decisions go in its
    ingress window's), and what an ideal coder would spend (`ideal`, bits)."""

    contexts: str                   # one of CONTEXTS
    bucket_s: float
    window_buckets: int
    bits: dict[int, list[int]]
    ideal: dict[int, float]


#: the adaptive code's contexts: `plain`, the request's passes so far (`min(passed, 2)`); `clocked`, those and its ingress
#: bucket against the release buckets around the step (`_Clock`); `egress`, whether its release buckets line up with
#: the replay's forks where it arrived now or a step later (`_Egress`), and the clock's where those can't tell
CONTEXTS = ("plain", "clocked", "egress")
#: the egress verdicts that tell the decision (`_Egress.context`: lines up now, a step later), each a prior of `_PRIOR`
#: observations of its decision in the adaptive code
_TELLS = {(1, 0): True, (0, 1): False, (0, 0): False}
_PRIOR = 8


def _model(contexts: str) -> _KT:
    return _KT({c: (0, _PRIOR) if v else (_PRIOR, 0) for c, v in _TELLS.items()} if contexts == "egress" else None)


def _coder(cert: Certified, contexts: str) -> Callable[[int, int, Open], tuple[int, ...]]:
    if contexts not in CONTEXTS:
        raise ValueError(f"contexts {contexts!r}: one of {CONTEXTS}")
    clock = _Clock(cert) if contexts != "plain" else None
    egress = _Egress(cert) if contexts == "egress" else None

    def context(i: int, passed: int, o: Open) -> tuple[int, ...]:
        if clock is None:
            return (min(passed, 2),)
        if egress is None:
            return (min(passed, 2),) + clock.context(i, o)
        e = egress.context(i, o)
        return e if e in _TELLS else e + clock.context(i, o)

    return context


def encode(trace: Trace, *, contexts: str = "egress", bucket_s: float = 0.1, window_buckets: int = 600) -> Code:
    """Code each decision, as the replay asks it, with the adaptive binary code in `contexts`. Refuses a trace the replay
    doesn't reproduce."""
    first = trace.first_scheduled()
    if any(f < 0 for f in first):
        raise ValueError(f"requests never scheduled: {[trace.ids[i] for i, f in enumerate(first) if f < 0][:8]}")
    cert = trace.certified(bucket_s)
    context = _coder(cert, contexts)
    rw = cert.windows(window_buckets)
    model, enc, ideal = _model(contexts), {}, {}
    passed = [0] * len(trace.requests)

    def decide(i: int, k: int, o: Open) -> bool:
        bit = k >= first[i]
        ctx = context(i, passed[i], o)
        f0, f1 = model.freqs(ctx)
        w = rw[i]
        enc.setdefault(w, _Encoder()).put(bit, f0, f1)
        ideal[w] = ideal.get(w, 0.0) - math.log2((f1 if bit else f0) / (f0 + f1))
        model.update(ctx, bit)
        passed[i] += 1
        return bit

    m = compare(replay(trace.engine, trace.requests, decide=decide).steps, trace.steps)
    if m.first_mismatch is not None:
        raise ValueError(f"the replay diverges from the trace at step {m.first_mismatch} ({m.matched}/{m.steps} steps match)")
    return Code(contexts, bucket_s, window_buckets, {w: e.finish() for w, e in enc.items()}, ideal)


def decode(cert: Certified, code: Code) -> Replay:
    """Replay from the residual advice and what the verifier holds without it (`Certified`, at the code's bucket)."""
    if cert.bucket_s != code.bucket_s:
        raise ValueError(f"the code is in {code.bucket_s} s buckets, the certified record in {cert.bucket_s} s")
    context = _coder(cert, code.contexts)
    rw = cert.windows(code.window_buckets)
    model, dec = _model(code.contexts), {}
    passed = [0] * len(cert.requests)

    def decide(i: int, k: int, o: Open) -> bool:
        ctx = context(i, passed[i], o)
        f0, f1 = model.freqs(ctx)
        w = rw[i]
        if w not in dec:
            dec[w] = _Decoder(code.bits.get(w, []))
        bit = dec[w].get(f0, f1)
        model.update(ctx, bit)
        passed[i] += 1
        return bit

    return replay(cert.engine, cert.requests, decide=decide)


# --- bits per certifier window ----------------------------------------------------------------------------------------


def width(n: int) -> int:
    """Bits for a value in [0, n)."""
    return math.ceil(math.log2(n)) if n > 1 else 0


def v0_widths(trace: Trace, *, window_buckets: int) -> dict[str, int]:
    """I1 v0's fields at fixed width (a stand-in for `advice.bits`): per request its id (an index), prompt_len,
    output_len and finish; per step its row count and release bucket; per row its request, kind, start, n, cache_hit and
    preempted."""
    e = trace.engine
    R = max(len(trace.requests), 1)
    req = width(R) + width(e.max_model_len + 1) + width(max((r.max_tokens for r in trace.requests), default=1) + 1) + 1
    row = width(R) + 1 + width(e.max_model_len) + width(e.max_num_batched_tokens + 1) + width(e.max_model_len) + 1
    step = width(e.max_num_seqs + 1) + width(window_buckets)
    return {"request": req, "row": row, "step": step}


def v0_advice(trace: Trace, v0: Any, *, bucket_s: float = 0.1) -> Any:
    """The trace as advice v0, built with its module `v0` (`verity.core.protocols.certifier.advice`): the
    trace's rows in its batch order, each hit's sources from the replay's block map, and each kept output released in its
    step's release bucket. v0 validates it as it builds it."""
    rp = replay(trace.engine, trace.requests, admit=trace.before_step)
    m = compare(rp.steps, trace.steps)
    if m.first_mismatch is not None:
        raise ValueError(f"the replay diverges from the trace at step {m.first_mismatch}, so it has no sources to give")
    first = trace.first_scheduled()
    order = sorted(range(len(trace.requests)), key=lambda i: (first[i], [r[0] for r in trace.steps[first[i]].rows].index(i)))
    idx = {i: k for k, i in enumerate(order)}
    reqs = tuple(v0.RequestAdvice(id=trace.ids[i], prompt_len=trace.requests[i].prompt_len,
                                  output_len=trace.requests[i].output_len, finish="eos" if trace.requests[i].eos else "length")
                 for i in order)
    kept, times, o = trace.kept_steps(), trace.output_times(), trace.origin(bucket_s)
    steps, release, at = [], [], [0] * len(reqs)
    for s, (st, mine) in enumerate(zip(trace.steps, rp.steps)):
        rows = []
        for i, start, n in st.rows:
            q = reqs[idx[i]]
            rows.append(v0.Row(request=idx[i], kind="decode" if n == 1 and start >= q.prompt_len else "prefill", start=start,
                               n=n, cache_hit=mine.cache_hit.get(i, 0), preempted=i in mine.resumed,
                               sources=tuple(v0.Source(idx[o], k) for o, k in mine.sources.get(i, ()))))
            if at[i] < len(kept[i]) and kept[i][at[i]] == s:
                t = times[i][at[i]]
                at[i] += 1
                release.append(0 if t is None else bucket(t, bucket_s) - o)
        steps.append(v0.StepAdvice(rows=tuple(rows)))
    return v0.Advice(requests=reqs, steps=tuple(steps), release=tuple(release))


def v0_parts(v0: Any, adv: Any) -> dict[str, int] | None:
    """I1 v0's bits by record (`header`, `requests`, `steps`, `release`), from the advice module's layout (`_layout`,
    None without one). The residual advice replaces only the step records: the verifier reads the rest from the
    certified record."""
    layout = getattr(v0, "_layout", None)
    if layout is None:
        return None
    lay = layout(adv)
    return {"header": lay.header, "requests": sum(lay.requests), "steps": sum(lay.steps),
            "release": lay.release * len(adv.release)}


def framed_bits(n: int) -> int:
    """A window's residual of `n` bits with the advice module's `uvar` length prefix (a u6 width, then the length), for
    a transport that doesn't frame windows."""
    return n + 6 + n.bit_length()


def table(trace: Trace, *, bucket_s: float = 0.1, window_buckets: int = 600, v0: Any = None) -> dict[str, Any]:
    """Per certifier window: I1 v0's bits (`advice.bits_by_window` when the module is given as `v0`, else
    `v0_widths`), the steps the replay matched, and the residual advice's bits under each code (Elias gamma; the adaptive
    code, unclocked and clocked), each code checked by decoding it back to the trace's steps."""
    truth = replay(trace.engine, trace.requests, admit=trace.before_step)
    adv = v0_advice(trace, v0, bucket_s=bucket_s) if v0 is not None else None
    charge = v0.bits_by_window(adv, window_buckets) if adv is not None else None
    ds, rp = decisions(trace)
    g = gamma_values(rp)
    codes = {"adaptive" if c == "plain" else f"adaptive_{c}": encode(trace, contexts=c, bucket_s=bucket_s,
                                                                     window_buckets=window_buckets)
             for c in CONTEXTS if c == "plain" or trace.clocked()}
    cert = trace.certified(bucket_s)
    decoded = {name: compare(decode(cert, c).steps, trace.steps).__dict__ for name, c in codes.items()}
    rw, sw = windows(trace, bucket_s=bucket_s, window_buckets=window_buckets)
    v0w = v0_widths(trace, window_buckets=window_buckets)
    rows = []
    for w in sorted(set(rw) | set(sw)):
        ks = [k for k, x in enumerate(sw) if x == w]
        rs = [i for i, x in enumerate(rw) if x == w]
        row = {"window": w, "requests": len(rs), "steps": len(ks), "rows": sum(len(trace.steps[k].rows) for k in ks),
               "matched": sum(1 for k in ks if k < len(truth.steps) and truth.steps[k].key() == trace.steps[k].key()),
               "forced": sum(1 for i in rs if rp.forced[i]), "decisions": sum(1 for d in ds if rw[d.request] == w),
               "v0_bits": charge.get(w, 0) if charge is not None else
                          len(rs) * v0w["request"] + sum(v0w["step"] + len(trace.steps[k].rows) * v0w["row"] for k in ks),
               "gamma_bits": sum(gamma_bits(g[i] + 1) for i in rs if g[i] is not None)}
        for name, c in codes.items():
            row[f"{name}_bits"] = len(c.bits.get(w, []))
            row[f"{name}_ideal"] = round(c.ideal.get(w, 0.0), 2)
        rows.append(row)
    return {"engine": dict(trace.engine.__dict__), "requests": len(trace.requests), "steps": len(trace.steps),
            "match_before_step": compare(truth.steps, trace.steps).__dict__, "decoded": decoded,
            "v0": "advice.bits_by_window" if charge is not None else {"widths": v0w},
            "v0_parts": v0_parts(v0, adv) if adv is not None else None, "bucket_s": bucket_s,
            "window_buckets": window_buckets, "windows": rows, "order_bits": round(order_bits(trace), 2),
            "passed": [x for x in g if x is not None]}


#: advice v0's module (`verity.core.protocols.certifier.advice`)
V0_PATH = Path(__file__).resolve().parents[3] / "verity" / "core" / "protocols" / "certifier" / "advice.py"


def load_v0(spec: str | Path | None) -> Any:
    """The advice module from `spec`, a path to its `advice.py`, else from `V0_PATH` when it is there, else None
    (`table` falls back to `v0_widths`)."""
    import importlib.util
    import sys
    if not spec:
        if not V0_PATH.exists():
            return None
        spec = V0_PATH
    s = importlib.util.spec_from_file_location("advice_v0", spec)
    assert s is not None and s.loader is not None
    sys.modules["advice_v0"] = mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    return mod


def read_any(p: Path, *, egress: str = "engine", ingress: str = "client") -> Trace:
    if p.is_dir():
        return read_corpus_trace(p, egress=egress, ingress=ingress)
    if egress != "engine" or ingress != "client":
        raise ValueError(f"{p}: a step trace has only its steps' release times (egress engine) and its own ingress")
    tok = p.with_suffix(".tokens.json")
    return read_step_trace(p, tokens=json.loads(tok.read_text()) if tok.exists() else None)


def main(argv: Sequence[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Bits per certifier window: I1 v0 against the replay's residual advice.")
    ap.add_argument("traces", nargs="+", type=Path,
                    help="step traces (*.jsonl; <name>.tokens.json beside one with prefix caching) or trace-corpus "
                         "trace dirs")
    ap.add_argument("--bucket-ms", type=float, default=100.0)
    ap.add_argument("--window-buckets", type=int, default=600)
    ap.add_argument("--v0", help="path to the advice module's advice.py (default: V0_PATH in the tree, else fixed "
                                 "widths)")
    ap.add_argument("--egress", choices=("engine", "client"), default="engine",
                    help="a trace-corpus output's release time: its step's (engine) or its streamed chunk's (client)")
    ap.add_argument("--ingress", choices=("client", "warden"), default="client",
                    help="a trace-corpus request's certified ingress: the client's send time (client) or the engine's "
                         "arrival stamp (warden: the floor a server admitting in the warden's release order would "
                         "give; today's warden certifies the release order, which a trace's .released view carries)")
    ap.add_argument("--out", type=Path, help="write {trace: table} as JSON")
    a = ap.parse_args(argv)
    v0 = load_v0(a.v0)
    out: dict[str, Any] = {}
    per_window = []
    print("| trace | requests | steps | replay matched | I1 v0 bits | v0 step records | gamma | adaptive | adaptive clocked "
          "| adaptive egress | best bits/request | order bits |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in a.traces:
        name = p.stem if p.is_file() else f"{p.resolve().parent.name}/{p.name}"
        tr = read_any(p, egress=a.egress, ingress=a.ingress)
        try:
            tb = out[name] = table(tr, bucket_s=a.bucket_ms / 1000, window_buckets=a.window_buckets, v0=v0)
        except (ValueError, RuntimeError) as e:
            try:
                m = compare(replay(tr.engine, tr.requests, admit=tr.before_step).steps, tr.steps)
                where = f"{m.matched}/{m.steps} (first mismatch {m.first_mismatch})"
                match: dict[str, Any] = m.__dict__
            except (ValueError, RuntimeError) as e2:
                where, match = f"replay refused: {e2}", {"error": str(e2)}
            out[name] = {"error": str(e), "requests": len(tr.requests), "steps": len(tr.steps), "match_before_step": match}
            print(f"| {name} | {len(tr.requests)} | {len(tr.steps)} | {where} | {e} |" + " |" * 7, flush=True)
            continue
        s = {k: sum(w[k] for w in tb["windows"] if w.get(k) is not None) for k in tb["windows"][0]}
        m = tb["match_before_step"]
        best = min(s[k] for k in ("adaptive_bits", "adaptive_clocked_bits", "adaptive_egress_bits") if k in s)
        steps = tb["v0_parts"]["steps"] if tb["v0_parts"] else "-"
        print(f"| {name} | {tb['requests']} | {tb['steps']} | {m['matched']}/{m['steps']} | "
              f"{s['v0_bits']} | {steps} | {s['gamma_bits']} | {s['adaptive_bits']} | {s.get('adaptive_clocked_bits', '-')} | "
              f"{s.get('adaptive_egress_bits', '-')} | {best / max(tb['requests'], 1):.2f} | {tb['order_bits']} |", flush=True)
        code = "adaptive_egress_bits" if "adaptive_egress_bits" in s else "adaptive_bits"
        for w in tb["windows"]:
            per_window.append(f"| {name} | {w['window']} | {w['requests']} | {w['steps']} | {w['matched']}/{w['steps']} | "
                              f"{w['decisions']} | {w['forced']} | {w['v0_bits']} | {w[code]} | {framed_bits(w[code])} |")
    print("\n| trace | window | requests | steps | replay matched | decisions | forced | I1 v0 bits | residual bits "
          "| residual framed |\n|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(per_window))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

