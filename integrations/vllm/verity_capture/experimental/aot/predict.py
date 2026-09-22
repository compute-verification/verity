"""The deterministic predictor: φ (+ hints) -> the structural execution schedule.

    schedule = predict_schedule(phi, hints)   # list[StepPlan]

A `StepPlan` is what `BuildProgram` consumes per engine step: the ordered token rows (request index, absolute position),
the sampled (logits) rows in sampler order, and for each sampled row whether the engine serves it (prompt complete after
the step) or discards it (partial-prefill chunk end).  Everything the predictor gets right costs zero advice; `EncodeHints`
records only the residual against the observed record.

PREDICTOR_ID = "dense-eager-fcfs/v2": a first-come-first-served simulation of vLLM's V1 scheduler as far as it is
reconstructible from φ alone —
    * requests are admitted in (arrive_step, workload order) while `len(running) < max_num_seqs`;
    * per step, the scheduler's running list comes first in admission order (one decode row each once their prompt is in
      the cache, else the next prefill chunk), then the newly admitted ones (their first chunk);
    * the model runner then orders the batch by a *stable sort on the per-request scheduled token count, ascending*
      (decodes first, the longest prefill chunk last) — v2, SOURCE-INSPECTED: `vllm/v1/worker/gpu/model_runner.py::
      sort_batch_req_ids` (see PREDICTORS below); v1 kept the scheduler order and mispredicts the one step of B8-mixed c256
      in which two new prompts of different length arrive together (MEASURED 2026-09-18: 1 step override, 121 naive bits → 0);
    * the step's token budget is `max_num_batched_tokens`; a prompt that does not fit is chunked (chunked prefill);
    * one logits row per request present in the step, at its last token row, in row order; it is served iff the request's
      prompt is complete after the step (vLLM's `discard_request_mask` rule);
    * a request stops after `served` tokens (`max_tokens` unless `hints.served` says otherwise).
A step listed in `hints.steps` replaces the simulated layout of that step entirely (the simulation state is re-synchronised
from the given rows); `hints.discarded` marks additional dropped samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .hints import Hints, RowRun, StepRows
from .phi import Phi

PREDICTOR_ID = "dense-eager-fcfs/v2"
# Every predictor the decoder can run.  `h.predictor` names the one the hints correct, so a decode with hints for v1 runs
# v1 — the corrections are meaningful only against the predictor that produced the residual.  v1 (no input-preparation sort)
# is kept as the MEASURED exercise of the non-empty hint path: on B8-mixed c256 it needs 24 step overrides (see REPORT.md).
PREDICTORS: dict[str, dict] = {"dense-eager-fcfs/v2": {"order": "sort"}, "dense-eager-fcfs/v1": {"order": "scheduler"},
                               "dense-eager-fcfs/x-legacy-batch": {"order": "persistent-batch"},
                               "dense-eager-fcfs/v3-kv": {"order": "sort", "kv_blocks": True},
                               "dense-eager-fcfs/v4-pool": {"order": "sort", "kv_blocks": True, "pool": True},
                               "dense-eager-fcfs/v5-seat": {"order": "sort", "kv_blocks": True, "pool": True, "seat": True}}
# v5-seat = v4-pool + ONE rule, SOURCE-INSPECTED 2026-09-18 at the pinned install (vllm/v1/core/sched/scheduler.py:590-603 and
# :816-819): under async scheduling a request that reached max_tokens in step t is still in `self.running` at schedule(t+1) —
# the running loop skips it (`num_computed_tokens + 2 - num_output_placeholders >= num_prompt_tokens + max_tokens` → `continue`,
# no rows) but the waiting loop's `num_running = len(self.running)` COUNTS it against `max_num_running_reqs`; update_from_output(t)
# removes it only after that schedule.  So a max_tokens stop holds its SEAT (not only its blocks, which v4-pool already held)
# through the next schedule; an EOS stop is removed after its placeholder step as before.  Found by C-codec's mix16 record
# (16 requests, max_num_seqs = 8 binding): v4-pool admits the next waiting request one step early each time a request stops at
# max_tokens while the seat cap binds (3 `steps` residual at 36/46/58; `corpus/gen-v5/`).  A held seat is still the last entry of
# `running` when it stopped last, so `running.pop()` may pick it as a preemption victim — modelled literally (counts as a preemption,
# its blocks return, it is not re-queued); that edge has not been observed.
# v4-pool = v3-kv with the KV cache modelled as vLLM's BlockPool instead of a per-request block count, SOURCE-INSPECTED 2026-09-18
# (block_pool.py::get_new_blocks / free_blocks / touch / cache_full_blocks / _maybe_evict_cached_block, kv_cache_utils.py::
# FreeKVCacheBlockQueue popleft_n / prepend_n / append_n, single_type_kv_cache_manager.py::get_num_blocks_to_allocate / free
# (reversed), kv_cache_manager.py::allocate_slots / get_computed_blocks, scheduler.py::update_from_output stale-output delivery):
#   * blocks are ref-counted: a prefix-cache hit references the producer's block (the pool holds DISTINCT blocks — v3-kv's
#     ⌈tokens/16⌉-per-request count overstates occupancy under prefix caching + KV pressure; C-codec's pp4 record);
#   * the free queue is ordered: allocation pops the FRONT; a freed block goes to the FRONT when it has no hash (LIFO reuse) and to
#     the BACK when it is hashed (FIFO → LRU eviction); a request's blocks are freed tail-first; popping a hashed block evicts its hash;
#   * a full block is hashed at *scheduling* time once all its tokens are known — prompt blocks by content, blocks that contain
#     generated tokens by (request, block index) (their content is a circuit value: only the SAME request can hit them, on a resume);
#     under async scheduling the tokens sampled in the previous step are not yet known (lag 1);
#   * admission looks the request's full blocks up in the hash map (at most (num_tokens − 1) // block_size), also blocks that sit
#     in the free queue (touched back: ref 0 → 1); capacity checks count those evictable hits as needed blocks;
#   * a request that stopped at max_tokens keeps its blocks through the NEXT schedule (the finish is processed after it; if that
#     schedule would be empty it is not an observed step and the blocks are free for the following one); an EOS stop frees them right
#     after its placeholder step (MEASURED pre6 steps 120–122 vs preempt4 159–160).
# Everything else is v3-kv.  With prefix caching off the pool degenerates to the v3-kv count plus the finish-timing rule.
# v3-kv = v2 + the KV block accounting of the V1 scheduler, SOURCE-INSPECTED 2026-09-18 at the pinned install
# (vllm/v1/core/sched/scheduler.py:686-760 running loop, :809-1170 waiting loop, :1440-1481 _preempt_request;
#  vllm/v1/core/kv_cache_manager.py:440-525 allocate_slots; single_type_kv_cache_manager.py:188-210; block_pool.py:190 null block).
# It needs φ.engine.num_gpu_blocks (the engine's KV capacity in blocks — a pre-execution engine fact; the RunHeader does not record
# it today, so it is supplied as an explicit baseline fact).  Rules: usable blocks = num_gpu_blocks - 1 (block 0 is the null block);
# a running request needs cdiv(computed + new, block_size) - held blocks; when they are not free, the LAST running request is
# preempted (running.pop(), its blocks freed, num_computed_tokens = 0, prepended to the waiting queue) until the request fits or it
# is itself the victim; a step with a preemption admits nothing; a waiting request is admitted only if its FULL sequence
# (prompt + generated so far; scheduler_reserve_full_isl = True) fits in the free blocks (watermark 0.0 = 0 blocks) and its chunk's
# blocks are allocated; a re-admitted (preempted) request recomputes prompt + generated tokens from position 0 in one chunk (budget
# permitting); a sample is served only when the request's whole sequence is computed.  With num_gpu_blocks unknown the accounting is
# skipped and v3-kv is v2.
# v2 is vLLM's mechanism, SOURCE-INSPECTED 2026-09-18 at the pinned install (vllm 0.28.1rc1.dev472+gd9105ea80,
# vllm/v1/worker/gpu/model_runner.py::sort_batch_req_ids, called from the per-step batch construction whose `req_ids` the
# observer reads as `execute_model_state.input_batch.req_ids`):
#     key = (not draft_tokens.get(r), num_scheduled_tokens[r] != decode_query_len, num_scheduled_tokens[r]);  sorted(...)  # stable
# i.e. without speculative decoding (decode_query_len = 1): decodes first, then ascending scheduled-token count, ties in the
# scheduler's `num_scheduled_tokens` insertion order (running list in admission order, then the newly admitted).  That is
# `_runner_order` below.  `x-legacy-batch` models the *other* runner in the same install (gpu_model_runner.py::_update_states:
# persistent InputBatch, lowest-hole filling, condense(), no reorder for FlashAttention); it is NOT the runner these profiles
# use — kept only as the MEASURED negative: the encoder charges it 563–773 naive bits on every b8 record (`corpus/enc-legacy/`).


@dataclass
class StepPlan:
    step: int
    rows: list[tuple[int, int]]                    # (request index, absolute position), in vLLM row order
    logits_rows: list[tuple[int, int]]             # (request index, position) sampled, in sampler row order
    served: list[bool]                             # per logits row
    phase: str = ""                                # diagnostic: prefill | decode | mixed
    hits: list[tuple[int, int, int, int]] = field(default_factory=list)   # (request, first pos, count, producer request): prefix-cache
    #                                               hits at this step's admission — K/V of those positions is the producer's (v4-pool)

    def runs(self) -> tuple[RowRun, ...]:
        out: list[RowRun] = []
        for r, p in self.rows:
            if out and out[-1].request == r and out[-1].first + out[-1].count == p:
                out[-1] = RowRun(r, out[-1].first, out[-1].count + 1)
            else:
                out.append(RowRun(r, p, 1))
        return tuple(out)

    def default_logits_rows(self) -> list[tuple[int, int]]:
        last: dict[int, int] = {}
        order: list[int] = []
        for r, p in self.rows:
            if r not in last:
                order.append(r)
            last[r] = p
        return [(r, last[r]) for r in order]


@dataclass
class _Req:
    idx: int
    prompt_len: int
    arrive: int
    target: int                     # tokens to serve
    computed: int = 0               # tokens in the KV cache
    served: int = 0                 # tokens returned so far
    admitted: bool = False
    done: bool = False
    blocks: int = 0                 # KV blocks held (v3-kv)
    preemptions: int = 0
    blk: list[int] = field(default_factory=list)   # block ids held, in position order (v4-pool)
    n_hashed: int = 0               # of those, how many leading blocks carry a hash (vLLM num_cached_block)
    served_steps: list[int] = field(default_factory=list)   # steps at which a served sample occurred (token knowledge under async lag)
    finished_at: int = -1           # step of the last row once done

    @property
    def total(self) -> int:         # vLLM request.num_tokens: prompt + generated so far (what a recompute re-feeds)
        return self.prompt_len + self.served

    def known_at(self, step: int, lag: int) -> int:
        """Tokens the scheduler knows at schedule(step): prompt + samples served at steps <= step - 1 - lag (async: lag 1)."""
        return self.prompt_len + sum(1 for s in self.served_steps if s <= step - 1 - lag)


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


class _Pool:
    """vLLM's BlockPool + FreeKVCacheBlockQueue as far as admission, preemption and prefix-cache hits depend on them (v4-pool)."""

    def __init__(self, num_blocks: int, caching: bool) -> None:
        self.free: list[int] = list(range(1, num_blocks))          # block 0 is the null block; front = next to allocate
        self.ref: dict[int, int] = {b: 0 for b in range(1, num_blocks)}
        self.key: dict[int, object] = {}                            # block -> hash key while cached
        self.cached: dict[object, list[int]] = {}                   # hash key -> blocks (first = the one a lookup returns)
        self.writer: dict[int, int] = {}                            # block -> request whose rows wrote its K/V
        self.caching = caching

    @property
    def num_free(self) -> int:
        return len(self.free)

    def evict(self, b: int) -> None:
        k = self.key.pop(b, None)
        if k is not None:
            lst = self.cached[k]
            lst.remove(b)
            if not lst:
                del self.cached[k]

    def get_new_blocks(self, n: int, writer: int) -> list[int]:
        assert n <= len(self.free), "pool: allocation beyond the free queue"
        ret, self.free = self.free[:n], self.free[n:]
        for b in ret:
            if self.caching:
                self.evict(b)
            self.ref[b] = 1
            self.writer[b] = writer
        return ret

    def touch(self, blocks: list[int]) -> None:
        for b in blocks:
            if self.ref[b] == 0:
                self.free.remove(b)
            self.ref[b] += 1

    def free_blocks(self, ordered: list[int]) -> None:
        first, last = [], []
        for b in ordered:
            self.ref[b] -= 1
            if self.ref[b] == 0:
                (last if (self.caching and b in self.key) else first).append(b)
        self.free = first + self.free + last

    def lookup(self, key: object) -> int | None:
        lst = self.cached.get(key)
        return lst[0] if lst else None

    def cache(self, b: int, key: object) -> None:
        if b in self.key:
            return
        self.key[b] = key
        self.cached.setdefault(key, []).append(b)

    def evictable(self, blocks: list[int]) -> int:
        return sum(1 for b in blocks if self.ref[b] == 0)


def _runner_order(rows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """vLLM's `sort_batch_req_ids` without speculative decoding: the per-request groups of the scheduler order, stably sorted by
    scheduled token count (key `(num != 1, num)` == ascending `num` since 1 is the minimum)."""
    groups: list[list[tuple[int, int]]] = []
    for rp in rows:
        if groups and groups[-1][0][0] == rp[0]:
            groups[-1].append(rp)
        else:
            groups.append([rp])
    groups.sort(key=len)              # stable
    return [rp for g in groups for rp in g]


class _Batch:
    """The legacy runner's persistent InputBatch as far as row order is concerned (`x-legacy-batch`): index -> request (None = hole)."""

    def __init__(self) -> None:
        self.slots: list[int | None] = []

    def update(self, finished: list[int], new: list[int]) -> None:
        removed: list[int] = []
        for r in finished:                                   # _update_states: finished_req_ids -> remove_request
            i = self.slots.index(r)
            self.slots[i] = None
            removed.append(i)
        removed.sort(reverse=True)                           # BatchUpdateBuilder.removed: descending; pop -> lowest
        for r in new:                                        # add_request: lowest hole, else append
            if removed:
                self.slots[removed.pop()] = r
            else:
                self.slots.append(r)
        # condense(): move the last live request down into the lowest remaining hole
        while removed:
            last = len(self.slots) - 1
            while last >= 0 and self.slots[last] is None:
                last -= 1
            empty = removed[-1]
            if empty >= last:
                break
            removed.pop()
            self.slots[empty], self.slots[last] = self.slots[last], None
        while self.slots and self.slots[-1] is None:
            self.slots.pop()

    def order(self, rows: list[tuple[int, int]]) -> list[tuple[int, int]]:
        by_req: dict[int, list[tuple[int, int]]] = {}
        for rp in rows:
            by_req.setdefault(rp[0], []).append(rp)
        out: list[tuple[int, int]] = []
        for r in self.slots:
            if r is not None and r in by_req:
                out.extend(by_req.pop(r))
        for rest in by_req.values():                         # requests the batch model does not know (should not happen)
            out.extend(rest)
        return out

    def resync(self, req_order: list[int]) -> None:
        seen: list[int] = []
        for r in req_order:
            if r not in seen:
                seen.append(r)
        self.slots = list(seen)


def _served_targets(phi: Phi, hints: Hints) -> dict[int, int]:
    t = {i: int(r.get("max_tokens") or 0) for i, r in enumerate(phi.logical_requests())}
    for s in hints.served:
        if s.request not in t:
            raise ValueError(f"hints.served names request {s.request}, the workload has {len(t)}")
        t[s.request] = s.served
    return t


def predict_schedule(phi: Phi, hints: Hints | None = None, *, max_steps: int = 1_000_000) -> list[StepPlan]:
    hints = hints or Hints(predictor=PREDICTOR_ID)
    pid = hints.predictor or PREDICTOR_ID
    if pid not in PREDICTORS:
        raise ValueError(f"hints are for predictor {hints.predictor!r}; known: {sorted(PREDICTORS)}")
    order = PREDICTORS[pid]["order"]
    batch = _Batch()                  # x-legacy-batch only: the legacy runner's persistent batch (index -> request)
    finished_prev: list[int] = []     # requests that finished in the previous step (removed at this step's update)
    reqs_json = phi.logical_requests()          # n > 1 forks are separate engine requests (phi.logical_requests)
    if not reqs_json:
        raise ValueError("φ.workload has no requests")
    targets = _served_targets(phi, hints)
    reqs = [_Req(i, len(r["prompt_token_ids"]), int(r.get("arrive_step") or 0), targets[i]) for i, r in enumerate(reqs_json)]
    max_seqs = int(phi.engine.get("max_num_seqs") or len(reqs))
    budget_total = int(phi.engine.get("max_num_batched_tokens") or 1 << 30)
    chunked = phi.engine.get("enable_chunked_prefill")
    # scheduler.py:624 / :1020: a prefill chunk is capped at long_prefill_token_threshold when that is > 0 (before the budget cap)
    long_cap = int(phi.engine.get("long_prefill_token_threshold") or 0)
    overrides: dict[int, StepRows] = {s.step: s for s in hints.steps}
    discarded = {(d.request, d.step) for d in hints.discarded}

    kv = bool(PREDICTORS[pid].get("kv_blocks")) and phi.engine.get("num_gpu_blocks") is not None
    block_size = int(phi.engine.get("block_size") or 16)
    # prefix caching (v3-kv, SOURCE-INSPECTED kv_cache_manager.py:246-296, :548-562; single_type_kv_cache_manager.py:447-495, :752-770):
    # at admission a request's leading FULL blocks whose chained content hash is cached are skipped (at most (num_tokens-1)//block_size
    # blocks: the last token is always recomputed for its logits); blocks are cached at *scheduling* time for every block that is full
    # after the scheduled chunk (so a request admitted later in the same step hits blocks another request computes in that step).
    # Only blocks made of prompt tokens are content-addressable from φ (a block with generated tokens hashes circuit values); no
    # eviction is modelled (a hit lost to eviction would surface as a `steps` residual).
    pc = bool(PREDICTORS[pid].get("kv_blocks")) and bool(phi.engine.get("enable_prefix_caching"))
    block_keys: list[list[int]] = []                       # request -> content key of its j-th block (prompt-only blocks)
    for rj in reqs_json:
        hk, keys, toks = 0, [], rj["prompt_token_ids"]
        for j in range(len(toks) // block_size):
            for tok in toks[j * block_size:(j + 1) * block_size]:
                hk = hash((hk, int(tok)))
            keys.append(hk)
        block_keys.append(keys)
    cached: dict[int, int] = {}                            # content key -> first producer (request index)

    def cache_after_schedule(r: _Req, n: int) -> None:
        if not pc:
            return
        full = min(r.computed + n, r.total) // block_size
        for j in range(min(full, len(block_keys[r.idx]))):
            cached.setdefault(block_keys[r.idx][j], r.idx)

    def hits_at_admission(r: _Req) -> int:
        if not pc or r.computed != 0:
            return 0
        limit = min((r.total - 1) // block_size, len(block_keys[r.idx]))
        h = 0
        while h < limit and block_keys[r.idx][h] in cached:
            h += 1
        return h
    free = (int(phi.engine["num_gpu_blocks"]) - 1) if kv else None          # block 0 is the null block
    preempt_queue: list[_Req] = []                                          # front of the waiting queue, most recent victim first

    # v4-pool: the explicit block pool (see PREDICTORS).  `kv` stays True (shared code paths), the count-based branches are skipped.
    pool_on = kv and bool(PREDICTORS[pid].get("pool"))
    pool = _Pool(int(phi.engine["num_gpu_blocks"]), pc) if pool_on else None
    seat_on = pool_on and bool(PREDICTORS[pid].get("seat"))   # v5-seat: a max_tokens stop keeps its running seat through the next schedule
    max_tok = [int(r.get("max_tokens") or 0) for r in reqs_json]
    lag = 1 if phi.engine.get("async_scheduling", True) else 0
    pending_free: list[_Req] = []                          # max_tokens stops: blocks returned after the next non-empty schedule

    def blk_key(r: _Req, j: int) -> object:
        return block_keys[r.idx][j] if j < len(block_keys[r.idx]) else ("gen", r.idx, j)

    def pool_cache(r: _Req, n: int, step: int) -> None:
        """cache_blocks at scheduling: every full block within min(computed + n, known tokens) that is not hashed yet."""
        if not pc:
            return
        full = min(r.computed + n, r.known_at(step, lag)) // block_size
        for j in range(r.n_hashed, min(full, len(r.blk))):
            pool.cache(r.blk[j], blk_key(r, j))
        r.n_hashed = max(r.n_hashed, min(full, len(r.blk)))

    def pool_hits(r: _Req) -> list[int]:
        """get_computed_blocks: the longest run of leading blocks whose key is cached (at most (num_tokens − 1) // block_size)."""
        if not pc or r.computed != 0 or r.blk:
            return []
        out: list[int] = []
        for j in range((r.total - 1) // block_size):
            b = pool.lookup(blk_key(r, j))
            if b is None:
                break
            out.append(b)
        return out

    def pool_free_request(r: _Req) -> None:
        pool.free_blocks(list(reversed(r.blk)))
        r.blk, r.n_hashed = [], 0

    def pool_resync(r: _Req) -> None:
        """override re-synchronisation: hold exactly cdiv(computed, block_size) blocks (allocate from / return to the queue)."""
        want = _cdiv(r.computed, block_size)
        if len(r.blk) < want:
            r.blk.extend(pool.get_new_blocks(min(want - len(r.blk), pool.num_free), r.idx))
        elif len(r.blk) > want:
            extra, r.blk = r.blk[want:], r.blk[:want]
            pool.free_blocks(list(reversed(extra)))
            r.n_hashed = min(r.n_hashed, len(r.blk))

    running: list[_Req] = []
    plans: list[StepPlan] = []
    step = 0
    retried_empty = False
    while step < max_steps:
        fresh = [r for r in reqs if not r.admitted and not r.done and r.arrive <= step and r not in preempt_queue]
        waiting = preempt_queue + fresh
        if not running and not waiting and step not in overrides:
            if all(r.done for r in reqs):
                break
            step += 1          # idle step: nothing to run yet (a later arrival); counted as an engine step number
            continue
        rows: list[tuple[int, int]] = []
        hits_now: list[tuple[int, int, int, int]] = []
        if step in overrides:
            ov = overrides[step]
            for run in ov.rows:
                rows.extend((run.request, run.first + j) for j in range(run.count))
            present = {run.request for run in ov.rows}
            if kv:
                # running requests the engine did not schedule can only have been preempted (or budget-starved) here
                for r in [r for r in running if r.idx not in present]:
                    running.remove(r)
                    if seat_on and r.done:                              # a held seat: finished, not preempted
                        if r in pending_free:
                            pending_free.remove(r)
                            pool_free_request(r)
                        continue
                    if pool_on:
                        pool_free_request(r)
                    else:
                        free += r.blocks
                    r.blocks, r.computed, r.admitted, r.preemptions = 0, 0, False, r.preemptions + 1
                    preempt_queue.insert(0, r)
            # requests the override admits join the scheduler's running list in workload (FCFS) order, not row order
            for run in sorted(ov.rows, key=lambda x: x.request):
                r = reqs[run.request]
                if not r.admitted:
                    r.admitted = True
                    running.append(r)
                    if r in preempt_queue:
                        preempt_queue.remove(r)
            plan = StepPlan(step, rows, [], [])
            plan.logits_rows = list(ov.logits_rows) if ov.logits_rows is not None else plan.default_logits_rows()
            batch.resync([run.request for run in ov.rows])
            finished_prev = []
        else:
            budget = budget_total
            new_reqs: list[int] = []
            preempted_now = False
            i = 0
            while i < len(running):
                r = running[i]
                if budget <= 0:
                    break
                if seat_on and r.done:
                    i += 1                                              # scheduler.py:590-603: reached max_tokens → skipped, keeps its seat
                    continue
                if r.computed < r.total:
                    n = r.total - r.computed                        # prefill remainder (or a preempted request's recompute remainder)
                    if 0 < long_cap < n:
                        n = long_cap
                    if n > budget:
                        if not chunked:
                            break
                        n = budget
                else:
                    n = 1
                if pool_on:
                    need = max(_cdiv(r.computed + n, block_size) - len(r.blk), 0)
                    victim = None
                    while need > pool.num_free:
                        victim = running.pop()                      # FCFS policy: the lowest-priority request is the last running one
                        preempted_now = True
                        if seat_on and victim.done:                 # a held seat popped as the victim: its blocks return, it is finished
                            if victim in pending_free:
                                pending_free.remove(victim)
                                pool_free_request(victim)
                            continue
                        pool_free_request(victim)
                        victim.blocks, victim.computed, victim.admitted, victim.preemptions = 0, 0, False, victim.preemptions + 1
                        preempt_queue.insert(0, victim)
                        if victim is r:
                            break
                    if victim is r:
                        break                                       # cannot schedule this request; nothing runs after it
                    r.blk.extend(pool.get_new_blocks(need, r.idx))
                    pool_cache(r, n, step)
                elif kv:
                    need = _cdiv(r.computed + n, block_size) - r.blocks
                    victim = None
                    while need > free:
                        victim = running.pop()                      # FCFS policy: the lowest-priority request is the last running one
                        free += victim.blocks
                        victim.blocks, victim.computed, victim.admitted, victim.preemptions = 0, 0, False, victim.preemptions + 1
                        preempt_queue.insert(0, victim)
                        preempted_now = True
                        if victim is r:
                            break
                    if victim is r:
                        break                                       # cannot schedule this request; nothing runs after it
                    free -= need
                    r.blocks += need
                if not pool_on:
                    cache_after_schedule(r, n)
                rows.extend((r.idx, r.computed + j) for j in range(n))
                budget -= n
                i += 1
            if not preempted_now:
                for r in waiting:
                    if len(running) >= max_seqs or budget <= 0:
                        break
                    if pool_on:
                        hit_blocks = pool_hits(r)
                        r.computed = len(hit_blocks) * block_size
                    else:
                        h = hits_at_admission(r)
                        r.computed = h * block_size                                         # prefix-cache hits (0 without caching)
                    n = r.total - r.computed
                    if 0 < long_cap < n:
                        n = long_cap
                    if n > budget:
                        if not chunked:
                            break
                        n = budget
                    if pool_on:
                        # get_num_blocks_to_allocate for a request without blocks: required − hit blocks, plus the hit blocks that sit
                        # in the free queue (touching them takes them out of it); full sequence first, then the chunk
                        ev = pool.evictable(hit_blocks)
                        need_full = max(_cdiv(r.total, block_size) - len(hit_blocks), 0) + ev
                        need = max(_cdiv(r.computed + n, block_size) - len(hit_blocks), 0) + ev
                        if need_full > pool.num_free or need > pool.num_free:
                            r.computed = 0
                            break
                        pool.touch(hit_blocks)
                        r.blk = list(hit_blocks)
                        r.n_hashed = len(hit_blocks)
                        r.blk.extend(pool.get_new_blocks(need - ev, r.idx))
                        for j, b in enumerate(hit_blocks):
                            hits_now.append((r.idx, j * block_size, block_size, pool.writer[b]))
                        pool_cache(r, n, step)
                    elif kv:
                        need = _cdiv(r.computed + n, block_size) - r.blocks                 # includes the hit blocks (shared, ref-counted)
                        if _cdiv(r.total, block_size) - r.blocks > free or need > free:     # reserve the full sequence; then the chunk
                            r.computed = 0
                            break
                        free -= need
                        r.blocks += need
                    if not pool_on:
                        cache_after_schedule(r, n)
                    r.admitted = True
                    running.append(r)
                    if r in preempt_queue:
                        preempt_queue.remove(r)
                    new_reqs.append(r.idx)
                    rows.extend((r.idx, r.computed + j) for j in range(n))
                    budget -= n
            if pool_on and not rows and (pending_free or (seat_on and any(r.done for r in running))) and not retried_empty:
                # an empty schedule is not an observed step: the finished requests' blocks (and seats, v5-seat) return and the same
                # step is scheduled again (scheduler loop: schedule(t+1) runs before update(t) frees the max_tokens stops of step t)
                for r in pending_free:
                    pool_free_request(r)
                pending_free = []
                if seat_on:
                    running = [r for r in running if not r.done]
                retried_empty = True
                continue
            if order == "sort":
                rows = _runner_order(rows)
            elif order == "persistent-batch":
                batch.update(finished_prev, new_reqs)
                finished_prev = []
                rows = batch.order(rows)
            plan = StepPlan(step, rows, [], [])
            plan.logits_rows = plan.default_logits_rows()
        if not rows:
            raise RuntimeError(f"predictor: step {step} has no rows (running={[r.idx for r in running]})")
        retried_empty = False
        plan.hits = hits_now
        # state update from the rows: KV length after the step, served/discarded samples, completion
        totals_before = {r.idx: r.total for r in reqs}
        last: dict[int, int] = {}
        for r_idx, p in rows:
            last[r_idx] = max(last.get(r_idx, -1), p)
        for r_idx, p in last.items():
            reqs[r_idx].computed = p + 1
            if kv and step in overrides:
                if pool_on:
                    pool_resync(reqs[r_idx])
                else:
                    held = _cdiv(reqs[r_idx].computed, block_size)
                    free -= held - reqs[r_idx].blocks
                    reqs[r_idx].blocks = held
        served_flags: list[bool] = []
        for r_idx, p in plan.logits_rows:
            r = reqs[r_idx]
            is_served = r.computed >= totals_before[r_idx] and (r_idx, step) not in discarded
            served_flags.append(is_served)
            if is_served:
                r.served += 1
                r.served_steps.append(step)
                if r.served >= r.target:
                    r.done = True
                    r.finished_at = step
        plan.served = served_flags
        kinds = {"prefill" if p < reqs[r].prompt_len else "decode" for r, p in rows}
        plan.phase = kinds.pop() if len(kinds) == 1 else "mixed"
        plans.append(plan)
        finished_prev.extend(r.idx for r in running if r.done)
        if pool_on:
            # update(t) runs after schedule(t+1): the blocks of a request that stopped at max_tokens in this step stay held through
            # the next schedule; an EOS stop (served < max_tokens) was detected one step earlier and its blocks return now
            for r in pending_free:
                pool_free_request(r)
            pending_free = []
            for r in running:
                if r.done and r.finished_at == step:                 # (a seat held from the previous step was handled then)
                    if r.served < max_tok[r.idx]:
                        pool_free_request(r)
                    else:
                        pending_free.append(r)
        elif kv:
            for r in running:
                if r.done:
                    free += r.blocks
                    r.blocks = 0
        if seat_on:
            # a request that stopped at max_tokens in THIS step keeps its seat through the next schedule (removed at the end of
            # that step); EOS stops (served < max_tokens) and seats held since the previous step leave now
            running = [r for r in running if not r.done or (r.finished_at == step and r.served >= max_tok[r.idx])]
        else:
            running = [r for r in running if not r.done]
        step += 1
    else:
        raise RuntimeError("predictor: step limit reached")
    return plans


def schedule_summary(plans: list[StepPlan]) -> dict:
    return {"steps": len(plans), "rows": sum(len(p.rows) for p in plans), "sampled": sum(len(p.logits_rows) for p in plans),
            "served": sum(sum(p.served) for p in plans), "phases": {k: sum(1 for p in plans if p.phase == k) for k in ("prefill", "decode", "mixed")}}
