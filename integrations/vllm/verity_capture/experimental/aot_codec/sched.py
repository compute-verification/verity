"""[R8 C-codec] Deterministic pre-execution scheduler model: the *predictor* behind A2/A3.

It reproduces the structural decisions of vLLM d9105ea80's V1 scheduler + V2 GPU model runner for the class covered by the corpus
(FCFS admission, chunked prefill under a token budget, max_num_seqs cap, prefix-cache admission hits, KV-block accounting with
FCFS preemption when the pool is known, no spec decoding):

  * step t: running requests first, in persistent (admission) order: decode-phase -> 1 token; prefill-phase -> min(remaining, budget)
    then waiting requests with arrive_step <= t, FCFS, while len(running) < max_num_seqs and budget > 0: min(prompt_len, budget)
  * row order in the model runner = stable sort by (num_scheduled != 1, num_scheduled)   [source-inspected: vllm/v1/worker/gpu/
    model_runner.py::sort_batch_req_ids -- "verification/decode -> short_extend -> prefill"; ties keep SchedulerOutput dict order]
  * logits rows: one per request in the batch, in row order; the sample is kept iff the chunk completes the prompt
  * a request is finished after the step in which it has processed prompt_len + n_generated - 1 tokens
  * KV pool (only when phi carries `num_gpu_blocks`, an operational fact -> phi2; SOURCE-INSPECTED vllm/v1/core/{block_pool,
    kv_cache_manager}.py + sched/scheduler.py at d9105ea80, MEASURED gen/preempt4 + gen/pp4): usable = num_gpu_blocks - 1 (null
    block); the pool holds the *distinct* blocks of the running requests -- cdiv(position, B) per request, minus prefix-cache hits, which
    reference blocks another request allocated (ref_cnt); a running request needing more than the free count preempts the *last* running
    request (running.pop() -> waiting front, num_computed = 0; if that is itself, nothing further is scheduled this step); a step that
    preempted admits nothing; otherwise a waiting request (preempted ones first, then arrivals FCFS) is admitted only if its whole known
    sequence (prompt + tokens sampled so far, minus its hit) fits, else admission stops; a resumed request re-prefills prompt + generated
    tokens in one chunk, after the same prefix lookup as a fresh admission.  The pool is modelled with *block identities* (`Pool`):
    the LRU free queue (block_pool.py::free_blocks: a freed request's blocks tail-first, unhashed ones prepended = LIFO reuse, hashed ones
    appended = FIFO eviction; get_new_blocks pops the front and evicts the popped block's hash), so a resumed request's own hashed blocks
    that nobody has reused yet are hits (MEASURED gen/pp4: resume with num_computed 80 = shared prefix + its own surviving block).  Block
    hashes: the prompt chain hashes for full prompt blocks (phi0), a private identity for blocks holding generated tokens (equal generated
    continuations of two requests are not derivable -> such a cross-request hit stays an escape).  The scheduler's KV watermark
    (`scheduler_config.watermark`, waiting requests only) defaults to 0.0 = disabled at this install (SOURCE-INSPECTED vllm/config/scheduler.py:167;
    an earlier comment said int(0.01 * num_blocks) -- wrong); a non-default watermark would be a phi0 engine fact and is not modelled.

The model is a plain function of (phi-level facts + decoded request facts); when the realized step differs, the codec escapes and the
simulator is *resynchronised to the realized step* (`apply`), so one deviation costs one escape, not a cascade.
"""
from __future__ import annotations


class Pool:
    """vLLM V1 BlockPool + FullAttentionManager with block identities (SOURCE-INSPECTED vllm/v1/core/{block_pool,kv_cache_manager,
    single_type_kv_cache_manager}.py at the pinned install).  Blocks 1..n (block 0 is the null block) start in the free queue in id order."""

    def __init__(self, n_usable: int, block_size: int, caching: bool, hashes: list[list[str]]) -> None:
        self.B = block_size
        self.caching = caching
        self.hashes = hashes  # prompt chain hashes per request (full prompt blocks)
        self.free_q: list[int] = list(range(1, n_usable + 1))  # front = index 0 = next block handed out / evicted
        self.ref: dict[int, int] = {}
        self.bhash: dict[int, tuple] = {}  # block -> key while cached
        self.cached: dict[tuple, list[int]] = {}  # key -> blocks holding it (several copies are possible), insertion order
        self.blocks: dict[int, list[int]] = {}  # request -> its blocks in order (hits first)
        self.ncached: dict[int, int] = {}  # request -> full blocks of it already cached (num_cached_block)

    def copy(self) -> "Pool":
        p = Pool.__new__(Pool)
        p.B, p.caching, p.hashes = self.B, self.caching, self.hashes
        p.free_q = list(self.free_q)
        p.ref = dict(self.ref)
        p.bhash = dict(self.bhash)
        p.cached = {k: list(v) for k, v in self.cached.items()}
        p.blocks = {k: list(v) for k, v in self.blocks.items()}
        p.ncached = dict(self.ncached)
        return p

    def key(self, i: int, k: int) -> tuple:
        h = self.hashes[i]
        return ("p", h[k]) if k < len(h) else ("g", i, k)

    def free_count(self) -> int:
        return len(self.free_q)

    def held(self, i: int) -> int:
        return len(self.blocks.get(i, ()))

    def need(self, i: int, ntok: int) -> int:
        """Blocks to allocate so that request i has slots for ntok tokens (its current blocks count, hits included)."""
        return max(-(-ntok // self.B) - self.held(i), 0)

    def lookup(self, i: int, known: int, extra: frozenset = frozenset()) -> list[int]:
        """Prefix-cache hit for a waiting request with `known` tokens: the longest run of leading blocks whose key is cached (or in
        `extra`, keys assumed allocated earlier in the same step), capped at known - 1 tokens (get_computed_blocks).  Block ids (0 for an
        `extra` key, resolved when the block exists)."""
        if not self.caching or i in self.blocks:
            return []
        out = []
        for k in range((known - 1) // self.B):
            key = self.key(i, k)
            lst = self.cached.get(key)
            if lst:
                out.append(lst[0])
            elif key in extra:
                out.append(0)
            else:
                break
        return out

    def evictable(self, blocks: list[int]) -> int:
        return sum(1 for b in blocks if b and self.ref.get(b, 0) == 0)

    def admit(self, i: int, hits: list[int]) -> None:
        """touch(): a hit block in the free queue leaves it; ref_cnt + 1; the request's block list starts with the hits."""
        for b in hits:
            if self.ref.get(b, 0) == 0:
                self.free_q.remove(b)
            self.ref[b] = self.ref.get(b, 0) + 1
        self.blocks[i] = list(hits)
        self.ncached[i] = len(hits)

    def alloc(self, i: int, ntok: int) -> None:
        """get_new_blocks(): pop the front of the free queue (evicting the popped block's hash) until request i has cdiv(ntok) blocks;
        then cache_blocks(): its full blocks up to ntok // B get their keys."""
        lst = self.blocks.setdefault(i, [])
        self.ncached.setdefault(i, 0)
        for _ in range(self.need(i, ntok)):
            if not self.free_q:
                break  # cannot happen on a step the model predicted; the realized step is replayed best-effort
            b = self.free_q.pop(0)
            self._evict(b)
            self.ref[b] = 1
            lst.append(b)
        if self.caching:
            full = min(ntok // self.B, len(lst))
            for k in range(self.ncached[i], full):
                key = self.key(i, k)
                self.bhash[lst[k]] = key
                self.cached.setdefault(key, []).append(lst[k])
            self.ncached[i] = max(self.ncached[i], full)

    def _evict(self, b: int) -> None:
        key = self.bhash.pop(b, None)
        if key is not None:
            lst = self.cached[key]
            lst.remove(b)
            if not lst:
                del self.cached[key]

    def free(self, i: int) -> None:
        """kv_cache_manager.free(): the request's blocks tail-first; ref_cnt - 1; blocks at 0 leave to the free queue -- without a hash
        (or caching off) prepended in that order, with a hash appended in that order (block_pool.py::free_blocks)."""
        first, last = [], []
        for b in reversed(self.blocks.pop(i, [])):
            self.ref[b] -= 1
            if self.ref[b] == 0:
                (last if (self.caching and b in self.bhash) else first).append(b)
        self.free_q = first + self.free_q + last
        self.ncached.pop(i, None)


class SchedState:
    def __init__(self, phi: dict, reqs: list[dict]) -> None:
        """reqs: per request {prompt_len, arrive_step, n_generated} (arrive/n_generated from phi or from decoded advice)."""
        eng = phi["engine"]
        self.max_num_seqs = int(eng["max_num_seqs"])
        self.budget_total = int(eng["max_num_batched_tokens"])
        self.chunked = bool(eng.get("chunked_prefill", True))
        self.thr = int(eng.get("long_prefill_token_threshold") or 0)  # chunk cap applied before the budget (0 = off)
        self.L = [int(q["prompt_len"]) for q in reqs]
        self.arrive = [int(q["arrive_step"]) for q in reqs]
        # tokens to process in total: prompt + n_generated - 1 (the last sampled token is never fed back) -- plus one placeholder decode
        # row when the request stopped on EOS under async scheduling (the scheduler runs a step ahead of the sampler; a max_tokens stop is
        # known in advance and gets no extra row).  SOURCE-INSPECTED vllm/v1/core/sched/async_scheduler.py; MEASURED on the eos4 record.
        self.async_sched = bool(eng.get("async_scheduling", True))
        self.target = [int(q["prompt_len"]) + max(int(q["n_generated"]), 1) - 1 + (1 if (self.async_sched and q.get("finish_reason") == "stop") else 0)
                       for q in reqs]
        self.nc = [0] * len(reqs)  # num_computed
        # tokens sampled so far per request (rows that reached the end of the prompt); with L it bounds what a row can legally
        # schedule: after a preemption vLLM re-prefills prompt + generated tokens in one chunk (MEASURED gen/preempt4)
        self.sampled = [0] * len(reqs)
        # prefix caching (vLLM V1, SOURCE-INSPECTED at the pinned install): a waiting request's num_computed at admission = the longest
        # run of leading *full* blocks whose chain hashes are already cached, capped at prompt_len - 1 (kv_cache_manager.get_computed_blocks:
        # `max_cache_hit_length = request.num_tokens - 1`); blocks are cached at *allocation*, up to min(computed + new, num_tokens)
        # (allocate_slots -> cache_blocks), so a request allocated earlier in the same step already counts.  Block hashes of the prompts
        # are phi0 content (prompt ids); hits against *generated* tokens are not derivable and stay escapes.  No eviction modelled.
        self.B = int(eng.get("block_size", 16))
        self.prefix = bool(eng.get("prefix_caching", False))
        self.hashes = [list(q.get("block_hashes") or []) for q in phi["requests"]] if self.prefix else [[] for _ in reqs]
        self.running: list[int] = []  # persistent order
        self.preempted: list[int] = []  # waiting-queue front, in queue order (most recently preempted first)
        kv = eng.get("num_gpu_blocks")
        self.usable_blocks = (int(kv) - 1) if kv else None  # block 0 is the null block; None -> pool not modelled
        # leading blocks of request i that are *references* to blocks another request allocated (prefix-cache hit at its last
        # admission): held once by the pool, not once per request -- MEASURED gen/pp4 (4 prompts sharing 4 blocks under a 20-block pool)
        self.hit = [0] * len(reqs)
        self.pool = Pool(self.usable_blocks, self.B, self.prefix, self.hashes) if self.usable_blocks is not None else None
        self._pred: tuple[list[list[int]], "Pool"] | None = None  # (rows, pool after the step) of the last predict() for this step
        self.finished = [False] * len(reqs)
        self.admitted = [False] * len(reqs)
        # requests whose last row was the previous step and whose finish the engine has not processed yet: under async scheduling
        # step t+1 is scheduled *before* update_from_output(t) runs (SOURCE-INSPECTED vllm/v1/engine/core.py::step_with_batch_queue), so a
        # length-stopped request still holds its KV blocks and its running slot for one more schedule (num_new_tokens == 0 -> skipped);
        # if that schedule would be empty the engine first drains the pending output and reschedules (no step is logged).  An EOS stop
        # is processed in update_from_output(k) while its placeholder row (step k+1) is in flight -> gone at k+2, no extra hold.
        # MEASURED gen/pre6 (resume at last row + 2 beside a running request) and gen/preempt4 (+1 when nothing else runs).
        self.stop = [q.get("finish_reason") == "stop" for q in reqs]
        self.draining: list[int] = []
        self.t = 0

    # -- prediction -------------------------------------------------------------------------------------------------------------
    def _cdiv(self, n: int) -> int:
        return -(-n // self.B)

    def predict(self) -> tuple[list[list[int]], list[list[int]]]:
        """Predicted (rows, logits) for step self.t given the current state; rows = [[req, num_computed, n_sched]...].  Pure (no state
        change): a predicted preemption is realised by `apply` when the realized rows confirm it.  A schedule that would be empty while
        finished requests still hold their blocks is retried after draining them (the engine does the same, logging no step)."""
        rows, logits, pool = self._predict(drain=False)
        if not rows and self.draining:
            rows, logits, pool = self._predict(drain=True)
        self._pred = (rows, pool) if pool is not None else None
        return rows, logits

    def _predict(self, drain: bool) -> tuple[list[list[int]], list[list[int]], "Pool | None"]:
        budget = self.budget_total
        sched: list[list[int]] = []
        pool = self.pool.copy() if self.pool is not None else None  # this step's allocations, on a copy (predict is pure)
        running = [j for j in self.running if not (drain and j in self.draining)]
        if drain and pool is not None:
            for j in self.draining:  # the engine drained their finish before rescheduling: blocks released
                pool.free(j)
        victims: list[int] = []  # this step's preemptions, in pop order
        k = 0
        while k < len(running) and budget > 0:
            i = running[k]
            rem = self.target[i] - self.nc[i]
            if rem <= 0:
                k += 1  # all tokens computed (finish not yet processed): skipped, keeps its slot and blocks
                continue
            # vLLM: num_new_tokens = tokens known (prompt + generated so far) - computed, capped by the chunk cap and the budget -- one
            # formula for a prefill chunk, a decode row (1) and a chunked re-prefill after preemption (MEASURED gen/pre6: 24-token chunks)
            n = self._chunk(self.L[i] + self.sampled[i] - self.nc[i], budget)
            if n <= 0:
                break
            if pool is not None:
                blocked = False
                while pool.need(i, self.nc[i] + n) > pool.free_count():
                    victim = running.pop()  # the last running request -- possibly the requester itself
                    victims.append(victim)
                    pool.free(victim)  # its blocks are released -- except those still referenced by another request
                    if victim == i:
                        # SOURCE-INSPECTED scheduler.py (running loop): the requester is preempted like any victim (status PREEMPTED, position 0,
                        # waiting front) and "No more request to preempt" ends the running loop; the step admits nothing (preempted_reqs non-empty).
                        # Until 10:30Z the model left the requester running un-preempted (never exercised on a record; pre-registered gen/mp3)
                        blocked = True
                        break
                if blocked:
                    break
                pool.alloc(i, self.nc[i] + n)
            budget -= n
            sched.append([i, self.nc[i], n])
            k += 1
        n_running = len(running)
        pending = [i for i in range(len(self.L)) if not self.admitted[i] and not self.finished[i] and self.arrive[i] >= 0]
        due = self.t
        if not sched and pending and not self.preempted and not victims and min(self.arrive[i] for i in pending) > self.t:
            # harness rule (verity_capture/arrivals.py): the engine never idles -- the next arrival is admitted at once
            due = min(self.arrive[i] for i in pending)
        cached = {j: min(self.nc[j], self.L[j]) for j in range(len(self.L)) if self.admitted[j]}
        for j, c0, n in sched:
            cached[j] = min(c0 + n, self.L[j])  # allocated earlier in this step -> already cached
        # waiting queue: this step's victims (each prepended -> running order among them), earlier preempted, then arrivals FCFS
        # scheduler.py:809 `if not preempted_reqs:` -- a step that preempted admits nothing (the victims wait for the next step)
        waiting = [] if victims else list(self.preempted) + [i for i in pending if self.arrive[i] <= due]
        for i in waiting:
            if n_running >= self.max_num_seqs or budget <= 0:
                break
            # admission lookup -- also for a resumed request: the prompt blocks another running request still holds are hits (MEASURED
            # gen/pp4: resume with num_computed 64) and so are its *own* hashed blocks that survived in the LRU free queue (hit 80 on gen/pp4)
            known = self.L[i] + self.sampled[i]  # prompt + tokens generated so far (a resumed request re-prefills them all)
            hits = pool.lookup(i, known) if pool is not None else []
            c0 = len(hits) * self.B if pool is not None else self._admission_nc(i, cached)
            n = self._chunk(known - c0, budget)
            if n <= 0:
                break
            if pool is not None:
                # scheduler_reserve_full_isl (default True): the whole known sequence must fit -- blocks to allocate = cdiv(known) minus the
                # hit blocks, plus those hit blocks that sit in the free queue (touching them removes them from it)
                if self._cdiv(known) - len(hits) + pool.evictable(hits) > pool.free_count():
                    break
                pool.admit(i, hits)
                pool.alloc(i, c0 + n)
            budget -= n
            n_running += 1
            sched.append([i, c0, n])
            cached[i] = min(c0 + n, self.L[i])
        rows = sorted(sched, key=lambda g: (g[2] != 1, g[2]))  # stable
        # a row samples a kept token iff it reaches the end of the request's *known* sequence (prompt + tokens generated so far): the
        # chunks of a re-prefill that stop short of it sample nothing (MEASURED gen/pre6), like a partial prompt chunk
        logits = [[g[0], 1 if g[1] + g[2] >= self.L[g[0]] + self.sampled[g[0]] else 0] for g in rows]
        return rows, logits, pool

    def _chunk(self, rem: int, budget: int) -> int:
        """Prefill tokens scheduled for a request with `rem` tokens left: cap at long_prefill_token_threshold (if > 0), then at the budget
        (chunked prefill); without chunked prefill the whole remainder or nothing."""
        if not self.chunked:
            return rem if rem <= budget else 0
        if 0 < self.thr < rem:
            rem = self.thr
        return min(rem, budget)

    def _preempt(self, i: int) -> None:
        """vLLM `_preempt_request`: blocks freed, num_computed = 0, request prepended to the waiting queue (running order among
        several victims of one step is preserved: the scheduler pops from the end and prepends each)."""
        self.running.remove(i)
        self.nc[i] = 0
        self.preempted.insert(0, i)

    def predict_n_steps(self) -> int:
        """Total steps of a clean simulation from the current state (does not mutate)."""
        import copy

        s = copy.deepcopy(self)
        n = 0
        while s.active():
            rows, _ = s.predict()
            if not rows:
                break
            s.apply(rows)
            n += 1
            if n > 1_000_000:
                raise RuntimeError("simulation runaway")
        return s.t

    # -- resynchronisation ------------------------------------------------------------------------------------------------------
    def apply(self, rows: list[list[int]]) -> None:
        """Advance the state by the *realized* step.  Newly admitted requests join the persistent batch in FCFS order
        (waiting-queue order = (arrive_step, submission index)), not in row order.  A running, unfinished request absent from the
        step was preempted (num_computed = 0, waiting front); a preempted request that reappears is resumed at the end of the batch
        (in waiting order).  Pure bookkeeping on decoded facts -- valid at every phi level."""
        present = {i for i, _, _ in rows}
        if self.pool is not None:
            if self._pred is not None and self._pred[0] == rows:
                self.pool = self._pred[1]  # the step went as predicted: its allocations were already made on the copy
            else:
                self._replay(rows)
        self._pred = None
        for i in self.draining:  # the engine processed their finish while this step ran: gone before the next schedule
            self.finished[i] = True
            if i in self.running:
                self.running.remove(i)
            if self.pool is not None:
                self.pool.free(i)  # (already released if the step was the drain-and-reschedule retry)
        self.draining = []
        for i in [j for j in reversed(self.running) if j not in present and not self.finished[j]]:
            self._preempt(i)  # reversed(running): the scheduler pops from the end, prepending each -> waiting order = running order
        resumed = [i for i in self.preempted if i in present]
        for i in resumed:
            self.preempted.remove(i)
            self.running.append(i)
        new = sorted({i for i, _, _ in rows if not self.admitted[i]}, key=lambda i: (self.arrive[i], i))
        for i in new:
            self.admitted[i] = True
            self.running.append(i)
        for i, c, n in rows:
            if i in new or i in resumed:
                self.hit[i] = c // self.B  # num_computed at (re)admission = whole blocks hit in the prefix cache (0 without a hit)
        for i, c, n in rows:
            self.nc[i] = c + n
            if c + n >= self.L[i] + self.sampled[i]:  # reached the end of the known sequence -> one token sampled (one row per request per step)
                self.sampled[i] += 1
        for i, c, n in rows:
            if self.nc[i] >= self.target[i]:
                if self.stop[i]:  # EOS: the placeholder row was the last; the finish was processed while it ran
                    self.finished[i] = True
                    if i in self.running:
                        self.running.remove(i)
                    if self.pool is not None:
                        self.pool.free(i)
                else:  # length stop: finish processed after the next schedule -> holds blocks and slot for one more step
                    self.draining.append(i)
        self.t += 1

    def _replay(self, rows: list[list[int]]) -> None:
        """Pool bookkeeping for a realized step that was not (or not exactly) predicted, in the scheduler's order: running requests in
        running order, each preemption (an absent unfinished running request, popped from the end) at the moment a requester needs the
        space; then the admissions in waiting order with whatever hit their position says.  Best effort where the realized step
        contradicts the pool model (blocks that cannot be found are skipped) -- exactness never depends on it, only prediction."""
        pool = self.pool
        present = {i: (c, n) for i, c, n in rows}
        victims = [j for j in reversed(self.running) if j not in present and not self.finished[j] and j not in self.draining]
        for i in self.running:
            if i not in present:
                continue
            c, n = present[i]
            while pool.need(i, c + n) > pool.free_count() and victims:
                pool.free(victims.pop(0))
            pool.alloc(i, c + n)
        for j in victims:
            pool.free(j)
        new = sorted([i for i in present if i not in self.running], key=lambda i: (self.arrive[i], i))
        for i in new:
            c, n = present[i]
            hits = pool.lookup(i, self.L[i] + self.sampled[i])[: c // self.B]
            pool.admit(i, hits)
            pool.alloc(i, c + n)

    def active(self) -> bool:
        return any(not self.finished[i] for i in range(len(self.L)))

    def _admission_nc(self, i: int, cached: dict[int, int]) -> int:
        """Prefix-cache hit (tokens) for request i against the cached prompt blocks `cached[j]` (tokens cached of request j)."""
        if not self.prefix or not self.hashes[i]:
            return 0
        max_blocks = (self.L[i] - 1) // self.B
        best = 0
        hi = self.hashes[i]
        for j, ctok in cached.items():
            if j == i or not self.hashes[j]:
                continue
            lim = min(ctok // self.B, max_blocks, len(hi), len(self.hashes[j]))
            k = 0
            while k < lim and hi[k] == self.hashes[j][k]:
                k += 1
            best = max(best, k)
        return best * self.B

    def expected_nc(self, i: int) -> int:
        """num_computed expected for request i's next row: its state if running; else (not yet admitted, or preempted and waiting) its
        predicted admission hit, counting as cached the prompts of admitted requests and of due, earlier-FCFS pending requests (assumed
        allocated before it in the same step)."""
        if self.admitted[i] and i not in self.preempted:
            return self.nc[i]
        earlier = [j for j in range(len(self.L)) if not self.admitted[j] and not self.finished[j] and j != i
                   and 0 <= self.arrive[j] <= self.t and (self.arrive[j], j) < (self.arrive[i], i)]
        if self.pool is not None:
            extra = frozenset(self.pool.key(j, k) for j in earlier for k in range(min(self.L[j], self.budget_total) // self.B))
            return len(self.pool.lookup(i, self.L[i] + self.sampled[i], extra)) * self.B
        cached = {j: min(self.nc[j], self.L[j]) for j in range(len(self.L)) if self.admitted[j]}
        for j in earlier:
            cached[j] = min(self.L[j], self.budget_total)
        return self._admission_nc(i, cached)

    def remaining_prompt(self, i: int) -> int:
        return max(self.L[i] - self.nc[i], 0)


def simulate(phi: dict, reqs: list[dict]) -> tuple[list[dict], list[list[list[int]]]]:
    """Clean simulation -> (steps as in ScheduleHint, logits)."""
    s = SchedState(phi, reqs)
    steps, logits = [], []
    while s.active():
        rows, lg = s.predict()
        if not rows:
            break
        steps.append({"rows": rows})
        logits.append(lg)
        s.apply(rows)
        if len(steps) > 1_000_000:
            raise RuntimeError("simulation runaway")
    return steps, logits
