"""A stand-in for vLLM V1's in-process scheduler, with the surface `engine.step_trace` wraps, so a step trace is recorded without
vLLM or a GPU.

It schedules like the V1 scheduler does in outline: running requests first (a decode is one token, a partial prefill continues),
then waiting ones in arrival order, within a token budget and a cap on running requests; a prompt longer than the budget left is
chunked.  A request emits one token at the step that completes its prompt and at every decode, and finishes at its EOS (the
scripted output reaches `eos`) or at `max_tokens`.  `preempt_at` = {step: request} preempts a running request at that schedule:
its computed tokens are dropped and it waits again, resumed from position 0.  The model runner's order is Model Runner V2's
`sort_batch_req_ids` (decodes first, then by chunk length, stably).  `prefix_block` = B turns prefix caching on: a request
admitted (first, or resumed) starts at the longest whole-block prefix of its tokens that some request, itself included, has
computed, short of its last token.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any


@dataclass
class FakeRequest:
    request_id: str
    prompt_token_ids: list[int]
    outputs: list[int]                          # what the model would sample, in order
    max_tokens: int
    eos_token_id: int = 2
    sampling_params: Any = None
    num_computed_tokens: int = 0
    emitted: list[int] = field(default_factory=list)
    status: str = "waiting"
    stop_reason: Any = None

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.emitted)

    def get_finished_reason(self):
        return {"stopped": 0, "length": 1, "aborted": 2}.get(self.status)


def runner_order(nst: dict[str, int]) -> list[str]:
    return sorted(nst, key=lambda r: (True, nst[r] != 1, nst[r]))


class FakeScheduler:
    def __init__(self, budget: int = 16, max_seqs: int = 4, preempt_at: dict[int, str] | None = None,
                 prefix_block: int | None = None) -> None:
        self.requests: dict[str, FakeRequest] = {}
        self.waiting: list[str] = []
        self.running: list[str] = []
        self.budget, self.max_seqs = budget, max_seqs
        self.preempt_at = dict(preempt_at or {})
        self.step = 0
        self.scheduler_config = SimpleNamespace(max_num_batched_tokens=budget, max_num_seqs=max_seqs, enable_chunked_prefill=True,
                                                policy="fcfs", async_scheduling=False)
        self.prefix_block = prefix_block
        self.cache_config = SimpleNamespace(block_size=prefix_block or 16, enable_prefix_caching=prefix_block is not None)
        self._cached: set[tuple[int, ...]] = set()      # the whole-block prefixes some request has computed
        self._seen: set[str] = set()
        self._resumed: set[str] = set()

    def has_requests(self) -> bool:
        return bool(self.waiting or self.running)

    def add_request(self, request: FakeRequest) -> None:
        self.requests[request.request_id] = request
        self.waiting.append(request.request_id)

    def schedule(self) -> Any:
        preempted = set()
        if (victim := self.preempt_at.pop(self.step, None)) is not None and victim in self.running:
            self.running.remove(victim)
            q = self.requests[victim]
            q.num_computed_tokens, q.status = 0, "waiting"
            self.waiting.insert(0, victim)
            preempted.add(victim)
            self._resumed.add(victim)
        nst: dict[str, int] = {}
        left = self.budget
        new, cached = [], []
        for rid in self.running:
            q = self.requests[rid]
            n = min(q.num_tokens - q.num_computed_tokens, left)
            if n > 0:
                nst[rid], left = n, left - n
                cached.append(rid)
        for rid in [r for r in self.waiting if r not in preempted]:
            if left <= 0 or len(self.running) >= self.max_seqs:
                break
            self.waiting.remove(rid)
            q = self.requests[rid]
            q.num_computed_tokens = self._hit(q)
            n = min(q.num_tokens - q.num_computed_tokens, left)
            nst[rid], left = n, left - n
            self.running.append(rid)
            q.status = "running"
            (cached if rid in self._seen else new).append(rid)
        resumed = set(cached) & self._resumed
        so = SimpleNamespace(
            scheduled_new_reqs=[SimpleNamespace(req_id=r, num_computed_tokens=self.requests[r].num_computed_tokens) for r in new],
            scheduled_cached_reqs=SimpleNamespace(req_ids=cached, num_computed_tokens=[self.requests[r].num_computed_tokens for r in cached],
                                                  resumed_req_ids=resumed),
            num_scheduled_tokens=nst, total_num_scheduled_tokens=sum(nst.values()), preempted_req_ids=preempted)
        self._resumed -= set(cached)
        self._seen |= set(nst)
        for rid, n in nst.items():
            q = self.requests[rid]
            q.num_computed_tokens += n
            if self.prefix_block:
                toks = q.prompt_token_ids + q.emitted
                for k in range(1, q.num_computed_tokens // self.prefix_block + 1):
                    self._cached.add(tuple(toks[:k * self.prefix_block]))
        self.step += 1
        return so

    def _hit(self, q: FakeRequest) -> int:
        if not self.prefix_block:
            return 0
        toks, b, k = q.prompt_token_ids + q.emitted, self.prefix_block, 0
        while (k + 1) * b < len(toks) and tuple(toks[:(k + 1) * b]) in self._cached:
            k += 1
        return k * b

    def execute(self, so: Any) -> Any:
        """The model runner: one sampled token per request whose computed tokens reached its known tokens."""
        order = runner_order(so.num_scheduled_tokens)
        sampled = []
        for rid in order:
            q = self.requests[rid]
            sampled.append([q.outputs[len(q.emitted)]] if q.num_computed_tokens >= q.num_tokens else [])
        return SimpleNamespace(req_ids=order, sampled_token_ids=sampled)

    def update_from_output(self, so: Any, mro: Any) -> dict[int, Any]:
        outs = []
        for rid, toks in zip(mro.req_ids, mro.sampled_token_ids):
            q = self.requests[rid]
            if not toks or q.status != "running":
                continue
            q.emitted += toks
            finish = None
            if toks[-1] == q.eos_token_id:
                q.status, finish = "stopped", 0
            elif len(q.emitted) >= q.max_tokens:
                q.status, finish = "length", 1
            if finish is not None:
                self.running.remove(rid)
            outs.append(SimpleNamespace(request_id=rid, new_token_ids=list(toks), finish_reason=finish, stop_reason=q.stop_reason))
        return {0: SimpleNamespace(outputs=outs)}

    def finish_requests(self, request_ids: Any, finished_status: Any) -> list[FakeRequest]:
        done = []
        for rid in ([request_ids] if isinstance(request_ids, str) else request_ids):
            q = self.requests[rid]
            if q.status in ("waiting", "running"):
                q.status = "aborted"
                (self.running if rid in self.running else self.waiting).remove(rid)
                done.append(q)
        return done


def drive(sched: FakeScheduler, arrivals: dict[int, list[FakeRequest]], *, abort_at: dict[int, str] | None = None) -> int:
    """Step the engine core loop (schedule, execute, update) until every request, arrived at its step, is done; returns the steps
    run.  `arrivals` = {k: requests added before the k-th loop iteration}; an idle engine waits for the next arrival."""
    pending = dict(sorted(arrivals.items()))
    abort_at = dict(abort_at or {})
    it = 0
    while pending or sched.has_requests():
        for r in pending.pop(it, []):
            sched.add_request(r)
        if (rid := abort_at.pop(it, None)) is not None:
            sched.finish_requests(rid, "FINISHED_ABORTED")
        if sched.has_requests():
            so = sched.schedule()
            sched.update_from_output(so, sched.execute(so))
        it += 1
    return it
