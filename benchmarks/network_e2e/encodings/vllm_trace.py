"""Step traces from vLLM V1's own scheduler on a fake model executor, on CPU: what `replay` is checked against.

`run` builds vLLM's `Scheduler` or `AsyncScheduler` as vLLM's tests do (`tests/v1/core/utils.py`, `create_scheduler`) and drives
it through the engine core's two loops (`EngineCore.step`, `EngineCore.step_with_batch_queue` with a batch queue of 2) on a
virtual clock. The executor is a fake: a row samples iff the scheduler marked it not a prefill chunk, and the token it samples
at output index j is the job's (EOS at its stop index). The GPU runs steps in order, each for `StepTime` of its shape, and
the engine core blocks only where it does: on an empty input queue, and on the oldest step's output.

With a draft (a `DraftEngine`) the scheduler is built with K speculative tokens and reads `mtp`'s settings (`make_scheduler`),
and the executor hands back what vLLM's runner does: for a row that samples, its token and then the drafts it kept, a
seeded count from 0 to the drafts the scheduler put in the row (none of a new or resumed request's, which are padding);
and, under sync scheduling, K draft ids for each request of a step once it has run, as the engine core passes them on.

It writes `verity-vllm/step-trace/v1` records (`verity_vllm.engine.step_trace`): a `header`, an `arrive` per request with the
schedules-with-tokens count it arrived before (`before_step`) and how many schedules with none came after that
(`after_empty`, when any), and a `step` per executed step at the time its output was processed, `drafts` the drafts each
of its rows carried. Records carry the engine's clock `t`, and the certifier's: `ingress_t`, when the ingress certifier
saw the request (its engine arrival less a frontend delay), and `release_t`, when the step's tokens left through the
egress certifier.

Needs vLLM importable (`VLLM_TARGET_DEVICE=cpu` without a GPU build) and a model config (`model`: a Hub id or a local
directory); `python -P vllm_trace.py OUT_DIR` writes `CASES` (`-P`: this directory's `code.py` would shadow the stdlib's).
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _load(name: str) -> Any:
    key = f"enc_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, Path(__file__).with_name(f"{name}.py"))
        assert spec is not None and spec.loader is not None
        sys.modules[key] = mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[key]


_replay = _load("replay")
STEP_TRACE, Engine, DraftEngine = _replay.STEP_TRACE, _replay.Engine, _replay.DraftEngine

EOS = 2
FINISH = {0: "stop", 1: "length", 2: "abort", 3: "error", 4: "repetition"}


@dataclass(frozen=True)
class Job:
    """One request: when it reaches the engine, its prompt, `max_tokens`, and the output index it samples EOS at (None: it
    runs to `max_tokens`). `tokens` (prompt then output) matters only with prefix caching."""

    arrival: float
    prompt_len: int
    max_tokens: int
    stop_at: int | None
    tokens: tuple[int, ...] | None = None
    ingress_delay: float = 0.0

    def token(self, j: int) -> int:
        if self.stop_at is not None and j == self.stop_at:
            return EOS
        if self.tokens is not None and self.prompt_len + j < len(self.tokens):
            return self.tokens[self.prompt_len + j]
        return 3 + j % 997


@dataclass(frozen=True)
class StepTime:
    """A step's GPU time, `base + per_token * T + per_row * rows` seconds, and the engine core's CPU time per schedule and
    per output it processes; `release` is the delay from processing a step's output to its tokens leaving through egress."""

    base: float = 0.012
    per_token: float = 4e-5
    per_row: float = 1e-4
    schedule: float = 4e-4
    update: float = 3e-4
    release: float = 2e-3

    def gpu(self, tokens: int, rows: int) -> float:
        return self.base + self.per_token * tokens + self.per_row * rows


def make_scheduler(engine: Engine, model: str, runner_v2: bool | None = None) -> Any:
    """vLLM's scheduler for `engine`. `runner_v2`: whether it takes its Model Runner V2 branches, set on the scheduler
    alone; None: as vLLM's environment says (`VLLM_USE_V2_MODEL_RUNNER`, which vLLM refuses on a node without Triton)."""
    os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")   # the executor is a fake: the model's own limit is moot
    import torch
    import vllm.envs as envs
    from vllm.config import CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig, SpeculativeConfig, VllmConfig
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.structured_output import StructuredOutputManager

    model_config = ModelConfig(model=model, trust_remote_code=False, dtype="float16", seed=0, skip_tokenizer_init=True,
                               max_model_len=engine.max_model_len)
    scheduler_config = SchedulerConfig(max_num_seqs=engine.max_num_seqs, max_num_batched_tokens=engine.max_num_batched_tokens,
                                       max_model_len=engine.max_model_len,
                                       long_prefill_token_threshold=engine.long_prefill_token_threshold,
                                       enable_chunked_prefill=engine.enable_chunked_prefill,
                                       async_scheduling=engine.async_scheduling, is_encoder_decoder=False,
                                       watermark=engine.watermark,
                                       scheduler_reserve_full_isl=engine.scheduler_reserve_full_isl)
    cache_config = CacheConfig(block_size=engine.block_size, gpu_memory_utilization=0.9, cache_dtype="auto",
                               enable_prefix_caching=engine.enable_prefix_caching)
    vllm_config = VllmConfig(scheduler_config=scheduler_config, model_config=model_config, cache_config=cache_config,
                             parallel_config=ParallelConfig())
    kv_cache_config = KVCacheConfig(num_blocks=engine.num_gpu_blocks, kv_cache_tensors=[], kv_cache_groups=[
        KVCacheGroupSpec(["layer"], FullAttentionSpec(block_size=engine.block_size, num_kv_heads=1, head_size=1,
                                                      dtype=torch.float32))])
    cache_config.num_gpu_blocks = engine.num_gpu_blocks
    if engine.num_speculative_tokens:
        # built as vLLM's scheduler tests build one, with no draft model to load; then given the method whose settings
        # the scheduler reads when it is made: `mtp`'s lookahead, drafting slots and trailing-block drop, which `header`
        # records as the scheduler holds them
        spec = SpeculativeConfig(model="ngram", num_speculative_tokens=engine.num_speculative_tokens)
        spec.method = "mtp"
        vllm_config.speculative_config = spec
    register_all_kvcache_specs(vllm_config)
    cls = AsyncScheduler if engine.async_scheduling else Scheduler
    sched = cls(vllm_config=vllm_config, kv_cache_config=kv_cache_config, block_size=engine.block_size, log_stats=False,
                structured_output_manager=StructuredOutputManager(vllm_config))
    sched.use_v2_model_runner = bool(envs.VLLM_USE_V2_MODEL_RUNNER) if runner_v2 is None else runner_v2
    return sched


def header(sched: Any, engine: Engine) -> dict[str, Any]:
    sc, cc = sched.scheduler_config, sched.cache_config
    spec = sched.vllm_config.speculative_config
    draft = {} if spec is None else {
        "speculative": {"method": spec.method, "num_speculative_tokens": int(spec.num_speculative_tokens),
                        "parallel_drafting": bool(spec.parallel_drafting),
                        "disable_eagle_block_drop": bool(spec.disable_eagle_block_drop)},
        "num_lookahead_tokens": int(sched.num_lookahead_tokens), "num_prefill_lookahead": int(sched.num_prefill_lookahead),
        "use_eagle_block_drop": bool(sched.use_eagle_block_drop), "draft_slots": int(spec.max_num_new_slots_for_drafting)}
    return {"kind": "header", "schema": STEP_TRACE, "source": "vllm-scheduler-fake-executor",
            "config": {"scheduler": type(sched).__name__, "max_num_batched_tokens": sc.max_num_batched_tokens,
                       "max_num_seqs": sc.max_num_seqs, "max_model_len": sched.max_model_len,
                       "enable_chunked_prefill": sc.enable_chunked_prefill,
                       "long_prefill_token_threshold": sc.long_prefill_token_threshold, "policy": str(sc.policy),
                       "async_scheduling": engine.async_scheduling, "block_size": cc.block_size,
                       "enable_prefix_caching": bool(cc.enable_prefix_caching), "num_gpu_blocks": cc.num_gpu_blocks,
                       "watermark": sc.watermark, "scheduler_reserve_full_isl": bool(sched.scheduler_reserve_full_isl),
                       **draft}}


def kept_drafts(seed: int, request: str, j: int, d: int, accept: float) -> int:
    """How many of the d drafts after output j of `request` the fake executor keeps: the drafts before the first one
    rejected, each kept with probability `accept`, a function of the seed, the request and j alone."""
    rng = random.Random(f"{seed}:{request}:{j}")
    k = 0
    while k < d and rng.random() < accept:
        k += 1
    return k


def run(engine: Engine, jobs: Sequence[Job], *, model: str, timing: StepTime = StepTime(), accept: float = 0.7,
        seed: int = 0, runner_v2: bool | None = None) -> list[dict[str, Any]]:
    """The step trace of `jobs` (sorted by arrival) through vLLM's scheduler (`make_scheduler`, with its `runner_v2`);
    with a draft, each draft is kept with probability `accept` (`kept_drafts`, from `seed`)."""
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
    from vllm.v1.request import Request

    if any(jobs[k].arrival > jobs[k + 1].arrival for k in range(len(jobs) - 1)):
        raise ValueError("jobs must be sorted by arrival")
    sched = make_scheduler(engine, model, runner_v2)
    init_none_hash(sha256)
    hasher = get_request_block_hasher(engine.block_size, sha256)
    recs: list[dict[str, Any]] = [header(sched, engine)]
    by_id = {f"r{k}": job for k, job in enumerate(jobs)}
    t = 0.0
    gpu_free = 0.0
    scheduled = 0
    empty = 0                       # schedules with no tokens since the last one with some
    nxt = 0
    queue: deque[tuple[Any, float, dict[str, bool], int | None]] = deque()
    K = engine.num_speculative_tokens

    def arrive() -> None:
        nonlocal nxt
        while nxt < len(jobs) and jobs[nxt].arrival <= t:
            job, rid = jobs[nxt], f"r{nxt}"
            sp = SamplingParams(max_tokens=job.max_tokens, ignore_eos=False)
            sp.update_from_generation_config({}, EOS)
            prompt = list(job.tokens[:job.prompt_len]) if job.tokens is not None else [3 + (nxt + p) % 997 for p in range(job.prompt_len)]
            sched.add_request(Request(request_id=rid, prompt_token_ids=prompt, sampling_params=sp, pooling_params=None,
                                      arrival_time=job.arrival, block_hasher=hasher))
            recs.append({"kind": "arrive", "request": rid, "before_step": scheduled,
                         **({"after_empty": empty} if empty else {}), "prompt_len": job.prompt_len,
                         "eos_token_id": EOS, "sampling": {"max_tokens": job.max_tokens, "stop": [], "ignore_eos": False},
                         "t": round(job.arrival, 6), "ingress_t": round(job.arrival - job.ingress_delay, 6)})
            nxt += 1

    def schedule() -> tuple[Any, dict[str, bool], int | None]:
        nonlocal t, scheduled, empty
        so = sched.schedule()
        t += timing.schedule
        sample = {rid: not sched.requests[rid].is_prefill_chunk for rid in so.num_scheduled_tokens}
        step = None
        if so.total_num_scheduled_tokens > 0:
            step = scheduled
            scheduled += 1
            empty = 0
        else:
            empty += 1
        return so, sample, step

    def execute(so: Any) -> float:
        nonlocal gpu_free
        if so.total_num_scheduled_tokens == 0:
            return t
        gpu_free = max(gpu_free, t) + timing.gpu(so.total_num_scheduled_tokens, len(so.num_scheduled_tokens))
        return gpu_free

    def sampled(so: Any, rid: str, fresh: set[str]) -> list[int]:
        """What the executor returns for a row that samples: the request's next output token, as its outputs stand when
        the step's output is processed, and then the drafts the row kept."""
        r = sched.requests.get(rid)
        j = 0 if r is None else r.num_output_tokens
        d = 0 if rid in fresh else len(so.scheduled_spec_decode_tokens.get(rid, ()))
        return [by_id[rid].token(j + m) for m in range(1 + kept_drafts(seed, rid, j, d, accept))]

    def update(so: Any, sample: dict[str, bool], step: int | None) -> None:
        nonlocal t
        ids = list(so.num_scheduled_tokens)
        cached = so.scheduled_cached_reqs
        fresh = {r.req_id for r in so.scheduled_new_reqs} | set(getattr(cached, "resumed_req_ids", None) or ())
        mro = ModelRunnerOutput(req_ids=ids, req_id_to_index={r: k for k, r in enumerate(ids)},
                                sampled_token_ids=[sampled(so, r, fresh) if sample[r] else [] for r in ids], logprobs=None,
                                prompt_logprobs_dict={}, pooler_output=[])
        starts = {}
        for r in so.scheduled_new_reqs:
            starts[r.req_id] = int(r.num_computed_tokens)
        for k, rid in enumerate(cached.req_ids):
            starts[rid] = int(cached.num_computed_tokens[k])
        outs = sched.update_from_output(so, mro)
        if K and not engine.async_scheduling and ids:
            sched.update_draft_token_ids(DraftTokenIds(ids, [[3 + m for m in range(K)] for _ in ids]))
        t += timing.update
        if step is None:
            return
        emitted: dict[str, int] = {}
        finished: dict[str, dict[str, Any]] = {}
        for eco in outs.values():
            for o in eco.outputs:
                if o.new_token_ids:
                    emitted[o.request_id] = len(o.new_token_ids)
                if o.finish_reason is not None:
                    finished[o.request_id] = {"reason": FINISH[int(o.finish_reason)], "stop_reason": o.stop_reason}
        new = [r.req_id for r in so.scheduled_new_reqs]
        recs.append({"kind": "step", "step": step, "T": int(so.total_num_scheduled_tokens), "order": "runner",
                     "rows": [[r, starts[r], int(so.num_scheduled_tokens[r])] for r in ids], "new": new,
                     "resumed": sorted(getattr(cached, "resumed_req_ids", None) or ()),
                     "cache_hit": {r.req_id: int(r.num_computed_tokens) for r in so.scheduled_new_reqs if r.num_computed_tokens},
                     "preempted": sorted(getattr(so, "preempted_req_ids", None) or ()), "emitted": emitted,
                     **({"drafts": {r: len(v) for r, v in so.scheduled_spec_decode_tokens.items() if v}} if K else {}),
                     "finished": finished, "t": round(t, 6), "release_t": round(t + timing.release, 6)})

    while nxt < len(jobs) or sched.has_requests() or queue:
        if not sched.has_requests() and not queue:
            t = max(t, jobs[nxt].arrival)
        arrive()
        if not engine.async_scheduling:
            if not sched.has_requests():
                continue
            so, sample, step = schedule()
            t = execute(so)
            update(so, sample, step)
            continue
        if sched.has_requests():
            so, sample, step = schedule()
            queue.appendleft((so, execute(so), sample, step))
            if len(queue) < 2 and (so.total_num_scheduled_tokens > 0 or sched.has_requests()):
                continue
        elif not queue:
            continue
        so, done, sample, step = queue.pop()
        t = max(t, done)
        update(so, sample, step)
    return recs


def workload(n: int, *, rate: float, prompt: tuple[int, int], output: tuple[int, int], max_tokens: int | None = None,
             p_eos: float = 0.8, shared_prefix: int = 0, vocab: int = 50, ingress_delay: tuple[float, float] = (0.003, 0.03),
             fifo: bool = True, one_token: float = 0.0, seed: int = 0) -> list[Job]:
    """`n` Poisson arrivals at `rate` per second, prompt and output lengths uniform in their ranges; a fraction `p_eos` stop on
    EOS before `max_tokens` (default: the output range's top). With `shared_prefix` > 0 every prompt starts with the same
    tokens, over a small vocabulary, so prefix caching has hits. Each request reaches the engine `ingress_delay` after the
    ingress certifier saw it; `fifo` keeps the certified order the engine's (a frontend that reorders makes `fifo` False).
    A fraction `one_token` of the prompts are one token: a request whose first row is a decode row."""
    rng = random.Random(seed)
    t, jobs, seen = 0.0, [], float("-inf")
    system = [3 + rng.randrange(vocab) for _ in range(shared_prefix)]
    for _ in range(n):
        t += rng.expovariate(rate)
        P = rng.randint(*prompt)
        if one_token and rng.random() < one_token:
            P = 1
        mt = max_tokens or output[1]
        out = rng.randint(*output)
        stop_at = min(out, mt) - 1 if rng.random() < p_eos and out < mt else None
        toks = None
        if shared_prefix:
            body = system[:P] + [3 + rng.randrange(vocab) for _ in range(max(P - shared_prefix, 0))]
            outs = [3 + rng.randrange(vocab) for _ in range(mt)]
            if stop_at is not None:
                outs[stop_at] = EOS
            toks = tuple(body + outs)
        ingress = t - rng.uniform(*ingress_delay)
        if fifo:
            ingress = seen = max(seen, ingress)
        jobs.append(Job(arrival=t, prompt_len=P, max_tokens=mt, stop_at=stop_at, tokens=toks, ingress_delay=t - ingress))
    return jobs


def tokens_of(jobs: Sequence[Job], recs: Sequence[dict[str, Any]]) -> dict[str, tuple[int, ...]]:
    """Each request's tokens as the certified I/O shows them (prompt, then the output it emitted), for `replay.load_step_trace`."""
    emitted: dict[str, int] = {}
    for r in recs:
        if r["kind"] == "step":
            for rid, k in r["emitted"].items():
                emitted[rid] = emitted.get(rid, 0) + k
    out = {}
    for k, job in enumerate(jobs):
        rid = f"r{k}"
        prompt = list(job.tokens[:job.prompt_len]) if job.tokens is not None else [3 + (k + p) % 997 for p in range(job.prompt_len)]
        out[rid] = tuple(prompt + [job.token(j) for j in range(emitted.get(rid, 0))])
    return out


#: small workloads that exercise the served paths: chunked prefill, decode batching, KV pressure with preemption, async
#: scheduling, and prefix caching (name -> (engine, workload kwargs))
CASES: dict[str, tuple[Engine, dict[str, Any]]] = {
    "sync-light": (Engine(max_num_seqs=8, max_num_batched_tokens=256, max_model_len=1024, num_gpu_blocks=2048),
                   dict(n=40, rate=4.0, prompt=(8, 600), output=(1, 120))),
    "async-light": (Engine(max_num_seqs=8, max_num_batched_tokens=256, max_model_len=1024, num_gpu_blocks=2048,
                           async_scheduling=True), dict(n=40, rate=4.0, prompt=(8, 600), output=(1, 120))),
    "async-mid": (Engine(max_num_seqs=16, max_num_batched_tokens=512, max_model_len=2048, num_gpu_blocks=2048,
                         async_scheduling=True), dict(n=60, rate=12.0, prompt=(8, 1500), output=(1, 200))),
    "sync-roomy": (Engine(max_num_seqs=8, max_num_batched_tokens=256, max_model_len=1024, num_gpu_blocks=2048),
                   dict(n=60, rate=40.0, prompt=(8, 600), output=(1, 120))),
    "sync-kv-pressure": (Engine(max_num_seqs=16, max_num_batched_tokens=256, max_model_len=1024, num_gpu_blocks=96),
                         dict(n=60, rate=60.0, prompt=(8, 400), output=(20, 300))),
    "async-roomy": (Engine(max_num_seqs=8, max_num_batched_tokens=256, max_model_len=1024, num_gpu_blocks=2048,
                           async_scheduling=True), dict(n=60, rate=40.0, prompt=(8, 600), output=(1, 120))),
    "async-kv-pressure": (Engine(max_num_seqs=16, max_num_batched_tokens=256, max_model_len=1024, num_gpu_blocks=96,
                                 async_scheduling=True), dict(n=60, rate=60.0, prompt=(8, 400), output=(20, 300))),
    "async-long-prefill": (Engine(max_num_seqs=32, max_num_batched_tokens=512, max_model_len=4096, num_gpu_blocks=1024,
                                  long_prefill_token_threshold=128, async_scheduling=True),
                           dict(n=80, rate=80.0, prompt=(8, 2000), output=(1, 200))),
    "async-prefix-cache": (Engine(max_num_seqs=16, max_num_batched_tokens=256, max_model_len=1024, num_gpu_blocks=160,
                                  enable_prefix_caching=True, async_scheduling=True),
                           dict(n=60, rate=50.0, prompt=(40, 300), output=(5, 150), shared_prefix=64, vocab=4)),
}
# with a draft of K = 1, 3 and 5 tokens, sync and async, with and without prefix caching: a KV pool under the requests'
# blocks and their lookahead (preemption), one-token prompts (a new decode row, padded) and prompts whose output runs
# into the context limit (a row's drafts are cut there)
_DRAFT = dict(n=40, prompt=(1, 120), one_token=0.25)
_SHARED = dict(shared_prefix=48, vocab=4)


def _draft(k: int, asy: bool, cache: bool = False, *, seqs: int = 16, budget: int = 96, blocks: int | None = None) -> DraftEngine:
    return DraftEngine(max_num_seqs=seqs, max_num_batched_tokens=budget, max_model_len=160,
                       num_gpu_blocks=blocks or (40 if cache else 48), enable_prefix_caching=cache, async_scheduling=asy,
                       num_speculative_tokens=k)


CASES |= {f"{'async' if asy else 'sync'}-draft{k}{'-prefix-cache' if cache else ''}": (
    _draft(k, asy, cache), _DRAFT | dict(rate=60.0, output=(1, 60)) | (_SHARED if cache else {}))
    for k in (1, 3, 5) for asy in (False, True) for cache in (False, True)}
# a token budget under the running rows' (the budget cuts a row's drafts, and leaves a request out of a step); and a
# light load of short outputs, where the engine goes quiet and a request arrives after a schedule with no tokens
CASES |= {
    "sync-draft1-tight": (_draft(1, False, seqs=8, budget=9, blocks=40), _DRAFT | dict(rate=60.0, output=(1, 60))),
    "sync-draft5-tight": (_draft(5, False, budget=32), _DRAFT | dict(rate=60.0, output=(1, 60))),
    "async-draft3-prefix-cache-tight": (_draft(3, True, True, budget=32, blocks=48),
                                        _DRAFT | dict(rate=60.0, output=(1, 60)) | _SHARED),
    "async-draft1-light": (_draft(1, True, seqs=4, budget=48, blocks=72), _DRAFT | dict(rate=10.0, output=(1, 12))),
    "async-draft5-prefix-cache-light": (_draft(5, True, True, seqs=4, budget=48, blocks=72),
                                        _DRAFT | dict(rate=10.0, output=(1, 12)) | _SHARED),
}


def exercised(t: Any) -> dict[str, Any]:
    """What a trace (`replay.Trace`) with a draft exercises of the scheduler's draft rules: its rows that carried drafts
    (`rows_with_drafts`); a new or resumed request's, which are padding (`padded`); the rows of a request that held K
    drafts, its last row having emitted, and carried fewer, because the row ends at the context limit (`cut_context`) or
    because the step's budget ran out (`cut_budget`); how often a row kept each count (`kept`); and the requests added
    after a schedule with no tokens (`after_empty`)."""
    K, L = t.engine.num_speculative_tokens, t.engine.max_model_len
    out = {"rows_with_drafts": 0, "padded": 0, "cut_context": 0, "cut_budget": 0}
    emitted: dict[int, bool] = {}
    for k, s in enumerate(t.steps):
        for i, start, n in s.rows:
            d = s.drafts.get(i, 0)
            out["rows_with_drafts"] += d > 0
            if i in s.new or i in s.resumed:
                out["padded"] += d > 0
            elif emitted.get(i) and d < K:
                out["cut_context" if start + n == L - 1 else "cut_budget"] += 1
            emitted[i] = bool(t.emitted[k].get(i))
    kept = [k for q in t.requests for k in q.kept or ()]
    return out | {"kept": {str(k): kept.count(k) for k in range(K + 1)}, "after_empty": sum(1 for p in t.passed or () if p)}


def check(recs: Sequence[dict[str, Any]], tokens: dict[str, Sequence[int]] | None = None) -> dict[str, Any]:
    """The replay against a trace of this module: whether its steps are the trace's (`match`: every step's rows in order
    and, with a draft, the drafts each row carried; else `first_mismatch` with both sides' rows, or `refused`), and what
    the trace exercises (`exercised`)."""
    t = _replay.load_step_trace(recs, tokens=tokens)
    steps = [r for r in recs if r["kind"] == "step"]
    out: dict[str, Any] = {
        "steps": len(steps), "requests": len(t.requests), "preemptions": sum(len(r["preempted"]) for r in steps),
        "cache_hits": sum(len(r["cache_hit"]) for r in steps),
        "finish": {f: sum(q["reason"] == f for r in steps for q in r["finished"].values()) for f in ("stop", "length")}}
    if t.engine.num_speculative_tokens:
        out |= exercised(t)
    try:
        rp = _replay.replay(t.engine, t.requests, admit=t.before_step, passed=t.passed)
    except (ValueError, RuntimeError, AssertionError) as e:
        return out | {"match": False, "refused": f"{type(e).__name__}: {e}"}
    m = _replay.compare(rp.steps, t.steps)
    same = [k for k in range(min(len(rp.steps), len(t.steps)))
            if rp.steps[k].rows != t.steps[k].rows or rp.steps[k].drafts != t.steps[k].drafts]
    first = same[0] if same else m.first_mismatch
    out |= {"match": first is None and m.ordered == m.steps, "matched": m.matched, "ordered": m.ordered,
            "first_mismatch": first}
    if first is not None:
        side = lambda ss: None if first >= len(ss) else {"rows": ss[first].rows, "drafts": ss[first].drafts}  # noqa: E731
        out |= {"replay": side(rp.steps), "trace": side(t.steps)}
    return out


def main(argv: Sequence[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path, help="directory to write <case>.jsonl (and <case>.tokens.json with prefix caching) into")
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--case", action="append", choices=sorted(CASES))
    ap.add_argument("--draft", action="store_true", help="the cases with a draft")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--requests", type=int, help="override each case's request count")
    ap.add_argument("--accept", type=float, default=0.7, help="with a draft, the probability each draft is kept")
    ap.add_argument("--runner-v2", action="store_true", help="the scheduler's Model Runner V2 branches: a resumed request "
                    "is handed over as a new one, so a step's `resumed` is empty")
    ap.add_argument("--check", action="store_true", help="replay each trace (`check`) into OUT/check.json; exit 1 unless "
                    "every replay's steps are its trace's")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    checks = {}
    for name in a.case or sorted(n for n in CASES if not a.draft or CASES[n][0].num_speculative_tokens):
        engine, kw = CASES[name]
        jobs = workload(seed=a.seed, **(kw | ({"n": a.requests} if a.requests else {})))
        recs = run(engine, jobs, model=a.model, accept=a.accept, seed=a.seed, runner_v2=a.runner_v2 or None)
        (a.out / f"{name}.jsonl").write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in recs))
        toks = tokens_of(jobs, recs) if engine.enable_prefix_caching else None
        if toks is not None:
            (a.out / f"{name}.tokens.json").write_text(json.dumps(toks))
        print(name, sum(r["kind"] == "step" for r in recs), "steps")
        if a.check:
            checks[name] = c = check(recs, toks)
            print(" ", "replay matches" if c["match"] else f"REPLAY DIFFERS: {c.get('refused') or c.get('first_mismatch')}")
    if a.check:
        (a.out / "check.json").write_text(json.dumps(checks, indent=1) + "\n")
        if not all(c["match"] for c in checks.values()):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
