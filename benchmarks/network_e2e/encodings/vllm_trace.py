"""Step traces from vLLM V1's own scheduler on a fake model executor, on CPU: what `replay` is checked against.

`run` builds vLLM's `Scheduler` or `AsyncScheduler` as vLLM's tests do (`tests/v1/core/utils.py`, `create_scheduler`) and drives
it through the engine core's two loops (`EngineCore.step`, `EngineCore.step_with_batch_queue` with a batch queue of 2) on a
virtual clock. The executor is a fake: a row samples iff the scheduler marked it not a prefill chunk, and the token it samples
at output index j is the job's (EOS at its stop index). The GPU runs steps in order, each for `StepTime` of its shape, and
the engine core blocks only where it does: on an empty input queue, and on the oldest step's output.

It writes `verity-vllm/step-trace/v1` records (`verity_vllm.engine.step_trace`): a `header`, an `arrive` per request with the
schedules-with-tokens count it arrived before (`before_step`), and a `step` per executed step at the time its output was
processed. Records carry the engine's clock `t`, and the certifier's: `ingress_t`, when the ingress certifier saw the request
(its engine arrival less a frontend delay), and `release_t`, when the step's tokens left through the egress certifier.

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
STEP_TRACE, Engine = _replay.STEP_TRACE, _replay.Engine

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


def make_scheduler(engine: Engine, model: str) -> Any:
    os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")   # the executor is a fake: the model's own limit is moot
    import torch
    import vllm.envs as envs
    from vllm.config import CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig
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
    register_all_kvcache_specs(vllm_config)
    cls = AsyncScheduler if engine.async_scheduling else Scheduler
    sched = cls(vllm_config=vllm_config, kv_cache_config=kv_cache_config, block_size=engine.block_size, log_stats=False,
                structured_output_manager=StructuredOutputManager(vllm_config))
    sched.use_v2_model_runner = bool(envs.VLLM_USE_V2_MODEL_RUNNER)
    return sched


def header(sched: Any, engine: Engine) -> dict[str, Any]:
    sc, cc = sched.scheduler_config, sched.cache_config
    return {"kind": "header", "schema": STEP_TRACE, "source": "vllm-scheduler-fake-executor",
            "config": {"scheduler": type(sched).__name__, "max_num_batched_tokens": sc.max_num_batched_tokens,
                       "max_num_seqs": sc.max_num_seqs, "max_model_len": sched.max_model_len,
                       "enable_chunked_prefill": sc.enable_chunked_prefill,
                       "long_prefill_token_threshold": sc.long_prefill_token_threshold, "policy": str(sc.policy),
                       "async_scheduling": engine.async_scheduling, "block_size": cc.block_size,
                       "enable_prefix_caching": bool(cc.enable_prefix_caching), "num_gpu_blocks": cc.num_gpu_blocks,
                       "watermark": sc.watermark, "scheduler_reserve_full_isl": bool(sched.scheduler_reserve_full_isl)}}


def run(engine: Engine, jobs: Sequence[Job], *, model: str, timing: StepTime = StepTime()) -> list[dict[str, Any]]:
    """The step trace of `jobs` (sorted by arrival) through vLLM's scheduler."""
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request

    if any(jobs[k].arrival > jobs[k + 1].arrival for k in range(len(jobs) - 1)):
        raise ValueError("jobs must be sorted by arrival")
    sched = make_scheduler(engine, model)
    init_none_hash(sha256)
    hasher = get_request_block_hasher(engine.block_size, sha256)
    recs: list[dict[str, Any]] = [header(sched, engine)]
    by_id = {f"r{k}": job for k, job in enumerate(jobs)}
    t = 0.0
    gpu_free = 0.0
    scheduled = 0
    nxt = 0
    queue: deque[tuple[Any, float, dict[str, int | None], int | None]] = deque()

    def arrive() -> None:
        nonlocal nxt
        while nxt < len(jobs) and jobs[nxt].arrival <= t:
            job, rid = jobs[nxt], f"r{nxt}"
            sp = SamplingParams(max_tokens=job.max_tokens, ignore_eos=False)
            sp.update_from_generation_config({}, EOS)
            prompt = list(job.tokens[:job.prompt_len]) if job.tokens is not None else [3 + (nxt + p) % 997 for p in range(job.prompt_len)]
            sched.add_request(Request(request_id=rid, prompt_token_ids=prompt, sampling_params=sp, pooling_params=None,
                                      arrival_time=job.arrival, block_hasher=hasher))
            recs.append({"kind": "arrive", "request": rid, "before_step": scheduled, "prompt_len": job.prompt_len,
                         "eos_token_id": EOS, "sampling": {"max_tokens": job.max_tokens, "stop": [], "ignore_eos": False},
                         "t": round(job.arrival, 6), "ingress_t": round(job.arrival - job.ingress_delay, 6)})
            nxt += 1

    def schedule() -> tuple[Any, dict[str, int | None], int | None]:
        nonlocal t, scheduled
        so = sched.schedule()
        t += timing.schedule
        sample: dict[str, int | None] = {}
        for rid in so.num_scheduled_tokens:
            r = sched.requests[rid]
            sample[rid] = None if r.is_prefill_chunk else by_id[rid].token(r.num_computed_tokens - r.num_prompt_tokens)
        step = None
        if so.total_num_scheduled_tokens > 0:
            step = scheduled
            scheduled += 1
        return so, sample, step

    def execute(so: Any) -> float:
        nonlocal gpu_free
        if so.total_num_scheduled_tokens == 0:
            return t
        gpu_free = max(gpu_free, t) + timing.gpu(so.total_num_scheduled_tokens, len(so.num_scheduled_tokens))
        return gpu_free

    def update(so: Any, sample: dict[str, int | None], step: int | None) -> None:
        nonlocal t
        ids = list(so.num_scheduled_tokens)
        mro = ModelRunnerOutput(req_ids=ids, req_id_to_index={r: k for k, r in enumerate(ids)},
                                sampled_token_ids=[[] if sample[r] is None else [sample[r]] for r in ids], logprobs=None,
                                prompt_logprobs_dict={}, pooler_output=[])
        starts = {}
        for r in so.scheduled_new_reqs:
            starts[r.req_id] = int(r.num_computed_tokens)
        cached = so.scheduled_cached_reqs
        for k, rid in enumerate(cached.req_ids):
            starts[rid] = int(cached.num_computed_tokens[k])
        outs = sched.update_from_output(so, mro)
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
             fifo: bool = True, seed: int = 0) -> list[Job]:
    """`n` Poisson arrivals at `rate` per second, prompt and output lengths uniform in their ranges; a fraction `p_eos` stop on
    EOS before `max_tokens` (default: the output range's top). With `shared_prefix` > 0 every prompt starts with the same
    tokens, over a small vocabulary, so prefix caching has hits. Each request reaches the engine `ingress_delay` after the
    ingress certifier saw it; `fifo` keeps the certified order the engine's (a frontend that reorders makes `fifo` False)."""
    rng = random.Random(seed)
    t, jobs, seen = 0.0, [], float("-inf")
    system = [3 + rng.randrange(vocab) for _ in range(shared_prefix)]
    for _ in range(n):
        t += rng.expovariate(rate)
        P = rng.randint(*prompt)
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


def main(argv: Sequence[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path, help="directory to write <case>.jsonl (and <case>.tokens.json with prefix caching) into")
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--case", action="append", choices=sorted(CASES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--requests", type=int, help="override each case's request count")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    for name in a.case or sorted(CASES):
        engine, kw = CASES[name]
        jobs = workload(seed=a.seed, **(kw | ({"n": a.requests} if a.requests else {})))
        recs = run(engine, jobs, model=a.model)
        (a.out / f"{name}.jsonl").write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in recs))
        if engine.enable_prefix_caching:
            (a.out / f"{name}.tokens.json").write_text(json.dumps(tokens_of(jobs, recs)))
        print(name, sum(r["kind"] == "step" for r in recs), "steps")


if __name__ == "__main__":
    main()
