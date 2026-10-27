"""A small step trace of varied traffic through vLLM's own scheduler, on CPU: the trace the derived encoding is measured on
before the trace corpus's real ones, and the ground truth its derivations are tested against.

It drives the pinned vLLM's `Scheduler` (or `AsyncScheduler`) and `KVCacheManager`, unmodified, with a fake model runner:
the runner returns one sampled id for every row that isn't a non-final prefill chunk (`Request.is_prefill_chunk`, the GPU
runner's discard rule) and nothing else runs. So chunked prefill, the prefix cache (sha256 block hashes, LRU eviction),
admission, preemption and the async scheduler's one-step lag are vLLM's; only the sampled ids and the step times are made
up. The sampled ids are a seeded function of (request, position), with a stop id at a seeded rate, and each step lasts
`base_ms + per_token_ms` times its tokens, so arrivals (Poisson, on that step clock) are admitted between steps as the
engine core admits them.

    python sched_trace.py --out trace.json [--seed 0] [--rate 3] [--seconds 150] [--num-blocks 35000] [--async]

It needs the pinned vLLM importable (`verity_bench.vllm.presets.VLLM_PIN`): on a GPU node's environment, or on any CPU from
the source tree with its `requirements/common.txt` and a CPU torch (nothing compiled: the scheduler is pure Python), with
the tree on PYTHONPATH. The trace's format is `derived.load_trace`'s.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
from pathlib import Path

LLAMA31_8B = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 4096, "intermediate_size": 14336,
              "num_attention_heads": 32, "num_hidden_layers": 32, "num_key_value_heads": 8, "vocab_size": 128256,
              "max_position_embeddings": 131072, "rms_norm_eps": 1e-5, "rope_theta": 500000.0, "head_dim": 128,
              "rope_scaling": {"factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                               "original_max_position_embeddings": 8192, "rope_type": "llama3"},
              "bos_token_id": 128000, "eos_token_id": [128001, 128008, 128009], "torch_dtype": "bfloat16"}
EOS = 128009                      # Llama-3.1-Instruct's tokenizer EOS, <|eot_id|>
STOP_IDS = [128001, 128008]       # the rest of its generation_config's eos_token_id, which vLLM adds as stop ids
SYSTEM_PROMPTS = (32, 100, 256, 500)


def scheduler(*, num_blocks: int, block_size: int, max_num_seqs: int, max_num_batched_tokens: int, max_model_len: int,
              async_scheduling: bool, enable_prefix_caching: bool):
    import torch
    from vllm.config import CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.structured_output import StructuredOutputManager

    model_dir = Path(tempfile.mkdtemp(prefix="llama31-8b-config-"))
    (model_dir / "config.json").write_text(json.dumps(LLAMA31_8B))
    model_config = ModelConfig(model=str(model_dir), dtype="bfloat16", seed=0, skip_tokenizer_init=True, max_model_len=max_model_len)
    scheduler_config = SchedulerConfig(max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens,
                                       max_model_len=max_model_len, enable_chunked_prefill=True,
                                       async_scheduling=async_scheduling, is_encoder_decoder=False)
    cache_config = CacheConfig(block_size=block_size, gpu_memory_utilization=0.9, cache_dtype="auto",
                               enable_prefix_caching=enable_prefix_caching)
    vllm_config = VllmConfig(scheduler_config=scheduler_config, model_config=model_config, cache_config=cache_config,
                             parallel_config=ParallelConfig())
    spec = FullAttentionSpec(block_size=block_size, num_kv_heads=8, head_size=128, dtype=torch.bfloat16)
    kv_cache_config = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[],
                                    kv_cache_groups=[KVCacheGroupSpec([f"model.layers.{i}.self_attn.attn" for i in range(32)], spec)])
    cache_config.num_gpu_blocks = num_blocks
    try:
        from vllm.v1.kv_cache_interface import register_all_kvcache_specs
        register_all_kvcache_specs(vllm_config)
    except ImportError:
        pass
    cls = AsyncScheduler if async_scheduling else Scheduler
    return cls(vllm_config=vllm_config, kv_cache_config=kv_cache_config, block_size=block_size, log_stats=False,
               structured_output_manager=StructuredOutputManager(vllm_config))


def workload(rng: random.Random, *, rate: float, seconds: float, max_model_len: int) -> list[dict]:
    """Poisson arrivals over `seconds`: prompts of 16-1024 ids (log-uniform), 40% behind one of four shared system prompts,
    10% an earlier prompt again, the rest fresh; max_tokens one of 16, 64, 128, 256, 512."""
    systems = [[rng.randrange(1000, 128000) for _ in range(n)] for n in SYSTEM_PROMPTS]
    out: list[dict] = []
    t = 0.0
    while True:
        t += rng.expovariate(rate)
        if t >= seconds:
            return out
        P = int(round(math.exp(rng.uniform(math.log(16), math.log(1024)))))
        kind = rng.random()
        if kind < 0.10 and out:
            prompt = list(rng.choice(out)["prompt_ids"])
        elif kind < 0.50:
            system = rng.choice(systems)
            prompt = (system + [rng.randrange(1000, 128000) for _ in range(max(P - len(system), 8))])[:max(P, len(system) + 8)]
        else:
            prompt = [rng.randrange(1000, 128000) for _ in range(P)]
        max_tokens = min(rng.choice((16, 64, 128, 256, 512)), max_model_len - len(prompt))
        out.append({"id": f"user-{len(out)}", "arrival_s": round(t, 6), "prompt_ids": prompt, "max_tokens": max_tokens})


def sampled_id(seed: int, request: int, position: int, stop_rate: float) -> int:
    r = random.Random(hash((seed, request, position)) & 0xFFFFFFFF)
    if r.random() < stop_rate:
        return EOS if r.random() < 0.9 else r.choice(STOP_IDS)
    return r.randrange(1000, 128000)


def run(*, seed: int = 0, rate: float = 3.0, seconds: float = 150.0, num_blocks: int = 35000, block_size: int = 16,
        max_num_seqs: int = 256, max_num_batched_tokens: int = 8192, max_model_len: int = 4096, async_scheduling: bool = False,
        enable_prefix_caching: bool = True, stop_rate: float = 1 / 150, base_ms: float = 10.0, per_token_ms: float = 0.04) -> dict:
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request, RequestStatus

    rng = random.Random(seed)
    reqs = workload(rng, rate=rate, seconds=seconds, max_model_len=max_model_len)
    index = {r["id"]: i for i, r in enumerate(reqs)}
    init_none_hash(sha256)
    hasher = get_request_block_hasher(block_size, sha256)
    sched = scheduler(num_blocks=num_blocks, block_size=block_size, max_num_seqs=max_num_seqs,
                      max_num_batched_tokens=max_num_batched_tokens, max_model_len=max_model_len,
                      async_scheduling=async_scheduling, enable_prefix_caching=enable_prefix_caching)
    objs: dict[str, Request] = {}
    produced = [0] * len(reqs)                    # samples the runner has returned (or will) for each request
    owner: dict[int, int] = {}                    # each KV block's computing request: rows run in scheduling order
    outputs: list[list[int]] = [[] for _ in reqs]
    steps: list[dict] = []
    clock, admitted, idle = 0.0, 0, 0
    inflight: list[tuple] = []
    depth = 2 if async_scheduling else 1

    def admit(now: float) -> None:
        nonlocal admitted
        while admitted < len(reqs) and reqs[admitted]["arrival_s"] <= now:
            r = reqs[admitted]
            sp = SamplingParams(max_tokens=r["max_tokens"], temperature=0.0, stop_token_ids=list(STOP_IDS))
            sp.update_from_generation_config({}, EOS)
            req = Request(request_id=r["id"], prompt_token_ids=list(r["prompt_ids"]), sampling_params=sp, pooling_params=None,
                          block_hasher=hasher)
            objs[r["id"]] = req
            sched.add_request(req)
            admitted += 1

    def finish_one() -> None:
        so, rows, out = inflight.pop(0)
        sched.update_from_output(so, out)

    while admitted < len(reqs) or sched.get_num_unfinished_requests() or inflight:
        admit(clock)
        if not sched.get_num_unfinished_requests() and not inflight:
            clock = reqs[admitted]["arrival_s"]
            continue
        so = sched.schedule()
        rows = []
        new = {d.req_id: d for d in so.scheduled_new_reqs}
        cached = so.scheduled_cached_reqs
        cached_start = dict(zip(cached.req_ids, cached.num_computed_tokens))
        resumed = set(getattr(cached, "resumed_req_ids", ()) or ())
        sampled = []
        for rid, n in so.num_scheduled_tokens.items():
            i = index[rid]
            start = int(new[rid].num_computed_tokens if rid in new else cached_start[rid])
            samples = not objs[rid].is_prefill_chunk
            blocks = sched.kv_cache_manager.get_block_ids(rid)[0]
            row = {"request": i, "start": start, "n": int(n), "samples": samples, "new": rid in new, "resumed": rid in resumed}
            if (rid in new or rid in resumed) and start:
                runs: list[list[int]] = []
                for b in blocks[:start // block_size]:
                    if runs and runs[-1][0] == owner[b]:
                        runs[-1][1] += block_size
                    else:
                        runs.append([owner[b], block_size])
                row["sources"] = runs
            for j in range(start // block_size, -(-(start + n) // block_size)):
                owner[blocks[j]] = i
            rows.append(row)
            if samples:
                tok = sampled_id(seed, i, produced[i], stop_rate)
                produced[i] += 1
                sampled.append([tok])
            else:
                sampled.append([])
        ids = list(so.num_scheduled_tokens)
        out = ModelRunnerOutput(req_ids=ids, req_id_to_index={r: k for k, r in enumerate(ids)}, sampled_token_ids=sampled,
                                logprobs=None, prompt_logprobs_dict={}, pooler_output=[])
        if not rows:
            while inflight:
                finish_one()
            sched.update_from_output(so, out)
            idle += 1
            if admitted < len(reqs):
                clock = max(clock, reqs[admitted]["arrival_s"])
            elif idle > 3 and not inflight:
                raise RuntimeError("the scheduler schedules nothing with requests left")
        else:
            idle = 0
            clock += (base_ms + per_token_ms * sum(r["n"] for r in rows)) / 1000.0
            steps.append({"t_end_s": round(clock, 6), "rows": rows})
            inflight.append((so, rows, out))
            if len(inflight) >= depth:
                finish_one()
        for rid, req in objs.items():
            i = index[rid]
            if req.is_finished() and "finish" not in reqs[i]:
                reqs[i]["finish"] = {RequestStatus.FINISHED_STOPPED: "eos", RequestStatus.FINISHED_LENGTH_CAPPED: "length"}.get(
                    req.status, str(req.status))
                reqs[i]["output_ids"] = list(req.output_token_ids)
                reqs[i]["preemptions"] = int(req.num_preemptions)
    for rid, req in objs.items():
        i = index[rid]
        if "finish" not in reqs[i]:
            raise RuntimeError(f"request {rid} never finished")
    config = {"vllm": "d9105ea8001e0a6d77a96327d17515bb5791fb36", "scheduler": "async" if async_scheduling else "sync",
              "block_size": block_size, "num_gpu_blocks": num_blocks, "max_num_seqs": max_num_seqs,
              "max_num_batched_tokens": max_num_batched_tokens, "max_model_len": max_model_len,
              "long_prefill_token_threshold": 0, "enable_prefix_caching": enable_prefix_caching, "watermark": 0.0,
              "eos_token_id": EOS, "stop_token_ids": STOP_IDS, "row_order": "scheduling"}
    workload_desc = {"seed": seed, "rate_per_s": rate, "seconds": seconds, "stop_rate": stop_rate, "base_ms": base_ms,
                     "per_token_ms": per_token_ms, "system_prompts": list(SYSTEM_PROMPTS)}
    return {"kind": "enc-derived/sched-trace/v2", "config": config, "workload": workload_desc, "requests": reqs, "steps": steps}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rate", type=float, default=3.0)
    ap.add_argument("--seconds", type=float, default=150.0)
    ap.add_argument("--num-blocks", type=int, default=35000)
    ap.add_argument("--max-num-batched-tokens", type=int, default=8192)
    ap.add_argument("--async", dest="async_scheduling", action="store_true")
    ap.add_argument("--no-prefix-caching", dest="prefix_caching", action="store_false")
    a = ap.parse_args(argv)
    trace = run(seed=a.seed, rate=a.rate, seconds=a.seconds, num_blocks=a.num_blocks, async_scheduling=a.async_scheduling,
                max_num_batched_tokens=a.max_num_batched_tokens, enable_prefix_caching=a.prefix_caching)
    a.out.write_text(json.dumps(trace, separators=(",", ":")))
    n_rows = sum(len(s["rows"]) for s in trace["steps"])
    print(json.dumps({"out": str(a.out), "requests": len(trace["requests"]), "steps": len(trace["steps"]), "rows": n_rows,
                      "preemptions": sum(r["preemptions"] for r in trace["requests"]),
                      "finish": {k: sum(r["finish"] == k for r in trace["requests"]) for k in ("eos", "length")}}))


if __name__ == "__main__":
    main()
