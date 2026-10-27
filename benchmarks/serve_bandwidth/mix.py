"""A served mix on vLLM for nsys's DRAM sampling: Llama-3.1 (8B on one GPU, 70B at `--tp 8`) in FP8 online
quantization with full CUDA graphs for decode (`kv_filler/serve.py`'s stock engine), serving `serve.trace`'s bursty
arrivals through the engine's step loop, with one NVTX range an engine step and one, `trace`, around the whole trace
(nsys's `--capture-range=nvtx --nvtx-capture=trace`).

A step's range runs from the scheduler's output to its `update_from_output` and is named
`s<i> p<prefill tokens> d<decode requests> kv<context tokens>`; a range `idle` covers each wait for the next arrival
with nothing to run. At `--tp 1` a range `x<i> ...` also covers each model step's launches in the runner (its
`execute_model` and `sample_tokens`), so `timeline.py` finds each kernel's step by its launch, with async scheduling
(vLLM's default, `--async-scheduling`) on or off. At TP > 1 the kernels launch in the workers, and `timeline.py` places
samples by the scheduler's ranges, which hold each step's GPU work only with async scheduling off. Writes `mix.json` (the engine, the weights' and KV's bytes a step per GPU, every step's record) under `--out`.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kv_filler"))

from serve import CAPTURE, trace  # noqa: E402


def per_gpu_bytes(cfg, tp: int) -> dict:
    """The bytes a step reads per GPU from what is registered: every linear's FP8 weights (vLLM's online FP8 keeps the
    LM head in BF16), and a token of context's KV in BF16."""
    h, layers = cfg.hidden_size, cfg.num_hidden_layers
    hd = getattr(cfg, "head_dim", None) or h // cfg.num_attention_heads
    q, kv = cfg.num_attention_heads * hd, cfg.num_key_value_heads * hd
    linear = layers * (h * (q + 2 * kv) + q * h + 3 * h * cfg.intermediate_size)
    return {"linear_weight_bytes": linear // tp, "lm_head_bytes": 2 * cfg.vocab_size * h // tp,
            "weight_bytes_per_step": linear // tp + 2 * cfg.vocab_size * h // tp,
            "kv_bytes_per_token": 2 * layers * 2 * kv // tp}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="/workspace/pouw/mvp-e2e/models/llama-3.1-8b-instruct")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--mem", type=float, default=0.85)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--burst-gap-s", type=float, default=3.0)
    ap.add_argument("--burst-mean", type=int, default=12)
    ap.add_argument("--prompts", type=lambda s: [int(v) for v in s.split(",")], default=[256, 1024, 2048])
    ap.add_argument("--outputs", type=lambda s: [int(v) for v in s.split(",")], default=[64, 256, 512])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-batched-tokens", type=int, default=8192)
    ap.add_argument("--max-seqs", type=int, default=max(CAPTURE))
    ap.add_argument("--async-scheduling", choices=("default", "on", "off"), default="default")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    import torch
    import vllm
    from verity_vllm.engine import handle
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    reqs = trace(a.seed, a.seconds, a.burst_gap_s, a.burst_mean, a.prompts, a.outputs)
    kw = {"model": a.model, "quantization": "fp8", "tensor_parallel_size": a.tp, "max_num_seqs": a.max_seqs,
          "enable_prefix_caching": False, "seed": 0, "max_model_len": max(a.prompts) + max(a.outputs) + 64,
          "max_num_batched_tokens": a.max_batched_tokens, "enforce_eager": False, "gpu_memory_utilization": a.mem,
          "compilation_config": {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": CAPTURE}}
    if a.async_scheduling != "default":
        kw["async_scheduling"] = a.async_scheduling == "on"
    t = time.perf_counter()
    llm = LLM(**kw)
    up = time.perf_counter() - t
    eng = llm.llm_engine
    sched = handle.scheduler(llm)
    cfg = eng.vllm_config.model_config.hf_config
    vocab = cfg.vocab_size
    steps: list[dict] = []
    open_: list = []
    orig_schedule, orig_update = sched.schedule, sched.update_from_output

    def label(so) -> tuple[int, int, int]:
        ns = so.num_scheduled_tokens
        kv = sum(int(sched.requests[r].num_computed_tokens) for r in ns if r in sched.requests)
        return sum(n for n in ns.values() if n > 1), sum(1 for n in ns.values() if n == 1), kv

    def schedule(*args, **kwargs):
        so = orig_schedule(*args, **kwargs)
        pre, dec, kv = label(so)
        rec = {"i": len(steps), "t0": time.perf_counter(), "prefill_tokens": pre, "decode_reqs": dec, "kv_tokens": kv,
               "tokens": int(so.total_num_scheduled_tokens)}
        steps.append(rec)
        open_.append((rec, torch.cuda.nvtx.range_start(f"s{rec['i']} p{pre} d{dec} kv{kv}")))
        return so

    def update_from_output(so, mro):
        if open_:
            rec, rid = open_.pop(0)
            torch.cuda.nvtx.range_end(rid)
            rec["t1"] = time.perf_counter()
        return orig_update(so, mro)

    sched.schedule, sched.update_from_output = schedule, update_from_output
    if a.tp == 1:
        # One range `x<i> ...` a model step, around the runner's launches (execute_model, then sample_tokens when
        # execute_model leaves the sampling to it): sequential with async scheduling too, so a kernel's step is the
        # range its launch falls in.
        runner = handle.model_runner(llm)
        ex_orig, st_orig = runner.execute_model, getattr(runner, "sample_tokens", None)
        xs: list = []

        def execute_model(*args, **kwargs):
            so = args[0] if args else kwargs["scheduler_output"]
            pre, dec, kv = label(so)
            xs.append(torch.cuda.nvtx.range_start(f"x{len(xs)} p{pre} d{dec} kv{kv}"))
            out = ex_orig(*args, **kwargs)
            if out is not None or st_orig is None:
                torch.cuda.nvtx.range_end(xs[-1])
                xs[-1] = None
            return out

        def sample_tokens(*args, **kwargs):
            out = st_orig(*args, **kwargs)
            if xs and xs[-1] is not None:
                torch.cuda.nvtx.range_end(xs[-1])
                xs[-1] = None
            return out

        runner.execute_model = execute_model
        if st_orig is not None:
            runner.sample_tokens = sample_tokens
    llm.generate([TokensPrompt(prompt_token_ids=[1000 + i for i in range(64)])],
                 SamplingParams(max_tokens=8, ignore_eos=True), use_tqdm=False)
    steps.clear()

    pending = sorted(reqs, key=lambda r: r["t"])
    k = 0
    first: dict[str, float] = {}
    done: dict[str, float] = {}
    arrive: dict[str, float] = {}
    torch.cuda.nvtx.range_push("trace")
    t0 = time.perf_counter()
    while k < len(pending) or eng.has_unfinished_requests():
        now = time.perf_counter() - t0
        while k < len(pending) and pending[k]["t"] <= now:
            r = pending[k]
            rr = random.Random(r["tokens_seed"])
            ids = [rr.randrange(1000, vocab - 1000) for _ in range(r["prompt"])]
            eng.add_request(r["id"], TokensPrompt(prompt_token_ids=ids),
                            SamplingParams(max_tokens=r["output"], ignore_eos=True, temperature=0.0))
            arrive[r["id"]] = t0 + r["t"]
            k += 1
        if eng.has_unfinished_requests():
            outs = eng.step()
            s1 = time.perf_counter()
            for o in outs:
                if o.outputs and len(o.outputs[0].token_ids) and o.request_id not in first:
                    first[o.request_id] = s1
                if o.finished:
                    done[o.request_id] = s1
        else:
            nxt = pending[k]["t"] + t0 if k < len(pending) else time.perf_counter()
            idle = torch.cuda.nvtx.range_start("idle")
            time.sleep(max(0.0, min(0.05, nxt - time.perf_counter())))
            torch.cuda.nvtx.range_end(idle)
    t_end = time.perf_counter()
    torch.cuda.nvtx.range_pop()

    ttft = sorted(first[r] - arrive[r] for r in first)
    out_tokens = {r["id"]: r["output"] for r in reqs}
    tpot = sorted((done[r] - first[r]) / (out_tokens[r] - 1) for r in done if r in first and out_tokens[r] > 1)
    res = {"format": "verity-serve-bandwidth/mix/v0", "args": {k_: v for k_, v in vars(a).items() if k_ != "out"},
           "vllm": vllm.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
           "engine_up_s": round(up, 1), "async_scheduling": eng.vllm_config.scheduler_config.async_scheduling,
           "kv_blocks": int(eng.vllm_config.cache_config.num_gpu_blocks), **per_gpu_bytes(cfg, a.tp),
           "trace": {"requests": len(reqs), "prompt_tokens": sum(r["prompt"] for r in reqs),
                     "output_tokens": sum(r["output"] for r in reqs), "wall_s": round(t_end - t0, 3)},
           "ttft_median_s": ttft[len(ttft) // 2] if ttft else None,
           "tpot_median_ms": 1e3 * tpot[len(tpot) // 2] if tpot else None,
           "t0": t0, "steps": [{**s, "t0": round(s["t0"] - t0, 6), "t1": round(s.get("t1", s["t0"]) - t0, 6)}
                               for s in steps]}
    (a.out / "mix.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k_: v for k_, v in res.items() if k_ != "steps"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
