"""Production throughput of one (model, configuration) on this pod (CONTRACT §16.7).

One process per (model, config): the batch-invariant flag, the compilation mode and the model
runner selection are process-lifetime state. Run ON THE POD::

    PYTHONPATH=. python -m verity_vllm_adapter.diagnostics.throughput --config C0 \
        --repo HuggingFaceTB/SmolLM2-135M --checkpoints manifests/checkpoints.json \
        --out /workspace/bench/SmolLM2-135M/C0

Configurations (which flags each one carries is written into ``result.json["config"]``):

* ``C0`` ordinary optimised vLLM: stock defaults (compiled + CUDA graphs), no batch-invariant flag.
* ``C1`` batch-invariant optimised: ``VLLM_BATCH_INVARIANT=1``, compiled + CUDA graphs.
* ``C2`` batch-invariant eager: ``VLLM_BATCH_INVARIANT=1``, ``enforce_eager=True``,
  ``VLLM_USE_V2_MODEL_RUNNER=1`` (the eager-production profile of the PASS rows).
* ``C2a`` (optional control) batch-invariant eager with the default model runner, to separate the
  cost of the V2 runner flag from the cost of eager execution.
* ``C1v2`` (optional control) = C1 + ``VLLM_USE_V2_MODEL_RUNNER=1``.

All configurations run the engine in-process (``VLLM_ENABLE_V1_MULTIPROCESSING=0``, the same
topology flag the adapter's production rows declare) so the scheduler's running-request count can be
read every step (achieved concurrency, §15.8). ``--vllm-bench`` additionally runs vLLM's own
``vllm bench throughput`` (stock multiprocessing topology) on the same shapes as a cross-check.

Workload: ``--num-prompts`` synthetic prompts of exactly ``input_len`` random token ids (seeded) with
``ignore_eos`` and ``max_tokens = output_len`` so every request generates exactly ``output_len``
tokens; greedy. Shapes ``128/128`` and ``512/64`` by default. All requests are added at t=0
(request rate inf) and the engine is stepped until all finish. A warm-up pass over the first shape
(compilation, CUDA-graph capture, allocator) is timed separately; then ``--reps`` timed passes per
shape. Per request: TTFT (first step that emitted its first token), end-to-end latency, both
relative to t=0 of the pass. Per pass: wall, total tokens/s (prompt + generated), generated
tokens/s, requests/s, steps, mean/max running requests per step (achieved concurrency), GPU
memory (torch allocated / reserved peaks, nvidia-smi used).

Kernel-path resolution: one profiled step after warm-up (``torch.profiler``, CUDA kernels)
lists the distinct kernel names actually launched, classified into matmul / rmsnorm / attention /
activation / rope families, so the C0 -> C1 change is attributed to the kernels the flag switched.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True, choices=["C0", "C1", "C2", "C2a", "C1v2"])
ap.add_argument("--repo", required=True)
ap.add_argument("--checkpoints", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--shapes", default="128:128,512:64")
ap.add_argument("--num-prompts", type=int, default=256)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--max-model-len", type=int, default=1024)
ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--profile-steps", type=int, default=3)
ap.add_argument("--vllm-bench", action="store_true", help="also run `vllm bench throughput` (stock multiprocessing) per shape")
args = ap.parse_args()

CONFIGS = {
    "C0": {"VLLM_BATCH_INVARIANT": None, "VLLM_USE_V2_MODEL_RUNNER": None, "enforce_eager": False,
           "label": "ordinary optimised vLLM (compiled + CUDA graphs, defaults)"},
    "C1": {"VLLM_BATCH_INVARIANT": "1", "VLLM_USE_V2_MODEL_RUNNER": None, "enforce_eager": False,
           "label": "batch-invariant optimised (VLLM_BATCH_INVARIANT=1, compiled + CUDA graphs)"},
    "C2": {"VLLM_BATCH_INVARIANT": "1", "VLLM_USE_V2_MODEL_RUNNER": "1", "enforce_eager": True,
           "label": "batch-invariant eager (VLLM_BATCH_INVARIANT=1, enforce_eager, VLLM_USE_V2_MODEL_RUNNER=1): the eager-production profile"},
    "C2a": {"VLLM_BATCH_INVARIANT": "1", "VLLM_USE_V2_MODEL_RUNNER": None, "enforce_eager": True,
            "label": "control: batch-invariant eager, default model runner"},
    "C1v2": {"VLLM_BATCH_INVARIANT": "1", "VLLM_USE_V2_MODEL_RUNNER": "1", "enforce_eager": False,
             "label": "control: batch-invariant optimised + VLLM_USE_V2_MODEL_RUNNER=1"},
}
cfg = CONFIGS[args.config]
ENV = {}
for k in ("VLLM_BATCH_INVARIANT", "VLLM_USE_V2_MODEL_RUNNER"):
    os.environ.pop(k, None)
    if cfg[k] is not None:
        os.environ[k] = cfg[k]
        ENV[k] = cfg[k]
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
ENV["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

from verity_vllm_adapter import profile as prof  # noqa: E402

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)


def nvidia_smi_used_mib() -> int | None:
    try:
        s = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        return int(s.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = (len(ys) - 1) * p
    lo, hi = int(np.floor(k)), int(np.ceil(k))
    return ys[lo] if lo == hi else ys[lo] + (ys[hi] - ys[lo]) * (k - lo)


def make_prompts(n: int, input_len: int, vocab: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    return rng.integers(100, vocab - 100, size=(n, input_len)).tolist()


def run_pass(llm: LLM, prompts: list[list[int]], output_len: int, tag: str, scheduler) -> dict:
    engine = llm.llm_engine
    sp = SamplingParams(temperature=0.0, top_p=1.0, top_k=0, max_tokens=output_len, ignore_eos=True, seed=None)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for i, ids in enumerate(prompts):
        engine.add_request(request_id=f"{tag}-{i}", prompt={"prompt_token_ids": ids}, params=sp)
    t_add = time.perf_counter() - t0
    first_tok: dict[str, float] = {}
    finish: dict[str, float] = {}
    gen_tokens: dict[str, int] = {}
    running_per_step: list[int] = []
    waiting_per_step: list[int] = []
    steps = 0
    smi_peak = 0
    while engine.has_unfinished_requests():
        outs = engine.step()
        now = time.perf_counter()
        steps += 1
        running_per_step.append(len(getattr(scheduler, "running", [])))
        waiting_per_step.append(len(getattr(scheduler, "waiting", [])))
        for o in outs:
            rid = o.request_id
            n = len(o.outputs[0].token_ids)
            if n > 0 and rid not in first_tok:
                first_tok[rid] = now - t0
            if o.finished:
                finish[rid] = now - t0
                gen_tokens[rid] = n
        if steps % 25 == 0:
            m = nvidia_smi_used_mib()
            if m:
                smi_peak = max(smi_peak, m)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    m = nvidia_smi_used_mib()
    if m:
        smi_peak = max(smi_peak, m)
    n_req = len(prompts)
    assert len(finish) == n_req, (len(finish), n_req)
    gen = sum(gen_tokens.values())
    prompt_tokens = sum(len(p) for p in prompts)
    e2e = list(finish.values())
    ttft = list(first_tok.values())
    tpot = [(finish[r] - first_tok[r]) / max(1, gen_tokens[r] - 1) for r in finish]
    return {
        "tag": tag,
        "requests": n_req,
        "input_len": len(prompts[0]),
        "output_len": output_len,
        "generated_tokens": gen,
        "prompt_tokens": prompt_tokens,
        "all_requests_generated_exactly_output_len": all(v == output_len for v in gen_tokens.values()),
        "wall_s": wall,
        "add_requests_s": t_add,
        "engine_steps": steps,
        "total_tokens_per_s": (gen + prompt_tokens) / wall,
        "generated_tokens_per_s": gen / wall,
        "requests_per_s": n_req / wall,
        "latency_e2e_s": {"p50": pct(e2e, 0.5), "p95": pct(e2e, 0.95), "p99": pct(e2e, 0.99), "mean": statistics.fmean(e2e), "max": max(e2e)},
        "ttft_s": {"p50": pct(ttft, 0.5), "p95": pct(ttft, 0.95), "mean": statistics.fmean(ttft)},
        "tpot_s": {"p50": pct(tpot, 0.5), "p95": pct(tpot, 0.95), "mean": statistics.fmean(tpot)},
        "concurrency": {
            "running_max": max(running_per_step),
            "running_mean": statistics.fmean(running_per_step),
            "running_mean_nonzero": statistics.fmean([x for x in running_per_step if x]) if any(running_per_step) else 0,
            "waiting_max": max(waiting_per_step),
            "steps_with_waiting": sum(1 for w in waiting_per_step if w),
        },
        "gpu_memory": {
            "torch_max_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
            "torch_max_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
            "nvidia_smi_used_peak_mib": smi_peak,
        },
    }


KERNEL_FAMILIES = [
    ("attention", re.compile(r"flash|fmha|attn|attention|paged", re.I)),
    ("matmul", re.compile(r"matmul_kernel_persistent|nvjet|xmma|cutlass|gemm|gemv|cublas|Cijk|sm\d+_.*gemm|ampere_.*gemm", re.I)),
    ("rmsnorm", re.compile(r"rms_norm|rmsnorm|fused_add_rms|triton_(red|per)_fused.*(rsqrt|mean|pow)", re.I)),
    ("activation", re.compile(r"act_and_mul|silu|triton_poi_fused.*(sigmoid|silu)", re.I)),
    ("rope", re.compile(r"rotary|rope|triton_poi_fused.*(cos|sin)", re.I)),
    ("kv_cache", re.compile(r"reshape_and_cache|kv_cache", re.I)),
    ("sampling", re.compile(r"argmax|softmax|topk|sample|gumbel|log_softmax", re.I)),
    ("embedding", re.compile(r"embedding|index_select|gather", re.I)),
    ("inductor_fused", re.compile(r"^triton_", re.I)),
    ("memcpy_memset", re.compile(r"memcpy|memset|Memcpy|Memset|copy", re.I)),
]


def classify(name: str) -> str:
    for fam, rx in KERNEL_FAMILIES:
        if rx.search(name):
            return fam
    return "other"


def profile_kernels(llm: LLM, prompts: list[list[int]], output_len: int, scheduler) -> dict:
    """Profile a short pass (few prompts) and list the CUDA kernels launched."""
    from torch.profiler import ProfilerActivity, profile

    engine = llm.llm_engine
    sp = SamplingParams(temperature=0.0, top_p=1.0, top_k=0, max_tokens=output_len, ignore_eos=True)
    for i, ids in enumerate(prompts):
        engine.add_request(request_id=f"prof-{i}", prompt={"prompt_token_ids": ids}, params=sp)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=False) as prof_:
        while engine.has_unfinished_requests():
            engine.step()
        torch.cuda.synchronize()
    counts: dict[str, int] = {}
    times: dict[str, float] = {}
    for e in prof_.events():
        if getattr(e, "device_type", None) is not None and str(e.device_type).endswith("CUDA") and e.name and not e.name.startswith(("cuda", "Runtime")):
            counts[e.name] = counts.get(e.name, 0) + 1
            times[e.name] = times.get(e.name, 0.0) + float(getattr(e, "device_time", 0.0) or getattr(e, "cuda_time", 0.0) or 0.0)
    kernels = sorted(({"name": k, "count": v, "device_time_us": times[k], "family": classify(k)} for k, v in counts.items()), key=lambda d: -d["device_time_us"])
    fam: dict[str, dict] = {}
    for k in kernels:
        f = fam.setdefault(k["family"], {"count": 0, "device_time_us": 0.0, "kernels": []})
        f["count"] += k["count"]
        f["device_time_us"] += k["device_time_us"]
        if len(f["kernels"]) < 12:
            f["kernels"].append(k["name"][:160])
    return {"prompts": len(prompts), "output_len": output_len, "distinct_kernels": len(kernels), "families": fam, "kernels": kernels[:80]}


def main() -> None:
    cps = prof.load_checkpoints(args.checkpoints)
    cp = cps[args.repo]
    local = cp["local_path"]
    kw = {
        "model": local,
        "tokenizer": local,
        "dtype": "bfloat16",
        "seed": 0,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": False,
    }
    if cfg["enforce_eager"]:
        kw["enforce_eager"] = True
    t0 = time.perf_counter()
    llm = LLM(**kw)
    build_s = time.perf_counter() - t0
    vc = llm.llm_engine.vllm_config
    scheduler = prof.get_scheduler(llm)
    vocab = int(getattr(vc.model_config.hf_config, "vocab_size", 32000))
    from vllm.ir.op import IrOp

    effective = {
        "compilation": prof.effective_compilation(llm),
        "cache": prof.cache_info(llm),
        "backends": prof.backends_info(llm),
        "ir_op_priority": {name: op.get_priority() for name, op in IrOp.registry.items()},
        "custom_ops": list(getattr(vc.compilation_config, "custom_ops", [])),
        "kv_cache_blocks_gpu": int(getattr(vc.cache_config, "num_gpu_blocks", 0) or 0),
        "block_size": int(vc.cache_config.block_size),
        "max_num_seqs": int(vc.scheduler_config.max_num_seqs),
        "max_num_batched_tokens": int(vc.scheduler_config.max_num_batched_tokens),
        "engine_build_s": build_s,
    }
    shapes = [tuple(int(x) for x in s.split(":")) for s in args.shapes.split(",")]
    result = {
        "schema": "verity-vllm/measure/throughput/v1",
        "config": {"id": args.config, "label": cfg["label"], "env": ENV, "enforce_eager": cfg["enforce_eager"],
                   "engine_kwargs": {k: v for k, v in kw.items() if k not in ("model", "tokenizer")}},
        "model": {"repo": args.repo, "revision": cp.get("revision"), "local_path": local, "weights_sha256": cp.get("safetensors_sha256")},
        "software": prof.software_info(None, None),
        "gpu": prof.gpu_info(),
        "effective": effective,
        "workload": {"num_prompts": args.num_prompts, "shapes": [{"input_len": a, "output_len": b} for a, b in shapes], "seed": args.seed, "greedy": True, "ignore_eos": True,
                     "request_rate": "inf (all requests added at t=0)"},
        "warmup": None,
        "passes": [],
        "kernels": None,
        "utc": prof.utc_stamp(),
    }
    # warm-up on the first shape (cold start: compile / graph capture / allocator)
    a, b = shapes[0]
    warm_prompts = make_prompts(args.num_prompts, a, vocab, args.seed + 999)
    result["warmup"] = run_pass(llm, warm_prompts, b, "warm", scheduler)
    print("WARMUP", json.dumps({k: result["warmup"][k] for k in ("wall_s", "total_tokens_per_s", "engine_steps")}), flush=True)
    # kernel path resolution (short profiled pass, 8 prompts, 4 output tokens)
    try:
        result["kernels"] = profile_kernels(llm, make_prompts(8, a, vocab, args.seed + 7), 4, scheduler)
        print("KERNELS", json.dumps({f: {"count": v["count"], "top": v["kernels"][:3]} for f, v in result["kernels"]["families"].items()}), flush=True)
    except Exception as e:  # noqa: BLE001
        result["kernels"] = {"error": repr(e)}
    for si, (a, b) in enumerate(shapes):
        prompts = make_prompts(args.num_prompts, a, vocab, args.seed + si)
        for rep in range(args.reps):
            llm.reset_prefix_cache()
            r = run_pass(llm, prompts, b, f"s{si}r{rep}", scheduler)
            r["rep"] = rep
            result["passes"].append(r)
            print("PASS", json.dumps({k: r[k] for k in ("tag", "wall_s", "total_tokens_per_s", "generated_tokens_per_s")}), json.dumps(r["concurrency"]), json.dumps(r["latency_e2e_s"]), flush=True)
            (OUT / "result.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    (OUT / "result.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    del llm
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    if args.vllm_bench:
        bench = []
        for si, (a, b) in enumerate(shapes):
            cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "throughput", "--model", local, "--tokenizer", local, "--dtype", "bfloat16",
                   "--dataset-name", "random", "--input-len", str(a), "--output-len", str(b), "--num-prompts", str(args.num_prompts),
                   "--max-model-len", str(args.max_model_len), "--gpu-memory-utilization", str(args.gpu_memory_utilization), "--seed", str(args.seed),
                   "--output-json", str(OUT / f"vllm_bench_{a}_{b}.json")]
            if cfg["enforce_eager"]:
                cmd.append("--enforce-eager")
            env = dict(os.environ)
            env.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)  # stock topology for the cross-check
            t1 = time.perf_counter()
            p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
            dt = time.perf_counter() - t1
            (OUT / f"vllm_bench_{a}_{b}.log").write_text(p.stdout[-20000:] + "\n--- stderr ---\n" + p.stderr[-20000:])
            js = None
            jp = OUT / f"vllm_bench_{a}_{b}.json"
            if jp.exists():
                js = json.loads(jp.read_text())
            bench.append({"input_len": a, "output_len": b, "returncode": p.returncode, "wall_incl_startup_s": dt, "result": js, "cmd": cmd})
            print("VLLM_BENCH", a, b, p.returncode, json.dumps(js)[:300] if js else p.stderr[-500:], flush=True)
        result["vllm_bench_throughput"] = bench
        (OUT / "result.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    print("DONE", args.repo, args.config, flush=True)


if __name__ == "__main__":
    main()
