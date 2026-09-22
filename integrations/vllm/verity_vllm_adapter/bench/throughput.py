"""Production throughput under one configuration (CONTRACT.md §16.7, §12). Runs ON THE POD.

One process per model x configuration: the configurations differ in environment flags that vLLM
reads at import / engine construction, so they cannot share a process. The harness itself is
identical for every configuration: an in-process engine (``VLLM_ENABLE_V1_MULTIPROCESSING=0``, so
the step loop can observe the scheduler; process topology only), the same fixed synthetic
workload (seeded random-token prompts, ``ignore_eos`` so every request generates exactly
``output_len`` tokens, greedy), the prefix cache reset before every timed pass so each pass is a
fresh prefill (recorded: ``cached_tokens`` must be 0), warm-up on its own pass and reported
separately, then ``--reps`` steady-state passes per shape.

Configurations (``CONFIGS``): the flags each carries are recorded verbatim in every row, and the
EFFECTIVE compilation / CUDA-graph mode, model runner, attention backend and FlashAttention
version come from the engine's config object and its log lines, never from the requested flags
(§15.4). Kernel paths are measured with ``torch.profiler`` on a short extra pass (``--profile``)
and grouped by family (gemm / rmsnorm / attention / ...), so "what the batch-invariant flag
switched" is read off the kernels that actually ran.

Row schema ``verity-vllm/bench/throughput/v1`` (one JSON object per line in ``rows.jsonl``, one
row per pass = model x config x shape x rep, plus the warm-up pass with ``phase = "warmup"``)::

    schema, worker, tag, utc,
    device {name, compute_capability, driver, pod, total_memory_gb},
    software {vllm, vllm_commit, torch, cuda, triton, flashinfer},
    model {repo, revision, local_path, dtype, num_layers, hidden_size, num_kv_heads, head_dim},
    config {id, label, env_flags {..}, harness_env {..}, engine_kwargs {..},
            effective {mode, cudagraph_mode, enforce_eager, model_runner, custom_ops,
                       attention_backend, attention_impl, flash_attn_version, log_lines [..]}},
    scheduler {max_num_seqs, max_num_batched_tokens, enable_prefix_caching, enable_chunked_prefill,
               block_size, num_gpu_blocks, kv_capacity_tokens},
    shape {input_len, output_len, requests},
    rep (0 = warm-up), phase ("warmup" | "steady"),
    wall_s, add_requests_s, engine_steps, prompt_tokens, generated_tokens,
    prompt_tokens_per_s, generated_tokens_per_s, total_tokens_per_s, requests_per_s,
    latency_s {e2e {p50, p95, mean, max}, ttft {p50, p95, mean, max}, tpot_ms {p50, p95, mean, max}},
    concurrency {cap, peak_achieved, mean_achieved, mean_tokens_per_step, max_tokens_per_step,
                 steps, preemptions},
    cached_tokens (sum of num_cached_tokens over the pass; must be 0),
    gpu_memory {torch_max_allocated_gb, torch_reserved_gb, nvidia_smi_used_mib}

Latencies are those of an offline batch: every request is submitted at t=0, so ``e2e`` includes
queueing behind the other 255 requests; ``ttft`` = first output token, ``tpot_ms`` = per-token
decode time after the first token. ``concurrency.peak_achieved`` / ``mean_achieved`` are the
number of requests the scheduler actually placed in a step (peak / mean over steps), not the cap.

Usage (pod)::

    /workspace/venv/bin/python -m verity_vllm_adapter.bench.throughput --config C0 \
        --repo Qwen/Qwen2.5-1.5B --model-path /workspace/hf/hub/models--Qwen--Qwen2.5-1.5B/snapshots/<rev> \
        --shapes 128x128,512x64 --num-requests 256 --reps 3 --profile --out /workspace/results/bench/<model>/C0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA = "verity-vllm/bench/throughput/v1"

# Environment the harness needs for every configuration (not a profile flag): the engine core
# runs in this process so the step loop can read the scheduler and torch.cuda memory counters.
HARNESS_ENV = {
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    # Required on this pod for EVERY configuration, C0 included: with the stock default the engine
    # warm-up JIT-compiles FlashInfer's top-k/top-p sampler and dies with `FileNotFoundError: 'ninja'`
    # (no ninja; nvcc 12.8 vs torch cu130). Inert for greedy requests (the V1/V2 samplers never call
    # it for temperature 0); the adapter declares the same flag for the same reason (profile.py).
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "HF_HOME": "/workspace/hf",
    "HF_HUB_OFFLINE": "1",
}

# The adapter's declared production profile (verity_vllm_adapter.profile.DECLARED_ENV) minus the
# two flags every configuration carries anyway (multiprocessing off, FlashInfer sampler off).
_ADAPTER_FLAGS = {
    "VLLM_BATCH_INVARIANT": "1",
    "VLLM_USE_V2_MODEL_RUNNER": "1",
}

CONFIGS: dict[str, dict[str, Any]] = {
    "C0": {
        "label": "ordinary optimised vLLM: stock defaults (compiled + CUDA graphs, default model runner, no batch-invariant flag)",
        "env": {},
        "enforce_eager": False,
    },
    "C1": {
        "label": "batch-invariant optimised: VLLM_BATCH_INVARIANT=1, compiled + CUDA graphs, default model runner",
        "env": {"VLLM_BATCH_INVARIANT": "1"},
        "enforce_eager": False,
    },
    "C1v2": {
        # In the pinned wheel Model Runner V2 is already the default for dense Qwen2/Llama
        # (config/vllm.py::use_v2_model_runner returns True unless an unsupported feature is on), so
        # this differs from C1 only in carrying the adapter's declared no-op flag; run once as a control.
        "label": "batch-invariant optimised + the adapter's declared VLLM_USE_V2_MODEL_RUNNER=1 (control: V2 is the wheel's default, see effective.model_runner of C0/C1): VLLM_BATCH_INVARIANT=1, VLLM_USE_V2_MODEL_RUNNER=1, compiled + CUDA graphs",
        "env": dict(_ADAPTER_FLAGS),
        "enforce_eager": False,
    },
    "C2": {
        "label": "batch-invariant eager (the adapter's eager production profile): VLLM_BATCH_INVARIANT=1, VLLM_USE_V2_MODEL_RUNNER=1, enforce_eager=True (no compilation, no CUDA graphs)",
        "env": dict(_ADAPTER_FLAGS),
        "enforce_eager": True,
    },
}

WARMUP_SEED = 0x5EED_7A9B
WORKLOAD_SEED = 20260907
WARMUP_REQUESTS = 32

# Kernel-name families for the profiler summary (ordered regexes on the CUDA kernel name; the
# first match wins, so the specific names (FlashAttention's cutlass kernel, vLLM's `vllm::` custom
# ops, the batch-invariant Triton kernels, inductor's `triton_*` codegen) come before the generic
# cuBLAS / aten patterns).
KERNEL_FAMILIES: list[tuple[str, str]] = [
    ("attention.flash_attn", r"flash::|FlashAttn|flash_fwd|Flash_fwd|fmha|flash_attn|combine_attn_seqk"),
    ("attention.flashinfer", r"flashinfer|BatchPrefill|BatchDecode|PrefillWithKV|DecodeWithKV"),
    ("attention.triton", r"kernel_unified_attention|kernel_paged_attention|triton_attn"),
    ("kv_cache.write", r"reshape_and_cache|kv_cache_update|concat_and_cache"),
    ("gemm.triton_batch_invariant", r"matmul_kernel_persistent|matmul_kernel_descriptor_persistent"),
    ("bmm.triton_batch_invariant", r"^bmm_kernel"),
    ("rmsnorm.inductor_generated", r"^triton_.*rms_norm"),
    ("activation.inductor_generated", r"^triton_.*(silu|gelu)"),
    ("inductor.generated", r"^triton_"),
    ("rmsnorm.cuda_custom_op", r"vllm::.*rms_norm"),
    ("activation.cuda_custom_op", r"vllm::.*(silu_and_mul|act_and_mul|gelu)"),
    ("rope.cuda_custom_op", r"vllm::.*rotary"),
    ("rmsnorm.triton_batch_invariant", r"^_rms_norm_kernel|rms_norm_kernel|rms_norm_residual"),
    ("softmax.triton_batch_invariant", r"^_log_softmax_kernel|^_softmax_kernel|^softmax_kernel"),
    ("mean.triton_batch_invariant", r"^mean_kernel"),
    ("gemm.cublas", r"nvjet|xmma|cutlass|gemm|gemv|splitK|cublas|Kernel<|ampere_|sm80_|sm90_"),
    ("softmax.aten", r"softmax"),
    ("sampler", r"gumbel|sampl|argmax|topk|top_k|top_p|rejected|draft_tokens|penalt"),
    ("reduce.aten", r"reduce_kernel"),
    ("elementwise.aten", r"elementwise_kernel|vectorized_elementwise|unrolled_elementwise|index_elementwise|CatArray|gather_kernel|scatter|copy_kernel|Fill"),
    ("memcpy", r"Memcpy|Memset|memcpy|memset"),
]


def classify_kernel(name: str) -> str:
    for fam, rx in KERNEL_FAMILIES:
        if re.search(rx, name):
            return fam
    return "other"


def parse_memory_lines(lines: list[str]) -> dict[str, float | None]:
    """Weights / KV-cache / compile figures from the engine's own log lines."""
    out: dict[str, float | None] = {"weights_gib": None, "kv_cache_gib": None, "torch_compile_s": None, "init_engine_s": None,
                                    "cudagraph_pool_gib": None}
    for l in lines:
        m = re.search(r"Model loading took ([\d.]+) GiB", l)
        if m:
            out["weights_gib"] = float(m.group(1))
        m = re.search(r"Available KV cache memory: ([\d.]+) GiB", l)
        if m:
            out["kv_cache_gib"] = float(m.group(1))
        # stock: "torch.compile took X s in total"; batch-invariant: "torch.compile and initial
        # profiling/warmup run together took X s in total"; both also appear as "(compilation: X s)"
        m = re.search(r"torch.compile (?:and .*?)?took ([\d.]+) s in total", l)
        if m:
            out["torch_compile_s"] = float(m.group(1))
        m = re.search(r"init engine .* took ([\d.]+) s(?: \(compilation: ([\d.]+) s\))?", l)
        if m:
            out["init_engine_s"] = float(m.group(1))
            if m.group(2) and out["torch_compile_s"] is None:
                out["torch_compile_s"] = float(m.group(2))
        m = re.search(r"CUDA graph pool memory: ([\d.]+) GiB \(actual\)", l)
        if m:
            out["cudagraph_pool_gib"] = float(m.group(1))
    return out


class LogCapture(logging.Handler):
    """Keep the vLLM log lines that state the effective execution mode."""

    KEYS = ("backend", "FlashAttention", "cudagraph", "CUDA graph", "compil", "Compil", "batch invar", "Batch invar",
            "BATCH_INVARIANT", "model runner", "Model Runner", "V2", "KV cache", "kv cache", "num_gpu_blocks",
            "Maximum concurrency", "Available KV", "torch.compile", "Dynamo", "eager", "Loading weights took",
            "Model loading took", "graph capturing", "Graph capturing", "cascade", "Cascade", "split", "Using ")

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return
        if any(k in msg for k in self.KEYS):
            self.lines.append(f"{record.name.split('.')[-1]}: {msg}"[:400])


def pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def stats(xs: list[float]) -> dict[str, float | None]:
    if not xs:
        return {"p50": None, "p95": None, "mean": None, "max": None}
    return {"p50": pct(xs, 0.5), "p95": pct(xs, 0.95), "mean": statistics.fmean(xs), "max": max(xs)}


def nvidia_smi_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        return int(out[0]) if out else None
    except Exception:  # noqa: BLE001
        return None


def make_prompts(n: int, length: int, vocab: int, seed: int, block_size: int) -> list[list[int]]:
    """Distinct random-token prompts sharing no aligned ``block_size`` block (prefix cache inert)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    prompts: list[list[int]] = []
    seen_blocks: set[tuple[int, ...]] = set()
    while len(prompts) < n:
        ids = rng.integers(3, vocab - 1, size=length).tolist()
        blocks = {tuple(ids[i : i + block_size]) for i in range(0, length - block_size + 1, block_size)}
        if blocks & seen_blocks:
            continue
        seen_blocks |= blocks
        prompts.append(ids)
    return prompts


class StepObserver:
    """Wrap ``scheduler.schedule`` to record, per engine step, how many requests / tokens the
    scheduler actually placed (achieved concurrency, §15.8) and preemptions."""

    def __init__(self, scheduler: Any) -> None:
        self.scheduler = scheduler
        self.orig = scheduler.schedule
        self.num_reqs: list[int] = []
        self.num_tokens: list[int] = []
        self.preempted = 0
        obs = self

        def schedule(*a: Any, **kw: Any) -> Any:
            out = obs.orig(*a, **kw)
            try:
                obs.num_reqs.append(len(out.num_scheduled_tokens))
                obs.num_tokens.append(int(out.total_num_scheduled_tokens))
                obs.preempted += len(out.preempted_req_ids or [])
            except Exception:  # noqa: BLE001
                pass
            return out

        scheduler.schedule = schedule

    def uninstall(self) -> None:
        self.scheduler.schedule = self.orig

    def summary(self) -> dict[str, Any]:
        steps = [n for n in self.num_reqs if n > 0]
        return {
            "peak_achieved": max(steps, default=0),
            "mean_achieved": statistics.fmean(steps) if steps else 0.0,
            "mean_tokens_per_step": statistics.fmean(self.num_tokens) if self.num_tokens else 0.0,
            "max_tokens_per_step": max(self.num_tokens, default=0),
            "steps": len(self.num_reqs),
            "preemptions": self.preempted,
        }


def run_pass(llm: Any, prompts: list[list[int]], out_len: int, scheduler: Any, sampling_params_cls: Any,
             tag: str) -> dict[str, Any]:
    engine = llm.llm_engine
    obs = StepObserver(scheduler)
    sp = sampling_params_cls(temperature=0.0, max_tokens=out_len, ignore_eos=True)
    t_add: dict[str, float] = {}
    t_first: dict[str, float] = {}
    t_done: dict[str, float] = {}
    n_out: dict[str, int] = {}
    cached: dict[str, int] = {}
    t0 = time.perf_counter()
    for i, ids in enumerate(prompts):
        rid = f"{tag}-{i}"
        t_add[rid] = time.perf_counter()
        engine.add_request(request_id=rid, prompt={"prompt_token_ids": ids}, params=sp)
    t_added = time.perf_counter()
    steps = 0
    while engine.has_unfinished_requests():
        outs = engine.step()
        now = time.perf_counter()
        steps += 1
        for o in outs:
            rid = o.request_id
            if rid not in t_first and o.outputs and len(o.outputs[0].token_ids) > 0:
                t_first[rid] = now
            if o.finished:
                t_done[rid] = now
                n_out[rid] = len(o.outputs[0].token_ids)
                cached[rid] = int(getattr(o, "num_cached_tokens", 0) or 0)
    t1 = time.perf_counter()
    obs.uninstall()
    missing = [r for r in t_add if r not in t_done]
    if missing:
        raise RuntimeError(f"{len(missing)} requests did not finish")
    wall = t1 - t0
    e2e = [t_done[r] - t_add[r] for r in t_add]
    ttft = [t_first[r] - t_add[r] for r in t_add if r in t_first]
    tpot = [(t_done[r] - t_first[r]) / (n_out[r] - 1) * 1000.0 for r in t_add if r in t_first and n_out[r] > 1]
    gen = sum(n_out.values())
    prompt_toks = sum(len(p) for p in prompts)
    short = [r for r, n in n_out.items() if n != out_len]
    return {
        "wall_s": wall,
        "add_requests_s": t_added - t0,
        "engine_steps": steps,
        "prompt_tokens": prompt_toks,
        "generated_tokens": gen,
        "prompt_tokens_per_s": prompt_toks / wall,
        "generated_tokens_per_s": gen / wall,
        "total_tokens_per_s": (prompt_toks + gen) / wall,
        "requests_per_s": len(prompts) / wall,
        "latency_s": {"e2e": stats(e2e), "ttft": stats(ttft), "tpot_ms": stats(tpot)},
        "concurrency": obs.summary(),
        "cached_tokens": sum(cached.values()),
        "requests_not_generating_output_len": len(short),
    }


def profile_kernels(llm: Any, prompts: list[list[int]], out_len: int, scheduler: Any, sampling_params_cls: Any) -> dict[str, Any]:
    """Short pass under torch.profiler (CUDA activity): kernel names grouped by family."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    llm.reset_prefix_cache()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=False) as p:
        run_pass(llm, prompts, out_len, scheduler, sampling_params_cls, "prof")
        torch.cuda.synchronize()
    kernels: dict[str, dict[str, Any]] = {}
    for ev in p.key_averages():
        # device-side events only (kernels, memcpy/memset); host-side aten/python frames excluded
        if not str(getattr(ev, "device_type", "")).endswith("CUDA"):
            continue
        cuda_us = float(getattr(ev, "self_device_time_total", 0.0))
        if cuda_us <= 0:
            continue
        name = ev.key
        kernels[name] = {"count": int(ev.count), "device_time_us": cuda_us, "family": classify_kernel(name)}
    fam: dict[str, dict[str, Any]] = {}
    for name, k in kernels.items():
        f = fam.setdefault(k["family"], {"count": 0, "device_time_us": 0.0, "kernels": []})
        f["count"] += k["count"]
        f["device_time_us"] += k["device_time_us"]
        f["kernels"].append(name)
    for f in fam.values():
        f["kernels"] = sorted(set(f["kernels"]), key=lambda n: -kernels[n]["device_time_us"])[:12]
    total = sum(f["device_time_us"] for f in fam.values()) or 1.0
    for f in fam.values():
        f["share_of_device_time"] = f["device_time_us"] / total
    return {
        "profiled_pass": {"requests": len(prompts), "input_len": len(prompts[0]), "output_len": out_len},
        "families": dict(sorted(fam.items(), key=lambda kv: -kv[1]["device_time_us"])),
        "top_kernels": sorted(({"name": n, **k} for n, k in kernels.items()), key=lambda k: -k["device_time_us"])[:40],
        "distinct_kernels": len(kernels),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--repo", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--revision", default=None, help="checkpoint revision (defaults to the snapshot dir name)")
    ap.add_argument("--shapes", default="128x128,512x64", help="comma list of <input_len>x<output_len>")
    ap.add_argument("--num-requests", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--worker", default="W4d")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--profile", action="store_true", help="also run a short torch.profiler pass and write kernels.json")
    ap.add_argument("--vllm-commit", default="d9105ea80")
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    for k, v in HARNESS_ENV.items():
        os.environ.setdefault(k, v)
    for k, v in cfg["env"].items():
        os.environ[k] = v
    # never inherit a declared flag the configuration does not carry
    for k in ("VLLM_BATCH_INVARIANT", "VLLM_USE_V2_MODEL_RUNNER"):
        if k not in cfg["env"]:
            os.environ.pop(k, None)

    import torch  # noqa: E402
    from vllm import LLM, SamplingParams  # noqa: E402

    from verity_vllm_adapter import profile as prof  # noqa: E402  (helpers only; apply_env is NOT called)

    # after `import vllm`: its dictConfig would otherwise replace the handler list
    cap = LogCapture()
    logging.getLogger("vllm").addHandler(cap)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    revision = args.revision or Path(args.model_path).name
    kw: dict[str, Any] = {
        "model": args.model_path,
        "tokenizer": args.model_path,
        "dtype": "bfloat16",
        "seed": 0,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": False,
    }
    if cfg["enforce_eager"]:
        kw["enforce_eager"] = True
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    llm = LLM(**kw)
    build_s = time.perf_counter() - t0
    scheduler = prof.get_scheduler(llm)
    vc = llm.llm_engine.vllm_config
    hf = vc.model_config.hf_config
    tok = llm.get_tokenizer()
    vocab = int(len(tok))
    cache = prof.cache_info(llm)
    num_gpu_blocks = int(getattr(vc.cache_config, "num_gpu_blocks", 0) or 0)
    fa_version = None
    for line in cap.lines:
        m = re.search(r"FlashAttention version (\d)", line)
        if m:
            fa_version = int(m.group(1))
    backend_line = next((l for l in cap.lines if "backend" in l.lower() and "Using" in l), None)
    backends = prof.backends_info(llm)
    effective = {
        **prof.effective_compilation(llm),
        "attention_backend": backend_line,
        "attention_impl": backends.get("attention"),
        "flash_attn_version": fa_version,
        "sampler": backends.get("sampler"),
        "log_lines": list(cap.lines),
    }
    sw = prof.software_info(None, args.vllm_commit)
    base = {
        "schema": SCHEMA,
        "worker": args.worker,
        "tag": args.tag,
        "device": prof.gpu_info(),
        "software": {"vllm": sw["vllm"]["version"], "vllm_commit": args.vllm_commit, **{k: sw["python"][k] for k in ("torch", "cuda", "triton", "flashinfer")}},
        "model": {
            "repo": args.repo,
            "revision": revision,
            "local_path": args.model_path,
            "dtype": "bfloat16",
            "num_layers": int(getattr(hf, "num_hidden_layers", 0)),
            "hidden_size": int(getattr(hf, "hidden_size", 0)),
            "num_kv_heads": int(getattr(hf, "num_key_value_heads", 0)),
            "head_dim": int(getattr(hf, "hidden_size", 0)) // max(int(getattr(hf, "num_attention_heads", 1)), 1),
        },
        "config": {
            "id": args.config,
            "label": cfg["label"],
            "env_flags": dict(cfg["env"]),
            "harness_env": {k: os.environ.get(k) for k in HARNESS_ENV},
            "engine_kwargs": {k: v for k, v in kw.items() if k not in ("model", "tokenizer")},
            "effective": {k: v for k, v in effective.items() if k != "log_lines"},
        },
        "scheduler": {**cache, "num_gpu_blocks": num_gpu_blocks, "kv_capacity_tokens": num_gpu_blocks * cache["block_size"]},
        "engine_memory": parse_memory_lines(cap.lines),
    }
    shapes = [tuple(int(x) for x in s.split("x")) for s in args.shapes.split(",") if s]
    block_size = cache["block_size"]

    # warm-up: its own pass on prompts from another seed (Triton JIT / inductor autotune / first
    # cudagraph replays); timed and reported as phase "warmup", never mixed with steady state.
    warm_prompts = make_prompts(WARMUP_REQUESTS, shapes[0][0], vocab, WARMUP_SEED, block_size)
    t_w = run_pass(llm, warm_prompts, shapes[0][1], scheduler, SamplingParams, "warm")
    warm_row = {**base, "shape": {"input_len": shapes[0][0], "output_len": shapes[0][1], "requests": WARMUP_REQUESTS},
                "rep": 0, "phase": "warmup", "utc": prof.utc_stamp(), **t_w,
                "gpu_memory": {"torch_max_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
                               "torch_reserved_gb": torch.cuda.memory_reserved() / 2**30,
                               "nvidia_smi_used_mib": nvidia_smi_used_mib()}}
    rows = [warm_row]
    for in_len, out_len in shapes:
        prompts = make_prompts(args.num_requests, in_len, vocab, WORKLOAD_SEED + in_len, block_size)
        for rep in range(1, args.reps + 1):
            reset_ok = bool(llm.reset_prefix_cache())
            t = run_pass(llm, prompts, out_len, scheduler, SamplingParams, f"s{in_len}x{out_len}r{rep}")
            row = {**base, "shape": {"input_len": in_len, "output_len": out_len, "requests": args.num_requests},
                   "rep": rep, "phase": "steady", "utc": prof.utc_stamp(), "prefix_cache_reset_ok": reset_ok, **t,
                   "gpu_memory": {"torch_max_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
                                  "torch_reserved_gb": torch.cuda.memory_reserved() / 2**30,
                                  "nvidia_smi_used_mib": nvidia_smi_used_mib()}}
            rows.append(row)
            print(f"[{args.config}] {args.repo} {in_len}x{out_len} rep{rep}: wall {t['wall_s']:.2f}s gen {t['generated_tokens_per_s']:.0f} tok/s "
                  f"prompt {t['prompt_tokens_per_s']:.0f} tok/s conc peak {t['concurrency']['peak_achieved']} mean {t['concurrency']['mean_achieved']:.1f} "
                  f"e2e p50 {t['latency_s']['e2e']['p50']:.2f}s p95 {t['latency_s']['e2e']['p95']:.2f}s cached {t['cached_tokens']}", flush=True)
    with (out / "rows.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    kern = None
    if args.profile:
        prof_prompts = make_prompts(16, 128, vocab, WORKLOAD_SEED + 999, block_size)
        kern = profile_kernels(llm, prof_prompts, 8, scheduler, SamplingParams)
        (out / "kernels.json").write_text(json.dumps(kern, indent=1) + "\n")
    engine = {
        **base,
        "engine_build_s": build_s,
        "warmup": {"requests": WARMUP_REQUESTS, "shape": list(shapes[0]), "wall_s": t_w["wall_s"], "seed": WARMUP_SEED},
        "workload_seed": WORKLOAD_SEED,
        "argv": sys.argv,
        "effective_log_lines": effective["log_lines"],
        "kernel_families": {k: {kk: vv for kk, vv in v.items() if kk != "kernels"} | {"top": v["kernels"][:5]} for k, v in (kern or {}).get("families", {}).items()},
        "utc": prof.utc_stamp(),
    }
    (out / "engine.json").write_text(json.dumps(engine, indent=1) + "\n")
    print(f"DONE {args.config} {args.repo}: build {build_s:.1f}s warmup {t_w['wall_s']:.2f}s rows {len(rows)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
