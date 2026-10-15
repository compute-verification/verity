#!/usr/bin/env python3
"""Stock vLLM's GPU time per engine step on one GPU, split into tensor-core GEMMs and everything else, from torch.profiler's
kernel records: the saturation-budget question's inference side (CMP-8, Daniel's #7).

  python vllm_split.py --model unsloth/Llama-3.1-8B --revision REV --precision bf16|fp8 --mode eager|graphs|compiled \
      --out OUT.json [--prefill 2048,8192] [--batch 32] [--context 1024] [--decode-steps 24]
  python vllm_split.py ... --ncu-step       # no profiler: one decode step inside the NVTX range ``ncu_step``, for ncu

Each engine step's ``GPUModelRunner.execute_model`` and ``sample_tokens`` run inside a profiler range
``vstep|<kind>|<tokens>|<requests>`` (kind ``decode`` when every scheduled request takes one token, else ``prefill``); a
kernel belongs to the step whose range holds its launch (the CUDA runtime or driver call with the kernel's correlation id,
a graph launch included). Phases, each profiled on its own after a warm-up: ``prefill<P>``, one request of P random tokens
and one output token; ``decode``, BATCH requests of CONTEXT random tokens and 1 + DECODE_STEPS output tokens, whose steps
with BATCH one-token requests are the decode steps. Modes: ``eager`` (enforce_eager), ``graphs`` (no torch.compile, full CUDA
graphs for decode at the batch size: the harness's graph mode) and ``compiled`` (vLLM's default compilation and graphs).

Per phase: each step's kernel time by category (`CATEGORIES`, first match on the kernel's name), the GPU span (first kernel
start to last kernel end), the host wall time of its ranges, and the top kernels. GPU kernel time is the sum of kernel
durations, so with one stream it is busy time; span minus kernel time is the GPU waiting on the host.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

#: (category, pattern) in order; the first match on the lower-cased kernel name wins. ``gemm`` is every tensor-core matmul
#: of the linears and the logits (cuBLAS/cuBLASLt's nvjet and xmma kernels, CUTLASS's scaled_mm), never attention's.
CATEGORIES = (
    ("kv_write", r"reshape_and_cache|cache_kernel|concat_and_cache"),
    ("attention", r"flash|fmha|attn|attention|paged|batchdecode|batchprefill|batch_decode|batch_prefill|merge_state|"
                  r"unified_attention|kernel_unified|decode_kernel|prefill_kernel"),
    ("norm", r"rms_norm|rmsnorm|layer_norm|layernorm|_norm_kernel|red_fused.*(rsqrt|mean|pow)"),
    ("act", r"silu|act_and_mul|gelu|swiglu|poi_fused.*(silu|sigmoid)"),
    ("rope", r"rotary|_rope|rope_"),
    ("quant", r"_quant|quant_|quantize"),
    ("gemm", r"gemm|cutlass|nvjet|xmma|cublas|sm\d+_|ampere_|hopper_|blackwell_|matmul|_mm_|splitk|s16816|s1688|f8f6f4|"
             r"qmma|hmma|scaled_mm|wgmma"),
    ("sampling", r"argmax|topk|top_k|sort|softmax|sample|gumbel|multinomial|argreduce|reduce_kernel|cumsum|scan|penalt|"
                 r"logit|exponential|uniform|bernoulli|max_kernel"),
    ("embedding", r"index_select|indexselect|embedding|gather|index_kernel|scatter"),
    ("elementwise", r"elementwise|vectorized|unrolled|fill|copy|cat|where|triton_|tensor_kernel|kernel_"),
)
_COMPILED = [(c, re.compile(p)) for c, p in CATEGORIES]
NONLINEAR = ("norm", "act", "rope", "attention", "sampling")


def category(name: str) -> str:
    low = name.lower()
    return next((c for c, p in _COMPILED if p.search(low)), "other")


def install_ranges(state: dict) -> None:
    """Wrap the model runner's step methods (before the engine is built) in profiler ranges, and in ncu mode put one decode
    step in the NVTX range ``ncu_step``."""
    import importlib

    runners = []
    for module in ("vllm.v1.worker.gpu_model_runner", "vllm.v1.worker.gpu.model_runner"):
        try:
            runners.append(importlib.import_module(module).GPUModelRunner)
        except (ImportError, AttributeError):
            pass
    if not runners:
        raise SystemExit("no GPUModelRunner to wrap in this vLLM")
    for runner in runners:
        _wrap_runner(runner, state)


def _wrap_runner(runner, state: dict) -> None:
    import torch

    execute, sample = runner.execute_model, getattr(runner, "sample_tokens", None)
    state.setdefault("wrapped", {})[f"{runner.__module__}.{runner.__qualname__}"] = 0

    def close_nvtx() -> None:
        if state.get("nvtx_open"):
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            state["nvtx_open"] = False

    def execute_model(self, scheduler_output, *a, **kw):
        state["wrapped"][f"{runner.__module__}.{runner.__qualname__}"] += 1
        n = int(scheduler_output.total_num_scheduled_tokens)
        r = len(scheduler_output.num_scheduled_tokens)
        kind = "decode" if r and n == r else ("prefill" if n else "idle")
        state["label"] = f"{kind}|{n}|{r}"
        if state.get("nvtx_arm") and kind == "decode" and r == state["batch"]:
            state["nvtx_seen"] = state.get("nvtx_seen", 0) + 1
            if state["nvtx_seen"] == state["nvtx_arm"]:
                torch.cuda.synchronize()
                torch.cuda.nvtx.range_push("ncu_step")
                state["nvtx_open"], state["nvtx_arm"] = True, 0
        with torch.profiler.record_function(f"vstep|{state['label']}"):
            out = execute(self, scheduler_output, *a, **kw)
        if sample is None:
            close_nvtx()
        return out

    def sample_tokens(self, *a, **kw):
        with torch.profiler.record_function(f"vstep|{state.get('label', 'idle|0|0')}|sample"):
            out = sample(self, *a, **kw)
        close_nvtx()
        return out

    runner.execute_model = execute_model
    if sample is not None:
        runner.sample_tokens = sample_tokens


def parse_trace(path: Path) -> dict:
    """Kernels of a chrome trace grouped by the ``vstep`` range holding their launch: {step index: {...}}."""
    events = json.loads(path.read_text())["traceEvents"]
    ranges, launches, kernels = [], {}, []
    for e in events:
        if e.get("ph") != "X":
            continue
        cat, name = e.get("cat", ""), e.get("name", "")
        if cat == "user_annotation" and name.startswith("vstep|"):
            ranges.append((float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0)), name))
        elif cat in ("cuda_runtime", "cuda_driver"):
            corr = e.get("args", {}).get("correlation")
            if corr is not None:
                launches[corr] = float(e["ts"])
        elif cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            kernels.append((float(e["ts"]), float(e.get("dur", 0)), name if cat == "kernel" else cat,
                            e.get("args", {}).get("correlation")))
    ranges.sort()
    steps: dict = {}
    unassigned = {"n": 0, "us": 0.0}
    starts = [r[0] for r in ranges]
    import bisect
    for ts, dur, name, corr in kernels:
        t = launches.get(corr)
        i = bisect.bisect_right(starts, t) - 1 if t is not None else -1
        if i < 0 or t > ranges[i][1]:
            unassigned["n"] += 1
            unassigned["us"] += dur
            continue
        label = ranges[i][2].removesuffix("|sample").removeprefix("vstep|")
        key = (i if not ranges[i][2].endswith("|sample") else _step_of_sample(ranges, i))
        s = steps.setdefault(key, {"label": label, "kernels": defaultdict(lambda: [0, 0.0]), "first": ts, "last": ts + dur,
                                   "host_us": 0.0})
        k = s["kernels"][name]
        k[0] += 1
        k[1] += dur
        s["first"], s["last"] = min(s["first"], ts), max(s["last"], ts + dur)
    for i, (a, b, name) in enumerate(ranges):
        key = i if not name.endswith("|sample") else _step_of_sample(ranges, i)
        if key in steps:
            steps[key]["host_us"] += b - a
    return {"steps": [steps[k] for k in sorted(steps)], "unassigned": unassigned}


def _step_of_sample(ranges: list, i: int) -> int:
    """A sample range belongs to the latest execute range before it."""
    j = i - 1
    while j >= 0 and ranges[j][2].endswith("|sample"):
        j -= 1
    return max(j, 0)


def summarize_step(s: dict) -> dict:
    by_cat: dict = defaultdict(float)
    for name, (_, us) in s["kernels"].items():
        by_cat[category(name)] += us
    total = sum(by_cat.values())
    return {"label": s["label"], "kernel_ms": total / 1e3, "span_ms": (s["last"] - s["first"]) / 1e3,
            "host_ms": s["host_us"] / 1e3, "by_category_ms": {c: round(v / 1e3, 4) for c, v in sorted(by_cat.items())}}


def summarize_phase(parsed: dict, batch: int | None) -> dict:
    steps = parsed["steps"]
    rows = [summarize_step(s) for s in steps]
    out: dict = {"steps": len(rows), "unassigned": parsed["unassigned"], "step_labels": [r["label"] for r in rows][:12]}
    want = [i for i, r in enumerate(rows) if (batch is None and r["label"].startswith("prefill"))
            or (batch is not None and r["label"] == f"decode|{batch}|{batch}")]
    if batch is not None and len(want) > 2:
        want = want[1:]
    sel = [rows[i] for i in want]
    if not sel:
        out["error"] = "no step of the wanted kind"
        return out
    cats = sorted({c for r in sel for c in r["by_category_ms"]})
    med = lambda xs: statistics.median(xs)  # noqa: E731
    per = {c: med([r["by_category_ms"].get(c, 0.0) for r in sel]) for c in cats}
    kernel = med([r["kernel_ms"] for r in sel])
    gemm = per.get("gemm", 0.0)
    out.update(selected=len(sel), kernel_ms=kernel, span_ms=med([r["span_ms"] for r in sel]),
               host_ms=med([r["host_ms"] for r in sel]), by_category_ms=per,
               gemm_share=gemm / kernel if kernel else None,
               nonlinear_ms=sum(per.get(c, 0.0) for c in NONLINEAR),
               non_gemm_ms=kernel - gemm)
    agg: dict = defaultdict(lambda: [0, 0.0])
    for i in want:
        for name, (n, us) in steps[i]["kernels"].items():
            agg[name][0] += n
            agg[name][1] += us
    top = sorted(agg.items(), key=lambda kv: -kv[1][1])[:40]
    out["top_kernels"] = [{"kernel": k[:160], "category": category(k), "per_step_ms": round(v[1] / 1e3 / len(sel), 4),
                           "per_step_launches": round(v[0] / len(sel), 2)} for k, v in top]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision")
    ap.add_argument("--precision", choices=("bf16", "fp8"), required=True)
    ap.add_argument("--mode", choices=("eager", "graphs", "compiled"), required=True)
    ap.add_argument("--prefill", default="2048,8192")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--decode-steps", type=int, default=24)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--ncu-step", action="store_true", help="no profiler: decode step 4 inside NVTX range ncu_step")
    ap.add_argument("--scratch", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    prefills = [int(x) for x in args.prefill.split(",") if x]
    max_len = max(prefills + [args.context + args.decode_steps + 8]) + 128
    state: dict = {"batch": args.batch}
    install_ranges(state)

    import torch
    from vllm import LLM, SamplingParams

    engine = dict(model=args.model, revision=args.revision, gpu_memory_utilization=args.gpu_mem, max_model_len=max_len,
                  max_num_batched_tokens=max_len, max_num_seqs=max(64, args.batch), enable_prefix_caching=False, seed=0,
                  enforce_eager=args.mode == "eager")
    if args.precision == "fp8":
        engine["quantization"] = "fp8"
    if args.mode == "graphs":
        engine["compilation_config"] = {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [args.batch]}
    t0 = time.perf_counter()
    llm = LLM(**engine)
    startup = time.perf_counter() - t0
    rng = random.Random(0)
    toks = lambda n: {"prompt_token_ids": [rng.randrange(1000, 120000) for _ in range(n)]}  # noqa: E731
    greedy = lambda n: SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)  # noqa: E731
    doc: dict = {"schema": "verity-pouw/saturation-vllm-split/v1", "run": os.environ.get("RESEARCH_RUN_ID"),
                 "model": args.model, "revision": args.revision, "precision": args.precision, "mode": args.mode,
                 "batch": args.batch, "context": args.context, "decode_steps": args.decode_steps, "prefills": prefills,
                 "engine": {k: v for k, v in engine.items() if k != "model"}, "startup_s": round(startup, 1),
                 "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                 "vllm": __import__("vllm").__version__, "categories": dict(CATEGORIES), "phases": {}}

    llm.generate([toks(64) for _ in range(args.batch)], greedy(8), use_tqdm=False)
    for p in prefills:
        llm.generate([toks(p)], greedy(1), use_tqdm=False)
    llm.generate([toks(args.context) for _ in range(args.batch)], greedy(4), use_tqdm=False)
    torch.cuda.synchronize()

    if args.ncu_step:
        state["nvtx_arm"] = 4
        llm.generate([toks(args.context) for _ in range(args.batch)], greedy(1 + 8), use_tqdm=False)
        torch.cuda.synchronize()
        doc["ncu_step"] = {"armed_step": 4, "closed": not state.get("nvtx_open", False),
                           "opened": state.get("nvtx_arm") == 0, "wrapped_runner_calls": state["wrapped"]}
        args.out.write_text(json.dumps(doc, indent=1))
        print(json.dumps(doc["ncu_step"]))
        return 0

    from torch.profiler import ProfilerActivity, profile
    scratch = args.scratch or Path(tempfile.mkdtemp())
    phases = [(f"prefill{p}", [toks(p)], greedy(1), None) for p in prefills]
    phases.append(("decode", [toks(args.context) for _ in range(args.batch)], greedy(1 + args.decode_steps), args.batch))
    for name, prompts, sp, batch in phases:
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            t = time.perf_counter()
            llm.generate(prompts, sp, use_tqdm=False)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t
        trace = scratch / f"{args.precision}-{args.mode}-{name}.json"
        prof.export_chrome_trace(str(trace))
        summary = summarize_phase(parse_trace(trace), batch)
        summary["generate_wall_s"] = round(wall, 4)
        doc["phases"][name] = summary
        trace.unlink()
        print(json.dumps({name: {k: summary.get(k) for k in ("selected", "kernel_ms", "span_ms", "gemm_share")}}), flush=True)
    doc["wrapped_runner_calls"] = state["wrapped"]
    args.out.write_text(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
