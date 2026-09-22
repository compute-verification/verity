#!/usr/bin/env python3
"""[REPLAY, additive] Trusted-replay baseline + instrumentation-ladder rungs (GUIDANCE §32.9A/.10).

Reruns TA1's B0 audited workload (SmolLM2-135M @ 93efa2f0, 1 request, 32 prompt
tokens, 8 greedy tokens) on the PRODUCTION vLLM path under one of several execution
configs, times repeated passes, and performs the trusted-replay consistency check
against the committed original A100 record (token ids + chosen-token logprobs from
``results/TA1-typed-b1-20260907/B0``, re-emitted by ``replay_b0_workload.py``).

Execution configs (``--exec``), the §32.10 ladder's uninstrumented rungs:

  stock-pure           stock default vLLM: no Verity env flags (BI=0, default model
                       runner, engine-core multiprocessing default), compiled +
                       CUDA graphs, ``LLM.generate`` driver. The "stock-vLLM
                       denominator" row.
  stock-inproc         stock kernels/compilation (BI=0, graphs on) but the declared
                       process flags (V2 runner, in-process engine core, flashinfer
                       sampler off) and the add_request/step driver every recorded
                       pass uses. Isolates process-topology + driver cost.
  bi-graphs            + batch-invariant kernels (VLLM_BATCH_INVARIANT=1), still
                       compiled + CUDA graphs. Isolates the BI-kernel cost, and
                       tests whether BI without eager reproduces the record.
  eager-stock-kernels  enforce_eager with stock kernels (BI=0). Isolates the
                       eager/graph cost with the kernel family unchanged.
  bi-eager             VLLM_BATCH_INVARIANT=1 + enforce_eager = the B0 semantic
                       profile's execution config, NO recorder. This is trusted
                       replay configuration (i).

Every config runs: engine build, warm-up on block-disjoint throwaway prompts
(production.py's construction), then ``--reps`` timed passes (prefix cache reset
before each), two determinism passes, the consistency check (timed), and optionally
one profiled pass (torch.profiler, summary JSON only).

Environment is set before any torch/vllm import; the effective values are recorded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

EXECS: dict[str, dict] = {
    "stock-pure": {
        "env": {"VLLM_BATCH_INVARIANT": "0"},
        "env_clear": ["VLLM_USE_V2_MODEL_RUNNER", "VLLM_ENABLE_V1_MULTIPROCESSING", "VLLM_USE_FLASHINFER_SAMPLER"],
        "enforce_eager": False,
        "driver": "generate",
    },
    "stock-inproc": {
        "env": {"VLLM_BATCH_INVARIANT": "0", "VLLM_USE_V2_MODEL_RUNNER": "1",
                "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_USE_FLASHINFER_SAMPLER": "0"},
        "env_clear": [],
        "enforce_eager": False,
        "driver": "steps",
    },
    "bi-graphs": {
        "env": {"VLLM_BATCH_INVARIANT": "1", "VLLM_USE_V2_MODEL_RUNNER": "1",
                "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_USE_FLASHINFER_SAMPLER": "0"},
        "env_clear": [],
        "enforce_eager": False,
        "driver": "steps",
    },
    "eager-stock-kernels": {
        "env": {"VLLM_BATCH_INVARIANT": "0", "VLLM_USE_V2_MODEL_RUNNER": "1",
                "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_USE_FLASHINFER_SAMPLER": "0"},
        "env_clear": [],
        "enforce_eager": True,
        "driver": "steps",
    },
    "bi-eager": {
        "env": {"VLLM_BATCH_INVARIANT": "1", "VLLM_USE_V2_MODEL_RUNNER": "1",
                "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_USE_FLASHINFER_SAMPLER": "0"},
        "env_clear": [],
        "enforce_eager": True,
        "driver": "steps",
    },
    # Stock kernels + the production stack's own trace-replay feature (upstream vLLM
    # d9105ea80: LLM(enable_trace_replay=True) + SamplingParams(trace_decode_token_ids));
    # NO Verity instrumentation. The strongest consistency check stock trusted replay can
    # perform: per-position chosen-token logprobs along the recorded trajectory.
    "stock-forced": {
        "env": {"VLLM_BATCH_INVARIANT": "0", "VLLM_USE_V2_MODEL_RUNNER": "1",
                "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_USE_FLASHINFER_SAMPLER": "0"},
        "env_clear": [],
        "enforce_eager": False,
        "driver": "steps",
        "forced": True,
    },
}


def apply_exec_env(name: str) -> dict[str, str | None]:
    cfg = EXECS[name]
    for k in cfg["env_clear"]:
        os.environ.pop(k, None)
    for k, v in cfg["env"].items():
        os.environ[k] = v
    os.environ.setdefault("HF_HOME", "/workspace/hf")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    keys = ["VLLM_BATCH_INVARIANT", "VLLM_USE_V2_MODEL_RUNNER",
            "VLLM_ENABLE_V1_MULTIPROCESSING", "VLLM_USE_FLASHINFER_SAMPLER"]
    return {k: os.environ.get(k) for k in keys}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec", dest="exec_name", required=True, choices=sorted(EXECS))
    ap.add_argument("--workload", required=True)
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--determinism-reps", type=int, default=2)
    ap.add_argument("--profile", action="store_true", help="one extra profiled pass (torch.profiler); summary JSON only")
    args = ap.parse_args()

    effective_env = apply_exec_env(args.exec_name)
    cfg = EXECS[args.exec_name]

    # env is final; now the heavy imports
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch  # noqa: E402
    from vllm import LLM, SamplingParams  # noqa: E402
    from verity_vllm_adapter import profile as prof  # noqa: E402  (setdefaults are no-ops here)
    from verity_vllm_adapter.driver import run_requests  # noqa: E402
    from verity_vllm_adapter.production import build_warmup, greedy_params  # noqa: E402
    from verity_vllm_sampler.workload import load_workload  # noqa: E402

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import replay_b0_workload as ref  # the committed original-record targets

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cp = prof.checkpoint_for("B0", args.checkpoints)
    prof.verify_local_weights(cp)
    wl = load_workload(args.workload)
    requests = list(wl.requests)
    assert [r.request_id for r in requests] == [ref.RECORDED_REQUEST_ID]

    forced = bool(cfg.get("forced"))
    kw = prof.engine_kwargs(cp, replay=False, enforce_eager=cfg["enforce_eager"])
    if forced:
        kw["enable_trace_replay"] = True
        kw["enable_prefix_caching"] = False  # same restriction as replay.py: forced positions must be computed
    t0 = time.perf_counter()
    llm = LLM(**kw)
    build_s = time.perf_counter() - t0

    logprobs = 1

    def params_for(r):
        if forced:
            return SamplingParams(temperature=0.0, max_tokens=len(ref.RECORDED_OUTPUT_TOKEN_IDS),
                                  logprobs=logprobs,
                                  trace_decode_token_ids=list(ref.RECORDED_OUTPUT_TOKEN_IDS))
        return greedy_params(r.max_tokens, logprobs)

    def one_pass_generate():
        prompts = [{"prompt_token_ids": list(r.prompt_token_ids)} for r in requests]
        params = [params_for(r) for r in requests]
        t = time.perf_counter()
        outs = llm.generate(prompts, params, use_tqdm=False)
        wall = time.perf_counter() - t
        gen = sum(len(o.outputs[0].token_ids) for o in outs)
        return outs, {"wall_s": wall, "generated_tokens": gen,
                      "prompt_tokens": sum(len(r.prompt_token_ids) for r in requests)}

    def one_pass_steps():
        outs_by_id, timing = run_requests(llm, requests, params_for)
        return [outs_by_id[r.request_id] for r in requests], timing

    one_pass = one_pass_generate if cfg["driver"] == "generate" else one_pass_steps

    def extract(outs):
        toks, lps = [], []
        for o in outs:
            seq = o.outputs[0]
            ids = list(seq.token_ids)
            toks.append(ids)
            row = []
            for i, tid in enumerate(ids):
                lp = None
                if seq.logprobs is not None and i < len(seq.logprobs):
                    entry = seq.logprobs[i].get(tid)
                    lp = float(entry.logprob) if entry is not None else None
                row.append(lp)
            lps.append(row)
        return toks, lps

    def reset_cache() -> bool:
        try:
            return bool(llm.reset_prefix_cache())
        except Exception as e:  # noqa: BLE001
            print(f"reset_prefix_cache failed: {e!r}", flush=True)
            return False

    # warm-up (cold start / compile / graph capture kept out of the timed passes)
    block_size = 16
    warm, warm_info = build_warmup(requests, 49152, block_size)
    if cfg["driver"] == "generate":
        wprompts = [{"prompt_token_ids": list(w.prompt_token_ids)} for w in warm]
        wparams = [greedy_params(w.max_tokens, logprobs) for w in warm]
        tw = time.perf_counter()
        llm.generate(wprompts, wparams, use_tqdm=False)
        warm_info["timing"] = {"wall_s": time.perf_counter() - tw}
    else:
        _, wt = run_requests(llm, warm, lambda r: greedy_params(r.max_tokens, logprobs))
        warm_info["timing"] = wt

    # timed passes
    passes = []
    resets = []
    first_outs = None
    for i in range(args.reps):
        resets.append(reset_cache())
        outs, timing = one_pass()
        toks, lps = extract(outs)
        if first_outs is None:
            first_outs = (toks, lps)
        passes.append({"timing": timing, "tokens": toks, "chosen_logprobs": lps})

    # run-to-run determinism (same config, same engine)
    det = []
    for i in range(args.determinism_reps):
        resets.append(reset_cache())
        outs, _ = one_pass()
        toks, lps = extract(outs)
        det.append({"tokens_equal_first_pass": toks == first_outs[0],
                    "logprobs_equal_first_pass": lps == first_outs[1]})

    # trusted-replay consistency check against the committed original record (timed)
    rec_tokens = [list(ref.RECORDED_OUTPUT_TOKEN_IDS)]
    rec_lps = [list(ref.RECORDED_CHOSEN_LOGPROBS)]
    tc0 = time.perf_counter()
    toks, lps = first_outs
    tokens_match = toks == rec_tokens
    first_diverging = None
    if not tokens_match:
        for k, (a, b) in enumerate(zip(toks[0], rec_tokens[0])):
            if a != b:
                first_diverging = k
                break
    lp_deltas = [abs(a - b) for a, b in zip(lps[0], rec_lps[0]) if a is not None]
    logprobs_match = lps == rec_lps
    compare_s = time.perf_counter() - tc0

    result = {
        "exec": args.exec_name,
        "driver": cfg["driver"],
        "effective_env": effective_env,
        "engine_kwargs": {k: v for k, v in kw.items() if k not in ("model", "tokenizer")},
        "engine_build_s": build_s,
        "warmup": warm_info,
        "prefix_cache_resets": resets,
        "passes": passes,
        "pass_wall_s": [p["timing"]["wall_s"] for p in passes],
        "pass_wall_s_median": sorted(p["timing"]["wall_s"] for p in passes)[len(passes) // 2],
        "run_to_run_determinism": det,
        "consistency_check": {
            "reference": "TA1 ta1eager A100 record (results/TA1-typed-b1-20260907/B0)",
            "recorded_output_token_ids": rec_tokens[0],
            "replayed_output_token_ids": toks[0],
            "tokens_bitexact": tokens_match,
            "first_diverging_output_index": first_diverging,
            "recorded_chosen_logprobs": rec_lps[0],
            "replayed_chosen_logprobs": lps[0],
            "chosen_logprobs_bitexact": logprobs_match,
            "max_abs_chosen_logprob_delta": max(lp_deltas) if lp_deltas else None,
            "comparison_wall_s": compare_s,
        },
    }
    try:
        result["effective_compilation"] = prof.effective_compilation(llm)
        result["cache"] = prof.cache_info(llm)
        result["backends"] = prof.backends_info(llm)
    except Exception as e:  # noqa: BLE001
        result["effective_compilation_error"] = repr(e)
    try:
        result["gpu"] = prof.gpu_info()
    except Exception as e:  # noqa: BLE001
        result["gpu_error"] = repr(e)

    if args.profile:
        from torch.profiler import ProfilerActivity, profile

        reset_cache()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof_ctx:
            one_pass()
        torch.cuda.synchronize()
        rows = prof_ctx.key_averages()
        kernels = []
        total_dev_us = 0.0
        launch_count = 0
        memcpy_us = 0.0
        memcpy_count = 0
        host_launch_us = 0.0
        host_launch_count = 0
        for r in rows:
            dev = float(getattr(r, "self_device_time_total", 0.0) or 0.0)
            name = str(r.key)
            if "Memcpy" in name or "Memset" in name:
                memcpy_us += dev
                memcpy_count += int(r.count)
                continue
            if dev > 0:
                total_dev_us += dev
                launch_count += int(r.count)
                kernels.append({"name": name[:120], "count": int(r.count), "self_device_us": dev})
            if name in ("cudaLaunchKernel", "cudaGraphLaunch", "cuLaunchKernel"):
                host_launch_us += float(r.self_cpu_time_total)
                host_launch_count += int(r.count)
        kernels.sort(key=lambda x: -x["self_device_us"])
        result["profile"] = {
            "note": "one extra pass, excluded from the timed reps; with CUDA graphs kernel events come from graph replays",
            "total_self_device_us": total_dev_us,
            "device_kernel_executions": launch_count,
            "memcpy_memset_us": memcpy_us,
            "memcpy_memset_count": memcpy_count,
            "host_launch_api_us": host_launch_us,
            "host_launch_api_count": host_launch_count,
            "top_kernels": kernels[:25],
        }

    (out / f"trusted-{args.exec_name}.json").write_text(json.dumps(result, indent=1) + "\n")
    ok = "OK" if tokens_match else "TOKENS DIFFER"
    print(f"[{args.exec_name}] median pass {result['pass_wall_s_median']:.4f}s; "
          f"tokens {ok}; logprobs bitexact={logprobs_match}; "
          f"max|dlogprob|={result['consistency_check']['max_abs_chosen_logprob_delta']}")


if __name__ == "__main__":
    main()
