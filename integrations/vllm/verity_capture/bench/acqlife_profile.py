"""R10 acq-life — attribution profile of the FROZEN R9 complete commitment path (native_host fast path + FA2 M1 stream + GPU-resident
256-B chunk tree) against the control arm, with torch.profiler (CPU + CUDA, stacks).  DIAGNOSTIC ONLY — never a headline timing; the
profiler and the markers inflate every number (labelled "under profiler").

Arms (same engine, same request, interleaved order control/complete/hooks_only, N reps each, all under one profiler session per run):
  control      Verity off (no hooks)
  complete     `NativeHostCommitter(fast_copy=True, gpu_tree=True)` + `HiddenM1Source(matAll)` — the +56.9 % path — subclassed HERE only to add
               `torch.profiler.record_function` markers (hook body, step begin/end, worker GPU-tree commit, FA2 wrapper); the frozen module is untouched
  hooks_only   R9's ablation (hooks fire and return) for the hook-dispatch floor on this pod

Attribution per engine step (chrome trace, main thread = the thread carrying `verity.engine_step`):
  serving thread   hook body (bookkeeping + runtime calls), memcpy/memset issue (CUPTI `cudaMemcpyAsync`/`cudaMemsetAsync` runtime events inside hooks),
                   step begin/end, FA2 wrapper glue (wrapper − tapped op), CUDA runtime stalls (`cudaStreamSynchronize`, `cudaEventSynchronize`,
                   `cudaMemcpy`, `cudaDeviceSynchronize`), allocator (`cudaMalloc`/`cudaFree`), launch API time (`cudaLaunchKernel` count and mean µs —
                   contention shows as a longer mean under the same kernel count), Python GC (gc.callbacks, separate clock), remainder = unattributed
  worker thread    `verity.gpu_tree` CPU time, its launch count
  GPU              kernel time by class (model / tree / copy-memset / FA2), busy union, tree∥model overlap

    PYTHONPATH=. /workspace/venv-cu129/bin/python -m verity_capture.bench.acqlife_profile --out /workspace/cp/acqlife/prof \
        --hidden-so /workspace/cp/fa2/build/matAll/verity_fa2_matAll.so --reps 2
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from verity_vllm_adapter import profile as prof  # noqa: F401

prof.apply_env()

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile, record_function  # noqa: E402

from verity_capture import vllm_adapter as va  # noqa: E402
from verity_capture.bench.commit_delta import baseline_state, assert_pristine, make_committer  # noqa: E402
from verity_capture.experimental.commit_integ.native_host import NativeHostCommitter  # noqa: E402

SCHEMA = "verity/acq-life/profile/v1"


class _GC:
    def __init__(self) -> None:
        self.events: list[tuple[float, str, int, int]] = []
        self._t0: float | None = None

    def __call__(self, phase: str, info: dict) -> None:
        now = time.perf_counter()
        if phase == "start":
            self._t0 = now
        else:
            self.events.append((now, "stop", info.get("generation", -1), int((now - (self._t0 or now)) * 1e6)))

    def summary(self, t0: float, t1: float) -> dict:
        ev = [e for e in self.events if t0 <= e[0] <= t1]
        return {"collections": len(ev), "total_us": sum(e[3] for e in ev), "by_gen": {g: sum(1 for e in ev if e[2] == g) for g in (0, 1, 2)}}


class ProfiledCommitter(NativeHostCommitter):
    """Markers only; every code path is the parent's."""

    def _make_hook(self, name: str):
        inner = super()._make_hook(name)

        def hook(module, args, output):
            with record_function("verity.hook"):
                return inner(module, args, output)

        return hook

    def _make_pre_hook(self, name: str):
        inner = super()._make_pre_hook(name)

        def pre(module, args):
            with record_function("verity.prehook"):
                return inner(module, args)

        return pre

    def _begin_step(self) -> None:
        with record_function("verity.step_begin"):
            super()._begin_step()

    def _end_step(self) -> None:
        with record_function("verity.step_end"):
            super()._end_step()

    def _commit_step_gpu(self, job) -> None:
        with record_function("verity.gpu_tree"):
            super()._commit_step_gpu(job)


def profiled_hidden_source(committer, so_path: str):
    from verity_capture.experimental.commit_integ.hidden_source import HiddenM1Source

    class ProfiledM1(HiddenM1Source):
        def _wrapper(self):
            inner = super()._wrapper()

            def varlen_fwd(*a, **kw):
                with record_function("verity.fa2wrap"):
                    return inner(*a, **kw)

            return varlen_fwd

    return ProfiledM1(committer, so_path)


def run_marked(engine: va.Engine, requests: list[dict], sampling: dict) -> dict:
    """commit_delta.run_workload with a `verity.engine_step` marker around every engine.step() (both arms; identical loop)."""
    eng = engine.llm.llm_engine
    pending = sorted(range(len(requests)), key=lambda i: (int(requests[i].get("arrive_step") or 0), i))
    tokens: dict[str, list[int]] = {}
    finished: set[str] = set()
    step_t: list[float] = []
    it = 0
    t0 = time.perf_counter()
    while pending or eng.has_unfinished_requests():
        while pending and int(requests[pending[0]].get("arrive_step") or 0) <= it:
            r = requests[pending.pop(0)]
            eng.add_request(request_id=r["request_id"], prompt={"prompt_token_ids": list(r["prompt_token_ids"])}, params=va.sampling_params(r, sampling))
        if not eng.has_unfinished_requests():
            r = requests[pending.pop(0)]
            eng.add_request(request_id=r["request_id"], prompt={"prompt_token_ids": list(r["prompt_token_ids"])}, params=va.sampling_params(r, sampling))
            continue
        with record_function(f"verity.engine_step#{it}"):
            outs = eng.step()
        step_t.append(time.perf_counter())
        for o in outs:
            for c in o.outputs:
                tokens[o.request_id] = [int(t) for t in c.token_ids]
            if o.finished:
                finished.add(o.request_id)
        it += 1
    t1 = time.perf_counter()
    walls = [step_t[0] - t0] + [b - a for a, b in zip(step_t, step_t[1:])]
    return {"wall_s": t1 - t0, "iterations": it, "step_wall_s": walls, "tokens": tokens, "t0": t0, "t1": t1}


# ---------------------------------------------------------------------------------------------------------------- trace analysis
RUNTIME_STALL = ("cudaStreamSynchronize", "cudaEventSynchronize", "cudaDeviceSynchronize", "cudaMemcpy", "cudaStreamQuery", "cudaEventQuery")
RUNTIME_ALLOC = ("cudaMalloc", "cudaFree", "cuMemAlloc", "cuMemFree", "cudaMallocAsync", "cudaFreeAsync", "cuMemCreate", "cuMemMap")
RUNTIME_LAUNCH = ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cuLaunchKernelEx")
RUNTIME_COPY = ("cudaMemcpyAsync", "cudaMemsetAsync", "cudaMemcpy2DAsync")
RUNTIME_EVENT = ("cudaEventRecord", "cudaStreamWaitEvent", "cudaEventCreate", "cudaEventDestroy", "cudaEventCreateWithFlags")


def kernel_class(name: str) -> str:
    n = name.lower()
    if "flash_fwd" in n or "flash" in n and "attn" in n:
        return "fa2"
    if any(s in n for s in ("chunk_leaves", "tree_level", "sha256", "sha_", "leaf", "merkle", "hidden_gpu")):
        return "tree"
    if n.startswith("memcpy") or n.startswith("memset") or "memcpy" in n:
        return "copy"
    return "model"


def _union(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    intervals.sort()
    tot, cs, ce = 0.0, intervals[0][0], intervals[0][1]
    for s, e in intervals[1:]:
        if s > ce:
            tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return tot + (ce - cs)


def _overlap(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    a.sort()
    b.sort()
    i = j = 0
    tot = 0.0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if e > s:
            tot += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def analyze(trace_path: Path, arm: str) -> dict:
    doc = json.loads(trace_path.read_text())
    ev = [e for e in doc["traceEvents"] if e.get("ph") == "X"]
    # record_function markers appear twice: `user_annotation` (CPU thread, tid = OS thread) and `gpu_user_annotation` (one per CUDA stream that
    # ran work inside the range, tid = stream id).  CPU accounting uses the former; GPU windows use the latter (kernels of step k are the
    # kernels inside step k's gpu_user_annotation ranges -- they lag the CPU window because the CPU runs ahead of the GPU).
    is_step = lambda e: str(e.get("name", "")).startswith("verity.engine_step#")  # noqa: E731
    steps = sorted([e for e in ev if is_step(e) and e.get("cat") == "user_annotation"], key=lambda e: e["ts"])
    if not steps:
        return {"error": "no engine_step markers"}
    gpu_steps = defaultdict(list)
    for e in ev:
        if is_step(e) and e.get("cat") == "gpu_user_annotation":
            gpu_steps[e["name"]].append((e["ts"], e["ts"] + e["dur"]))
    main_tid = steps[0]["tid"]
    per_step = []
    kernels = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    runtime = [e for e in ev if e.get("cat") == "cuda_runtime"]
    user = [e for e in ev if e.get("cat") == "user_annotation"]
    cpu_ops = [e for e in ev if e.get("cat") == "cpu_op"]
    worker_tids = {e["tid"] for e in user if e["name"] == "verity.gpu_tree"}
    for k, st in enumerate(steps):
        s0, s1 = st["ts"], st["ts"] + st["dur"]
        w = lambda e: s0 <= e["ts"] < s1  # noqa: E731
        g = gpu_steps.get(st["name"], [])
        g0, g1 = (min(a for a, _b in g), max(b for _a, b in g)) if g else (s0, s1)
        wg = lambda e: g0 <= e["ts"] < g1  # noqa: E731
        u = [e for e in user if w(e)]
        main_u = [e for e in u if e["tid"] == main_tid]
        hook = [e for e in main_u if e["name"] in ("verity.hook", "verity.prehook")]
        hook_sum = sum(e["dur"] for e in hook)
        hook_iv = [(e["ts"], e["ts"] + e["dur"]) for e in hook]
        rt_main = [e for e in runtime if w(e) and e["tid"] == main_tid]

        def in_hook(e: dict) -> bool:
            t = e["ts"]
            return any(a <= t < b for a, b in hook_iv)

        rt_in_hook = [e for e in rt_main if in_hook(e)]
        copy_issue = sum(e["dur"] for e in rt_in_hook if e["name"] in RUNTIME_COPY)
        fa2w = [e for e in main_u if e["name"] == "verity.fa2wrap"]
        fa2_ops = [e for e in cpu_ops if w(e) and e["tid"] == main_tid and "varlen_fwd" in e["name"]]
        fa2_glue = (sum(e["dur"] for e in fa2w) - sum(e["dur"] for e in fa2_ops)) if fa2w else 0.0
        fa2_op_cpu = sum(e["dur"] for e in fa2_ops)
        sb = sum(e["dur"] for e in main_u if e["name"] in ("verity.step_begin", "verity.step_end"))
        launches = [e for e in rt_main if e["name"] in RUNTIME_LAUNCH]
        stalls = [e for e in rt_main if e["name"] in RUNTIME_STALL]
        allocs = [e for e in rt_main if e["name"] in RUNTIME_ALLOC]
        evs = [e for e in rt_main if e["name"] in RUNTIME_EVENT]
        other_rt = [e for e in rt_main if e["name"] not in RUNTIME_LAUNCH + RUNTIME_STALL + RUNTIME_ALLOC + RUNTIME_EVENT + RUNTIME_COPY]
        # worker thread(s)
        wk = [e for e in u if e["tid"] in worker_tids and e["name"] == "verity.gpu_tree"]
        wk_launch = [e for e in runtime if w(e) and e["tid"] in worker_tids and e["name"] in RUNTIME_LAUNCH]
        wk_stall = [e for e in runtime if w(e) and e["tid"] in worker_tids and e["name"] in RUNTIME_STALL]
        # GPU (windowed by the GPU-side annotation of this step)
        ks = [e for e in kernels if wg(e)]
        by = defaultdict(float)
        iv = defaultdict(list)
        nk = defaultdict(int)
        for e in ks:
            c = kernel_class(e["name"])
            by[c] += e["dur"]
            nk[c] += 1
            iv[c].append((e["ts"], e["ts"] + e["dur"]))
        busy = _union([(e["ts"], e["ts"] + e["dur"]) for e in ks])
        # main-thread cpu_ops inside hooks (what "bookkeeping" is made of) and outside (the model's own Python/aten dispatch)
        ops_main = [e for e in cpu_ops if w(e) and e["tid"] == main_tid]
        ops_in_hook = [e for e in ops_main if in_hook(e)]
        top_in_hook = defaultdict(float)
        for e in ops_in_hook:
            top_in_hook[e["name"]] += e["dur"]
        per_step.append({
            "step": k, "wall_us": st["dur"], "gpu_window_us": g1 - g0, "hooks": len(hook), "hook_body_us": hook_sum, "copy_issue_us_in_hooks": copy_issue, "copy_calls_in_hooks": sum(1 for e in rt_in_hook if e["name"] in RUNTIME_COPY),
            "cpu_ops_in_hooks_us": sum(e["dur"] for e in ops_in_hook), "cpu_ops_in_hooks_top": dict(sorted(top_in_hook.items(), key=lambda kv: -kv[1])[:8]),
            "gpu_kernel_counts": dict(nk), "gpu_model_span_us": (max(b for a, b in iv["model"]) - min(a for a, b in iv["model"])) if iv.get("model") else 0.0,
            "bookkeeping_us": hook_sum - sum(e["dur"] for e in rt_in_hook), "runtime_in_hooks_us": sum(e["dur"] for e in rt_in_hook),
            "fa2_wrapper_glue_us": fa2_glue, "fa2_wrapper_calls": len(fa2w), "fa2_op_cpu_us": fa2_op_cpu, "fa2_op_calls": len(fa2_ops), "step_begin_end_us": sb,
            "cpu_ops_main": len(ops_main), "cpu_ops_main_us": sum(e["dur"] for e in ops_main if not in_hook(e) and e["name"] not in ("verity.hook",)),
            "launch_calls": len(launches), "launch_us": sum(e["dur"] for e in launches), "launch_mean_us": statistics.fmean(e["dur"] for e in launches) if launches else 0.0,
            "launch_p90_us": sorted(e["dur"] for e in launches)[int(0.9 * len(launches))] if launches else 0.0,
            "stall_calls": len(stalls), "stall_us": sum(e["dur"] for e in stalls), "stall_names": dict(sorted(((n, sum(1 for e in stalls if e["name"] == n)) for n in {e["name"] for e in stalls}))),
            "alloc_calls": len(allocs), "alloc_us": sum(e["dur"] for e in allocs), "event_api_calls": len(evs), "event_api_us": sum(e["dur"] for e in evs),
            "other_runtime_us": sum(e["dur"] for e in other_rt), "other_runtime_names": dict(sorted(((n, sum(1 for e in other_rt if e["name"] == n)) for n in {e["name"] for e in other_rt}))),
            "worker_gpu_tree_cpu_us": sum(e["dur"] for e in wk), "worker_launches": len(wk_launch), "worker_stall_us": sum(e["dur"] for e in wk_stall), "worker_stall_calls": len(wk_stall),
            "gpu_kernel_us": dict(by), "gpu_kernels": len(ks), "gpu_busy_us": busy, "gpu_tree_model_overlap_us": _overlap(iv.get("tree", []), iv.get("model", []) + iv.get("fa2", [])),
            "gpu_copy_model_overlap_us": _overlap(iv.get("copy", []), iv.get("model", []) + iv.get("fa2", [])),
        })
    return {"arm": arm, "main_tid": main_tid, "worker_tids": sorted(worker_tids), "steps": per_step, "trace": str(trace_path), "events": len(ev)}


def mean_over(steps: list[dict], sel: list[int]) -> dict:
    rows = [s for s in steps if s["step"] in sel]
    out = {}
    if not rows:
        return out
    for k in rows[0]:
        v = [r[k] for r in rows]
        if isinstance(v[0], (int, float)):
            out[k] = statistics.fmean(v)
        elif isinstance(v[0], dict) and k == "gpu_kernel_us":
            out[k] = {c: statistics.fmean(r[k].get(c, 0.0) for r in rows) for c in {c for r in rows for c in r[k]}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="verity_capture/profiles/workload_cov_b0_b1_c256.json")
    ap.add_argument("--checkpoints", default="manifests/checkpoints.json")
    ap.add_argument("--case", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden-so", default=None)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--engine-arg", action="append", default=None)
    ap.add_argument("--no-stack", action="store_true")
    ap.add_argument("--arms", default="control,complete,hooks_only")
    ap.add_argument("--analyze-only", action="store_true", help="re-run analyze() over the traces already in --out (no engine)")
    ap.add_argument("--switch-interval", type=float, default=None, help="diagnostic: sys.setswitchinterval (GIL hand-off latency hypothesis)")
    args = ap.parse_args()
    if args.switch_interval is not None:
        sys.setswitchinterval(args.switch_interval)
        print(f"[prof] sys.setswitchinterval({args.switch_interval})", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.analyze_only:
        doc = json.loads((out / "profile_results.json").read_text())
        for row in doc["results"]:
            row["analysis"] = an = analyze(Path(row["analysis"]["trace"]), row["arm"])
            dec = mean_over(an["steps"], list(range(3, 15)))
            print(f"[prof] rep {row['rep']} {row['arm']:10s}: decode-mean step {dec.get('wall_us', 0):.0f} µs gpu-window {dec.get('gpu_window_us', 0):.0f}; hooks {dec.get('hooks', 0):.0f} body {dec.get('hook_body_us', 0):.0f} µs "
                  f"(cpu_ops-in-hooks {dec.get('cpu_ops_in_hooks_us', 0):.0f}, runtime-in-hooks {dec.get('runtime_in_hooks_us', 0):.0f}, copies {dec.get('copy_calls_in_hooks', 0):.0f}/{dec.get('copy_issue_us_in_hooks', 0):.0f} µs); "
                  f"fa2 glue {dec.get('fa2_wrapper_glue_us', 0):.0f}; launches {dec.get('launch_calls', 0):.0f}×{dec.get('launch_mean_us', 0):.2f} µs; stalls {dec.get('stall_calls', 0):.0f}/{dec.get('stall_us', 0):.0f} µs; "
                  f"alloc {dec.get('alloc_calls', 0):.0f}; cpu_ops main (outside hooks) {dec.get('cpu_ops_main_us', 0):.0f} µs; worker tree cpu {dec.get('worker_gpu_tree_cpu_us', 0):.0f}; "
                  f"GPU {{{', '.join(f'{c}: {v:.0f}' for c, v in sorted(dec.get('gpu_kernel_us', {}).items()))}}} busy {dec.get('gpu_busy_us', 0):.0f} model-span {dec.get('gpu_model_span_us', 0):.0f} "
                  f"tree∩model {dec.get('gpu_tree_model_overlap_us', 0):.0f} copy∩model {dec.get('gpu_copy_model_overlap_us', 0):.0f}", flush=True)
        (out / "profile_results.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
        return
    wl = va.load_workload(args.workload)
    engine_args = dict(wl.get("engine_args_required") or {})
    engine_args.update(va.parse_engine_args(args.engine_arg))
    eng = va.build_engine(args.checkpoints, max_num_seqs=8, case=va.case_of(wl, args.case), engine_args=engine_args)
    reqs, sampling = wl["requests"], wl["sampling"]
    base = baseline_state(eng)
    for i in range(2):
        r = run_marked(eng, va.throwaway_requests(eng, reqs, seed=100 + i), sampling)
        print(f"[prof] warm-up {i}: {r['wall_s']:.3f}s", flush=True)
    gcw = _GC()
    gc.callbacks.append(gcw)

    class A:
        chunk, hash_threads, arena_mb = 256, 2, 512

    com = ProfiledCommitter(eng, run_id="prof", chunk=256, hash_threads=2, staging="arena", arena_bytes=512 << 20, fast_copy=True, gpu_tree=True)
    ext = com._gpu_ext_load()                      # JIT-build / load CMT-2's tree extension HERE (main thread) so no worker step can silently fail
    print(f"[prof] gpu tree ext loaded: {ext}", flush=True)
    if args.hidden_so:
        com.attach_source(profiled_hidden_source(com, args.hidden_so))
    com.register_weights()
    com.install()
    r = run_marked(eng, va.throwaway_requests(eng, reqs, seed=200), sampling)
    rc = com.finalize()
    com.uninstall()
    com.reset()
    assert_pristine(eng, base)
    print(f"[prof] warm-up complete: {r['wall_s']:.3f}s roots {rc.root_ready_delay_s * 1e3:.1f} ms errors {rc.extra.get('errors')}", flush=True)
    hooks_only = make_committer("hooks_only", eng, A, "prof") if "hooks_only" in args.arms else None
    if hooks_only is not None:
        hooks_only.install()
        run_marked(eng, va.throwaway_requests(eng, reqs, seed=201), sampling)
        hooks_only.finalize()
        hooks_only.uninstall()
        hooks_only.reset()
        assert_pristine(eng, base)

    results = []
    tokens_ref = None
    arms = args.arms.split(",")
    for rep in range(args.reps):
        for arm in (arms if rep % 2 == 0 else arms[::-1]):
            gc.collect()
            torch.cuda.synchronize()
            ms0 = torch.cuda.memory_stats()
            c = None
            if arm == "complete":
                com.reset()
                com.install()
                c = com
            elif arm == "hooks_only":
                hooks_only.reset()
                hooks_only.install()
                c = hooks_only
            acts = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            with profile(activities=acts, with_stack=not args.no_stack, record_shapes=False, profile_memory=False) as p:
                r = run_marked(eng, reqs, sampling)
                if c is not None:
                    rcx = c.finalize()
                    c.uninstall()
                torch.cuda.synchronize()
            assert_pristine(eng, base)
            ms1 = torch.cuda.memory_stats()
            trace = out / f"trace_{arm}_{rep}.json"
            p.export_chrome_trace(str(trace))
            if tokens_ref is None:
                tokens_ref = r["tokens"]
            an = analyze(trace, arm)
            row = {"arm": arm, "rep": rep, "wall_s_under_profiler": r["wall_s"], "step_wall_s": r["step_wall_s"], "tokens_equal": r["tokens"] == tokens_ref,
                   "gc": gcw.summary(r["t0"], r["t1"]),
                   "allocator_delta": {k: ms1.get(k, 0) - ms0.get(k, 0) for k in ("num_alloc_retries", "num_device_alloc", "num_device_free", "num_sync_all_streams")},
                   "analysis": an}
            if c is not None:
                row["commit"] = {"root_ready_ms": rcx.root_ready_delay_s * 1e3, "spans_total_s": rcx.spans["total_s"], "gpu_tree_ms": rcx.extra.get("gpu_tree_ms"), "errors": rcx.extra.get("errors")}
            results.append(row)
            dec = mean_over(an["steps"], list(range(3, 15)))
            print(f"[prof] rep {rep} {arm:10s}: wall {r['wall_s'] * 1e3:.1f} ms (under profiler); decode-mean step {dec.get('wall_us', 0):.0f} µs; hooks {dec.get('hooks', 0):.0f} "
                  f"hook_body {dec.get('hook_body_us', 0):.0f} µs (bookkeeping {dec.get('bookkeeping_us', 0):.0f}, copy-issue {dec.get('copy_issue_us_in_hooks', 0):.0f} in {dec.get('copy_calls_in_hooks', 0):.0f} calls); "
                  f"fa2 glue {dec.get('fa2_wrapper_glue_us', 0):.0f}; launches {dec.get('launch_calls', 0):.0f} mean {dec.get('launch_mean_us', 0):.2f} µs; stalls {dec.get('stall_us', 0):.0f} µs/{dec.get('stall_calls', 0):.0f}; "
                  f"alloc {dec.get('alloc_calls', 0):.0f}; worker tree cpu {dec.get('worker_gpu_tree_cpu_us', 0):.0f} µs; GPU {dec.get('gpu_kernel_us', {})} busy {dec.get('gpu_busy_us', 0):.0f} overlap {dec.get('gpu_tree_model_overlap_us', 0):.0f}; "
                  f"gc {row['gc']}; alloc-delta {row['allocator_delta']}", flush=True)
            (out / "profile_results.json").write_text(json.dumps({"schema": SCHEMA, "label": "under torch.profiler — diagnostic, not headline", "results": results}, indent=1, default=str) + "\n")
    com.close()
    if hooks_only is not None:
        hooks_only.close()
    print(f"[prof] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
