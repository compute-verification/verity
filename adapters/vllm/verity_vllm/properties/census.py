"""Kernel census (DESIGN §8): every device kernel the bare run launches must be accounted for by a pattern.

Two halves, one CLI:

  GPU pod (torch, vLLM):
    verity-vllm census --workload workloads/workload_32x16_1req.json --out DIR
        Builds the engine exactly like `bench/m1_capture.py` (κ, `vllm_adapter.build_engine`), runs the uninstrumented
        throwaway warm-up (Triton JIT / cuBLAS / allocator / CUPTI), then runs the workload request with **no observer, no
        Triton hook, no module hooks**: every `engine.step()` is wrapped in its own `torch.profiler.profile(activities=
        [CPU, CUDA], record_shapes=False)` (the pattern of `out/curve/e10/kern_profile.py:183-199`), so the device kernels
        of engine step k are attributed to step k without any marker inside the runner.  Writes DIR/census.json: per step
        the device kernel names (demangled, as Kineto reports them) with count / total device µs / first grid+block, the
        sampled tokens after the step, plus the run-level histogram; the per-step Chrome traces go to DIR/traces/ (gz).
        Steps that launch no device kernel (the engine's trailing bookkeeping step) are listed but not numbered.

  laptop (torch-free):
    verity-vllm census --reconcile DIR/census.json --log data/logs/m1.jsonl.gz \
        [--profile vllm_d9105ea80_sm89_eager] [--report out/capture/reports/census_m1]
        Resolves the M1 log with `bench/resolve_log.resolve_log` and, per step, collects the kernel names every resolution
        part declares (`Compute.kernels` / `Transparent.kernels` over `resolver.parts_of`) plus the profile's
        `kernel_allowlist`.  Every census kernel of the same step must be matched by at least one declared name
        (`KERNEL_MATCH`: declared name -> regexes over the demangled kernel name; substring fallback).  Reports the residue
        (census kernels no declaration matches), the reverse (declared names never seen in the step), claim multiplicity
        (a kernel matched by declarations of more than one specialization is flagged) and the count comparison
        (kernels in the census vs raw events whose resolution declares the kernel).  Writes <report>.json and <report>.md.
        Exit status 1 when the residue is non-empty or a launch count differs.  `--propose-allowlist a,b` treats extra declared
        names as allowlisted (what-if for profile proposals; recorded separately in the report, the pinned profile is not touched).

Launch counts: per step, every declared kernel must launch on the device exactly as many times as the step has root events
whose resolution declares it.  Claiming by name alone lets an untraced launch of a declared kernel (a JIT or cuteDSL kernel
that leaves no dispatcher event) overwrite a Call's output in place: the fold sees no read-before-write and the residue stays
zero.  A mismatch is refused by name: the kernel, its device count against its
declaring events, and the buffers (allocations) those events write.  `kernel_allowlist.LAUNCH_COUNT_ALLOWLIST` exempts a
declared name with its reason, which the report records.  Two cases are not counted: launches the frozen `KERNEL_ALLOWLIST`
claims (copies, fills, the runner's metadata kernels), which the residue accepts at any count with or without a declaration,
and a declared name with no launch in the step, an alternative the step did not take (`fp8_mqa_logits` or
`fp8_paged_mqa_logits`), which stays in `declared_never_seen`.

Kernel names: Kineto reports demangled CUDA names (`void vllm::fused_add_rms_norm_kernel<c10::BFloat16, 8>(...)`,
`flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<64, 64, 128, 4, ...>, ...>`, `Memcpy DtoH (Device -> Pinned)`, `Memset
(Device)`, Triton kernels by their function name `matmul_kernel_persistent`).  The declared names in `patterns.py` and the
profile's allowlist are short tokens; `KERNEL_MATCH` is the explicit bridge.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_vllm import settings
from verity_vllm.config import UsageError, option, workload_arg
from verity_vllm.observe.kernel_trace import kernels_of_trace_events, load_trace, summarize_kernels
from verity_vllm.properties.record import seal

# ---------------------------------------------------------------------------------------------------------
# declared name -> regexes over the demangled kernel name (torch-free)
# ---------------------------------------------------------------------------------------------------------

#: The explicit matching table.  Keys are the names `patterns.py` declares in `kernels=` and the names of the profile's
#: `kernel_allowlist`; values are regexes tried with `re.search` on the Kineto kernel name.  A declared name absent here
#: falls back to a whole-token substring search (`_fallback_regex`).  `category` (below) is used for the multiplicity check.
KERNEL_MATCH: dict[str, tuple[str, ...]] = {
    # --- compute kernels declared by the b1 patterns ---
    "matmul_kernel_persistent": (r"^matmul_kernel_persistent(\b|$)",),                    # Triton, batch-invariant GEMM
    "_rms_norm_kernel": (r"^_rms_norm_kernel(\b|$)",),                                     # Triton, layer-0 input norm
    "fused_add_rms_norm_kernel": (r"vllm::fused_add_rms_norm_kernel<",),                  # _C.fused_add_rms_norm
    "rotary_embedding_kernel": (r"vllm::rotary_embedding_kernel<",),                      # _C.rotary_embedding
    "reshape_and_cache_flash_kernel": (r"vllm::reshape_and_cache_flash_kernel<",),        # _C_cache_ops.reshape_and_cache_flash
    "flash_fwd_splitkv_kernel": (r"(^|\W)flash_fwd_splitkv_kernel<",),                    # _vllm_fa2_C.varlen_fwd (num_splits=1)
    "FlashAttnFwdSm90": (r"flash::FlashAttnFwdSm90<",),                                    # _vllm_fa3_C.fwd (FA3, sm90; num_splits=1 under BI)
    "act_and_mul_kernel": (r"vllm::act_and_mul_kernel<",),                                # _C.silu_and_mul
    "indexSelectLargeIndex": (r"indexSelectLargeIndex<",),                                # aten.embedding / index_select
    "indexSelectSmallIndex": (r"indexSelectSmallIndex<",),
    "_gumbel_sample_kernel": (r"^_gumbel_sample_kernel(\b|$)",),                          # Triton, V2 sampler stage 1
    "ArgMaxOps": (r"reduce_kernel<.*ArgMaxOps<",),                                        # aten.argmax (sampler stage 2)
    "gather_kernel": (r"_scatter_gather_elementwise_kernel<", r"(?<![A-Za-z_])gather_kernel"),  # aten.gather (sampler stage 3)
    "index_elementwise_kernel": (r"index_elementwise_kernel<",),                          # aten.index.Tensor
    # torch >= 2.13 index_select fast path (contiguous rows, > 16 indices): the prefill embedding lookup.  Declared by NO
    # pattern yet (embedding.aten_embedding declares indexSelect{Large,Small}Index) -> reported as residue; see census_m1.md.
    "vectorized_gather_kernel": (r"vectorized_gather_kernel<",),
    # V2 runner post-sample request-state update (vllm/v1/worker/gpu/input_batch.py: appends the sampled ids to
    # all_token_ids, bumps total_len / num_computed_tokens; integer pointers only).  Runs AFTER runner.sample() returns,
    # i.e. outside the observer's step window; not in the pinned profile's allowlist -> residue; see census_m1.md.
    "_post_update_kernel": (r"^_post_update_kernel(\b|$)",),
    # --- transparent / infrastructure (patterns.COPY_KERNELS / FILL_KERNELS and the profile allowlist) ---
    "Memcpy": (r"^Memcpy ",),
    "Memset": (r"^Memset ",),
    "FillFunctor": (r"FillFunctor<",),
    "fill_kernel": (r"fill_kernel",),
    "vectorized_elementwise_kernel": (r"vectorized_elementwise_kernel<",),
    "elementwise_kernel": (r"(?<![A-Za-z_])elementwise_kernel<",),                        # not vectorized_/unrolled_
    "unrolled_elementwise_kernel": (r"unrolled_elementwise_kernel<",),
    "index_put": (r"index_put",),
    "arange": (r"arange",),
    "cub::": (r"cub::",),
    "DeviceScan": (r"DeviceScan",),
    "DeviceReduce": (r"DeviceReduce",),
    "CatArrayBatchedCopy": (r"CatArrayBatchedCopy",),
    "masked_fill": (r"masked_fill",),
    "compare": (r"(^|\W)compare",),
    "where": (r"(^|\W)where_",),
    "cumsum": (r"cumsum",),
    "_scan": (r"_scan",),
    "sort": (r"(^|\W)sort",),
    "bitonic": (r"bitonic",),
    "radix": (r"radix",),
    # V2 model runner input preparation (Triton, integer pointers only; `TransparentTritonMetadata`)
    "_apply_write_kernel": (r"^_apply_write_kernel(\b|$)",),
    "_prepare_prefill_inputs_kernel": (r"^_prepare_prefill_inputs_kernel(\b|$)",),
    "_prepare_pos_seq_lens_kernel": (r"^_prepare_pos_seq_lens_kernel(\b|$)",),
    "_combine_sampled_and_draft_tokens_kernel": (r"^_combine_sampled_and_draft_tokens_kernel(\b|$)",),
    "_gather_block_tables_kernel": (r"^_gather_block_tables_kernel(\b|$)",),
    "_compute_slot_mappings_kernel": (r"^_compute_slot_mappings_kernel(\b|$)",),
    "_get_num_sampled_and_rejected_kernel": (r"^_get_num_sampled_and_rejected_kernel(\b|$)",),
}

#: Declared names that stand for a *computation* (a registered specialization's kernel).  A census kernel matched by
#: declared names of two different specializations is an ambiguity the report flags; matches shared between a compute
#: name and infrastructure names, or among infrastructure names, are fine (e.g. a fill is both `FillFunctor` and
#: `vectorized_elementwise_kernel`).
COMPUTE_KERNELS: frozenset[str] = frozenset({
    "matmul_kernel_persistent", "_rms_norm_kernel", "fused_add_rms_norm_kernel", "rotary_embedding_kernel",
    "flash_fwd_splitkv_kernel", "FlashAttnFwdSm90", "act_and_mul_kernel", "indexSelectLargeIndex", "indexSelectSmallIndex",
    "_gumbel_sample_kernel", "ArgMaxOps", "gather_kernel", "vectorized_gather_kernel",
})

_COMPILED: dict[str, list[re.Pattern]] = {}


def _fallback_regex(name: str) -> str:
    return r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])"


def regexes_for(declared: str) -> list[re.Pattern]:
    rs = _COMPILED.get(declared)
    if rs is None:
        pats = KERNEL_MATCH.get(declared) or (_fallback_regex(declared),)
        rs = [re.compile(p) for p in pats]
        _COMPILED[declared] = rs
    return rs


def matches(declared: str, kernel_name: str) -> bool:
    return any(r.search(kernel_name) for r in regexes_for(declared))


def claims_for(kernel_name: str, declared: set[str] | frozenset[str]) -> list[str]:
    """The declared names (sorted) whose regexes match `kernel_name`."""
    return sorted(d for d in declared if matches(d, kernel_name))


# ---------------------------------------------------------------------------------------------------------
# GPU half: profile the bare run step by step
# ---------------------------------------------------------------------------------------------------------

def run_census(workload_path: str, out_dir: str, checkpoints: str, *, max_tokens: int | None, warmup: bool,
               max_num_seqs: int, keep_traces: bool, case: str | None = None, engine_args: dict | None = None) -> dict:
    import shutil

    import torch
    from torch.profiler import ProfilerActivity, profile

    from verity_vllm.engine import engine_profile as prof

    prof.apply_env()
    from verity_vllm.engine import vllm_adapter as va

    out = Path(out_dir)
    (out / "traces").mkdir(parents=True, exist_ok=True)
    wl = va.load_workload(workload_path, max_tokens)
    case = va.case_of(wl, case)
    if (gpu_util := settings.VERITY_CENSUS_GPU_UTIL) is not None:
        # [coverage] a census re-run that shares the GPU with another engine: the KV-cache size does not enter the kernel counts
        # (no preemption at these workload sizes); the resolved value is on record in this document's engine_kwargs
        engine_args = {**(engine_args or {}), "gpu_memory_utilization": gpu_util}
    eng = va.build_engine(checkpoints, max_num_seqs=max_num_seqs, case=case, engine_args=engine_args, target=va.target_of_workload(wl))
    print(f"[census] engine built in {eng.build_s:.1f}s (case {case}, {eng.cp['repo']})", flush=True)
    versions = va.versions_doc(eng)
    (out / "versions.json").write_text(json.dumps(versions, indent=1) + "\n")
    engine = eng.llm.llm_engine

    def profiled_run(requests: list[dict], tag: str) -> tuple[list[dict], dict[str, list[int]]]:
        """The bare engine's request loop with one profiler session per engine step (no wrappers at all).

        [coverage D87, adapter-level] the same step-indexed arrival rule as `verity_vllm.engine.arrivals.run_requests_arrivals`: request
        r is admitted right before forward r["arrive_step"] (all 0 = the historical loop); an idle engine admits the next arrival at
        once.  The census must profile the batch composition the capture saw, or its per-step kernel counts cannot align with the
        log (G1: "census steps not aligned with log steps" on every B8-mixed cell).  Iterations that launched no device kernel
        (vLLM V1 returns the previous forward's sample, so the loop ends with one drain iteration) carry no census information and
        are not steps: `engine_step` indexes forwards, the unit the log, the observer and the gates count."""
        arrivals = sorted(requests, key=lambda r: int(r.get("arrive_step") or 0))
        steps: list[dict] = []
        tokens: dict[str, list[int]] = {}
        k = 0            # forward index (steps with device kernels; the admission clock, as in arrivals.py)
        it = 0           # engine.step() iterations
        n_empty = 0
        t_run0 = time.perf_counter()
        realized: dict[str, dict] = {}   # per request: planned arrive_step, admit iteration, first forward, finish forward, n_generated

        def _admit(r: dict) -> None:
            engine.add_request(request_id=r["request_id"], prompt={"prompt_token_ids": list(r["prompt_token_ids"])},
                               params=va.sampling_params(r, wl["sampling"]))
            realized[r["request_id"]] = {"arrive_step": int(r.get("arrive_step") or 0), "arrive_step_realized": k, "admit_iter": it,
                                         "first_step": k, "finish_step": None, "finish_reason": None, "n_generated": None}

        while arrivals or engine.has_unfinished_requests():
            while arrivals and int(arrivals[0].get("arrive_step") or 0) <= k:
                _admit(arrivals.pop(0))
            if not engine.has_unfinished_requests():
                _admit(arrivals.pop(0))
                continue
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as pr:
                outs = engine.step()
                torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            it += 1
            tr = out / "traces" / f"{tag}-engine_step{k}.json"
            pr.export_chrome_trace(str(tr))
            evs = load_trace(tr)
            ks = kernels_of_trace_events(evs)
            for o in outs:
                tokens[o.request_id] = [int(t) for t in o.outputs[0].token_ids]
                if getattr(o, "finished", False) and o.request_id in realized and realized[o.request_id]["finish_step"] is None:
                    realized[o.request_id].update({"finish_step": k - 1, "finish_iter": it - 1, "n_generated": len(o.outputs[0].token_ids),
                                                   "finish_reason": getattr(o.outputs[0], "finish_reason", None)})
            if not ks:
                tr.unlink()
                n_empty += 1
                continue
            if keep_traces:
                with open(tr, "rb") as fi, gzip.open(str(tr) + ".gz", "wb") as fo:
                    shutil.copyfileobj(fi, fo)
            tr.unlink()
            cpu_ops = Counter(e.get("name", "") for e in evs if e.get("ph") == "X" and e.get("cat") == "cpu_op")
            steps.append({"engine_step": k, "iteration": it - 1, "wall_s": wall, "n_kernels": len(ks), "kernels": summarize_kernels(ks),
                          "device_us": round(sum(float(e.get("dur", 0.0) or 0.0) for e in ks), 1),
                          "cpu_ops_total": sum(cpu_ops.values()), "cpu_ops_distinct": len(cpu_ops),
                          "tokens_after": {rid: list(t) for rid, t in tokens.items()},
                          "trace": f"traces/{tag}-engine_step{k}.json.gz" if keep_traces else None})
            print(f"[census] {tag} engine step {k}: {len(ks)} device kernels ({steps[-1]['device_us']} us), "
                  f"{sum(cpu_ops.values())} cpu ops, wall {wall * 1e3:.1f} ms", flush=True)
            k += 1
        print(f"[census] {tag}: {k} engine steps ({it} iterations, {n_empty} without device kernels) in {time.perf_counter() - t_run0:.2f}s", flush=True)
        profiled_run.last_realized = {"engine_steps": k, "step_iterations": it, "empty_iterations": n_empty, "requests": realized}
        return steps, tokens

    if warmup:
        warm = va.throwaway_requests(eng, wl["requests"], seed=va.WARMUP_SEED)
        profiled_run(warm, "warmup")      # profiler + CUPTI warm-up too; discarded
        torch.cuda.synchronize()
    steps, tokens = profiled_run(wl["requests"], "census")

    # number the steps that launched device work (= the M1 log's steps: execute_model with scheduled tokens)
    numbered = []
    n = 0
    for s in steps:
        if s["n_kernels"]:
            s["step"] = n
            n += 1
            numbered.append(s)
        else:
            s["step"] = None
    hist: Counter = Counter()
    us: Counter = Counter()
    for s in numbered:
        for name, d in s["kernels"].items():
            hist[name] += d["count"]
            us[name] += d["us"]
    run_hist = OrderedDict((name, {"count": c, "us": round(us[name], 1)}) for name, c in
                           sorted(hist.items(), key=lambda kv: (-kv[1], kv[0])))
    acc = {rid: wl["accepted"].get(rid) for rid in tokens}
    doc = {
        "schema": "verity-capture/census/v1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workload": {"source": wl.get("source"), "sha256": wl.get("sha256"), "requests": wl["requests"], "sampling": wl["sampling"]},
        "engine_kwargs": {k: v for k, v in eng.kwargs.items() if k not in ("model", "tokenizer")},
        "env_flags": eng.env_flags,
        "versions": versions,
        "method": "uninstrumented engine (no observer / Triton hook / module hooks); one torch.profiler.profile(activities=[CPU, CUDA], "
                  "record_shapes=False) per engine.step() after an uninstrumented+profiled throwaway warm-up; device kernels = Kineto "
                  "trace events with cat in {kernel, gpu_memcpy, gpu_memset}; steps with no device kernel are not numbered",
        "engine_steps": len(steps),
        "steps_with_device_work": len(numbered),
        "steps": steps,
        # [campaign runner, spec §5] the realized arrivals of THIS phase, in the capture's units (forward indices), so experiment.json can
        # check that every phase ran the same concrete workload instead of trusting the shared spec digest alone
        "realized": getattr(profiled_run, "last_realized", None),
        "run_histogram": run_hist,
        "tokens": tokens,
        "accepted_tokens": acc,
        "tokens_equal_accepted": all(acc[r] is not None and tokens[r] == acc[r] for r in tokens),
    }
    (out / "census.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"[census] {len(numbered)} steps with device work / {len(steps)} engine steps; tokens_equal_accepted="
          f"{doc['tokens_equal_accepted']}; {len(run_hist)} distinct kernel names -> {out / 'census.json'}", flush=True)
    return doc


# ---------------------------------------------------------------------------------------------------------
# laptop half: reconcile against the resolved M1 log (torch-free)
# ---------------------------------------------------------------------------------------------------------


def declared_kernels_of_step(resolutions: list, roots: list) -> tuple[dict[str, set[str]], dict[str, set[int]], dict[str, int]]:
    """From one step's resolutions: declared kernel -> {pattern names}, declared kernel -> {root event ids that declare it},
    declared kernel -> number of parts declaring it."""
    from verity_vllm.observe.fold.resolver import Compute, Transparent, parts_of

    pats: dict[str, set[str]] = {}
    events: dict[str, set[int]] = {}
    parts_n: Counter = Counter()
    for root, r in zip(roots, resolutions):
        for p in parts_of(r):
            if not isinstance(p, (Compute, Transparent)):
                continue
            pat = getattr(p, "pattern", "") or getattr(r, "pattern", "")
            for k in p.kernels:
                pats.setdefault(k, set()).add(pat)
                events.setdefault(k, set()).add(root.id)
                parts_n[k] += 1
    return pats, events, dict(parts_n)


def declaring_events(resolutions: list, roots: list, kernel: str) -> list[dict]:
    """The step's root events whose resolution declares `kernel`: id, op, module and the allocations its declaring parts write."""
    from verity_vllm.observe.fold.resolver import Compute, Transparent, parts_of

    out = []
    for root, r in zip(roots, resolutions):
        parts = [p for p in parts_of(r) if isinstance(p, (Compute, Transparent)) and kernel in p.kernels]
        if not parts:
            continue
        written = [o for p in parts for o in (p.outputs if isinstance(p, Compute) else (p.dst,))]
        allocs = sorted({v.alloc for o in written if (v := getattr(o, "view", None) or getattr(o, "own", None)) is not None})
        out.append({"event": root.id, "op": root.op, "module": getattr(root.enter, "module", None), "writes": allocs})
    return out


def count_line(m: dict) -> str:
    """One launch-count refusal (or exemption) as a line: kernel, device count vs declaring events, the buffers they write."""
    evs = m.get("events") or []
    where = "; ".join(f"event {e['event']} {e['op']} {e['module'] or '-'} writes allocs {e['writes']}" for e in evs[:4])
    more = f"; +{len(evs) - 4} more" if len(evs) > 4 else ""
    return (f"step {m['step']}: {m['kernel']} launched {m['device_count']} times on the device against {m['declaring_events']} "
            f"declaring events (buffers {m.get('buffers')}{': ' + where + more if where else ''})")


def reconcile(census: dict, log_path: str, profile_name: str, proposed_allowlist: tuple[str, ...] = ()) -> dict:
    """`proposed_allowlist`: extra declared names treated as allowlisted (what-if; L5 may not edit the profile).  The
    report records them separately (`proposed_allowlist`) so the measured residue against the pinned profile stays visible."""
    from verity_vllm.observe.fold.resolve_log import load_profile, resolve_log

    profile, prof_mod = load_profile(profile_name)
    # protected: a profile cannot allowlist kernels by itself
    from verity_vllm.properties.kernel_allowlist import KERNEL_ALLOWLIST as FROZEN_ALLOWLIST
    from verity_vllm.properties.kernel_allowlist import LAUNCH_COUNT_ALLOWLIST
    allow = set(FROZEN_ALLOWLIST) | set(proposed_allowlist)
    profile_extra_allowlist = sorted(set(profile.kernel_allowlist) - set(FROZEN_ALLOWLIST))
    per_step_roots: dict[int, list] = {}

    def on_step(se, sr):
        per_step_roots[sr.step] = se.roots

    header, results, _roots, _notes = resolve_log(log_path, profile, None, on_step)
    by_step = {sr.step: sr for sr in results}
    census_steps = [s for s in census["steps"] if s.get("step") is not None]
    steps_out = []
    residue_total: Counter = Counter()
    kernel_rows: "OrderedDict[str, dict]" = OrderedDict()   # kernel name -> {counts per step, claims}
    ambiguous: list[dict] = []
    count_mismatches: list[dict] = []
    count_exempt: Counter = Counter()                        # allowlisted declared name -> steps whose count differed
    for cs in census_steps:
        i = cs["step"]
        sr = by_step.get(i)
        if sr is None or sr.error:
            steps_out.append({"step": i, "error": f"log step {i} missing or unresolved: {getattr(sr, 'error', None)}"})
            continue
        pats, events, parts_n = declared_kernels_of_step(sr.resolutions, per_step_roots[i])
        declared = set(pats) | allow
        residue = []
        seen_declared: set[str] = set()
        claims_by_kernel: dict[str, list[str]] = {}
        for name, d in cs["kernels"].items():
            cl = claims_for(name, declared)
            claims_by_kernel[name] = cl
            seen_declared.update(cl)
            row = kernel_rows.setdefault(name, {"counts": {}, "us": {}, "claims": set(), "patterns": set(), "grid": d.get("grid"),
                                                "block": d.get("block"), "cat": d.get("cat")})
            row["counts"][i] = d["count"]
            row["us"][i] = d["us"]
            row["claims"].update(cl)
            for c in cl:
                row["patterns"].update(pats[c] if c in pats else {"allowlist"})
            if not cl:
                residue.append({"kernel": name, "count": d["count"], "us": d["us"]})
                residue_total[name] += d["count"]
            comp = {c for c in cl if c in COMPUTE_KERNELS}
            if len(comp) > 1:
                ambiguous.append({"step": i, "kernel": name, "compute_claims": sorted(comp)})
        never_seen = sorted(set(pats) - seen_declared)
        allow_never_seen = sorted(allow - seen_declared)
        infrastructure = {n for n in cs["kernels"] if claims_for(n, allow)}
        counts = {}
        step_mismatches = []
        for k in sorted(pats):
            census_count = sum(d["count"] for n, d in cs["kernels"].items() if matches(k, n))
            counts[k] = {"census_kernels": census_count, "declaring_events": len(events[k]), "declaring_parts": parts_n[k],
                         "patterns": sorted(pats[k])}
            launched = {n: d["count"] for n, d in cs["kernels"].items() if matches(k, n) and n not in infrastructure}
            if sum(launched.values()) != census_count:
                counts[k]["infrastructure"] = census_count - sum(launched.values())
            if not launched or sum(launched.values()) == len(events[k]):
                continue
            if k in LAUNCH_COUNT_ALLOWLIST:
                counts[k]["exempt"] = LAUNCH_COUNT_ALLOWLIST[k]
                count_exempt[k] += 1
                continue
            evs = declaring_events(sr.resolutions, per_step_roots[i], k)
            step_mismatches.append({"step": i, "kernel": k, "device_kernels": launched, "device_count": sum(launched.values()),
                                    "declaring_events": len(events[k]), "patterns": sorted(pats[k]),
                                    "buffers": sorted({a for e in evs for a in e["writes"]}), "events": evs})
        count_mismatches.extend(step_mismatches)
        steps_out.append({"step": i, "phase": sr.phase_hint, "rows": sr.rows, "log_events": sr.events, "log_roots": sr.roots,
                          "census_kernels": cs["n_kernels"], "census_distinct": len(cs["kernels"]), "device_us": cs.get("device_us"),
                          "declared_by_patterns": sorted(pats), "residue": residue,
                          "declared_never_seen": never_seen, "allowlist_never_seen": allow_never_seen,
                          "counts": counts, "count_mismatches": [m["kernel"] for m in step_mismatches]})
    rows_out = []
    for name, row in kernel_rows.items():
        rows_out.append({"kernel": name, "cat": row["cat"], "grid": row["grid"], "block": row["block"],
                         "count_step0": row["counts"].get(0, 0),
                         "count_decode": sorted({c for s, c in row["counts"].items() if s > 0}),
                         "count_total": sum(row["counts"].values()), "us_total": round(sum(row["us"].values()), 1),
                         "steps_present": sorted(row["counts"]), "claims": sorted(row["claims"]),
                         "patterns": sorted(row["patterns"]), "residue": not row["claims"]})
    total_kernels = sum(cs["n_kernels"] for cs in census_steps)
    rep = {
        "schema": "verity-capture/census-reconciliation/v2",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "census": {"utc": census.get("utc"), "workload_sha256": (census.get("workload") or {}).get("sha256"),
                   "engine_steps": census.get("engine_steps"), "steps_with_device_work": len(census_steps),
                   "tokens_equal_accepted": census.get("tokens_equal_accepted"), "method": census.get("method"),
                   "versions": {k: (census.get("versions") or {}).get(k) for k in ("torch", "triton", "vllm", "cuda", "gpu")}},
        "log": log_path, "log_profile_id": header.profile_id if header else None, "profile": profile.id,
        "log_steps": len(results), "census_steps": len(census_steps),
        "steps_aligned": len(results) == len(census_steps),
        "totals": {"census_kernels": total_kernels, "distinct_kernel_names": len(kernel_rows),
                   "residue_kernels": sum(residue_total.values()), "residue_names": sorted(residue_total),
                   "ambiguous": len(ambiguous), "log_unsupported": sum(len(sr.unsupported) for sr in results),
                   "count_mismatches": len(count_mismatches),
                   "count_mismatch_kernels": sorted({m["kernel"] for m in count_mismatches}),
                   "count_exempt": dict(sorted(count_exempt.items()))},
        "kernels": rows_out,
        "steps": steps_out,
        "ambiguous": ambiguous[:64],
        "count_mismatches": count_mismatches,
        "count_allowlist": {k: LAUNCH_COUNT_ALLOWLIST[k] for k in sorted(count_exempt)},
        "match_table": {k: list(v) for k, v in KERNEL_MATCH.items()},
        "compute_kernels": sorted(COMPUTE_KERNELS),
        "allowlist": sorted(allow), "profile_extra_allowlist": profile_extra_allowlist,
        "proposed_allowlist": sorted(proposed_allowlist),
    }
    rep["residue_zero"] = rep["totals"]["residue_kernels"] == 0 and rep["steps_aligned"] and all("error" not in s for s in steps_out)
    rep["counts_equal"] = not count_mismatches
    return seal(rep, rep["residue_zero"] and rep["counts_equal"])


def reconciliation_markdown(rep: dict) -> str:
    L: list[str] = []
    L.append(f"# Kernel census × resolution: `{Path(rep['log']).name}` × profile `{rep['profile']}`")
    L.append("")
    L.append(f"- reconciled {rep['utc']}; census taken {rep['census']['utc']} (torch {rep['census']['versions'].get('torch')}, "
             f"triton {rep['census']['versions'].get('triton')}, vLLM {rep['census']['versions'].get('vllm')}, GPU "
             f"{(rep['census']['versions'].get('gpu') or {}).get('name') if isinstance(rep['census']['versions'].get('gpu'), dict) else rep['census']['versions'].get('gpu')})")
    L.append(f"- method: {rep['census']['method']}")
    t = rep["totals"]
    L.append(f"- census: {rep['census_steps']} steps with device work ({rep['census']['engine_steps']} engine steps), "
             f"{t['census_kernels']} device kernels, {t['distinct_kernel_names']} distinct names; tokens == accepted record: "
             f"{rep['census']['tokens_equal_accepted']}")
    L.append(f"- log: {rep['log_steps']} steps (profile_id `{rep['log_profile_id']}`), Unsupported {t['log_unsupported']}; steps aligned: {rep['steps_aligned']}")
    L.append(f"- **residue: {t['residue_kernels']} kernels** ({', '.join(f'`{n}`' for n in t['residue_names']) or 'none'}); "
             f"ambiguous compute claims: {t['ambiguous']}  →  **residue_zero = {rep['residue_zero']}**")
    L.append(f"- **launch counts: {t['count_mismatches']} refused** ({', '.join(f'`{n}`' for n in t['count_mismatch_kernels']) or 'none'}); "
             f"exempt by the launch-count allowlist: {', '.join(f'`{k}` ({n} steps)' for k, n in t['count_exempt'].items()) or 'none'}"
             f"  →  **counts_equal = {rep['counts_equal']}**")
    L.append("")
    if rep["count_mismatches"]:
        L.append("## Launch counts refused")
        L.append("")
        for m in rep["count_mismatches"][:64]:
            L.append(f"- {count_line(m)}")
        L.append("")
    if rep["count_allowlist"]:
        L.append("## Launch counts exempt (`kernel_allowlist.LAUNCH_COUNT_ALLOWLIST`)")
        L.append("")
        for k, why in rep["count_allowlist"].items():
            L.append(f"- `{k}`: {why}")
        L.append("")
    L.append("## Kernels (all steps)")
    L.append("")
    L.append("| kernel (Kineto name, truncated) | cat | step 0 | decode steps | total | device µs | claimed by (declared name → pattern) |")
    L.append("| --- | --- | ---: | ---: | ---: | ---: | --- |")
    for r in rep["kernels"]:
        dec = "/".join(str(c) for c in r["count_decode"]) if r["count_decode"] else "-"
        claims = ", ".join(f"`{c}`" for c in r["claims"]) if r["claims"] else "**RESIDUE**"
        pats = ", ".join(f"`{p}`" for p in r["patterns"])
        name = r["kernel"]
        if len(name) > 110:
            name = name[:107] + "..."
        L.append(f"| `{name}` | {r['cat']} | {r['count_step0']} | {dec} | {r['count_total']} | {r['us_total']} | {claims} → {pats} |")
    L.append("")
    L.append("## Per step")
    L.append("")
    L.append("| step | phase | rows | census kernels | distinct | device µs | log events | residue | declared never seen |")
    L.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for s in rep["steps"]:
        if "error" in s:
            L.append(f"| {s['step']} | ERROR | | | | | | {s['error']} | |")
            continue
        res = ", ".join(f"`{r['kernel'][:60]}`×{r['count']}" for r in s["residue"]) or "0"
        nv = ", ".join(f"`{k}`" for k in s["declared_never_seen"]) or "-"
        L.append(f"| {s['step']} | {s['phase']} | {s['rows']} | {s['census_kernels']} | {s['census_distinct']} | {s['device_us']} | "
                 f"{s['log_events']} | {res} | {nv} |")
    L.append("")
    L.append("## Count comparison (declared compute / transparent kernels; step 0 = prefill, step 1 = first decode)")
    L.append("")
    L.append("`census` = kernels of that name in the profiler; `events` = raw root events of the step whose resolution declares the "
             "name; `parts` = resolution parts declaring it (RoPE declares its kernel twice — q and k —, the KV update twice — K and V —, "
             "the attention scope once per distinct T, the two-stage sampler three times for its Triton kernel).")
    L.append("")
    steps = [s for s in rep["steps"] if "error" not in s]
    if steps:
        names = sorted({k for s in steps for k in s["counts"]})
        L.append("| declared | " + " | ".join(f"step {s['step']} census/events/parts" for s in steps[:2]) + " | decode steps 2.. census (set) | patterns |")
        L.append("| --- | " + " | ".join("---" for _ in steps[:2]) + " | --- | --- |")
        for k in names:
            cells = []
            for s in steps[:2]:
                c = s["counts"].get(k)
                cells.append(f"{c['census_kernels']}/{c['declaring_events']}/{c['declaring_parts']}" if c else "-")
            rest = sorted({s["counts"][k]["census_kernels"] for s in steps[2:] if k in s["counts"]})
            pats = sorted({p for s in steps for p in (s["counts"].get(k) or {}).get("patterns", [])})
            L.append(f"| `{k}` | " + " | ".join(cells) + f" | {rest} | {', '.join(f'`{p}`' for p in pats)} |")
        L.append("")
    if rep["ambiguous"]:
        L.append("## Ambiguous compute claims")
        L.append("")
        for a in rep["ambiguous"][:32]:
            L.append(f"- step {a['step']}: `{a['kernel'][:100]}` ← {a['compute_claims']}")
        L.append("")
    L.append("## Matching table (`census.KERNEL_MATCH`)")
    L.append("")
    L.append("| declared name | regexes |")
    L.append("| --- | --- |")
    for k, v in rep["match_table"].items():
        L.append(f"| `{k}` | " + ", ".join(f"`{p}`" for p in v) + " |")
    L.append("")
    L.append(f"Allowlist (profile `kernel_allowlist`): {', '.join(f'`{a}`' for a in rep['allowlist'])}")
    if rep.get("proposed_allowlist"):
        L.append("")
        L.append(f"Proposed allowlist additions applied in this what-if reconciliation (NOT in the pinned profile): "
                 f"{', '.join(f'`{a}`' for a in rep['proposed_allowlist'])}")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Options:
    workload: str | None = option("--workload", type=workload_arg, default=None, help="GPU half: run the profiled bare workload")
    out: str | None = option("--out", default=None, help="GPU half: output directory (census.json, traces/)")
    checkpoints: str = option("--checkpoints", default="manifests/checkpoints.json")
    max_tokens: int | None = option("--max-tokens", type=int, default=None)
    no_warmup: bool = option("--no-warmup", default=False, action="store_true")
    max_num_seqs: int = option("--max-num-seqs", type=int, default=8)
    case: str | None = option("--case", default=None, help="checkpoint role (B0 / B1); default: the workload file's, else B0")
    no_traces: bool = option("--no-traces", default=False, action="store_true",
                             help="GPU half: do not keep the per-step Chrome traces")
    engine_arg: list[str] = option("--engine-arg", action="append", default=[],
                                   help="GPU half: LLM(**kw) override key=value (JSON value), repeatable")
    reconcile: str | None = option("--reconcile", default=None, help="laptop half: path to census.json")
    log: str = option("--log", default="data/logs/m1.jsonl.gz")
    profile: str = option("--profile", default="vllm_d9105ea80_sm89_eager")
    report: str = option("--report", default="out/capture/reports/census_m1", help="laptop half: report base path (.json/.md)")
    propose_allowlist: str = option("--propose-allowlist", default="",
                                    help="laptop half: comma list of declared names to treat as allowlisted (what-if)")


def main(args: Options) -> int:
    if args.reconcile:
        census = json.loads(Path(args.reconcile).read_text())
        rep = reconcile(census, args.log, args.profile, tuple(x for x in args.propose_allowlist.split(",") if x))
        base = Path(args.report)
        base.parent.mkdir(parents=True, exist_ok=True)
        Path(str(base) + ".json").write_text(json.dumps(rep, indent=1) + "\n")
        Path(str(base) + ".md").write_text(reconciliation_markdown(rep))
        t = rep["totals"]
        print(f"[census] steps {rep['census_steps']} (log {rep['log_steps']}), kernels {t['census_kernels']}, distinct {t['distinct_kernel_names']}, "
              f"residue {t['residue_kernels']} {t['residue_names']}, ambiguous {t['ambiguous']} -> residue_zero={rep['residue_zero']}; "
              f"launch counts refused {t['count_mismatches']} {t['count_mismatch_kernels']} -> counts_equal={rep['counts_equal']}")
        for s in rep["steps"]:
            if "error" in s:
                print(f"  step {s['step']}: {s['error']}")
            elif s["residue"]:
                print(f"  step {s['step']}: residue " + "; ".join(f"{r['kernel'][:80]} x{r['count']}" for r in s["residue"]))
        for m in rep["count_mismatches"]:
            print(f"  {count_line(m)}")
        print(f"wrote {base}.json and {base}.md")
        return 0 if rep["ok"] else 1
    if not args.workload or not args.out:
        raise UsageError("either --reconcile CENSUS.json or --workload W --out DIR")
    run_census(args.workload, args.out, args.checkpoints, max_tokens=args.max_tokens, warmup=not args.no_warmup,
               max_num_seqs=args.max_num_seqs, keep_traces=not args.no_traces, case=args.case, engine_args=_engine_args(args.engine_arg))
    return 0


def _engine_args(items: list[str]) -> dict:
    from verity_vllm.engine import vllm_adapter as va  # torch import only on the GPU half
    return va.parse_engine_args(items)
