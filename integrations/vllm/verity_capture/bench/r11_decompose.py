"""R11 collector-meas — DIAGNOSTIC decomposition of the remaining complete-commitment delta of the native gather collector on Llama-3.2-1B.

ONE profiled run (torch.profiler CPU+CUDA, no stacks) with arms control / candidate (`native_collect_gather`, or the kind given with
`--kind`) / `hooks_only` interleaved, N reps each; the candidate is subclassed HERE only to add `record_function` markers around the
collection body (`verity.hook`), the gather flush (`verity.flush`), step begin/end, the worker's GPU tree (`verity.gpu_tree`) and the FA2
wrapper (`verity.fa2wrap`).  The collector itself is untouched (markers wrap its calls).  Trace analysis is R10's
`acqlife_profile.analyze` with the gather kernel classified as `copy`.

NEVER a headline number: the profiler inflates the CPU side ~2x (R10: unprofiled +5.97 ms/step vs +13.0 ms under the no-stack profiler).
The output table gives each component in µs per decode step under the profiler AND its share of the profiled delta; the REPORT scales
the shares onto the unprofiled delta measured by r11_llama_measure.sh.

    PYTHONPATH=. python -m verity_capture.bench.r11_decompose --case LLAMA32_1B --hidden-so .../verity_fa2_matReq.so \
        --engine-arg gpu_memory_utilization=0.5 --reps 2 --out /workspace/cp/out-r11/collector-meas/decompose
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any

from verity_vllm_adapter import profile as prof  # noqa: F401

prof.apply_env()

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile, record_function  # noqa: E402

from verity_capture import vllm_adapter as va  # noqa: E402
from verity_capture.bench import acqlife_profile as AP  # noqa: E402
from verity_capture.bench.commit_delta import assert_pristine, baseline_state, make_committer  # noqa: E402
from verity_capture.commit import native_collect as NC  # noqa: E402

SCHEMA = "verity/r11/collector-meas/decompose/v1"

_orig_kernel_class = AP.kernel_class


def kernel_class(name: str) -> str:
    n = name.lower()
    if "gather_copy" in n or "native_gather" in n:
        return "copy"
    return _orig_kernel_class(name)


AP.kernel_class = kernel_class


class MarkedGather(NC.NativeCollectCommitter):
    """Markers only: the wrapper body is the parent's code path (stage/collect + flush at the legal points) with record_function around it."""

    def _make_wrapper(self, orig, name: str, pre: bool):
        me = self
        collect = self._nc.stage if self.gather else self._nc.collect
        flush = self._nc.flush if (self.gather and name.rsplit(".", 1)[-1] in NC.GATHER_FLUSH_LEAVES) else None
        Tensor = torch.Tensor
        nid_out = self._intern(name)
        nids_k = tuple(self._intern(f"{name}/{k}") for k in range(8))
        nid_in = self._intern(f"{name}/in") if pre else None
        nid_st = self._intern(f"{name}/sampled_token_ids")

        def collect_out(out):
            if isinstance(out, Tensor):
                collect(nid_out, out)
            elif isinstance(out, (tuple, list)):
                for k, o in enumerate(out):
                    if isinstance(o, Tensor):
                        collect(nids_k[k], o)
            else:
                st = getattr(out, "sampled_token_ids", None)
                if isinstance(st, Tensor):
                    collect(nid_st, st)
            if flush is not None:
                with record_function("verity.flush"):
                    flush()

        if pre:
            def fwd(*a, **kw):
                if me._cur is not None:
                    with record_function("verity.prehook"):
                        try:
                            if a and isinstance(a[0], Tensor):
                                collect(nid_in, a[0])
                        except Exception as e:  # noqa: BLE001
                            me.errors.append(f"{name}/in: {e!r}"[:300])
                out = orig(*a, **kw)
                if me._cur is not None:
                    with record_function("verity.hook"):
                        try:
                            collect_out(out)
                        except Exception as e:  # noqa: BLE001
                            me.errors.append(f"{name}: {e!r}"[:300])
                return out
        else:
            def fwd(*a, **kw):
                out = orig(*a, **kw)
                if me._cur is not None:
                    with record_function("verity.hook"):
                        try:
                            collect_out(out)
                        except Exception as e:  # noqa: BLE001
                            me.errors.append(f"{name}: {e!r}"[:300])
                return out

        return fwd

    def _acquire(self, name: str, t: Any, job) -> None:      # FA2 M1 export buffers + sampler ids
        with record_function("verity.acquire"):
            super()._acquire(name, t, job)

    def _begin_step(self) -> None:
        with record_function("verity.step_begin"):
            super()._begin_step()

    def _end_step(self) -> None:
        with record_function("verity.step_end"):
            super()._end_step()

    def _commit_step(self, job) -> None:
        with record_function("verity.gpu_tree"):
            super()._commit_step(job)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="verity_capture/profiles/workload_cov_b0_b1_c256.json")
    ap.add_argument("--checkpoints", default="manifests/checkpoints.json")
    ap.add_argument("--case", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden-so", default=None)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--engine-arg", action="append", default=None)
    ap.add_argument("--kind", default="native_collect_gather", help="candidate committer kind (native_collect_gather | native_collect)")
    ap.add_argument("--arms", default="control,candidate,hooks_only")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wl = va.load_workload(args.workload)
    engine_args = dict(wl.get("engine_args_required") or {})
    engine_args.update(va.parse_engine_args(args.engine_arg))
    eng = va.build_engine(args.checkpoints, max_num_seqs=8, case=va.case_of(wl, args.case), engine_args=engine_args)
    reqs, sampling = wl["requests"], wl["sampling"]
    base = baseline_state(eng)
    for i in range(2):
        r = AP.run_marked(eng, va.throwaway_requests(eng, reqs, seed=100 + i), sampling)
        print(f"[decomp] warm-up {i}: {r['wall_s']:.3f}s", flush=True)
    gcw = AP._GC()
    gc.callbacks.append(gcw)

    class A:
        chunk, hash_threads, arena_mb, profile_hooks = 256, 2, 512, False

    com = MarkedGather(eng, run_id="prof", arena_bytes=512 << 20, wrap=True, gather=(args.kind == "native_collect_gather"))
    ext = com._gpu_ext_load()
    print(f"[decomp] gpu tree ext loaded: {ext}; candidate {com.name}", flush=True)
    if args.hidden_so:
        from verity_capture.experimental.commit_integ.hidden_source import HiddenM1Source

        class ProfiledM1(HiddenM1Source):
            def _wrapper(self):
                inner = super()._wrapper()

                def varlen_fwd(*a, **kw):
                    with record_function("verity.fa2wrap"):
                        return inner(*a, **kw)

                return varlen_fwd

        com.attach_source(ProfiledM1(com, args.hidden_so, ns=Path(args.hidden_so).stem))
    com.register_weights()
    com.install()
    r = AP.run_marked(eng, va.throwaway_requests(eng, reqs, seed=200), sampling)
    rc = com.finalize()
    com.uninstall()
    com.reset()
    assert_pristine(eng, base)
    print(f"[decomp] warm-up candidate: {r['wall_s']:.3f}s root-ready {rc.root_ready_delay_s * 1e3:.1f} ms errors {rc.extra.get('errors')} "
          f"native {rc.extra.get('native_collect')}", flush=True)
    hooks_only = make_committer("hooks_only", eng, A, "prof") if "hooks_only" in args.arms else None
    if hooks_only is not None:
        hooks_only.install()
        AP.run_marked(eng, va.throwaway_requests(eng, reqs, seed=201), sampling)
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
            c = None
            if arm == "candidate":
                com.reset()
                com.install()
                c = com
            elif arm == "hooks_only":
                hooks_only.reset()
                hooks_only.install()
                c = hooks_only
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=False, record_shapes=False, profile_memory=False) as p:
                r = AP.run_marked(eng, reqs, sampling)
                rcx = None
                if c is not None:
                    rcx = c.finalize()
                    c.uninstall()
                torch.cuda.synchronize()
            assert_pristine(eng, base)
            trace = out / f"trace_{arm}_{rep}.json"
            p.export_chrome_trace(str(trace))
            if tokens_ref is None:
                tokens_ref = r["tokens"]
            an = AP.analyze(trace, arm)
            # extra: flush / acquire marker sums per step on the main thread (not in R10's analyze)
            doc = json.loads(trace.read_text())
            ev = [e for e in doc["traceEvents"] if e.get("ph") == "X"]
            steps = sorted([e for e in ev if str(e.get("name", "")).startswith("verity.engine_step#") and e.get("cat") == "user_annotation"], key=lambda e: e["ts"])
            main_tid = steps[0]["tid"] if steps else None
            for k, st in enumerate(steps):
                s0, s1 = st["ts"], st["ts"] + st["dur"]
                fl = [e for e in ev if e.get("cat") == "user_annotation" and e["tid"] == main_tid and s0 <= e["ts"] < s1 and e["name"] == "verity.flush"]
                aq = [e for e in ev if e.get("cat") == "user_annotation" and e["tid"] == main_tid and s0 <= e["ts"] < s1 and e["name"] == "verity.acquire"]
                if k < len(an["steps"]):
                    an["steps"][k]["flush_calls"] = len(fl)
                    an["steps"][k]["flush_us"] = sum(e["dur"] for e in fl)
                    an["steps"][k]["acquire_calls"] = len(aq)
                    an["steps"][k]["acquire_us"] = sum(e["dur"] for e in aq)
            del doc, ev
            row = {"arm": arm, "rep": rep, "wall_s_under_profiler": r["wall_s"], "step_wall_s": r["step_wall_s"], "tokens_equal": r["tokens"] == tokens_ref,
                   "gc": gcw.summary(r["t0"], r["t1"]), "analysis": an}
            if rcx is not None:
                row["commit"] = {"root": rcx.run_root.hex(), "root_ready_ms": rcx.root_ready_delay_s * 1e3, "spans_total_s": rcx.spans["total_s"],
                                 "gpu_tree_ms": rcx.extra.get("gpu_tree_ms"), "native_collect": rcx.extra.get("native_collect"), "errors": rcx.extra.get("errors")}
            results.append(row)
            dec = AP.mean_over(an["steps"], list(range(3, 15)))
            print(f"[decomp] rep {rep} {arm:10s}: wall {r['wall_s'] * 1e3:.1f} ms (under profiler); decode-mean step {dec.get('wall_us', 0):.0f} µs; hooks {dec.get('hooks', 0):.0f} "
                  f"hook_body {dec.get('hook_body_us', 0):.0f} µs (bookkeeping {dec.get('bookkeeping_us', 0):.0f}, runtime-in-hooks {dec.get('runtime_in_hooks_us', 0):.0f}); "
                  f"flush {dec.get('flush_calls', 0):.0f}/{dec.get('flush_us', 0):.0f} µs; acquire {dec.get('acquire_calls', 0):.0f}/{dec.get('acquire_us', 0):.0f} µs; "
                  f"fa2 glue {dec.get('fa2_wrapper_glue_us', 0):.0f}; launches {dec.get('launch_calls', 0):.0f} mean {dec.get('launch_mean_us', 0):.2f} µs; "
                  f"cpu_ops main (outside hooks) {dec.get('cpu_ops_main_us', 0):.0f} µs; worker tree cpu {dec.get('worker_gpu_tree_cpu_us', 0):.0f} µs; "
                  f"GPU {dec.get('gpu_kernel_us', {})} busy {dec.get('gpu_busy_us', 0):.0f} tree∩model {dec.get('gpu_tree_model_overlap_us', 0):.0f} copy∩model {dec.get('gpu_copy_model_overlap_us', 0):.0f}; gc {row['gc']}", flush=True)
            (out / "profile_results.json").write_text(json.dumps({"schema": SCHEMA, "label": "under torch.profiler — diagnostic, not headline", "kind": args.kind,
                                                                    "results": results}, indent=1, default=str) + "\n")
    com.close()
    if hooks_only is not None:
        hooks_only.close()
    # decomposition table: decode-step means (steps 3..14) over reps, candidate vs control
    def dec_mean(arm):
        rows = [AP.mean_over(x["analysis"]["steps"], list(range(3, 15))) for x in results if x["arm"] == arm]
        keys = {k for r_ in rows for k, v in r_.items() if isinstance(v, (int, float))}
        return {k: statistics.fmean(r_.get(k, 0.0) for r_ in rows) for k in keys} | {"gpu_kernel_us": {c: statistics.fmean(r_.get("gpu_kernel_us", {}).get(c, 0.0) for r_ in rows) for c in {c for r_ in rows for c in r_.get("gpu_kernel_us", {})}}}

    ctl, cand = dec_mean("control"), dec_mean("candidate")
    hk = dec_mean("hooks_only") if hooks_only is not None else {}
    delta = cand["wall_us"] - ctl["wall_us"]
    comp = {
        "hook_entry_dispatch (hooks_only arm delta; the candidate wraps forward instead: ablation says ~0.1 us/call)": (hk.get("wall_us", ctl["wall_us"]) - ctl["wall_us"]) if hk else None,
        "hook body: pybind stage/collect + plan check (hook markers minus runtime calls and flush inside)": cand["hook_body_us"] - cand["runtime_in_hooks_us"] - cand.get("flush_us", 0.0),
        "gather flush launches (host: ring write + cudaLaunchKernel)": cand.get("flush_us", 0.0),
        "FA2 export acquire (M1 buffers: cudaMemcpyAsync/stage in _acquire)": cand.get("acquire_us", 0.0),
        "FA2 wrapper glue (fa2wrap minus tapped op)": cand["fa2_wrapper_glue_us"],
        "step begin/end (arena take, plan export, event record, queue)": cand["step_begin_end_us"],
        "model's own aten dispatch slowdown outside hooks (interference)": cand["cpu_ops_main_us"] - ctl["cpu_ops_main_us"],
        "launch API slowdown (same kernel count)": cand["launch_us"] - ctl["launch_us"],
        "GC": None,
    }
    known = sum(v for v in comp.values() if v is not None) - (comp["hook_entry_dispatch (hooks_only arm delta; the candidate wraps forward instead: ablation says ~0.1 us/call)"] or 0.0)
    comp["other / unattributed (delta - sum of the above, excl. hook_entry row)"] = delta - known
    gpu = {"copy/gather kernels us/step (GPU, serialised into the serving stream)": cand["gpu_kernel_us"].get("copy", 0.0) - ctl["gpu_kernel_us"].get("copy", 0.0),
           "tree kernels us/step (GPU, worker side stream)": cand["gpu_kernel_us"].get("tree", 0.0),
           "fa2 kernel delta us/step (tapped kernel vs installed)": cand["gpu_kernel_us"].get("fa2", 0.0) - ctl["gpu_kernel_us"].get("fa2", 0.0),
           "model kernels us/step control": ctl["gpu_kernel_us"].get("model", 0.0), "gpu busy us/step control": ctl["gpu_busy_us"], "gpu busy us/step candidate": cand["gpu_busy_us"],
           "tree∩model overlap us": cand["gpu_tree_model_overlap_us"], "copy∩model overlap us": cand["gpu_copy_model_overlap_us"]}
    table = {"schema": SCHEMA, "label": "under profiler (no stacks); decode-step means steps 3..14 over reps; us per step", "kind": args.kind,
             "control_step_us": ctl["wall_us"], "candidate_step_us": cand["wall_us"], "delta_us": delta, "delta_pct_under_profiler": delta / ctl["wall_us"] * 100,
             "components_us": comp, "components_share_of_delta": {k: (v / delta if v is not None else None) for k, v in comp.items()}, "gpu": gpu,
             "counts": {"hooks": cand["hooks"], "flushes": cand.get("flush_calls", 0), "acquires": cand.get("acquire_calls", 0), "fa2_wrapper_calls": cand["fa2_wrapper_calls"],
                        "launches_control": ctl["launch_calls"], "launches_candidate": cand["launch_calls"], "gpu_kernels_control": ctl["gpu_kernels"], "gpu_kernels_candidate": cand["gpu_kernels"]}}
    (out / "decomposition.json").write_text(json.dumps(table, indent=1, default=str) + "\n")
    print(json.dumps(table, indent=1, default=str), flush=True)
    print(f"[decomp] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
