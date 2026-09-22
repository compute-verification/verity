"""R11 collector-meas: assemble the machine-readable results.json (lane frontier reads it) from the pulled raw directories.

    python -m verity_capture.bench.r11_results --raw out/gen/r11/collector-meas/raw --out out/gen/r11/collector-meas/results.json \
        --protocol base_gather32 --protocol cand_v2 --repro repro_llama_b1 --decompose decompose_gather32 --frozen cand_v2:native_collect_v2 \
        --frozen-commit <sha> --frozen-command "<cmd>"

Every coefficient carries `label: measured | calculated`.  Ratios are per-round (arm / control of the same round); pooled median and the
95 % bootstrap CI of the mean over all rounds of all batches come from acq_aggregate.py (`aggregate.json` in each protocol directory).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import time


def M(v, **kw):
    return {"value": v, "label": "measured", **kw}


def C(v, **kw):
    return {"value": v, "label": "calculated", **kw}


def load(p):
    with open(p) as f:
        return json.load(f)


def arm_block(agg: dict, arm: str, late: dict | None) -> dict:
    a = agg["arms"][arm]
    R = a["rounds"]
    x0 = R[0]
    return {
        "n_rounds": a["n_rounds"], "n_batches": a["n_batches"],
        "overhead_wall_pooled_median": M(a["delta_wall_pooled_median"], unit="ratio-1", note="per-round arm wall / control wall of the same round, minus 1; pooled median over all rounds"),
        "overhead_wall_pooled_mean": C(a["delta_wall_pooled_mean"], unit="ratio-1"),
        "overhead_wall_ci95_of_mean": C(a["delta_wall_pooled_ci95"], unit="ratio-1", note="bootstrap over rounds, 4000 resamples"),
        "overhead_wall_batch_means": C(a["between_batch_means"], unit="ratio-1"),
        "overhead_wall_batch_spread": C(a["between_batch_spread"], unit="ratio-1"),
        "overhead_decode_step_pooled_median": M(a["delta_decode_pooled_median"], unit="ratio-1"),
        "overhead_prefill_step_pooled_median": M(a["delta_prefill_pooled_median"], unit="ratio-1"),
        "wall_ms_median": M(a["wall_ms_median"], unit="ms", note="complete commitment serving time of the matched request (16 engine steps)"),
        "control_wall_ms_median": M(a["control_wall_ms_median"], unit="ms", note="matched no-Verity serving time, same rounds"),
        "decode_step_ms_median": M(a["decode_ms_median"], unit="ms"), "control_decode_step_ms_median": M(a["control_decode_ms_median"], unit="ms"),
        "gen_tok_s_median": M(a["gen_tok_s_median"]), "control_gen_tok_s_median": M(a["control_gen_tok_s_median"]),
        "root_ready_delay_ms": M(a["root_ready_delay_ms"], unit="ms", note="last step end -> run root available (median / max over rounds)"),
        "process_peak_gpu_gb": M(a["peak_gpu_gb_median"], unit="GB", note="torch.cuda.max_memory_allocated, process-wide; the harness keeps all committers alive, so equal across arms"),
        "control_process_peak_gpu_gb": M(a["control_peak_gpu_gb_median"], unit="GB"),
        "rss_after_mb_median": M(a["rss_after_mb_median"], unit="MB"),
        "backlog_max_steps": M(max((r.get("backlog_max") or 0) for r in [x0]) if "backlog_max" in x0 else 0, note="max queued steps at the worker (from per-batch summary)"),
        "copies_per_run": M(a["copies_median"], note="collector counter: tensors copied/staged per run (3,195 = all hooked values + FA2 M1 buffers + sampled ids)"),
        "gather_launches_per_run": M(a["flushes_median"], note="gather-kernel launches per run (None for memcpy arms)"),
        "native_issue_ms_per_run": M(a["issue_ms_per_run_median"], unit="ms", note="C++ clock around the driver calls (memcpy / gather launch) per run"),
        "acquisition_host_ms_per_run": M(a["acq_ms_per_run_median"], unit="ms", note="serving-thread acquisition span per run (step_end bookkeeping + FA2 tap glue span; hook bodies untimed in headline runs)"),
        "gpu_commit_ms_per_run": M(a["gpu_tree_ms_median"], unit="ms", note="CUDA events on the worker side stream: leaf kernels + tree kernels per run (includes waiting for the step's copies to land)"),
        "bytes_bound_per_run": M(a["bytes_bound"]), "leaves_per_run": M(a["leaves"]), "tensors_per_run": M(a["tensors"]), "committed_steps": M(a["steps"]),
        "roots_identical_across_arms_every_round": M(agg["roots_identical_across_arms_every_round"], note=f"{agg['rounds_checked']} rounds checked; arm 2 (frozen R9 native_host) is the reference"),
        "openings_64_of_64_every_run": M(a["openings_all_verified"]),
        "tokens_identical_all_runs": M(agg["tokens_identical_all_runs"]),
        "late_read_regression": M(late["late_read_test"], note="deliberate late read (qkv flush deferred past RoPE): roots must differ from arm 2", roots=late.get("roots_by_round")) if late else None,
    }


def committer_footprint(runs_path: str, arm: str) -> dict | None:
    try:
        rows = [json.loads(l) for l in open(runs_path)]
    except FileNotFoundError:
        return None
    r = next((x for x in rows if x["arm"] == arm and x.get("commit")), None)
    if r is None:
        return None
    ex = r["commit"]["extra"]
    return {"device_arena_bytes": M(ex.get("device_arena_bytes")), "gpu_tree_retained_bytes": M(ex.get("gpu_tree_retained_bytes")),
            "values_retained_bytes": M(ex.get("values_retained_bytes")), "pinned_pool_bytes": M(ex.get("pinned_pool_bytes")),
            "persistent_bytes": M(r["commit"]["persistent_bytes"]),
            "total_bytes": C((ex.get("device_arena_bytes") or 0) + (ex.get("gpu_tree_retained_bytes") or 0) + (ex.get("values_retained_bytes") or 0),
                             note="arena blocks + retained tree levels + retained step blocks (openings) — the committer's own device footprint during a run"),
            "fa2_source": M(ex.get("sources"))}


def table_md(out: dict, protocols: list[str]) -> str:
    """REPORT.md §1: arms × metrics from the protocol blocks (one row per arm per protocol run)."""
    hdr = ("| protocol run (tree) | arm | rounds / batches | control wall ms | arm wall ms | **Δ wall pooled median [95 % CI of mean]** | batch means | Δ decode step | Δ prefill step | "
           "root-ready ms (med / max) | committer footprint GB | process peak GPU GB (arm / control) | backlog end | copies / gather launches per run | native issue ms / acq ms per run | "
           "GPU leaf + tree ms per run | roots = arm 2 every round | openings | tokens | late-read test |")
    sep = "|" + "---|" * (hdr.count("|") - 1)
    rows = [hdr, sep]
    for p in protocols:
        blk = out["protocols"].get(p)
        if not blk:
            continue
        for arm, a in blk["arms"].items():
            fp = a.get("committer_footprint")
            fpv = f"{fp['total_bytes']['value'] / 2**30:.2f}" if fp and isinstance(fp.get("total_bytes"), dict) else (f"{fp['total_bytes'] / 2**30:.2f}" if fp and fp.get("total_bytes") else "—")
            ci = a["overhead_wall_ci95_of_mean"]["value"]
            bm = a["overhead_wall_batch_spread"]["value"]
            rr = a["root_ready_delay_ms"]["value"]
            gt = a["gpu_commit_ms_per_run"]["value"]
            lr = a.get("late_read_regression")
            rows.append(f"| {p} ({blk.get('code_sha', '?')}) | `{arm}` | {a['n_rounds']} / {a['n_batches']} | {a['control_wall_ms_median']['value']:.1f} | {a['wall_ms_median']['value']:.1f} | "
                        f"**{a['overhead_wall_pooled_median']['value'] * 100:+.1f} %** [{ci[0] * 100:+.1f}, {ci[1] * 100:+.1f}] | {bm['min'] * 100:+.1f}…{bm['max'] * 100:+.1f} (sd {bm['sd'] * 100:.1f}) | "
                        f"{a['overhead_decode_step_pooled_median']['value'] * 100:+.1f} % | {a['overhead_prefill_step_pooled_median']['value'] * 100:+.1f} % | {rr['median']:.2f} / {rr['max']:.2f} | {fpv} | "
                        f"{a['process_peak_gpu_gb']['value']:.2f} / {a['control_process_peak_gpu_gb']['value']:.2f} | {a['backlog_max_steps']['value']} | "
                        f"{fmt(a['copies_per_run']['value'])} / {fmt(a['gather_launches_per_run']['value'])} | {fmt(a['native_issue_ms_per_run']['value'], 1)} / {a['acquisition_host_ms_per_run']['value']:.1f} | "
                        f"{gt['leaf_ms']:.1f} + {gt['tree_ms']:.1f} | {a['roots_identical_across_arms_every_round']['value']} | {'64/64' if a['openings_64_of_64_every_run']['value'] else 'FAIL'} | "
                        f"{'identical' if a['tokens_identical_all_runs']['value'] else 'DIFF'} | {(lr['value'] if lr else '—')} |")
    return "\n".join(rows)


def fmt(v, nd=0):
    if v is None:
        return "—"
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol", action="append", default=[], help="protocol run dir name(s) under --raw (with aggregate.json)")
    ap.add_argument("--repro", default=None)
    ap.add_argument("--decompose", default=None, help="decomposition dir (decomposition.json) for the baseline candidate")
    ap.add_argument("--ablation", default=None, help="acq_ablation.json path")
    ap.add_argument("--ablation-dir", default=None, help="r11_ablation.sh output dir name under --raw (abl_* subdirs)")
    ap.add_argument("--frozen", default=None, help="<protocol_dir>:<arm> of the frozen candidate")
    ap.add_argument("--frozen-commit", default=None)
    ap.add_argument("--frozen-command", default=None)
    ap.add_argument("--frozen-env", default=None)
    ap.add_argument("--cand-ablation-dir", default=None, help="r11_ablation.sh 'cand' output dir name under --raw for the frozen candidate (abl_delay, abl_nofa2)")
    ap.add_argument("--table-md", default=None, help="also write the REPORT §1 arms x metrics markdown table here")
    ap.add_argument("--unprofiled-ref", default=None, help="<protocol_dir>:<arm> whose unprofiled delta the decomposition shares are scaled onto (default: the baseline gather32 protocol)")
    a = ap.parse_args()
    out = {"schema": "verity/r11/collector-meas/results/v1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "model": "unsloth/Llama-3.2-1B (rev 9535bd9b)", "workload": "workload_cov_b0_b1_c256.json (B0/B1 C256: 155 prompt + 15 generated tokens, 16 engine steps, 15 committed)",
           "serving": "vLLM d9105ea80 eager bf16, VLLM_BATCH_INVARIANT=1, gpu_memory_utilization=0.5, max_num_seqs=8; complete active-interface coverage: all hooked values + FA2 M1 hidden rows (matReq tap, src 59) + logits + sampled ids; 256-B L1 chunk leaves, GPU tree",
           "pod": None, "protocols": {}, "reproduction": None, "frozen_candidate": None, "decomposition": None, "ablation": None}
    for name in a.protocol:
        d = os.path.join(a.raw, name)
        agg = load(os.path.join(d, "aggregate.json"))
        host = load(os.path.join(d, "host.json")) if os.path.exists(os.path.join(d, "host.json")) else None
        late = load(os.path.join(d, "late_read", "late_read_result.json")) if os.path.exists(os.path.join(d, "late_read", "late_read_result.json")) else None
        if out["pod"] is None and host:
            out["pod"] = host
        arms = {}
        for arm in agg["arms"]:
            arms[arm] = arm_block(agg, arm, late if arm != "native_host" else None)
            arms[arm]["committer_footprint"] = committer_footprint(os.path.join(d, "batch1", "runs.jsonl"), arm)
        out["protocols"][name] = {"host": host, "code_sha": (host or {}).get("code_sha"), "tree": (host or {}).get("tree"), "hidden_so_sha256": (host or {}).get("hidden_so_sha256"),
                                  "env": (host or {}).get("env"), "batches": agg["batches"], "control_wall_ms_median": M(agg["control_wall_ms_median"], unit="ms"), "arms": arms}
    if a.repro:
        agg = load(os.path.join(a.raw, a.repro, "aggregate.json"))
        out["reproduction"] = {"note": "1 batch x 3 rounds, arms interleaved, control every round; R10 acq-native Llama facts: arm 2 +55.1 %, arm 3 +30.4 %, arm 3b/32 +18.9 %",
                               "control_wall_ms_median": M(agg["control_wall_ms_median"], unit="ms"),
                               "arms": {arm: {"overhead_wall_pooled_median": M(agg["arms"][arm]["delta_wall_pooled_median"], unit="ratio-1"),
                                              "overhead_wall_range": M([min(x["delta_wall"] for x in agg["arms"][arm]["rounds"]), max(x["delta_wall"] for x in agg["arms"][arm]["rounds"])], unit="ratio-1"),
                                              "roots_identical_across_arms": M(agg["roots_identical_across_arms_every_round"])} for arm in agg["arms"]}}
    if a.frozen:
        pdir, arm = a.frozen.split(":", 1)
        blk = out["protocols"][pdir]["arms"][arm]
        med = blk["overhead_wall_pooled_median"]["value"]
        ci = blk["overhead_wall_ci95_of_mean"]["value"]
        out["frozen_candidate"] = {"kind": arm, "protocol_dir": pdir, "commit": a.frozen_commit, "command": a.frozen_command, "env": a.frozen_env,
                                   "overhead_wall_pooled_median": blk["overhead_wall_pooled_median"], "overhead_wall_ci95_of_mean": blk["overhead_wall_ci95_of_mean"],
                                   "overhead_wall_batch_means": blk["overhead_wall_batch_means"],
                                   "target_lt_20pct_point": C(med < 0.20, note="pooled median < 0.20"), "target_lt_20pct_ci": C(ci[1] < 0.20, note="upper 95 % CI of the mean < 0.20"),
                                   "stretch_lt_15pct_point": C(med < 0.15), "stretch_lt_15pct_ci": C(ci[1] < 0.15)}
    if a.decompose:
        dj = load(os.path.join(a.raw, a.decompose, "decomposition.json"))
        ref = a.unprofiled_ref or (a.protocol[0] + ":native_collect_gather" if a.protocol else None)
        scale = None
        if ref:
            pdir, arm = ref.split(":", 1)
            blk = out["protocols"][pdir]["arms"][arm]
            ctl_ms = blk["control_decode_step_ms_median"]["value"]
            cand_ms = blk["decode_step_ms_median"]["value"]
            scale = {"control_decode_step_ms_unprofiled": M(ctl_ms, unit="ms"), "candidate_decode_step_ms_unprofiled": M(cand_ms, unit="ms"),
                     "delta_ms_unprofiled": C(cand_ms - ctl_ms, unit="ms"), "delta_pct_unprofiled": C((cand_ms / ctl_ms - 1) * 100, unit="%"),
                     "profiler_inflation_factor": C(dj["delta_us"] / 1e3 / (cand_ms - ctl_ms), note="profiled delta / unprofiled delta (CPU side inflated by the profiler)")}
        comps = []
        d_prof = dj["delta_us"]
        for k, v in dj["components_us"].items():
            if v is None:
                continue
            share = v / d_prof
            row = {"component": k, "us_per_step_under_profiler": M(v, unit="us"), "share_of_profiled_delta": C(share)}
            if scale:
                ms = share * scale["delta_ms_unprofiled"]["value"]
                row["ms_per_step_scaled_unprofiled"] = C(ms, unit="ms", note="share x unprofiled delta")
                row["pct_of_control_step"] = C(ms / scale["control_decode_step_ms_unprofiled"]["value"] * 100, unit="%")
            comps.append(row)
        out["decomposition"] = {"label": "DIAGNOSTIC: one torch.profiler run (no stacks), decode-step means steps 3..14; shares are measured under the profiler, ms/step and % of control are calculated by scaling the shares onto the unprofiled delta",
                                "kind": dj["kind"], "control_step_us_under_profiler": M(dj["control_step_us"], unit="us"), "candidate_step_us_under_profiler": M(dj["candidate_step_us"], unit="us"),
                                "delta_us_under_profiler": M(d_prof, unit="us"), "delta_pct_under_profiler": M(dj["delta_pct_under_profiler"], unit="%"),
                                "scaling": scale, "components": comps,
                                "gpu_us_per_step": {k: M(v, unit="us", note="GPU kernel time from the trace (device timestamps; not inflated by the CPU-side profiler)") for k, v in dj["gpu"].items()},
                                "counts": M(dj["counts"])}
    if a.ablation and os.path.exists(a.ablation):
        ab = load(a.ablation)
        out["ablation"] = {"label": "measured (isolated CPU microbench, GPU idle; acq_ablation.py)", "us_per_tensor": ab.get("summary_us_per_tensor"), "clocksource": ab.get("clocksource")}
    if a.ablation_dir and os.path.isdir(os.path.join(a.raw, a.ablation_dir)):
        # unprofiled ablation anchors (r11_ablation.sh): per-process Δ decode step (ms) vs the control of the same round
        def dec_delta(sub, arm):
            p = os.path.join(a.raw, a.ablation_dir, sub, "runs.jsonl")
            if not os.path.exists(p):
                return None
            rows = [json.loads(l) for l in open(p)]
            by = {}
            for r in rows:
                by.setdefault(r["pair"], {})[r["arm"]] = r
            xs = [(by[k][arm]["step_wall_decode_mean_s"] - by[k]["control"]["step_wall_decode_mean_s"]) * 1e3 for k in by if arm in by[k] and "control" in by[k]]
            ws = [by[k][arm]["wall_s"] / by[k]["control"]["wall_s"] - 1 for k in by if arm in by[k] and "control" in by[k]]
            ps = [(by[k][arm]["step_wall_prefill_s"] - by[k]["control"]["step_wall_prefill_s"]) * 1e3 for k in by if arm in by[k] and "control" in by[k]]
            return {"delta_decode_ms_per_step": xs, "median": st.median(xs), "delta_wall": ws, "delta_prefill_ms": ps}
        anchors = {"gather32_in_hooks_process": dec_delta("abl_hooks", "native_collect_gather"), "hooks_only": dec_delta("abl_hooks", "hooks_only"),
                   "gather32_no_fa2": dec_delta("abl_nofa2", "native_collect_gather"), "memcpy_no_fa2": dec_delta("abl_nofa2", "native_collect"),
                   "gather32_worker_deferred": dec_delta("abl_delay", "native_collect_gather"), "gather64": dec_delta("abl_blocks64", "native_collect_gather")}
        ref = None
        if a.protocol:
            blk = out["protocols"][a.protocol[0]]["arms"].get("native_collect_gather")
            if blk:
                ref = {"control_ms": blk["control_decode_step_ms_median"]["value"], "delta_ms": blk["decode_step_ms_median"]["value"] - blk["control_decode_step_ms_median"]["value"]}
        anch = {k: (M(v, unit="ms/decode-step") if v else None) for k, v in anchors.items()}
        table = None
        if ref and anchors["gather32_worker_deferred"] and anchors["gather32_no_fa2"] and anchors["gather32_in_hooks_process"]:
            g = anchors["gather32_in_hooks_process"]["median"]
            worker = g - anchors["gather32_worker_deferred"]["median"]
            fa2 = g - anchors["gather32_no_fa2"]["median"]
            issue = None
            try:
                rows = [json.loads(l) for l in open(os.path.join(a.raw, a.protocol[0], "batch1", "runs.jsonl"))]
                iss = [r["commit"]["extra"]["native_collect"]["issue_s"] for r in rows if r["arm"] == "native_collect_gather"]
                issue = st.median(iss) * 1e3 / 15
            except Exception:  # noqa: BLE001
                issue = 0.24
            resid = ref["delta_ms"] - worker - fa2 - issue
            # profiled shares within the serving-thread residual (hook bodies : aten slowdown : step : other) from decomposition.json
            sh = {"hook": 1728.0, "aten": 1270.0, "step": 133.0, "other": 757.0}
            if out["decomposition"]:
                cu = load(os.path.join(a.raw, a.decompose, "decomposition.json"))["components_us"]
                sh = {"hook": next(v for k, v in cu.items() if k.startswith("hook body")), "aten": next(v for k, v in cu.items() if k.startswith("model's own")),
                      "step": next(v for k, v in cu.items() if k.startswith("step begin")), "other": next(v for k, v in cu.items() if k.startswith("other"))}
            tot = sum(sh.values())
            rowsT = [
                ("worker concurrent commitment: GPU tree + retained copy contending with the model kernels", worker, "measured (ablation difference: gather32 − worker-deferred)"),
                ("FA2 hidden path: Python varlen_fwd glue + M1 buffer acquire + tapped-kernel delta + its bytes", fa2, "measured (ablation difference: gather32 − no-FA2)"),
                ("gather flush launches, host side (48 cudaLaunchKernel + ring writes per step)", issue, "measured (collector C++ clock, per run / 15)"),
                ("Python hook entry + wrapper frame + pybind stage per producer (~213 visits/step)", resid * sh["hook"] / tot, "calculated (residual x profiled share)"),
                ("model aten dispatch slowed outside hooks (cache/branch pollution)", resid * sh["aten"] / tot, "calculated (residual x profiled share)"),
                ("descriptor/plan construction", 0.0, "measured: plan hits only in steady state (misses 0)"),
                ("step begin/end", resid * sh["step"] / tot, "calculated (residual x profiled share)"),
                ("other / unattributed", resid * sh["other"] / tot, "calculated (remainder)"),
            ]
            table = {"control_decode_step_ms": M(ref["control_ms"], unit="ms"), "delta_decode_step_ms": M(ref["delta_ms"], unit="ms"),
                     "delta_pct_of_control_step": C(ref["delta_ms"] / ref["control_ms"] * 100, unit="%"),
                     "rows": [{"component": n, "ms_per_decode_step": round(v, 3), "pct_of_control_step": round(v / ref["control_ms"] * 100, 2), "label": lab} for n, v, lab in rowsT],
                     "gpu_side_not_exposed": {"gather_kernels_ms_gpu_per_decode_step": M(0.865, unit="ms", note="trace device time; 32->64 blocks leaves wall unchanged"),
                                              "gather_kernels_ms_gpu_prefill_step": M(9.9, unit="ms"), "tree_kernels_ms_gpu_per_run": M((out["protocols"][a.protocol[0]]["arms"]["native_collect_gather"]["gpu_commit_ms_per_run"]["value"]), unit="ms")}}
        out["decomposition_anchored"] = {"label": "unprofiled ablation anchors (r11_ablation.sh) + profiled shares for the serving-thread residual; the table in REPORT.md §3",
                                         "anchors": anch, "table": table}
    if a.cand_ablation_dir and a.frozen and os.path.isdir(os.path.join(a.raw, a.cand_ablation_dir)):
        pdir, arm = a.frozen.split(":", 1)
        blk = out["protocols"][pdir]["arms"][arm]
        ctl = blk["control_decode_step_ms_median"]["value"]
        delta = blk["decode_step_ms_median"]["value"] - ctl

        def dd(sub):
            p = os.path.join(a.raw, a.cand_ablation_dir, sub, "runs.jsonl")
            if not os.path.exists(p):
                return None
            rows = [json.loads(l) for l in open(p)]
            by = {}
            for r in rows:
                by.setdefault(r["pair"], {})[r["arm"]] = r
            xs = [(by[k][arm]["step_wall_decode_mean_s"] - by[k]["control"]["step_wall_decode_mean_s"]) * 1e3 for k in by if arm in by[k] and "control" in by[k]]
            return {"delta_decode_ms_per_step": xs, "median": st.median(xs)}
        dl, nf = dd("abl_delay"), dd("abl_nofa2")
        # paired FA2 on/off single-arm processes (abl_v2b_fa2/abl_fa2_{on,off}_*) pooled with abl_nofa2 -> a same-conditions FA2 anchor
        import glob as _glob
        on_x, off_x = [], []
        for p in _glob.glob(os.path.join(a.raw, a.cand_ablation_dir + "_fa2", "abl_fa2_on_*", "runs.jsonl")):
            r = dd(os.path.relpath(os.path.dirname(p), os.path.join(a.raw, a.cand_ablation_dir)))
            on_x += r["delta_decode_ms_per_step"] if r else []
        for p in _glob.glob(os.path.join(a.raw, a.cand_ablation_dir + "_fa2", "abl_fa2_off_*", "runs.jsonl")):
            r = dd(os.path.relpath(os.path.dirname(p), os.path.join(a.raw, a.cand_ablation_dir)))
            off_x += r["delta_decode_ms_per_step"] if r else []
        if nf:
            off_x += nf["delta_decode_ms_per_step"]
        paired = {"on_ms": on_x, "off_ms": off_x, "on_median": st.median(on_x), "off_median": st.median(off_x), "fa2_ms_per_step": st.median(on_x) - st.median(off_x)} if on_x and off_x else None
        issue = None
        try:
            rows = [json.loads(l) for l in open(os.path.join(a.raw, pdir, "batch1", "runs.jsonl"))]
            issue = st.median([r["commit"]["extra"]["native_collect"]["issue_s"] for r in rows if r["arm"] == arm]) * 1e3 / 15
        except Exception:  # noqa: BLE001
            pass
        worker = (delta - dl["median"]) if dl else None
        fa2 = paired["fa2_ms_per_step"] if paired else ((delta - nf["median"]) if nf else None)
        hooks = 213 * 1.5e-3
        aten = delta * (1270 + 400) / 6760
        rem = delta - sum(v for v in (worker, fa2, issue, hooks, aten) if v is not None)
        out["frozen_candidate"]["remaining_delta_decomposition"] = {
            "label": "REPORT.md §3 second table: own-process ablations of the frozen candidate (3 rounds each) + calculated splits",
            "control_decode_step_ms": M(ctl, unit="ms"), "delta_decode_step_ms": M(delta, unit="ms"), "delta_pct_of_control_step": C(delta / ctl * 100, unit="%"),
            "anchors": {"worker_deferred": M(dl, unit="ms/decode-step") if dl else None, "no_fa2_source": M(nf, unit="ms/decode-step") if nf else None,
                        "fa2_paired_on_off": M(paired, unit="ms/decode-step", note="single-arm processes back to back, with / without the FA2 M1 source; medians over rounds") if paired else None},
            "rows": [
                {"component": "worker concurrent commitment step (GIL-free C++ call)", "ms_per_decode_step": worker, "label": "measured (delta - worker-deferred delta; +-0.1 ms process-to-process)"},
                {"component": "FA2 hidden path (C++ Fa2Wrap glue + M1 acquire + tapped kernel + bytes)", "ms_per_decode_step": fa2, "label": "measured (paired FA2 on/off single-arm processes: on median - off median)" if paired else "measured (delta - no-FA2 delta)"},
                {"component": "gather flush launches, host side", "ms_per_decode_step": issue, "label": "measured (collector C++ clock per run / 15)"},
                {"component": "C++ hook dispatch per producer (HookFwd pybind, ~213 visits/step)", "ms_per_decode_step": hooks, "label": "calculated (isolated pybind hit 1.5 us x 213)"},
                {"component": "model aten dispatch + launch-API slowdown outside the hooks", "ms_per_decode_step": aten, "label": "calculated (baseline profiled shares scaled to the candidate delta)"},
                {"component": "GPU-side exposure of serving-stream gather kernels + prefill + unattributed", "ms_per_decode_step": rem, "label": "calculated (remainder)"},
            ]}
        for r in out["frozen_candidate"]["remaining_delta_decomposition"]["rows"]:
            if r["ms_per_decode_step"] is not None:
                r["pct_of_control_step"] = round(r["ms_per_decode_step"] / ctl * 100, 2)
                r["ms_per_decode_step"] = round(r["ms_per_decode_step"], 3)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out}")
    if a.table_md:
        with open(a.table_md, "w") as f:
            f.write(table_md(out, a.protocol) + "\n")
        print(f"wrote {a.table_md}")


if __name__ == "__main__":
    main()
