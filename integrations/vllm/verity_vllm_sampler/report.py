"""Assemble W3's raw results into the contract §9 layout using W2's row specs (`verity_vllm.rows`).

    PYTHONPATH=<harness>/vllm-poc:<sampler>/vllm-poc python -m verity_vllm_sampler.report \
        --raw fixtures/results/w3-sampler-nightly --out-root fixtures/results \
        --wheel-sha <sha256 of the vllm wheel> [--adapter-revision <git sha>] [--pod-hours <h>]

`--raw` holds prod/*.json, analysis/*.json, negative/*.json + negative/n08_*/n08.json,
reproducibility.json (compare.py), validate_rng.json, checkpoint_qwen2.5-1.5b.json.

Writes fixtures/results/<experiment_id>/{table1,table3,table4}.jsonl, manifest.json, log.txt and
w3_detail.json (everything the §9 fields cannot carry: per-variant/seed counters, the §15.3 split
"trajectory followed" vs "unforced sampler agrees", certification, batching sweep, stopping relations).

Two Table 1 rows are produced, both with `result`/`first_blocker` derived by W2's §10 rule:
  * R5 (token selection + stopping): token_selection = independent re-derivation over every generated
    position of the analysed runs; relation_checks.independent = the certified subset (what the registry
    checker would report exact), relation_checks.same_runtime = vLLM's kernels on the same rows;
    boundary_noninterference = unforced replay sampler token vs production token; negative = N08/N09.
  * R1 production side (batching): boundary_noninterference = stock production outputs under caps
    1/8/32 and two request orderings vs cap 64; token_selection = the cap-64 record replayed under a
    different batch composition (cap 8) when that analysis exists; no negatives -> PARTIAL at best.
Commitments, authentication and coverage are W1's and are NOT_RUN here (zero attempted -> PARTIAL).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import time
from typing import Any

from verity_vllm import rows as R

from . import stopping as ST

POD_USD_PER_HOUR = 0.74  # PLAN.md: vllm-4090b


def jload(p: str) -> Any:
    with open(p) as f:
        return json.load(f)


def build_profile(prod: dict[str, Any], ckpt: dict[str, Any], wheel_sha: str, adapter_rev: str,
                  seeds: list[int], lock: dict[str, str]) -> dict[str, Any]:
    p = prod["profile"]
    return {
        "schema": "verity-vllm/profile/v1",
        "model": {"repo": ckpt["repo"], "revision": ckpt["revision"], "weights_sha256": ckpt["safetensors_sha256"]},
        "vllm": {"version": p["vllm"], "wheel_sha256": wheel_sha, "commit": p.get("vllm_commit")},
        "python": {"version": p["python"], "torch": p["torch"], "cuda": p["cuda"], "triton": lock.get("triton"),
                   "flashinfer": lock.get("flashinfer-python") or lock.get("flashinfer")},
        "gpu": {"name": p["gpu"], "count": 1, "driver": "580.126.20", "compute_capability": p["compute_capability"],
                "pod": "vllm-4090b upgk7qho31vo2q"},
        "dtype": "bfloat16", "quantization": None,
        "backends": {"attention": "vllm default under VLLM_BATCH_INVARIANT=1 (batch-invariant path)",
                     "linear": "batch_invariant Triton persistent matmul", "moe": None,
                     "sampler": "v2 gumbel_sample (Triton Philox4x32-10) + apply_top_k_top_p"},
        "parallelism": {"tensor_parallel_size": 1, "pipeline_parallel_size": 1},
        "compilation": {"mode": p["compilation_mode"], "cudagraph_mode": p["cudagraph_mode"], "enforce_eager": p["enforce_eager"],
                        "model_runner": "v2" if p["use_v2_model_runner"] else "v1"},
        "engine_flags": {"VLLM_BATCH_INVARIANT": "1", "async_scheduling": p.get("async_scheduling"),
                         "enable_chunked_prefill": p.get("enable_chunked_prefill"), "enable_prefix_caching": p.get("enable_prefix_caching"),
                         "max_model_len": p["max_model_len"], "kv_cache_memory_bytes": prod["engine_kwargs"].get("kv_cache_memory_bytes"),
                         "max_num_batched_tokens": p.get("max_num_batched_tokens")},
        "sampling": {"greedy": True, "temperature": 0.0, "top_p": 1.0, "top_k": -1, "seeds": seeds,
                     "stochastic_variants": {"temp": {"temperature": 1.0}, "topk_topp": {"temperature": 1.0, "top_p": 0.9, "top_k": 50}}},
        "adapter": {"revision": adapter_rev, "interception": "module-hooks (W3 Sampler.__call__ wrap for logits rows; no commitments)"},
    }


def profile_id(profile: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()[:12]


def lock_versions(path: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if path and os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if "==" in line:
                k, v = line.split("==", 1)
                out[k.lower()] = v
            elif " @ " in line:
                out[line.split(" @ ", 1)[0].lower()] = line.split(" @ ", 1)[1]
    return out


def stopping_relations(prods: dict[str, dict[str, Any]], eos: int) -> dict[str, Any]:
    """Run stopping.py's record relations over every production record; returns pass/fail counts."""
    out = {"output_presence": [0, 0], "finish_reason": [0, 0], "no_early_eos": [0, 0], "failures": []}
    for name, prod in sorted(prods.items()):
        mml = prod["engine_kwargs"].get("max_model_len")
        # the recorder stores the run-wide extra SamplingParams fields (stop_token_ids for the stop-probe run) once
        # at run level; the relations read them from the record, so give each record its run's extras
        run_extra = prod.get("sampling_extra") or {}
        for rec0 in prod["records"]:
            rec = {**rec0, "sampling_extra": {**run_extra, **(rec0.get("sampling_extra") or {})}} if run_extra else rec0
            for key, fn in (("output_presence", lambda r: ST.check_output_presence(r)),
                            ("finish_reason", lambda r: ST.check_finish_reason(r, eos, mml)),
                            ("no_early_eos", lambda r: ST.check_no_early_eos(r, eos))):
                ok, problems = fn(rec)
                out[key][1] += 1
                out[key][0] += bool(ok)
                if not ok:
                    out["failures"].append({"run": name, "index": rec["index"], "relation": key, "problems": problems})
    finish = {}
    for name, prod in prods.items():
        for rec in prod["records"]:
            fr = rec.get("client_finish_reason") or rec["finish_reason"]
            finish[fr] = finish.get(fr, 0) + 1
    out["finish_reasons_seen"] = finish
    out["all_pass"] = not out["failures"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--wheel-sha", required=True)
    ap.add_argument("--lock", default=None, help="requirements lock for triton/flashinfer versions")
    ap.add_argument("--adapter-revision", default=None)
    ap.add_argument("--pod-hours", type=float, default=None, help="GPU hours billed to W3's pod so far")
    ap.add_argument("--stamp", default=None)
    ap.add_argument("--primary-suffix", default="_fresh_auto",
                    help="analyses whose name ends with this are the R5 token-selection evidence (default: the "
                         "fresh-prefill records, every request num_cached_tokens == 0 on both sides); other `_auto` "
                         "analyses are reported per run but not aggregated")
    ap.add_argument("--negatives-filter", default="fresh", help="substring selecting the Table 3 negatives (others listed in detail)")
    args = ap.parse_args()
    raw = args.raw
    adapter_rev = args.adapter_revision or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

    prods = {os.path.basename(p)[:-5]: jload(p) for p in glob.glob(os.path.join(raw, "prod", "*.json"))}
    ref = prods.get("greedy_c64_fresh") or prods["greedy_c64_r0"]
    primary_suffix = args.primary_suffix
    ckpt = jload(os.path.join(raw, "checkpoint_qwen2.5-1.5b.json"))
    repro = jload(os.path.join(raw, "reproducibility.json"))
    rng = jload(os.path.join(raw, "validate_rng.json")) if os.path.exists(os.path.join(raw, "validate_rng.json")) else None
    analyses = {os.path.basename(p)[:-5]: jload(p) for p in glob.glob(os.path.join(raw, "analysis", "select_*.json"))}
    # compare_rows.py outputs (bit-exact row comparison of two replay dumps of one trajectory)
    row_cmps = {os.path.basename(p)[5:-5]: jload(p) for p in glob.glob(os.path.join(raw, "analysis", "rows_*.json"))}
    negatives: dict[str, dict[str, Any]] = {os.path.basename(p)[:-5]: jload(p) for p in glob.glob(os.path.join(raw, "negative", "*.json"))}
    for d in glob.glob(os.path.join(raw, "negative", "n08_*")):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "n08.json")):
            negatives[os.path.basename(d)] = jload(os.path.join(d, "n08.json"))
    lock = lock_versions(args.lock)

    profile = build_profile(ref, ckpt, args.wheel_sha, adapter_rev, [1, 2, 3], lock)
    pid = profile_id(profile)
    stamp = args.stamp or time.strftime("%Y%m%dT%H%MZ", time.gmtime())
    exp5 = R.experiment_id("R5", pid, stamp)
    out = os.path.join(args.out_root, exp5)
    os.makedirs(out, exist_ok=True)
    common5 = R.Common(exp5, "R5", pid, "W3", adapter_rev)
    common1 = R.Common(exp5, "R1", pid, "W3", adapter_rev)  # same experiment dir; case differs

    # ---- token selection over the analysed runs (the `_auto` analyses = vLLM's own dispatch)
    tok = {"exact": 0, "attempted": 0}
    ind_cert = {"exact": 0, "attempted": 0}
    srt = {"exact": 0, "attempted": 0}
    unforced = {"exact": 0, "attempted": 0}
    per_variant: dict[str, Any] = {}
    trajectory = {"requests": 0, "attempted": 0}
    first_divs: dict[str, Any] = {}
    cross_batch: dict[str, Any] | None = None
    cross_runs: dict[str, Any] = {}
    primary_names: list[str] = []
    for name, a in sorted(analyses.items()):
        c = a["counters"]
        pc = a.get("prefix_cache") or {}
        entry = {"vllm_production": a.get("vllm_production"), "vllm_replay": a.get("vllm_replay"), "force_mode": a.get("force_mode"),
                 "independent": c["independent"], "same_runtime": c["same_runtime"],
                 "unforced_sampler_agrees": c.get("unforced_sampler_agrees", c["replay_resample"]),
                 "trajectory_followed": a.get("trajectory_followed"), "trace_replay_rows": a.get("trace_replay_rows"),
                 "trace_replay_changed_rows": a.get("trace_replay_changed_rows"),
                 "min_margin": a["min_margin"], "median_margin": a.get("median_margin"),
                 "first_divergence": a["first_divergence"], "checker_seconds": a["checker_seconds"], "topk_topp_impl": a.get("topk_topp_impl"),
                 "topk_ties": a.get("topk_ties"),
                 "prefix_cache": {k: pc.get(k) for k in ("production_recorded", "replay_recorded", "production_requests_with_cached_prefix",
                                                          "replay_requests_with_cached_prefix", "positions_from_cached_prefix_requests",
                                                          "all_fresh_prefill_both_sides")},
                 "primary": name.endswith(primary_suffix)}
        per_variant[name] = entry
        if "_at_c" in name and name.endswith("_auto"):
            # cap-64 record replayed under another batch composition (R1): aggregate across runs
            cross_runs[name] = entry
            if cross_batch is None:
                cross_batch = {"independent": {"exact": 0, "attempted": 0, "certified": 0, "certified_exact": 0},
                               "same_runtime": {"exact": 0, "attempted": 0}, "unforced_sampler_agrees": {"exact": 0, "attempted": 0}}
            for cls in ("independent", "same_runtime", "unforced_sampler_agrees"):
                for k in cross_batch[cls]:
                    cross_batch[cls][k] += int(entry[cls].get(k, 0))
            continue
        if not name.endswith(primary_suffix):
            continue  # `_triton`/`_unpinned` re-analyses are diagnostics; non-primary `_auto` runs are reported per run only
        primary_names.append(name)
        tok["exact"] += c["independent"]["exact"]
        tok["attempted"] += c["independent"]["attempted"]
        ind_cert["exact"] += c["independent"]["certified_exact"]
        ind_cert["attempted"] += c["independent"]["certified"]
        srt["exact"] += c["same_runtime"]["exact"]
        srt["attempted"] += c["same_runtime"]["attempted"]
        u = c.get("unforced_sampler_agrees", c["replay_resample"])
        unforced["exact"] += u["exact"]
        unforced["attempted"] += u["attempted"]
        tf = a.get("trajectory_followed") or {}
        trajectory["requests"] += int(tf.get("requests", 0))
        trajectory["attempted"] += int(tf.get("attempted", 0))
        for k, v in (a["first_divergence"] or {}).items():
            first_divs.setdefault(k, {"run": name, **v})

    # ---- stopping relations over every production record (structural field of the R5 row)
    eos = int(ref["eos_token_id"])
    stopping = stopping_relations(prods, eos)
    eos_sel = []
    for name, a in analyses.items():
        prod = prods.get(name.replace("select_", "").rsplit("_", 1)[0])
        if prod is None or not name.endswith(primary_suffix):
            continue
        recs = {r["request_id"]: r for r in prod["records"]}
        for row in a["positions"]:
            rec = recs[row["request_id"]]
            if row["k"] == len(rec["output_token_ids"]) - 1 and (rec.get("client_finish_reason") or rec["finish_reason"]) == "stop":
                eos_sel.append({"run": name, "index": rec["index"], "k": row["k"], "expected": row["expected"],
                                "independent": row["independent"], "certified": row["certified"], "ok": row["independent"] == row["expected"]})
    stopping["eos_selection_at_stop_positions"] = {"checked": len(eos_sel), "exact": sum(e["ok"] for e in eos_sel), "rows": eos_sel}
    structural = {"result": "ok" if stopping["all_pass"] and all(e["ok"] for e in eos_sel) else "fail",
                  "failing_invariant": None if stopping["all_pass"] and all(e["ok"] for e in eos_sel)
                  else (stopping["failures"][0]["relation"] if stopping["failures"] else "eos_selection")}

    # ---- Table 3 (N08 x2, N09 x2); inconclusive (coincidental value match) probes are not attempts
    table3 = []
    neg_sel = {k: v for k, v in negatives.items() if args.negatives_filter in k} or negatives
    for name, n in sorted(neg_sel.items()):
        if n["negative_id"] == "N08":
            att, rej = int(bool(n.get("attempted", True))), int(bool(n.get("rejected")))
            ua = att - rej
        else:
            rej = int(n["rejected"])
            att = rej + int(n["attempted"] - n["rejected"] - n["coincidental_value_matches"])
            ua = att - rej
        table3.append(R.require_valid(3, {**common5.fields(), "negative_id": n["negative_id"], "attempted": att, "rejected": rej,
                                          "unexpectedly_accepted": ua,
                                          "reproducer": f"fixtures/results/{os.path.basename(os.path.normpath(raw))}/negative/{name}"
                                                        + ("/n08.json" if n["negative_id"] == "N08" else ".json")}))
    negative = R.negative_summary(table3)

    caps = repro.get("concurrency", {})
    peaks = {k: {c: v["peak_concurrency_achieved"] for c, v in d.items() if isinstance(v, dict)} for k, d in caps.items()}
    peak_max = max((p for d in peaks.values() for p in d.values()), default=0)
    gen_positions_ref = sum(r["num_output_tokens"] for r in ref["records"])
    # the primary (R5) production records: achieved concurrency of the recorded pass itself
    primary_prods = {n.replace("select_", "").rsplit("_", 1)[0]: prods.get(n.replace("select_", "").rsplit("_", 1)[0]) for n in primary_names}
    primary_prods = {k: v for k, v in primary_prods.items() if v is not None}
    peak_primary = max((p["peak_concurrency_achieved"] for p in primary_prods.values()), default=peak_max)
    prefix_cache_prod = {name: {"recorded": all(r.get("num_cached_tokens") is not None for r in p["records"]),
                                "requests_with_cached_prefix": p.get("requests_with_cached_prefix",
                                                                     sorted(r["index"] for r in p["records"] if r.get("cached_prefix"))),
                                "label": (p.get("cached_prefix_label") or {}).get("source"),
                                "execution": (p.get("execution") or {}).get("note"), "warmup": p.get("warmup") is not None}
                        for name, p in sorted(prods.items())}
    replay_restrictions = [
        "replay engine in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0); same kernels and graph mode as production",
        "logits rows copied to host every step (replay reference only; no commitments)",
    ]
    if any(p.get("execution") for p in primary_prods.values()):
        replay_restrictions.append("workload requests 11 and 13 share one leading 16-token block: request 13 is recorded and "
                                   "replayed in its own fresh engine so that every request is a fresh prefill (num_cached_tokens 0)")
    fm = {a.get("force_mode") for a in analyses.values()}
    if "trace_replay" in fm:
        replay_restrictions.append("trajectory injected by vLLM trace replay (enable_trace_replay + SamplingParams.trace_decode_token_ids)")
    if "hook" in fm:
        replay_restrictions.append("vLLM 0.28.0 has no trace replay: trajectory injected by overwriting sampled_token_ids in a Sampler.__call__ wrap")

    row5 = R.table1_row(
        common5,
        workload={"requests": 16, "generated_positions": tok["attempted"], "input_len_range": [32, 2048],
                  "peak_concurrency_achieved": peak_primary},
        production_changes="flags:[VLLM_BATCH_INVARIANT=1]",
        replay_restrictions=replay_restrictions,
        token_selection=tok,
        boundary_noninterference={"identical": unforced["exact"] == unforced["attempted"] and unforced["attempted"] > 0,
                                  "compared": unforced["attempted"], "mismatched_scalars": unforced["attempted"] - unforced["exact"],
                                  "reference_method": "unforced sampler token on the replay-reference logits row vs production token "
                                                      "(sampler runs before the trace overwrite); production-visible outputs only"},
        required_coverage={"bound": 0, "required": 0},
        authentication={"valid": 0, "attempted": 0},
        relation_checks={"same_runtime": srt, "independent": ind_cert},
        structural=structural,
        negative=negative,
    )

    # ---- R1 production side: batching sweep
    compared = mismatched = 0
    cap_detail = {}
    for key, d in caps.items():
        for c, v in d.items():
            if isinstance(v, dict) and v.get("vs_cap64"):
                compared += v["vs_cap64"]["positions_compared"]
                mismatched += v["vs_cap64"]["positions_compared"] - v["vs_cap64"]["positions_matching_prefix"]
                cap_detail[f"{key}@{c}"] = {"peak": v["peak_concurrency_achieved"], "identical_requests": v["vs_cap64"]["identical_requests"],
                                            "requests": v["vs_cap64"]["requests"], "first_divergence": v["vs_cap64"]["first_divergence"]}
    tok1 = {"exact": cross_batch["independent"]["exact"], "attempted": cross_batch["independent"]["attempted"]} if cross_batch else {"exact": 0, "attempted": 0}
    rel1 = ({"same_runtime": dict(cross_batch["same_runtime"]),
             "independent": {"exact": cross_batch["independent"]["certified_exact"], "attempted": cross_batch["independent"]["certified"]}}
            if cross_batch else {"same_runtime": {"exact": 0, "attempted": 0}, "independent": {"exact": 0, "attempted": 0}})
    row1 = R.table1_row(
        common1,
        workload={"requests": 16, "generated_positions": gen_positions_ref, "input_len_range": [32, 2048],
                  "peak_concurrency_achieved": peak_max},
        production_changes="flags:[VLLM_BATCH_INVARIANT=1]",
        replay_restrictions=replay_restrictions + ["R1 production side only: no adapter capture, no commitments"],
        token_selection=tok1,
        boundary_noninterference={"identical": mismatched == 0 and compared > 0, "compared": compared, "mismatched_scalars": mismatched,
                                  "reference_method": "stock production outputs at max_num_seqs 1/8/32 and orderings reversed/shuffled "
                                                      "vs the cap-64 record (same seeds); positions after the first divergence count as mismatched"},
        required_coverage={"bound": 0, "required": 0},
        authentication={"valid": 0, "attempted": 0},
        relation_checks=rel1,
        structural={"result": "ok", "failing_invariant": None},
        negative={"rejected": 0, "attempted": 0},
    )
    R.write_jsonl(os.path.join(out, "table1.jsonl"), [row5, row1])
    R.write_jsonl(os.path.join(out, "table3.jsonl"), table3)

    # ---- Table 4
    t4 = []
    ov = repro.get("overhead", {})
    hours = args.pod_hours
    spend = {"gpu_hours": hours, "usd": None if hours is None else round(hours * POD_USD_PER_HOUR, 2)}
    null_t4 = dict(replay_commit_overhead_ratio=None, replay_vs_production_gpu_work=None,
                   transparent_check_cost={"cpu_s": None, "gpu_s": None},
                   peak_replay_memory={"device_gb": None, "host_gb": None, "delta_vs_reference": None},
                   committed_logical_bytes=None, processed_bytes={"hashed": None, "copied": None, "retained": None, "emitted": None},
                   description_size_bytes=None, constructor_inputs_bytes=None, logical_size={"gates": None, "vus": None, "rus": None},
                   lookup_cost_s=None, validation_cost_s=None,
                   moe_padding={"inactive_units": None, "inactive_checks": None, "inactive_work_s": None, "inactive_bytes": None})
    for key, o in ov.items():
        t4.append(R.table4_row(common5, partition_id=f"production:{key}", instrumentation_mode="production recording (logprobs=1, outputs kept)",
                               production_overhead_ratio=o["production_overhead_ratio"],
                               **null_t4,
                               restrictions=["generation phase only, engine init excluded", "single stock run vs median of fresh-process repetitions",
                                             f"samples {o.get('recorded_generate_s_samples')} vs stock {o['stock_generate_s']:.3f}s"],
                               spend=spend))
    for name, a in sorted(analyses.items()):
        if not name.endswith(primary_suffix):
            continue
        prod = prods.get(name.replace("select_", "").rsplit("_", 1)[0])
        rep_gen = (a.get("replay_timing") or {}).get("generate_s")
        t4.append(R.table4_row(common5, partition_id=f"sampler.select:{name}", instrumentation_mode="replay reference (logits hook) + checker",
                               production_overhead_ratio=None,
                               **{**null_t4, "transparent_check_cost": {"cpu_s": a["checker_seconds"]["independent_total"],
                                                                        "gpu_s": a["checker_seconds"]["same_runtime_total"]},
                                  "replay_vs_production_gpu_work": (rep_gen / prod["timing"]["generate_s"]) if (rep_gen and prod) else None},
                               restrictions=[f"independent checker: numpy over one 151936-row per position, "
                                             f"{a['checker_seconds']['independent_per_position']*1e3:.1f} ms/position",
                                             "same_runtime checker: one vLLM kernel launch pair per position (GPU)",
                                             "replay_vs_production_gpu_work = replay wall-clock incl. per-step host copy of logits / production wall-clock"],
                               spend=spend))
    R.write_jsonl(os.path.join(out, "table4.jsonl"), t4)

    manifest = {"profile": profile, "profile_id": pid, "experiment_id": exp5, "worker": "W3", "cases": ["R5", "R1(production side)"],
                "checkpoint": ckpt, "workload": "fixtures/results/w3-sampler/workload.json", "raw": os.path.basename(os.path.normpath(raw)),
                "production_runs": sorted(prods), "analyses": sorted(analyses), "primary_analyses": primary_names,
                "negatives": sorted(negatives), "table3_negatives": sorted(neg_sel),
                "vllm": {"versions_in_production_records": repro.get("vllm_versions"), "commit": repro.get("vllm_commit")},
                "prefix_cache": prefix_cache_prod,
                "notes": ["Table 1 row R5 covers token selection + stopping; commitments/authentication/coverage are W1's and NOT_RUN "
                          "(zero attempted -> PARTIAL by the §10 rule).",
                          "Table 1 row R1 is the production-side batching sweep plus one cross-batch replay analysis.",
                          f"R5 evidence = analyses ending in {primary_suffix!r}: production records whose every request was a fresh "
                          "prefill (num_cached_tokens == 0 recorded by the engine, throwaway warm-up only), replayed with the same split; "
                          "earlier records (natural order, one engine) have request 13 served 16 prompt tokens from request 11's block "
                          "and are labelled cached_prefix=true; their analyses are kept per run."]}
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    detail = {
        "schema": "verity-vllm/w3-detail/v1", "experiment_id": exp5,
        "token_selection": {"independent_all_positions": tok, "independent_certified": ind_cert, "same_runtime": srt,
                            "unforced_sampler_agrees": unforced, "trajectory_followed": trajectory,
                            "first_divergences": first_divs, "per_run": per_variant},
        "rng_validation": rng,
        "production_reproducibility": {k: {"bit_identical_across_repetitions": v["bit_identical_across_repetitions"], "runs": v["runs"]}
                                       for k, v in repro["repetitions"].items()},
        "batching": {"outputs_invariant_to_batching": {k: v["outputs_invariant_to_batching"] for k, v in caps.items()},
                     "peak_concurrency_achieved": peaks, "per_cap": cap_detail, "cross_batch_replay": cross_batch,
                     "cross_batch_runs": cross_runs},
        "stopping": stopping,
        "negatives": negatives,
        "prefix_cache_per_production_run": prefix_cache_prod,
        # compare_rows.py: are the sampler-input logits rows of one recorded trajectory bit-identical between two
        # replays (cached-prefix r0 dump vs fresh-prefill dump; cap-64 dump vs cross-batch dump)?
        "logits_row_comparisons": {k: {**{kk: v.get(kk) for kk in ("label", "vllm_a", "vllm_b", "requests", "positions", "identical_rows",
                                                                    "rows_differing", "max_abs_diff_bf16_as_f32", "first_diff", "all_identical")},
                                       "a": os.path.basename(os.path.normpath(v["a"])), "b": os.path.basename(os.path.normpath(v["b"])),
                                       # None = that dump carries no cache witness (older hook), not "no cached request"
                                       "cached_requests_a": [pr["request_id"] for pr in v.get("per_request", []) if pr.get("cached_a")]
                                       if any(pr.get("cached_a") is not None for pr in v.get("per_request", [])) else None,
                                       "cached_requests_b": [pr["request_id"] for pr in v.get("per_request", []) if pr.get("cached_b")]
                                       if any(pr.get("cached_b") is not None for pr in v.get("per_request", [])) else None}
                                   for k, v in sorted(row_cmps.items())},
        "overhead": ov,
        "verdicts": {"R5": {"result": row5["result"], "first_blocker": row5["first_blocker"], "reasons": R.verdict(row5).reasons},
                     "R1": {"result": row1["result"], "first_blocker": row1["first_blocker"], "reasons": R.verdict(row1).reasons}},
    }
    with open(os.path.join(out, "w3_detail.json"), "w") as f:
        json.dump(detail, f, indent=1)
    with open(os.path.join(out, "log.txt"), "w") as f:
        f.write(f"{exp5}\nassembled {stamp} from {raw}; vllm {profile['vllm']}\n")
        f.write(f"R5 {row5['result']} ({row5['first_blocker']}); R1 {row1['result']} ({row1['first_blocker']})\n")
        f.write(f"token_selection {tok}  certified {ind_cert}  same_runtime {srt}  unforced {unforced}  trajectory {trajectory}\n")
        for k, v in repro["repetitions"].items():
            f.write(f"reps {k}: bit_identical={v['bit_identical_across_repetitions']}\n")
        for k, v in caps.items():
            f.write(f"caps {k}: invariant={v['outputs_invariant_to_batching']} peaks={peaks[k]}\n")
        for k, v in per_variant.items():
            f.write(f"select {k}{' [primary]' if v['primary'] else ''}: ind {v['independent']} srt {v['same_runtime']} "
                    f"unforced {v['unforced_sampler_agrees']} traj {v['trajectory_followed']} cache {v['prefix_cache']}\n")
        for k, v in prefix_cache_prod.items():
            f.write(f"prod {k}: cached_prefix_requests={v['requests_with_cached_prefix']} recorded={v['recorded']} "
                    f"label={'yes' if v['label'] else 'no'} split={'yes' if v['execution'] else 'no'} warmup={v['warmup']}\n")
        f.write(f"stopping: {json.dumps({k: v for k, v in stopping.items() if k != 'eos_selection_at_stop_positions'})}\n")
        for r in table3:
            f.write(f"table3 {r['negative_id']}: {r['attempted']}/{r['rejected']}/{r['unexpectedly_accepted']}\n")
    print("REPORT_DONE", out, "profile_id", pid, "R5", row5["result"], row5["first_blocker"], "| R1", row1["result"], row1["first_blocker"])


if __name__ == "__main__":
    main()
