"""R10 acq-native: aggregate the interleaved multi-arm batches (`commit_delta --arms`) into per-arm medians with within-batch CIs and
between-batch spread, plus the diagnostics the REPORT table needs.  Reads `<raw>/<label>/runs.jsonl` (+ `summary.json`), writes JSON and a
markdown table.  Every number is "measured" unless labelled "calculated" (ratios / medians of measured samples).

    python -m verity_capture.bench.acq_aggregate --raw out/gen/r10/acq-native/raw --glob 'b0_*arm_batch*' --out out/gen/r10/acq-native/raw/aggregate_b0.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import statistics as st
from collections import defaultdict


def _boot_ci(xs, n=4000, seed=0):
    if len(xs) < 2:
        return [min(xs), max(xs)] if xs else [float("nan"), float("nan")]
    rng = random.Random(seed)
    ms = sorted(st.mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return [ms[int(0.025 * n)], ms[int(0.975 * n) - 1]]


def load_batch(d):
    rows = [json.loads(l) for l in open(os.path.join(d, "runs.jsonl"))]
    by_round = defaultdict(dict)
    for r in rows:
        by_round[r["pair"]][r["arm"]] = r
    return rows, by_round


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dirs = sorted(d for d in glob.glob(os.path.join(a.raw, a.glob)) if os.path.exists(os.path.join(d, "runs.jsonl")))
    per_arm = defaultdict(lambda: {"batches": {}, "rounds": []})
    control_wall = []
    roots = defaultdict(set)
    tokens_ok = True
    for d in dirs:
        label = os.path.basename(d)
        rows, by_round = load_batch(d)
        arms = sorted({r["arm"] for r in rows} - {"control"})
        for rnd, arm_rows in sorted(by_round.items()):
            c = arm_rows.get("control")
            if c is None:
                continue
            control_wall.append(c["wall_s"])
            tokens_ok &= all(r.get("tokens_equal_ref", True) for r in arm_rows.values())
            for arm in arms:
                r = arm_rows.get(arm)
                if r is None:
                    continue
                cm = r.get("commit") or {}
                ex = cm.get("extra") or {}
                nc = ex.get("native_collect") or {}
                dec_c = c["step_wall_decode_mean_s"]
                per_arm[arm]["rounds"].append({
                    "batch": label, "round": rnd,
                    "delta_wall": r["wall_s"] / c["wall_s"] - 1.0,
                    "delta_decode": r["step_wall_decode_mean_s"] / dec_c - 1.0,
                    "delta_prefill": r["step_wall_prefill_s"] / c["step_wall_prefill_s"] - 1.0,
                    "wall_ms": r["wall_s"] * 1e3, "control_wall_ms": c["wall_s"] * 1e3,
                    "decode_ms": r["step_wall_decode_mean_s"] * 1e3, "control_decode_ms": dec_c * 1e3,
                    "gen_tok_s": r["generated_tokens_per_s"], "control_gen_tok_s": c["generated_tokens_per_s"],
                    "root_ready_delay_ms": (cm.get("root_ready_delay_s") or 0.0) * 1e3,
                    "peak_gpu_gb": (r.get("gpu_mem") or {}).get("torch_max_allocated_gb"),
                    "control_peak_gpu_gb": (c.get("gpu_mem") or {}).get("torch_max_allocated_gb"),
                    "rss_after_mb": ((r.get("host_mem_kb") or {}).get("rss_after") or 0) / 1024,
                    "acq_s": (cm.get("spans_total_s") or {}).get("acquisition"),
                    "hash_s": (cm.get("spans_total_s") or {}).get("hashing"),
                    "mov_s": (cm.get("spans_total_s") or {}).get("movement"),
                    "gpu_tree_ms": ex.get("gpu_tree_ms"),
                    "openings": (cm.get("openings") or {}).get("verified"), "openings_n": (cm.get("openings") or {}).get("n"),
                    "bytes_bound": cm.get("bytes_bound"), "leaves": cm.get("leaves"), "tensors": cm.get("tensors"), "steps": cm.get("steps"),
                    "copies": nc.get("copies"), "copy_bytes": nc.get("bytes"), "memsets": nc.get("memsets"), "flushes": nc.get("flushes"),
                    "issue_s": nc.get("issue_s"), "plans": nc.get("plans"), "misses": nc.get("misses"),
                    "root_equal_expected": r.get("root_equal_expected"),
                })
                roots[(label, rnd)].add((cm.get("run_root") or "?")[:16])
        for arm in arms:
            xs = [x["delta_wall"] for x in per_arm[arm]["rounds"] if x["batch"] == label]
            if xs:
                per_arm[arm]["batches"][label] = {"n": len(xs), "mean": st.mean(xs), "median": st.median(xs), "ci95": _boot_ci(xs)}
    out = {"label": "aggregate of interleaved multi-arm batches (each batch = separate process; arms rotate within a round)",
           "batches": [os.path.basename(d) for d in dirs], "control_wall_ms_median": st.median(control_wall) * 1e3 if control_wall else None,
           "tokens_identical_all_runs": tokens_ok,
           "roots_identical_across_arms_every_round": all(len(v) == 1 for v in roots.values()), "rounds_checked": len(roots), "arms": {}}
    for arm, d in per_arm.items():
        R = d["rounds"]
        dw = [x["delta_wall"] for x in R]
        dd = [x["delta_decode"] for x in R]
        bm = [b["mean"] for b in d["batches"].values()]
        mid = lambda k: st.median([x[k] for x in R if x[k] is not None]) if any(x[k] is not None for x in R) else None
        out["arms"][arm] = {
            "n_rounds": len(R), "n_batches": len(d["batches"]),
            "delta_wall_pooled_median": st.median(dw), "delta_wall_pooled_mean": st.mean(dw), "delta_wall_pooled_ci95": _boot_ci(dw),
            "delta_decode_pooled_median": st.median(dd), "delta_decode_pooled_ci95": _boot_ci(dd),
            "delta_prefill_pooled_median": st.median([x["delta_prefill"] for x in R]),
            "between_batch_means": {k: v["mean"] for k, v in d["batches"].items()},
            "between_batch_spread": {"min": min(bm), "max": max(bm), "sd": st.pstdev(bm) if len(bm) > 1 else 0.0} if bm else None,
            "within_batch_ci95": {k: v["ci95"] for k, v in d["batches"].items()},
            "wall_ms_median": mid("wall_ms"), "decode_ms_median": mid("decode_ms"), "gen_tok_s_median": mid("gen_tok_s"),
            "control_wall_ms_median": mid("control_wall_ms"), "control_decode_ms_median": mid("control_decode_ms"), "control_gen_tok_s_median": mid("control_gen_tok_s"),
            "root_ready_delay_ms": {"median": mid("root_ready_delay_ms"), "max": max(x["root_ready_delay_ms"] for x in R)},
            "peak_gpu_gb_median": mid("peak_gpu_gb"), "control_peak_gpu_gb_median": mid("control_peak_gpu_gb"), "rss_after_mb_median": mid("rss_after_mb"),
            "acq_ms_per_run_median": (mid("acq_s") or 0) * 1e3, "hash_ms_per_run_median": (mid("hash_s") or 0) * 1e3, "mov_ms_per_run_median": (mid("mov_s") or 0) * 1e3,
            "gpu_tree_ms_median": {k: st.median([x["gpu_tree_ms"][k] for x in R if x["gpu_tree_ms"]]) for k in ("leaf_ms", "tree_ms")} if any(x["gpu_tree_ms"] for x in R) else None,
            "openings_all_verified": all(x["openings"] == x["openings_n"] == 64 for x in R),
            "bytes_bound": R[0]["bytes_bound"], "leaves": R[0]["leaves"], "tensors": R[0]["tensors"], "steps": R[0]["steps"],
            "copies_median": mid("copies"), "copy_bytes_median": mid("copy_bytes"), "memsets_median": mid("memsets"), "flushes_median": mid("flushes"),
            "issue_ms_per_run_median": (mid("issue_s") or 0) * 1e3, "plans_median": mid("plans"), "misses_median": mid("misses"),
            "root_equal_expected_all": all(x["root_equal_expected"] in (True, None) for x in R),
            "rounds": R,
        }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    # markdown
    print(f"batches: {out['batches']}; control wall median {out['control_wall_ms_median']:.1f} ms; tokens identical: {tokens_ok}; "
          f"roots identical across arms every round: {out['roots_identical_across_arms_every_round']} ({out['rounds_checked']} rounds)")
    print("| arm | rounds / batches | ΔT wall pooled median [95% CI of mean] | between-batch means (min…max, sd) | Δ decode median | root-ready (median / max ms) | peak GPU GB (arm / control) | copies/run | flushes/run | acq ms/run | GPU tree leaf+tree ms/run |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for arm, s in out["arms"].items():
        bb = s["between_batch_spread"]
        gt = s["gpu_tree_ms_median"]
        print(f"| {arm} | {s['n_rounds']} / {s['n_batches']} | **{s['delta_wall_pooled_median']*100:+.1f} %** [{s['delta_wall_pooled_ci95'][0]*100:+.1f}, {s['delta_wall_pooled_ci95'][1]*100:+.1f}] "
              f"| {bb['min']*100:+.1f}…{bb['max']*100:+.1f} (sd {bb['sd']*100:.1f}) | {s['delta_decode_pooled_median']*100:+.1f} % | {s['root_ready_delay_ms']['median']:.2f} / {s['root_ready_delay_ms']['max']:.2f} "
              f"| {s['peak_gpu_gb_median']:.3f} / {s['control_peak_gpu_gb_median']:.3f} | {s['copies_median'] or '—'} | {s['flushes_median'] or '—'} | {s['acq_ms_per_run_median']:.1f} "
              f"| {(gt['leaf_ms'] if gt else 0):.1f}+{(gt['tree_ms'] if gt else 0):.1f} |")


if __name__ == "__main__":
    main()
