"""Run the `sampler.select` relation over a replay logits dump and report exact/attempted.

    python -m verity_vllm_sampler.analyze --production prod.json --logits-dir logits_dir --out report.json \
        [--same-runtime] [--topk-topp-impl auto|triton|torch]

For every (request, output position k) with a recorded token:
  * independent: numpy re-derivation (`select.independent_select`), with certification;
  * same_runtime (GPU, optional): vLLM's own kernels on the committed row.
Also compares the replay sampler's own choice on the replayed logits (`sampled_before_force`)
with the production token: that is the "replay reproduces production token selection" check,
which mixes model batch-invariance and sampler determinism, hence reported separately.

Reports the first divergent position (expected/actual, margin, runner-up) per class.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import numpy as np

from . import select as S


def load_dump(logits_dir: str) -> dict[str, Any]:
    with open(os.path.join(logits_dir, "index.json")) as f:
        return json.load(f)


def sampler_rows_per_step(prod: dict[str, Any]) -> dict[int, int]:
    """Rows the production sampler saw at each engine step, from the uninstrumented record: every
    running request emits one token per step, so tokens arriving at step s (`token_arrival_step`)
    count the sampled rows of that step.  Mid-prefill chunk rows are also in the kernel's batch but
    emit no token; they occur only in the first steps, where the batch is largest anyway."""
    counts: dict[int, int] = {}
    for rec in prod["records"]:
        for s in rec.get("token_arrival_step", []):
            counts[s] = counts.get(s, 0) + 1
    return counts


def resolve_topk_ties(topk_ties: str | None):
    """Return (per-position rule function, label).

    topk_ties: "exact_k" | "keep_all" | None (unpinned, certify only where both agree) |
    "from_batch:<n>" — the rule production used at each position from its batch size: exact_k
    (Triton) at >= n rows, keep_all (torch sort) below; 0.28.0 switches at n = 8."""
    if isinstance(topk_ties, str) and topk_ties.startswith("from_batch"):
        thr = int(topk_ties.split(":", 1)[1]) if ":" in topk_ties else 8

        def rule(batch_rows: int | None) -> str | None:
            if batch_rows is None:
                return None
            return "exact_k" if batch_rows >= thr else "keep_all"

        return rule, f"from_batch:{thr}"
    return (lambda _b: topk_ties), (topk_ties or "unpinned")


def analyze(prod: dict[str, Any], dump: dict[str, Any], logits_dir: str, same_runtime: bool,
            topk_topp_impl: str = "auto", worker: str = "W3", topk_ties: str | None = "exact_k") -> dict[str, Any]:
    dtype = dump["logits_dtype"]
    V = dump["vocab_size"]
    tie_rule, tie_label = resolve_topk_ties(topk_ties)
    rows_per_step = sampler_rows_per_step(prod)
    ties_used: dict[str, int] = {}
    rows_out: list[dict[str, Any]] = []
    counters = {
        "independent": {"attempted": 0, "exact": 0, "certified": 0, "certified_exact": 0, "uncertified_exact": 0},
        "same_runtime": {"attempted": 0, "exact": 0},
        "replay_resample": {"attempted": 0, "exact": 0},
    }
    first_div: dict[str, Any] = {}
    margins: list[float] = []
    t_ind = 0.0
    t_rt = 0.0
    recs = {r["request_id"]: r for r in prod["records"]}
    for rid, info in dump["requests"].items():
        rec = recs[rid]
        sp = rec["sampling"]
        seed = rec.get("seed")
        arr = np.load(os.path.join(logits_dir, info["file"]))
        recorded = info["recorded"]
        plen = info["prompt_len"]
        for i, pos in enumerate(info["positions"]):
            k = pos - (plen - 1)
            if k >= len(recorded):
                continue
            expected = int(recorded[k])
            raw = arr[i]
            logits = S.logits_from_bytes(raw.tobytes(), dtype) if dtype in ("bfloat16", "bf16") else raw.astype(np.float32)
            arrival = rec.get("token_arrival_step") or []
            batch_rows = rows_per_step.get(arrival[k]) if k < len(arrival) else None
            ties = tie_rule(batch_rows)
            ties_used[ties or "unpinned"] = ties_used.get(ties or "unpinned", 0) + 1
            t0 = time.perf_counter()
            r_ind = S.independent_select(logits, sp.get("temperature", 1.0), sp.get("top_k", 0), sp.get("top_p", 1.0), seed, pos,
                                         topk_ties=ties)
            t_ind += time.perf_counter() - t0
            row: dict[str, Any] = {
                "request_id": rid, "index": rec["index"], "k": k, "pos": pos, "expected": expected,
                "independent": r_ind.predicted, "certified": r_ind.certified, "margin": r_ind.margin,
                "runner_up": r_ind.runner_up, "reason": r_ind.reason,
                "topk_ties": ties or "unpinned", "production_batch_rows": batch_rows,
                "replay_resample": int(info["sampled_before_force"][i]),
            }
            c = counters["independent"]
            c["attempted"] += 1
            ok = r_ind.predicted == expected
            c["exact"] += ok
            c["certified"] += r_ind.certified
            c["certified_exact"] += ok and r_ind.certified
            c["uncertified_exact"] += ok and not r_ind.certified
            if not ok and "independent" not in first_div:
                first_div["independent"] = {**row, "details": r_ind.to_json()}
            if r_ind.margin is not None and np.isfinite(r_ind.margin):
                margins.append(float(r_ind.margin))
            cr = counters["replay_resample"]
            cr["attempted"] += 1
            cr["exact"] += row["replay_resample"] == expected
            if row["replay_resample"] != expected and "replay_resample" not in first_div:
                first_div["replay_resample"] = dict(row)
            if same_runtime:
                t0 = time.perf_counter()
                r_rt = S.same_runtime_select(logits, sp.get("temperature", 1.0), sp.get("top_k", 0), sp.get("top_p", 1.0),
                                             seed, pos, topk_topp_impl=topk_topp_impl)
                t_rt += time.perf_counter() - t0
                row["same_runtime"] = r_rt.predicted
                cs = counters["same_runtime"]
                cs["attempted"] += 1
                cs["exact"] += r_rt.predicted == expected
                if r_rt.predicted != expected and "same_runtime" not in first_div:
                    first_div["same_runtime"] = {**row, "details": r_rt.to_json()}
            rows_out.append(row)
    # contract §15.3: the two facts are separate fields
    counters["unforced_sampler_agrees"] = dict(counters["replay_resample"])
    # prefix-cache witness on both sides (contract §15.4/§15.8): per request, prompt tokens served from the
    # cache in production (recorded, or labelled from a measured re-run) and in this replay
    cache_rows = []
    for rid, info in dump["requests"].items():
        rec = recs[rid]
        rep = ((dump.get("num_cached_tokens") or {}).get(rid) or {}).get("replay")
        if rep is None:
            rep = ((dump.get("replay_outputs") or {}).get(rid) or {}).get("num_cached_tokens")
        cache_rows.append({"index": rec["index"], "prompt_len": rec["prompt_len"], "production": rec.get("num_cached_tokens"),
                           "replay": rep, "positions": sum(1 for p in info["positions"] if p - (info["prompt_len"] - 1) < len(info["recorded"]))})
    cache_rows.sort(key=lambda r: r["index"])
    prefix_cache = {
        "per_request": cache_rows,
        # `*_recorded = False` means the side carries no witness (older recorder / dump): its
        # `*_requests_with_cached_prefix` list is then "unknown", not "none"
        "production_recorded": all(r["production"] is not None for r in cache_rows),
        "replay_recorded": all(r["replay"] is not None for r in cache_rows),
        "production_requests_with_cached_prefix": [r["index"] for r in cache_rows if (r["production"] or 0) > 0]
        if all(r["production"] is not None for r in cache_rows) else None,
        "replay_requests_with_cached_prefix": [r["index"] for r in cache_rows if (r["replay"] or 0) > 0]
        if all(r["replay"] is not None for r in cache_rows) else None,
        "positions_from_cached_prefix_requests": sum(r["positions"] for r in cache_rows if (r["production"] or 0) > 0 or (r["replay"] or 0) > 0),
        "all_fresh_prefill_both_sides": all(not (r["production"] or 0) and not (r["replay"] or 0) and r["production"] is not None and r["replay"] is not None
                                            for r in cache_rows),
    }
    return {
        "schema": "verity-vllm/sampler-select-report/v1",
        "worker": worker,
        "variant": prod["variant"], "seed": prod["seed"], "max_num_seqs": prod["max_num_seqs"],
        "vllm_production": prod["profile"].get("vllm"), "vllm_replay": dump.get("vllm_version") or dump.get("profile", {}).get("vllm"),
        "force_mode": dump.get("force_mode", "hook"),
        "trajectory_followed": dump.get("trajectory_followed"),
        "trace_replay_rows": {k: dump.get(k) for k in ("trace_replay_rows_changed_by_vllm", "trace_replay_rows_followed",
                                                        "trace_replay_rows_token_mismatch", "forcing_events", "rows_seen")},
        "trace_replay_changed_rows": dump.get("trace_replay_changed_rows"),
        "prefix_cache": prefix_cache,
        "replay_warmup": dump.get("warmup"),
        "production_execution": (prod.get("execution") or {}).get("note"),
        "logits_dtype": dtype, "vocab_size": V,
        "replay_timing": dump.get("timing"), "replay_init_s": dump.get("timing_init_s"),
        "production_timing": prod.get("timing"),
        "counters": counters,
        "first_divergence": first_div,
        "min_margin": min(margins) if margins else None,
        "median_margin": float(np.median(margins)) if margins else None,
        "checker_seconds": {"independent_total": t_ind, "same_runtime_total": t_rt,
                            "independent_per_position": t_ind / max(1, counters["independent"]["attempted"])},
        "topk_topp_impl": topk_topp_impl if same_runtime else None,
        "topk_ties": tie_label,
        "topk_ties_used": ties_used,
        "production_batch_rows_per_step": {str(s): n for s, n in sorted(rows_per_step.items())},
        "hook": dump.get("hook"),
        "positions": rows_out,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--production", required=True)
    ap.add_argument("--logits-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--same-runtime", action="store_true")
    ap.add_argument("--topk-topp-impl", default="auto")
    ap.add_argument("--topk-ties", default="exact_k",
                    help="top-k tie rule production used: exact_k = Triton kernel (nightly at every batch size; 0.28.0 at "
                         ">= 8 rows), keep_all = torch sort path (0.28.0 below 8 rows), unpinned = certify only where both "
                         "agree, from_batch:<n> = per position from the production batch size (exact_k at >= n rows)")
    args = ap.parse_args()
    if args.topk_ties not in ("exact_k", "keep_all", "unpinned") and not args.topk_ties.startswith("from_batch"):
        ap.error(f"bad --topk-ties {args.topk_ties!r}")
    with open(args.production) as f:
        prod = json.load(f)
    dump = load_dump(args.logits_dir)
    rep = analyze(prod, dump, args.logits_dir, args.same_runtime, args.topk_topp_impl,
                  topk_ties=None if args.topk_ties == "unpinned" else args.topk_ties)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=1)
    c = rep["counters"]
    tf = rep.get("trajectory_followed") or {}
    print(f"ANALYZE_DONE {args.out} independent {c['independent']['exact']}/{c['independent']['attempted']} "
          f"(certified {c['independent']['certified']}) same_runtime {c['same_runtime']['exact']}/{c['same_runtime']['attempted']} "
          f"unforced_sampler_agrees {c['unforced_sampler_agrees']['exact']}/{c['unforced_sampler_agrees']['attempted']} "
          f"trajectory_followed={tf.get('requests')}/{tf.get('attempted')} min_margin={rep['min_margin']} "
          f"topk_ties={rep['topk_ties']} used={rep['topk_ties_used']} "
          f"cached_prefix prod={rep['prefix_cache']['production_requests_with_cached_prefix']} "
          f"replay={rep['prefix_cache']['replay_requests_with_cached_prefix']} "
          f"all_fresh_both={rep['prefix_cache']['all_fresh_prefill_both_sides']}")


if __name__ == "__main__":
    main()
