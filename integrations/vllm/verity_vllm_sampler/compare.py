"""Production-side reproducibility report over the recorded runs (CPU).

    python -m verity_vllm_sampler.compare --prod-dir <dir with prod/*.json> --out reproducibility.json

Answers, per variant/seed:
  * fresh-process repetitions (r0, r1, r2 at cap 64): are output ids bit-identical?  per request;
    first divergent (request, position) with expected/actual and the chosen-token logprobs there;
  * concurrency caps {1, 8, 32, 64}: achieved peak concurrency, and whether outputs change vs cap 64;
  * request orderings at cap 8;
  * Table 4 production overhead: recorded (logprobs=1 + outputs kept) vs stock (--no-record).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any


def load_all(prod_dir: str) -> dict[str, dict[str, Any]]:
    out = {}
    for p in sorted(glob.glob(os.path.join(prod_dir, "*.json"))):
        with open(p) as f:
            out[os.path.basename(p)[:-5]] = json.load(f)
    return out


def outputs_by_index(run: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {r["index"]: r for r in run["records"]}


def diff_runs(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    ra, rb = outputs_by_index(a), outputs_by_index(b)
    per_req = []
    identical = 0
    first = None
    total_positions = 0
    matching_positions = 0
    for idx in sorted(ra):
        x, y = ra[idx]["output_token_ids"], rb[idx]["output_token_ids"]
        same = x == y
        identical += same
        n = max(len(x), len(y))
        total_positions += len(x)
        k = next((i for i in range(n) if i >= len(x) or i >= len(y) or x[i] != y[i]), None)
        matching_positions += len(x) if k is None else k
        per_req.append({"index": idx, "identical": same, "len_a": len(x), "len_b": len(y), "first_divergent_k": k,
                        "finish_a": ra[idx]["finish_reason"], "finish_b": rb[idx]["finish_reason"]})
        if k is not None and first is None:
            lpa = (ra[idx].get("chosen_logprobs") or [None] * n)
            lpb = (rb[idx].get("chosen_logprobs") or [None] * n)
            first = {"index": idx, "k": k, "pos": ra[idx]["prompt_len"] - 1 + k,
                     "a_token": x[k] if k < len(x) else None, "b_token": y[k] if k < len(y) else None,
                     "a_logprob": lpa[k] if k < len(lpa) else None, "b_logprob": lpb[k] if k < len(lpb) else None,
                     "a_prefix_matches": k}
    return {"requests": len(per_req), "identical_requests": identical, "all_identical": identical == len(per_req),
            "positions_compared": total_positions, "positions_matching_prefix": matching_positions,
            "first_divergence": first, "per_request": per_req}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs = load_all(args.prod_dir)
    versions = sorted({r["profile"].get("vllm") for r in runs.values()})
    report: dict[str, Any] = {"schema": "verity-vllm/production-reproducibility/v1", "vllm_versions": versions,
                              "vllm_commit": sorted({str(r["profile"].get("vllm_commit")) for r in runs.values()}), "runs": {}}
    for name, run in runs.items():
        report["runs"][name] = {"variant": run["variant"], "seed": run["seed"], "max_num_seqs": run["max_num_seqs"],
                                "order": run.get("order"), "recording": run.get("recording"),
                                "peak_concurrency_achieved": run["peak_concurrency_achieved"],
                                "engine_steps": run["engine_steps"], "generate_s": run["timing"]["generate_s"],
                                "init_s": run.get("timing_init_s"),
                                "generated_positions": sum(r["num_output_tokens"] for r in run["records"]),
                                "finish_reasons": {fr: sum(1 for r in run["records"] if (r.get("client_finish_reason") or r["finish_reason"]) == fr)
                                                   for fr in {(r.get("client_finish_reason") or r["finish_reason"]) for r in run["records"]}}}

    # repetitions
    reps: dict[str, Any] = {}
    for key in sorted({n.rsplit("_r", 1)[0] for n in runs if "_c64_r" in n and not n.startswith("abort")}):
        names = [f"{key}_r{i}" for i in range(3) if f"{key}_r{i}" in runs]
        if len(names) < 2:
            continue
        base = runs[names[0]]
        reps[key] = {"runs": names, "vs_r0": {n: diff_runs(base, runs[n]) for n in names[1:]}}
        reps[key]["bit_identical_across_repetitions"] = all(d["all_identical"] for d in reps[key]["vs_r0"].values())
    report["repetitions"] = reps

    # concurrency caps
    caps: dict[str, Any] = {}
    for key in ("greedy", "temp_s1", "topk_topp_s1"):
        ref = runs.get(f"{key}_c64_r0")
        if ref is None:
            continue
        caps[key] = {}
        for cap in (1, 8, 32, 64):
            r = runs.get(f"{key}_c{cap}_r0")
            if r is None:
                continue
            caps[key][str(cap)] = {"peak_concurrency_achieved": r["peak_concurrency_achieved"], "engine_steps": r["engine_steps"],
                                   "generate_s": r["timing"]["generate_s"],
                                   "vs_cap64": diff_runs(ref, r) if cap != 64 else None}
        for extra in (f"{key}_c8_reversed", f"{key}_c8_shuffled"):
            if extra in runs:
                caps[key][extra] = {"peak_concurrency_achieved": runs[extra]["peak_concurrency_achieved"],
                                    "order": runs[extra]["order"], "vs_cap64": diff_runs(ref, runs[extra])}
        caps[key]["outputs_invariant_to_batching"] = all(
            (v.get("vs_cap64") is None) or v["vs_cap64"]["all_identical"] for v in caps[key].values())
    report["concurrency"] = caps

    # overhead (Table 4).  The first stochastic process of a batch pays the inference-time Triton JIT of
    # _topk_topp_kernel/_gumbel_sample_kernel (observed 8.6 s vs 1.5 s), so use the median over the
    # recorded repetitions and report every sample.
    ov: dict[str, Any] = {}
    for key in ("greedy_c64", "topk_topp_s1_c64", "temp_s1_c64"):
        stock = runs.get(f"{key}_norecord")
        recs = [runs[n] for n in (f"{key}_r0", f"{key}_r1", f"{key}_r2", f"{key}_r0_again") if n in runs]
        if recs and stock:
            samples = [r["timing"]["generate_s"] for r in recs]
            med = float(sorted(samples)[len(samples) // 2])
            ov[key] = {"recorded_generate_s": med, "recorded_generate_s_samples": samples,
                       "stock_generate_s": stock["timing"]["generate_s"],
                       "production_overhead_ratio": med / stock["timing"]["generate_s"],
                       "note": "recording = logprobs=1 + output ids kept + JSON; stock = same requests, nothing kept; "
                               "median over fresh-process repetitions (first stochastic process includes Triton JIT)"}
    report["overhead"] = ov

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print("COMPARE_DONE")
    for k, v in reps.items():
        print(f"  reps {k}: bit_identical={v['bit_identical_across_repetitions']}")
    for k, v in caps.items():
        print(f"  caps {k}: invariant={v['outputs_invariant_to_batching']} peaks={ {c: x['peak_concurrency_achieved'] for c, x in v.items() if isinstance(x, dict)} }")
    for k, v in ov.items():
        print(f"  overhead {k}: {v['production_overhead_ratio']:.3f}")


if __name__ == "__main__":
    main()
