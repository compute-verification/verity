#!/usr/bin/env python3
"""[REPLAY, additive] Regenerate the trusted-replay/ladder table (GUIDANCE §32.9A/.10)
from the pod records under docs/data/replay-baseline/. Writes ladder.json next to them.

    python scripts/replay_table.py --data docs/data/replay-baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# The original audited record's numbers (TA1 ta1eager, A100 80GB PCIe pod 48d33cb7c95b,
# driver 570.172.08), from docs/data/verified-backend/cost-integrity/b0-store-timings.json
# and fixtures/results/TA1-typed-b1-20260907/B0/replay.json.
TA1 = {
    "production_recorded_wall_s": 0.22638288233429193,
    "production_recording_off_mean_s": 0.22351685,
    "replay_run_s": 3.2699036011472344,
    "replay_forward_s": 1.0966054312884808,
    "replay_snapshot_s": 0.22910496965050697,
    "replay_emit_s": 2.1286681285127997,
    "weights_seconds": 0.9867311576381326,
    "commitments_seconds": 0.679677676409483,
}


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="docs/data/replay-baseline")
    args = ap.parse_args()
    d = Path(args.data)

    trusted = {}
    for f in sorted(d.glob("trusted-*.json")):
        doc = json.loads(f.read_text())
        cc = doc["consistency_check"]
        prof = doc.get("profile") or {}
        trusted[doc["exec"]] = {
            "median_wall_s": doc["pass_wall_s_median"],
            "walls_s": doc["pass_wall_s"],
            "engine_build_s": doc["engine_build_s"],
            "tokens_bitexact_vs_record": cc["tokens_bitexact"],
            "chosen_logprobs_bitexact_vs_record": cc["chosen_logprobs_bitexact"],
            "first_diverging_output_index": cc["first_diverging_output_index"],
            "max_abs_chosen_logprob_delta": cc["max_abs_chosen_logprob_delta"],
            "comparison_wall_s": cc["comparison_wall_s"],
            "run_to_run_bitexact": all(
                x["tokens_equal_first_pass"] and x["logprobs_equal_first_pass"]
                for x in doc["run_to_run_determinism"]),
            "profile_device_us": prof.get("total_self_device_us"),
            "profile_kernel_executions": prof.get("device_kernel_executions"),
            "profile_memcpy_us": prof.get("memcpy_memset_us"),
            "profile_memcpy_count": prof.get("memcpy_memset_count"),
        }
        # pre-divergence deltas (the tolerance a stock check would need before the argmax flips)
        div = cc["first_diverging_output_index"]
        n = div if div is not None else len(cc["replayed_chosen_logprobs"])
        pre = [abs(a - b) for a, b in
               zip(cc["replayed_chosen_logprobs"][:n], cc["recorded_chosen_logprobs"][:n])]
        trusted[doc["exec"]]["prediv_max_abs_chosen_logprob_delta"] = max(pre) if pre else 0.0

    pt = json.loads((d / "production-timings.json").read_text())
    rp = {r: json.loads((d / f"replay-{r}.json").read_text()) for r in ("nocap", "ref", "commit")}

    prod_recorded = pt["recorded_pass"]["wall_s"]
    stock = trusted["stock-pure"]["median_wall_s"]
    stock_inproc = trusted["stock-inproc"]["median_wall_s"]

    def rung(name, wall, note, label):
        return {
            "rung": name, "wall_s": wall,
            "x_vs_stock_pure": wall / stock,
            "x_vs_stock_inproc": wall / stock_inproc,
            "x_vs_profile_pass": wall / prod_recorded,
            "note": note, "evidence": label,
        }

    commit_total = (rp["commit"]["timing"]["run_s"] + rp["commit"]["timing"]["weights"]["seconds"]
                    + rp["commit"]["commitments"]["seconds"])
    ladder = [
        rung("R0 stock default vLLM (multiproc, LLM.generate)", stock,
             "compiled + CUDA graphs, cuBLAS/stock kernels, BI=0", "measured"),
        rung("R0b stock kernels, in-process driver (declared process flags)", stock_inproc,
             "V2 runner + in-proc engine + step driver; kernels unchanged", "measured"),
        rung("R1a + batch-invariant kernels (graphs kept)", trusted["bi-graphs"]["median_wall_s"],
             "BI triton GEMMs inside compiled/graphs; does NOT reproduce the record", "measured"),
        rung("R1b eager, stock kernels (boundary exposure without BI)", trusted["eager-stock-kernels"]["median_wall_s"],
             "graphs+fusion off; does NOT reproduce the record", "measured"),
        rung("R2 semantic profile = BI + eager, no recorder (trusted-replay config i)",
             trusted["bi-eager"]["median_wall_s"],
             "reproduces the record bit-exactly (tokens + chosen logprobs)", "measured"),
        rung("R2' + production recorder (StepTrace + outputs)",
             sum(t["wall_s"] for t in pt["recording_on"]) / len(pt["recording_on"]),
             f"recording on/off ratio {pt['production_overhead_ratio']:.3f}", "measured"),
        rung("R3 + forced-trace replay driver, no capture", rp["nocap"]["timing"]["run_s"],
             "enable_trace_replay + trace_decode_token_ids; no interception", "measured"),
        rung("R4 + interior capture, hash-only", rp["ref"]["timing"]["run_s"],
             f"forward {rp['ref']['timing']['capture']['forward_s']:.3f} snapshot "
             f"{rp['ref']['timing']['capture']['snapshot_s']:.3f} emit(hash) "
             f"{rp['ref']['timing']['capture']['emit_s']:.3f}", "measured"),
        rung("R5 + interior capture, values stored", rp["commit"]["timing"]["run_s"],
             f"forward {rp['commit']['timing']['capture']['forward_s']:.3f} snapshot "
             f"{rp['commit']['timing']['capture']['snapshot_s']:.3f} emit(store) "
             f"{rp['commit']['timing']['capture']['emit_s']:.3f}", "measured"),
        rung("R6 + commitments (10 Merkle trees) + weights tree (hash mode)", commit_total,
             f"trees {rp['commit']['commitments']['seconds']:.3f}s, weights "
             f"{rp['commit']['timing']['weights']['seconds']:.3f}s on top of R5", "measured"),
    ]

    out = {
        "pod": "replay-a100 ux6af2w0s5avo0 (A100 80GB PCIe, driver 580.159.04, cu129 wheel 7aa52ac7)",
        "workload": "TA1 B0: 1 request, 32 prompt tokens, 8 greedy tokens (dc159809-...)",
        "denominators": {
            "stock_pure_median_s": stock,
            "stock_inproc_median_s": stock_inproc,
            "profile_pass_recorded_s": prod_recorded,
            "ta1_original_recorded_s": TA1["production_recorded_wall_s"],
        },
        "trusted": trusted,
        "ladder": ladder,
        "production_timings": {
            "recording_off": [t["wall_s"] for t in pt["recording_off"]],
            "recording_on": [t["wall_s"] for t in pt["recording_on"]],
            "overhead_ratio": pt["production_overhead_ratio"],
            "engine_build_s": pt["engine_build_s"],
        },
        "ta1_original": TA1,
    }
    (d / "ladder.json").write_text(json.dumps(out, indent=1) + "\n")
    for r in ladder:
        print(f"{r['rung']:<70} {r['wall_s']:>8.4f}s  {r['x_vs_stock_pure']:>7.1f}x stock  {r['x_vs_profile_pass']:>6.2f}x pass")


if __name__ == "__main__":
    main()
