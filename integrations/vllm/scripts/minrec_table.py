#!/usr/bin/env python
"""MINREC §3/§4 tables from a `minimal_recording_ab.py` campaign directory (lane MINREC, 2026-09-09).

    python scripts/minrec_table.py results/MINREC/MINREC-B0-20260909T1930Z [--json out.json]

Reads <dir>/{stock-hash,profile-hash,profile-file,profile-none}/passes.jsonl, pairs arms within each process by
repetition (paired_ci of the harness), and prints:
  * per-arm wall (mean, SD, N), paired delta vs the process base (A or B) with 95 % t-CI, bytes copied/hashed/written,
    per-request and per-token deltas;
  * c_rec_min(a) (= A' − A), c_rec_min(b) (= C:block − B), c_rec_layer, c_rec_diag (= D − B) as ratios of the NATIVE pass
    (arm A, stock) and of the profile pass (arm B) — the cost model's two denominators (cost-result-format.md §2);
  * per-row / per-byte rates and their dc-1m extrapolation (rows and bytes rules; ANALYTICAL sha256 floor) against the
    dc-1m native denominator N = 198,532 GPU-s (docs/data/frontier/dc-1m/cost_model.json).
Every number here is MEASURED on the campaign's pod except the extrapolations, which are labelled.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from minimal_recording_ab import paired_ci  # noqa: E402

DC1M = {"native_gpu_s": 198532.21410921024, "positions": 999_999, "c_rec_exported": 0.05830481017206791,
        "boundary": {"whole": {"rows_per_pos": 0, "bytes_per_pos": 0}, "layer": {"rows_per_pos": 189, "bytes_per_pos": 1.55e6},
                     "block": {"rows_per_pos": 658, "bytes_per_pos": 7.22e6}, "operator": {"rows_per_pos": 3574, "bytes_per_pos": 24.3e6}},
        "source": "first-stage-boundary.md §3.2 (ANALYTICAL formula census); dc-1m/cost_model.json denominator.central.native_gpu_s"}
SHA256_GBPS = 1.86  # CPROF asymptote (cost_model.v4 cprof_beta_us_per_kb note), one EPYC thread


def load(run_dir: Path):
    rows = [json.loads(l) for l in (run_dir / "passes.jsonl").read_text().splitlines() if l.strip()]
    return [r for r in rows if not r.get("warmup") and not r.get("discard")]


def by_arm(rows):
    out: dict[str, dict[int, dict]] = {}
    for r in rows:
        out.setdefault(r["arm"], {})[r["rep"]] = r
    return out


def arm_stats(arms, base):
    reps = sorted(arms[base])
    res = {}
    for arm, byrep in arms.items():
        common = [k for k in reps if k in byrep]
        x = [arms[base][k]["wall_s"] for k in common]
        y = [byrep[k]["wall_s"] for k in common]
        rec = lambda key: statistics.mean((byrep[k].get("recorder") or {}).get(key, 0) or 0 for k in common)  # noqa: E731
        toks = statistics.mean(byrep[k]["prompt_tokens"] + byrep[k]["generated_tokens"] for k in common)
        gen = statistics.mean(byrep[k]["generated_tokens"] for k in common)
        st = {"n": len(common), "wall_mean": statistics.mean(y), "wall_sd": statistics.stdev(y), "tokens": toks, "generated": gen,
              "rows": rec("rows"), "leaves": rec("leaves"), "bytes_copied": rec("bytes_copied"), "bytes_hashed": rec("bytes_hashed"),
              "bytes_written": rec("bytes_written"), "record_bytes": rec("production_record_bytes"), "steptrace_bytes": rec("steptrace_bytes"),
              "gpu_util": statistics.mean((byrep[k].get("telemetry") or {}).get("gpu_util_mean", 0) or 0 for k in common),
              "sm_clock": statistics.mean((byrep[k].get("telemetry") or {}).get("sm_clock_mean", 0) or 0 for k in common),
              "tokens_identical": all(byrep[k]["tokens_digest"] == arms[base][k]["tokens_digest"] for k in common)}
        t = [(byrep[k].get("recorder") or {}).get("timing_s") for k in common]
        if t and t[0]:
            st["recorder_timing"] = {kk: statistics.mean(x.get(kk, 0) for x in t) for kk in t[0]}
        if arm != base:
            st["paired"] = paired_ci(x, y)
        res[arm] = st
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    C = Path(a.campaign)
    runs = {}
    for name, base in (("stock-hash", "A"), ("profile-hash", "B"), ("profile-file", "B"), ("profile-none", "B")):
        if (C / name / "passes.jsonl").exists():
            runs[name] = arm_stats(by_arm(load(C / name)), base)
    nreq = 4
    A = runs["stock-hash"]["A"]["wall_mean"]
    B = runs["profile-hash"]["B"]["wall_mean"]
    out = {"campaign": str(C), "runs": runs, "native_pass_s": A, "profile_pass_s": B, "profile_tax_ratio": B / A}

    print(f"# {C.name}: native pass A = {A:.4f} s (N={runs['stock-hash']['A']['n']}, SD {runs['stock-hash']['A']['wall_sd']:.4f}); "
          f"profile pass B = {B:.4f} s (N={runs['profile-hash']['B']['n']}, SD {runs['profile-hash']['B']['wall_sd']:.4f}); profile tax B/A = {B / A:.2f}× "
          f"(cross-process, same GPU)\n")
    print("| process / sink | arm | N | wall s (mean ± SD) | Δ vs base s [95 % CI] | ratio [95 % CI] | Δ / request ms | Δ / token µs | rows (leaves) | bytes copied | bytes hashed | bytes written | GPU util % | tokens = base |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, res in runs.items():
        for arm, st in res.items():
            p = st.get("paired")
            d = f"{p['mean_diff']:+.4f} [{p['ci95_diff'][0]:+.4f}, {p['ci95_diff'][1]:+.4f}]" if p else "base"
            r = f"{p['ratio']:.3f} [{p['ci95_ratio_delta'][0]:.3f}, {p['ci95_ratio_delta'][1]:.3f}]" if p else "1"
            dr = f"{1000 * p['mean_diff'] / nreq:+.2f}" if p else "—"
            dt = f"{1e6 * p['mean_diff'] / st['tokens']:+.1f}" if p else "—"
            print(f"| {name} | {arm} | {st['n']} | {st['wall_mean']:.4f} ± {st['wall_sd']:.4f} | {d} | {r} | {dr} | {dt} | {st['rows']:.0f} ({st['leaves']:.0f}) | "
                  f"{st['bytes_copied']:,.0f} | {st['bytes_hashed']:,.0f} | {st['bytes_written']:,.0f} | {st['gpu_util']:.1f} | {'yes' if st['tokens_identical'] else 'NO'} |")
    print()
    # cost-model units
    ph = runs["profile-hash"]
    sh = runs["stock-hash"]
    cm = {}

    def unit(label, delta, ci, rows, bytes_):
        cm[label] = {"delta_s": delta, "ci95_s": ci, "ratio_native": delta / A, "ci95_ratio_native": [ci[0] / A, ci[1] / A],
                     "ratio_profile": delta / B, "ci95_ratio_profile": [ci[0] / B, ci[1] / B],
                     "s_per_row": (delta / rows) if rows else None, "ns_per_byte": (1e9 * delta / bytes_) if bytes_ else None}

    pA = sh["A'"]["paired"]
    unit("c_rec_min(a) whole-request = A' − A (production recorder, stock)", pA["mean_diff"], pA["ci95_diff"], 0, 0)
    pB = ph["B'"]["paired"]
    unit("production recorder on the profile = B' − B", pB["mean_diff"], pB["ci95_diff"], 0, 0)
    for arm, lab in (("C:block", "c_rec_min(b) P3/cu=block = C:block − B"), ("C:layer", "c_rec_min(layer) = C:layer − B"), ("D", "c_rec_diag = D − B (capture.py hash-only)")):
        p = ph[arm]["paired"]
        unit(lab, p["mean_diff"], p["ci95_diff"], ph[arm]["rows"], ph[arm]["bytes_copied"])
    for run, lab in (("profile-none", "rung copy-only"), ("profile-file", "rung copy+hash+write")):
        if run in runs:
            for arm in ("C:block", "C:layer"):
                p = runs[run][arm]["paired"]
                unit(f"{arm} {lab} = {arm} − B ({run})", p["mean_diff"], p["ci95_diff"], runs[run][arm]["rows"], runs[run][arm]["bytes_copied"])
    out["cost_model_units"] = cm
    print("| quantity (MEASURED, this pod) | Δ s [95 % CI] | × native pass (A) [CI] | × profile pass (B) [CI] | µs / row | ns / byte |")
    print("|---|---|---|---|---|---|")
    for k, v in cm.items():
        pr = f"{1e6 * v['s_per_row']:.1f}" if v["s_per_row"] else "—"
        pb = f"{v['ns_per_byte']:.2f}" if v["ns_per_byte"] else "—"
        print(f"| {k} | {v['delta_s']:+.4f} [{v['ci95_s'][0]:+.4f}, {v['ci95_s'][1]:+.4f}] | {v['ratio_native']:+.3f} [{v['ci95_ratio_native'][0]:+.3f}, {v['ci95_ratio_native'][1]:+.3f}] | "
              f"{v['ratio_profile']:+.4f} [{v['ci95_ratio_profile'][0]:+.4f}, {v['ci95_ratio_profile'][1]:+.4f}] | {pr} | {pb} |")
    print()
    # dc-1m extrapolation
    N = DC1M["native_gpu_s"]
    P = DC1M["positions"]
    ex = {}
    blk = cm["c_rec_min(b) P3/cu=block = C:block − B"]
    lay = cm["c_rec_min(layer) = C:layer − B"]
    dia = cm["c_rec_diag = D − B (capture.py hash-only)"]
    tok_a = pA["mean_diff"] / sh["A'"]["tokens"]
    tok_a_hi = pA["ci95_diff"][1] / sh["A'"]["tokens"]
    ex["whole: per-token rule (A' − A per token × 1e6 tokens) / N"] = {"value": tok_a * P / N, "ci95_high": tok_a_hi * P / N, "label": "EXTRAPOLATED from a measurement at noise"}
    for plan, c in (("block", blk), ("layer", lay)):
        b = DC1M["boundary"][plan]
        ex[f"{plan}: rows rule ({1e6 * c['s_per_row']:.1f} µs/row × {b['rows_per_pos']} rows/pos × P) / N"] = {"value": c["s_per_row"] * b["rows_per_pos"] * P / N, "label": "EXTRAPOLATED (this implementation)"}
        ex[f"{plan}: bytes rule ({c['ns_per_byte']:.2f} ns/B × {b['bytes_per_pos'] / 1e6:.2f} MB/pos × P) / N"] = {"value": c["ns_per_byte"] * 1e-9 * b["bytes_per_pos"] * P / N, "label": "EXTRAPOLATED (this implementation)"}
        ex[f"{plan}: ANALYTICAL floor, sha256 at {SHA256_GBPS} GB/s one thread"] = {"value": b["bytes_per_pos"] * P / (SHA256_GBPS * 1e9) / N, "label": "ANALYTICAL lower bound (bytes must be hashed once)"}
        ex[f"{plan}: ANALYTICAL floor, 16 host threads (one per TP rank)"] = {"value": b["bytes_per_pos"] * P / (16 * SHA256_GBPS * 1e9) / N, "label": "ANALYTICAL lower bound"}
    ex["diagnostic: rows rule (D − B per leaf × 3,574 rows/pos × P) / N"] = {"value": dia["s_per_row"] * DC1M["boundary"]["operator"]["rows_per_pos"] * P / N, "label": "EXTRAPOLATED"}
    ex["exported c_rec (dc-1m/cost_model.json, rows rule from the production-recorder delta)"] = {"value": DC1M["c_rec_exported"], "label": "EXTRAPOLATED (v2 export)"}
    out["dc1m"] = ex
    print(f"| dc-1m always-on recording term (× N = {N:,.0f} GPU-s) | value | label |")
    print("|---|---|---|")
    for k, v in ex.items():
        hi = f" (95 % high {v['ci95_high']:.2e})" if "ci95_high" in v else ""
        print(f"| {k} | {v['value']:.3e}{hi} | {v['label']} |")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
