"""Re-price A-GKR with the two measured GPU buckets (laptop side, no torch):

    python -m backends.gkr.packed.reprice_a --sumcheck results/r20260922-065412-73be/bench.json \
        --logup results/<run>/logup_bench.json --out backends/numerical/reports/a_kernels_h100

`explore.tensor` design A on the H100 at 0.172 utilisation, with (1) the `gkr_checker` matmul bucket + its SIMT Hadamard
bucket replaced by the measured six-layer packed sumcheck (direct 2^28 layer tables) and (2) the `logup_fractional_gkr`
matmul bucket + the `logup_hadamards` + `logup_tree_build` SIMT buckets replaced by the measured logUp GKR (one 2^27-leaf
instance, scaled to the model's leaf count over all five tables).  Everything else (Ligero encoding, row combination,
hashing, hint generation, PRG) stays modelled.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sumcheck", type=Path, required=True, help="bench.json of the packed sumcheck run (layer model)")
    ap.add_argument("--logup", type=Path, required=True, help="logup_bench.json of the logUp run")
    ap.add_argument("--sumcheck-k", type=int, default=3)
    ap.add_argument("--logup-k", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("backends/numerical/reports/a_kernels_h100"))
    a = ap.parse_args(argv)
    sys.path.insert(0, "backends/numerical/python")
    from verity_numerical.bench import contract
    from verity_numerical.explore import tensor as tx
    from verity_numerical.security import accounting as acc

    sc = json.loads(a.sumcheck.read_text())
    lu = json.loads(a.logup.read_text())
    L = sc["layers"][str(a.sumcheck_k)]
    lrec = next(r for r in lu["results"] if r["k"] == a.logup_k)
    sc_run = json.loads(a.sumcheck.with_name("result.json").read_text())["run_id"] if a.sumcheck.with_name("result.json").exists() else None
    lu_run = json.loads(a.logup.with_name("result.json").read_text())["run_id"] if a.logup.with_name("result.json").exists() else None

    fs = tx.FIELDS["babybear"]
    hr = tx.HASH_RATES["blake3_simt"]["bytes_per_s"]
    gpu = tx.GPUS["H100"]
    # the model's leaves over all tables (the measurement is one 2^27-leaf instance)
    sfc = tx._small_field(fs)
    d, lig = tx._ligero(-128, fs, "A", tx.B_TARGET)
    tables = sfc.tables()
    cand = acc.CandidateA(K=tx.K_TARGET, B=tx.B_TARGET, trace_field=fs.option.trace, challenge_field=fs.option.trace.extension(d), ligero=lig,
                          hash_queries=tx.fl.HASH_QUERIES, tables=tables, inputs_per_unit=sfc.columns_committed_a,
                          query_wires_per_unit=sfc.queries_on_computed_wires, depth=acc.DepthOptions(commit_query_wires=True))
    model_leaves = sum(acc._shards(cand, tb)[0] * 2 ** acc._shards(cand, tb)[2] for tb in tables)
    meas_leaves = 1 << lu["n"]
    leaf_scale = model_leaves / meas_leaves * cand.lookup_repetitions
    logup_meas = lrec["total_ms"] / 1e3
    logup_scaled = logup_meas * leaf_scale
    tail_ms = lrec["tail_levels_ms"]
    # host launch overhead per round: the slope of level time vs level index over the smallest levels (device work there is ~0)
    lv = {int(k): v for k, v in lrec["levels_ms"].items()}
    slope_ms = (lv[12] - lv[4]) / 8 if 12 in lv and 4 in lv else 0.0
    n_rounds = lu["n"] * (lu["n"] - 1) // 2
    logup_device_est = max(0.0, logup_meas - slope_ms * n_rounds / 1e3) * leaf_scale

    rows = {}
    for dname in ("A", "A-hintfree-depth630"):
        p = tx.price_design(tx.DESIGNS[dname], fs, -128)
        base = p.times(gpu, "H100", 0.172, hr, "rs_matrix_ntt", fs)
        simt_rate = gpu.simt_field_mults_per_s / fs.simt_mult_vs_goldilocks
        model_gkr = p.matmul["gkr_checker"] / (gpu.peak_macs_per_s * 0.172) + p.simt_field_mults["gkr_hadamards"] / simt_rate
        model_logup = (p.matmul["logup_fractional_gkr"] / (gpu.peak_macs_per_s * 0.172)
                       + (p.simt_field_mults["logup_hadamards"] + p.simt_field_mults["logup_tree_build"]) / simt_rate)
        g = {"model@0.172": {"overhead": base["overhead.vs_native_peak"], "t.total": base["t.total"], "t.gkr_checker": model_gkr, "t.logup": model_logup}}
        for tag in ("padded", "unpadded"):
            meas_sc = L[f"batch_seconds.{tag}"]
            total = base["t.total"] - model_gkr - model_logup + meas_sc + logup_scaled
            g[f"measured ({tag} sumcheck)"] = {"overhead": contract.overhead(total / tx.B_TARGET, tx.K_TARGET), "t.total": total,
                                              "t.gkr_checker": meas_sc, "t.logup": logup_scaled,
                                              "share.sumcheck": meas_sc / total, "share.logup": logup_scaled / total,
                                              "share.modelled": 1 - (meas_sc + logup_scaled) / total}
            total_e = base["t.total"] - model_gkr - model_logup + meas_sc + logup_device_est
            g[f"estimate ({tag}, logUp device time)"] = {"overhead": contract.overhead(total_e / tx.B_TARGET, tx.K_TARGET), "t.total": total_e,
                                                        "t.gkr_checker": meas_sc, "t.logup": logup_device_est,
                                                        "note": f"logUp with the host launch overhead removed ({slope_ms:.2f} ms/round x {n_rounds} rounds, "
                                                                "from the slope of the smallest levels): an ESTIMATE of what CUDA graphs / a C++ launcher would leave"}
        rows[dname] = g
    out = {"source": {"sumcheck_bench": str(a.sumcheck), "sumcheck_run_id": sc_run, "logup_bench": str(a.logup), "logup_run_id": lu_run,
                      "commit": sc.get("software", {}).get("commit")},
           "labels": {"proof_class": "ARITHMETIC_DIAGNOSTIC", "authentication": "excluded", "security.target": -128,
                      "hardware": sc.get("hardware"), "software": sc.get("software")},
           "sumcheck": {"k": a.sumcheck_k, "layer_model": L, "direct_2^28": [r for r in sc["results"] if r["log2_t"] == 28]},
           "logup": {"k": a.logup_k, "measured": lrec, "shape": {k: lu[k] for k in ("units", "lookups", "Nw", "Nt", "n")},
                     "model_leaves": model_leaves, "measured_leaves": meas_leaves, "leaf_scale": leaf_scale,
                     "seconds_per_batch_scaled": logup_scaled, "us_per_vu_scaled": logup_scaled * 1e6 / tx.B_TARGET,
                     "launch_overhead_ms_per_round": slope_ms, "rounds": n_rounds, "seconds_per_batch_device_estimate": logup_device_est,
                     "launch_overhead_note": f"{tail_ms:.0f} ms of the {lrec['total_ms']:.0f} ms are the {lu['n'] - 7} smallest levels "
                                             "(host launch overhead of ~0.6 ms per round, not device work)"},
           "repriced": rows,
           "omitted": ["Ligero encoding (modelled)", "row combination (modelled)", "hashing (modelled)", "hint generation (modelled)",
                       "relation checker not assembled", "logUp: the four small tables scaled from the 16-bit instance by leaf count",
                       "Fiat-Shamir host round trips (coins were independent of the transcript)"]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(f"logUp measured {logup_meas*1e3:.1f} ms x {leaf_scale:.3f} = {logup_scaled*1e3:.1f} ms per batch = {logup_scaled*1e6/tx.B_TARGET:.1f} us/VU; "
          f"launch overhead {slope_ms:.2f} ms/round x {n_rounds} -> device estimate {logup_device_est*1e3:.1f} ms")
    for dname, g in rows.items():
        for label, v in g.items():
            print(f"{dname:22} {label:28} overhead {v['overhead']:.3g}  t.total {v['t.total']:.3f} s  sumcheck {v['t.gkr_checker']*1e3:.0f} ms  logUp {v['t.logup']*1e3:.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
