"""Re-price A-GKR at the v1 and v2 checker shapes with the a-fusion measured buckets (laptop side, no torch):

    uv run --package verity-numerical python -m backends.gkr.packed.reprice_v2 \
        --sumcheck backends/numerical/reports/a_fusion/bench_fused_k3.json \
        --logup-v1 backends/numerical/reports/a_fusion/logup_graph_v1.json \
        --logup-v2 backends/numerical/reports/a_fusion/logup_graph_v2.json \
        --out backends/numerical/reports/a_kernels_h100_v2

For each variant (``VERITY_CHECKER_VARIANT`` in a subprocess) ``explore.tensor`` prices design A on the H100 at 0.172
utilisation; the ``gkr_checker`` matmul + ``gkr_hadamards`` SIMT buckets are replaced by the measured fused sumcheck
(the 2^26 / 2^28 k=3 linear fit applied to the variant's layer tables ``units x 2^s_in``), and the ``logup_*`` buckets
by the measured CUDA-graphed logUp (v1: the one 2^27 instance scaled to the model's leaves; v2: the ten measured table
instances plus the model's 11th (TNORM_HI, 2^21) taken as the measured TNORM instance).  Everything else stays modelled.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

UNITS = 393_216
B = 4096
K = 1536

_MODEL_SCRIPT = r"""
import json, sys
from verity_numerical.explore import tensor as tx
from verity_numerical.security import accounting as acc, census
fs = tx.FIELDS["babybear"]; gpu = tx.GPUS["H100"]; hr = tx.HASH_RATES["blake3_simt"]["bytes_per_s"]
out = {"variant": census.CHECKER_VARIANT, "shape": list(acc.CHECKER_SHAPE_A)}
simt_rate = gpu.simt_field_mults_per_s / fs.simt_mult_vs_goldilocks
sfc = tx._small_field(fs); d, lig = tx._ligero(-128, fs, "A", tx.B_TARGET); tables = sfc.tables()
cand = acc.CandidateA(K=tx.K_TARGET, B=tx.B_TARGET, trace_field=fs.option.trace, challenge_field=fs.option.trace.extension(d), ligero=lig,
                      hash_queries=tx.fl.HASH_QUERIES, tables=tables, inputs_per_unit=sfc.columns_committed_a,
                      query_wires_per_unit=sfc.queries_on_computed_wires, depth=acc.DepthOptions(commit_query_wires=True))
out["tables"] = [{"name": tb.name, "rows": tb.rows, "q_per_unit": tb.queries_per_unit, "n": acc._shards(cand, tb)[2], "shards": acc._shards(cand, tb)[0]} for tb in tables]
out["unit"] = {"mul": sfc.mul, "columns_committed_a": sfc.columns_committed_a}
out["designs"] = {}
for dname in ("A", "A-hintfree-depth630"):
    p = tx.price_design(tx.DESIGNS[dname], fs, -128)
    base = p.times(gpu, "H100", 0.172, hr, "rs_matrix_ntt", fs)
    model_gkr = p.matmul["gkr_checker"] / (gpu.peak_macs_per_s * 0.172) + p.simt_field_mults["gkr_hadamards"] / simt_rate
    model_logup = (p.matmul["logup_fractional_gkr"] / (gpu.peak_macs_per_s * 0.172)
                   + (p.simt_field_mults["logup_hadamards"] + p.simt_field_mults["logup_tree_build"]) / simt_rate)
    out["designs"][dname] = {"t.total": base["t.total"], "overhead": base["overhead.vs_native_peak"], "t.gkr_checker": model_gkr,
                             "t.logup": model_logup, "modelled": base["t.modelled"]}
json.dump(out, sys.stdout)
"""


def model(variant: str) -> dict:
    env = dict(os.environ, VERITY_CHECKER_VARIANT=variant)
    r = subprocess.run([sys.executable, "-c", _MODEL_SCRIPT], capture_output=True, text=True, env=env, check=True)
    return json.loads(r.stdout)


def overhead(seconds_per_vu: float) -> float:
    sys.path.insert(0, "backends/numerical/python")
    from verity_numerical.bench import contract

    return contract.overhead(seconds_per_vu, K)


def sumcheck_fit(sc: dict, k: int, tag: str) -> tuple[float, float]:
    pts = sorted((1 << r["log2_t"], r[tag]["device_ms"] / 1e3) for r in sc["results"] if r["k"] == k)
    (t0, y0), (t1, y1) = pts[0], pts[-1]
    a = (y1 - y0) / (t1 - t0)
    return a, y0 - a * t0


def layer_seconds(fit: tuple[float, float], tables: list[int]) -> float:
    a, b = fit
    return sum(a * t + b for t in tables)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sumcheck", type=Path, required=True)
    ap.add_argument("--logup-v1", type=Path, required=True)
    ap.add_argument("--logup-v2", type=Path, required=True)
    ap.add_argument("--logup-v2-concurrent", type=Path, default=None, help="logup_graph_bench.json of the --modes concurrent run")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("backends/numerical/reports/a_kernels_h100_v2"))
    a = ap.parse_args(argv)
    sc = json.loads(a.sumcheck.read_text())
    lu1 = json.loads(a.logup_v1.read_text())
    lu2 = json.loads(a.logup_v2.read_text())
    fits = {"fused": sumcheck_fit(sc, a.k, "fused"), "baseline": sumcheck_fit(sc, a.k, "baseline")}
    direct = next(r for r in sc["results"] if r["log2_t"] == 28 and r["k"] == a.k)

    v1t = lu1["suites"]["v1"]["tables"][0]
    v2 = lu2["suites"]["v2"]
    v2s = lu2["suites"].get("v2-splitu")
    result = {"source": {"sumcheck": str(a.sumcheck), "logup_v1": str(a.logup_v1), "logup_v2": str(a.logup_v2), "commit": sc.get("commit")},
              "labels": {"proof_class": "ARITHMETIC_DIAGNOSTIC", "authentication": "excluded", "security.target": -128, "B": B, "K": K,
                         "hardware": sc.get("hardware"), "torch": sc.get("torch")},
              "sumcheck": {"k": a.k, "fit_seconds_per_element_and_intercept": fits,
                           "direct_2^28": {tag: direct[tag]["device_ms"] for tag in ("baseline", "fused")},
                           "direct_2^28_x_hbm_floor": direct["fused"]["x_hbm_floor"], "direct_2^28_frac_int8_peak": direct["fused"]["frac_int8_peak"],
                           "pieces_2^28": direct.get("pieces")},
              "variants": {}}
    for variant, lu_meas in (("v1", None), ("v2", None)):
        m = model(variant)
        shape = m["shape"]
        padded = [UNITS << L["s_in"] for L in shape]
        unpadded = [UNITS * L["width"] for L in shape]
        sc_rows = {}
        for tag, fit in fits.items():
            sc_rows[tag] = {"batch_seconds.padded": layer_seconds(fit, padded), "batch_seconds.unpadded": layer_seconds(fit, unpadded)}
        extra_shapes = {}
        if variant == "v2":
            # the a-v2 exporter's circuit (PROTOCOL.md 14.4): two checker layers (s_out=5, s_in=9, assertions), (9, 9)
            exp_tables = [UNITS << 9, UNITS << 9]
            extra_shapes["exporter_2_layers_s_in_9"] = {tag: layer_seconds(fit, exp_tables) for tag, fit in fits.items()}
        # logUp
        if variant == "v1":
            model_leaves = sum(t["shards"] * 2 ** t["n"] for t in m["tables"])
            meas_leaves = v1t["leaves"]
            scale = model_leaves / meas_leaves
            lg = {"measured_instance_ms": {"level": v1t["level"]["device_ms"], "level_wall": v1t["level"]["wall_ms"],
                                           "round": v1t["round"]["device_ms"], "round_wall": v1t["round"]["wall_ms"],
                                           "eager": v1t.get("eager", {}).get("device_ms")},
                  "measured_leaves": meas_leaves, "model_leaves": model_leaves, "leaf_scale": scale,
                  "batch_seconds": {"level": v1t["level"]["device_ms"] / 1e3 * scale, "round": v1t["round"]["device_ms"] / 1e3 * scale},
                  "note": "one 2^27-leaf instance (306 lookups/unit, 65,536-row table) scaled to the model's leaves over its 5 tables"}
        else:
            per_table = {t["table"]: {"n": t["n"], "leaves": t["leaves"], "rows": t["rows"], "lookups_per_unit": t["lookups_per_unit"],
                                      "level_ms": t["level"]["device_ms"], "level_wall_ms": t["level"]["wall_ms"], "round_ms": t["round"]["device_ms"],
                                      "tree_ms": t["level"]["tree_ms"], "table_share_of_leaves": t["table_share_of_leaves"]} for t in v2["tables"]}
            tnorm = per_table["TNORM"]
            tot_level = sum(t["level_ms"] for t in per_table.values()) + tnorm["level_ms"]        # + TNORM_HI (model's 11th table)
            tot_round = sum(t["round_ms"] for t in per_table.values()) + tnorm["round_ms"]
            lg = {"per_table": per_table, "measured_10_tables_ms": {"level": v2["totals"]["level"]["device_ms"], "round": v2["totals"]["round"]["device_ms"],
                                                                  "level_wall": v2["totals"]["level"]["wall_ms"]},
                  "tnorm_hi_added_as_tnorm_ms": tnorm["level_ms"],
                  "batch_seconds": {"level": tot_level / 1e3, "round": tot_round / 1e3},
                  "sequential_instances_note": "'level'/'round' = the instances one after another (each pays its ~20 latency-bound small levels)",
                  "shift": {"committed_ms": per_table["SHIFT"]["level_ms"], "committed_tree_ms": per_table["SHIFT"]["tree_ms"],
                            "committed_table_share_of_leaves": per_table["SHIFT"]["table_share_of_leaves"]},
                  "note": "ten table instances measured separately (checker_min 3.1 tables); each instance pays ~20 latency-bound small levels"}
            if a.logup_v2_concurrent is not None:
                cc = json.loads(a.logup_v2_concurrent.read_text())["suites"]["v2"]["totals"]["concurrent"]
                n_inst = cc["streams"]
                lg["concurrent"] = {"measured_10_tables_ms": cc["device_ms"], "wall_ms": cc["wall_ms"], "streams": n_inst,
                                    "per_instance_device_ms_in_parallel": cc["per_instance_device_ms_in_parallel"],
                                    "scaled_to_11_tables_ms": cc["device_ms"] * (n_inst + 1) / n_inst,
                                    "note": "the ten instances on ten CUDA streams (level graphs, protocol unchanged); TNORM_HI as an 11th "
                                            "instance taken pro rata"}
                lg["batch_seconds"]["concurrent"] = cc["device_ms"] * (n_inst + 1) / n_inst / 1e3
            if v2s:
                sp_tabs = {t["table"]: t for t in v2s["tables"]}
                lg["shift"]["split_u_ms"] = {n: sp_tabs[n]["level"]["device_ms"] for n in ("SHIFT_u_lo", "SHIFT_u_hi") if n in sp_tabs}
                lg["shift"]["split_u_total_ms"] = sum(lg["shift"]["split_u_ms"].values())
                lg["split_u_suite_total_ms"] = v2s["totals"]["level"]["device_ms"]
        rows = {}
        for dname, dm in m["designs"].items():
            g = {"model@0.172": {"overhead": dm["overhead"], "t.total": dm["t.total"], "t.gkr_checker": dm["t.gkr_checker"], "t.logup": dm["t.logup"]}}
            for sc_tag in ("baseline", "fused"):
                for lg_tag in lg["batch_seconds"]:
                    meas_sc = sc_rows[sc_tag]["batch_seconds.padded"]
                    meas_lg = lg["batch_seconds"][lg_tag]
                    total = dm["t.total"] - dm["t.gkr_checker"] - dm["t.logup"] + meas_sc + meas_lg
                    g[f"measured sumcheck={sc_tag} logup={lg_tag}"] = {
                        "overhead": overhead(total / B), "t.total": total, "t.gkr_checker": meas_sc, "t.logup": meas_lg,
                        "share.sumcheck": meas_sc / total, "share.logup": meas_lg / total, "share.modelled": 1 - (meas_sc + meas_lg) / total}
            rows[dname] = g
        result["variants"][variant] = {"shape": shape, "layer_tables_padded": padded, "layer_tables_unpadded": unpadded, "unit": m["unit"],
                                       "tables": m["tables"], "sumcheck": sc_rows, "sumcheck_extra_shapes": extra_shapes, "logup": lg, "repriced": rows}
    result["omitted"] = [
        "Ligero encoding, row combination, hashing, hint generation, PRG: modelled (explore.tensor at 0.172 utilisation)",
        ("authentication (excluded), zero-knowledge (none), Fiat-Shamir host round trips: the level-graph numbers use coins independent of the "
         "transcript; the per-round-graph numbers (Fiat-Shamir compatible) are the 'round' rows"),
        "sumcheck: coins drawn ahead, the whole layer as one CUDA graph; the fold+Hadamard fusion is measured slower and left off",
        ("v2 logUp: ten instances from checker_min 3.1 (T_range16 at 10/unit; the model has 14/unit and an 11th table TNORM_HI, added as a second TNORM); "
         "SHIFT is committed per proof (3.54M multiplicity elements + 3.54M table leaves) -- the multiplicity column's Ligero cost is inside the modelled encoding"),
        "v2 sumcheck: the census layer shape (4 layers: assert + depths 3, 2, 1 at s_in 9, 9, 9, 8); the a-v2 exporter's 2-layer shape is the extra row",
        "relation checker not assembled on the GPU; kernels on random tables",
    ]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    for variant, v in result["variants"].items():
        print(f"== {variant}: sumcheck padded fused {v['sumcheck']['fused']['batch_seconds.padded']*1e3:.1f} ms (baseline "
              f"{v['sumcheck']['baseline']['batch_seconds.padded']*1e3:.1f}); logUp level {v['logup']['batch_seconds']['level']*1e3:.1f} ms, "
              f"round {v['logup']['batch_seconds']['round']*1e3:.1f} ms")
        for dname, g in v["repriced"].items():
            for label, r in g.items():
                print(f"  {dname:22} {label:44} overhead {r['overhead']:.3g}  t.total {r['t.total']:.3f} s  sumcheck {r['t.gkr_checker']*1e3:.0f} ms  logUp {r['t.logup']*1e3:.0f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
