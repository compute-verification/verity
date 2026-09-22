"""Re-price Candidate B at the checker-v2 witness shape with every bucket measured on the RTX 4090 (b-v2 lane).

    uv run --frozen --package verity-numerical python backends/direct/encode/reprice_v2.py \
        --bench backends/direct/encode/results_v2_<run>.json --v1 backends/numerical/reports/b_measured_4090.json \
        --out backends/numerical/reports/b_measured_4090_v2.json --md backends/numerical/reports/b_measured_4090_v2.md

Same method as ``reprice_measured.py``: ``explore.tensor``'s cost model at ``VERITY_CHECKER_VARIANT=v2`` with the commitment,
IRS row combination, linear test (structured, M = 2 + the table-side exceptional rows), quadratic tests (F_t triples and the
F_c helper triples), helper inversions and column openings replaced by the measurements of ``v2_bench.py``; the model still
prices the r^T A scalar / pattern generation (``simt:constraint_combination``), witness / hint generation and prover
randomness.  Label: ``ARITHMETIC_DIAGNOSTIC`` -- not an end-to-end proof.  A100 / H100 rows are labelled SCALINGS (INT32
issue-rate ratio), not measurements.  Also: the batch-size crossover at which v2 beats v1 (the fixed per-proof table cost
against the per-VU saving).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["VERITY_CHECKER_VARIANT"] = "v2"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from reprice_measured import GPUS, HASH_4090, INT32_OPS_PER_S, UTIL, _best
from verity_numerical.bench import contract
from verity_numerical.explore import tensor as t

NATIVE_4096 = 4096


def pick(bench: dict, variant: str | None = None) -> dict:
    """The measured rows for the protocol-as-written shape (``variant=None``) or variant (b) (``"public_coefficients"``)."""
    rs = bench["results"]

    def var(r):
        return r.get("variant") == variant if variant else "variant" not in r
    commit = _best([r for r in rs if r.get("bucket") == "commitment" and r.get("keep") is True], shape="v2")
    irs = _best([r for r in rs if r.get("bucket") == "irs_row_combination"], shape="v2")
    quad_ft = _best([r for r in rs if r.get("bucket") == "quadratic_test"], shape="v2")
    # primary: coset 1 read from the cosets store the fused commitment keeps anyway (keep=True is the commitment row used;
    # the openings read the same store); the re-encoding variant is kept in the detail table
    quad_fc = _best([r for r in rs if r.get("bucket") == "quadratic_test_fc" and r["coset1_source"] == "store"])
    quad_fc_reenc = _best([r for r in rs if r.get("bucket") == "quadratic_test_fc" and r["coset1_source"] == "re-encoded"])
    lin = _best([r for r in rs if r.get("bucket") == "linear_test_structured" and var(r)])
    inv = _best([r for r in rs if r.get("bucket") == "batch_inverse" and var(r)])
    op = _best([r for r in rs if r.get("name") == "open_columns_gather_from_store"])
    paths = _best([r for r in rs if r.get("name") == "open_columns_merkle_paths_host"])
    return {"commitment": commit, "irs_row_combination": irs, "quadratic_test_ft": quad_ft, "quadratic_test_fc": quad_fc,
            "quadratic_test_fc_reencode": quad_fc_reenc, "linear_test": lin, "batch_inverse": inv, "open_columns": op, "merkle_paths": paths}


def measured_seconds(d: dict) -> dict:
    return {"commitment": d["commitment"]["ms_median"] / 1e3,
            "matmul:ligero_row_combination": (d["linear_test"]["ms_median"] + d["irs_row_combination"]["ms_median"]) / 1e3,
            "simt:quadratic_hadamards": (d["quadratic_test_ft"]["ms_median"] + d["quadratic_test_fc"]["ms_median"]) / 1e3,
            "simt:helper_inversions": d["batch_inverse"]["ms_median"] / 1e3,
            "extra:openings": (d["open_columns"]["ms_median"] + d["merkle_paths"]["ms_median"]) / 1e3}


def reprice(bench: dict, shape: dict, v1: dict) -> dict:
    fs = t.FIELDS["babybear"]
    p = t.price_design(t.DESIGNS["B"], fs, -128)                    # v2 census (env var above)
    rows_out = []
    for variant in (None, "public_coefficients"):
        d = pick(bench, variant)
        m = measured_seconds(d)
        sh = shape["helper_rows" if variant is None else "public_coefficients"]
        for gkey, gpu in GPUS.items():
            scale = INT32_OPS_PER_S[gkey] / INT32_OPS_PER_S["RTX4090"]
            tm = p.times(gpu, gkey, UTIL, HASH_4090["B"], "rs_matrix_ntt", fs, measured={k: v / scale for k, v in m.items()})
            label = (("MEASURED buckets on RTX 4090 at the checker-v2 shape (commitment, structured linear test, IRS, F_t + F_c quadratic tests, "
                      "inversions, openings); r^T A scalars, witness / hint generation, PRG modelled" + (f"; table side: {variant} (linear test, inversions at its shape; commitment / IRS / openings at the helper_rows row count: upper bound)" if variant else ""))
                     if gkey == "RTX4090" else
                     f"SCALING: measured buckets / {scale:.2f} (INT32 issue-rate ratio), modelled buckets at {gkey} peaks" + (f"; table side: {variant}" if variant else ""))
            rows_out.append({"design": "B", "checker": "v2", "table_side": variant or "helper_rows", "field": "babybear", "target_log2": -128,
                             "B": t.B_TARGET, "K": t.K_TARGET, "ligero_rows": sh["rows"]["total"], "ligero_rows_model": p.ligero_rows,
                             "committed_bytes_per_vu": sh["committed_bytes_per_vu"], "gpu": gkey, "label": label, "simt_scale": scale,
                             "measured_ms": {k: v * 1e3 for k, v in m.items()},
                             "detail": {k: {kk: vv for kk, vv in v.items() if kk != "ms_all"} for k, v in d.items()},
                             "t.total_ms": tm["t.total"] * 1e3, "t.commitment_ms": tm["t.commitment"] * 1e3, "t.matmul_ms": tm["t.matmul"] * 1e3,
                             "t.simt_field_ms": tm["t.simt_field"] * 1e3, "t.extra_ms": tm["t.extra"] * 1e3, "t.hint_gen_ms": tm["t.hint_gen"] * 1e3,
                             "t.prg_ms": tm["t.prg"] * 1e3, "t.modelled_ms": {k: v * 1e3 for k, v in tm["t.modelled"].items()},
                             "measured_fraction": tm["measured_fraction"], "seconds_per_vu": tm["seconds_per_vu"],
                             "overhead.vs_native_peak": tm["overhead.vs_native_peak"]})
    # -- v1 reference row (b_measured_4090.json, the same kernels at the v1 shape) ------------------------------------------------
    v1best = min((r for r in v1["rows"] if r["design"] == "B" and r["gpu"] == "RTX4090" and r.get("leaf_hash") == "blake3" and r.get("openings") == "store"),
                 key=lambda r: r["t.total_ms"])
    # -- crossover: fixed per-proof table cost vs per-VU saving --------------------------------------------------------------------------
    best = rows_out[0]
    sh = shape["helper_rows"]
    d = best["detail"]
    table_rows = sh["rows"]["multiplicity"] + sh["rows"]["table_inverse"]
    fixed_ms = (table_rows * d["commitment"]["us_per_row"] / 1e3 + table_rows * d["irs_row_combination"]["us_per_row"] / 1e3
                + sh["table_rows_total"] * d["batch_inverse"]["ns_per_element"] / 1e6 + d["linear_test"]["split_ms"]["exceptional_generic"]
                + table_rows * d["open_columns"]["us_per_row"] / 1e3)
    marginal_us = (best["t.total_ms"] - fixed_ms) / t.B_TARGET * 1e3
    v1_us = v1best["t.total_ms"] / t.B_TARGET * 1e3
    b_star = fixed_ms * 1e3 / (v1_us - marginal_us) if v1_us > marginal_us else float("inf")
    rows_cross = sh["rows"]["fixed_per_proof"] / (v1best["ligero_rows"] / t.B_TARGET - sh["rows"]["per_vu_marginal"])
    crossover = {"v2_fixed_ms_per_proof": fixed_ms, "v2_marginal_us_per_vu": marginal_us, "v1_us_per_vu": v1_us, "B_star_time": b_star,
                 "B_star_rows": rows_cross, "v2_table_rows": table_rows, "v2_table_elements_inverted": sh["table_rows_total"],
                 "note": "fixed = table rows' commitment + IRS + openings (us/row measured) + 6.26M table inversions (ns/element measured) + the "
                         "exceptional linear-test rows (measured split); marginal = (total - fixed) / 4096; v1 per VU from b_measured_4090.json; "
                         "both at 2^-128 (t = 192 is B-independent at this scale), so time(v2, B) = fixed + marginal B vs time(v1, B) = v1_us B"}
    return {"meta": {"lane": "b-v2", "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "run_ids": {"v2_bench": bench.get("run_id"), **v1["meta"]["run_ids"]}, "device": bench["device"],
                     "checker_variant": "v2", "utilisation_modelled_matmul": UTIL, "native_seconds_per_vu": contract.native_seconds_per_vu(t.K_TARGET),
                     "proof_class": "ARITHMETIC_DIAGNOSTIC", "all_bit_exact": bench["all_bit_exact"], "checks": bench["checks"],
                     "scope": "vu-level re-pricing at the checker-v2 relation: tensor.py cost model (VERITY_CHECKER_VARIANT=v2) with the "
                              "commitment, IRS, structured linear test, F_t and F_c quadratic tests, helper inversions and openings replaced by "
                              "standalone measurements on the RTX 4090 at the V2_SHAPE.md shape (B = 4096, K = 1536, 2^-128, BabyBear / "
                              "BabyBear^6, HVZK rows included, authentication excluded)",
                     "omitted": ["not an end-to-end proof: buckets measured separately on a synthetic v2-shape witness",
                                 "witness / hint generation and r^T A scalars modelled",
                                 "BabyBear word-class repair rows modelled per checker_min.md (F4 limbs, sign tightening, TNORM_HI)",
                                 "transcript serialisation, verifier work, network"]},
            "shape": shape, "v1_reference": {k: v1best[k] for k in ("t.total_ms", "ligero_rows", "measured_ms", "t.modelled_ms", "overhead.vs_native_peak", "measured_fraction")},
            "crossover": crossover, "rows": rows_out}


def markdown(rep: dict) -> str:
    dev = rep["meta"]["device"]
    best = rep["rows"][0]
    bvar = next(r for r in rep["rows"] if r["table_side"] == "public_coefficients" and r["gpu"] == "RTX4090")
    v1 = rep["v1_reference"]
    sh = rep["shape"]["helper_rows"]
    d = best["detail"]
    mv1 = v1["measured_ms"]
    best["measured_ms"]
    lin, irs, qft, qfc, qfs, inv, op, mp, c = (d["linear_test"], d["irs_row_combination"], d["quadratic_test_ft"], d["quadratic_test_fc"],
                                              d["quadratic_test_fc_reencode"], d["batch_inverse"], d["open_columns"], d["merkle_paths"], d["commitment"])
    v1_lin = 7.15
    v1_irs = mv1["matmul:ligero_row_combination"] - v1_lin
    mod1, mod2 = sum(v1["t.modelled_ms"].values()), sum(best["t.modelled_ms"].values())
    cr = rep["crossover"]
    L = [f"# Candidate B on the RTX 4090 at the checker-v2 shape, every prover bucket measured (b-v2 lane, {rep['meta']['generated'][:10]})", "",
         (f"**Label: `ARITHMETIC_DIAGNOSTIC`-level re-pricing, scope vu.**  {rep['meta']['scope']}.  Device: {dev['gpu']} ({dev['sms']} SMs, driver "
         f"{dev['nvidia_smi_driver']}, CUDA runtime {dev['cuda_runtime_version']}, cupy {dev['cupy']}).  Run `{rep['meta']['run_ids']['v2_bench']}` (v2 shape); v1 "
         f"reference runs `{rep['meta']['run_ids']['commit']}`, `{rep['meta']['run_ids']['buckets']}`, `{rep['meta']['run_ids']['lintest']}`.  CUDA-event timings, "
         f"warm, median of 7; every kernel bit-exact against its numpy / Python reference in the run (`all_bit_exact = {rep['meta']['all_bit_exact']}`), including "
         "the new `F_c`-triple quadratic kernel (`buckets_v2.py`).  Layout and counts: `backends/direct/encode/V2_SHAPE.md`."), "",
         "## v1 vs v2, same kernels, same card (B = 4096, K = 1536, 2^-128, BabyBear^6, HVZK rows, Blake3 leaves, openings from the kept store)", "",
         "| | v1 shape (`b_measured_4090.md`) | v2 shape (this run) | v2 / v1 |", "|---|---|---|---|",
         f"| Ligero rows x [16384, 4096] | 310,602 | **{sh['rows']['total']:,}** (34,093 witness + 58,710 helper + 11,408 table-side + 6 blinding) | {sh['rows']['total'] / 310602:.2f}x |",
         f"| committed bytes / VU | 1,164.7 kB | **{sh['committed_bytes_per_vu'] / 1e3:.1f} kB** | {sh['committed_bytes_per_vu'] / 1164743:.2f}x |",
         f"| quadratic rows | 33,485 `F_t` | {sh['quadratic']['ft_triple_rows']:,} `F_t` + {sh['quadratic']['fc_triple_rows']:,} `F_c` (6 coordinate rows each) | |",
         f"| helpers inverted | 125.9M | **{sh['helpers_to_invert'] / 1e6:.1f}M** (37.4M queries + 6.26M table rows) | {sh['helpers_to_invert'] / 125895743:.2f}x |",
         f"| linear test structure | M = 3, X ~ 130-256 | M = 2, X = {sh['lintest']['exceptional_rows']:,} (table-side inverse rows) | |",
         f"| commitment (fused encode -> Blake3 -> root, cosets kept) | {mv1['commitment']:.1f} ms | **{c['ms_median']:.1f} ms** ({c['us_per_row']:.3f} us/row) | {c['ms_median'] / mv1['commitment']:.2f}x |",
         f"| linear test (structured, lazy) | {v1_lin:.2f} ms | **{lin['ms_median']:.2f} ms** (combine {lin['split_ms']['combine_multi']:.2f} + exceptional {lin['split_ms']['exceptional_generic']:.2f}) | {lin['ms_median'] / v1_lin:.2f}x |",
         f"| IRS row combination | {v1_irs:.1f} ms | **{irs['ms_median']:.2f} ms** | {irs['ms_median'] / v1_irs:.2f}x |",
         f"| quadratic test | {mv1['simt:quadratic_hadamards']:.1f} ms | **{qft['ms_median'] + qfc['ms_median']:.1f} ms** (`F_t` {qft['ms_median']:.2f} + `F_c` helper triples {qfc['ms_median']:.2f} with coset 1 read from the kept store; {qfs['ms_median']:.2f} if re-encoded) | {(qft['ms_median'] + qfc['ms_median']) / mv1['simt:quadratic_hadamards']:.2f}x |",
         f"| helper inversions | {mv1['simt:helper_inversions']:.1f} ms | **{inv['ms_median']:.1f} ms** ({inv['ns_per_element']:.2f} ns/element, {inv['traffic_floor_ms'] / inv['ms_median']:.2f} of traffic floor) | {inv['ms_median'] / mv1['simt:helper_inversions']:.2f}x |",
         f"| openings (t = 192 + paths) | {mv1['extra:openings']:.1f} ms | **{op['ms_median'] + mp['ms_median']:.1f} ms** | |",
         f"| modelled remainder (r^T A scalars, hints, PRG) | {mod1:.1f} ms | {mod2:.1f} ms | |",
         f"| **total per 4096-VU batch** | **{v1['t.total_ms']:.1f} ms** ({v1['measured_fraction']:.0%} measured) | **{best['t.total_ms']:.1f} ms** ({best['measured_fraction']:.0%} measured) | **{best['t.total_ms'] / v1['t.total_ms']:.2f}x** |",
         f"| **overhead.vs_native_peak** | **{v1['overhead.vs_native_peak']:.3g}** | **{best['overhead.vs_native_peak']:.3g}** | |",
         f"| per VU | {v1['t.total_ms'] / 4096 * 1e3:.1f} us | {best['t.total_ms'] / 4096 * 1e3:.1f} us | |", "",
         (f"Variant (b), public table coefficients (`V2_SHAPE.md` section 4; MODELLED layout, measured at its shape): {bvar['t.total_ms']:.1f} ms, overhead "
         f"{bvar['overhead.vs_native_peak']:.3g} ({bvar['ligero_rows']:,} rows, {bvar['committed_bytes_per_vu'] / 1e3:.1f} kB / VU)."), "",
         "## Re-pricing rows", "", "~~~text",
         f"{'table side':20} {'gpu':8} {'total ms':>9} {'commit':>7} {'lin+irs':>8} {'quad':>6} {'inv':>6} {'open':>6} {'model':>6} {'meas%':>6} {'overhead':>9}  label"]
    for r in rep["rows"]:
        mm = r["measured_ms"]
        s = r["simt_scale"]
        L.append(f"{r['table_side']:20} {r['gpu']:8} {r['t.total_ms']:9.1f} {r['t.commitment_ms']:7.1f} {mm['matmul:ligero_row_combination'] / s:8.1f} "
                 f"{mm['simt:quadratic_hadamards'] / s:6.1f} {mm['simt:helper_inversions'] / s:6.1f} {r['t.extra_ms']:6.1f} {sum(r['t.modelled_ms'].values()):6.1f} "
                 f"{r['measured_fraction']:6.0%} {r['overhead.vs_native_peak']:9.3g}  {r['label']}")
    L += ["~~~", "",
          "## Batch-size crossover", "",
          (f"v2 pays the tables per proof: {cr['v2_table_rows']:,} table-side Ligero rows (multiplicities + inverses) and {cr['v2_table_elements_inverted'] / 1e6:.2f}M table "
          f"inversions, plus the exceptional linear-test rows: **{cr['v2_fixed_ms_per_proof']:.1f} ms fixed per proof** (measured per-row / per-element rates).  The rest is "
          f"**{cr['v2_marginal_us_per_vu']:.1f} us per VU** against v1's {cr['v1_us_per_vu']:.1f} us per VU.  time(v2, B) < time(v1, B) for **B > {cr['B_star_time']:.0f} VUs** "
          f"(in committed rows the crossover is B > {cr['B_star_rows']:.0f}).  At B = 4096 the fixed part is {cr['v2_fixed_ms_per_proof'] / best['t.total_ms']:.0%} of the batch; at "
          f"B = 64 v2 would cost {(cr['v2_fixed_ms_per_proof'] + cr['v2_marginal_us_per_vu'] * 64 / 1e3):.1f} ms against v1's {cr['v1_us_per_vu'] * 64 / 1e3:.1f} ms.  Below the "
          "crossover the SHIFT table (57% of the rows) should be its split-u variant (`checker_min.md` section 4)."), "",
          "## What the measurements say", "",
          (f"* **The relation change is worth {v1['t.total_ms'] / best['t.total_ms']:.1f}x on B's measured buckets**, close to the 2.98x row ratio: every bucket is per-row streaming "
          "work, so 3x fewer committed rows is 3x less time; the tables cost 11% of the rows and one exceptional-row pass in the linear test."),
          (f"* **Inversions shrink, not grow**: v1's 320 range-chunk lookups per unit were the helpers; v2's 95 lookups + 6.26M table rows are {sh['helpers_to_invert'] / 125895743:.2f}x "
          f"as many elements.  Still {inv['traffic_floor_ms'] / inv['ms_median']:.2f} of the traffic floor."),
          (f"* **The `F_c` helper triples are the quadratic test now**: {qfc['ms_median']:.1f} ms for {qfc['rows']:,} rows reading coset 1 from the kept store "
          f"({qfc['us_per_row']:.2f} us/row) -- the store the fused commitment already keeps for the openings; {qfs['ms_median']:.1f} ms if the "
          f"coset is re-encoded instead ({6 + 3} encoder calls per row on average)."),
          (f"* The linear test's exceptional rows (the table-side inverses, {lin['exceptional_rows']:,} through the general path) are {lin['split_ms']['exceptional_generic']:.1f} ms of its "
          f"{lin['ms_median']:.1f}; variant (b) removes them (X = 1,632)."),
          ("* Still modelled: r^T A scalars / pattern (O(rows + pattern)), witness / hint generation (v2 has 15 hints per unit, v1 225: the CPU anchor's 4.7 s per 4,096 "
          "units is a v1 number), prover randomness; the BabyBear repair rows are checker-min's modelled deltas (stated in `V2_SHAPE.md` section 2)."), "",
          *(_blocked_section(rep) if rep["meta"].get("inverse_blocked") else []),
          *(_radix4_section(rep) if rep["meta"].get("radix4") else []),
          "## Files", "",
          ("`backends/direct/encode/V2_SHAPE.md`, `v2_shape.py` / `v2_shape.json`, `buckets_v2.py` (the `F_c`-triple kernel + reference), `v2_bench.py`, "
          f"`results_v2_{rep['meta']['run_ids']['v2_bench']}.json`, `reprice_v2.py` (this report); ledger `ledger/b-v2.jsonl`.")]
    return "\n".join(L) + "\n"


def apply_blocked_inverse(bench: dict, inverse: dict, run_id: str) -> dict:
    """Replace the ``batch_inverse`` rows' timings by the blocked-Montgomery kernel's (``inverse_bench.py``): the ratio best blocked /
    best plain measured at the v2 helper count is applied to every ``batch_inverse`` row (variant (b) has a smaller N)."""
    rows = inverse["bench"]["v2"]
    plain = min((r for r in rows if r["g"] == 1), key=lambda r: r["ms_median"])
    blocked = min((r for r in rows if r["g"] > 1), key=lambda r: r["ms_median"])
    ratio = blocked["ms_median"] / plain["ms_median"]
    for r in bench["results"]:
        if r.get("bucket") == "batch_inverse":
            r["ms_median_plain"] = r["ms_median"]
            r["ms_median"] *= ratio
            r["ns_per_element"] *= ratio
            r["bytes_moved"] = int(r["bytes_moved"] * blocked["bytes_moved"] / plain["bytes_moved"])
            r["traffic_floor_ms"] *= blocked["bytes_moved"] / plain["bytes_moved"]
            r["name"] += f"+blocked_G{blocked['g']}"
            r["blocked_run"] = run_id
    v1rows = inverse["bench"].get("v1", [])
    v1p = min((r for r in v1rows if r["g"] == 1), key=lambda r: r["ms_median"], default=None)
    v1b = min((r for r in v1rows if r["g"] > 1), key=lambda r: r["ms_median"], default=None)
    return {"run": run_id, "g": blocked["g"], "threads": blocked["threads"], "exact": all(inverse["exact"].values()),
            "v2_plain_ms": plain["ms_median"], "v2_blocked_ms": blocked["ms_median"], "v2_floor_plain_ms": plain["traffic_floor_ms"],
            "v2_floor_blocked_ms": blocked["traffic_floor_ms"], "ratio": ratio,
            "v1_plain_ms": v1p and v1p["ms_median"], "v1_blocked_ms": v1b and v1b["ms_median"], "v1_blocked_g": v1b and v1b["g"],
            "v1_floor_blocked_ms": v1b and v1b["traffic_floor_ms"]}


def _blocked_section(rep: dict) -> list[str]:
    b = rep["meta"]["inverse_blocked"]
    v1 = (f"  At the v1 helper count (125.9M) the same kernel is {b['v1_blocked_ms']:.1f} ms (G = {b['v1_blocked_g']}, {b['v1_floor_blocked_ms'] / b['v1_blocked_ms']:.2f} of "
          f"its floor) against 28.8 ms: v1's total becomes {116.4 - 28.8 + b['v1_blocked_ms']:.1f} ms." if b.get("v1_blocked_ms") else "")
    return ["## Inversions: blocked Montgomery trick (`BatchInverterBlocked`, run `%s`)" % b["run"], "",
            (f"One stored prefix per G = {b['g']} elements instead of one per element: traffic (3 + 2/G) x 24 N bytes instead of 5 x 24 N, "
             f"(4 - 1/G) `F_c` products per element instead of 3.  At the v2 helper count: **{b['v2_blocked_ms']:.2f} ms** ({b['v2_floor_blocked_ms'] / b['v2_blocked_ms']:.2f} of its "
             f"traffic floor {b['v2_floor_blocked_ms']:.2f} ms) against {b['v2_plain_ms']:.2f} ms for the plain kernel ({b['v2_floor_plain_ms'] / b['v2_plain_ms']:.2f} of {b['v2_floor_plain_ms']:.2f}): "
             f"{1 / b['ratio']:.2f}x.  Bit-exact against `ref.batch_inverse` at N = 1000 / 4097 / 70000, G = 2 / 4 / 8 (`exact = {b['exact']}`)."
             + v1 + "  The inversions bucket above uses the blocked kernel (the `batch_inverse` rows carry `ms_median_plain`)."), ""]


def apply_radix4(bench: dict, radix4: dict, run_id: str) -> dict:
    """Scale the ``commitment`` rows by the radix-4 / radix-2 fused-commitment ratio measured in the same run at the v2 row count
    (``radix4_bench.py``; keep=True, the configuration the report uses)."""
    rs = [r for r in radix4["results"] if r["bucket"] == "commitment" and r["rows"] == 104217 and r["keep"]]
    r2 = next(r for r in rs if not r["radix4"])
    r4 = next(r for r in rs if r["radix4"])
    ratio = r4["ms_median"] / r2["ms_median"]
    for r in bench["results"]:
        if r.get("bucket") == "commitment":
            r["ms_median_radix2"] = r["ms_median"]
            r["ms_median"] *= ratio
            r["us_per_row"] *= ratio
            r["name"] += "_radix4"
            r["radix4_run"] = run_id
    enc = [r for r in radix4["results"] if r["bucket"] == "encoder" and r["rows"] == 104217]
    e2 = min((r for r in enc if not r["radix4"]), key=lambda r: r["ms_median"])
    e4 = min((r for r in enc if r["radix4"]), key=lambda r: r["ms_median"])
    v1 = [r for r in radix4["results"] if r["bucket"] == "commitment" and r["rows"] == 310602 and r["keep"]]
    return {"run": run_id, "ratio": ratio, "v2_commit_radix2_ms": r2["ms_median"], "v2_commit_radix4_ms": r4["ms_median"],
            "v2_encoder_radix2_ms": e2["ms_median"], "v2_encoder_radix4_ms": e4["ms_median"], "exact": all(radix4["exact"].values()),
            "v1_commit_radix2_ms": next(r["ms_median"] for r in v1 if not r["radix4"]), "v1_commit_radix4_ms": next(r["ms_median"] for r in v1 if r["radix4"])}


def _radix4_section(rep: dict) -> list[str]:
    b = rep["meta"]["radix4"]
    return ["## Encoder: radix-4 shared-memory stages (`RSEncoderSIMT(radix4=True)`, run `%s`)" % b["run"], "",
            (f"The shared-memory stages (m < THREADS; the register stages m >= THREADS were already free of shared-memory traffic) are fused in pairs: "
             f"the quad is closed under both radix-2 stages, so the arithmetic is unchanged (bit-exact, `exact = {b['exact']}`) with half the shared-memory "
             f"traffic and half the `__syncthreads`.  Encoder alone at 104,217 rows: {b['v2_encoder_radix2_ms']:.2f} -> **{b['v2_encoder_radix4_ms']:.2f} ms** "
             f"({b['v2_encoder_radix2_ms'] / b['v2_encoder_radix4_ms']:.2f}x); fused commitment (Blake3, cosets kept): {b['v2_commit_radix2_ms']:.2f} -> "
             f"**{b['v2_commit_radix4_ms']:.2f} ms** ({1 / b['ratio']:.2f}x; the hash and the store write are unchanged).  At the v1 row count: "
             f"{b['v1_commit_radix2_ms']:.1f} -> {b['v1_commit_radix4_ms']:.1f} ms.  The commitment bucket above is the run-`{rep['meta']['run_ids']['v2_bench']}` "
             f"measurement scaled by {b['ratio']:.3f} (`ms_median_radix2` kept on the row); `commit_stream.py`'s default encoder is now radix-4."), ""]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radix4", default=None, help="radix4_bench.py result: use the radix-4 encoder's fused-commitment ratio")
    ap.add_argument("--radix4-run", default="")
    ap.add_argument("--bench", required=True)
    ap.add_argument("--inverse", default=None, help="inverse_bench.py result: use the blocked kernel's timings for the inversions bucket")
    ap.add_argument("--inverse-run", default="")
    ap.add_argument("--v1", default=str(HERE.parents[1] / "numerical/reports/b_measured_4090.json"))
    ap.add_argument("--shape", default=str(HERE / "v2_shape.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--md", default=None)
    a = ap.parse_args(argv)
    bench = json.loads(Path(a.bench).read_text())
    shape = json.loads(Path(a.shape).read_text())
    v1 = json.loads(Path(a.v1).read_text())
    blocked = apply_blocked_inverse(bench, json.loads(Path(a.inverse).read_text()), a.inverse_run) if a.inverse else None
    radix4 = apply_radix4(bench, json.loads(Path(a.radix4).read_text()), a.radix4_run) if a.radix4 else None
    rep = reprice(bench, shape, v1)
    rep["meta"]["inverse_blocked"] = blocked
    rep["meta"]["radix4"] = radix4
    for r in rep["rows"]:
        print(f"{r['table_side']:20} {r['gpu']:8} {r['t.total_ms']:8.1f} ms  {r['measured_fraction']:.0%} measured  overhead {r['overhead.vs_native_peak']:.3g}")
    print("crossover:", json.dumps({k: v for k, v in rep["crossover"].items() if k != "note"}))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
    if a.md:
        Path(a.md).write_text(markdown(rep))


if __name__ == "__main__":
    main()
