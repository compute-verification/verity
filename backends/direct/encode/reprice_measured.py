"""Re-price B-Ligero (and A's shared buckets) on the RTX 4090 with every measured bucket (b-commit lane).

    uv run --frozen --package verity-numerical python backends/direct/encode/reprice_measured.py \
        --commit backends/direct/encode/results_commit_<run>.json --buckets backends/direct/encode/results_buckets_<run>.json \
        --out backends/numerical/reports/b_measured_4090.json [--md backends/numerical/reports/b_measured_4090.md]

Every row is ``explore.tensor``'s cost model with buckets replaced by measurements (``Priced.times(measured=...)``):

  commitment                 fused encode -> column hash -> root (commit_stream; Blake3 leaf or SHA-256 leaf, both given)
  matmul:ligero_row_combination  the linear test (7 encoder calls per row + product-accumulate) + the IRS row combination
  simt:quadratic_hadamards   the quadratic test (3 encoder calls per product row + product-accumulate)
  simt:helper_inversions     the F_c batch inversion of the helpers
  extra:openings             t = 192 column openings (gather from the kept cosets store, or re-encode) + Merkle paths

Still MODELLED (listed per row as ``t.modelled``): the sparse constraint combination ``r^T A`` (``simt:constraint_combination``),
witness/hint generation (``hint_gen``; no GPU witness kernel exists on main) and prover randomness (``prg``).  Label:
``ARITHMETIC_DIAGNOSTIC``-level -- not an end-to-end proof: the buckets are measured separately on resident pseudo-random
data.  A100 / H100 rows are SCALINGS (all measured buckets are integer-SIMT-compute-bound on the 4090: the INT32 issue-rate
ratio, A100 0.47 / H100 0.81; the modelled buckets as ``Priced.times`` prices them), labelled as such -- not measurements.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from verity_numerical.bench import contract
from verity_numerical.explore import tensor as t

UTIL = 0.172
HASH_4090 = {"B": 446e9, "A": 425e9}
GPUS = {"RTX4090": t.GPU_RTX4090, "A100": t.GPUS["A100"], "H100": t.GPUS["H100"]}
#: INT32 issue rate (lanes per SM x SMs x boost clock): the measured buckets are integer SIMT work (Montgomery multiplies,
#: SHA-256 / Blake3 rounds).  Ada has 64 INT32 lanes per SM (128 FP32), Ampere/Hopper 64 (the FP32 ratio would be 0.24 / 0.81).
INT32_OPS_PER_S = {"RTX4090": 128 * 64 * 2.52e9, "A100": 108 * 64 * 1.41e9, "H100": 132 * 64 * 1.98e9}


def _best(results: list[dict], **match) -> dict:
    rows = [r for r in results if all(r.get(k) == v for k, v in match.items())]
    if not rows:
        raise KeyError(match)
    return min(rows, key=lambda r: r["ms_median"])


def measured_buckets(commit: dict, buckets: dict, shape: str, hash: str, openings: str, lintest: dict | None = None,
                     lintest_config: str = "linear_test_structured_M4_X256") -> tuple[dict, dict]:
    """(measured seconds per batch keyed for ``Priced.times``, detail table rows).  ``lintest`` (b-lintest lane): the structured
    linear test's results file; its ``lintest_config`` row replaces the general-path linear test."""
    cres, bres = commit["results"], buckets["results"]
    fused = _best(cres, shape=shape, hash=hash, mode="fused", keep=openings == "store")
    detail = {"commitment": fused}
    m = {"commitment": fused["ms_median"] / 1e3}
    irs = _best(bres, bucket="irs_row_combination", shape=shape)
    detail["irs_row_combination"] = irs
    if shape == "B":
        lin = _best(bres, bucket="linear_test", shape=shape)
        if lintest is not None:
            detail["linear_test_general_path"] = lin
            lin = _best(lintest["results"], name=lintest_config)
        quad = _best(bres, bucket="quadratic_test", shape=shape)
        inv = _best(bres, bucket="batch_inverse", shape=shape)
        detail.update({"linear_test": lin, "quadratic_test": quad, "batch_inverse": inv})
        m["matmul:ligero_row_combination"] = (lin["ms_median"] + irs["ms_median"]) / 1e3
        m["simt:quadratic_hadamards"] = quad["ms_median"] / 1e3
        m["simt:helper_inversions"] = inv["ms_median"] / 1e3
    else:
        m["matmul:ligero_row_combination"] = irs["ms_median"] / 1e3          # A has no quadratic test; its linear test is inside the GKR claims
    name = "open_columns_gather_from_store" if openings == "store" else "open_columns_encode_gather"
    op = _best(bres, name=name, shape=shape)
    paths = _best(bres, name="open_columns_merkle_paths_host", shape=shape)
    detail.update({"open_columns": op, "merkle_paths": paths})
    m["extra:openings"] = (op["ms_median"] + paths["ms_median"]) / 1e3
    return m, detail


def reprice(commit: dict, buckets: dict, run_ids: dict, lintest: dict | None = None, lintest_config: str = "linear_test_structured_M4_X256") -> dict:
    fs = t.FIELDS["babybear"]
    rows_out = []
    for dname in ("B", "A"):
        p = t.price_design(t.DESIGNS[dname], fs, -128)
        for hash in ("blake3", "sha256"):
            for openings in ("store", "reencode"):
                m, detail = measured_buckets(commit, buckets, dname, hash, openings, lintest, lintest_config)
                base = {"design": dname, "field": "babybear", "target_log2": -128, "B": t.B_TARGET, "K": t.K_TARGET, "ligero_rows": p.ligero_rows,
                        "leaf_hash": hash, "openings": openings, "measured_ms": {k: v * 1e3 for k, v in m.items()},
                        "detail": {k: {kk: vv for kk, vv in v.items() if kk != "ms_all"} for k, v in detail.items()}}
                for gkey, gpu in GPUS.items():
                    scale = INT32_OPS_PER_S[gkey] / INT32_OPS_PER_S["RTX4090"]
                    tm = p.times(gpu, gkey, UTIL, HASH_4090[dname], "rs_matrix_ntt", fs, measured={k: v / scale for k, v in m.items()})
                    label = (("MEASURED buckets on RTX 4090 (commitment, tests, inversions, openings); r^T A generation, witness generation, PRG modelled"
                              + ("; linear test STRUCTURED (b-lintest)" if lintest is not None else ""))
                             if gkey == "RTX4090" else
                             f"SCALING: measured buckets / {scale:.2f} (INT32 issue-rate ratio; they are compute-bound on the 4090), modelled buckets at {gkey} peaks")
                    rows_out.append({**base, "gpu": gkey, "label": label, "simt_scale": scale,
                                     "t.total_ms": tm["t.total"] * 1e3, "t.commitment_ms": tm["t.commitment"] * 1e3,
                                     "t.matmul_ms": tm["t.matmul"] * 1e3, "t.simt_field_ms": tm["t.simt_field"] * 1e3,
                                     "t.extra_ms": tm["t.extra"] * 1e3, "t.hint_gen_ms": tm["t.hint_gen"] * 1e3, "t.prg_ms": tm["t.prg"] * 1e3,
                                     "t.modelled_ms": {k: v * 1e3 for k, v in tm["t.modelled"].items()},
                                     "measured_fraction": tm["measured_fraction"],
                                     "seconds_per_vu": tm["seconds_per_vu"], "overhead.vs_native_peak": tm["overhead.vs_native_peak"]})
        # the previous state of the model for comparison (encode_gpu.json: measured encoder + measured SHA-256, rest modelled)
        enc = _best(commit["results"], shape=dname, mode="encode_only")
        tm = p.times(t.GPU_RTX4090, "RTX4090", UTIL, HASH_4090[dname], "rs_matrix_ntt", fs, encoder_seconds_per_row=1.0176e-7)
        rows_out.append({"design": dname, "gpu": "RTX4090", "leaf_hash": "sha256", "openings": "n/a",
                         "label": "PREVIOUS (encode_gpu.json): measured encoder 31.6 ms + measured SHA-256 446 GB/s, other buckets modelled",
                         "t.total_ms": tm["t.total"] * 1e3, "measured_fraction": tm["measured_fraction"],
                         "t.modelled_ms": {k: v * 1e3 for k, v in tm["t.modelled"].items()},
                         "seconds_per_vu": tm["seconds_per_vu"], "overhead.vs_native_peak": tm["overhead.vs_native_peak"],
                         "encode_only_in_l2_slices_ms": enc["ms_median"]})
    return {"meta": {"lane": "b-commit" if lintest is None else "b-commit + b-lintest (structured linear test)", "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "run_ids": run_ids,
                     "device": commit["device"], "utilisation_modelled_matmul": UTIL,
                     "native_seconds_per_vu": contract.native_seconds_per_vu(t.K_TARGET),
                     "proof_class": "ARITHMETIC_DIAGNOSTIC",
                     "linear_test": ("general path (b-commit): 14 transforms per row" if lintest is None else
                                     f"structured (b-lintest, LINTEST.md): {lintest_config} = per-row F_c scalars x fixed rows; "
                                     "r^T A never materialised; the general path's 149.4 ms row is kept in detail.linear_test_general_path"),
                     "scope": "vu-level re-pricing: tensor.py cost model with the commitment, IRS row combination, linear and quadratic tests, "
                              "helper inversions and column openings replaced by standalone measurements on the RTX 4090 (B = 4096, K = 1536, "
                              "2^-128, BabyBear / BabyBear^6, HVZK rows included, authentication excluded)",
                     "omitted": ["not an end-to-end proof: buckets measured separately on resident pseudo-random data",
                                 "r^T A generation (sparse per-unit constraint combination): modelled SIMT (simt:constraint_combination)",
                                 "witness / hint generation on GPU: modelled (no GPU witness kernel on main; CPU anchor 4.7 s per 4,096 units)",
                                 "prover randomness (randomizers, blinding rows, HM96 strings): modelled at 100 GB/s",
                                 "transcript serialisation, verifier work, network"]},
            "rows": rows_out}


def table(rep: dict) -> str:
    lines = [f"{'design':6} {'gpu':8} {'leaf':7} {'open':8} {'total ms':>9} {'commit':>7} {'tests+irs':>9} {'inv':>6} {'open':>6} {'model':>6} {'meas%':>6} {'overhead':>9}  label"]
    for r in rep["rows"]:
        if "t.commitment_ms" not in r:
            lines.append(f"{r['design']:6} {r['gpu']:8} {r['leaf_hash']:7} {r['openings']:8} {r['t.total_ms']:9.1f} {'':7} {'':9} {'':6} {'':6} "
                         f"{sum(r['t.modelled_ms'].values()):6.1f} {r['measured_fraction']:6.0%} {r['overhead.vs_native_peak']:9.3g}  {r['label']}")
            continue
        meas = r["measured_ms"]
        tests = meas.get("matmul:ligero_row_combination", 0.0) + meas.get("simt:quadratic_hadamards", 0.0)
        lines.append(f"{r['design']:6} {r['gpu']:8} {r['leaf_hash']:7} {r['openings']:8} {r['t.total_ms']:9.1f} {r['t.commitment_ms']:7.1f} "
                     f"{tests / r['simt_scale']:9.1f} {meas.get('simt:helper_inversions', 0.0) / r['simt_scale']:6.1f} {r['t.extra_ms']:6.1f} "
                     f"{sum(r['t.modelled_ms'].values()):6.1f} {r['measured_fraction']:6.0%} {r['overhead.vs_native_peak']:9.3g}  {r['label']}")
    return "\n".join(lines)


def markdown(rep: dict, commit: dict, buckets: dict) -> str:
    dev = rep["meta"]["device"]
    b = [r for r in rep["rows"] if r["design"] == "B" and r["gpu"] == "RTX4090" and "t.commitment_ms" in r]
    best = min(b, key=lambda r: r["t.total_ms"])
    prev = next(r for r in rep["rows"] if r["design"] == "B" and "PREVIOUS" in r["label"])
    L = [f"# Candidate B on the RTX 4090, every prover bucket measured ({rep['meta']['lane']} lane, {rep['meta']['generated'][:10]})", "",
         f"**Label: `ARITHMETIC_DIAGNOSTIC`-level re-pricing, scope vu.**  {rep['meta']['scope']}.  Device: {dev['gpu']} "
         f"({dev['sms']} SMs, driver {dev['nvidia_smi_driver']}, CUDA runtime {dev['cuda_runtime_version']}, cupy {dev['cupy']}).  Runs: "
         + ", ".join(f"`{v}`" for v in rep["meta"]["run_ids"].values()) + ".  CUDA-event timings, warm, median of 7; every kernel bit-exact "
         "against a numpy / Python reference before timing.", "",
         "## Measured buckets, B shape (310,602 Ligero rows x [16384, 4096] BabyBear; 33,485 product rows; 125.9M helpers; t = 192)", "",
         "| bucket | kernel | ms / batch | us / row | floor | fraction of floor | bit-exact |", "|---|---|---|---|---|---|---|"]
    d = best["detail"]
    c = d["commitment"]
    L.append(f"| commitment (encode -> {best['leaf_hash']} column hash -> root) | `commit_stream` fused, {c['slice_rows']}-row L2 slices, CUDA graph"
             f"{', cosets kept (15.3 GB store)' if c['keep'] else ''} | **{c['ms_median']:.1f}** | {c['us_per_row']:.3f} | "
             f"encode {commit['unfused_ms_4090_reference']['B']['encode']} + hash unfused = {c['unfused_ms_reference']:.1f} ms; HBM read floor {c['read_floor_ms']:.1f} ms | "
             f"{c['speedup_vs_unfused']:.2f}x faster than unfused | yes (vs unfused path + CPU) |")
    i = d["irs_row_combination"]
    L.append(f"| IRS row combination v = r^T W (k-wide, F_c scalar per row) | `combine_scalar`, chunk {i['chunk_rows']} | **{i['ms_median']:.1f}** | {i['us_per_row']:.3f} | "
             f"read 5.09 GB once = {i['read_floor_ms']:.1f} ms | {i['read_floor_ms'] / i['ms_median']:.2f} of HBM floor ({i['g_mont_muls_per_s']:.0f} G mont-mul/s) | yes |")
    q = d["quadratic_test"]
    L.append(f"| quadratic test p_0 (8,192-point domain) | 3 encoder calls (coset 1) per product row + `combine_quadratic`, {q['slice_rows']}-row slices | "
             f"**{q['ms_median']:.1f}** | {q['us_per_row']:.3f} per product row | 6 transforms per product row | -- | yes |")
    ln = d["linear_test"]
    if "linear_test_general_path" in d:
        lg = d["linear_test_general_path"]
        sp = ln["split_ms"]
        L.append(f"| linear test q_add, STRUCTURED (b-lintest, `LINTEST.md`): r^T A = per-row F_c scalars x {ln['M']} fixed rows + {ln['exceptional_rows']} exceptional rows | "
                 f"`lintest_structured`: one fused pass over W (`combine_multi`, {sp['combine_multi']:.2f} ms) + {6 * ln['M']} encoder calls ({sp['encode_V']:.2f}) + F_c pointwise ({sp['pointwise']:.2f}) "
                 f"+ general path for the exceptional rows ({sp['exceptional_generic']:.2f}) | **{ln['ms_median']:.1f}** | {ln['us_per_row']:.4f} | read W once = {ln['read_floor_ms']:.1f} ms; 0 transforms per row | "
                 f"{ln['fraction_of_read_floor']:.2f} of HBM floor; **{lg['ms_median'] / ln['ms_median']:.0f}x** faster than the general path ({lg['ms_median']:.1f} ms, same run) | yes (vs the general GPU path and numpy) |")
    else:
        L.append(f"| linear test q_add (8,192-point domain) | 7 encoder calls per row (6 coordinate rows of r^T A + the witness row, coset 1) + `combine_linear`, "
                 f"{ln['slice_rows']}-row slices | **{ln['ms_median']:.1f}** | {ln['us_per_row']:.3f} | 14 transforms per row = 3.5x the commitment's encoding | "
                 f"{ln['us_per_row'] / c['us_per_row']:.1f}x the fused commitment per row | yes (r^T A generation NOT included) |")
    inv = d["batch_inverse"]
    L.append(f"| helper inversions (125.9M F_c elements) | `batch_inverse`, Montgomery trick, {inv['threads']} threads x {inv['segment']} | **{inv['ms_median']:.1f}** | "
             f"{inv['ns_per_element']:.2f} ns / element | traffic {inv['traffic_floor_ms']:.1f} ms | {inv['traffic_floor_ms'] / inv['ms_median']:.2f} of HBM floor | yes |")
    op, mp = d["open_columns"], d["merkle_paths"]
    L.append(f"| column openings (t = 192 columns + paths) | `{op['name']}` + host Merkle paths | **{op['ms_median'] + mp['ms_median']:.1f}** | -- | "
             f"{op['opened_bytes'] / 1e6:.0f} MB opened | -- | yes |")
    L += ["", "## Re-pricing (all rows: `explore.tensor` with the buckets above measured; `t.modelled` = what the model still prices)", "", "~~~text", table(rep), "~~~", "",
          f"B on the 4090 with every bucket measured: **{best['t.total_ms']:.0f} ms per batch, overhead {best['overhead.vs_native_peak']:.3g}** "
          f"(leaf {best['leaf_hash']}, openings from the kept store), {best['measured_fraction']:.0%} of the time measured; the previous state "
          f"(encoder + SHA-256 measured, rest modelled) said {prev['t.total_ms']:.0f} ms / {prev['overhead.vs_native_peak']:.3g}.  Modelled remainder: "
          + ", ".join(f"{k} {v:.1f} ms" for k, v in best["t.modelled_ms"].items()) + ".", "",
          "## What the measurements say", "",
          ((f"* **The linear test was B's bucket; the structure of r^T A removes it.**  Under the protocol's powers-of-rho combination every unit's "
            f"constraint pattern makes r^T A on a Ligero row a per-row F_c scalar times one of {ln['M']} fixed rows (`LINTEST.md`), so q = sum_m enc(E_m) * "
            f"enc(sum_i s_im W_i): one memory-bound pass over W instead of 14 transforms per row.  Measured {ln['ms_median']:.1f} ms "
            f"({ln['fraction_of_read_floor']:.2f} of the read floor; M = {ln['M']}, {ln['exceptional_rows']} exceptional rows through the general path) against "
            f"{d['linear_test_general_path']['ms_median']:.1f} ms for the general path in the same run; now {ln['ms_median'] / best['t.total_ms']:.0%} of the batch.  "
            f"Same test, same transcript: a prover-side algorithm, given a unit-major constraint enumeration and rows holding whole units (both public conventions).")
           if "linear_test_general_path" in d else
           (f"* **The linear test is B's bucket, not hashing or encoding.**  q_add needs the deg < k interpolant of every row of r^T A (an F_c row: "
            f"6 BabyBear coordinate rows), evaluated on 8,192 points, times the re-encoded witness row: 14 NTT-size transforms per row against the "
            f"commitment's 4.  Measured {ln['ms_median']:.0f} ms = {ln['ms_median'] / best['t.total_ms']:.0%} of the batch.  The model priced the whole "
            f"row-combination bucket as one tensor-core matmul of 5.4e11 MACs (9.5 ms at utilisation 0.172) and did not count the r^T A encodings at all.")),
          (f"* **Commitment: {c['ms_median']:.1f} ms fused** ({c['speedup_vs_unfused']:.2f}x vs unfused encode + hash), the codeword never in HBM (peak device memory "
          f"{c['device_memory_in_use_bytes'] / 2**30:.1f} GiB).  Blake3 with an unrolled schedule hashes the 20.4 GB codeword in 15.4 ms; SHA-256 in 30 ms at the "
          f"INT32 issue rate.  The SIMT NTT encoder (34-36 ms in L2 slices) is now the commitment's bottleneck; its SIMT floor is ~14 ms."),
          (f"* IRS row combination {i['ms_median']:.1f} ms ({i['read_floor_ms'] / i['ms_median']:.2f} of the read floor: memory-bound), quadratic test {q['ms_median']:.1f} ms, "
          f"helper inversions {inv['ms_median']:.1f} ms ({inv['traffic_floor_ms'] / inv['ms_median']:.2f} of traffic floor), openings {op['ms_median'] + mp['ms_median']:.1f} ms: "
          "small buckets; no tensor-core operation anywhere in B."),
          "* Still modelled: r^T A generation (the sparse constraint combination), witness generation (no GPU kernel exists), prover randomness.",
          "", "## Files", "",
          ("`backends/direct/encode/commit_stream.py` (fused commitment), `buckets.py` (bucket kernels), `buckets_ref.py` (references), `commit_bench.py`, "
           "`buckets_bench.py`, `reprice_measured.py` (this report), raw `results_commit_*.json` / `results_buckets_*.json`; ledger `ledger/b-commit.jsonl`.")]
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--buckets", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--lintest", default=None, help="b-lintest results_lintest_<run>.json: structured linear test replaces the general path")
    ap.add_argument("--lintest-config", default="linear_test_structured_M4_X256")
    a = ap.parse_args(argv)
    commit = json.loads(Path(a.commit).read_text())
    buckets = json.loads(Path(a.buckets).read_text())
    lintest = json.loads(Path(a.lintest).read_text()) if a.lintest else None
    run_ids = {"commit": commit.get("run_id"), "buckets": buckets.get("run_id")}
    if lintest:
        run_ids["lintest"] = lintest.get("run_id")
    rep = reprice(commit, buckets, run_ids, lintest, a.lintest_config)
    print(table(rep))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
    if a.md:
        Path(a.md).write_text(markdown(rep, commit, buckets))


if __name__ == "__main__":
    main()
