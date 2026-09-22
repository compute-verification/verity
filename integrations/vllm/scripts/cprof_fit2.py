#!/usr/bin/env python3
"""CPERF 2026-09-17 §3 deliverable 3: refit the commitment cost model from bench-1.json (runs on the laptop, no store needed).

    python scripts/cprof_fit2.py --bench docs/data/commitment-prof-2026-09-17/bench-1.json \
        --census docs/data/commitment-prof-2026-09-17/census-compact.json --out docs/data/commitment-prof-2026-09-17/cost-model.json

Two models on the same bytes (the 64 RU trees, 68,184 leaves, 123,675,904 B):
  production   T = L * a_prod + B * beta        (a_prod from ablate.1_production minus the streaming term)
  packed       T = L * a_packed + B * beta      (least squares over the layout sweep: 6 leaf granularities, same bytes)
beta = the contiguous SHA-256 streaming rate measured on the same bytes (ablate.3a).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def lstsq_affine(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    slope = sxy / sxx
    return slope, my - slope * mx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--census", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    b = json.loads(Path(a.bench).read_text())
    c = json.loads(Path(a.census).read_text())
    L = b["ru_material"]["leaves"]
    B = b["ru_material"]["bytes"]
    abl, lay, wf, e2e = b["ablate"], b["layout"], b["waterfall"], b["e2e"]
    beta_s_per_byte = abl["3a_contiguous_sha256_per_ru"]["median_s"] / B
    # packed: least squares over the granularity sweep (same bytes, leaves vary)
    pts = [(lay[k]["leaves"], lay[k]["median_s"]) for k in ("row_leaves_as_is", "chunk_1KiB", "chunk_4KiB", "chunk_16KiB", "chunk_64KiB", "one_leaf_per_ru")]
    a_packed, c_packed = lstsq_affine([p[0] for p in pts], [p[1] for p in pts])
    pred = {k: lay[k]["leaves"] * a_packed + c_packed for k in lay if isinstance(lay[k], dict) and "leaves" in lay[k]}
    ratio = {k: round(pred[k] / lay[k]["median_s"], 3) for k in pred}
    a_prod = (abl["1_production"]["median_s"] - B * beta_s_per_byte) / L
    a_prod_noop = (abl["2_noop_hash_pipeline_intact"]["median_s"]) / L
    prov = b["provenance"]
    doc = {
        "schema": "veritor/cperf/cost-model/v2", "label": "[M] coefficients fitted on measured rows; predictions [C]",
        "hardware": {k: prov[k] for k in ("cpu", "sha_ni", "openssl", "python", "blake3", "gpu")}, "weight_class": "light (shared worker cp-cov-b, heavy campaign job resident)",
        "threads": 1, "store": b.get("store", "/workspace/cp/cperf/B0/commit"), "leaves": L, "bytes": B,
        "beta_sha256_streaming": {"s_per_byte": beta_s_per_byte, "us_per_KB": beta_s_per_byte * 1e9 / 1.0, "GB_per_s": 1 / beta_s_per_byte / 1e9,
                                   "source": "ablate.3a contiguous sha256 over each RU's values.bin"},
        "production": {"formula": "T_ru = L * a_prod + B * beta  (+ T_index 7.2 ms/RU + T_read)", "a_prod_us_per_leaf": a_prod * 1e6,
                       "a_prod_noop_us_per_leaf": a_prod_noop * 1e6, "T_index_s_per_RU": wf["index_parse"]["median_s"] / b["ru_material"]["trees"],
                       "T_read_s_total": wf["read_memmap_to_bytes"]["median_s"] - wf["index_parse"]["median_s"],
                       "split_of_a_prod": {"identity.parse+sort+ids_digest_us_per_leaf": wf["ids_parse_sort_digest"]["median_s"] / L * 1e6,
                                           "leaf_framing_python_us_per_leaf": wf["leaf_framing_noop"]["median_s"] / L * 1e6,
                                           "leaf_sha_arith_us_per_leaf": (wf["leaf_hash"]["median_s"] - wf["leaf_framing_noop"]["median_s"]) / L * 1e6,
                                           "fold_python_us_per_leaf": wf["merkle_fold_noop"]["median_s"] / L * 1e6,
                                           "fold_node_sha_us_per_leaf": (wf["merkle_fold_production"]["median_s"] - wf["merkle_fold_noop"]["median_s"]) / L * 1e6},
                       "e2e_build_commitments_s": e2e["build_commitments_recompute=True"]["median_s"],
                       "e2e_noop_hash_s": e2e["build_commitments_NOOP_HASH_recompute=True"]["median_s"],
                       "max_gain_from_any_hash_change_s": e2e["build_commitments_recompute=True"]["median_s"] - e2e["build_commitments_NOOP_HASH_recompute=True"]["median_s"]},
        "packed": {"formula": "T_ru = L * a_packed + c ;  c ~= B * beta", "a_packed_us_per_leaf": a_packed * 1e6, "c_s": c_packed, "B_beta_s": B * beta_s_per_byte,
                   "fit_points": pts, "prediction_over_measurement": ratio,
                   "python_floor_us_per_leaf": abl["4c_packed_noop"]["median_s"] / L * 1e6, "fold_only_s_total": abl["5d_trusted_index_packed_fold"]["median_s"]},
        "census_link": {"B_0_raw": c["plan_request_RU"]["expected_volume"]["B_0_raw"], "L_0": c["plan_request_RU"]["expected_volume"]["L_0"],
                        "sum_B_r": c["plan_request_RU"]["expected_volume"]["sum_B_r_raw"], "sum_L_r": c["plan_request_RU"]["expected_volume"]["sum_L_r"],
                        "T(q)_production_s": {str(q): q * (L * a_prod + B * beta_s_per_byte) for q in (1, 0.5, 0.1)},
                        "T(q)_packed_s": {str(q): q * (L * a_packed + c_packed) for q in (1, 0.5, 0.1)}},
    }
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({k: doc[k] for k in ("beta_sha256_streaming", "production", "packed")}, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
