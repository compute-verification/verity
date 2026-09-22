#!/usr/bin/env python3
"""CPROF: fit the commitment cost model from the profiling artefacts and validate it (§32.11).

Model (single CPU thread, the current implementation's shape):

    T_commit(layout) = L * alpha_leaf  +  beta * B  +  N_int * alpha_merkle      [seconds]

    L         tree leaves of the layout
    B         bytes committed (hashed once per tree build)
    N_int     internal hashes = node + lift count (= sum over levels; ~ L - 1)
    alpha_leaf   per-leaf overhead of ``hashing.leaf_hash`` (Python framing + SHA-256 init/final)
    beta         per-byte SHA-256 streaming cost
    alpha_merkle per-internal-hash cost of ``MerkleTree.__init__`` (node/lift hash + its share
                 of id parse + sort)

Coefficients come from ``hash-micro.json`` (microbenchmarks); validation compares the model's
prediction against every measured layout point in ``layouts.json`` and ``store-profile.json``
(hash_s + fold_s; the read term is reported apart because it is page-cache dependent).

Usage:
    python scripts/cprof_fit.py --dir docs/data/commitment-prof \
        --out docs/data/commitment-prof/cost-model.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def node_count(leaf_count: int) -> int:
    total, width = 0, leaf_count
    while width > 1:
        total += width // 2 + (width % 2)
        width = (width + 1) // 2
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = Path(args.dir)

    micro = json.loads((d / "hash-micro.json").read_text())
    layouts = json.loads((d / "layouts.json").read_text())
    store = json.loads((d / "store-profile.json").read_text())
    attrib = json.loads((d / "attrib.json").read_text()) if (d / "attrib.json").exists() else None

    fit = micro["fit"]
    alpha_leaf = fit["alpha_leaf_us"] / 1e6
    beta = fit["beta_us_per_kb_bigbuffer"] / 1024 / 1e6
    # real canonical ids cost ~5x the micro's short synthetic ids to parse; MerkleTree parses
    # every id, so the fold coefficient depends on the id family (attrib.json measures both)
    alpha_merkle = (attrib["merkle_real_ids_us_per_leaf"] if attrib else fit["alpha_merkle_us_per_leaf"]) / 1e6

    def predict(leaves: int, nbytes: int) -> float:
        return leaves * alpha_leaf + nbytes * beta + node_count(leaves) * alpha_merkle

    rows = []
    for r in layouts["weights"] + layouts["ru"]:
        if "hash_s" not in r or r.get("leaves") is None:
            continue
        measured = r["hash_s"] + r["fold_s"]
        pred = predict(r["leaves"], r["bytes"])
        rows.append({"variant": r["variant"], "tree": r.get("tree"), "leaves": r["leaves"],
                     "bytes": r["bytes"], "measured_s": measured, "predicted_s": pred,
                     "ratio": pred / measured if measured else None})
    for t in store["trees"]:
        if "hash_s" not in t:
            continue
        measured = t["hash_s"] + t["fold_s"]
        pred = predict(t["leaves"], t["bytes"])
        rows.append({"variant": f"store/{t['kind']}", "tree": t["tree"], "leaves": t["leaves"],
                     "bytes": t["bytes"], "measured_s": measured, "predicted_s": pred,
                     "ratio": pred / measured if measured else None})

    ratios = [r["ratio"] for r in rows if r["ratio"]]
    doc = {
        "schema": "veritor/cprof/cost-model/v1",
        "model": "T_commit(L, B) = L*alpha_leaf + beta*B + N_int(L)*alpha_merkle  [s, 1 CPU thread]",
        "coefficients": {
            "alpha_leaf_us": fit["alpha_leaf_us"],
            "beta_us_per_kb": fit["beta_us_per_kb_bigbuffer"],
            "alpha_merkle_us": alpha_merkle * 1e6,
            "alpha_merkle_us_synthetic_short_ids": fit["alpha_merkle_us_per_leaf"],
            "parse_us_per_real_ru_id": attrib["parse_us_per_id"] if attrib else None,
            "parse_us_per_weight_row_id": attrib["parse_us_per_weight_row_id"] if attrib else None,
            "node_hash_us": fit["alpha_node_us"],
            "ru_index_parse_ms_per_ru": (attrib["ru_index_parse_all_s"] / attrib["ru_count"] * 1e3) if attrib else None,
            "sha256_asymptote_gb_per_s": fit["sha256_asymptote_gb_per_s"],
            "cpu": micro["cpu"],
            "openssl": micro["openssl"],
        },
        "evidence": {
            "coefficients": "measured (microbenchmark, hash-micro.json)",
            "validation_points": "measured (layouts.json, store-profile.json, same machine)",
            "model_applications": "modeled from measured components",
        },
        "excluded_terms": {
            "read": "values.bin -> bytes copy; page-cache dependent, reported per store in store-profile.json",
            "materialisation": "D2H + tobytes; gpu-side.json weights_stage",
            "ids_digest_and_json": "per-tree record writing; sub-ms at B0 scale",
        },
        "validation": {"points": rows,
                       "ratio_min": min(ratios), "ratio_max": max(ratios),
                       "ratio_mean": sum(ratios) / len(ratios)},
    }
    out = Path(args.out)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"coefficients": doc["coefficients"],
                      "ratio_range": [doc["validation"]["ratio_min"], doc["validation"]["ratio_max"]],
                      "points": len(rows)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
