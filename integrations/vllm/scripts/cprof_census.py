#!/usr/bin/env python3
"""CPERF 2026-09-17 §1: the logical commitment census of a B0 capture store, by protocol stage.

Reads ``index.json`` / ``boundary.json`` / ``weights/index.json`` of a commit-mode store (no bytes are hashed) and
reports, per tree and per protocol class, the distinct logical values, raw value bytes, committed (framed) preimage
bytes, leaf count, leaf-size distribution and Merkle/tree metadata bytes.

Protocol classes (docs/PROTOCOL.md steps 1/3; record_v5 plan P1/cu=whole = one RU per request, first boundary =
designated outputs):

  (a) unconditional first stage      boundary trees (served tokens = v_out(C) = y, plus the request's inputs/params)
                                      [+ the weights root when the checkpoint is not pre-registered]
  (b) second stage, per selected RU  the ``ru/<request>/<step>`` trees of the request (interior rows, KV rows,
                                      the sampled-token copies)
  (c) prescribed inputs               prompt token ids, sampling params, seed (in the boundary tree) and the weights
                                      (v_in(C)): already bound by the request / the checkpoint registration

Two finer RU plans are ALSO priced from the same rows (labelled [C] in the study):
  RU = (request, step)   v_out(step) = KV-cache rows written this step (k after RoPE, v) + the sampled token
  layer / block plans     verity_vllm_adapter.boundary.PLAN_BOUNDARY row counts (2 / 7 rows per layer per position + 1)

Usage:
    python scripts/cprof_census.py --store <run dir> [--out census.json] [--ops]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
POC = HERE.parent
if str(POC) not in sys.path:
    sys.path.insert(0, str(POC))

from verity_vllm import hashing, identity  # noqa: E402
from verity_vllm.capture import CaptureReader  # noqa: E402

LEAF_TAG_B = len(hashing.LEAF_TAG.encode())
NODE_TAG_B = len(hashing.NODE_TAG.encode())
LIFT_TAG_B = len(hashing.LIFT_TAG.encode())
IDS_TAG_B = len(hashing.IDS_TAG.encode())

# Rows that cross a (request, step) RU boundary = the KV cache written this step and read by later steps' attention.
# From the real op list (job 0917T222921-8c9d6f): k after RoPE is its own row ``layers.<l>.self_attn.rotary_emb.rope/port=1``
# ([192] bf16 = 384 B); v is the last 192 elements of the ``layers.<l>.self_attn.qkv_proj.gemm/port=0`` row ([960] bf16 =
# 1920 B) -- the leaf is the whole qkv row, so the v slice is counted as 384 B of a 1920 B leaf (KV_V_SLICE_BYTES).
KV_K_OP = re.compile(r"layers\.\d+\.self_attn\.rotary_emb\.rope")
KV_K_PORT = 1
KV_V_OP = re.compile(r"layers\.\d+\.self_attn\.qkv_proj\.gemm")
KV_V_SLICE_BYTES = 384
SAMPLER_OP = "sampler.select"


def leaf_preimage_bytes(cid: str, dtype: str, shape, nbytes: int) -> int:
    """Bytes actually fed to SHA-256 for one leaf: u32 tag len + tag + 4 x (u64 len + part)."""
    return 4 + LEAF_TAG_B + (8 + len(cid.encode())) + (8 + len(dtype.encode())) + (8 + len(hashing.shape_text(shape).encode())) + (8 + nbytes)


def tree_meta_bytes(leaf_count: int, ids) -> dict:
    """Merkle metadata: internal node preimages, digests stored per level, ids digest preimage."""
    nodes = lifts = 0
    width = leaf_count
    while width > 1:
        nodes += width // 2
        lifts += width % 2
        width = (width + 1) // 2
    ids_pre = 4 + IDS_TAG_B + sum(8 + len(c.encode()) for c in ids)
    return {
        "node_hashes": nodes, "lift_hashes": lifts,
        "node_preimage_bytes": nodes * (4 + NODE_TAG_B + 2 * (8 + 32)) + lifts * (4 + LIFT_TAG_B + 8 + 32),
        "digest_bytes_all_levels": 32 * (leaf_count + nodes + lifts),
        "path_digests_per_opening": nodes and (leaf_count - 1).bit_length(),
        "ids_digest_preimage_bytes": ids_pre,
        "id_bytes_total": sum(len(c.encode()) for c in ids),
    }


def size_hist(sizes) -> dict:
    c = Counter(sizes)
    return {str(k): v for k, v in sorted(c.items())} if len(c) <= 24 else {
        "distinct_sizes": len(c), "min": min(sizes), "median": statistics.median(sizes), "max": max(sizes)}


def census_ru(reader: CaptureReader, rid: str, step: int, ops: Counter) -> dict:
    ru = reader.ru(rid, step)
    entries = ru.index.entries
    ids = [e.canonical_id for e in entries]
    raw = framed = 0
    kv_b = kv_l = samp_b = samp_l = derived_l = derived_b = kv_v_slices = 0
    sizes = []
    by_op = defaultdict(lambda: [0, 0])
    for e in entries:
        raw += e.nbytes
        framed += leaf_preimage_bytes(e.canonical_id, e.dtype, e.shape, e.nbytes)
        sizes.append(e.nbytes)
        cid = identity.parse(e.canonical_id)
        op = f"{cid.op}/port={cid.port}"
        by_op[op][0] += 1
        by_op[op][1] += e.nbytes
        ops[(op, e.dtype, tuple(e.shape), e.nbytes)] += 1
        if e.derived is not None:
            derived_l += 1
            derived_b += e.nbytes
        if cid.op == SAMPLER_OP:
            samp_l += 1
            samp_b += e.nbytes
        elif KV_K_OP.fullmatch(cid.op or "") and cid.port == KV_K_PORT:
            kv_l += 1
            kv_b += e.nbytes
        elif KV_V_OP.fullmatch(cid.op or ""):
            kv_v_slices += 1
            kv_b += KV_V_SLICE_BYTES  # v lives inside the qkv row; the whole 1920 B leaf stays second-stage
    meta = tree_meta_bytes(len(entries), identity.tree_sorted(ids))
    index_bytes = (ru.dir / "index.json").stat().st_size
    values_bytes = (ru.dir / "values.bin").stat().st_size
    return {
        "tree": ru.tree_id, "request_id": rid, "step": step, "phase": ru.index.phase, "positions": len(ru.index.positions),
        "leaves": len(entries), "raw_bytes": raw, "framed_preimage_bytes": framed, "values_bin_bytes": values_bytes,
        "index_json_bytes": index_bytes, "leaf_sizes": size_hist(sizes), "meta": meta,
        "kv_rows": {"leaves": kv_l, "bytes": kv_b, "v_slices_inside_qkv_leaves": kv_v_slices}, "sampled_token": {"leaves": samp_l, "bytes": samp_b},
        "derived": {"leaves": derived_l, "bytes": derived_b},
        "by_op": {k: {"leaves": v[0], "bytes": v[1]} for k, v in sorted(by_op.items())},
    }


def census_boundary(reader: CaptureReader, rid: str) -> dict:
    b = reader.boundary(rid)
    values = b.leaves(reader.profile_id)
    rows = []
    cls = defaultdict(lambda: [0, 0, 0])
    for v in values:
        cid = identity.parse(v.canonical_id)
        field = getattr(cid, "field", None) or getattr(cid, "op", None) or v.canonical_id.rsplit("/", 1)[-1]
        f = str(field)
        if "prompt" in f:
            c = "c_prescribed_input_prompt"
        elif "output" in f or "chosen" in f or "logprob" in f:
            c = "a_output_y"
        elif "param" in f or "seed" in f or "step" in f or "finish" in f or "max_tokens" in f or "profile" in f:
            c = "c_prescribed_params"
        else:
            c = "other"
        fr = leaf_preimage_bytes(v.canonical_id, v.dtype, v.shape, len(v.data))
        cls[c][0] += 1
        cls[c][1] += len(v.data)
        cls[c][2] += fr
        rows.append({"id_tail": v.canonical_id.split("/", 3)[-1][:80], "dtype": v.dtype, "nbytes": len(v.data), "class": c})
    ids = identity.tree_sorted([v.canonical_id for v in values])
    return {
        "tree": f"boundary/{rid}", "request_id": rid, "leaves": len(values), "raw_bytes": sum(len(v.data) for v in values),
        "framed_preimage_bytes": sum(leaf_preimage_bytes(v.canonical_id, v.dtype, v.shape, len(v.data)) for v in values),
        "meta": tree_meta_bytes(len(values), ids), "prompt_len": b.prompt_len, "outputs": len(b.output_token_ids), "steps": len(b.steps),
        "classes": {k: {"leaves": v[0], "raw_bytes": v[1], "framed_bytes": v[2]} for k, v in sorted(cls.items())},
        "rows": rows,
    }


def census_weights(reader: CaptureReader) -> dict:
    w = reader.weights
    idx = w.index
    leaves = 0
    raw = framed = 0
    sizes = []
    ids = []
    for e in idx.entries:
        if e.layout.kind == "whole":
            leaves += 1
            raw += e.nbytes
            framed += leaf_preimage_bytes(e.canonical_id, e.dtype, e.shape, e.nbytes)
            sizes.append(e.nbytes)
            ids.append(e.canonical_id)
        else:
            rb = e.row_nbytes
            for start in e.block_starts():
                end = min(start + e.layout.rows_per_leaf, e.rows)
                cid = e.block_id(start)
                n = (end - start) * rb
                leaves += 1
                raw += n
                framed += leaf_preimage_bytes(cid, e.dtype, e.block_shape(start, end), n)
                sizes.append(n)
                ids.append(cid)
    return {
        "tree": "weights", "format_version": idx.format_version, "tensors": len(idx.entries), "leaves": leaves, "raw_bytes": raw,
        "framed_preimage_bytes": framed, "leaf_sizes": size_hist(sizes), "meta": tree_meta_bytes(leaves, identity.tree_sorted(ids)),
        "values_bin_bytes": (reader.run_dir / "weights" / "values.bin").stat().st_size if (reader.run_dir / "weights" / "values.bin").exists() else None,
        "index_json_bytes": (reader.run_dir / "weights" / "index.json").stat().st_size,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ops", action="store_true", help="print the distinct (op, dtype, shape, nbytes) rows")
    args = ap.parse_args()
    t0 = time.perf_counter()
    reader = CaptureReader(args.store)
    ops: Counter = Counter()
    rus = [census_ru(reader, r, s, ops) for r, s in reader.ru_keys()]
    bounds = [census_boundary(reader, r) for r in reader.request_ids() if reader.has_boundary(r)]
    weights = census_weights(reader) if reader.has_weights() else None
    commitments = reader.commitments
    commitments_bytes = (reader.run_dir / "commitments.json").stat().st_size if (reader.run_dir / "commitments.json").exists() else None

    def agg(rows, keys=("leaves", "raw_bytes", "framed_preimage_bytes")):
        out = {k: sum(r[k] for r in rows) for k in keys}
        out["trees"] = len(rows)
        out["node_hashes"] = sum(r["meta"]["node_hashes"] + r["meta"]["lift_hashes"] for r in rows)
        out["node_preimage_bytes"] = sum(r["meta"]["node_preimage_bytes"] for r in rows)
        out["ids_digest_preimage_bytes"] = sum(r["meta"]["ids_digest_preimage_bytes"] for r in rows)
        out["id_bytes_total"] = sum(r["meta"]["id_bytes_total"] for r in rows)
        return out

    per_request = {}
    for r in rus:
        d = per_request.setdefault(r["request_id"], {"steps": 0, "leaves": 0, "raw_bytes": 0, "framed_preimage_bytes": 0, "kv_bytes": 0, "kv_leaves": 0,
                                                     "sampled_leaves": 0, "sampled_bytes": 0, "node_hashes": 0, "index_json_bytes": 0})
        d["steps"] += 1
        d["leaves"] += r["leaves"]
        d["raw_bytes"] += r["raw_bytes"]
        d["framed_preimage_bytes"] += r["framed_preimage_bytes"]
        d["kv_bytes"] += r["kv_rows"]["bytes"]
        d["kv_leaves"] += r["kv_rows"]["leaves"]
        d["sampled_leaves"] += r["sampled_token"]["leaves"]
        d["sampled_bytes"] += r["sampled_token"]["bytes"]
        d["node_hashes"] += r["meta"]["node_hashes"] + r["meta"]["lift_hashes"]
        d["index_json_bytes"] += r["index_json_bytes"]

    ru_all = agg(rus)
    b_all = agg(bounds)
    by_phase = defaultdict(lambda: {"trees": 0, "leaves": 0, "raw_bytes": 0, "framed_preimage_bytes": 0, "positions": 0})
    for r in rus:
        p = by_phase[r["phase"]]
        p["trees"] += 1
        p["leaves"] += r["leaves"]
        p["raw_bytes"] += r["raw_bytes"]
        p["framed_preimage_bytes"] += r["framed_preimage_bytes"]
        p["positions"] += r["positions"]

    # --- protocol classes under the record_v5 plan (one RU per request, first boundary = designated outputs) ---
    b_out = sum(b["classes"].get("a_output_y", {}).get("raw_bytes", 0) for b in bounds)
    b_out_l = sum(b["classes"].get("a_output_y", {}).get("leaves", 0) for b in bounds)
    b_in = sum(v["raw_bytes"] for b in bounds for k, v in b["classes"].items() if k.startswith("c_"))
    b_in_l = sum(v["leaves"] for b in bounds for k, v in b["classes"].items() if k.startswith("c_"))
    plan_request = {
        "plan": "record_v5 P1/cu=whole: RU = request; first boundary = designated outputs (boundary tree)",
        "a_unconditional": {"trees": len(bounds), "leaves": b_all["leaves"], "raw_bytes": b_all["raw_bytes"], "framed_bytes": b_all["framed_preimage_bytes"],
                            "of_which_outputs_y": {"leaves": b_out_l, "raw_bytes": b_out}, "node_hashes": b_all["node_hashes"],
                            "note": "B_0/L_0 excluding the weights root; add weights below if the checkpoint is not pre-registered"},
        "b_per_selected_RU": {rid: {"B_r_raw": d["raw_bytes"], "B_r_framed": d["framed_preimage_bytes"], "L_r": d["leaves"], "steps": d["steps"],
                                    "node_hashes": d["node_hashes"], "redundant_sampled_token_copies": d["sampled_leaves"]} for rid, d in per_request.items()},
        "b_sum": {"B_raw": ru_all["raw_bytes"], "B_framed": ru_all["framed_preimage_bytes"], "L": ru_all["leaves"], "RUs": len(per_request)},
        "c_prescribed_inputs": {"prompt_params_seed": {"leaves": b_in_l, "raw_bytes": b_in},
                                "weights": ({"leaves": weights["leaves"], "raw_bytes": weights["raw_bytes"], "framed_bytes": weights["framed_preimage_bytes"],
                                             "format_version": weights["format_version"]} if weights else None)},
    }
    N_R = len(per_request) or 1
    mean_Br = ru_all["raw_bytes"] / N_R
    mean_Lr = ru_all["leaves"] / N_R
    plan_request["expected_volume"] = {"formula": "B(q) = B_0 + q * sum_r B_r ; L(q) = L_0 + q * sum_r L_r",
                                       "B_0_raw": b_all["raw_bytes"], "L_0": b_all["leaves"], "sum_B_r_raw": ru_all["raw_bytes"], "sum_L_r": ru_all["leaves"],
                                       "mean_B_r_raw": mean_Br, "mean_L_r": mean_Lr,
                                       "at_q": {str(q): {"B_raw": b_all["raw_bytes"] + q * ru_all["raw_bytes"], "L": b_all["leaves"] + q * ru_all["leaves"]} for q in (1.0, 0.5, 0.25, 0.1, 0.01)}}

    # --- RU = (request, step): v_out(step) = KV rows + sampled token (first stage), the rest second stage ---
    kv_b = sum(r["kv_rows"]["bytes"] for r in rus)
    kv_l = sum(r["kv_rows"]["leaves"] for r in rus)
    sp_b = sum(r["sampled_token"]["bytes"] for r in rus)
    sp_l = sum(r["sampled_token"]["leaves"] for r in rus)
    positions = sum(r["positions"] for r in rus)
    layers = len({m.group(1) for k in ops for m in [re.match(r"layers\.(\d+)\.", k[0])] if m})
    plan_step = {
        "plan": "RU = (request, step) = the ru/<request>/<step> trees as stored; v_out(step) = KV rows (k row after RoPE + v slice of the qkv row) + sampled token",
        "kv_rule": f"k: {KV_K_OP.pattern} port {KV_K_PORT} (whole leaf); v: {KV_V_SLICE_BYTES} B slice of {KV_V_OP.pattern} port 0 [C]",
        "positions_total": positions, "decoder_layers_seen": layers,
        "coarser_first_boundaries_C": {
            "note": "verity_vllm_adapter.boundary.PLAN_BOUNDARY row sets priced from the row sizes above (DERIVED, not measured)",
            "layer_plan_bytes_per_position": layers * 2 * 1152 + 1152, "block_plan_bytes_per_position": layers * (1920 + 1152 + 384 + 1152 + 1152 + 1152 + 1152) + 1152,
            "layer_plan_total_bytes": positions * (layers * 2 * 1152 + 1152), "block_plan_total_bytes": positions * (layers * (1920 + 1152 + 384 + 1152 + 1152 + 1152 + 1152) + 1152),
            "layer_plan_leaves": positions * (layers * 2 + 1), "block_plan_leaves": positions * (layers * 7 + 1),
        },
        "a_unconditional": {"boundary": {"leaves": b_all["leaves"], "raw_bytes": b_all["raw_bytes"]}, "kv_rows": {"leaves": kv_l, "raw_bytes": kv_b},
                            "sampled_tokens": {"leaves": sp_l, "raw_bytes": sp_b}, "B_0_raw": b_all["raw_bytes"] + kv_b + sp_b, "L_0": b_all["leaves"] + kv_l + sp_l},
        "b_second_stage_sum": {"B_raw": ru_all["raw_bytes"] - kv_b - sp_b, "L": ru_all["leaves"] - kv_l - sp_l, "RUs": len(rus)},
        "per_RU": {r["tree"]: {"phase": r["phase"], "positions": r["positions"], "L_r": r["leaves"] - r["kv_rows"]["leaves"] - r["sampled_token"]["leaves"],
                               "B_r_raw": r["raw_bytes"] - r["kv_rows"]["bytes"] - r["sampled_token"]["bytes"], "v_out_bytes": r["kv_rows"]["bytes"] + r["sampled_token"]["bytes"]} for r in rus},
    }

    doc = {
        "schema": "veritor/cperf/census/v1", "store": str(args.store), "profile_id": reader.profile_id, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.perf_counter() - t0, 3),
        "framing": {"leaf": hashing.FRAME_DESCRIPTION, "leaf_tag": hashing.LEAF_TAG, "fixed_overhead_per_leaf_bytes_excluding_id": 4 + LEAF_TAG_B + 8 + 8 + 8 + 8,
                    "note": "framed_preimage_bytes = bytes SHA-256 actually consumes per leaf (tag + id + dtype + shape text + value with length prefixes)"},
        "totals": {"ru": ru_all, "ru_by_phase": dict(by_phase), "boundary": b_all, "weights": weights, "commitments_json_bytes": commitments_bytes,
                   "commitments_trees": len((commitments or {}).get("trees", {})) if commitments else None,
                   "ru_index_json_bytes": sum(r["index_json_bytes"] for r in rus), "ru_values_bin_bytes": sum(r["values_bin_bytes"] for r in rus)},
        "plan_request_RU": plan_request,
        "plan_step_RU": plan_step,
        "boundary_trees": [{k: v for k, v in b.items() if k != "rows"} for b in bounds],
        "boundary_rows_example": bounds[0]["rows"] if bounds else None,
        "ru_trees": [{k: v for k, v in r.items() if k != "by_op"} for r in rus],
        "ru_by_op_example_decode": next((r["by_op"] for r in rus if r["phase"] == "decode"), None),
        "ru_by_op_example_prefill": next((r["by_op"] for r in rus if r["phase"] == "prefill"), None),
        "distinct_rows": [{"op_port": k[0], "dtype": k[1], "shape": list(k[2]), "nbytes": k[3], "count": v} for k, v in sorted(ops.items(), key=lambda kv: kv[0][0])],
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
    compact = {k: doc[k] for k in ("totals", "plan_request_RU", "plan_step_RU")}
    compact["plan_step_RU"] = {k: v for k, v in compact["plan_step_RU"].items() if k != "per_RU"}
    compact["boundary_rows_example"] = doc["boundary_rows_example"]
    if args.ops:
        compact["distinct_rows"] = doc["distinct_rows"]
    print("CPERF-CENSUS-JSON " + json.dumps(compact, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
