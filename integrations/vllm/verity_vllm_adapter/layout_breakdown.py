"""Commitment layout breakdown of a bundle (contract section 16.4).

Per operator family: distinct logical values (leaves), bytes hashed / copied / retained, leaf count,
and the bytes an *opening* of one selected check costs -- for the GEMM families both the same-runtime
whole-row check and the independent coordinate check, under capture format v1 (whole-tensor weight
leaves) and v2 (one leaf per parameter row, section 16.2).  Then the conditional cost of replaying
one selected request and the amortised cost under stated request-sampling plans.

Inputs: the commit bundle (``ru/*/index.json`` are enumerated; ``values.bin`` is never read), its
``replay.json`` (timings), the production ``timings.json``, optionally the coordinate-check summary
(measured opened bytes and checker work) and the verify report (same-runtime input bytes).

    PYTHONPATH=. python -m verity_vllm_adapter.layout_breakdown <commit_bundle> --out <dir> \
        [--coordinate-summary <coordinate_summary.json>] [--production <production_dir>]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from verity_vllm import identity
from verity_vllm.capture import CaptureReader
from verity_vllm.commit import range_path_shape
from verity_vllm.description import Description
from verity_vllm_adapter.coordinate_check import GEMM_TYPES, family_of

HASH_BYTES = 32


def family_key(node: Any) -> str:
    if node.op_type in GEMM_TYPES:
        return f"gemm/{family_of(node)}"
    fam = node.extra.get("family") or node.op_type or "other"
    return str(fam)


def path_bytes_single(leaf_count: int) -> int:
    """Bytes of one Merkle path in a tree of ``leaf_count`` leaves (one sibling hash per level)."""
    return HASH_BYTES * max(1, math.ceil(math.log2(max(2, leaf_count))))


def path_bytes_range(leaf_count: int, lo: int, hi: int) -> int:
    """Path bytes of a :class:`RangeOpening` over ``[lo, hi)`` in a tree of ``leaf_count`` leaves."""
    return HASH_BYTES * sum(int(left) + int(right) for left, right in range_path_shape(leaf_count, lo, hi))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bundle")
    ap.add_argument("--out", required=True)
    ap.add_argument("--coordinate-summary", default=None)
    ap.add_argument("--production", default=None)
    ap.add_argument("--rates", default="1e-2,1e-3,1e-4")
    args = ap.parse_args(argv)
    t_start = time.perf_counter()
    reader = CaptureReader(args.bundle)
    description = Description(reader.description)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes = {n.op: n for n in description.nodes}
    fam_of_op: dict[str, str] = {}
    per_family: dict[str, dict[str, Any]] = defaultdict(lambda: {"leaves": 0, "bytes": 0, "ops": set(), "phases": defaultdict(int), "row_bytes": defaultdict(int)})
    per_request: dict[str, dict[str, Any]] = defaultdict(lambda: {"leaves": 0, "bytes": 0, "rus": 0, "positions": 0})
    per_phase = defaultdict(lambda: {"leaves": 0, "bytes": 0, "positions": 0})
    store_values = None
    ru_count = 0
    for ru in reader.rus():
        ru_count += 1
        idx = ru.index
        if store_values is None:
            store_values = (ru.dir / "values.bin").exists() and idx.values_nbytes > 0
        preq = per_request[idx.request_id]
        preq["rus"] += 1
        preq["positions"] += len(idx.positions)
        per_phase[idx.phase]["positions"] += len(idx.positions)
        for e in idx.entries:
            cid = e.canonical_id
            op = cid.split("/op=", 1)[1].split("/", 1)[0] if "/op=" in cid else identity.parse(cid).op
            fam = fam_of_op.get(op)
            if fam is None:
                node = nodes.get(op)
                fam = fam_of_op[op] = family_key(node) if node is not None else f"unknown/{op}"
            d = per_family[fam]
            d["leaves"] += 1
            d["bytes"] += e.nbytes
            d["ops"].add(op)
            d["phases"][idx.phase] += 1
            d["row_bytes"][e.nbytes] += 1
            preq["leaves"] += 1
            preq["bytes"] += e.nbytes
            per_phase[idx.phase]["leaves"] += 1
            per_phase[idx.phase]["bytes"] += e.nbytes
    total_bytes = sum(d["bytes"] for d in per_family.values())
    total_leaves = sum(d["leaves"] for d in per_family.values())

    # --- weights: v1 whole-tensor leaves vs v2 row leaves ---
    widx = reader.weights.index
    w_fam: dict[str, dict[str, Any]] = defaultdict(lambda: {"tensors": 0, "bytes": 0, "leaves_v1": 0, "leaves_v2": 0, "params": []})
    param_family: dict[str, str] = {}
    for n in description.nodes:
        for a in n.args:
            if a.source == "weight" and a.param:
                param_family.setdefault(a.param, family_key(n))
    for e in widx.entries:
        fam = param_family.get(e.param, "weights/unbound")
        d = w_fam[fam]
        d["tensors"] += 1
        d["bytes"] += e.nbytes
        d["leaves_v1"] += 1
        rows = e.shape[0] if len(e.shape) >= 2 and e.shape[0] > 1 else 1
        d["leaves_v2"] += rows
        d["params"].append({"param": e.param, "shape": list(e.shape), "nbytes": e.nbytes, "rows_v2": rows})
    w_total_leaves_v1 = sum(d["leaves_v1"] for d in w_fam.values())
    w_total_leaves_v2 = sum(d["leaves_v2"] for d in w_fam.values())
    w_total_bytes = sum(d["bytes"] for d in w_fam.values())

    # --- bytes opened per selected check, per GEMM family (analytic, from the shapes) ---
    coord = json.loads(Path(args.coordinate_summary).read_text()) if args.coordinate_summary else None
    opened: dict[str, Any] = {}
    for fam, d in sorted(per_family.items()):
        if not fam.startswith("gemm/"):
            continue
        ops = sorted(d["ops"])
        node = nodes[ops[0]]
        N = int(description.resolve_shape(node.port_shape(0))[0])
        warg = node.arg("w")
        wid = str(identity.weight_id(warg.param or "", warg.shard))
        we = reader.weights.entry(wid)
        K = int(we.shape[1]) if len(we.shape) >= 2 else 0
        ru_leaves_typical = max(1, round(total_leaves / max(1, ru_count)))
        x_bytes, out_bytes, w_bytes = 2 * K, 2 * N, we.nbytes
        p_ru = path_bytes_single(ru_leaves_typical)
        p_w1 = path_bytes_single(w_total_leaves_v1)
        p_w2_row = path_bytes_single(w_total_leaves_v2)
        # the row block of this parameter in the v2 leaf order: bound the range path by the shape helper
        p_w2_range = path_bytes_range(w_total_leaves_v2, 0, N)
        entry: dict[str, Any] = {
            "K": K, "N": N, "ops": len(ops), "weight_param": we.param, "weight_bytes": w_bytes,
            "same_runtime_row": {
                "v1": {"values": x_bytes + out_bytes + w_bytes, "paths": 2 * p_ru + p_w1, "note": "x row + out row + whole W (one leaf)"},
                "v2": {"values": x_bytes + out_bytes + w_bytes, "paths": 2 * p_ru + p_w2_range, "note": "x row + out row + whole W as one RangeOpening over its N row leaves"},
            },
            "independent_coordinate": {
                "v1": {"values": x_bytes + out_bytes + w_bytes, "paths": 2 * p_ru + p_w1, "note": "no coordinate-addressable weight leaf: the whole W tensor is opened for one column"},
                "v2": {"values": x_bytes + out_bytes + 2 * K, "paths": 2 * p_ru + p_w2_row, "note": "x row + out row (the committed leaf holding the coordinate) + one W row leaf (2K bytes) + paths"},
            },
        }
        for cls in ("same_runtime_row", "independent_coordinate"):
            for ver in ("v1", "v2"):
                c = entry[cls][ver]
                c["total"] = c["values"] + c["paths"]
        entry["reduction_independent_v1_to_v2"] = entry["independent_coordinate"]["v1"]["total"] / max(1, entry["independent_coordinate"]["v2"]["total"])
        if coord is not None:
            fkey = fam.split("/", 1)[1]
            measured = {k: v for k, v in coord.get("by_family_phase", {}).items() if k.startswith(fkey + "/")}
            if measured:
                entry["measured_independent_v2"] = {
                    k: {
                        "opened_bytes_per_coordinate": v.get("opened_bytes_per_coordinate"),
                        "opened_bytes_w": v.get("opened_bytes_w", 0) / max(1, v.get("attempted", 1)),
                        "opened_bytes_paths": v.get("opened_bytes_paths", 0) / max(1, v.get("attempted", 1)),
                        "checker_seconds_per_coordinate": v.get("checker_seconds_per_coordinate"),
                        "exact": v.get("exact"), "attempted": v.get("attempted"),
                    }
                    for k, v in measured.items()
                }
            sr = (coord.get("same_runtime") or {}).get("by_family", {})
            measured_sr = {k: v for k, v in sr.items() if k.startswith(fkey + "/")}
            if measured_sr:
                entry["measured_same_runtime_v2"] = {k: {"input_bytes_per_row": v.get("input_bytes_per_row"), "checker_seconds_per_row": v.get("checker_seconds_per_row"), "exact": v.get("exact"), "attempted": v.get("attempted")} for k, v in measured_sr.items()}
        opened[fam] = entry

    # --- conditional and amortised replay cost ---
    replay = json.loads((Path(args.bundle) / "replay.json").read_text()) if (Path(args.bundle) / "replay.json").exists() else {}
    timing = replay.get("timing", {})
    cap_t = timing.get("capture", {})
    driver = timing.get("driver", {})
    n_req = max(1, len(per_request))
    prod = json.loads((Path(args.production) / "timings.json").read_text()) if args.production and (Path(args.production) / "timings.json").exists() else {}
    prod_wall = (prod.get("recorded_pass") or {}).get("wall_s") or (prod.get("recording_off") or {}).get("wall_s")
    replay_wall = driver.get("wall_s")
    forward_s = cap_t.get("forward_s")
    per_req_bytes = total_bytes / n_req
    per_req_positions = sum(p["positions"] for p in per_request.values()) / n_req
    cost = {
        "requests": n_req,
        "replay_wall_s_total": replay_wall,
        "replay_forward_s_total": forward_s,
        "replay_capture_s": {k: cap_t.get(k) for k in ("snapshot_s", "sync_s", "emit_s", "forward_s")},
        "production_wall_s_total": prod_wall,
        "per_request": {
            "committed_bytes": per_req_bytes,
            "leaves": total_leaves / n_req,
            "positions": per_req_positions,
            "replay_wall_s": (replay_wall / n_req) if replay_wall else None,
            "replay_forward_gpu_s": (forward_s / n_req) if forward_s else None,
            "production_wall_s": (prod_wall / n_req) if prod_wall else None,
        },
        "note": "the replay of a selected request re-runs the whole batch it was served in (the per-request figure is the batch's cost divided by its 16 requests; a per-request replay would recompute only that request's steps, whose cost is bounded by the same figure under batch invariance)",
        "amortised": [],
    }
    for r in [float(x) for x in args.rates.split(",") if x.strip()]:
        cost["amortised"].append({
            "plan": f"sample each served request with probability q={r:g}; a selected request is replayed under the eager-production profile and ALL its interiors are committed and bound (this bundle's layout); unselected requests cost nothing beyond production",
            "q": r,
            "expected_committed_bytes_per_served_request": r * per_req_bytes,
            "expected_replay_wall_s_per_served_request": (r * replay_wall / n_req) if replay_wall else None,
            "expected_replay_gpu_s_per_served_request": (r * forward_s / n_req) if forward_s else None,
            "ratio_to_production_wall": (r * replay_wall / max(prod_wall, 1e-9)) if (replay_wall and prod_wall) else None,
        })

    doc = {
        "schema": "verity-vllm/layout-breakdown/v1",
        "bundle": str(args.bundle),
        "profile_id": reader.profile_id,
        "store_values": bool(store_values),
        "weights_format_version": widx.format_version,
        "conversion": json.loads((Path(args.bundle) / "weights" / "conversion.json").read_text()) if (Path(args.bundle) / "weights" / "conversion.json").exists() else None,
        "totals": {"replay_units": ru_count, "leaves": total_leaves, "bytes": total_bytes, "weights_bytes": w_total_bytes, "weights_leaves_v1": w_total_leaves_v1, "weights_leaves_v2": w_total_leaves_v2},
        "by_family": {
            fam: {
                "logical_values": d["leaves"],
                "leaves": d["leaves"],
                "hashed_bytes": d["bytes"],
                "copied_bytes": d["bytes"],
                "retained_bytes": d["bytes"] if store_values else 0,
                "share_of_bytes": d["bytes"] / max(1, total_bytes),
                "ops": len(d["ops"]),
                "phases": dict(d["phases"]),
                "row_bytes": {str(k): v for k, v in sorted(d["row_bytes"].items())},
            }
            for fam, d in sorted(per_family.items(), key=lambda kv: -kv[1]["bytes"])
        },
        "by_phase": {k: dict(v) for k, v in per_phase.items()},
        "weights_by_family": {fam: {k: v for k, v in d.items() if k != "params"} | {"params": d["params"][:6]} for fam, d in sorted(w_fam.items())},
        "opened_bytes_per_check": opened,
        "replay_cost": cost,
        "seconds": time.perf_counter() - t_start,
    }
    (out_dir / "layout_breakdown.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"families": len(per_family), "leaves": total_leaves, "bytes": total_bytes, "weights_leaves": [w_total_leaves_v1, w_total_leaves_v2], "seconds": round(doc["seconds"], 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
