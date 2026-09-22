#!/usr/bin/env python
"""First-stage boundary per candidate plan (lane COST-D, 2026-09-09; docs/vllm-poc/first-stage-boundary.md).

Paper §5.2: before the challenge the prover commits (i) the designated outputs, (ii) every value that crosses
a replay-unit (RU) boundary — cross-request/shared values (weights, when more than one copy) and RU interface
values.  Interior values are acquired only for the requests the challenge selects.  This script CUTS the
adapter's program description (``description.json`` of a recorded run: 274 op templates, per-row output
widths, producer -> consumer edges) under each candidate plan's RU/CU assignment and counts the rows and
bytes that cross, per position, against what the diagnostic capture records (every kernel-output row).

Every number printed is labelled: MEASURED = counted from the record given on the command line (its
description graph, its ``capture_summary.json`` row/byte totals, its ``phase0.json`` boundary tree);
ANALYTICAL = derived from the plan and the public geometry (the datacenter-1m workload has no record).

    PYTHONPATH=. python scripts/first_stage_boundary.py \
        fixtures/results/PAD-INT/PADINT-B0-20260909T095529Z-EOS0 [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# --------------------------------------------------------------------------------------------------
# plan -> RU/CU assignment of a program node.  A node is identified by (op, layer, family).  The unit
# key returned must be equal for two nodes iff they are in the same RU (per request, per position where
# the plan is position-granular).  ``pos`` is a symbolic position tag so per-position plans separate.
# --------------------------------------------------------------------------------------------------

PRE_OPS = ("input_layernorm", "qkv_proj", "rotary_emb")
ATTN_OPS = ("self_attn.attn",)


def unit_request(node, pos):
    return "request"


def unit_layer(node, pos):
    # RU = (request, decoder layer), all positions.  embed is its own unit; norm + lm_head + sampler = final.
    if node["layer"] in ("norm", "lm_head", "sampler"):
        return "final"
    return f"L{node['layer']}"


def unit_block(node, pos):
    # P3 / cu=block: one CU per root call of Serve: Embedding{t}, LayerPre{l,t}, Attention{l,t},
    # LayerPost{l,t}, Final{t} (compact-planning.md §5.3.2: 47 Embedding, 1,410 x LayerPre/Attention/
    # LayerPost, 16 Final for the 47-position B0).
    layer = node["layer"]
    if layer in ("embed",):
        return f"Embedding@{pos}"
    if layer in ("norm", "lm_head", "sampler"):
        return f"Final@{pos}"
    op = node["op"]
    if any(k in op for k in PRE_OPS):
        return f"LayerPre{layer}@{pos}"
    if any(k in op for k in ATTN_OPS):
        return f"Attention{layer}@{pos}"
    return f"LayerPost{layer}@{pos}"


def unit_operator(node, pos):
    # cu=operator / the fine coordinate plans: RU = one operator instance = one kernel-output row.
    return f"{node['op']}@{pos}"


PLANS = {
    # name: (unit fn, plan families it stands for)
    "request": (unit_request, "whole-request RU: root plan, cu=whole, P1/cu=whole, P3/cu=whole (coordinate or operator VUs)"),
    "layer": (unit_layer, "layer RU (LayerPost{C} unit of the B0 records; INT's `layer`)"),
    "block": (unit_block, "P3 / cu=block: one CU per root call (LayerPre / Attention / LayerPost per layer x position)"),
    "operator": (unit_operator, "cu=operator, P1/cu=operator, P3/cu=operator: one RU per kernel-output row"),
}


def load_graph(desc_path: Path):
    d = json.loads(desc_path.read_text())
    nodes = {n["op"]: n for n in d["nodes"]}
    return d, nodes


def row_bits(node, port):
    if port == 0 or port is None:
        return node["decompositions"][0]["interface"]["output_bits"]
    for o in node.get("outputs") or []:
        if o["port"] == port:
            n = 1
            for s in o["shape"]:
                n *= s
            return n * 16  # bf16 extra ports
    raise KeyError((node["op"], port))


def n_ports(node):
    return int(node.get("ports") or 1)


def cut(nodes, unit_fn, positions_all: int, positions_sampled: int):
    """Count the rows/bytes that cross RU boundaries.  Positions: ``positions_all`` rows for
    positions=all nodes, ``positions_sampled`` for lm_head / sampler nodes.  Attention consumes k/v of
    every earlier position of the same layer: for per-position plans those rows are producer-side boundary
    rows already (counted once at the producer) -- the cut counts each produced row once if ANY consumer
    is in another unit."""
    consumers = defaultdict(set)  # (producer op, port) -> {consumer op}
    for n in nodes.values():
        for a in n["args"]:
            if a.get("source") == "node":
                consumers[(a["op"], a.get("port", 0))].add(n["op"])
    per_pos_rows = per_pos_bytes = 0.0
    designated_rows = designated_bytes = 0.0
    total_rows = total_bytes = 0.0
    for n in nodes.values():
        mult = positions_sampled if n.get("positions") == "sampled" else positions_all
        for port in range(n_ports(n)):
            bits = row_bits(n, port)
            total_rows += mult
            total_bytes += mult * bits / 8
            cons = consumers.get((n["op"], port), set())
            crosses = False
            for c in cons:
                # per-position plans: a consumer at a LATER position (attention's k/v reads) is another unit
                same_pos = unit_fn(n, "t") == unit_fn(nodes[c], "t")
                if not same_pos:
                    crosses = True
                    break
                if nodes[c]["family"] == "attn" and unit_fn(n, "t") != unit_fn(n, "t'"):
                    crosses = True  # k/v rows are read by attention at later positions (other CUs)
                    break
            if n["family"] == "sampler":
                designated_rows += mult
                designated_bytes += mult * 4  # token id, 4 B (the record's int32)
                continue
            if crosses:
                per_pos_rows += mult
                per_pos_bytes += mult * bits / 8
    return dict(
        boundary_rows=per_pos_rows,
        boundary_bytes=per_pos_bytes,
        designated_rows=designated_rows,
        designated_bytes=designated_bytes,
        capture_rows=total_rows,
        capture_bytes=total_bytes,
    )


def dc1m_analytical():
    """datacenter-1m workload (Qwen3-235B-A22B, 94 layers, H=4096, 64Q/4KV heads, hd=128, MoE): the same
    cuts by formula (no record exists).  Widths in bf16 words per position per layer."""
    NL, H, NQ, NKV, HD = 94, 4096, 64, 4, 128
    P, NOUT = 999_999, 200_000
    rows_per_pos, bytes_per_row = 3_574, 6_800  # populations.json census (datacenter-1m.md §2)
    full_bytes = rows_per_pos * bytes_per_row  # 24.3 MB / position, 24.36 TB / request
    w = 2
    layer_cut = NL * 2 * H * w + H * w  # residual + block output per layer (+ embedding out): 2 rows/layer
    # 7 rows / layer as on the B0 graph: qkv row (v read by Attention), RoPE q, RoPE k, attention out,
    # block (MoE) out, and the residual twice (LayerPre -> LayerPost, LayerPost -> next LayerPre)
    block_cut = NL * ((NQ + 2 * NKV) * HD + NQ * HD + NKV * HD + NQ * HD + H + 2 * H) * w
    return {
        "request": dict(boundary_rows=0, boundary_bytes=0, designated_rows=P + 1, designated_bytes=(P + 1) * 4,
                        capture_rows=rows_per_pos * P, capture_bytes=full_bytes * P, positions=P),
        "layer": dict(boundary_rows=(2 * NL + 1) * P, boundary_bytes=layer_cut * P, designated_rows=P + 1,
                      designated_bytes=(P + 1) * 4, capture_rows=rows_per_pos * P, capture_bytes=full_bytes * P, positions=P),
        "block": dict(boundary_rows=7 * NL * P, boundary_bytes=block_cut * P, designated_rows=P + 1,
                      designated_bytes=(P + 1) * 4, capture_rows=rows_per_pos * P, capture_bytes=full_bytes * P, positions=P),
        "operator": dict(boundary_rows=rows_per_pos * P, boundary_bytes=full_bytes * P, designated_rows=P + 1,
                         designated_bytes=(P + 1) * 4, capture_rows=rows_per_pos * P, capture_bytes=full_bytes * P, positions=P),
    }


def fmt_bytes(b):
    for unit, div in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if b >= div:
            return f"{b / div:,.2f} {unit}"
    return f"{b:,.0f} B"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("record", type=Path, help="e2e record dir with adapter/selective/{description,capture_summary,phase0}.json")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    sel = a.record / "adapter" / "selective"
    d, nodes = load_graph(sel / "description.json")
    cs = json.loads((sel / "capture_summary.json").read_text())
    p0 = json.loads((sel / "phase0.json").read_text())
    st = json.loads((sel / "steptrace.json").read_text())
    n_req = p0["requests"]
    positions = sum(v for v in st["summary"]["steps_per_request"].values()) if "steps_per_request" in st["summary"] else None
    # positions processed = prompt tokens + generated - 1 per request (the last sampled token is not forwarded)
    prod = json.loads((a.record / "production_timings.json").read_text())["recorded_pass"]
    P_all = prod["prompt_tokens"] + prod["generated_tokens"] - n_req
    P_out = prod["generated_tokens"]
    out = {"record": str(a.record), "model": d["model"], "profile_id": d["profile_id"], "requests": n_req,
           "positions_all": P_all, "positions_sampled": P_out,
           "measured": {"capture_rows": cs["leaves"], "capture_bytes": cs["bytes_captured"],
                        "phase0_boundary_leaves": p0["boundary_leaves"], "phase0_boundary_leaf_bytes": p0["boundary_leaf_bytes"],
                        "phase0_boundaries_s": p0["timing"]["phase0_boundaries_s"],
                        "phase0_weights_write_s": p0["timing"]["phase0_weights_write_s"],
                        "phase0_commit_s": p0["timing"]["phase0_commit_s"],
                        "weights_tree_leaves": p0["weights"]["tree_leaves"], "weights_bytes": p0["weights"]["bytes"],
                        "capture_forward_s": cs["timing"]["forward_s"], "capture_emit_s": cs["timing"]["emit_s"],
                        "capture_snapshot_s": cs["timing"]["snapshot_s"]},
           "plans": {}}
    for name, (fn, desc) in PLANS.items():
        r = cut(nodes, fn, P_all, P_out)
        r["families"] = desc
        out["plans"][name] = r
    out["dc1m_analytical"] = dc1m_analytical()

    print(f"record {a.record}  model {d['model']}  requests {n_req}  positions {P_all} (+{P_out} output positions)")
    print(f"MEASURED capture: {cs['leaves']:,} rows, {fmt_bytes(cs['bytes_captured'])} "
          f"({cs['leaves'] / P_all:,.1f} rows/position, {fmt_bytes(cs['bytes_captured'] / P_all)}/position); "
          f"graph model {out['plans']['operator']['capture_rows']:,.0f} rows, {fmt_bytes(out['plans']['operator']['capture_bytes'])}")
    print(f"MEASURED phase-0 boundary tree today (cu=whole): {p0['boundary_leaves']} leaves, {p0['boundary_leaf_bytes']} B for {n_req} requests")
    print()
    print("| plan | RU boundary rows / position | RU boundary bytes / position | share of full capture | designated outputs | label |")
    print("|---|---:|---:|---:|---|---|")
    for name, r in out["plans"].items():
        share = r["boundary_bytes"] / r["capture_bytes"] if r["capture_bytes"] else 0
        print(f"| {name} | {r['boundary_rows'] / P_all:,.1f} | {fmt_bytes(r['boundary_bytes'] / P_all)} | {share:.1%} | "
              f"{r['designated_rows']:,.0f} tokens ({fmt_bytes(r['designated_bytes'])}) | ANALYTICAL cut of the MEASURED graph |")
    print()
    print("datacenter-1m (ANALYTICAL, formula census):")
    print("| plan | boundary rows / position | boundary bytes / position | per request | share of full store |")
    print("|---|---:|---:|---:|---:|")
    for name, r in out["dc1m_analytical"].items():
        P = r["positions"]
        share = r["boundary_bytes"] / r["capture_bytes"]
        print(f"| {name} | {r['boundary_rows'] / P:,.0f} | {fmt_bytes(r['boundary_bytes'] / P)} | {fmt_bytes(r['boundary_bytes'])} | {share:.2%} |")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1, sort_keys=True))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
