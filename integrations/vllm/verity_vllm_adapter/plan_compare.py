"""Plan comparison on one fixed slice (workstream C, experiment 3, decision 5; RC 2026-09-07).

The slice: **one decode step of one request** of an eager B1 bundle -- every node instance at position
``t`` (368 rows: 1 embedding, 57 normed rows, 56 residual rows, 112 GEMM rows, 28 bias adds, 56 RoPE
rows, 28 attention rows, 28 SiLU rows, 1 logits row, 1 token) -- with the same circuit, configuration,
protected output (the token sampled at ``t``, a check output bound to the production record) and the
same bound values around it (the weights, the token at ``t``, the K/V rows of positions ``< t``).  The
canonical values are the same under every plan (memo §2: leaves and units never define value
identities); the plans differ in which of them are *committed* (interfaces), how the commitment is
grouped, and what one obligation re-executes:

1. **fine**: the bundle's own plan -- every row a VU, one RU tree per (request, step); every
   obligation is one registered same-runtime relation (the sampler: W3's independent one).
2. **composite**: one VU per decoder layer at ``t`` (plus one embedding VU and one head VU =
   final norm + lm_head + sampler).  Interfaces = the 4 rows a layer hands on (down_proj out, the
   post-attention residual, the roped K row and the bias-added qkv row the attention of later
   positions reads); the other 9 rows per layer are interior and *not committed*.  The obligation
   re-executes the layer from its openings by chaining the registered relations in dataflow order
   (interior inputs come from the recomputation, never from an opening) and compares the 4 output
   rows with the committed ones.
3. **regrouped**: plan 1's VUs with another commitment-unit grouping -- (a) one tree per
   (request, step, layer), (b) one tree per request over all its steps.  Computed from the
   bundle's leaf index (leaf hashing is identical; only tree nodes, roots and path lengths change);
   labelled analytical, not re-executed.
4. **coordinate** (W3c): one VU per GEMM output word (the 113 GEMM/lm_head rows of the slice =
   797,056 coordinates), the row commitment unchanged, weights opened per row block (format v2); a
   sample of coordinates is checked with W4b's independent chain model (``coordinate_check``) and
   the slice total is extrapolated from the sample and labelled so.  Non-GEMM rows keep plan 1.

Columns per plan: logical values (circuit rows / scalars, the same for all; committed rows /
bytes per plan), committed bytes (leaf values + tree nodes + roots), witness bytes (values +
paths) to check the whole slice, checked obligations, ``Bound`` (vacuous here: every protected
output is a check output, root ``out_bits = 0`` -- reported as such), reconstruction work (rows
recomputed but not committed) and authentication work (path hashes).

    # GPU (torch venv): plans 1 and 2 executed on the slice
    PYTHONPATH=. python -m verity_vllm_adapter.plan_compare <bundle> --out <dir> --mode gpu [--request <id>] [--t <pos>]
    # CPU (Python 3.12 + veritor): plan 4 sample + plan 3 arithmetic + the table
    PYTHONPATH=.:../src python -m verity_vllm_adapter.plan_compare <bundle> --out <dir> --mode cpu --coordinates-per-family 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from verity_vllm import dtypes, identity
from verity_vllm import relations as R
from verity_vllm.capture import WEIGHTS_TREE_ID, CaptureReader, Value, ru_tree_id
from verity_vllm.commit import Opening, RangeOpening, TreeCache, load_commitments, verify_opening_in
from verity_vllm.description import Description, NodeInstance
from verity_vllm_adapter.memo_cache import MemoCache

HASH = 32


def path_bytes(o: Opening | RangeOpening) -> int:
    return o.path_bytes() if isinstance(o, RangeOpening) else sum(HASH for p in o.path if p is not None)


def path_hashes(o: Opening | RangeOpening) -> int:
    """Hash evaluations to authenticate the opening: one per path level (Opening) or one per leaf plus
    two per level (RangeOpening, the verifier rehashes every block leaf)."""
    if isinstance(o, RangeOpening):
        return (o.hi - o.lo) + 2 * len(o.path)
    return len(o.path)


# --- the slice --------------------------------------------------------------------------------------------


def choose_slice(reader: CaptureReader, request: str | None, t: int | None) -> tuple[str, int, int]:
    """(request_id, t, step): the last decode step of the chosen request (default: the request with the
    longest prompt, so the attention history is the slice's largest bound input)."""
    if request is None:
        replayed = {r for r, _ in reader.ru_keys()}  # a selective bundle carries every boundary but only J's RUs
        request = max((r for r in reader.request_ids() if r in replayed), key=lambda r: reader.boundary(r).prompt_len)
    b = reader.boundary(request)
    decode_steps = [s for s in b.steps if s.phase == "decode"]
    if not decode_steps:
        raise SystemExit(f"{request} has no decode step")
    if t is None:
        step = decode_steps[-1]
        t = int(step.positions[0])
    else:
        step = b.step_of(t)
        if step is None or step.phase != "decode":
            raise SystemExit(f"t={t} is not a decode position of {request}")
    return request, t, int(step.step)


def slice_instances(description: Description, reader: CaptureReader, rid: str, t: int) -> list[NodeInstance]:
    b = reader.boundary(rid)
    return [inst for inst in description.instances(b) if inst.t == t]


def layer_of(inst: NodeInstance) -> str:
    return inst.node.layer  # "0".."27", "norm", "embed" ...


def is_interface_composite(inst: NodeInstance) -> bool:
    """Plan 2's committed rows: what leaves a layer (or the embedding) and what later positions read."""
    nd = inst.node
    if nd.op_type == "embedding":
        return True
    if nd.op_type == "gemm" and "down_proj" in str(nd.extra.get("module_path", "")):
        return True
    if nd.op_type == "rmsnorm" and "post_attention_layernorm" in nd.op and inst.port == 1:
        return True  # the residual handed to the next layer
    if nd.op_type == "rope" and inst.port == 1:
        return True  # roped K row: read by attention at every later position
    if nd.op_type == "residual_add":
        return True  # bias-added qkv row: attention reads its V slice at every later position
    if nd.op_type == "sampler":
        return True  # the protected output (a boundary leaf too)
    return False


# --- plan 1: fine ---------------------------------------------------------------------------------------------


def run_fine(description: Description, reader: CaptureReader, commitments: dict[str, Any], cache: TreeCache,
             insts: list[NodeInstance], log: Any) -> dict[str, Any]:
    profile_id = reader.profile_id
    ids = [i.output_id(profile_id) for i in insts]
    t0 = time.perf_counter()
    report = R.check(description, reader, commitments, ids, cache=cache)
    wall = time.perf_counter() - t0
    # witness: every opening a VU used (output + inputs), values + paths; weights apart
    values = paths = hashes = 0
    w_values = w_paths = 0
    openings = 0
    by_family: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r, inst in zip(report.results, insts, strict=True):
        boundary = reader.boundary(inst.request_id)
        resolved = description.resolve_args(inst, boundary, profile_id)
        oids = [r.canonical_id]
        for ref in resolved.values():
            if isinstance(ref, dict) and "const" in ref:
                continue
            oids.extend(ref if isinstance(ref, list) else [ref])
        for oid in oids:
            o = cache.open(oid)
            openings += 1
            vb, pb, hb = len(o.value), path_bytes(o), path_hashes(o)
            values += vb
            paths += pb
            hashes += hb
            if oid.startswith("w="):
                w_values += vb
                w_paths += pb
        fam = by_family[inst.node.op_type]
        fam["n"] += 1
        fam["checker_s"] += r.checker_seconds
        fam["input_bytes"] += r.input_bytes
    log(f"plan fine: {len(ids)} VUs checked in {wall:.1f}s; statuses {report.statuses()}; witness {values / 1e6:.1f} MB values + {paths / 1e3:.1f} KB paths "
        f"(weights {w_values / 1e6:.1f} MB); {openings} openings, {hashes} path hashes")
    return {
        "plan": "fine", "vus": len(ids), "checked": len(ids), "statuses": report.statuses(), "all_exact": report.all_exact(),
        "checker_seconds_total": sum(r.checker_seconds for r in report.results), "wall_seconds": wall,
        "witness": {"value_bytes": values, "path_bytes": paths, "witness_bytes": values + paths, "openings": openings, "path_hashes": hashes,
                    "weights_value_bytes": w_values, "weights_path_bytes": w_paths, "activation_value_bytes": values - w_values},
        "families": {k: dict(v) for k, v in by_family.items()},
        "reconstruction": {"rows_recomputed_uncommitted": 0, "note": "every value read directly from its leaf"},
        "results": [r.to_json() for r in report.results],
    }


# --- plan 2: composite layer VUs -------------------------------------------------------------------------------


def composite_units(insts: list[NodeInstance]) -> dict[str, list[NodeInstance]]:
    """VU key -> node instances (dataflow order is the description's node order within a layer)."""
    units: dict[str, list[NodeInstance]] = defaultdict(list)
    for inst in insts:
        nd = inst.node
        if nd.op_type == "embedding":
            units["embedding"].append(inst)
        elif nd.layer.isdigit():
            units[f"layer.{int(nd.layer):02d}"].append(inst)
        else:
            units["head"].append(inst)  # norm.rmsnorm (2 ports), lm_head, sampler
    return dict(sorted(units.items()))


def topo_order(description: Description, insts: list[NodeInstance]) -> list[NodeInstance]:
    """Node instances of one VU ordered so every interior producer precedes its consumers."""
    by_op: dict[str, list[NodeInstance]] = defaultdict(list)
    for i in insts:
        by_op[i.node.op].append(i)
    ops = set(by_op)
    done: set[str] = set()
    order: list[NodeInstance] = []

    def visit(op: str) -> None:
        if op in done or op not in ops:
            return
        for a in description.node(op).args:
            if a.source == "node" and a.op in ops:
                visit(a.op)
        done.add(op)
        order.extend(sorted(by_op[op], key=lambda i: i.port))

    for op in by_op:
        visit(op)
    return order


def run_composite(description: Description, reader: CaptureReader, commitments: dict[str, Any], cache: TreeCache,
                  insts: list[NodeInstance], log: Any) -> dict[str, Any]:
    profile_id = reader.profile_id
    rid = insts[0].request_id
    boundary = reader.boundary(rid)
    units = composite_units(insts)
    results = []
    tot_values = tot_paths = tot_hashes = tot_openings = 0
    tot_w_values = 0
    tot_interior = 0
    tot_checker = 0.0
    t_all = time.perf_counter()
    for key, members in units.items():
        t0 = time.perf_counter()
        ordered = topo_order(description, members)
        recomputed: dict[str, Value] = {}
        interface_out = [i for i in ordered if is_interface_composite(i)]
        interior = [i for i in ordered if not is_interface_composite(i)]
        opened_ids: set[str] = set()
        values = paths = hashes = 0
        w_values = 0
        checker_s = 0.0
        problems: list[str] = []
        exact = 0
        for inst in ordered:
            cid = inst.output_id(profile_id)
            rel = R.REGISTRY.get(inst.node.relation)
            if rel is None or not rel.implemented:
                problems.append(f"{cid}: relation {inst.node.relation} not available")
                break
            inputs: dict[str, Any] = {"__symbols__": description.symbols}
            ok = True
            for name, ref in description.resolve_args(inst, boundary, profile_id).items():
                if isinstance(ref, dict) and "const" in ref:
                    inputs[name] = ref["const"]
                    continue
                refs = ref if isinstance(ref, list) else [ref]
                vals: list[Value] = []
                for r_ in refs:
                    if r_ in recomputed:  # interior of this VU: reconstructed, never opened
                        vals.append(recomputed[r_])
                        continue
                    o = cache.open(r_)
                    if not verify_opening_in(commitments, o):
                        problems.append(f"{r_}: authentication failed")
                        ok = False
                        break
                    if r_ not in opened_ids:
                        opened_ids.add(r_)
                        values += len(o.value)
                        paths += path_bytes(o)
                        hashes += path_hashes(o)
                        if r_.startswith("w="):
                            w_values += len(o.value)
                    vals.append(o.as_value())
                if not ok:
                    break
                inputs[name] = vals if isinstance(ref, list) else vals[0]
            if not ok:
                break
            dtype, shape = description.leaf_spec(inst.node, inst.port)
            t1 = time.perf_counter()
            try:
                predicted = rel.checker(inst, inputs)
            except Exception as e:  # noqa: BLE001
                problems.append(f"{cid}: {type(e).__name__}: {e}"[:200])
                break
            checker_s += time.perf_counter() - t1
            recomputed[cid] = Value(cid, dtype, tuple(shape), predicted, origin="reconstructed")
            if is_interface_composite(inst):
                o = cache.open(cid)
                if not verify_opening_in(commitments, o):
                    problems.append(f"{cid}: output authentication failed")
                    break
                values += len(o.value)
                paths += path_bytes(o)
                hashes += path_hashes(o)
                opened_ids.add(cid)
                if o.value == predicted:
                    exact += 1
                else:
                    problems.append(f"{cid}: recomputed row != committed row (first byte {R.first_divergence(predicted, o.value)['byte']})")
        wall = time.perf_counter() - t0
        status = "exact" if not problems and exact == len(interface_out) else "mismatch"
        results.append({"vu": key, "status": status, "nodes": len(ordered), "interface_rows": len(interface_out), "interior_rows": len(interior),
                        "interface_exact": exact, "problems": problems[:3], "checker_seconds": checker_s, "wall_seconds": wall,
                        "witness": {"value_bytes": values, "path_bytes": paths, "openings": len(opened_ids), "path_hashes": hashes, "weights_value_bytes": w_values}})
        tot_values += values
        tot_paths += paths
        tot_hashes += hashes
        tot_openings += len(opened_ids)
        tot_w_values += w_values
        tot_interior += len(interior)
        tot_checker += checker_s
        log(f"plan composite {key}: {status} {exact}/{len(interface_out)} interface rows, {len(interior)} interior rows recomputed, "
            f"witness {values / 1e6:.1f} MB, {wall:.2f}s")
    statuses: dict[str, int] = defaultdict(int)
    for r_ in results:
        statuses[r_["status"]] += 1
    return {
        "plan": "composite", "vus": len(units), "checked": len(units), "statuses": dict(statuses), "all_exact": all(r_["status"] == "exact" for r_ in results),
        "checker_seconds_total": tot_checker, "wall_seconds": time.perf_counter() - t_all,
        "witness": {"value_bytes": tot_values, "path_bytes": tot_paths, "witness_bytes": tot_values + tot_paths, "openings": tot_openings, "path_hashes": tot_hashes,
                    "weights_value_bytes": tot_w_values, "activation_value_bytes": tot_values - tot_w_values},
        "reconstruction": {"rows_recomputed_uncommitted": tot_interior,
                           "note": "interior rows recomputed by the registered same-runtime relations chained in dataflow order; "
                                   "reconstruction never overwrote a bound value (interface rows are opened and compared)"},
        "results": results,
    }


# --- plan 3: regrouped commitment units (analytical) -------------------------------------------------------


def tree_nodes(leaves: int) -> int:
    return max(2 * leaves - 1, 1)


def path_len(leaves: int) -> int:
    return math.ceil(math.log2(leaves)) if leaves > 1 else 0


def regrouped(description: Description, reader: CaptureReader, rid: str, step: int, insts: list[NodeInstance], leaf_bytes: int) -> dict[str, Any]:
    profile_id = reader.profile_id
    n = len(insts)
    by_layer: dict[str, int] = defaultdict(int)
    for i in insts:
        by_layer[i.node.layer] += 1
    ru = reader.ru(rid, step)
    ru_leaves = len(ru.index.entries)
    assert ru_leaves == n, (ru_leaves, n)
    # (b) one tree per request over all its steps
    all_leaves = sum(len(reader.ru(rid, s).index.entries) for _, s in reader.ru_keys(rid))
    plans = {
        "per_step (bundle)": {"trees": 1, "leaves_per_tree": [n], "path_len": path_len(n), "roots": 1},
        "per_step_per_layer": {"trees": len(by_layer), "leaves_per_tree": sorted(by_layer.values()), "path_len": max(path_len(k) for k in by_layer.values()),
                               "roots": len(by_layer)},
        "per_request_all_steps": {"trees": 1, "leaves_per_tree": [all_leaves], "path_len": path_len(all_leaves), "roots": 1,
                                  "note": f"the request's {len(reader.ru_keys(rid))} RUs folded into one tree of {all_leaves} leaves; the slice's share of roots is 1/{len(reader.ru_keys(rid))}"},
    }
    out = {}
    for name, p in plans.items():
        nodes = sum(tree_nodes(k) for k in p["leaves_per_tree"])
        out[name] = {
            **p, "tree_node_bytes": nodes * HASH, "root_bytes": p["roots"] * HASH,
            "committed_bytes": leaf_bytes + nodes * HASH,
            "witness_path_bytes_per_vu": p["path_len"] * HASH,
            "witness_path_bytes_slice_all_vus": n * p["path_len"] * HASH,
            "path_hashes_slice_all_vus": n * p["path_len"],
            "leaf_hashes": n,
        }
    return {"plan": "regrouped", "vus": n, "leaf_value_bytes": leaf_bytes, "groupings": out,
            "note": "analytical from the bundle's leaf index: leaf hashing and every relation are plan 1's; only tree nodes, roots and path lengths change "
                    "(RU = (request, step) in the bundle); a coarser grouping lengthens paths, a finer one multiplies roots the statement must carry"}


# --- plan 4: coordinate VUs (W3c) ----------------------------------------------------------------------------


def run_coordinate(description: Description, reader: CaptureReader, commitments: dict[str, Any], cache: TreeCache,
                   insts: list[NodeInstance], per_family: int, seed: int, arch: str, log: Any) -> dict[str, Any]:
    import random

    from verity_vllm_adapter import coordinate_check as CC

    profile_id = reader.profile_id
    rng = random.Random(seed)
    gemms = [i for i in insts if i.node.op_type in CC.GEMM_TYPES]
    fams: dict[str, list[NodeInstance]] = defaultdict(list)
    for i in gemms:
        fams[CC.family_of(i.node)].append(i)
    population = 0
    for i in gemms:
        population += int(description.resolve_shape(i.node.port_shape(i.port))[0])
    if not reader.has_weights() or all(reader.weights.is_leaf(str(identity.weight_id(i.node.arg("w").param or "", None))) for i in gemms[:1]):
        return {"plan": "coordinate", "vus": population, "checked": 0, "status": "structural",
                "note": "weights are whole-tensor leaves (format v1): no coordinate-addressable opening; convert with verity_vllm.weights_v2"}
    results = []
    for fam, members in sorted(fams.items()):
        for _ in range(per_family):
            inst = rng.choice(members)
            N = int(description.resolve_shape(inst.node.port_shape(inst.port))[0])
            j = rng.randrange(N)
            t0 = time.perf_counter()
            res = CC.check_coordinate(description, reader, commitments, cache, inst.output_id(profile_id), j, python_gate=False, arch=arch)
            wall = time.perf_counter() - t0
            d = res.to_json()
            d["wall_seconds"] = wall
            results.append(d)
        fr = [r_ for r_ in results if r_["family"] == fam]
        log(f"plan coordinate {fam}: {sum(1 for r_ in fr if r_['status'] == 'exact')}/{len(fr)} exact, opened {np.median([r_['opened_bytes'].get('total', 0) for r_ in fr]):.0f} B, "
            f"checker {np.median([r_['checker_seconds'] for r_ in fr]) * 1e3:.2f} ms, wall {np.median([r_['wall_seconds'] for r_ in fr]):.3f}s")
    statuses: dict[str, int] = defaultdict(int)
    for r_ in results:
        statuses[r_["status"]] += 1
    # extrapolation to the whole slice: per family, coordinates x median opened bytes / checker seconds
    extrap_bytes = 0.0
    extrap_checker = 0.0
    extrap_hashes = 0
    per_fam = {}
    for fam, members in fams.items():
        fr = [r_ for r_ in results if r_["family"] == fam and r_["status"] == "exact"]
        coords = sum(int(description.resolve_shape(i.node.port_shape(i.port))[0]) for i in members)
        if not fr:
            per_fam[fam] = {"coordinates": coords, "sample": 0}
            continue
        ob = float(np.median([r_["opened_bytes"]["total"] for r_ in fr]))
        cs = float(np.median([r_["checker_seconds"] for r_ in fr]))
        # paths: out row + x row (RU tree) + one weight row block (weights tree)
        ru_leaves = len(reader.ru(members[0].request_id, reader.boundary(members[0].request_id).step_of(members[0].t).step).index.entries)
        w_leaves = int(commitments["trees"][WEIGHTS_TREE_ID]["leaf_count"])
        hashes = 2 * path_len(ru_leaves) + path_len(w_leaves)
        per_fam[fam] = {"coordinates": coords, "sample": len(fr), "opened_bytes_median": ob, "checker_seconds_median": cs, "path_hashes_per_coordinate": hashes,
                        "K": fr[0]["K"], "N": fr[0]["N"]}
        extrap_bytes += coords * ob
        extrap_checker += coords * cs
        extrap_hashes += coords * hashes
    return {
        "plan": "coordinate", "vus": population, "gemm_rows": len(gemms), "checked": len(results), "statuses": dict(statuses),
        "all_exact": all(r_["status"] == "exact" for r_ in results), "families": per_fam,
        "extrapolated_slice_all_coordinates": {"witness_bytes": extrap_bytes, "checker_seconds": extrap_checker, "path_hashes": extrap_hashes,
                                               "label": "EXTRAPOLATED from the sample's medians x coordinate counts; not measured"},
        "note": "W3c's coordinate VU: opens the output row, the input row and ONE weight row block (2K bytes) per coordinate; the independent chain model "
                "(gemm.bf16.v1, validated on the A100) predicts one word.  Only GEMM/lm_head rows have coordinate VUs; the slice's other rows keep plan 1.",
        "results": results,
    }


# --- driver -------------------------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["gpu", "cpu"], required=True)
    ap.add_argument("--request", default=None)
    ap.add_argument("--t", type=int, default=None)
    ap.add_argument("--coordinates-per-family", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--arch", default="sm_80")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / f"plan_compare_{args.mode}.log"

    def log(*a: Any) -> None:
        msg = f"[{time.strftime('%H:%M:%S')}] " + " ".join(str(x) for x in a)
        print(msg, flush=True)
        with log_path.open("a") as f:
            f.write(msg + "\n")

    reader = CaptureReader(Path(args.bundle))
    description = Description(reader.description)
    commitments = load_commitments(reader)
    rid, t, step = choose_slice(reader, args.request, args.t)
    insts = slice_instances(description, reader, rid, t)
    profile_id = reader.profile_id
    leaf_bytes = 0
    scalars = 0
    rows_by_type: dict[str, int] = defaultdict(int)
    for i in insts:
        dt, shape = description.leaf_spec(i.node, i.port)
        n = int(np.prod(shape)) if shape else 1
        scalars += n
        leaf_bytes += n * (dtypes.itemsize(dt) or 0)
        rows_by_type[i.node.op_type] += 1
    committed_composite = [i for i in insts if is_interface_composite(i)]
    composite_bytes = 0
    for i in committed_composite:
        dt, shape = description.leaf_spec(i.node, i.port)
        composite_bytes += (int(np.prod(shape)) if shape else 1) * (dtypes.itemsize(dt) or 0)
    slice_doc = {
        "request_id": rid, "t": t, "step": step, "prompt_len": reader.boundary(rid).prompt_len, "phase": "decode",
        "protected_output": f"the token sampled at t={t} (boundary leaf output_token, a check output bound to production.jsonl)",
        "bound_inputs": "weights (v2 row blocks), the token at t, the K/V rows (roped K, bias-added qkv) of positions < t committed in earlier RUs",
        "circuit_rows": len(insts), "circuit_scalars": scalars, "circuit_row_bytes": leaf_bytes, "rows_by_op_type": dict(rows_by_type),
        "committed_rows": {"fine": len(insts), "composite": len(committed_composite), "regrouped": len(insts), "coordinate": len(insts)},
        "committed_value_bytes": {"fine": leaf_bytes, "composite": composite_bytes, "regrouped": leaf_bytes, "coordinate": leaf_bytes},
        "bound": {"bits": 0, "vacuous": True,
                  "reason": "every protected output of the forced-token replay is a check output (compared with the production record at the boundary before "
                            "sampling), so the root's out_bits is 0 and Bound(C, I, theta) = 0 for every plan; the bound does not separate the plans"},
    }
    log(f"slice: request {rid[:8]} t={t} step={step}: {len(insts)} rows / {scalars} scalars / {leaf_bytes / 1e6:.2f} MB; composite commits {len(committed_composite)} rows / {composite_bytes / 1e6:.2f} MB")
    cache = MemoCache(reader)
    doc: dict[str, Any] = {"schema": "verity-vllm/plan-compare/v1", "bundle": str(args.bundle), "profile_id": profile_id, "mode": args.mode, "slice": slice_doc,
                           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if args.mode == "gpu":
        for k_, v in (reader.manifest.get("engine_flags") or {}).items():
            os.environ.setdefault(k_, str(v))
        import verity_vllm_adapter.relations_gpu as gpu_rel

        doc["relations_installed"] = gpu_rel.install()
        import torch

        doc["device"] = torch.cuda.get_device_name(0)
        # warm-up: the first CUDA launch of each kernel is not the plan's cost
        R.check(description, reader, commitments, [i.output_id(profile_id) for i in insts[:40]], cache=cache)
        doc["fine"] = run_fine(description, reader, commitments, cache, insts, log)
        doc["composite"] = run_composite(description, reader, commitments, cache, insts, log)
        # the tree the slice's RU actually has: nodes and paths (for plan 1's committed bytes)
        n = len(insts)
        doc["fine"]["committed"] = {"rows": n, "value_bytes": leaf_bytes, "tree_node_bytes": tree_nodes(n) * HASH, "roots": 1, "root_bytes": HASH,
                                    "committed_bytes": leaf_bytes + tree_nodes(n) * HASH, "leaf_hashes": n}
        m = len(committed_composite)
        doc["composite"]["committed"] = {"rows": m, "value_bytes": composite_bytes, "tree_node_bytes": tree_nodes(m) * HASH, "roots": 1, "root_bytes": HASH,
                                         "committed_bytes": composite_bytes + tree_nodes(m) * HASH, "leaf_hashes": m,
                                         "note": "the bundle commits every row (plan 1); plan 2's commitment is the 4-rows-per-layer subset, sized from the same leaves"}
    else:
        os.environ.setdefault("VERITY_ARCH", args.arch)
        doc["regrouped"] = regrouped(description, reader, rid, step, insts, leaf_bytes)
        doc["coordinate"] = run_coordinate(description, reader, commitments, cache, insts, args.coordinates_per_family, args.seed, args.arch, log)
    (out / f"plan_compare_{args.mode}.json").write_text(json.dumps(doc, indent=1, default=str) + "\n")
    log("DONE")


if __name__ == "__main__":
    main()
