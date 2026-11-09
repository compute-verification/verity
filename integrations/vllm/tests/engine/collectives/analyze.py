"""Torch-free evidence over the per-rank raw logs of `tp.capture` (numpy only; runs on the pod or on small logs).

    python -m tests.engine.collectives.analyze inventory RUN_DIR [--out PATH]        collectives per rank: op, step, phase, module,
                                                                                    operand/result views, ordinal, position in the layer
    python -m tests.engine.collectives.analyze roots RUN_DIR --tp1-log LOG [--out]   Root events per rank vs the TP=1 log: replicated /
                                                                                    dim0 / dim1 sharded, fused (qkv, gate_up) shards
    python -m tests.engine.collectives.analyze body RUN_DIR [--out]                  numerical body of every snapshotted collective:
                                                                                    all_reduce == bf16(f32(a)+f32(b)) RNE, both ranks equal;
                                                                                    all_gather == concat(shards, dim)
    python -m tests.engine.collectives.analyze determinism RUN_A RUN_B [--out]       run-to-run: snapshot sha256 of every collective's
                                                                                    operands/results aligned by (rank, step, op, ordinal),
                                                                                    plus every snapshot entry of every step
    python -m tests.engine.collectives.analyze unsupported RES.json [RES.json ...]   the exact Unsupported list of `resolve_log` outputs

`RUN_DIR` is a `tp2_capture` output: `log.rank<r>.jsonl.gz`, `rank<r>/snapshots.json`, `rank<r>/values/<step>/<sha>.bin`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from verity_vllm.observe.events import DispatchEnter, DispatchExit, Root, RunHeader, StepStart, TritonLaunch, ViewDesc, views_in
from verity_vllm.observe.log import read_log

COLLECTIVE_OPS = ("vllm.all_reduce", "vllm.all_gather", "vllm.reduce_scatter", "c10d.", "_c10d_functional.", "_C.vocab_parallel_embedding",
                  "vllm.all_reduce_symmetric")


def is_collective_op(op: str) -> bool:
    return any(op.startswith(p) for p in COLLECTIVE_OPS)


def rank_logs(run: Path) -> dict[int, Path]:
    out = {}
    for p in sorted(run.glob("log.rank*.jsonl.gz")):
        m = re.match(r"log\.rank(\d+)\.jsonl\.gz$", p.name)
        if m:
            out[int(m.group(1))] = p
    if not out:
        raise SystemExit(f"no log.rank<r>.jsonl.gz under {run}")
    return out


def view_json(v: ViewDesc | None) -> dict | None:
    if v is None:
        return None
    return {"alloc": v.alloc, "offset": v.offset, "shape": list(v.shape), "strides": list(v.strides), "dtype": v.dtype, "numel": v.numel}


def first_view(x: Any) -> ViewDesc | None:
    for v in views_in(x):
        return v
    return None


def position_of(module: str | None, op: str) -> str:
    """Where a collective sits, from the module-path hint of the forward hooks (diagnostic only)."""
    if not module:
        return "?"
    m = re.match(r"^model\.layers\.(\d+)\.self_attn\.o_proj$", module)
    if m:
        return f"layer {m.group(1)}: after o_proj (attention output, row-parallel)"
    m = re.match(r"^model\.layers\.(\d+)\.mlp\.down_proj$", module)
    if m:
        return f"layer {m.group(1)}: after down_proj (MLP output, row-parallel)"
    if module.endswith("embed_tokens"):
        return "embedding: sum of the vocab-sharded lookups" if "all_reduce" in op else "embedding: fused masked lookup of this rank's vocab shard"
    if "logits_processor" in module or module.endswith("lm_head"):
        return "logits: gather of the vocab-sharded lm_head output"
    return module


# ---------------------------------------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------------------------------------


def scan_log(path: Path) -> dict[str, Any]:
    """One pass: header, per-step collective events (with results), op histogram, roots, steps."""
    header: RunHeader | None = None
    enters: dict[int, DispatchEnter] = {}
    exits: dict[int, Any] = {}
    coll: list[DispatchEnter] = []
    hist: Counter = Counter()
    tri: Counter = Counter()
    roots: list[Root] = []
    steps: list[StepStart] = []
    n = 0
    children: dict[int, list[str]] = defaultdict(list)
    for ev in read_log(path):
        n += 1
        if isinstance(ev, RunHeader):
            header = ev
        elif isinstance(ev, DispatchEnter):
            hist[ev.op] += 1
            if is_collective_op(ev.op):
                coll.append(ev)
                enters[ev.id] = ev
            if ev.parent is not None and ev.parent in enters:
                children[ev.parent].append(ev.op)
        elif isinstance(ev, DispatchExit):
            if ev.id in enters:
                exits[ev.id] = ev.outs
        elif isinstance(ev, TritonLaunch):
            tri[ev.fn] += 1
            if ev.parent is not None and ev.parent in enters:
                children[ev.parent].append("triton:" + ev.fn.rsplit(".", 1)[-1])
        elif isinstance(ev, Root):
            roots.append(ev)
        elif isinstance(ev, StepStart):
            steps.append(ev)
    return {"header": header, "collectives": coll, "exits": exits, "children": children, "hist": hist, "triton": tri, "roots": roots,
            "steps": steps, "events": n}


def inventory(run: Path) -> dict[str, Any]:
    doc: dict[str, Any] = {"schema": "verity-capture/tp2-inventory/v0", "run": str(run), "ranks": {}}
    for r, p in rank_logs(run).items():
        s = scan_log(p)
        h = s["header"]
        ordinal: Counter = Counter()
        rows = []
        for ev in s["collectives"]:
            k = (ev.step, ev.op)
            o = ordinal[k]
            ordinal[k] += 1
            a0 = first_view(ev.args)
            out = first_view(s["exits"].get(ev.id))
            arg_views = [view_json(v) for v in views_in(ev.args)]
            rows.append({"id": ev.id, "step": ev.step, "phase": ev.phase, "op": ev.op, "ordinal": o, "module": ev.module,
                         "position": position_of(ev.module, ev.op), "arg0": view_json(a0), "out": view_json(out), "arg_views": arg_views,
                         "out_variant": out is None and s["exits"].get(ev.id) is None,
                         "kwargs": {k: (v if not isinstance(v, ViewDesc) else view_json(v)) for k, v in (ev.kwargs or {}).items()},
                         "scalar_args": [a for a in ev.args if not isinstance(a, (ViewDesc, list, tuple, dict))],
                         "children": s["children"].get(ev.id, [])})
        per_step: dict[int, Counter] = defaultdict(Counter)
        for row in rows:
            per_step[row["step"]][row["op"]] += 1
        by_op = Counter(row["op"] for row in rows)
        by_pos = Counter(re.sub(r"layer \d+", "layer i", row["position"]) for row in rows)
        rk = {"log": p.name, "events": s["events"], "steps": len(s["steps"]), "roots": len(s["roots"]),
              "root_numel": sum(x.view.numel for x in s["roots"]), "profile_id": h.profile_id if h else None,
              "tp2_notes": (h.notes.get("tp2") if h else None) and {k: v for k, v in h.notes["tp2"].items() if k != "facts"},
              "dispatch_ops": sum(s["hist"].values()), "distinct_ops": len(s["hist"]), "triton_launches": sum(s["triton"].values()),
              "collectives_total": len(rows), "collectives_by_op": dict(by_op), "collectives_by_position": dict(by_pos),
              "collectives_per_step": {str(k): dict(v) for k, v in sorted(per_step.items())},
              "op_histogram": dict(s["hist"].most_common()), "triton_histogram": dict(s["triton"].most_common()),
              "collectives": rows}
        doc["ranks"][str(r)] = rk
    return doc


# ---------------------------------------------------------------------------------------------------------
# roots vs the TP=1 log
# ---------------------------------------------------------------------------------------------------------


def roots_of(path: Path) -> dict[str, Root]:
    out: dict[str, Root] = {}
    for ev in read_log(path):
        if isinstance(ev, Root):
            out.setdefault(ev.name, ev)
        elif isinstance(ev, StepStart):
            break
    return out


def classify_shard(full: tuple[int, ...], part: tuple[int, ...], world: int, name: str, cfg: dict) -> str:
    if full == part:
        return "replicated"
    if len(full) == len(part) == 2:
        if part[0] * world == full[0] and part[1] == full[1]:
            if "qkv_proj" in name:
                return "dim0: fused [q | k | v] rows, each split by heads across ranks (QKVParallelLinear)"
            if "gate_up_proj" in name:
                return "dim0: fused [gate | up] rows, each halved (MergedColumnParallelLinear)"
            if "embed_tokens" in name or "lm_head" in name:
                return "dim0: contiguous vocab range of this rank (VocabParallelEmbedding, padded vocab)"
            return "dim0: contiguous output-feature slice (ColumnParallelLinear)"
        if part[1] * world == full[1] and part[0] == full[0]:
            return "dim1: contiguous input-feature slice (RowParallelLinear)"
    if len(full) == len(part) == 1 and part[0] * world == full[0]:
        if "qkv_proj" in name:
            return "dim0: fused [q | k | v] bias, each split by heads"
        return "dim0: output-feature slice (bias)"
    if len(full) == len(part) == 5 and part[2] * world == full[2]:
        return "KV cache: heads split across ranks"
    return f"other: {list(full)} -> {list(part)}"


def roots_report(run: Path, tp1_log: Path) -> dict[str, Any]:
    full = roots_of(tp1_log)
    doc: dict[str, Any] = {"schema": "verity-capture/tp2-roots/v0", "run": str(run), "tp1_log": str(tp1_log), "tp1_roots": len(full),
                           "tp1_root_numel": sum(r.view.numel for r in full.values()), "ranks": {}}
    for r, p in rank_logs(run).items():
        part = roots_of(p)
        world = 2
        for ev in read_log(p):
            if isinstance(ev, RunHeader):
                world = int((ev.notes.get("tp2") or {}).get("world_size", 2))
            break
        rows = []
        kinds: Counter = Counter()
        numel_part = 0
        for name, ev in part.items():
            f = full.get(name)
            numel_part += ev.view.numel
            kind = classify_shard(f.view.shape, ev.view.shape, world, name, {}) if f else "not in TP=1 log"
            kinds[kind.split(":")[0]] += 1
            rows.append({"name": name, "role": ev.role, "tp1_shape": list(f.view.shape) if f else None, "shape": list(ev.view.shape),
                         "dtype": ev.view.dtype, "kind": kind})
        missing = sorted(set(full) - set(part))
        doc["ranks"][str(r)] = {"roots": len(part), "root_numel": numel_part, "numel_ratio_vs_tp1": round(numel_part / max(1, doc["tp1_root_numel"]), 4),
                                "kinds": dict(kinds), "missing_vs_tp1": missing, "extra_vs_tp1": sorted(set(part) - set(full)),
                                "layer0_and_globals": [x for x in rows if ".layers." not in x["name"] or ".layers.0." in x["name"]],
                                "rows": rows}
    return doc


# ---------------------------------------------------------------------------------------------------------
# snapshots, numerical body, determinism
# ---------------------------------------------------------------------------------------------------------


def load_index(run: Path, rank: int) -> list[dict]:
    p = run / f"rank{rank}" / "snapshots.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())["entries"]


def entry_bytes(run: Path, rank: int, e: dict) -> bytes | None:
    f = e.get("file")
    if not f:
        return None
    p = run / f"rank{rank}" / f
    return p.read_bytes() if p.exists() else None


def view_array(raw: bytes, e: dict) -> np.ndarray:
    """The logical elements of a snapshotted view (spanned-storage-bytes layout) as a uint16/uint32/... array in logical order."""
    v = ViewDesc.from_json(e["view"])
    width = v.itemsize
    kind = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[width]
    flat = np.frombuffer(raw, dtype=kind)
    # spanned bytes start at the view's first element (min offset); strides are in elements
    base = v.offset * width - int(e["byte_offset"])
    assert base % width == 0
    strides = tuple(s * width for s in v.strides)
    return np.lib.stride_tricks.as_strided(flat[base // width:], shape=v.shape, strides=strides, writeable=False).copy()


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_rne(f: np.ndarray) -> np.ndarray:
    u = f.astype(np.float32).view(np.uint32).astype(np.uint64)
    nan = np.isnan(f)
    bias = ((u >> 16) & 1) + 0x7FFF
    out = ((u + bias) >> 16).astype(np.uint16)
    out[nan] = 0x7FC0
    return out


def collective_entries(run: Path, inv_rank: dict) -> dict[tuple, dict]:
    """(step, op, ordinal) -> {'arg0': entry, 'out': entry} from this rank's snapshot index."""
    by_event = defaultdict(dict)
    rank = int(re.search(r"rank(\d+)", inv_rank["log"]).group(1))
    for e in load_index(run, rank):
        by_event[e["event"]][e["path"]] = e
    out = {}
    for row in inv_rank["collectives"]:
        ents = by_event.get(row["id"], {})
        out[(row["step"], row["op"], row["ordinal"])] = {"arg0": ents.get("args[0]"), "out": ents.get("outs"), "row": row}
    return out


def body_report(run: Path) -> dict[str, Any]:
    inv = inventory(run)
    ranks = sorted(int(r) for r in inv["ranks"])
    ents = {r: collective_entries(run, inv["ranks"][str(r)]) for r in ranks}
    keys = sorted(set.intersection(*(set(ents[r]) for r in ranks)))
    doc: dict[str, Any] = {"schema": "verity-capture/tp2-body/v0", "run": str(run), "ranks": ranks, "checked": [], "summary": Counter()}
    for k in keys:
        step, op, o = k
        rows = {r: ents[r][k] for r in ranks}
        res: dict[str, Any] = {"step": step, "op": op, "ordinal": o, "module": rows[ranks[0]]["row"]["module"]}
        outs = {r: rows[r]["out"] for r in ranks}
        ins = {r: rows[r]["arg0"] for r in ranks}
        if any(e is None or not e.get("file") for e in list(outs.values()) + list(ins.values())):
            res["status"] = "no snapshot bytes (root alloc, oversize, or non-snapshot step)"
            doc["checked"].append(res)
            doc["summary"]["unchecked"] += 1
            continue
        res["out_sha_equal_across_ranks"] = len({outs[r]["sha256"] for r in ranks}) == 1
        res["in_sha"] = {str(r): ins[r]["sha256"][:16] for r in ranks}
        res["out_sha"] = {str(r): outs[r]["sha256"][:16] for r in ranks}
        try:
            a = {r: view_array(entry_bytes(run, r, ins[r]), ins[r]) for r in ranks}
            b = {r: view_array(entry_bytes(run, r, outs[r]), outs[r]) for r in ranks}
            dt = ViewDesc.from_json(ins[ranks[0]]["view"]).dtype
            res["dtype"] = dt
            res["shape_in"], res["shape_out"] = list(a[ranks[0]].shape), list(b[ranks[0]].shape)
            if op.startswith("vllm.all_reduce") and dt == "bfloat16":
                acc = np.zeros(a[ranks[0]].shape, np.float32)
                for r in ranks:
                    acc = acc + bf16_to_f32(a[r])
                pred = f32_to_bf16_rne(acc)
                res["body"] = "bf16(sum_r f32(shard_r)) round-to-nearest-even"
                res["body_matches_rank_outs"] = {str(r): bool(np.array_equal(pred, b[r])) for r in ranks}
                # sensitivity: would a different rounding/order be distinguishable?  (2 ranks: a+b is commutative; the only freedom is
                # the accumulation width/rounding)
                pred_bf16_chain = a[ranks[0]]
                for r in ranks[1:]:
                    pred_bf16_chain = f32_to_bf16_rne(bf16_to_f32(pred_bf16_chain) + bf16_to_f32(a[r]))
                res["bf16_chain_equals_f32_sum"] = bool(np.array_equal(pred_bf16_chain, pred))
                mism = int(np.count_nonzero(pred != b[ranks[0]]))
                res["mismatching_elements_vs_rank0_out"] = mism
                res["shards_identical"] = bool(np.array_equal(a[ranks[0]], a[ranks[-1]]))
            elif op.startswith("vllm.all_gather"):
                dim = rows[ranks[0]]["row"]["scalar_args"][0] if rows[ranks[0]]["row"]["scalar_args"] else -1
                dim = int(dim)
                pred = np.concatenate([a[r] for r in ranks], axis=dim)
                res["body"] = f"concat(shard_0..shard_{len(ranks) - 1}, dim={dim})"
                res["body_matches_rank_outs"] = {str(r): bool(np.array_equal(pred, b[r])) for r in ranks}
            else:
                res["body"] = "not modelled here"
            res["status"] = "ok"
        except Exception as e:  # noqa: BLE001
            res["status"] = f"error: {e!r}"
        doc["checked"].append(res)
        ok = res.get("status") == "ok" and all(res.get("body_matches_rank_outs", {}).values()) and res["out_sha_equal_across_ranks"]
        doc["summary"][("body_ok" if ok else "body_mismatch") if "body_matches_rank_outs" in res else "unmodelled"] += 1
    doc["summary"] = dict(doc["summary"])
    return doc


def determinism_report(a: Path, b: Path) -> dict[str, Any]:
    inv_a, inv_b = inventory(a), inventory(b)
    doc: dict[str, Any] = {"schema": "verity-capture/tp2-determinism/v0", "run_a": str(a), "run_b": str(b), "ranks": {}}
    for r in sorted(set(inv_a["ranks"]) & set(inv_b["ranks"])):
        ra, rb = inv_a["ranks"][r], inv_b["ranks"][r]
        ea, eb = collective_entries(a, ra), collective_entries(b, rb)
        keys = sorted(set(ea) & set(eb))
        coll = Counter()
        mism = []
        for k in keys:
            for part in ("arg0", "out"):
                x, y = ea[k][part], eb[k][part]
                if x is None or y is None or not x.get("sha256") or not y.get("sha256"):
                    coll[f"{part}_unsnapshotted"] += 1
                elif x["sha256"] == y["sha256"]:
                    coll[f"{part}_equal"] += 1
                else:
                    coll[f"{part}_differ"] += 1
                    mism.append({"key": list(k), "part": part, "a": x["sha256"][:16], "b": y["sha256"][:16]})
        # every snapshot entry of every step, aligned by (step, per-step ordinal of (path))
        def all_entries(run: Path, rank: str = r) -> dict[tuple, str]:
            out = {}
            seen: Counter = Counter()
            for e in load_index(run, int(rank)):
                key0 = (e.get("step"), e["path"], e.get("dtype"), tuple(e.get("shape", [])))
                out[key0 + (seen[key0],)] = e.get("sha256")
                seen[key0] += 1
            return out
        sa, sb = all_entries(a), all_entries(b)
        common = set(sa) & set(sb)
        eq = sum(1 for k in common if sa[k] == sb[k])
        per_step_differ: Counter = Counter()
        for k in common:
            if sa[k] != sb[k]:
                per_step_differ[k[0]] += 1
        doc["ranks"][r] = {"collectives_compared": len(keys), "collectives_only_in_a": len(set(ea) - set(eb)), "collectives_only_in_b": len(set(eb) - set(ea)),
                           "collective_shas": dict(coll), "collective_mismatches": mism[:20],
                           "all_snapshot_entries": {"a": len(sa), "b": len(sb), "common_keys": len(common), "equal": eq, "differ": len(common) - eq,
                                                    "differ_per_step": {str(k): v for k, v in sorted(per_step_differ.items())}},
                           "events_a": ra["events"], "events_b": rb["events"], "dispatch_ops_a": ra["dispatch_ops"], "dispatch_ops_b": rb["dispatch_ops"]}
    ta, tb = a / "tokens.json", b / "tokens.json"
    if ta.exists() and tb.exists():
        ga = [q["generated_token_ids"] for q in json.loads(ta.read_text())["requests"]]
        gb = [q["generated_token_ids"] for q in json.loads(tb.read_text())["requests"]]
        doc["tokens_equal"] = ga == gb
    return doc


# ---------------------------------------------------------------------------------------------------------
# Unsupported list
# ---------------------------------------------------------------------------------------------------------


def unsupported_report(paths: list[Path]) -> dict[str, Any]:
    doc: dict[str, Any] = {"schema": "verity-capture/tp2-unsupported/v0", "resolutions": {}}
    for p in paths:
        rep = json.loads(p.read_text())
        ops: Counter = Counter()
        reasons: dict[str, Counter] = defaultdict(Counter)
        for u in rep.get("unsupported_examples", []):
            reasons[u["op"]][u.get("reason", "")] += 1
        hist = rep.get("op_histogram") or rep.get("histogram") or {}
        for op, d in hist.items():
            if isinstance(d, dict):
                n = d.get("unsupported") or sum(v for k, v in d.items() if k.startswith("Unsupported") or k == "unsupported")
                if n:
                    ops[op] = n
        doc["resolutions"][str(p)] = {"totals": rep.get("totals"), "unsupported_ops": dict(ops.most_common()),
                                      "reasons": {k: dict(v) for k, v in reasons.items()}, "roots": rep.get("roots"),
                                      "header_mismatches": rep.get("header_mismatches"), "by_spec": rep.get("by_spec"),
                                      "quarantine_specs": rep.get("quarantine_specs"), "notes": rep.get("notes", [])[:10],
                                      "examples": rep.get("unsupported_examples", [])[:12],
                                      "steps": [{k: s.get(k) for k in ("step", "phase", "rows", "roots", "events", "computes", "transparent", "unsupported", "error")}
                                                for s in rep.get("steps", [])]}
    return doc


# ---------------------------------------------------------------------------------------------------------


def _dump(doc: dict, out: str | None, default: Path) -> None:
    p = Path(out) if out else default
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"-> {p}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("inventory"); s.add_argument("run"); s.add_argument("--out")
    s = sub.add_parser("roots"); s.add_argument("run"); s.add_argument("--tp1-log", required=True); s.add_argument("--out")
    s = sub.add_parser("body"); s.add_argument("run"); s.add_argument("--out")
    s = sub.add_parser("determinism"); s.add_argument("run_a"); s.add_argument("run_b"); s.add_argument("--out")
    s = sub.add_parser("unsupported"); s.add_argument("res", nargs="+"); s.add_argument("--out")
    a = ap.parse_args(argv)
    if a.cmd == "inventory":
        doc = inventory(Path(a.run))
        for r, rk in doc["ranks"].items():
            print(f"rank {r}: {rk['events']} events, {rk['steps']} steps, {rk['roots']} roots ({rk['root_numel']} elems), "
                  f"{rk['dispatch_ops']} dispatch ops ({rk['distinct_ops']} distinct), {rk['triton_launches']} triton; "
                  f"collectives {rk['collectives_total']} {rk['collectives_by_op']}\n  positions: {rk['collectives_by_position']}")
        _dump(doc, a.out, Path(a.run) / "tp2_inventory.json")
    elif a.cmd == "roots":
        doc = roots_report(Path(a.run), Path(a.tp1_log))
        for r, rk in doc["ranks"].items():
            print(f"rank {r}: {rk['roots']} roots vs TP=1 {doc['tp1_roots']}; numel ratio {rk['numel_ratio_vs_tp1']}; kinds {rk['kinds']}; "
                  f"missing {rk['missing_vs_tp1'][:5]} extra {rk['extra_vs_tp1'][:5]}")
        _dump(doc, a.out, Path(a.run) / "tp2_roots.json")
    elif a.cmd == "body":
        doc = body_report(Path(a.run))
        print(f"body: {doc['summary']}")
        for c in doc["checked"][:8]:
            print("  ", {k: v for k, v in c.items() if k in ("step", "op", "ordinal", "module", "status", "body", "body_matches_rank_outs",
                                                                "out_sha_equal_across_ranks", "shape_in", "shape_out", "bf16_chain_equals_f32_sum")})
        _dump(doc, a.out, Path(a.run) / "tp2_body.json")
    elif a.cmd == "determinism":
        doc = determinism_report(Path(a.run_a), Path(a.run_b))
        for r, rk in doc["ranks"].items():
            print(f"rank {r}: collectives {rk['collectives_compared']} {rk['collective_shas']}; all entries {rk['all_snapshot_entries']}")
        print(f"tokens_equal={doc.get('tokens_equal')}")
        _dump(doc, a.out, Path(a.run_a) / "tp2_determinism.json")
    elif a.cmd == "unsupported":
        doc = unsupported_report([Path(p) for p in a.res])
        for p, d in doc["resolutions"].items():
            print(f"{p}: totals {d['totals']}\n  unsupported ops {d['unsupported_ops']}\n  reasons {d['reasons']}")
        _dump(doc, a.out, Path(a.res[0]).parent / "tp2_unsupported.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
