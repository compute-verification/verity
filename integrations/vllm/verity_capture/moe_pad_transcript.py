"""R17-42 Stage 3a -- the PADDED (fixed-slot) MoE assignment of a recorded row, read off the MOE-01 record (MOE-PAD-SPEC s7.3).

The record's collector writes the MoE of one (token, layer) as MOE-01 rows: one `MoeRouterTopK_v1{E,TOPK,VPT}` instance (outputs `alpha[TOPK]`
f32, `ids[TOPK]` i32), TOPK `MoeExpertGemm_v1` / `MoeExpertGemmW_v1` PAIR rows whose second operand ref points INTO the router's output at
element `TOPK + k` (the pair's expert id `ids[k]`; the W row's third ref at element `k` is `alpha[k]`), and one `MoeSum_v1` over the TOPK
W outputs.  The pair index `p = t*TOPK + k` with `e = ids[p]` IS the packed -> fixed mapping: the padded block of the same (token, layer) has
E slots; slot `e = ids[k]` is ENABLED (`a_e = a_parent AND [e selected]`, R17-43 s5 / LIFTING-SPEC v0.6 A12) and carries `(x, alpha[k], the
pair's output)`; every other slot is DISABLED: its inputs `x` and `W_e` are wired in unchanged (the same Active activation words, the
checkpoint's weights -- NOT replaced by bot), and only the values it produces internally (`g_e`, up/gate, silu, `z_e`) are `bot` -- by
enablement, never by execution.  Those disabled words are explicitly MATERIALIZED and committed (s7; `materialize_bot_slots.py` measures it).

This module builds that assignment for every (step, token, layer) of a row directory and checks what the relations say about it:
  (a) every router instance has exactly TOPK pair rows (Gemm and GemmW), one per k, and MoeSum reads exactly those W outputs in k order;
  (b) at snapshot steps the recorded `ids` are TOPK DISTINCT values in [0, E) (the router's masked argmax) -- an out-of-range or repeated
      id is reported BY NAME, never mapped to a default slot (s2 FixedSelect: INV, not slot 0);
  (c) the population: expanded (E slot bodies per token-layer) vs compact (TOPK executed pairs), reported as separate quantities (s6),
      and the count of DISABLED slots' internally produced module-boundary words = (E - TOPK) * (H + 1) per token-layer exactly (z_e + g_e;
      the Stage 2 artefact's number, at scale) -- the words a padded Commit materializes as enc(bot).
Nothing here changes a row of record: the row keeps its MOE-01 label (R17-42 s8); the output `moe_pad_assignment.json` is evidence of the
padded construction's assignment on a real record.

    python -m verity_capture.moe_pad_transcript <row dir> [--out moe_pad_assignment.json] [--examples 3]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time
from typing import Any

import numpy as np

SPEC_RE = re.compile(r"^(?P<fam>[A-Za-z0-9_]+)\{(?P<args>[^}]*)\}$")
ROUTER, GEMM, GEMMW, SUM = "MoeRouterTopK_v1", "MoeExpertGemm_v1", "MoeExpertGemmW_v1", "MoeSum_v1"


def spec_args(spec: str) -> tuple[str, dict[str, int]]:
    m = SPEC_RE.match(spec or "")
    if not m:
        return spec.split("{")[0], {}
    args = {}
    for kv in m.group("args").split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            try:
                args[k.strip()] = int(v)
            except ValueError:
                args[k.strip()] = v.strip()
    return m.group("fam"), args


def _refs(r: dict) -> list:
    refs = r.get("operand_refs") or []
    if isinstance(refs, str):
        refs = json.loads(refs.replace("'", '"'))
    return refs


def read_instances(path: str):
    op = open
    if path.endswith(".gz"):
        import gzip
        op = gzip.open
    with op(path, "rt") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build(row_dir: str, *, examples: int = 3) -> dict[str, Any]:
    t0 = time.time()
    inst_path = os.path.join(row_dir, "match", "instances.jsonl")
    if not os.path.exists(inst_path):
        inst_path += ".gz"
    routers: dict[int, dict] = {}                                   # router node -> instance
    pairs: dict[int, dict[int, dict[str, dict]]] = collections.defaultdict(dict)   # router node -> k -> {"gemm": inst, "gemmw": inst}
    sums: dict[int, dict] = {}                                      # router node (via its W rows) -> MoeSum instance
    w_out_node_to_router_k: dict[int, tuple[int, int]] = {}
    fam_counts: collections.Counter = collections.Counter()
    envelope: dict[str, int] = {}
    anomalies: list[dict] = []
    KEEP = ("node", "step", "event", "spec", "name", "request", "position", "output_views", "operand_refs")
    moe: list[tuple[str, dict[str, int], dict]] = []
    for r in read_instances(inst_path):
        fam, args = spec_args(r.get("spec", ""))
        if fam not in (ROUTER, GEMM, GEMMW, SUM):
            continue
        fam_counts[fam] += 1
        if fam == ROUTER:
            envelope.setdefault("E", args.get("E")); envelope.setdefault("TOPK", args.get("TOPK")); envelope.setdefault("VPT", args.get("VPT"))
        elif fam == GEMM:
            envelope.setdefault("H", args.get("K")); envelope.setdefault("I2", args.get("N"))
        elif fam == GEMMW:
            envelope.setdefault("I", args.get("K"))
        moe.append((fam, args, {k: r.get(k) for k in KEEP}))
    topk = envelope.get("TOPK")
    for fam, args, r in moe:
        if fam == ROUTER:
            routers[int(r["node"])] = r
        elif fam in (GEMM, GEMMW):
            refs = _refs(r)
            if len(refs) < 2 or not refs[1] or refs[1][0][0] != "n":
                anomalies.append({"kind": "pair-without-ids-ref", "node": r["node"], "spec": r["spec"]})
                continue
            _, rnode, lo, hi = refs[1][0]
            if topk is None or hi - lo != 1 or lo < topk:
                anomalies.append({"kind": "ids-ref-not-a-single-id-element", "node": r["node"], "ref": refs[1][0]})
                continue
            k = lo - topk
            slot = pairs[int(rnode)].setdefault(k, {})
            key = "gemm" if fam == GEMM else "gemmw"
            if key in slot:
                anomalies.append({"kind": f"duplicate-{key}-row", "router": rnode, "k": k, "node": r["node"]})
            slot[key] = r
            if fam == GEMMW:
                w_out_node_to_router_k[int(r["node"])] = (int(rnode), k)
                if len(refs) >= 3 and refs[2] and refs[2][0][0] == "n":
                    _, anode, alo, ahi = refs[2][0]
                    if int(anode) != int(rnode) or alo != k or ahi != k + 1:
                        anomalies.append({"kind": "alpha-ref-not-the-pairs-k", "router": rnode, "k": k, "ref": refs[2][0]})
    for fam, args, r in moe:
        if fam != SUM:
            continue
        refs = _refs(r)
        ins = [tuple(x) for x in (refs[0] if refs else [])]
        rks = [w_out_node_to_router_k.get(int(x[1])) for x in ins]
        if not rks or any(v is None for v in rks):
            anomalies.append({"kind": "moesum-operand-not-a-w-row", "node": r["node"]})
            continue
        rnodes = {v[0] for v in rks}
        if len(rnodes) != 1:
            anomalies.append({"kind": "moesum-mixes-routers", "node": r["node"], "routers": sorted(rnodes)})
            continue
        rnode = rnodes.pop()
        ks = [v[1] for v in rks]
        if ks != list(range(len(ks))):
            anomalies.append({"kind": "moesum-operands-not-in-k-order", "router": rnode, "ks": ks})
        sums[rnode] = r
    del moe
    E, TOPK, H, I = envelope.get("E"), envelope.get("TOPK"), envelope.get("H"), envelope.get("I")
    if not routers or E is None or TOPK is None:
        raise SystemExit(f"no MoE router instances in {inst_path}")

    # snapshot values (ids / alpha) where the record has them
    store = None
    try:
        from verity_capture.compare import SnapshotStore
        from verity_capture.events import ViewDesc
        store = SnapshotStore(os.path.join(row_dir, "match", "capture"))
    except Exception as ex:  # noqa: BLE001
        anomalies.append({"kind": "no-snapshot-store", "error": f"{type(ex).__name__}: {ex}"[:200]})

    def router_values(r: dict):
        if store is None:
            return None, None
        ov = r.get("output_views") or []
        if len(ov) < 2:
            return None, None
        try:
            va = ViewDesc(**{k: v for k, v in ov[0].items() if k in ("alloc", "offset", "shape", "strides", "dtype", "device")})
            vi = ViewDesc(**{k: v for k, v in ov[1].items() if k in ("alloc", "offset", "shape", "strides", "dtype", "device")})
            sa, si = store.find(va, at=int(r["event"])), store.find(vi, at=int(r["event"]))
            if sa is None or si is None:
                return None, None
            alpha = store.extract(sa, va).astype(np.uint32).view(np.float32).tolist()
            ids = store.extract(si, vi).astype(np.int64).tolist()
            return ids, alpha
        except Exception as ex:  # noqa: BLE001
            anomalies.append({"kind": "snapshot-read-failed", "router": r["node"], "error": f"{type(ex).__name__}: {ex}"[:160]})
            return None, None

    per_step: dict[int, dict[str, int]] = collections.defaultdict(lambda: collections.Counter())
    n_complete = n_incomplete = n_with_values = n_ids_ok = 0
    ids_bad: list[dict] = []
    ex_out: list[dict] = []
    slot_hist = collections.Counter()
    for rnode, r in sorted(routers.items()):
        step = int(r.get("step", -1))
        slots = pairs.get(rnode, {})
        complete = sorted(slots) == list(range(TOPK)) and all({"gemm", "gemmw"} <= set(v) for v in slots.values()) and rnode in sums
        per_step[step]["token_layers"] += 1
        per_step[step]["pairs_executed"] += sum(1 for v in slots.values() if "gemmw" in v)
        if complete:
            n_complete += 1
        else:
            n_incomplete += 1
            anomalies.append({"kind": "incomplete-token-layer", "router": rnode, "step": step, "ks": sorted(slots), "has_sum": rnode in sums})
        ids, alpha = router_values(r)
        rec: dict[str, Any] = {"step": step, "request": r.get("request"), "position": r.get("position"), "layer": (r.get("name") or "").split("/")[-2] if "/" in (r.get("name") or "") else None,
                               "router_node": rnode, "pairs": {k: {"gemm": v.get("gemm", {}).get("node"), "gemmw": v.get("gemmw", {}).get("node")} for k, v in sorted(slots.items())}}
        if ids is not None:
            n_with_values += 1
            ok = len(ids) == TOPK and all(0 <= e < E for e in ids) and len(set(ids)) == TOPK
            rec["ids"], rec["alpha"] = ids, alpha
            if ok:
                n_ids_ok += 1
                for e in ids:
                    slot_hist[e] += 1
                # the padded assignment of this (token, layer): slot e <- pair k; the rest bot (assigned, never executed)
                rec["enabled_slots"] = {e: {"k": k, "a": 1, "x": "wired (the token's activation)", "g": "alpha[%d]" % k, "z": "gemmw node %s output" % slots.get(k, {}).get("gemmw", {}).get("node")} for k, e in enumerate(ids)}
                rec["disabled_slots"] = sorted(set(range(E)) - set(ids))          # a_e = 0: x and W_e wired in unchanged; g_e, up/gate, silu, z_e = bot
                rec["n_disabled_internal_module_words"] = (E - TOPK) * ((H or 0) + 1)
            else:
                ids_bad.append({"router": rnode, "step": step, "ids": ids, "why": "out-of-range" if any(not 0 <= e < E for e in ids) else "repeated" if len(set(ids)) != TOPK else "count"})
        if len(ex_out) < examples and complete and ids is not None:
            ex_out.append(rec)
    n_tl = len(routers)
    out = {"schema": "moe-pad-transcript/v1", "row_dir": row_dir, "envelope": envelope, "families": dict(fam_counts),
           "token_layers": n_tl, "complete": n_complete, "incomplete": n_incomplete,
           "with_recorded_ids": n_with_values, "ids_admissible": n_ids_ok, "ids_inadmissible": ids_bad[:50], "n_ids_inadmissible": len(ids_bad),
           "population": {"expanded_slot_bodies": n_tl * E, "compact_executed_pairs": sum(c["pairs_executed"] for c in per_step.values()),
                          "disabled_slots": n_tl * (E - TOPK), "disabled_internal_module_words": n_tl * (E - TOPK) * ((H or 0) + 1),
                          "disabled_internal_fine_words": n_tl * (E - TOPK) * (3 * (I or 0) + (H or 0) + 1),
                          "expansion_factor": E / TOPK, "per_step": {str(s): dict(c) for s, c in sorted(per_step.items())}},
           "slot_usage_histogram": {str(e): slot_hist.get(e, 0) for e in range(E)} if slot_hist else None,
           "anomalies": anomalies[:200], "n_anomalies": len(anomalies), "examples": ex_out, "wall_s": round(time.time() - t0, 1)}
    structural_ok = n_incomplete == 0 and not ids_bad and not [a for a in anomalies if a["kind"] not in ("no-snapshot-store", "snapshot-read-failed")]
    # OK only when the ids of EVERY token-layer were read off the record (rev / revb F-r17b-44 (1)); a structurally complete map whose
    # ids were not all recorded is STRUCTURE-ONLY by name -- never OK
    out["verdict"] = ("OK" if n_with_values == n_tl else "STRUCTURE-ONLY") if structural_ok else "ANOMALIES"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("row_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--examples", type=int, default=3)
    a = ap.parse_args(argv)
    out = build(a.row_dir, examples=a.examples)
    path = a.out or os.path.join(a.row_dir, "moe_pad_assignment.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    p = out["population"]
    print(f"[moe-pad] {out['verdict']}: envelope {out['envelope']} token-layers {out['token_layers']} complete {out['complete']} "
          f"incomplete {out['incomplete']}; recorded ids {out['with_recorded_ids']} admissible {out['ids_admissible']} inadmissible {out['n_ids_inadmissible']}; "
          f"population expanded {p['expanded_slot_bodies']:,} slot bodies vs compact {p['compact_executed_pairs']:,} pairs (x{p['expansion_factor']:g}); "
          f"disabled slots' internal module words {p['disabled_internal_module_words']:,}; anomalies {out['n_anomalies']}; wrote {path} ({out['wall_s']} s)")
    return 0 if out["verdict"] == "OK" else 2


if __name__ == "__main__":
    sys.exit(main())
