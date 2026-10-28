"""The global (B >= 2) composition of request manifests and the merge of a TP row's rank manifests -- v1's reading (TABLE-RULES global),
over the components `required.request_manifest` builds.  A composed header names every part's partition, never one part's for all."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Mapping

from verity_vllm.query.manifest import format as fmt
from verity_vllm.query.required import _as_list, collective_kind

__all__ = ["compose_global", "merge_ranks"]



def compose_global(fixture: Mapping[str, Any], parts: Mapping[str, dict[str, Any]], reqs: list[dict[str, Any]], *, workload_digest: str | None,
                   rank: int = 0, ranks: int = 1) -> dict[str, Any]:
    """`parts` = {request label: component manifest}; `reqs` = the declared requests in (arrive_step, index) order with `request_id` = label."""
    component_digests = sorted({p["program_digest"] for p in parts.values()})
    pdig = str(workload_digest) if workload_digest else fmt.components_fold_digest(component_digests)
    pdig_kind = "workload" if workload_digest else "components-fold"
    by_engine: dict[int, list[tuple[str, int]]] = {}
    for r in reqs:
        for s in range(int(r["max_tokens"])):
            by_engine.setdefault(int(r.get("arrive_step", 0)) + s, []).append((r["request_id"], s))
    hidden_family = parts[reqs[0]["request_id"]]["query"]["attention_hidden_family"]
    ids: list[dict[str, Any]] = []
    engine_of = {(rid, s): e for e, members in by_engine.items() for rid, s in members}
    unreturned = [dict(row, program_digest=pdig, component_digest=row["program_digest"], engine_step=engine_of.get((rid, row["step"]), row.get("engine_step")))
                  for rid, p in parts.items() for row in p.get("unreturned") or ()]
    seen_w: set = set()
    for rid, p in parts.items():
        for row in p["identities"]:
            if row["family"] == "weights":
                if (row["rank"], row["op_path"]) in seen_w:
                    continue
                seen_w.add((row["rank"], row["op_path"]))
                ids.append(dict(row, program_digest=pdig, component_digest=row["program_digest"]))
    engine_steps_out: list[dict[str, Any]] = []
    for e in sorted(by_engine):
        members = by_engine[e]
        rows_of: dict[str, int] = {}
        for rid, s in members:
            emb = next((x for x in parts[rid]["identities"] if x["family"] == "input_token_ids" and x["step"] == s), None)
            rows_of[rid] = int(emb["rows"]) if emb else 0
        attn_union: dict[str, dict[str, Any]] = {}
        for rid, s in members:
            for row in parts[rid]["identities"]:
                if row["step"] != s or row["family"] == "weights":
                    continue
                g = dict(row, program_digest=pdig, component_digest=row["program_digest"], engine_step=e)
                if row["family"] == hidden_family:
                    a = attn_union.setdefault(row["op_path"], {"requests": [], "n_seq": 0, "max_rows": 0, "T_max": 0, "module": row.get("attention_module")})
                    a["requests"].append(rid); a["n_seq"] += 1
                    a["max_rows"] = max(a["max_rows"], int(row["geometry"]["rows"])); a["T_max"] = max(a["T_max"], int(row["geometry"]["seqlen_k"]))
                else:
                    ids.append(g)
        for op, a in sorted(attn_union.items()):
            comp = parts[a["requests"][0]]
            pre = next(x for x in comp["identities"] if x["family"] == hidden_family and x["op_path"] == op and x["step"] == 0)
            dec = next((x for x in comp["identities"] if x["family"] == hidden_family and x["op_path"] == op and x["step"] == 1), None)
            D = int(pre["geometry"]["D"])
            if hidden_family[:3] == "fa3":       # FA3 packs a KV group's query heads into its rows: HB = KVH, M = rows * NH / KVH
                KVH = int(pre["geometry"]["HB"])
                NH = KVH * (int(pre["geometry"]["M"]) // max(1, int(pre["geometry"]["rows"])))
            else:                                # FA2: a prefill's slabs are its query heads, a swapped decode's its KV heads
                NH = int(pre["geometry"]["HB"])
                KVH = int(dec["geometry"]["HB"]) if dec is not None and int(dec["geometry"]["M"]) > 1 else NH
            fn = fmt.fa3_geometry if hidden_family[:3] == "fa3" else fmt.fa2_geometry
            HB, M, NB, BN, gname = fn(a["T_max"], a["max_rows"] * a["n_seq"], NH, KVH, D, a["n_seq"])
            ids.append({"family": hidden_family, "program_digest": pdig, "request_id": "*", "requests": a["requests"], "step": "*", "engine_step": e, "invocation": 0,
                        "op_path": op, "output_member": gname, "element_range": [0, fmt.mat_total_words(HB, M, NB, BN, D)], "rank": rank, "dtype": "u32",
                        "geometry": {"HB": HB, "M": M, "NB": NB, "BN": BN, "D": D, "seqlen_k": a["T_max"], "rows": a["max_rows"] * a["n_seq"], "n_seq": a["n_seq"]},
                        "attention_module": a["module"]})
        engine_steps_out.append({"engine_step": e, "members": [{"request_id": rid, "step": s, "rows": rows_of[rid]} for rid, s in members],
                                 "rows": sum(rows_of.values()), "n_seq": len(members)})
    ids.sort(key=fmt.row_key)
    fam: Counter = Counter(r["family"] for r in ids)
    ref: dict[str, Any] = {}
    for r in ids:
        if r["family"] in ("input_token_ids", "sampled_token_ids") and r["family"] not in ref:
            ref[r["family"]] = {"op_path": r["op_path"], "output_member": r["output_member"], "per_row": 1}
    q0 = parts[reqs[0]["request_id"]]["query"]
    return {"schema": fmt.SCHEMA, "identity_key": list(fmt.IDENTITY_KEY), "program_digest": pdig, "program_digest_kind": pdig_kind, "component_digests": component_digests,
            "global": True, "addressing": "request-local", "composition_reference": ref,
            "fixture": {k: v for k, v in fixture.items() if k in ("schema", "fixture", "model", "revision", "sampling", "expected_engine_steps", "case")}
                       | ({"row_id": fixture["sweep"].get("row_id"), "seed": fixture["sweep"].get("seed")} if isinstance(fixture.get("sweep"), dict) else {}),
            "requests_declared": [{k: r.get(k) for k in ("request_id", "workload_request_id", "index", "prompt_len", "max_tokens", "arrive_step")} for r in reqs],
            "components": {rid: {"program_digest": p["program_digest"], "steps": len(p["steps"]), "LP": p["program"]["LP"], "T": p["program"]["T"],
                                 "workload_request_id": next(r.get("workload_request_id") for r in reqs if r["request_id"] == rid),
                                 "correspondence": p.get("correspondence")} for rid, p in parts.items()},
            # the source of the owning modules across the components: one source (the record on every component, or the fallback on every one)
            "correspondence": {"source": sorted({(p.get("correspondence") or {}).get("source", "v1_annotations") for p in parts.values()}),
                               "digests": sorted({d for p in parts.values() if (d := (p.get("correspondence") or {}).get("digest"))})},
            "complete": all(p.get("complete", True) for p in parts.values()),
            "unmodelled": {k: v for p in parts.values() for k, v in (p.get("unmodelled") or {}).items()},
            "query": _with_partitions(dict(q0), "by_component", {rid: p["query"] for rid, p in parts.items()}),
            "op_path_aliases": parts[reqs[0]["request_id"]]["op_path_aliases"],
            "ranks": ranks, "rank": rank,
            "engine_steps": engine_steps_out, "steps": sorted(by_engine), "requests": [r["request_id"] for r in reqs],
            "populations": {"identities": len(ids), "by_family": dict(fam), "engine_steps": len(by_engine), "requests": len(reqs)},
            "manifest_digest": fmt.manifest_digest(ids), "identities": ids, **fmt.unreturned_block(unreturned)}


def _with_partitions(q: dict[str, Any], key: str, queries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """A composed header names every part's partition and word check (`key`: `by_component` / `by_rank`), never one part's for all."""
    for field in ("partition", "word_check"):
        q.pop(field, None)
        parts = {k: h[field] for k, h in queries.items() if h.get(field)}
        if parts:
            q[field] = {key: dict(sorted(parts.items()))}
    return q


def merge_ranks(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """The manifest of a TP workload from its W rank manifests (v1's rank rule): ranks 0..W-1 each exactly once under one world, one
    Program digest per rank, the same engine-step composition on every rank.  Every row keeps its rank's digest (`program_digest_per_rank`);
    the header digest is the labelled fold of the rank digests (`tp-ranks-fold`).  `tp_peer_binding`: every collective output and rank-local
    partial of rank r has each peer's partial at the same (request, step, invocation, module) -- an unbound one makes the merge incomplete."""
    worlds = sorted({int(p.get("ranks") or 1) for p in parts})
    world = worlds[-1]
    ranks = [int(p.get("rank", 0)) for p in parts]
    if len(worlds) != 1 or world < 2:
        raise ValueError(f"merge_ranks: the parts declare worlds {worlds} (one world W >= 2 on every rank part)")
    if sorted(ranks) != list(range(world)):
        raise ValueError(f"merge_ranks: parts cover ranks {ranks}, world {world} needs every rank 0..{world - 1} exactly once")
    per_rank = {str(r): p["program_digest"] for r, p in zip(ranks, parts)}
    if len(set(per_rank.values())) != world:
        raise ValueError(f"merge_ranks: two ranks carry one Program digest ({per_rank}); a rank Program is specific to its rank")
    if len({json.dumps(p.get("engine_steps") if p.get("global") else p.get("steps"), sort_keys=True) for p in parts}) != 1:
        raise ValueError("merge_ranks: the rank parts declare different step compositions (not the same workload on every rank)")
    aliases: dict[str, list] = {}
    for p in parts:
        for k, v in (p.get("op_path_aliases") or {}).items():
            want = [str(v[0]), int(v[1])]
            if aliases.setdefault(k, want) != want:
                raise ValueError(f"merge_ranks: the parts disagree on alias {k!r} ({aliases[k]} vs {want})")
    comp = sorted(per_rank.values())
    ids = sorted((r for p in parts for r in p["identities"]), key=fmt.row_key)
    binding = fmt.tp_peer_binding(parts, collective_kind)
    unmodelled = {k: v for p in parts for k, v in (p.get("unmodelled") or {}).items()}
    if binding["unbound"]:
        unmodelled[f"tp peer binding: {binding['n_unbound']} collective output(s) / partial(s) without the peer identity "
                   f"(first: {binding['unbound'][0]})"] = binding["n_unbound"]
    p0 = parts[ranks.index(0)]
    return {"schema": fmt.SCHEMA, "identity_key": list(fmt.IDENTITY_KEY), "program_digest": fmt.tp_ranks_fold_digest(comp),
            "program_digest_kind": "tp-ranks-fold", "component_digests": comp, "program_digest_per_rank": per_rank,
            "parts": [{"program_digest": p["program_digest"], "request_id": p["requests"], "steps": p["steps"], "rank": r} for r, p in zip(ranks, parts)],
            **{k: p0[k] for k in ("global", "addressing", "composition_reference", "fixture", "requests_declared", "components", "engine_steps") if k in p0},
            "correspondence": {"source": sorted({s for p in parts for s in _as_list((p.get("correspondence") or {}).get("source", "v1_annotations"))}),
                               "digests": sorted({d for p in parts for d in _as_list((p.get("correspondence") or {}).get("digest") or
                                                                                        (p.get("correspondence") or {}).get("digests") or [])})},
            "query": _with_partitions(dict(p0["query"]), "by_rank",
                                      {str(r): p["query"] for r, p in zip(ranks, parts)}),
            "op_path_aliases": aliases, "ranks": world,
            "steps": sorted({s for p in parts for s in p["steps"]}), "requests": list(dict.fromkeys(rq for p in parts for rq in p["requests"])),
            "complete": all(p.get("complete", True) for p in parts) and not binding["unbound"], "unmodelled": unmodelled,
            "populations": {"identities": len(ids), "by_family": dict(Counter(r["family"] for r in ids))},
            "manifest_digest": fmt.manifest_digest(ids), "identities": ids, "tp_peer_binding": binding,
            **fmt.unreturned_block([r for p in parts for r in p.get("unreturned") or ()])}
