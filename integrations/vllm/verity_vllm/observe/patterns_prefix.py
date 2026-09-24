"""verity_vllm/observe/patterns_prefix.py — the Program-level reading of a prefix-linkage record (coverage row M1).  Torch-free.

This module does NOT construct Program parts (fold.py / patterns.py are frozen for the window).  It states, from
`prefix_cache.extract_linkage`'s record, what the Program must contain for a run with prefix caching to be *closed*, and
computes the concrete obligations the checker has to discharge:

  * `StateEdge`  — one per reused KV slot: the consumer request's attention input at `position` (every layer) IS the k/v
                   output of the producer row (`producer_request`, `producer_step`, `producer_row`).  In the fold this edge
                   already exists implicitly: attention reads `kv_slots(block_table)` and the memory model resolves each slot
                   to its last writer (the `reshape_and_cache_flash` Transparent move of the producing step), which G5's
                   independent last-writer map reproduces.  What is missing is the *request-level* statement of it.
  * `AliasObligation` — one per reused position: the consumer's DECLARED prompt token at `position` (spec_tokens.json /
                   the workload) must equal the producer row's token.  The declared leaf has no gate of its own in the
                   Program (no `Root("prompt_token", prompt[B][p])` is ever emitted for a cached position), so without this
                   obligation the interface is over-declared: mutating the declared leaf changes nothing.  Token identity is
                   necessary for the request-level claim `tokens_B = M(prompt_B)`; it is not the linkage.
  * `ExternalState` — a reused slot whose writer is not in the log (warm server, previous run).  No same-run edge exists; the
                   Program needs an explicit `cached_kv` Input whose value is committed to the producing run's k/v outputs
                   (block-content digests keyed by the block's token hash) — never a silent prescribed input.

`closure(linkage)` folds these into one verdict per warm request and for the run:
    closed            every reused slot has a same-run StateEdge and every AliasObligation holds
    alias-violated    linked, but a declared token differs from the producer's (the consumer's claim is about another prompt)
    external-state    at least one reused slot has no writer in the log
    cold              no reuse (the control run)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

SCHEMA = "verity-capture/prefix-obligations/v1"


@dataclass(frozen=True)
class StateEdge:
    consumer_request: str
    consumer_first_step: int
    position: int
    slot: int
    physical_block: int
    producer_request: str
    producer_step: int
    producer_row: int
    producer_instances: tuple[Any, ...] = ()      # instance ids at (producer_step, producer_row) when instances.jsonl was given


@dataclass(frozen=True)
class AliasObligation:
    consumer_request: str
    position: int
    declared_token: int | None                    # None: the declaration was not available to the extractor
    producer_request: str
    producer_token: int | None
    holds: bool | None                            # None: undecidable without the declaration


@dataclass(frozen=True)
class ExternalState:
    consumer_request: str
    position: int
    slot: int
    physical_block: int | None


def obligations(linkage: dict) -> dict:
    """The obligations of one linkage record: {"schema", "requests": {rid: {...}}, "run": {...}}."""
    out: dict[str, Any] = {"schema": SCHEMA, "requests": {}, "run": {}}
    n_edges = n_alias = n_alias_fail = n_alias_open = n_external = 0
    statuses: list[str] = []
    for rid, r in (linkage.get("requests") or {}).items():
        edges: list[StateEdge] = []
        aliases: list[AliasObligation] = []
        external: list[ExternalState] = []
        if not r.get("warm"):
            out["requests"][rid] = {"status": "cold", "state_edges": [], "alias_obligations": [], "external_state": [],
                                    "declared_leaves_without_gate": 0, "prompt_rows_with_gates": r.get("prompt_rows_at_first_step")}
            statuses.append("cold")
            continue
        for blk in r.get("reused_blocks") or []:
            pi = blk.get("producer_instances") or {}
            inst = tuple(pi.get("nodes") or []) if isinstance(pi, dict) else tuple(g.get("id") for g in pi if g.get("id") is not None)
            for s in blk.get("slots") or []:
                w = s.get("writer")
                if w is None:
                    external.append(ExternalState(rid, int(s["position"]), int(s["slot"]), blk.get("physical_block")))
                    continue
                edges.append(StateEdge(rid, int(r["first_step"]), int(s["position"]), int(s["slot"]), int(blk["physical_block"]),
                                       str(w["request_id"]), int(w["step"]), int(w["row"]),
                                       tuple(s["producer_instances"]) if s.get("producer_instances") else inst))
                ti = s.get("token_identical")
                holds = None if ti is None else bool(ti and s.get("position_identical", True))
                aliases.append(AliasObligation(rid, int(s["position"]), None if ti is None else (w.get("token") if ti else None),
                                               str(w["request_id"]), w.get("token"), holds))
        n_fail = sum(1 for a in aliases if a.holds is False)
        n_open = sum(1 for a in aliases if a.holds is None)
        status = ("external-state" if external else "alias-violated" if n_fail else "closed")
        ncomp = int(r.get("num_computed_at_first_step") or 0)
        out["requests"][rid] = {
            "status": status, "num_computed_at_first_step": ncomp,
            "declared_leaves_without_gate": ncomp,             # positions < num_computed never get a prompt Root in the log
            "prompt_rows_with_gates": r.get("prompt_rows_at_first_step"),
            "producers": r.get("producers") or [],
            "state_edges": [{**asdict(e), "producer_instances": list(e.producer_instances)} for e in edges],
            "alias_obligations": [asdict(a) for a in aliases],
            "external_state": [asdict(x) for x in external],
            "counts": {"state_edges": len(edges), "alias_obligations": len(aliases), "alias_failed": n_fail, "alias_undecided": n_open,
                       "external_state": len(external)}}
        statuses.append(status)
        n_edges += len(edges)
        n_alias += len(aliases)
        n_alias_fail += n_fail
        n_alias_open += n_open
        n_external += len(external)
    warm = [s for s in statuses if s != "cold"]
    run_status = ("cold" if not warm else "external-state" if "external-state" in warm else
                  "alias-violated" if "alias-violated" in warm else "closed")
    out["run"] = {"status": run_status, "warm_requests": len(warm), "state_edges": n_edges, "alias_obligations": n_alias,
                  "alias_failed": n_alias_fail, "alias_undecided": n_alias_open, "external_state": n_external}
    return out


def closure(linkage: dict) -> str:
    return obligations(linkage)["run"]["status"]


def interface_of(linkage: dict, rid: str) -> dict:
    """The corrected request-level interface of one request: which declared prompt positions are its own gates, which are
    aliases of another request's leaves (through the state edges), which are external state."""
    r = (linkage.get("requests") or {}).get(rid)
    if r is None:
        raise KeyError(rid)
    ob = obligations(linkage)["requests"][rid]
    alias = {e["position"]: (e["producer_request"], e["producer_step"]) for e in ob["state_edges"]}
    ext = sorted(x["position"] for x in ob["external_state"])
    ncomp = int(r.get("num_computed_at_first_step") or 0)
    declared = r.get("declared_prompt_len")
    own = list(range(ncomp, int(declared))) if declared is not None else None
    return {"request_id": rid, "own_prompt_positions": own, "aliased_positions": {str(p): list(v) for p, v in sorted(alias.items())},
            "external_positions": ext, "status": ob["status"],
            "claim": (f"tokens[{rid}] = M(prompt[{rid}]) holds iff, for every aliased position p, declared prompt[{rid}][p] == "
                      f"producer's prompt[p] (alias obligation) AND the attention inputs at p are the producer rows' k/v (state edge); "
                      f"external positions have no same-run producer: a committed `cached_kv` Input is required")}


def blocker_text(linkage: dict) -> str:
    """The precise remaining blocker for the checker, from the record (for status files / the mechanism row)."""
    ob = obligations(linkage)
    run = ob["run"]
    if run["status"] == "cold":
        return "no prefix reuse observed (cold run): nothing to close."
    lines = [f"prefix closure: {run['status']} — {run['warm_requests']} warm request(s), {run['state_edges']} same-run state edges, "
             f"{run['alias_obligations']} alias obligations ({run['alias_failed']} failed, {run['alias_undecided']} undecided), "
             f"{run['external_state']} external-state slots."]
    for rid, r in ob["requests"].items():
        if r["status"] == "cold":
            continue
        prods = ", ".join(f"{p['request_id'][:8]}@step{p['step']}" for p in r["producers"]) or "—"
        lines.append(f"  {rid[:8]}: {r['declared_leaves_without_gate']} declared prompt leaves without a gate (positions < "
                     f"{r['num_computed_at_first_step']}), {r['prompt_rows_with_gates']} with gates; producers {prods}; "
                     f"edges {r['counts']['state_edges']}, alias failed {r['counts']['alias_failed']}, external {r['counts']['external_state']}")
    lines.append("  Program needs: (a) a request-level state edge `attention_inputs[consumer][p] := kv_out[producer_row]` for every reused "
                 "position (the fold has it as a memory edge; the interface does not say it), and (b) the alias obligation "
                 "`declared prompt[consumer][p] == prompt[producer][p]` on the DECLARED leaves — or the consumer's interface declared as "
                 "(cached_kv from producer rows, own tokens).  No silent prescribed KV input; an external-state slot requires an explicit "
                 "`cached_kv` Input committed to the producing run's outputs.")
    return "\n".join(lines)


__all__ = ["SCHEMA", "StateEdge", "AliasObligation", "ExternalState", "obligations", "closure", "interface_of", "blocker_text"]
