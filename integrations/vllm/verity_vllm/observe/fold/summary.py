"""A fold's JSON-able summary for reports (`fold.summarize`)."""

from __future__ import annotations

from verity_vllm.observe.fold.resolver import spec_label

#: the fail-closed reference-integrity rule id (definition and evaluation over persisted summaries:
#: `verity_vllm.check.gates.REF_INTEGRITY_RULE` / `gates.fold_integrity`); quoted in `summarize()` so every fold summary names it
REFERENCE_INTEGRITY_RULE = "R1/2026-09-18"


def summarize(result) -> dict:
    """A JSON-able summary of a fold (`fold.FoldResult`) for reports."""
    return {"digest": result.digest, "params": [[n, repr(t)] for n, t in result.params], "ret_type": repr(result.body.ret_type) if result.body else None,
            "nodes": len(result.body.nodes) if result.body else 0, "instances": len(result.instances),
            "report": result.report.to_dict(), "memory": result.memory, "steps": result.steps, "errors": len(result.errors),
            "error_classes": _hist(e["class"] for e in result.errors), "error_examples": result.errors[:16],
            "notes": result.notes[:64], "timings": result.timings, "sampled": result.sampled, "discarded": result.discarded,
            # recompute wiring evidence + the fail-closed eligibility rule (see `gates.fold_integrity`)
            "refed": len(result.refed), "refed_examples": result.refed[:16],
            "reference_integrity": {"rule": REFERENCE_INTEGRITY_RULE, "eligible": result.reference_eligible,
                                    "errors": len(result.errors), "unsupported": int(getattr(result.report, "unsupported", 0) or 0)},
            "spec": spec_label(result.spec) if result.spec is not None else None,
            # the tensor-parallel facts of a rank fold (None / empty on a TP1 log): rank/world, the collectives executed (one
            # record each, own view as repr) and the `peers_<i>` root-parameter layout -- the collective_received roots the fold synthesised
            "tp": result.tp,
            "collectives": [{**{k: v for k, v in c.items() if k != "own_view"}, "own_view": repr(c.get("own_view"))} for c in result.collectives],
            "collectives_by_kind": _hist(c["kind"] for c in result.collectives),
            "peers_layout": result.peers_layout, "witness_layout": result.witness_layout, "supplied": result.supplied}


def _hist(items) -> dict[str, int]:
    h: dict[str, int] = {}
    for i in items:
        h[i] = h.get(i, 0) + 1
    return dict(sorted(h.items()))
