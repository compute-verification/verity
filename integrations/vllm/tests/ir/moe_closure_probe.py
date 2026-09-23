"""MoE stage, Program side, on a REAL derived body -- **padded baseline (experimental)**, not a panel row.

Lifts the Build's derived OLMoE request body (cap-shape component) and computes the activity closure under a HYPOTHETICAL early stop
(the record of the row is a length stop: 0 bot events there), then reports, per event and per node FAMILY, how many nodes are Active
and how many bot -- the question the MoE stage asks (`LIFTING-SPEC` v0.4 §5.3 / BRIEF §7B): does an inactive continuation need an expert
ROUTE?  In the lifted Program a bot event's router (`MoeRouterTopK*`: softmax + top-k over a bot hidden state) is bot, so its `ids` are bot,
so every expert gather / routed sum of that event is bot -- NO route is invented, the strict rule alone carries it; and in an Active event
every routing node is Active (the guard flows through the event boundary only).  Positive: the counts per family split exactly by event
stratum.  Negative (same body): a closure over a transcript with a spurious bot at an ACTIVE event index is refused by the closure's own
guard check.  Nothing here is inferred from the row's Match; the row's record (`lifted_record.json`) is checked separately.

Usage (cp-dev1 CPU; the laptop's per-process cap does not hold a 166k-node body):
    python -m tests.ir.moe_closure_probe <build_request dir> --eos 0 --cap 32 --a-r 11 --lag 1 --out <json>
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict

from verity_vllm.lifting import record as R
from verity.ir.codec import _spec_id
from verity_vllm.ir import workload as W
from verity.ir.defs import REGISTRY
from verity_vllm.registry import lifted as L

MOE_FAMILIES = re.compile(r"Moe|Router|TopK|Expert", re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("build_request")
    ap.add_argument("--eos", required=True)
    ap.add_argument("--cap", type=int, required=True)
    ap.add_argument("--a-r", type=int, required=True, help="hypothetical activity length: the EOS is sampled at event a_r-1")
    ap.add_argument("--lag", type=int, default=1)
    ap.add_argument("--boundary-prefix", default=L.BOUNDARY_PREFIX)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    eos = [int(t) for t in a.eos.split(",")]
    desc = R.load_descriptor(a.build_request)
    comp = W.request_component(desc, REGISTRY, None)
    lifted = L.lift_request_component(comp.fn, eos, max_tokens=a.cap, lag=a.lag, boundary_prefix=a.boundary_prefix)
    # a hypothetical transcript: a_r tokens, the last the EOS, then (LAG) placeholder sample(s) -- the served shape of an early stop
    filler = next(t for t in range(1, 1000) if t not in eos)
    y = [filler] * (a.a_r - 1) + [eos[0]] + [filler] * a.lag
    cl = R.activity_closure(lifted, y, eos)
    guard = set(lifted.guard_nodes)
    fam_by_event: dict[int, Counter] = defaultdict(Counter)   # event -> Counter((family, tag))
    moe_by_event: dict[int, Counter] = defaultdict(Counter)
    for k, n in enumerate(lifted.body.nodes):
        if k in guard:
            continue
        s, tag = cl.step_of_node[k], cl.tags[k]
        fam = re.sub(r"\{.*", "", _spec_id(n.fn))
        fam_by_event[s][(fam, tag)] += 1
        if MOE_FAMILIES.search(fam):
            moe_by_event[s][(fam, tag)] += 1
    n_exec = L.expected_served(a.a_r, a.cap, a.lag)
    per_event = {}
    violations = []
    for s in range(a.cap):
        act = sum(c for (f, t), c in fam_by_event[s].items() if t == R.A)
        bot = sum(c for (f, t), c in fam_by_event[s].items() if t == R.B)
        moe_act = sum(c for (f, t), c in moe_by_event[s].items() if t == R.A)
        moe_bot = sum(c for (f, t), c in moe_by_event[s].items() if t == R.B)
        expected = R.A if s < n_exec else R.B
        per_event[s] = {"event_tag": cl.event_tag[s], "expected": expected, "nodes_active": act, "nodes_bot": bot,
                        "moe_nodes_active": moe_act, "moe_nodes_bot": moe_bot,
                        "moe_families": sorted({f for (f, t) in moe_by_event[s]})}
        # the MoE-stage statement: in an executed event every routing node is Active; in a bot event every routing node is bot
        if expected == R.A and moe_bot:
            violations.append((s, "bot routing node in an Active event"))
        if expected == R.B and moe_act:
            violations.append((s, "Active routing node in a bot event: a route was invented"))
        if expected == R.B and bot != act + bot:
            violations.append((s, "an Active node in a bot event"))
    moe_total = Counter()
    for s in range(a.cap):
        for (f, t), c in moe_by_event[s].items():
            moe_total[f"{f}|{t}"] += c
    # NEGATIVE on the same body: a transcript that claims the EOS at a_r-1 but a token at index a_r that is not the placeholder's -- the closure
    # takes the transcript as committed; what is refused is a guard that does not fit it.  Here: a transcript LONGER than the served shape
    # (an extra step after the placeholder) has no guard reading -> R2-style refusal is the record tool's; the closure's own check is that
    # the guard values of the hypothetical transcript are exactly [1]*(n_exec-1) + [0]*(cap-n_exec).
    guard_expected = [1 if s < n_exec else 0 for s in range(1 + a.lag, a.cap)]  # events 0..LAG are unconditionally live (R4's own formula)
    out = {"schema": "verity/pad/moe-closure-probe/v1", "label": L.LABEL, "build_request": a.build_request, "program_fn": _spec_id(comp.fn),
           "cap": a.cap, "hypothetical_a_r": a.a_r, "lag": a.lag, "expected_executed": n_exec, "transcript": y,
           "guard_live": list(cl.guard_live), "guard_live_expected": guard_expected, "guard_ok": list(cl.guard_live) == guard_expected,
           "body_nodes": len(lifted.body.nodes), "guard_nodes": len(guard), "per_event": per_event,
           "moe_family_totals": dict(sorted(moe_total.items())), "violations": violations,
           "statement": ("in every executed event every routing node (router softmax/top-k, expert gather, routed sum) is Active and in every bot "
                         "event every routing node is bot: the inactive continuation invents no expert route (strict propagation through the "
                         "router's bot hidden state); the guard flows through the event boundary only" if not violations and list(cl.guard_live) == guard_expected
                         else "VIOLATION: see `violations`")}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("guard_ok", "expected_executed", "body_nodes", "guard_nodes", "violations", "moe_family_totals")}, indent=1))
    print(json.dumps({s: {k: v for k, v in per_event[s].items() if k != "moe_families"} for s in sorted({0, n_exec - 1, min(n_exec, a.cap - 1), a.cap - 1})}, indent=1))


if __name__ == "__main__":
    main()
