"""[R15 stoch] B>1 stochastic record: per engine step, the REALIZED live sampler batch vs the DECLARED live count (arrivals + caps),
and the mask kernel's constexpr S vs splits_for(realized live, num_SMs).  Evidence for RULING M-0571 (3): S(s) = SplitsFor_v1(|live(s)|, num_SMs)."""
import json, sys, collections
from verity_capture.log import read_log
from verity_capture.events import StepStart, TritonLaunch
from verity_capture.experimental.cb_a.derive_step import declared_live_counts
from verity_vllm.registry.topp_split import splits_for
log, wl, out, nsm = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 142
ev = list(read_log(log)); w = json.load(open(wl))
cap = {r["request_id"]: int(r["max_tokens"]) for r in w["requests"]}
arrive = {r["request_id"]: int(r.get("arrive_step", 0) or 0) for r in w["requests"]}
n_live = declared_live_counts(w["requests"])
served = collections.Counter(); S_at = {}; live_at = {}; rows_at = {}
for e in ev[1:]:
    if isinstance(e, StepStart):
        live_at[e.step] = len(e.logits_rows); rows_at[e.step] = [rid for rid, _ in e.logits_rows]
        for rid, _ in e.logits_rows: served[rid] += 1
    elif isinstance(e, TritonLaunch) and e.phase == "sample" and e.fn.endswith("_topp_sb_mask_kernel"):
        S_at[e.step] = (e.constexprs or {}).get("S")
steps = []
for s in sorted(live_at):
    d = n_live[s] if s < len(n_live) else None
    steps.append({"step": s, "live_realized": live_at[s], "live_declared": d, "S_kernel": S_at.get(s), "S_splits_for_realized": splits_for(live_at[s], nsm),
                  "S_splits_for_declared": None if d is None else splits_for(d, nsm), "requests": [r[:8] for r in rows_at[s]]})
bad = [x for x in steps if x["S_kernel"] != x["S_splits_for_realized"] or x["live_realized"] != x["live_declared"]]
flips = [(x["step"], x["live_realized"], x["S_kernel"]) for i, x in enumerate(steps) if i == 0 or x["S_kernel"] != steps[i-1]["S_kernel"]]
rep = {"schema": "verity-r15/stoch-s-schedule-check/v1", "log": log, "workload": wl, "num_sms": nsm, "engine_steps": len(steps),
       "sampled_rows": sum(x["live_realized"] for x in steps),
       "served_vs_cap": {r[:8]: {"served": served[r], "cap": cap[r], "arrive": arrive[r]} for r in cap},
       "every_request_served_to_cap": all(served[r] == cap[r] for r in cap),
       "steps_with_disagreement": bad, "n_disagree": len(bad), "S_flips_(step,live,S)": flips,
       "verdict": "S(s) == SplitsFor(|live(s)|, num_SMs) at every step and realized live == declared live" if not bad else "DISAGREEMENT (see steps_with_disagreement)",
       "steps": steps}
json.dump(rep, open(out, "w"), indent=1)
print(rep["verdict"], "| steps", len(steps), "rows", rep["sampled_rows"], "to-cap", rep["every_request_served_to_cap"], "flips", flips)
