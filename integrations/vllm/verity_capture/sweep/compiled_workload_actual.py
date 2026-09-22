"""[R16 compiled, brief §11 / report M-0171] WHAT THE COMPILED COMMIT RUN ACTUALLY DID, derived from the run's own binding map -- not from the
workload declaration: per request the number of engine steps in which one of its P_compiled identities was BOUND (= tokens generated: one
decode step per token, the prefill step included), the engine step it first appears in (observed arrival), the prompt length (the
`ids` identity's element range at its step 0), and the achieved concurrency = the maximum number of distinct requests bound in one engine
step (observed chronology, the binding map's per-step roots), cross-checked against the workload's caps / arrivals and the run's
generated_tokens total (runs.jsonl).  Output: workload_actual.json beside the arm (schema verity/compiled-workload-actual/v1).

  python -m verity_capture.sweep.compiled_workload_actual --arm <commit_treatment_r1 dir> --workload <workload.json> [--out <file>]
  python -m verity_capture.sweep.compiled_workload_actual --from-actual <workload_actual.json> --workload <workload.json> [--runs <runs.jsonl>] [--out <file>]
      (re-derives the arrivals-rule section from an existing record when the binding map is no longer beside it)

ARRIVALS RULE (R17 F-r17-17, rev M-0765).  The declaration's `arrive_step` is a LOOP ITERATION (a `step()` call of
verity_capture.arrivals.run_requests_arrivals); the binding map observes the FORWARD index of the request's first bound identity.  They
differ by the iterations that ran no forward (a drain iteration returning a finish under the engine's output lag) and by the driver's
declared idle rule (a request is admitted at once when every earlier request has finished).  `arrivals_rule` replays that loop from the
declaration + the observed per-request forward counts (one forward per generated token, prefill included) under the structural output lag
(vLLM V1 = 1), names per request the rule that admitted it (`declared` at its iteration | `idle` earlier because the engine was empty)
and the predicted forward index, and compares with the observed one: a request whose observed arrival is neither is a DEVIATION.  The
replayed iteration count is tied to runs.jsonl `iterations`; the replayed forward count to the binding map's engine steps.  `consistent`
folds the rule (0 deviations, iteration count tied); `cross_check.arrive_equal_declared` stays the raw (units-differing) comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone

REQ_RE = re.compile(r"^r(\d+)$")


def _sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


ARRIVALS_RULE_TEXT = ("declared: request r is admitted right before loop iteration arrive_step[r]; idle: admitted earlier, at once, because every "
                      "earlier request had finished and the engine was empty (verity_capture.arrivals.run_requests_arrivals). Forward index = "
                      "iterations before admission that ran a forward; an iteration that only returns a finish (output lag) runs none.")


def replay_arrivals(decl_arr: list[int], n_fwd: list[int], lag: int) -> dict:
    """Replay run_requests_arrivals' loop: declared arrival iterations + per-request forward counts -> per request the admitting rule,
    the admission iteration and the forward index at admission; the loop's iteration and forward totals.  Pure, torch-free."""
    n = len(decl_arr)
    pending = sorted(range(n), key=lambda i: (int(decl_arr[i]), i))
    it = fwd = 0
    remaining: dict[int, int] = {}
    finish_at: dict[int, int] = {}
    finished_iter: dict[int, int] = {}
    out: dict[int, dict] = {}

    def admit(i: int, rule: str) -> None:
        out[i] = {"rule": rule, "admit_iter": it, "forward_at_admission": fwd, "declared_iter": int(decl_arr[i])}
        if rule == "idle":
            out[i]["earlier_finished_iter"] = max((finished_iter[j] for j in finished_iter), default=None)
        if int(n_fwd[i]) > 0:
            remaining[i] = int(n_fwd[i])
        else:
            finish_at[i] = it + lag

    guard = 0
    while pending or remaining or finish_at:
        guard += 1
        if guard > 10_000_000:
            raise RuntimeError("replay_arrivals: loop did not terminate")
        while pending and int(decl_arr[pending[0]]) <= it:
            admit(pending.pop(0), "declared")
        if not remaining and not finish_at:
            if not pending:
                break
            admit(pending.pop(0), "idle")             # idle gap: the driver brings the next arrival forward without a step()
            continue
        if remaining:                                 # this step() runs a forward
            for i in list(remaining):
                remaining[i] -= 1
                if remaining[i] == 0:
                    del remaining[i]
                    finish_at[i] = it + lag
            fwd += 1
        for i in [i for i, t in finish_at.items() if t <= it]:
            del finish_at[i]
            finished_iter[i] = it
        it += 1
    return {"per_request": out, "iterations": it, "forwards": fwd, "lag": lag}


def arrivals_rule(rids: list[str], observed_arr: list[int], gen: list[int], decl_arr: list[int], engine_steps: int | None,
                  iterations_runs_jsonl: int | None) -> dict:
    """The F-r17-17 statement: per request the declared iteration, the rule that admitted it and the forward index it predicts, against
    the observed forward index; totals tied to runs.jsonl iterations and the observed engine steps.  Lag 1 (vLLM V1 structural) unless
    only lag 0 ties the iteration count."""
    n = min(len(rids), len(observed_arr), len(gen), len(decl_arr))
    if n == 0:
        return {"applicable": False, "why": "no requests"}
    tried = []
    chosen = None
    for lag in (1, 0):
        R = replay_arrivals(list(decl_arr[:n]), list(gen[:n]), lag)
        tie = (R["iterations"] == int(iterations_runs_jsonl)) if iterations_runs_jsonl is not None else None
        tried.append({"lag": lag, "iterations": R["iterations"], "forwards": R["forwards"], "iterations_tie_runs_jsonl": tie})
        if chosen is None and (tie is True or (tie is None and lag == 1)):
            chosen = (lag, R)
    if chosen is None:
        lag, R = 1, replay_arrivals(list(decl_arr[:n]), list(gen[:n]), 1)
        lag_status = "UNTIED: neither lag 1 nor lag 0 reproduces runs.jsonl iterations (lag 1 shown)"
    else:
        lag, R = chosen
        lag_status = ("structural vLLM V1 lag 1, tied to runs.jsonl iterations" if lag == 1 and tried[0]["iterations_tie_runs_jsonl"] else
                      "structural vLLM V1 lag 1 (runs.jsonl iterations absent: untied)" if lag == 1 else "lag 0 ties runs.jsonl iterations (lag 1 does not)")
    per = []
    by_rule = {"declared": 0, "idle": 0, "DEVIATION": 0}
    for k in range(n):
        p = R["per_request"][k]
        pred = p["forward_at_admission"]
        obs = int(observed_arr[k])
        rule = p["rule"] if obs == pred else "DEVIATION"
        by_rule[rule] += 1
        ent = {"request_id": rids[k], "declared_iter": p["declared_iter"], "admit_iter_replayed": p["admit_iter"], "rule": rule,
               "forward_predicted": pred, "forward_observed": obs}
        if p["rule"] == "idle":
            ent["why"] = f"every earlier request had finished (last finish returned at iteration {p.get('earlier_finished_iter')}); admitted at iteration {p['admit_iter']} < declared {p['declared_iter']}"
        if rule == "DEVIATION":
            ent["why"] = f"observed forward {obs} != predicted {pred} under the {p['rule']} rule"
        per.append(ent)
    forwards_tie = (R["forwards"] == int(engine_steps)) if engine_steps is not None else None
    return {"applicable": True, "rule": ARRIVALS_RULE_TEXT, "units": "declaration: loop iterations (step() calls); observation: forward index (binding map step order)",
            "lag": lag, "lag_status": lag_status, "replays_tried": tried,
            "iterations_replayed": R["iterations"], "iterations_runs_jsonl": iterations_runs_jsonl,
            "forwards_replayed": R["forwards"], "engine_steps_observed": engine_steps, "forwards_tie_engine_steps": forwards_tie,
            "no_forward_iterations": R["iterations"] - R["forwards"], "requests": per, "by_rule": by_rule,
            "arrive_equal_declared_under_rule": by_rule["DEVIATION"] == 0,
            "holds": by_rule["DEVIATION"] == 0 and forwards_tie is not False and (tried[0]["iterations_tie_runs_jsonl"] is not False or lag == 0)}


def _runs(arm_or_path: str) -> tuple[dict, str | None]:
    rp = arm_or_path if os.path.isfile(arm_or_path) else os.path.join(arm_or_path, "runs.jsonl")
    runs = []
    if os.path.isfile(rp):
        for line in open(rp):
            try:
                runs.append(json.loads(line))
            except Exception:  # noqa: BLE001
                pass
    run = next((r for r in runs if r.get("arm") in ("instrumented", "treatment")), runs[0] if runs else {})
    return run, (rp if os.path.isfile(rp) else None)


def rederive_from_actual(actual_path: str, workload: str, runs_path: str | None) -> dict:
    """F-r17-17 (3) for a record whose binding map is no longer beside it: the observed facts are the existing record's; only the
    arrivals-rule section is (re)derived and `consistent` re-folded.  The original derivation is kept under `rederived_from`."""
    A = json.load(open(actual_path))
    if A.get("schema") != "verity/compiled-workload-actual/v1":
        raise SystemExit(f"{actual_path} is not a verity/compiled-workload-actual/v1 record")
    want = (A.get("source") or {}).get("workload_sha256")
    if want and _sha(workload) != want:
        raise SystemExit(f"REFUSED: {workload} sha256 {_sha(workload)[:12]} != the record's executed workload {str(want)[:12]} (source.workload_sha256)")
    W = json.load(open(workload))
    reqs = W.get("requests") or []
    decl_arr = [int(r.get("arrive_step") or 0) for r in reqs]
    run, rp = _runs(runs_path or os.path.join(os.path.dirname(os.path.abspath(actual_path)), "runs.jsonl"))
    it = run.get("iterations", run.get("engine_steps"))
    AR = arrivals_rule(A["request_ids"], A["arrive_steps"], A["generated_lengths"], decl_arr, A.get("engine_steps"), int(it) if it is not None else None)
    A["arrivals_rule"] = AR
    A.setdefault("cross_check", {})["arrive_equal_declared_under_rule"] = AR.get("arrive_equal_declared_under_rule")
    A["consistent_before_arrivals_rule"] = A.get("consistent")
    A["consistent"] = bool(A.get("consistent") is not False and AR.get("holds") is not False)
    A["rederived_from"] = {"workload_actual": os.path.basename(actual_path), "workload_actual_sha256": _sha(actual_path), "derived_utc_original": A.get("derived_utc"),
                           "runs_jsonl": rp, "workload_sha256": _sha(workload), "tool_sha256": _sha(os.path.abspath(__file__)), "what": "arrivals_rule + consistent only (F-r17-17 (3)); observed facts unchanged",
                           "derived_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    return A


def derive(arm: str, workload: str) -> dict:
    bm_path = next((p for p in (os.path.join(arm, "binding_map_p0.json"), *sorted(__import__("glob").glob(os.path.join(arm, "binding_map_p*.json")))) if os.path.isfile(p)), None)
    if bm_path is None:
        raise SystemExit(f"compiled_workload_actual: no binding_map_p*.json under {arm}")
    B = json.load(open(bm_path))
    entries = B.get("entries") or []
    W = json.load(open(workload))
    reqs = W.get("requests") or []
    # engine steps in order of first appearance of their step_root (the map is written in commit order: step 0 first)
    order: dict[str, int] = {}
    per_req_steps: dict[str, set] = {}
    per_step_reqs: dict[int, set] = {}
    first_step: dict[str, int] = {}
    prompt_len: dict[str, int] = {}
    ids_member = None
    for e in entries:
        ident = e.get("identity") or {}
        rid = str(ident.get("request_id"))
        if rid == "*" or e.get("kind") == "registered-input":
            continue
        root = e.get("step_root")
        if root is None:
            continue
        if root not in order:
            order[root] = len(order)
        es = order[root]
        per_req_steps.setdefault(rid, set()).add(es)
        per_step_reqs.setdefault(es, set()).add(rid)
        first_step.setdefault(rid, es)
        if ident.get("output_member") == "ids" and rid not in prompt_len:     # the request's FIRST ids identity = its prefill row (the prompt)
            er = ident.get("element_range") or [0, 0]
            prompt_len[rid] = int(er[1]) - int(er[0])
            ids_member = ident.get("op_path")
    rids = sorted(per_req_steps, key=lambda r: (int(REQ_RE.match(r).group(1)) if REQ_RE.match(r) else 1 << 30, r))
    gen = [len(per_req_steps[r]) for r in rids]
    arr = [first_step[r] for r in rids]
    conc = max((len(v) for v in per_step_reqs.values()), default=0)
    engine_steps = len(order)
    # the run's own totals (runs.jsonl, the arm's instrumented run) and the declaration, for the cross-check
    run, rp = _runs(arm)
    rp = rp or os.path.join(arm, "runs.jsonl")
    decl_caps = [int(r.get("max_tokens") or 0) for r in reqs]
    decl_arr = [int(r.get("arrive_step") or 0) for r in reqs]
    decl_lp = [len(r.get("prompt_token_ids") or []) for r in reqs]
    commit_steps = (run.get("commit") or {}).get("steps")
    cross = {"generated_tokens_runs_jsonl": run.get("generated_tokens"), "engine_steps_runs_jsonl": run.get("engine_steps"), "commit_steps_runs_jsonl": commit_steps,
             "sum_generated_lengths": sum(gen), "generated_total_equal": (run.get("generated_tokens") == sum(gen)) if run.get("generated_tokens") is not None else None,
             "steps_equal_commit_steps": (int(commit_steps) == engine_steps) if commit_steps is not None else None,
             "engine_steps_note": "the engine's own step counter (runs.jsonl engine_steps) may exceed the committed steps by the terminal step that binds no identity (no token produced)",
             "declared_output_caps": decl_caps, "generated_equal_caps": gen == decl_caps if len(gen) == len(decl_caps) else None,
             "declared_arrive_steps": decl_arr, "arrive_equal_declared": arr == decl_arr if len(arr) == len(decl_arr) else None,
             "declared_input_lengths": decl_lp, "input_lengths_equal_declared": [prompt_len.get(r) for r in rids] == decl_lp if len(rids) == len(decl_lp) else None}
    it = run.get("iterations", run.get("engine_steps"))
    AR = arrivals_rule(rids, arr, gen, decl_arr, engine_steps, int(it) if it is not None else None)
    cross["arrive_equal_declared_under_rule"] = AR.get("arrive_equal_declared_under_rule")
    return {"schema": "verity/compiled-workload-actual/v1", "derived_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": {"binding_map": os.path.basename(bm_path), "binding_map_sha256": _sha(bm_path), "entries_n": len(entries), "program_digest": B.get("program_digest"),
                       "runs_jsonl": os.path.relpath(rp, arm) if os.path.isfile(rp) else None, "workload": os.path.basename(workload), "workload_sha256": _sha(workload),
                       "tool_sha256": _sha(os.path.abspath(__file__))},
            "request_ids": rids, "generated_lengths": gen, "arrive_steps": arr, "input_lengths": [prompt_len.get(r) for r in rids],
            "output_caps": decl_caps, "achieved_concurrency": conc, "engine_steps": engine_steps,
            "requests_per_step": [len(per_step_reqs.get(s, ())) for s in range(engine_steps)],
            "concurrency_basis": "observed: distinct requests with a BOUND P_compiled identity per engine step (binding map step roots, in commit order)",
            "generated_basis": f"observed: engine steps in which the request has a bound identity (one token per step; prefill step included); prompt length = the element range of the request's first '{ids_member}' ids identity (its prefill row)",
            "cross_check": cross,
            "arrivals_rule": AR,
            "consistent": bool(cross["generated_total_equal"] is not False and cross["steps_equal_commit_steps"] is not False and cross["input_lengths_equal_declared"] is not False
                               and AR.get("holds") is not False)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm")
    ap.add_argument("--from-actual", help="an existing workload_actual.json: re-derive only arrivals_rule + consistent (no binding map needed)")
    ap.add_argument("--runs", help="runs.jsonl for --from-actual (default: beside the record)")
    ap.add_argument("--workload", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    if bool(a.arm) == bool(a.from_actual):
        ap.error("exactly one of --arm / --from-actual")
    R = derive(a.arm, a.workload) if a.arm else rederive_from_actual(a.from_actual, a.workload, a.runs)
    out = a.out or (os.path.join(a.arm, "workload_actual.json") if a.arm else a.from_actual)
    json.dump(R, open(out, "w"), indent=1)
    AR = R.get("arrivals_rule") or {}
    print(f"[workload_actual] {out}: requests {len(R['request_ids'])} generated {R['generated_lengths']} arrive {R['arrive_steps']} "
          f"concurrency {R['achieved_concurrency']} engine_steps {R['engine_steps']} consistent {R['consistent']} "
          f"(generated == caps {R['cross_check']['generated_equal_caps']}, arrivals == declared {R['cross_check']['arrive_equal_declared']} raw / "
          f"{AR.get('arrive_equal_declared_under_rule')} under the rule: by_rule {AR.get('by_rule')}, lag {AR.get('lag')}, iterations replayed "
          f"{AR.get('iterations_replayed')} vs runs.jsonl {AR.get('iterations_runs_jsonl')}, forwards {AR.get('forwards_replayed')} vs observed {AR.get('engine_steps_observed')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
