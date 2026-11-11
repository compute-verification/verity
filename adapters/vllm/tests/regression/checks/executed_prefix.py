"""(d) The executed prefix of record re-derives: `executed_prefix.executed_prefix_of_record` run with the CURRENT code over the
record's own facts (the driver's served token ids per request from `match/capture/tokens.json` -- `match/control/tokens.json` when the
capture dir was moved -- the workload's caps and request order, the Program's declared EOS ids and LAG from
`build_workload/workload_program.json.stop_rule`) must give, per request, the `served_output_len` / `executed_prefix_len` /
`cap_exhausted` / `eos_stop` the record states (tokens.json's own columns, and G6's `lengths_by_request` in `global_match.json`), the
same EOS-stopped set as `global_match.json.lag.eos_stop_requests`, and `ok`.

Pure and fast (ms).  WEAK by the reference's own admission: no reference row exercises an EOS stop (`lag.exercised = false` everywhere), so
the LAG arithmetic (a_r + LAG) is never exercised by these rows -- only the length-stop branch and the fail-closed rules on the declared
inputs are.  A v2 that changes the rule for EOS stops will pass this check and must be covered by a synthetic unit test instead
(`tests/program/test_continuation.py`, `tests/commit`).
"""
from __future__ import annotations

import json

from .. import resolver

NAME = "executed_prefix"
TIER = "T0"
STRENGTH = "weak"
INPUTS = ("match/capture/tokens.json | match/control/tokens.json", "build_workload/workload_program.json", "global_match.json", "workloads/<row>.json")


def _tokens_rel(row: resolver.Row) -> str | None:
    for rel in ("match/capture/tokens.json", "match/control/tokens.json"):
        if row.has(rel):
            return rel
    return None


def applies(row: resolver.Row) -> str | None:
    if _tokens_rel(row) is None:
        return "no tokens.json (served tokens) on this row"
    if not row.has("build_workload/workload_program.json"):
        return "no GP-01 workload_program.json (declared EOS / LAG) on this row"
    return None


def _order(fx: dict) -> dict[str, str]:
    """workload request_id (UUID) -> r<k> by position in the workload manifest."""
    return {str(r.get("request_id")): f"r{i}" for i, r in enumerate(fx.get("requests") or [])}


def _facts(row: resolver.Row) -> tuple[dict, dict, dict, list, dict]:
    fx = json.load(open(row.workload()))
    order = _order(fx)
    caps = {order[str(r["request_id"])]: int(r["max_tokens"]) for r in fx["requests"]}
    tok = row.json(_tokens_rel(row))
    wp = row.json("build_workload/workload_program.json")
    stop = wp.get("stop_rule") or {}
    return fx, order, caps, tok["requests"], stop


def expected(row: resolver.Row) -> dict:
    fx, order, caps, treqs, stop = _facts(row)
    per = {}
    for e in treqs:
        rid = order.get(str(e.get("request_id")))
        if rid is None:
            continue
        per[rid] = {"served_output_len": e.get("served_output_len", len(e.get("generated_token_ids") or [])), "executed_prefix_len": e.get("executed_prefix_len"),
                    "cap_exhausted": e.get("cap_exhausted"), "eos_stop": e.get("eos_stop_before_cap")}
    # (the tokens file's path is NOT part of the contract: the vault mirror holds it under match/control/, the live dirs under match/capture/)
    out = {"lag": stop.get("lag"), "eos_of_record": sorted(int(x) for x in (stop.get("eos_token_ids") or [])),
           "requests": per, "ok": True, "any_eos_stop": any(v.get("eos_stop") for v in per.values())}
    if row.has("global_match.json"):
        gm = row.json("global_match.json")
        out["eos_stop_requests"] = sorted((gm.get("lag") or {}).get("eos_stop_requests") or [])
        g6 = next((c for c in gm.get("checks") or [] if c.get("id") == "G6"), None) or {}
        lens = g6.get("lengths_by_request") or {}
        table = (gm.get("request_ids") or {}).get("table") or {}
        out["g6_lengths"] = {table.get(k, k): {"served_output_len": v.get("served_output_len"), "executed_prefix_len": v.get("executed_prefix_len")}
                             for k, v in lens.items() if isinstance(v, dict)}
    return out


def actual(row: resolver.Row) -> dict:
    from verity_vllm.commit.executed_prefix import executed_prefix_of_record
    fx, order, caps, treqs, stop = _facts(row)
    served = {order[str(e["request_id"])]: [int(t) for t in (e.get("generated_token_ids") or [])] for e in treqs if str(e.get("request_id")) in order}
    epr = executed_prefix_of_record(sorted(caps, key=lambda r: int(r[1:])), served, caps=caps, eos_of_record=stop.get("eos_token_ids"), lag=stop.get("lag"))
    per = {rid: {"served_output_len": r["served_output_len"], "executed_prefix_len": r["executed_prefix_len"], "cap_exhausted": r["cap_exhausted"], "eos_stop": r["eos_stop"]}
           for rid, r in epr["requests"].items()}
    out = {"tokens_source": _tokens_rel(row), "lag": epr["lag"], "eos_of_record": epr["eos_of_record"], "requests": per, "ok": epr["ok"],
           "any_eos_stop": epr["any_eos_stop"], "problems": epr["problems"][:8],
           "eos_stop_requests": sorted(r for r, v in per.items() if v["eos_stop"]),
           "g6_lengths": {rid: {"served_output_len": v["served_output_len"], "executed_prefix_len": v["executed_prefix_len"]} for rid, v in per.items()},
           "source": "recomputed"}
    return out
