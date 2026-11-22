"""(T0) The sampler VALUE check of a stochastic row reproduces (a row's Match's `rcs`, `match_summary.json` `stoch_value_check`):
`match_stoch.json` -- every sampled row of the record recomputed with the Program's own sampler Definition (`GumbelTopPTokenSelect_v1{V,S}`,
the registry interpreter walking the V lanes) from the Match record's snapshots of the device sampler chain, under the DECLARED seed /
temperature / top-p.  Lifted: its verdict and reasons, the summary (engine steps in the record, sampled rows, rows checked, the row tally,
completeness, device-tied facts, masked rows, the Program's S agreement, the sample ledger), the declared facts, the per-request S the Build
derived, and per sampled row (step, row, request, event, pos, served token, recomputed token, V, S, chain, declared / device facts,
status) as a sha256; plus the snapshot steps the stochastic Match took (`match/run.json` `snapshot_steps`: EVERY engine step -- the value
check is complete only when every sampled row is snapshotted).

Read from the record kept on the node or, in candidate mode, from the candidate's own `match_stoch.json` + `match/run.json`, lifted the
same way.  STRONG: a sampler Definition that moves a lane, a sample read at another counter, an S derived differently, an unsnapshotted
step -- each moves a status, a count or the verdict.

Applies to stochastic rows only (a greedy row has no match_stoch.json: its Match's rcs = -1).
"""
from __future__ import annotations

import hashlib

from .. import resolver

NAME = "stoch_value"
TIER = "T0"          # inputs ~180 MB (the capture log, snapshots.json, the sampler chain's values); the lane walk is ~30 CPU-s per sampled row
STRENGTH = "strong"
INPUTS = ("match_stoch.json", "match/run.json", "match/capture/{log.jsonl.gz,snapshots.json,values/<sampler chain>}", "build_request*/result.json",
          "build_workload/workload_program.json", "workloads/<row>.json")

SUMMARY_KEYS = ("engine_steps_in_record", "sampled_steps_in_log", "steps_chain_absent", "sampled_rows_total", "rows_checked", "complete", "rows",
                "device_facts_all_match", "device_facts_rows", "masked_rows_all_equal", "program_S_all_agree", "program_S_rows", "draw_ledger_ok",
                "lag_slot_exercised", "requests_with_eos_before_cap", "requests_at_cap")
ROW_KEYS = ("step", "row", "request_id", "event", "pos", "served", "token", "V", "S", "chain", "program_S", "S_agree", "declared", "device",
            "device_facts_match", "masked_row_equal_lanes", "masked_row_equal", "status")


def applies(row: resolver.Row) -> str | None:
    return None if row.has("match_stoch.json") else "no match_stoch.json on this row (a greedy row: no sampler value check, rcs = -1)"


def _sha(obj) -> str:
    return hashlib.sha256(resolver.canonical(obj).encode()).hexdigest()


def lift(ms: dict, snapshot_steps) -> dict:
    """The contract from a match_stoch.json and the Match's snapshot steps."""
    s = ms.get("summary") or {}
    rows = [{k: r.get(k) for k in ROW_KEYS} for r in ms.get("rows") or []]
    return {"verdict": s.get("verdict"), "reasons": list(ms.get("reasons") or []), "summary": {k: s.get(k) for k in SUMMARY_KEYS},
            "declared_facts": ms.get("declared_facts"), "header_vs_manifest": list(ms.get("header_vs_manifest") or []),
            "program_splits": ms.get("program_splits"), "evaluator": (ms.get("evaluator") or {}).get("name"),
            "rows_n": len(rows), "rows_sha256": _sha(rows),
            "snapshot_steps": sorted(int(x) for x in snapshot_steps) if snapshot_steps is not None else None}


def expected(row: resolver.Row) -> dict:
    run = row.json("match/run.json") if row.has("match/run.json") else {}
    return lift(row.json("match_stoch.json"), run.get("snapshot_steps"))


def actual(row: resolver.Row) -> dict:
    if kept := resolver.kept_on_the_node(row, "match_stoch.json"):
        return kept
    if row.candidate_root() is not None:
        ms_p = row.candidate("match_stoch.json")
        if ms_p is None:
            return {"source": "candidate", "verdict": resolver.not_applicable("the candidate has no match_stoch.json")}
        run_p = row.candidate("match/run.json")
        run = resolver.read_json(run_p) if run_p is not None else {}
        return dict(lift(resolver.read_json(ms_p), run.get("snapshot_steps")), source="candidate")
    raise resolver.NotResolvable(f"{row.key}: no match_stoch.json kept or in a candidate; the sampler value check is not recomputed")
