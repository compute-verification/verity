"""The stochastic row's sampler value check (`checks/stoch_value.py`): what it lifts from a match_stoch.json, pure (no store, no GPU)."""
from __future__ import annotations

import copy

from .checks import stoch_value as SV
from . import resolver


def _doc(n: int = 3) -> dict:
    rows = [{"step": s, "row": 0, "request_id": "r0", "event": s, "pos": 255 + s, "served": 100 + s, "token": 100 + s, "V": 128256, "S": 32,
             "chain": ["_topp_sb_stats_kernel", "_topp_sb_mask_kernel", "_temperature_kernel"], "program_S": 32, "S_agree": True,
             "declared": {"seed": 7, "temperature": 0.8, "top_p": 0.95, "pos": 255 + s}, "device": {"seed": 7, "pos": 255 + s},
             "device_facts_match": True, "masked_row_equal_lanes": 128256, "masked_row_equal": True, "status": "exact-match",
             "eval_seconds": 27.5 + s, "evaluator": "lane"} for s in range(n)]
    return {"summary": {"engine_steps_in_record": n, "sampled_steps_in_log": n, "steps_chain_absent": 0, "sampled_rows_total": n,
                        "rows_checked": n, "complete": True, "rows": {"exact-match": n, "mismatch": 0, "no-snapshot": 0, "skipped": 0},
                        "device_facts_all_match": True, "verdict": "PASS", "seconds": 66.0},
            "reasons": [], "declared_facts": {"r0": {"seed": 7}}, "header_vs_manifest": [], "program_splits": {"r0": {"derived": 142}},
            "evaluator": {"name": "lane", "eval_seconds_total": 61.3}, "utc": "2026-09-23T20:39:27Z", "rows": rows}


def test_timings_and_paths_are_not_the_contract() -> None:
    a, b = _doc(), _doc()
    b["summary"]["seconds"] = 999.0
    b["utc"] = "later"
    b["evaluator"]["eval_seconds_total"] = 1.0
    for r in b["rows"]:
        r["eval_seconds"] = 0.1
    assert SV.lift(a, range(3)) == SV.lift(b, range(3))


def test_a_moved_token_or_step_changes_the_contract() -> None:
    base = SV.lift(_doc(), range(3))
    moved = _doc()
    moved["rows"][1]["token"] = 5
    moved["rows"][1]["status"] = "mismatch"
    assert SV.lift(moved, range(3))["rows_sha256"] != base["rows_sha256"]
    assert SV.lift(_doc(), [0, 1])["snapshot_steps"] != base["snapshot_steps"]
    fewer = copy.deepcopy(_doc())
    fewer["summary"]["rows_checked"] = 2
    assert resolver.diff(base, SV.lift(fewer, range(3)))


def test_a_greedy_row_does_not_apply() -> None:
    class Greedy:
        def has(self, rel):
            return False
    assert "greedy" in SV.applies(Greedy())
