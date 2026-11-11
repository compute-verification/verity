"""`test_stoch_draw_ledger`'s sample ledger on the two B8 stochastic records of record, read from their evidence dirs (`EVID`,
written on a GPU pod, which no tracked file holds). Skipped where they are absent.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from verity_vllm.check.replay.stoch_recompute import draw_accounting

EVID = Path(__file__).resolve().parents[2] / "out" / "gen" / "sweep" / "evidence" / "stoch"


@pytest.mark.parametrize("row, att, n_rows, V", [
    ("llama32-1b__bf16__l40s__tp1__b8__i1024__o128__mixed__stoch-t0.8-p0.95__bi-eager", "r17_1627_b8_int5_3b9e301c9", 526, 128256),
    ("qwen25-15b__bf16__l40s__tp1__b8__i1024__o128__mixed__stoch-t0.8-p0.95__bi-eager", "r17_1846_b8_int1_88dd07fd3", 388, 151936),
])
def test_the_two_b8_stoch_records_of_record_carry_a_clean_ledger_every_request_to_its_cap_r5_not_exercised(row, att, n_rows, V):
    """The B8 stochastic rows of record (evidence dirs): every sampled row is exact-match, so match_stoch.json's rows ARE the whole
    ledger; each of the 8 requests drew exactly cap times at pos prompt_len - 1 + k, one per step; no request stopped at EOS -> the
    lag slot was not scheduled on either record and the ledger says R5 is NOT MEASURED there (the negative is named, not passed)."""
    att_dir = EVID / row / att
    ms = att_dir / "match_stoch.json"
    if not ms.exists():
        pytest.skip(f"evidence dir absent: {att_dir}")
    st = json.loads(ms.read_text())
    rows = [r for r in st["rows"] if r.get("request_id")]
    assert len(rows) == n_rows == st["summary"]["sampled_rows_total"] and all(r["status"] == "exact-match" for r in rows)
    d: dict[str, list[dict]] = {}
    for r in rows:
        d.setdefault(r["request_id"], []).append({"step": r["step"], "row": r["row"], "pos": r["pos"], "served": r["served"]})
    wl = Path(__file__).resolve().parents[2] / "workloads" / f"{row}.json"
    gp = json.loads(gzip.open(att_dir / "build_workload_program.json.gz", "rt").read())
    lag_decl = (gp.get("stop_rule") or {}).get("eos_lag_rows")
    out = draw_accounting(d, wl, att_dir, lag_attested=(1, "RunHeader.profile.engine_kwargs.async_scheduling=True (the row_pod's vLLM v1 default; "
                                                                 "GM-01 G6 read the same, verified_eos_steps [])"))
    assert out["problems"] == [], out["problems"]
    assert out["stop_rule"]["eos_token_ids"] and out["stop_rule"]["eos_lag_rows_declared"] == lag_decl
    assert len(out["requests"]) == 8 and out["requests_at_cap"] == 8 and out["lag_slot_exercised"] is False
    wlj = json.loads(wl.read_text())
    for q in wlj["requests"]:
        e = out["requests"][q["request_id"]]
        assert e["draws"] == q["max_tokens"] and e["first_pos"] == q["prompt_len"] - 1 and e["last_pos"] == q["prompt_len"] - 2 + q["max_tokens"]
        assert e["one_draw_per_step"] and e["ordered_set_equal"] and e["finish"] == "length" and e["drawn_not_expected"] == [] == e["expected_not_drawn"]
    assert sum(e["draws"] for e in out["requests"].values()) == n_rows
    assert out["measured"].startswith("R5 NOT MEASURED")
