"""served_commit's TP decode driver (`benchmarks/served_commit/tp_decode.py`): its batches rotate the row's request, `phases`
reads TTFT, prefill and the decode TPOT from a run's step walls and outputs as `run_workload` records them (an empty first
step, chunked prefill, requests that end a step apart), and its gate names each failure. No engine."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

DRIVER = Path(__file__).resolve().parents[2] / "served_commit" / "tp_decode.py"


def _driver():
    spec = importlib.util.spec_from_file_location("served_commit_tp_decode", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_requests_at_rotates_the_rows_prompt():
    D = _driver()
    wl = {"requests": [{"request_id": "x", "prompt_token_ids": list(range(10)), "max_tokens": 4, "prompt_text": "t"}]}
    reqs = D.requests_at(wl, 3, 4)
    assert [r["prompt_token_ids"][:2] for r in reqs] == [[0, 1], [4, 5], [8, 9]]
    assert all(sorted(r["prompt_token_ids"]) == list(range(10)) for r in reqs)
    assert [(r["request_id"], r["max_tokens"], r["arrive_step"]) for r in reqs] == [("b3-r0", 4, 0), ("b3-r1", 4, 0), ("b3-r2", 4, 0)]
    assert "prompt_text" not in reqs[0]


def test_phases_one_prefill_step():
    D = _driver()
    p = D.phases([0.001, 0.5, 0.2, 0.2, 0.3], [0, 2, 2, 2, 2], 2)
    assert p["prefill_s"] == pytest.approx(0.501)
    assert p["ttft_mean_s"] == pytest.approx(0.501)
    assert (p["decode_steps"], p["tpot_s"], p["tpot_median_s"]) == (3, pytest.approx(0.7 / 3), pytest.approx(0.2))


def test_phases_chunked_prefill_counts_decode_once_every_request_has_started():
    D = _driver()
    # four requests prefilled two at a time; the first two end one step before the others
    p = D.phases([0.1, 0.1, 0.05, 0.05, 0.05], [2, 4, 4, 4, 2], 4)
    assert p["prefill_s"] == pytest.approx(0.2)
    assert p["ttft_mean_s"] == pytest.approx((2 * 0.1 + 2 * 0.2) / 4)
    assert (p["decode_steps"], p["tpot_s"]) == (2, pytest.approx(0.05))


def test_phases_without_every_first_token():
    D = _driver()
    p = D.phases([0.1, 0.1], [1, 1], 2)
    assert (p["prefill_s"], p["ttft_mean_s"], p["tpot_s"], p["decode_steps"]) == (None, None, None, 0)


def test_gate_names_each_failure():
    D = _driver()
    ok = {"run_root": "00" * 64, "errors_n": 0, "errors": [], "learned_only_steps": 0, "identity_ok": True, "identity_violations": []}
    fin = [{**ok, "rank": 0}, {**ok, "rank": 1}]
    assert D.gate(True, fin, 2, identity=True) == {"passed": True, "why": []}
    bad = [{**ok, "rank": 0, "run_root": "00" * 32}, {**ok, "rank": 1, "errors_n": 2, "errors": ["e"], "identity_ok": False}]
    why = D.gate(False, bad, 2, identity=True)["why"]
    assert why[0].startswith("tokens differ") and "rank 0: a root of 32 bytes" in why
    assert any(w.startswith("rank 1: 2 collector errors") for w in why) and any(w.startswith("rank 1: identity check False") for w in why)
    assert D.gate(True, [{**ok, "rank": 0, "identity_ok": None}, {**ok, "rank": 1}], 2, identity=False)["passed"]
    assert D.gate(True, fin[:1], 2, identity=False)["why"] == ["ranks [0] finalized of 2"]


def test_summarize_pairs_the_arm_with_its_control():
    D = _driver()

    def row(arm, pair, tpot, gate=True):
        return {"arm": arm, "batch": 8, "pair": pair, "tokens_equal_ref": True, "gate": {"passed": gate},
                "phases": {"tpot_s": tpot, "ttft_mean_s": 2 * tpot, "prefill_s": 3 * tpot, "decode_steps": 30}}

    rows = [row("control", 0, 0.2), row("rows", 0, 0.3), row("control", 1, 0.25), row("rows", 1, 0.25)]
    s = D.summarize(rows, "rows", [8])["8"]
    assert s["control"]["tpot_s"] == pytest.approx(0.225) and s["rows"]["tpot_s"] == pytest.approx(0.275)
    assert s["tpot_ratio"] == {"median": pytest.approx(1.25), "min": pytest.approx(1.0), "max": pytest.approx(1.5), "pairs": 2}
    assert s["gates_passed"] and s["control_tokens_equal"] and s["decode_steps"] == [30]
