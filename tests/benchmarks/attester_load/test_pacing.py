"""`benchmarks/attester_load/pacing.py`: a request's outcome under a network attester's schedule, a run's replay and what
it writes, on made-up requests."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from verity_bench import kit

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("attester_load_pacing", ROOT / "benchmarks/attester_load/pacing.py")
P = sys.modules[_spec.name] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

STAMP = {"card": "none", "gpus": [], "driver": None, "cuda": None, "node": "test", "commit": None}
WIDE = P.Schedule(bucket_ms=100, slots=4, delay=10, cell_bytes=4096, lanes=4, ingress_cells=64, ingress_buckets=8,
                  egress_cells=64, max_tokens_cap=40)


def user(*turns: tuple[float, float, float, int, int]) -> dict:
    """A vLLM bench `result.json` of one user, a turn each (start, time to first token, latency, prompt ids, tokens)."""
    t0, ttft, lat, p, k = zip(*turns)
    return {"model_id": "made-up", "start_times": list(t0), "ttfts": list(ttft), "latencies": list(lat),
            "input_lens": list(p), "output_lens": list(k)}


def test_the_card_s_schedule_is_read_from_the_card() -> None:
    s = P.Schedule.of_card()
    card = json.loads(P.CARD.read_text())["value"]["schedule"]
    assert (s.bucket_ms, s.slots, s.delay, s.lanes, s.max_tokens_cap) == (card["bucket_ms"], card["slots"], card["delay"],
                                                                          card["lanes"], card["max_tokens_cap"])
    assert set(P.schedules(s)) >= {"card"} and P.schedules(s)["card"] == s


def test_an_id_request_s_cells_grow_with_its_prompt() -> None:
    assert P.request_cells(1, 8, 4096) == 1
    many = P.request_cells(4000, 8, 256)
    assert many > 4000 * 11 // 256 and many == P.request_cells(4000, 8, 256)


def test_an_on_time_request_is_answered_when_its_end_cell_leaves() -> None:
    why, through = P.outcome(WIDE, ttft=0.2, latency=0.5, prompt_ids=10, tokens=12)
    assert why is None and through == pytest.approx(0.05 + (10 + 12 // 4) * 0.1 + P.PACED_MS / 1000)
    assert P.outcome(replace(WIDE, bucket_ms=1000, paced_ms=200), 0.2, 0.5, 10, 12)[1] == pytest.approx(0.5 + 13 + 0.2)


def test_the_server_gets_the_request_at_its_end_s_bucket_and_the_delay_counts_from_admission() -> None:
    s = replace(WIDE, cell_bytes=256, slots=1)
    cells = P.request_cells(30, 12, 256)
    assert cells == 3
    on_time = (s.delay - (cells - 1)) * 0.1
    assert P.outcome(s, on_time - 1e-9, on_time, 30, 12)[0] is None
    assert P.outcome(s, on_time + 1e-3, on_time + 1e-3, 30, 12)[0] == "late"


@pytest.mark.parametrize("change, served, reason", [
    ({}, (0.2, 0.5, 10, 41), "cap"),
    ({"cell_bytes": 256, "ingress_cells": 4}, (0.2, 0.5, 400, 12), "ingress"),
    ({"egress_cells": 12}, (0.2, 0.5, 10, 12), "limit"),
    ({}, (1.1, 1.2, 10, 12), "late"),
    ({}, (0.2, 9.0, 10, 12), "late"),
])
def test_a_request_fails_the_first_bound_it_meets(change, served, reason) -> None:
    ttft, lat, p, k = served
    assert P.outcome(replace(WIDE, **change), ttft, lat, p, k)[0] == reason


def test_a_replay_moves_a_user_s_later_turns_by_what_its_earlier_turns_added() -> None:
    passes = [[user((0.0, 0.2, 0.5, 10, 12), (0.5, 0.2, 0.5, 10, 12))]]
    r = P.replay(WIDE, passes)
    through = (10 + 3 + 1) * 0.1
    assert (r.passes, r.requests, r.concurrent) == (1, 2, 1)
    assert r.ratios == pytest.approx((2 * through / 1.0,))
    assert r.percent == pytest.approx(100 * (2 * through - 1)) and not any(r.failed.values())


def test_a_schedule_under_which_a_request_fails_has_no_figure() -> None:
    r = P.replay(WIDE, [[user((0.0, 0.2, 0.5, 10, 12)), user((0.0, 1.5, 2.0, 10, 12))]])
    assert r.failed["late"] == 1 and r.percent is None and r.spread is None


def test_a_schedule_with_fewer_lanes_than_the_run_s_concurrency_is_refused() -> None:
    passes = [[user((0.0, 0.2, 0.5, 10, 12)), user((0.1, 0.2, 0.5, 10, 12))]]
    with pytest.raises(SystemExit, match="lanes"):
        P.replay(replace(WIDE, lanes=1), passes)


def test_passes_are_read_in_lifetime_then_repetition_order(tmp_path) -> None:
    for life, rep, k in ((2, 1, 30), (1, 2, 20), (1, 1, 10)):
        d = tmp_path / f"prod.{life}/agentic@4/rep-{rep}/user-0"
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps(user((0.0, 0.1, 0.2, 5, k))))
    assert [p[0]["output_lens"] for p in P.passes_of(tmp_path, "prod", "agentic@4")] == [[10], [20], [30]]
    with pytest.raises(SystemExit, match="no pass"):
        P.passes_of(tmp_path, "prod", "agentic@1")


def test_a_run_is_fetched_from_its_record_s_passes_alone(tmp_path) -> None:
    asked = {}

    class Store:
        def get_attempt(self, run):
            return {"outputs": {"run_record": "art:rec"}} if run == "r1" else None

        def fetch(self, art, to, paths):
            asked.update(art=art, paths=paths)
            return to

    assert P.fetched("r1", tmp_path, "prod", "agentic@4", store=Store) == tmp_path / "out"
    assert asked == {"art": "art:rec", "paths": ["out/prod.*/agentic@4/rep-*/user-*/result.json"]}
    with pytest.raises(SystemExit, match="no record"):
        P.fetched("r2", tmp_path, "prod", "agentic@4", store=Store)


def test_each_schedule_is_written_as_the_metric_marked_modeled(monkeypatch, tmp_path) -> None:
    passes = [[user((0.0, 0.2, 0.5, 10, 12)), user((0.0, 1.5, 2.0, 10, 12))]]
    doc = P.document("r1", "none", "prod", "agentic@4", {"slow": replace(WIDE, delay=30), "fast": WIDE}, passes)
    monkeypatch.setattr(kit, "host_stamp", lambda: STAMP)
    assert P.write(doc, tmp_path) == 2
    got = {m["member"]: m for m in json.loads((tmp_path / "result.json").read_text())["measurements"]}
    assert set(got) == {"slow/made-up/none", "fast/made-up/none"}
    assert {m["name"] for m in got.values()} == {P.METRIC} and {m["mark"] for m in got.values()} == {"modeled"}
    assert got["slow/made-up/none"]["value"] > 0 and got["fast/made-up/none"]["value"] is None
    assert "fails at least 1 of 2 requests" in got["fast/made-up/none"]["why"]
    assert "| fast |" in P.render(doc) and "none: fails" in P.render(doc)
