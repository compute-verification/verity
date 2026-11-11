"""served_cheats, the four-switch cheating server against a served record. The stand-in endpoint streams GLM-5.3's
tokens and answers /v1/models and /tokenize, so `certifier_serve.plan_of` plans it; each switch's row says what the
audit was expected to do and what it did, and the timing switch states the budget. One honest run end to end on the
injected clock (slow) checks the stand-in and the recipe: honest serving is accepted."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, rel: str):
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sc = _load("network_traces_served_cheats", "benchmarks/network_traces/served_cheats.py")
cs = sc.cs


def _accepted(windows: int = 3) -> dict:
    return {"verdict": "accepted", "windows": windows, "network_verdict": {"accepted": True, "refused": []},
            "audit_accepted": True, "salt_accepted": True, "salt_again": "refused: again",
            "rows": [{"window": w, "n2": {"accepted": True, "codes": [], "public": {}}} for w in range(windows)]}


def _refused(codes: list[str], window: int = 0, windows: int = 3) -> dict:
    rows = [{"window": w, "n2": {"accepted": w != window, "codes": codes if w == window else [], "public": {}}}
            for w in range(windows)]
    return {"verdict": "refused", "windows": windows, "network_verdict": {"accepted": False, "refused": [window]},
            "audit_accepted": True, "salt_accepted": True, "salt_again": "refused: again", "rows": rows}


def test_the_stand_in_streams_glm_tokens_and_is_planned_as_a_served_record():
    """The stand-in answers /v1/models and /tokenize and streams GLM-5.3's tokens, so `plan_of` plans it: a served
    profile over at least one window, every request padded to the body's tokens."""
    async def go():
        server = await asyncio.start_server(sc.StandIn(4, merges=lambda r: r % 2 == 1), "127.0.0.1", 0)
        url = "http://%s:%d" % server.sockets[0].getsockname()[:2]
        try:
            probed, report, plan = await asyncio.to_thread(cs.plan_of, sc.plan_args("served"), url, lambda: 0.0)
        finally:
            server.close()
        return probed, report, plan

    probed, report, plan = asyncio.run(go())
    assert plan["windows"] >= 1 and plan["occupancy"]["kind"] == "served"
    assert len(plan["requests"]) == sc.REQUESTS
    tau = sc.ar.Setup(report).egress_params.bucket_ms / 1000.0
    assert tau > 0 and probed["responses"]
    # the timing switch states the budget tau/600 .. tau/240 bits an interval from the plan's own egress interval
    row = sc._timing_row(report)
    assert row["switch"] == "timing" and f"{tau / 600:.3g}" in row["expected"] and f"{tau / 240:.3g}" in row["expected"]


def test_each_switch_row_says_the_expected_and_the_actual():
    """tokens is not caught under count-only N2 (it waits on the I/O tie); a refused token run would read its codes.
    length is refused by the counts, failures accepted and charged, and the baseline accepted."""
    assert sc._baseline_row(_accepted())["ok"] is True
    tok = sc._tokens_row(_accepted())
    assert tok["caught"] is False and "not caught" in tok["actual"] and "I/O tie" in tok["actual"]
    assert sc._tokens_row(_refused(["io_tie.content"]))["caught"] is True
    length = sc._length_row(_refused(["served.counts"]), _refused(["served.counts"]))
    assert length["caught"] is True and "refused" in length["actual"]
    assert sc._length_row(_accepted(), _refused(["served.counts"]))["caught"] is False
    fail = sc._failures_row(_accepted(), 8)
    assert fail["ok"] is True and fail["charged"] is True and "8 connection" in fail["actual"]
    assert sc._failures_row(_refused(["served.counts"]), 8)["ok"] is False
    assert "expected:" in sc.table({"baseline": sc._baseline_row(_accepted())})


def test_an_errored_run_is_a_reported_row_not_a_crash():
    """A run that couldn't complete (an {"error": ...} check) is reported, not raised."""
    err = {"error": "RuntimeError('boom')"}
    assert "boom" in sc._tokens_row(err)["actual"]
    assert "boom" in sc._length_row(err, _refused(["c"]))["actual"]
    assert sc._failures_row(err, None)["ok"] is False and "boom" in sc._failures_row(err, None)["actual"]


@pytest.mark.slow
def test_honest_serving_against_the_stand_in_is_accepted_end_to_end(tmp_path):
    """The recipe end to end on the injected clock: the honest baseline is accepted, and the length cheat's late event
    is refused by the counts. Builds the network-records verifier, so it is slow."""
    rows = asyncio.run(sc._serve(tmp_path, ("length",), "served"))
    assert rows["baseline"]["ok"] is True, rows["baseline"]["actual"]
    assert "late refused" in rows["length"]["actual"], rows["length"]["actual"]
