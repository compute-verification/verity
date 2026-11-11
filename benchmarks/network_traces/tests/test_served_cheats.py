"""served_cheats, the four-switch cheating server. The proxy's stand-in endpoint streams GLM-5.3's tokens and answers
/v1/models and /tokenize, so `certifier_serve.plan_of` plans it. The failures switch runs whole on a `VirtualClock`:
its control is accepted with every connection served, and its cheat's connections fail on the relay's certified
record at the cells chosen, which the relay's check names. Each switch's row says what was expected and what
happened, an errored run is a row, and the timing switch's row reads the timing test's result. One honest run of
the proxy end to end (slow) checks the stand-in and the recipe: honest serving is accepted, and each length cheat's
event is in its run's record."""
from __future__ import annotations

import asyncio
import importlib.util
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
HONEST = [sc.MERGED_TOKENS] * sc.REQUESTS


def _accepted(events: list[int] = HONEST, windows: int = 3) -> dict:
    check = {"verdict": "accepted", "windows": windows, "network_verdict": {"accepted": True, "refused": []},
             "audit_accepted": True, "salt_accepted": True, "salt_again": "refused: again",
             "rows": [{"window": w, "n2": {"accepted": True, "codes": [], "public": {}}} for w in range(windows)]}
    return {"check": check, "events": events}


def _refused(codes: list[str], events: list[int] = HONEST, window: int = 0, windows: int = 3) -> dict:
    rows = [{"window": w, "n2": {"accepted": w != window, "codes": codes if w == window else [], "public": {}}}
            for w in range(windows)]
    check = {"verdict": "refused", "windows": windows, "network_verdict": {"accepted": False, "refused": [window]},
             "audit_accepted": True, "salt_accepted": True, "salt_again": "refused: again", "rows": rows}
    return {"check": check, "events": events}


def test_the_stand_in_streams_glm_tokens_and_is_planned_as_a_served_record():
    """The stand-in answers /v1/models and /tokenize and streams GLM-5.3's tokens, so `plan_of` plans it: a served
    profile over at least one window, every request of the plan."""
    async def go():
        server = await asyncio.start_server(sc.StandIn(1, merges=lambda r: r % 2 == 1), "127.0.0.1", 0)
        url = "http://%s:%d" % server.sockets[0].getsockname()[:2]
        try:
            return await asyncio.to_thread(cs.plan_of, sc.plan_args("served"), url, lambda: 0.0)
        finally:
            server.close()

    probed, _, plan = asyncio.run(go())
    assert plan["windows"] >= 1 and plan["occupancy"]["kind"] == "served"
    assert len(plan["requests"]) == sc.REQUESTS and probed["responses"]


def test_the_failures_switch_fails_connections_on_the_relays_record_and_its_control_is_accepted(tmp_path):
    """With no failure the relay serves every connection, certifies its three records and its check accepts. With
    `FAILED`'s servers withholding, exactly those connections fail late at the cells chosen, on a record that is
    still certified, and the relay's check refuses the run naming each. The row reports both, and the table's charge
    is the calculator's for this run's connections and failures."""
    runs = sc.relay(tmp_path)
    control, cheat = runs["control"], runs["failed"]
    assert control["check"]["accepted"], control["check"]["problems"]
    assert len(control["log"]) == sc.REQUESTS and not any(c["failed"] for c in control["log"])
    assert control["check"]["summary"]["ended"] == sc.REQUESTS
    failed = {c["ordinal"]: (c["failed"], c["failed_at"]) for c in cheat["log"] if c["failed"]}
    assert failed == {k + 1: ("late", at) for k, at in sc.FAILED.items()}
    assert not cheat["check"]["accepted"]
    for ordinal, (_, at) in failed.items():
        assert any(f"connection {ordinal} failed closed (late) at egress cell {at}" in p
                   for p in cheat["check"]["problems"]), cheat["check"]["problems"]
    summary = cheat["check"]["summary"]
    assert summary["certified_windows"] == len(sc.ws.CERTIFIED) and summary["failed"] == ["fail"]
    assert summary["ended"] == sc.REQUESTS - len(sc.FAILED)

    row = sc._failures_row(runs)
    assert row["ok"] is True and row["accepted"] is False and row["failed"] == len(sc.FAILED)
    assert "REFUSED" in row["actual"] and "ACCEPTED with none failed" in row["actual"]
    cells, n = sc.RELAY.egress_cells, sc.REQUESTS
    assert row["charged_bits"] == {
        "scan": sc.LB.terms(n, 1, cells=cells, steps=sc.RELAY.window).cells,
        "lengths_held": sc.LB.terms(n, 1, cells=cells, steps=sc.RELAY.window, changes=("lengths",),
                                    failures=len(sc.FAILED)).cells}
    assert row["charged_bits"]["scan"] > row["charged_bits"]["lengths_held"] > 0


def test_a_failures_control_that_isnt_an_honest_run_is_not_a_result(tmp_path):
    """The row is a result only when the run with no failure was accepted: a relay that turned connections away, or
    a control whose check refused, makes the cheat's verdict say nothing."""
    runs = sc.relay(tmp_path)
    runs["control"]["check"] = runs["control"]["check"] | {"accepted": False}
    row = sc._failures_row(runs)
    assert row["ok"] is False and "NOT as an honest run with none failed" in row["actual"]


def test_each_proxy_row_says_the_expected_and_the_actual():
    """The baseline is ok only when accepted. tokens reads its verdict from the run: an accepted run is not caught,
    and says the I/O tie is not run here. length is caught only when both the late and the early run are refused, and
    carries each run's events a connection and N2's profile."""
    assert sc._baseline_row(_accepted())["ok"] is True
    assert sc._baseline_row(_refused(["a-code"]))["ok"] is False
    tok = sc._tokens_row(_accepted(), _accepted())
    assert tok["caught"] is False and "not caught" in tok["actual"] and "I/O tie is not run here" in tok["actual"]
    caught = sc._tokens_row(_refused(["a-code"]), _accepted())
    assert caught["caught"] is True and "a-code" in caught["actual"] and "not caught" not in caught["actual"]
    late, early = HONEST[:-1] + [HONEST[-1] + 1], [HONEST[0] - 1] + HONEST[1:]
    length = sc._length_row(_refused(["a-code"], late), _refused(["b-code"], early), _accepted(), "planned")
    assert length["caught"] is True and "planned profile" in length["actual"]
    assert str(late) in length["actual"] and str(early) in length["actual"] and "b-code" in length["actual"]
    assert sc._length_row(_accepted(late), _refused(["b-code"], early), _accepted(), "served")["caught"] is False
    assert "expected:" in sc.table({"baseline": sc._baseline_row(_accepted())})


def test_an_errored_run_is_a_reported_row_not_a_crash():
    """A run that couldn't complete (an {"error": ...} check) is reported with its error, not raised."""
    err = {"check": {"error": "RuntimeError('boom')"}, "events": None}
    assert "boom" in sc._baseline_row(err)["actual"] and sc._baseline_row(err)["ok"] is False
    assert "boom" in sc._tokens_row(err, _accepted())["actual"] and sc._tokens_row(err, _accepted())["error"]
    row = sc._length_row(err, _refused(["c"]), _accepted(), "served")
    assert "boom" in row["actual"] and row["error"] and row["caught"] is False


def test_the_timing_row_reads_the_timing_tests_figure_against_its_budget():
    """The row states the budget's two ends at the result's interval, the figure with its bounds, and whether the
    ceiling is under the budget; a ceiling over it says how many scored intervals show the budget. A run with no
    result, a failed guard or a case that is no measurement is an error, never a figure."""
    goal = sc.channel.target(0.1)
    result = {"value": 0.01, "lower": 0.0, "upper": 0.05, "n": 600, "ceiling": 0.05, "exit": 0, "problems": [],
              "best": {"attack": "burst", "observer": "spin", "stamp": "kernel"}, "guard": {"ok": True},
              "target": goal | {"met": [False, False]}}
    row = sc._timing_row(result)
    lo, hi = sorted(goal["bits"])
    assert f"{lo:.3g} to {hi:.3g}" in row["expected"] and "0.1 s" in row["expected"]
    assert "0.01 bits an interval" in row["actual"] and "not under the budget" in row["actual"]
    assert str(sc.channel.samples_to_show(hi)["least"]) in row["actual"] and "error" not in row
    under = sc._timing_row(result | {"ceiling": lo / 2, "target": goal | {"met": [True, True]}})
    assert "under both ends of the budget" in under["actual"] and "scored intervals to show" not in under["actual"]
    for broken in (result | {"exit": 1, "guard": {"ok": False}}, result | {"exit": 2, "problems": ["a tick late"]},
                   {"error": "the timing test wrote no result (exit 1): boom"}):
        assert sc._timing_row(broken)["error"]
    assert "a tick late" in sc._timing_row(result | {"exit": 2, "problems": ["a tick late"]})["actual"]


@pytest.mark.slow
def test_honest_serving_through_the_proxy_is_accepted_and_each_length_cheat_is_in_its_record(tmp_path):
    """The proxy's recipe end to end on the injected clock, which builds the Lean verifier, so it is slow: the honest
    run is accepted with every connection's events the plan's, the late run's record holds one event more on one
    connection and the early run's one fewer."""
    rows = asyncio.run(sc._serve(tmp_path, ("length",), "served"))
    assert rows["baseline"]["ok"] is True, rows["baseline"]["actual"]
    assert str(HONEST) in rows["baseline"]["actual"]
    assert str(HONEST[:-1] + [HONEST[-1] + 1]) in rows["length"]["actual"], rows["length"]["actual"]
    assert str([HONEST[0] - 1] + HONEST[1:]) in rows["length"]["actual"], rows["length"]["actual"]
    assert "error" not in rows["length"]
