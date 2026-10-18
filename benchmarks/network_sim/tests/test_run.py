"""The night's summary and its expected verdicts (`run.py`), on summaries in the shape mockdc and lab_run write them,
with the recorded night's verdicts and timings (`r20261018-110501-432f`): those are the expected ones, and a differing
verdict or answer, a step that exited nonzero, a missing summary or a proxy tick past its slack each fails the night,
named in `summary.txt`."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from verity.core.protocols.certifier import calibration

HERE = Path(__file__).resolve().parents[1]
COVERAGE = "status:coverage-never-established"
CHEATED = ["occupancy differs from the declared profile", "a frame is not a committed crossing"]
LINE = 2.4e12
HELD = 21577915695104.0
SALT = {"accepted": True, "violations": [], "ingress_salt_frames": [1, 1, 1], "egress_salt_frames": 0,
        "egress_holds_salt": False, "block_received": True}
EXITS = {"demo": 0, "lab --blocks 5": 0, "mockdc": 0, "mockdc-salt": 0, "lab --fleet --blocks 2": 0, "fleet": 0,
         "fleet-swapped": 0, "lab default": 0, "python": 0, "workload/honest": 0, "workload/flip-bit": 0,
         "workload/late": 0}
WAN = [["east.wan.cert-west.wan.cert", "wan/east-west=1e+11 bit/s (declared)",
        "wan/east-west-latency=0.02 s (declared)"]]


@pytest.fixture(scope="module")
def run():
    spec = importlib.util.spec_from_file_location("network_sim_run", HERE / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rows(*refused) -> list[list]:
    bad = {(link, w): (raw, com) for link, w, raw, com in refused}
    return [[link, w, *bad.get((link, w), (None, None))] for link in ("responses", "requests") for w in range(3)]


def timing(late: float, lag: float, over: int = 0) -> dict:
    return {"slack_ms": 25.0, "tick_late_ms_max": {"egress": late, "ingress": late - 0.06},
            "ticks_over_slack": {"egress": over, "ingress": 0},
            "wire_bucket_lag_ms": {"p50": lag, "p99": lag + 1.0, "p99.9": lag + 2.5, "max": lag + 2.5}}


# Each block's worst proxy tick and median wire lag, in ms.
TIMES = {
    "mockdc": {"b0": (4.172, 5.639), "b1": (4.172, 9.977), "b2": (3.543, 5.551), "b3": (4.132, 9.833),
               "b4": (3.798, 5.425)},
    "mockdc-salt": {"b0": (4.359, 5.496), "b1": (3.856, 10.02), "b2": (4.361, 5.454), "b3": (3.439, 9.88),
                    "b4": (3.815, 5.475)},
    "fleet": {"east.b0": (3.295, 35.344), "east.b1": (3.439, 35.243), "west.b0": (3.819, 5.561),
              "west.b1": (3.661, 9.961)},
    "fleet-swapped": {"east.b0": (3.574, 35.291), "east.b1": (4.025, 35.15), "west.b0": (3.783, 5.533),
                      "west.b1": (3.805, 9.785)},
    "honest": {"b0": (4.12, 5.562), "b1": (4.157, 10.011), "b2": (3.561, 5.621), "b3": (3.874, 9.968)},
    "flip-bit": {"b0": (4.238, 5.469), "b1": (3.597, 9.99), "b2": (3.634, 5.513), "b3": (3.545, 9.918)},
    "late": {"b0": (3.605, 5.548), "b1": (3.598, 9.998), "b2": (3.424, 5.54), "b3": (4.318, 10.019)},
}


def account(advice: float, refused: float = 0.0, serviced: float = 0.0, blocks: int = 1) -> dict:
    return {"sound": not refused, "covert_bits": advice + refused + serviced, "timing_bits": advice,
            "refused_bits": refused, "serviced_bits": serviced, "carried_bits": blocks * HELD}


def out(b: str, w: int) -> list[str]:
    return [f"port {b}.cert:out [{w * 6000}, {(w + 1) * 6000}): {why}" for why in CHEATED]


def mockdc_block(strategy: str, b: str, times: tuple[float, float], against: bool | None) -> dict:
    """A block's entry as mockdc writes it, for the four strategies a fleet's blocks play."""
    verdicts, refused, acc = {
        "honest": (rows(), [], account(24.0)),
        "bits": (rows(), [], account(24.0)),
        "late": (rows(["responses", 0, COVERAGE, COVERAGE]), out(b, 0), account(20.0, LINE)),
        "extra": (rows(["responses", 2, "mismatch", "mismatch"]), out(b, 2), account(20.0, LINE)),
    }[strategy]
    return {"strategy": strategy, "accepted": all(r[2] is None for r in verdicts), "verdicts": verdicts,
            "datacenter": {"accepted": not refused, "refused": refused, "account": acc}} | timing(*times) | (
        {} if against is None else {"against_honest": {"wire_is_honests": against}})


def mockdc_summary(run: str, salt: bool) -> dict:
    t = TIMES[run]
    blocks = {"b0": mockdc_block("honest", "b0", t["b0"], None), "b1": mockdc_block("bits", "b1", t["b1"], True),
              "b2": mockdc_block("late", "b2", t["b2"], False), "b3": mockdc_block("extra", "b3", t["b3"], False),
              "b4": {"strategy": "maintenance", "accepted": False,
                     "verdicts": rows(["responses", 1, "no-verdict", "no-verdict"],
                                      ["requests", 1, "no-verdict", "no-verdict"]),
                     "datacenter": {"accepted": True, "refused": [], "account": account(16.0, serviced=HELD),
                                    "unplanned": {"accepted": False, "account": account(16.0, 2 * LINE), "refused": [
                                        f"port b4.cert:{d} [6000, 12000): no record: the certifier was down or its "
                                        "record was withheld" for d in ("out", "in")]}}} | timing(*t["b4"])}
    if salt:
        for b, x in blocks.items():
            x["salt"] = SALT if b != "b4" else SALT | {"accepted": False, "violations": [[1, "salt"]],
                                                        "ingress_salt_frames": [1, 1]}
    return {"declaration": "leaf_spine", "wan": [],
            "plan": {"epoch": [0, 18000], "maintenance": [["b4", [6000, 12000]]]}, "blocks": blocks, "exit": {b: {"proxy": 0, "peer": 0, "server": 0} for b in blocks},
            "records": {"ports": sorted(f"b{k}.cert:{d}" for k in range(5) for d in ("in", "out")), "of": 10,
                        "carried": 26, "refused": out("b2", 0) + out("b3", 2),
                        "account": account(104.0, 2 * LINE, HELD, blocks=5)}}


def fleet_summary(run: str) -> dict:
    """The fleet's run, `honest` on east.b0 and the others in turn, or with the strategies swapped between the sites."""
    t = TIMES[run]
    plays = ({"east.b0": "honest", "east.b1": "late", "west.b0": "bits", "west.b1": "extra"} if run == "fleet" else
             {"east.b0": "bits", "east.b1": "extra", "west.b0": "honest", "west.b1": "late"})
    honest = next(b for b, s in plays.items() if s == "honest")
    blocks = {b: mockdc_block(s, b, t[b], None if b == honest else s == "bits") for b, s in plays.items()}
    refused = [r for b, x in blocks.items() for r in x["datacenter"]["refused"]]
    return {"declaration": "east+west", "wan": WAN, "plan": {"epoch": [0, 18000], "maintenance": []},
            "blocks": blocks, "exit": {b: {"proxy": 0, "peer": 0, "server": 0} for b in blocks},
            "records": {"ports": sorted(f"{b}.cert:{d}" for b in blocks for d in ("in", "out")), "of": 8,
                        "carried": 22, "refused": refused, "account": account(88.0, 2 * LINE, blocks=4)}}


def workload_block(times: tuple[float, float], tamper: str | None = None, verdicts: list | None = None,
                   n3: list[str] = ()) -> dict:
    verdicts = verdicts or rows()
    return {"tamper": tamper, "certifier": {"accepted": all(r[2] is None for r in verdicts), "verdicts": verdicts},
            "n3": {"accepted": not n3, "violations": list(n3)}} | timing(*times)


def starved(b: str, first: int) -> list[str]:
    return ["values " + b + ": the unit committed no values",
            f"missing {b}:responses: 1 declared events, the first at position {first}",
            f"missing {b}:requests: the request of session {b}.1"]


def workload_summaries() -> dict[str, dict]:
    answers = {"answers": [16, 16], "reference": [16, 16]}
    h, f, late = TIMES["honest"], TIMES["flip-bit"], TIMES["late"]
    return {
        "honest": {"blocks": {b: workload_block(h[b]) for b in h},
                   "n3": {"ok": True, "codes": {}, "violations": None}} | answers,
        "flip-bit": {"blocks": {
            "b0": workload_block(f["b0"]),
            "b1": workload_block(f["b1"], "flip-bit", n3=["content b1:responses: the event at position 0 (message 4) "
                                                          "isn't the declared one"]),
            "b2": workload_block(f["b2"], n3=["content b2:requests: the request of session b2.0 isn't the declared "
                                              "one"]),
            "b3": workload_block(f["b3"])}, "n3": {"ok": False, "codes": {"content": 2}, "violations": None}} | answers,
        "late": {"blocks": {
            "b0": workload_block(late["b0"], n3=starved("b0", 6)),
            "b1": workload_block(late["b1"], n3=starved("b1", 1)),
            "b2": workload_block(late["b2"], "late", rows(["responses", 0, COVERAGE, COVERAGE]), starved("b2", 0)),
            "b3": workload_block(late["b3"], n3=starved("b3", 0))},
            "n3": {"ok": False, "codes": {"missing": 9, "values": 4}, "violations": None},
            "answers": None, "reference": [16, 16]},
    }


def night(out: Path) -> None:
    summaries = {"mockdc": mockdc_summary("mockdc", False), "mockdc-salt": mockdc_summary("mockdc-salt", True),
                 "fleet": fleet_summary("fleet"), "fleet-swapped": fleet_summary("fleet-swapped")}
    summaries |= {f"workload/{name}": s for name, s in workload_summaries().items()}
    for name, s in summaries.items():
        (out / name).mkdir(parents=True)
        (out / name / "summary.json").write_text(json.dumps(s))


def edit(path: Path, change) -> None:
    s = json.loads(path.read_text())
    change(s)
    path.write_text(json.dumps(s))


def test_the_expected_strategies_and_window_are_mockdcs(run):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "path", list(sys.path))
        spec = importlib.util.spec_from_file_location("network_sim_mockdc", HERE / "mockdc.py")
        m = importlib.util.module_from_spec(spec)
        mp.setitem(sys.modules, spec.name, m)
        spec.loader.exec_module(m)
        assert run.STRATEGIES == m.STRATEGIES
        assert m.records.cadence(m.ar.Setup(calibration.calibrate(m.trace()))).window_ticks == run.WINDOW


def test_the_recorded_runs_verdicts_are_the_expected_ones(run, tmp_path, capsys):
    night(tmp_path)
    assert run.finish(tmp_path, EXITS) == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["ok"] and summary["mismatches"] == [] and summary["ticks_over_slack"] == 0
    assert summary["worst_tick"] == {"late_ms": 4.361, "slack_ms": 25.0, "over": 0, "lag_ms": 5.454,
                                     "at": "mockdc-salt b2"}
    b4 = summary["runs"]["mockdc"]["blocks"]["b4"]
    assert b4["verdicts"]["charged"] == ["timing", "serviced"] and b4["verdicts"]["unplanned_charged"] == ["timing", "refused"]
    assert b4["account"]["covert_bits"] == 16.0 + HELD and b4["unplanned_account"]["covert_bits"] == 16.0 + 2 * LINE
    fleet = summary["runs"]["fleet"]
    assert fleet["declaration"] == "east+west" and fleet["wan"] == WAN
    assert fleet["records"]["verdicts"] == {
        "ports": ["east.b0.cert:in", "east.b0.cert:out", "east.b1.cert:in", "east.b1.cert:out", "west.b0.cert:in",
                  "west.b0.cert:out", "west.b1.cert:in", "west.b1.cert:out"], "of": 8,
        "refused": out("east.b1", 0) + out("west.b1", 2), "sound": False, "charged": ["timing", "refused"]}
    assert summary["runs"]["mockdc"]["records"]["verdicts"]["charged"] == ["timing", "refused", "serviced"]
    text = (tmp_path / "summary.txt").read_text()
    assert text == capsys.readouterr().out
    lines = text.splitlines()
    assert lines[0].split() == ["run", "block", "case", "certifier", "audit", "records", "/", "N3", "covert", "budget",
                                "(bits)", "wire", "lag", "worst", "tick"]
    assert len(lines) == 1 + 2 * 6 + 2 * 5 + 3 * 4 + 5 and all(line.endswith(" ok") for line in lines[1:35])
    row = {tuple(line.split()[:2]): " ".join(line.split()) for line in lines[1:35]}
    assert row["mockdc", "b0"] == "mockdc b0 honest accepted records accepted 24 sound 5.6 ms 4.2/25 ms ok"
    assert row["mockdc", "b2"] == ("mockdc b2 late refused responses[0] status:coverage-never-established records "
                                   "refused out [0, 6000) 2.4e+12 NOT sound 5.6 ms 3.5/25 ms ok")
    assert row["mockdc", "b4"] == ("mockdc b4 maintenance refused responses[1] no-verdict, requests[1] no-verdict records "
                                   "accepted; unplanned refused out, in [6000, 12000) 2.16e+13 sound; unplanned 4.8e+12 "
                                   "NOT sound 5.4 ms 3.8/25 ms ok")
    assert row["mockdc", "all"] == ("mockdc all 10 ports records refused b2 out [0, 6000); b3 out [12000, 18000) "
                                    "2.64e+13 NOT sound ok")
    assert row["mockdc-salt", "b1"] == ("mockdc-salt b1 bits accepted; salt accepted records accepted 24 sound 10.0 ms "
                                        "3.9/25 ms ok")
    assert "no-verdict; salt refused [1] records accepted;" in row["mockdc-salt", "b4"]
    assert row["fleet", "east.b1"] == ("fleet east.b1 late refused responses[0] status:coverage-never-established "
                                       "records refused out [0, 6000) 2.4e+12 NOT sound 35.2 ms 3.4/25 ms ok")
    assert row["fleet", "west.b0"] == "fleet west.b0 bits accepted records accepted 24 sound 5.6 ms 3.8/25 ms ok"
    assert row["fleet", "all"] == ("fleet all 8 ports records refused east.b1 out [0, 6000); west.b1 out "
                                   "[12000, 18000) 4.8e+12 NOT sound ok")
    assert row["fleet-swapped", "all"] == ("fleet-swapped all 8 ports records refused east.b1 out [12000, 18000); "
                                           "west.b1 out [0, 6000) 4.8e+12 NOT sound ok")
    assert row["workload/flip-bit", "b2"] == "workload/flip-bit b2 - accepted N3 refused (content) 5.5 ms 3.6/25 ms ok"
    assert row["workload/late", "b2"] == ("workload/late b2 late refused responses[0] status:coverage-never-established "
                                          "N3 refused (missing, values) 5.5 ms 3.4/25 ms ok")
    assert lines[35:] == [
        "",
        "answers: honest [16, 16] equal the reference; flip-bit [16, 16] equal the reference; late none",
        "worst proxy tick: 4.4 ms late against 25 ms of slack (mockdc-salt b2), 0 ticks past the slack in the night",
        "steps: " + ", ".join(f"{s} 0" for s in EXITS),
        "every verdict is the expected one"]


def test_a_differing_verdict_a_failed_step_a_missing_summary_or_a_late_tick_fails_the_night(run, tmp_path):
    night(tmp_path)
    edit(tmp_path / "mockdc" / "summary.json",
         lambda s: s["blocks"]["b1"]["against_honest"].update(wire_is_honests=False))
    edit(tmp_path / "mockdc-salt" / "summary.json",
         lambda s: s["blocks"]["b4"].update(salt=SALT | {"ingress_salt_frames": [1, 1]}))
    edit(tmp_path / "fleet" / "summary.json", lambda s: s["records"].update(of=10))
    edit(tmp_path / "fleet-swapped" / "summary.json", lambda s: s.pop("records"))
    edit(tmp_path / "workload" / "honest" / "summary.json",
         lambda s: s["blocks"]["b3"]["ticks_over_slack"].update(egress=2))
    edit(tmp_path / "workload" / "flip-bit" / "summary.json",
         lambda s: (s["blocks"]["b2"]["n3"].update(violations=[]), s.update(answers=[16, 15])))
    (tmp_path / "workload" / "late" / "summary.json").unlink()
    assert run.finish(tmp_path, EXITS | {"mockdc": 1}) == 1
    summary = json.loads((tmp_path / "summary.json").read_text())
    m = summary["mismatches"]
    assert m[:3] == ["mockdc exited 1", "mockdc b1 wire_is_honests: expected True, got False",
                     "mockdc-salt b4 salt: expected {'accepted': False, 'violations': [[1, 'salt']], "
                     "'ingress_salt_frames': [1, 1], 'egress_salt_frames': 0, 'egress_holds_salt': False, "
                     "'block_received': True}, got {'accepted': True, 'violations': [], 'ingress_salt_frames': [1, 1], "
                     "'egress_salt_frames': 0, 'egress_holds_salt': False, 'block_received': True}"]
    assert m[3:] == ["fleet every port of: expected 8, got 10",
                     "fleet-swapped: no record check on every port",
                     "workload/honest b3: 2 proxy ticks past the 25 ms slack",
                     "workload/flip-bit b2 n3: expected ['content'], got []",
                     "workload/flip-bit answers: expected 'equal', got 'differ' ([16, 15] against [16, 16])",
                     "workload/late: no summary.json"]
    bad = {(run_, b) for run_, r in summary["runs"].items() for b, x in r["blocks"].items() if not x["ok"]}
    assert bad == {("mockdc", "b1"), ("mockdc-salt", "b4"), ("workload/honest", "b3"), ("workload/flip-bit", "b2")}
    assert not summary["runs"]["fleet"]["records"]["ok"] and "records" not in summary["runs"]["fleet-swapped"]
    text = (tmp_path / "summary.txt").read_text()
    assert "workload/late" not in text.split("\n\n")[0] and text.count("MISMATCH\n") == 5
    assert ["fleet-swapped", "all"] not in [line.split()[:2] for line in text.splitlines()]
    assert text.endswith("9 MISMATCHES:\n" + "".join(f"  {x}\n" for x in m))
