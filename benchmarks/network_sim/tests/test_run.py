"""The night's summary and its expected verdicts (`run.py`), on summaries in the shape mockdc and lab_run write them,
with their recorded runs' verdicts (`r20261018-100310-685b`, and `r20261018-084303-dae6` for the workload): those are
the expected ones, and a differing verdict or answer, a step that exited nonzero, a missing summary or a proxy tick past
its slack each fails the night, named in `summary.txt`."""

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
EXITS = {"demo": 0, "lab --blocks 5": 0, "mockdc": 0, "mockdc-salt": 0, "lab default": 0, "python": 0,
         "workload/honest": 0, "workload/flip-bit": 0, "workload/late": 0}


@pytest.fixture(scope="module")
def run():
    spec = importlib.util.spec_from_file_location("network_sim_run", HERE / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rows(*refused) -> list[list]:
    bad = {(link, w): (raw, com) for link, w, raw, com in refused}
    return [[link, w, *bad.get((link, w), (None, None))] for link in ("responses", "requests") for w in range(3)]


def timing(late: float, over: int = 0) -> dict:
    return {"slack_ms": 25.0, "tick_late_ms_max": {"egress": late, "ingress": late - 0.06},
            "ticks_over_slack": {"egress": over, "ingress": 0}}


def account(advice: float, refused: float = 0.0, serviced: float = 0.0) -> dict:
    return {"sound": not refused, "covert_bits": advice + refused + serviced, "timing_bits": advice,
            "refused_bits": refused, "serviced_bits": serviced, "carried_bits": HELD}


def mockdc_block(strategy: str, verdicts: list, refused: list, acc: dict, late: float, **more) -> dict:
    return {"strategy": strategy, "accepted": all(r[2] is None for r in verdicts), "verdicts": verdicts,
            "datacenter": {"accepted": not refused, "refused": refused, "account": acc}} | timing(late) | more


def mockdc_summary(salt: bool) -> dict:
    def out(b: str, w: int) -> list[str]:
        return [f"port {b}.cert:out [{w * 6000}, {(w + 1) * 6000}): {why}" for why in CHEATED]

    down = [f"port b4.cert:{d} [6000, 12000): no record: the certifier was down or its record was withheld"
            for d in ("out", "in")]
    blocks = {
        "b0": mockdc_block("honest", rows(), [], account(24.0), 4.177),
        "b1": mockdc_block("bits", rows(), [], account(24.0), 3.507, against_honest={"wire_is_honests": True}),
        "b2": mockdc_block("late", rows(["responses", 0, COVERAGE, COVERAGE]), out("b2", 0), account(20.0, LINE), 3.483,
                           against_honest={"wire_is_honests": False}),
        "b3": mockdc_block("extra", rows(["responses", 2, "mismatch", "mismatch"]), out("b3", 2), account(20.0, LINE),
                           3.397, against_honest={"wire_is_honests": False}),
        "b4": mockdc_block("maintenance", rows(["responses", 1, "no-verdict", "no-verdict"],
                                               ["requests", 1, "no-verdict", "no-verdict"]),
                           [], account(16.0, serviced=HELD), 3.989),
    }
    blocks["b4"]["datacenter"]["unplanned"] = {"accepted": False, "refused": down, "account": account(16.0, 2 * LINE)}
    if salt:
        for b, x in blocks.items():
            x["salt"] = SALT if b != "b4" else SALT | {"accepted": False, "violations": [[1, "salt"]],
                                                        "ingress_salt_frames": [1, 1]}
    return {"declaration": "leaf_spine", "plan": {"epoch": [0, 18000], "maintenance": [["b4", [6000, 12000]]]},
            "blocks": blocks, "exit": {b: {"proxy": 0, "peer": 0, "server": 0} for b in blocks}}


def workload_block(late: float, tamper: str | None = None, verdicts: list | None = None, n3: list[str] = ()) -> dict:
    verdicts = verdicts or rows()
    return {"tamper": tamper, "certifier": {"accepted": all(r[2] is None for r in verdicts), "verdicts": verdicts},
            "n3": {"accepted": not n3, "violations": list(n3)}} | timing(late)


def starved(b: str, first: int) -> list[str]:
    return ["values " + b + ": the unit committed no values",
            f"missing {b}:responses: 1 declared events, the first at position {first}",
            f"missing {b}:requests: the request of session {b}.1"]


def workload_summaries() -> dict[str, dict]:
    answers = {"answers": [16, 16], "reference": [16, 16]}
    return {
        "honest": {"blocks": {"b0": workload_block(3.645), "b1": workload_block(4.062), "b2": workload_block(3.704),
                              "b3": workload_block(2.904)}, "n3": {"ok": True, "codes": {}, "violations": None}} | answers,
        "flip-bit": {"blocks": {
            "b0": workload_block(3.435),
            "b1": workload_block(3.509, "flip-bit", n3=["content b1:responses: the event at position 0 (message 4) isn't "
                                                        "the declared one"]),
            "b2": workload_block(3.533, n3=["content b2:requests: the request of session b2.0 isn't the declared one"]),
            "b3": workload_block(2.035)}, "n3": {"ok": False, "codes": {"content": 2}, "violations": None}} | answers,
        "late": {"blocks": {
            "b0": workload_block(3.386, n3=starved("b0", 6)), "b1": workload_block(2.442, n3=starved("b1", 1)),
            "b2": workload_block(2.132, "late", rows(["responses", 0, COVERAGE, COVERAGE]), starved("b2", 0)),
            "b3": workload_block(2.803, n3=starved("b3", 0))},
            "n3": {"ok": False, "codes": {"missing": 9, "values": 4}, "violations": None},
            "answers": None, "reference": [16, 16]},
    }


def night(out: Path) -> None:
    for name, salt in (("mockdc", False), ("mockdc-salt", True)):
        (out / name).mkdir(parents=True)
        (out / name / "summary.json").write_text(json.dumps(mockdc_summary(salt)))
    for name, s in workload_summaries().items():
        (out / "workload" / name).mkdir(parents=True)
        (out / "workload" / name / "summary.json").write_text(json.dumps(s))


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
    assert summary["worst_tick"] == {"late_ms": 4.177, "slack_ms": 25.0, "over": 0, "at": "mockdc b0"}
    b4 = summary["runs"]["mockdc"]["blocks"]["b4"]
    assert b4["verdicts"]["charged"] == ["timing", "serviced"] and b4["verdicts"]["unplanned_charged"] == ["timing", "refused"]
    assert b4["account"]["covert_bits"] == 16.0 + HELD and b4["unplanned_account"]["covert_bits"] == 16.0 + 2 * LINE
    text = (tmp_path / "summary.txt").read_text()
    assert text == capsys.readouterr().out
    lines = text.splitlines()
    assert lines[0].split() == ["run", "block", "case", "certifier", "audit", "records", "/", "N3", "covert", "budget",
                                "(bits)", "worst", "tick"]
    assert len(lines) == 1 + 5 + 5 + 3 * 4 + 5 and all(line.endswith(" ok") for line in lines[1:23])
    row = {tuple(line.split()[:2]): " ".join(line.split()) for line in lines[1:23]}
    assert row["mockdc", "b0"] == "mockdc b0 honest accepted records accepted 24 sound 4.2/25 ms ok"
    assert row["mockdc", "b2"] == ("mockdc b2 late refused responses[0] status:coverage-never-established records "
                                   "refused out [0, 6000) 2.4e+12 NOT sound 3.5/25 ms ok")
    assert row["mockdc", "b4"] == ("mockdc b4 maintenance refused responses[1] no-verdict, requests[1] no-verdict records "
                                   "accepted; unplanned refused out, in [6000, 12000) 2.16e+13 sound; unplanned 4.8e+12 "
                                   "NOT sound 4.0/25 ms ok")
    assert row["mockdc-salt", "b1"] == "mockdc-salt b1 bits accepted; salt accepted records accepted 24 sound 3.5/25 ms ok"
    assert "no-verdict; salt refused [1] records accepted;" in row["mockdc-salt", "b4"]
    assert row["workload/flip-bit", "b2"] == "workload/flip-bit b2 - accepted N3 refused (content) 3.5/25 ms ok"
    assert row["workload/late", "b2"] == ("workload/late b2 late refused responses[0] status:coverage-never-established "
                                          "N3 refused (missing, values) 2.1/25 ms ok")
    assert lines[24:] == [
        "answers: honest [16, 16] equal the reference; flip-bit [16, 16] equal the reference; late none",
        "worst proxy tick: 4.2 ms late against 25 ms of slack (mockdc b0), 0 ticks past the slack in the night",
        "steps: " + ", ".join(f"{s} 0" for s in EXITS),
        "every verdict is the expected one"]


def test_a_differing_verdict_a_failed_step_a_missing_summary_or_a_late_tick_fails_the_night(run, tmp_path):
    night(tmp_path)
    edit(tmp_path / "mockdc" / "summary.json",
         lambda s: s["blocks"]["b1"]["against_honest"].update(wire_is_honests=False))
    edit(tmp_path / "mockdc-salt" / "summary.json",
         lambda s: s["blocks"]["b4"].update(salt=SALT | {"ingress_salt_frames": [1, 1]}))
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
    assert m[3:] == ["workload/honest b3: 2 proxy ticks past the 25 ms slack",
                     "workload/flip-bit b2 n3: expected ['content'], got []",
                     "workload/flip-bit answers: expected 'equal', got 'differ' ([16, 15] against [16, 16])",
                     "workload/late: no summary.json"]
    bad = {(run_, b) for run_, r in summary["runs"].items() for b, x in r["blocks"].items() if not x["ok"]}
    assert bad == {("mockdc", "b1"), ("mockdc-salt", "b4"), ("workload/honest", "b3"), ("workload/flip-bit", "b2")}
    text = (tmp_path / "summary.txt").read_text()
    assert "workload/late" not in text.split("\n\n")[0] and text.count("MISMATCH\n") == 4
    assert text.endswith("7 MISMATCHES:\n" + "".join(f"  {x}\n" for x in m))
