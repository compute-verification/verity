from __future__ import annotations

import json
from pathlib import Path

import pytest
import run as R

CARD = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
UNMET = "challenger shares node-a with the developer's workers: Faithful and Secret unmet"
ON = {"host": "node-a", "served_node": True, "from": "the served run's own files"}
BASE = f"Llama-3.2-1B-INT8 on {CARD}, remote"
MEMBER = f"{BASE}, coins simulated ({UNMET}), composed on node-a, the workload's node"
STATED = {"composed_on": ON}


@pytest.fixture
def here(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r-app")
    (tmp_path / "run").mkdir()
    monkeypatch.setattr(R.kit, "host_stamp", lambda: {"card": CARD, "commit": "0" * 40})
    ticks = iter((100.0, 4600.0))
    monkeypatch.setattr(R, "clock", lambda: next(ticks))
    config = tmp_path / "remote.toml"
    config.write_text('[workload.params]\nmodel = "verity-int8/Llama-3.2-1B-INT8"\n')
    return config


def stand_in(monkeypatch: pytest.MonkeyPatch, lines: list[dict] | None, rc: int,
             report: dict | None = STATED) -> list[list[str]]:
    seen = []

    def application(argv: list[str]) -> int:
        seen.append(argv)
        state = Path(argv[argv.index("--state") + 1])
        if lines is not None:
            (state / "record").mkdir(parents=True)
            (state / "record" / "verdicts.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
        if report is not None:
            (state / R.REPORT).parent.mkdir(parents=True, exist_ok=True)
            (state / R.REPORT).write_text(json.dumps(report))
        return rc

    monkeypatch.setattr(R, "application", application)
    return seen


def line(verdict: str = "accept", coins: str | None = "simulated", simulated: list[str] | None = None,
         **components: str) -> dict:
    x = {"block": 0, "components": {"spot": "accept", "pearl-plus": "accept", "inference-only": "accept", **components},
         "product": "verity/remote/v1", "verdict": verdict, "simulated": [UNMET] if simulated is None else simulated}
    return x if coins is None else {**x, "coins": coins}


def measurements(config: Path) -> dict[str, dict]:
    doc = json.loads((config.parent / "run" / "result.json").read_text())
    return {m["name"]: m for m in doc["measurements"]}


def test_an_accepting_run_writes_its_count_and_minutes_under_its_member(here, monkeypatch):
    seen = stand_in(monkeypatch, [line()], 0)
    assert R.main([str(here), "--periods", "1"]) == 0
    assert seen[0][0] == str(here) and seen[0][1] == "--state" and seen[0][3:] == ["--periods", "1"]
    m = measurements(here)
    assert (m[R.ACCEPTED]["value"], m[R.ACCEPTED]["of"], m[R.ACCEPTED]["unit"]) == (4, 4, "verdicts")
    assert (m[R.MINUTES]["value"], m[R.MINUTES]["unit"]) == (75.0, "minutes")
    assert {x["member"] for x in m.values()} == {MEMBER}
    assert [x["composed_on"] for x in m.values()] == [ON, ON]
    assert {x["mark"] for x in m.values()} == {"measured"}
    doc = json.loads((here.parent / "run" / "out" / "application_run.json").read_text())
    assert (doc["accepted"], doc["of"], doc["minutes"], doc["verdicts"], doc["composed_on"]) == (4, 4, 75.0, [line()], ON)


def test_a_run_whose_verdicts_do_not_all_accept_has_its_count_and_no_minutes(here, monkeypatch):
    stand_in(monkeypatch, [line("reject", **{"pearl-plus": "reject"})], 1)
    assert R.main([str(here)]) == 1
    m = measurements(here)
    assert (m[R.ACCEPTED]["value"], m[R.ACCEPTED]["of"]) == (2, 4)
    assert m[R.MINUTES]["value"] is None
    assert m[R.MINUTES]["why"] == "2 of 4 verdicts accept: only a run whose verdicts all accept has minutes"


def test_a_run_with_no_verdict_has_no_number(here, monkeypatch):
    stand_in(monkeypatch, None, 2, None)
    assert R.main([str(here)]) == 2
    m = measurements(here)
    assert [(x["value"], x["why"]) for x in m.values()] == [(None, "verity run exited 2 with no verdict")] * 2
    assert {x["member"] for x in m.values()} == {f"{BASE}, no verdict, so no coins, composed on: not stated"}
    assert [x["composed_on"] for x in m.values()] == [None, None]


@pytest.mark.parametrize("lines, said", [
    ([line(coins="development", simulated=[])], "coins development"),
    ([line(), line(coins="development", simulated=[])], "coins disagree (development, simulated)"),
    ([line(), line(coins=None)], "coins not stated"),
    ([line(), line(simulated=["another premise"])], "coins simulated, simulated premises disagree"),
    ([{k: v for k, v in line().items() if k != "simulated"}], "coins simulated, simulated premises not stated"),
])
def test_the_member_names_the_coins_the_record_states_and_says_where_it_states_none_or_disagrees(here, monkeypatch,
                                                                                               lines, said):
    stand_in(monkeypatch, lines, 0)
    R.main([str(here)])
    assert {x["member"] for x in measurements(here).values()} == {
        f"{BASE}, {said}, composed on node-a, the workload's node"}


@pytest.mark.parametrize("report, said", [
    ({"composed_on": {**ON, "host": "auditor-1", "served_node": False}}, "composed on auditor-1, not the workload's node"),
    (None, "composed on: not stated"),
    ({"protocols": {}}, "composed on: not stated"),
    ({"composed_on": {"served_node": False}}, "composed on: host not stated"),
    ({"composed_on": {"host": "node-a"}}, "composed on node-a, whether it is the workload's node not stated"),
])
def test_the_member_says_where_the_record_was_composed_as_the_record_states_it(here, monkeypatch, report, said):
    stand_in(monkeypatch, [line()], 0, report)
    R.main([str(here)])
    m = measurements(here)
    assert {x["member"] for x in m.values()} == {f"{BASE}, coins simulated ({UNMET}), {said}"}
    assert [x["composed_on"] for x in m.values()] == [(report or {}).get("composed_on")] * 2


def test_the_state_is_the_runs_own(here, monkeypatch, capsys):
    stand_in(monkeypatch, [line()], 0)
    with pytest.raises(SystemExit):
        R.main([str(here), "--state", "/elsewhere"])
    assert "--state is the run's own" in capsys.readouterr().err
