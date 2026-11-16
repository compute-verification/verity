"""The harness kit's writer: the result, its measurements and its validation, which research's own check takes, with what
it refuses."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from kit_standins import RUN, SHA, _result
from kit_standins import no_gpu as no_gpu
from verity_bench import kit

# ---- the writer --------------------------------------------------------------------------------------------------------------


def test_the_writer_appends_measurements_in_the_form_a_reader_of_numbers_takes(tmp_path, no_gpu):
    from research import result as research_result

    out = tmp_path / "run" / "out"
    w = kit.writer(out)
    stamp = kit.host_stamp()
    first = w.measurement("served_tok_s", 4120.5, "tok/s", mark="measured")
    assert first == {"name": "served_tok_s", "value": 4120.5, "unit": "tok/s", "mark": "measured", "commit": SHA, "run": RUN,
                     "ci95": None, "target": None, "verdict": None, "inputs": ["measured"], "host": stamp}
    assert _result(out) == {"schema": kit.RESULT_SCHEMA, "run_id": RUN, "measurements": [first]}
    second = kit.writer(out).measurement("serving_cost", 0.031, "ratio", interval=(0.029, 0.034), target=0.05,
                                         verdict="within target", inputs=("served_tok_s", "stock_tok_s"), mark="measured")
    assert (second["ci95"], second["target"], second["verdict"]) == ([0.029, 0.034], 0.05, "within target")
    assert second["inputs"] == ["measured", "served_tok_s", "stock_tok_s"] and second["host"]["card"] == "none"
    w.measurement("reps", 20, "", inputs="served_tok_s", mark="measured")
    w.measurement("stock_tok_s", None, "tok/s", target=4000, verdict="cannot tell", mark="measured")
    doc = _result(out)
    assert [(m["name"], m["value"]) for m in doc["measurements"]] == [
        ("served_tok_s", 4120.5), ("serving_cost", 0.031), ("reps", 20), ("stock_tok_s", None)]
    assert doc["measurements"][2]["inputs"] == ["measured", "served_tok_s"]
    assert all(m.keys() >= {"value", "unit", "commit", "run"} and m["host"] == stamp for m in doc["measurements"])
    check = research_result.check_doc(out, doc, RUN)
    assert check.valid and check.errors == [] and list(out.iterdir()) == [out / kit.RESULT]


def test_the_writer_appends_to_the_result_a_runs_validation_is_in(tmp_path, no_gpu):
    envelope = {"schema": kit.RESULT_SCHEMA, "run_id": RUN, "workload_fingerprint": {"door": "run"},
                "validation": {"status": "passed", "checker": "a benchmark", "detail": "", "evidence": []},
                "measurement_files": []}
    (tmp_path / kit.RESULT).write_text(json.dumps(envelope))
    kit.writer(tmp_path).measurement("gamma", 0.5, "ratio", mark="measured")
    doc = _result(tmp_path)
    assert {k: doc[k] for k in envelope} == envelope and [m["name"] for m in doc["measurements"]] == ["gamma"]


def test_the_writer_writes_the_validation_block_a_benchmarks_own_result_holds(tmp_path, no_gpu):
    """The block is, key for key, the one a benchmark that writes its run's whole result itself puts under `validation`
    for the same gates: `passed` beside the evidence's path relative to the run's directory, and `failed` with none."""

    from research import result as research_result

    run_dir = tmp_path / "run"
    (run_dir / "out").mkdir(parents=True)
    (run_dir / "out" / "census.json").write_text("{}")
    detail = "worst in-domain share 0.00120 (threshold 0.0025)"
    block = kit.writer(run_dir).validation("passed", "pearlc-census", detail, ["out/census.json"])
    assert block == {"status": "passed", "checker": "pearlc-census", "detail": detail, "evidence": ["out/census.json"]}
    assert list(block) == ["status", "checker", "detail", "evidence"]
    assert _result(run_dir) == {"schema": kit.RESULT_SCHEMA, "run_id": RUN, "measurements": [], "validation": block}
    check = research_result.check(run_dir, RUN)
    assert check.present and check.valid and check.errors == [] and check.validation_status == "passed"
    assert sorted(run_dir.iterdir()) == [run_dir / "out", run_dir / kit.RESULT]
    other = tmp_path / "other"
    assert kit.writer(other).validation("failed", "pearlc-k256-gate", "2 of 4 parts hold", evidence=()) == {
        "status": "failed", "checker": "pearlc-k256-gate", "detail": "2 of 4 parts hold", "evidence": []}
    check = research_result.check(other, RUN)
    assert check.valid and check.validation_status == "failed" and check.doc == _result(other)
    # the status is the whole guard: a detail says what passed or failed in the gates' own words, and one may say nothing
    for i, said in enumerate(("", "3/4 gates passed", "every part passes", "failing: the k256 part", "FAIL: gate 3",
                              "worst accepted fragment share 0.0012", "12 inputs, 0 rejected tiles", "1 failure")):
        assert kit.writer(tmp_path / str(i)).validation("failed", "", said, ("a/b.json",)) == {
            "status": "failed", "checker": "", "detail": said, "evidence": ["a/b.json"]}


def test_a_validation_and_the_measurements_share_a_result_in_either_order(tmp_path, no_gpu):
    from research import result as research_result

    first, second = tmp_path / "validation-first", tmp_path / "measurements-first"
    w = kit.writer(first)
    block = w.validation("passed", "a benchmark", "4 of 4 gates hold", ["out/gates.json"])
    gamma = w.measurement("gamma", 0.5, "ratio", target=1.0, verdict="within target", mark="measured")
    reps = kit.writer(first).measurement("reps", 20, "", mark="measured")
    assert _result(first) == {"schema": kit.RESULT_SCHEMA, "run_id": RUN, "measurements": [gamma, reps], "validation": block}
    w = kit.writer(second)
    w.measurement("gamma", 0.5, "ratio", target=1.0, verdict="within target", mark="measured")
    kit.writer(second).validation("passed", "a benchmark", "4 of 4 gates hold", ["out/gates.json"])
    w.measurement("reps", 20, "", mark="measured")
    assert _result(second) == _result(first)
    for out in (first, second):
        check = research_result.check(out, RUN)
        assert check.valid and check.errors == [] and check.validation_status == "passed"
    # into a result the run wrote without one, beside what it already holds
    envelope = {"schema": kit.RESULT_SCHEMA, "run_id": RUN, "workload_fingerprint": {"door": "run"},
                "measurements": [{"name": "wall", "value": 1.5, "unit": "s"}], "measurement_files": []}
    third = tmp_path / "enveloped"
    third.mkdir()
    (third / kit.RESULT).write_text(json.dumps(envelope))
    kit.writer(third).validation("failed", "a benchmark", "1 of 4 gates hold")
    assert _result(third) == envelope | {"validation": {"status": "failed", "checker": "a benchmark",
                                                       "detail": "1 of 4 gates hold", "evidence": []}}


def test_under_a_research_run_the_result_is_the_run_directorys_where_research_reads_it(tmp_path, no_gpu, monkeypatch):
    """`research` reads a run's result from `$RESEARCH_RUN_DIR/result.json` alone, so a result under the run's `out/` is
    absent to it: every writer of a run, whatever its `out`, appends to the one file in the run's directory."""

    from research import result as research_result

    run_dir, elsewhere = tmp_path / "run", tmp_path / "elsewhere"
    out = run_dir / "out"
    out.mkdir(parents=True)
    # outside a run (the variable unset, or empty) the result is `out`'s, and research finds none in the run's directory
    kit.writer(out).measurement("wall", 1.5, "s", mark="measured")
    assert [m["name"] for m in _result(out)["measurements"]] == ["wall"] and list(run_dir.iterdir()) == [out]
    assert not research_result.check(run_dir, RUN).present
    monkeypatch.setenv("RESEARCH_RUN_DIR", "")
    assert kit.writer(elsewhere).out == elsewhere
    (out / kit.RESULT).unlink()
    # under one it is the run directory's, whatever `out` is, and the evidence is relative to that directory
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(run_dir))
    first, second = kit.writer(out), kit.writer(elsewhere)
    assert first.out == second.out == run_dir
    wall = first.measurement("wall", 1.5, "s", mark="measured")
    block = second.validation("passed", "a benchmark", "4 of 4 gates passed", ["out/gates.json"])
    gamma = second.measurement("gamma", 0.5, "ratio", mark="measured")
    reps = kit.writer(run_dir).measurement("reps", 20, "", mark="measured")
    assert _result(run_dir) == {"schema": kit.RESULT_SCHEMA, "run_id": RUN, "measurements": [wall, gamma, reps],
                                "validation": block}
    assert sorted(run_dir.iterdir()) == [out, run_dir / kit.RESULT] and list(out.iterdir()) == [] and not elsewhere.exists()
    check = research_result.check(run_dir, RUN)
    assert check.present and check.valid and check.errors == [] and check.validation_status == "passed"
    assert check.doc == _result(run_dir)
    # one result a run: its writers hold each other to one measurement a name and one validation
    with pytest.raises(ValueError, match="already holds a measurement named wall"):
        second.measurement("wall", 2.0, "s", mark="measured")
    with pytest.raises(ValueError, match="already holds a validation"):
        first.validation("failed", "another checker", "")
    assert _result(run_dir) == check.doc


def test_the_writer_refuses_a_second_validation_and_one_that_isnt_a_runs_own_gates(tmp_path, no_gpu, monkeypatch):
    blank = tmp_path / "blank"
    w = kit.writer(blank)
    ok = {"status": "passed", "checker": "a benchmark", "detail": "4 of 4 gates hold", "evidence": ["out/gates.json"]}
    for status in ("not_run", "PASS", "Passed", "accepted", "sealed", "ok", "", True, None, 1):
        with pytest.raises(ValueError, match="status is 'passed' or 'failed', whether the run's own gates held, not .* a "
                                             "verdict of record or a seal is never a run's validation"):
            w.validation(**ok | {"status": status})
    for what in ("checker", "detail"):
        for text in (None, 3, b"bytes", ["4 of 4 gates hold"]):
            with pytest.raises(ValueError, match=f"a validation's {what} is a string"):
                w.validation(**ok | {what: text})
    for evidence in ("out/gates.json", b"out/gates.json", None, 5, {"out/gates.json"}, iter(["out/gates.json"])):
        with pytest.raises(ValueError, match="evidence is a list of paths relative to"):
            w.validation(**ok | {"evidence": evidence})
    for rel in (3, None, "", "/abs/gates.json", "../gates.json", "out/../../gates.json", Path("out/gates.json")):
        with pytest.raises(ValueError, match="evidence is a list of paths relative to .* and inside it, each a string"):
            w.validation(**ok | {"evidence": ["out/gates.json", rel]})
    assert not blank.exists()
    # one validation a result, whoever wrote the first
    w.validation(**ok)
    w.measurement("gamma", 0.5, "ratio", mark="measured")
    before = (blank / kit.RESULT).read_bytes()
    for again in (w, kit.writer(blank)):
        with pytest.raises(ValueError, match="already holds a validation: a result holds one, the run's own"):
            again.validation("failed", "another checker", "1 of 4 gates hold")
    assert (blank / kit.RESULT).read_bytes() == before and list(blank.iterdir()) == [blank / kit.RESULT]
    for i, held in enumerate(({"status": "failed", "checker": "a benchmark", "detail": "", "evidence": []},
                              {"status": "not_run", "checker": None, "detail": None, "evidence": []})):
        enveloped = tmp_path / str(i)
        enveloped.mkdir()
        text = json.dumps({"schema": kit.RESULT_SCHEMA, "run_id": RUN, "validation": held})
        (enveloped / kit.RESULT).write_text(text)
        with pytest.raises(ValueError, match="already holds a validation"):
            kit.writer(enveloped).validation(**ok)
        assert (enveloped / kit.RESULT).read_text() == text
    # and only into this run's result
    monkeypatch.setenv("RESEARCH_RUN_ID", "run-2")
    with pytest.raises(ValueError, match="holds run run-1's measurements, and this run is run-2"):
        kit.writer(blank).validation(**ok)
    assert (blank / kit.RESULT).read_bytes() == before


def test_the_writer_refuses_a_verdict_of_record_and_a_seal(tmp_path, no_gpu):
    w = kit.writer(tmp_path)
    w.measurement("kept", 1.0, "s", target=2.0, verdict="within target", mark="measured")
    before = (tmp_path / kit.RESULT).read_bytes()
    for verdict in ("PASS", "FAIL", "accept", "Accepted", "rejected", "passed: within target", "the verifier accepts",
                    "sealed", "seal", "passing", "Failing", "rejection", "the proof's acceptance", "a failure", "accepting",
                    "rejecting: over target", "sealing"):
        with pytest.raises(ValueError, match="reads as a verdict of record or a seal .* nothing under benchmarks/ writes"):
            w.measurement("gate", 1.0, "s", target=2.0, verdict=verdict, mark="measured")
    with pytest.raises(ValueError, match="'within target' has no target"):
        w.measurement("gate", 1.0, "s", verdict="within target", mark="measured")
    for verdict in (True, 1, "", "  "):
        with pytest.raises(ValueError, match="says in words where its value stands against its target"):
            w.measurement("gate", 1.0, "s", target=2.0, verdict=verdict, mark="measured")
    for value in (True, "accept", "PASS", "sha512:" + "0" * 128, b"\x00", {"verdict": "accept"}, math.nan, math.inf):
        with pytest.raises(ValueError, match="value is a finite number.*a verdict of record or a seal is never a measurement"):
            w.measurement("gate", value, "verdict", mark="measured")
    assert (tmp_path / kit.RESULT).read_bytes() == before and list(tmp_path.iterdir()) == [tmp_path / kit.RESULT]
    # a target check that only holds one of those words inside another is no verdict of record
    for verdict in ("within target", "over target", "meets", "misses", "cannot tell", "surpasses target", "acceptable"):
        w.measurement(f"gate {verdict}", 1.0, "s", target=2.0, verdict=verdict, mark="measured")


def test_the_writer_refuses_what_isnt_a_measurement_of_this_run(tmp_path, no_gpu, monkeypatch):
    w = kit.writer(tmp_path)
    w.measurement("a", 1, "s", mark="measured")
    with pytest.raises(ValueError, match="already holds a measurement named a"):
        kit.writer(tmp_path).measurement("a", 2, "s", mark="measured")
    for kw, why in (({"name": ""}, "has a name"), ({"name": 3}, "has a name"), ({"unit": None}, "unit is a string"),
                    ({"inputs": ["a", 3]}, "inputs are the names"), ({"inputs": [""]}, "inputs are the names"),
                    ({"interval": (1.0,)}, r"interval is \(low, high\)"), ({"interval": (2.0, 1.0)}, r"interval is \(low, high\)"),
                    ({"interval": 0.5}, r"interval is \(low, high\)"),
                    ({"interval": (0.0, math.inf)}, "interval is a finite number"), ({"target": "low"}, "target is a finite number")):
        with pytest.raises(ValueError, match=why):
            w.measurement(**{"name": "b", "value": 1.0, "unit": "s"} | kw, mark="measured")
    assert [m["name"] for m in _result(tmp_path)["measurements"]] == ["a"]
    monkeypatch.setenv("RESEARCH_RUN_ID", "run-2")
    with pytest.raises(ValueError, match="holds run run-1's measurements, and this run is run-2"):
        kit.writer(tmp_path).measurement("b", 1, "s", mark="measured")
    for i, text in enumerate(('{"schema": "verity-bench/overhead/v1"}', "[]", "not json")):
        other = tmp_path / str(i)
        other.mkdir()
        (other / kit.RESULT).write_text(text)
        with pytest.raises(ValueError, match=f"isn't (a {kit.RESULT_SCHEMA} result|JSON)"):
            kit.writer(other).measurement("b", 1, "s", mark="measured")
        assert (other / kit.RESULT).read_text() == text


def test_a_measurement_with_no_value_says_why_and_only_it_holds_a_why(tmp_path, no_gpu):
    w = kit.writer(tmp_path)
    absent = w.measurement("capacity", None, "MAC", why="the record holds no registered device", mark="measured")
    assert (absent["value"], absent["why"]) == (None, "the record holds no registered device")
    assert "why" not in w.measurement("verified", 5, "MAC", mark="measured")
    assert "why" not in w.measurement("free", None, "MAC", mark="modeled")
    for value, why in ((1, "a value beside it"), (None, ""), (None, "  "), (None, 3)):
        with pytest.raises(ValueError, match="why says in words why it has no value, beside a value of None"):
            w.measurement("gate", value, "MAC", why=why, mark="measured")
    assert [m.get("why") for m in _result(tmp_path)["measurements"]] == ["the record holds no registered device", None, None]


def test_every_measurement_carries_its_mark_as_a_field_and_first_among_its_inputs(tmp_path, no_gpu):
    """The mark is required, a value of None's too, and one of `MARKS`; the result holds it as `mark` and as the first of
    `inputs`, where the console's Metrics tab reads the first word, so a first input that already opens with it is kept
    as given and one that opens with another mark is refused."""

    from research import result as research_result

    assert kit.MARKS == ("measured", "modeled", "partial")
    w = kit.writer(tmp_path)
    timed = w.measurement("wall", 1.5, "s", inputs=["measured_tok_s"], mark="measured")
    said = w.measurement("total", 2.0, "s", inputs=["modeled, not measured: one block type", "r1"], mark="modeled")
    part = w.measurement("proving", 3.0, "s", inputs="partial: 2 block type(s) with no figure", mark="partial")
    absent = w.measurement("capacity", None, "MAC", why="the record holds no registered device", mark="modeled")
    assert [(m["mark"], m["inputs"]) for m in (timed, said, part, absent)] == [
        ("measured", ["measured", "measured_tok_s"]), ("modeled", ["modeled, not measured: one block type", "r1"]),
        ("partial", ["partial: 2 block type(s) with no figure"]), ("modeled", ["modeled"])]
    doc = _result(tmp_path)
    assert [m["mark"] for m in doc["measurements"]] == ["measured", "modeled", "partial", "modeled"]
    assert research_result.check_doc(tmp_path, doc, RUN).valid
    before = (tmp_path / kit.RESULT).read_bytes()
    for given in ({}, {"why": "no value to give"}):
        with pytest.raises(ValueError, match=r"gate's mark is one of .*, not None: pass mark="):
            w.measurement("gate", None, "s", **given)
    for mark in (None, "", "Measured", "modelled", "estimated", "measured: a timed run", 1, True, ("measured",)):
        with pytest.raises(ValueError, match="gate's mark is one of 'measured', 'modeled', 'partial', how its value was "
                                             r"had \(a value of None takes the one it would have had\), not .*: pass mark="):
            w.measurement("gate", None if mark is None else 1.0, "s", mark=mark)
    for first, mark in (("modeled: scaled from one machine", "measured"), ("measured", "partial"), ("partial: one", "modeled")):
        with pytest.raises(ValueError, match="gate's first input opens with the mark .* give the one mark it has"):
            w.measurement("gate", 1.0, "s", inputs=[first], mark=mark)
    assert (tmp_path / kit.RESULT).read_bytes() == before


def test_a_measurement_over_a_period_and_a_scope_holds_each_as_its_own_field_and_one_a_name_period_and_over(tmp_path, no_gpu):
    """`period` and `over` are fields of the measurement, after its mark, where given and only there; each is one of the
    kit's words or refused with the fix. A result holds one measurement a name, period and `over`: the same name over
    another period or scope is another measurement, and a second of one is refused by all three."""

    from research import result as research_result

    assert kit.PERIODS == ("phase", "week", "month", "year") and kit.OVER == ("world", "datacenter")
    w, stamp = kit.writer(tmp_path), kit.host_stamp()
    year = w.measurement("gb", 14240.0, "GB a world year", mark="modeled", period="year", over="world")
    assert year == {"name": "gb", "value": 14240.0, "unit": "GB a world year", "mark": "modeled", "period": "year",
                    "over": "world", "commit": SHA, "run": RUN, "ci95": None, "target": None, "verdict": None,
                    "inputs": ["modeled"], "host": stamp}
    week = w.measurement("gb", 273.1, "GB a datacenter week", mark="modeled", period="week", over="datacenter")
    phase = w.measurement("gb", None, "GB a phase", mark="modeled", period="phase", why="no phase's rows")
    world = w.measurement("gb", 3.0, "GB", mark="modeled", over="world")
    bare = w.measurement("gb", 2.0, "GB", mark="modeled")
    assert [(m.get("period"), m.get("over")) for m in (week, phase, world, bare)] == [
        ("week", "datacenter"), ("phase", None), (None, "world"), (None, None)]
    assert list(phase)[:5] == ["name", "value", "unit", "mark", "period"] and "over" not in phase
    assert "period" not in world and not bare.keys() & {"period", "over"}
    doc = _result(tmp_path)
    assert doc["measurements"] == [year, week, phase, world, bare]
    assert research_result.check_doc(tmp_path, doc, RUN).valid
    before = (tmp_path / kit.RESULT).read_bytes()
    for held in (year, week, phase, world, bare):
        with pytest.raises(ValueError, match=f"already holds a measurement named gb at period={held.get('period')!r}, "
                                             f"over={held.get('over')!r} and member=None: a result holds one a name, "
                                             "period, over and member"):
            kit.writer(tmp_path).measurement("gb", 1.0, "GB", mark="modeled", period=held.get("period"),
                                             over=held.get("over"))
    for period in ("day", "Year", "years", "", "world", 7, True):
        with pytest.raises(ValueError, match=r"gate's period is one of 'phase', 'week', 'month', 'year', the monitoring "
                                             r"period its value is over, or None where it is over none, not .*: pass "
                                             r"period=\.\.\. with one of them"):
            w.measurement("gate", 1.0, "GB", mark="modeled", period=period)
    for over in ("dc", "World", "datacentre", "", "year", 1):
        with pytest.raises(ValueError, match=r"gate's over is one of 'world', 'datacenter', the compute its value is over, "
                                             r"or None where it is over no such scope, not .*: pass over=\.\.\. with one "
                                             "of them"):
            w.measurement("gate", 1.0, "GB", mark="modeled", over=over)
    assert (tmp_path / kit.RESULT).read_bytes() == before


def test_a_measurement_of_one_member_of_a_set_holds_it_as_its_own_field_and_one_a_name_and_member(tmp_path, no_gpu):
    """`member` is a field of the measurement, after its spans, where given and only there, a non-empty string. The same
    name for another member is another measurement, and a second of one name and member is refused."""

    from research import result as research_result

    w = kit.writer(tmp_path)
    a = w.measurement("gap", 2, "premises", target=0, mark="measured", member="Pouw.Guarantees.Sound")
    b = w.measurement("gap", 0, "premises", target=0, mark="measured", member="Pous.Guarantees.Sound")
    yearly = w.measurement("gap", 1, "premises", mark="measured", period="year", member="Pouw.Guarantees.Sound")
    bare = w.measurement("gap", 3, "premises", mark="measured")
    assert list(a)[:5] == ["name", "value", "unit", "mark", "member"] and a["member"] == "Pouw.Guarantees.Sound"
    assert list(yearly)[4:6] == ["period", "member"] and "member" not in bare
    doc = _result(tmp_path)
    assert doc["measurements"] == [a, b, yearly, bare]
    assert research_result.check_doc(tmp_path, doc, RUN).valid
    before = (tmp_path / kit.RESULT).read_bytes()
    with pytest.raises(ValueError, match="already holds a measurement named gap at period=None, over=None and "
                                         "member='Pous.Guarantees.Sound': a result holds one a name, period, over and member"):
        kit.writer(tmp_path).measurement("gap", 1, "premises", mark="measured", member="Pous.Guarantees.Sound")
    for member in ("", "  ", 7, True, ["lean"]):
        with pytest.raises(ValueError, match=r"gate's member names which member of a set its value is for, a non-empty "
                                             r"string, or None where it is for no one member, not .*"):
            w.measurement("gate", 1, "premises", mark="measured", member=member)
    with pytest.raises(ValueError, match="'member' is a key the measurement already has a meaning for"):
        w.measurement("gate", 1, "premises", mark="measured", extra={"member": "lean"})
    assert (tmp_path / kit.RESULT).read_bytes() == before


def test_a_measurement_with_no_why_is_written_byte_for_byte_as_it_was_and_one_with_a_why_reads(tmp_path, no_gpu):
    from research import result as research_result

    w, stamp = kit.writer(tmp_path), kit.host_stamp()
    w.measurement("served_tok_s", 4120.5, "tok/s", interval=(4100, 4140), target=4000, verdict="within target", inputs="reps",
                  mark="measured")
    w.measurement("stock_tok_s", None, "tok/s", mark="measured")
    entries = [{"name": "served_tok_s", "value": 4120.5, "unit": "tok/s", "mark": "measured", "commit": SHA, "run": RUN,
                "ci95": [4100, 4140], "target": 4000, "verdict": "within target", "inputs": ["measured", "reps"], "host": stamp},
               {"name": "stock_tok_s", "value": None, "unit": "tok/s", "mark": "measured", "commit": SHA, "run": RUN,
                "ci95": None, "target": None, "verdict": None, "inputs": ["measured"], "host": stamp}]
    doc = {"schema": kit.RESULT_SCHEMA, "run_id": RUN, "measurements": entries}
    assert (tmp_path / kit.RESULT).read_text() == json.dumps(doc, indent=1) + "\n"
    said = w.measurement("capacity", None, "MAC", why="the record holds no registered device", mark="measured")
    assert list(said) == [*entries[1], "why"]
    doc["measurements"].append(said)
    assert (tmp_path / kit.RESULT).read_text() == json.dumps(doc, indent=1) + "\n"
    check = research_result.check_doc(tmp_path, doc, RUN)
    assert check.valid and check.errors == []


def test_a_measurement_carries_its_benchmarks_extra_keys_and_none_that_the_entry_already_means(tmp_path, no_gpu):
    from research import result as research_result

    w = kit.writer(tmp_path)
    held = w.measurement("share", 0.12, "of the read rate", target=0.14, verdict="within target; the verifier held no "
                         "checkpoint", mark="measured", extra={"bound": False})
    assert held["bound"] is False and list(held)[-1] == "bound"
    assert "bound" not in w.measurement("bare", 0.12, "of the read rate", mark="measured")
    for key in ("value", "verdict", "mark", "why", "period", "over", 3):
        with pytest.raises(ValueError, match="already has a meaning for"):
            w.measurement(f"clash {key}", 1, "MAC", mark="measured", extra={key: False})
    with pytest.raises(ValueError, match="'period' is a key the measurement already has a meaning for"):
        w.measurement("clash at a year", 1, "MAC", mark="measured", period="year", extra={"period": "week"})
    with pytest.raises(ValueError, match="is a JSON value"):
        w.measurement("nan", 1, "MAC", mark="measured", extra={"bound": math.nan})
    doc = _result(tmp_path)
    assert [m["name"] for m in doc["measurements"]] == ["share", "bare"]
    assert research_result.check_doc(tmp_path, doc, RUN).valid
