"""The harness kit's writer's fingerprint and files, the keys and shapes PoUW's own writer writes, with what it refuses."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from kit_standins import RUN, SHA, _result
from kit_standins import no_gpu as no_gpu
from verity_bench import kit

# ---- the fingerprint and the files ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("sha", [SHA, None])
def test_the_fingerprint_and_the_files_are_the_keys_and_shapes_pouws_own_writer_writes(tmp_path, no_gpu, monkeypatch, sha):
    """For the same gates, numbers and record, the kit's result holds the keys of `theirs`, the document PoUW's writer
    of a run's whole result writes, each the same value; the kit's measurements hold more than its name, value and
    unit, and those three are the same. `research` takes both, and reads the same fingerprint and files from each.
    PoUW's suite holds its writers to the kit's keys (tests/benchmarks/pouw/test_run_result.py)."""

    from research import result as research_result

    if sha is None:
        monkeypatch.delenv("RESEARCH_SOURCE_SHA")
    run_dir = tmp_path / "run"
    out = run_dir / "out"
    out.mkdir(parents=True)
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(run_dir))
    theirs = {"schema": "research/result/v0.1", "run_id": RUN, "workload_fingerprint": {"source_sha": sha},
              "validation": {"status": "passed", "checker": "a benchmark", "detail": "4 of 4 gates hold",
                             "evidence": ["out"]},
              "measurements": [{"name": "gamma", "value": 0.5, "unit": "ratio"},
                               {"name": "wall", "value": 1.5, "unit": "s"}],
              "measurement_files": ["out"], "artifacts": []}
    w = kit.writer(out)
    assert w.fingerprint() == {"source_sha": sha}
    assert w.measurement_file("out") == "out"
    w.validation("passed", "a benchmark", "4 of 4 gates hold", ["out"])
    w.measurement("gamma", 0.5, "ratio", mark="measured")
    kit.writer(run_dir).measurement("wall", 1.5, "s", mark="measured")
    ours = _result(run_dir)
    assert ours.keys() == theirs.keys() and {k: ours[k] for k in ours if k != "measurements"} == {
        k: theirs[k] for k in theirs if k != "measurements"}
    assert [{k: m[k] for k in ("name", "value", "unit")} for m in ours["measurements"]] == theirs["measurements"]
    for doc in (ours, theirs):
        check = research_result.check_doc(run_dir, doc, RUN)
        assert check.valid and check.errors == [] and check.artifacts_missing == []
        assert check.fingerprint == {"source_sha": sha} and check.validation_status == "passed"


def test_each_writer_of_a_run_adds_to_its_fingerprint_and_its_files_which_research_takes(tmp_path, no_gpu, monkeypatch):
    from research import result as research_result

    run_dir = tmp_path / "run"
    (run_dir / "out" / "proofs").mkdir(parents=True)
    (run_dir / "out" / "bench.json").write_text("{}")
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(run_dir))
    first, second = kit.writer(run_dir / "out"), kit.writer(tmp_path / "elsewhere")
    first.measurement("wall", 1.5, "s", mark="measured")
    assert first.artifact("./out/proofs/") == "out/proofs"
    assert second.fingerprint(script="checks.py", args=["--k", "256"]) == {"source_sha": SHA, "script": "checks.py",
                                                                           "args": ["--k", "256"]}
    assert first.measurement_file("out/bench.json") == "out/bench.json"
    assert first.fingerprint(mutations_run=3)["mutations_run"] == 3
    doc = _result(run_dir)
    assert list(doc) == ["schema", "run_id", "measurements", "workload_fingerprint", *kit.FILES]
    assert doc["workload_fingerprint"] == {"source_sha": SHA, "script": "checks.py", "args": ["--k", "256"],
                                           "mutations_run": 3}
    assert (doc["measurement_files"], doc["artifacts"]) == (["out/bench.json"], ["out/proofs"])
    check = research_result.check(run_dir, RUN)
    assert check.valid and check.errors == [] and check.artifacts_missing == [] and check.doc == doc
    # into a result the run wrote itself, beside what it already holds
    envelope = {"schema": kit.RESULT_SCHEMA, "run_id": RUN, "workload_fingerprint": {"door": "run"},
                "measurement_files": ["out/bench.json"]}
    monkeypatch.delenv("RESEARCH_RUN_DIR")
    (tmp_path / kit.RESULT).write_text(json.dumps(envelope))
    kit.writer(tmp_path).artifact("out/proofs")
    assert _result(tmp_path) == envelope | {"workload_fingerprint": {"door": "run", "source_sha": SHA},
                                            "measurements": [], "artifacts": ["out/proofs"]}


def test_the_writer_refuses_a_file_outside_the_result_or_named_twice_and_a_fingerprint_it_cant_hold(tmp_path, no_gpu):
    w = kit.writer(tmp_path)
    w.measurement_file("out/bench.json")
    w.fingerprint(script="run.py")
    before = (tmp_path / kit.RESULT).read_bytes()
    for name in (w.measurement_file, w.artifact):
        for rel in (3, None, "", "/abs/bench.json", "../bench.json", "out/../../bench.json", Path("out/x.json"), b"x"):
            with pytest.raises(ValueError, match="are paths relative to .* and inside it, each a string"):
                name(rel)
        for again in ("out/bench.json", "./out/bench.json", "out//bench.json"):
            with pytest.raises(ValueError, match="already names out/bench.json: a file is named once"):
                name(again)
    with pytest.raises(ValueError, match="source_sha is the commit research run shipped"):
        w.fingerprint(source_sha="d" * 40)
    with pytest.raises(ValueError, match="already holds script: a fingerprint holds each once"):
        kit.writer(tmp_path).fingerprint(mode="fast", script="other.py")
    for value in (object(), math.nan, {1, 2}, b"bytes"):
        with pytest.raises(ValueError, match="fingerprint's mode is a JSON value"):
            w.fingerprint(mode=value)
    assert (tmp_path / kit.RESULT).read_bytes() == before
    for i, (key, held) in enumerate((("workload_fingerprint", ["run.py"]), ("measurement_files", "out/bench.json"),
                                     ("artifacts", [3]))):
        other = tmp_path / str(i)
        other.mkdir()
        text = json.dumps({"schema": kit.RESULT_SCHEMA, "run_id": RUN, key: held})
        (other / kit.RESULT).write_text(text)
        for call in (lambda other=other: kit.writer(other).fingerprint(), lambda other=other: kit.writer(other).artifact("x")):
            with pytest.raises(ValueError, match=f"{key} (isn't an object|aren't a list of paths)"):
                call()
        assert (other / kit.RESULT).read_text() == text


def test_a_result_that_names_no_fingerprint_and_no_file_is_written_byte_for_byte_as_it_was(tmp_path, no_gpu):
    w, stamp = kit.writer(tmp_path), kit.host_stamp()
    w.validation("passed", "a benchmark", "4 of 4 gates hold", ["out/gates.json"])
    w.measurement("wall", 1.5, "s", mark="measured")
    text = (tmp_path / kit.RESULT).read_text()
    assert text == json.dumps({"schema": kit.RESULT_SCHEMA, "run_id": RUN, "measurements": [
        {"name": "wall", "value": 1.5, "unit": "s", "mark": "measured", "commit": SHA, "run": RUN, "ci95": None,
         "target": None, "verdict": None, "inputs": ["measured"], "host": stamp}], "validation": {
        "status": "passed", "checker": "a benchmark", "detail": "4 of 4 gates hold", "evidence": ["out/gates.json"]}},
        indent=1) + "\n"
    assert not {"workload_fingerprint", *kit.FILES} & json.loads(text).keys()


def test_a_writer_of_a_kind_writes_it_and_refuses_a_result_of_another(tmp_path, no_gpu):
    """`research` stores a result as the artifact kind its `kind` names (`bench-result/v1` where it has none), so a
    benchmark whose results are stored as their own kind writes it through the writer, and one run writes one kind."""

    from research import result as research_result

    kind = "verity-steady-point/v1"
    kit.writer(tmp_path, kind).measurement("rate", 2.5, "1/s", mark="measured")
    kit.writer(tmp_path).validation("passed", "a benchmark", "4 of 4 gates hold")
    held = _result(tmp_path)
    assert held["kind"] == kind and [m["name"] for m in held["measurements"]] == ["rate"]
    assert research_result.check(tmp_path, RUN).valid
    assert kit.writer(tmp_path, kind).fingerprint(point="p")["point"] == "p"
    with pytest.raises(ValueError, match="writes one kind"):
        kit.writer(tmp_path, "verity-vllm-benchmark/v1").measurement("wall", 1.5, "s", mark="measured")
    with pytest.raises(ValueError, match="NAME/vN"):
        kit.writer(tmp_path / "other", "a steady point").measurement("wall", 1.5, "s", mark="measured")
    kit.writer(tmp_path / "plain").measurement("wall", 1.5, "s", mark="measured")
    assert "kind" not in _result(tmp_path / "plain")
