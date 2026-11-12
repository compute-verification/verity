"""The reader of a served two-stage run's record (`verity_bench.served_record`): the record is exactly what the
command's `served_share` takes; a time or a count of GPUs the run's files don't hold is None there and named, and
`served_share` then claims no share and names the same fields; a record whose files hold everything gives a share."""

from __future__ import annotations

import json

import pytest

from verity_bench import exfiltration as E
from verity_bench import served_record as R


def session(session_s, prove_total_s, **more):
    return {"session_s": session_s, "circuit_sha512": "c", "statement_digest": "d", "sigma": "s",
            "timing": {"e2e_s": session_s - 1, "witness_s": 2.0, "prove_total_s": prove_total_s, "k_log": 20, "units": 4,
                       "proof_bytes": 1}, "verify_s": 0.5, "lean_verify_s": 3.0, "lean_partition": True, **more}


def second(tmp_path, summary):
    d = tmp_path / "second"
    d.mkdir(parents=True, exist_ok=True)
    (d / R.SUMMARY).write_text(json.dumps(summary))
    return d


def load(tmp_path, name, wall_s, gpus):
    f = tmp_path / f"{name}.json"
    f.write_text(json.dumps({"tag": name, "rate": 1.0, "requests": 8, "wall_s": wall_s,
                             "gpu": None if gpus is None else {"gpus": gpus, "busy_gpu_s": 1.0}}))
    return f


def as_main_writes(tmp_path):
    """A second stage's summary with the fields the two-stage audit writes: each session's seconds, and no more of
    what the record takes."""
    return second(tmp_path, {
        "sample": [3], "second_stage": {"registration": "r", "law": {}, "draw": [1], "window_root": "w"},
        "interior": {"population": 8, "run": "interior-run", "launches": 1, "members": [], "registered": [], "links": [],
                     "returns": []},
        "audit": {"checks_before_draw": {}, "refused": [], "audit": {"accepted": True}, "draw": [1], "draw_counts": {},
                  "negatives": {}, "circuits": {}, "timing": {"m1": session(40.0, 31.5), "m0": session(50.0, 38.25)}},
        "accepted": True})


def test_a_record_as_main_writes_it_holds_the_sessions_seconds_and_names_what_it_lacks(tmp_path):
    record, absent = R.read(as_main_writes(tmp_path), run="r1")
    assert tuple(record) == R.FIELDS
    assert record == {"run": "r1", "prove": [{"seconds": 50.0, "gpus": None}, {"seconds": 40.0, "gpus": None}],
                      "staging": None, "replay": None, "interior": None, "serving": None, "stock": None}
    assert [g["field"] for g in absent] == ["prove.gpus", "staging", "replay", "interior", "serving", "stock"]
    assert all(g["why"] for g in absent) and "not the GPUs it held" in absent[0]["why"]
    # the prover's own timer is the other reading of a session's seconds, and nothing else of the record moves
    timer, same = R.read(as_main_writes(tmp_path), run="r1", seconds="prove_total_s")
    assert [e["seconds"] for e in timer["prove"]] == [38.25, 31.5] and same == absent
    assert {k: v for k, v in timer.items() if k != "prove"} == {k: v for k, v in record.items() if k != "prove"}
    with pytest.raises(ValueError, match="one of session_s, prove_total_s"):
        R.read(as_main_writes(tmp_path), seconds="e2e_s")
    # the command claims no share from it, and names each field the reader named
    share = E.served_share(record)
    assert share["value"] is None and share["mark"] is None and share["why"].startswith("not measured: r1's record lacks ")
    named = {k for g in share["missing"] for k in g["lacks"]}
    assert named == {"prove.gpus", "staging", "replay", "interior", "serving", "stock"}
    assert {g["field"] for g in absent} == named


def test_the_serve_and_stock_are_a_loads_meta_json(tmp_path):
    record, absent = R.read(as_main_writes(tmp_path), run="r1", serving=load(tmp_path, "two-stage", 600.0, 8),
                            stock=load(tmp_path, "compiled", 540.0, 8))
    assert record["serving"] == {"seconds": 600.0, "gpus": 8} and record["stock"] == {"seconds": 540.0, "gpus": 8}
    assert [g["field"] for g in absent] == ["prove.gpus", "staging", "replay", "interior"]
    # a load served without NVML's bindings holds no count of GPUs: the entry's is None and named
    record, absent = R.read(as_main_writes(tmp_path), run="r1", serving=load(tmp_path, "blind", 600.0, None))
    assert record["serving"] == {"seconds": 600.0, "gpus": None}
    assert [g["field"] for g in absent][-2:] == ["serving.gpus", "stock"]
    # a serving cost goes in only where no stock serve is given
    cost, _ = R.read(as_main_writes(tmp_path), run="r1", serving_cost=0.05, serving_cost_source="a run, a load")
    assert (cost["serving_cost"], cost["serving_cost_source"]) == (0.05, "a run, a load")
    both, _ = R.read(as_main_writes(tmp_path), run="r1", stock=load(tmp_path, "compiled", 540.0, 8), serving_cost=0.05)
    assert "serving_cost" not in both


def test_a_record_whose_files_hold_every_field_gives_the_share(tmp_path):
    d = second(tmp_path, {
        "audit": {"timing": {"m0": session(50.0, 38.25, gpus=1), "m1": session(40.0, 31.5, gpus=2)},
                  "staging": {"before_draw_s": 6.0, "drawn_s": 4.0, "public_files": 2, "gpus": 1}},
        "replay": {"seconds": 12.0, "gpus": 8}, "interior_commit": {"seconds": 3.0, "gpus": 8}})
    record, absent = R.read(d, run="r2", serving=load(tmp_path, "two-stage", 600.0, 8),
                            stock=load(tmp_path, "compiled", 540.0, 8))
    assert absent == [] and tuple(record) == R.FIELDS
    assert record["prove"] == [{"seconds": 50.0, "gpus": 1}, {"seconds": 40.0, "gpus": 2}]
    assert record["staging"] == [{"seconds": 10.0, "gpus": 1}]
    assert (record["replay"], record["interior"]) == ([{"seconds": 12.0, "gpus": 8}], [{"seconds": 3.0, "gpus": 8}])
    share = E.served_share(record)
    served = 600.0 * 8
    audit = (50.0 + 80.0) + 10.0 + 96.0 + 24.0 + (served - 540.0 * 8)
    assert share["missing"] == [] and share["mark"] == "measured" and share["value"] == pytest.approx(audit / served)


def test_a_directory_that_is_no_second_stages_is_refused(tmp_path):
    with pytest.raises(ValueError, match="holds no `summary.json`"):
        R.read(tmp_path)
    with pytest.raises(ValueError, match="holds no `audit`"):
        R.read(second(tmp_path, {"refused": "the sample"}))
    record, absent = R.read(second(tmp_path, {"audit": {"timing": {}}}))
    assert record["run"] is None and record["prove"] is None and [g["field"] for g in absent][:2] == ["run", "prove"]


def test_the_command_prints_the_record_and_names_what_it_lacks(tmp_path, capsys):
    assert R.main(["--second", str(as_main_writes(tmp_path)), "--run", "r1"]) == 0
    out, err = capsys.readouterr()
    assert json.loads(out)["prove"][0] == {"seconds": 50.0, "gpus": None}
    assert err.count("served_record: the record lacks `") == 6 and "`prove.gpus`" in err and "`replay`" in err
    assert R.main(["--second", str(tmp_path / "none")]) == 2
    assert "holds no `summary.json`" in capsys.readouterr().err
