"""The reader of a served two-stage run's record (`verity_bench.served_record`): the record holds what the command's
`served_share` takes under its names; a time or a count of GPUs the run's files don't hold is None there and named,
and `served_share` then claims no share and names the same fields; a record whose files hold everything gives a
share; and what the audit showed is counted from the files, with the runs' ids and commits and what holds their files
to each other."""

from __future__ import annotations

import json

import pytest

from verity_bench import exfiltration as E
from verity_bench import served_record as R

COMMIT = "c0ffee" + "0" * 34


def session(session_s, prove_total_s, **more):
    return {"session_s": session_s, "circuit_sha512": "c", "statement_digest": "d", "sigma": "s",
            "timing": {"e2e_s": session_s - 1, "witness_s": 2.0, "prove_total_s": prove_total_s, "k_log": 20, "units": 4,
                       "proof_bytes": 1}, "verify_s": 0.5, "lean_verify_s": 3.0, "lean_partition": True, **more}


def second(tmp_path, summary, name="summary.json"):
    d = tmp_path / "second"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(summary))
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


def shares(record):
    return {k: record[k] for k in R.FIELDS}


def test_a_record_as_main_writes_it_holds_the_sessions_seconds_and_names_what_it_lacks(tmp_path):
    record, absent = R.read(as_main_writes(tmp_path), run="r1")
    assert shares(record) == {"run": "r1", "prove": [{"seconds": 50.0, "gpus": None}, {"seconds": 40.0, "gpus": None}],
                              "staging": None, "replay": None, "interior": None, "serving": None, "stock": None}
    assert record[R.SOURCE] is None and record["commit"] is None
    assert [g["field"] for g in absent] == [R.SOURCE, "prove.gpus", "staging", "replay", "interior", "serving", "stock"]
    assert all(g["why"] for g in absent) and "not the GPUs it held" in absent[1]["why"]
    # the command claims no share from it, and names each field the reader named
    share = E.served_share(record)
    assert share["value"] is None and share["mark"] is None and share["why"].startswith("not measured: r1's record lacks ")
    named = {k for g in share["missing"] for k in g["lacks"]}
    assert named == {"prove.gpus", "staging", "replay", "interior", "serving", "stock"}
    assert {g["field"] for g in absent} - {R.SOURCE} == named


def test_the_serve_and_stock_are_a_loads_meta_json(tmp_path):
    record, absent = R.read(as_main_writes(tmp_path), run="r1", serving=load(tmp_path, "two-stage", 600.0, 8),
                            stock=load(tmp_path, "compiled", 540.0, 8))
    assert record["serving"] == {"seconds": 600.0, "gpus": 8} and record["stock"] == {"seconds": 540.0, "gpus": 8}
    assert [g["field"] for g in absent] == [R.SOURCE, "prove.gpus", "staging", "replay", "interior"]
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
    """The driver's fields, each under its name: a session's seconds by its own source, and a part that started no
    process on a card on 0 GPUs, which the command reads as no GPU-seconds."""

    d = second(tmp_path, {
        "audit": {"timing": {"m0": session(50.0, 38.25, gpus=1, busy_s=45.0, prove_seconds_source="busy"),
                             "m1": session(40.0, 31.5, gpus=2, busy_s=30.0, prove_seconds_source="busy")},
                  "staging": {"before_draw_s": 6.0, "drawn_s": 4.0, "public_files": 2, "gpus": 0}},
        "replay": {"seconds": 12.0, "gpus": 8, "ready_s": 130.0}, "interior_commit": {"seconds": 3.0, "gpus": 0}})
    record, absent = R.read(d, run="r2", serving=load(tmp_path, "two-stage", 600.0, 8),
                            stock=load(tmp_path, "compiled", 540.0, 8))
    assert absent == [] and record[R.SOURCE] == "busy"
    assert record["prove"] == [{"seconds": 45.0, "gpus": 1}, {"seconds": 30.0, "gpus": 2}]
    assert record["staging"] == [{"seconds": 10.0, "gpus": 0}]
    assert (record["replay"], record["interior"]) == ([{"seconds": 12.0, "gpus": 8}], [{"seconds": 3.0, "gpus": 0}])
    share = E.served_share(record)
    served = 600.0 * 8
    audit = (45.0 + 60.0) + 0.0 + 96.0 + 0.0 + (served - 540.0 * 8)
    assert share["missing"] == [] and share["mark"] == "measured" and share["value"] == pytest.approx(audit / served)
    assert share["parts"]["staging"]["gpu_s"] == 0.0 and share["parts"]["staging"]["mark"] == "measured"
    with pytest.raises(ValueError, match="on -1 GPUs: an entry is seconds of 0 or more on a whole count of GPUs, 0 or more"):
        E.served_share(record | {"staging": [{"seconds": 10.0, "gpus": -1}]})


def test_a_sessions_seconds_are_those_its_own_source_names(tmp_path):
    """`busy` reads the proof server's clock on the session's prover and `session` the audit's on the whole session;
    the record's source is `session` where any session's is, absent where a session names none, and no other word is
    taken."""

    def timing(*sources):
        return {"audit": {"timing": {f"m{i}": session(50.0 - i, 30.0, gpus=1, busy_s=20.0 + i,
                                                      **({} if s is None else {R.SOURCE: s}))
                                     for i, s in enumerate(sources)}}}

    record, absent = R.read(second(tmp_path, timing("busy", "session")), run="r3")
    assert record["prove"] == [{"seconds": 20.0, "gpus": 1}, {"seconds": 49.0, "gpus": 1}] and record[R.SOURCE] == "session"
    assert R.SOURCE not in [g["field"] for g in absent]
    record, absent = R.read(second(tmp_path, timing("busy", None)), run="r3")
    assert record["prove"] == [{"seconds": 20.0, "gpus": 1}, {"seconds": 49.0, "gpus": 1}] and record[R.SOURCE] is None
    assert absent[0]["field"] == R.SOURCE and "read as `session_s`" in absent[0]["why"]
    with pytest.raises(ValueError, match="a session's `prove_seconds_source` is timer, not one of busy, session"):
        R.read(second(tmp_path, timing("busy", "timer")), run="r3")
    # a source that names seconds the session doesn't hold: the entry's are None, and named
    lost = timing("busy")
    del lost["audit"]["timing"]["m0"]["busy_s"]
    record, absent = R.read(second(tmp_path, lost), run="r3")
    assert record["prove"] == [{"seconds": None, "gpus": 1}]
    assert [(g["field"], g["why"]) for g in absent][0] == ("prove.seconds", "`audit.timing` holds no `busy_s` for a session")


def run_dir(tmp_path, name, files, commit=COMMIT, source=True):
    """A run's directory as it is fetched: its record beside its `out/`."""

    d = tmp_path / name
    for path, content in files.items():
        (d / path).parent.mkdir(parents=True, exist_ok=True)
        (d / path).write_text(content if isinstance(content, str) else json.dumps(content))
    (d / "job.json").write_text(json.dumps({
        "run": {"id": name, "remote": {"source_sha": commit}},
        "store": {"conditions": {"source": {"commit": commit, "dirty": False} if source else None}}}))
    return d


LAYOUT = {"format": "f", "window": 1, "check": "Select{V=8}", "per_prompt": 0, "per_output": 1, "inline": True,
          "replay": 'Inlined{template="Select{V=8}",N=2}', "prompt": 4, "outputs": 2, "spans": 8,
          "values": {"b.ingress": {"root": "a"}, "b.egress": {"root": "b"}}, "seconds": 0}
REGISTRATION = "ab" * 64
WITHHELD = {"accepted": False, "interior": ["statements"], "replay_sample": [], "statements": ["two-stage-form"],
            "registered": {"interior": None, "replay_sample": None}}


def served(tmp_path, *, outcome=WITHHELD, held=True, verdicts=None, units=(2,), layout=LAYOUT):
    """A served run's four runs' directories, each holding what that run publishes."""

    verdicts = verdicts or {"lean-verify/m0": True, "lean-verify/m1": True, "lean-draw/m0": True, "lean-draw/m1": True,
                            "m0-verifier/m0": True, "m0-verifier/m1": {"accepted": True}}
    stage1 = {"commit": layout, "registration": REGISTRATION, "rule": {"replay": "rate:1/8"},
              "draw": {"k": 1, "law": "subset", "population": 8, "units": list(units)}}
    summary = {
        "first_stage": {"registration": REGISTRATION, "units": [2]}, "unlinked": False, "held": held, "outcome": outcome,
        "accepted": False, "record": {"failed": ["registered"]}, "pins": {"m0": True, "m1": True},
        "round_two": {"ok": True, "codes": []}, "second_stage": {"draw": [1, 2, 3]},
        "interior": {"population": 16, "registered": ["logits"], "links": [["m0", "logits", "logits"]], "returns": []},
        "audit": {"checks_before_draw": {"a": True, "b": True}, "negatives": {"n0": True, "n1": {"refused": True}},
                  "audit": {"accepted": True, "complete": True, "registered_reads": True, "verdicts": verdicts},
                  "timing": {"m0": session(50.0, 38.25), "m1": session(40.0, 31.5)}}}
    return {
        "second": run_dir(tmp_path, "r20261113-000004-0004", {"out/second.json": summary,
                                                             "out/machine.txt": "30\nMem: 251\n" + "d" * 40 + "\nsecond rc=0\n"}),
        "first": run_dir(tmp_path, "r20261113-000002-0002", {
            "out/stage1/p4-o2/stage1.json": stage1,
            "out/check.txt": json.dumps({"registration": REGISTRATION[:16], "error": None, "session_error": None,
                                         "coins_reason": None}, indent=1) + "\nFIRST: ACCEPTED\n"}),
        "serve": run_dir(tmp_path, "r20261113-000001-0001", {"out/boundary/w1/p4-o2/commit.json": layout,
                                                            "out/boundary/w1/p8-o2/commit.json": layout | {"prompt": 8}}),
        "replay": run_dir(tmp_path, "r20261113-000003-0003", {"out/summary.json": {"replay_wall_s": 0.5, "bit_exact": True}},
                          commit="e" * 40, source=False)}


def test_what_the_audit_showed_is_counted_from_the_four_runs_files(tmp_path):
    record, absent = R.read(**served(tmp_path))
    assert (record["run"], record["commit"]) == ("r20261113-000004-0004", COMMIT)
    assert {k: (r["run"], r["commit"]) for k, r in record["runs"].items()} == {
        "serve": ("r20261113-000001-0001", COMMIT), "first": ("r20261113-000002-0002", COMMIT),
        "replay": ("r20261113-000003-0003", "e" * 40), "second": ("r20261113-000004-0004", COMMIT)}
    # the replay's seconds are its own run's, which holds no count of GPUs
    assert record["replay"] == [{"seconds": 0.5, "gpus": None}]
    assert [g["field"] for g in absent if g["field"].startswith("replay")] == ["replay.gpus"]
    shown = record["audit"]
    assert shown["verdicts"] == {"lean-draw": {"accepted": 2, "of": 2}, "lean-verify": {"accepted": 2, "of": 2},
                                 "m0-verifier": {"accepted": 2, "of": 2}}
    assert shown["checks"] == {"checks_before_draw": {"accepted": 2, "of": 2}, "pins": {"accepted": 2, "of": 2},
                               "negatives": {"accepted": 2, "of": 2}, "registered_reads": {"accepted": 1, "of": 1},
                               "complete": {"accepted": 1, "of": 1}, "round_two": {"accepted": 1, "of": 1, "codes": []}}
    assert shown["outcome"] == {"accepted": False, "codes": ["two-stage-form"], "refused": None} and shown["held"] is True
    assert (shown["sessions"], shown["proof_blocks"], shown["cores"], shown["sizes"]) == (2, 3, 30, [20])
    assert shown["seconds"] == {"session_s": 90.0, "busy_s": None, "prove_total_s": 69.75, "lean_verify_s": 6.0,
                                "verify_s": 1.0}
    assert shown["layout"] == {k: LAYOUT[k] for k in R.LAYOUT if k in LAYOUT} | {"values": ["b.egress", "b.ingress"]}
    assert shown["interior"] == {"registered": ["logits"], "links": 1, "returns": 0, "population": 16, "unlinked": False}
    assert shown["first_stage"] == {"registration": REGISTRATION, "rule": "rate:1/8", "sampled": [2], "of": 8,
                                    "check": {"error": None, "session_error": None, "coins_reason": None,
                                              "said": "FIRST: ACCEPTED"}}
    assert [t["holds"] for t in shown["ties"]] == [True, True, True, True]
    # the one-stage record over the interior is kept apart: it is not the stream's outcome
    assert shown["one_stage_record"] == {"accepted": False, "record": {"failed": ["registered"]}}


def test_files_that_are_not_one_runs_are_said_not_to_hold(tmp_path):
    dirs = served(tmp_path, units=(5,))
    (dirs["serve"] / "out/boundary/w1/p4-o2/commit.json").write_text(json.dumps(LAYOUT | {"spans": 9}))
    record, _ = R.read(**dirs)
    assert [t["holds"] for t in record["audit"]["ties"]] == [True, False, True, False]
    # with the second stage alone nothing is held to anything, and nothing is said to hold
    alone, _ = R.read(dirs["second"])
    assert [t["holds"] for t in alone["audit"]["ties"]] == [None, None, None, None]
    assert alone["audit"]["layout"] is None and alone["audit"]["first_stage"] is None
    assert alone["runs"] == {"serve": None, "first": None, "replay": None, "second": record["runs"]["second"]}


def test_a_run_whose_record_names_no_commit_says_why(tmp_path):
    d = run_dir(tmp_path, "r20261113-000004-0004", {"out/second.json": {"audit": {"timing": {}}}}, commit=None, source=False)
    assert R.run_of(d) == {"run": "r20261113-000004-0004", "commit": None, "dirty": False,
                           "why": "its record names no commit", "machine": None}
    # its machine's own line is the next place
    (d / "out/machine.txt").write_text("8\n" + "d" * 40 + "\n")
    assert R.run_of(d)["commit"] == "d" * 40
    # a file alone names neither a run nor a commit
    record, absent = R.read(d / "out/second.json")
    assert (record["run"], record["commit"], record["runs"]["second"]) == (None, None, None)
    assert absent[0]["field"] == "run"


def test_a_directory_that_is_no_second_stages_is_refused(tmp_path):
    with pytest.raises(ValueError, match="holds none of out/second.json, .*: it is not a second stage's run"):
        R.read(tmp_path)
    with pytest.raises(ValueError, match="holds no `audit`"):
        R.read(second(tmp_path, {"refused": "the sample"}))
    with pytest.raises(ValueError, match="it is not a first stage's run"):
        R.read(as_main_writes(tmp_path), first=tmp_path / "second")
    record, absent = R.read(second(tmp_path, {"audit": {"timing": {}}}))
    assert record["run"] is None and record["prove"] is None and [g["field"] for g in absent][:2] == ["run", "prove"]


def test_the_command_prints_the_record_and_names_what_it_lacks(tmp_path, capsys):
    assert R.main(["--second", str(as_main_writes(tmp_path)), "--run", "r1"]) == 0
    out, err = capsys.readouterr()
    assert json.loads(out)["prove"][0] == {"seconds": 50.0, "gpus": None}
    assert err.count("served_record: the record lacks `") == 7 and "`prove.gpus`" in err and "`replay`" in err
    assert R.main(["--second", str(tmp_path / "none")]) == 2
    assert "it is not a second stage's run" in capsys.readouterr().err
