"""The two records the Match driver (`verity_vllm.pipeline.match`) writes before its stages run.  `write_manifest`'s run.json:
every key from the options, the plan, the profile's configuration and the workload, written as it returns it; an earlier
phase's notes and realized shape kept, and nothing else of it.  `write_spec_for`'s spec.json, with `experiment`'s three functions
stood in: what it is built from (the checkpoint the case names in the manifest, the workload, the options), the flag and the
digest of a CPU phase that writes it after the execution, its line in the log, and the error it returns in place of raising."""
from __future__ import annotations

import dataclasses
import json
import logging
import re
import sys
import types
from pathlib import Path

import pytest

from verity_vllm.pipeline import cli
from verity_vllm.pipeline import match as RC
from verity_vllm.pipeline.report import workload_summary
from verity_vllm.records.schema_ids import MATCH_RUN_SCHEMA

GPU, CPU = "/pod/py311", "/pod/py312"
C = {"model": "org/model", "profile": "prof"}
#: the one option no flag sets: how `main` derived the snapshots' byte limit
[DERIVED] = [f.name for f in dataclasses.fields(RC.Options) if not f.init]


def _workload(tmp_path, **more):
    path = tmp_path / "wl.json"
    path.write_text(json.dumps({"requests": [{"request_id": "r0", "prompt_len": 5, "max_tokens": 4}], **more}))
    return str(path)


def _options(tmp_path, workload, *given, case="B1"):
    return cli.parse("match", ["--config-id", "cid", "--profile", "prof", "--workload", workload, "--case", case,
                               "--out", str(tmp_path / "runs" / "one"), "--python-gpu", GPU, "--python-cpu", CPU, *given], {})


@pytest.fixture
def stood_in(monkeypatch):
    """The profile's configuration, the HF configuration and the command line, stood in; gives what the first two were asked."""
    asked = types.SimpleNamespace(profile=[], hf=[])

    def profile_config(name):
        asked.profile.append(name)
        return dict(C)

    def hf_config_of(a, given):
        asked.hf.append(given)
        return {"hidden_size": 8}

    monkeypatch.setattr(RC, "profile_config", profile_config)
    monkeypatch.setattr(RC, "_hf_config_of", hf_config_of)
    monkeypatch.setattr(sys, "argv", ["match", "--case", "B1"])
    return asked


def test_the_manifest_holds_each_key_from_the_options_the_plan_the_profile_and_the_workload(tmp_path, stood_in):
    wl = _workload(tmp_path, execution="compiled")
    a = _options(tmp_path, wl, "--seed", "7", "--k", "3", "--chains", "2", "--max-num-seqs", "2", "--engine-arg",
                 "max_num_batched_tokens=64", "--family", "fam", "--checkpoint-dir", "/w", "--phase", "gpu", "--skip", "fold")
    plan = RC.Plan(a)
    man = RC.write_manifest(plan, a)
    assert stood_in.profile == ["prof"] and stood_in.hf == [C]
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", man["utc"])
    assert man == {
        "schema": MATCH_RUN_SCHEMA, "utc": man["utc"], "config_id": "cid", "profile": "prof", "case": "B1", "workload_path": wl,
        "workload": workload_summary(wl), "C": C, "hf_config": {"hidden_size": 8}, "laptop": False, "snapshot_steps": [0, 1, 2, 3],
        "tier_b_steps": [], "verify_weights": True, "parallel_replay": 4, "phase": "gpu",
        "sampling": {"seed": 7, "k": 3, "chains": 2, "full_replay": False}, "python_gpu": GPU, "python_cpu": CPU,
        "engine_args_requested": {"max_num_seqs": 2, "enforce_eager": False, "max_num_batched_tokens": 64},
        "declared_execution": "compiled", "engine_args_required_by_workload": {},
        "snapshot_max_bytes": 33554432, DERIVED: None, "snapshot_max_bytes_op": None,
        "checkpoints": "manifests/checkpoints.json", "checkpoint_dir": "/w", "artifacts": plan.artifacts_override(),
        "skip": ["fold", "fold_2", "fold_other"], "argv": ["--case", "B1"], "family": "fam", "mechanism_notes": [], "notes": [],
    }
    assert plan.p("run").read_text() == json.dumps(man, indent=1) + "\n"


def test_a_given_snapshot_limit_the_derived_one_and_a_full_replay_are_written_as_given(tmp_path, stood_in):
    a = _options(tmp_path, _workload(tmp_path), "--snapshot-max-bytes", "1024", "--snapshot-max-bytes-op", "512", "--full-replay")
    setattr(a, DERIVED, {"from": "the widths"})
    man = RC.write_manifest(RC.Plan(a), a)
    assert (man["snapshot_max_bytes"], man[DERIVED], man["snapshot_max_bytes_op"]) == (1024, {"from": "the widths"}, 512)
    assert man["sampling"]["full_replay"] is True and man["tier_b_steps"] == [0, 1, 2, 3]


def test_an_earlier_phase_s_notes_and_realized_shape_are_kept_and_nothing_else_of_it(tmp_path, stood_in):
    a = _options(tmp_path, _workload(tmp_path))
    plan = RC.Plan(a)
    plan.run.mkdir(parents=True)
    plan.p("run").write_text(json.dumps({"notes": ["n"], "mechanism_notes": ["m"], "realized": {"steps": 3}, "phase": "cpu", "family": "f"}))
    man = RC.write_manifest(plan, a)
    assert (man["notes"], man["mechanism_notes"], man["realized"]) == (["n"], ["m"], {"steps": 3})
    assert (man["phase"], man["family"]) == ("all", None)
    assert json.loads(plan.p("run").read_text()) == man


def test_empty_earlier_notes_and_shape_leave_the_fresh_ones(tmp_path, stood_in):
    a = _options(tmp_path, _workload(tmp_path))
    plan = RC.Plan(a)
    plan.run.mkdir(parents=True)
    plan.p("run").write_text(json.dumps({"notes": [], "mechanism_notes": None, "realized": {}}))
    man = RC.write_manifest(plan, a)
    assert (man["notes"], man["mechanism_notes"], "realized" in man) == ([], [], False)


@pytest.fixture
def spec_seen(monkeypatch):
    """`experiment`'s build, write and digest of the spec, stood in; gives what the first two were given and the note to return."""
    seen = types.SimpleNamespace(built=[], written=[], note={"written": True, "digest": "0123456789abcdef"})

    def build_spec(**given):
        seen.built.append(given)
        return {"config_id": given["config_id"], "digest": "d0"}

    def write_spec(run, spec):
        seen.written.append((run, spec))
        if isinstance(seen.note, Exception):
            raise seen.note
        return seen.note

    monkeypatch.setattr(RC.EXP, "build_spec", build_spec)
    monkeypatch.setattr(RC.EXP, "write_spec", write_spec)
    monkeypatch.setattr(RC.EXP, "digest_doc", lambda doc: "digest of " + json.dumps(doc, sort_keys=True))
    return seen


def _checkpoints(tmp_path, entries):
    path = tmp_path / "ck.json"
    path.write_text(json.dumps({"checkpoints": entries}))
    return str(path)


def test_the_spec_is_built_from_the_case_s_checkpoint_the_workload_and_the_options(tmp_path, spec_seen):
    one = {"role": "B1", "repo": "org/one", "revision": None, "files": ["a"], "sha256": "s", "digest": "dg", "local_path": "/w", "other": 1}
    ck = _checkpoints(tmp_path, [{"role": "B0", "repo": "org/zero"}, one, {"role": "B1", "repo": "org/later"}])
    wl = _workload(tmp_path, execution="compiled")
    a = _options(tmp_path, wl, "--checkpoints", ck, "--checkpoint-dir", "/w", "--max-num-seqs", "2", "--engine-arg",
                 "max_num_batched_tokens=64", "--family", "fam")
    plan = RC.Plan(a)
    assert RC.write_spec_for(plan, a, {"hf_config": {"hidden_size": 8}}) is spec_seen.note
    assert spec_seen.built == [{
        "config_id": "cid", "case": "B1", "profile": "prof", "workload_path": wl, "workload_doc": json.loads(Path(wl).read_text()),
        "checkpoint": {"repo": "org/one", "role": "B1", "files": ["a"], "sha256": "s", "digest": "dg", "local_path": "/w"},
        "checkpoint_dir": "/w", "hf_config": {"hidden_size": 8}, "engine_args": {"max_num_batched_tokens": 64}, "max_num_seqs": 2,
        "sampling": {}, "tp": 1, "dtype": None, "snapshot_steps": [0, 1, 2, 3], "snapshot_max_bytes": None,
        "harness": {"enforce_eager": False, "execution": "compiled", "driver": "verity_vllm.pipeline.match", "python_gpu": GPU,
                    "family": "fam"},
    }]
    assert spec_seen.written == [(plan.run, {"config_id": "cid", "digest": "d0"})]


@pytest.mark.parametrize("case, entries, picked", [
    ("B1", [{"role": "B1-fp8", "repo": "x"}], {"role": "B1-fp8", "repo": "x"}),
    ("B0", [{"role": "B0-fp8", "repo": "x"}], None),
    ("B0", [{"role": "B1", "repo": "y"}, {"role": "B0", "repo": "x"}], {"role": "B0", "repo": "x"}),
    ("B1", [{"repo": "x"}], None),
])
def test_only_case_b1_takes_a_checkpoint_whose_role_begins_with_its_name(tmp_path, spec_seen, case, entries, picked):
    a = _options(tmp_path, _workload(tmp_path), "--checkpoints", _checkpoints(tmp_path, entries), case=case)
    RC.write_spec_for(RC.Plan(a), a, {})
    assert spec_seen.built[0]["checkpoint"] == picked


@pytest.mark.parametrize("text", [None, "not JSON", "[1]", '{"checkpoints": [1]}'])
def test_a_manifest_that_is_absent_or_not_one_names_no_checkpoint(tmp_path, spec_seen, text):
    path = tmp_path / "ck.json"
    if text is not None:
        path.write_text(text)
    a = _options(tmp_path, _workload(tmp_path), "--checkpoints", str(path))
    assert RC.write_spec_for(RC.Plan(a), a, {}) is spec_seen.note
    assert spec_seen.built[0]["checkpoint"] is None and spec_seen.built[0]["hf_config"] is None


@pytest.mark.parametrize("tp, given", [(None, 1), (4, 4), (0, 1)])
def test_the_spec_s_tensor_parallel_size_is_the_options_else_one(tmp_path, spec_seen, tp, given):
    a = _options(tmp_path, _workload(tmp_path))
    if tp is not None:
        a.tp = tp
    RC.write_spec_for(RC.Plan(a), a, {})
    assert spec_seen.built[0]["tp"] == given


@pytest.mark.parametrize("phase, before, flagged", [("cpu", False, True), ("cpu", True, False), ("all", False, False), ("gpu", False, False)])
def test_a_cpu_phase_that_finds_no_spec_writes_one_flagged_as_after_the_execution(tmp_path, spec_seen, phase, before, flagged):
    a = _options(tmp_path, _workload(tmp_path), "--phase", phase)
    plan = RC.Plan(a)
    if before:
        plan.run.mkdir(parents=True)
        (plan.run / "spec.json").write_text("{}")
    RC.write_spec_for(plan, a, {})
    [(_run, spec)] = spec_seen.written
    if flagged:
        body = {"config_id": "cid", "reconstructed_after_execution": True}
        assert spec == {**body, "digest": "digest of " + json.dumps(body, sort_keys=True)}
    else:
        assert spec == {"config_id": "cid", "digest": "d0"}


@pytest.mark.parametrize("note, line", [
    ({"written": True, "digest": "0123456789abcdef", "existing_digest": "ffff"}, "[spec] written digest=0123456789ab"),
    ({"written": False, "existing_digest": "fedcba9876543210"}, "[spec] kept digest=fedcba987654"),
    ({"written": False, "existing_digest": "fedcba9876543210", "same": False},
     "[spec] kept digest=fedcba987654  (this phase would have produced a DIFFERENT spec; not rewritten)"),
    ({"written": False, "same": True}, "[spec] kept digest="),
])
def test_the_spec_s_line_in_the_log_says_written_or_kept_its_digest_and_a_difference(tmp_path, spec_seen, caplog, note, line):
    caplog.set_level(logging.INFO, logger=RC.__name__)
    spec_seen.note = note
    a = _options(tmp_path, _workload(tmp_path))
    RC.write_spec_for(RC.Plan(a), a, {})
    assert [r.getMessage() for r in caplog.records if r.name == RC.__name__ and r.getMessage().startswith("[spec]")] == [line]


def test_a_spec_that_cannot_be_written_is_an_error_returned_and_logged(tmp_path, spec_seen, caplog):
    caplog.set_level(logging.INFO, logger=RC.__name__)
    spec_seen.note = OSError("no room")
    a = _options(tmp_path, _workload(tmp_path))
    assert RC.write_spec_for(RC.Plan(a), a, {}) == {"written": False, "error": "OSError: no room"}
    assert [(r.levelname, r.getMessage()) for r in caplog.records if r.name == RC.__name__] == [("WARNING", "[spec] not written: OSError: no room")]


def test_a_workload_that_is_not_an_object_ends_the_driver_and_is_no_error_of_the_spec(tmp_path, spec_seen):
    path = tmp_path / "wl.json"
    path.write_text("[1]")
    a = _options(tmp_path, str(path))
    with pytest.raises(SystemExit, match="not an object"):
        RC.write_spec_for(RC.Plan(a), a, {})
    assert spec_seen.built == []
