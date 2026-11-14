"""A `match --run-dir` re-run (`verity_vllm.pipeline.match`) takes what its run.json records, the tier-b steps of a `--full-replay`
run among them unless `--tier-b-steps` names others.  `--no-verify-weights` and `--chains` reach the capture's command and the
manifest, and `--keep-traces` no stage's.  Its CPU phase with every stage skipped still writes the run's manifest, its spec and the
realized shape of its capture.  What the driver says of a run goes to the log: a stage's command and status, and each record it
could not write.  The phase's last word on the realized shape, the shape or that it was not written, is a line on stderr as it is
written, which a pod's script shows.  A run directory with no run.json, and a recorded workload this checkout lacks, are refused
by name.  A re-run takes its checkpoint, its family and its engine arguments from run.json unless they are given.  A
checkpoint's configuration is the file `--hf-config` names, else its directory's, else the hub's newest snapshot's."""
from __future__ import annotations

import json
import re
import shlex
import sys

import pytest

from verity_vllm.config import UsageError
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import match as RC


def _run_dir(tmp_path, **recorded):
    run = tmp_path / "run"
    run.mkdir()
    (run / RC.G.ARTIFACTS.get("run", "run.json")).write_text(json.dumps(
        {"case": "B0", "workload_path": str(tmp_path / "wl.json"), "profile": "p", "config_id": "x", **recorded}))
    return run


@pytest.mark.parametrize("recorded, steps", [
    ({"snapshot_steps": [0, 1, 2, 3], "tier_b_steps": [0, 2]}, [0, 2]),     # the run's own steps, not the default's
    ({"snapshot_steps": [0, 1], "tier_b_steps": [0, 1]}, [0, 1]),
    ({"snapshot_steps": [0, 1, 2, 3, 4]}, [0, 1, 2, 3]),                    # none recorded: step 0 and three decode steps
    ({"snapshot_steps": [0, 1], "tier_b_steps": []}, [0, 1]),               # a run without --full-replay recorded none
])
def test_a_run_dir_rerun_replays_the_tier_b_steps_its_run_json_records(tmp_path, recorded, steps):
    a = cli.parse("match", ["--run-dir", str(_run_dir(tmp_path, **recorded)), "--phase", "cpu", "--full-replay"], {})
    RC._defaults_from_run_dir(a)
    assert RC.Plan(a).tier_b_steps == steps
    a.full_replay = False
    assert RC.Plan(a).tier_b_steps == []


def _rerun_plan(tmp_path, recorded, full_replay):
    run = _run_dir(tmp_path, snapshot_steps=[0, 1, 2, 3], tier_b_steps=recorded)
    a = cli.parse("match", ["--run-dir", str(run), "--phase", "cpu", *(["--full-replay"] if full_replay else [])], {})
    RC._defaults_from_run_dir(a)
    return RC.Plan(a)


@pytest.mark.parametrize("recorded, steps", [
    ([""], [0, 1, 2, 3]),       # joins to nothing: every snapshot step
    (" ", [0]),                 # names no step: step 0 alone
    ("0,1", [0, 1]),            # a string's characters, joined and split again
    (["1,2"], [0, 1, 2]),
    (["3", 1], [0, 1, 3]),
])
def test_a_run_json_s_tier_b_steps_are_the_comma_list_they_join_to(tmp_path, recorded, steps):
    assert _rerun_plan(tmp_path, recorded, True).tier_b_steps == steps


@pytest.mark.parametrize("recorded", [[""], " ", "0,1", ["1,2"], [0, 2]])
def test_a_rerun_without_the_full_replay_has_no_tier_b_steps(tmp_path, recorded):
    assert _rerun_plan(tmp_path, recorded, False).tier_b_steps == []


@pytest.mark.parametrize("full_replay", [True, False])
@pytest.mark.parametrize("recorded", [[1.0], [1.5], [True], [None], [[1]], [{"x": 1}], ["x"]])
def test_a_run_json_s_tier_b_step_that_is_no_integer_s_text_is_refused_with_or_without_the_replay(tmp_path, recorded, full_replay):
    with pytest.raises(ValueError, match="invalid literal for int"):
        _rerun_plan(tmp_path, recorded, full_replay)


@pytest.mark.parametrize("given, steps", [
    ([], [0, 2]),                                   # no flag: the steps the run.json records
    (["--tier-b-steps", "1,3"], [0, 1, 3]),         # a flag wins over the record, and step 0 is always replayed
    (["--tier-b-steps", "2,7"], [0, 2]),            # a step that is no snapshot step is dropped
    (["--tier-b-steps", ""], [0, 1, 2, 3]),         # an empty list: every snapshot step
    (["--tier-b-steps", RC.DEFAULT_TIER_B_STEPS], [0, 2]),   # the default's own text reads as no flag
])
def test_tier_b_steps_on_the_command_line_win_over_the_ones_a_run_json_records(tmp_path, given, steps):
    run = _run_dir(tmp_path, snapshot_steps=[0, 1, 2, 3], tier_b_steps=[0, 2])
    a = cli.parse("match", ["--run-dir", str(run), "--phase", "cpu", "--full-replay", *given], {})
    RC._defaults_from_run_dir(a)
    assert RC.Plan(a).tier_b_steps == steps


def test_a_first_run_s_tier_b_steps_are_the_flag_s_else_step_0_and_three_decode_steps(tmp_path):
    def steps(*given):
        return RC.Plan(cli.parse("match", ["--out", str(tmp_path), "--snapshot-steps", "0,1,2,3,4", "--full-replay", *given], {})).tier_b_steps

    assert (steps(), steps("--tier-b-steps", "4"), steps("--tier-b-steps", "")) == ([0, 1, 2, 3], [0, 4], [0, 1, 2, 3, 4])


def test_the_capture_verifies_the_weights_unless_a_flag_says_otherwise_and_keeping_traces_changes_no_command(tmp_path):
    (tmp_path / "wl.json").write_text(json.dumps({"requests": [{"request_id": "r0", "prompt_len": 5, "max_tokens": 4}]}))
    run = _run_dir(tmp_path, snapshot_steps=[0, 1])

    def cmd(stage, *given):
        a = cli.parse("match", ["--run-dir", str(run), *given], {})
        RC._defaults_from_run_dir(a)
        return RC.Plan(a).cmd(stage)

    unverified = cmd("capture", "--no-verify-weights")
    assert "--verify-weights" not in unverified and cmd("capture") == [*unverified, "--verify-weights"]
    assert cmd("capture", "--keep-traces") == cmd("capture") and cmd("census", "--keep-traces") is None


CPU_STAGES = ("resolve", "census_reconcile", "nonint_compare", "compare", "fold", "fold_2", "root_policy", "golden", "wiring",
              "replay_a", "replay_chain", "gates")
EAGER_ROW = "llama32-1b__bf16__l40s__tp1__b3__i8__o4__burst__greedy__bi-eager"


def _request(i, prompt_len, first, finish):
    return {"request_id": f"r{i}", "prompt_len": prompt_len, "max_tokens": 4, "n_generated": finish - first + 1, "first_step": first,
            "finish_step": finish, "finish_reason": "length"}


def _capture(run, requests, **doc):
    (run / "capture").mkdir(exist_ok=True)
    (run / "capture" / "tokens.json").write_text(json.dumps({"requests": requests, **doc}))


def _cpu_phase(run, *more):
    return cli.main(["match", "--run-dir", str(run), "--phase", "cpu", "--skip", ",".join(CPU_STAGES), *more])


def test_a_cpu_phase_with_every_stage_skipped_writes_the_manifest_the_spec_and_the_realized_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(RC, "profile_config", lambda name: None)
    (tmp_path / "wl.json").write_text(json.dumps({"requests": [{"request_id": f"r{i}", "prompt_len": n, "max_tokens": 4}
                                                               for i, n in enumerate((5, 3, 8))], "sweep": {"row_id": EAGER_ROW}}))
    run = _run_dir(tmp_path, snapshot_steps=[0, 1], snapshot_max_bytes=777)
    # r0 runs steps 0..3, r1 steps 2..4, r2 steps 4..5: two requests at once at steps 2, 3 and 4, never three
    _capture(run, [_request(0, 5, 0, 3), _request(1, 3, 2, 4), _request(2, 8, 4, 5)], engine_steps=6)
    assert _cpu_phase(run) == 0
    man = json.loads((run / "run.json").read_text())
    assert (man["config_id"], man["profile"], man["case"], man["phase"]) == ("x", "p", "B0", "cpu")
    assert (man["snapshot_steps"], man["snapshot_max_bytes"]) == ([0, 1], 777)
    assert man["declared_execution"] == "bi-eager" and man["tier_b_steps"] == []
    assert man["skip"] == sorted({*CPU_STAGES, "fold_other"})          # `fold` names its group too
    real = man["realized"]
    assert (real["prompt_lens"], real["n_generated"], real["max_concurrent_requests"]) == ([5, 3, 8], [4, 3, 2], 2)
    assert (real["engine_steps_capture"], real["engine_steps_control"], real["snapshot_steps_realized"]) == (6, None, None)
    spec = json.loads((run / "spec.json").read_text())
    assert spec["reconstructed_after_execution"] is True
    stages = json.loads((run / RC.G.ARTIFACTS["stages"]).read_text())["stages"]
    assert [(s["stage"], s["status"], s["reason"]) for s in stages] == [(st, "skipped", "--skip") for st in CPU_STAGES]
    # the weights were verified and three chains recorded, unless a flag says otherwise
    assert (man["verify_weights"], man["sampling"]["chains"]) == (True, RC.DEFAULT_CHAINS) == (True, 3)
    assert _cpu_phase(run, "--no-verify-weights", "--chains", "5") == 0
    man = json.loads((run / "run.json").read_text())
    assert (man["verify_weights"], man["sampling"]["chains"]) == (False, 5)


def test_a_request_that_never_ran_leaves_no_peak_and_no_capture_leaves_no_realized_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(RC, "profile_config", lambda name: None)
    (tmp_path / "wl.json").write_text(json.dumps({"requests": [{"request_id": "r0", "prompt_len": 5, "max_tokens": 4}]}))
    run = _run_dir(tmp_path, snapshot_steps=[0, 1], snapshot_max_bytes=777)
    assert _cpu_phase(run) == 0
    assert "realized" not in json.loads((run / "run.json").read_text())
    _capture(run, [_request(0, 5, 0, 3), {"request_id": "r1", "prompt_len": 3, "max_tokens": 4}])
    assert _cpu_phase(run) == 0
    real = json.loads((run / "run.json").read_text())["realized"]
    assert real["max_concurrent_requests"] is None and real["prompt_lens"] == [5, 3] and real["n_generated"] == [4, 0]
    assert RC._max_concurrent([]) is None


@pytest.mark.parametrize("reqs, peak", [
    ([{"first_step": 0, "finish_step": 2}, {"first_step": 1, "finish_step": 3}, {"first_step": 3, "finish_step": 3}], 2),
    ([{"first_step": 0, "finish_step": 0}, {"first_step": 2, "finish_step": 2}], 1),
    ([{"first_step": 0, "finish_step": 2}, {"first_step": None, "finish_step": 2}], None),
    ([{"first_step": 0}], None),
])
def test_the_peak_of_requests_in_one_step_is_none_unless_every_request_states_both_its_steps(reqs, peak):
    assert RC._max_concurrent(reqs) == peak


def test_every_request_s_steps_are_read_before_any_is_found_missing():
    with pytest.raises(AttributeError, match="'int' object has no attribute 'get'"):
        RC._max_concurrent([{"first_step": None}, 7])


NESTED = "[" * 100_000 + "]" * 100_000      # JSON nested past what Python's parser reads: `json` raises RecursionError, no ValueError


@pytest.mark.parametrize("text", ["{not JSON", NESTED, '{"steps": [1e999]}'], ids=["not JSON", "nested past the parser", "an infinite step"])
def test_a_cpu_phase_takes_a_record_it_cannot_read_as_no_record(tmp_path, monkeypatch, text):
    """The checkpoint manifest, the control's tokens and the capture's kept steps are best-effort reads: one that is not JSON, is
    nested past the parser or keeps an infinite step leaves the spec and the realized shape as they are with no such file."""
    monkeypatch.setattr(RC, "profile_config", lambda name: None)
    (tmp_path / "wl.json").write_text(json.dumps({"requests": [{"request_id": "r0", "prompt_len": 5, "max_tokens": 4}]}))
    written = {}
    for name in ("none", "unread"):
        (tmp_path / name).mkdir()
        run = _run_dir(tmp_path / name, workload_path=str(tmp_path / "wl.json"), snapshot_steps=[0, 1], snapshot_max_bytes=777)
        _capture(run, [_request(0, 5, 0, 3)], engine_steps=4)
        if name == "unread":
            (run / "control").mkdir()
            for file in (tmp_path / "checkpoints.json", run / "control" / "tokens.json", run / "capture" / "snapshots.json"):
                file.write_text(text)
        assert _cpu_phase(run, "--checkpoints", str(tmp_path / "checkpoints.json")) == 0
        written[name] = (json.loads((run / "run.json").read_text())["realized"], json.loads((run / "spec.json").read_text()))
    assert written["unread"] == written["none"]
    real, spec = written["none"]
    assert (real["engine_steps_capture"], real["engine_steps_control"], real["snapshot_steps_realized"]) == (4, None, None)
    assert spec["configuration"]["checkpoint"] == {}


@pytest.mark.parametrize("text", ["{not JSON", NESTED], ids=["not JSON", "nested past the parser"])
def test_a_capture_whose_tokens_cannot_be_parsed_leaves_no_realized_shape_and_no_word_of_one(tmp_path, monkeypatch, caplog, capsys, text):
    run = _one_request_run(tmp_path, monkeypatch)
    (run / "capture" / "tokens.json").write_text(text)
    with caplog.at_level("INFO", logger="verity_vllm"):
        assert _cpu_phase(run) == 0
    assert "realized" not in json.loads((run / "run.json").read_text())
    assert capsys.readouterr().err == "" and not [line for line in _said(caplog) if "[realized]" in line]


def _said(caplog, level=None):
    return [r.getMessage() for r in caplog.records if r.name == "verity_vllm.pipeline.match" and level in (None, r.levelname)]


def _one_request_run(tmp_path, monkeypatch):
    monkeypatch.setattr(RC, "profile_config", lambda name: None)
    (tmp_path / "wl.json").write_text(json.dumps({"requests": [{"request_id": "r0", "prompt_len": 5, "max_tokens": 4}]}))
    run = _run_dir(tmp_path, snapshot_steps=[0, 1], snapshot_max_bytes=777)
    _capture(run, [_request(0, 5, 0, 3)], engine_steps=4)
    return run


def test_a_stage_that_runs_is_logged_with_its_command_and_then_its_status(tmp_path, monkeypatch, caplog, capsys):
    """The phase's last line, the realized shape, is on stderr as it is written, with no head before `[realized]`."""
    run = _one_request_run(tmp_path, monkeypatch)
    cmd = [sys.executable, "-c", "pass"]
    planned = RC.Plan.cmd
    monkeypatch.setattr(RC.Plan, "cmd", lambda self, stage: cmd if stage == "resolve" else planned(self, stage))
    assert cli.main(["match", "--run-dir", str(run), "--phase", "cpu", "--skip", ",".join(s for s in CPU_STAGES if s != "resolve")]) == 0
    resolve = next(s for s in json.loads((run / RC.G.ARTIFACTS["stages"]).read_text())["stages"] if s["stage"] == "resolve")
    assert (resolve["status"], resolve["cmd"], resolve["returncode"]) == ("ok", shlex.join(cmd), 0)
    said = _said(caplog)
    spec = json.loads((run / "spec.json").read_text())
    assert said == [f"[spec] written digest={spec['digest'][:12]}", f"[resolve] {shlex.join(cmd)}", f"[resolve] ok in {resolve['seconds']} s"]
    assert _said(caplog, "INFO") == said
    printed = capsys.readouterr()
    assert printed.out == ""
    assert printed.err == "[realized] prompt_lens=[5] n_generated=[4] steps capture/control=4/None max_concurrent=1\n"


def test_a_record_that_cannot_be_written_is_a_warning_each_time_and_the_phase_ends_on_its_stages(tmp_path, monkeypatch, caplog, capsys):
    """The run's spec, its realized shape and its experiment index inform a reader and gate nothing: a failure to write one is
    said where it happened (the realized shape is written at the phase's start, before the gates and at its end, and the end's
    failure is on stderr as it is written), and the phase's exit code is still its stages' verdict."""
    run = _one_request_run(tmp_path, monkeypatch)

    def no_room(*args, **kw):
        raise OSError("no room")

    for owner, name in ((RC.EXP, "write_spec"), (RC, "write_realized"), (RC.EXP, "write_experiment")):
        monkeypatch.setattr(owner, name, no_room)
    gates = {"stage": "gates", "status": "ok", "returncode": None, "seconds": 0.0, "log": None, "cmd": "in-process", "reason": None}
    monkeypatch.setattr(RC, "gates_stage", lambda plan, a: gates)
    assert cli.main(["match", "--run-dir", str(run), "--phase", "cpu", "--skip", ",".join(s for s in CPU_STAGES if s != "gates")]) == 0
    assert _said(caplog) == _said(caplog, "WARNING") == [
        "[spec] not written: OSError: no room", *["[realized] not written: OSError: no room"] * 2, "[experiment] not written: OSError: no room"]
    printed = capsys.readouterr()
    assert (printed.out, printed.err) == ("", "[realized] not written: OSError: no room\n")
    assert not (run / "spec.json").exists() and "realized" not in json.loads((run / "run.json").read_text())
    monkeypatch.setattr(RC, "gates_stage", lambda plan, a: {**gates, "status": "failed"})
    assert cli.main(["match", "--run-dir", str(run), "--phase", "cpu", "--skip", ",".join(s for s in CPU_STAGES if s != "gates")]) == 1


@pytest.mark.parametrize("workload, steps", [
    ({"requests": [{"request_id": "r0", "prompt_len": 5}]}, 4),                                       # no length named: steps 0 to 3
    ({"requests": [{"request_id": "r0", "prompt_len": 5, "max_tokens": 6}]}, 6),
    ({"requests": [{"request_id": "r0", "prompt_len": 5, "max_tokens": 6}], "expected_engine_steps": 9}, 9),
])
def test_every_step_of_a_workload_is_its_declared_steps_else_its_longest_request_else_four(tmp_path, caplog, workload, steps):
    (tmp_path / "wl.json").write_text(json.dumps(workload))
    a = cli.parse("match", ["--run-dir", str(_run_dir(tmp_path)), "--phase", "cpu", "--snapshot-steps", "all"], {})
    RC._defaults_from_run_dir(a)
    assert RC.Plan(a).steps == list(range(steps))
    fell_back = "requests" in workload and "max_tokens" not in workload["requests"][0]
    assert _said(caplog, "WARNING") == ["[plan] --snapshot-steps all: workload max_tokens unknown, falling back to 0,1,2,3"] * fell_back
    a.max_tokens = 2                                                          # a `--max-tokens` given wins over the workload's
    assert RC.Plan(a).steps == [0, 1]


def test_a_profile_with_no_configuration_reads_as_none_and_the_log_says_why(caplog):
    assert RC.profile_config("no-such-profile") is None
    (said,) = _said(caplog, "INFO")
    assert said.startswith("profile no-such-profile: no configuration (")


def test_an_hf_config_that_is_no_checkpoint_s_reads_as_none_and_the_log_warns(tmp_path, caplog):
    (tmp_path / "config.json").write_text("{}")
    a = cli.parse("match", ["--hf-config", str(tmp_path / "config.json")], {})
    assert RC._hf_config_of(a, None) is None
    (said,) = _said(caplog, "WARNING")
    assert re.fullmatch(re.escape(f"{tmp_path / 'config.json'}: no configuration read, so provenance falls back to \"hand\" (") + r"\w+: .+\)", said)
    assert RC._hf_config_of(cli.parse("match", [], {}), None) is None and len(_said(caplog)) == 1      # none named: nothing to say


def test_a_run_dir_with_no_run_json_and_a_recorded_workload_this_checkout_lacks_are_refused_by_name(tmp_path, monkeypatch):
    (tmp_path / "empty").mkdir()
    refusal = f"--run-dir {tmp_path / 'empty'}: no run.json (the GPU phase writes it)"
    with pytest.raises(UsageError, match=f"^{re.escape(refusal)}$"):
        RC._defaults_from_run_dir(cli.parse("match", ["--run-dir", str(tmp_path / "empty")], {}))
    run = _run_dir(tmp_path, workload_path="workloads/gone.json")               # as the GPU phase's checkout named it
    monkeypatch.chdir(tmp_path)
    with pytest.raises(UsageError, match="^workload workloads/gone.json not found in this checkout$"):
        RC._defaults_from_run_dir(cli.parse("match", ["--run-dir", str(run)], {}))
    (tmp_path / "workloads").mkdir()
    (tmp_path / "workloads" / "gone.json").write_text("{}")
    a = cli.parse("match", ["--run-dir", str(run)], {})
    RC._defaults_from_run_dir(a)
    assert (a.workload, a.out) == ("workloads/gone.json", str(run))
    (tmp_path / "unnamed").mkdir()
    a = cli.parse("match", ["--run-dir", str(_run_dir(tmp_path / "unnamed", workload_path=None))], {})
    RC._defaults_from_run_dir(a)                                                # a run.json that names no workload: `main` asks for one
    assert a.workload is None


def test_a_run_dir_rerun_takes_its_checkpoint_its_family_and_its_engine_arguments_from_run_json_unless_they_are_given(tmp_path):
    engine = {"max_num_seqs": 2, "enforce_eager": True, "seed": 7, "name": "x"}
    run = _run_dir(tmp_path, checkpoint_dir="/ck/snap", checkpoints="ck.json", family="fam", engine_args_requested=engine)
    a = cli.parse("match", ["--run-dir", str(run)], {})
    RC._defaults_from_run_dir(a)
    assert (a.checkpoint_dir, a.checkpoints, a.family) == ("/ck/snap", "ck.json", "fam")
    assert (a.max_num_seqs, a.engine_arg) == (2, ["seed=7", 'name="x"'])       # but for the two the driver has options of its own for
    given = ["--checkpoint-dir", "/ck/other", "--checkpoints", "other.json", "--family", "f2", "--max-num-seqs", "4", "--engine-arg", "seed=1"]
    b = cli.parse("match", ["--run-dir", str(run), *given], {})
    RC._defaults_from_run_dir(b)
    assert (b.checkpoint_dir, b.checkpoints, b.family, b.max_num_seqs, b.engine_arg) == ("/ck/other", "other.json", "f2", 4, ["seed=1"])
    (tmp_path / "plain").mkdir()
    c = cli.parse("match", ["--run-dir", str(_run_dir(tmp_path / "plain"))], {})
    RC._defaults_from_run_dir(c)                                                # a run.json that records none of them
    assert (c.checkpoint_dir, c.checkpoints, c.family, c.max_num_seqs, c.engine_arg) == (None, RC.DEFAULT_CHECKPOINTS, None, 8, [])


HF_CONFIG = {"model_type": "llama", "hidden_size": 2048, "num_hidden_layers": 16, "num_attention_heads": 32, "num_key_value_heads": 8,
             "head_dim": 64, "intermediate_size": 8192, "vocab_size": 128256, "rms_norm_eps": 1e-5, "rope_theta": 500000.0,
             "max_position_embeddings": 131072}


def test_a_checkpoint_s_configuration_is_the_named_file_s_then_its_directory_s_then_the_hub_s_newest_snapshot_s(tmp_path, monkeypatch):
    def config(directory, hidden):
        directory.mkdir(parents=True)
        (directory / "config.json").write_text(json.dumps({**HF_CONFIG, "hidden_size": hidden}))
        return str(directory / "config.json")

    def hidden(*given, C):
        read = RC._hf_config_of(cli.parse("match", [*given], {}), C)
        return read and read["H"]

    named, held, model = config(tmp_path / "named", 64), tmp_path / "held", {"model": "org/x"}
    config(held, 128)
    for home, sizes in ((tmp_path / "hf", (256, 512)), (tmp_path / "home" / ".cache" / "huggingface", (1024,))):
        for name, size in zip("ab", sizes):
            config(home / "hub" / "models--org--x" / "snapshots" / (name * 40), size)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert hidden("--hf-config", named, "--checkpoint-dir", str(held), C=model) == 64
    assert hidden("--checkpoint-dir", str(held), C=model) == hidden("--checkpoint-dir", str(held), C=None) == 128
    # no file named and no directory that holds one: the profile's repository under the hub, the last snapshot by name
    assert hidden(C=model) == hidden("--checkpoint-dir", str(tmp_path / "none"), C=model) == hidden("--checkpoint-dir", named, C=model) == 512
    assert [hidden(C=C) for C in (None, {}, {"model": "org/y"})] == [None] * 3
    assert hidden("--hf-config", str(tmp_path / "none.json"), "--checkpoint-dir", str(held), C=model) is None     # the named file or none
    monkeypatch.delenv("HF_HOME")
    assert hidden(C=model) == 1024                                              # the cache under the home directory
