"""A `match --run-dir` re-run (`verity_vllm.pipeline.match`) takes what its run.json records: the tier-b steps of a `--full-replay`
run among them, which no flag sets.  Its CPU phase with every stage skipped still writes the run's manifest, its spec and the
realized shape of its capture.  What the driver says of a run goes to the log: a stage's command and status, and each record it
could not write.  The phase's last word on the realized shape, the shape or that it was not written, is a line on stderr as it is
written, which a pod's script shows."""
from __future__ import annotations

import json
import shlex
import sys

import pytest

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


def test_no_flag_sets_the_tier_b_steps():
    with pytest.raises(SystemExit):
        cli.parse("match", ["--tier-b-steps", "0,1"], {})
    assert cli.parse("match", [], {}).tier_b_steps == RC.TIER_B_STEPS


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
    assert said.startswith(f"{tmp_path / 'config.json'}: no configuration read, so provenance falls back to \"hand\" (")
    assert RC._hf_config_of(cli.parse("match", [], {}), None) is None and len(_said(caplog)) == 1      # none named: nothing to say
