"""A `match --run-dir` re-run (`verity_vllm.pipeline.match`) takes what its run.json records: the tier-b steps of a `--full-replay`
run among them, which no flag sets.  Its CPU phase with every stage skipped still writes the run's manifest, its spec and the
realized shape of its capture."""
from __future__ import annotations

import json

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

