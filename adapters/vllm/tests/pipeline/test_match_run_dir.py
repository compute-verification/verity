"""A `match --run-dir` re-run takes what its run.json records: the tier-b steps of a `--full-replay` run among them, which no
flag sets."""
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
