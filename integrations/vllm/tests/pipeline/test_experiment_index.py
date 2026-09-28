"""[D88] spec.json is written once and never rewritten; experiment.json keeps the four records apart and references files, not copies."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from verity_vllm.pipeline import experiment as EXP
from verity_vllm.pipeline import research_tools

REPO = Path(__file__).resolve().parents[2]
WL = REPO / "workloads" / "workload_cov_b0_b8-mixed_c256.json"


def _spec(**over):
    wl = json.loads(WL.read_text())
    kw = dict(config_id="B0-cov_b0_b8-mixed_c256", case="B0", profile="vllm_d9105ea80_sm89_eager", workload_path=str(WL), workload_doc=wl,
              checkpoint={"repo": "HuggingFaceTB/SmolLM2-135M", "revision": "93efa2f0", "role": "B0"}, checkpoint_dir="/x/snap",
              hf_config={"hidden_size": 576, "torch_dtype": "bfloat16"}, engine_args={"max_num_batched_tokens": 2048}, max_num_seqs=8,
              sampling={}, snapshot_steps=[0, 1, 2], snapshot_max_bytes=256 << 20)
    kw.update(over)
    return EXP.build_spec(**kw)


def test_spec_is_pre_execution_and_concrete():
    s = _spec()
    assert s["known_before_execution"] is True
    w = s["workload"]
    assert w["serving_profile"] == "B8-mixed" and w["context_class"] == "C256" and w["n_requests"] == 8
    r = w["requests"][0]
    assert r["prompt_len"] == len(r["prompt_token_ids"]) and "max_tokens" in r and "arrive_step" in r
    assert w["sampling"]["variant"] == "greedy" and "variants" in w
    assert s["configuration"]["dtype"] == "bfloat16" and s["configuration"]["engine_args"] == {"max_num_batched_tokens": 2048}
    assert s["initial_state"]["kind"] == "fresh"
    # nothing observed: no realized / generated keys anywhere in the spec
    blob = json.dumps(s)
    for bad in ("generated_token_ids", "finish_step", "engine_steps_capture", "n_generated\""):
        assert bad not in blob


def test_spec_dtype_is_the_served_dtype():
    """`configuration.dtype`: explicit driver value > engine-argument override (`engine_args_required.dtype`) > HF torch_dtype."""
    assert _spec()["configuration"]["dtype"] == "bfloat16"                                              # HF fallback (no override)
    s = _spec(engine_args={"max_num_batched_tokens": 2048, "dtype": "float16"})
    assert s["configuration"]["dtype"] == "float16" and s["configuration"]["engine_args"]["dtype"] == "float16"
    assert _spec(engine_args={"dtype": "float16"}, dtype="bfloat16")["configuration"]["dtype"] == "bfloat16"   # explicit wins
    assert _spec(engine_args={"dtype": "float16"})["digest"] != _spec()["digest"]


def test_code_version_is_dirty_for_an_edit_in_every_directory_of_core(tmp_path: Path):
    """The dirty check covers each directory that ships core (`research_tools.CORE_DIRS`), not one hand-written path, and
    every other directory the integration imports (`research_tools.CODE_DIRS`)."""
    assert set(research_tools.CORE_DIRS) <= set(research_tools.CODE_DIRS)
    files = [Path("integrations/vllm/verity_vllm/x.py"), *(Path(d) / "x.py" for d in research_tools.CODE_DIRS)]
    for f in files:
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text("A = 0\n")
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-C", str(tmp_path)]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "x"], check=True)
    integration = tmp_path / "integrations" / "vllm"
    assert EXP.code_version(integration)["dirty"] is False
    for f in files:
        (tmp_path / f).write_text("A = 1\n")
        assert EXP.code_version(integration)["dirty"] is True, f
        (tmp_path / f).write_text("A = 0\n")


def test_spec_written_once(tmp_path: Path):
    run = tmp_path / "run"
    n1 = EXP.write_spec(run, _spec())
    assert n1["written"] and (run / "spec.json").exists()
    n2 = EXP.write_spec(run, _spec(snapshot_steps=[0]))          # a CPU re-run with other arguments
    assert n2["written"] is False and n2["same"] is False
    assert json.loads((run / "spec.json").read_text())["digest"] == n1["digest"]   # untouched


def test_experiment_index_separates_records(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    EXP.write_spec(run, _spec())
    (run / "run.json").write_text(json.dumps({"config_id": "B0-x", "workload_path": str(WL), "checkpoint_dir": "/x/snap", "verify_weights": True,
                                              "sampling": {"seed": 0, "k": 3}, "profile": "p", "realized": {"prompt_lens": [1], "n_generated": [2]}}))
    (run / "capture").mkdir()
    (run / "capture" / "snapshots.json").write_text(json.dumps({"snapshots": [
        {"event": 1, "step": 0, "path": "a", "shape": [4, 4]}, {"event": 2, "step": 0, "path": "b", "shape": [9000, 9000], "skipped": "spanned bytes 1 > snapshot_max_bytes 0"}]}))
    (run / "gates.json").write_text(json.dumps({"verdict": "PASS", "constants": {"TIER_A_K": 3}}))
    (run / "replay_a.json").write_text(json.dumps({"k": 3, "vu_population": {"total": 10, "oversize": 1}}))
    (run / "stages.json").write_text(json.dumps([{"stage": "fold", "status": "ok", "seconds": 1.5}, {"stage": "gates", "status": "ok", "seconds": 0.5}]))
    (run / "program.json").write_text("{}")
    doc = EXP.write_experiment(run)
    assert set(doc) >= {"specification", "execution", "representation", "checks"}
    assert doc["specification"]["spec"]["present"] and doc["specification"]["spec_digest"]
    assert doc["execution"]["realized"] == {"prompt_lens": [1], "n_generated": [2]}
    assert doc["execution"]["observation_log"]["present"] is False                 # referenced, absent, not invented
    tv = doc["representation"]["transcript_values"]
    assert tv["tensors_captured"] == 1 and tv["tensors_skipped_oversize"] == 1 and tv["vu_population"]["oversize"] == 1
    assert doc["representation"]["program"]["present"] and "sha256" in doc["representation"]["program"]
    assert doc["checks"]["results"]["verdict"] == "PASS" and doc["checks"]["costs"]["total_seconds"] == 2.0
    assert doc["checks"]["classification"] is None                                 # the campaign's judgement, not the run's
    assert (run / "experiment.json").exists()


def test_realized_consistency_names_the_phase_that_ignored_the_arrivals(tmp_path):
    """Spec §5 of the campaign runner: the same spec digest does not prove every phase honoured the arrivals; the check compares each
    phase's own realized record with the capture's and names the phase + request that differ."""
    import json
    from verity_vllm.pipeline import experiment as EXP
    run = tmp_path / "run"
    (run / "capture").mkdir(parents=True); (run / "control").mkdir(); (run / "census").mkdir(); (run / "nonint" / "c_bare").mkdir(parents=True)
    reqs = {"r0": {"first_step": 0, "finish_step": 12, "n_generated": 13, "arrive_step_realized": 0},
            "r1": {"first_step": 3, "finish_step": 21, "n_generated": 19, "arrive_step_realized": 3}}
    cap = {"timing": {"engine_steps": 38, "requests": reqs}}
    (run / "capture" / "tokens.json").write_text(json.dumps(cap))
    (run / "control" / "tokens.json").write_text(json.dumps(cap))
    (run / "nonint" / "c_bare" / "tokens.json").write_text(json.dumps(cap))
    # a census that admitted everything at step 0 (the pre-a9a13179f behaviour): 24 forwards, r1 prefilled at forward 0
    bad = {"engine_steps": 24, "requests": {"r0": dict(reqs["r0"]), "r1": {**reqs["r1"], "first_step": 0, "finish_step": 18, "arrive_step_realized": 0}}}
    (run / "census" / "census.json").write_text(json.dumps({"steps": [{} for _ in range(24)], "realized": bad}))
    rc = EXP.realized_consistency(run)
    assert rc["status"] == "inconsistent" and rc["capture_engine_steps"] == 38
    assert "census: engine_steps 24 != capture 38" in rc["problems"]
    assert any(p.startswith("census: request r1 first_step 0 != capture 3") for p in rc["problems"])
    assert all(not p.startswith(("control", "nonint")) for p in rc["problems"])
    assert set(rc["phases_compared"]) == {"control", "nonint/c_bare", "census"} and "nonint/a_observer" in rc["phases_missing"]
    # the same census, fixed: consistent
    (run / "census" / "census.json").write_text(json.dumps({"steps": [{} for _ in range(38)], "realized": {"engine_steps": 38, "requests": reqs}}))
    assert EXP.realized_consistency(run)["status"] == "consistent"
