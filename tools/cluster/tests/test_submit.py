"""`cluster submit`, the queue's side of `research run --queue`: a declared job is placed and gets its node executor's wrapper."""
import json
from pathlib import Path

import pytest

from cluster import cli, kinds, model, submit

NEBIUS = str(Path(__file__).parents[1] / "descriptions" / "nebius.toml")
T = 1_790_000_000


def _job(**kw):
    return submit.job(**{"run_id": "r1", "project": "verity", "submitter": "lane-x", "max_s": 600, "custody_s": 900, **kw})


def test_a_cpu_allocation_covers_the_jobs_time_and_its_custody_and_a_gpu_lease_the_workload_alone():
    j = _job()
    assert j.id == "rr-r1" and j.max_s == 1500 and j.kind == "batch"
    assert _job(gpus=1).max_s == 600


def test_a_cpu_job_goes_home_in_a_capped_pinned_scope_with_cuda_hidden():
    out = submit.resolve(model.load(NEBIUS), _job(cpus=4, memory_gib=16), "r1", T)
    assert out["node"] == out["machine"] == "vy-nebius-1"
    w = out["wrap"]
    assert w[:3] == ["env", "GPU_LEASE_WHO=lane-x", "RESEARCH_RUN_ID=r1"] and "CUDA_VISIBLE_DEVICES=" in w
    assert w[w.index("systemd-run"):][:7] == ["systemd-run", "--user", "--scope", "--quiet", "--collect", "--unit", "cluster-r1"]
    assert "MemoryMax=16G" in w and "MemorySwapMax=0" in w
    assert w[-3:] == ["taskset", "-c", "96-191"], "the owner runs outside node 1's reserved pools, at its own priority"
    assert out["workload_wrap"] == []


def test_a_guest_gpu_job_goes_to_node_2_its_runner_on_its_shares_cpus_and_its_workload_alone_in_gpu_lease():
    out = submit.resolve(model.load(NEBIUS), _job(gpus=1, cpus=4, memory_gib=64), "r1", T)
    assert out["node"] == "vy-nebius-2" and "Kueue" in out["why"]
    assert "systemd-run" in out["wrap"] and out["wrap"][-6:] == ["taskset", "-c", "48-95", "nice", "-n", "19"]
    assert submit.GPU_LEASE not in out["wrap"], "custody runs in the CPU allocation, not under the GPU lease"
    assert out["workload_wrap"] == [submit.GPU_LEASE, "1", "--wait", "--max-min", "10", "--mem-gb", "64", "--preemptible", "--"]
    assert out["job"]["node"] == "vy-nebius-2" and out["job"]["max_s"] == 600


def test_quiet_and_pinned_gpus_reach_gpu_lease():
    out = submit.resolve(model.load(NEBIUS), _job(project="pous", gpus=2, quiet=True, preempt="never", gpu_ids=(3, 4)), "r1", T)
    lease = out["workload_wrap"]
    assert "--timed" in lease and "--preemptible" not in lease and lease[-3:] == ["--on", "3,4", "--"]
    assert out["wrap"][-3] == "taskset" and "nice" not in out["wrap"], "the owner runs at its own priority"


def test_refusals_name_every_nodes_reason():
    with pytest.raises(submit.Refused) as e:
        submit.resolve(model.load(NEBIUS), _job(gpus=1, preempt="never"), "r1", T)
    assert "vy-nebius-1" in str(e.value) and "vy-nebius-2" in str(e.value)
    with pytest.raises(submit.Refused, match="not h100"):
        submit.resolve(model.load(NEBIUS), _job(gpus=1, gpu_model="h100"), "r1", T)
    with pytest.raises(submit.Refused, match="unknown priority class"):
        submit.resolve(model.load(NEBIUS), _job(priority="urgent"), "r1", T)


def test_a_pinned_node_is_the_only_one_tried():
    c = model.load(NEBIUS)
    assert submit.resolve(c, _job(cpus=4, node="vy-nebius-2"), "r1", T)["node"] == "vy-nebius-2"
    with pytest.raises(submit.Refused) as e:
        submit.resolve(c, _job(gpus=1, node="vy-nebius-1"), "r1", T)
    assert "vy-nebius-2" not in str(e.value)


def test_the_cli_prints_the_placement_or_the_refusal(capsys):
    base = ["submit", NEBIUS, "--run-id", "r1", "--now", str(T), "--project", "verity", "--submitter", "lane-x", "--max-s", "600",
            "--custody-s", "900"]
    assert cli.main([*base, "--gpus", "1", "--gpu-model", "rtx-pro-6000", "--mem-gb", "64"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["machine"] == "vy-nebius-2" and out["job"]["gpu_model"] == "rtx-pro-6000"
    assert cli.main([*base, "--gpus", "1", "--preempt", "never"]) == 3
    assert "refused" in json.loads(capsys.readouterr().out)
    assert cli.main(["submit", NEBIUS, "--run-id", "r2", "--now", str(T), "--project", "verity", "--submitter", "w",
                     "--kind", "lean-audit"]) == 0
    assert json.loads(capsys.readouterr().out)["max_s"] == 1200


KINDS = Path(__file__).parents[1] / "kinds"


def test_the_repos_kinds_load_and_lean_audit_is_proofs_cpu_m():
    reg = kinds.load(KINDS)
    k = reg["lean-audit"]
    assert (k.owner, k.phase, k.max_wall_min, k.restart) == ("proofs", "cpu-m", 20, "idempotent") and k.question
    assert k.source["file"] == "kinds/proofs.toml" and len(k.source["sha256"]) == 64
    assert kinds.kinds_dir(NEBIUS) == KINDS.resolve()


def test_a_registered_kind_supplies_what_the_job_leaves_unsaid_and_names_itself_in_the_lease():
    out = submit.submit(model.load(NEBIUS), kinds.load(KINDS), run_id="r1", now=T, kind="lean-audit", project="verity",
                        submitter="proofs-w", custody_s=900)
    assert (out["node"], out["max_s"], out["job"]["cpus"], out["job"]["memory_gib"], out["job"]["max_s"]) == \
        ("vy-nebius-1", 1200, 8, 32, 2100)
    assert out["kind"] == "lean-audit" and out["question"].startswith("Do core's Lean proofs") and out["warnings"] == []
    assert out["kind_source"]["file"] == "kinds/proofs.toml" and "GPU_LEASE_WHO=lean-audit:proofs-w" in out["wrap"]
    assert out["disk_gb"] == kinds.DISK_GB


def test_the_held_rules_warn_and_never_refuse(tmp_path):
    reg = kinds.load(KINDS)
    c = model.load(NEBIUS)
    with pytest.raises(submit.Refused, match="longest run"):
        submit.submit(c, reg, run_id="r1", now=T, project="verity", submitter="w")
    out = submit.submit(c, reg, run_id="r1", now=T, project="verity", submitter="w", max_s=600)
    assert any("adhoc" in w for w in out["warnings"]) and any("no research question" in w for w in out["warnings"])
    out = submit.submit(c, reg, run_id="r1", now=T, kind="lean-audit", project="verity", submitter="w", cpus=4, max_s=3600)
    assert any("not its phase cpu-m" in w for w in out["warnings"]) and any("over its max_wall_min" in w for w in out["warnings"])
    (tmp_path / "x.toml").write_text('[kinds.big]\nphase = "gpu1"\nmax_wall_min = 90\nquestion = "q"\n')
    w = kinds.warnings(kinds.load(tmp_path)["big"], "big", question="q", gpus=1, cpus=8, memory_gib=64, max_s=5400, preempt="never")
    assert w == ["a GPU job over 60 min: split it into chunks", "a GPU kind that isn't preemptible", "declares no outputs"]


def test_a_malformed_registry_names_its_file(tmp_path):
    for body, want in (('[kinds.a]\nphase = "gpu9"\nmax_wall_min = 5\n', "phase must be one of"),
                       ('[kinds.a]\nphase = "cpu-s"\n', "max_wall_min"),
                       ('[kinds.a]\nphase = "cpu-s"\nmax_wall_min = 5\nnext = "b"\n', "isn't registered")):
        (tmp_path / "o.toml").write_text(body)
        with pytest.raises(ValueError, match=want):
            kinds.load(tmp_path)
    (tmp_path / "o.toml").write_text('[kinds.a]\nphase = "cpu-s"\nmax_wall_min = 5\n')
    (tmp_path / "p.toml").write_text('[kinds.a]\nphase = "cpu-s"\nmax_wall_min = 5\n')
    with pytest.raises(ValueError, match="already o's"):
        kinds.load(tmp_path)
    (tmp_path / "p.toml").write_text('[kinds.b]\nphase = "cpu-s"\nmax_wall_min = 5\ndisk_gb = 300\n')
    reg = kinds.load(tmp_path)
    assert reg["b"].disk_gb == 300 and reg["a"].disk_gb is None
    out = submit.submit(model.load(NEBIUS), reg, run_id="r1", now=T, kind="b", project="verity", submitter="w")
    assert out["disk_gb"] == 300
    (tmp_path / "p.toml").write_text('[kinds.b]\nphase = "cpu-s"\nmax_wall_min = 5\ndisk_gb = "lots"\n')
    with pytest.raises(ValueError, match="disk_gb"):
        kinds.load(tmp_path)
