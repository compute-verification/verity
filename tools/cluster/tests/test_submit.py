"""`cluster submit`, the queue's side of `research run --queue`: a declared job is placed and gets its node executor's wrapper."""
import json
from pathlib import Path

import pytest

from cluster import cli, model, submit

NEBIUS = str(Path(__file__).parents[1] / "descriptions" / "nebius.toml")
T = 1_790_000_000


def _job(**kw):
    return submit.job(**{"run_id": "r1", "project": "verity", "submitter": "lane-x", "max_s": 600, "custody_s": 900, **kw})


def test_the_allocation_covers_the_jobs_time_and_its_custody():
    j = _job()
    assert j.id == "rr-r1" and j.max_s == 1500 and j.kind == "batch"


def test_a_cpu_job_goes_home_in_a_capped_pinned_scope_with_cuda_hidden():
    out = submit.resolve(model.load(NEBIUS), _job(cpus=4, memory_gib=16), "r1", T)
    assert out["node"] == out["machine"] == "vy-nebius-1"
    w = out["wrap"]
    assert w[:3] == ["env", "GPU_LEASE_WHO=lane-x", "RESEARCH_RUN_ID=r1"] and "CUDA_VISIBLE_DEVICES=" in w
    assert w[w.index("systemd-run"):][:7] == ["systemd-run", "--user", "--scope", "--quiet", "--collect", "--unit", "cluster-r1"]
    assert "MemoryMax=16G" in w and "MemorySwapMax=0" in w
    assert w[-3:] == ["taskset", "-c", "96-191"], "the owner runs outside node 1's reserved pools, at its own priority"


def test_a_guest_gpu_job_goes_to_node_2_through_gpu_lease_on_its_shares_cpus():
    out = submit.resolve(model.load(NEBIUS), _job(gpus=1, cpus=4, memory_gib=64), "r1", T)
    assert out["node"] == "vy-nebius-2" and "Kueue" in out["why"]
    w = out["wrap"]
    lease = w[w.index(submit.GPU_LEASE):w.index("--")]
    assert lease == [submit.GPU_LEASE, "1", "--wait", "--max-min", "25", "--mem-gb", "64", "--preemptible"]
    assert w[w.index("--") + 1:] == ["taskset", "-c", "48-95", "nice", "-n", "19"]
    assert out["job"]["node"] == "vy-nebius-2" and out["job"]["max_s"] == 1500


def test_quiet_and_pinned_gpus_reach_gpu_lease():
    out = submit.resolve(model.load(NEBIUS), _job(project="pous", gpus=2, quiet=True, preempt="never", gpu_ids=(3, 4)), "r1", T)
    w = out["wrap"]
    lease = w[w.index(submit.GPU_LEASE):w.index("--")]
    assert "--timed" in lease and "--preemptible" not in lease and lease[-2:] == ["--on", "3,4"]
    assert w[-1] == "--", "the owner of node 2 runs unpinned and at its own priority, as gpu-lease leases do today"


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
