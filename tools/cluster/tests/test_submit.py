"""`cluster submit`, the queue's side of `research run --queue`: a declared job is placed and gets its node executor's wrapper."""
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from cluster import cli, kinds, model, submit

NEBIUS = str(Path(__file__).parents[1] / "descriptions" / "nebius.toml")
T = 1_790_000_000


def _job(**kw):
    return submit.job(**{"run_id": "r1", "project": "verity", "submitter": "lane-x", "max_s": 600, "custody_s": 900, **kw})


def _n1_executor(c, executor):
    return replace(c, nodes={**c.nodes, "vy-nebius-1": replace(c.nodes["vy-nebius-1"], gpu_executor=executor)})


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
    assert "AllowedCPUs=96-127" in w and w[w.index("--"):][1:6] == ["sh", "-c", submit.CPUSET_CHECK, "cpuset-check", "96-127"], \
        "the scope holds the pin, so a workload's own taskset can't leave it, and the wrapper checks that it does"
    assert w[-3:] == ["taskset", "-c", "96-127"], "the owner runs outside node 1's checks and provers pools, at its own priority"
    assert "CPUWeight=idle" in w, "on node 1 a CPU job yields its CPUs to the GPU jobs' host threads that share them"
    assert out["workload_wrap"] == []
    guest = submit.resolve(model.load(NEBIUS), _job(project="pous", cpus=4, memory_gib=16, node="vy-nebius-1"), "r1", T)
    assert guest["wrap"][-6:] == ["taskset", "-c", "96-127", "nice", "-n", "19"], "a guest's share narrows to the unpooled CPUs"


def test_a_gpu_host_on_node_1_runs_at_a_gpu_jobs_weight_and_takes_no_gpus_from_the_queue():
    """A job that takes gpu-lease itself around each GPU step (proofs' rec-step provers) is a GPU job's host: at CPUWeight=idle its
    verifier and replays between leases wait on CPU time the other jobs leave, and its timings measure that wait."""
    c, reg = model.load(NEBIUS), kinds.load(KINDS)
    out = submit.submit(c, reg, run_id="r1", now=T, project="verity", submitter="w", max_s=600, memory_gib=16, cpus=4,
                        node="vy-nebius-1", gpu_host=True)
    assert "CPUWeight=idle" not in out["wrap"] and "AllowedCPUs=96-127" in out["wrap"] and out["workload_wrap"] == []
    assert out["gpu_host"] is True
    plain = submit.submit(c, reg, run_id="r1", now=T, project="verity", submitter="w", max_s=600, memory_gib=16, cpus=4,
                          node="vy-nebius-1")
    assert "CPUWeight=idle" in plain["wrap"] and plain["gpu_host"] is False
    with pytest.raises(submit.Refused, match="takes its GPUs with gpu-lease itself"):
        submit.submit(c, reg, run_id="r1", now=T, project="verity", submitter="w", max_s=600, memory_gib=16, gpus=1, gpu_host=True)


def test_the_cpuset_check_starts_the_job_only_in_a_scope_whose_cpuset_is_the_pin():
    def check(pin):
        return subprocess.run(["sh", "-c", submit.CPUSET_CHECK, "cpuset-check", pin, "printf", "%s|", "a b", "c"],
                              capture_output=True, text=True)
    held = subprocess.run(["sh", "-c", 'cat "/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/cpuset.cpus" 2>/dev/null'],
                          capture_output=True, text=True).stdout.strip()
    ok = check(held)
    assert (ok.returncode, ok.stdout) == (0, "a b|c|"), ok.stderr
    bad = check("0-99999")
    assert bad.returncode == 70 and bad.stdout == "" and "does not hold CPUs 0-99999" in bad.stderr, \
        "where systemd dropped AllowedCPUs= (cpuset not delegated), nothing starts"


def test_a_guest_gpu_job_goes_to_node_2_its_runner_on_its_shares_cpus_and_its_workload_alone_in_gpu_lease():
    out = submit.resolve(model.load(NEBIUS), _job(gpus=1, cpus=4, memory_gib=64), "r1", T)
    assert out["node"] == "vy-nebius-2" and "pinned there" in out["why"]
    assert "systemd-run" in out["wrap"] and out["wrap"][-6:] == ["taskset", "-c", "48-95", "nice", "-n", "19"]
    assert "AllowedCPUs=48-95" in out["wrap"], "the runner's scope holds the share's CPUs; the workload keeps taskset's pin in its lease"
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
    n2 = submit.resolve(c, _job(cpus=4, node="vy-nebius-2"), "r1", T)
    assert n2["node"] == "vy-nebius-2" and "CPUWeight=idle" not in n2["wrap"], "node 2's guest CPU jobs run at nice 19 instead"
    with pytest.raises(submit.Refused) as e:
        submit.resolve(c, _job(gpus=1, quiet=True, node="vy-nebius-1"), "r1", T)
    assert "vy-nebius-2" not in str(e.value)


def test_node_1s_gpus_take_a_pinned_job_through_its_own_gpu_lease_only_with_the_executor_switched_on():
    c = model.load(NEBIUS)
    assert c.nodes["vy-nebius-1"].gpu_executor == "n1-lease", "the cutover: the description switches node 1's executor on"
    with pytest.raises(submit.Refused, match="gpu_executor"):
        submit.resolve(_n1_executor(c, ""), _job(gpus=1, node="vy-nebius-1"), "r1", T)
    out = submit.resolve(c, _job(gpus=1, cpus=4, memory_gib=64, node="vy-nebius-1"), "r1", T)
    assert out["node"] == "vy-nebius-1" and "systemd-run" in out["wrap"] and submit.GPU_LEASE_N1 not in out["wrap"]
    assert "CPUWeight=idle" not in out["wrap"], "a GPU job's host threads take the CPUs a CPU job yields"
    assert out["workload_wrap"] == ["env", "GPU_LEASE_DIR=/run/gpu-lease", "GPU_LEASE_ALLOWED_FILE=/run/gpu-lease/pool.gpus",
                                    submit.GPU_LEASE_N1, "1", "--wait", "--max-min", "10", "--mem-gb", "64", "--preemptible", "--"]
    with pytest.raises(submit.Refused, match="no timed windows"):
        submit.resolve(c, _job(gpus=1, quiet=True, node="vy-nebius-1"), "r1", T)


def test_node_1_refuses_a_lease_that_isnt_preemptible_beyond_the_provers_own_quota():
    """n1_lease.py borrows past `provers`' nominal quota only for preemptible waiters, so a larger non-preemptible lease would wait
    in gpu-lease forever (circuits' 8-GPU load-time run, 7 Oct)."""
    c = model.load(NEBIUS)
    own = c.workstream(submit.N1_POOL_QUEUE, "vy-nebius-1").gpus
    with pytest.raises(submit.Refused, match=f"isn't --preemptible at most provers' own quota, {own} GPUs"):
        submit.resolve(c, _job(gpus=own + 1, memory_gib=64, preempt="never", node="vy-nebius-1"), "r1", T)
    assert submit.resolve(c, _job(gpus=own, memory_gib=64, preempt="never", node="vy-nebius-1"), "r1", T)["node"] == "vy-nebius-1"
    assert submit.resolve(c, _job(gpus=8, memory_gib=64, node="vy-nebius-1"), "r1", T)["node"] == "vy-nebius-1", \
        "a preemptible lease may borrow the rest"


def test_an_unpinned_gpu_job_skips_node_1s_pool_which_the_submit_path_cant_see():
    c = model.load(NEBIUS)
    out = submit.resolve(c, _job(gpus=1, cpus=4, memory_gib=64), "r1", T)
    assert out["node"] == "vy-nebius-2" and submit.UNSEEN_POOL in out["why"]
    with pytest.raises(submit.Refused, match="pinned there") as e:
        submit.resolve(c, _job(gpus=1, preempt="never"), "r1", T)
    assert "must be preemptible" in str(e.value), "node 2's reason too, so the lane learns it may pin node 1"
    assert submit.resolve(c, _job(cpus=4), "r1", T)["node"] == "vy-nebius-1", "a CPU job still goes home"


def test_the_cli_prints_the_placement_or_the_refusal(capsys):
    base = ["submit", NEBIUS, "--run-id", "r1", "--now", str(T), "--project", "verity", "--submitter", "lane-x", "--max-s", "600",
            "--custody-s", "900"]
    assert cli.main([*base, "--gpus", "1", "--gpu-model", "rtx-pro-6000", "--mem-gb", "64"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["machine"] == "vy-nebius-2" and out["job"]["gpu_model"] == "rtx-pro-6000"
    assert cli.main([*base, "--gpus", "1", "--mem-gb", "64", "--preempt", "never"]) == 3
    assert "preemptible" in json.loads(capsys.readouterr().out)["refused"]
    assert cli.main(["submit", NEBIUS, "--run-id", "r2", "--now", str(T), "--project", "verity", "--submitter", "w",
                     "--kind", "lean-audit"]) == 0
    assert json.loads(capsys.readouterr().out)["max_s"] == 1200
    assert cli.main(["submit", NEBIUS, "--run-id", "r3", "--now", str(T), "--project", "verity", "--submitter", "w",
                     "--kind", "lean-audit", "--node", "vy-nebius-1", "--gpu-host"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["gpu_host"] is True and "CPUWeight=idle" not in out["wrap"]


KINDS = Path(__file__).parents[1] / "kinds"


def test_the_repos_kinds_load_and_lean_audit_is_leans_cpu_m():
    reg = kinds.load(KINDS)
    k = reg["lean-audit"]
    assert (k.owner, k.phase, k.max_wall_min, k.restart) == ("lean", "cpu-m", 20, "idempotent") and k.question
    assert k.source["file"] == "kinds/lean.toml" and len(k.source["sha256"]) == 64
    assert kinds.kinds_dir(NEBIUS) == KINDS.resolve()


def test_a_registered_kind_supplies_what_the_job_leaves_unsaid_and_names_itself_in_the_lease():
    out = submit.submit(model.load(NEBIUS), kinds.load(KINDS), run_id="r1", now=T, kind="lean-audit", project="verity",
                        submitter="proofs-w", custody_s=900)
    assert (out["node"], out["max_s"], out["job"]["cpus"], out["job"]["memory_gib"], out["job"]["max_s"]) == \
        ("vy-nebius-1", 1200, 8, 32, 2100)
    assert out["kind"] == "lean-audit" and out["question"].startswith("Do core's Lean proofs") and out["warnings"] == []
    assert out["kind_source"]["file"] == "kinds/lean.toml" and "GPU_LEASE_WHO=lean-audit:proofs-w" in out["wrap"]
    assert out["disk_gb"] == kinds.DISK_GB


def test_the_held_rules_warn_and_never_refuse(tmp_path):
    reg = kinds.load(KINDS)
    c = model.load(NEBIUS)
    with pytest.raises(submit.Refused, match="longest run"):
        submit.submit(c, reg, run_id="r1", now=T, project="verity", submitter="w", memory_gib=8)
    out = submit.submit(c, reg, run_id="r1", now=T, project="verity", submitter="w", max_s=600, memory_gib=8)
    assert any("adhoc" in w for w in out["warnings"]) and any("no research question" in w for w in out["warnings"])
    out = submit.submit(c, reg, run_id="r1", now=T, kind="lean-audit", project="verity", submitter="w", cpus=4, max_s=3600)
    assert any("not its phase cpu-m" in w for w in out["warnings"]) and any("over its max_wall_min" in w for w in out["warnings"])
    (tmp_path / "x.toml").write_text('[kinds.big]\nphase = "gpu1"\nmax_wall_min = 90\nquestion = "q"\n')
    w = kinds.warnings(kinds.load(tmp_path)["big"], "big", question="q", gpus=1, cpus=8, memory_gib=64, max_s=5400, preempt="never")
    assert w == ["a GPU job over 60 min: split it into chunks", "a GPU kind that isn't preemptible", "declares no outputs"]


def test_a_job_of_an_unregistered_kind_names_its_memory_cap():
    c, reg = model.load(NEBIUS), kinds.load(KINDS)
    with pytest.raises(submit.Refused, match=r"--mem-gb G.*'adhoc' isn't registered.*MemoryMax"):
        submit.submit(c, reg, run_id="r1", now=T, project="verity", submitter="w", max_s=600)
    out = submit.submit(c, reg, run_id="r1", now=T, project="verity", submitter="w", max_s=600, memory_gib=96)
    assert out["job"]["memory_gib"] == 96 and "MemoryMax=96G" in out["wrap"]
    out = submit.submit(c, reg, run_id="r1", now=T, kind="lean-audit", project="verity", submitter="w")
    assert out["job"]["memory_gib"] == 32, "a registered kind's shape still gives its memory"


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
