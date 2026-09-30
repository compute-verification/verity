"""The Nebius cluster's SkyPilot and Kueue files (pods/nebius/sky/): the queues' quotas fit the server, every job template names a
queue and a priority class kueue.yaml defines, the quiet window's labels match, and usage_report reads Kueue v0.19's objects.
The fixtures are ClusterQueues, Workloads and Pods captured from Kueue v0.19.6 on a local k3s with the queues applied: circuits
running one Build, provers running a quiet bench and holding a dev job pending."""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

from research.pods import registry

SKY = Path(registry.__file__).resolve().parent / "nebius" / "sky"
FIX = Path(__file__).resolve().parent / "fixtures" / "nebius"
NODE = {"nvidia.com/gpu": 8, "cpu": 192, "memory": 1690 * 2**30}       # vy-nebius-1's allocatable, less the guest kernel's share
CONTROLLER = {"cpu": 2, "memory": 8 * 2**30}                          # sky-config.yaml's jobs controller


def usage_report():
    spec = importlib.util.spec_from_file_location("usage_report", SKY / "usage_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def documents(path: Path) -> list[str]:
    return [d for d in path.read_text().split("\n---\n") if "kind:" in d]


def quotas(path: Path) -> dict[str, dict[str, float]]:
    ur = usage_report()
    out = {}
    for d in documents(path):
        if "kind: ClusterQueue" in d:
            name = re.search(r"metadata: \{name: (\S+)\}", d).group(1)
            out[name] = {m[0]: ur.quantity(m[1]) for m in re.findall(r"\{name: (\S+), nominalQuota: ([^,}]+)", d)}
    return out


def test_the_queues_fit_the_server_and_config_runs_never_borrow():
    q = quotas(SKY / "kueue.yaml")
    assert set(q) == {"circuits", "provers"}
    assert (q["circuits"]["nvidia.com/gpu"], q["provers"]["nvidia.com/gpu"]) == (4, 4)       # cluster_up lowers provers' while GPUs 4-7 are direct
    assert (q["circuits"]["cpu"], q["provers"]["cpu"]) == (144, 48)
    for res, cap in NODE.items():
        assert sum(x[res] for x in q.values()) <= cap, res
    assert sum(x["memory"] for x in q.values()) + CONTROLLER["memory"] <= NODE["memory"]
    text = (SKY / "kueue.yaml").read_text()
    circuits = next(d for d in documents(SKY / "kueue.yaml") if "metadata: {name: circuits}" in d and "ClusterQueue" in d)
    assert circuits.count("borrowingLimit: 0") == 3 and "borrowingLimit" not in text.split("metadata: {name: provers}")[1]
    assert text.count("cohortName: nebius") == 2 and text.count("    reclaimWithinCohort: Any") == 2
    values = dict(re.findall(r"metadata: \{name: (\S+)\}\nvalue: (\d+)", text))
    assert sorted(values, key=lambda k: -int(values[k])) == ["capture", "sweep-night", "circuits", "prover-bench", "dev"]
    # an unquoted ": " inside a one-line value is invalid YAML (kubectl refused a description that had one); the tests read this
    # file with regexes, so check it here
    assert not [ln for ln in text.splitlines() if re.match(r"^\s*description: [^'\"].*: ", ln)]


def test_every_job_template_names_a_defined_queue_and_priority():
    text = (SKY / "kueue.yaml").read_text()
    local = set(re.findall(r"kind: LocalQueue\nmetadata: \{name: (\S+), namespace: default\}", text))
    priorities = set(re.findall(r"kind: WorkloadPriorityClass\nmetadata: \{name: (\S+)\}", text))
    seen = {}
    for f in sorted((SKY / "jobs").glob("*.yaml")):
        t = f.read_text()
        queue = re.search(r"kueue\.x-k8s\.io/queue-name: (\S+)", t).group(1)
        prio = re.search(r"kueue\.x-k8s\.io/priority-class: (\S+)", t).group(1)
        assert queue in local and prio in priorities, f.name
        assert "provision_timeout: -1" in t and "infra: ssh/vy-nebius" in t and "budget-line:" in t, f.name
        seen[f.stem] = (queue, prio)
    assert seen == {"config-run": ("circuits", "circuits"), "config-run-row": ("circuits", "circuits"),
                    "port-capture": ("circuits", "capture"), "prover-dev": ("provers", "dev"), "prover-bench": ("provers", "prover-bench")}


def test_the_config_run_holds_a_gpu_only_for_its_commit(tmp_path):
    import subprocess
    text = (SKY / "jobs" / "config-run.yaml").read_text()
    head, build, gpu = text.split("\n---\n")
    assert "accelerators" not in build and 'verity.dev/build: "true"' in build and "CUDA_VISIBLE_DEVICES=" in build
    assert "row stage build \"$ROW\" \"$ROLE\" \"$REPO\" \"$REVISION\" --config-run 1 --replay-k 460" in build
    assert "row stage commit \"$ROW\" \"$ROLE\" \"$REPO\" \"$REVISION\" --config-run 1 --replay-k 460" in gpu
    assert "accelerators: RTXPRO6000:1" in gpu and "row stage match" not in text                     # config runs have no Match
    assert build.count("kueue.x-k8s.io/queue-name: circuits") == gpu.count("kueue.x-k8s.io/queue-name: circuits") == 1
    assert 'ov.noisy: "true"' in gpu and "--out" in build and "--out" in gpu                          # bootstrap writes outside the tree
    for part in (build, gpu):
        assert "GUARD_PRESERVE_FILE: /workspace/jobs/dm/guard.preserve" in part and "HOT_ROOT: /workspace/jobs/hot" in part
    sub = (SKY / "submit.sh").read_text()
    (tmp_path / "cls.py").write_text(sub[sub.index("<<'PY'\n") + 7:sub.index("\nPY\n")])
    def render(*args):
        out = subprocess.run(["python3", str(tmp_path / "cls.py"), str(SKY / "jobs" / "config-run.yaml"), *args],
                             capture_output=True, text=True, check=True).stdout
        return out.split("\n---\n")[1:]
    b2, g2 = render("tp2", "", "")
    assert "  memory: 160" in b2 and "  memory: 384" in g2 and "accelerators: RTXPRO6000:2" in g2
    b3, g3 = render("moe", "300", "")                                                                  # a per-cell figure wins
    assert "  memory: 300" in b3 and 'BUILD_RAM_BUDGET_GB: "300"' in b3 and "  memory: 384" in g3


def test_benches_are_labelled_noisy_and_the_quiet_hour_holds_circuits():
    bench = (SKY / "jobs" / "prover-bench.yaml").read_text()
    assert 'ov.noisy: "true"' in bench and 'OV_NOISY: "true"' in bench and "podAntiAffinity" not in bench
    quiet = (SKY / "quiet_hour.sh").read_text()
    assert "clusterqueue circuits" in quiet and '"stopPolicy\\":\\"$policy' in quiet
    up = (SKY / "cluster_up.sh").read_text()
    assert "hold:12:30 release:13:30" in up and "quiet_hour.sh" in up


def test_jobs_write_only_under_workspace_jobs_and_set_their_thread_count():
    for f in sorted((SKY / "jobs").glob("*.yaml")):
        t = f.read_text()
        cpus = re.search(r"^  cpus: (\d+)", t, re.M).group(1)
        assert f'OMP_NUM_THREADS: "{cpus}"' in t, f.name                                  # SkyPilot's runner sets 1
        assert "RESEARCH_RUNS: /workspace/jobs/runs" in t and "workdir" not in t, f.name
        assert "--python 3.12" in t, f.name                                               # the image's python is 3.10


def test_the_pouw_queue_joins_the_same_cohort_on_its_own_server():
    text = (SKY / "kueue-pouw.yaml").read_text()
    assert "cohortName: nebius" in text and "nodeLabels: {verity.dev/server: vy-nebius-2}" in text
    values = dict(re.findall(r"metadata: \{name: (\S+)\}\nvalue: (\d+)", text))
    assert int(values["pouw-timed"]) > int(values["pouw"]) and "withinClusterQueue: LowerPriority" in text
    up = (SKY / "cluster_up.sh").read_text()
    assert "VY_QUEUES=kueue-pouw.yaml VY_DIRECT_GPUS=none" in up and "lendingLimit: 0" in text     # node 2: its own cluster, same code


def test_usage_report_reads_kueue_objects():
    ur = usage_report()
    load = lambda n: json.loads((FIX / f"{n}.json").read_text())      # noqa: E731
    util = ur.parse_dcgm('DCGM_FI_DEV_GPU_UTIL{gpu="0",UUID="GPU-x",pod="bench-q",namespace="default",container="c"} 40\n'
                         'DCGM_FI_DEV_GPU_UTIL{gpu="1",UUID="GPU-y",pod="bench-q",namespace="default",container="c"} 60\n'
                         'DCGM_FI_DEV_GPU_UTIL{gpu="2",UUID="GPU-z"} 0\n')
    assert util == {("default", "bench-q"): 50.0}
    created = max(w["metadata"]["creationTimestamp"] for w in load("workloads")["items"])
    now = ur.time.mktime(ur.time.strptime(created, "%Y-%m-%dT%H:%M:%SZ")) - ur.time.timezone + 120
    q = ur.queues(load("clusterqueues"), load("workloads"), load("pods"), util, now)
    assert q["circuits"]["nominal"] == {"nvidia.com/gpu": 4, "cpu": 142, "memory": 1280 * 2**30}
    assert (q["circuits"]["usage"]["nvidia.com/gpu"], q["circuits"]["admitted"], q["circuits"]["pending"]) == (1, 1, 0)
    assert (q["provers"]["admitted"], q["provers"]["pending"], q["provers"]["gpu_util_mean"]) == (1, 1, 50.0)
    assert q["provers"]["oldest_pending_s"] == 120


def test_usage_summary_suggests_the_p90_as_nominal():
    ur = usage_report()
    gi = 2**30
    lines = [{"t": 1000 + i, "queues": {"provers": {"nominal": {"nvidia.com/gpu": 4, "cpu": 46, "memory": 384 * gi},
                                                     "usage": {"nvidia.com/gpu": g, "cpu": 10 * g, "memory": 100 * gi * g},
                                                     "borrowed": {"nvidia.com/gpu": max(0, g - 4), "cpu": 0, "memory": 0},
                                                     "admitted": g, "pending": int(g == 6), "oldest_pending_s": 600 * (g == 6),
                                                     "gpu_util_mean": 5.0}}}
             for i, g in enumerate([1, 2, 2, 3, 3, 3, 4, 4, 5, 6])]
    out = ur.summarize(lines, 7, now=2000).splitlines()
    gpu = next(r for r in out if r.startswith("provers    GPU"))
    assert gpu.split() == ["provers", "GPU", "4", "3.3", "5.0", "1.0", "5"]
    mem = next(r for r in out if r.startswith("provers    GiB"))
    assert mem.split()[-1] == "512"
    assert "work waited in 10% of samples (longest 10 min)" in out[-2] and "GPU utilization of its pods 5%" in out[-2]
    assert ur.summarize(lines, 7, now=10**9) == "no samples in the last 7 days"


def test_provers_gives_up_one_gpu_per_gpu_held_for_direct_runs(tmp_path):
    import subprocess
    up = (SKY / "cluster_up.sh").read_text()
    render = up[up.index("<<'PY'\n") + len("<<'PY'\n"):up.index("\nPY\n")]
    (tmp_path / "render.py").write_text(render)
    for direct, want in ((4, 0), (0, 4), (6, 0)):
        out = subprocess.run(["python3", str(tmp_path / "render.py"), str(SKY / "kueue.yaml"), str(direct)], capture_output=True, text=True, check=True).stdout
        provers = out.split("metadata: {name: provers}")[1]
        assert re.search(r"\{name: nvidia.com/gpu, nominalQuota: (\d+)", provers).group(1) == str(want)
        assert out.split("metadata: {name: provers}")[0] == (SKY / "kueue.yaml").read_text().split("metadata: {name: provers}")[0]
    pouw = subprocess.run(["python3", str(tmp_path / "render.py"), str(SKY / "kueue-pouw.yaml"), "0"], capture_output=True, text=True, check=True).stdout
    assert pouw == (SKY / "kueue-pouw.yaml").read_text()


def test_every_template_runs_from_a_job_private_tree_and_bootstraps_outside_it():
    for f in sorted((SKY / "jobs").glob("*.yaml")):
        t = f.read_text()
        tasks = t.count("\nsetup: |")
        assert t.count('JOB_SRC=/workspace/jobs/src/$(hostname)') == tasks >= 1, f.name        # uid 1000 can't write the synced tree
        assert t.count("export SRC=/workspace/jobs/src/$(hostname)") == tasks, f.name
        for line in (ln for ln in t.replace("\\\n", " ").splitlines() if "pod_bootstrap.sh" in ln):
            assert "env -u REPO" in line and re.search(r'--out "?/workspace/jobs/', line), (f.name, line)
    capture = (SKY / "jobs" / "port-capture.yaml").read_text()
    assert "pod_bootstrap.sh --cpu --out" in capture                                           # a capture needs no native taps


def test_submit_names_the_tools_it_is_missing(tmp_path):
    import os
    import shutil
    import subprocess
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("bash", "git", "python3", "ssh", "curl", "dirname", "sed", "cat", "tr", "head", "awk", "grep"):
        if path := shutil.which(tool):
            os.symlink(path, bin_dir / tool)
    r = subprocess.run(["bash", str(SKY / "submit.sh"), "prover-dev", "x"], capture_output=True, text=True,
                       env={"PATH": str(bin_dir), "HOME": str(tmp_path)})
    assert r.returncode == 2 and "missing on this machine: uv rsync" in r.stderr
