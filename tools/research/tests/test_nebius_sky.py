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
    assert (q["circuits"]["nvidia.com/gpu"], q["provers"]["nvidia.com/gpu"]) == (5, 3)       # tonight: coverage is the binding workstream
    assert (q["circuits"]["cpu"], q["provers"]["cpu"]) == (112, 64)           # checks and pinned benches run outside Kueue
    for res, cap in NODE.items():
        assert sum(x[res] for x in q.values()) <= cap, res
    assert sum(x["memory"] for x in q.values()) + CONTROLLER["memory"] <= NODE["memory"]
    text = (SKY / "kueue.yaml").read_text()
    circuits = next(d for d in documents(SKY / "kueue.yaml") if "metadata: {name: circuits}" in d and "ClusterQueue" in d)
    assert "{name: nvidia.com/gpu, nominalQuota: 5, borrowingLimit: 2}" in circuits and "borrowWithinCohort: {policy: Never}" in circuits
    provers = text.split("metadata: {name: provers}")[1].split("\n---\n")[0]
    assert provers.count("borrowingLimit: 0") == 3 == provers.count("nominalQuota")              # a borrowed bench is evicted by reclaim
    assert text.count("    withinClusterQueue: Never") == 2 and "LowerPriority" not in text   # never evict a running Build
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
    for direct, want in ((4, 0), (0, 3), (2, 1)):
        out = subprocess.run(["python3", str(tmp_path / "render.py"), str(SKY / "kueue.yaml"), str(direct)], capture_output=True, text=True, check=True).stdout
        provers = out.split("metadata: {name: provers}")[1]
        assert re.search(r"\{name: nvidia.com/gpu, nominalQuota: (\d+)", provers).group(1) == str(want)
        assert out.split("metadata: {name: provers}")[0] == (SKY / "kueue.yaml").read_text().split("metadata: {name: provers}")[0]
    pouw = subprocess.run(["python3", str(tmp_path / "render.py"), str(SKY / "kueue-pouw.yaml"), "0"], capture_output=True, text=True, check=True).stdout
    assert pouw == (SKY / "kueue-pouw.yaml").read_text()


def test_the_coverage_build_task_has_no_gpu_but_finds_the_hosts_cuda_driver():
    """A GPU-less pod has no driver libraries, yet vLLM's CUDA platform (#536) loads libcuda.so.1 at import: the build task puts the
    host's copy, which cluster_up.sh maintains, on LD_LIBRARY_PATH."""
    head, build, gpu = (SKY / "jobs" / "config-run.yaml").read_text().split("\n---\n")
    assert "accelerators" not in build and "LD_LIBRARY_PATH=/workspace/jobs/cuda-driver" in build
    assert "accelerators: RTXPRO6000:1" in gpu and "cuda-driver" not in gpu
    assert "/workspace/jobs/cuda-driver" in (SKY / "cluster_up.sh").read_text()


def test_each_coverage_task_is_one_stage_attempt_whose_tool_reads_the_row():
    """A two-task cell publishes what `row chain` does: each task's `row stage` is the workload of one `research run --tool vllm.<stage>`
    Attempt, as plain argv (a `bash -c` string hides the row from the Tool's key params), and the Commit cites the Build's artifact."""
    import shlex

    from research.store import tools_registry
    repo = Path(__file__).resolve().parents[3]
    row = "smollm2-135m__bf16__rtxpro6000__tp1__b2__i64__o16__single__greedy__bi-eager"
    env = {"PY": "/venv/bin/python", "ROW": row, "ROLE": "SMOLLM2", "REPO": "HuggingFaceTB/SmolLM2-135M", "REVISION": "abc123"}
    _, build, gpu = (SKY / "jobs" / "config-run.yaml").read_text().split("\n---\n")
    for stage, task in (("build", build), ("commit", gpu)):
        run = task[task.index("\nrun: |"):task.index("\nconfig:")].replace("\\\n", " ")
        line = next(ln for ln in run.splitlines() if "-m research run" in ln)
        assert f'--tool vllm.{stage} --project verity --cwd "$SRC"' in line and "bash -c" not in line
        argv = line.split(" -- ", 1)[1]
        for k, v in env.items():
            argv = argv.replace(f'"${k}"', v)
        _, key = tools_registry.load(f"vllm.{stage}", repo).resolve_params(shlex.split(argv), repo)
        assert (key["stage"], key["row"], key["batch"], key["revision"]) == (stage, row, 2, "abc123")
    assert "two-task-build-run" in build and "two-task-build-run" in gpu and '--input "build=$build_art"' in gpu


def test_gpu_tasks_keep_compiled_kernels_per_tree_on_the_host(tmp_path):
    """Each cell's Commit otherwise JIT-compiles vLLM's batch-invariant Triton kernels and rebuilds hidden_gpu_tree again (most of a
    small cell's 436 s GPU hold; 83-88 s with both). Eager rows only for the Triton cache, and the shared JIT dirs only for trees with
    #561's staged sources."""
    import subprocess
    for f in ("config-run.yaml", "config-run-row.yaml"):
        gpu = (SKY / "jobs" / f).read_text().split("\n---\n")[-1]
        run = gpu[gpu.index("\nrun: |"):gpu.index("\nconfig:")]
        start = run.index('  case "$ROW" in *__bi-eager)')
        block = "\n".join(ln[2:] for ln in run[start:].split("\n")[:4])
        assert run.index("job_tree.sh\" --mine") < start and block.endswith("fi"), f
        for row, fixed, want in (("m__bi-eager", True, "/workspace/jobs/cache/triton/T|/workspace/jobs/cache/jit/hidden_gpu"),
                                 ("m__bi-compiled", True, "|/workspace/jobs/cache/jit/hidden_gpu"),
                                 ("m__bi-eager", False, "/workspace/jobs/cache/triton/T|")):
            tree = tmp_path / ("fixed" if fixed else "old") / "T"
            jit = tree / "integrations/vllm/verity_vllm/commit/committer/native_jit.py"
            jit.parent.mkdir(parents=True, exist_ok=True)
            jit.write_text("def staged_sources(): ...\n" if fixed else "def jit_load(): ...\n")
            out = subprocess.run(["bash", "-c", block + '\necho "${TRITON_CACHE_DIR:-}|${HIDDEN_GPU_BUILD:-}"'], capture_output=True,
                                 text=True, check=True, env={"PATH": "/usr/bin:/bin", "ROW": row, "SRC": str(tree)}).stdout.strip()
            assert out == want, (f, row, fixed, out)


def test_submit_caps_what_waits_in_kueue_until_the_dispatcher_runs(tmp_path):
    """A job waiting for quota holds one of SkyPilot's 8 launch slots: at most 4 coverage cells and 6 jobs in all may wait."""
    import subprocess
    sub = (SKY / "submit.sh").read_text()
    (tmp_path / "waiting.py").write_text(sub[sub.index("\n", sub.index("<<'WAITING'")) + 1:sub.index("\nWAITING\n")])
    assert sub.index("<<'WAITING'") > sub.index("<<'PY'") and sub.index("<<'WAITING'") < sub.index("sky jobs launch -y")

    def check(template, queues, finished=0):
        items = [{"spec": {"queueName": q}, "status": {}} for q in queues]
        items += [{"spec": {"queueName": "circuits"}, "status": {"admission": {"clusterQueue": "circuits"}}}] * 3
        items += [{"spec": {"queueName": "circuits"}, "status": {"conditions": [{"type": "Finished", "status": "True"}]}}] * finished
        (tmp_path / "wl.json").write_text(json.dumps({"items": items}))
        return subprocess.run(["python3", str(tmp_path / "waiting.py"), str(tmp_path / "wl.json"), template, "4", "6"],
                              capture_output=True, text=True)
    assert check("config-run", ["circuits"] * 3, finished=5).returncode == 0
    r = check("config-run", ["circuits"] * 4)
    assert r.returncode == 4 and "4 coverage cells" in r.stderr
    assert check("prover-bench", ["circuits"] * 4).returncode == 0                 # the cell cap is coverage's; provers has its own slot
    assert check("prover-bench", ["circuits"] * 4 + ["provers"] * 2).returncode == 4
    (tmp_path / "wl.json").write_text("not json")
    assert subprocess.run(["python3", str(tmp_path / "waiting.py"), str(tmp_path / "wl.json"), "config-run", "4", "6"]).returncode == 0


def test_every_template_runs_from_a_job_tree_and_bootstraps_outside_it():
    """uid 1000 can't write the synced tree, so every task runs from a copy. vLLM tasks share one copy per tree content (their taps
    then compile once) and bootstrap through vllm_bootstrap.sh; prover tasks, which write into their tree, keep one copy per pod."""
    for f in sorted((SKY / "jobs").glob("*.yaml")):
        t = f.read_text()
        tasks = t.count("\nsetup: |")
        if f.stem.startswith("prover-"):
            assert t.count('JOB_SRC=/workspace/jobs/src/$(hostname)') == tasks >= 1, f.name
            assert t.count("export SRC=/workspace/jobs/src/$(hostname)") == tasks, f.name
            continue
        assert t.count('job_tree.sh" "$SRC")') == t.count('job_tree.sh" --mine)') == tasks >= 1, f.name
        assert "pod_bootstrap.sh" not in t and t.count('vllm_bootstrap.sh" ') == tasks, f.name
        for line in (ln for ln in t.splitlines() if 'vllm_bootstrap.sh" ' in ln):
            assert re.search(r'--out "?/workspace/jobs/', line), (f.name, line)
    capture = (SKY / "jobs" / "port-capture.yaml").read_text()
    assert 'vllm_bootstrap.sh" --cpu --out' in capture and "pod-$(hostname)" in capture     # no native taps; $CMD gets its own copy


def test_a_job_tree_is_one_copy_per_content_and_a_passed_bootstrap_skips_the_lock(tmp_path):
    """job_tree.sh: two pods with the same synced tree get the same copy, made once; a changed file makes a new one. vllm_bootstrap.sh:
    the second call with the same tree, arguments and venv skips pod_bootstrap.sh without opening the host-wide lock."""
    import os
    import shutil
    import subprocess
    synced = tmp_path / "synced"
    sky = synced / "tools/research/src/research/pods/nebius/sky"
    sky.mkdir(parents=True)
    for s in ("job_tree.sh", "vllm_bootstrap.sh"):
        shutil.copy(SKY / s, sky / s)
    ops = synced / "integrations/vllm/verity_vllm/ops"
    ops.mkdir(parents=True)
    (ops / "pod_bootstrap.sh").write_text('echo run >> "$WORK/count"; echo BOOTSTRAP-OK\n')
    jobs, trees = tmp_path / "jobs", tmp_path / "jobs" / "src"
    (jobs / "venv312/lib/python3.12/site-packages/torch").mkdir(parents=True)
    env = {**os.environ, "VY_JOB_TREES": str(trees), "VY_JOBS": str(jobs), "HF_HOME": str(tmp_path / "hf")}

    def tree(src: Path) -> str:
        return subprocess.run(["bash", str(src / "tools/research/src/research/pods/nebius/sky/job_tree.sh"), str(src)], env=env,
                              capture_output=True, text=True, check=True).stdout.strip()

    a, b = tree(synced), tree(synced)
    assert a == b and Path(a, ".copied").exists() and Path(a).parent == trees
    assert subprocess.run(["bash", str(sky / "job_tree.sh"), "--mine"], env=env, capture_output=True, text=True, check=True).stdout.strip() == a
    (synced / "integrations/vllm/new.py").write_text("x = 1\n")
    assert tree(synced) != a

    boot = Path(a) / "tools/research/src/research/pods/nebius/sky/vllm_bootstrap.sh"
    first = subprocess.run(["bash", str(boot), "--cases", "B0", "--cpu", "--out", str(jobs / "b")], env=env, capture_output=True, text=True, check=True)
    assert first.stdout.strip().endswith("BOOTSTRAP-OK") and (jobs / "count").read_text() == "run\n"
    (jobs / ".vllm-bootstrap.lock").unlink()
    again = subprocess.run(["bash", str(boot), "--cases", "B0", "--cpu", "--out", str(jobs / "b")], env=env, capture_output=True, text=True, check=True)
    assert "cached" in again.stdout and again.stdout.strip().endswith("BOOTSTRAP-OK")
    assert (jobs / "count").read_text() == "run\n" and not (jobs / ".vllm-bootstrap.lock").exists()      # no rerun, no lock taken
    other = subprocess.run(["bash", str(boot), "--cases", "B1", "--cpu", "--out", str(jobs / "b")], env=env, capture_output=True, text=True, check=True)
    assert "cached" not in other.stdout and (jobs / "count").read_text() == "run\nrun\n"                  # another role bootstraps


def test_submit_names_the_tools_it_is_missing(tmp_path):
    import os
    import shutil
    import subprocess
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("bash", "git", "python3", "ssh", "curl", "dirname", "sed", "cat", "tr", "head", "awk", "grep"):
        if path := shutil.which(tool):
            os.symlink(path, bin_dir / tool)
    r = subprocess.run(["bash", str(SKY / "submit.sh"), "prover-dev", "x", "--allow-stale"], capture_output=True, text=True,
                       env={"PATH": str(bin_dir), "HOME": str(tmp_path), "VY_INFRA_REMOTE": str(tmp_path / "no-remote")})  # no network
    assert r.returncode == 2 and "missing on this machine: uv rsync" in r.stderr


def test_submit_refuses_templates_that_differ_from_infra_nebius(tmp_path):
    import os
    import shutil
    import subprocess
    rel = "tools/research/src/research/pods/nebius/sky"
    git = lambda cwd, *a: subprocess.run(["git", "-C", str(cwd), *a], check=True, capture_output=True, text=True).stdout  # noqa: E731
    work = tmp_path / "work"
    shutil.copytree(SKY, work / rel, ignore=shutil.ignore_patterns("__pycache__"))
    git(tmp_path, "init", "-q", "--bare", "origin.git")
    git(work.parent, "init", "-q", "-b", "infra/nebius", str(work))
    for k, v in (("user.email", "t@example.com"), ("user.name", "t")):
        git(work, "config", k, v)
    git(work, "add", "-A")
    git(work, "commit", "-qm", "shared infra")
    git(work, "remote", "add", "origin", str(tmp_path / "origin.git"))
    git(work, "push", "-q", "origin", "infra/nebius")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("bash", "git", "python3", "ssh", "curl", "dirname", "sed", "cat", "tr", "head", "awk", "grep", "realpath", "ls", "uv"):
        if path := shutil.which(tool):
            os.symlink(path, bin_dir / tool)
    env = {"PATH": str(bin_dir), "HOME": str(tmp_path)}                    # no rsync: a submission that passes the check stops at it

    def submit(*args):
        return subprocess.run(["bash", str(work / rel / "submit.sh"), "prover-dev", "x", *args], capture_output=True, text=True, env=env)

    r = submit()
    assert r.returncode == 2 and "missing on this machine" in r.stderr and "STALE" not in r.stderr, r.stderr
    tip = git(work, "log", "-1", "--format=%h").strip()
    template = work / rel / "jobs" / "prover-dev.yaml"
    template.write_text(template.read_text().replace("cpus: 8", "cpus: 9"))
    r = submit()
    assert r.returncode == 3 and "STALE TEMPLATES" in r.stderr and "jobs/prover-dev.yaml" in r.stderr and tip in r.stderr
    r = submit("--allow-stale")
    assert r.returncode == 2 and "--allow-stale: submitting anyway" in r.stderr
    git(work, "checkout", "-q", "--", ".")
    (work / rel / "jobs" / "mine.yaml").write_text("name: mine\n")
    r = subprocess.run(["bash", str(work / rel / "submit.sh"), "mine", "x"], capture_output=True, text=True, env=env)
    assert r.returncode == 3 and "not on origin/infra/nebius (a private copy)" in r.stderr
    other = tmp_path / "other"                                            # the shared branch moves on: this checkout is behind
    git(tmp_path, "clone", "-q", "-b", "infra/nebius", str(tmp_path / "origin.git"), str(other))
    for k, v in (("user.email", "t@example.com"), ("user.name", "t")):
        git(other, "config", k, v)
    (other / rel / "jobs" / "prover-bench.yaml").write_text("changed upstream\n")
    git(other, "commit", "-qam", "a newer template")
    git(other, "push", "-q", "origin", "infra/nebius")
    r = submit()
    assert r.returncode == 3 and "a newer template" in r.stderr and "git merge origin/infra/nebius" in r.stderr


def test_every_task_has_the_venvs_tools_on_path_and_captures_skip_the_flashinfer_sampler():
    for f in sorted((SKY / "jobs").glob("*.yaml")):
        t = f.read_text()
        runs = t.split("\nrun: |\n")[1:]
        assert runs and all("export PATH=/workspace/jobs/venv312/bin:$PATH" in r.split("\nconfig:")[0] for r in runs), f.name
    capture = (SKY / "jobs" / "port-capture.yaml").read_text()
    assert 'VLLM_USE_FLASHINFER_SAMPLER: "0"' in capture.split("\nsecrets:")[0]           # an env default, so --env overrides it


def test_a_preempted_job_can_still_publish_its_attempt():
    for f in sorted((SKY / "jobs").glob("*.yaml")):
        t = f.read_text()
        tasks = t.count("\nrun: |")
        assert t.count("terminationGracePeriodSeconds: 300") == tasks, f.name        # the default 30 s killed a publish mid-way
        assert t.count("RESEARCH_STORE: /workspace/jobs/store") == tasks, f.name    # the pod's own home dies with the pod


def test_backfill_takes_its_credentials_on_stdin_only():
    import subprocess
    script = SKY / "backfill_attempts.sh"
    text = script.read_text()
    assert "head -n 4 > \"$creds\"" in text and "R2_ENDPOINT=" not in text.replace("export R2_ENDPOINT", "")
    r = subprocess.run(["bash", str(script)], input="only-one-line\n", capture_output=True, text=True)
    assert r.returncode == 2 and "want the 4 credential lines on stdin" in r.stderr
