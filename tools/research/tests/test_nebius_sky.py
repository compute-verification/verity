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
            out[name] = {m[0]: ur.quantity(m[1].strip("\"")) for m in re.findall(r"\{name: (\S+), nominalQuota: ([^,}]+)", d)}
    return out


def test_the_queues_fit_the_server_and_config_runs_never_borrow():
    q = quotas(SKY / "kueue.yaml")
    assert set(q) == {"deployments-gpu", "deployments-cpu", "circuits", "provers", "backfill"}
    assert set(q["backfill"].values()) == set(q["circuits"].values()) == {0}                          # backfill borrows; circuits drains
    assert (q["deployments-gpu"]["nvidia.com/gpu"], q["deployments-cpu"]["nvidia.com/gpu"], q["provers"]["nvidia.com/gpu"]) == (6, 0, 2)
    assert q["deployments-gpu"]["cpu"] == 4 * q["deployments-gpu"]["nvidia.com/gpu"]                   # 4 vCPU per GPU: never waits on CPU
    assert (q["deployments-cpu"]["cpu"], q["provers"]["cpu"]) == (100, 64)
    assert sum(x["cpu"] for x in q.values()) + CONTROLLER["cpu"] <= NODE["cpu"]              # every admitted pod can be placed           # checks and pinned benches run outside Kueue
    for res, cap in NODE.items():
        assert sum(x[res] for x in q.values()) <= cap, res
    assert sum(x["memory"] for x in q.values()) + CONTROLLER["memory"] <= NODE["memory"]
    text = (SKY / "kueue.yaml").read_text()
    gpuq = next(d for d in documents(SKY / "kueue.yaml") if "metadata: {name: deployments-gpu}" in d and "ClusterQueue" in d)
    cpuq = next(d for d in documents(SKY / "kueue.yaml") if "metadata: {name: deployments-cpu}" in d and "ClusterQueue" in d)
    assert '{name: nvidia.com/gpu, nominalQuota: "6", borrowingLimit: "2"}' in gpuq and '{name: nvidia.com/gpu, nominalQuota: "0", borrowingLimit: "0"}' in cpuq
    for d in (gpuq, cpuq):
        assert "borrowWithinCohort: {policy: LowerPriority, maxPriorityThreshold: 10}" in d and "reclaimWithinCohort: LowerPriority" in d
    provers = text.split("metadata: {name: provers}")[1].split("\n---\n")[0]
    borrow = {r: usage_report().quantity(v) for r, v in re.findall(r"\{name: (\S+), nominalQuota: [^,]+, borrowingLimit: \"([^\"]+)\"\}", provers)}
    others = {res: sum(x[res] for name, x in q.items() if name != "provers") for res in ("nvidia.com/gpu", "cpu", "memory")}
    assert borrow == others                    # proofs borrows every idle GPU, with the CPU and memory to run them
    assert text.count("    withinClusterQueue: Never") == 5 and text.count("policy: LowerPriority") == 2   # never evict a running Build
    assert text.count("cohortName: nebius") == 5 and text.count("    reclaimWithinCohort: Any") == 1
    backfill = next(d for d in documents(SKY / "kueue.yaml") if "metadata: {name: backfill}" in d and "ClusterQueue" in d)
    assert "reclaimWithinCohort: Never" in backfill and "borrowWithinCohort: {policy: Never}" in backfill and "borrowingLimit" not in backfill
    values = dict(re.findall(r"metadata: \{name: (\S+)\}\nvalue: (\d+)", text))
    assert sorted(values, key=lambda k: -int(values[k])) == ["capture", "sweep-night", "circuits-gpu", "circuits", "prover-bench", "dev", "backfill"]
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
    assert seen == {"config-run": ("deployments-cpu", "circuits"), "config-run-row": ("deployments-gpu", "circuits-gpu"),
                    "commit-pack": ("deployments-gpu", "circuits-gpu"),
                    "port-capture": ("deployments-gpu", "capture"), "prover-dev": ("provers", "dev"), "prover-bench": ("provers", "prover-bench"),
                    "store-push": ("backfill", "backfill"), "store-fetch": ("backfill", "backfill")}


def test_the_config_run_holds_a_gpu_only_for_its_commit(tmp_path):
    import subprocess
    text = (SKY / "jobs" / "config-run.yaml").read_text()
    head, build, gpu, replay = text.split("\n---\n")
    assert "accelerators" not in replay and "queue-name: deployments-cpu" in replay and "row stage replay" in replay
    assert 'if [ "$REPLAY_DEFERRED" != 1 ]; then' in replay and "REPLAY_DEFERRED: auto" in replay and "REPLAY_DEFERRED: auto" in gpu
    assert '[ "$REPLAY_DEFERRED" != 1 ] || deferred=(--replay-deferred 1)' in gpu and '"${deferred[@]}"' in gpu
    detect = "grep -q VLLM_REPLAY"
    assert gpu.count(detect) == 1 and replay.count(detect) == 2                        # setup and run resolve it as the Commit does
    assert 'if [ "$rc" = 0 ] && [ -s "$SWEEP_DIR/$ROW/config_record.json" ]; then\n    python3 - "$SWEEP_DIR/$ROW" << \'EOF\'' in replay
    assert "rm -rf" not in replay                                     # the keep check alone deletes a bundle (the next test)
    assert "accelerators" not in build and 'verity.dev/build: "true"' in build and "CUDA_VISIBLE_DEVICES=" in build
    assert "row stage build \"$ROW\" \"$ROLE\" \"$REPO\" \"$REVISION\" --config-run 1 --replay-k 460" in build
    assert "row stage commit \"$ROW\" \"$ROLE\" \"$REPO\" \"$REVISION\" --config-run 1 --replay-k 460" in gpu
    assert "accelerators: RTXPRO6000:1" in gpu and "row stage match" not in text                     # config runs have no Match
    assert "kueue.x-k8s.io/priority-class: circuits-gpu" in gpu and "kueue.x-k8s.io/priority-class: circuits\n" in build  # Commits first
    assert "kueue.x-k8s.io/priority-class: circuits-gpu" in replay                                    # replays drain bundles before Builds
    assert build.count("kueue.x-k8s.io/queue-name: deployments-cpu") == gpu.count("kueue.x-k8s.io/queue-name: deployments-gpu") == 1
    assert 'ov.noisy: "true"' in gpu and "--out" in build and "--out" in gpu                          # bootstrap writes outside the tree
    for part in (build, gpu):
        assert "GUARD_PRESERVE_FILE: /workspace/jobs/dm/guard.preserve" in part
    sub = (SKY / "submit.sh").read_text()
    (tmp_path / "cls.py").write_text(sub[sub.index("<<'PY'\n") + 7:sub.index("\nPY\n")])
    def render(*args):
        out = subprocess.run(["python3", str(tmp_path / "cls.py"), str(SKY / "jobs" / "config-run.yaml"), *args, *[""] * (4 - len(args))],
                             capture_output=True, text=True, check=True).stdout
        return out.split("\n---\n")[1:]
    b2, g2, r2 = render("tp2", "", "")
    assert "  memory: 160" in b2 and "  memory: 340" in g2 and "accelerators: RTXPRO6000:2" in g2 and "  cpus: 8 " in g2
    b3, g3, r3 = render("moe", "300", "", "96")                                                                  # a per-cell figure wins
    assert "  memory: 300" in b3 and 'BUILD_RAM_BUDGET_GB: "300"' in b3 and "  memory: 170 " in g3
    assert "  memory: 96 " in r3 and "  memory: 64 " in r2                                                  # VY_REPLAY_MEMORY


def test_a_replay_bundle_goes_only_with_its_complete_slim_keep(tmp_path):
    """A replay task must not delete a passed row's sealed bundle as soon as config_record.json exists: a tree that predates
    keep-leaves holds nothing else of the leaves a later CPU replay reads. A bundle goes only when
    commit/replay_slim_p<pair>/ is its complete keep (cpu_replay.keep's slim.json: this bundle's MANIFEST.json, dump_sha256, every
    file's size and sha256); any doubt keeps it, with one line in row.log. config-run.yaml and n2_commit.sh run the same lines."""
    import hashlib
    import subprocess
    import textwrap

    text = (SKY / "jobs" / "config-run.yaml").read_text()
    run = textwrap.dedent(text.split("\n---\n")[3].split("\nrun: |\n", 1)[1].split("\n\nconfig:", 1)[0]) + "\n"
    try:
        import yaml
        assert [d for d in yaml.safe_load_all(text) if d and d.get("name") == "replay"][0]["run"] == run
    except ImportError:
        pass
    assert subprocess.run(["bash", "-n"], input=run, text=True).returncode == 0
    body = run.split("<< 'EOF'", 1)[1].split("\n", 1)[1].split("\nEOF\n", 1)[0]
    n2 = (SKY.parent / "n2_commit.sh").read_text()
    assert n2.split("discard_kept_bundles() {", 1)[1].split("<< 'EOF'", 1)[1].split("\n", 1)[1].split("\nEOF\n", 1)[0] == body
    assert 'then discard_kept_bundles "$row_dir"; fi' in n2 and "rm -rf \"$row_dir\"/commit/replay_bundle_p" not in n2
    (tmp_path / "discard.py").write_text(body)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

    def row(name: str, pairs=(0,), keep: dict[int, dict] | None = None) -> Path:
        """A replayed row with a sealed bundle per pair, and for each pair in `keep` its slim keep, slim.json fields overridden."""
        r = tmp_path / name
        for p in pairs:
            b = r / "commit" / f"replay_bundle_p{p}"
            (b / "store").mkdir(parents=True)
            (b / "store" / "leaves").write_bytes(b"L" * 4096)
            (b / "MANIFEST.json").write_text(json.dumps({"files": {"store/leaves": {"sha256": sha(b / "store" / "leaves"), "bytes": 4096}}}))
        for p, over in (keep or {}).items():
            k = r / "commit" / f"replay_slim_p{p}"
            (k / "store").mkdir(parents=True)
            (k / "store" / "opened").write_bytes(b"S" * 64)
            (k / "binding_map.json").write_text("{}\n")
            files = {f: {"sha256": sha(k / f), "bytes": (k / f).stat().st_size} for f in ("binding_map.json", "store/opened")}
            doc = {"schema": "verity-vllm/replay-slim/v1", "pair": p, "files": files,
                   "bundle_manifest_sha256": sha(r / "commit" / f"replay_bundle_p{p}" / "MANIFEST.json"),
                   "dump_sha256": hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), **over}
            (k / "slim.json").write_text(json.dumps(doc, separators=(",", ":")) + "\n")
        return r

    def discard(r: Path) -> list[str]:
        out = subprocess.run(["python3", str(tmp_path / "discard.py"), str(r)], capture_output=True, text=True, check=True).stdout
        assert out.splitlines() == [ln.split(" ", 1)[1] for ln in (r / "row.log").read_text().splitlines()]
        return out.splitlines()

    bundle = lambda r, p=0: r / "commit" / f"replay_bundle_p{p}"
    r = row("kept-whole", keep={0: {}})
    [line] = discard(r)
    assert not bundle(r).exists() and "deleted" in line and (r / "commit" / "replay_slim_p0" / "slim.json").is_file()
    r = row("no-keep")
    [line] = discard(r)
    assert bundle(r).is_dir() and "kept: no slim keep" in line
    r = row("two", pairs=(0, 1), keep={0: {}})
    assert [ln.split(" kept")[0].split(" deleted")[0] for ln in discard(r)] == [f"replay bundle {bundle(r, p)}" for p in (0, 1)]
    assert not bundle(r, 0).exists() and bundle(r, 1).is_dir()
    for name, over, why in [("other-bundle", {"bundle_manifest_sha256": "0" * 64}, "made from another bundle"),
                            ("dump", {"dump_sha256": "0" * 64}, "dump_sha256"),
                            ("pair", {"pair": 1}, "keep of pair 0"),
                            ("schema", {"schema": "verity-vllm/replay-slim/v0"}, "replay-slim/v1"),
                            ("empty", {"files": {}, "dump_sha256": hashlib.sha256(b"{}").hexdigest()}, "lists files")]:
        r = row(name, keep={0: over})
        [line] = discard(r)
        assert bundle(r).is_dir() and why in line, (name, line)
    r = row("changed-file", keep={0: {}})
    (r / "commit" / "replay_slim_p0" / "store" / "opened").write_bytes(b"T" * 64)
    assert "1 of" in discard(r)[0] and bundle(r).is_dir()
    r = row("missing-file", keep={0: {}})
    (r / "commit" / "replay_slim_p0" / "binding_map.json").unlink()
    assert "incomplete (FileNotFoundError" in discard(r)[0] and bundle(r).is_dir()
    r = row("blank-slim", keep={0: {}})
    (r / "commit" / "replay_slim_p0" / "slim.json").write_text("")
    assert "incomplete (JSONDecodeError" in discard(r)[0] and bundle(r).is_dir()
    r = row("partial", keep={0: {}})
    bundle(r).rename(r / "commit" / "replay_bundle_p0.partial")
    assert "not a sealed bundle" in discard(r)[0] and (r / "commit" / "replay_bundle_p0.partial").is_dir()


def test_benches_are_labelled_noisy_and_the_quiet_hour_holds_circuits():
    bench = (SKY / "jobs" / "prover-bench.yaml").read_text()
    assert 'ov.noisy: "true"' in bench and 'OV_NOISY: "true"' in bench and "podAntiAffinity" not in bench
    quiet = (SKY / "quiet_hour.sh").read_text()
    assert "QUEUES=${VY_QUIET_QUEUES:-deployments-cpu deployments-gpu circuits}" in quiet and '"stopPolicy\\":\\"$2' in quiet
    up = (SKY / "cluster_up.sh").read_text()
    assert "hold:12:30 release:13:30" in up and "quiet_hour.sh" in up


def test_every_server_script_parses():
    """A quoting slip in cluster_up.sh (an apostrophe inside a single-quoted printf) broke the whole file and nothing ran it."""
    import subprocess
    for f in sorted([*SKY.glob("*.sh"), *SKY.parent.glob("*.sh"), *(SKY.parent.parent / "sh").glob("*.sh")]):
        r = subprocess.run(["bash", "-n", str(f)], capture_output=True, text=True)
        assert r.returncode == 0, f"{f.name}: {r.stderr}"


QUEUES = ["deployments-cpu", "deployments-gpu", "provers", "backfill", "circuits"]


def stop_policy_node(tmp_path: Path, start: dict[str, str]):
    """vy-nebius-1's stopPolicy writers (disk_guard.sh, quiet_hour.sh) run against fakes of `df`, `date` and `kubectl`, the policy
    of each queue a file. Returns run(script, verb, used %, HHMM) -> stdout, and policies() -> {queue: stopPolicy}."""
    import subprocess
    bin_dir, kq = tmp_path / "bin", tmp_path / "cq"
    for d in (bin_dir, kq):
        d.mkdir()
    (bin_dir / "df").write_text('#!/bin/sh\necho "Filesystem 1024-blocks Used Available Capacity Mounted"\n'
                                'echo "/dev/vdc 100 $DF_USED $((100 - DF_USED)) x /workspace"\n')
    (bin_dir / "date").write_text('#!/bin/sh\n[ "$2" = +%H%M ] && { echo "$HM"; exit 0; }\nexec /bin/date "$@"\n')
    (bin_dir / "kubectl").write_text('#!/bin/sh\ncase $1 in\n  get) cat "$KQ/$3" ;;\n'
                                     '  patch) echo "$6" | sed \'s/.*stopPolicy":"\\([A-Za-z]*\\)".*/\\1/\' > "$KQ/$3" ;;\nesac\n')
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    for q, p in start.items():
        (kq / q).write_text(p)

    def run(script: str, verb: str, used: int = 70, hm: str = "1400") -> str:
        env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "KQ": str(kq), "DF_USED": str(used), "HM": hm, "VY_KUBECTL": str(bin_dir / "kubectl"),
               "VY_DISK_STATE": str(tmp_path / "disk-held"), "VY_DISK_OUTBOX": str(tmp_path / "outbox"),
               "VY_QUIET_STATE": str(tmp_path / "quiet-held")}
        r = subprocess.run(["sh", str(SKY / script), verb], env=env, capture_output=True, text=True)
        assert r.returncode == 0 and r.stderr == "", r.stderr
        return r.stdout

    return run, lambda: {q: (kq / q).read_text().strip() for q in start}


def test_the_disk_guard_holds_every_queue_at_80_and_releases_only_its_own_holds(tmp_path):
    """vy-disk-guard: at 80% it holds every queue that isn't held already and writes one alert note; while the disk stays full it
    re-holds a queue someone released; under 75% it releases only its own holds, and not inside the quiet hour."""
    up = (SKY / "cluster_up.sh").read_text()
    assert "disk_guard.sh\" /usr/local/bin/vy-disk-guard" in up and "vy-disk-guard.timer" in up and "OnUnitActiveSec=1min" in up
    outbox, queues = tmp_path / "outbox", QUEUES
    run, policies = stop_policy_node(tmp_path, {q: "Hold" if q == "circuits" else "None" for q in queues})  # circuits is in its quiet hour
    kq = tmp_path / "cq"

    def tick(used: int, hm: str = "1400") -> str:
        return run("disk_guard.sh", "check", used, hm)

    assert tick(79) == "" and set(policies().values()) == {"None", "Hold"} and not outbox.exists()
    assert "held deployments-cpu deployments-gpu provers backfill" in tick(80)
    assert set(policies().values()) == {"Hold"} and len(list(outbox.iterdir())) == 1
    note = next(outbox.iterdir()).read_text()
    assert "lane: infra" in note and "nothing was deleted" in note and "provers" in note
    (kq / "provers").write_text("None")                                   # released by hand while the disk is still full
    assert tick(81) == "" and policies()["provers"] == "Hold" and len(list(outbox.iterdir())) == 1
    assert tick(77) == "" and set(policies().values()) == {"Hold"}        # between the marks: nothing changes
    assert tick(70, hm="1245") == "" and set(policies().values()) == {"Hold"}   # the quiet hour releases its own
    assert "released deployments-cpu deployments-gpu provers backfill" in tick(70)
    assert policies() == {**{q: "None" for q in queues}, "circuits": "Hold"} and len(list(outbox.iterdir())) == 2


def test_the_quiet_hour_releases_only_the_holds_it_set(tmp_path):
    """vy-quiet-hour's 13:30Z release used to set None on all three of its queues, so it cleared the disk guard's hold and a lane's
    hold by hand. Now its 12:30 hold takes only the queues nobody holds, and its release frees only those, and none the disk guard
    has since taken as its own."""
    run, policies = stop_policy_node(tmp_path, {q: "Hold" if q == "deployments-gpu" else "None" for q in QUEUES})  # a lane's hold
    kq = tmp_path / "cq"
    assert run("quiet_hour.sh", "release") == "" and run("quiet_hour.sh", "status").startswith("held by the quiet hour: \n")
    out = run("quiet_hour.sh", "hold")
    assert "deployments-gpu stopPolicy=Hold, held by someone else: left alone" in out
    assert set(policies().values()) == {"None", "Hold"} and policies()["circuits"] == policies()["deployments-cpu"] == "Hold"
    run("quiet_hour.sh", "hold")
    assert (tmp_path / "quiet-held").read_text().split() == ["deployments-cpu", "circuits"]
    (kq / "circuits").write_text("None")                                 # released by hand inside the hour, and then the disk fills
    assert "held provers backfill circuits" in run("disk_guard.sh", "check", 82, "1241")
    out = run("quiet_hour.sh", "release")
    assert "deployments-cpu stopPolicy=None" in out and "circuits stopPolicy=Hold, held by the disk guard: left alone" in out
    assert policies() == {**{q: "Hold" for q in QUEUES}, "deployments-cpu": "None"}
    assert (tmp_path / "quiet-held").read_text() == ""
    assert "released provers backfill circuits" in run("disk_guard.sh", "check", 70, "1331")
    assert policies() == {**{q: "None" for q in QUEUES}, "deployments-gpu": "Hold"}

    assert "held deployments-cpu provers backfill circuits" in run("disk_guard.sh", "check", 81, "1210")    # the next day
    assert "held by someone else" in run("quiet_hour.sh", "hold") and (tmp_path / "quiet-held").read_text() == ""
    run("quiet_hour.sh", "release")
    assert set(policies().values()) == {"Hold"}
    run("disk_guard.sh", "check", 70, "1245")
    assert set(policies().values()) == {"Hold"}                         # the guard keeps its holds through the hour
    run("disk_guard.sh", "check", 70, "1330")
    assert policies() == {**{q: "None" for q in QUEUES}, "deployments-gpu": "Hold"}


def test_jobs_write_only_under_workspace_jobs_and_set_their_thread_count():
    for f in sorted((SKY / "jobs").glob("*.yaml")):
        t = f.read_text()
        for task in t.split("\n---\n"):
            if "\nresources:" not in task:
                continue
            cpus = int(re.search(r"^  cpus: (\d+)", task, re.M).group(1))
            threads = int(re.search(r'^  OMP_NUM_THREADS: "(\d+)"', task, re.M).group(1))    # SkyPilot's runner sets 1
            assert threads >= cpus, f.name                                                   # threads may burst past the request
            if "queue-name: deployments-gpu" in task:
                gpus = int(re.search(r"accelerators: RTXPRO6000:(\d+)", task).group(1))
                assert cpus == 4 * gpus, f.name                                                 # the GPU queue's fixed share
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
    for direct, want in ((4, 0), (0, 2), (1, 1)):
        out = subprocess.run(["python3", str(tmp_path / "render.py"), str(SKY / "kueue.yaml"), str(direct)], capture_output=True, text=True, check=True).stdout
        provers = out.split("metadata: {name: provers}")[1]
        assert re.search(r"\{name: nvidia.com/gpu, nominalQuota: (\d+)", provers).group(1) == str(want)
        assert out.split("metadata: {name: provers}")[0] == (SKY / "kueue.yaml").read_text().split("metadata: {name: provers}")[0]
    pouw = subprocess.run(["python3", str(tmp_path / "render.py"), str(SKY / "kueue-pouw.yaml"), "0"], capture_output=True, text=True, check=True).stdout
    assert pouw == (SKY / "kueue-pouw.yaml").read_text()


def test_the_coverage_build_task_has_no_gpu_but_finds_the_hosts_cuda_driver():
    """A GPU-less pod has no driver libraries, yet vLLM's CUDA platform loads libcuda.so.1 at import: the build task puts the
    host's copy, which cluster_up.sh maintains, on LD_LIBRARY_PATH."""
    head, build, gpu, _ = (SKY / "jobs" / "config-run.yaml").read_text().split("\n---\n")
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
    _, build, gpu, _ = (SKY / "jobs" / "config-run.yaml").read_text().split("\n---\n")
    for stage, task in (("build", build), ("commit", gpu)):
        run = task[task.index("\nrun: |"):task.index("\nconfig:")].replace("\\\n", " ")
        line = next(ln for ln in run.splitlines() if "-m research run" in ln or '"${research[@]}" run ' in ln)
        assert f'--tool vllm.{stage} --project verity --cwd "$SRC"' in line and "bash -c" not in line
        argv = line.split(" -- ", 1)[1].replace('${deferred[@]+"${deferred[@]}"}', '')
        for k, v in env.items():
            argv = argv.replace(f'"${k}"', v)
        _, key = tools_registry.load(f"vllm.{stage}", repo).resolve_params(shlex.split(argv), repo)
        assert (key["stage"], key["row"], key["batch"], key["revision"]) == (stage, row, 2, "abc123")
    assert "two-task-build-run" in build and "two-task-build-run" in gpu and '--input "build=$build_art"' in gpu


def test_gpu_tasks_keep_compiled_kernels_per_tree_on_the_host(tmp_path):
    """Each cell's Commit otherwise JIT-compiles vLLM's batch-invariant Triton kernels and rebuilds hidden_gpu_tree again (most of a
    small cell's 436 s GPU hold; 83-88 s with both). Eager rows only for the Triton cache, and the shared JIT dirs only for trees whose
    native_jit stages its sources."""
    import subprocess
    for f in ("config-run.yaml", "config-run-row.yaml"):
        gpu = next(p for p in (SKY / "jobs" / f).read_text().split("\n---\n") if "accelerators:" in p)
        run = gpu[gpu.index("\nrun: |"):gpu.index("\nconfig:")]
        start = run.index('  case "$ROW" in *__bi-eager)')
        block = "\n".join(ln[2:] for ln in run[start:].split("\n")[:4])
        assert run.index("job_tree.sh\" --mine") < start and block.endswith("fi"), f
        for row, fixed, want in (("m__bi-eager", True, "/workspace/jobs/cache/triton/T|/workspace/jobs/cache/jit/hidden_gpu"),
                                 ("m__bi-compiled", True, "|/workspace/jobs/cache/jit/hidden_gpu"),
                                 ("m__bi-eager", False, "/workspace/jobs/cache/triton/T|")):
            tree = tmp_path / ("fixed" if fixed else "old") / "T"
            jit = tree / "adapters/vllm/verity_vllm/commit/committer/native_jit.py"
            jit.parent.mkdir(parents=True, exist_ok=True)
            jit.write_text("def staged_sources(): ...\n" if fixed else "def jit_load(): ...\n")
            out = subprocess.run(["bash", "-c", block + '\necho "${TRITON_CACHE_DIR:-}|${HIDDEN_GPU_BUILD:-}"'], capture_output=True,
                                 text=True, check=True, env={"PATH": "/usr/bin:/bin", "ROW": row, "SRC": str(tree)}).stdout.strip()
            assert out == want, (f, row, fixed, out)


def test_submit_caps_what_waits_in_kueue_until_the_dispatcher_runs(tmp_path):
    """A SkyPilot job waiting for quota holds one of its 8 launch slots: at most 4 coverage cells and 6 jobs in all may wait. The
    dispatcher's Kueue Jobs hold none, so any number of them waiting blocks nothing."""
    import subprocess
    sub = (SKY / "submit.sh").read_text()
    (tmp_path / "waiting.py").write_text(sub[sub.index("\n", sub.index("<<'WAITING'")) + 1:sub.index("\nWAITING\n")])
    assert sub.index("<<'WAITING'") > sub.index("<<'PY'") and sub.index("<<'WAITING'") < sub.index("sky jobs launch -y")

    def check(template, queues, finished=0):
        pod = {"ownerReferences": [{"kind": "Pod", "name": "sky-head"}]}
        items = [{"metadata": pod, "spec": {"queueName": q}, "status": {}} for q in queues]
        items += [{"metadata": {"ownerReferences": [{"kind": "Job", "name": "nd-x"}]}, "spec": {"queueName": "deployments-gpu"},
                   "status": {}}] * 40                                                  # the dispatcher's waiting Kueue Jobs
        items += [{"spec": {"queueName": "circuits"}, "status": {"admission": {"clusterQueue": "circuits"}}}] * 3
        items += [{"spec": {"queueName": "circuits"}, "status": {"conditions": [{"type": "Finished", "status": "True"}]}}] * finished
        (tmp_path / "wl.json").write_text(json.dumps({"items": items}))
        return subprocess.run(["python3", str(tmp_path / "waiting.py"), str(tmp_path / "wl.json"), template, "4", "6"],
                              capture_output=True, text=True)
    assert check("config-run", ["circuits"] * 3, finished=5).returncode == 0
    r = check("config-run", ["circuits"] * 4)
    assert r.returncode == 4 and "4 vLLM deployments" in r.stderr
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
        if f.stem == "commit-pack":           # a pilot: each Commit it runs is config-run's gpu task script, job tree included
            assert tasks == 0 and "job_tree.sh" not in t and 'pilot=$SRC/sky/commit_pack.py' in t, f.name
            continue
        if f.stem in ("store-push", "store-fetch"):   # only import `research` from the tree, never write it
            assert tasks == 0 and "job_tree.sh" not in t and 'PYTHONDONTWRITEBYTECODE: "1"' in t, f.name
            continue
        assert t.count('job_tree.sh" "$SRC")') == t.count('job_tree.sh" --mine)') == tasks >= 1, f.name
        assert "pod_bootstrap.sh" not in t and t.count('vllm_bootstrap.sh" ') == tasks, f.name
        for line in (ln for ln in t.splitlines() if 'vllm_bootstrap.sh" ' in ln):
            assert re.search(r'--out "?/workspace/jobs/', line), (f.name, line)
    capture = (SKY / "jobs" / "port-capture.yaml").read_text()
    assert 'vllm_bootstrap.sh" --cpu --out' in capture and "pod-$(hostname)" in capture     # no native taps; $CMD gets its own copy


def test_store_fetch_lays_each_artifact_out_under_its_hex_through_a_scratch_store_it_removes(tmp_path):
    """store-fetch's run step with `research` from this checkout and a directory remote in place of R2: a tree (PATHS picking its
    files) and a file artifact land under $OUT/$NAME/<hex>/, fetched.tsv names each file's path and sha256, the scratch store is
    gone, a directory older than KEEP_DAYS goes, and an id that isn't a full art id fails before anything is fetched. An id it
    can't fetch, or whose fetch writes no file, is named and fails the job, and the others still land."""
    import hashlib
    import os
    import subprocess
    import sys
    import textwrap

    from research.store.local import LocalStore

    vault, toml, jobs, out = tmp_path / "vault", tmp_path / "store.toml", tmp_path / "jobs", tmp_path / "out"
    toml.write_text(f'[remote]\ntype = "fs"\npath = "{vault}"\n')
    (tmp_path / "repo").mkdir()
    origin = LocalStore(tmp_path / "origin", config_path=toml, repo=tmp_path / "repo")
    (tmp_path / "t" / "rep1").mkdir(parents=True)
    (tmp_path / "t" / "rep1" / "a.json").write_text("{}\n")
    (tmp_path / "t" / "b.log").write_text("log\n")
    (tmp_path / "one.bin").write_bytes(b"\x01" * 64)
    tree = origin.put_artifact("run-files/v1", {"n": 1}, tree=tmp_path / "t")
    one = origin.put_artifact("proof/v1", {"n": 2}, file=tmp_path / "one.bin")
    assert origin.push(tree)["preserved"] and origin.push(one)["preserved"]

    (jobs / "bin").mkdir(parents=True)
    (jobs / "bin" / "uv").write_text(f'#!/bin/sh\nshift 5\nexec "{sys.executable}" "$@"\n')     # `uv run --no-project --python 3.12 python`
    (jobs / "bin" / "uv").chmod(0o755)
    (out / "old").mkdir(parents=True)
    os.utime(out / "old", (0, 0))
    text = (SKY / "jobs" / "store-fetch.yaml").read_text()
    run = textwrap.dedent(text.split("\nrun: |\n", 1)[1].split("\n\nconfig:", 1)[0]).replace("/workspace/jobs", str(jobs))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("R2_", "AWS_", "RESEARCH_"))}
    env.update(SRC=str(SKY.parents[6]), RESEARCH_STORE_CONFIG=str(toml), RESEARCH_REPO=str(tmp_path / "repo"), OUT=str(out),
               KEEP_DAYS="2", NAME="console-1", ARTS=f"{tree} {one}", PATHS="rep1/*")

    r = subprocess.run(["bash", "-c", run], capture_output=True, text=True, env=env, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    dest = out / "console-1"
    files = [f"{tree[4:]}/rep1/a.json", f"{one[4:]}/one.bin"]
    assert sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()) == sorted([*files, "fetched.tsv"])
    assert not (dest / ".store").exists() and not (out / "old").exists() and "2 of 2 artifact(s), 2 file(s)" in r.stdout
    listed = [ln.split("\t") for ln in (dest / "fetched.tsv").read_text().splitlines()]
    assert listed == [[a, str(dest / f), hashlib.sha256((dest / f).read_bytes()).hexdigest()] for a, f in zip((tree, one), files)]

    r = subprocess.run(["bash", "-c", run], capture_output=True, text=True, env={**env, "ARTS": f"{tree} {one[:12]}", "NAME": "x"})
    assert r.returncode == 2 and "is not a full art id" in r.stderr and not (out / "x").exists()

    missing = "art:" + "0" * 64
    r = subprocess.run(["bash", "-c", run], capture_output=True, text=True, env={**env, "ARTS": f"{missing} {one}", "NAME": "y"})
    assert r.returncode == 1 and f"could not fetch {missing}" in r.stderr, r.stderr
    assert f"{missing}\t-\tnot fetched" in (out / "y" / "fetched.tsv").read_text().splitlines()
    assert (out / "y" / one[4:] / "one.bin").exists(), "an id the remote lacks doesn't stop the others"

    (tmp_path / "quiet" / "bin").mkdir(parents=True)
    (tmp_path / "quiet" / "bin" / "uv").write_text("#!/bin/sh\nexit 0\n")                     # a fetch that exits 0 and writes nothing
    (tmp_path / "quiet" / "bin" / "uv").chmod(0o755)
    quiet = run.replace(str(jobs), str(tmp_path / "quiet"))
    r = subprocess.run(["bash", "-c", quiet], capture_output=True, text=True, env={**env, "ARTS": one, "NAME": "z"})
    assert r.returncode == 1 and f"could not fetch {one}" in r.stderr and "0 of 1 artifact(s)" in r.stdout, r.stderr


def test_a_per_pod_copy_goes_once_research_run_exits(tmp_path):
    """prover-bench, prover-dev and port-capture run $CMD in a copy of the tree of their own (it writes there), ~4.7k inodes each that
    nothing reads once the pod is gone, and hundreds pile up on a node. The run step removes its copy after `research run` exits,
    keeps that exit status, and leaves the copy in place while research run runs from it. port-capture keeps the shared copy."""
    import os
    import shutil
    import subprocess
    import textwrap

    jobs, synced = tmp_path / "jobs", tmp_path / "synced"
    (synced / "tools/research/src/research/pods/nebius/sky").mkdir(parents=True)
    shutil.copy(SKY / "job_tree.sh", synced / "tools/research/src/research/pods/nebius/sky/job_tree.sh")
    (synced / "marker").write_text("x\n")
    (jobs / "bin").mkdir(parents=True)
    (jobs / "bin" / "uv").write_text('#!/bin/sh\n[ -f "$SRC/marker" ] && [ "$(pwd)" = "$SRC" ] || exit 99\nexit "$RC"\n')
    (jobs / "bin" / "uv").chmod(0o755)
    host = subprocess.run(["hostname"], capture_output=True, text=True, check=True).stdout.strip()
    env = {**os.environ, "SRC": str(synced), "VY_JOB_TREES": str(jobs / "src"), "CUDA_HOME": str(tmp_path / "cuda"), "CAMPAIGN": "c",
           "CMD": "true"}
    shared = subprocess.run(["bash", str(SKY / "job_tree.sh"), str(synced)], capture_output=True, text=True, env=env, check=True).stdout.strip()

    def step(doc: dict, key: str, rc: int) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", "-c", doc[key].replace("/workspace/jobs", str(jobs))], capture_output=True, text=True,
                              env={**env, "RC": str(rc)}, cwd=tmp_path)

    for name, copy in (("prover-bench", jobs / "src" / host), ("prover-dev", jobs / "src" / host),
                       ("port-capture", jobs / "src" / f"pod-{host}")):
        text = (SKY / "jobs" / f"{name}.yaml").read_text()
        doc = {"setup": textwrap.dedent(text.split("\nsetup: |\n", 1)[1].split("\n\nrun: |\n", 1)[0]),
               "run": textwrap.dedent(text.split("\nrun: |\n", 1)[1].split("\n\nconfig:", 1)[0])}
        for rc in (0, 3):
            if name != "port-capture":
                assert step(doc, "setup", 0).returncode == 0 and (copy / "marker").is_file(), name
            r = step(doc, "run", rc)
            assert r.returncode == rc, (name, rc, r.stderr)
            assert not copy.exists(), (name, rc)
            assert (Path(shared) / "marker").is_file(), name


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
    ops = synced / "adapters/vllm/verity_vllm/ops"
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
    (synced / "adapters/vllm/new.py").write_text("x = 1\n")
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


def test_a_leased_bootstrap_takes_its_gpu_before_the_host_lock_so_others_bootstrap_while_it_waits(tmp_path):
    """A leased job's pod (GPU_LEASE set) has no GPU of its own: a real GPU-mode bootstrap runs under the lease tool, which it calls
    before it opens the host-wide lock. The fake lease tool stands for a pool with no GPU free yet: when called it finds the lock free
    and runs another job's CPU bootstrap to completion, then grants GPU 3. Taking the lease inside the lock would find it held
    instead. A cached bootstrap and a --cpu one never call the tool."""
    import os
    import shutil
    import subprocess
    tree = tmp_path / "tree"
    sky = tree / "tools/research/src/research/pods/nebius/sky"
    sky.mkdir(parents=True)
    shutil.copy(SKY / "vllm_bootstrap.sh", sky / "vllm_bootstrap.sh")
    ops = tree / "adapters/vllm/verity_vllm/ops"
    ops.mkdir(parents=True)
    (ops / "pod_bootstrap.sh").write_text('echo "$2 cvd=${CUDA_VISIBLE_DEVICES-unset}" >> "$WORK/seen"; echo BOOTSTRAP-OK\n')
    jobs = tmp_path / "jobs"
    (jobs / "venv312/lib/python3.12/site-packages").mkdir(parents=True)
    lease = tmp_path / "gpu-lease"
    lease.write_text(
        '#!/usr/bin/env bash\necho "$*" >> "$VY_JOBS/lease"\n'
        'if flock -n "$VY_JOBS/.vllm-bootstrap.lock" true; then\n'
        '  echo free >> "$VY_JOBS/lock"\n'
        '  env -u VY_BOOTSTRAP_LEASED bash "$OTHER" --cases B1 --cpu --out "$VY_JOBS/b" >> "$VY_JOBS/other"\n'
        'else\n  echo held >> "$VY_JOBS/lock"\nfi\n'
        'while [ "$1" != -- ]; do shift; done; shift\nexport CUDA_VISIBLE_DEVICES=3\nexec "$@"\n')
    lease.chmod(0o755)
    env = {**os.environ, "VY_JOBS": str(jobs), "HF_HOME": str(tmp_path / "hf"), "GPU_LEASE": str(lease), "CUDA_VISIBLE_DEVICES": "",
           "OTHER": str(sky / "vllm_bootstrap.sh")}
    env.pop("VY_BOOTSTRAP_LEASED", None)

    def boot(*args: str) -> str:
        return subprocess.run(["bash", str(sky / "vllm_bootstrap.sh"), *args, "--out", str(jobs / "b")], env=env, capture_output=True,
                              text=True, check=True).stdout

    assert boot("--cases", "B0").strip().endswith("BOOTSTRAP-OK")
    assert (jobs / "lease").read_text() == f"1 --wait --max-min 60 --mem-gb 0 -- bash {sky / 'vllm_bootstrap.sh'} --cases B0 --out " \
        f"{jobs / 'b'}\n"
    assert (jobs / "lock").read_text() == "free\n" and (jobs / "other").read_text().strip().endswith("BOOTSTRAP-OK")
    assert (jobs / "seen").read_text() == "B1 cvd=\nB0 cvd=3\n"         # the other job's bootstrap ran while this one waited
    assert "cached" in boot("--cases", "B0") and (jobs / "lease").read_text().count("\n") == 1
    assert "cached" in boot("--cases", "B1", "--cpu") and (jobs / "lease").read_text().count("\n") == 1


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


STORE_PAIR = {"AWS_ACCESS_KEY_ID": "parent-id", "AWS_SECRET_ACCESS_KEY": "parent-secret"}      # dummies for the store's own key pair
CANNOT_REMOVE = {"DeleteObject", "DeleteObjects", "CopyObject", "UploadPartCopy"}
READS = {"HeadObject", "GetObject", "GetBucketLocation", "ListObjectsV1", "ListObjectsV2"}


def _submit(tmp_path, template, **shell):
    """submit.sh for `template` from this checkout, in a shell that holds the store's own key pair, with stand-ins on PATH for what
    reaches a network: `sky` records each call's arguments and the store's variables in its environment (where the SkyPilot client
    reads a `--secret NAME` from, and what it sends an API server on this port), `python3` records `research pods ...` and runs
    anything else, and `curl` says the API server answers.  `shell` adds to the environment; None takes a variable out.  Returns
    the finished process, the `sky` calls and the `research pods` calls."""
    import os
    import shutil
    import subprocess
    import sys

    tmp_path.mkdir(parents=True, exist_ok=True)
    bin_dir, calls = tmp_path / "bin", tmp_path / "sky-calls"
    bin_dir.mkdir()
    calls.mkdir()
    for tool in ("bash", "git", "dirname", "sed", "cat", "tr", "head", "awk", "grep", "mktemp", "rm", "ls"):
        os.symlink(shutil.which(tool), bin_dir / tool)
    (tmp_path / "sky.py").write_text(
        "import json, os, sys\n"
        "calls = os.environ['SKY_CALLS']\n"
        "store = {k: v for k, v in os.environ.items() if k.startswith(('AWS_', 'R2_'))}\n"
        "with open(os.path.join(calls, f'{len(os.listdir(calls)):02d}.json'), 'w') as f:\n"
        "    json.dump({'argv': sys.argv[1:], 'env': store}, f)\n"
        "if sys.argv[1:3] == ['api', 'info']:\n"
        "    print('http://127.0.0.1:46580')\n")
    stand_ins = {"sky": f'exec "{sys.executable}" "{tmp_path / "sky.py"}" "$@"',
                 "python3": 'if [ "$1 $2 $3" = "-m research pods" ]; then shift 2; echo "$*" >> "$PODS_CALLS"; exit 0; fi\n'
                            f'exec "{sys.executable}" "$@"',
                 "curl": "exit 0", "ssh": "exit 0", "rsync": "exit 0", "uv": "exit 0"}
    for name, body in stand_ins.items():
        (bin_dir / name).write_text(f"#!/bin/bash\n{body}\n")
        (bin_dir / name).chmod(0o755)
    env = {"PATH": str(bin_dir), "HOME": str(tmp_path), "TMPDIR": str(tmp_path), "VY_INFRA_REMOTE": str(tmp_path / "no-remote"),
           "SKY_CALLS": str(calls), "PODS_CALLS": str(tmp_path / "pods-calls"), "RESEARCH_STORE_CONFIG": str(tmp_path / "absent.toml"),
           "R2_ENDPOINT": "https://" + "0" * 32 + ".r2.cloudflarestorage.com", "R2_BUCKET": "bkt", **STORE_PAIR, **shell}
    r = subprocess.run(["bash", str(SKY / "submit.sh"), template, "job-1", "--allow-stale", "--env", "NAME=n"], capture_output=True, text=True,
                       env={k: v for k, v in env.items() if v is not None})
    pods = (tmp_path / "pods-calls").read_text().splitlines() if (tmp_path / "pods-calls").exists() else []
    return r, [json.loads(p.read_text()) for p in sorted(calls.iterdir())], pods


def _signed_claims(token: str) -> dict | None:
    """The claims of a temporary key's session token, or None with no token or one the store's own secret did not sign. A token that
    is no JWT at all raises."""
    import base64

    from research.store import credentials as C

    if not token:
        return None
    jwt = base64.b64decode(token).decode()[len("jwt/"):]
    claims = C.decode_jwt(jwt)[1]
    return claims if C.sign_jwt_hs256(claims, STORE_PAIR["AWS_SECRET_ACCESS_KEY"]) == jwt else None


def test_submit_hands_a_job_a_key_minted_for_it_and_no_sky_call_the_stores_own(tmp_path):
    """What submit.sh gives SkyPilot. The launch names five secrets and carries no value; the three AWS_* values it reads from its
    environment are a temporary key signed with the store's own secret, with custody's actions for a template that pushes and the
    reader's for store-fetch, none of which deletes or copies, for 24 hours or VY_STORE_TTL. No `sky` call has the store's own
    secret in its environment, the launch included: the SkyPilot client sends an API server on this port what it finds there."""
    import base64
    import hashlib

    from research.store import credentials as C
    from research.store import custody as CU

    for template, access, actions, ttl in (("store-push", "push", CU.ACTIONS, None), ("prover-dev", "push", CU.ACTIONS, "7d"),
                                           ("store-fetch", "fetch", CU.READ_ACTIONS, "90m")):
        r, sky, pods = _submit(tmp_path / template, template, VY_STORE_TTL=ttl)
        assert r.returncode == 0, r.stderr
        assert [c["argv"][:2] for c in sky] == [["api", "info"], ["jobs", "launch"], ["jobs", "queue"]], sky
        info, launch, queue = sky
        assert [launch["argv"][i + 1] for i, a in enumerate(launch["argv"]) if a == "--secret"] == \
            ["R2_ENDPOINT", "R2_BUCKET", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"], launch["argv"]
        key = launch["env"]
        assert sorted(key) == ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "R2_BUCKET", "R2_ENDPOINT"]
        claims = _signed_claims(key["AWS_SESSION_TOKEN"])
        assert claims and claims["actions"] == list(actions) and "scope" not in claims and claims["bucket"] == "bkt", (template, claims)
        assert not CANNOT_REMOVE & set(claims["actions"]) and (access == "push" or set(claims["actions"]) <= READS), (template, claims)
        assert claims["exp"] - claims["iat"] == C.parse_ttl(ttl or "24h")
        jwt = base64.b64decode(key["AWS_SESSION_TOKEN"])[len("jwt/"):]                # R2 derives the temporary secret from the token
        assert key["AWS_SECRET_ACCESS_KEY"] == hashlib.sha256(jwt).hexdigest() and key["AWS_ACCESS_KEY_ID"] == claims["iss"]
        assert not any(key[k] in a for k in ("AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN") for a in launch["argv"])
        for call in (info, queue):
            assert sorted(call["env"]) == ["R2_BUCKET", "R2_ENDPOINT"], call
        assert STORE_PAIR["AWS_SECRET_ACCESS_KEY"] not in json.dumps(sky) + r.stdout + r.stderr
        assert f"job-1 gets a {access} key until " in r.stderr and key["AWS_SESSION_TOKEN"] not in r.stdout + r.stderr
        assert pods[0].startswith("pods sync vy-nebius-1 "), pods


def test_submit_refuses_when_it_cannot_mint_the_jobs_key(tmp_path):
    """Never what the shell holds in the key's place. Without the store's own pair, with a temporary key (which cannot mint another,
    and which a job would otherwise get whole) or with a lifetime R2 won't sign, the script exits 5 before any `sky` call or sync."""
    cases = {"no-pair": dict.fromkeys(STORE_PAIR), "temporary": {"AWS_SESSION_TOKEN": "dummy"}, "a-week-and-a-day": {"VY_STORE_TTL": "8d"}}
    for case, shell in cases.items():
        r, sky, pods = _submit(tmp_path / case, "prover-dev", **shell)
        assert r.returncode == 5 and "submit.sh: no store key for the job" in r.stderr, (case, r.stderr)
        assert sky == [] and pods == [], (case, sky, pods)


def test_a_jobs_key_pushes_or_fetches_and_a_delete_is_refused(tmp_path, monkeypatch):
    """What a job can do with the key submit.sh gave it, on a bucket that holds each request to the actions in its session token, as
    R2 does (FakeS3's `authorize`; a request without a token the store's own secret signed is refused too). store-push's run step
    preserves the job store's pending artifact and store-fetch's run step lays it out with the read-only key, no request of either
    denied. Through the environment's remote, which is how a job reads its key, the push key also uploads in parts; a DELETE
    with either key answers 403 and the object stays, where a key that names DeleteObject removes it; and the fetch key cannot
    write. The run steps name the bucket in a store.toml here: the environment's remote takes the store's real bucket alone."""
    import hashlib
    import os
    import subprocess
    import sys
    import textwrap
    import tomllib
    import urllib.error

    import pytest

    from research.store import credentials as C
    from research.store import remote, remote_s3
    from research.store.local import LocalStore
    from test_store import FakeS3

    s3 = FakeS3(authorize=lambda action, headers: action in (_signed_claims(headers.get("x-amz-security-token") or "") or {}).get("actions", ()))
    try:
        keys = {}
        for template in ("store-push", "store-fetch"):
            r, sky, _ = _submit(tmp_path / template, template, R2_ENDPOINT=s3.endpoint, R2_ACCOUNT_ID="0" * 32)
            assert r.returncode == 0, r.stderr
            keys[template] = sky[1]["env"]

        toml, jobs, out = tmp_path / "store.toml", tmp_path / "jobs", tmp_path / "out"
        toml.write_text(f'[remote]\ntype = "s3"\nbucket = "bkt"\nendpoint = "{s3.endpoint}"\n'
                        'access_key_id_env = "AWS_ACCESS_KEY_ID"\nsecret_access_key_env = "AWS_SECRET_ACCESS_KEY"\n')
        (tmp_path / "repo").mkdir()
        (tmp_path / "t").mkdir()
        (tmp_path / "t" / "a.json").write_text("{}\n")
        art = LocalStore(jobs / "store", config_path=tmp_path / "absent.toml", repo=tmp_path / "repo").put_artifact(
            "run-files/v1", {"n": 1}, tree=tmp_path / "t")
        tools = {"bin/uv": f'shift 5\nexec "{sys.executable}" "$@"',                 # `uv run --no-project --python 3.12 python`
                 "venv312/bin/nice": 'shift 2\nexec "$@"', "venv312/bin/ionice": 'shift 2\nexec "$@"'}
        for name, body in tools.items():
            (jobs / name).parent.mkdir(parents=True, exist_ok=True)
            (jobs / name).write_text(f"#!/bin/sh\n{body}\n")
            (jobs / name).chmod(0o755)

        def step(template: str, **envs: str) -> subprocess.CompletedProcess:
            text = (SKY / "jobs" / f"{template}.yaml").read_text()
            run = textwrap.dedent(text.split("\nrun: |\n", 1)[1].split("\n\nconfig:", 1)[0]).replace("/workspace/jobs", str(jobs))
            env = {k: v for k, v in os.environ.items() if not k.startswith(("R2_", "AWS_", "RESEARCH_"))}
            env.update({k: v for k, v in keys[template].items() if k.startswith("AWS_")})
            env.update(SRC=str(SKY.parents[6]), RESEARCH_STORE_CONFIG=str(toml), RESEARCH_REPO=str(tmp_path / "repo"),
                       RESEARCH_STORE=str(jobs / "store"), **envs)
            return subprocess.run(["bash", "-c", run], capture_output=True, text=True, env=env, cwd=tmp_path)

        manifest = f"manifests/{art[4:]}.json"
        r = step("store-push", MODE="push", PUSH_JOBS="2", OUT=str(out / "push"))
        assert r.returncode == 0 and manifest in s3.objects, r.stdout + r.stderr
        r = step("store-fetch", ARTS=art, PATHS="", NAME="n", KEEP_DAYS="2", OUT=str(out / "fetch"))
        assert r.returncode == 0 and (out / "fetch" / "n" / art[4:] / "a.json").read_text() == "{}\n", r.stdout + r.stderr
        assert not [entry for entry in s3.log if entry[0] == "DENIED"], s3.log

        monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
        monkeypatch.delenv("RESEARCH_STORE_CONFIG")
        pod = tmp_path / "store.pod.toml"                              # pins this bucket's digest, as the tracked file pins the store's
        pod.write_text(remote.POD_CONFIG.read_text().replace(tomllib.loads(remote.POD_CONFIG.read_text())["remote"]["bucket_sha256"],
                                                             hashlib.sha256(b"bkt").hexdigest()))
        monkeypatch.setattr(remote, "POD_CONFIG", pod)
        monkeypatch.setattr(remote_s3.S3Remote, "multipart_min", 8192)
        monkeypatch.setattr(remote_s3.S3Remote, "part_size", 4096)

        def job_remote(template: str) -> remote_s3.S3Remote:
            for k, v in keys[template].items():
                monkeypatch.setenv(k, v)
            r = remote.load_remote()
            assert r.session_token == keys[template]["AWS_SESSION_TOKEN"]
            return r

        big = tmp_path / "big.bin"
        big.write_bytes(bytes(range(256)) * 36)
        assert job_remote("store-push").put("objects/big", big)["action"] == "created" and s3.etags["objects/big"].endswith("-3")
        assert not [entry for entry in s3.log if entry[0] == "DENIED"], s3.log
        for template in ("store-push", "store-fetch"):
            with pytest.raises(urllib.error.HTTPError) as refused:
                job_remote(template).delete(manifest)
            assert refused.value.code == 403 and manifest in s3.objects
        with pytest.raises(urllib.error.HTTPError) as refused:
            job_remote("store-fetch").put("objects/new", tmp_path / "t" / "a.json")
        assert refused.value.code == 403 and "objects/new" not in s3.objects
        assert [entry[1] for entry in s3.log if entry[0] == "DENIED"] == ["DeleteObject", "DeleteObject", "PutObject"], s3.log
        own = C.mint_local(s3.endpoint, "0" * 32, *STORE_PAIR.values(), "bkt", "object-read-write", actions=["DeleteObject"], ttl_s=60)
        deleter = remote_s3.S3Remote(s3.endpoint, "bkt", own["accessKeyId"], own["secretAccessKey"], session_token=own["sessionToken"])
        assert deleter.delete(manifest) and manifest not in s3.objects, "the bucket deletes for a key that names the action"
    finally:
        s3.close()


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


def test_a_quiet_commit_is_stopped_and_a_progressing_one_is_not(tmp_path):
    """The Commit's watchdog: when commit.log hasn't changed for COMMIT_STALL_S (Gemma-2-2B Commits held GPUs 50-90 min silent) it
    lowers the run's timeout through `research tele set-timeout`, whose harness stops the workload's session and records the attempt
    as cancelled, and exits 86; a run the harness doesn't stop is killed with its workload's group (status.json `pgid`) after
    COMMIT_STALL_GRACE_S; a Commit whose log keeps changing runs to its own exit. Clocks are shims (each `date +%s` call advances
    400 s, `sleep` returns at once, `stat` reads the log's mtime from a file), and each Commit exits on a FIFO or a kill, so nothing
    here waits on wall-clock time."""
    import os
    import subprocess
    gpu = (SKY / "jobs" / "config-run.yaml").read_text().split("\n---\n")[2]
    fn = "\n".join(ln[2:] for ln in gpu[gpu.index("\n  # >>> commit watchdog") + 1:gpu.index("  # <<< commit watchdog")].splitlines())
    assert 'watch_commit "$SWEEP_DIR/$ROW/commit.log" "$RESEARCH_RUNS/$run_id" "${research[@]}" run --id "$run_id"' in gpu
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "sleep").write_text("#!/bin/sh\nexit 0\n")
    (bin_dir / "date").write_text('#!/bin/sh\nn=$(cat "$CLOCK"); echo $((n + 400)) > "$CLOCK"; [ "$1" = +%s ] && echo "$n" || echo T\n')
    # a progressing log (CALLS set) has the clock's next reading as its mtime, and its Commit exits at the 5th poll (the FIFO it reads);
    # a quiet one has mtime 0
    (bin_dir / "stat").write_text('#!/bin/sh\nif [ -n "$CALLS" ]; then c=$(($(cat "$CALLS") + 1)); echo $c > "$CALLS"\n'
                                  '[ $c -eq 5 ] && echo > "$FIFO"; cat "$CLOCK"; else echo 0; fi\n')
    # the runner's harness, stopping the workload (the FIFO it reads) when HARNESS_FIFO is set
    (bin_dir / "research").write_text('#!/bin/sh\necho "$*" >> "$CALLED"\n[ -z "$HARNESS_FIFO" ] || echo > "$HARNESS_FIFO"\n')
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    log, run_dir, called = tmp_path / "row" / "commit.log", tmp_path / "runs" / "r1", tmp_path / "called"
    log.parent.mkdir()
    run_dir.mkdir(parents=True)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "CLOCK": str(tmp_path / "clock"), "CALLED": str(called), "COMMIT_STALL_S": "900",
           "COMMIT_STALL_POLL_S": "0"}

    def watch(cmd: str, **extra: str) -> tuple[str, str]:
        (tmp_path / "clock").write_text("1000\n")
        for f in (called, log.parent / "row.log"):
            f.unlink(missing_ok=True)
        r = subprocess.run(["bash", "-c", f"research=({bin_dir}/research)\n{fn}\nwatch_commit {log} {run_dir} {cmd}; echo rc=$?"],
                           capture_output=True, text=True, env=env | extra)
        return r.stdout.strip().splitlines()[-1], (log.parent / "row.log").read_text() if (log.parent / "row.log").exists() else ""

    fifo = tmp_path / "workload"
    os.mkfifo(fifo)
    rc, row_log = watch(f"cat {fifo}", HARNESS_FIFO=str(fifo), COMMIT_STALL_GRACE_S="1000000000")
    assert rc == "rc=86" and "stopping the Commit" in row_log and "killing it" not in row_log
    assert called.read_text().split() == ["tele", "set-timeout", "--attempt-dir", str(run_dir), "--timeout-s", "1", "--actor", "guardian",
                                          "--reason", "commit", "watchdog:", "commit.log", "unchanged", "for", "1200", "s"]
    workload = subprocess.Popen(["tail", "-f", "/dev/null"], start_new_session=True)
    (run_dir / "status.json").write_text(f'{{"pgid": {workload.pid}}}')
    rc, row_log = watch("tail -f /dev/null", COMMIT_STALL_GRACE_S="0")
    assert rc == "rc=86" and "killing it" in row_log and workload.wait() == -9
    (tmp_path / "calls").write_text("0\n")
    rc, row_log = watch(f"cat {fifo}", CALLS=str(tmp_path / "calls"), FIFO=str(fifo), COMMIT_STALL_GRACE_S="0")
    assert rc == "rc=0" and row_log == "" and not called.exists()


def test_the_store_eviction_drops_only_preserved_copies_hourly_as_ubuntu_and_fails_on_a_tool_error(tmp_path):
    """vy-store-evict: cluster_up.sh and `research deploy` install it hourly as ubuntu (the job store's owner, store_evict.service); it runs `research data evict --runs`
    against the job store and runs dir toward its free-space mark, logs one line, treats a mark it can't reach as done (nothing
    more is preserved), and fails only when the tool's output isn't its JSON."""
    import subprocess
    up = (SKY / "cluster_up.sh").read_text()
    assert "store_evict.sh\" /usr/local/bin/vy-store-evict" in up and "vy-store-evict-research.timer" in up
    assert "User=ubuntu" in (SKY / "store_evict.service").read_text() and "OnUnitActiveSec=1h" in (SKY / "store_evict.timer").read_text()
    research = (SKY / "store_evict_research.service").read_text()
    assert "User=research" in research and "VY_EVICT_STORE=/workspace/research/store VY_EVICT_RUNS=/workspace/research/runs" in research
    tool, bin_dir = tmp_path / "tool", tmp_path / "bin"
    (tool / "research").mkdir(parents=True)
    bin_dir.mkdir()
    (tool / "research" / "__init__.py").write_text("")
    (tool / "research" / "__main__.py").write_text(
        "import json, os, sys\n"
        "open(os.environ['FAKE_ARGV'], 'w').write(json.dumps({'argv': sys.argv[1:], 'store': os.environ.get('RESEARCH_STORE')}))\n"
        "mode = os.environ['FAKE_MODE']\n"
        "if mode == 'error':\n    print('research data evict: [Errno 1] Operation not permitted'); sys.exit(1)\n"
        "print(json.dumps({'store': {'evicted_bytes': 184.6e9, 'reached_target': mode == 'reached'}, 'runs': {'evicted_bytes': 3e9}}))\n"
        "sys.exit(0 if mode == 'reached' else 1)\n")
    (bin_dir / "df").write_text('#!/bin/sh\necho " Avail"\necho " 2161"\n')
    (bin_dir / "df").chmod(0o755)

    def run(mode: str, **extra):
        env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "VY_RESEARCH_TOOL": str(tool), "FAKE_MODE": mode, "FAKE_ARGV": str(tmp_path / "argv.json"),
               "VY_EVICT_GIT": str(tmp_path / "no-git"), **extra}
        return subprocess.run(["bash", str(SKY / "store_evict.sh")], env=env, capture_output=True, text=True)

    r = run("reached")
    assert r.returncode == 0 and "evicted 184.6 GB of store blobs, 3.0 GB of run files; target reached: True" in r.stdout
    seen = json.loads((tmp_path / "argv.json").read_text())
    assert seen["store"] == "/workspace/jobs/store"
    assert seen["argv"] == ["data", "evict", "--target-free-gb", "2500", "--runs", "--runs-dir", "/workspace/jobs/runs", "--min-mb", "1e12", "--json"]
    r = run("short")
    assert r.returncode == 0 and "target reached: False" in r.stdout
    r = run("error")
    assert r.returncode == 1 and "Operation not permitted" in r.stderr
    r = run("reached", VY_EVICT_STORE="/workspace/research/store", VY_EVICT_RUNS="/workspace/research/runs")   # research's own instance
    seen = json.loads((tmp_path / "argv.json").read_text())
    assert r.returncode == 0 and seen["store"] == "/workspace/research/store" and seen["argv"][6] == "/workspace/research/runs"


def test_researchs_store_eviction_also_deletes_unused_source_trees_and_says_what_it_kept(tmp_path):
    """Something must prune /workspace/research/src, or a node runs out of inodes: research's unit (its drop-in, from cluster_up.sh
    and deploy.toml) passes `--src-dir` with the capabilities its /proc scan needs, and the journal line counts what went and why
    the rest stayed, the reason it deleted nothing included."""
    import subprocess
    conf = (SKY / "store_evict_src.conf").read_text()
    assert "Environment=VY_EVICT_SRC=/workspace/research/src VY_EVICT_ORIGIN=/workspace/research/git/verity.git\n" in conf and "AmbientCapabilities=CAP_SYS_PTRACE CAP_DAC_READ_SEARCH\n" in conf
    dest = "/etc/systemd/system/vy-store-evict-research.service.d/src.conf"
    assert f'"$SRC/store_evict_src.conf" {dest}' in (SKY / "cluster_up.sh").read_text()
    import tomllib
    deploy = (SKY.parent / "deploy.toml").read_text()
    files = tomllib.loads(deploy)["file"]
    src_conf = [f for f in files if f["path"] == dest]
    assert [f["src"] for f in src_conf] == ["tools/research/src/research/pods/nebius/sky/store_evict_src.conf"]
    unit_nodes = {n for f in files if f["path"] == "/etc/systemd/system/vy-store-evict-research.service" for n in f["nodes"]}
    assert set(src_conf[0]["nodes"]) == unit_nodes and "vy-nebius-cpu-12" in unit_nodes
    for u in ("store_evict.service", "store_evict.timer", "store_evict_research.service", "store_evict_research.timer"):
        assert f'"$SRC/$u"' in (SKY / "cluster_up.sh").read_text() and f'src = "tools/research/src/research/pods/nebius/sky/{u}"' in deploy
    assert 'ExecCondition=/bin/grep -qs "timed False" /workspace/pouw/fill/status.txt' in (SKY / "store_evict_timed.conf").read_text()
    assert deploy.count('path = "/usr/local/bin/vy-store-evict"\nsrc = "tools/research/src/research/pods/nebius/sky/store_evict.sh"\nmode = "755"\n') == 2, \
        "a fresh install gets the repository's mode (644), which systemd can't exec (203/EXEC)"
    unit = (SKY / "store_evict_research.service").read_text()   # cpu-1's copy of the unit cluster_up.sh writes on node 1
    assert "User=research\n" in unit and "VY_EVICT_STORE=/workspace/research/store VY_EVICT_RUNS=/workspace/research/runs" in unit
    assert "ExecStart=/usr/local/bin/vy-store-evict\n" in unit and "OnUnitActiveSec=30min\n" in (SKY / "store_evict_research.timer").read_text()
    tool, bin_dir = tmp_path / "tool", tmp_path / "bin"
    (tool / "research").mkdir(parents=True)
    bin_dir.mkdir()
    (tool / "research" / "__init__.py").write_text("")
    (tool / "research" / "store").mkdir()
    (tool / "research" / "store" / "cli.py").write_text('ap.add_argument("--src-dir")\n')
    (tool / "research" / "__main__.py").write_text(
        "import json, os, sys\n"
        "open(os.environ['FAKE_ARGV'], 'w').write(json.dumps(sys.argv[1:]))\n"
        "open(os.environ['FAKE_ARGV'] + '.cwd', 'w').write(os.getcwd())\n"
        "kept = json.loads(os.environ.get('FAKE_KEPT') or json.dumps([{'tree': 'c'}] * 3))\n"
        "src = {'deleted': [{'tree': 'a'}, {'tree': 'b'}], 'deleted_bytes': 31e9, 'deleted_inodes': 812345, 'kept': kept,\n"
        "       'kept_by': {'1': 1, '2': 2}, 'error': os.environ.get('FAKE_ERROR')}\n"
        "err = {'error': os.environ['FAKE_STORE_ERROR']} if os.environ.get('FAKE_STORE_ERROR') else {}\n"
        "print(json.dumps({'store': {'evicted_bytes': 0, 'reached_target': False}, 'runs': {'evicted_bytes': 0}, 'src': src, **err}))\n"
        "sys.exit(1)\n")
    (bin_dir / "df").write_text('#!/bin/sh\necho " Avail"\necho " 1443"\n')
    (bin_dir / "df").chmod(0o755)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "VY_RESEARCH_TOOL": str(tool), "FAKE_ARGV": str(tmp_path / "argv.json"),
           "VY_EVICT_STORE": "/workspace/research/store", "VY_EVICT_RUNS": "/workspace/research/runs", "VY_EVICT_SRC": "/workspace/research/src",
           "VY_EVICT_GIT": str(tmp_path / "no-git")}
    r = subprocess.run(["bash", str(SKY / "store_evict.sh")], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads((tmp_path / "argv.json").read_text()) == ["data", "evict", "--target-free-gb", "2500", "--runs", "--runs-dir",
                                                                 "/workspace/research/runs", "--src-dir", "/workspace/research/src", "--min-mb", "1e12",
                                                                 "--json"]
    assert "target reached: False; src trees: deleted 2 (31.0 GB, 812345 inodes), kept 3 (1:1 2:2); no store blob" in r.stdout
    # a pass that keeps trees under condition 2 says which process named each
    kept = [{"tree": t, "why": ["2: process 7 (du) uses it", *(["2: process 9 (lake) uses it"] if t == "c" else [])]} for t in "cde"]
    r = subprocess.run(["bash", str(SKY / "store_evict.sh")], env={**env, "FAKE_KEPT": json.dumps(kept)}, capture_output=True, text=True)
    assert "kept 3 (1:1 2:2); trees a process names: 7 (du) 3, 9 (lake) 1; no store blob" in r.stdout
    r = subprocess.run(["bash", str(SKY / "store_evict.sh")], env={**env, "VY_EVICT_ORIGIN": "/workspace/research/git/verity.git",
                                                                     "FAKE_ERROR": "2: /proc/1 is unreadable"}, capture_output=True, text=True)
    assert json.loads((tmp_path / "argv.json").read_text())[-5:-2] == ["--src-origin", "/workspace/research/git/verity.git", "--min-mb"]
    assert "kept 3 (1:1 2:2), nothing more deleted: 2: /proc/1 is unreadable; no store blob" in r.stdout
    assert r.returncode == 1 and "could not run" in r.stderr, "a pass that can't establish its conditions fails the unit (cpu-2, 16 Oct)"
    r = subprocess.run(["bash", str(SKY / "store_evict.sh")], env={**env, "FAKE_ERROR": "another pass holds /workspace/research/src/.evict.lock"},
                       capture_output=True, text=True)
    assert r.returncode == 0 and "nothing more deleted: another pass holds" in r.stdout, "an overlapping pass is not a fault"
    r = subprocess.run(["bash", str(SKY / "store_evict.sh")], env={**env, "FAKE_STORE_ERROR": "store pass: OperationalError: readonly"},
                       capture_output=True, text=True)
    assert "src trees: deleted 2 (31.0 GB, 812345 inodes)" in r.stdout and "; FAILED store pass: OperationalError: readonly" in r.stdout
    assert r.returncode == 1 and "the source-tree pass ran without it" in r.stderr, "a store it can't open fails the unit, after the trees"
    assert (tmp_path / "argv.json.cwd").read_text() == "/", "python -m puts its cwd first on sys.path: never a writable one"
    (tool / "research" / "store" / "cli.py").unlink()
    r = subprocess.run(["bash", str(SKY / "store_evict.sh")], env=env, capture_output=True, text=True)
    assert "--src-dir" not in json.loads((tmp_path / "argv.json").read_text()), "a tool too old for it still evicts the rest"
    assert r.returncode == 0 and "; src trees: not evicted, tool has no --src-dir; no store blob" in r.stdout


def test_researchs_store_eviction_prunes_the_pycache_of_sources_that_are_gone(tmp_path):
    """Nothing else prunes research's PYTHONPYCACHEPREFIX, where the bytecode of evicted source trees and uv build envs takes a
    node's inodes by the million. The hourly pass removes each mirrored directory whose source no longer exists, and keeps the rest."""
    import subprocess
    assert "Environment=VY_EVICT_PYCACHE=/workspace/research/cache/pycache\n" in (SKY / "store_evict_src.conf").read_text()
    tool, bin_dir = tmp_path / "tool", tmp_path / "bin"
    (tool / "research" / "store").mkdir(parents=True)
    bin_dir.mkdir()
    (tool / "research" / "__init__.py").write_text("")
    (tool / "research" / "__main__.py").write_text(
        "import json\nprint(json.dumps({'store': {'evicted_bytes': 0, 'reached_target': True}, 'runs': {'evicted_bytes': 0}}))\n")
    (bin_dir / "df").write_text('#!/bin/sh\necho " Avail"\necho " 1443"\n')
    (bin_dir / "df").chmod(0o755)
    live, src = tmp_path / "src" / "live" / "pkg", tmp_path / "src"
    live.mkdir(parents=True)
    pyc = tmp_path / "pycache"
    mirror = lambda p: pyc / str(p).lstrip("/")
    for d, n in ((live, 2), (src / "gone" / "pkg", 3), (live / "removed", 1)):
        mirror(d).mkdir(parents=True)
        for i in range(n):
            (mirror(d) / f"m{i}.cpython-312.pyc").write_bytes(b"x")
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "VY_RESEARCH_TOOL": str(tool), "VY_EVICT_GIT": str(tmp_path / "no-git"),
           "VY_EVICT_PYCACHE": str(pyc)}
    r = subprocess.run(["bash", str(SKY / "store_evict.sh")], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "; pycache: removed 2 dirs (4 files) whose source is gone (tool)" in r.stdout, r.stdout
    assert sorted(p.name for p in mirror(live).iterdir()) == ["m0.cpython-312.pyc", "m1.cpython-312.pyc"]
    assert not mirror(src / "gone").exists() and mirror(src / "live").is_dir()


def test_researchs_store_eviction_also_removes_what_nothing_else_does_and_says_what_stayed(tmp_path):
    """Request directories without a run dir, the clones of runs that are over, the store's lock files and uv's editable shards
    took a node's inodes by the million with nothing removing them. Research's unit points a pass at each (its drop-in, beside the
    capabilities their /proc scan needs), the journal line counts what each removed and why the rest stayed, naming the first path
    in use and the runs whose request or clone went (no lock file or shard: they hold nothing), and one that could not run fails
    the unit. A variable left empty switches its removal off."""
    import subprocess
    conf = (SKY / "store_evict_src.conf").read_text()
    assert "Environment=VY_EVICT_REQUESTS=/workspace/research/requests VY_EVICT_SCRATCH=/workspace/research/scratch\n" in conf
    assert "Environment=VY_EVICT_LOCKS=1 VY_EVICT_UV=/workspace/cache/uv\n" in conf and "AmbientCapabilities=" in conf
    assert " UV_CACHE_DIR=/workspace/cache/uv " in (SKY.parent / "vm_setup.sh").read_text(), "the cache the nodes' uv writes is the one the pass reads"
    tool, bin_dir = tmp_path / "tool", tmp_path / "bin"
    (tool / "research" / "store").mkdir(parents=True)
    bin_dir.mkdir()
    (tool / "research" / "__init__.py").write_text("")
    flags = ("--requests-dir", "--scratch-dir", "--locks", "--uv-cache", "--uv-keep-h")
    (tool / "research" / "store" / "cli.py").write_text("".join(f'ap.add_argument("{f}")\n' for f in flags))
    (tool / "research" / "__main__.py").write_text(
        "import json, os, sys\n"
        "open(os.environ['FAKE_ARGV'], 'w').write(json.dumps(sys.argv[1:]))\n"
        "left = json.loads(os.environ['FAKE_LEFT'])\n"
        "print(json.dumps({'store': {'evicted_bytes': 0, 'reached_target': False}, 'runs': {'evicted_bytes': 0},\n"
        "                  **{k: v for k, v in left.items() if '--' + {'requests': 'requests-dir', 'clones': 'scratch-dir', 'locks': 'locks',\n"
        "                                                           'uv': 'uv-cache'}[k] in sys.argv}}))\n"
        "sys.exit(1)\n")
    (bin_dir / "df").write_text('#!/bin/sh\necho " Avail"\necho " 1443"\n')
    (bin_dir / "df").chmod(0o755)
    shard = "/workspace/cache/uv/sdists-v9/editable/0123456789abcdef"
    a, b = "r20330518-033300-0a1b", "r20330518-033301-0a1b"
    left = {"requests": {"deleted": 2, "deleted_bytes": 3e6, "deleted_inodes": 12, "kept": 3, "kept_by": {"recent": 2, "run": 1}, "kept_first": [],
                         "deleted_first": [f"/workspace/research/requests/{a}", f"/workspace/research/requests/{b}"], "error": None},
            "clones": {"deleted": 21, "deleted_bytes": 4.5e9, "deleted_inodes": 189264, "kept": 0, "kept_by": {}, "kept_first": [],
                       "deleted_first": [f"/workspace/research/scratch/{a}"], "error": None},
            "locks": {"deleted": 316915, "deleted_bytes": 0, "deleted_inodes": 316915, "kept": 4, "kept_by": {"recent": 4}, "kept_first": [],
                      "deleted_first": ["/workspace/research/store/locks/blob-" + "0" * 64 + ".lock"], "error": None},
            "uv": {"deleted": 5, "deleted_bytes": 2e8, "deleted_inodes": 51, "kept": 9, "kept_by": {"2": 1, "recent": 6, "retired": 2},
                   "kept_first": [{"path": shard, "why": "2: its lock is held (.lock)"}], "error": None}}
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "VY_RESEARCH_TOOL": str(tool), "FAKE_ARGV": str(tmp_path / "argv.json"),
           "VY_EVICT_STORE": "/workspace/research/store", "VY_EVICT_RUNS": "/workspace/research/runs", "VY_EVICT_GIT": str(tmp_path / "no-git"),
           "VY_EVICT_REQUESTS": "/workspace/research/requests", "VY_EVICT_SCRATCH": "/workspace/research/scratch", "VY_EVICT_LOCKS": "1",
           "VY_EVICT_UV": "/workspace/cache/uv", "FAKE_LEFT": json.dumps(left)}

    def run(**extra):
        r = subprocess.run(["bash", str(SKY / "store_evict.sh")], env={**env, **extra}, capture_output=True, text=True)
        return r, json.loads((tmp_path / "argv.json").read_text())

    r, argv = run()
    assert r.returncode == 0, r.stderr
    assert argv == ["data", "evict", "--target-free-gb", "2500", "--runs", "--runs-dir", "/workspace/research/runs",
                    "--requests-dir", "/workspace/research/requests", "--scratch-dir", "/workspace/research/scratch", "--locks",
                    "--uv-cache", "/workspace/cache/uv", "--min-mb", "1e12", "--json"]
    assert (f"target reached: False; requests: deleted 2 (0.0 GB, 12 inodes), kept 3 (recent:2 run:1), gone: {a} {b}"
            f"; clones: deleted 21 (4.5 GB, 189264 inodes), kept 0 (none), gone: {a} ...; store locks: deleted 316915 (0.0 GB, 316915 inodes), kept 4 (recent:4)"
            f"; uv cache: deleted 5 (0.2 GB, 51 inodes), kept 9 (2:1 recent:6 retired:2), e.g. {shard}: 2: its lock is held (.lock)"
            "; no store blob") in r.stdout, r.stdout
    r, argv = run(VY_EVICT_UV="", VY_EVICT_LOCKS="0", VY_EVICT_UV_KEEP_H="72")
    assert r.returncode == 0 and "--uv-cache" not in argv and "--uv-keep-h" not in argv and "--locks" not in argv, "left empty: switched off"
    assert argv[7:11] == ["--requests-dir", "/workspace/research/requests", "--scratch-dir", "/workspace/research/scratch"]
    assert "uv cache" not in r.stdout and "store locks" not in r.stdout and "; clones: deleted 21 " in r.stdout
    r, argv = run(VY_EVICT_UV_KEEP_H="72")
    assert argv[12:16] == ["--uv-cache", "/workspace/cache/uv", "--uv-keep-h", "72"]
    failed = {**left, "uv": {**left["uv"], "deleted": 0, "error": "2: /proc/1 is unreadable"}}
    r, argv = run(FAKE_LEFT=json.dumps(failed))
    assert "kept 9 (2:1 recent:6 retired:2), e.g. " in r.stdout and ", nothing more deleted: 2: /proc/1 is unreadable; no store blob" in r.stdout
    assert r.returncode == 1 and "could not run" in r.stderr and "; clones: deleted 21 " in r.stdout, "the unit fails, after the line"
    (tool / "research" / "store" / "cli.py").write_text('ap.add_argument("--requests-dir")\n')
    r, argv = run()
    assert r.returncode == 0 and argv[7:9] == ["--requests-dir", "/workspace/research/requests"] and "--locks" not in argv
    assert (f"; requests: deleted 2 (0.0 GB, 12 inodes), kept 3 (recent:2 run:1), gone: {a} {b}; clones: not removed, tool has no --scratch-dir"
            "; store locks: not removed, tool has no --locks; uv cache: not removed, tool has no --uv-cache; no store blob") in r.stdout


def test_the_store_eviction_keeps_the_fixture_registry_of_the_newest_shipped_trees_and_else_evicts_no_blob(tmp_path):
    """Node 1's runs have no private remote and the public export can lack a fixture, so the evictor passes the union of
    `fixtures/artifacts.json` at the newest `refs/research/src/*` commits as `--keep-registry`; with no registry to read, or a tool
    without the flag, it evicts no store blob or run file (`--min-mb 1e12`) and says why."""
    import subprocess
    git_dir = tmp_path / "verity.git"
    work = tmp_path / "work"
    (work / "fixtures").mkdir(parents=True)

    def git(*a, date=0):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=work, check=True, capture_output=True,
                       env={"PATH": "/usr/bin:/bin", "GIT_COMMITTER_DATE": f"@{1_700_000_000 + date}", "GIT_AUTHOR_DATE": f"@{1_700_000_000 + date}"})

    subprocess.run(["git", "init", "-q", "--bare", str(git_dir)], check=True)
    git("init", "-q")
    for n, reg in enumerate(({"art:old": "a"}, {"art:new": "b", "previous": {"art:new": ["art:gone"]}})):
        (work / "fixtures" / "artifacts.json").write_text(json.dumps(reg))
        git("add", ".")
        git("commit", "-q", "-m", str(n), date=n)
        git("push", "-q", str(git_dir), f"HEAD:refs/research/src/c{n}")
    tool, bin_dir = tmp_path / "tool", tmp_path / "bin"
    (tool / "research" / "store").mkdir(parents=True)
    bin_dir.mkdir()
    (tool / "research" / "__init__.py").write_text("")
    (tool / "research" / "store" / "cli.py").write_text('ap.add_argument("--keep-registry")\n')
    (tool / "research" / "__main__.py").write_text(
        "import json, os, sys\n"
        "a = sys.argv[1:]\n"
        "reg = json.load(open(a[a.index('--keep-registry') + 1])) if '--keep-registry' in a else None\n"
        "open(os.environ['FAKE_ARGV'], 'w').write(json.dumps({'argv': a, 'registry': reg}))\n"
        "print(json.dumps({'store': {'evicted_bytes': 0, 'reached_target': False}, 'runs': {'evicted_bytes': 0}}))\n")
    (bin_dir / "df").write_text('#!/bin/sh\necho " Avail"\necho " 1741"\n')
    (bin_dir / "df").chmod(0o755)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "VY_RESEARCH_TOOL": str(tool), "FAKE_ARGV": str(tmp_path / "argv.json"),
           "VY_EVICT_GIT": str(git_dir)}

    def run(**extra):
        r = subprocess.run(["bash", str(SKY / "store_evict.sh")], env={**env, **extra}, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return r.stdout, json.loads((tmp_path / "argv.json").read_text())

    out, seen = run()
    assert seen["registry"] == {"art:old": "a", "art:new": "b"} and "--min-mb" not in seen["argv"] and "; kept 2 registry fixtures" in out
    out, seen = run(VY_EVICT_KEEP_REFS="1")
    assert seen["registry"] == {"art:new": "b"}, "only the newest commits' registries"
    out, seen = run(VY_EVICT_GIT=str(tmp_path / "no-git"))
    assert seen["argv"][-3:] == ["--min-mb", "1e12", "--json"] and "--keep-registry" not in seen["argv"]
    assert "; no store blob or run file evicted:" in out
    (tool / "research" / "store" / "cli.py").write_text("")
    out, seen = run()
    assert seen["argv"][-3:] == ["--min-mb", "1e12", "--json"] and "no store blob or run file evicted: tool has no --keep-registry (tool)" in out


def test_retention_gc_runs_hourly_on_the_nodes_outside_a_timed_window(tmp_path):
    """The retention policy is applied on a schedule, so a node never sits at 80% with terabytes collectable while the fill
    runner holds every queued run (the merge gates among them) for a `research retention gc` by hand. It reads the node's
    store, whose catalog records what custody verified on the remote: without it gc keeps every preserved path. It is installed
    wherever research runs are: the nodes whose runs the store eviction trims, and those `--on auto` places runs on."""
    import subprocess
    import tomllib

    from research import auto
    unit, timer = (SKY / "retention_gc.service").read_text(), (SKY / "retention_gc.timer").read_text()
    assert "User=research\n" in unit and "python3 -m research retention gc --apply'" in unit and "OnUnitActiveSec=1h\n" in timer
    assert "Environment=RESEARCH_STORE=/workspace/research/store\n" in unit
    deploy = (SKY.parent / "deploy.toml").read_text()
    doc = tomllib.loads(deploy)
    files = {f["path"]: f for f in doc["file"]}
    holds_runs = set(files["/etc/systemd/system/vy-store-evict-research.timer"]["nodes"]) | {m for m in doc["nodes"] if auto.CPU_NODE.fullmatch(m)}
    for u, path in (("retention_gc.service", "vy-retention-gc.service"), ("retention_gc.timer", "vy-retention-gc.timer")):
        f = files[f"/etc/systemd/system/{path}"]
        assert f["src"] == f"tools/research/src/research/pods/nebius/sky/{u}" and f["run"] == ["systemd:vy-retention-gc.timer"]
        assert set(f["nodes"]) == holds_runs, "the collector runs on every node of deploy.toml that holds research runs"
    cond = re.search(r"^ExecCondition=/bin/sh -c '(.*)'$", unit, re.M)[1]
    status = tmp_path / "status.txt"
    gate = lambda: subprocess.run(["sh", "-c", cond.replace("/workspace/pouw/fill/status.txt", str(status))]).returncode
    assert gate() == 0, "node 1 has no fill status: it runs"
    status.write_text("2026-10-19T06:40:00Z free GPUs 0/8, timed True, window waiting False\n")
    assert gate() != 0, "node 2 inside a timed window: skipped"
    status.write_text("2026-10-19T06:40:00Z free GPUs 0/8, timed False, window waiting False\n")
    assert gate() == 0


def test_every_cpu_node_the_retention_collector_runs_on_gets_researchs_store_eviction_whole():
    """A node that holds research runs gets a source tree for each commit `research run` ships there, and only research's
    store eviction with src.conf deletes them: under the retention collector alone the node fills. So every CPU node the
    collector is on gets the eviction, and none gets its service without the script it runs, its timer and src.conf."""
    import tomllib

    from research import auto
    files = tomllib.loads((SKY.parent / "deploy.toml").read_text())["file"]

    def on(path):
        return {n for f in files if f["path"] == path for n in f["nodes"]}

    unit = "/etc/systemd/system/vy-store-evict-research"
    service, script = on(f"{unit}.service"), on("/usr/local/bin/vy-store-evict")
    holds_runs = {n for n in on("/etc/systemd/system/vy-retention-gc.timer") if auto.CPU_NODE.fullmatch(n)}
    assert len(holds_runs) > 2, "the retention collector's CPU nodes"
    assert not holds_runs - service, (f"{sorted(holds_runs - service)} hold research runs without research's store eviction: add them "
                                      f"to the `nodes` of its script, {unit}.service, .timer and .service.d/src.conf")
    assert not service - script, f"{sorted(service - script)} get the service without the script it runs: add them to its `nodes`"
    for part in (f"{unit}.timer", f"{unit}.service.d/src.conf"):
        assert on(part) == service, f"{part} and the service it belongs to are on different nodes: {sorted(on(part) ^ service)}"
