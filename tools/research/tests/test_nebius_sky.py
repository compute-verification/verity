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
            out[name] = {m[0]: ur.quantity(m[1]) for m in re.findall(r"\{name: (\S+), nominalQuota: (\S+)\}", d)}
    return out


def test_the_queues_fit_the_server_with_room_for_the_controller():
    q = quotas(SKY / "kueue.yaml")
    assert set(q) == {"circuits", "provers"}
    assert q["circuits"]["nvidia.com/gpu"] == q["provers"]["nvidia.com/gpu"] == 4
    for res, cap in NODE.items():
        assert sum(x[res] for x in q.values()) + CONTROLLER.get(res, 0) <= cap, res
    text = (SKY / "kueue.yaml").read_text()
    assert text.count("cohortName: nebius") == 2 and text.count("    reclaimWithinCohort: Any") == 2 and "  borrowingLimit:" not in text
    values = dict(re.findall(r"metadata: \{name: (\S+)\}\nvalue: (\d+)", text))
    assert sorted(values, key=lambda k: -int(values[k])) == ["sweep-night", "circuits", "prover-bench", "dev"]


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
    assert seen == {"config-run": ("circuits", "circuits"), "port-capture": ("circuits", "circuits"),
                    "prover-dev": ("provers", "dev"), "prover-bench": ("provers", "prover-bench")}


def test_the_quiet_window_keys_on_the_label_config_runs_carry():
    bench = (SKY / "jobs" / "prover-bench.yaml").read_text()
    assert "requiredDuringSchedulingIgnoredDuringExecution" in bench and 'matchLabels: {verity.dev/build: "true"}' in bench
    assert 'verity.dev/build: "true"' in (SKY / "jobs" / "config-run.yaml").read_text()
    assert "verity.dev/build" not in (SKY / "jobs" / "port-capture.yaml").read_text()


def test_the_pouw_queue_joins_the_same_cohort_on_its_own_server():
    text = (SKY / "kueue-pouw.yaml").read_text()
    assert "cohortName: nebius" in text and "nodeLabels: {verity.dev/server: vy-nebius-2}" in text
    assert "kueue-pouw.yaml" not in (SKY / "cluster_up.sh").read_text()


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
