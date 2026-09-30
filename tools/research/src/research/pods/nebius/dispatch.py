"""vy-nebius-1's Kueue dispatcher: plain Kubernetes Jobs from the sky/jobs templates, no SkyPilot, keeping every queue deep.

    dispatch.py submit TEMPLATE KEY --tree PATH [--queue Q] [--priority P] [--class C] [--env K=V ...]   one item, now
    dispatch.py tick [--dry-run]      one pass: advance chains, requeue exit 99, top every queue up from the ready files
    dispatch.py loop [--every 60]     tick forever (tmux `node1-dispatch` on the server, as research)
    dispatch.py status                queues, this dispatcher's jobs, GPU-busy and CPU-busy (DCGM / node-exporter, 10 and 60 min)

Runs on the server with a Python that has PyYAML (`/workspace/jobs/venv312/bin/python`) and research's kubeconfig.

**Templates.** Each task of `sky/jobs/<TEMPLATE>.yaml` becomes one batch/v1 Job labelled `kueue.x-k8s.io/queue-name` (Kueue admits
it by quota and requeues it itself when it is preempted). The Job runs the task's `setup` then `run` with the task's envs, as the
image's uid 1000, with /workspace mounted, and the R2 store credentials from the Secret `research-r2` (the templates' `secrets`).
Tasks of a multi-task template (config-run: build, then gpu) chain: the next task is submitted when the previous one succeeds.
A task that exits 99 is resubmitted (a chunk that ended early or was told to yield); any other failure ends the item.
Every task starts under `taskset -c $VY_DISPATCH_CPUS` (96-191): 0-7 are k3s's and 8-95 the RC's three train-check slots.

**Ready files.** `$VY_DISPATCH_ROOT/ready/<workstream>.jsonl` (default root /workspace/jobs/dispatch), one item per line, which the
workstream's owner appends to (never rewrites a line it has; a changed item is a new id):

    {"id": "g017", "template": "config-run", "tree": "/workspace/research/trees/<lane>", "queue": "circuits",
     "priority": "circuits", "class": "small", "rank": 0, "env": {"ROW": "...", "ROLE": "B0", "REPO": "...", "REVISION": "..."}}

`queue` and `priority` default to the template's labels; `class` is submit.sh's VY_ROW_CLASS table (config-run's two tasks);
`resources` ({"<task>": {"cpus": N, "memory": GB, "gpus": N}}) overrides it. Order: higher `rank` first; within a rank the
workstreams take turns, each in file order. A queue is topped up while it has fewer than its depth of pending workloads (anyone's;
VY_DISPATCH_DEPTH, default circuits=4,provers=2,backfill=4). A chain's next task is submitted regardless of depth.

**Record.** `log.jsonl` gets one line per submit and per end (job, item key, task, queue, rc, wall); `done.jsonl` one per ended item.
Inside the Job, the template's own `research run` writes the Attempt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATES = HERE / "sky" / "jobs"
ROOT = Path(os.environ.get("VY_DISPATCH_ROOT", "/workspace/jobs/dispatch"))
IMAGE = os.environ.get("VY_DISPATCH_IMAGE", "us-docker.pkg.dev/sky-dev-465/skypilotk8s/skypilot-gpu:202605150216")
CPUS = os.environ.get("VY_DISPATCH_CPUS", "96-191")
SERVER = os.environ.get("VY_DISPATCH_SERVER", "vy-nebius-1")
SECRET = "research-r2"
REQUEUE_RC = 99
MAX_TRIES = 50
DEFAULT_DEPTH = "circuits=4,provers=2,backfill=4"
DEFAULT_PRIORITY = {"backfill": "dev"}
# submit.sh's VY_ROW_CLASS table: (build GB, gpu GB, GPUs) for config-run's two tasks
CLASSES = {"small": (64, 64, 1), "dense": (160, 192, 1), "long": (160, 256, 1), "moe": (256, 384, 1), "b64": (256, 512, 1),
           "tp2": (160, 384, 2)}
PROMETHEUS = os.environ.get("VY_PROMETHEUS", "http://10.43.136.178")
LABEL = "verity.dev/dispatch"


def kubectl(*args: str, stdin: str | None = None) -> str:
    r = subprocess.run(["kubectl", *args], input=stdin, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"kubectl {' '.join(args[:3])}: {r.stderr.strip()}")
    return r.stdout


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def append(name: str, rec: dict) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    with open(ROOT / name, "a") as f:
        f.write(json.dumps({"t": now_utc(), **rec}, sort_keys=True) + "\n")


def load_template(name: str) -> tuple[list[dict], str]:
    import yaml

    path = TEMPLATES / f"{name}.yaml"
    text = path.read_text()
    docs = [d for d in yaml.safe_load_all(text) if d]
    tasks = [d for d in docs if "run" in d] if len(docs) > 1 else docs
    return tasks, hashlib.sha256(text.encode()).hexdigest()[:12]


def dns(s: str, n: int) -> str:
    return re.sub(r"[^a-z0-9-]", "-", s.lower()).strip("-")[:n].strip("-") or "x"


def task_resources(item: dict, task: dict, idx: int, ntasks: int) -> tuple[int, int, int]:
    res = task.get("resources") or {}
    cpus, mem = int(res.get("cpus", 8)), int(res.get("memory", 32))
    acc = str(res.get("accelerators") or "")
    gpus = int(acc.split(":")[1]) if ":" in acc else 0
    cls = item.get("class")
    if cls:
        if cls not in CLASSES or ntasks != 2:
            raise ValueError(f"class {cls!r}: one of {sorted(CLASSES)}, for a two-task template")
        build, gpu, n = CLASSES[cls]
        if idx == 0:
            mem = build
        else:
            mem, gpus = gpu, n
    o = (item.get("resources") or {}).get(task.get("name") or str(idx)) or {}
    return int(o.get("cpus", cpus)), int(o.get("memory", mem)), int(o.get("gpus", gpus))


def render(item: dict, key: str, idx: int, tries: int) -> dict:
    """The batch/v1 Job for task `idx` of the item's template."""
    tasks, tsha = load_template(item["template"])
    task = tasks[idx]
    cpus, mem, gpus = task_resources(item, task, idx, len(tasks))
    envs = {k: v for k, v in (task.get("envs") or {}).items()}
    envs.update({k: str(v) for k, v in (item.get("env") or {}).items()})
    envs["SRC"] = item["tree"]
    envs["OMP_NUM_THREADS"] = str(cpus)
    if "BUILD_RAM_BUDGET_GB" in envs:
        envs["BUILD_RAM_BUDGET_GB"] = str(mem)
    missing = sorted(k for k, v in envs.items() if v is None)
    if missing:
        raise ValueError(f"template {item['template']} needs env {', '.join(missing)}")
    pod_labels = dict((((task.get("config") or {}).get("ssh") or {}).get("pod_config") or {}).get("metadata", {}).get("labels") or {})
    pod_labels.update({k: str(v) for k, v in ((task.get("resources") or {}).get("labels") or {}).items()})
    if item.get("queue"):
        pod_labels["kueue.x-k8s.io/queue-name"] = item["queue"]
        if item["queue"] in DEFAULT_PRIORITY:
            pod_labels["kueue.x-k8s.io/priority-class"] = DEFAULT_PRIORITY[item["queue"]]
    if item.get("priority"):
        pod_labels["kueue.x-k8s.io/priority-class"] = item["priority"]
    queue = pod_labels.get("kueue.x-k8s.io/queue-name")
    if not queue:
        raise ValueError("no queue: the template has no kueue.x-k8s.io/queue-name label and the item names none")
    khash = hashlib.sha256(key.encode()).hexdigest()[:10]
    tname = dns(task.get("name") or "task", 8)
    name = f"nd-{dns(key.split('/')[0], 14)}-{khash}-{tname}-{tries}"
    ours = {LABEL: "node1", "verity.dev/key": khash, "verity.dev/task": str(idx), "verity.dev/try": str(tries),
            "verity.dev/workstream": dns(key.split("/")[0], 63)}
    labels = {**pod_labels, **ours}
    script = "set -euo pipefail\nexport HOME=/tmp/home && mkdir -p $HOME\n" \
             f"( {task.get('setup') or 'true'}\n)\n{task['run']}"
    requests = {"cpu": str(cpus), "memory": f"{mem}G"}
    limits = {}
    if gpus:
        requests["nvidia.com/gpu"] = limits["nvidia.com/gpu"] = str(gpus)
    pod_spec = {
        "restartPolicy": "Never",
        "nodeSelector": {"verity.dev/server": SERVER},
        "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
        "securityContext": {"runAsUser": 1000, "runAsGroup": 1000},
        "terminationGracePeriodSeconds": int(((((task.get("config") or {}).get("ssh") or {}).get("pod_config") or {})
                                              .get("spec") or {}).get("terminationGracePeriodSeconds", 300)),
        "containers": [{
            "name": "main", "image": IMAGE, "imagePullPolicy": "IfNotPresent",
            "command": ["taskset", "-c", CPUS, "bash", "-c", script],
            "env": [{"name": k, "value": str(v)} for k, v in sorted(envs.items())],
            "envFrom": [{"secretRef": {"name": SECRET}}],
            "resources": {"requests": requests, **({"limits": limits} if limits else {})},
            "volumeMounts": [{"name": "workspace", "mountPath": "/workspace"}, {"name": "dshm", "mountPath": "/dev/shm"}],
        }],
        "volumes": [{"name": "workspace", "hostPath": {"path": "/workspace", "type": "Directory"}},
                    {"name": "dshm", "emptyDir": {"medium": "Memory"}}],
    }
    if gpus:
        pod_spec["runtimeClassName"] = "nvidia"
    return {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"name": name, "labels": labels,
                     "annotations": {"verity.dev/item": key, "verity.dev/template": f"{item['template']}@{tsha}",
                                     "verity.dev/item-json": json.dumps(item, sort_keys=True)}},
        # the pods carry no kueue.x-k8s.io labels: Kueue manages the Job, and its plain-pod integration must not take them too
        "spec": {"suspend": True, "backoffLimit": 0, "ttlSecondsAfterFinished": 6 * 3600,
                 "template": {"metadata": {"labels": {k: v for k, v in labels.items() if not k.startswith("kueue.x-k8s.io/")}},
                              "spec": pod_spec}},
    }


def submit(item: dict, key: str, idx: int = 0, tries: int = 0, dry: bool = False) -> str:
    job = render(item, key, idx, tries)
    name = job["metadata"]["name"]
    if dry:
        print(json.dumps(job, indent=1))
        return name
    kubectl("create", "-f", "-", stdin=json.dumps(job))
    c = job["spec"]["template"]["spec"]["containers"][0]
    append("log.jsonl", {"ev": "submit", "job": name, "key": key, "task": idx, "try": tries,
                         "queue": job["metadata"]["labels"]["kueue.x-k8s.io/queue-name"],
                         "priority": job["metadata"]["labels"].get("kueue.x-k8s.io/priority-class"),
                         "template": job["metadata"]["annotations"]["verity.dev/template"], "tree": item["tree"],
                         "requests": c["resources"]["requests"]})
    print(f"submitted {name} ({key} task {idx} try {tries})")
    return name


def done_keys() -> set[str]:
    p = ROOT / "done.jsonl"
    return {json.loads(line)["key"] for line in p.read_text().splitlines() if line.strip()} if p.exists() else set()


def ready_items() -> list[tuple[int, str, str, dict]]:
    """[(rank, workstream, key, item)] in dispatch order: rank desc, then workstreams take turns, each in file order."""
    per_ws: dict[str, list[tuple[int, str, dict]]] = {}
    for f in sorted((ROOT / "ready").glob("*.jsonl")):
        ws = f.stem
        for n, line in enumerate(f.read_text().splitlines()):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                key = f"{ws}/{item['id']}"
            except (ValueError, KeyError) as e:
                key, item = f"{ws}/line{n + 1}", {"_error": f"unparsable: {e}"}
            per_ws.setdefault(ws, []).append((int(item.get("rank", 0)), key, item))
    out = []
    for rank in sorted({r for items in per_ws.values() for r, _, _ in items}, reverse=True):
        lists = [(ws, [(k, it) for r, k, it in per_ws[ws] if r == rank]) for ws in sorted(per_ws)]
        for i in range(max(len(items) for _, items in lists)):
            out.extend((rank, ws, *items[i]) for ws, items in lists if i < len(items))
    return out


def job_state(job: dict, pods: dict) -> tuple[str, int | None]:
    st = job.get("status") or {}
    if st.get("succeeded"):
        return "succeeded", 0
    if st.get("failed"):
        rc = None
        for p in pods.get(job["metadata"]["name"], []):
            for cs in (p.get("status") or {}).get("containerStatuses") or []:
                term = (cs.get("state") or {}).get("terminated")
                if term:
                    rc = term.get("exitCode")
        return "failed", rc
    if job["spec"].get("suspend"):
        return "pending", None
    return "running", None


def tick(dry: bool = False) -> None:
    jobs = json.loads(kubectl("get", "jobs", "-l", f"{LABEL}=node1", "-o", "json"))["items"]
    pods: dict[str, list] = {}
    for p in json.loads(kubectl("get", "pods", "-l", f"{LABEL}=node1", "-o", "json"))["items"]:
        pods.setdefault((p["metadata"].get("labels") or {}).get("job-name", ""), []).append(p)
    done = done_keys()
    inflight: set[str] = set()
    for j in jobs:
        key = j["metadata"]["annotations"]["verity.dev/item"]
        state, rc = job_state(j, pods)
        if state in ("pending", "running") or j["metadata"]["labels"].get("verity.dev/seen") != "1":
            inflight.add(key)
        if state not in ("succeeded", "failed") or j["metadata"]["labels"].get("verity.dev/seen") == "1":
            continue
        item = json.loads(j["metadata"]["annotations"]["verity.dev/item-json"])
        idx, tries = int(j["metadata"]["labels"]["verity.dev/task"]), int(j["metadata"]["labels"]["verity.dev/try"])
        st = j.get("status") or {}
        wall = None
        if st.get("startTime") and st.get("completionTime"):
            ts = [time.mktime(time.strptime(st[k], "%Y-%m-%dT%H:%M:%SZ")) for k in ("startTime", "completionTime")]
            wall = int(ts[1] - ts[0])
        append("log.jsonl", {"ev": "end", "job": j["metadata"]["name"], "key": key, "task": idx, "try": tries, "state": state,
                             "rc": rc, "wall_s": wall})
        ntasks = len(load_template(item["template"])[0])
        nxt = None
        if state == "succeeded" and idx + 1 < ntasks:
            nxt = (idx + 1, 0)
        elif state == "failed" and rc == REQUEUE_RC and tries + 1 < MAX_TRIES:
            nxt = (idx, tries + 1)
        if not dry:
            if nxt:
                submit(item, key, *nxt)
            else:
                append("done.jsonl", {"key": key, "state": state, "rc": rc, "task": idx, "job": j["metadata"]["name"]})
                inflight.discard(key)
            kubectl("label", "job", j["metadata"]["name"], "verity.dev/seen=1", "--overwrite")
    wls = json.loads(kubectl("get", "workloads", "-o", "json"))["items"]
    pending: dict[str, int] = {}
    for w in wls:
        if not (w.get("status") or {}).get("admission") and not any(
                c.get("type") == "Finished" and c.get("status") == "True" for c in (w.get("status") or {}).get("conditions") or []):
            q = (w.get("spec") or {}).get("queueName", "")
            pending[q] = pending.get(q, 0) + 1
    depth = dict(kv.split("=") for kv in os.environ.get("VY_DISPATCH_DEPTH", DEFAULT_DEPTH).split(","))
    for _, ws, key, item in ready_items():
        if key in done or key in inflight:
            continue
        if "_error" in item:
            append("done.jsonl", {"key": key, "state": "rejected", "why": item["_error"]})
            done.add(key)
            continue
        try:
            q = item.get("queue") or render(item, key, 0, 0)["metadata"]["labels"]["kueue.x-k8s.io/queue-name"]
            if not Path(item["tree"]).is_dir():
                raise ValueError(f"no tree {item['tree']}")
        except (ValueError, KeyError, OSError) as e:
            append("done.jsonl", {"key": key, "state": "rejected", "why": str(e)})
            done.add(key)
            print(f"rejected {key}: {e}")
            continue
        if pending.get(q, 0) >= int(depth.get(q, 0)):
            continue
        submit(item, key, dry=dry)
        pending[q] = pending.get(q, 0) + 1
        inflight.add(key)


def prom(q: str) -> float | None:
    try:
        url = f"{PROMETHEUS}/api/v1/query?" + urllib.parse.urlencode({"query": q})
        r = json.load(urllib.request.urlopen(url, timeout=10))["data"]["result"]
        return float(r[0]["value"][1]) if r else None
    except Exception:  # noqa: BLE001
        return None


def busy() -> dict:
    out = {}
    for w in ("10m", "60m"):
        out[f"gpu_util_{w}"] = prom(f"avg(max by (gpu) (avg_over_time(DCGM_FI_DEV_GPU_UTIL[{w}])))")
        out[f"gpu_busy_{w}"] = prom(f"100 * avg(max by (gpu) (avg_over_time((DCGM_FI_DEV_GPU_UTIL > bool 0)[{w}:1m])))")
        out[f"cpu_busy_{w}"] = prom(f"100 * (1 - avg(rate(node_cpu_seconds_total{{mode=\"idle\"}}[{w}])))")
    return {k: (round(v, 1) if v is not None else None) for k, v in out.items()}


def status() -> None:
    print(kubectl("get", "clusterqueue"), end="")
    print(kubectl("get", "jobs", "-l", f"{LABEL}=node1", "-L", "kueue.x-k8s.io/queue-name"), end="")
    print(json.dumps(busy()))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("template")
    s.add_argument("key", help="workstream/id")
    s.add_argument("--tree", required=True)
    s.add_argument("--queue")
    s.add_argument("--priority")
    s.add_argument("--class", dest="cls")
    s.add_argument("--env", action="append", default=[])
    s.add_argument("--resources", type=json.loads, help='JSON: {"<task>": {"cpus": N, "memory": GB, "gpus": N}}')
    s.add_argument("--dry-run", action="store_true")
    t = sub.add_parser("tick")
    t.add_argument("--dry-run", action="store_true")
    lp = sub.add_parser("loop")
    lp.add_argument("--every", type=int, default=60)
    sub.add_parser("status")
    ns = ap.parse_args(argv)
    if ns.cmd == "submit":
        item = {"id": ns.key.split("/", 1)[-1], "template": ns.template, "tree": ns.tree,
                "env": dict(e.split("=", 1) for e in ns.env)}
        for k, v in (("queue", ns.queue), ("priority", ns.priority), ("class", ns.cls), ("resources", ns.resources)):
            if v:
                item[k] = v
        submit(item, ns.key, dry=ns.dry_run)
    elif ns.cmd == "tick":
        tick(ns.dry_run)
    elif ns.cmd == "loop":
        while True:
            try:
                tick()
                print(f"{now_utc()} tick {json.dumps(busy())}", flush=True)
            except Exception as e:  # noqa: BLE001  (a failed tick is retried next time, never the end of the loop)
                print(f"{now_utc()} tick failed: {e}", file=sys.stderr, flush=True)
            time.sleep(ns.every)
    else:
        status()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
