"""vy-nebius-1's Kueue dispatcher: plain Kubernetes Jobs from the sky/jobs templates, no SkyPilot, keeping every queue deep.

    dispatch.py submit TEMPLATE KEY --tree PATH [--queue Q] [--priority P] [--class C] [--env K=V ...]   one item, now
    dispatch.py tick [--dry-run]      one pass: advance chains, requeue exit 99, top every queue up from the ready files
    dispatch.py loop [--every 60]     tick forever (tmux `node1-dispatch` on the server, as research)
    dispatch.py status                queues, this dispatcher's jobs, GPU-busy and CPU-busy (DCGM / node-exporter, 10 and 60 min)

Runs on the server with a Python that has PyYAML (`/workspace/jobs/venv312/bin/python`) and research's kubeconfig.

**Templates.** Each task of `sky/jobs/<TEMPLATE>.yaml` becomes one batch/v1 Job labelled `kueue.x-k8s.io/queue-name` (Kueue admits
it by quota and requeues it itself when it is preempted). The Job runs the task's `setup` then `run` with the task's envs, as the
image's uid 1000, with /workspace mounted, and the R2 store credentials from the Secret `research-r2` (the templates' `secrets`).
Tasks of a multi-task template (config-run: build, plan, gpu, replay) chain: the next task is submitted when the previous one succeeds.
A task that exits 99 is resubmitted (a chunk that ended early or was told to yield); any other failure ends the item.
An item is pinned to its template's content at its first submit: the item carries `template_sha` (the `name@sha` its submit event
logs) and a copy is kept at `$VY_DISPATCH_ROOT/templates.by-sha/<sha>/<template>.yaml`. Every later task, resubmission and spooled
Commit of its chain renders from that copy, never from the live `sky/jobs/`, so a template refreshed mid-chain applies only to items
submitted after it. A Job whose pinned copy is gone is held (left unseen, printed each tick), never chained on another template.
Every task starts under `taskset -c`: a `provers` task on $VY_PROVER_CPUS (128-175, three 16-core slices: the floor of 2 GPUs plus 1
borrowed), every other task on $VY_DISPATCH_CPUS (96-127,176-191), so vLLM jobs never run on a prover's slice (Daniel 1 Oct 01:35Z;
note:20261001T0137Z-handoff-from-proofs-flock-fp-vllm-jobs-on-prover-slices). 0-7 are k3s's, 8-95 the RC's three train-check slots.

**Ready files.** One JSON file per item, `$VY_READY/<workstream>/<id>.json` (default /workspace/jobs/ready, which the steward's
exporter counts as `vy_ready_jobs{lane}` for the idle alert), written by the workstream's owner (write a temp name, then rename):

    {"template": "config-run", "tree": "/workspace/research/trees/<lane>", "queue": "circuits", "priority": "circuits",
     "class": "small", "rank": 0, "env": {"ROW": "...", "ROLE": "B0", "REPO": "...", "REVISION": "..."}}

The item's id is the file's stem. On submit the file moves to `$VY_DISPATCH_ROOT/taken/<workstream>/` (default root
/workspace/jobs/dispatch), so the ready directory holds only work not yet submitted.
`queue` and `priority` default to the template's labels (`backfill` to priority `backfill`); `class` is submit.sh's VY_ROW_CLASS
table (config-run's build and gpu tasks, found by name); `resources` ({"<task>": {"cpus": N, "memory": GB, "gpus": N}}) overrides
it. Order: higher `rank` first; within a rank the workstreams take turns, each oldest file first. A queue is topped up while it has
fewer than its depth of pending workloads (anyone's; VY_DISPATCH_DEPTH, default circuits=4,provers=2,backfill=4). A chain's next
task is submitted regardless of depth.

**GPU leases.** An item with `"lease": "wrap"` (or whose `workstream/template` is in VY_LEASE_CLASSES, comma-separated) leases its
GPUs per process instead of binding them to its pod (n1_lease.py): the pod requests no `nvidia.com/gpu`, so Kueue books only its CPU
and memory, and runs with hostPID, runtime class nvidia, `NVIDIA_VISIBLE_DEVICES=all` and `/run/gpu-lease` mounted; the task's
`setup` runs first, then its `run` under `gpu-lease N --wait --max-min M --mem-gb 0` (M: the item's `lease_max_min`, else
VY_LEASE_MAX_MIN, 120), which sets CUDA_VISIBLE_DEVICES. `"lease": "self"` sets the pod up the same way and leaves the leasing to
the job (`"$GPU_LEASE" 1 --wait -- ...` around each GPU phase), so it can release the GPU for a CPU phase and lease one again.
At most VY_LEASE_DEPTH (5) leased tasks per queue are in flight, since an admitted one may wait for a GPU while Kueue holds its CPUs.

**Record.** `log.jsonl` gets one line per submit and per end (job, item key, task, queue, rc, wall); `done.jsonl` one per ended item.
Both are append-only: a wrong outcome is corrected by a later `done.jsonl` line for the key (and an `ev: correct` log line) that
names the line it corrects in `corrects` and cites its evidence; an item's outcome is its last `done.jsonl` line (`status`).
Inside the Job, the template's own `research run` writes the Attempt.

**Packing (PACK_COMMITS=1; default 0).** config-run's gpu task of an eligible item (`packable`: TP1, a model of <= 4B on
PACK_MODELS, batch <= 8, a tree with the deferred replay; never TP2, Gemma-2, Pythia or 7B+) is spooled, rendered as its Job would
be, to `$PACK_SPOOL/queue/` (default <root>/pack) instead of submitted, and still-suspended gpu Jobs of eligible items are taken
off Kueue into the spool (PACK_ADOPT, default 1). At most PACK_PODS (1 during node 1's storage emergency; infra 3:36 PM PDT)
`commit-pack` Jobs run at once (sky/jobs/commit-pack.yaml, its pilot sky/commit_pack.py), each running up to 3 spooled Commits on its
one GPU under MPS. With /workspace at PACK_DISK_PAUSE (80%) or fuller, nothing new is spooled, adopted or started (the Commit goes
out as a normal Job), and the pilot claims nothing. Their `results/` come back into the
chain whatever the flag: rc 0 submits the next task (the CPU replay), 99 spools it again, and anything else (an MPS fault included)
resubmits the task once as a normal 1-GPU Job with `unpacked` set on the item; a Commit that then fails ends as any failure does.
Entries claimed by a pod that is gone count as a fault. Rollback: PACK_COMMITS=0, then let the pods finish or delete them.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATES = HERE / "sky" / "jobs"
ROOT = Path(os.environ.get("VY_DISPATCH_ROOT", "/workspace/jobs/dispatch"))
PINNED = ROOT / "templates.by-sha"
IMAGE = os.environ.get("VY_DISPATCH_IMAGE", "us-docker.pkg.dev/sky-dev-465/skypilotk8s/skypilot-gpu:202605150216")
READY = Path(os.environ.get("VY_READY", "/workspace/jobs/ready"))
CPUS = os.environ.get("VY_DISPATCH_CPUS", "96-127,176-191")
PROVER_CPUS = os.environ.get("VY_PROVER_CPUS", "128-175")
SERVER = os.environ.get("VY_DISPATCH_SERVER", "vy-nebius-1")
SECRET = "research-r2"
REQUEUE_RC = 99
MAX_TRIES = 50
DEFAULT_DEPTH = "circuits=4,provers=2,backfill=4"
DEFAULT_PRIORITY = {"backfill": "backfill"}
PROMETHEUS = os.environ.get("VY_PROMETHEUS", "http://10.43.136.178")
LABEL = "verity.dev/dispatch"
LEASE_DIR = "/run/gpu-lease"
LEASE_BIN = os.environ.get("VY_GPU_LEASE", "/workspace/verity-guest/bin/gpu-lease")
PACK_COMMITS = os.environ.get("PACK_COMMITS", "0") == "1"
PACK_ADOPT = os.environ.get("PACK_ADOPT", "1") == "1"
PACK = Path(os.environ.get("PACK_SPOOL", str(ROOT / "pack")))
PACK_PODS = int(os.environ.get("PACK_PODS", "1"))
PACK_SLOTS = 3
PACK_TEMPLATE = "commit-pack"
PACK_MODELS = ("smollm2-135m", "smollm2-360m", "tinyllama-11b", "qwen25-05b", "qwen25-15b", "llama32-1b", "phi3-mini", "qwen3-4b")
PACK_MAX_BATCH = 8
PACK_DISK_PAUSE = float(os.environ.get("PACK_DISK_PAUSE", "0.80"))
QUIET = ((12, 30), (13, 30))


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


def template_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def load_template(name: str, sha: str | None = None) -> tuple[list[dict], str]:
    """The template's tasks and sha: sky/jobs/<name>.yaml as it is now, or with `sha` exactly that content (its pinned copy, else
    the live file while it still hashes to `sha`)."""
    import yaml

    live = TEMPLATES / f"{name}.yaml"
    if sha is None:
        text = live.read_text()
    else:
        text = next((t for t in (p.read_text() for p in (PINNED / sha / f"{name}.yaml", live) if p.is_file())
                     if template_sha(t) == sha), None)
        if text is None:
            raise ValueError(f"template {name}@{sha}: no copy in {PINNED / sha}, and sky/jobs/{name}.yaml has other content")
    docs = [d for d in yaml.safe_load_all(text) if d]
    tasks = [d for d in docs if "run" in d] if len(docs) > 1 else docs
    return tasks, template_sha(text)


def pin(item: dict, dry: bool = False) -> dict:
    """The item pinned to its template as it is now, keeping a copy under PINNED (an item's first submit); a pinned item as is."""
    if item.get("template_sha") is not None:
        return item
    text = (TEMPLATES / f"{item['template']}.yaml").read_text()
    sha = template_sha(text)
    dst = PINNED / sha / f"{item['template']}.yaml"
    if not dry and not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
        tmp.write_text(text)
        tmp.rename(dst)
    return {**item, "template_sha": sha}


def pinned_item(item: dict, template: str) -> dict:
    """`item` pinned to `template` (a Job's or spool entry's `name@sha`): Jobs rendered before pinning carry it only there."""
    return item if item.get("template_sha") is not None else {**item, "template_sha": template.rpartition("@")[2]}


def job_item(job: dict) -> dict:
    """The item a Job was rendered from, pinned to that Job's template."""
    a = job["metadata"]["annotations"]
    return pinned_item(json.loads(a["verity.dev/item-json"]), a.get("verity.dev/template", ""))


def dns(s: str, n: int) -> str:
    return re.sub(r"[^a-z0-9-]", "-", s.lower()).strip("-")[:n].strip("-") or "x"


def classes() -> dict[str, tuple[int, int, int]]:
    """submit.sh's VY_ROW_CLASS table, {class: (build GB, gpu GB, GPUs)} for config-run's build and gpu tasks, read from submit.sh
    itself."""
    m = re.search(r"(?ms)^table = (\{.*?\})\s*(#[^\n]*)?\n(?!\s)", (HERE / "sky" / "submit.sh").read_text())
    if not m:
        raise ValueError("no VY_ROW_CLASS table in sky/submit.sh")
    return ast.literal_eval(re.sub(r"#[^\n]*", "", m.group(1)))


def task_resources(item: dict, task: dict, idx: int, ntasks: int) -> tuple[int, int, int]:
    res = task.get("resources") or {}
    cpus, mem = int(res.get("cpus", 8)), int(res.get("memory", 32))
    acc = str(res.get("accelerators") or "")
    gpus = int(acc.split(":")[1]) if ":" in acc else 0
    cls, name = item.get("class"), task.get("name")
    if cls:
        table = classes()
        if cls not in table or ntasks not in (2, 3, 4):
            raise ValueError(f"class {cls!r}: one of {sorted(table)}, for a pipeline template (build and gpu, and config-run's "
                             "plan and replay)")
        build, gpu, n = table[cls]
        if name == "build":
            mem = build
        elif name == "gpu":
            mem, gpus, cpus = gpu, n, 4 * n          # submit.sh: the GPU queue's fixed 4 vCPU per GPU
        # config-run's 0-GPU plan and replay tasks keep their template resources, as submit.sh does without VY_*_MEMORY
    o = (item.get("resources") or {}).get(name or str(idx)) or {}
    return int(o.get("cpus", cpus)), int(o.get("memory", mem)), int(o.get("gpus", gpus))


def render(item: dict, key: str, idx: int, tries: int) -> dict:
    """The batch/v1 Job for task `idx` of the item's template (its pinned content, when the item is pinned)."""
    tasks, tsha = load_template(item["template"], item.get("template_sha"))
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
    lease = lease_mode(item, key) if gpus else None
    run = task["run"]
    if lease:
        labels["verity.dev/lease"] = lease
        envs.update({"NVIDIA_VISIBLE_DEVICES": "all", "NVIDIA_DRIVER_CAPABILITIES": "compute,utility", "GPU_LEASE_DIR": LEASE_DIR,
                     "GPU_LEASE_ALLOWED_FILE": f"{LEASE_DIR}/pool.gpus", "GPU_LEASE_WHO": dns(key.split("/")[0], 63),
                     "GPU_LEASE_N": str(gpus), "GPU_LEASE": LEASE_BIN})
        if lease == "wrap":
            m = int(item.get("lease_max_min") or os.environ.get("VY_LEASE_MAX_MIN", "120"))
            run = f'"$GPU_LEASE" {gpus} --wait --max-min {m} --mem-gb 0 -- bash -c {shlex.quote("set -euo pipefail; " + run)}'
        run = "umask 000\n" + run
        gpus = 0
    script = "set -euo pipefail\nexport HOME=/tmp/home && mkdir -p $HOME\n" \
             f"( {task.get('setup') or 'true'}\n)\n{run}"
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
            "command": ["taskset", "-c", PROVER_CPUS if queue == "provers" else CPUS, "bash", "-c", script],
            "env": [{"name": k, "value": str(v)} for k, v in sorted(envs.items())],
            "envFrom": [{"secretRef": {"name": SECRET}}],
            "resources": {"requests": requests, **({"limits": limits} if limits else {})},
            "volumeMounts": [{"name": "workspace", "mountPath": "/workspace"}, {"name": "dshm", "mountPath": "/dev/shm"}],
        }],
        "volumes": [{"name": "workspace", "hostPath": {"path": "/workspace", "type": "Directory"}},
                    {"name": "dshm", "emptyDir": {"medium": "Memory"}}],
    }
    if gpus or lease:
        pod_spec["runtimeClassName"] = "nvidia"
    if lease:
        pod_spec["hostPID"] = True
        pod_spec["volumes"].append({"name": "gpu-lease", "hostPath": {"path": LEASE_DIR, "type": "Directory"}})
        pod_spec["containers"][0]["volumeMounts"].append({"name": "gpu-lease", "mountPath": LEASE_DIR})
    return {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"name": name, "labels": labels,
                     "annotations": {"verity.dev/item": key, "verity.dev/template": f"{item['template']}@{tsha}",
                                     "verity.dev/item-json": json.dumps(item, sort_keys=True)}},
        # the pods carry no kueue.x-k8s.io labels: Kueue manages the Job, and its plain-pod integration must not take them too.
        # podReplacementPolicy Failed: a Job Kueue preempts and resumes starts its next pod only once the old one has ended, never
        # beside it in the same row (note:20261001T0026Z-finding-from-mps-pack-preempted-commit-runs-twice)
        "spec": {"suspend": True, "backoffLimit": 0, "ttlSecondsAfterFinished": 6 * 3600, "podReplacementPolicy": "Failed",
                 "template": {"metadata": {"labels": {k: v for k, v in labels.items() if not k.startswith("kueue.x-k8s.io/")}},
                              "spec": pod_spec}},
    }


def lease_mode(item: dict, key: str) -> str | None:
    """"wrap", "self" or None: the item's `lease`, else wrap when its workstream/template is in VY_LEASE_CLASSES."""
    if item.get("lease"):
        if item["lease"] not in ("wrap", "self"):
            raise ValueError(f"lease {item['lease']!r}: wrap or self")
        return item["lease"]
    classes = {c.strip() for c in os.environ.get("VY_LEASE_CLASSES", "").split(",") if c.strip()}
    return "wrap" if f"{key.split('/')[0]}/{item['template']}" in classes else None


def submit(item: dict, key: str, idx: int = 0, tries: int = 0, dry: bool = False) -> str:
    item = pin(item, dry)
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


def quiet(t: float | None = None) -> bool:
    g = time.gmtime(time.time() if t is None else t)
    return QUIET[0] <= (g.tm_hour, g.tm_min) < QUIET[1]


def packable(item: dict, key: str, idx: int) -> str | None:
    """Why task `idx` of the item isn't packed, or None when it is (the cutover plan's eligibility)."""
    if item.get("template") != "config-run":
        return "not a config-run"
    if lease_mode(item, key):
        return "leased (its GPU comes from gpu-lease)"
    tasks, _ = load_template(item["template"], item.get("template_sha"))
    if (tasks[idx].get("name") or "") != "gpu":
        return "not the gpu task"
    if item.get("unpacked"):
        return f"unpacked ({item['unpacked']})"
    env = item.get("env") or {}
    if str(env.get("REPLAY_DEFERRED", "auto")) == "0":
        return "REPLAY_DEFERRED=0 (replays on its GPU)"
    parts = str(env.get("ROW", "")).split("__")
    if parts[0] not in PACK_MODELS:
        return f"model {parts[0]!r} is not on the packing list"
    if "tp1" not in parts:
        return "not TP1"
    batch = next((int(p[1:]) for p in parts if re.fullmatch(r"b\d+", p)), None)
    if batch is None or batch > PACK_MAX_BATCH:
        return f"batch {batch} (packing takes <= {PACK_MAX_BATCH})"
    if task_resources(item, tasks[idx], idx, len(tasks))[2] != 1:
        return "not a 1-GPU task"
    tools = Path(item["tree"]) / "integrations/vllm/verity_vllm/pipeline/research_tools.py"
    try:
        if "VLLM_REPLAY" not in tools.read_text():
            return "the tree has no deferred replay"
    except OSError:
        return f"no {tools}"
    return None


def spool(item: dict, key: str, idx: int, tries: int = 0, dry: bool = False) -> str:
    """Queue task `idx` for a commit-pack pod: the Job it would be, as its env and script."""
    item = pin(item, dry)
    job = render(item, key, idx, tries)
    name = job["metadata"]["name"]
    c = job["spec"]["template"]["spec"]["containers"][0]
    entry = {"key": key, "item": item, "idx": idx, "tries": tries, "job": name, "spooled": now_utc(),
             "template": job["metadata"]["annotations"]["verity.dev/template"],
             "env": {e["name"]: e["value"] for e in c["env"]}, "script": c["command"][-1]}
    if dry:
        print(json.dumps(entry, indent=1))
        return name
    pack_tree()
    q = PACK / "queue"
    tmp = q / f".{name}.tmp"
    tmp.write_text(json.dumps(entry, sort_keys=True))
    tmp.rename(q / f"{name}.json")
    append("log.jsonl", {"ev": "spool", "job": name, "key": key, "task": idx, "try": tries, "template": entry["template"],
                         "tree": item["tree"]})
    print(f"spooled {name} ({key} task {idx} try {tries}) for commit-pack")
    return name


def pack_tree() -> None:
    """The spool's directories, writable by the pack pods (uid 1000) as well as this dispatcher."""
    for d in ("queue", "claimed", "results/done", "logs", "stats"):
        (PACK / d).mkdir(parents=True, exist_ok=True)
    for d in (PACK, *(PACK / d for d in ("queue", "claimed", "results", "results/done", "logs", "stats"))):
        if d.stat().st_uid == os.getuid():
            d.chmod(0o777)


def disk_full() -> bool:
    u = shutil.disk_usage(os.environ.get("PACK_DISK_PATH", "/workspace"))
    return u.used / u.total >= PACK_DISK_PAUSE


def route(item: dict, key: str, idx: int, tries: int = 0, dry: bool = False) -> str:
    if PACK_COMMITS and packable(item, key, idx) is None and not disk_full():
        return spool(item, key, idx, tries, dry)
    return submit(item, key, idx, tries, dry)


def pack_keys() -> set[str]:
    """Keys whose task is in the spool: queued, claimed or with a result not yet read back."""
    keys = set()
    for f in [*PACK.glob("queue/*.json"), *PACK.glob("claimed/*/*.json"), *PACK.glob("results/*.json")]:
        try:
            keys.add(json.loads(f.read_text())["key"])
        except (OSError, ValueError, KeyError):
            continue
    return keys


def pack_results(pods: list[dict], dry: bool = False) -> None:
    """Read commit-pack results back into their chains; entries claimed by a pod that is gone are a fault."""
    live = {p["metadata"]["name"] for p in pods if (p.get("status") or {}).get("phase") in ("Pending", "Running")}
    for d in sorted(p for p in PACK.glob("claimed/*") if p.is_dir()):
        if d.name in live:
            continue
        for f in d.glob("*.json"):
            if dry:
                print(f"would requeue {f.name}: pod {d.name} is gone")
                continue
            res = PACK / "results" / f.name
            if not res.exists():
                entry = json.loads(f.read_text())
                res.parent.mkdir(parents=True, exist_ok=True)
                res.write_text(json.dumps({"name": f.stem, "key": entry["key"], "idx": entry["idx"], "tries": entry["tries"],
                                           "entry": entry, "rc": None, "fault": True, "pod": d.name,
                                           "reason": f"pod {d.name} ended without a result"}, sort_keys=True))
            f.unlink(missing_ok=True)
        if not dry and not any(d.iterdir()):
            d.rmdir()
    for r in sorted(PACK.glob("results/*.json")):
        res = json.loads(r.read_text())
        entry = res["entry"]
        item, key, idx, tries, rc = entry["item"], entry["key"], entry["idx"], entry["tries"], res.get("rc")
        item = pinned_item(item, entry.get("template", ""))
        try:
            ntasks = len(load_template(item["template"], item["template_sha"])[0])
        except (OSError, ValueError) as e:
            print(f"held {r.name} ({key}): {e}", file=sys.stderr)
            continue
        if dry:
            print(f"would read back {r.name}: rc {rc} fault {res.get('fault')}, on {item['template']}@{item['template_sha']}")
            continue
        append("log.jsonl", {"ev": "end", "job": entry["job"], "key": key, "task": idx, "try": tries,
                             "state": "succeeded" if rc == 0 else "failed", "rc": rc, "wall_s": res.get("wall_s"),
                             "packed": res.get("pod"), "fault": bool(res.get("fault")), "reason": res.get("reason")})
        if rc == 0 and idx + 1 < ntasks:
            route(item, key, idx + 1, 0)
        elif rc == REQUEUE_RC and not res.get("fault") and tries + 1 < MAX_TRIES:
            route(item, key, idx, tries + 1)
        elif rc != 0 and not item.get("unpacked"):
            why = res.get("reason") or f"rc {rc}"
            append("log.jsonl", {"ev": "unpack", "key": key, "task": idx, "why": why, "pod": res.get("pod")})
            submit({**item, "unpacked": f"{res.get('pod')}: {why}"[:200]}, key, idx, tries + 1)
        else:
            append("done.jsonl", {"key": key, "state": "succeeded" if rc == 0 else "failed", "rc": rc, "task": idx,
                                  "job": entry["job"], "packed": res.get("pod")})
        done_dir = PACK / "results" / "done"
        done_dir.mkdir(parents=True, exist_ok=True)
        r.rename(done_dir / r.name)


def pack_adopt(jobs: list[dict], dry: bool = False) -> None:
    """Move still-suspended gpu Jobs of eligible items into the spool, keeping it at most two pods' slots deep."""
    if disk_full():
        return
    room = 2 * PACK_PODS * PACK_SLOTS - len(list(PACK.glob("queue/*.json")))
    for j in sorted(jobs, key=lambda j: j["metadata"].get("creationTimestamp", "")):
        if room <= 0:
            return
        if not j["spec"].get("suspend") or (j.get("status") or {}).get("startTime"):
            continue
        item = job_item(j)
        idx, tries = int(j["metadata"]["labels"]["verity.dev/task"]), int(j["metadata"]["labels"]["verity.dev/try"])
        name, key = j["metadata"]["name"], j["metadata"]["annotations"]["verity.dev/item"]
        try:
            if packable(item, key, idx) is not None:
                continue
        except (OSError, ValueError):
            continue
        if dry:
            print(f"would adopt {name} ({key})")
            room -= 1
            continue
        fresh = json.loads(kubectl("get", "job", name, "-o", "json"))
        if not fresh["spec"].get("suspend") or (fresh.get("status") or {}).get("startTime"):
            continue
        kubectl("delete", "job", name, "--wait=false")
        append("log.jsonl", {"ev": "adopt", "job": name, "key": key, "task": idx, "try": tries})
        spool(item, key, idx, tries)
        room -= 1


def pack_pods(jobs: list[dict], dry: bool = False) -> None:
    """Keep up to PACK_PODS commit-pack Jobs while the spool holds more than the running pods' free slots."""
    if quiet() or disk_full():
        return
    active = [j for j in jobs if j["metadata"]["annotations"].get("verity.dev/template", "").startswith(PACK_TEMPLATE + "@")
              and not (j.get("status") or {}).get("succeeded") and not (j.get("status") or {}).get("failed")]
    claimed = len(list(PACK.glob("claimed/*/*.json")))
    queued = len(list(PACK.glob("queue/*.json")))
    if queued and len(active) < PACK_PODS and queued > PACK_SLOTS * len(active) - claimed:
        submit_pack_pod(dry=dry)


def submit_pack_pod(priority: str | None = None, dry: bool = False) -> str:
    item = {"id": "p", "template": PACK_TEMPLATE, "tree": str(HERE), "env": {"PACK_SPOOL": str(PACK)}}
    if priority:
        item["priority"] = priority
    key = f"{PACK_TEMPLATE}/p{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}{os.getpid() % 100:02d}"
    if not dry:
        pack_tree()
    return submit(item, key, dry=dry)


def done_keys() -> set[str]:
    p = ROOT / "done.jsonl"
    return {json.loads(line)["key"] for line in p.read_text().splitlines() if line.strip()} if p.exists() else set()


def ready_items() -> list[tuple[str, dict, Path]]:
    """[(key, item, file)] in dispatch order: rank desc, then workstreams take turns, each oldest file first."""
    per_ws: dict[str, list[tuple[int, str, dict, Path]]] = {}
    for d in sorted(p for p in READY.glob("*") if p.is_dir()):
        files = []
        for f in d.glob("*.json"):
            try:
                files.append((f.stat().st_mtime, f))
            except OSError:
                continue
        for _, f in sorted(files):
            key = f"{d.name}/{f.stem}"
            try:
                item = {**json.loads(f.read_text()), "id": f.stem}
            except (ValueError, OSError) as e:
                item = {"id": f.stem, "_error": f"unparsable: {e}"}
            per_ws.setdefault(d.name, []).append((int(item.get("rank", 0) or 0), key, item, f))
    out = []
    for rank in sorted({r for items in per_ws.values() for r, *_ in items}, reverse=True):
        lists = [[(k, it, f) for r, k, it, f in per_ws[ws] if r == rank] for ws in sorted(per_ws)]
        for i in range(max(len(items) for items in lists)):
            out.extend(items[i] for items in lists if i < len(items))
    return out


def set_aside(f: Path, key: str, where: str) -> None:
    dst = ROOT / where / key.split("/")[0]
    dst.mkdir(parents=True, exist_ok=True)
    try:
        f.rename(dst / f"{f.stem}.{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    except OSError as e:
        print(f"could not move {f} to {dst}: {e}", file=sys.stderr)


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
    pod_list = json.loads(kubectl("get", "pods", "-l", f"{LABEL}=node1", "-o", "json"))["items"]
    for p in pod_list:
        pods.setdefault((p["metadata"].get("labels") or {}).get("job-name", ""), []).append(p)
    pack_results(pod_list, dry)
    done = done_keys()
    inflight: set[str] = pack_keys()
    leased: dict[str, int] = {}
    for j in jobs:
        key = j["metadata"]["annotations"]["verity.dev/item"]
        state, rc = job_state(j, pods)
        if state in ("pending", "running") and j["metadata"]["labels"].get("verity.dev/lease"):
            q = j["metadata"]["labels"].get("kueue.x-k8s.io/queue-name", "")
            leased[q] = leased.get(q, 0) + 1
        if state in ("pending", "running") or j["metadata"]["labels"].get("verity.dev/seen") != "1":
            inflight.add(key)
        if state not in ("succeeded", "failed") or j["metadata"]["labels"].get("verity.dev/seen") == "1":
            continue
        item = job_item(j)
        idx, tries = int(j["metadata"]["labels"]["verity.dev/task"]), int(j["metadata"]["labels"]["verity.dev/try"])
        try:
            ntasks = len(load_template(item["template"], item["template_sha"])[0])
        except (OSError, ValueError) as e:
            print(f"held {j['metadata']['name']} ({key}): {e}", file=sys.stderr)
            continue
        st = j.get("status") or {}
        wall = None
        if st.get("startTime") and st.get("completionTime"):
            ts = [time.mktime(time.strptime(st[k], "%Y-%m-%dT%H:%M:%SZ")) for k in ("startTime", "completionTime")]
            wall = int(ts[1] - ts[0])
        nxt = None
        if state == "succeeded" and idx + 1 < ntasks:
            nxt = (idx + 1, 0)
        elif state == "failed" and rc == REQUEUE_RC and tries + 1 < MAX_TRIES:
            nxt = (idx, tries + 1)
        if dry:
            then = "the item ends" if not nxt else \
                f"then {render(item, key, *nxt)['metadata']['name']} (task {nxt[0]} of {ntasks})"
            print(f"would end {j['metadata']['name']} ({key}, {state} rc {rc}) on {item['template']}@{item['template_sha']}: {then}")
        else:
            append("log.jsonl", {"ev": "end", "job": j["metadata"]["name"], "key": key, "task": idx, "try": tries, "state": state,
                                 "rc": rc, "wall_s": wall})
            if nxt:
                route(item, key, *nxt)
            else:
                append("done.jsonl", {"key": key, "state": state, "rc": rc, "task": idx, "job": j["metadata"]["name"]})
                inflight.discard(key)
            kubectl("label", "job", j["metadata"]["name"], "verity.dev/seen=1", "--overwrite")
    if PACK_COMMITS and PACK_ADOPT:
        pack_adopt(jobs, dry)
    wls = json.loads(kubectl("get", "workloads", "-o", "json"))["items"]
    pending: dict[str, int] = {}
    for w in wls:
        if not (w.get("status") or {}).get("admission") and not any(
                c.get("type") == "Finished" and c.get("status") == "True" for c in (w.get("status") or {}).get("conditions") or []):
            q = (w.get("spec") or {}).get("queueName", "")
            pending[q] = pending.get(q, 0) + 1
    depth = dict(kv.split("=") for kv in os.environ.get("VY_DISPATCH_DEPTH", DEFAULT_DEPTH).split(","))
    for key, item, f in ready_items():
        try:
            if key in done or key in inflight:
                raise ValueError("duplicate id: this item was already submitted (a changed item needs a new id)")
            if "_error" in item:
                raise ValueError(item["_error"])
            job0 = render(item, key, 0, 0)
            q = job0["metadata"]["labels"]["kueue.x-k8s.io/queue-name"]
            if not Path(item["tree"]).is_dir():
                raise ValueError(f"no tree {item['tree']}")
        except (ValueError, KeyError, OSError, IndexError) as e:
            print(f"rejected {key}: {e}")
            if not dry:
                append("log.jsonl", {"ev": "rejected", "key": key, "why": str(e)})
                if key not in done and key not in inflight:
                    append("done.jsonl", {"key": key, "state": "rejected", "why": str(e)})
                done.add(key)
                set_aside(f, key, "rejected")
            continue
        if pending.get(q, 0) >= int(depth.get(q, 0)):
            continue
        is_leased = bool(job0["metadata"]["labels"].get("verity.dev/lease"))
        if is_leased and leased.get(q, 0) >= int(os.environ.get("VY_LEASE_DEPTH", "5")):
            continue
        try:
            submit(item, key, dry=dry)
        except RuntimeError as e:
            print(f"rejected {key}: {e}")
            append("log.jsonl", {"ev": "rejected", "key": key, "why": str(e)})
            append("done.jsonl", {"key": key, "state": "rejected", "why": str(e)})
            set_aside(f, key, "rejected")
            continue
        if not dry:
            set_aside(f, key, "taken")
        pending[q] = pending.get(q, 0) + 1
        if is_leased:
            leased[q] = leased.get(q, 0) + 1
        inflight.add(key)
    if PACK_COMMITS:
        pack_pods(json.loads(kubectl("get", "jobs", "-l", f"{LABEL}=node1", "-o", "json"))["items"], dry)


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


def outcomes() -> dict[str, dict]:
    """Each ended item's outcome: its last done.jsonl line (a correction supersedes the line it names)."""
    p = ROOT / "done.jsonl"
    return {r["key"]: r for r in (json.loads(line) for line in p.read_text().splitlines() if line.strip())} if p.exists() else {}


def status() -> None:
    print(kubectl("get", "clusterqueue"), end="")
    print(kubectl("get", "jobs", "-l", f"{LABEL}=node1", "-L", "kueue.x-k8s.io/queue-name"), end="")
    states: dict[str, int] = {}
    for r in outcomes().values():
        states[r.get("state", "?")] = states.get(r.get("state", "?"), 0) + 1
    print(json.dumps({"items": states}, sort_keys=True))
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
    s.add_argument("--task", type=int, default=0, help="start the chain at this task (resume after an earlier task succeeded)")
    s.add_argument("--try", dest="tries", type=int, default=0, help="the try number in the Job name (a resubmission)")
    s.add_argument("--template-sha", help="render from this pinned template (resume a chain on the template it started on)")
    s.add_argument("--dry-run", action="store_true")
    sp = sub.add_parser("spool", help="queue one item's task for a commit-pack pod now, whatever PACK_COMMITS says")
    sp.add_argument("template")
    sp.add_argument("key", help="workstream/id")
    sp.add_argument("--tree", required=True)
    sp.add_argument("--priority")
    sp.add_argument("--env", action="append", default=[])
    sp.add_argument("--task", type=int, default=1)
    sp.add_argument("--dry-run", action="store_true")
    pp = sub.add_parser("pack-pod", help="submit one commit-pack Job now")
    pp.add_argument("--priority")
    pp.add_argument("--dry-run", action="store_true")
    t = sub.add_parser("tick")
    t.add_argument("--dry-run", action="store_true")
    lp = sub.add_parser("loop")
    lp.add_argument("--every", type=int, default=60)
    sub.add_parser("status")
    ns = ap.parse_args(argv)
    if ns.cmd == "submit":
        item = {"id": ns.key.split("/", 1)[-1], "template": ns.template, "tree": ns.tree,
                "env": dict(e.split("=", 1) for e in ns.env)}
        for k, v in (("queue", ns.queue), ("priority", ns.priority), ("class", ns.cls), ("resources", ns.resources),
                     ("template_sha", ns.template_sha)):
            if v:
                item[k] = v
        submit(item, ns.key, ns.task, ns.tries, dry=ns.dry_run)
    elif ns.cmd == "spool":
        item = {"id": ns.key.split("/", 1)[-1], "template": ns.template, "tree": ns.tree,
                "env": dict(e.split("=", 1) for e in ns.env), **({"priority": ns.priority} if ns.priority else {})}
        why = packable(item, ns.key, ns.task)
        if why:
            raise SystemExit(f"not packable: {why}")
        spool(item, ns.key, ns.task, dry=ns.dry_run)
    elif ns.cmd == "pack-pod":
        submit_pack_pod(ns.priority, dry=ns.dry_run)
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
