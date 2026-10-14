"""vy-nebius-1's Kueue dispatcher: plain Kubernetes Jobs from the sky/jobs templates, no SkyPilot, keeping every queue deep.

    dispatch.py submit TEMPLATE KEY --tree PATH [--queue Q] [--priority P] [--class C] [--env K=V ...]   one item, now
    dispatch.py tick [--dry-run]      one pass: advance chains, requeue exit 99, top every queue up from the ready files
    dispatch.py loop [--every 60]     tick forever, each tick a fresh `loop --once` process (unit vy-node1-dispatch, as research)
    dispatch.py status                queues, this dispatcher's jobs, GPU-busy and CPU-busy (DCGM / node-exporter, 10 and 60 min)
    dispatch.py pack-stops | pack-lift MODEL --by circuits --why TEXT    models stopped from packing; lift one (circuits only)

Runs on the server with a Python that has PyYAML (`/workspace/jobs/venv312/bin/python`) and research's kubeconfig.
Every invocation reads its settings (the VY_* and PACK_* below) from `$VY_DISPATCH_ROOT/dispatch.env` (`KEY=VALUE` lines, `#`
comments), under any the caller's environment sets: a one-off `submit`, such as n2_build.sh's and n2_commit.sh's over ssh from
node 2, renders its Job with the loop's CPU slices and lease pools, not the defaults. The loop is one invocation per tick: each tick
runs in a fresh process from the environment the loop started with, so an edit to dispatch.env applies at the next tick and a key
removed from it drops out, with no restart (2 Oct: a lend of cores 128-159 to provers never reached a loop that had read the file
once, at its start).

**Templates.** Each task of `sky/jobs/<TEMPLATE>.yaml` becomes one batch/v1 Job labelled `kueue.x-k8s.io/queue-name` (Kueue admits
it by quota and requeues it itself when it is preempted). The Job runs the task's `setup` then `run` with the task's envs, as the
image's uid 1000, with /workspace mounted, and the R2 store credentials from the Secret `research-r2` (the templates' `secrets`).
Tasks of a multi-task template (config-run: build, then gpu) chain: the next task is submitted when the previous one succeeds.
A task that exits 99 is resubmitted (a chunk that ended early or was told to yield); any other failure ends the item.
An error while handling one finished Job (routing its next task, labelling it seen) is that Job's: it is logged as
`item-failed`, the Job stays unseen for the next tick, and the tick goes on to the other Jobs and the ready files; the loop then
prints `tick failed`, which the steward's failed-tick alert counts (2 Oct: three AlreadyExists stalls in 10 h each stopped every tick).
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
table (config-run's two tasks); `resources` ({"<task>": {"cpus": N, "memory": GB, "gpus": N}}) overrides it. Order: higher `rank`
first; within a rank the workstreams take turns, each oldest file first. A queue is topped up while it has fewer than its depth of
pending workloads (anyone's; VY_DISPATCH_DEPTH, default circuits=4,provers=2,backfill=4). A chain's next task is submitted
regardless of depth.

**GPU leases.** An item with `"lease": "wrap"` (or whose `workstream/template` is in VY_LEASE_CLASSES, comma-separated) leases its
GPUs per process instead of binding them to its pod (n1_lease.py): the pod requests no `nvidia.com/gpu`, so Kueue books only its CPU
and memory, and runs with hostPID, runtime class nvidia, `NVIDIA_VISIBLE_DEVICES=all` and `/run/gpu-lease` mounted; the task's
`setup` runs first, then its `run` under `gpu-lease N --wait --max-min M --mem-gb 0` (M: the item's `lease_max_min`, else
VY_LEASE_MAX_MIN, 120), which sets CUDA_VISIBLE_DEVICES. `"lease": "self"` sets the pod up the same way and leaves the leasing to
the job (`"$GPU_LEASE" 1 --wait -- ...` around each GPU phase), so it can release the GPU for a CPU phase and lease one again.
At most VY_LEASE_DEPTH (5) leased tasks per queue are in flight, since an admitted one may wait for a GPU while Kueue holds its CPUs.
A leased pod's GPUs come from its queue's pool: VY_LEASE_HOSTDIRS (`QUEUE=DIR,...`, one n1_lease.py per pool) names the host
directory mounted at `/run/gpu-lease`, by default `deployments-gpu=/run/gpu-lease-circuits` so circuits' Commits lease from holders
in their own quota; any other queue mounts /run/gpu-lease, the `provers` pool. The default holds for every caller (the loop, a
`dispatch.py submit` over ssh, a script that imports this module); set the variable empty to map no queue.

**Record.** `log.jsonl` gets one line per submit and per end (job, item key, task, queue, rc, wall), and an `exists` line when the
task's Job was already created, as this task and try, by another path; `done.jsonl` one per ended item.
Both are append-only: a wrong outcome is corrected by a later `done.jsonl` line for the key (and an `ev: correct` log line) that
names the line it corrects in `corrects` and cites its evidence; an item's outcome is its last `done.jsonl` line (`status`).
Inside the Job, the template's own `research run` writes the Attempt.

**Packing (PACK_COMMITS=1; default 0).** config-run's gpu task of an eligible item (`packable`: TP1, a model of <= 4B on
PACK_MODELS, batch <= 8, a tree with the deferred replay; never TP2, Gemma-2, Pythia or 7B+) is spooled, rendered as its Job would
be, to `$PACK_SPOOL/queue/` (default <root>/pack) instead of submitted, and still-suspended gpu Jobs of eligible items are taken
off Kueue into the spool (PACK_ADOPT, default 1). At most PACK_PODS (1 during node 1's storage emergency; infra 3:36 PM PDT)
`commit-pack` Jobs run at once (sky/jobs/commit-pack.yaml, its pilot sky/commit_pack.py), each running up to 3 spooled Commits on its
one GPU under MPS; one starts only when a queued entry passes the pilot's replay-bundle cap with every claimed entry counted as
running, since a pod that can claim nothing idles 3 min and exits (66 of them ran no Commit on Oct 1). With /workspace at
PACK_DISK_PAUSE (80%) or fuller, nothing new is spooled, adopted or started (the Commit goes
out as a normal Job), and the pilot claims nothing. Their `results/` come back into the
chain whatever the flag: rc 0 submits the next task (the CPU replay), 99 spools it again, and anything else (an MPS fault included)
resubmits the task once as a normal 1-GPU Job with `unpacked` set on the item; a Commit that then fails ends as any failure does.
Entries claimed by a pod that is gone count as a fault. Rollback: PACK_COMMITS=0, then let the pods finish or delete them.
A packed Commit's replay task is judged (circuits' yes, 10:55 PM PDT Sep 30): it passes only when it exits 0 and the row's
`config_record.json`, written by that replay, is outcome PASS with replay k = evaluated = equal = 460 and nothing mismatched or
unevaluated (`packed_replay`: `commit-replay` exits 0 on a PASS that carries a named gap, so the exit alone is not enough).
Anything short ends the item failed and stops packing for its model (`stop_model`): an entry in `$PACK_SPOOL/stopped-models.json`
(append-only, the last entry per model decides), an `ev: pack-stop` line and an alert note for node1-dispatcher in PACK_ALERT_DIR
(the steward's alert_pull.sh delivers it), after which `packable` refuses the model. Only circuits lifts a stop
(`dispatch.py pack-lift MODEL --by circuits ...`). Each packed Commit's Attempt is labelled `ov.mps_packed` by its pilot.
From 12:10Z to the quiet hour's end (13:30Z; Kueue holds node 1's queues at 12:10Z) nothing is spooled, adopted or started: queued
entries go back out as normal Jobs, pack Jobs not yet started are deleted, and from 12:33Z any pack Job still active is deleted, so
its Commits requeue unpacked (the pilot itself drains at 12:28Z).

**Spilling to node 2 (while `$VY_DISPATCH_ROOT/n2-spill.on` exists).** A GPU Job of this dispatcher that Kueue has held
VY_N2_SPILL_WAIT_S (120 s) without ever starting it, whose item carries an `n2` block, goes to vy-nebius-2's fill queue when
node 2's fill runner would start it now: a GPU whose gpu-lease lock is free, not in its keep-free, no Commit, pous or
higher-prio Verity GPU job queued for it, 2+ GPUs free (the runner's rule for Verity GPU jobs), no timed window running or
waiting, and no booked window within the block's max_min. At most one per tick, never in node 1's hold (12:10-13:30Z):

    "n2": {"cmd": "...", "owner": "bc-...", "max_min": 25, "gpus": 1, "cpus": 16, "mem_gb": 96, "prio": 5, "pool": "provers",
           "cpuset": "160-175", "on": "0-3", "cwd": "/..."}      (cmd, owner and max_min (<= 30) required; gpus 1)

The fill job (`n2spill-<Job>.sh`) runs `bash -c cmd` (`pool: provers`: under /usr/local/bin/vy-provers, with VY_PROVERS_CPUS
= `cpuset` when given) with N2_SPILL_ITEM (the item's JSON on node 2), N2_SPILL_KEY, N2_SPILL_JOB and N2_SPILL_TASK set. The
Kueue workload is deactivated (spec.active=false, only at the resourceVersion that showed it unadmitted), and the spill is
recorded in `<root>/n2-spill/<Job>.json` and an `ev: n2-spill` log line. Each tick reads node 2's `result.json` for every
outstanding spill: exit 0 ends the task as the Job would have (the Job is deleted, the chain continues); any other exit, a third
start (it was stopped twice), or a fill job gone from node 2 without a result reactivates the workload, so node 1 runs it
(`ev: n2-spill-back`). Rollback: remove the flag; outstanding spills still resolve.

**The queue's placement (while `$VY_DISPATCH_ROOT/place.on` exists).** After `submit` creates a Job of a workstream in
VY_PLACE_WORKSTREAMS (circuits' `vllm-epoch-run`; circuits' yes, 1 Oct 5:15 AM PDT), it asks the queue
(`cluster submit` from $VY_QUEUE_SRC, default /workspace/verity-guest/cluster-src) where that Job would go, with the Job's
requests and its research question, and appends the answer to $VY_PLACED (/workspace/verity-guest/placed.jsonl,
`executor: kueue`). The question is the item's `question`, else its key family's kind in the queue's `kinds/circuits.toml`
(the first kind one of whose `keys` globs matches the item id). It only records: the Job is rendered and created before
the queue is asked, so its spec, env and tree never depend on the answer, and a queue that refuses or doesn't answer is
recorded as such. Rollback: remove the flag.
"""
from __future__ import annotations

import argparse
import ast
import calendar
import fnmatch
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import tomllib
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATES = HERE / "sky" / "jobs"
ROOT = Path(os.environ.get("VY_DISPATCH_ROOT", "/workspace/jobs/dispatch"))


def load_settings(path: Path) -> None:
    """`KEY=VALUE` lines of `path` into os.environ, each only where the caller's environment doesn't set it; no file, no change."""
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return
    for line in lines:
        k, sep, v = line.strip().partition("=")
        if sep and k and not k.startswith("#"):
            os.environ.setdefault(k.strip(), v.strip())


CALLER_ENV = dict(os.environ)   # before dispatch.env: what each of the loop's ticks starts from
load_settings(ROOT / "dispatch.env")
PINNED = ROOT / "templates.by-sha"
IMAGE = os.environ.get("VY_DISPATCH_IMAGE", "us-docker.pkg.dev/sky-dev-465/skypilotk8s/skypilot-gpu:202605150216")
READY = Path(os.environ.get("VY_READY", "/workspace/jobs/ready"))
CPUS = os.environ.get("VY_DISPATCH_CPUS", "96-127,176-191")
PROVER_CPUS = os.environ.get("VY_PROVER_CPUS", "128-175")
SERVER = os.environ.get("VY_DISPATCH_SERVER", "vy-nebius-1")
SECRET = "research-r2"
REQUEUE_RC = 99
TICK_FAILED = 3     # `loop --once`'s exit status after it has printed `tick failed: ...`
MAX_TRIES = 50
DEFAULT_DEPTH = "circuits=4,provers=2,backfill=4"
DEFAULT_PRIORITY = {"backfill": "backfill"}
PROMETHEUS = os.environ.get("VY_PROMETHEUS", "http://10.43.136.178")
LABEL = "verity.dev/dispatch"
LEASE_DIR = "/run/gpu-lease"
LEASE_HOSTDIRS = dict(kv.split("=", 1) for kv in os.environ.get("VY_LEASE_HOSTDIRS", "deployments-gpu=/run/gpu-lease-circuits")
                      .split(",") if "=" in kv)
LEASE_BIN = os.environ.get("VY_GPU_LEASE", "/workspace/verity-guest/bin/gpu-lease")
PACK_COMMITS = os.environ.get("PACK_COMMITS", "0") == "1"
PACK_ADOPT = os.environ.get("PACK_ADOPT", "1") == "1"
PACK = Path(os.environ.get("PACK_SPOOL", str(ROOT / "pack")))
PACK_PODS = int(os.environ.get("PACK_PODS", "1"))
PACK_SLOTS = 3
PACK_TEMPLATE = "commit-pack"
PACK_MODELS = ("smollm2-135m", "smollm2-360m", "tinyllama-11b", "qwen25-05b", "qwen25-15b", "llama32-1b", "phi3-mini", "qwen3-4b",
               # a model joins only once a packed twin's run root equals its unpacked row's
               "qwen3-06b", "qwen25-05b-instruct", "falcon3-1b", "qwen25-coder-15b", "r1-distill-qwen-15b", "qwen3-17b", "smollm2-17b",
               "qwen25-3b", "llama32-3b")
PACK_MAX_BATCH = 8
PACK_DISK_PAUSE = float(os.environ.get("PACK_DISK_PAUSE", "0.80"))
PACK_REPLAY_K = 460
PACK_ALERTS = Path(os.environ.get("PACK_ALERT_DIR", "/workspace/monitoring/alerts/outbox"))
QUIET = ((12, 30), (13, 30))
PACK_HOLD = ((12, 10), QUIET[1])     # node 1's daily quota cutover and quiet hour: Kueue holds every queue at 12:10Z
PACK_DRAIN = (12, 33)                # pack pods must be gone by 12:35Z
N2 = os.environ.get("VY_N2", "vy-n2")
SPILL = ROOT / "n2-spill"
SPILL_FLAG = Path(os.environ.get("VY_N2_SPILL_FLAG", str(ROOT / "n2-spill.on")))
SPILL_WAIT_S = int(os.environ.get("VY_N2_SPILL_WAIT_S", "120"))
SPILL_TRIES = 2
SPILL_SLACK_MIN = 2                  # node 2's runner starts a queued job at its own poll, a little after this tick
N2_FILL = os.environ.get("VY_N2_FILL", "/workspace/pouw/fill")
N2_LEASES = "/run/gpu-lease"
N2_SPILL = os.environ.get("VY_N2_SPILL_DIR", "/workspace/verity-guest/spill")
VY_PROVERS = "/usr/local/bin/vy-provers"
PLACE_FLAG = Path(os.environ.get("VY_DISPATCH_PLACE_FLAG", str(ROOT / "place.on")))
QUEUE_SRC = Path(os.environ.get("VY_QUEUE_SRC", "/workspace/verity-guest/cluster-src"))
PLACED = Path(os.environ.get("VY_PLACED", "/workspace/verity-guest/placed.jsonl"))
PLACE_TIMEOUT_S = 30
PLACE_WORKSTREAMS = {w for w in os.environ.get("VY_PLACE_WORKSTREAMS", "vllm-epoch-run").split(",") if w}


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
    """submit.sh's VY_ROW_CLASS table, {class: (build GB, gpu GB, GPUs)} for config-run's two tasks, read from submit.sh itself."""
    m = re.search(r"(?ms)^table = (\{.*?\})\s*(#[^\n]*)?\n(?!\s)", (HERE / "sky" / "submit.sh").read_text())
    if not m:
        raise ValueError("no VY_ROW_CLASS table in sky/submit.sh")
    return ast.literal_eval(re.sub(r"#[^\n]*", "", m.group(1)))


def task_resources(item: dict, task: dict, idx: int, ntasks: int) -> tuple[int, int, int]:
    res = task.get("resources") or {}
    cpus, mem = int(res.get("cpus", 8)), int(res.get("memory", 32))
    acc = str(res.get("accelerators") or "")
    gpus = int(acc.split(":")[1]) if ":" in acc else 0
    cls = item.get("class")
    if cls:
        table = classes()
        if cls not in table or ntasks not in (2, 3):
            raise ValueError(f"class {cls!r}: one of {sorted(table)}, for a pipeline template (build, gpu, and config-run's replay)")
        build, gpu, n = table[cls]
        if idx == 0:
            mem = build
        elif idx == 1:
            mem, gpus, cpus = gpu, n, 4 * n          # submit.sh: the GPU queue's fixed 4 vCPU per GPU
        # config-run's replay task (idx 2) keeps its template resources, as submit.sh does without VY_REPLAY_MEMORY
    o = (item.get("resources") or {}).get(task.get("name") or str(idx)) or {}
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
        pod_spec["volumes"].append({"name": "gpu-lease", "hostPath": {"path": LEASE_HOSTDIRS.get(queue, LEASE_DIR), "type": "Directory"}})
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


def kind_of(key: str) -> dict:
    """The kind in the queue's kinds/circuits.toml whose `keys` globs first match the item id, with its name; {} for none."""
    try:
        kinds = tomllib.loads((QUEUE_SRC / "kinds" / "circuits.toml").read_text()).get("kinds", {})
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    stem = key.split("/")[-1]
    for name, d in kinds.items():
        if any(fnmatch.fnmatchcase(stem, g) for g in d.get("keys", ())):
            return {"name": name, **d}
    return {}


def quantity(v, unit: float) -> float:
    """A Kubernetes quantity ("500m", "8", "64Gi", "96G") in `unit` (1 for CPUs, 1e9 for GB)."""
    s = str(v)
    for suffix, mult in (("m", 1e-3), ("Ki", 2**10), ("Mi", 2**20), ("Gi", 2**30), ("Ti", 2**40), ("k", 1e3), ("M", 1e6),
                         ("G", 1e9), ("T", 1e12)):
        if s.endswith(suffix):
            return float(s[:-len(suffix)]) * mult / unit
    return float(s) / unit


def place(item: dict, key: str, job: dict) -> dict:
    """The queue's placement of a created Job, appended to PLACED; never raises (a failure is the row's `refused` or `error`)."""
    req = job["spec"]["template"]["spec"]["containers"][0]["resources"]["requests"]
    kind = kind_of(key)
    question = str(item.get("question") or kind.get("question") or "")
    row = {"t": now_utc(), "kind": "dispatch", "key": key, "job": job["metadata"]["name"], "family": kind.get("name"),
           "queue": job["metadata"]["labels"].get("kueue.x-k8s.io/queue-name"), "question": question, "executor": "kueue"}
    cmd = [sys.executable, "-m", "cluster", "submit", str(QUEUE_SRC / "descriptions" / "nebius.toml"),
           "--run-id", row["job"], "--now", str(int(time.time())), "--project", "verity", "--submitter", "circuits",
           "--question", question, "--gpus", str(int(quantity(req.get("nvidia.com/gpu", 0), 1))),
           "--cpus", str(max(1, round(quantity(req.get("cpu", 1), 1)))),
           "--mem-gb", str(max(1, round(quantity(req.get("memory", "1G"), 1e9)))), "--preempt", "requeue", "--node", SERVER]
    if kind:
        cmd += ["--kind", kind["name"]]
    if job["spec"].get("activeDeadlineSeconds"):
        cmd += ["--max-s", str(job["spec"]["activeDeadlineSeconds"])]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=PLACE_TIMEOUT_S,
                           env={**os.environ, "PYTHONPATH": str(QUEUE_SRC / "src")})
        out = json.loads(r.stdout.strip().splitlines()[-1])
        if "refused" in out:
            row["refused"] = out["refused"]
        else:
            row.update(node=out.get("node"), why=out.get("why"), question=out.get("question") or question)
    except (OSError, subprocess.TimeoutExpired, ValueError, IndexError) as e:
        row["error"] = f"the queue gave no placement: {type(e).__name__}"
    try:
        PLACED.parent.mkdir(parents=True, exist_ok=True)
        with open(PLACED, "a") as f:
            f.write(json.dumps(row) + "\n")
    except OSError as e:
        print(f"placed.jsonl: {e}", file=sys.stderr)
    return row


def submitted(name: str, key: str, idx: int, tries: int) -> bool:
    """Whether Job `name` is already this item's task `idx`, try `tries`: submitted by another path (by hand, or a feeder) before the
    tick routed the task before it. Without this, the tick's `create` fails on every pass and nothing after that Job is handled."""
    try:
        meta = json.loads(kubectl("get", "job", name, "-o", "json"))["metadata"]
    except (RuntimeError, ValueError, KeyError):
        return False
    labels = meta.get("labels") or {}
    return ((meta.get("annotations") or {}).get("verity.dev/item"), labels.get("verity.dev/task"), labels.get("verity.dev/try")) \
        == (key, str(idx), str(tries))


def submit(item: dict, key: str, idx: int = 0, tries: int = 0, dry: bool = False) -> str:
    item = pin(item, dry)
    job = render(item, key, idx, tries)
    name = job["metadata"]["name"]
    if dry:
        print(json.dumps(job, indent=1))
        return name
    try:
        kubectl("create", "-f", "-", stdin=json.dumps(job))
    except RuntimeError as e:
        if "AlreadyExists" not in str(e) or not submitted(name, key, idx, tries):
            raise
        append("log.jsonl", {"ev": "exists", "job": name, "key": key, "task": idx, "try": tries})
        print(f"{name} ({key} task {idx} try {tries}) already exists: counted as submitted")
        return name
    c = job["spec"]["template"]["spec"]["containers"][0]
    append("log.jsonl", {"ev": "submit", "job": name, "key": key, "task": idx, "try": tries,
                         "queue": job["metadata"]["labels"]["kueue.x-k8s.io/queue-name"],
                         "priority": job["metadata"]["labels"].get("kueue.x-k8s.io/priority-class"),
                         "template": job["metadata"]["annotations"]["verity.dev/template"], "tree": item["tree"],
                         "requests": c["resources"]["requests"]})
    print(f"submitted {name} ({key} task {idx} try {tries})")
    if PLACE_FLAG.exists() and key.split("/")[0] in PLACE_WORKSTREAMS:
        place(item, key, job)
    return name


def quiet(t: float | None = None) -> bool:
    g = time.gmtime(time.time() if t is None else t)
    return QUIET[0] <= (g.tm_hour, g.tm_min) < QUIET[1]


def held(t: float | None = None) -> bool:
    g = time.gmtime(time.time() if t is None else t)
    return PACK_HOLD[0] <= (g.tm_hour, g.tm_min) < PACK_HOLD[1]


def draining(t: float | None = None) -> bool:
    g = time.gmtime(time.time() if t is None else t)
    return PACK_DRAIN <= (g.tm_hour, g.tm_min) < PACK_HOLD[1]


def model_of(item: dict) -> str:
    return str((item.get("env") or {}).get("ROW", "")).split("__")[0]


def stops_file() -> Path:
    return PACK / "stopped-models.json"


def stop_entries() -> list[dict]:
    """The stop list's entries, oldest first ([] when there is none); raises when the file can't be read as a list."""
    p = stops_file()
    if not p.exists():
        return []
    entries = json.loads(p.read_text())
    if not isinstance(entries, list):
        raise ValueError(f"{p} is not a JSON list")
    return entries


def stopped_models() -> dict[str, dict] | None:
    """{model: its stop entry} for every model whose last entry is a stop, or None when the list can't be read."""
    try:
        entries = stop_entries()
    except (OSError, ValueError):
        return None
    out: dict[str, dict] = {}
    for e in entries:
        if e.get("lift"):
            out.pop(e.get("model"), None)
        else:
            out[e.get("model")] = e
    return out


def append_stop(entry: dict) -> None:
    pack_tree()
    entries = stop_entries()
    p = stops_file()
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps([*entries, {"t": now_utc(), **entry}], indent=1, sort_keys=True) + "\n")
    tmp.rename(p)


def stop_model(model: str, key: str, run: str | None, reason: str, **extra) -> None:
    """Stop packing `model`: a stop entry, an `ev: pack-stop` line and an alert note for node1-dispatcher."""
    try:
        append_stop({"model": model, "key": key, "run": run, "reason": reason, **extra})
    except (OSError, ValueError) as e:     # an unreadable list already stops every model (packable)
        extra = {**extra, "unrecorded": str(e)}
    append("log.jsonl", {"ev": "pack-stop", "model": model, "key": key, "run": run, "reason": reason, **extra})
    print(f"pack-stop {model}: {reason} ({key})", file=sys.stderr)
    stamp = time.strftime("%Y%m%dT%H%MZ", time.gmtime())
    stem = f"{stamp}-alert-pack-stop-{dns(model, 40)}-{hashlib.sha256(key.encode()).hexdigest()[:8]}"
    facts = "\n".join(f"- **{k}:** {v}" for k, v in (("Item", f"`{key}`"), ("Runs", f"replay `{run}`, Commit `{extra.get('commit_run')}`"),
                                                       ("Pack pod", f"`{extra.get('pod')}`"), ("Replay Job", f"`{extra.get('job')}`")))
    note = (f"---\nid: {stem}\ncampaign: one-pool\nlane: node1-dispatcher\nkind: finding\nstatus: open\nrepo: compute-verification/verity\n"
            f"origin: node 1's dispatcher (dispatch.py stop_model; MPS Commit packing, circuits' rule of 10:55 PM PDT Sep 30)\n---\n\n"
            f"# MPS packing stopped for {model}: a packed Commit's replay failed\n\n{facts}\n- **Why:** {reason}\n"
            f"- **Effect:** the item ended failed, and `packable` refuses {model} from now on (`{stops_file()}`). Other models still pack.\n"
            f"- **Lift (circuits only):** on node 1, `cd /workspace/jobs/dispatch/infra/nebius && /workspace/jobs/venv312/bin/python "
            f"dispatch.py pack-lift {model} --by circuits --why '...'`\n")
    try:
        PACK_ALERTS.mkdir(parents=True, exist_ok=True)
        tmp = PACK_ALERTS / f".{stem}.md.tmp"
        tmp.write_text(note)
        tmp.rename(PACK_ALERTS / f"{stem}.md")
    except OSError as e:
        append("log.jsonl", {"ev": "pack-stop-alert-failed", "model": model, "key": key, "why": str(e)})
        print(f"pack-stop {model}: no alert note in {PACK_ALERTS}: {e}", file=sys.stderr)


def lift_model(model: str, by: str, why: str) -> None:
    if not by.startswith("circuits"):
        raise SystemExit(f"only circuits lifts a packing stop (--by {by!r})")
    if model not in (stopped_models() or {}):
        raise SystemExit(f"{model!r} is not stopped")
    append_stop({"model": model, "lift": True, "by": by, "why": why})
    append("log.jsonl", {"ev": "pack-lift", "model": model, "by": by, "why": why})
    print(f"lifted the packing stop on {model}")


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
    stops = stopped_models()
    if stops is None:
        return f"the packing stop list {stops_file()} can't be read"
    if parts[0] in stops:
        return f"model {parts[0]!r} stopped packing: {stops[parts[0]].get('reason')} (only circuits lifts it: dispatch.py pack-lift)"
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
    if PACK_COMMITS and not held() and packable(item, key, idx) is None and not disk_full():
        return spool(item, key, idx, tries, dry)
    return submit(item, key, idx, tries, dry)


def row_dir(job: dict) -> Path:
    env = {e["name"]: e["value"] for e in job["spec"]["template"]["spec"]["containers"][0].get("env") or []}
    return Path(env.get("SWEEP_DIR", "")) / env.get("ROW", "")


def packed_replay(job: dict, state: str, rc: int | None) -> tuple[str | None, dict]:
    """(why a packed Commit's ended replay task falls short of 460/460, else None; the record's replay counts). It passes only when it
    exited 0 and the row's config_record.json, written since the Job started, is PASS with k = evaluated = equal = PACK_REPLAY_K and
    nothing mismatched or unevaluated."""
    rec_path = row_dir(job) / "config_record.json"
    try:
        started = (job.get("status") or {}).get("startTime")
        if started and rec_path.stat().st_mtime < calendar.timegm(time.strptime(started, "%Y-%m-%dT%H:%M:%SZ")):
            raise ValueError(f"it predates the replay Job (started {started}): this replay wrote no record")
        rec = json.loads(rec_path.read_text())
    except (OSError, ValueError) as e:
        rec, err = {}, f"no config_record.json of this replay ({e})"
    else:
        err = None
    rp = rec.get("replay") or {}
    counts = {"outcome": rec.get("outcome"), **{k: rp.get(k) for k in ("k", "evaluated", "equal", "mismatch_n", "not_evaluated_n")}}
    k = PACK_REPLAY_K
    if state != "succeeded":
        return f"the replay task failed (rc {rc})", counts
    if err:
        return err, counts
    if (rec.get("outcome"), rp.get("k"), rp.get("evaluated"), rp.get("equal")) == ("PASS", k, k, k) \
            and not rp.get("mismatch_n") and not rp.get("not_evaluated_n"):
        return None, counts
    return (f"replay {rp.get('equal')}/{rp.get('evaluated')} equal of k={rp.get('k')} (mismatch {rp.get('mismatch_n')}, not evaluated "
            f"{rp.get('not_evaluated_n')}, grade {rp.get('grade')}), outcome {rec.get('outcome')}: packing needs {k}/{k} and PASS"), counts


def run_ids(job: dict, pod_list: list[dict]) -> tuple[str | None, str | None]:
    """(the replay's research run id, from its pod's log; the Commit's, from <row>/pipeline-commit-run), None where unknown."""
    try:
        commit = (row_dir(job) / "pipeline-commit-run").read_text().strip() or None
    except OSError:
        commit = None
    replay = None
    for p in pod_list:
        try:
            m = re.search(r"research: run (\S+) ->", kubectl("logs", p["metadata"]["name"], "--limit-bytes=1000000"))
        except RuntimeError:
            continue
        replay = m.group(1) if m else replay
    return replay, commit


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
                             "packed": res.get("pod"), "fault": bool(res.get("fault")), "reason": res.get("reason"),
                             "label": res.get("label")})
        if rc == 0 and idx + 1 < ntasks:
            route({**item, "packed": res.get("pod")}, key, idx + 1, 0)     # its replay is judged by circuits' 460/460 rule (tick)
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
    if disk_full() or held():
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


def active_pack_jobs(jobs: list[dict]) -> list[dict]:
    return [j for j in jobs if j["metadata"]["annotations"].get("verity.dev/template", "").startswith(PACK_TEMPLATE + "@")
            and not (j.get("status") or {}).get("succeeded") and not (j.get("status") or {}).get("failed")]


def pack_hold(jobs: list[dict], dry: bool = False) -> None:
    """In the hold (PACK_HOLD): queued spool entries go out as normal Jobs, pack Jobs not yet started are deleted, and from
    PACK_DRAIN every pack Job still active is deleted (pack_results requeues its claimed Commits unpacked)."""
    if not held():
        return
    for f in sorted(PACK.glob("queue/*.json")):
        if dry:
            print(f"would unspool {f.name}: the pack hold")
            continue
        mine = PACK / "unspooled" / f.name
        mine.parent.mkdir(parents=True, exist_ok=True)
        try:
            f.rename(mine)
        except OSError:
            continue
        entry = json.loads(mine.read_text())
        item = pinned_item(entry["item"], entry.get("template", ""))
        append("log.jsonl", {"ev": "unspool", "job": entry["job"], "key": entry["key"], "task": entry["idx"], "why": "pack hold"})
        submit(item, entry["key"], entry["idx"], entry["tries"])
        mine.unlink(missing_ok=True)
    for j in active_pack_jobs(jobs):
        name, started = j["metadata"]["name"], bool((j.get("status") or {}).get("startTime"))
        if (started and not draining()) or j["metadata"].get("deletionTimestamp"):
            continue
        if dry:
            print(f"would delete {name}: the pack hold")
            continue
        try:
            kubectl("delete", "job", name, "--wait=false")
        except RuntimeError as e:
            print(f"pack hold: {e}", file=sys.stderr)
            continue
        append("log.jsonl", {"ev": "pack-hold-delete", "job": name, "started": started})


def pack_pods(jobs: list[dict], dry: bool = False) -> None:
    """Keep up to PACK_PODS commit-pack Jobs while the spool holds more than the running pods' free slots and a new pod could
    claim one of its entries."""
    if quiet() or held() or disk_full():
        return
    active = active_pack_jobs(jobs)
    claimed = len(list(PACK.glob("claimed/*/*.json")))
    queued = len(list(PACK.glob("queue/*.json")))
    if queued and len(active) < PACK_PODS and queued > PACK_SLOTS * len(active) - claimed:
        if not pack_fits():
            if dry:
                print("would start no pack pod: no queued entry fits the replay-bundle cap")
            return
        submit_pack_pod(dry=dry)


def read_entries(pattern: str) -> list[dict]:
    out = []
    for f in PACK.glob(pattern):
        try:
            out.append(json.loads(f.read_text()))
        except (OSError, ValueError):
            continue
    return out


def pack_fits() -> bool:
    """Whether some queued entry passes the pilot's own claim rule (sky/commit_pack.py's bundle_ok), on the bundles waiting now
    and with every claimed entry counted as running."""
    spec = importlib.util.spec_from_file_location("dispatch_commit_pack", HERE / "sky" / "commit_pack.py")
    pilot = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pilot     # its dataclass looks its module up by name while the module runs
    try:
        spec.loader.exec_module(pilot)
    finally:
        sys.modules.pop(spec.name, None)
    running, waiting = read_entries("claimed/*/*.json"), None
    for entry in read_entries("queue/*.json"):
        if pilot.batch(entry) < pilot.BUNDLE_BATCH:
            return True
        waiting = pilot.bundle_bytes(pilot.BUNDLE_GLOBS) if waiting is None else waiting
        if pilot.bundle_ok(entry, waiting, running):
            return True
    return False


def submit_pack_pod(priority: str | None = None, dry: bool = False) -> str:
    item = {"id": "p", "template": PACK_TEMPLATE, "tree": str(HERE), "env": {"PACK_SPOOL": str(PACK)}}
    if priority:
        item["priority"] = priority
    key = f"{PACK_TEMPLATE}/p{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}{os.getpid() % 100:02d}"
    if not dry:
        pack_tree()
    return submit(item, key, dry=dry)


N2_PROBE = r'''
import fcntl, glob, json, os, re, sys
fill, leases, spill, names = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
def read(p):
    try:
        with open(p, errors="replace") as f:
            return f.read()
    except OSError:
        return None
def held(p):
    try:
        fd = os.open(p, os.O_RDONLY)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        return False
    except OSError:
        return True
    finally:
        os.close(fd)
gpus = sorted(int(m.group(1)) for m in (re.fullmatch(r"(\d+)\.lock", os.path.basename(p)) for p in glob.glob(leases + "/*.lock")) if m)
queued = []
for p in sorted(glob.glob(fill + "/queue/*")):
    m = re.search(r"^#\s*fill:\s*(.*)$", (read(p) or "")[:4000], re.M) if not os.path.basename(p).startswith(".") else None
    if m:
        queued.append([os.path.basename(p), m.group(1)])
dirs = sorted(d for d in os.listdir(fill) if os.path.isdir(os.path.join(fill, d)))
spills = {}
for n in names:
    where = next((d for d in dirs if os.path.isfile(os.path.join(fill, d, n + ".sh"))), None)
    spills[n] = {"where": where, "result": read(os.path.join(spill, n, "result.json"))}
print(json.dumps({"gpus": gpus, "busy": [i for i in gpus if held(f"{leases}/{i}.lock")], "keep_free": read(fill + "/keep-free"),
                  "windows": read(fill + "/windows"), "status": (read(fill + "/status.txt") or "").split("\n")[0],
                  "queued": queued, "spills": spills}))
'''

N2_WRITE = r'''
import json, os, sys
fill, spill = sys.argv[1], sys.argv[2]
d = json.load(sys.stdin)
n = d["name"]
if os.path.exists(os.path.join(spill, n, "result.json")) or any(
        os.path.isfile(os.path.join(fill, q, n + ".sh")) for q in os.listdir(fill) if os.path.isdir(os.path.join(fill, q))):
    sys.exit(n + " is on node 2 already")
os.makedirs(os.path.join(spill, n), exist_ok=True)
with open(os.path.join(spill, n, "item.json"), "w") as f:
    json.dump(d["item"], f, sort_keys=True)
tmp = os.path.join(fill, "queue", "." + n + ".sh.tmp")
with open(tmp, "w") as f:
    f.write(d["script"])
os.chmod(tmp, 0o755)
os.rename(tmp, os.path.join(fill, "queue", n + ".sh"))
print("queued", n)
'''


def n2_python(code: str, *args: str, stdin: str | None = None) -> str:
    """`python3 -c code args...` on node 2 over ssh, its stdout; raises RuntimeError when it fails."""
    try:
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", N2, shlex.join(["python3", "-c", code, *args])],
                           input=stdin, capture_output=True, text=True, timeout=60, check=False)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"ssh {N2}: {e}") from None
    if r.returncode:
        raise RuntimeError(f"ssh {N2}: rc {r.returncode}: {r.stderr.strip()[-300:]}")
    return r.stdout


def n2_block(item: dict) -> dict:
    """The item's `n2` block with its defaults; raises ValueError when it can't become a node-2 fill job."""
    b = item.get("n2")
    if not isinstance(b, dict) or not b.get("cmd") or not b.get("owner"):
        raise ValueError("no n2 block with a cmd and an owner")
    b = {"gpus": 1, "cpus": 8, "prio": 0, **b}
    for k in ("gpus", "cpus", "prio", "max_min", "mem_gb"):
        if k in b:
            b[k] = int(b[k])
    if b["gpus"] != 1:
        raise ValueError(f"n2.gpus {b['gpus']}: a node-2 fill job other than a Commit holds 1 GPU")
    if not 1 <= b.get("max_min", 0) <= 30:
        raise ValueError(f"n2.max_min {b.get('max_min')}: 1-30, the fill runner's cap")
    if b.get("pool") not in (None, "provers"):
        raise ValueError(f"n2.pool {b['pool']!r}: provers or none")
    for k in ("owner", "on", "cpuset"):
        if k in b and not re.fullmatch(r"[\w.,-]+", str(b[k])):
            raise ValueError(f"n2.{k} {b[k]!r}")
    return b


def spill_script(name: str, job: str, key: str, idx: int, queue: str, b: dict) -> str:
    """Node 2's fill job for a spilled task. It keeps its state in N2_SPILL/<name>/: tries (each start), result.json and exit
    (written once it ended for good). A rerun after that only repeats the exit; the third start ends it as stopped (rc null)."""
    d = f"{N2_SPILL}/{name}"
    run = f"bash -c {shlex.quote(b['cmd'])}"
    if b.get("pool") == "provers":
        run = f"{VY_PROVERS} {run}"
    env = {"N2_SPILL_ITEM": f"{d}/item.json", "N2_SPILL_KEY": key, "N2_SPILL_JOB": job, "N2_SPILL_TASK": str(idx),
           **({"VY_PROVERS_CPUS": str(b["cpuset"])} if b.get("cpuset") else {})}
    hdr = (f"# fill: owner={b['owner']} gpus={b['gpus']} max_min={b['max_min']} cpus={b['cpus']} project=verity prio={b['prio']}"
           + (f" mem_gb={b['mem_gb']}" if b.get("mem_gb") else "") + (f" on={b['on']}" if b.get("on") else ""))
    return f"""#!/usr/bin/env bash
{hdr}
# node 1's {key} (task {idx}, Job {job}), held by Kueue in {queue}: spilled by dispatch.py (infra/nebius). Exit 0 ends the task
# on node 1; any other exit, or a third start (stopped twice), sends it back to node 1's queue (node 1's tick reads result.json).
set -uo pipefail
d={shlex.quote(d)}
[ ! -s "$d/result.json" ] || exit "$(cat "$d/exit" 2> /dev/null || echo 1)"
n=$(( $(cat "$d/tries" 2> /dev/null || echo 0) + 1 ))
echo "$n" > "$d/tries"
result() {{
  echo "$2" > "$d/exit"
  printf '{{"rc": %s, "tries": %s, "fill_job": "%s", "t": "%s"}}\\n' "$1" "$n" "${{FILL_JOB:-}}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$d/.result.json"
  mv "$d/.result.json" "$d/result.json"
}}
[ "$n" -le {SPILL_TRIES} ] || {{ result null 1; exit 1; }}
export {' '.join(f'{k}={shlex.quote(v)}' for k, v in env.items())}
cd {shlex.quote(b.get('cwd') or d)} || {{ result 2 2; exit 2; }}
rc=0
{run} || rc=$?
case $rc in
  99) echo $((n - 1)) > "$d/tries"; exit 99 ;;
  124 | 137 | 143) exit 143 ;;
esac
result "$rc" "$rc"
exit "$rc"
"""


def gpu_list(s: str | None) -> set[int]:
    """GPU indices as the fill runner reads them (`7`, `0,1`, `2-7`), before any `# comment`."""
    out: set[int] = set()
    for part in re.split(r"[,\s]+", (s or "").split("#")[0].strip()):
        lo, _, hi = part.partition("-")
        if lo.isdigit() and (not hi or hi.isdigit()):
            out.update(range(int(lo), int(hi or lo) + 1))
    return out


def booked(text: str | None) -> list[tuple[float, float]]:
    """The fill runner's booked windows (`2026-10-01T10:00Z 30 # who`), as (start, end) epoch seconds."""
    out = []
    for line in (text or "").splitlines():
        f = line.split("#")[0].split()
        try:
            s = datetime.fromisoformat(f[0])
            s = (s if s.tzinfo else s.replace(tzinfo=UTC)).timestamp()
            out.append((s, s + 60 * float(f[1] if len(f) > 1 else 30)))
        except (IndexError, ValueError):
            continue
    return out


def n2_room(probe: dict, b: dict, t: float) -> tuple[list[int], str | None]:
    """(node 2's idle GPUs the block's fill job could take, None), or ([], why node 2's runner wouldn't start it now): keep-free,
    the runner's order (Commit guests, then pous, then Verity by prio), its 2+ free GPUs for a Verity GPU job, and the windows."""
    status = probe.get("status") or ""
    if "timed False, window waiting False" not in status:
        return [], f"node 2's runner: {status[:160] or 'no status'}"
    gpus, busy, keep = probe["gpus"], set(probe["busy"]), gpu_list(probe.get("keep_free"))
    on = gpu_list(str(b["on"])) if b.get("on") else None
    idle = [i for i in gpus if i not in busy and i not in keep and (on is None or i in on)]
    if len(gpus) - len(busy) < 2:
        return [], f"{len(gpus) - len(busy)} GPU free on node 2: its runner starts a Verity GPU job only with 2+"
    ahead = 0
    for name, h in probe.get("queued") or []:
        f = dict(re.findall(r"(\w+)=(\S+)", h))
        g = int(f["gpus"]) if f.get("gpus", "").isdigit() else 1
        prio = int(f["prio"]) if f.get("prio", "").isdigit() else 0
        if g and (name.startswith("verity-commit-") or f.get("project", "pous") != "verity" or prio >= b["prio"]):
            ahead += g
    if len(idle) - ahead < b["gpus"]:
        return [], f"node 2's idle GPUs {idle or '-'} (keep-free {sorted(keep) or '-'}) go to {ahead} queued GPU job(s) ahead"
    end = t + 60 * (b["max_min"] + SPILL_SLACK_MIN)
    for s, e in booked(probe.get("windows")):
        if not (end <= s or t >= e):
            return [], f"node 2's window at {time.strftime('%H:%MZ', time.gmtime(s))} is within max_min {b['max_min']}"
    return idle, None


def spill_records() -> list[dict]:
    out = []
    for f in sorted(SPILL.glob("*.json")):
        try:
            out.append(json.loads(f.read_text()))
        except (OSError, ValueError) as e:
            print(f"n2-spill: {f} unreadable: {e}", file=sys.stderr)
    return out


def write_json(p: Path, d: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(d, sort_keys=True))
    tmp.rename(p)


def settle(rec: dict, **kw) -> None:
    write_json(SPILL / "done" / f"{rec['job']}.json", {**rec, **kw, "settled": now_utc()})
    (SPILL / f"{rec['job']}.json").unlink(missing_ok=True)


def ts(s: str) -> float:
    return calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ"))


def spill_candidates(jobs: list[dict], wls: list[dict], t: float) -> list[tuple[dict, dict, dict, dict]]:
    """(Job, workload, item, n2 block) for each GPU Job of this dispatcher that Kueue has held SPILL_WAIT_S and never started,
    whose item has an n2 block, and that was never spilled before; oldest workload first."""
    by_name = {j["metadata"]["name"]: j for j in jobs}
    out = []
    for w in sorted(wls, key=lambda w: w["metadata"].get("creationTimestamp", "")):
        st = w.get("status") or {}
        if (w.get("spec") or {}).get("active") is False or st.get("admission") or any(
                c.get("type") == "Finished" and c.get("status") == "True" for c in st.get("conditions") or []):
            continue
        name = next((r.get("name") for r in w["metadata"].get("ownerReferences") or [] if r.get("kind") == "Job"), None)
        j = by_name.get(name)
        if j is None or not j["spec"].get("suspend") or (j.get("status") or {}).get("startTime") or j["metadata"].get("deletionTimestamp"):
            continue
        if t - ts(w["metadata"]["creationTimestamp"]) < SPILL_WAIT_S:
            continue
        req = ((j["spec"]["template"]["spec"]["containers"][0].get("resources") or {}).get("requests") or {})
        if not int(req.get("nvidia.com/gpu", 0) or 0):
            continue
        if (SPILL / f"{name}.json").exists() or (SPILL / "done" / f"{name}.json").exists():
            continue
        item = job_item(j)
        try:
            b = n2_block(item)
        except (ValueError, TypeError):
            continue
        out.append((j, w, item, b))
    return out


def spill(j: dict, w: dict, item: dict, b: dict, idle: list[int], t: float, dry: bool = False) -> bool:
    """Deactivate the held workload (only while it is still unadmitted at the resourceVersion just read), then queue its fill job
    on node 2; a node-2 write that fails reactivates it at once."""
    name, wl = j["metadata"]["name"], w["metadata"]["name"]
    key, labels = j["metadata"]["annotations"]["verity.dev/item"], j["metadata"]["labels"]
    idx, tries, queue = int(labels["verity.dev/task"]), int(labels["verity.dev/try"]), (w.get("spec") or {}).get("queueName", "")
    n2 = f"n2spill-{name}"
    held_s = int(t - ts(w["metadata"]["creationTimestamp"]))
    if dry:
        print(f"would spill {name} ({key}, held {held_s} s in {queue}) to node 2 as {n2} (idle GPUs {idle})")
        return True
    rec = {"job": name, "workload": wl, "key": key, "task": idx, "try": tries, "queue": queue, "n2": n2, "spilled": now_utc(),
           "held_s": held_s, "idle": idle, "item": item}
    fresh = json.loads(kubectl("get", "workload", wl, "-o", "json"))
    if (fresh.get("spec") or {}).get("active") is False or (fresh.get("status") or {}).get("admission"):
        return False
    write_json(SPILL / f"{name}.json", rec)
    try:
        kubectl("patch", "workload", wl, "--type", "merge", "-p",
                json.dumps({"metadata": {"resourceVersion": fresh["metadata"]["resourceVersion"]}, "spec": {"active": False}}))
    except RuntimeError as e:
        (SPILL / f"{name}.json").unlink(missing_ok=True)
        print(f"n2-spill {name}: not deactivated, it stays on node 1: {e}", file=sys.stderr)
        return False
    try:
        n2_python(N2_WRITE, N2_FILL, N2_SPILL, stdin=json.dumps({"name": n2, "item": item, "script": spill_script(n2, name, key, idx, queue, b)}))
    except RuntimeError as e:
        spill_back(rec, f"node 2 took no fill job: {e}")
        return False
    append("log.jsonl", {"ev": "n2-spill", "job": name, "key": key, "task": idx, "try": tries, "queue": queue, "workload": wl,
                         "n2": n2, "held_s": held_s, "idle": idle, "max_min": b["max_min"], "owner": b["owner"], "pool": b.get("pool")})
    print(f"n2-spill {name} ({key}, held {held_s} s in {queue}) -> node 2 {n2}")
    return True


def spill_back(rec: dict, why: str) -> bool:
    """Back to node 1's queue: the workload active again. False when Kueue couldn't be told (retried next tick)."""
    try:
        kubectl("patch", "workload", rec["workload"], "--type", "merge", "-p", json.dumps({"spec": {"active": True}}))
        outcome = "back"
    except RuntimeError as e:
        if "not found" not in str(e).lower():
            print(f"n2-spill {rec['job']}: not reactivated yet: {e}", file=sys.stderr)
            return False
        outcome = "gone"
        append("done.jsonl", {"key": rec["key"], "state": "failed", "task": rec["task"], "job": rec["job"], "n2": rec["n2"],
                              "why": f"{why}; its node-1 Job is gone"})
    append("log.jsonl", {"ev": "n2-spill-back", "job": rec["job"], "key": rec["key"], "n2": rec["n2"], "why": why, "outcome": outcome})
    print(f"n2-spill {rec['job']}: back to node 1 ({why})")
    settle(rec, outcome=outcome, why=why)
    return True


def spill_done(rec: dict, res: dict) -> None:
    """A spilled task that exited 0 on node 2: its node-1 Job goes, and the chain continues as it would from that Job."""
    item, key, idx = rec["item"], rec["key"], rec["task"]
    try:
        ntasks = len(load_template(item["template"], item["template_sha"])[0])
        kubectl("delete", "job", rec["job"], "--ignore-not-found", "--wait=false")
    except (OSError, ValueError, RuntimeError) as e:
        print(f"n2-spill {rec['job']}: held ({e}); retried next tick", file=sys.stderr)
        return
    append("log.jsonl", {"ev": "end", "job": rec["job"], "key": key, "task": idx, "try": rec["try"], "state": "succeeded", "rc": 0,
                         "n2": rec["n2"], "fill_job": res.get("fill_job"), "tries": res.get("tries")})
    if idx + 1 < ntasks:
        route(item, key, idx + 1, 0)
    else:
        append("done.jsonl", {"key": key, "state": "succeeded", "rc": 0, "task": idx, "job": rec["job"], "n2": rec["n2"]})
    settle(rec, outcome="done", rc=0, fill_job=res.get("fill_job"))


def spill_results(probe: dict, recs: list[dict], dry: bool = False) -> None:
    for rec in recs:
        s = (probe.get("spills") or {}).get(rec["n2"]) or {}
        if not s.get("result"):
            where = s.get("where") or ""
            if where in ("queue", "running") or where.startswith("held"):
                if rec.get("missing"):
                    write_json(SPILL / f"{rec['job']}.json", {**rec, "missing": 0})
                continue
            if rec.get("missing", 0) + 1 < 2:
                if not dry:
                    write_json(SPILL / f"{rec['job']}.json", {**rec, "missing": rec.get("missing", 0) + 1})
                continue
            why = f"node 2 has no result, and its fill job is {'in ' + where if where else 'gone'}"
            print(f"would send {rec['job']} back: {why}") if dry else spill_back(rec, why)
            continue
        try:
            res = json.loads(s["result"])
        except ValueError:
            res = {"rc": None, "unreadable": s["result"][:200]}
        if dry:
            print(f"would read back {rec['n2']}: {res}")
        elif res.get("rc") == 0:
            spill_done(rec, res)
        else:
            stopped = " (stopped twice)" if res.get("rc") is None and "unreadable" not in res else ""
            spill_back(rec, f"node 2 rc {res.get('rc')} after {res.get('tries')} start(s){stopped}")


SPILL_WHY: dict[str, str] = {}


def n2_spill(jobs: list[dict], wls: list[dict], dry: bool = False, t: float | None = None) -> set[str]:
    """Read back every outstanding spill, then spill at most one held GPU Job to node 2 (while SPILL_FLAG exists, outside node 1's
    hold). The workloads it deactivated."""
    t = time.time() if t is None else t
    recs = spill_records()
    cands = spill_candidates(jobs, wls, t) if SPILL_FLAG.exists() and not held(t) else []
    if not recs and not cands:
        return set()
    try:
        probe = json.loads(n2_python(N2_PROBE, N2_FILL, N2_LEASES, N2_SPILL, *[r["n2"] for r in recs]))
    except (RuntimeError, ValueError) as e:
        print(f"n2-spill: node 2 unreadable, nothing spilled or read back: {e}", file=sys.stderr)
        return set()
    if recs:
        spill_results(probe, recs, dry)
    for j, w, item, b in cands:
        name = j["metadata"]["name"]
        idle, why = n2_room(probe, b, t)
        if why:
            if dry or SPILL_WHY.get(name) != why:
                print(f"n2-spill {name}: stays on node 1: {why}")
            SPILL_WHY[name] = why
            continue
        return {w["metadata"]["name"]} if spill(j, w, item, b, idle, t, dry) and not dry else set()
    return set()


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


def end_job(j: dict, key: str, state: str, rc: int | None, pods: dict, inflight: set[str], dry: bool = False) -> None:
    """A finished Job of this dispatcher's, not yet seen: log its end, then route its chain's next task or end its item, and
    label it seen."""
    item = job_item(j)
    idx, tries = int(j["metadata"]["labels"]["verity.dev/task"]), int(j["metadata"]["labels"]["verity.dev/try"])
    try:
        tasks = load_template(item["template"], item["template_sha"])[0]
    except (OSError, ValueError) as e:
        print(f"held {j['metadata']['name']} ({key}): {e}", file=sys.stderr)
        return
    ntasks, name = len(tasks), j["metadata"]["name"]
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
    judged = not nxt and bool(item.get("packed")) and (tasks[idx].get("name") or "") == "replay"
    why, counts = packed_replay(j, state, rc) if judged else (None, {})
    if dry:
        then = "the item ends" if not nxt else \
            f"then {render(item, key, *nxt)['metadata']['name']} (task {nxt[0]} of {ntasks})"
        verdict = "" if not judged else f"; packed replay {'PASS' if why is None else 'FAIL, would stop ' + model_of(item)}: {counts}"
        print(f"would end {name} ({key}, {state} rc {rc}) on {item['template']}@{item['template_sha']}: {then}{verdict}")
        return
    append("log.jsonl", {"ev": "end", "job": name, "key": key, "task": idx, "try": tries, "state": state,
                         "rc": rc, "wall_s": wall})
    if judged:
        append("log.jsonl", {"ev": "packed-replay", "job": name, "key": key, "pod": item["packed"],
                             "verdict": "pass" if why is None else "fail", "why": why, **counts})
    if nxt:
        route(item, key, *nxt)
    else:
        end = {"key": key, "state": state, "rc": rc, "task": idx, "job": name, **({"packed": item["packed"]} if judged else {})}
        if why:
            end.update(state="failed", why=f"packed replay: {why}")
            replay_run, commit_run = run_ids(j, pods.get(name, []))
            stop_model(model_of(item), key, replay_run, why, commit_run=commit_run, pod=item["packed"], job=name)
        append("done.jsonl", end)
        inflight.discard(key)
    kubectl("label", "job", name, "verity.dev/seen=1", "--overwrite")


def tick(dry: bool = False) -> list[str]:
    """One pass. The finished Jobs whose handling raised (`name: error`): each stays unseen, so the next tick tries it again."""
    failed: list[str] = []
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
        try:
            end_job(j, key, state, rc, pods, inflight, dry)
        except Exception as e:  # noqa: BLE001  (one Job's error is that Job's: it stays unseen for the next tick, the rest go on)
            name, why = j["metadata"]["name"], f"{type(e).__name__}: {e}"
            print(f"item-failed {name} ({key}): {why}", file=sys.stderr, flush=True)
            failed.append(f"{name}: {why}")
            if not dry:
                append("log.jsonl", {"ev": "item-failed", "job": name, "key": key, "why": why})
    if PACK_COMMITS and PACK_ADOPT:
        pack_adopt(jobs, dry)
    pack_hold(jobs, dry)
    wls = json.loads(kubectl("get", "workloads", "-o", "json"))["items"]
    spilled = n2_spill(jobs, wls, dry)
    pending: dict[str, int] = {}
    for w in wls:
        if w["metadata"]["name"] in spilled or (w.get("spec") or {}).get("active") is False:
            continue
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
    return failed


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


class _Stop(Exception):
    pass


def run_loop(tick: Callable[[], int], every: float, sleep: Callable[[float], None] | None = None) -> int:
    """`loop`: a tick (its exit status) every `every` s. SIGTERM, the unit's stop (KillMode=mixed sends it to this process
    alone), lets the tick in flight finish and then ends the loop; between ticks it ends it at once."""
    busy = stop = False

    def term(signum, frame) -> None:
        nonlocal stop
        stop = True
        if not busy:
            raise _Stop

    prev = signal.signal(signal.SIGTERM, term)
    try:
        while not stop:
            busy = True
            rc = tick()
            busy = False
            if rc not in (0, TICK_FAILED):    # it died before saying so (an import error in a redeployed copy, a kill)
                print(f"{now_utc()} tick failed: exit {rc}", file=sys.stderr, flush=True)
            if not stop:
                (sleep or time.sleep)(every)
    except _Stop:
        pass
    finally:
        signal.signal(signal.SIGTERM, prev)
    return 0


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
    s.add_argument("--lease", choices=("wrap", "self"), help="the item's \"lease\" (the docstring's GPU leases)")
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
    pl = sub.add_parser("pack-lift", help="let a model stopped by a failed packed replay pack again (circuits only)")
    pl.add_argument("model")
    pl.add_argument("--by", required=True)
    pl.add_argument("--why", required=True)
    sub.add_parser("pack-stops", help="the models stopped from packing, with their stop entries")
    t = sub.add_parser("tick")
    t.add_argument("--dry-run", action="store_true")
    lp = sub.add_parser("loop")
    lp.add_argument("--every", type=int, default=60)
    lp.add_argument("--once", action="store_true", help="one of the loop's ticks, in this process (what the loop runs each tick)")
    sub.add_parser("status")
    ns = ap.parse_args(argv)
    if ns.cmd == "submit":
        item = {"id": ns.key.split("/", 1)[-1], "template": ns.template, "tree": ns.tree,
                "env": dict(e.split("=", 1) for e in ns.env)}
        for k, v in (("queue", ns.queue), ("priority", ns.priority), ("class", ns.cls), ("resources", ns.resources),
                     ("template_sha", ns.template_sha), ("lease", ns.lease)):
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
    elif ns.cmd == "pack-lift":
        lift_model(ns.model, ns.by, ns.why)
    elif ns.cmd == "pack-stops":
        print(json.dumps(stopped_models(), indent=1, sort_keys=True))
    elif ns.cmd == "tick":
        tick(ns.dry_run)
    elif ns.cmd == "loop" and ns.once:
        try:
            failed = tick()
        except Exception as e:  # noqa: BLE001  (a failed tick is retried next time, never the end of the loop)
            print(f"{now_utc()} tick failed: {e}", file=sys.stderr, flush=True)
            return TICK_FAILED
        if failed:      # the rest of the tick ran; the steward's failed-tick alert counts this line
            print(f"{now_utc()} tick failed: {len(failed)} Job(s) left unseen: {'; '.join(failed)[:500]}", file=sys.stderr, flush=True)
            return TICK_FAILED
        print(f"{now_utc()} tick {json.dumps(busy())}", flush=True)
    elif ns.cmd == "loop":
        return run_loop(lambda: subprocess.run([sys.executable, str(Path(__file__).resolve()), "loop", "--once"],
                                               env=CALLER_ENV).returncode, ns.every)
    else:
        status()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
