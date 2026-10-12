#!/usr/bin/env python3
"""vy-nebius-1's process-level GPU leases: node 2's `gpu-lease` (the same lock files and rules) on GPUs that Kubernetes lends it.

    n1_lease.py loop [--every 5]     as research on node 1 (unit vy-n1-lease@POOL, one per pool)
    n1_lease.py tick | status

A job leases a GPU for its GPU phase only: its pod requests no `nvidia.com/gpu`, runs with `hostPID`, runtime class nvidia and
`NVIDIA_VISIBLE_DEVICES=all`, mounts `$GPU_LEASE_DIR`, and runs `gpu-lease N --wait ... -- CMD`, which sets CUDA_VISIBLE_DEVICES
(the dispatcher's `lease` items, dispatch.py). It may release the GPU for a CPU phase and lease one again later.

**The pool.** A GPU is gpu-lease's only while a holder pod (`gpu-pool-*`, 1 GPU, a Kueue Job in `$VY_POOL_QUEUE`) holds it from
the device plugin, so Kubernetes never hands it to another pod. Every other GPU is *fenced*: a `flock -x` on its lock file, held by
a detached process whose owner line says who=kubernetes, so no lease can take it, whatever list a waiting gpu-lease read at its
start. A fence outlives this controller, so a crash fails closed. `pool.gpus` (the allowed file) lists every index.
A holder requests its GPU and nothing else: waiting pods hold their queue's CPU and memory quota, so a holder that needed any could
wait on the very pods it serves. It is BestEffort, so evicted first under memory pressure, and an evicted holder's lease is outside
the pool (below).
Each tick, with the brain absent (local rules):
  - grow: when the waiters want more than the free pool GPUs and pending holders cover, submit holders (at most `$VY_POOL_MAX`,
    and never past the queue's nominal GPU quota less what its other admitted Workloads reserve: a holder on borrowed quota is
    Kueue's to reclaim mid-lease; with the quota unread, no growth). While every waiter is `--preemptible`, holders may borrow
    `$VY_POOL_BORROW` GPUs more (default 0), until `$VY_POOL_BORROW_UNTIL` (UTC) if set: a Commit that reclaims one evicts its
    holder, and the lease on it is then outside the pool and stopped. A holder's GPU is unfenced once its pod runs and names its UUID;
  - shrink: fence a pool GPU that has been free for `$VY_POOL_IDLE_S` with nobody waiting for it, then delete its holder;
    delete pending holders nobody waits for.
With `$VY_POOL_POLICY` set (a command printing this pool's keys of the cluster policy in force, `cluster/policy/v1` JSON:
`python3 -m cluster policy show --key vy-nebius-1.lease-pool.provers --json` from infra/cluster), the pool's `max`, `borrow` and
`borrow-until` come from it instead, read again every `POLICY_EVERY_S`, so an override (`cluster policy override`) takes effect
without a restart (a read that takes longer than `POLICY_TIMEOUT_S` fails, and everything it started is killed). A key it lacks keeps the environment's value; while it fails or answers a `max` or `borrow` that isn't a
whole number of the node's GPUs, the three it last answered stay (the environment's, before it ever has). Each change is logged
(`policy`), and the three in force are written to `$GPU_LEASE_DIR/n1-lease.limits` at each read with where they came from,
`from=policy`, `last-policy` or `environment` (`cluster policy diff` reads them there).
**Heartbeat and expiry.** Each tick reads every lease (lock holders from /proc/locks, the owner line, the pod from the holder's
cgroup) into `n1-lease.json`; that record is the lease's heartbeat. A lease held `$VY_LEASE_EXPIRY_GRACE_S` past its `until=` (a
hung job gpu-lease's own timeout didn't end) has its pod deleted. A lease on a GPU outside the pool (its holder was deleted by
hand, or Kueue reclaimed it) has its pod deleted at once: Kubernetes may already have given that GPU to someone else. A lease
held by a host process (a `research run --queue` job, in no pod) is signalled instead, its gpu-lease scope else its owner pid:
SIGTERM, then SIGKILL once `KILL_AFTER_S` have passed, as gpu-lease's own preemption does.
**Pools.** One controller serves one queue's pool, named `$VY_POOL_NAME` (its holders' `verity.dev/gpu-pool` label; default
node1, the `provers` pool in /run/gpu-lease). A second pool runs as a second controller with its own `$GPU_LEASE_DIR`,
`$VY_POOL_QUEUE`, `$VY_POOL_PRIORITY` and `$VY_POOL_NAME` (circuits': /run/gpu-lease-circuits, `deployments-gpu`, `circuits-gpu`,
circuits): each fences, in its own lock files, every GPU its own holders don't hold, so a waiter gets only its own pool's GPUs.
`$VY_POOL_QUEUE` is the ClusterQueue whose quota the holders take; `$VY_POOL_LOCAL_QUEUE` (default the same name) is the
LocalQueue they are submitted to. Circuits' holders use `deployments-gpu-pool`, so the Commit pacer's hold on the
`deployments-gpu` LocalQueue never holds them back (sky/release.py).
**Blocked.** While `$GPU_LEASE_DIR/blocked` exists (gpu_stray.py writes it when a process of the pool's pods runs on a GPU it
holds no lease for; a person removes it once that's cleared), the pool grants nothing new: leases already held go on, every
other pool GPU is fenced and its holder deleted, and pending holders are cancelled.
**The brain.** With `$VY_LEASE_BRAIN` set (a command, e.g. `ssh vy-cluster cluster grant --node vy-nebius-1`), each tick sends
it the state as JSON on stdin and reads `{"grants": {"<pid>": [index, ...]}, "pool_target": N}`. While it answers, this controller
holds `agent.lock`, so every request queues for a grant (gpu-lease agent mode); after `BRAIN_MISSES` failures in a row it drops the
lock and the requests fall back to gpu-lease's local rules in place.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DIR = Path(os.environ.get("GPU_LEASE_DIR", "/run/gpu-lease"))
QUEUE = os.environ.get("VY_POOL_QUEUE", "provers")
LOCAL_QUEUE = os.environ.get("VY_POOL_LOCAL_QUEUE", QUEUE)   # the LocalQueue its holders are submitted to, on ClusterQueue QUEUE
PRIORITY = os.environ.get("VY_POOL_PRIORITY", "dev")
POOL_MAX = int(os.environ.get("VY_POOL_MAX", "3"))
IDLE_S = int(os.environ.get("VY_POOL_IDLE_S", "120"))
GRACE_S = int(os.environ.get("VY_LEASE_EXPIRY_GRACE_S", "120"))
BORROW = int(os.environ.get("VY_POOL_BORROW", "0"))
BORROW_UNTIL = os.environ.get("VY_POOL_BORROW_UNTIL", "")
POLICY = os.environ.get("VY_POOL_POLICY", "")
POLICY_EVERY_S = 60
POLICY_TIMEOUT_S = 5
KILL_AFTER_S = 30
BRAIN = os.environ.get("VY_LEASE_BRAIN", "")
BRAIN_MISSES = 2
SERVER = os.environ.get("VY_DISPATCH_SERVER", "vy-nebius-1")
IMAGE = os.environ.get("VY_DISPATCH_IMAGE", "us-docker.pkg.dev/sky-dev-465/skypilotk8s/skypilot-gpu:202605150216")
KUBECTL = os.environ.get("VY_KUBECTL", "sudo -n k3s kubectl").split()
POOL_LABEL = "verity.dev/gpu-pool"
POOL_NAME = os.environ.get("VY_POOL_NAME", "node1")
FENCE_CMD = "n1-lease-fence"


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def epoch(s: str | None) -> float | None:
    try:
        return datetime.strptime(s or "", "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def gpu_count(field: str, v, cap: int) -> int:
    """A policy's `max` or `borrow`: a whole number of GPUs from 0 to `cap`, as an int or a string of digits; ValueError for
    anything else (-1, 1.5, Infinity, "8x")."""
    if isinstance(v, str) and v.isascii() and v.isdigit():
        v = int(v)
    if type(v) is not int or not 0 <= v <= cap:
        raise ValueError(f"{field} {v!r} isn't a whole number of GPUs from 0 to {cap}")
    return v


def communicate(p: subprocess.Popen, timeout: float) -> tuple[str, str]:
    return p.communicate(timeout=timeout)


def policy_output(cmd: str) -> str:
    """`cmd`'s stdout, run by a shell in a session of its own whose whole process group is killed after `POLICY_TIMEOUT_S`, so
    nothing it started (an ssh, a python3 under `cd ... &&`) outlives it; CalledProcessError on a nonzero exit."""
    with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                          start_new_session=True) as p:
        try:
            out, err = communicate(p, POLICY_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            raise
    if p.returncode:
        raise subprocess.CalledProcessError(p.returncode, cmd, out, err)
    return out


def kubectl(*args: str, stdin: str | None = None) -> str:
    return subprocess.run([*KUBECTL, *args], input=stdin, capture_output=True, text=True, check=True).stdout


def gpu_table() -> dict[str, int]:
    """UUID -> nvidia-smi index."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True, text=True,
                         check=True).stdout
    return {u.strip(): int(i) for i, u in (line.split(",") for line in out.splitlines() if line.strip())}


def owner_fields(line: str) -> dict[str, str]:
    head = line.split(" cmd=", 1)
    f = dict(kv.split("=", 1) for kv in head[0].split() if "=" in kv)
    if len(head) == 2:
        f["cmd"] = head[1]
    return f


def write(path: Path, text: str) -> None:
    """Atomically, mode 666: the pods' gpu-lease (uid 1000) and this controller (research) both rewrite these files."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}")
    tmp.write_text(text)
    os.chmod(tmp, 0o666)
    os.replace(tmp, path)


def lock_path(i: int) -> Path:
    p = DIR / f"{i}.lock"
    if not p.exists():
        fd = os.open(p, os.O_CREAT | os.O_WRONLY, 0o666)
        os.fchmod(fd, 0o666)
        os.close(fd)
    return p


def lock_holders() -> dict[int, list[int]]:
    """inode -> the pids /proc/locks names for its flocks: the process that took each lock, which for gpu-lease is a `flock`
    helper long gone (the lock lives on in the lease's descriptors), so a lease is attributed by its owner line's pid instead.
    The kernel hands /proc/locks out a page per read, so with locks coming and going one read can miss a lock held throughout
    (node 1: 182 lines, 3 pages): use it only where missing a lock costs a tick, and `locked` where it must not."""
    out: dict[int, list[int]] = {}
    for line in Path("/proc/locks").read_text().splitlines():
        f = line.split()
        if len(f) >= 6 and f[1] == "FLOCK":
            out.setdefault(int(f[5].split(":")[-1]), []).append(int(f[4]))
    return out


def locked(path: Path) -> bool:
    """Whether a process holds a flock on `path`: a non-blocking exclusive flock fails exactly then.  The file is opened without
    following a link or waiting on a FIFO (the lease dir is 0777); one this can't open counts as not locked."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return False
    except BlockingIOError:
        return True
    finally:
        os.close(fd)


def cmdline(pid: int) -> list[str]:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").split("\0")
    except OSError:
        return []


def pod_uid_of(pid: int) -> str | None:
    try:
        m = re.search(r"pod([0-9a-f_-]{36})", Path(f"/proc/{pid}/cgroup").read_text())
    except OSError:
        return None
    return m.group(1).replace("_", "-") if m else None


def fence(i: int, t: float) -> int | None:
    """Hold GPU i's lock from a detached process, its pid; None (holding nothing) when a lease has it."""
    lock = lock_path(i)
    p = subprocess.Popen(["flock", "-n", "-x", str(lock), "sh", "-c", "echo locked; exec sleep 2147483647", FENCE_CMD],
                         stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True)
    held = p.stdout.readline().strip() == b"locked"
    p.stdout.close()
    if not held:
        p.wait()
        return None
    write(DIR / f"{i}.owner", f"pid={p.pid} run=- who=kubernetes uuid=- since={iso(t)} until={iso(t + 86400)} "
                              f"cmd={FENCE_CMD}: not in node 1's lease pool (n1_lease.py)\n")
    return p.pid


def unfence(pids: list[int]) -> None:
    """Stop each fence and wait (up to 2 s) until its whole process group is gone, so its GPU's lock is free on return."""
    for pid in pids:
        try:
            os.killpg(pid, signal.SIGTERM)
        except OSError:
            pass
    for pid in pids:
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass
        for _ in range(200):
            try:
                os.killpg(pid, 0)
            except OSError:
                break
            time.sleep(0.01)


def holder_job(t: float) -> dict:
    name = f"gpu-pool-{int(t * 1000)}" if POOL_NAME == "node1" else f"gpu-pool-{POOL_NAME}-{int(t * 1000)}"
    labels = {POOL_LABEL: POOL_NAME}
    return {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"name": name, "labels": {**labels, "kueue.x-k8s.io/queue-name": LOCAL_QUEUE,
                                              "kueue.x-k8s.io/priority-class": PRIORITY}},
        "spec": {"suspend": True, "backoffLimit": 0, "ttlSecondsAfterFinished": 600, "template": {
            "metadata": {"labels": labels},
            "spec": {"restartPolicy": "Never", "nodeSelector": {"verity.dev/server": SERVER}, "runtimeClassName": "nvidia",
                     "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
                     "securityContext": {"runAsUser": 1000, "runAsGroup": 1000}, "terminationGracePeriodSeconds": 1,
                     "containers": [{"name": "hold", "image": IMAGE, "imagePullPolicy": "IfNotPresent",
                                     "command": ["sh", "-c", "nvidia-smi --query-gpu=uuid --format=csv,noheader; exec sleep 2147483647"],
                                     "resources": {"requests": {"nvidia.com/gpu": "1"}, "limits": {"nvidia.com/gpu": "1"}}}]}}},
    }


def _true(w: dict, kind: str) -> bool:
    return any(c.get("type") == kind and c.get("status") == "True" for c in (w.get("status") or {}).get("conditions") or [])


def unborrowed(obs: dict, holders: set[str]) -> int | None:
    """GPUs the pool queue can give holders without borrowing: its nominal quota less the GPUs its other admitted Workloads
    reserve. None when the ClusterQueue or the Workloads weren't read."""
    cq, workloads = obs.get("cq"), obs.get("workloads")
    if cq is None or workloads is None:
        return None
    nominal = sum(int(r.get("nominalQuota", 0)) for rg in cq["spec"].get("resourceGroups") or [] for f in rg.get("flavors") or []
                  for r in f.get("resources") or [] if r.get("name") == "nvidia.com/gpu")
    other = 0
    for w in workloads:
        if ((w.get("status") or {}).get("admission") or {}).get("clusterQueue") != QUEUE or _true(w, "Finished") \
                or not _true(w, "QuotaReserved"):
            continue
        if any(o.get("name") in holders for o in w["metadata"].get("ownerReferences") or []):
            continue
        for ps in w["spec"].get("podSets") or []:
            for c in ps["template"]["spec"].get("containers") or []:
                other += int(((c.get("resources") or {}).get("requests") or {}).get("nvidia.com/gpu", 0) or 0) * int(ps.get("count", 1))
    return max(0, nominal - other)


def waiters() -> list[dict]:
    """gpu-lease's live wait files (a waiter locks its file before naming it `wait.*`, and holds it while it waits), oldest first."""
    out = []
    for f in sorted(DIR.glob("wait.*")):
        try:
            if not locked(f):
                continue
            w = owner_fields(f.read_text().strip())
        except (OSError, ValueError):
            continue
        out.append({"file": f.name, **w, "n": int(w.get("n", 1))})
    return out


class Controller:
    def __init__(self) -> None:
        self.table: dict[str, int] = {}
        self.uuid_of_pod: dict[str, str] = {}
        self.free_since: dict[int, float] = {}
        self.outside: dict[int, int] = {}
        self.stopping: dict[int, float] = {}           # host pid -> when it was first signalled
        self.brain_misses = BRAIN_MISSES
        self.agent_fd: int | None = None
        self.events: list[dict] = []
        self.limits: dict = {"max": POOL_MAX, "borrow": BORROW, "borrow-until": BORROW_UNTIL}
        self.policy_at: float | None = None
        self.policy_error: str | None = None
        self.policy_last: dict | None = None           # the limits $VY_POOL_POLICY last answered

    def pool_limits(self, t: float, dry: bool = False) -> dict:
        """The pool's max, borrow and borrow-until: `$VY_POOL_POLICY`'s while it answers with whole numbers of the node's GPUs
        and a UTC borrow-until, else those it last answered, else the environment's. Nothing the command prints raises out of
        here."""
        if not POLICY or (self.policy_at is not None and t - self.policy_at < POLICY_EVERY_S):
            return self.limits
        self.policy_at = t
        env = {"max": POOL_MAX, "borrow": BORROW, "borrow-until": BORROW_UNTIL}
        got, err = dict(env), None
        try:
            doc = json.loads(policy_output(POLICY))
            if doc.get("schema") != "cluster/policy/v1":
                raise ValueError(f"schema {doc.get('schema')!r}, not cluster/policy/v1")
            seen: dict = {}
            for k in doc["keys"]:
                f = k["key"].rsplit(".", 1)[-1]
                if f in env and f in seen:
                    raise ValueError(f"two keys give {f}: name one pool")
                if f in env:
                    seen[f] = k["value"]
            for f, v in seen.items():
                if f == "borrow-until" and epoch(str(v)) is None:
                    raise ValueError(f"borrow-until {v!r} isn't %Y-%m-%dT%H:%M:%SZ")
                got[f] = str(v) if f == "borrow-until" else gpu_count(f, v, len(self.table))
        except Exception as e:  # noqa: BLE001  (a bad policy must never stop the loop)
            got, err = self.policy_last or env, f"{type(e).__name__}: {e}"[:200]
        else:
            self.policy_last = got
        source = "policy" if not err else "last-policy" if self.policy_last else "environment"
        if not dry:
            if got != self.limits or err != self.policy_error:
                self.log(t, "policy", limits=got, **({"error": err, "fallback": source} if err else {}))
            try:
                write(DIR / "n1-lease.limits", " ".join(f"{k}={v}" for k, v in got.items()) + f" from={source}\n")
            except OSError as e:
                self.log(t, "limits-write-failed", error=str(e)[:200])
        self.limits, self.policy_error = got, err
        return got

    def log(self, t: float, ev: str, **kw) -> None:
        rec = {"t": iso(t), "ev": ev, **kw}
        self.events.append(rec)
        with open(DIR / "n1-lease.log", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

    def observe(self) -> dict:
        jobs = json.loads(kubectl("get", "jobs", "-l", f"{POOL_LABEL}={POOL_NAME}", "-o", "json"))["items"]
        pods = json.loads(kubectl("get", "pods", "-o", "json"))["items"]
        try:
            cq = json.loads(kubectl("get", "clusterqueue", QUEUE, "-o", "json"))
            workloads = json.loads(kubectl("get", "workloads", "-A", "-o", "json"))["items"]
        except (subprocess.CalledProcessError, ValueError, KeyError):
            cq, workloads = None, None
        return {"jobs": jobs, "pods": pods, "cq": cq, "workloads": workloads}

    def stop(self, t: float, lease: dict, why: str) -> None:
        """End a lease: delete its pod, or signal a host process's scope (else its owner pid), SIGKILL after `KILL_AFTER_S`."""
        if lease.get("pod") or not (lease.get("pid") and lease.get("owner_alive")):
            self.log(t, why, lease=lease)
            if lease.get("pod"):
                kubectl("delete", "pod", lease["pod"], "--wait=false")
            return
        pid = lease["pid"]
        sig = signal.SIGKILL if t - self.stopping.setdefault(pid, t) >= KILL_AFTER_S else signal.SIGTERM
        self.log(t, why, lease=lease, signal=sig.name)
        if lease.get("scope") and subprocess.run(["systemctl", "--user", "kill", f"--signal={sig.name}", f"{lease['scope']}.scope"],
                                                 capture_output=True).returncode == 0:
            return
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError) as e:
            self.log(t, "signal-failed", pid=pid, error=str(e))

    def tick(self, t: float, obs: dict | None = None, dry: bool = False) -> dict:
        self.events = []
        if not self.table:
            self.table = gpu_table()
        lim = self.pool_limits(t, dry)
        obs = obs or self.observe()
        n = len(self.table)
        pods_by_uid = {p["metadata"].get("uid"): p for p in obs["pods"]}
        pods_by_job: dict[str, list[dict]] = {}
        for p in obs["pods"]:
            pods_by_job.setdefault((p["metadata"].get("labels") or {}).get("job-name", ""), []).append(p)
        pool: dict[int, str] = {}                      # index -> holder Job
        pending: list[str] = []                        # holders without a GPU in the pool yet
        unmapped = 0                                   # of those, running but with no UUID read yet
        for j in obs["jobs"]:
            name = j["metadata"]["name"]
            if (j.get("status") or {}).get("failed") or (j.get("status") or {}).get("succeeded"):
                continue
            running = [p for p in pods_by_job.get(name, []) if (p.get("status") or {}).get("phase") == "Running"]
            if running:
                pod = running[0]["metadata"]["name"]
                if pod not in self.uuid_of_pod:
                    try:
                        first = kubectl("logs", pod).strip().splitlines()
                        if first and first[0].startswith("GPU-"):
                            self.uuid_of_pod[pod] = first[0].strip()
                    except subprocess.CalledProcessError:
                        pass
                i = self.table.get(self.uuid_of_pod.get(pod, ""))
                if i is not None:
                    pool[i] = name
                    continue
                unmapped += 1
            pending.append(name)
        held = lock_holders()
        gpus = []
        for i in range(n):
            lock = lock_path(i)
            pids = held.get(lock.stat().st_ino, [])
            fences = [p for p in pids if FENCE_CMD in " ".join(cmdline(p)) and str(lock) in cmdline(p)]
            leased = len(pids) > len(fences)
            try:
                owner = (DIR / f"{i}.owner").read_text().strip()
            except OSError:
                owner = ""
            g = {"index": i, "uuid": next((u for u, k in self.table.items() if k == i), None), "holder": pool.get(i),
                 "fence_pids": fences}
            if leased:
                f = owner_fields(owner) if owner else {}
                pid = int(f["pid"]) if f.get("pid", "").isdigit() else None
                uid = pod_uid_of(pid) if pid else None
                pod = pods_by_uid.get(uid) if uid else None
                g["lease"] = {"who": f.get("who"), "run": f.get("run"), "since": f.get("since"), "until": f.get("until"),
                              "pid": pid, "scope": f.get("scope"), "pod": pod["metadata"]["name"] if pod else None,
                              "owner_alive": bool(pid) and Path(f"/proc/{pid}").exists(), "seen": iso(t)}
            gpus.append(g)
        want = waiters()
        try:
            blocked = (DIR / "blocked").read_text().strip() or "blocked"
        except OSError:
            blocked = None
        brain = self.ask_brain(t, gpus, want, pending) if BRAIN and not blocked else None
        for g in gpus:
            i, lease = g["index"], g.get("lease")
            if g["holder"] is None and lease:
                self.outside[i] = self.outside.get(i, 0) + 1
                if self.outside[i] >= 2 and not unmapped and not dry:
                    self.stop(t, {**lease, "gpu": i}, "lease-outside-pool")
                continue
            self.outside.pop(i, None)
            if lease and lease.get("until") and t > (epoch(lease["until"]) or t) + GRACE_S and not dry:
                self.stop(t, {**lease, "gpu": i}, "expired")
            if g["holder"] is None and not g["fence_pids"] and not lease and not dry:
                pid = fence(i, t)
                if pid:
                    g["fence_pids"] = [pid]
                    self.log(t, "fenced", gpu=i)
            if g["holder"] is not None and g["fence_pids"] and not dry and not blocked:
                unfence(g["fence_pids"])
                g["fence_pids"] = []
                self.log(t, "unfenced", gpu=i, holder=g["holder"])
            if g["holder"] is not None and not lease:
                self.free_since.setdefault(i, t)
            else:
                self.free_since.pop(i, None)
        if blocked and not dry:
            for g in gpus:
                i = g["index"]
                if g["holder"] is None or g.get("lease"):
                    continue
                if not g["fence_pids"] and (pid := fence(i, t)):
                    g["fence_pids"] = [pid]
                if g["fence_pids"]:
                    kubectl("delete", "job", g["holder"], "--wait=false", "--ignore-not-found")
                    self.log(t, "blocked-shrink", gpu=i, holder=g["holder"], why=blocked[:200])
                    g["holder"] = None
                    self.free_since.pop(i, None)
            for name in pending:
                kubectl("delete", "job", name, "--wait=false", "--ignore-not-found")
                self.log(t, "cancel", holder=name, why="blocked")
        free = sorted(i for i in self.free_since if i in pool)
        wanted = sum(w["n"] for w in want)
        target = brain.get("pool_target") if brain else None
        if target is None:
            target = len(pool) - len(free) + wanted
        self.stopping = {p: s for p, s in self.stopping.items() if Path(f"/proc/{p}").exists()}
        cap = unborrowed(obs, {j["metadata"]["name"] for j in obs["jobs"]})
        borrow = lim["borrow"] if want and all(w.get("preempt") == "1" for w in want) and \
            t < (epoch(lim["borrow-until"]) or float("inf")) else 0
        limit = min(lim["max"], cap + borrow) if cap is not None else len(pool) + len(pending)
        target = max(0, min(limit, int(target)))
        if blocked:
            target = len(pool) + len(pending)
        grow = target - len(pool) - len(pending)
        if grow > 0 and not dry:
            for k in range(grow):
                job = holder_job(t + k / 1000)
                kubectl("create", "-f", "-", stdin=json.dumps(job))
                self.log(t, "grow", holder=job["metadata"]["name"], waiting=wanted, pool=len(pool))
        elif grow < 0 and not dry:
            extra = -grow
            for name in pending[:extra]:
                kubectl("delete", "job", name, "--wait=false")
                self.log(t, "cancel", holder=name)
                extra -= 1
            for i in free:
                if extra <= 0:
                    break
                if t - self.free_since[i] < IDLE_S and brain is None:
                    continue
                pid = fence(i, t)
                if pid:
                    gpus[i]["fence_pids"], gpus[i]["holder"] = [pid], None
                    kubectl("delete", "job", pool[i], "--wait=false")
                    self.log(t, "shrink", gpu=i, holder=pool[i])
                    extra -= 1
        state = {"t": iso(t), "node": SERVER, "mode": "brain" if self.agent_fd is not None else "local", "pool": len(pool),
                 "pool_name": POOL_NAME, "blocked": blocked, "pending_holders": pending, "waiting": want, "gpus": gpus,
                 "events": self.events}
        if not dry:
            write(DIR / "pool.gpus", ",".join(str(i) for i in range(n)) + "\n")
            write(DIR / "n1-lease.json", json.dumps(state, indent=1) + "\n")
        return state

    def ask_brain(self, t: float, gpus: list[dict], want: list[dict], pending: list[str]) -> dict | None:
        req = {"t": iso(t), "node": SERVER, "gpus": gpus, "waiting": want, "pending_holders": pending,
               "pool_max": self.limits["max"]}
        try:
            out = subprocess.run(BRAIN, shell=True, input=json.dumps(req), capture_output=True, text=True, timeout=20,
                                 check=True).stdout
            ans = json.loads(out)
        except (subprocess.SubprocessError, ValueError, OSError) as e:
            self.brain_misses += 1
            if self.brain_misses >= BRAIN_MISSES and self.agent_fd is not None:
                os.close(self.agent_fd)
                self.agent_fd = None
                self.log(t, "brain-down", why=str(e)[:200])
            return None
        self.brain_misses = 0
        if self.agent_fd is None:
            import fcntl
            fd = os.open(DIR / "agent.lock", os.O_CREAT | os.O_RDWR, 0o666)
            fcntl.flock(fd, fcntl.LOCK_EX)
            self.agent_fd = fd
            self.log(t, "brain-up")
        live = {w.get("pid") for w in want}
        for pid, idx in (ans.get("grants") or {}).items():
            if str(pid) in live:
                write(DIR / f"grant.{pid}", ",".join(str(i) for i in idx) + "\n")
        return ans


#: what `setup` runs as root on one file of DIR: DIR is 0777, so any local user can plant a symlink or a hard link there, or swap a
#: file for one after it was listed; only the regular file it opens without following a link, with no other name, is made 0666
FIX_MODE = """import os, stat, sys
fd = os.open(sys.argv[1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
st = os.fstat(fd)
if stat.S_ISREG(st.st_mode) and st.st_nlink == 1:
    os.fchmod(fd, 0o666)
"""


def setup() -> None:
    if not DIR.is_dir() or (DIR.stat().st_mode & 0o7777) != 0o777:
        subprocess.run(["sudo", "-n", "install", "-d", "-m", "0777", str(DIR)], check=True)
        subprocess.run(["sudo", "-n", "chmod", "0777", str(DIR)], check=True)
    for f in DIR.iterdir():
        st = f.lstat()
        if stat.S_ISREG(st.st_mode) and (st.st_mode & 0o666) != 0o666:
            subprocess.run(["sudo", "-n", "python3", "-c", FIX_MODE, str(f)], check=False)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tick")
    t.add_argument("--dry-run", action="store_true")
    lp = sub.add_parser("loop")
    lp.add_argument("--every", type=float, default=5)
    sub.add_parser("status")
    ns = ap.parse_args(argv)
    if ns.cmd == "status":
        print((DIR / "n1-lease.json").read_text())
        return 0
    if not getattr(ns, "dry_run", False):
        setup()
    c = Controller()
    if ns.cmd == "tick":
        s = c.tick(time.time(), dry=ns.dry_run)
        print(json.dumps({k: s[k] for k in ("mode", "pool", "pending_holders")}))
        return 0
    while True:
        try:
            c.tick(time.time())
        except (subprocess.CalledProcessError, OSError, ValueError) as e:
            print(f"{iso(time.time())} tick failed: {e}", file=sys.stderr, flush=True)
        time.sleep(ns.every)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
