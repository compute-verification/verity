"""commit-pack's pilot: up to PACK_SLOTS Commits at once on this pod's one GPU, as clients of an MPS daemon of the pod's own.

    python3 commit_pack.py      (the commit-pack template's run; stdlib only)

The dispatcher (dispatch.py, PACK_COMMITS=1) spools each eligible Commit, config-run's gpu task rendered for its item, as
`$PACK_SPOOL/queue/<name>.json` ({"key", "item", "idx", "tries", "env", "script"}). This pod claims one by renaming it into
`claimed/<pod>/` (a rename has one winner), runs its script as a subprocess with the MPS client environment, and writes
`results/<name>.json` when it ends; the dispatcher reads that back into the item's chain. It takes the next entry as each one
finishes, and stops claiming at PACK_CLAIM_UNTIL_S, in the quiet hour (12:30-13:30Z), or after an MPS fault; it exits once
nothing runs and the queue has been empty for PACK_IDLE_S, or at PACK_MAX_WALL_S, when whatever still runs is stopped.
Storage (infra and circuits, 3:25-3:36 PM PDT Sep 30): it claims nothing while /workspace is PACK_DISK_PAUSE (80%) full or more,
and no Commit of batch 8 or more while the sealed replay bundles waiting on the node pass PACK_BUNDLE_CAP_GB (150).

**MPS.** `nvidia-cuda-mps-control -d` with its pipe and log directories under the pod's scratch ($TMPDIR/commit-pack-<pod>, removed at
exit). Each client gets CUDA_MPS_PINNED_DEVICE_MEM_LIMIT (0=28G) and CUDA_MPS_ACTIVE_THREAD_PERCENTAGE (100), and the Commit
GPU_UTIL (vLLM's gpu_memory_utilization, 0.28) and REPLAY_DEFERRED=1.

**Faults.** A fatal GPU fault in one MPS client can take down every client on the GPU, so a fault is pod-wide: the control daemon or
its server dying, an error in the MPS logs, or a client ending with a CUDA fault in its log. Every Commit still running is then
stopped and reported `fault: true`, and the pod claims nothing more; the dispatcher requeues each once, unpacked (as it does any
packed Commit that fails).

**Record.** `stats/<pod>.jsonl` every PACK_STATS_S: running Commits, the GPU's utilization and memory used (nvidia-smi).
`logs/<pod>/<name>.log` is each Commit's wrapper output; the Commit's own `research run` writes its Attempt.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

SPOOL = Path(os.environ.get("PACK_SPOOL", "/workspace/jobs/dispatch/pack"))
POD = os.environ.get("HOSTNAME") or "pod"
SLOTS = int(os.environ.get("PACK_SLOTS", "3"))
MAX_WALL_S = int(os.environ.get("PACK_MAX_WALL_S", "7200"))
CLAIM_UNTIL_S = int(os.environ.get("PACK_CLAIM_UNTIL_S", str(MAX_WALL_S - 1800)))
IDLE_S = int(os.environ.get("PACK_IDLE_S", "180"))
POLL_S = float(os.environ.get("PACK_POLL_S", "5"))
STATS_S = float(os.environ.get("PACK_STATS_S", "15"))
STOP_GRACE_S = int(os.environ.get("PACK_STOP_GRACE_S", "60"))
CLIENT_ENV = {
    "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT": os.environ.get("PACK_MEM_LIMIT", "0=28G"),
    "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": os.environ.get("PACK_THREAD_PERCENTAGE", "100"),
    "GPU_UTIL": os.environ.get("PACK_GPU_UTIL", "0.28"),
    "REPLAY_DEFERRED": "1",
}
DISK = os.environ.get("PACK_DISK_PATH", "/workspace")
DISK_PAUSE = float(os.environ.get("PACK_DISK_PAUSE", "0.80"))
BUNDLE_GLOBS = os.environ.get("PACK_BUNDLE_GLOBS", "/workspace/jobs/cov/*/*/commit/replay_bundle_p*,"
                              "/workspace/cp/sweep/*/commit/replay_bundle_p*,"
                              "/workspace/jobs/dispatch/pack/golden/*/commit/replay_bundle_p*").split(",")
BUNDLE_CAP = float(os.environ.get("PACK_BUNDLE_CAP_GB", "150")) * 1e9
BUNDLE_BATCH = 8
FAULT_EXIT = 87
QUIET = ((12, 30), (13, 30))
CLIENT_FAULT = re.compile(r"illegal memory access|unspecified launch failure|CUDA_ERROR_MPS|cudaErrorMps|MPS server|"
                          r"CUDA-capable device\(s\) is/are busy|uncorrectable ECC|GPU is lost|Xid", re.IGNORECASE)
MPS_LOG_FAULT = re.compile(r"exited with status [1-9-]|fault|\bERROR\b|failed to", re.IGNORECASE)


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def quiet(t: float) -> bool:
    g = time.gmtime(t)
    return QUIET[0] <= (g.tm_hour, g.tm_min) < QUIET[1]


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True))
    tmp.rename(path)


def batch(entry: dict) -> int:
    parts = str((entry.get("env") or {}).get("ROW", "")).split("__")
    return next((int(p[1:]) for p in parts if re.fullmatch(r"b\d+", p)), 0)


def bundle_bytes(globs: list[str]) -> int:
    """Bytes of the sealed replay bundles waiting on the node (a Commit's replay deletes its own)."""
    total = 0
    for g in globs:
        for d in Path("/").glob(g.lstrip("/")):
            for root, _, files in os.walk(d):
                for name in files:
                    try:
                        total += os.lstat(os.path.join(root, name)).st_size
                    except OSError:
                        continue
    return total


def _accepts(f: Path, accept) -> bool:
    try:
        return bool(accept(json.loads(f.read_text())))
    except (OSError, ValueError):
        return False


def claim(spool: Path, pod: str, accept=lambda entry: True) -> tuple[str, dict] | None:
    """The oldest queued entry that `accept`s, renamed into claimed/<pod>/, or None when there is none."""
    mine = spool / "claimed" / pod
    mine.mkdir(parents=True, exist_ok=True)
    entries = []
    for f in (spool / "queue").glob("*.json"):
        try:
            entries.append((f.stat().st_mtime, f))
        except OSError:
            continue
    for _, f in sorted(entries):
        if not _accepts(f, accept):
            continue
        dst = mine / f.name
        try:
            f.rename(dst)
        except OSError:
            continue
        return f.stem, json.loads(dst.read_text())
    return None


@dataclass
class Client:
    name: str
    entry: dict
    proc: subprocess.Popen
    log: Path
    slot: int
    started: float = field(default_factory=time.time)


def tail(path: Path, n: int = 256 * 1024) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def client_fault(c: Client) -> str | None:
    env = c.entry.get("env") or {}
    texts = [tail(c.log)]
    if env.get("SWEEP_DIR") and env.get("ROW"):
        texts.append(tail(Path(env["SWEEP_DIR"]) / env["ROW"] / "commit.log"))
    for t in texts:
        m = CLIENT_FAULT.search(t)
        if m:
            return f"client fault: {m.group(0)!r}"
    return None


class Mps:
    def __init__(self, scratch: Path):
        self.pipe, self.logs = scratch / "mps-pipe", scratch / "mps-log"
        self.env = {"CUDA_MPS_PIPE_DIRECTORY": str(self.pipe), "CUDA_MPS_LOG_DIRECTORY": str(self.logs)}
        self.offsets: dict[Path, int] = {}

    def start(self) -> None:
        for d in (self.pipe, self.logs):
            d.mkdir(parents=True, exist_ok=True)
        subprocess.run(["nvidia-cuda-mps-control", "-d"], env={**os.environ, **self.env}, check=True)
        for _ in range(50):
            if self.alive():
                return
            time.sleep(0.2)
        raise RuntimeError("nvidia-cuda-mps-control -d started no daemon")

    def alive(self) -> bool:
        """The control daemon of this pod's pipe directory is running (a /proc scan: the image need not have pgrep)."""
        for p in Path("/proc").glob("[0-9]*/cmdline"):
            try:
                argv = p.read_bytes().split(b"\0")
            except OSError:
                continue
            if any(a.endswith(b"nvidia-cuda-mps-control") for a in argv[:2]):
                return (self.pipe / "control").exists()
        return False

    def log_fault(self) -> str | None:
        """The first fault line written to the MPS logs since the last call."""
        for p in sorted(self.logs.glob("*.log")):
            try:
                with open(p, "rb") as f:
                    f.seek(self.offsets.get(p, 0))
                    new = f.read().decode("utf-8", "replace")
                    self.offsets[p] = f.tell()
            except OSError:
                continue
            for line in new.splitlines():
                if MPS_LOG_FAULT.search(line):
                    return f"MPS {p.name}: {line.strip()[:300]!r}"
        return None

    def stop(self) -> None:
        subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", text=True, env={**os.environ, **self.env},
                       capture_output=True, timeout=30, check=False)


def gpu_sample() -> dict:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30, check=False)
        util, mem = (float(x) for x in r.stdout.strip().splitlines()[0].split(","))
        return {"gpu_util": util, "mem_used_mib": mem}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {"gpu_util": None, "mem_used_mib": None}


def stop_client(c: Client, grace_s: int) -> None:
    try:
        os.killpg(c.proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        c.proc.wait(grace_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(c.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        c.proc.wait()


def _terminate(signum, frame):
    raise SystemExit(128 + signum)


def main() -> int:
    signal.signal(signal.SIGTERM, _terminate)   # the pilot is the container's PID 1: without a handler it ignores SIGTERM
    os.umask(0)   # the dispatcher (another uid) moves and deletes what the pilot writes in the spool; the Commits keep 022
    t0 = time.time()
    scratch = Path(os.environ.get("TMPDIR") or "/tmp") / f"commit-pack-{POD}"
    logs = SPOOL / "logs" / POD
    logs.mkdir(parents=True, exist_ok=True)
    stats = SPOOL / "stats" / f"{POD}.jsonl"
    stats.parent.mkdir(parents=True, exist_ok=True)
    mps = Mps(scratch)
    running: dict[str, Client] = {}
    finished: list[str] = []
    fault: str | None = None
    idle_since, last_stats = time.time(), 0.0
    gates_at, disk_ok, big_ok = 0.0, True, True
    print(f"{now_utc()} commit-pack {POD}: slots {SLOTS}, max wall {MAX_WALL_S}s, client env {CLIENT_ENV}", flush=True)

    def report(c: Client, rc: int | None, fault_: bool, reason: str) -> None:
        write_json(SPOOL / "results" / f"{c.name}.json",
                   {"name": c.name, "key": c.entry["key"], "idx": c.entry["idx"], "tries": c.entry["tries"], "entry": c.entry,
                    "rc": rc, "fault": fault_, "reason": reason, "pod": POD, "slot": c.slot, "log": str(c.log),
                    "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(c.started)), "ended": now_utc(),
                    "wall_s": round(time.time() - c.started, 1)})
        (SPOOL / "claimed" / POD / f"{c.name}.json").unlink(missing_ok=True)
        print(f"{now_utc()} {c.name} ({c.entry['key']}) rc={rc} fault={fault_} {reason}", flush=True)

    try:
        mps.start()
    except (OSError, subprocess.SubprocessError, RuntimeError) as e:
        print(f"{now_utc()} MPS did not start: {e}; claiming nothing", flush=True)
        shutil.rmtree(scratch, ignore_errors=True)
        return FAULT_EXIT
    try:
        while True:
            for name, c in list(running.items()):
                rc = c.proc.poll()
                if rc is None:
                    continue
                del running[name]
                why = client_fault(c) if rc else None
                if why:
                    fault = fault or f"{name}: {why}"
                    report(c, rc, True, why)
                else:
                    report(c, rc, False, "exit")
                    finished.append(name)
            if not fault:
                if not mps.alive():
                    fault = "the MPS control daemon is gone"
                else:
                    fault = mps.log_fault()
            if fault:
                print(f"{now_utc()} MPS fault: {fault}; stopping {len(running)} Commit(s)", flush=True)
                for c in running.values():
                    stop_client(c, STOP_GRACE_S)
                    report(c, None, True, f"stopped: pod fault ({fault})")
                running.clear()
                break
            age = time.time() - t0
            if age >= MAX_WALL_S:
                for c in running.values():
                    stop_client(c, STOP_GRACE_S)
                    report(c, None, True, f"stopped: max wall {MAX_WALL_S}s")
                running.clear()
                break
            if len(running) < SLOTS and time.time() - gates_at >= 60:
                gates_at = time.time()
                u = shutil.disk_usage(DISK)
                disk_ok = u.used / u.total < DISK_PAUSE
                big_ok = bundle_bytes(BUNDLE_GLOBS) <= BUNDLE_CAP
                if not disk_ok or not big_ok:
                    print(f"{now_utc()} holding claims: disk {u.used / u.total:.1%} (pause at {DISK_PAUSE:.0%}), "
                          f"batch >= {BUNDLE_BATCH} {'allowed' if big_ok else 'refused: bundles over the cap'}", flush=True)
            accept = (lambda e: True) if big_ok else (lambda e: batch(e) < BUNDLE_BATCH)
            while (len(running) < SLOTS and age < CLAIM_UNTIL_S and disk_ok and not quiet(time.time())
                   and (got := claim(SPOOL, POD, accept))):
                name, entry = got
                slot = min(set(range(SLOTS)) - {c.slot for c in running.values()})
                log = logs / f"{name}.log"
                env = {**os.environ, **{k: str(v) for k, v in (entry.get("env") or {}).items()}, **mps.env, **CLIENT_ENV,
                       "OV_PACKED": POD}
                with open(log, "ab") as out:
                    proc = subprocess.Popen(["bash", "-c", entry["script"]], env=env, stdout=out, stderr=subprocess.STDOUT,
                                            start_new_session=True, umask=0o022)
                running[name] = Client(name, entry, proc, log, slot)
                print(f"{now_utc()} started {name} ({entry['key']}) in slot {slot}, pid {proc.pid}", flush=True)
            if running or (disk_ok and any(_accepts(f, accept) for f in (SPOOL / "queue").glob("*.json"))):
                idle_since = time.time()
            elif time.time() - idle_since >= IDLE_S:
                break
            if time.time() - last_stats >= STATS_S:
                last_stats = time.time()
                with open(stats, "a") as f:
                    f.write(json.dumps({"t": now_utc(), "running": sorted(running), **gpu_sample()}) + "\n")
            time.sleep(POLL_S)
    finally:
        for c in running.values():
            stop_client(c, STOP_GRACE_S)
            report(c, None, True, "stopped: the pilot is exiting")
        try:
            mps.stop()
        except (OSError, subprocess.SubprocessError):
            pass
        for p in sorted(mps.logs.glob("*.log")):
            print(f"--- {p.name}\n{tail(p, 16 * 1024)}", flush=True)
        shutil.rmtree(scratch, ignore_errors=True)
    print(f"{now_utc()} commit-pack {POD}: done, {len(finished)} Commit(s) ended normally, fault={fault}", flush=True)
    return FAULT_EXIT if fault else 0


if __name__ == "__main__":
    sys.exit(main())
