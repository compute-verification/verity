"""window.py stage|prove UNITS_FILE: prove_time.sh's stage and prove steps for many block types on a shared node, inside host
limits the job enforces itself. Run from the tree (research run --on NODE --cwd source) under `taskset -c LIST nice -n N`.

UNITS_FILE has one block type a line, in priority order, tab-separated: `UNIT@N[,N...]` (prove_time.sh's unit and its Ns,
stage.py's `full` and `full/D` included), then the estimates admission reads for each of its statements: peak RSS (GB), disk
(GB), CPU seconds to stage and device memory (GiB). `#` starts a comment.

stage  each UNIT@N its own `prove_time.sh STEPS=stage PT_DEVICE=cpu` into $FLOCK_WORK/data, started in the file's order while
       it admits: the running stagings' RSS (each the larger of its estimate and its peak) plus its estimate within
       WIN_MEM_GB x 0.9, FLOCK_WORK's size plus the running ones' disk estimates plus its own within WIN_DISK_GB x 0.95, and its
       estimated end (seconds x WIN_SLOW) before WIN_UNTIL. Once one waits for memory, only stagings of at most
       WIN_BACKFILL_GB start ahead of it. One that can never fit is cut, with its reason. Every 2 s: the stagings' RSS over
       WIN_MEM_GB x 0.95 kills the newest; FLOCK_WORK over WIN_DISK_GB x 0.97 kills the newest and starts nothing more;
       WIN_UNTIL kills every staging. A failed or killed statement's directory is removed, and a statement whose circuit
       equals an earlier N's has its circuit.txt hard-linked to that one. The build step (flock-circuit with its CUDA prover)
       runs first, beside the stagings. FLOCK_WORK stays for `prove`.
prove  the statements of UNITS_FILE staged under FLOCK_WORK, a unit's statements on one GPU, in WIN_GPUS groups balanced
       by estimated time, each group one `prove_time.sh STEPS=prove PT_DEVICE=gpu` on its own slice of the job's cores
       under `gpu-lease 1` (the tree's gpu_lease.sh where none is installed). Every 2 s: a group whose GPU holds more than
       WIN_DEV_GIB has its prover killed (every prover, once nvidia-smi goes unread for 30 s), and our processes' RSS over
       WIN_MEM_GB x 0.97 kills the largest. If the job stops, its groups are killed. Then FLOCK_WORK is removed, unless
       WIN_KEEP=1.

Limits (env): the job's affinity at most WIN_CPUS cores and its niceness at least WIN_NICE (refused otherwise), WIN_MEM_GB,
WIN_DISK_GB, WIN_UNTIL (UTC, e.g. 2026-11-10T01:30Z), FLOCK_WORK (under /workspace/tmp), WIN_DEV_GIB, WIN_GPUS; WIN_DEVICE=cpu
builds and proves on the CPU (a rehearsal off the GPU node). Records under
$RESEARCH_RUN_DIR/out: window.log, stage.tsv or prove.tsv, and each prove_time.sh run's own out/."""

from __future__ import annotations

import calendar
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path.cwd()
PT = REPO / "benchmarks/served_units/prove_time.sh"
OUT = Path(os.environ.get("RESEARCH_RUN_DIR", "/tmp/served-units-window")) / "out"
PAGE = os.sysconf("SC_PAGE_SIZE")
GB = 1e9


def env(k: str, default: str | None = None) -> str:
    v = os.environ.get(k, default)
    if v is None:
        raise SystemExit(f"window.py: set {k}")
    return v


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)
    with open(OUT / "window.log", "a") as f:
        f.write(line + "\n")


@dataclass
class Unit:
    unit: str
    ns: list[str]
    rss: float
    disk: float
    secs: float
    dev: float


def read_units(path: Path) -> list[Unit]:
    units = []
    for ln in path.read_text().splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        spec, rss, disk, secs, dev = ln.rstrip("\n").split("\t")
        u, _, ns = spec.rpartition("@")
        units.append(Unit(u, ns.split(","), float(rss), float(disk), float(secs), float(dev)))
    return units


def name(unit: str, n: str) -> str:
    """prove_time.sh's name(): the statement's directory name."""
    s = re.sub(r"-+", "-", re.sub(r"[^a-z0-9.\n-]", "-", unit.translate(str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                                                                                         "abcdefghijklmnopqrstuvwxyz"))))
    s = s[:-1] if s.endswith("-") else s
    if len(s) > 96:
        s = f"{s[:80]}-{hashlib.sha256(unit.encode()).hexdigest()[:12]}"
    return f"{s}-n{n.replace('/', '-')}"


def key() -> str:
    """prove_time.sh's KEY: the staging tree's digest that names each statement's directory."""
    return subprocess.run(["bash", "-c", "find benchmarks/served_units/stage.py verity/ml integrations/vllm/verity_vllm "
                           "tools/verity/circuit_check/src -name '*.py' -type f | LC_ALL=C sort | xargs sha256sum | sha256sum "
                           "| cut -c1-16"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def procs() -> dict[int, tuple[int, int]]:
    """Every process: pid -> (ppid, RSS bytes)."""
    t = {}
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                stat = Path(f"/proc/{p}/stat").read_text()
                ppid = int(stat[stat.rindex(")") + 2:].split()[1])
                t[int(p)] = (ppid, int(Path(f"/proc/{p}/statm").read_text().split()[1]) * PAGE)
            except (OSError, ValueError, IndexError):
                pass
    return t


def tree(root: int, t: dict[int, tuple[int, int]]) -> list[int]:
    kids: dict[int, list[int]] = {}
    for p, (pp, _r) in t.items():
        kids.setdefault(pp, []).append(p)
    out, todo = [], [root]
    while todo:
        p = todo.pop()
        if p in t:
            out.append(p)
        todo += kids.get(p, [])
    return out


def du(path: Path) -> int:
    seen, n = set(), 0
    for dirpath, _dirs, files in os.walk(path):
        for f in files:
            try:
                st = os.lstat(os.path.join(dirpath, f))
            except OSError:
                continue
            if (st.st_dev, st.st_ino) not in seen:
                seen.add((st.st_dev, st.st_ino))
                n += st.st_blocks * 512
    return n


def limits(mem_default: str) -> tuple[float, list[int]]:
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) > int(env("WIN_CPUS")):
        raise SystemExit(f"window.py: {len(cpus)} CPUs in the job's affinity, more than WIN_CPUS={env('WIN_CPUS')}: run under taskset -c")
    if os.nice(0) < int(env("WIN_NICE", "0")):
        raise SystemExit(f"window.py: niceness {os.nice(0)}, below WIN_NICE={env('WIN_NICE')}: run under nice -n {env('WIN_NICE')}")
    if not env("FLOCK_WORK").startswith("/workspace/tmp/"):
        raise SystemExit(f"window.py: FLOCK_WORK {env('FLOCK_WORK')} is not under /workspace/tmp/")
    return float(env("WIN_MEM_GB", mem_default)) * GB, cpus


def until() -> float:
    return calendar.timegm(time.strptime(env("WIN_UNTIL"), "%Y-%m-%dT%H:%MZ"))


@dataclass
class Job:
    label: str
    proc: subprocess.Popen
    started: float
    rss: float = 0.0
    disk: float = 0.0
    secs: float = 0.0
    data: Path | None = None
    unit: Unit | None = None
    n: str = ""
    peak: float = 0.0
    rss_now: float = 0.0
    killed: str = ""
    gpu: str = ""


def start(label: str, args: list[str], run_dir: Path, extra_env: dict[str, str], prefix: list[str] | None = None) -> subprocess.Popen:
    run_dir.mkdir(parents=True, exist_ok=True)
    e = {**os.environ, "RESEARCH_RUN_DIR": str(run_dir), **extra_env}
    logf = open(run_dir / "job.log", "w")
    log(f"start {label}")
    return subprocess.Popen([*(prefix or []), "bash", str(PT), *args], cwd=REPO, env=e, stdout=logf, stderr=subprocess.STDOUT,
                            start_new_session=True)


def kill(job: Job, why: str) -> None:
    job.killed = why
    try:
        os.killpg(job.proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    log(f"killed {job.label}: {why}")


def stage(units: list[Unit]) -> int:
    mem, cpus = limits("600")
    disk_cap, deadline = float(env("WIN_DISK_GB")) * GB, until()
    slow, backfill = float(env("WIN_SLOW", "1.3")), float(env("WIN_BACKFILL_GB", "8")) * GB
    fw = Path(env("FLOCK_WORK"))
    fw.mkdir(parents=True, exist_ok=True)
    k = key()
    log(f"stage: {len(units)} types, {len(cpus)} CPUs, niceness {os.nice(0)}, memory {mem / GB:.0f} GB, disk {disk_cap / GB:.0f} GB, "
        f"until {env('WIN_UNTIL')}, FLOCK_WORK {fw}, key {k}")
    one = {"FLOCK_WORK": str(fw), "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    prep = subprocess.run(["bash", str(PT), "STEPS=prep", "PT_DEVICE=cpu"], cwd=REPO, capture_output=True, text=True,
                          env={**os.environ, **one, "RESEARCH_RUN_DIR": str(OUT / "prep")})
    if not (fw / "py/bin/python").exists():
        log(f"prep failed, no venv: {prep.stdout[-2000:]}{prep.stderr[-2000:]}")
        return 1
    build = Job("build", start("build", ["STEPS=build", f"PT_DEVICE={env('WIN_DEVICE', 'gpu')}"], OUT / "build", {"FLOCK_WORK": str(fw)}), time.time(),
                rss=16 * GB, disk=20 * GB)
    pending = [(u, n) for u in units for n in u.ns]
    running: list[Job] = [build]
    rows, n_done, last_du, used, closed = [], 0, 0.0, 0, ""
    while pending or running:
        now = time.time()
        for j in [j for j in running if j.proc.poll() is not None]:
            running.remove(j)
            last_du = 0.0
            if not j.unit:
                ok = (fw / "flock-gpu-link/flock/target/release/flock-circuit").exists()
                log(f"build {'done' if ok else 'FAILED'} in {now - j.started:.0f} s (see {OUT}/build)")
                continue
            done = (j.data / ".done").exists() and not j.killed
            rec = json.loads((j.data / "stage.json").read_text()) if done else {}
            why = j.killed or ("" if done else next((ln for ln in (OUT / "units" / j.label / "job.log").read_text().splitlines()
                                                    if "FAILED" in ln), "failed"))
            if not done:
                shutil.rmtree(j.data, ignore_errors=True)
            else:
                n_done += 1
                for other in j.unit.ns:
                    od = fw / "data" / f"{name(j.unit.unit, other)}-{k}"
                    if other != j.n and (od / ".done").exists() and \
                            json.loads((od / "stage.json").read_text())["circuit_sha512"] == rec["circuit_sha512"]:
                        tmp = j.data / "circuit.txt.link"
                        os.link(od / "circuit.txt", tmp)
                        os.replace(tmp, j.data / "circuit.txt")
                        log(f"{j.label}'s circuit.txt is {od.name}'s, hard-linked")
                        break
            rows.append([j.unit.unit, j.n, str(rec.get("n", "")), j.data.name, "staged" if done else f"failed: {why}",
                         str(rec.get("ands", "")), str(rec.get("k_log", "")), str(rec.get("unit_nnz", "")),
                         str(rec.get("circuit_bytes", "")), f"{now - j.started:.0f}", f"{j.peak / GB:.1f}"])
            log(f"{'staged' if done else 'FAILED'} {j.label} {j.unit.unit}@{j.n} in {now - j.started:.0f} s, peak "
                f"{j.peak / GB:.1f} GB{'' if done else ': ' + why}")
        t = procs()
        for j in running:
            j.rss_now = sum(t[p][1] for p in tree(j.proc.pid, t))
            j.peak = max(j.peak, j.rss_now)
        if now - last_du > 10:
            used, last_du = du(fw), now
        stagings = [j for j in running if j.unit and not j.killed]
        if sum(j.rss_now for j in running) > 0.95 * mem and stagings:
            kill(max(stagings, key=lambda j: j.started), f"host memory {sum(j.rss_now for j in running) / GB:.0f} GB over "
                 f"{0.95 * mem / GB:.0f}")
        elif used > 0.97 * disk_cap and stagings:
            kill(max(stagings, key=lambda j: j.started), f"FLOCK_WORK {used / GB:.0f} GB over {0.97 * disk_cap / GB:.0f}")
            closed = closed or "disk"
        if now > deadline:
            for j in stagings:
                kill(j, f"WIN_UNTIL {env('WIN_UNTIL')}")
            closed = closed or "time"
        blocked_mem = False
        for u, n in list(pending):
            reason = closed
            if not reason:
                mem_in_use = sum(max(j.rss, j.peak) for j in running)
                if now + u.secs * slow > deadline:
                    reason = "time"
                elif used + sum(j.disk for j in running if j.unit) + u.disk * GB > 0.95 * disk_cap:
                    reason = "disk" if not any(j.unit for j in running) else "wait"
                elif mem_in_use + u.rss * GB > 0.9 * mem or (blocked_mem and u.rss * GB > backfill):
                    reason = "memory" if not running else "wait"
                    blocked_mem = True
                elif len(running) >= len(cpus) - 2:
                    reason = "wait"
            if reason == "wait":
                continue
            pending.remove((u, n))
            if reason:
                rows.append([u.unit, n, "", name(u.unit, n), f"cut: {reason}", "", "", "", "", "", ""])
                log(f"cut {u.unit}@{n}: {reason}")
                continue
            label = f"{len(rows) + len(running):03d}"
            d = fw / "data" / f"{name(u.unit, n)}-{k}"
            running.append(Job(label, start(f"{label} {u.unit}@{n}", ["STEPS=stage", "PT_DEVICE=cpu", f"PT_UNITS={u.unit}@{n}",
                                                                         "PT_JOBS=1"], OUT / "units" / label, one),
                               now, rss=u.rss * GB, disk=u.disk * GB, secs=u.secs, data=d, unit=u, n=n))
        time.sleep(2)
    (OUT / "stage.tsv").write_text("unit\tn_spec\tn\tdir\tstatus\tands\tk_log\tunit_nnz\tcircuit_bytes\tseconds\tpeak_gb\n" +
                                   "".join("\t".join(r) + "\n" for r in rows))
    used = du(fw)
    built = (fw / "flock-gpu-link/flock/target/release/flock-circuit").exists()
    log(f"stage done: {n_done} statements staged, {sum(r[4].startswith('cut') for r in rows)} cut, "
        f"{sum(r[4].startswith('failed') for r in rows)} failed; FLOCK_WORK {used / GB:.1f} GB; prover {'built' if built else 'NOT built'}")
    return 0 if built else 1


def prove(units: list[Unit]) -> int:
    mem, cpus = limits("268")
    fw, k = Path(env("FLOCK_WORK")), key()
    ngpu, dev_mib = int(env("WIN_GPUS", "8")), float(env("WIN_DEV_GIB", "45")) * 1024
    on_gpu = env("WIN_DEVICE", "gpu") == "gpu"
    if on_gpu and not shutil.which("nvidia-smi"):
        raise SystemExit("window.py: no nvidia-smi on PATH, so no device-memory limit: prove on a GPU node, or WIN_DEVICE=cpu")
    binary = fw / "flock-gpu-link/flock/target/release/flock-circuit"
    if not binary.exists():
        log(f"no prover at {binary}: building it inside the window")
        subprocess.run(["bash", str(PT), "STEPS=build", f"PT_DEVICE={env('WIN_DEVICE', 'gpu')}"], cwd=REPO,
                       env={**os.environ, "FLOCK_WORK": str(fw), "RESEARCH_RUN_DIR": str(OUT / "build")}, check=True)
    staged = []
    for u in units:
        ns = [n for n in u.ns if (fw / "data" / f"{name(u.unit, n)}-{k}" / ".done").exists()]
        if ns:
            recs = [json.loads((fw / "data" / f"{name(u.unit, n)}-{k}" / "stage.json").read_text()) for n in ns]
            est = sum(10 + 3 * r["circuit_bytes"] / GB + (int(env("WARM", "1")) + int(env("RUNS", "3"))) *
                      (4 + 2 ** max(r["k_log"] - 23, 0)) for r in recs)
            staged.append((u, ns, est))
        else:
            log(f"not staged, skipped: {u.unit}")
    groups: list[list] = [[] for _ in range(ngpu)]
    for item in sorted(staged, key=lambda x: -x[2]):
        min(groups, key=lambda g: sum(x[2] for x in g)).append(item)
    scratch = Path(tempfile.mkdtemp(prefix="window-bin-"))
    path = os.environ["PATH"]
    if not shutil.which("gpu-lease"):
        (scratch / "gpu-lease").symlink_to(REPO / "tools/research/src/research/pods/sh/gpu_lease.sh")
        path = f"{scratch}:{path}"
    per = len(cpus) // ngpu
    jobs = []
    log(f"prove: {len(staged)} types in {ngpu} groups, {per} CPUs each, host memory {mem / GB:.0f} GB, device {dev_mib / 1024:.0f} GiB a GPU")
    for g, items in enumerate(groups):
        if not items:
            continue
        units_arg = " ".join(f"{u.unit}@{','.join(ns)}" for u, ns, _e in items)
        lease = int(env("LEASE_MIN", "30"))
        cores = ",".join(map(str, cpus[g * per:(g + 1) * per]))
        log(f"group {g}: {len(items)} types, estimated {sum(e for *_x, e in items) / 60:.1f} min, lease {lease} min, cores {cores}")
        p = start(f"gpu group {g}", ["STEPS=prove", f"PT_DEVICE={env('WIN_DEVICE', 'gpu')}", f"PT_UNITS={units_arg}", f"LEASE_MIN={lease}",
                                     f"VCPUS={env('VCPUS', '2')}", f"WARM={env('WARM', '1')}", f"RUNS={env('RUNS', '3')}"],
                  OUT / f"gpu{g}", {"FLOCK_WORK": str(fw), "PATH": path, "GPU_LEASE_WHO": env("GPU_LEASE_WHO", "servedcard")},
                  prefix=["taskset", "-c", cores])
        jobs.append(Job(f"gpu{g}", p, time.time()))
    try:
        watch(jobs, mem, dev_mib, on_gpu)
    finally:
        t = procs()
        for j in jobs:
            for p in tree(j.proc.pid, t) if j.proc.poll() is None else ():
                try:
                    os.kill(p, signal.SIGKILL)
                except OSError:
                    pass
    rows = []
    for j in jobs:
        s = OUT / j.label / "out" / "summary.json"
        log(f"{j.label} exit {j.proc.returncode}")
        if s.exists():
            for st, rec in json.loads(s.read_text())["statements"].items():
                rows.append([j.label, st, str(rec.get("n")), str(rec.get("k_log")), str(rec.get("prove_total_s", "")),
                             json.dumps({x: rec[x] for x in rec if x not in ("n", "k_log", "prove_total_s")})])
    (OUT / "prove.tsv").write_text("group\tstatement\tn\tk_log\tprove_total_s\trecord\n" + "".join("\t".join(r) + "\n" for r in rows))
    shutil.rmtree(scratch, ignore_errors=True)
    if env("WIN_KEEP", "0") != "1":
        freed = du(fw)
        shutil.rmtree(fw, ignore_errors=True)
        log(f"removed FLOCK_WORK {fw} ({freed / GB:.1f} GB)")
    log(f"prove done: {len(rows)} statements with a summary")
    return 0


def watch(jobs: list[Job], mem: float, dev_mib: float, on_gpu: bool) -> None:
    """Until every group exits: kill a group's prover past dev_mib on its GPU, and the largest of our processes past
    0.97 x mem. Device memory unread for 30 s in a row kills every prover, since the limit can't be held."""
    blind = 0
    while any(j.proc.poll() is None for j in jobs):
        t = procs()
        ours = {p for j in jobs for p in tree(j.proc.pid, t)}
        used: dict[str, str] = {}
        if on_gpu:
            try:
                q = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=10)
                used = dict(ln.replace(" ", "").split(",") for ln in q.stdout.splitlines() if "," in ln)
            except (subprocess.TimeoutExpired, OSError):
                pass
            blind = 0 if used else blind + 1
            if blind == 15:
                log("device memory unread for 30 s: killing every prover")
                dev_mib = -1.0
        for j in jobs:
            for p in tree(j.proc.pid, t) if not j.gpu else ():
                try:
                    m = re.search(rb"(?:^|\0)GPU_LEASE_UUID=([^\0]+)", Path(f"/proc/{p}/environ").read_bytes())
                except OSError:
                    m = None
                if m:
                    j.gpu = m[1].decode().split(",")[0]
                    log(f"{j.label} holds {j.gpu}")
                    break
            mib = float(used.get(j.gpu, 0)) if j.gpu else 0
            if mib > dev_mib:
                for p in tree(j.proc.pid, t):
                    try:
                        if b"\0prove\0" in Path(f"/proc/{p}/cmdline").read_bytes():
                            os.kill(p, signal.SIGKILL)
                            log(f"killed {j.label}'s prover pid {p}: {mib:.0f} MiB on {j.gpu}, over {dev_mib:.0f}")
                    except OSError:
                        pass
        total = sum(t[p][1] for p in ours)
        if total > 0.97 * mem:
            big = max((p for p in ours if t[p][1] > GB), key=lambda p: t[p][1], default=None)
            if big:
                os.kill(big, signal.SIGKILL)
                log(f"killed pid {big} ({t[big][1] / GB:.0f} GB): our RSS {total / GB:.0f} GB over {0.97 * mem / GB:.0f}")
        time.sleep(2)


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in ("stage", "prove"):
        raise SystemExit("window.py: usage: window.py stage|prove UNITS_FILE")
    OUT.mkdir(parents=True, exist_ok=True)
    units = read_units(Path(sys.argv[2]))
    return stage(units) if sys.argv[1] == "stage" else prove(units)


if __name__ == "__main__":
    sys.exit(main())
