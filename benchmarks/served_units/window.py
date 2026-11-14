"""window.py stage|prove UNITS_FILE: prove_time.sh's stage and prove steps for many block types on a shared node, inside host
limits the job enforces itself. Run from the tree (research run --on NODE --cwd source) under `taskset -c LIST nice -n N`.

UNITS_FILE has one block type a line, in priority order, tab-separated: `UNIT@N[,N...]` (prove_time.sh's unit and its Ns,
stage.py's `full` and `full/D` included), then the estimates admission reads for each of its statements: peak RSS (GB), disk
(GB), CPU seconds to stage and device memory (GiB). `#` starts a comment.

stage  each UNIT@N its own `prove_time.sh STEPS=stage PT_DEVICE=cpu` into $FLOCK_WORK/data, started in the file's order while
       it admits: the running stagings' RSS (each the larger of its estimate and its peak) plus its estimate within
       WIN_MEM_GB x 0.9, FLOCK_WORK's size plus the running ones' disk estimates plus its own within WIN_DISK_GB x 0.95 and
       leaving WIN_FREE_GB (default 50) free on its filesystem, and its estimated end (seconds x WIN_SLOW) before WIN_UNTIL.
       Once one waits for memory, only stagings of at most WIN_BACKFILL_GB start ahead of it. One that can never fit is cut,
       with its reason. Every 2 s: the stagings' RSS over WIN_MEM_GB x 0.95 kills the newest; FLOCK_WORK over WIN_DISK_GB x
       0.97, or under WIN_FREE_GB free, kills the newest and starts nothing more; WIN_UNTIL kills every staging. A failed
       or killed statement's directory is removed, and a statement whose circuit equals an earlier N's has its circuit.txt
       hard-linked to that one. The build step (flock-circuit with its CUDA prover) runs first, beside the stagings, unless
       FLOCK_WORK already holds a built prover. FLOCK_WORK stays for `prove`. Where /dev/shm is mounted noexec, a FLOCK_WORK
       there holds `py` and `flock-gpu-link` as symlinks to a built FLOCK_WORK on disk, and only the statements in RAM.
prove  the statements of UNITS_FILE staged under FLOCK_WORK, and under each WIN_ALSO directory (colon-separated
       FLOCK_WORKs, each statement linked file by file into FLOCK_WORK/data so its sessions are written there), a unit's
       statements on one GPU, in WIN_GPUS groups balanced by estimated time, each group one `prove_time.sh STEPS=prove
       PT_DEVICE=gpu` (WARM=1 RUNS=1 unless set) on its own slice of the job's cores under `gpu-lease 1` (the tree's
       gpu_lease.sh where none is installed). A statement's estimate is its seconds in WIN_SECONDS (an earlier prove's
       out/seconds.tsv); one it doesn't hold finished takes WIN_UNMEASURED_S (default 900) or its unfinished seconds there,
       whichever is more, and its unit starts first in its group. Without WIN_SECONDS the estimate is a model of the
       circuit's size and k_log. As each statement's session set ends, its KEEP files (its public files and every session's
       records; WIN_KEEP_ALSO adds colon-separated globs under its directory) are copied to out/kept/NAME/, with its
       prover's exit status (out/kept/NAME/exit): a statement finished only on 0. Every 2 s: a group whose GPU holds
       more than WIN_DEV_GIB has its prover killed (every prover, once nvidia-smi goes unread for 30 s), and our
       processes' RSS, plus a FLOCK_WORK in RAM with its sessions, over WIN_MEM_GB x 0.97 kills the largest process. Then, or as soon as the job stops (SIGTERM included), its groups are
       killed and FLOCK_WORK is removed (its links, not what they point to), unless WIN_KEEP=1. A stopped `stage` kills its
       stagings and keeps FLOCK_WORK.

Limits (env, each declared in settings.py with its default): the job's affinity at most WIN_CPUS cores and its niceness at least WIN_NICE (refused otherwise), WIN_MEM_GB,
WIN_DISK_GB, WIN_UNTIL (UTC, e.g. 2026-11-10T01:30Z), FLOCK_WORK (under /workspace/tmp, or under /dev/shm, where what it
holds counts as host memory beside the processes' RSS), WIN_MEM_PEER (stage: another job's pid, whose process tree's RSS
counts against WIN_MEM_GB too, so two stagings share one memory grant), WIN_DEV_GIB, WIN_GPUS; WIN_DEVICE=cpu builds and proves on the CPU
(a rehearsal off the GPU node). Once a statement waits for disk, later ones of more than WIN_BACKFILL_DISK_GB (default 0)
wait behind it. Records under
$RESEARCH_RUN_DIR/out: window.log, stage.tsv or prove.tsv (with prove's seconds.tsv and kept/), and each prove_time.sh run's
own out/."""

from __future__ import annotations

import atexit
import calendar
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


def _settings():
    spec = importlib.util.spec_from_file_location("served_units_settings", Path(__file__).with_name("settings.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


env = _settings().get
REPO = Path.cwd()
PT = REPO / "benchmarks/served_units/prove_time.sh"
#: what the staging tree's digest covers, the Python under each: prove_time.sh's KEY_PATHS, the same list
KEY_PATHS = "benchmarks/served_units/stage.py verity/circuits verity/definitions verity/commitments verity/protocols adapters/vllm/verity_vllm tools/verity/circuit_check/src"
OUT = Path(env("RESEARCH_RUN_DIR")) / "out"
GB = 1e9
RAM = "/dev/shm/"  # a FLOCK_WORK here is host memory, counted with the processes' RSS against WIN_MEM_GB
#: a proved statement's files prove copies to $OUT/kept/NAME/, globs under its directory, for a verifier run elsewhere: its
#: public files and every session's records. Never inst.bin (the witness), nor circuit.txt (gigabytes: the verifier's node
#: restages it from the same commit and checks it against stage.json's circuit_sha512). WIN_KEEP_ALSO adds globs.
KEEP = ("pub.bin", "stage.json", "pin.txt", "partition.json", "tables", "sessions/index.jsonl", "sessions/*/session.json",
        "sessions/*/circuit.rep*.bin", "sessions/*/verifier.json")


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
    """prove_time.sh's KEY: the staging tree's digest that names each statement's directory. Like prove_time.sh, it refuses
    a tree without one of KEY_PATHS, whose digest would lack what was under that path, and raises where a step of the
    digest's pipeline fails, whose message it lets through."""
    for p in KEY_PATHS.split():
        if not (REPO / p).exists():
            raise SystemExit(f"window.py: no {p} in {REPO}, a path of the staging key: correct KEY_PATHS in "
                             "benchmarks/served_units/prove_time.sh and window.py together")
    return subprocess.run(["bash", "-o", "pipefail", "-c", f"find {KEY_PATHS} -name '*.py' -type f | LC_ALL=C sort "
                           "| xargs sha256sum | sha256sum | cut -c1-16"], cwd=REPO, stdout=subprocess.PIPE, text=True,
                          check=True).stdout.strip()


def procs() -> dict[int, tuple[int, int]]:
    """Every process: pid -> (ppid, RSS bytes), from one `ps`, which passes over a process that exits while it reads."""
    out = subprocess.run(["ps", "-e", "-o", "pid=,ppid=,rss="], capture_output=True, text=True, check=True).stdout
    return {int(pid): (int(ppid), int(rss) * 1024) for pid, ppid, rss in (ln.split() for ln in out.splitlines())}


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
    """The bytes `path` holds, each hard-linked file once (GNU du), less what is removed while du reads: du reports such a
    file on stderr and exits 1, and its total stands."""
    r = subprocess.run(["du", "-s", "-B1", str(path)], capture_output=True, text=True)
    if not r.stdout.strip():
        raise RuntimeError(f"du {path}: {r.stderr.strip()}")
    return int(r.stdout.split()[0])


def limits(mem_default: str) -> tuple[float, list[int]]:
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) > int(env("WIN_CPUS")):
        raise SystemExit(f"window.py: {len(cpus)} CPUs in the job's affinity, more than WIN_CPUS={env('WIN_CPUS')}: run under taskset -c")
    if os.nice(0) < int(env("WIN_NICE")):
        raise SystemExit(f"window.py: niceness {os.nice(0)}, below WIN_NICE={env('WIN_NICE')}: run under nice -n {env('WIN_NICE')}")
    if not env("FLOCK_WORK").startswith(("/workspace/tmp/", RAM)):
        raise SystemExit(f"window.py: FLOCK_WORK {env('FLOCK_WORK')} is under neither /workspace/tmp/ nor {RAM}")
    return float(env("WIN_MEM_GB") or mem_default) * GB, cpus


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


def stop(jobs: list[Job]) -> None:
    """Kill every job's process group: each runs in its own session, which the runner's stop doesn't reach."""
    for j in jobs:
        try:
            os.killpg(j.proc.pid, signal.SIGKILL)
            log(f"stopped {j.label}")
        except ProcessLookupError:
            log(f"{j.label}: no process left in its group")


def kill(job: Job, why: str) -> None:
    job.killed = why
    try:
        os.killpg(job.proc.pid, signal.SIGKILL)
        log(f"killed {job.label}: {why}")
    except ProcessLookupError:
        log(f"{job.label} had already exited when {why}")


def stage(units: list[Unit]) -> int:
    mem, cpus = limits("600")
    disk_limit, deadline = float(env("WIN_DISK_GB")) * GB, until()
    slow, backfill = float(env("WIN_SLOW")), float(env("WIN_BACKFILL_GB")) * GB
    reserve, disk_backfill = float(env("WIN_FREE_GB")) * GB, float(env("WIN_BACKFILL_DISK_GB")) * GB
    fw = Path(env("FLOCK_WORK"))
    in_ram = str(fw).startswith(RAM)
    peer = int(env("WIN_MEM_PEER") or 0)
    fw.mkdir(parents=True, exist_ok=True)
    k, free = key(), shutil.disk_usage(fw).free
    if env("WIN_DEVICE") == "gpu" and not env("SM"):
        raise SystemExit("window.py: set SM, the GPUs' compute capability the prover is built for (90 for H200, 120 for RTX PRO 6000)")
    if free < reserve + 20 * GB:
        raise SystemExit(f"window.py: {free / GB:.0f} GB free under {fw}, under WIN_FREE_GB={reserve / GB:.0f} plus the build's 20")
    t0 = procs()
    log(f"stage: {len(units)} types, {len(cpus)} CPUs, niceness {os.nice(0)}, memory {mem / GB:.0f} GB, disk {disk_limit / GB:.0f} GB "
        f"({free / GB:.0f} GB free, {reserve / GB:.0f} GB kept free), until {env('WIN_UNTIL')}, FLOCK_WORK {fw}, key {k}"
        + (f", sharing WIN_MEM_GB with pid {peer}'s tree ({sum(t0[p][1] for p in tree(peer, t0)) / GB:.1f} GB now)" if peer else ""))
    one = {"FLOCK_WORK": str(fw), "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    prep = subprocess.run(["bash", str(PT), "STEPS=prep", "PT_DEVICE=cpu"], cwd=REPO, capture_output=True, text=True,
                          env={**os.environ, **one, "RESEARCH_RUN_DIR": str(OUT / "prep")})
    if not (fw / "py/bin/python").exists():
        log(f"prep failed, no venv: {prep.stdout[-2000:]}{prep.stderr[-2000:]}")
        return 1
    running: list[Job] = []
    if (fw / "flock-gpu-link/flock/target/release/flock-circuit").exists():
        log("prover already built under FLOCK_WORK: no build")
    else:
        running.append(Job("build", start("build", ["STEPS=build", f"PT_DEVICE={env('WIN_DEVICE')}"], OUT / "build",
                                          {"FLOCK_WORK": str(fw)}), time.time(), rss=16 * GB, disk=20 * GB))
    pending = [(u, n) for u in units for n in u.ns]
    atexit.register(stop, running)
    rows, n_done, last_du, used, closed, seq = [], 0, 0.0, 0, "", 0
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
        peer_rss = sum(t[p][1] for p in tree(peer, t)) if peer else 0
        if now - last_du > 10:
            used, free, last_du = du(fw), shutil.disk_usage(fw).free, now
        stagings = [j for j in running if j.unit and not j.killed]
        held = sum(j.rss_now for j in running) + (used if in_ram else 0) + peer_rss
        if held > 0.95 * mem and stagings:
            kill(max(stagings, key=lambda j: j.started), f"host memory {held / GB:.0f} GB over {0.95 * mem / GB:.0f}")
        elif (used > 0.97 * disk_limit or free < reserve) and stagings:
            kill(max(stagings, key=lambda j: j.started), f"FLOCK_WORK {used / GB:.0f} GB (limit {0.97 * disk_limit / GB:.0f}), "
                 f"{free / GB:.0f} GB free (floor {reserve / GB:.0f})")
            closed = closed or "disk"
        if now > deadline:
            for j in stagings:
                kill(j, f"WIN_UNTIL {env('WIN_UNTIL')}")
            closed = closed or "time"
        blocked_mem = blocked_disk = False
        for u, n in list(pending):
            reason = closed
            if not reason:
                mem_in_use = sum(max(j.rss, j.peak) for j in running) + (used if in_ram else 0) + peer_rss
                if now + u.secs * slow > deadline:
                    reason = "time"
                elif (sum(j.disk for j in running if j.unit) + u.disk * GB > min(0.95 * disk_limit - used, free - reserve)
                      or (blocked_disk and u.disk * GB > disk_backfill)):
                    reason = "disk" if not any(j.unit for j in running) else "wait"
                    blocked_disk = blocked_disk or reason == "wait"
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
            seq += 1
            label = f"{seq:03d}"
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


def read_seconds(path: str) -> dict[str, tuple[float, bool]]:
    """An earlier prove's seconds.tsv (WIN_SECONDS): each statement's seconds, and whether it finished in them (its prover
    exited 0)."""
    if not path:
        return {}
    rows = [ln.split("\t") for ln in Path(path).read_text().splitlines()[1:] if ln.strip()]
    return {r[0]: (float(r[2]), r[3] == "1") for r in rows}


def statements_run() -> list[tuple[str, str, float, bool, str]]:
    """Each statement a group started: (name, group, seconds, finished, its prover's exit status), from prove_time.sh's
    start, done and exit files. It finished when its session set ended with the prover's exit status 0; an unfinished
    one's seconds are those so far, or to its end (killed or failed), and its status "" while it runs."""
    rows = []
    for s in sorted(OUT.glob("gpu*/out/[cg]pu/*/start")):
        d, done, ex = s.parent, s.parent / "done", s.parent / "exit"
        t0, t1 = s.read_text().strip(), done.read_text().strip() if done.exists() else ""
        rc = ex.read_text().strip() if t1 and ex.exists() else ""
        if t0:
            rows.append((d.name, d.parts[-4], (float(t1) if t1 else time.time()) - float(t0), rc == "0", rc))
    return rows


def keep(fw: Path, k: str, kept: set[str]) -> None:
    """Each statement whose session set ended, not yet in `kept`: its KEEP files, and WIN_KEEP_ALSO's globs, copied to
    $OUT/kept/NAME/, with its prover's exit status (`exit`), so a verifier tells a killed or failed prover's sessions from
    a finished one's."""
    globs = KEEP + tuple(filter(None, env("WIN_KEEP_ALSO").split(":")))
    for done in sorted(OUT.glob("gpu*/out/[cg]pu/*/done")):
        nm = done.parent.name
        if nm in kept:
            continue
        src, dst = fw / "data" / f"{nm}-{k}", OUT / "kept" / nm
        dst.mkdir(parents=True, exist_ok=True)
        if (done.parent / "exit").exists():
            shutil.copyfile(done.parent / "exit", dst / "exit")
        for g in globs:
            for f in sorted(src.glob(g)):
                t = dst / f.relative_to(src)
                t.parent.mkdir(parents=True, exist_ok=True)
                if f.is_dir():
                    shutil.copytree(f, t, dirs_exist_ok=True)
                else:
                    shutil.copyfile(f, t)
        kept.add(nm)


def prove(units: list[Unit]) -> int:
    mem, cpus = limits("268")
    fw, k = Path(env("FLOCK_WORK")), key()
    ngpu, dev_mib = int(env("WIN_GPUS")), float(env("WIN_DEV_GIB")) * 1024
    on_gpu = env("WIN_DEVICE") == "gpu"
    if on_gpu and not shutil.which("nvidia-smi"):
        raise SystemExit("window.py: no nvidia-smi on PATH, so no device-memory limit: prove on a GPU node, or WIN_DEVICE=cpu")
    for also in filter(None, env("WIN_ALSO").split(":")):
        link_also(fw, Path(also))
    binary = fw / "flock-gpu-link/flock/target/release/flock-circuit"
    if not binary.exists():
        log(f"no prover at {binary}: building it inside the window")
        subprocess.run(["bash", str(PT), "STEPS=build", f"PT_DEVICE={env('WIN_DEVICE')}"], cwd=REPO,
                       env={**os.environ, "FLOCK_WORK": str(fw), "RESEARCH_RUN_DIR": str(OUT / "build")}, check=True)
    measured, floor = read_seconds(env("WIN_SECONDS")), float(env("WIN_UNMEASURED_S"))
    staged = []
    for u in units:
        ns = [n for n in u.ns if (fw / "data" / f"{name(u.unit, n)}-{k}" / ".done").exists()]
        if ns:
            est, fresh = 0.0, False
            for n in ns:
                r = json.loads((fw / "data" / f"{name(u.unit, n)}-{k}" / "stage.json").read_text())
                model = 10 + 3 * r["circuit_bytes"] / GB + (int(env("WARM")) + int(env("RUNS"))) * \
                    (4 + 2 ** max(r["k_log"] - 23, 0))
                s, fin = measured.get(name(u.unit, n), (0.0, False))
                est += s if fin else max(s, floor) if measured else model
                fresh = fresh or not fin
            staged.append((u, ns, est, fresh))
        else:
            log(f"not staged, skipped: {u.unit}")
    if measured:
        log(f"estimates: {sum(not x[3] for x in staged)} of {len(staged)} types measured in {env('WIN_SECONDS')}; an unmeasured "
            f"statement {floor:.0f} s or its unfinished seconds there, whichever is more, and started first in its group")
    other = sorted({d.name.rsplit("-", 1)[1] for d in (fw / "data").glob("*-*") if (d / ".done").exists()} - {k})
    if other:
        log(f"statements staged under key(s) {', '.join(other)}, not this tree's {k}: prove from the staging's commit")
    ram = fw if str(fw).startswith(RAM) else None
    if ram:
        held = du(fw)
        if held >= 0.9 * mem:
            raise SystemExit(f"window.py: FLOCK_WORK holds {held / GB:.0f} GB of host memory, too much of WIN_MEM_GB={mem / GB:.0f}")
        log(f"FLOCK_WORK holds {held / GB:.0f} GB of host memory: {(mem - held) / GB:.0f} GB left for the provers and sessions")
    groups: list[list] = [[] for _ in range(ngpu)]
    for item in sorted(staged, key=lambda x: -x[2]):
        min(groups, key=lambda g: sum(x[2] for x in g)).append(item)
    for items in groups:
        items.sort(key=lambda x: (not x[3], -x[2]))
    scratch = Path(tempfile.mkdtemp(prefix="window-bin-"))
    path = env("PATH")
    if not shutil.which("gpu-lease"):
        (scratch / "gpu-lease").symlink_to(REPO / "tools/research/src/research/pods/sh/gpu_lease.sh")
        path = f"{scratch}:{path}"
    per = len(cpus) // ngpu
    jobs = []
    log(f"prove: {len(staged)} types in {ngpu} groups, {per} CPUs each, host memory {mem / GB:.0f} GB, device {dev_mib / 1024:.0f} GiB a GPU")
    for g, items in enumerate(groups):
        if not items:
            continue
        units_arg = " ".join(f"{u.unit}@{','.join(ns)}" for u, ns, *_x in items)
        lease = int(env("LEASE_MIN"))
        cores = ",".join(map(str, cpus[g * per:(g + 1) * per]))
        log(f"group {g}: {len(items)} types ({sum(x[3] for x in items)} unmeasured, first), estimated "
            f"{sum(x[2] for x in items) / 60:.1f} min, lease {lease} min, cores {cores}")
        p = start(f"gpu group {g}", ["STEPS=prove", f"PT_DEVICE={env('WIN_DEVICE')}", f"PT_UNITS={units_arg}", f"LEASE_MIN={lease}",
                                     f"VCPUS={env('VCPUS')}", f"WARM={env('WARM')}", f"RUNS={env('RUNS')}"],
                  OUT / f"gpu{g}", {"FLOCK_WORK": str(fw), "PATH": path, "GPU_LEASE_WHO": env("GPU_LEASE_WHO")},
                  prefix=["taskset", "-c", cores])
        jobs.append(Job(f"gpu{g}", p, time.time()))
    kept: set[str] = set()
    try:
        watch(jobs, mem, dev_mib, on_gpu, ram, lambda: keep(fw, k, kept))
    finally:
        stop(jobs)
        shutil.rmtree(scratch, ignore_errors=True)
        try:
            keep(fw, k, kept)
            ran = statements_run()
            (OUT / "seconds.tsv").write_text("name\tgroup\tseconds\tdone\texit\n" +
                                             "".join(f"{n}\t{g}\t{s:.1f}\t{int(f)}\t{rc}\n" for n, g, s, f, rc in ran))
            log(f"kept {len(kept)} statements' files in out/kept ({du(OUT / 'kept') / GB:.2f} GB); {sum(r[3] for r in ran)} "
                f"of {len(ran)} started statements finished with exit 0, {sum(bool(r[4]) and not r[3] for r in ran)} "
                "ended otherwise (out/seconds.tsv)")
        finally:
            if env("WIN_KEEP") != "1":
                freed = du(fw)
                shutil.rmtree(fw, ignore_errors=True)
                log(f"removed FLOCK_WORK {fw} ({freed / GB:.1f} GB)")
    rows = []
    for j in jobs:
        s = OUT / j.label / "out" / "summary.json"
        log(f"{j.label} exit {j.proc.returncode}")
        if s.exists():
            for st, rec in json.loads(s.read_text())["statements"].items():
                rows.append([j.label, st, str(rec.get("n")), str(rec.get("k_log")), str(rec.get("prove_total_s", "")),
                             json.dumps({x: rec[x] for x in rec if x not in ("n", "k_log", "prove_total_s")})])
    (OUT / "prove.tsv").write_text("group\tstatement\tn\tk_log\tprove_total_s\trecord\n" + "".join("\t".join(r) + "\n" for r in rows))
    log(f"prove done: {len(rows)} statements with a summary")
    return 0


def link_also(fw: Path, also: Path) -> None:
    """Each statement staged under `also` (another FLOCK_WORK) and not under `fw`, as a directory of `fw`'s data whose
    files link to its own, so its sessions are written under `fw`; and `also`'s venv and prover where `fw` has none."""
    n = 0
    for d in sorted(also.glob("data/*-*")):
        t = fw / "data" / d.name
        if (d / ".done").exists() and not t.exists():
            t.mkdir(parents=True)
            for f in d.iterdir():
                (t / f.name).symlink_to(f)
            n += 1
    for x in ("py", "flock-gpu-link"):
        if not (fw / x).exists() and (also / x).exists():
            (fw / x).symlink_to(also / x)
    log(f"linked {n} statements staged under {also}")


def watch(jobs: list[Job], mem: float, dev_mib: float, on_gpu: bool, ram: Path | None = None,
          tick: Callable[[], None] | None = None) -> None:
    """Until every group exits: kill a group's prover past dev_mib on its GPU, and the largest of our processes past
    0.97 x mem, counting a FLOCK_WORK in RAM (`ram`, its statements and the sessions written beside them) with them.
    Device memory unread for 30 s in a row kills every prover, since the limit can't be held. `tick` runs each round."""
    blind, held, last, unread = 0, du(ram) if ram else 0, time.time(), set()
    while any(j.proc.poll() is None for j in jobs):
        if tick:
            tick()
        if ram and time.time() - last > 10:
            held, last = du(ram), time.time()
        t = procs()
        ours = {p for j in jobs for p in tree(j.proc.pid, t)}
        used: dict[str, str] = {}
        if on_gpu:
            try:
                q = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=10)
                used = dict(ln.replace(" ", "").split(",") for ln in q.stdout.splitlines() if "," in ln)
            except (subprocess.TimeoutExpired, OSError) as e:
                log(f"nvidia-smi unread ({type(e).__name__}): no device memory this round")
            blind = 0 if used else blind + 1
            if blind == 15:
                log("device memory unread for 30 s: killing every prover")
                dev_mib = -1.0
        for j in jobs:
            for p in tree(j.proc.pid, t) if not j.gpu else ():
                try:
                    m = re.search(rb"(?:^|\0)GPU_LEASE_UUID=([^\0]+)", Path(f"/proc/{p}/environ").read_bytes())
                except OSError as e:
                    if p not in unread:
                        log(f"{j.label}: pid {p}'s environment unread ({e.strerror}), passed over")
                    unread.add(p)
                    continue
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
                    except OSError as e:
                        log(f"{j.label}: pid {p} not killed ({e.strerror}): it exited since the scan")
        total = sum(t[p][1] for p in ours) + held
        if total > 0.97 * mem:
            big = max((p for p in ours if t[p][1] > GB), key=lambda p: t[p][1], default=None)
            if big:
                os.kill(big, signal.SIGKILL)
                log(f"killed pid {big} ({t[big][1] / GB:.0f} GB): our RSS and FLOCK_WORK in RAM {total / GB:.0f} GB over "
                    f"{0.97 * mem / GB:.0f}")
        time.sleep(2)


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in ("stage", "prove"):
        raise SystemExit("window.py: usage: window.py stage|prove UNITS_FILE")
    OUT.mkdir(parents=True, exist_ok=True)
    units = read_units(Path(sys.argv[2]))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    return stage(units) if sys.argv[1] == "stage" else prove(units)


if __name__ == "__main__":
    sys.exit(main())
