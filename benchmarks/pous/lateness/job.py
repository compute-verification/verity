"""One job on a GPU node (2 GPUs from the queue): build `lat`, start the responder on GPU 0 of the job, then for
each condition start its load and time challenges from three challengers in turn: on this node over loopback (the
on-node lower bound), from a CPU node and from a second GPU node over the VPC (the two-machine path). Stdlib only.

Conditions: `quiet` (no load of ours; the node's other tenants as they are); `gemm` (back-to-back bf16 8192^3 GEMMs on
both GPUs, and the job's other declared CPUs spinning or copying memory); `hbm` (`gemm` plus 2 GiB device-to-device
copies); `pcie` (`hbm` plus 64 MiB pinned H2D and D2H copies, both directions at once); `pcie1m` (the same in 1 MiB
copies); `vllm` (vLLM generation on the responder's GPU). Those loads run in processes of their own, so the GPU
time-slices between their CUDA contexts and the responder's; `gemm-in`, `hbm-in`, `pcie-in` and `pcie1m-in` run the
same GPU load inside the responder's process (one context, where stream priority orders the work), the second GPU's and
the CPUs' load as before; `pcie-in-sm` is `pcie-in` challenged only on the SM-kernel and host paths (no copy engine). The responder and the local challenger each hold a whole core
of the job's cpuset that no load touches, and the load takes the job's declared CPU count less those four. Everything goes to $OUT: per cell `<cond>/<challenger>.bin|json`, the load
reports, the responder's stats, the environment, and `summary.json`."""

from __future__ import annotations

import argparse
import io
import json
import os
import signal
import struct
import subprocess
import sys
import tarfile
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SSH = ["ssh", "-i", os.path.expanduser("~/.ssh/vy-cluster"), "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
       "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20"]
NODE_IP = "10.80.0.42"
PORT = 47310
REMOTES = {"cpu1": ("research@10.80.0.89", 47311), "node1": ("research@10.80.0.18", 47312)}
TAUS_US = (50, 100, 150, 300, 840)
CUDA = "/usr/local/cuda"
NVCC = f"{CUDA}/bin/nvcc"
VENV = "/workspace/jobs/venv312/bin/python"
VLLM_CUDA = "/workspace/jobs/cuda-12.9"  # the toolkit the venv's torch (cu129) JIT-compiles with
MODEL = "/workspace/hf/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct/snapshots/a2856192dd7c25b842431f39c179a6c2c2f627d1"
REC = struct.Struct("<IIIHH")
OPS = {1: "gpu", 5: "gpu_sm", 2: "host", 3: "ping"}
LOADS = {"gemm": ["gemm=8192"], "hbm": ["gemm=8192", "d2d_mib=2048"],
         "pcie": ["gemm=8192", "d2d_mib=2048", "pcie_mib=64", "pcie=both"],
         "pcie1m": ["gemm=8192", "d2d_mib=2048", "pcie_mib=1", "pcie=both"]}
IN_PROCESS = {f"{k}-in": v for k, v in LOADS.items()}
IN_PROCESS["pcie-in-sm"] = LOADS["pcie"]
# The challenge ops a condition's probes cycle through, where not the default "1,5,1,5,2".
PROBE_OPS = {"pcie-in-sm": "5,2"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sh(cmd: list[str], **kw) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, **kw).stdout


def siblings(cpu: int) -> set[int]:
    text = open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read().strip()
    out: set[int] = set()
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def plan_cpus() -> tuple[int, int, list[int]]:
    """The responder's CPU, the local challenger's, and the load's: two whole cores reserved, the rest to the load."""
    cpus = sorted(os.sched_getaffinity(0))
    whole = [c for c in cpus if siblings(c) <= set(cpus) and c == min(siblings(c))]
    if len(whole) < 2:
        raise SystemExit(f"need two whole cores in the job's cpuset, got {cpus}")
    resp, chal = whole[0], whole[1]
    reserved = siblings(resp) | siblings(chal)
    declared = int(os.environ.get("OMP_NUM_THREADS") or len(cpus))
    return resp, chal, [c for c in cpus if c not in reserved][: max(0, declared - len(reserved))]


def busy_ticks() -> dict[int, tuple[int, int]]:
    out = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3].isdigit():
            f = line.split()
            v = list(map(int, f[1:9]))
            out[int(f[0][3:])] = (sum(v) - v[3] - v[4], sum(v))
    return out


def busy_frac(a: dict, b: dict, cpus) -> float:
    busy = sum(b[c][0] - a[c][0] for c in cpus)
    total = sum(b[c][1] - a[c][1] for c in cpus)
    return busy / total if total else 0.0


def gpu_sample() -> str:
    return sh(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw,clocks.sm,clocks.mem,"
               "pcie.link.gen.current,pcie.link.width.current", "--format=csv,noheader"])


def summarize(path: str) -> dict:
    data = open(path, "rb").read()
    by_op: dict[int, list[int]] = {}
    flags = {"resp_preempted": 0, "chal_preempted": 0, "lost": 0, "bad_data": 0}
    for e, _s, _d, op, f in REC.iter_unpack(data):
        by_op.setdefault(op, []).append(e)
        flags["resp_preempted"] += bool(f & 1)
        flags["chal_preempted"] += bool(f & 2)
        flags["lost"] += bool(f & 4)
        flags["bad_data"] += bool(f & 8)
    out: dict = {"flags": flags}
    for op, name in OPS.items():
        xs = sorted(by_op.get(op, []))
        if not xs:
            continue
        q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))] / 1000  # noqa: E731
        out[name] = {"n": len(xs), "p50_us": q(0.5), "p99_us": q(0.99), "p999_us": q(0.999), "max_us": xs[-1] / 1000,
                     "late": {str(t): sum(1 for x in xs if x > t * 1000) for t in TAUS_US}}
    return out


class Load:
    """A condition's load processes, each in its own process group, stopped by group."""

    def __init__(self, cond: str, lat: str, cpus: list[int], celldir: str, scratch: str):
        self.procs: list[subprocess.Popen] = []
        self.stop_file = os.path.join(scratch, f"stop-{cond}")
        self.ready_file = os.path.join(scratch, f"ready-{cond}")
        half = len(cpus) // 2
        cpu_arg = ",".join(map(str, cpus))
        hogs = [f"spin_cpus={','.join(map(str, cpus[:half]))}", f"mem_cpus={','.join(map(str, cpus[half:]))}"]
        if cond in LOADS:
            for dev in (0, 1):
                self.spawn(["taskset", "-c", cpu_arg, lat, "load", f"device={dev}", *LOADS[cond], *(hogs if dev == 0 else []),
                            f"report={celldir}/load-gpu{dev}.jsonl"], f"{celldir}/load-gpu{dev}.log")
        elif cond in IN_PROCESS:
            self.spawn(["taskset", "-c", cpu_arg, lat, "load", "device=1", *IN_PROCESS[cond], f"report={celldir}/load-gpu1.jsonl"],
                       f"{celldir}/load-gpu1.log")
            self.spawn(["taskset", "-c", cpu_arg, lat, "load", *hogs, f"report={celldir}/load-cpu.jsonl"], f"{celldir}/load-cpu.log")
        elif cond == "vllm":
            env = dict(os.environ, HF_HOME="/workspace/hf", HF_HUB_OFFLINE="1", CUDA_HOME=VLLM_CUDA,
                       PATH=f"{os.path.dirname(VENV)}:{VLLM_CUDA}/bin:{os.environ.get('PATH', '')}",
                       CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
            self.spawn(["taskset", "-c", cpu_arg, VENV, os.path.join(HERE, "vllm_load.py"), MODEL, self.ready_file, self.stop_file,
                        f"{celldir}/vllm.jsonl"], f"{celldir}/vllm.log", env=env)

    def spawn(self, cmd: list[str], logfile: str, env=None) -> None:
        self.procs.append(subprocess.Popen(cmd, stdout=open(logfile, "w"), stderr=subprocess.STDOUT, env=env, start_new_session=True))

    def wait_ready(self, cond: str, timeout_s: float) -> bool:
        if cond != "vllm":
            time.sleep(10 if self.procs else 0)
            return all(p.poll() is None for p in self.procs)
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if os.path.exists(self.ready_file):
                time.sleep(20)
                return all(p.poll() is None for p in self.procs)
            if any(p.poll() is not None for p in self.procs):
                return False
            time.sleep(2)
        return False

    def stop(self) -> list[int | None]:
        open(self.stop_file, "w").close()
        for p in self.procs:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p in self.procs:
            try:
                p.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        return [p.returncode for p in self.procs]


def local_probe(lat: str, cpu: int, celldir: str, pings: int, rounds: int, max_s: int, extra: tuple[str, ...] = ()) -> dict:
    out = os.path.join(celldir, "local")
    p = subprocess.run([lat, "probe", f"to=127.0.0.1:{PORT}", f"cpu={cpu}", f"pings={pings}", f"rounds={rounds}", f"max_s={max_s}",
                        *extra, f"out={out}"],
                       capture_output=True, text=True, timeout=1800)
    return {"rc": p.returncode, "stderr": p.stderr[-2000:]}


def remote_probe(name: str, src: bytes, celldir: str, pings: int, rounds: int, max_s: int, extra: tuple[str, ...] = (),
                 out: str | None = None, port: int = PORT, to: str | None = None, lport: int | None = None) -> dict:
    dest, default_lport = REMOTES[name]
    lport = lport or default_lport
    out = out or name
    to = to or f"{NODE_IP}:{port}"
    script = ("set -e; d=$(mktemp -d /tmp/lat-probe.XXXXXX); trap 'rm -rf \"$d\"' EXIT; cat > \"$d/lat.cu\"; "
              "g++ -O2 -std=c++17 -x c++ -o \"$d/lat\" \"$d/lat.cu\" -lpthread; "
              f"\"$d/lat\" probe to={to} lport={lport} cpu=auto pings={pings} rounds={rounds} max_s={max_s} "
              f"{' '.join(extra)} out=\"$d/{out}\" >&2 || echo \"probe rc=$?\" >&2; "
              f"(uptime; nproc) > \"$d/{out}.host\"; tar -C \"$d\" -cf - {out}.bin {out}.json {out}.host")
    p = subprocess.run([*SSH, dest, script], input=src, capture_output=True, timeout=1800)
    if p.returncode == 0 and p.stdout:
        tarfile.open(fileobj=io.BytesIO(p.stdout)).extractall(celldir)
    return {"rc": p.returncode, "stderr": p.stderr.decode(errors="replace")[-2000:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="quiet,gemm,hbm,pcie,pcie1m,vllm")
    ap.add_argument("--challengers", default="local,cpu1,node1")
    ap.add_argument("--pings", type=int, default=20000)
    ap.add_argument("--rounds-local", type=int, default=300000)
    ap.add_argument("--rounds-remote", type=int, default=500000)
    ap.add_argument("--rounds-quiet-cpu1", type=int, default=1000000)
    ap.add_argument("--rounds-node1", type=int, default=100000)
    ap.add_argument("--max-s-local", type=int, default=50)
    ap.add_argument("--max-s-remote", type=int, default=120)
    ap.add_argument("--max-s-quiet-cpu1", type=int, default=240)
    ap.add_argument("--max-s-node1", type=int, default=40)
    ap.add_argument("--vllm-timeout", type=float, default=600)
    ns = ap.parse_args()
    out = os.environ.get("OUT") or sys.exit("$OUT is unset: run under research run")
    scratch = tempfile.mkdtemp(prefix="lat-")
    src = open(os.path.join(HERE, "lat.cu"), "rb").read()
    lat = os.path.join(scratch, "lat")
    log("build")
    subprocess.run([NVCC, "-O2", "-std=c++17", "-arch=sm_120", "-o", lat, os.path.join(HERE, "lat.cu"), "-lcublas",
                    "-Xlinker", f"-rpath={CUDA}/lib64"], check=True)
    resp_cpu, chal_cpu, load_cpus = plan_cpus()
    env = {"hostname": os.uname().nodename, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "affinity": sorted(os.sched_getaffinity(0)), "responder_cpu": resp_cpu, "challenger_cpu": chal_cpu,
           "load_cpus": load_cpus, "nice": os.nice(0), "loadavg": open("/proc/loadavg").read().strip(),
           "nvidia_smi": gpu_sample(), "nvcc": sh([NVCC, "--version"]).strip().splitlines()[-1],
           "cmdline": open("/proc/cmdline").read().strip(), "argv": sys.argv}
    json.dump(env, open(os.path.join(out, "env.json"), "w"), indent=1)
    log(f"cpus: responder {resp_cpu}, challenger {chal_cpu}, load {load_cpus}")

    peers = [f"peer={dest.split('@')[1]}:{lport}" for dest, lport in REMOTES.values()]

    def start_responder(args: list[str], tag: str) -> subprocess.Popen:
        ready = os.path.join(scratch, f"responder-ready-{tag}")
        p = subprocess.Popen([lat, "serve", f"port={PORT}", f"cpu={resp_cpu}", f"ready={ready}",
                              f"stats={out}/responder-stats-{tag}.json", *peers, *args],
                             stdout=open(f"{out}/responder-{tag}.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
        while not os.path.exists(ready) or not open(ready).read().strip():
            if p.poll() is not None:
                sys.exit(f"responder exited {p.returncode}")
            time.sleep(0.5)
        log(f"responder {tag} up: {open(ready).read().strip()} {' '.join(args)}")
        return p

    def stop_responder(p: subprocess.Popen) -> None:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait(60)

    resp_args: list[str] = []
    responder = start_responder(resp_args, "plain")
    summary: dict = {"env": env, "cells": {}}
    try:
        for cond in ns.conditions.split(","):
            want = IN_PROCESS.get(cond, [])
            if want != resp_args:
                stop_responder(responder)
                resp_args = want
                responder = start_responder(resp_args, cond)
                time.sleep(5)
            celldir = os.path.join(out, cond)
            os.makedirs(celldir, exist_ok=True)
            log(f"{cond}: start load")
            load = Load(cond, lat, load_cpus, celldir, scratch)
            ok = load.wait_ready(cond, ns.vllm_timeout)
            cells = summary["cells"].setdefault(cond, {"load_ok": ok})
            if not ok:
                log(f"{cond}: load failed to start, skipped")
                cells["load_rc"] = load.stop()
                continue
            cells["gpu_during"] = gpu_sample()
            extra = (f"ops={PROBE_OPS[cond]}",) if cond in PROBE_OPS else ()
            for who in ns.challengers.split(","):
                a = busy_ticks()
                t0 = time.time()
                if who == "local":
                    r = local_probe(lat, chal_cpu, celldir, ns.pings, ns.rounds_local, ns.max_s_local, extra)
                    name = "local"
                else:
                    rounds, max_s = ((ns.rounds_node1, ns.max_s_node1) if who == "node1" else
                                     (ns.rounds_quiet_cpu1, ns.max_s_quiet_cpu1) if cond == "quiet" else
                                     (ns.rounds_remote, ns.max_s_remote))
                    r = remote_probe(who, src, celldir, ns.pings, rounds, max_s, extra)
                    name = who
                b = busy_ticks()
                r.update(seconds=round(time.time() - t0, 1), responder_cpu_busy=busy_frac(a, b, [resp_cpu]),
                         load_cpus_busy=busy_frac(a, b, load_cpus) if load_cpus else None,
                         node_busy=busy_frac(a, b, list(b)), loadavg=open("/proc/loadavg").read().strip())
                binpath = os.path.join(celldir, f"{name}.bin")
                if os.path.exists(binpath):
                    r["summary"] = summarize(binpath)
                cells[name] = r
                log(f"{cond}/{name}: {json.dumps(r.get('summary', r))[:400]}")
            cells["gpu_after"] = gpu_sample()
            cells["load_rc"] = load.stop()
            json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    finally:
        stop_responder(responder)
        json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
        subprocess.run(["rm", "-rf", scratch])
    log("done")


if __name__ == "__main__":
    main()
