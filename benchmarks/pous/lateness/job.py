"""One `research run` on vy-nebius-2 (2 GPUs from the queue): build `lat`, start the responder on GPU 0 of the job, then for
each condition start its load and time challenges from three challengers in turn: on this node over loopback (the
on-node lower bound), from vy-nebius-cpu-1 and from vy-nebius-1 over the VPC (the two-machine path). Stdlib only.

Conditions: `quiet` (no load of ours; the node's other tenants as they are), `hbm` (bf16 GEMMs and device-to-device
copies on both GPUs, every other CPU of the job spinning or copying memory), `pcie` (`hbm` plus pinned H2D and D2H copies
on both GPUs), `vllm` (vLLM generation on the responder's GPU). The responder and the local challenger each hold a whole
core of the job's cpuset that no load touches. Everything goes to $OUT: per cell `<cond>/<challenger>.bin|json`, the load
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
VENV = "/workspace/jobs/venv312/bin/python"
MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
REC = struct.Struct("<IIIHH")


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
    return resp, chal, [c for c in cpus if c not in reserved]


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
    for op, name in ((1, "gpu"), (2, "host"), (3, "ping")):
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
        if cond in ("hbm", "pcie"):
            pcie = ["pcie_mib=64", "pcie=both"] if cond == "pcie" else []
            for dev in (0, 1):
                hogs = [f"spin_cpus={','.join(map(str, cpus[:half]))}", f"mem_cpus={','.join(map(str, cpus[half:]))}"] if dev == 0 else []
                self.spawn(["taskset", "-c", cpu_arg, lat, "load", f"device={dev}", "gemm=8192", "d2d_mib=2048", *pcie, *hogs,
                            f"report={celldir}/load-gpu{dev}.jsonl"], f"{celldir}/load-gpu{dev}.log")
        elif cond == "vllm":
            env = dict(os.environ, HF_HOME="/workspace/hf", HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
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


def local_probe(lat: str, cpu: int, celldir: str, pings: int, rounds: int) -> dict:
    out = os.path.join(celldir, "local")
    p = subprocess.run([lat, "probe", f"to=127.0.0.1:{PORT}", f"cpu={cpu}", f"pings={pings}", f"rounds={rounds}", f"out={out}"],
                       capture_output=True, text=True, timeout=1800)
    return {"rc": p.returncode, "stderr": p.stderr[-2000:]}


def remote_probe(name: str, src: bytes, celldir: str, pings: int, rounds: int) -> dict:
    dest, lport = REMOTES[name]
    script = ("set -e; d=$(mktemp -d /tmp/lat-probe.XXXXXX); trap 'rm -rf \"$d\"' EXIT; cat > \"$d/lat.cu\"; "
              "g++ -O2 -std=c++17 -x c++ -o \"$d/lat\" \"$d/lat.cu\" -lpthread; "
              f"\"$d/lat\" probe to={NODE_IP}:{PORT} lport={lport} cpu=auto pings={pings} rounds={rounds} out=\"$d/{name}\" >&2 "
              "|| echo \"probe rc=$?\" >&2; "
              f"(uptime; nproc) > \"$d/{name}.host\"; tar -C \"$d\" -cf - {name}.bin {name}.json {name}.host")
    p = subprocess.run([*SSH, dest, script], input=src, capture_output=True, timeout=1800)
    if p.returncode == 0 and p.stdout:
        tarfile.open(fileobj=io.BytesIO(p.stdout)).extractall(celldir)
    return {"rc": p.returncode, "stderr": p.stderr.decode(errors="replace")[-2000:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="quiet,hbm,pcie,vllm")
    ap.add_argument("--challengers", default="local,cpu1,node1")
    ap.add_argument("--pings", type=int, default=20000)
    ap.add_argument("--rounds-local", type=int, default=300000)
    ap.add_argument("--rounds-remote", type=int, default=500000)
    ap.add_argument("--rounds-quiet-cpu1", type=int, default=1000000)
    ap.add_argument("--rounds-node1", type=int, default=100000)
    ap.add_argument("--vllm-timeout", type=float, default=600)
    ns = ap.parse_args()
    out = os.environ.get("OUT") or sys.exit("$OUT is unset: run under research run")
    scratch = tempfile.mkdtemp(prefix="lat-")
    src = open(os.path.join(HERE, "lat.cu"), "rb").read()
    lat = os.path.join(scratch, "lat")
    log("build")
    subprocess.run(["nvcc", "-O2", "-std=c++17", "-arch=sm_120", "-o", lat, os.path.join(HERE, "lat.cu"), "-lcublas"], check=True)
    resp_cpu, chal_cpu, load_cpus = plan_cpus()
    env = {"hostname": os.uname().nodename, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "affinity": sorted(os.sched_getaffinity(0)), "responder_cpu": resp_cpu, "challenger_cpu": chal_cpu,
           "load_cpus": load_cpus, "nice": os.nice(0), "loadavg": open("/proc/loadavg").read().strip(),
           "nvidia_smi": gpu_sample(), "nvcc": sh(["nvcc", "--version"]).strip().splitlines()[-1],
           "cmdline": open("/proc/cmdline").read().strip(), "argv": sys.argv}
    json.dump(env, open(os.path.join(out, "env.json"), "w"), indent=1)
    log(f"cpus: responder {resp_cpu}, challenger {chal_cpu}, load {load_cpus}")

    ready = os.path.join(scratch, "ready")
    peers = [f"peer={dest.split('@')[1]}:{lport}" for dest, lport in REMOTES.values()]
    responder = subprocess.Popen([lat, "serve", f"port={PORT}", f"cpu={resp_cpu}", f"ready={ready}",
                                  f"stats={out}/responder-stats.json", *peers],
                                 stdout=open(f"{out}/responder.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
    while not os.path.exists(ready) or not open(ready).read().strip():
        if responder.poll() is not None:
            sys.exit(f"responder exited {responder.returncode}")
        time.sleep(0.5)
    log(f"responder up: {open(ready).read().strip()}")

    summary: dict = {"env": env, "cells": {}}
    try:
        for cond in ns.conditions.split(","):
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
            for who in ns.challengers.split(","):
                a = busy_ticks()
                t0 = time.time()
                if who == "local":
                    r = local_probe(lat, chal_cpu, celldir, ns.pings, ns.rounds_local)
                    name = "local"
                else:
                    rounds = ns.rounds_node1 if who == "node1" else ns.rounds_quiet_cpu1 if cond == "quiet" else ns.rounds_remote
                    r = remote_probe(who, src, celldir, ns.pings, rounds)
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
        os.killpg(responder.pid, signal.SIGTERM)
        responder.wait(30)
        json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
        subprocess.run(["rm", "-rf", scratch])
    log("done")


if __name__ == "__main__":
    main()
