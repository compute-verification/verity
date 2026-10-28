"""The two-machine network path alone, as one CPU job on a GPU node: `lat serve` built with g++ (answers from
host memory, no GPU) on a core of the job, then challengers on a CPU node and a second GPU node in turn, each in
variants that separate the network's causes: back-to-back pings, pings paced 1 ms and 300 µs apart (adaptive interrupt
moderation follows the packet rate), and 4 KiB answers in 1,472-byte datagrams (fragmented on cpu-1's MTU-1450 path)
and in 1,422-byte ones (whole), back to back and paced, and a five-minute ping series 1 ms apart (how the path's
modes change with time), then that series on three paths at once: cpu-1 and node 1 to node 2, and cpu-1 to a responder
on node 1 (`cpu1-to-node1/`), with each machine's CPUs, receive softirqs and NIC interrupts sampled once a second
(`cpusample.py`, `<machine>-cpus.jsonl`). `--only NAME,...` runs only those variants (`ping-series-concurrent` the
concurrent one). Writes `<challenger>/<variant>.bin|json` and `summary.json` to $OUT."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from job import HERE, REMOTES, SSH, busy_ticks, busy_frac, log, plan_cpus, remote_probe, summarize  # noqa: E402

PORT = 47320
# (name, pings, rounds, max_s, extra probe arguments)
VARIANTS = [
    ("ping-b2b", 200000, 0, 60, ("k=111", "gap_us=2000")),
    ("ping-1ms", 40000, 0, 60, ("k=1", "gap_us=1000")),
    ("ping-300us", 60000, 0, 60, ("k=1", "gap_us=300")),
    ("host4k-1472-b2b", 0, 150000, 45, ("ops=2", "dgram=1472")),
    ("host4k-1422-b2b", 0, 150000, 45, ("ops=2", "dgram=1422")),
    ("host4k-1422-1ms", 0, 40000, 50, ("ops=2", "dgram=1422", "k=1", "gap_us=1000")),
    ("ping-series-1ms", 0, 400000, 300, ("ops=3", "k=1", "gap_us=1000")),
]
# The series on three paths at once, cpu-1 and node 1 to node 2 and cpu-1 to node 1 (a responder there): which
# machine's side the slow episodes follow.
CONCURRENT = ("ping-series-concurrent", 0, 400000, 300, ("ops=3", "k=1", "gap_us=1000"))
CROSS_PORT, CROSS_LPORT = 47330, 47313


def remote_serve(src: bytes, seconds: int, log_path: str, who: str = "node1", port: int = CROSS_PORT,
                 peers: tuple[str, ...] = (f"{REMOTES['cpu1'][0].split('@')[1]}:{CROSS_LPORT}",),
                 host_mib: int = 1024) -> subprocess.Popen:
    """`lat serve` on `who` (default the second GPU node) for `seconds`, built and run in a scratch dir that it removes on exit.
    The log gets `PID n` (for `remote_stop`: a killed ssh client leaves the remote server running) and `READY` once it
    is bound (`remote_ready`)."""
    dest = REMOTES[who][0]
    script = ("set -e; d=$(mktemp -d /tmp/lat-serve.XXXXXX); trap 'rm -rf \"$d\"' EXIT; cat > \"$d/lat.cu\"; "
              "g++ -O2 -std=c++17 -x c++ -o \"$d/lat\" \"$d/lat.cu\" -lpthread; "
              f"timeout -s TERM {seconds} \"$d/lat\" serve port={port} cpu=auto host_mib={host_mib} ready=\"$d/ready\" "
              f"{' '.join(f'peer={p}' for p in peers)} & pid=$!; echo \"PID $pid\"; "
              "while [ ! -s \"$d/ready\" ] && kill -0 $pid 2>/dev/null; do sleep 0.2; done; "
              "[ -s \"$d/ready\" ] && echo READY; wait $pid || true")
    p = subprocess.Popen([*SSH, dest, script], stdin=subprocess.PIPE, stdout=open(log_path, "w"), stderr=subprocess.STDOUT)
    p.stdin.write(src)
    p.stdin.close()
    return p


def remote_ready(proc: subprocess.Popen, log_path: str) -> bool:
    """Whether `remote_serve`'s server bound its port: False once the remote script ended without `READY`."""
    while "READY" not in open(log_path).read():
        if proc.poll() is not None:
            return "READY" in open(log_path).read()
        time.sleep(0.5)
    return True


def remote_stop(who: str, log_path: str) -> None:
    """Stop `remote_serve`'s server by the PID it logged."""
    m = re.search(r"^PID (\d+)$", open(log_path).read(), re.M)
    if m:
        subprocess.run([*SSH, REMOTES[who][0], f"kill {m.group(1)} 2>/dev/null || true"], timeout=60)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated variant names (default: all)")
    ns = ap.parse_args()
    only = set(ns.only.split(",")) if ns.only else None
    out = os.environ.get("OUT") or sys.exit("$OUT is unset: run under research run")
    scratch = tempfile.mkdtemp(prefix="latnet-")
    src = open(os.path.join(HERE, "lat.cu"), "rb").read()
    lat = os.path.join(scratch, "lat")
    subprocess.run(["g++", "-O2", "-std=c++17", "-x", "c++", "-o", lat, os.path.join(HERE, "lat.cu"), "-lpthread"], check=True)
    resp_cpu, _chal, _load = plan_cpus()
    ready = os.path.join(scratch, "ready")
    peers = [f"peer={dest.split('@')[1]}:{lport}" for dest, lport in REMOTES.values()]
    responder = subprocess.Popen([lat, "serve", f"port={PORT}", f"cpu={resp_cpu}", f"ready={ready}", "host_mib=1024",
                                  f"stats={out}/responder-stats.json", *peers],
                                 stdout=open(f"{out}/responder.log", "w"), stderr=subprocess.STDOUT)
    while not os.path.exists(ready) or not open(ready).read().strip():
        if responder.poll() is not None:
            sys.exit(f"responder exited {responder.returncode}")
        time.sleep(0.5)
    summary: dict = {"responder_cpu": resp_cpu, "affinity": sorted(os.sched_getaffinity(0)), "nice": os.nice(0), "cells": {}}
    try:
        for who in REMOTES:
            d = os.path.join(out, who)
            os.makedirs(d, exist_ok=True)
            for name, pings, rounds, max_s, extra in VARIANTS:
                if only is not None and name not in only:
                    continue
                a = busy_ticks()
                r = remote_probe(who, src, d, pings, rounds, max_s, ("warm=500", *extra), out=name, port=PORT)
                b = busy_ticks()
                r.update(responder_cpu_busy=busy_frac(a, b, [resp_cpu]), node_busy=busy_frac(a, b, list(b)),
                         loadavg=open("/proc/loadavg").read().strip())
                if os.path.exists(os.path.join(d, f"{name}.bin")):
                    r["summary"] = summarize(os.path.join(d, f"{name}.bin"))
                summary["cells"][f"{who}/{name}"] = r
                log(f"{who}/{name}: {json.dumps(r.get('summary', r))[:300]}")
                json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
        name, pings, rounds, max_s, extra = CONCURRENT
        if only is None or name in only:
            server = remote_serve(src, max_s + 90, os.path.join(out, "node1-responder.log"))
            sampler = open(os.path.join(HERE, "cpusample.py"), "rb").read()
            samplers = [subprocess.Popen([sys.executable, os.path.join(HERE, "cpusample.py"), str(max_s + 40)],
                                         stdout=open(os.path.join(out, "node2-cpus.jsonl"), "w"))]
            for who, (dest, _lport) in REMOTES.items():
                s = subprocess.Popen([*SSH, dest, f"python3 - {max_s + 40}"], stdin=subprocess.PIPE,
                                     stdout=open(os.path.join(out, f"{who}-cpus.jsonl"), "w"))
                s.stdin.write(sampler)
                s.stdin.close()
                samplers.append(s)
            node1_ip = REMOTES["node1"][0].split("@")[1]
            paths = {"cpu1": ("cpu1", {}), "node1": ("node1", {}),
                     "cpu1-to-node1": ("cpu1", {"to": f"{node1_ip}:{CROSS_PORT}", "lport": CROSS_LPORT})}
            results: dict = {}

            def probe(cell: str) -> None:
                who, kw = paths[cell]
                d = os.path.join(out, cell)
                os.makedirs(d, exist_ok=True)
                results[cell] = remote_probe(who, src, d, pings, rounds, max_s, ("warm=500", *extra), out=name, port=PORT,
                                             **kw)

            threads = [threading.Thread(target=probe, args=(cell,)) for cell in paths]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            server.wait()
            for s in samplers:
                s.wait()
            for cell, r in results.items():
                path = os.path.join(out, cell, f"{name}.bin")
                if os.path.exists(path):
                    r["summary"] = summarize(path)
                summary["cells"][f"{cell}/{name}"] = r
                log(f"{cell}/{name}: {json.dumps(r.get('summary', r))[:300]}")
            json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    finally:
        responder.terminate()
        responder.wait(30)
        subprocess.run(["rm", "-rf", scratch])


if __name__ == "__main__":
    main()
