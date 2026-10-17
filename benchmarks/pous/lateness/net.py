"""The two-machine network path alone, as one CPU `research run` on vy-nebius-2: `lat serve` built with g++ (answers from
host memory, no GPU) on a core of the job, then challengers on vy-nebius-cpu-1 and vy-nebius-1 in turn, each in
variants that separate the network's causes: back-to-back pings, pings paced 1 ms and 300 µs apart (adaptive interrupt
moderation follows the packet rate), and 4 KiB answers in 1,472-byte datagrams (fragmented on cpu-1's MTU-1450 path)
and in 1,422-byte ones (whole), back to back and paced. Writes `<challenger>/<variant>.bin|json` and `summary.json` to $OUT."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from job import HERE, REMOTES, busy_ticks, busy_frac, log, plan_cpus, remote_probe, summarize  # noqa: E402

PORT = 47320
# (name, pings, rounds, max_s, extra probe arguments)
VARIANTS = [
    ("ping-b2b", 200000, 0, 60, ("k=111", "gap_us=2000")),
    ("ping-1ms", 40000, 0, 60, ("k=1", "gap_us=1000")),
    ("ping-300us", 60000, 0, 60, ("k=1", "gap_us=300")),
    ("host4k-1472-b2b", 0, 150000, 45, ("ops=2", "dgram=1472")),
    ("host4k-1422-b2b", 0, 150000, 45, ("ops=2", "dgram=1422")),
    ("host4k-1422-1ms", 0, 40000, 50, ("ops=2", "dgram=1422", "k=1", "gap_us=1000")),
]


def main() -> None:
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
    finally:
        responder.terminate()
        responder.wait(30)
        subprocess.run(["rm", "-rf", scratch])


if __name__ == "__main__":
    main()
