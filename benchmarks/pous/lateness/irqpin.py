"""Does keeping the receiving CPU awake remove the VPC path's slow state? One CPU job on a GPU node that
drives a second GPU node and a CPU node over ssh; the first node itself carries no traffic.

The guests idle with HLT (no cpuidle driver, no MWAIT), so an idle vCPU's wake-up is the host's. cpu-1's virtio NIC
steers a flow's receive queue to the CPU that sends it, which our busy-polling challenger keeps awake; node 1's mlx5 VF
hashes each UDP 4-tuple to one of 11 channels, whose interrupt is on CPU 96 + channel, idle or not.

1. `lat serve` on cpu-1 (port 47340), peers node 1's candidate source ports.
2. On node 1, with the challenger on one fixed idle CPU, each candidate source port's 2,000 pings, twice from new
   sockets: the channel whose interrupt count rises both times is that port's receive channel (`discovery.json`).
3. Two ports L1, L2 with different receive channels r1, r2. Phase 1 times
   L1 with its challenger on CPU 96 + r1 (`on`) and L2 on the fixed CPU (`off`) at once; phase 2 swaps them. 1 ms-paced
   pings (or `--ops 2`, 4 KiB answers from cpu-1's host memory), `--seconds` per phase. Each machine's CPUs are sampled
   once a second (`cpusample.py`).
Writes `phase<N>/<on|off>.bin|json`, `discovery.json`, `<machine>-cpus.jsonl` and `summary.json` to $OUT."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from job import HERE, REMOTES, SSH, log, remote_probe, summarize  # noqa: E402
from net import remote_ready, remote_serve, remote_stop  # noqa: E402

PORT = 47340
LPORTS = tuple(range(47341, 47357))
NODE1_IRQ_CPU0, NODE1_CHANNELS = 96, 11

DISCOVER = r'''
import json, os, shutil, subprocess, sys, tempfile, time
to, lports, pings = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], int(sys.argv[3])
src = sys.stdin.buffer.read()
d = tempfile.mkdtemp(prefix="lat-irq.")
try:
    open(d + "/lat.cu", "wb").write(src)
    subprocess.run(["g++", "-O2", "-std=c++17", "-x", "c++", "-o", d + "/lat", d + "/lat.cu", "-lpthread"], check=True)
    def stat():
        return {int(l.split()[0][3:]): list(map(int, l.split()[1:9])) for l in open("/proc/stat")
                if l.startswith("cpu") and l[3].isdigit()}
    def irqs():
        lines = open("/proc/interrupts").read().splitlines()
        ncpu = len(lines[0].split())
        return {l.split()[-1].split("@")[0]: sum(int(x) for x in l.split()[1:1 + ncpu]) for l in lines if "mlx5_comp" in l}
    s0 = stat(); time.sleep(2); s1 = stat()
    idle = {c: (s1[c][3] - s0[c][3]) / max(1, sum(s1[c]) - sum(s0[c])) for c in s1}
    irq_cpus = set(range(96, 107))
    fixed = max((c for c in idle if c not in irq_cpus and c != 0), key=lambda c: idle[c])
    res = []
    for rep in range(2):
        for L in lports:
            a = irqs()
            p = subprocess.run([d + "/lat", "probe", "to=" + to, "lport=%d" % L, "cpu=%d" % fixed, "pings=%d" % pings,
                                "rounds=0", "k=1", "gap_us=1000", "warm=50", "max_s=20", "out=" + d + "/disc"],
                               capture_output=True)
            b = irqs()
            res.append({"lport": L, "pass": rep, "rc": p.returncode, "delta": {k: b[k] - a[k] for k in b},
                        "stderr": p.stderr.decode()[-300:]})
    rfs = open("/proc/sys/net/core/rps_sock_flow_entries").read().strip()
    print(json.dumps({"fixed_cpu": fixed, "fixed_idle": idle[fixed], "rps_sock_flow_entries": rfs, "ports": res}))
finally:
    shutil.rmtree(d)
'''


def discover(src: bytes, to: str, pings: int) -> dict:
    cmd = f"python3 -c {shlex.quote(DISCOVER)} {to} {','.join(map(str, LPORTS))} {pings}"
    p = subprocess.run([*SSH, REMOTES["node1"][0], cmd], input=src, capture_output=True, timeout=900)
    if p.returncode:
        sys.exit(f"discovery failed rc={p.returncode}: {p.stderr.decode(errors='replace')[-2000:]}")
    return json.loads(p.stdout.decode().strip().splitlines()[-1])


def receive_channels(disc: dict, pings: int) -> dict[int, int | None]:
    """Each port's receive channel. A port's pings raise two channels' interrupts above the channel's median over all
    ports (the background): its receive channel, fixed by the 4-tuple's RSS hash, and its transmit channel, which each
    new socket draws at random (no XPS map here). Each port is timed twice from new sockets, so its receive channel is
    the one raised both times; None where that is not exactly one channel."""
    chans = sorted({int(k.removeprefix("mlx5_comp")) for p in disc["ports"] for k in p["delta"]})
    def get(p: dict, ch: int) -> int:
        return p["delta"].get(f"mlx5_comp{ch}", 0)
    med = {ch: sorted(get(p, ch) for p in disc["ports"])[len(disc["ports"]) // 2] for ch in chans}
    raised: dict[int, list[set[int]]] = {}
    for p in disc["ports"]:
        raised.setdefault(p["lport"], []).append(
            {ch for ch in chans if get(p, ch) - med[ch] > pings * 3 // 4} if p["rc"] == 0 else set())
    rx: dict[int, int | None] = {}
    for L, sets in raised.items():
        common = set.intersection(*sets) if len(sets) > 1 else set()
        rx[L] = common.pop() if len(common) == 1 else None
    return rx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=180)
    ap.add_argument("--discovery-pings", type=int, default=2000)
    ap.add_argument("--ops", default="3", help="the phases' challenge ops: 3 pings (default), 2 4 KiB host-memory answers")
    ap.add_argument("--dgram", type=int, default=1472,
                    help="answer datagram bytes; 1472 fragments on cpu-1's MTU-1450 link, 1422 does not")
    ns = ap.parse_args()
    out = os.environ.get("OUT") or sys.exit("$OUT is unset: run under research run")
    src = open(os.path.join(HERE, "lat.cu"), "rb").read()
    cpu1_ip, node1_ip = (REMOTES[w][0].split("@")[1] for w in ("cpu1", "node1"))
    to = f"{cpu1_ip}:{PORT}"
    total = 120 + 2 * (ns.seconds + 60)
    server_log = os.path.join(out, "cpu1-responder.log")
    server = remote_serve(src, total, server_log, who="cpu1", port=PORT, peers=tuple(f"{node1_ip}:{L}" for L in LPORTS))
    if not remote_ready(server, server_log):
        sys.exit(f"cpu-1's responder did not start: {open(server_log).read()[-500:]}")
    sampler = open(os.path.join(HERE, "cpusample.py"), "rb").read()
    samplers = []
    for who in REMOTES:
        s = subprocess.Popen([*SSH, REMOTES[who][0], f"python3 - {total}"], stdin=subprocess.PIPE,
                             stdout=open(os.path.join(out, f"{who}-cpus.jsonl"), "w"))
        s.stdin.write(sampler)
        s.stdin.close()
        samplers.append(s)
    summary: dict = {"to": to, "phases": {}}
    try:
        disc = discover(src, to, ns.discovery_pings)
        json.dump(disc, open(os.path.join(out, "discovery.json"), "w"), indent=1)
        fixed = disc["fixed_cpu"]
        rx = receive_channels(disc, ns.discovery_pings)
        log(f"discovery: fixed CPU {fixed}, receive channel by port {rx}")
        picks: list[tuple[int, int]] = []
        for L, ch in rx.items():
            if ch is not None and all(ch != c for _l, c in picks):
                picks.append((L, ch))
            if len(picks) == 2:
                break
        if len(picks) < 2:
            sys.exit(f"no two ports with distinct, unambiguous receive channels: {rx}")
        (l1, r1), (l2, r2) = picks
        summary.update(fixed_cpu=fixed, rx_channel_by_port=rx, ports={"L1": [l1, r1], "L2": [l2, r2]})
        plan = {1: {"on": (l1, NODE1_IRQ_CPU0 + r1), "off": (l2, fixed)},
                2: {"on": (l2, NODE1_IRQ_CPU0 + r2), "off": (l1, fixed)}}
        for phase, cells in plan.items():
            d = os.path.join(out, f"phase{phase}")
            os.makedirs(d, exist_ok=True)
            results: dict = {}

            def probe(role: str) -> None:
                L, cpu = cells[role]
                results[role] = remote_probe("node1", src, d, 0, 1000000, ns.seconds,
                                             ("warm=500", f"ops={ns.ops}", f"dgram={ns.dgram}", "k=1", "gap_us=1000",
                                              f"cpu={cpu}"),
                                             out=role, to=to, lport=L)

            threads = [threading.Thread(target=probe, args=(role,)) for role in cells]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            for role, r in results.items():
                path = os.path.join(d, f"{role}.bin")
                if os.path.exists(path):
                    r["summary"] = summarize(path)
                r["lport"], r["cpu"] = cells[role]
                summary["phases"][f"{phase}/{role}"] = r
                log(f"phase {phase} {role} (port {cells[role][0]}, CPU {cells[role][1]}): "
                    f"{json.dumps(r.get('summary', r))[:300]}")
            json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    finally:
        remote_stop("cpu1", server_log)
        for s in samplers:
            s.terminate()
        server.wait(60)
        for s in samplers:
            s.wait(30)
        json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
