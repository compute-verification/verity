"""Sample a machine's CPUs once a second for `seconds` and print one JSON line per sample (stdlib only, so it runs on
any node as `python3 - SECONDS < cpusample.py`): per CPU the cumulative busy and softirq jiffies (/proc/stat), NET_RX
softirqs (/proc/softirqs), each NIC interrupt's count (/proc/interrupts, mlx5 and virtio lines, nonzero CPUs only) and
ksoftirqd/N's utime + stime, which grows when the kernel defers receive processing to a thread that competes with
user work."""

import json
import os
import re
import sys
import time


def ksoftirqd() -> dict[int, str]:
    out = {}
    for pid in os.listdir("/proc"):
        if pid.isdigit():
            try:
                m = re.fullmatch(r"ksoftirqd/(\d+)", open(f"/proc/{pid}/comm").read().strip())
            except OSError:
                continue
            if m:
                out[int(m.group(1))] = f"/proc/{pid}/stat"
    return out


def sample(threads: dict[int, str]) -> dict:
    busy, soft = [], []
    running = 0
    for line in open("/proc/stat"):
        f = line.split()
        if re.fullmatch(r"cpu\d+", f[0]):
            v = list(map(int, f[1:9]))
            busy.append(sum(v) - v[3] - v[4])
            soft.append(v[6])
        elif f[0] == "procs_running":
            running = int(f[1])
    net_rx = []
    for line in open("/proc/softirqs"):
        if line.strip().startswith("NET_RX:"):
            net_rx = list(map(int, line.split()[1:]))
    irqs = {}
    for line in open("/proc/interrupts"):
        if "mlx5" in line or "virtio" in line:
            f = line.split()
            counts = [int(x) for x in f[1:1 + len(busy)]]
            irqs[f"{f[0].rstrip(':')} {f[-1]}"] = {str(c): n for c, n in enumerate(counts) if n}
    kso = {}
    for cpu, path in threads.items():
        try:
            f = open(path).read().rsplit(")", 1)[1].split()
            kso[str(cpu)] = int(f[11]) + int(f[12])
        except OSError:
            pass
    return {"t": time.time(), "busy": busy, "softirq": soft, "net_rx": net_rx, "irqs": irqs, "ksoftirqd": kso,
            "procs_running": running}


def main() -> None:
    seconds = float(sys.argv[1])
    threads = ksoftirqd()
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        print(json.dumps(sample(threads), separators=(",", ":")), flush=True)
        time.sleep(1.0)


if __name__ == "__main__":
    main()
