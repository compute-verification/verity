"""How busy a node's CPUs are and how many GPUs `gpu-lease` has out: one JSON line.

    python3 quiet.py [--cpus 0-119] [--max 0.02 --wait-min 30]

Each sample is a 3 s /proc/stat difference over --cpus, by range (0-63, 64-95, 96-119 and the rest, the slots and
sockets of vy-nebius-2), with `gpu-lease status`'s held count. With --max it samples until the --cpus' busy fraction is
at most that and no GPU is held, or --wait-min has passed, and says which (`quiet`).
"""

import argparse
import json
import subprocess
import time

RANGES = {"0-63": range(0, 64), "64-95": range(64, 96), "96-119": range(96, 120), "120-191": range(120, 192)}


def stat() -> dict[int, tuple[int, int]]:
    out = {}
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu") and line[3].isdigit():
                v = [int(x) for x in line.split()[1:]]
                out[int(line.split()[0][3:])] = (sum(v) - v[3] - v[4], sum(v))
    return out


def cpus(spec: str) -> list[int]:
    out: list[int] = []
    for part in filter(None, spec.split(",")):
        lo, _, hi = part.partition("-")
        out += range(int(lo), int(hi or lo) + 1)
    return out


def busy(a: dict, b: dict, members) -> float | None:
    members = [c for c in members if c in a and c in b]
    used = sum(b[c][0] - a[c][0] for c in members)
    total = sum(b[c][1] - a[c][1] for c in members)
    return round(used / total, 4) if total else None


def held() -> int | None:
    try:
        text = subprocess.run(["gpu-lease", "status"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    return sum(1 for line in text.splitlines() if line.split()[1:2] == ["held"])


def sample(spec: str) -> dict:
    a = stat()
    t0 = time.time()
    time.sleep(3)
    b = stat()
    return {"t_unix": round(t0, 3), "busy": busy(a, b, cpus(spec)),
            "by_range": {k: busy(a, b, r) for k, r in RANGES.items()}, "gpus_held": held()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cpus", default="0-119")
    p.add_argument("--max", type=float)
    p.add_argument("--wait-min", type=float, default=30)
    ns = p.parse_args()
    start = time.time()
    while True:
        s = sample(ns.cpus)
        if ns.max is None:
            break
        s["quiet"] = s["busy"] is not None and s["busy"] <= ns.max and s["gpus_held"] == 0
        if s["quiet"] or time.time() - start >= 60 * ns.wait_min:
            s["waited_s"] = round(time.time() - start, 1)
            break
    print(json.dumps({"cpus": ns.cpus, **s}), flush=True)


if __name__ == "__main__":
    main()
