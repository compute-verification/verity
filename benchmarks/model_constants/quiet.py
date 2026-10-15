"""How busy a node's CPUs are, how many GPUs `gpu-lease` has out and who holds the merge-check slots: one JSON line.

    python3 quiet.py [--cpus 0-119] [--slots a,b] [--max 0.03 --sustain 3 --gpus-held 0 --wait-min 30 --exit]

Each sample is a 3 s /proc/stat difference over --cpus, by range (0-63, 64-95, 96-119 and the rest, the slots and
sockets of vy-nebius-2), with `gpu-lease status`'s held count and each --slots lock's holder (slot.py's `holder`,
read from /proc/locks; nothing here takes a lock). With --max it samples until --sustain samples in a row have the
--cpus at most that busy, exactly --gpus-held GPUs held and every slot free, or until --until (a UNIX time; else
--wait-min from now), and says which (`quiet`); --exit then exits 3 when it isn't.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/verity/check"))
from slot import holder  # noqa: E402

RANGES = {"0-63": range(0, 64), "64-95": range(64, 96), "96-119": range(96, 120), "120-191": range(120, 192)}
LOCKS = Path("/workspace/research/locks")


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


def sample(spec: str, slots: list[str]) -> dict:
    a = stat()
    t0 = time.time()
    time.sleep(3)
    b = stat()
    return {"t_unix": round(t0, 3), "busy": busy(a, b, cpus(spec)),
            "by_range": {k: busy(a, b, r) for k, r in RANGES.items()}, "gpus_held": held(),
            "slots": {s: holder(LOCKS / f"check-{s}.lock") for s in slots}}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cpus", default="0-119")
    p.add_argument("--slots", default="a,b")
    p.add_argument("--max", type=float)
    p.add_argument("--sustain", type=int, default=1)
    p.add_argument("--gpus-held", type=int, default=0)
    p.add_argument("--wait-min", type=float, default=30)
    p.add_argument("--until", type=float, help="a UNIX time to wait until instead of --wait-min")
    p.add_argument("--exit", action="store_true")
    ns = p.parse_args()
    slots = [s for s in ns.slots.split(",") if s]
    start, run = time.time(), 0
    until = ns.until if ns.until is not None else start + 60 * ns.wait_min
    while True:
        s = sample(ns.cpus, slots)
        if ns.max is None:
            break
        ok = (s["busy"] is not None and s["busy"] <= ns.max and s["gpus_held"] == ns.gpus_held
              and all(v is None for v in s["slots"].values()))
        run = run + 1 if ok else 0
        s["quiet"] = run >= ns.sustain
        if s["quiet"] or time.time() >= until:
            s["waited_s"] = round(time.time() - start, 1)
            break
    print(json.dumps({"cpus": ns.cpus, **s}), flush=True)
    if ns.exit and ns.max is not None and not s["quiet"]:
        sys.exit(3)


if __name__ == "__main__":
    main()
