"""Busy fraction of each socket's CPUs, excluding a given list, between two reads of /proc/stat.

    python3 cpuload.py snap > a.json; ...; python3 cpuload.py diff a.json --exclude 48-95

Host DRAM rates on a shared node depend on what else is running; this records how busy the other CPUs were.
"""

import json
import sys
from pathlib import Path


def stat() -> dict[int, list[int]]:
    with open("/proc/stat") as f:
        return {int(l.split()[0][3:]): [int(x) for x in l.split()[1:]] for l in f if l.startswith("cpu") and l[3].isdigit()}


def cpus(spec: str) -> set[int]:
    out: set[int] = set()
    for part in filter(None, spec.split(",")):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def node_cpus() -> dict[int, set[int]]:
    nodes = {}
    for d in sorted(Path("/sys/devices/system/node").glob("node[0-9]*")):
        nodes[int(d.name[4:])] = cpus((d / "cpulist").read_text().strip())
    return nodes


def main() -> None:
    if sys.argv[1] == "snap":
        json.dump(stat(), sys.stdout)
        return
    with open(sys.argv[2]) as f:
        a = {int(k): v for k, v in json.load(f).items()}
    b = stat()
    exclude = cpus(sys.argv[4]) if len(sys.argv) > 4 else set()
    out = {}
    for n, members in node_cpus().items():
        busy = total = 0
        for c in members - exclude:
            d = [y - x for x, y in zip(a[c], b[c])]
            idle = d[3] + d[4]
            busy += sum(d) - idle
            total += sum(d)
        out[f"node{n}_other_busy"] = round(busy / total, 3) if total else None
    print(json.dumps(out))


if __name__ == "__main__":
    main()
