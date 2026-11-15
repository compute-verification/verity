"""Each card's prover on its card's NUMA node, for stage_prove.sh.

pin CARDS GPUS_CSV TOPOLOGY_JSON
    GPUS_CSV is `nvidia-smi --query-gpu=index,uuid,pci.bus_id --format=csv,noheader`. Each card's NUMA node is its PCI
    device's numa_node under SYSFS (default /sys), each node's CPUs its cpulist there. Every card gets the same number
    of CPUs, the most each node's share of the job's affinity allows, from its own node's; TOPOLOGY_JSON records the
    job's CPUs, every node's CPUs and each card's index, UUID, bus, node and CPUs. Prints window.py's WIN_CARD_CPUS.
    Exits 1, naming why, on a card with no node (numa_node -1) or a node holding none of the job's CPUs.
check TOPOLOGY_JSON OUT
    Reads each prover group OUT/gpu*/: its card (gpu-lease-usage.jsonl) and its prover's and verifier's CPUs
    (out/host.txt's first line), writes OUT/pinning.tsv a row a group, prints each group off its card's node, and exits
    1 if any is.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def cpulist(text: str) -> list[int]:
    r: list[int] = []
    for part in filter(None, text.strip().split(",")):
        a, _, b = part.partition("-")
        r += range(int(a), int(b or a) + 1)
    return r


def pin(cards: list[str], gpus_csv: str, topology: Path, sysfs: Path, job: list[int]) -> str:
    gpus = {}
    for ln in filter(None, gpus_csv.splitlines()):
        i, u, bus = (x.strip() for x in ln.split(","))
        if i in cards:
            f = sysfs / "bus/pci/devices" / bus.lower()[-12:] / "numa_node"
            if not f.exists():
                raise SystemExit(f"numa.py: card {i}'s bus {bus} has no {f}")
            gpus[i] = {"index": int(i), "uuid": u, "bus": bus, "numa_node": int(f.read_text())}
    nodes = {p.name[4:]: cpulist((p / "cpulist").read_text()) for p in sorted((sysfs / "devices/system/node").glob("node[0-9]*"))}
    by: dict[int, list[str]] = {}
    for c in cards:
        if c not in gpus or gpus[c]["numa_node"] < 0:
            raise SystemExit(f"numa.py: card {c} has no NUMA node in nvidia-smi and {sysfs}")
        by.setdefault(gpus[c]["numa_node"], []).append(c)
    mine = {n: [x for x in nodes.get(str(n), []) if x in job] for n in by}
    share = min(len(mine[n]) // len(cs) for n, cs in by.items())
    if share < 1:
        raise SystemExit(f"numa.py: the job's CPUs {job[0]}-{job[-1]} hold none of some card's node: "
                         + "; ".join(f"node {n} (cards {','.join(cs)}): {len(mine[n])}" for n, cs in by.items()))
    cpus = {c: mine[n][k * share:(k + 1) * share] for n, cs in by.items() for k, c in enumerate(cs)}
    topology.write_text(json.dumps({"job_cpus": job, "nodes": nodes,
                                    "cards": [{**gpus[c], "cpus": cpus[c]} for c in cards]}, indent=1) + "\n")
    return ";".join(f"{c}={','.join(map(str, cpus[c]))}" for c in cards)


def check(topology: Path, out: Path) -> int:
    t = json.loads(topology.read_text())
    node_of = {str(c["index"]): str(c["numa_node"]) for c in t["cards"]}
    rows, off = ["group\tcard\tnuma_node\tprover_cpus\tverifier_cpus\ton_node"], 0
    for g in sorted(out.glob("gpu[0-9]*"), key=lambda p: int(p.name[3:])):
        usage, host = g / "gpu-lease-usage.jsonl", g / "out/host.txt"
        if not (usage.exists() and host.exists()):
            continue
        card = str(json.loads(usage.read_text().splitlines()[0])["gpus"][0]["index"])
        f = dict(kv.partition("=")[::2] for kv in host.read_text().splitlines()[0].split())
        prover, verifier = cpulist(f.get("prover", "")), cpulist(f.get("verifier", ""))
        n = node_of.get(card, "?")
        ok = n != "?" and set(prover + verifier) <= set(t["nodes"][n])
        off += not ok
        rows.append(f"{g.name}\t{card}\t{n}\t{f.get('prover', '')}\t{f.get('verifier', '')}\t{int(ok)}")
        if not ok:
            print(f"{g.name}: card {card} on node {n}, prover CPUs {f.get('prover')} verifier {f.get('verifier')}")
    (out / "pinning.tsv").write_text("\n".join(rows) + "\n")
    return 1 if off else 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["pin"] and len(a) == 4:
        print(pin(a[1].split(","), Path(a[2]).read_text(), Path(a[3]), Path(os.environ.get("SYSFS", "/sys")),
                  sorted(os.sched_getaffinity(0))))
    elif a[:1] == ["check"] and len(a) == 3:
        sys.exit(check(Path(a[1]), Path(a[2])))
    else:
        raise SystemExit(__doc__)
