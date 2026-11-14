"""Per-queue usage on the Nebius cluster, for the weekly retune of kueue.yaml's nominal quotas.

    usage_report.py sample [--kubectl CMD]          one JSON line: each ClusterQueue's nominal quota, usage, borrowing, admitted and
                                                    pending workloads, and its pods' mean GPU utilization (DCGM)
    usage_report.py summarize FILE [--days 7]       per queue over the window: mean and p90 usage against nominal, how often work
                                                    waited, how much was borrowed; and the p90 as a suggested nominal

cluster_up.sh installs `sample` on the server as a 5-minute systemd timer appending to /workspace/usage/queues.jsonl, so the record
doesn't depend on any agent VM. GPU utilization comes from the GPU operator's dcgm-exporter (DCGM_FI_DEV_GPU_UTIL, labelled with
the pod that holds the GPU), read through the API server's service proxy. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
import time
from pathlib import Path

RESOURCES = ("nvidia.com/gpu", "cpu", "memory")
DCGM_PROXY = "/api/v1/namespaces/gpu-operator/services/dcgm-exporter:9400/proxy/metrics"
_SUFFIX = {"Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40, "k": 1e3, "M": 1e6, "G": 1e9, "T": 1e12, "m": 1e-3}


def quantity(q: str | int | float) -> float:
    """A Kubernetes quantity as a number: cpu in cores, memory in bytes."""
    s = str(q)
    for suf in sorted(_SUFFIX, key=len, reverse=True):
        if s.endswith(suf):
            return float(s[: -len(suf)]) * _SUFFIX[suf]
    return float(s)


def queues(cqs: dict, workloads: dict, pods: dict, gpu_util: dict[tuple[str, str], float], now: float) -> dict:
    out: dict[str, dict] = {}
    for cq in cqs.get("items", []):
        name = cq["metadata"]["name"]
        nominal = {r["name"]: quantity(r["nominalQuota"]) for g in cq["spec"].get("resourceGroups", []) for f in g["flavors"]
                   for r in f["resources"]}
        usage, borrowed = dict.fromkeys(RESOURCES, 0.0), dict.fromkeys(RESOURCES, 0.0)
        for f in (cq.get("status") or {}).get("flavorsUsage") or []:
            for r in f["resources"]:
                usage[r["name"]] = usage.get(r["name"], 0.0) + quantity(r.get("total", 0))
                borrowed[r["name"]] = borrowed.get(r["name"], 0.0) + quantity(r.get("borrowed", 0))
        out[name] = {"nominal": nominal, "usage": usage, "borrowed": borrowed, "admitted": 0, "pending": 0, "oldest_pending_s": 0.0,
                     "gpu_util": []}
    local_to_cq: dict[tuple[str, str], str] = {}
    for w in workloads.get("items", []):
        conds = {c["type"]: c["status"] for c in (w.get("status") or {}).get("conditions") or []}
        if conds.get("Finished") == "True":
            continue
        cq = ((w.get("status") or {}).get("admission") or {}).get("clusterQueue")
        lq = (w["metadata"]["namespace"], w["spec"].get("queueName", ""))
        if cq:
            local_to_cq[lq] = cq
        if cq and conds.get("Admitted") == "True" and cq in out:
            out[cq]["admitted"] += 1
    for w in workloads.get("items", []):
        conds = {c["type"]: c["status"] for c in (w.get("status") or {}).get("conditions") or []}
        if conds.get("Finished") == "True" or conds.get("QuotaReserved") == "True":
            continue
        cq = local_to_cq.get((w["metadata"]["namespace"], w["spec"].get("queueName", ""))) or w["spec"].get("queueName", "")
        if cq in out:
            out[cq]["pending"] += 1
            created = time.mktime(time.strptime(w["metadata"]["creationTimestamp"], "%Y-%m-%dT%H:%M:%SZ")) - time.timezone
            out[cq]["oldest_pending_s"] = max(out[cq]["oldest_pending_s"], now - created)
    for p in pods.get("items", []):
        lq = (p["metadata"]["namespace"], (p["metadata"].get("labels") or {}).get("kueue.x-k8s.io/queue-name", ""))
        cq = local_to_cq.get(lq, lq[1])
        u = gpu_util.get((p["metadata"]["namespace"], p["metadata"]["name"]))
        if cq in out and u is not None:
            out[cq]["gpu_util"].append(u)
    for q in out.values():
        vals = q.pop("gpu_util")
        q["gpu_util_mean"] = round(sum(vals) / len(vals), 1) if vals else None
    return out


def parse_dcgm(text: str) -> dict[tuple[str, str], float]:
    """{(namespace, pod): mean DCGM_FI_DEV_GPU_UTIL over that pod's GPUs} from the exporter's Prometheus text."""
    per: dict[tuple[str, str], list[float]] = {}
    for line in text.splitlines():
        if not line.startswith("DCGM_FI_DEV_GPU_UTIL{"):
            continue
        labels, _, value = line.partition("} ")
        kv = dict(part.split("=", 1) for part in labels[len("DCGM_FI_DEV_GPU_UTIL{"):].split(",") if "=" in part)
        kv = {k: v.strip('"') for k, v in kv.items()}
        if kv.get("pod"):
            per.setdefault((kv.get("namespace", ""), kv["pod"]), []).append(float(value.split()[0]))
    return {k: sum(v) / len(v) for k, v in per.items()}


def sample(kubectl: list[str], now: float | None = None) -> dict:
    def get(*args: str) -> dict:
        return json.loads(subprocess.run([*kubectl, *args, "-o", "json"], capture_output=True, text=True, check=True).stdout)

    now = time.time() if now is None else now
    r = subprocess.run([*kubectl, "get", "--raw", DCGM_PROXY], capture_output=True, text=True)
    util = parse_dcgm(r.stdout) if r.returncode == 0 else {}
    doc = {"t": round(now), "dcgm": r.returncode == 0,
           "queues": queues(get("get", "clusterqueues"), get("get", "workloads", "-A"), get("get", "pods", "-A"), util, now)}
    return doc


def p90(xs: list[float]) -> float:
    s = sorted(xs)
    return s[max(0, math.ceil(0.9 * len(s)) - 1)] if s else 0.0


def summarize(lines: list[dict], days: float, now: float) -> str:
    window = [d for d in lines if d["t"] >= now - days * 86400]
    if not window:
        return f"no samples in the last {days:g} days"
    names = sorted({q for d in window for q in d["queues"]})
    unit = {"nvidia.com/gpu": ("GPU", 1), "cpu": ("vCPU", 1), "memory": ("GiB", 2**30)}
    rows = [f"{len(window)} samples, {days:g} days to {time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime(now))}",
            f"{'queue':<10} {'resource':<5} {'nominal':>8} {'mean':>8} {'p90':>8} {'borrowed p90':>13} {'suggest':>8}"]
    for n in names:
        qs = [d["queues"][n] for d in window if n in d["queues"]]
        for res, (label, div) in unit.items():
            nominal = qs[-1]["nominal"].get(res, 0.0) / div
            used = [q["usage"].get(res, 0.0) / div for q in qs]
            bor = [q["borrowed"].get(res, 0.0) / div for q in qs]
            suggest = math.ceil(p90(used)) if res != "memory" else math.ceil(p90(used) / 64) * 64
            rows.append(f"{n:<10} {label:<5} {nominal:>8.0f} {sum(used) / len(used):>8.1f} {p90(used):>8.1f} {p90(bor):>13.1f} {suggest:>8}")
        waited = sum(1 for q in qs if q["pending"]) / len(qs)
        util = [q["gpu_util_mean"] for q in qs if q.get("gpu_util_mean") is not None]
        rows.append(f"{n:<10} work waited in {waited:.0%} of samples (longest {max(q['oldest_pending_s'] for q in qs) / 60:.0f} min); "
                    f"GPU utilization of its pods {'%.0f%%' % (sum(util) / len(util)) if util else 'unknown'}")
    rows.append("retune: set each ClusterQueue's nominalQuota in kueue.yaml (keep their sum under the node's allocatable) and "
                "`kubectl apply -f kueue.yaml`")
    return "\n".join(rows)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="usage_report.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--kubectl", default="kubectl", help="the kubectl command line (on the server: 'k3s kubectl')")
    m = sub.add_parser("summarize")
    m.add_argument("file", type=Path)
    m.add_argument("--days", type=float, default=7.0)
    ns = ap.parse_args(argv)
    if ns.cmd == "sample":
        print(json.dumps(sample(shlex.split(ns.kubectl)), separators=(",", ":")))
        return 0
    lines = [json.loads(ln) for ln in ns.file.read_text().splitlines() if ln.strip()]
    print(summarize(lines, ns.days, time.time()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
