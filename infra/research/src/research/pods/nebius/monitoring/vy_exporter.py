"""Kueue's queue state and pod-to-queue map as Prometheus metrics, for Grafana's per-queue panels and the idle-GPU alerts.

    python3 vy_exporter.py [--listen ADDR:PORT] [--kubectl CMD] [--ready DIR]

A host service on vy-nebius-1 (host_up.sh installs it), read by the host's own kubectl. SkyPilot's Prometheus finds it through the
selectorless Service in exporter.yaml, so no in-cluster credentials are needed. Kueue's own metrics endpoint wants RBAC for a scrape
token, and kube-state-metrics doesn't carry the queue label. Every scrape reads fresh state (at most every 20 s).

    vy_queue_nominal{queue,resource}   vy_queue_borrowing_limit{queue,resource}   vy_queue_usage{queue,resource}
    vy_queue_borrowed{queue,resource}  vy_queue_pending{queue}   vy_queue_admitted{queue}   vy_queue_stopped{queue}
    vy_pod_queue{holder_namespace,holder_pod,queue} 1      each running pod Kueue admitted, and its queue
    vy_ready_jobs{lane}                                     job files in <ready>/<lane>/ (the dispatcher's ready files)
    vy_exporter_ok                                          1 when the last read of Kueue succeeded

Labels on pods are `holder_*`, not `pod`/`namespace`, which Prometheus's service discovery sets to the exporter's own target.
CPU quantities are in cores, memory in bytes, GPUs as counts. Stdlib only.
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import re
import shlex
import subprocess
import threading
import time
from typing import Any

_UNITS = {"Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40, "k": 1e3, "M": 1e6, "G": 1e9, "T": 1e12, "m": 1e-3}


def quantity(q: Any) -> float:
    m = re.fullmatch(r"([0-9.]+)([A-Za-z]*)", str(q).strip())
    if not m:
        return 0.0
    return float(m.group(1)) * _UNITS.get(m.group(2), 1.0)


def _esc(v: Any) -> str:
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def render(clusterqueues: dict, pods: dict, ready: dict[str, int], ok: bool = True) -> str:
    """The exposition text for ClusterQueue and Pod lists (kubectl -o json) and ready-file counts per lane."""
    rows: dict[str, list[str]] = {}

    def add(name: str, labels: dict[str, Any], value: float) -> None:
        lab = ",".join(f'{k}="{_esc(v)}"' for k, v in labels.items())
        rows.setdefault(name, []).append(f"{name}{{{lab}}} {value:g}")

    for cq in clusterqueues.get("items", []):
        q = cq["metadata"]["name"]
        spec, st = cq.get("spec") or {}, cq.get("status") or {}
        for rg in spec.get("resourceGroups") or []:
            for fl in rg.get("flavors") or []:
                for r in fl.get("resources") or []:
                    add("vy_queue_nominal", {"queue": q, "resource": r["name"]}, quantity(r.get("nominalQuota", 0)))
                    if "borrowingLimit" in r:
                        add("vy_queue_borrowing_limit", {"queue": q, "resource": r["name"]}, quantity(r["borrowingLimit"]))
        for fu in st.get("flavorsUsage") or []:
            for r in fu.get("resources") or []:
                add("vy_queue_usage", {"queue": q, "resource": r["name"]}, quantity(r.get("total", 0)))
                add("vy_queue_borrowed", {"queue": q, "resource": r["name"]}, quantity(r.get("borrowed", 0)))
        add("vy_queue_pending", {"queue": q}, float(st.get("pendingWorkloads") or 0))
        add("vy_queue_admitted", {"queue": q}, float(st.get("admittedWorkloads") or 0))
        add("vy_queue_stopped", {"queue": q}, float((spec.get("stopPolicy") or "None") != "None"))
    for p in pods.get("items", []):
        md = p.get("metadata") or {}
        q = (md.get("labels") or {}).get("kueue.x-k8s.io/queue-name")
        if q and (p.get("status") or {}).get("phase") == "Running":
            add("vy_pod_queue", {"holder_namespace": md.get("namespace", ""), "holder_pod": md.get("name", ""), "queue": q}, 1)
    for lane, n in sorted(ready.items()):
        add("vy_ready_jobs", {"lane": lane}, float(n))
    add("vy_exporter_ok", {}, float(ok))
    return "".join(f"# TYPE {name} gauge\n" + "\n".join(lines) + "\n" for name, lines in rows.items())


def ready_counts(root: str) -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        lanes = sorted(os.listdir(root))
    except OSError:
        return out
    for lane in lanes:
        d = os.path.join(root, lane)
        if os.path.isdir(d):
            out[lane] = sum(1 for f in os.listdir(d) if not f.startswith(".") and os.path.isfile(os.path.join(d, f)))
    return out


class State:
    def __init__(self, kubectl: list[str], ready: str) -> None:
        self.kubectl, self.ready, self.lock = kubectl, ready, threading.Lock()
        self.at, self.text = 0.0, ""

    def get(self, what: list[str]) -> dict:
        r = subprocess.run([*self.kubectl, "get", *what, "-o", "json"], capture_output=True, text=True, timeout=30, check=True)
        return json.loads(r.stdout)

    def metrics(self) -> str:
        with self.lock:
            if time.monotonic() - self.at > 20:
                try:
                    text = render(self.get(["clusterqueues"]), self.get(["pods", "-A", "-l", "kueue.x-k8s.io/queue-name"]),
                                  ready_counts(self.ready))
                except (OSError, subprocess.SubprocessError, ValueError):
                    text = render({}, {}, ready_counts(self.ready), ok=False)
                self.at, self.text = time.monotonic(), text
            return self.text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--listen", default="127.0.0.1:9402")
    ap.add_argument("--kubectl", default="k3s kubectl")
    ap.add_argument("--ready", default="/workspace/jobs/ready")
    a = ap.parse_args()
    state = State(shlex.split(a.kubectl), a.ready)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?")[0] != "/metrics":
                self.send_error(404)
                return
            body = state.metrics().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            pass

    host, port = a.listen.rsplit(":", 1)
    http.server.ThreadingHTTPServer((host, int(port)), Handler).serve_forever()


if __name__ == "__main__":
    main()
