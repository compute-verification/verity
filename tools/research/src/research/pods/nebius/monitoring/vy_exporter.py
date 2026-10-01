"""Kueue's queue state and pod-to-queue map as Prometheus metrics, for Grafana's per-queue panels and the idle-GPU alerts.

    python3 vy_exporter.py [--listen ADDR:PORT] [--kubectl CMD] [--ready DIR] [--watch FILE]

A host service on vy-nebius-1 (host_up.sh installs it), read by the host's own kubectl. SkyPilot's Prometheus finds it through the
selectorless Service in exporter.yaml, so no in-cluster credentials are needed. Kueue's own metrics endpoint wants RBAC for a scrape
token, and kube-state-metrics doesn't carry the queue label. Every scrape reads fresh state (at most every 20 s).

    vy_queue_nominal{queue,resource}   vy_queue_borrowing_limit{queue,resource}   vy_queue_usage{queue,resource}
    vy_queue_borrowed{queue,resource}  vy_queue_pending{queue}   vy_queue_admitted{queue}   vy_queue_stopped{queue}
    vy_pod_queue{holder_namespace,holder_pod,queue} 1      each running pod Kueue admitted, and its queue
    vy_ready_jobs{lane}                                     job files in <ready>/<lane>/ (the dispatcher's ready files)
    vy_exporter_ok                                          1 when the last read of Kueue succeeded

With --watch, also `cluster watch`'s last findings (tools/cluster's watch.py; node 1's vy-stall-watch timer writes FILE whole):

    vy_job_stalled{node,owner,handle,usergroup,lane,key,name,run,goal,state,why}   a job with no progress for the window, or a
                                            goal job waiting that long for a grant: its seconds without progress (else waiting)
    vy_watch_ok{node,why}                   1 when the watch read the node; 0 with its error (node="registry": the Slack registry)
    vy_watch_watched{node}   vy_watch_unwatched{node}           the jobs it judged, and those it couldn't
    vy_watch_age_seconds                    since the watch ran (+Inf when FILE is missing or unreadable)

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


WATCH_LABELS = ("node", "owner", "handle", "usergroup", "lane", "key", "name", "run", "goal", "state", "why")


def read_watch(path: str) -> dict | None:
    try:
        with open(path) as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) and doc.get("schema") == "cluster/watch/v1" else None


def render_watch(doc: dict | None, now: float) -> str:
    """The exposition text for a `cluster watch --json` document (None: missing or unreadable)."""
    rows: dict[str, list[str]] = {}

    def add(name: str, labels: dict[str, Any], value: float) -> None:
        lab = ",".join(f'{k}="{_esc(v)}"' for k, v in labels.items())
        val = "+Inf" if value == float("inf") else f"{value:g}"
        rows.setdefault(name, []).append(f"{name}{{{lab}}} {val}" if lab else f"{name} {val}")

    if doc is None:
        add("vy_watch_age_seconds", {}, float("inf"))
    else:
        add("vy_watch_age_seconds", {}, max(0.0, now - float(doc.get("t") or 0)))
        if doc.get("error"):
            add("vy_watch_ok", {"node": "registry", "why": str(doc["error"])[:200]}, 0)
        for n in doc.get("nodes") or []:
            node = n.get("node", "?")
            add("vy_watch_ok", {"node": node, "why": str(n.get("error") or "")[:200]}, float(not n.get("error")))
            if n.get("error"):
                continue
            add("vy_watch_watched", {"node": node}, float(n.get("watched") or 0))
            add("vy_watch_unwatched", {"node": node}, float(len(n.get("unwatched") or ())))
            for f in n.get("findings") or []:
                f = {**f, "key": f.get("job")}
                labels = {k: "-" if f.get(k) in (None, "") else f[k] for k in WATCH_LABELS}
                add("vy_job_stalled", labels, float(f.get("quiet_s") or f.get("for_s") or 1))
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
    def __init__(self, kubectl: list[str], ready: str, watch: str | None = None) -> None:
        self.kubectl, self.ready, self.watch, self.lock = kubectl, ready, watch, threading.Lock()
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
            return self.text + (render_watch(read_watch(self.watch), time.time()) if self.watch else "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--listen", default="127.0.0.1:9402")
    ap.add_argument("--kubectl", default="k3s kubectl")
    ap.add_argument("--ready", default="/workspace/jobs/ready")
    ap.add_argument("--watch", default=None, help="cluster watch's --out file (vy-stall-watch.timer writes it)")
    a = ap.parse_args()
    state = State(shlex.split(a.kubectl), a.ready, a.watch)

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
