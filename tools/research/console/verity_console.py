#!/usr/bin/env python3
"""Verity's panels for the site's live console (`/admin/live`), recomputed from their live sources and sent in the site's panel
format (Project store `internal/live-console/panel-format.md`; the inventory is `internal/live-console/verity-panels.md`).

    verity_console.py --group store [--store DIR] [--research-src DIR] [--out DIR] [--key FILE] [--no-refresh]
    verity_console.py --group node1 [--out DIR] [--key FILE]

- `store`, on the control pod: the evidence store's `check` Attempts of the last day and their `landed` labels. The rest of the
  store's results are the site's Results view, which reads the store's index.
- `node1`, on vy-nebius-1: its Prometheus (DCGM and node-exporter, with the steward's definitions from `util_collect.py`, a copy of
  `internal/lanes/nebius-infra/tools/util_collect.py`) and the Kueue samples `vy-usage` appends to /workspace/usage/queues.jsonl;
  and vy-nebius-2's POUS sampler, read over `ssh vy-n2` and reduced to per-minute rows there.

Every panel is checked against the format (fields, lengths, cell types, 64 KiB) and written to OUT/<name>.json. When the key file
exists (default ~/.config/verity/panels.key) each valid panel is also PUT to the site; without it the run is a dry run. A panel
whose source fails this run is neither written nor sent, so the site keeps the last good one. Prints one JSON line per run; exit 1
when a panel failed, was invalid, or wasn't accepted.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
URL = "https://website-docs-sage.vercel.app"
FIELDS = ("id", "title", "kind", "unit", "y_scale", "target", "target_label", "stacked", "note", "updated_at", "source", "columns", "rows")
REQUIRED = {"id", "title", "kind", "updated_at", "source", "columns", "rows"}
MAX_BODY = 64 * 1024
ID = re.compile(r"[a-z0-9-]+/[a-z0-9-]+")
ISO = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d(:\d\d(\.\d+)?)?(Z|[+-]\d\d:\d\d)")
GPUS, CORES = 8, 192
GPU_TARGET, CPU_TARGET = 60, 70  # Daniel's node-1 targets (docs/gpu-utilization-postmortem.md)
PROM = "/api/v1/namespaces/skypilot/services/skypilot-prometheus-server:80/proxy/api/v1"
QUEUES = Path("/workspace/usage/queues.jsonl")


def iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def isoz(s: str | None) -> str | None:
    """A store timestamp (`…T12:54:39.764Z`) in the format's form, to the second."""
    if not s:
        return None
    try:
        return iso(datetime.fromisoformat(s.replace("Z", "+00:00")))
    except ValueError:
        return None


def sig(v, n: int = 4):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return float(f"{v:.{n}g}")


def ratio(a, b, n: int = 3):
    return sig(a / b, n) if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b else None


def panel(name: str, title: str, kind: str, columns: list, rows: list, *, source: str, updated_at: str, unit: str | None = None,
          note: str | None = None, y_scale: str | None = None, producer: str = "verity", target: float | None = None,
          target_label: str | None = None, stacked: bool | None = None) -> dict:
    p = {"id": f"{producer}/{name}", "title": title, "kind": kind, "unit": unit, "y_scale": y_scale, "target": target,
         "target_label": target_label, "stacked": stacked, "note": note,
         "updated_at": updated_at, "source": source, "columns": columns, "rows": rows}
    return {k: p[k] for k in FIELDS if p[k] is not None}


def problems(p: dict) -> list[str]:
    """Everything the site's validator would refuse (panel-format.md §1)."""
    out = [f"unknown field `{k}`" for k in p if k not in FIELDS]
    out += [f"missing `{k}`" for k in sorted(REQUIRED - set(p))]
    if out:
        return out

    def text(k: str, hi: int) -> None:
        if k in p and not (isinstance(p[k], str) and 1 <= len(p[k]) <= hi):
            out.append(f"`{k}` must be 1 to {hi} characters")

    if not (isinstance(p["id"], str) and ID.fullmatch(p["id"])):
        out.append("`id` must be producer/name in lowercase letters, digits and dashes")
    text("title", 120), text("unit", 20), text("note", 500), text("source", 300)
    if p["kind"] not in ("table", "line", "bar"):
        out.append("`kind` must be table, line or bar")
    if "y_scale" in p and (p["y_scale"] not in ("linear", "log") or p["kind"] == "table"):
        out.append("`y_scale` must be linear or log, on a line or bar chart")
    if "target" in p and (p["kind"] == "table" or isinstance(p["target"], bool) or not isinstance(p["target"], (int, float))):
        out.append("`target` must be a number, on a line or bar chart")
    if "target_label" in p and ("target" not in p or not (isinstance(p["target_label"], str) and 1 <= len(p["target_label"]) <= 40)):
        out.append("`target_label` must be 1 to 40 characters, with a target")
    if "stacked" in p and (p["kind"] != "bar" or not isinstance(p["stacked"], bool)):
        out.append("`stacked` must be true or false, on a bar chart")
    if not (isinstance(p["updated_at"], str) and ISO.fullmatch(p["updated_at"])):
        out.append("`updated_at` must be ISO 8601 with a zone")
    cols, rows = p["columns"], p["rows"]
    if not (isinstance(cols, list) and 1 <= len(cols) <= 20 and all(isinstance(c, str) and 1 <= len(c) <= 60 for c in cols)
            and len(set(cols)) == len(cols)):
        out.append("`columns` must be 1 to 20 distinct names of up to 60 characters")
    elif p["kind"] != "table" and len(cols) < 2:
        out.append("a chart needs at least 2 columns")
    if not isinstance(rows, list) or len(rows) > 1000:
        out.append("`rows` must be a list of at most 1000 rows")
    else:
        for i, r in enumerate(rows):
            if not isinstance(r, list) or len(r) != len(cols):
                out.append(f"row {i} has {len(r) if isinstance(r, list) else '?'} cells, not {len(cols)}")
                continue
            for j, c in enumerate(r):
                ok = c is None or (isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c)) or (
                    isinstance(c, str) and len(c) <= 200 and (j == 0 or p["kind"] == "table"))
                if not ok:
                    out.append(f"row {i} column {j}: {'a series cell must be a number or null' if isinstance(c, str) else 'bad cell'}")
    size = len(json.dumps(p).encode())
    if size > MAX_BODY:
        out.append(f"{size} bytes, over 64 KiB")
    return out


def fit(p: dict) -> dict:
    """A table over 64 KiB: its long strings cut shorter, then its rows cut, saying so in `note`."""
    if p["kind"] != "table":
        return p
    for cap in (200, 140, 100, 70):
        if len(json.dumps(p).encode()) <= MAX_BODY:
            return p
        p["rows"] = [[c[:cap - 1] + "…" if isinstance(c, str) and len(c) > cap else c for c in r] for r in p["rows"]]
    n, note = len(p["rows"]), p.get("note", "")
    while len(json.dumps(p).encode()) > MAX_BODY and p["rows"]:
        p["rows"] = p["rows"][:int(len(p["rows"]) * 0.9)]
        p["note"] = (f"The first {len(p['rows'])} of {n} rows (the site's 64 KiB limit). " + note)[:500]
    return p


def publish(p: dict, url: str, key: str) -> str:
    req = urllib.request.Request(f"{url}/api/panels/{p['id']}", data=json.dumps(p).encode(), method="PUT",
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return str(r.status)
        except urllib.error.HTTPError as e:
            if e.code != 503 or attempt == 2:
                return f"{e.code} {e.read()[:300].decode(errors='replace')}"
        except (urllib.error.URLError, OSError) as e:
            if attempt == 2:
                return f"unreachable: {e}"
        time.sleep(5)
    return "unreachable"


def write(out: Path, p: dict) -> None:
    dest = out / f"{p['id'].split('/', 1)[1]}.json"
    tmp = dest.with_name(f".{dest.name}.tmp")
    tmp.write_text(json.dumps(p, indent=1))
    os.replace(tmp, dest)


# ---------------------------------------------------------------------------------------------------------------- store group

def store_panels(ns, now: datetime, errors: list) -> tuple[list[dict], dict]:
    sys.path[:0] = [str(ns.research_src)]
    from research.store.local import LocalStore  # noqa: PLC0415

    store = LocalStore(ns.store) if ns.store else LocalStore()
    info = {"refresh": "skipped (--no-refresh)"}
    if not ns.no_refresh:
        try:
            rep = store.refresh(jobs=8)
            info["refresh"] = f"+{rep['attempts']} attempts, +{rep['labels']} labels, {len(rep['errors'])} errors, {rep['seconds']:.0f} s"
        except Exception as e:  # noqa: BLE001  a stale catalog still renders; the run line says so
            info["refresh"] = f"failed: {type(e).__name__}: {e}"
    try:
        return [check_panel(store.index, now)], info
    except Exception as e:  # noqa: BLE001  the run line names it
        errors.append(f"merge-checks: {type(e).__name__}: {e}")
        return [], info


def machine_of(doc: dict) -> str | None:
    ex = doc.get("execution") or {}
    return ex.get("machine") or (ex.get("remote") or {}).get("machine") or ex.get("host")


def check_panel(idx, now: datetime) -> dict:
    """Every `check` Attempt of the last 24 hours, with the main commit that landed it from its `landed` label (`research merge`)."""
    since = iso(now - timedelta(hours=24))
    landed = dict(idx.sql("SELECT target, value FROM labels WHERE key = 'landed' AND by = 'research-merge' ORDER BY ts"))
    rows, ends = [], []
    for run, start, wall, rc, state, commit, mj in idx.sql(
            "SELECT id, start_utc, wall_s, rc, state, source_commit, manifest_json FROM attempts WHERE tool = 'check' AND start_utc >= ? "
            "ORDER BY start_utc DESC", (since,)):
        result = "pass" if state == "done" and rc == 0 else state if state in ("running", "cancelled") else f"fail (rc {rc})"
        rows.append([run, isoz(start), sig(wall / 60, 3) if wall else None, result, (commit or "")[:8] or None,
                     (landed.get(run) or "")[:9] or None, machine_of(json.loads(mj)) if mj else None])
        if start and wall:
            ends.append(datetime.fromisoformat(start.replace("Z", "+00:00")) + timedelta(seconds=wall))
    return panel("merge-checks", "Every `check` in the last 24 hours", "table",
                 ["Run", "Started (UTC)", "Minutes", "Result", "Commit", "Landed as", "Machine"], rows,
                 updated_at=iso(max(ends)) if ends else iso(now), source="evidence store: tool=check Attempts and their landed labels",
                 note="Landed as is main's commit whose Check: trailer names the run (its `landed` label); the others are checks of "
                      "branches or of trains that didn't land. Attempts are published when a check ends, so one in progress isn't listed.")


# ---------------------------------------------------------------------------------------------------------------- node1 group

def kraw(path: str) -> dict:
    env = dict(os.environ, KUBECONFIG=os.environ.get("KUBECONFIG", str(Path.home() / ".kube" / "config")))
    return json.loads(subprocess.run(["kubectl", "get", "--raw", path], check=True, capture_output=True, text=True, timeout=120,
                                     env=env).stdout)


def prom_range(query: str, start: float, end: float, step: int = 60) -> list[dict]:
    qs = urllib.parse.urlencode({"query": query, "start": int(start), "end": int(end), "step": step})
    return kraw(f"{PROM}/query_range?{qs}")["data"]["result"]


def prom_now(query: str) -> list[dict]:
    return kraw(f"{PROM}/query?{urllib.parse.urlencode({'query': query})}")["data"]["result"]


def node1_panels(ns, now: datetime, errors: list) -> tuple[list[dict], dict]:
    sys.path.insert(0, str(HERE))
    import util_collect as uc  # noqa: PLC0415  the steward's utilization definitions

    end = now.timestamp()
    start = (now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=23)).timestamp()
    prom = {k: prom_range(q, start, end) for k, q in uc.Q.items()}
    qlines = [json.loads(ln) for ln in QUEUES.read_text().splitlines() if ln.strip().startswith("{")] if QUEUES.exists() else []
    src = "vy-nebius-1: Prometheus (DCGM, node-exporter) and Kueue samples (vy-usage, /workspace/usage/queues.jsonl)"
    out = []

    summary = uc.summarize_n1(prom, start, end)
    alloc = uc.kueue_allocated(qlines, start)
    hours = summary["hours"]
    cur = uc.hour(end)
    hrows = []
    for h in sorted(hours, reverse=True):
        b = hours[h]
        cpu = 100 * b.get("cpu_core_h_busy", 0) / (b["cpu_h_observed"] * CORES) if b.get("cpu_h_observed") else None
        hrows.append([h + (" (so far)" if h == cur else ""), sig(b.get("gpu_h_observed"), 3), sig(alloc.get(h, 0.0), 3) if qlines else None,
                      sig(b.get("gpu_h_held_or_busy"), 3), sig(b.get("gpu_h_busy"), 3), sig(b.get("gpu_h_effective"), 3),
                      sig(b.get("gpu_h_idle"), 3), sig(cpu, 3), sig(b.get("ram_peak_gib"), 3)])
    out.append(panel("node1-hours", "vy-nebius-1 per hour: GPU-hours and CPU", "table",
                     ["Hour (UTC)", "GPU-h", "Kueue-allocated", "Held or busy", "Busy", "Util-weighted", "Idle", "CPU busy %",
                      "RAM peak GiB"], hrows, updated_at=iso(now), source=src,
                     note="The steward's definitions: busy = DCGM utilization above 0 in a minute; held = 1 GiB or more of GPU memory "
                          "in use; util-weighted = the sum of utilization; idle = GPU-hours neither held nor busy. Kueue-allocated is "
                          "GPU-hours admitted to jobs, from 5-minute samples. The last 24 hours, newest first."))

    tot = summary["total"]
    o, b, h = tot.get("gpu_h_observed", 0), tot.get("gpu_h_busy", 0), tot.get("gpu_h_held_or_busy", 0)
    a = sum(alloc.values())
    split = [("Busy", b), ("GPU memory held, not busy", h - b), ("Allocated by Kueue, no GPU memory held", max(a - h, 0)),
             ("Neither allocated nor held", o - max(a, h))]
    srows = [[k, sig(v, 3), sig(100 * v / o, 3) if o else None] for k, v in split] + [["All GPU-hours", sig(o, 3), 100.0 if o else None]]
    out.append(panel("node1-split", "vy-nebius-1: where the last 24 hours of GPU-hours went", "table",
                     ["Where the GPU-hours went", "GPU-h", "Share %"], srows, updated_at=iso(now), source=src,
                     note=f"Since {uc.hour(start)}, the postmortem's split. Held counts any process on a GPU, inside Kueue or not. "
                          "Busy is generous: utilization above 0 in a minute."))

    util = {r["metric"].get("gpu"): {int(float(t)): float(v) for t, v in r["values"]} for r in prom["gpu_util"]}
    fb = {r["metric"].get("gpu"): {int(float(t)): float(v) for t, v in r["values"]} for r in prom["gpu_fb_mib"]}
    cpu = {int(float(t)): float(v) for r in prom["cpu_busy"] for t, v in r["values"]}
    buckets: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for g, ser in util.items():
        for t, u in ser.items():
            bk = buckets[t // 300 * 300]
            bk["busy"].append(100.0 * (u > 0))
            bk["util"].append(u)
            bk["held"].append(100.0 * (u > 0 or fb.get(g, {}).get(t, 0.0) >= 1024))
    for t, c in cpu.items():
        buckets[t // 300 * 300]["cpu"].append(100 * c / CORES)
    for d in qlines:
        if d.get("t", 0) >= start:
            gpus = sum(q.get("usage", {}).get("nvidia.com/gpu", 0.0) for q in d.get("queues", {}).values())
            buckets[int(d["t"]) // 300 * 300]["alloc"].append(100 * gpus / GPUS)
    mean = lambda xs: sig(sum(xs) / len(xs), 3) if xs else None  # noqa: E731
    urows = [[iso(datetime.fromtimestamp(t, timezone.utc)), mean(v["busy"]), mean(v["util"]), mean(v["held"]), mean(v["alloc"]),
              mean(v["cpu"]), GPU_TARGET, CPU_TARGET] for t, v in sorted(buckets.items())]
    out.append(panel("node1-utilization", "vy-nebius-1 utilization, last 24 hours", "line",
                     ["time", "GPU busy %", "GPU util %", "GPU held %", "Kueue-allocated %", "CPU busy %", "GPU busy target %",
                      "CPU busy target %"], urows, unit="%", updated_at=iso(now), source=src,
                     note="5-minute means over the 8 GPUs and 192 vCPUs. GPU busy: utilization above 0 (the steward's measure, and "
                          "Daniel's 60% target); GPU util: mean utilization; held: 1 GiB or more of memory in use; Kueue-allocated: "
                          "GPUs admitted to jobs. CPU busy: non-idle, non-iowait cores (target 70%)."))

    if qlines:
        last = qlines[-1]
        qrows = []
        for q, v in sorted(last.get("queues", {}).items()):
            n, u, bo = v.get("nominal", {}), v.get("usage", {}), v.get("borrowed", {})
            gi = lambda d: sig(d.get("memory", 0) / 2**30, 3)  # noqa: E731
            qrows.append([q, u.get("nvidia.com/gpu"), n.get("nvidia.com/gpu"), bo.get("nvidia.com/gpu"), u.get("cpu"), n.get("cpu"),
                          gi(u), gi(n), v.get("admitted"), v.get("pending"), sig((v.get("oldest_pending_s") or 0) / 60, 3),
                          sig(v.get("gpu_util_mean"), 3)])
        out.append(panel("node1-queues", "vy-nebius-1 Kueue queues now", "table",
                         ["Queue", "GPUs used", "GPUs nominal", "GPUs borrowed", "CPUs used", "CPUs nominal", "Memory used GiB",
                          "Memory nominal GiB", "Admitted", "Pending", "Oldest pending (min)", "Pods' mean GPU util %"], qrows,
                         updated_at=iso(datetime.fromtimestamp(last["t"], timezone.utc)), source=src,
                         note="Each ClusterQueue's quota and use from vy-usage's latest sample (every 5 minutes)."))
    else:
        errors.append(f"node1-queues: no samples in {QUEUES}")

    try:
        pods = kraw("/api/v1/pods")["items"]
        queue_of = {(p["metadata"]["namespace"], p["metadata"]["name"]): (p["metadata"].get("labels") or {}).get("kueue.x-k8s.io/queue-name")
                    for p in pods}
    except Exception as e:  # noqa: BLE001  the GPU table goes out without queue names
        queue_of = {}
        errors.append(f"node1-gpus: pod list: {type(e).__name__}: {e}")
    now_q = {"util": "DCGM_FI_DEV_GPU_UTIL", "fb": "DCGM_FI_DEV_FB_USED", "engine": "DCGM_FI_PROF_GR_ENGINE_ACTIVE",
             "power": "DCGM_FI_DEV_POWER_USAGE"}
    gpu: dict[str, dict] = defaultdict(dict)
    for k, q in now_q.items():
        for r in prom_now(q):
            m = r["metric"]
            d = gpu[m.get("gpu", "?")]
            d[k] = float(r["value"][1])
            if m.get("pod"):
                d["pod"], d["ns"] = m["pod"], m.get("namespace")
    grows = []
    for g in sorted(gpu, key=lambda x: int(x) if x.isdigit() else 99):
        d = gpu[g]
        holder = d.get("pod") or ("(no pod: direct or leaked)" if d.get("fb", 0) >= 1024 else None)
        grows.append([int(g) if g.isdigit() else g, sig(d.get("util"), 3), sig(100 * d["engine"], 3) if "engine" in d else None,
                      sig(d.get("fb", 0) / 1024, 3), sig(d.get("power"), 3), holder, queue_of.get((d.get("ns"), d.get("pod")))])
    out.append(panel("node1-gpus", "vy-nebius-1 GPUs now", "table",
                     ["GPU", "Util %", "Engine active %", "Memory GiB", "Power W", "Held by pod", "Kueue queue"], grows,
                     updated_at=iso(now), source=src,
                     note="Each GPU's latest DCGM reading (1-minute scrapes) and the pod DCGM names as holding it. Engine active is "
                          "DCGM_FI_PROF_GR_ENGINE_ACTIVE, closer to real SM occupancy than utilization."))
    mstart = (now - timedelta(hours=MINUTES_H)).timestamp()
    minutes: dict[str, list[tuple] | None] = {"n1": None, "n2": None}
    try:
        minutes["n1"] = n1_minutes(mstart, end)
        out += owner_panels("node1", "vy-nebius-1", minutes["n1"], now, src.split(" and Kueue")[0] + ", DCGM's pod labels")
    except Exception as e:  # noqa: BLE001  the other panels still go out
        errors.append(f"node1-owners: {type(e).__name__}: {e}")
    try:
        minutes["n2"] = n2_minutes(mstart)
        out.append(hours_panel("node2", "vy-nebius-2", minutes["n2"], now,
                               f"vy-nebius-2: POUS's sampler ({N2_UTIL}), read over ssh from vy-nebius-1"))
    except Exception as e:  # noqa: BLE001  the other panels still go out
        errors.append(f"node2-hours: {type(e).__name__}: {e}")
    try:
        read = {"node 1": minutes["n1"], "node 2": minutes["n2"]}
        try:
            spans = {"node 1": n1_lease_spans(now.timestamp())}
        except Exception as e:  # noqa: BLE001  node 1's hours wait for a pass that knows its leases; node 2's go on
            errors.append(f"held-idle node 1: {type(e).__name__}: {e}")
            read["node 1"], spans = None, {}
        rows = usage_rows(read, spans)
        readings = idle_readings(rows, now)
        try:
            add_phases(readings, now)
        except Exception as e:  # noqa: BLE001  the readings still go out, without phases
            errors.append(f"held-idle phases: {type(e).__name__}: {e}")
        if ns.key.exists():
            record_usage(rows, readings, now, mstart)
        out.append(idle_panel(readings, now))
    except Exception as e:  # noqa: BLE001  the other panels still go out
        errors.append(f"held-idle: {type(e).__name__}: {e}")
    try:
        pool = json.loads(POOL.read_text()) if POOL.exists() else {}
        out += pool_panels(pool, n1_from_prom(summary, urows, grows, now), now, errors, minutes)
    except Exception as e:  # noqa: BLE001  the node-1 panels still go out
        errors.append(f"pool: {type(e).__name__}: {e}")
    try:
        out += hillclimb_panels(now, errors)
    except Exception as e:  # noqa: BLE001  the node-1 panels still go out
        errors.append(f"hillclimb: {type(e).__name__}: {e}")
    try:
        out += dot_overhead_panels(now, errors) + vllm_overhead_panels(now, errors)
    except Exception as e:  # noqa: BLE001  the node-1 panels still go out
        errors.append(f"prover-overhead: {type(e).__name__}: {e}")
    try:
        out += overnight_panels(errors)
    except Exception as e:  # noqa: BLE001  the node-1 panels still go out
        errors.append(f"overnight: {type(e).__name__}: {e}")
    return out, {"queue_samples": len(qlines), "pool_file": POOL.exists(),
                 "hillclimb_files": len(list(HILLCLIMB.glob("*.json"))) if HILLCLIMB.is_dir() else 0,
                 "overnight_files": len(list(OVERNIGHT.glob("*.json"))) if OVERNIGHT.is_dir() else 0}


# ---------------------------------------------------------------------------------------------------------------- proofs hill climb

HILLCLIMB = Path("/workspace/usage/hillclimb")  # proofs' roll-ups, one `verity/hillclimb-rollup/v0` file per subcircuit
SUBCIRCUIT = re.compile(r"[a-z0-9]+/[A-Za-z0-9_]+/K=\d+")
HILLCLIMB_COLUMNS = ["Step", "Label", "Commit", "Run", "Time (UTC)", "Overhead", "Throughput VU/s", "GPU util", "GPU-held s/VU",
                     "Byte-identical", "Flags", "VUs", "Session s/VU", "Setup s", "Verify s/statement", "Scheme", "GPU",
                     "vCPU", "Dot", "Prove-only s/VU"]


def hillclimb_panels(now: datetime, errors: list) -> list[dict]:
    """One table per subcircuit, a row per optimization step. The title is the subcircuit id, which the site parses."""
    out = []
    for f in sorted(HILLCLIMB.glob("*.json")) if HILLCLIMB.is_dir() else []:
        try:
            doc = json.loads(f.read_text())
            sub = doc["subcircuit"]
            sid = sub["id"]
            if doc.get("schema") != "verity/hillclimb-rollup/v0" or not SUBCIRCUIT.fullmatch(sid):
                raise ValueError(f"schema {doc.get('schema')!r}, subcircuit {sid!r}")
            text = lambda v, n=200: str(v)[:n] if v not in (None, "") else None  # noqa: E731
            rows = []
            for p in sorted(doc.get("points") or [], key=lambda p: p.get("step", 0)):
                hw = p.get("hardware") or {}
                rows.append([p.get("step"), text(p.get("label")), text(p.get("commit"), 12), text(p.get("run_id")),
                             isoz(p.get("time_utc")), sig(p.get("overhead")), sig(p.get("throughput_vu_per_s")),
                             sig(p.get("gpu_util")), sig(p.get("gpu_held_s_per_vu")),
                             {True: "yes", False: "no"}.get(p.get("byte_identical")), text(" ".join(p.get("flags") or [])),
                             p.get("vus") if isinstance(p.get("vus"), int) else None, sig(p.get("t_prove_s_per_vu")),
                             sig(p.get("setup_s")), sig(p.get("verify_s_per_statement")), text(p.get("scheme"), 60),
                             text(hw.get("gpu")), hw.get("vcpu") if isinstance(hw.get("vcpu"), int) else None,
                             text(sub.get("dot")), sig(p.get("t_prove_only_s_per_vu"))])
            name = "hillclimb-" + re.sub(r"[^a-z0-9]+", "-", sid.lower()).strip("-")
            when = isoz(doc.get("updated_utc")) or iso(datetime.fromtimestamp(f.stat().st_mtime, timezone.utc))
            out.append(fit(panel(name, sid, "table", HILLCLIMB_COLUMNS, rows, updated_at=when,
                                 source=f"proofs' hill-climb roll-up (vy-nebius-1 {f}), from hillclimb.json records",
                                 note="overhead = datasheet dense peak ÷ (2·K·VUs ÷ session time), the session being the whole interactive "
                                      "proof job, verifier round trips included (proofs, 4:47 PM PDT Sep 30). Prove-only s/VU is the "
                                      "prover buckets alone. One row per step.")))
        except Exception as e:  # noqa: BLE001  one bad file doesn't stop the others
            errors.append(f"hillclimb {f.name}: {type(e).__name__}: {e}")
    return out


# ------------------------------------------------------------------------------------------------- private-value prover overhead

DOT_DTYPES = ("bf16", "e4m3", "nvf4", "mxf4")
DOT_KS = (2048, 4096, 8192, 16384)
N1_STORE = Path("/workspace/research/store/catalog.sqlite")  # node 1's evidence-store catalog, read as it is
VLLM_CAMPAIGN = "private-overheads-oct2"


def point_mode(p: dict) -> str:
    """A hill-climb point's prover: `zk` (private values, `--zk`) or `m0`, the default when the point doesn't say."""
    return "zk" if str(p.get("mode") or "").strip().lstrip("-").lower() == "zk" else "m0"


def dot_points(errors: list) -> dict[tuple[str, int], dict[str, list[dict]]]:
    """Every byte-identical point with an overhead, per (dtype, K) cell and mode, oldest first."""
    cells: dict[tuple[str, int], dict[str, list[dict]]] = defaultdict(lambda: {"zk": [], "m0": []})
    for f in sorted(HILLCLIMB.glob("*.json")) if HILLCLIMB.is_dir() else []:
        try:
            doc = json.loads(f.read_text())
            sub = doc["subcircuit"]
            key = (str(sub.get("dtype") or "").lower(), sub.get("K"))
            for p in doc.get("points") or []:
                if sig(p.get("overhead")) is None or p.get("byte_identical") is False or not isoz(p.get("time_utc")):
                    continue
                cells[key][point_mode(p)].append(p)
        except Exception as e:  # noqa: BLE001  one bad file doesn't stop the others
            errors.append(f"prover-overhead {f.name}: {type(e).__name__}: {e}")
    for modes in cells.values():
        for ps in modes.values():
            ps.sort(key=lambda p: isoz(p.get("time_utc")))
    return cells


def dot_overhead_panels(now: datetime, errors: list) -> list[dict]:
    cells = dot_points(errors)
    src = "proofs' hill-climb roll-ups (vy-nebius-1 /workspace/usage/hillclimb), 74-gemm-hill.sh points from hillclimb.json"
    rows, when = [], []
    for d in DOT_DTYPES:
        for k in DOT_KS:
            zk, m0 = cells.get((d, k), {}).get("zk", []), cells.get((d, k), {}).get("m0", [])
            zl, ml = (zk[-1] if zk else None), (m0[-1] if m0 else None)
            best = lambda ps: min((p["overhead"] for p in ps), default=None)  # noqa: E731
            last = max([isoz(p["time_utc"]) for p in (zl, ml) if p], default=None)
            when += [last] if last else []
            rows.append([d.upper(), k, sig(zl and zl["overhead"]), sig(best(zk)), zl.get("step") if zl else None,
                         sig(ml and ml["overhead"]), sig(best(m0)), ratio(zl and zl["overhead"], ml and ml["overhead"]),
                         len(zk), len(m0), last])
    any_zk = any(r[8] for r in rows)
    note = ("Overhead = census dense peak ÷ (2·K·VUs ÷ session time), the session being the whole interactive proof job. "
            "Latest = the cell's newest byte-identical point of that prover; best = its lowest. ZK ÷ M0 compares the two latests. "
            "A point is ZK when its record says mode zk; otherwise M0.")
    if not any_zk:
        note = "No --zk points yet. " + note
    out = [panel("prover-overhead-dot-products", "Private-value (--zk) prover overhead per dot-product cell, M0 beside it", "table",
                 ["Dtype", "K", "ZK latest (×)", "ZK best (×)", "ZK step", "M0 latest (×)", "M0 best (×)", "ZK ÷ M0",
                  "ZK points", "M0 points", "Latest (UTC)"],
                 rows, updated_at=max(when, default=None) or iso(now), source=src, note=note[:500])]
    series = [(f"{d.upper()} K={k}", cells[(d, k)]["zk"]) for d in DOT_DTYPES for k in DOT_KS if cells.get((d, k), {}).get("zk")]
    times = sorted({isoz(p["time_utc"]) for _, ps in series for p in ps})
    by = [{isoz(p["time_utc"]): sig(p["overhead"]) for p in ps} for _, ps in series]
    out.append(panel("prover-overhead-dot-products-steps", "Private-value (--zk) prover overhead per cell, by hill-climb step",
                     "line", ["Time (UTC)", *[n for n, _ in series]] if series else ["Time (UTC)", "no --zk points yet"],
                     [[t, *[b.get(t) for b in by]] for t in times], unit="× peak", y_scale="log",
                     updated_at=times[-1] if times else iso(now), source=src,
                     note="Each --zk hill-climb point at the time it was measured, one line per cell; lower is better; log scale."))
    return out


def vllm_points(path: Path | None = None) -> list[dict]:
    """The vLLM proof-unit-weighted overhead points: attempts labelled `ov.ws prover`, `ov.metric overhead` and an `ov.config`
    of `vllm/<model>/<all|prefill|decode>/<zk|m0>`; the latest assertion of each key wins."""
    path = path or N1_STORE
    if not path.exists():
        return []
    import sqlite3  # noqa: PLC0415
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    try:
        rows = db.execute("SELECT target, key, value, ts FROM labels WHERE target IN "
                          "(SELECT target FROM labels WHERE key = 'ov.config' AND value LIKE 'vllm/%') "
                          "AND (key LIKE 'ov.%' OR key = 'campaign') ORDER BY ts").fetchall()
    finally:
        db.close()
    by: dict[str, dict] = defaultdict(dict)
    for target, key, value, ts in rows:
        by[target][key] = value
        by[target]["_ts"] = ts
    out = []
    for target, v in by.items():
        try:
            value = float(v.get("ov.value"))
        except (TypeError, ValueError):
            continue
        if v.get("ov.ws") != "prover" or v.get("ov.metric") != "overhead" or v.get("campaign", VLLM_CAMPAIGN) != VLLM_CAMPAIGN:
            continue
        try:
            attempt = int(v.get("ov.attempt") or 0)
        except ValueError:
            attempt = 0
        out.append({"target": target, "config": v["ov.config"], "attempt": attempt, "value": value,
                    "line": v.get("ov.line"), "when": isoz(v["_ts"])})
    return out


def vllm_overhead_panels(now: datetime, errors: list) -> list[dict]:
    try:
        pts = vllm_points()
    except Exception as e:  # noqa: BLE001  an empty panel still goes out
        errors.append(f"prover-overhead-vllm: {type(e).__name__}: {e}")
        pts = []
    src = f"node 1's evidence store: campaign {VLLM_CAMPAIGN}, ov.ws prover, ov.metric overhead, ov.config vllm/*"
    configs = sorted({p["config"] for p in pts})[:19]
    latest: dict[tuple[str, int], dict] = {}
    for p in sorted(pts, key=lambda p: p["when"] or ""):
        latest[(p["config"], p["attempt"])] = p
    attempts = sorted({a for _, a in latest})
    when = max((p["when"] for p in pts if p["when"]), default=None) or iso(now)
    out = [panel("prover-overhead-vllm", "vLLM prover overhead, weighted over every proof unit of the run", "line",
                 ["attempt", *configs] if configs else ["attempt", "no points yet"],
                 [[a, *[sig(latest[(c, a)]["value"]) if (c, a) in latest else None for c in configs]] for a in attempts],
                 unit="× native", y_scale="log", updated_at=when, source=src,
                 note=("Prover seconds summed over the run's q-word units ÷ the run's native GPU seconds; x is the hill-climb step "
                       "(ov.attempt), one line per model, phase and prover. TP1 B1 1024/128 greedy. Lower is better; log scale."
                       + ("" if configs else " No points yet.")))]
    rows = []
    for c in configs:
        mine = sorted((p for (cc, _), p in latest.items() if cc == c), key=lambda p: p["attempt"])
        last = mine[-1]
        parts = c.split("/")
        other = "/".join(parts[:-1] + ["m0" if parts[-1] == "zk" else "zk"])
        twin = max((p for (cc, _), p in latest.items() if cc == other), key=lambda p: p["attempt"], default=None)
        rows.append([c, last["attempt"], (last["line"] or "")[:200] or None, sig(last["value"]),
                     sig(min(p["value"] for p in mine)),
                     ratio(last["value"], twin["value"]) if twin and parts[-1] == "zk" else None, len(mine), last["when"]])
    out.append(panel("prover-overhead-vllm-latest", "vLLM prover overhead: latest step per model, phase and prover", "table",
                     ["Config", "Step", "Step label", "Latest (× native)", "Best (× native)", "ZK ÷ M0", "Steps", "Labelled (UTC)"],
                     rows, updated_at=when, source=src,
                     note="One row per ov.config; ZK ÷ M0 compares a --zk config's latest step with its M0 twin's."))
    return out


# ------------------------------------------------------------------------------------------------------- overnight goal numbers

OVERNIGHT = Path("/workspace/usage/overnight")  # one `verity/overnight-numbers/v0` file per owner, for the console's Overnight view
OWNER = re.compile(r"[a-z0-9-]{1,40}")
GOAL_KEY = re.compile(r"[a-z0-9-]{1,40}")


GOAL_NOTES = OVERNIGHT / "notes" / "goals.json"  # docs/goals.md's status note and checked state per goal (verity/overnight-goal-notes/v0)
NOTES_SOURCE = "docs/goals.md status note"


def goal_note_rows(n: dict) -> list[list]:
    """A goal's status note, split into cells of at most 200 characters at spaces, then its checked state."""
    words, chunks = (n.get("note") or "").split(), [""]
    for w in words:
        if chunks[-1] and len(chunks[-1]) + 1 + len(w) > 200:
            chunks.append("")
        chunks[-1] = (chunks[-1] + " " + w).strip()[:200]
    t = isoz(n.get("as_of"))
    rows = [[n["key"], "status" if i == 0 else "status, continued", c, None, t, NOTES_SOURCE] for i, c in enumerate(chunks) if c]
    return rows + [[n["key"], "done", "yes" if n.get("done") else "no", None, t, NOTES_SOURCE]]


def goal_notes(errors: list) -> tuple[list[dict], str | None]:
    if not GOAL_NOTES.is_file():
        return [], None
    try:
        doc = json.loads(GOAL_NOTES.read_text())
        if doc.get("schema") != "verity/overnight-goal-notes/v0":
            raise ValueError(f"schema {doc.get('schema')!r}")
        notes = [n for n in doc.get("goals") or [] if isinstance(n.get("owner"), str) and OWNER.fullmatch(n["owner"])
                 and isinstance(n.get("key"), str) and GOAL_KEY.fullmatch(n["key"])]
        return notes, isoz(doc.get("updated_utc"))
    except Exception as e:  # noqa: BLE001  owners' files still publish without the notes
        errors.append(f"overnight notes: {type(e).__name__}: {e}")
        return [], None


def overnight_panels(errors: list) -> list[dict]:
    """One table per owner: a row per number, filed under the goal's key, which the site's Overnight view reads. A goal's status
    note from docs/goals.md stands in for the owner's rows under that key unless the owner's file is at least as new as the note.
    A file's optional "title" names its panel."""
    owners: dict[str, dict] = {}
    for f in sorted(OVERNIGHT.glob("*.json")) if OVERNIGHT.is_dir() else []:
        try:
            doc = json.loads(f.read_text())
            owner = doc.get("owner")
            if doc.get("schema") != "verity/overnight-numbers/v0" or not isinstance(owner, str) or not OWNER.fullmatch(owner):
                raise ValueError(f"schema {doc.get('schema')!r}, owner {owner!r}")
            rows = []
            for n in doc.get("numbers") or []:
                v = n.get("value")
                if not (isinstance(n.get("goal"), str) and GOAL_KEY.fullmatch(n["goal"]) and isinstance(n.get("metric"), str)):
                    raise ValueError(f"number {json.dumps(n)[:120]}: goal must be a lowercase key and metric a string")
                if isinstance(v, bool) or not (v is None or isinstance(v, (int, float, str))):
                    raise ValueError(f"number {n['metric'][:60]!r}: value must be a number, a string or null")
                rows.append([n["goal"], n["metric"][:200], (v if isinstance(v, int) else sig(v, 6)) if isinstance(v, (int, float)) else (v[:200] if v else None),
                             (str(n["unit"])[:20] if n.get("unit") else None), isoz(n.get("as_of")), (str(n["source"])[:200] if n.get("source") else None)])
            when = isoz(doc.get("updated_utc")) or iso(datetime.fromtimestamp(f.stat().st_mtime, timezone.utc))
            title = doc.get("title")
            if title is not None and not (isinstance(title, str) and 0 < len(title) <= 120):
                raise ValueError("title must be a string of 1 to 120 characters")
            owners[owner] = {"rows": rows, "when": when, "source": f"{owner}'s {f} on vy-nebius-1", "title": title}
        except Exception as e:  # noqa: BLE001  one bad file doesn't stop the others
            errors.append(f"overnight {f.name}: {type(e).__name__}: {e}")
    notes, notes_when = goal_notes(errors)
    for n in notes:
        o = owners.setdefault(n["owner"], {"rows": [], "when": None, "source": None})
        t = isoz(n.get("as_of"))
        if any(r[0] == n["key"] for r in o["rows"]) and (t is None or (o["when"] and o["when"] >= t)):
            continue
        o["rows"] = [r for r in o["rows"] if r[0] != n["key"]] + goal_note_rows(n)
        o["notes"] = True
    out = []
    for owner, o in sorted(owners.items()):
        when = max((w for w in (o["when"], notes_when if o.get("notes") else None) if w), default=iso(datetime.now(timezone.utc)))
        source = " and ".join(s for s in (o["source"], f"{NOTES_SOURCE}s ({GOAL_NOTES} on vy-nebius-1)" if o.get("notes") else None) if s)
        out.append(fit(panel(f"overnight-{owner}", o.get("title") or f"{owner}: overnight goal numbers", "table",
                             ["Goal", "Metric", "Value", "Unit", "As of (UTC)", "Source"], o["rows"], updated_at=when, source=source,
                             note="Numbers for the overnight set (Project store docs/goals.md), one row per number, filed under the goal's "
                                  "key: the owner's file, or the goal's docs/goals.md status note and checked state where that is newer.")))
    return out


# ---------------------------------------------------------------------------------------------------------- GPU minutes (both nodes)

N2_HOST = "vy-n2"
N2_UTIL = "/workspace/pouw/infra/util"  # POUS's own 10 s sampler on vy-nebius-2, one file a UTC day; read, never written
MINUTES_H = 25  # a day of hours plus the hour before the first one, for rolling last-hour figures
NO_POD = "(no pod: direct or leaked)"
KUEUE_POD = re.compile(r"^nd-(.+?)-[0-9a-f]{10}-")
SKY_POD = re.compile(r"^(.+?)-\d+-[0-9a-f]{8}-head$")
N2_MINUTES = r"""
import datetime as dt, glob, json, os, sys
start = float(sys.argv[1])
out = {}
for f in sorted(glob.glob(sys.argv[2] + "/*.jsonl")):
    if os.path.getmtime(f) < start:
        continue
    for ln in open(f):
        try:
            d = json.loads(ln)
            t = dt.datetime.strptime(d["t"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()
        except (ValueError, KeyError):
            continue
        if t < start:
            continue
        for g in d.get("gpus") or []:
            who = next((x[4:] for x in (g.get("lease") or "").split() if x.startswith("who=")), None)
            held = bool(who) or bool(g.get("procs")) or (g.get("mem_mib") or 0) >= 1024
            k = (int(t) // 60 * 60, g["i"])
            c = out.setdefault(k, [0, 0, 0, 0, 0, {}])
            if d.get("timed") or g.get("util") is None:
                c[4] += 1
            else:
                c[2 * (not held) + (not g["util"] > 0)] += 1
            h = who or ("(process, no lease)" if held else None)
            if h:
                c[5][h] = c[5].get(h, 0) + 1
json.dump([[m, i, *(round(x / sum(c[:5]), 3) for x in c[:4]), max(c[5], key=c[5].get) if c[5] else None,
            round(c[4] / sum(c[:5]), 3)] for (m, i), c in sorted(out.items())], sys.stdout, separators=(",", ":"))
"""


def owner_n1(namespace: str | None, pod: str) -> str:
    """A node-1 pod's job family: Kueue jobs by their name, SkyPilot managed jobs by their name's first word."""
    if m := KUEUE_POD.match(pod):
        return f"Kueue {m.group(1)}"
    if m := SKY_POD.match(pod):
        return f"SkyPilot {m.group(1).split('-')[0]}-*"
    if namespace == "vy-system":
        return f"direct ({pod})"
    return pod[:40]


def owner_n2(who: str) -> str:
    """A node-2 lease's holder: its kind (fill, verity, ...) and the agent's short id."""
    m = re.search(r"bc-[0-9a-f]{8}", who)
    kind = who.split(":", 1)[0] if ":" in who else None
    return " ".join(x for x in (kind, m.group(0) if m else who[:40]) if x)


def n1_minutes(start: float, end: float) -> list[tuple]:
    """(minute, GPU, held and busy, held and idle, free and busy, free and idle, holder), the four as shares of the minute.

    DCGM at 60 s gives one reading a minute, so each minute is wholly one of the four: busy = utilization > 0, held = a pod on the
    GPU or >= 1 GiB used. Prometheus is read at each minute's 59th second, the minute's own scrape, so every pass reads a minute
    alike."""
    start = int(start) // 60 * 60 + 59
    util = prom_range("max by (gpu, pod, namespace) (DCGM_FI_DEV_GPU_UTIL)", start, end)
    fb = prom_range("max by (gpu) (DCGM_FI_DEV_FB_USED)", start, end)
    mem = {(r["metric"].get("gpu"), int(float(t)) // 60 * 60): float(v) for r in fb for t, v in r["values"]}
    seen: dict[tuple, tuple] = {}
    for r in util:
        m = r["metric"]
        pod = m.get("pod") or None
        for t, v in r["values"]:
            k = (int(float(t)) // 60 * 60, m.get("gpu"))
            b, o = seen.get(k, (False, None))
            seen[k] = (b or float(v) > 0, o or (owner_n1(m.get("namespace"), pod) if pod else None))
    out = []
    for (t, g), (b, o) in seen.items():
        held = bool(o) or mem.get((g, t), 0.0) >= 1024
        q = [0.0] * 4
        q[2 * (not held) + (not b)] = 1.0
        out.append((t, int(g) if str(g).isdigit() else g, *q, o or (NO_POD if held else None)))
    return sorted(out, key=lambda r: (r[0], str(r[1])))


def n2_minutes(start: float) -> list[tuple]:
    """The same rows for vy-nebius-2 from POUS's 10 s sampler, each share the fraction of the minute's samples, the holder the
    most of its samples name, and an eighth: the unmeasured share, samples that read no utilization (a timed window's, which
    read no NVML, or a `util` nvidia-smi couldn't read, null)."""
    doc = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", N2_HOST, "python3", "-", str(int(start)), N2_UTIL],
                         input=N2_MINUTES, check=True, capture_output=True, text=True, timeout=120).stdout
    return [(*r[:6], owner_n2(r[6]) if r[6] and not r[6].startswith("(") else r[6], *r[7:]) for r in json.loads(doc)]


def raw_busy(minutes: list[tuple], at: float, span: float = 3600) -> float | None:
    """Share of observed GPU-time in (at - span, at] with utilization above 0; unmeasured time isn't observed."""
    xs = [r for r in minutes if at - span < r[0] + 60 <= at]
    seen = sum(r[2] + r[3] + r[4] + r[5] for r in xs)
    return sum(r[2] + r[4] for r in xs) / seen if seen else None


def split_rows(minutes: list[tuple], key) -> dict:
    """GPU-hours by key(row, held): held and busy, held and idle, busy with no holder, and neither."""
    acc: dict = defaultdict(lambda: defaultdict(float))
    for r in minutes:
        for held, parts in ((True, (("held_busy", r[2]), ("held_idle", r[3]))), (False, (("free_busy", r[4]), ("free_idle", r[5])))):
            if not parts[0][1] and not parts[1][1]:
                continue
            a = acc[key(r, held)]
            for k, v in parts:
                a[k] += v / 60
                a["obs"] += v / 60
                if k.endswith("busy"):
                    a["busy"] += v / 60
    return acc


def hours_panel(name: str, host: str, minutes: list[tuple], now: datetime, src: str) -> dict:
    cut = now.timestamp() - 24 * 3600
    cur = uc_hour(now.timestamp())
    by_hour = split_rows([r for r in minutes if r[0] >= cut], lambda r, _: uc_hour(r[0]))
    hrows = []
    for h in sorted(by_hour, reverse=True):
        a = by_hour[h]
        hrows.append([h + (" (so far)" if h == cur else ""), sig(a["obs"], 3), sig(100 * a["busy"] / a["obs"], 3) if a["obs"] else None,
                      sig(a["held_busy"], 3), sig(a["held_idle"], 3), sig(a["free_idle"] + a["free_busy"], 3), sig(a["free_busy"], 3)])
    return panel(f"{name}-hours", f"{host} per hour: busy, held idle, not held", "table",
                 ["Hour (UTC)", "GPU-h observed", "Raw busy %", "Held and busy", "Held and idle", "Not held", "of which busy"], hrows,
                 updated_at=iso(now), source=src,
                 note="GPU-hours from POUS's 10 s sampler. Raw busy: the share of samples with utilization above 0 (node 1's "
                      "rule, DCGM's one reading a minute, on node 2's samples). Held: a lease, a process, or 1 GiB of memory in use. Not held: none of those; 'of which busy' is "
                      "the part with utilization anyway. The last 24 hours, newest first.")


def owner_panels(name: str, host: str, minutes: list[tuple], now: datetime, src: str) -> list[dict]:
    out = []
    cut = now.timestamp() - 24 * 3600
    day = [r for r in minutes if r[0] >= cut]
    cur = uc_hour(now.timestamp())
    tot = split_rows(day, lambda r, held: r[6] if held else None)
    orows = []
    for o, a in sorted(((o, a) for o, a in tot.items() if o), key=lambda x: -(x[1]["held_busy"] + x[1]["held_idle"])):
        held = a["held_busy"] + a["held_idle"]
        orows.append([o, sig(held, 3), sig(a["held_busy"], 3), sig(a["held_idle"], 3), sig(100 * a["held_busy"] / held, 3) if held else None])
    free = tot.get(None)
    if free:
        orows.append(["Not held", sig(free["obs"], 3), sig(free["free_busy"], 3), sig(free["free_idle"], 3), None])
    allo = sum(a["obs"] for a in tot.values())
    orows.append(["All GPU-hours", sig(allo, 3), sig(sum(a["held_busy"] + a["free_busy"] for a in tot.values()), 3),
                  sig(sum(a["held_idle"] + a["free_idle"] for a in tot.values()), 3), None])
    out.append(panel(f"{name}-owners", f"{host}: GPU-hours by holder, last 24 hours", "table",
                     ["Holder", "GPU-h", "Busy", "Idle", "Busy share of held %"], orows[:200], updated_at=iso(now), source=src,
                     note="Holder is the pod DCGM names on the GPU: Kueue jobs by job name, SkyPilot managed jobs by their name's first "
                          "word. Held: a pod on the GPU, or 1 GiB used. Busy: utilization above 0 in DCGM's 60 s reading. 'Not held' has no holder; its busy part is utilization with "
                          "no pod on the GPU."))
    ohour = split_rows(day, lambda r, held: (uc_hour(r[0]), r[6] if held else None))
    ohrows = []
    for (h, o), a in sorted(ohour.items(), key=lambda x: (x[0][0], x[0][1] is not None, x[1]["held_busy"] + x[1]["held_idle"]), reverse=True):
        label = h + (" (so far)" if h == cur else "")
        if o is None:
            ohrows.append([label, "Not held", None, None, sig(a["obs"], 3)])
        else:
            ohrows.append([label, o, sig(a["held_busy"], 3), sig(a["held_idle"], 3), None])
    out.append(fit(panel(f"{name}-owner-hours", f"{host} per hour and holder: held and busy, held and idle, not held", "table",
                         ["Hour (UTC)", "Holder", "Held and busy", "Held and idle", "Not held"], ohrows[:1000], updated_at=iso(now),
                         source=src, note="GPU-hours per UTC hour and holder, from per-minute DCGM readings; newest first.")))
    return out


def uc_hour(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:00Z")


# ---------------------------------------------------------------------------------------------------------------- held idle (infra/*)

IDLE_READINGS = Path("/workspace/usage/held-idle-hourly.jsonl")  # one line per completed UTC hour, node and lane; infra's relay alerts on it
IDLE_TARGETS = {("node 1", "circuits"): 40.0}  # the 11:30 AM PDT Oct 1 set: circuits' Commits at or below 40% held idle on node 1
POOL_UNLEASED = "lease pool, not leased"  # cluster.usage.POOL_UNLEASED
N1_LANES = [(re.compile(p), lane) for p, lane in (
    (r"^gpu-pool-circuits-", "circuits"), (r"^gpu-pool-\d", POOL_UNLEASED),  # the provers pool serves every lane
    (r"^Kueue (vllm-|commit-pack|circuits-|mps-golden|n2-build)", "circuits"), (r"^Kueue (proofs-|backend-sweep|assumption-)", "proofs"),
    (r"^SkyPilot (cfgtp2|sm120|vcd)-", "circuits"), (r"^SkyPilot (m0|tcdefs|as)-", "proofs"))]
N2_LANES = {"bc-17cc41f1": "infra", "bc-8416bc72": "proofs", "bc-b8aaadaa": "circuits", "bc-e90634dd": "compute-accounting",
            "bc-15ada664": "memory-accounting", "bc-ecea50f6": "network-accounting", "bc-ddee017b": "console"}


def lane_of(node: str, holder: str) -> str:
    """The lane a holder's GPU-time counts to: node 1's by job name, node 2's by its lease's who (who_lane); anything else by its
    label."""
    if node == "node 1":
        return next((lane for rx, lane in N1_LANES if rx.search(holder)), holder)
    return who_lane(holder)


def who_lane(who: str) -> str:
    """The lane a gpu-lease `who` counts to: the lane its first field names, after a `keeper:` or `fill:` prefix if it has one
    (`keeper:memory-accounting`, `memory-accounting:erase-calib`), else the coordinator's id (`bc-xxxxxxxx`, N2_LANES), else
    the who itself."""
    m = re.match(r"(?:(?:keeper|fill)[: ])?([a-z-]+)(?=[:_ ]|$)", who)
    if m and m.group(1) in N2_LANES.values():
        return m.group(1)
    m = re.search(r"bc-[0-9a-f]{8}", who)
    return N2_LANES.get(m.group(0), who) if m else who


def kind_of(node: str, holder: str) -> str:
    """A holder's job family, the `kind` its GPU-time counts to: node 1's holder label (owner_n1's family), a lease pool's own pod
    as `gpu-lease pool`; node 2's lease by its `who=` prefix (fill, verity, ...), `gpu-lease` when it has none."""
    if holder.startswith("("):
        return holder
    if node == "node 1":
        return "gpu-lease pool" if POOL_HOLDER.match(holder) else holder
    first, _, rest = holder.partition(" ")
    return first if rest else "gpu-lease"


#: n1_lease.py's pools on node 1, each with the lane a lease it granted counts to when no record names the lease's who
N1_LEASE_POOLS = {"/run/gpu-lease-circuits": "circuits", "/run/gpu-lease": "proofs"}
N1_LEASE_USAGE = Path("/workspace/research/lease-usage.jsonl")  # gpu-lease's GPU_LEASE_USAGE_LOG default: node 1's finished leases
N1_POOL_DIR = "/run/gpu-lease"  # n1_lease.py's GPU_LEASE_DIR default
N1_POOL_IDLE_S = 120  # n1_lease.py's VY_POOL_IDLE_S default: a pool GPU free this long, with nobody waiting, leaves the pool
N1_LOOP_ENV = ("GPU_LEASE_DIR", "VY_POOL_IDLE_S", "VY_LEASE_BRAIN")  # the only variables read from a loop's environment
POOL_HOLDER = re.compile(r"^gpu-pool-")
NODE_NAMES = {"node 1": "vy-nebius-1", "node 2": "vy-nebius-2"}  # the held-idle file's names for the usage record's nodes
USAGE_RECORD = Path("/workspace/usage/usage-hourly.jsonl")  # cluster.usage.RECORD: both nodes' rows, `cluster usage` reads it


def usage_module():
    """`cluster.usage`, the one usage computation: the copy deploy.toml ships beside this script, else the checkout's; never
    another `cluster` on the path."""
    for p in (HERE, HERE.parents[2] / "infra" / "cluster" / "src"):
        if (p / "cluster" / "usage.py").is_file():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            break
    else:
        raise ModuleNotFoundError(f"no cluster/usage.py beside the console or in a checkout: install deploy.toml's tree "
                                  f"{HERE / 'cluster'} (research deploy install vy-nebius-1 --by WHO {HERE / 'cluster'})",
                                  name="cluster")
    from cluster import usage  # noqa: PLC0415  node 1's python3 has no cluster installed
    return usage


def _n1_lease_loop(argv: list[bytes]) -> bool:
    args = [a.decode(errors="replace") for a in argv]
    return bool(args) and Path(args[0]).name.startswith("python") and any(
        Path(a).name == "n1_lease.py" and args[i + 1:i + 2] == ["loop"] for i, a in enumerate(args))


def n1_pool_loops(proc: Path = Path("/proc")) -> dict[str, set[tuple[str, str]]]:
    """Per pool directory, the (VY_POOL_IDLE_S, VY_LEASE_BRAIN) of each `n1_lease.py loop` running for it, read from its
    /proc cmdline and environment as n1_lease.py reads them, with its defaults. Only N1_LOOP_ENV's variables are read. A loop
    whose environment can't be read could be any pool's, so it raises LookupError."""
    loops: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for p in proc.glob("[0-9]*"):
        try:
            if not _n1_lease_loop((p / "cmdline").read_bytes().split(b"\0")):
                continue
            raw = (p / "environ").read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError as e:
            raise LookupError(f"can't read the environment of n1_lease.py loop {p.name}: {type(e).__name__}") from None
        if not raw:  # exited since its cmdline was read
            continue
        env = {}
        for kv in raw.split(b"\0"):
            k, sep, v = kv.partition(b"=")
            if sep and k.decode(errors="replace") in N1_LOOP_ENV:
                env[k.decode()] = v.decode(errors="replace")
        loops[str(Path(env.get("GPU_LEASE_DIR", N1_POOL_DIR)))].add(
            (env.get("VY_POOL_IDLE_S", str(N1_POOL_IDLE_S)), env.get("VY_LEASE_BRAIN", "")))
    return loops


def n1_pool_idle(pool: str, loops: dict[str, set[tuple[str, str]]]) -> int:
    """The seconds `pool`'s loop waits with a GPU free before it shrinks, from n1_pool_loops; LookupError when that isn't one
    known number: no loop runs for the pool, its loops disagree, or a brain decides its shrinks (they don't wait)."""
    found = loops.get(str(Path(pool)), set())
    if not found:
        raise LookupError(f"no n1_lease.py loop runs for {pool}, so how long its GPUs sit free before a shrink is unknown")
    if len(found) > 1:
        raise LookupError(f"{pool}'s {len(found)} n1_lease.py loops differ in VY_POOL_IDLE_S or VY_LEASE_BRAIN")
    [(idle, brain)] = found
    if brain:
        raise LookupError(f"{pool}'s loop runs with VY_LEASE_BRAIN, whose shrinks don't wait VY_POOL_IDLE_S")
    try:
        return int(idle)
    except ValueError:
        raise LookupError(f"{pool}'s loop has a VY_POOL_IDLE_S that isn't a whole number of seconds") from None


def _utc(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def n1_leases(now: float, pools: dict = N1_LEASE_POOLS, usage_log: Path = N1_LEASE_USAGE) -> list[tuple]:
    """(GPU, start, end, who) for each lease node 1's gpu-lease granted: since to end of each finished one's
    `gpu-lease/usage/v1` record in usage_log, and since to now of each live one in a pool's n1-lease.json that no record has."""
    out = []
    for ln in usage_log.read_text().splitlines() if usage_log.exists() else ():
        try:
            d = json.loads(ln)
            if d.get("schema") == "gpu-lease/usage/v1" and d.get("who"):
                a, b = _utc(d["since"]), _utc(d["end"])
                out += [(int(g["index"]), a, b, d["who"]) for g in d["gpus"]]
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
    done = {(g, a, who) for g, a, _, who in out}
    for d in pools:
        f = Path(d) / "n1-lease.json"
        for g in (json.loads(f.read_text()).get("gpus") or []) if f.exists() else ():
            lease = g.get("lease") or {}
            try:
                a = _utc(lease["since"])
            except (ValueError, KeyError, TypeError):
                continue
            if lease.get("who") and (g["index"], a, lease["who"]) not in done:
                out.append((g["index"], a, max(a, now), lease["who"]))
    return out


def by_lease(spans: list[tuple], leases: list[tuple]) -> list[tuple]:
    """Each pool span (GPU, start, end, pool's lane) as the leases on its GPU that overlap it, clipped to it, each counting to
    its who's lane (who_lane). A span no lease overlaps keeps its pool's lane: nobody's who is known for it. In a span some
    lease overlaps, the time no lease covers is the pool's, leased to nobody, so no span holds it and the minute's reading
    counts it: the pool's pod there is cluster.usage's POOL_UNLEASED."""
    out = []
    for g, a, b, lane in spans:
        mine = [(g, max(a, x), min(b, y), who_lane(who)) for h, x, y, who in leases if h == g and x < b and y > a]
        out += mine or [(g, a, b, lane)]
    return out


def n1_lease_spans(now: float, pools: dict = N1_LEASE_POOLS, idle: dict | None = None,
                   usage_log: Path = N1_LEASE_USAGE) -> list[tuple]:
    """(GPU, start, end, lane) for each lease a node-1 pool granted, from n1_lease.py's log and state in each pool's directory,
    each pool span split by the leases gpu-lease records in it (by_lease over n1_leases).

    The log records pool membership, not leases: a GPU joins the pool (`unfenced`) for a waiting lease and leaves it (`shrink`)
    once it has been free for the pool's idle seconds with nobody waiting, so its lease ran from the unfence to the shrink less
    that idle (to within the pool's 5 s tick; a GPU leased again before it shrinks is one span, the gap in it). `blocked-shrink`
    and `fenced` end it where they stand. A GPU still in the pool is leased until now if n1-lease.json shows a lease on it, else
    until the idle before the state's time. `idle` maps each pool to its idle seconds; by default each pool's running loop
    gives it (n1_pool_idle), and a pool with a log whose idle isn't known raises LookupError."""
    spans, loops = [], None
    for d, lane in pools.items():
        log = Path(d) / "n1-lease.log"
        if not log.exists():
            continue
        if idle is None:
            loops = n1_pool_loops() if loops is None else loops
            idle_s = n1_pool_idle(d, loops)
        else:
            idle_s = idle[d]
        start: dict[int, float] = {}
        for ln in log.read_text().splitlines():
            try:
                e = json.loads(ln)
                t, g = datetime.strptime(e["t"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp(), e.get("gpu")
            except (ValueError, KeyError, TypeError):
                continue
            if g is None:
                continue
            if e.get("ev") == "unfenced":
                start.setdefault(g, t)
            elif e.get("ev") in ("shrink", "blocked-shrink", "fenced") and g in start:
                a = start.pop(g)
                b = t - idle_s if e["ev"] == "shrink" else t
                if b > a:
                    spans.append((g, a, b, lane))
        state = json.loads((Path(d) / "n1-lease.json").read_text()) if start and (Path(d) / "n1-lease.json").exists() else {}
        st = datetime.strptime(state["t"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp() if state else now
        leased = {g["index"] for g in state.get("gpus") or [] if g.get("lease")}
        for g, a in start.items():
            b = now if g in leased else st - idle_s
            if b > a:
                spans.append((g, a, b, lane))
    return by_lease(spans, n1_leases(now, pools, usage_log)) if spans else []


def usage_rows(minutes: dict, spans: dict | None = None) -> list[dict]:
    """`cluster.usage.hourly`'s rows (per UTC hour, node, owner and kind: GPU-seconds held, busy, idle, unmeasured, useful) for
    each node whose minutes were read. Owners are lanes (lane_of), kinds job families (kind_of). spans: per node, (GPU, start,
    end, lane) of its pools' leases (n1_lease_spans), each a `gpu-lease` holding of its lane; on such a node a pool's own
    pod outside every lease is the module's POOL_UNLEASED, and the provers pool's pod is owned by POOL_UNLEASED too."""
    usage = usage_module()
    out = []
    for node, rows in minutes.items():
        if rows is None:
            continue
        leases = (spans or {}).get(node)
        out += usage.hourly(NODE_NAMES[node], rows,
                            None if leases is None else [usage.Lease(g, a, b, lane, "gpu-lease") for g, a, b, lane in leases],
                            attribute=lambda h, node=node: (lane_of(node, h), kind_of(node, h)),
                            pooled=lambda h: bool(POOL_HOLDER.match(h)))
    return out


def idle_readings(rows: list[dict], now: datetime) -> list[dict]:
    """Per UTC hour, node and lane: held GPU-hours and the held-idle share, with the change from the lane's hour before
    (`cluster.usage.held_idle` over usage_rows; its rules: a lease's GPU is held for its lane for exactly the lease, busy by
    that GPU's reading, unmeasured without one or in a timed window, never idle)."""
    return usage_module().held_idle(rows, now.timestamp(), IDLE_TARGETS, {v: k for k, v in NODE_NAMES.items()})


COV_TIMELINES = Path("/workspace/jobs/cov")  # <key>/<row>/timeline.jsonl: circuits' rows on node 1 (verity-sweep/timeline/v1)
LEASE_PHASES = ("engine.build", "prep.warmup_instrumented", "validate.weights_of_record_digests")
#: a lease open longer than this is a Commit that died holding it, and counts no further
LEASE_OPEN_MAX_S = 4 * 3600


def _lease(spans: dict, a: float, b: float) -> tuple[float, float, list[tuple[str, float, float]], int]:
    return (a, b, [("work" if s.startswith("work.") else s, x, y) for s, xs in spans.items() for x, y in xs
                   if s in LEASE_PHASES or s.startswith("work.")], 1)


def tp_lease(row: Path, w0: float, w1: float | None, now: float) -> tuple[float, float, list, int] | None:
    """The lease of a Commit window whose timeline has no `gpu_lease.*` span (a TP Commit writes only `commit.tp_commit`), from
    the files gpu-lease writes in the row as it goes: `gpu_lease.started` to `gpu_lease.released`'s `t`, on one GPU per rank it
    released (gm357: 145 s on 2 GPUs). A lease still open runs to the window's end, or to `now` up to LEASE_OPEN_MAX_S, on the
    row's `__tpN__` GPUs. Files an earlier attempt left in the row start before the window and don't count."""
    try:
        a = datetime.strptime((row / "gpu_lease.started").read_text().strip(), "%Y-%m-%dT%H:%M:%SZ")
    except (OSError, ValueError):
        return None
    a = a.replace(tzinfo=timezone.utc).timestamp()
    if a < w0 or (w1 is not None and a > w1):
        return None
    try:
        rel = json.loads((row / "gpu_lease.released").read_text())
    except (OSError, ValueError):
        rel = None
    if isinstance(rel, dict) and isinstance(rel.get("t"), (int, float)) and rel["t"] >= a:
        return (a, rel["t"] if w1 is None else min(rel["t"], w1), [], max(1, len(rel.get("ranks") or [])))
    tp = re.search(r"__tp(\d+)__", row.name)
    return (a, w1 if w1 is not None else max(a, min(now, a + LEASE_OPEN_MAX_S)), [], int(tp.group(1)) if tp else 1)


def commit_leases(paths, now: float) -> list[tuple[float, float, list[tuple[str, float, float]], int]]:
    """(start, end, [(phase, a, b)...], GPUs) for every Commit's GPU lease in these timelines.

    circuits' definition (`/workspace/research/circuits-held-idle/hi-lease.py`): inside a `stage.commit` window the lease runs
    from the end of `gpu_lease.wait` to the end of `gpu_lease.release`, on one GPU, and its phases are LEASE_PHASES plus `work`
    (every `work.*` span). A window with no `gpu_lease.*` span is a TP Commit's, whose lease `tp_lease` reads from the row; its
    phases are the window's spans, as for one GPU, and `phases_by_hour` counts them on each of the lease's GPUs. Every
    Commit attempt in a timeline counts, not only its last.  An hour's line is written once, as the hour ends, so a lease still
    open at `now` runs to `now` (up to LEASE_OPEN_MAX_S), its open phases with it: a Commit writes nothing while a work phase
    runs (gm176's took 2,077 s)."""
    out = []
    for p in paths:
        try:
            evs = [json.loads(ln) for ln in Path(p).read_text().splitlines() if ln.strip()]
        except (OSError, ValueError):
            continue
        opened: dict = {}
        spans: dict = {}
        inside = released = False
        w0 = None

        def tp(w1: float | None) -> None:
            if isinstance(w0, (int, float)) and not any(s.startswith("gpu_lease.") for s in (*spans, *(s for _, s in opened))):
                lease = tp_lease(Path(p).parent, w0, w1, now)
                if lease:
                    a, b, _, gpus = lease
                    live = {s: list(xs) for s, xs in spans.items()}
                    for (_, s), x in opened.items():
                        live.setdefault(s, []).append((x, b))
                    out.append((*_lease(live, a, b)[:3], gpus))

        for e in evs:
            stage, ev, t = e.get("stage"), e.get("ev"), e.get("t")
            if stage == "stage.commit":
                if inside and ev == "end" and isinstance(t, (int, float)):
                    tp(t)
                inside, opened, released = ev == "start", {}, False
                if inside:
                    spans, w0 = {}, t
                continue
            if not inside or not isinstance(t, (int, float)):
                continue
            if ev == "start":
                opened[(e.get("pid"), stage)] = t
            elif ev == "end" and (e.get("pid"), stage) in opened:
                spans.setdefault(stage, []).append((opened.pop((e.get("pid"), stage)), t))
                if stage == "gpu_lease.release" and "gpu_lease.wait" in spans and not released:
                    out.append(_lease(spans, spans["gpu_lease.wait"][0][1], t))
                    released = True
        if inside and not released and "gpu_lease.wait" in spans:
            a = spans["gpu_lease.wait"][0][1]
            b = max(a, min(now, a + LEASE_OPEN_MAX_S))
            live = {s: list(xs) for s, xs in spans.items()}
            for (_, s), x in opened.items():
                live.setdefault(s, []).append((x, b))
            out.append(_lease(live, a, b))
        elif inside:
            tp(None)
    return out


def phases_by_hour(leases: list) -> dict[str, dict]:
    """Per UTC hour, the GPU-hours of the Commit leases in it by phase, each span clipped to its lease and the hour and counted
    on each of its lease's GPUs; `other` is the leased time no phase covers, `leases` how many leases the hour touches."""
    out: dict[str, dict] = {}
    for a, b, phases, gpus in leases:
        h = int(a) // 3600 * 3600
        while h < b:
            lo, hi = max(a, h), min(b, h + 3600)
            if hi > lo:
                d = out.setdefault(uc_hour(h), {"leases": 0, **{s: 0.0 for s in (*LEASE_PHASES, "work", "other")}})
                d["leases"] += 1
                covered = 0.0
                for s, x, y in phases:
                    c = max(0.0, min(y, hi) - max(x, lo))
                    d[s] += c * gpus / 3600
                    covered += c
                d["other"] += max(0.0, hi - lo - covered) * gpus / 3600
            h += 3600
    return {k: {s: (round(v, 3) if s != "leases" else v) for s, v in d.items()} for k, d in out.items()}


def add_phases(readings: list[dict], now: datetime, root: Path = COV_TIMELINES, hours: int = MINUTES_H) -> None:
    """Node 1 circuits' readings get `phases_gpu_h` from the Commit timelines written in the readings' window, and with it
    `outside_leases`: the lane's held GPU-hours no Commit lease covers (a whole-GPU pod, a job that isn't a Commit)."""
    since = now.timestamp() - (hours + 1) * 3600
    paths = [p for p in root.glob("*/*/timeline.jsonl") if p.stat().st_mtime >= since]
    by_hour = phases_by_hour(commit_leases(paths, now.timestamp()))
    for r in readings:
        if r["node"] == "node 1" and r["lane"] == "circuits":
            p = dict(by_hour.get(r["hour"]) or {"leases": 0, **{s: 0.0 for s in (*LEASE_PHASES, "work", "other")}})
            leased = sum(v for s, v in p.items() if s != "leases")
            r["phases_gpu_h"] = {**p, "outside_leases": round(max(0.0, r["held_gpu_h"] - leased), 3)}


def phases_text(p: dict | None) -> str | None:
    if not p:
        return None
    names = {"engine.build": "build", "prep.warmup_instrumented": "warm-up", "validate.weights_of_record_digests": "digests"}
    return (", ".join(f"{names.get(s, s)} {sig(p[s], 2)}" for s in (*LEASE_PHASES, "work", "other")) + f" ({p['leases']} leases)"
            + f"; outside Commit leases {sig(p.get('outside_leases'), 2)}")


def record_usage(rows: list[dict], readings: list[dict], now: datetime, since: float, idle_path: Path = IDLE_READINGS,
                 usage_path: Path = USAGE_RECORD) -> tuple[int, int]:
    """Append each complete hour the minutes covered whole: its held-idle lines once per (hour, node, lane), its usage rows once
    per (hour, node); returns how many of each it wrote."""
    usage = usage_module()
    t = now.timestamp()
    return (usage.record(readings, idle_path, ("hour", "node", "lane"), t, since),
            usage.record(rows, usage_path, ("hour", "node"), t, since))


def idle_panel(readings: list[dict], now: datetime) -> dict:
    rows = [[r["hour"] + ("" if r["complete"] else " (so far)"), r["node"], r["lane"], sig(r["held_gpu_h"], 3),
             sig(r["held_idle_gpu_h"], 3), sig(r["held_idle_pct"], 3), sig(r.get("unmeasured_gpu_h"), 3), r["target_pct"],
             r["change_pts"], phases_text(r.get("phases_gpu_h"))]
            for r in sorted(sorted(readings, key=lambda r: (r["lane"] != "(all holders)", -r["held_gpu_h"])),
                            key=lambda r: (r["hour"], r["node"]), reverse=True)
            if r["held_gpu_h"] >= 0.05 or (r.get("unmeasured_gpu_h") or 0) >= 0.05 or r["lane"] == "(all holders)"]
    return fit(panel("held-idle-hourly", "Held but idle, per hour, node and lane", "table",
                     ["Hour (UTC)", "Node", "Lane", "Held GPU-h", "Held idle GPU-h", "Held idle %", "Unmeasured GPU-h", "Target %",
                      "Change (pts)", "Commit leases by phase (GPU-h)"],
                     rows[:1000], updated_at=iso(now),
                     source="vy-nebius-1's DCGM (per minute) and n1_lease.py's pool logs, vy-nebius-2's POUS sampler (10 s)",
                     note="Held idle %: a lane's measured held GPU-hours with utilization 0 (cluster.usage). Node 1: lanes by "
                          "job name; a pool's lease is held for its time, busy by its GPU's DCGM reading, unmeasured without one; "
                          "the pool holder outside a lease is 'lease pool, not leased'. Node 2: by coordinator id; timed-window "
                          "samples are unmeasured. Change: points against the hour before; infra alerts a lane above target that "
                          "improves by under 5 points. Phases: node 1 circuits' Commit leases' timelines. Newest first."))


# ---------------------------------------------------------------------------------------------------------------- pool (infra/*)

POOL = Path("/workspace/usage/infra-pool.json")  # infra's file: node 2 (and later node 1), every 5 minutes, written atomically
POOL_HISTORY = HERE / "pool-history.jsonl"      # this script's own 24 h of pool samples, one line a run
USEFUL_TARGET = 85  # Daniel, 2:06 PM PDT Sep 30: both nodes at >= 85% useful GPU busy by noon Oct 1 PDT
TARGET_BY = "noon Oct 1 PDT"
TARGET_END = datetime(2026, 10, 1, 19, 0, tzinfo=timezone.utc)
WATERMARK_GPU_H = 12
POOL_STALE_S = 900


def pct(v):
    return sig(100 * v, 3) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def n1_from_prom(summary: dict, util_rows: list, gpu_rows: list, now: datetime) -> dict:
    """Node 1 in the pool file's shape, from this script's own Prometheus reading, until infra's file carries n1."""
    recent = [r for r in util_rows if r[0] >= iso(now - timedelta(hours=1))]
    last = util_rows[-1] if util_rows else None
    mean = lambda j, rows: (sum(r[j] for r in rows if r[j] is not None) / max(1, sum(r[j] is not None for r in rows)) / 100) if rows else None  # noqa: E731
    return {"gpu_busy_5m": last[1] / 100 if last and last[1] is not None else None, "gpu_busy_1h": mean(1, recent),
            "gpu_held_1h": mean(3, recent), "useful_share_1h": None, "cpu_busy_5m": last[5] / 100 if last and last[5] is not None else None,
            "cpu_busy_1h": mean(5, recent),
            "gpus": [{"index": g[0], "busy_5m": (g[1] or 0) / 100, "holder": g[5], "class": g[6]} for g in gpu_rows],
            "_from": "prometheus"}


def held_idle(node: dict):
    """Held-idle share: infra's field when present, else GPUs with a holder and no work in the last 5 minutes."""
    if isinstance(node.get("held_idle_1h"), (int, float)):
        return node["held_idle_1h"]
    if isinstance(node.get("gpu_held_1h"), (int, float)) and isinstance(node.get("gpu_busy_1h"), (int, float)):
        return max(0.0, node["gpu_held_1h"] - node["gpu_busy_1h"])
    gpus = node.get("gpus") or []
    return sum(1 for g in gpus if g.get("holder") and (g.get("busy_5m") or 0) < 0.05) / len(gpus) if gpus else None


def useful(node: dict):
    b, u = node.get("gpu_busy_1h"), node.get("useful_share_1h")
    if not isinstance(b, (int, float)):
        return None
    return b * u if isinstance(u, (int, float)) else b


def pool_panels(pool: dict, n1: dict | None, now: datetime, errors: list, minutes: dict | None = None) -> list[dict]:
    nodes = dict(pool.get("nodes") or {k: v for k, v in pool.items() if k in ("n1", "n2")})
    if "n1" not in nodes and n1:
        nodes["n1"] = n1
    gen = pool.get("generated_at") or iso(now)
    src = f"infra's {POOL} (generated {gen}) and vy-nebius-1's Prometheus"
    names = {"n1": "node 1", "n2": "node 2"}
    order = [k for k in ("n1", "n2") if k in nodes]
    out = []

    busy_rows = []
    for k in order:
        d = nodes[k]
        u, b, hi = useful(d), d.get("gpu_busy_1h"), held_idle(d)
        filler = (b - u) if isinstance(b, (int, float)) and isinstance(u, (int, float)) else None
        busy_rows.append([f"{names[k]} GPU", pct(u), pct(filler), pct(hi)])
    for k in order:
        d = nodes[k]
        busy_rows.append([f"{names[k]} CPU", pct(d.get("cpu_busy_1h", d.get("cpu_busy_5m"))), None, None])
    out.append(panel("pool-busy", "Both nodes, last hour: useful work, other busy, held idle", "bar", ["node", "useful", "busy, not useful", "held idle"],
                     busy_rows, producer="infra", unit="%", target=USEFUL_TARGET, target_label="useful GPU target", stacked=True,
                     updated_at=iso(now), source=src,
                     note="GPU: useful busy (busy times infra's useful share), busy on filler, and held idle (memory held or a lease, no work). "
                          "CPU: busy. Node 1's useful share is its busy share until infra's file carries n1."))

    hist = []
    if POOL_HISTORY.exists():
        cutoff = iso(now - timedelta(hours=24))
        hist = [h for h in (json.loads(ln) for ln in POOL_HISTORY.read_text().splitlines() if ln.strip()) if h.get("t", "") >= cutoff]
    hist.append({"t": iso(now), **{k: {"useful": useful(nodes[k]), "cpu": nodes[k].get("cpu_busy_5m")} for k in order}})
    tmp = POOL_HISTORY.with_name(f".{POOL_HISTORY.name}.tmp")
    tmp.write_text("".join(json.dumps(h) + "\n" for h in hist))
    os.replace(tmp, POOL_HISTORY)
    minutes = minutes or {}
    at = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()  # noqa: E731
    raw = lambda k, t: pct(raw_busy(minutes[k], t)) if minutes.get(k) else None  # noqa: E731
    cols = ["time"] + [c for k in ("n1", "n2") for c in (f"{names[k]} useful GPU %", f"{names[k]} raw GPU busy %")] + \
           [f"{names[k]} CPU %" for k in ("n1", "n2")]
    urows = [[h["t"]] + [c for k in ("n1", "n2") for c in (pct((h.get(k) or {}).get("useful")), raw(k, at(h["t"])))] +
             [pct((h.get(k) or {}).get("cpu")) for k in ("n1", "n2")] for h in hist]
    out.append(panel("pool-utilization", "Useful and raw GPU busy, and CPU busy, both nodes, last 24 hours", "line", cols, urows,
                     producer="infra", unit="%", target=USEFUL_TARGET, target_label="useful GPU target", updated_at=iso(now), source=src,
                     note="One point per 5-minute run, each over the hour before it. Useful: infra's useful GPU busy. Raw: the share of "
                          "readings with utilization above 0 (node 1: DCGM every 60 s; node 2: POUS's sampler every 10 s), the same "
                          "rule on both nodes. History starts when this panel did."))

    qrows = []
    for k in order:
        d = nodes[k]
        qrows.append([names[k], sig(d.get("ready_gpu_h"), 3), WATERMARK_GPU_H, d.get("waiting"), "yes" if d.get("timed_window") else "no"])
    tot = [n for n in (nodes[k].get("ready_gpu_h") for k in order) if isinstance(n, (int, float))]
    qrows.append(["both", sig(sum(tot), 3) if tot else None, 2 * WATERMARK_GPU_H,
                  sum(nodes[k].get("waiting") or 0 for k in order), ""])
    out.append(panel("pool-queue", "Work waiting: ready GPU-hours against the watermark", "table",
                     ["node", "ready GPU-h", "watermark GPU-h", "jobs waiting", "timed window now"], qrows, producer="infra",
                     updated_at=iso(now), source=src,
                     note=f"Ready GPU-h is queued work that could start now. Below the {WATERMARK_GPU_H} GPU-h watermark, a node can run dry."))

    grows = []
    for k in order:
        for g in nodes[k].get("gpus") or []:
            grows.append([names[k], g.get("index"), pct(g.get("busy_5m")), g.get("holder"), g.get("class")])
    out.append(panel("pool-gpus", "Every GPU now: busy and who holds it", "table", ["node", "GPU", "busy % (5 min)", "holder", "class"],
                     grows, producer="infra", updated_at=iso(now), source=src, note="The latest 5-minute reading per GPU."))

    krows = []
    for k in order:
        for r in nodes[k].get("kinds") or []:
            if str(r.get("kind", "")).startswith("timed:"):
                continue
            krows.append([names[k], r.get("kind"), (", ".join(o[:11] for o in r.get("owners") or []) or None) and ", ".join(o[:11] for o in r.get("owners") or [])[:200],
                          sig(r.get("leased_gpu_h"), 3), pct(r.get("efficiency")), sig(r.get("idle_gpu_h"), 3)])
    krows.sort(key=lambda r: -(r[5] or 0))
    if krows:
        out.append(panel("pool-kinds", "Job kinds, last 24 hours", "table",
                         ["node", "kind", "owners", "leased GPU-h", "efficiency %", "idle GPU-h"], krows[:200], producer="infra",
                         updated_at=iso(now), source=src))

    left_h = (TARGET_END - now).total_seconds() / 3600
    trows = []
    for k in order:
        u = useful(nodes[k])
        prev = [h for h in hist if h["t"] <= iso(now - timedelta(hours=1)) and (h.get(k) or {}).get("useful") is not None]
        trend = (u - prev[-1][k]["useful"]) if prev and isinstance(u, (int, float)) else None
        status = ("met" if u >= USEFUL_TARGET / 100 else "behind" if (trend or 0) <= 0 else "on track") if isinstance(u, (int, float)) else "no data"
        r = raw(k, now.timestamp())
        trows.append([f"{names[k]} useful GPU busy", f">= {USEFUL_TARGET}%", f"{pct(u)}%" if u is not None else None,
                      f"{r}%" if r is not None else None,
                      f"{'+' if (trend or 0) >= 0 else ''}{sig(100 * trend, 2)} pts/h" if trend is not None else None, status, TARGET_BY,
                      f"{sig(left_h, 3)} h" if left_h > 0 else "ended"])
    out.append(panel("targets", "Daniel's targets: how each is going", "table",
                     ["target", "goal", "now (last hour)", "raw busy, util > 0 (last hour)", "change over the hour", "status", "by",
                      "time left"], trows, producer="infra",
                     updated_at=iso(now), source=src,
                     note="Met: at or above the goal now. On track: below, and rising over the last hour. Behind: below and flat or falling."))
    age = (now - datetime.fromisoformat(gen.replace("Z", "+00:00"))).total_seconds() if isinstance(gen, str) else 0
    if age > POOL_STALE_S:
        errors.append(f"pool: {POOL} is {int(age)} s old")
    return out


# ---------------------------------------------------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--group", choices=("store", "node1"), required=True)
    ap.add_argument("--out", type=Path, default=HERE / "outbox", help="where each panel's JSON is written")
    ap.add_argument("--key", type=Path, default=Path("~/.config/verity/panels.key").expanduser(),
                    help="the panels:write key; absent means a dry run")
    ap.add_argument("--url", default=os.environ.get("PANELS_URL", URL))
    ap.add_argument("--store", type=Path, help="the evidence store's local root (default: $RESEARCH_STORE or ~/.research/store)")
    ap.add_argument("--research-src", type=Path, default=HERE / "research-src", help="a directory holding the `research` package")
    ap.add_argument("--no-refresh", action="store_true", help="read the store's catalog as it is")
    ns = ap.parse_args(argv)
    t0, now, errors = time.time(), datetime.now(timezone.utc).replace(microsecond=0), []
    ns.out.mkdir(parents=True, exist_ok=True)
    try:
        panels, info = (store_panels if ns.group == "store" else node1_panels)(ns, now, errors)
    except Exception as e:  # noqa: BLE001  nothing is sent; the site keeps the last good panels
        panels, info = [], {}
        errors.append(f"{ns.group}: {type(e).__name__}: {e} | {traceback.format_exc(limit=3).splitlines()[-3:]}")
    key = ns.key.read_text().strip() if ns.key.is_file() else None
    sent: dict[str, str] = {}
    for p in panels:
        bad = problems(p)
        if bad:
            errors.append(f"{p.get('id')}: invalid: {'; '.join(bad)}")
            continue
        write(ns.out, p)
        if key:
            sent[p["id"]] = publish(p, ns.url, key)
    refused = {k: v for k, v in sent.items() if v not in ("200", "201")}
    line = {"at": iso(now), "group": ns.group, "seconds": round(time.time() - t0, 1), "panels": len(panels),
            "published": (f"{len(sent) - len(refused)} of {len(sent)}" if key else "dry run (no key)"), **info,
            "refused": refused or None, "errors": errors or None}
    (ns.out / f"_last-run-{ns.group}.json").write_text(json.dumps(line, indent=1))
    print(json.dumps(line), flush=True)
    return 1 if errors or refused else 0


if __name__ == "__main__":
    sys.exit(main())
