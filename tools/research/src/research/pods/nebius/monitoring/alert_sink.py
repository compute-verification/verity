"""Grafana's alert notifications as markdown notes: a webhook receiver on vy-nebius-1's loopback (host_up.sh installs it).

    python3 alert_sink.py [--listen 127.0.0.1:9403] [--dir /workspace/monitoring/alerts]

Grafana's contact point `markdown-files` (alerting.yaml) POSTs each notification to /grafana. Its firing alerts become notes,
<dir>/outbox/<YYYYMMDDTHHMMZ>-alert-<rule>-<id>.md, with the lane contract's front matter, for the infra coordinator's lane
(`lane: infra`; the steward's alert_pull.sh also copies each to node1-dispatcher, which owns utilization). Each alert names the pod's SkyPilot job
and, from the job name's prefix (lanes.tsv), its owning lane. A Commit whose timeline shows `validate.sampled_replay` open is running
its CPU replay while holding the GPU, by design until Daniel rules on moving replay off the GPU: those alerts go in a separate note
tagged `replay-on-gpu`, apart from real stalls (they still count as idle in every busy % number). Every notification, resolved ones
included, is also a line of <dir>/alerts.jsonl. Stdlib only.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import http.server
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

LANE = "infra"          # the infra coordinator (bc-17cc41f1) since 18:45Z; verity-root before
REPLAY = "validate.sampled_replay"
RUNS = "/workspace/jobs/runs"
ROW_GLOBS = ("/workspace/cp/*/{row}/timeline.jsonl", "/workspace/jobs/*/{row}/timeline.jsonl", "/workspace/jobs/*/*/{row}/timeline.jsonl")
SKY_PY = "/home/research/.local/share/uv/tools/skypilot/bin/python"
HERE = Path(__file__).resolve().parent

Context = Callable[[str], dict[str, Any]]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:48] or "alert"


def lane_table(path: Path) -> list[tuple[str, str]]:
    out = []
    try:
        for ln in path.read_text().splitlines():
            if ln.strip() and not ln.startswith("#") and "\t" in ln:
                prefix, lane = ln.split("\t", 1)
                out.append((prefix, lane.strip()))
    except OSError:
        pass
    return sorted(out, key=lambda x: -len(x[0]))


def lane_of(job: str | None, table: list[tuple[str, str]]) -> str | None:
    return next((lane for prefix, lane in table if job and job.startswith(prefix)), None)


def job_id_of(pod: str) -> int | None:
    """SkyPilot names a job's pod <task-or-job>-<job id>-<8 hex>-head."""
    m = re.fullmatch(r".+-(\d+)-[0-9a-f]{8}-head", pod)
    return int(m.group(1)) if m else None


def open_stages(timeline: Path) -> set[str]:
    """Stages that started and haven't ended, by (pid, stage), in a verity-sweep timeline."""
    live: dict[tuple[Any, str], bool] = {}
    try:
        for ln in timeline.read_text().splitlines():
            try:
                e = json.loads(ln)
            except ValueError:
                continue
            key = (e.get("pid"), e.get("stage"))
            if e.get("ev") == "start":
                live[key] = True
            elif e.get("ev") == "end":
                live.pop(key, None)
    except OSError:
        pass
    return {stage for _, stage in live}


def pod_row(pod: str, runs: str = RUNS, since_s: float = 2 * 86400) -> str | None:
    """The row of the newest research run whose events name this pod (row stage / row run)."""
    now = time.time()
    try:
        dirs = sorted((d for d in os.scandir(runs) if d.is_dir()), key=lambda d: d.stat().st_mtime, reverse=True)
    except OSError:
        return None
    for d in dirs[:400]:
        if now - d.stat().st_mtime > since_s:
            break
        try:
            with open(os.path.join(d.path, "events.jsonl")) as f:
                if f"@{pod}:" not in f.read(8192):
                    continue
            j = json.load(open(os.path.join(d.path, "job.json")))
        except (OSError, ValueError):
            continue
        row = ((j.get("store") or {}).get("params") or {}).get("row")
        cmd = (j.get("execution") or {}).get("command") or []
        if not row and "row" in cmd:
            rest = cmd[cmd.index("row") + 1:]
            row = rest[2] if rest[:1] == ["stage"] and len(rest) > 2 else rest[1] if rest[:1] == ["run"] and len(rest) > 1 else None
        return row
    return None


def row_timeline(row: str, globs: tuple[str, ...] = ROW_GLOBS) -> Path | None:
    found = [Path(p) for g in globs for p in glob.glob(g.format(row=row))]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def sky_jobs() -> dict[int, str]:
    """job id -> job name, from the API server on this host (SkyPilot's SDK), {} when it can't be read."""
    code = ("import json, sky\nfrom sky import jobs\n"
            "print(json.dumps({r['job_id']: r['job_name'] for r in sky.get(jobs.queue(refresh=False, skip_finished=False, all_users=True))}))")
    try:
        r = subprocess.run([SKY_PY, "-c", code], capture_output=True, text=True, timeout=60)
        return {int(k): v for k, v in json.loads(r.stdout.strip().splitlines()[-1]).items()}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return {}


def host_context(table_path: Path = HERE / "lanes.tsv") -> Context:
    table, names = lane_table(table_path), sky_jobs()

    def ctx(pod: str) -> dict[str, Any]:
        jid = job_id_of(pod)
        job = names.get(jid) if jid is not None else None
        row = pod_row(pod)
        tl = row_timeline(row) if row else None
        return {"job": job, "job_id": jid, "lane": lane_of(job, table), "row": row, "open": open_stages(tl) if tl else set()}
    return ctx


def notes(payload: dict[str, Any], now: float, ctx: Context | None = None) -> list[tuple[str, str]]:
    """(file name, markdown) per class of firing alert in a notification: real stalls, and replay-on-gpu."""
    alerts = [a for a in payload.get("alerts") or [] if a.get("status") == "firing"]
    if not alerts:
        return []
    stamp = time.strftime("%Y%m%dT%H%MZ", time.gmtime(now))
    rule = (payload.get("commonLabels") or {}).get("alertname") or (alerts[0].get("labels") or {}).get("alertname", "alert")
    info = {id(a): (ctx((a.get("labels") or {})["pod"]) if ctx and (a.get("labels") or {}).get("pod") else {}) for a in alerts}
    replay = [a for a in alerts if REPLAY in (info[id(a)].get("open") or ())]
    stalls = [a for a in alerts if a not in replay]
    out = []
    for group, tag in ((stalls, None), (replay, "replay-on-gpu")):
        if not group:
            continue
        ident = hashlib.sha256(json.dumps(sorted(a.get("fingerprint", "") for a in group)).encode()).hexdigest()[:8]
        name = f"{stamp}-alert-{_slug(tag or rule)}-{ident}.md"
        head = f"# {rule}{' [replay-on-gpu]' if tag else ''}: {len(group)} firing on vy-nebius-1"
        lines = ["---", f"id: {name[:-3]}", "campaign: nebius-monitoring", f"lane: {LANE}", "kind: finding", "status: open",
                 *([f"tags: [{tag}]"] if tag else []), "repo: compute-verification/verity",
                 "origin: grafana on vy-nebius-1 (alert_sink.py; rules in pods/nebius/monitoring/alerting.yaml)", "---", "", head, ""]
        if tag:
            lines += [f"These Commits are in `{REPLAY}`: the CPU replay runs while the pod still holds its GPU, by design until Daniel rules "
                      "on moving replay off the GPU. Not a stall, but the GPU counts as idle.", ""]
        for a in group:
            lab, ann, c = a.get("labels") or {}, a.get("annotations") or {}, info[id(a)]
            where = ", ".join(f"{k} {lab[k]}" for k in ("gpu", "pod", "namespace", "queue") if lab.get(k))
            owner = "; ".join(x for x in (f"job {c['job']}" if c.get("job") else "", f"lane {c['lane']}" if c.get("lane") else "",
                                          f"row {c['row']}" if c.get("row") else "") if x)
            text = ann.get("description") or ann.get("summary") or ""
            lines.append(f"- **{where or lab.get('alertname', rule)}**{f' ({owner})' if owner else ''}, since "
                         f"{a.get('startsAt', '?')[:19]}Z: {text}".rstrip(": "))
        lines += ["", "Open http://127.0.0.1:3000 through `ssh -N -L 3000:127.0.0.1:3000 research@81.85.2.165` (Nebius folder)."]
        out.append((name, "\n".join(lines) + "\n"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--listen", default="127.0.0.1:9403")
    ap.add_argument("--dir", default="/workspace/monitoring/alerts")
    a = ap.parse_args()
    outbox = os.path.join(a.dir, "outbox")
    os.makedirs(outbox, exist_ok=True)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            try:
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            except ValueError:
                self.send_error(400)
                return
            now = time.time()
            with open(os.path.join(a.dir, "alerts.jsonl"), "a") as f:
                f.write(json.dumps({"received": now, "payload": payload}) + "\n")
            firing = any(x.get("status") == "firing" for x in payload.get("alerts") or [])
            for name, text in notes(payload, now, host_context() if firing else None):
                path = os.path.join(outbox, name)
                with open(path + ".tmp", "w") as f:
                    f.write(text)
                os.replace(path + ".tmp", path)
            self.send_response(204)
            self.end_headers()

        def log_message(self, *args: Any) -> None:
            pass

    host, port = a.listen.rsplit(":", 1)
    http.server.ThreadingHTTPServer((host, int(port)), Handler).serve_forever()


if __name__ == "__main__":
    main()
