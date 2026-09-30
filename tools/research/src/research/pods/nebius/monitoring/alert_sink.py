"""Grafana's alert notifications as markdown notes: a webhook receiver on vy-nebius-1's loopback (host_up.sh installs it).

    python3 alert_sink.py [--listen 127.0.0.1:9403] [--dir /workspace/monitoring/alerts]

Grafana's contact point `markdown-files` (alerting.yaml) POSTs each notification to /grafana. A notification with at least one firing
alert becomes one note, <dir>/outbox/<YYYYMMDDTHHMMZ>-alert-<rule>-<id>.md, with the lane contract's front matter, for root's lane
(`lane: verity-root`). The steward's channel sync copies new notes into the store and research-notes. Every notification, resolved
ones included, is also a line of <dir>/alerts.jsonl. The server keeps the record whether or not an agent VM is up to carry it on.
Stdlib only.
"""
from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import os
import re
import time
from typing import Any

LANE = "verity-root"


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:48] or "alert"


def note(payload: dict[str, Any], now: float) -> tuple[str, str] | None:
    """(file name, markdown) for a notification with firing alerts, else None."""
    alerts = payload.get("alerts") or []
    firing = [a for a in alerts if a.get("status") == "firing"]
    if not firing:
        return None
    stamp = time.strftime("%Y%m%dT%H%MZ", time.gmtime(now))
    rule = (payload.get("commonLabels") or {}).get("alertname") or (firing[0].get("labels") or {}).get("alertname", "alert")
    ident = hashlib.sha256(json.dumps(sorted(a.get("fingerprint", "") for a in firing)).encode()).hexdigest()[:8]
    name = f"{stamp}-alert-{_slug(rule)}-{ident}.md"
    lines = [
        "---",
        f"id: {name[:-3]}",
        "campaign: nebius-monitoring",
        f"lane: {LANE}",
        "kind: finding",
        "status: open",
        "repo: danielreuter/verity",
        "origin: grafana on vy-nebius-1 (alert_sink.py; rules in pods/nebius/monitoring/alerting.yaml)",
        "---",
        "",
        f"# {rule}: {len(firing)} firing on vy-nebius-1",
        "",
    ]
    for a in firing:
        lab = a.get("labels") or {}
        ann = a.get("annotations") or {}
        where = ", ".join(f"{k} {lab[k]}" for k in ("gpu", "pod", "namespace", "queue") if lab.get(k))
        text = ann.get("description") or ann.get("summary") or ""
        lines.append(f"- **{where or lab.get('alertname', rule)}**, since {a.get('startsAt', '?')[:19]}Z: {text}".rstrip(": "))
    lines += ["", "Open http://127.0.0.1:3000 through `ssh -N -L 3000:127.0.0.1:3000 research@81.85.2.165` (Nebius folder)."]
    return name, "\n".join(lines) + "\n"


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
            made = note(payload, now)
            if made:
                path = os.path.join(outbox, made[0])
                with open(path + ".tmp", "w") as f:
                    f.write(made[1])
                os.replace(path + ".tmp", path)
            self.send_response(204)
            self.end_headers()

        def log_message(self, *args: Any) -> None:
            pass

    host, port = a.listen.rsplit(":", 1)
    http.server.ThreadingHTTPServer((host, int(port)), Handler).serve_forever()


if __name__ == "__main__":
    main()
