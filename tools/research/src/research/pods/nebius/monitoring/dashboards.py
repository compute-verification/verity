"""The Grafana dashboards on vy-nebius-1, pinned by grafana.com id, revision and sha256 (Node Exporter Full is 469 KB, over the
repository's per-file limit, so grafana_up.sh fetches them on the server and this prepares them for provisioning).

    python3 dashboards.py OUT_DIR      fetch every pin, check its sha256, write OUT_DIR/<file> ready for Grafana's file provider

Preparing a dashboard: every `${DS_...}` import placeholder becomes the provisioned Prometheus data source; the old `graph` panels
(Angular, gone in Grafana 12) become `timeseries`; and DCGM queries select one scrape Service, because SkyPilot's Prometheus scrapes
the one dcgm-exporter pod through two Services (`dcgm-exporter`, `nvidia-dcgm-exporter`) and would sample every GPU twice.
Stdlib only.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

PINS = [
    {"file": "nvidia-dcgm.json", "id": 12239, "revision": 2,
     "sha256": "7964f18e887229776641132095f41df561b8e29323e6be3d4d2f7c681f2e4dfe"},
    {"file": "node-exporter-full.json", "id": 1860, "revision": 45,
     "sha256": "184c6b7409f306da75525d7772f71945b10cea23ad16b5d78c4698ea0ea51986"},
]
DATASOURCE = {"type": "prometheus", "uid": "prometheus"}
DCGM_SERVICE = "nvidia-dcgm-exporter"
PLACEHOLDER = re.compile(r"^\$\{DS_[A-Z0-9_]+\}$")


def url(pin: dict[str, Any]) -> str:
    return f"https://grafana.com/api/dashboards/{pin['id']}/revisions/{pin['revision']}/download"


def fetch(pin: dict[str, Any]) -> bytes:
    with urllib.request.urlopen(url(pin), timeout=60) as r:
        body = r.read()
    got = hashlib.sha256(body).hexdigest()
    if got != pin["sha256"]:
        raise ValueError(f"dashboard {pin['id']} revision {pin['revision']}: sha256 {got}, pinned {pin['sha256']}")
    return body


def _walk(node: Any) -> Any:
    if isinstance(node, dict):
        out = {k: _walk(v) for k, v in node.items()}
        if out.get("type") == "graph" and "targets" in out:
            out["type"] = "timeseries"
        if isinstance(out.get("expr"), str) and "DCGM_" in out["expr"]:
            out["expr"] = re.sub(r'(DCGM_\w+)\{', r'\1{service="%s", ' % DCGM_SERVICE, out["expr"])
        return out
    if isinstance(node, list):
        return [_walk(v) for v in node]
    if isinstance(node, str) and PLACEHOLDER.match(node):
        return dict(DATASOURCE)
    return node


def prepare(body: bytes) -> str:
    d = _walk(json.loads(body))
    d.pop("__inputs", None)
    d.pop("__requires", None)
    d["id"] = None
    return json.dumps(d, indent=1) + "\n"


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    out = Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    for pin in PINS:
        (out / pin["file"]).write_text(prepare(fetch(pin)))
        print(f"dashboard {pin['id']} r{pin['revision']} -> {out / pin['file']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
