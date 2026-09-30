"""The Nebius monitoring stack (research/pods/nebius/monitoring): Grafana on vy-nebius-1's loopback only, its admin password never
in argv or a manifest, and dashboards pinned by revision and sha256 and prepared for provisioning."""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
from pathlib import Path

from research.pods import registry

MON = Path(registry.__file__).resolve().parent / "nebius" / "monitoring"


def dashboards():
    spec = importlib.util.spec_from_file_location("nebius_dashboards", MON / "dashboards.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_grafana_listens_on_the_hosts_loopback_only():
    y = (MON / "grafana.yaml").read_text()
    assert "hostNetwork: true" in y and '{name: GF_SERVER_HTTP_ADDR, value: "127.0.0.1"}' in y
    assert '{name: GF_SERVER_HTTP_PORT, value: "3000"}' in y and "httpGet: {host: 127.0.0.1" in y
    assert "kind: Service" not in y and "hostPort" not in y and "NodePort" not in y      # nothing on the public IP
    assert "strategy: {type: Recreate}" in y and "dnsPolicy: ClusterFirstWithHostNet" in y
    assert re.search(r"image: grafana/grafana:\d+\.\d+\.\d+\n", y)                       # a pinned release, never latest
    assert "url: http://skypilot-prometheus-server.skypilot.svc.cluster.local" in y and "uid: prometheus" in y
    assert "kueue.x-k8s.io" not in y and "requests: {cpu: 100m, memory: 256Mi}" in y


def test_the_admin_password_comes_only_from_stdin_and_a_secret():
    y = (MON / "grafana.yaml").read_text()
    assert "secretKeyRef: {name: grafana-admin, key: password}" in y
    assert not re.search(r"GF_SECURITY_ADMIN_PASSWORD, value:", y)
    sh = (MON / "grafana_up.sh").read_text()
    assert "IFS= read -r line" in sh and '--from-file=password="$pw"' in sh and "--password-from-stdin" in sh
    assert "--from-literal" not in sh and "set -x" not in sh
    subprocess.run(["bash", "-n", str(MON / "grafana_up.sh")], check=True)
    r = subprocess.run(["bash", str(MON / "grafana_up.sh")], input="", capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", "VY_KUBECTL": "false"})
    assert r.returncode == 2 and "want the admin password on stdin" in r.stderr


def test_every_dashboard_is_pinned_by_revision_and_sha256():
    d = dashboards()
    assert {p["id"] for p in d.PINS} == {12239, 1860}                                     # NVIDIA DCGM, Node Exporter Full
    for p in d.PINS:
        assert re.fullmatch(r"[0-9a-f]{64}", p["sha256"]) and isinstance(p["revision"], int) and p["file"].endswith(".json")
        assert d.url(p) == f"https://grafana.com/api/dashboards/{p['id']}/revisions/{p['revision']}/download"


def test_a_dashboard_is_prepared_for_provisioning():
    d = dashboards()
    raw = {"__inputs": [{"name": "DS_PROMETHEUS"}], "id": 7, "uid": "x",
           "templating": {"list": [{"name": "instance", "datasource": "${DS_PROMETHEUS}"}]},
           "panels": [{"type": "graph", "datasource": "${DS_PROMETHEUS}",
                       "targets": [{"expr": 'avg(DCGM_FI_DEV_GPU_TEMP{instance=~"${instance}", gpu=~"${gpu}"})'}]},
                      {"type": "stat", "targets": [{"expr": "node_load1"}], "datasource": "${ds_prometheus}"}]}
    out = json.loads(d.prepare(json.dumps(raw).encode()))
    assert "__inputs" not in out and out["id"] is None and out["uid"] == "x"
    assert out["templating"]["list"][0]["datasource"] == d.DATASOURCE and out["panels"][0]["datasource"] == d.DATASOURCE
    assert out["panels"][0]["type"] == "timeseries" and out["panels"][1]["type"] == "stat"
    assert out["panels"][0]["targets"][0]["expr"] == \
        'avg(DCGM_FI_DEV_GPU_TEMP{service="nvidia-dcgm-exporter", instance=~"${instance}", gpu=~"${gpu}"})'
    assert out["panels"][1]["datasource"] == "${ds_prometheus}"                          # a dashboard's own variable stays
