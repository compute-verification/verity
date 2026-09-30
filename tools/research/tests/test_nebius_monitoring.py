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


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"nebius_{name}", MON / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_exporter_publishes_each_queue_and_which_pod_is_in_it(tmp_path):
    ex = load("vy_exporter")
    assert ex.quantity("500m") == 0.5 and ex.quantity("384Gi") == 384 * 2**30 and ex.quantity(5) == 5
    cqs = {"items": [{"metadata": {"name": "circuits"},
                      "spec": {"stopPolicy": "Hold", "resourceGroups": [{"flavors": [{"resources": [
                          {"name": "nvidia.com/gpu", "nominalQuota": 5, "borrowingLimit": 2}, {"name": "cpu", "nominalQuota": "112"},
                          {"name": "memory", "nominalQuota": "1280Gi"}]}]}]},
                      "status": {"pendingWorkloads": 2, "admittedWorkloads": 3,
                                 "flavorsUsage": [{"resources": [{"name": "nvidia.com/gpu", "total": "3", "borrowed": "0"}]}]}}]}
    pods = {"items": [{"metadata": {"name": "gpu-7-head", "namespace": "default", "labels": {"kueue.x-k8s.io/queue-name": "circuits"}},
                       "status": {"phase": "Running"}},
                      {"metadata": {"name": "gpu-8-head", "namespace": "default", "labels": {"kueue.x-k8s.io/queue-name": "circuits"}},
                       "status": {"phase": "Pending"}}]}
    (tmp_path / "coverage").mkdir()
    for n in ("a.sh", "b.sh", ".tmp"):
        (tmp_path / "coverage" / n).write_text("#")
    text = ex.render(cqs, pods, ex.ready_counts(str(tmp_path)))
    for line in ('vy_queue_nominal{queue="circuits",resource="nvidia.com/gpu"} 5',
                 'vy_queue_borrowing_limit{queue="circuits",resource="nvidia.com/gpu"} 2',
                 f'vy_queue_nominal{{queue="circuits",resource="memory"}} {1280 * 2**30:g}',
                 'vy_queue_usage{queue="circuits",resource="nvidia.com/gpu"} 3', 'vy_queue_pending{queue="circuits"} 2',
                 'vy_queue_admitted{queue="circuits"} 3', 'vy_queue_stopped{queue="circuits"} 1',
                 'vy_pod_queue{holder_namespace="default",holder_pod="gpu-7-head",queue="circuits"} 1',
                 'vy_ready_jobs{lane="coverage"} 2', "vy_exporter_ok{} 1"):
        assert line in text.splitlines(), line
    assert "gpu-8-head" not in text and ex.ready_counts(str(tmp_path / "absent")) == {}
    assert "vy_exporter_ok{} 0" in ex.render({}, {}, {}, ok=False)


def test_a_firing_notification_becomes_one_markdown_note_for_root():
    sink = load("alert_sink")
    firing = {"commonLabels": {"alertname": "Pod holds a GPU at 0%"},
              "alerts": [{"status": "firing", "fingerprint": "f1", "startsAt": "2026-09-30T15:40:00.000Z",
                          "labels": {"alertname": "Pod holds a GPU at 0%", "gpu": "5", "pod": "gpu-209-head", "namespace": "default"},
                          "annotations": {"description": "Pod default/gpu-209-head has held GPU 5 at 0% for 10 minutes"}},
                         {"status": "resolved", "fingerprint": "f2", "labels": {"gpu": "1"}}]}
    name, text = sink.note(firing, 1790783000)
    assert re.fullmatch(r"\d{8}T\d{4}Z-alert-pod-holds-a-gpu-at-0-[0-9a-f]{8}\.md", name)
    assert text.startswith("---\nid: " + name[:-3] + "\n") and "\nlane: verity-root\n" in text and "\nkind: finding\n" in text
    assert "gpu 5, pod gpu-209-head, namespace default" in text and "since 2026-09-30T15:40:00Z" in text and "GPU 1" not in text
    assert sink.note({"alerts": [{"status": "resolved"}]}, 0) is None and sink.note({}, 0) is None


def test_the_idle_rules_deliver_to_the_markdown_sink_and_mute_the_quiet_hour():
    a = (MON / "alerting.yaml").read_text()
    assert "url: http://127.0.0.1:9403/grafana" in a and "type: webhook" in a
    assert 'start_time: "12:30"' in a and 'end_time: "13:30"' in a and "location: UTC" in a and "mute_time_intervals: [quiet-hour]" in a
    idle, held = a.split("- uid: nebius-gpu-idle-work-waiting")[1].split("- uid: nebius-gpu-held-at-zero")
    assert "for: 15m" in idle and 'expression: "$B < 5 && $V > 0"' in idle and "DCGM_FI_PROF_GR_ENGINE_ACTIVE" in idle
    assert "(sum(vy_queue_pending) or vector(0)) + (sum(vy_ready_jobs) or vector(0))" in idle
    assert "for: 10m" in held and 'pod!=""' in held and "evaluator: {type: lt, params: [1]}" in held
    assert a.count("datasourceUid: prometheus") == 3 and a.count("noDataState: OK") == 2
    y = (MON / "grafana.yaml").read_text()
    assert "mountPath: /etc/grafana/provisioning/alerting" in y and "configMap: {name: grafana-alerting}" in y
    up = (MON / "grafana_up.sh").read_text()
    assert 'cp "$here/nebius-overview.json" "$dash/"' in up and "create configmap grafana-alerting" in up


def test_the_overview_has_a_busy_panel_per_queue_on_the_provisioned_data_source():
    d = json.loads((MON / "nebius-overview.json").read_text())
    assert d["uid"] == "nebius-overview" and d["editable"] is False
    for p in d["panels"]:
        assert p["datasource"] == dashboards().DATASOURCE and all(t["datasource"] == p["datasource"] for t in p["targets"])
    busy = next(p for p in d["panels"] if p["title"].startswith("GPU busy % per Kueue queue"))
    expr = busy["targets"][0]["expr"]
    assert "group_left (queue)" in expr and '"pod", "$1", "holder_pod"' in expr and busy["targets"][0]["legendFormat"] == "{{queue}}"


def test_the_host_services_listen_privately_and_prometheus_finds_the_exporter():
    sh = (MON / "host_up.sh").read_text()
    subprocess.run(["bash", "-n", str(MON / "host_up.sh")], check=True)
    assert "--listen $ip:9402" in sh and "--listen 127.0.0.1:9403" in sh and "0.0.0.0" not in sh and "User=research" in sh
    e = (MON / "exporter.yaml").read_text()
    assert 'prometheus.io/scrape: "true"' in e and "addresses: [{ip: NODE_IP}]" in e and "selector:" not in e
