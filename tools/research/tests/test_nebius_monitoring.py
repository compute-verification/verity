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


def test_a_firing_notification_becomes_markdown_notes_naming_the_job_and_its_lane(tmp_path):
    sink = load("alert_sink")
    assert sink.job_id_of("gpu-209-ce1b86e4-head") == 209 and sink.job_id_of("m0-v3-a13-198-ce1b86e4-head") == 198
    assert sink.job_id_of("grafana-7d9f") is None
    table = sink.lane_table(MON / "lanes.tsv")
    assert sink.lane_of("cov-g003", table) == "vllm-epoch-run" and sink.lane_of("m0-v3-a15b", table) == "flock-netlist (M0)"
    assert sink.lane_of("sm120-qwen05-gb2-b1-2", table) is None and sink.lane_of(None, table) is None
    runs = tmp_path / "runs"
    (runs / "r1").mkdir(parents=True)
    (runs / "r1" / "events.jsonl").write_text('{"id": "run#1@gpu-209-ce1b86e4-head:11:22"}\n')
    (runs / "r1" / "job.json").write_text(json.dumps({"store": {"params": {"row": "phi3__b8"}}}))
    (runs / "r2").mkdir()
    (runs / "r2" / "events.jsonl").write_text('{"id": "run#1@gpu-5-ce1b86e4-head:1:2"}\n')
    (runs / "r2" / "job.json").write_text(json.dumps({"execution": {"command": ["py", "-m", "cli", "row", "stage", "commit", "q__b1", "R"]}}))
    assert sink.pod_row("gpu-209-ce1b86e4-head", str(runs)) == "phi3__b8" and sink.pod_row("gpu-5-ce1b86e4-head", str(runs)) == "q__b1"
    assert sink.pod_row("gpu-7-ce1b86e4-head", str(runs)) is None
    tl = tmp_path / "timeline.jsonl"
    tl.write_text("".join(json.dumps(e) + "\n" for e in (
        {"pid": 1, "stage": "engine.build", "ev": "start"}, {"pid": 1, "stage": "engine.build", "ev": "end"},
        {"pid": 1, "stage": "validate.sampled_replay", "ev": "start"})))
    assert sink.open_stages(tl) == {"validate.sampled_replay"} and sink.open_stages(tmp_path / "absent") == set()
    ctx = {"gpu-209-ce1b86e4-head": {"job": "cov-g003", "lane": "vllm-epoch-run", "row": "phi3__b8", "open": {"validate.sampled_replay"}},
           "gpu-214-ce1b86e4-head": {"job": "cov-k23-2", "lane": "vllm-epoch-run", "row": "q__b1", "open": {"committer_setup"}}}
    alert = lambda pod, gpu, fp: {"status": "firing", "fingerprint": fp, "startsAt": "2026-09-30T15:40:00.000Z",  # noqa: E731
                                  "labels": {"alertname": "Pod holds a GPU at 0%", "gpu": gpu, "pod": pod, "namespace": "default"},
                                  "annotations": {"description": f"Pod default/{pod} has held GPU {gpu} at 0% for 10 minutes"}}
    payload = {"commonLabels": {"alertname": "Pod holds a GPU at 0%"},
               "alerts": [alert("gpu-209-ce1b86e4-head", "5", "f1"), alert("gpu-214-ce1b86e4-head", "3", "f2"),
                          {"status": "resolved", "fingerprint": "f3", "labels": {"gpu": "1"}}]}
    (stall_name, stall), (replay_name, replay) = sink.notes(payload, 1790783000, ctx.get)
    assert re.fullmatch(r"\d{8}T\d{4}Z-alert-pod-holds-a-gpu-at-0-[0-9a-f]{8}\.md", stall_name)
    assert re.fullmatch(r"\d{8}T\d{4}Z-alert-replay-on-gpu-[0-9a-f]{8}\.md", replay_name)
    for name, text in ((stall_name, stall), (replay_name, replay)):
        assert text.startswith("---\nid: " + name[:-3] + "\n") and "\nlane: infra\n" in text and "\nkind: finding\n" in text
    assert "gpu 3, pod gpu-214-ce1b86e4-head, namespace default** (job cov-k23-2; lane vllm-epoch-run; row q__b1)" in stall
    assert "gpu-209" not in stall and "GPU 1" not in stall and "tags:" not in stall
    assert "\ntags: [replay-on-gpu]\n" in replay and "[replay-on-gpu]: 1 firing" in replay and "gpu 5, pod gpu-209-ce1b86e4-head" in replay
    assert sink.notes({"alerts": [{"status": "resolved"}]}, 0) == [] and sink.notes({}, 0) == []
    assert len(sink.notes({"alerts": [alert("x-1-abcdef12-head", "0", "f9")]}, 0)) == 1          # no context: one plain note


def test_the_idle_rules_deliver_to_the_markdown_sink_and_mute_the_quiet_hour():
    a = (MON / "alerting.yaml").read_text()
    assert "url: http://127.0.0.1:9403/grafana" in a and "type: webhook" in a
    assert 'start_time: "12:30"' in a and 'end_time: "13:30"' in a and "location: UTC" in a and "mute_time_intervals: [quiet-hour]" in a
    idle, held = a.split("- uid: nebius-gpu-idle-work-waiting")[1].split("- uid: nebius-gpu-held-at-zero")
    assert "for: 15m" in idle and 'expression: "$B < 5 && $V > 0"' in idle and "DCGM_FI_PROF_GR_ENGINE_ACTIVE" in idle
    assert '(sum(vy_queue_pending{queue!~"shadow-.*"}) or vector(0)) + (sum(vy_ready_jobs) or vector(0))' in idle
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
