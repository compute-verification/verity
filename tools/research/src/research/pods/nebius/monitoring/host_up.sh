#!/usr/bin/env bash
# Install or update the monitoring stack's two host services on vy-nebius-1, as research (sudo), from a checkout on the server:
#   vy-exporter    vy_exporter.py on the node's private address, :9402 (Kueue queues, pod-to-queue map, ready files)
#   vy-alert-sink  alert_sink.py on 127.0.0.1:9403 (Grafana's webhook -> markdown notes in /workspace/monitoring/alerts/outbox)
# and exporter.yaml, which tells SkyPilot's Prometheus where the exporter is. Run grafana_up.sh after it (the alerts need both).
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
K=${VY_KUBECTL:-sudo k3s kubectl}
ip=$(ip -4 -o addr show eth0 | awk '{print $4}' | cut -d/ -f1)
[ -n "$ip" ] || { echo "host_up: no IPv4 address on eth0" >&2; exit 2; }
lib=/usr/local/lib/vy-monitoring
sudo install -d -m 755 "$lib"
sudo install -m 644 "$here/vy_exporter.py" "$here/alert_sink.py" "$here/lanes.tsv" "$lib/"
sudo install -d -o research -g research -m 755 /workspace/monitoring/alerts /workspace/monitoring/alerts/outbox
sudo tee /etc/systemd/system/vy-exporter.service > /dev/null <<EOF
[Unit]
Description=Kueue queue metrics for Prometheus (vy_exporter.py)
After=k3s.service
[Service]
ExecStart=/usr/bin/python3 $lib/vy_exporter.py --listen $ip:9402 --kubectl "k3s kubectl"
Restart=always
RestartSec=10
[Install]
WantedBy=multi-user.target
EOF
sudo tee /etc/systemd/system/vy-alert-sink.service > /dev/null <<EOF
[Unit]
Description=Grafana alert notifications as markdown notes (alert_sink.py)
[Service]
User=research
ExecStart=/usr/bin/python3 $lib/alert_sink.py --listen 127.0.0.1:9403 --dir /workspace/monitoring/alerts
Restart=always
RestartSec=10
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable -q vy-exporter.service vy-alert-sink.service
sudo systemctl restart vy-exporter.service vy-alert-sink.service
$K create namespace monitoring --dry-run=client -o yaml | $K apply -f - > /dev/null
sed "s/NODE_IP/$ip/" "$here/exporter.yaml" | $K apply -f -
for _ in $(seq 15); do curl -sf "http://$ip:9402/metrics" > /dev/null && break; sleep 1; done
curl -sf "http://$ip:9402/metrics" | grep -E "^vy_exporter_ok|^vy_queue_pending"
curl -sf -o /dev/null -w "alert sink: %{http_code}\n" -X POST -H 'Content-Type: application/json' -d '{"alerts": []}' http://127.0.0.1:9403/grafana
