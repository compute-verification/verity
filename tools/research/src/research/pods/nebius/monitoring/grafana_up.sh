#!/usr/bin/env bash
# Install or update Grafana on vy-nebius-1 (grafana.yaml), from a checkout on the server, as research. The admin password comes on
# stdin, one line, never argv or the log:
#
#   research pods ssh vy-nebius-1 -- bash <tree>/tools/research/src/research/pods/nebius/monitoring/grafana_up.sh < password-file
#
# Run host_up.sh first (the exporter and the alert sink the rules use). It creates the admin Secret, fetches the pinned dashboards
# (dashboards.py) and adds nebius-overview.json into the grafana-dashboards ConfigMap, puts alerting.yaml in grafana-alerting, applies
# grafana.yaml, waits for the rollout, sets the admin password in Grafana's database too (GF_SECURITY_ADMIN_PASSWORD applies only
# when that database is first created), and checks http://127.0.0.1:3000. Running it again changes only what changed.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
K=${VY_KUBECTL:-sudo k3s kubectl}
pw=$(mktemp)
dash=$(mktemp -d)
trap 'rm -rf "$pw" "$dash"' EXIT
chmod 600 "$pw"
IFS= read -r line || true
printf '%s' "$line" > "$pw"
unset line
[ -s "$pw" ] || { echo "grafana_up: want the admin password on stdin" >&2; exit 2; }
sudo install -d -o 472 -g 0 -m 750 /workspace/monitoring/grafana
$K create namespace monitoring --dry-run=client -o yaml | $K apply -f - > /dev/null
$K -n monitoring create secret generic grafana-admin --from-file=password="$pw" --dry-run=client -o yaml | $K apply -f - > /dev/null
python3 "$here/dashboards.py" "$dash"
cp "$here/nebius-overview.json" "$dash/"
$K -n monitoring create configmap grafana-dashboards --from-file="$dash" --dry-run=client -o yaml \
  | $K apply --server-side --force-conflicts -f -                                             # 469 KB: too big for client-side apply
$K -n monitoring create configmap grafana-alerting --from-file=alerting.yaml="$here/alerting.yaml" --dry-run=client -o yaml \
  | $K apply -f - > /dev/null
$K apply -f "$here/grafana.yaml"
$K -n monitoring rollout restart deploy/grafana > /dev/null                                   # provisioning is read at start
$K -n monitoring rollout status deploy/grafana --timeout=300s
$K -n monitoring exec -i deploy/grafana -- grafana cli --homepath /usr/share/grafana admin reset-admin-password --password-from-stdin \
  < "$pw" > /dev/null
for _ in $(seq 30); do curl -sf http://127.0.0.1:3000/api/health > /dev/null && break; sleep 2; done
curl -sf http://127.0.0.1:3000/api/health
echo "grafana_up: http://127.0.0.1:3000 on this host; from a laptop: ssh -N -L 3000:127.0.0.1:3000 research@81.85.2.165"
