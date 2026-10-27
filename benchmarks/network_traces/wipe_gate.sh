#!/usr/bin/env bash
# Network accounting's side of a memory wipe: one warden session through vy-challenger-1's gate (`warden-rehearse --scan
# --prompts`, the toy's class) spanning the wipe's session, under the lane's gate key (GATE_KEY), inside a `research run`
# on the machine that holds it (vy-nebius-cpu-2), cwd = the source tree (`--cwd source`); everything into $OUT.
#
# With AT (a UTC time) it first waits until then, so a run launched early fires on the clock with no VM awake: give the
# run a --timeout past AT, MINUTES and the fetch. Launched more than LATE_S (60) seconds after AT it sends nothing. It
# sends the gate REQUESTS prompts of the toy's class (`demo_stub.py prompts`), the toy (`demo_stub.py serve`) already
# listening on PEER's loopback PORT, and writes the gate's answer to start.json; a refusal (another rehearsal still runs)
# is a visible failure, never retried. Then `status` every 15 s until the run ends (status.json), and its bundle with
# witness/, the lane's own (`fetch`), to bundle.tgz. Exits with the gate run's rc.
#
# 400 requests, the gate's most, are 259 link-windows at its fixed seed (`warden_serve.prompt_traffic`): 828.8 s from
# the session's origin, which is about 15 s after the request at HEIGHT 12. A wipe whose [t0, t1) lies within
# [origin + 1 s, origin + 827.8 s) is covered; then
#
#   python benchmarks/network_traces/wipe_egress.py $OUT/bundle.tgz --t0-ns T0 --t1-ns T1 --out egress.json
#
# writes its `wipe-egress/v1` record.
set -euo pipefail
: "${OUT:?}"
GATE_KEY=${GATE_KEY:-$HOME/.ssh/vy-gate-network-accounting}
CHALLENGER=${CHALLENGER:-research@204.12.178.84}
PEER=${PEER:-vy-dc-h200-2} PORT=${PORT:-18776} REQUESTS=${REQUESTS:-400} HEIGHT=${HEIGHT:-12} MINUTES=${MINUTES:-20}
LATE_S=${LATE_S:-60}
SSH=(ssh -T -o BatchMode=yes -o IdentitiesOnly=yes -o ServerAliveInterval=15 -i "$GATE_KEY" "$CHALLENGER")
REQUEST="warden-rehearse --peer $PEER --port $PORT --minutes $MINUTES --requests $REQUESTS --height $HEIGHT --scan"
REQUEST+=" --prompts --tokens 24"

if [ -n "${AT:-}" ]; then
  at=$(date -u -d "$AT" +%s) || { echo "AT $AT isn't a time"; exit 2; }
  now=$(date -u +%s)
  [ "$now" -le $((at + LATE_S)) ] || { echo "$(date -u +%FT%TZ): past AT $AT by more than $LATE_S s: nothing sent"; exit 3; }
  echo "$(date -u +%FT%TZ): waiting until $AT"
  [ "$now" -ge "$at" ] || sleep $((at - now))
fi
echo "$(date -u +%FT%TZ): $REQUEST"
PYTHONPATH=$PWD python3 benchmarks/network_traces/demo_stub.py prompts --count "$REQUESTS" | "${SSH[@]}" "$REQUEST" \
  | tee "$OUT/start.json"
RUN=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["run"])' "$OUT/start.json")
echo "$(date -u +%FT%TZ): gate run $RUN"
until "${SSH[@]}" "status $RUN" > "$OUT/status.json" \
    && python3 -c 'import json, sys; sys.exit(json.load(open(sys.argv[1]))["end"] is None)' "$OUT/status.json"; do
  sleep 15
done
"${SSH[@]}" "fetch $RUN" > "$OUT/bundle.tgz"
python3 - "$OUT/status.json" "$OUT/bundle.tgz" <<'EOF'
import json, sys, tarfile
end = json.load(open(sys.argv[1]))["end"]
with tarfile.open(sys.argv[2], "r:gz") as tar:
    dep = json.load(tar.extractfile(next(m for m in tar if m.name.endswith("/out/deployment.json"))))
n, w = dep["windows"], dep["params"]["window"] * dep["params"]["bucket_ms"] / 1000
print(json.dumps({"run": end.get("run"), "rc": end.get("rc"), "ended_utc": end.get("ended_utc"),
                  "accepted": (end.get("verdict") or {}).get("accepted"), "origin_unix": dep["origin_unix"],
                  "windows": n, "session_s": round(n * w, 1)}))
sys.exit(end.get("rc") or 0)
EOF
