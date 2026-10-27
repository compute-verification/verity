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
# witness/, the lane's own (`fetch`), to KEEP/RUN/fetch.tar.gz (0700, outside the run dir: custody publishes $OUT), and
# to bundle.tgz only what `fetch` gives any other key, the bundle less witness/. Exits with the gate run's rc.
#
# 1700 requests, the gate's most (`auditor.WARDEN_REQUESTS`), on the gate's most prompts, 400 (`WARDEN_PROMPTS`; request
# j sends the (j mod n)-th), are 1064 link-windows at its fixed seed (`warden_serve.prompt_traffic`): 3404.8 s from the
# session's origin, which is about 16 s after the request at HEIGHT 13. At about 5 one-time signatures a link-window
# they take about 5300 of height 13's 8192 (height 12's 4096 sign about 819 windows). A gate pinned before 1700 takes
# 400 requests at most, 259 link-windows: give it REQUESTS=400 HEIGHT=12 MINUTES=20. A wipe whose [t0, t1) lies within
# [origin + 1 s, origin + 3403.8 s) is covered; then
#
#   python benchmarks/network_traces/wipe_egress.py $OUT/bundle.tgz --t0-ns T0 --t1-ns T1 --out egress.json
#
# writes its `wipe-egress/v1` record.
set -euo pipefail
: "${OUT:?}"
GATE_KEY=${GATE_KEY:-$HOME/.ssh/vy-gate-network-accounting}
CHALLENGER=${CHALLENGER:-research@204.12.178.84}
PEER=${PEER:-vy-dc-h200-2} PORT=${PORT:-18776} REQUESTS=${REQUESTS:-1700} HEIGHT=${HEIGHT:-13} MINUTES=${MINUTES:-60}
PROMPTS=$((REQUESTS < 400 ? REQUESTS : 400))
LATE_S=${LATE_S:-60}
SSH=(ssh -T -o BatchMode=yes -o IdentitiesOnly=yes -o ServerAliveInterval=15 -i "$GATE_KEY" "$CHALLENGER")
KEEP=$(realpath -m "${KEEP:-/workspace/research/lanes/network-accounting/gate}")
case "$KEEP/" in "$(realpath -m "${RESEARCH_RUN_DIR:-$OUT}")/"*|"$(realpath -m "$OUT")/"*)
  echo "KEEP $KEEP is inside the run dir, which custody publishes: give a directory outside it"; exit 2;; esac
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
PYTHONPATH=$PWD python3 benchmarks/network_traces/demo_stub.py prompts --count "$PROMPTS" | "${SSH[@]}" "$REQUEST" \
  | tee "$OUT/start.json"
RUN=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["run"])' "$OUT/start.json")
echo "$(date -u +%FT%TZ): gate run $RUN"
until "${SSH[@]}" "status $RUN" > "$OUT/status.json" \
    && python3 -c 'import json, sys; sys.exit(json.load(open(sys.argv[1]))["end"] is None)' "$OUT/status.json"; do
  sleep 15
done
mkdir -p "$KEEP/$RUN" && chmod 700 "$KEEP" "$KEEP/$RUN"
(umask 077 && "${SSH[@]}" "fetch $RUN" > "$KEEP/$RUN/fetch.tar.gz")
python3 - "$OUT/status.json" "$KEEP/$RUN/fetch.tar.gz" "$OUT/bundle.tgz" <<'EOF'
import json, sys, tarfile
end = json.load(open(sys.argv[1]))["end"]
with tarfile.open(sys.argv[2], "r:gz") as tar, tarfile.open(sys.argv[3], "w:gz") as public:
    for m in tar:
        if "witness" not in m.name.split("/") and (m.isfile() or m.isdir()):
            public.addfile(m, tar.extractfile(m) if m.isfile() else None)
with tarfile.open(sys.argv[3], "r:gz") as tar:
    dep = json.load(tar.extractfile(next(m for m in tar if m.name.endswith("/out/deployment.json"))))
n, w = dep["windows"], dep["params"]["window"] * dep["params"]["bucket_ms"] / 1000
print(json.dumps({"run": end.get("run"), "rc": end.get("rc"), "ended_utc": end.get("ended_utc"),
                  "accepted": (end.get("verdict") or {}).get("accepted"), "origin_unix": dep["origin_unix"],
                  "windows": n, "session_s": round(n * w, 1)}))
sys.exit(end.get("rc") or 0)
EOF
