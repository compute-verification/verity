#!/usr/bin/env bash
# ASK-64's per-step round (percall_rtt.py) on the serving node (vy-nebius-2), against one session of the challenger on its
# auditor host.  Start that session first, its layout a live window of ROUNDS rounds (default 131072); its tunnel opens
# 127.0.0.1:PORT here, which this waits for with `ss -ltn` (never a probe connect: the challenger serves the first
# connection only).  Extra arguments go to `percall_rtt.py run`.
#   research run --on vy-challenger-1 --project verity --source . --cwd source --timeout 30m --declared-output 'out/*' \
#     --env PEER=vy-nebius-2 --env PORT=7811 --env SCHEME=verity/pouw/epoch-coins/live/v1 --env EPOCH=1 \
#     --env FAMILIES="$(python3 benchmarks/pouw/served_h1/coins.py families --rounds 131072)" -- bash benchmarks/live_challenger/run.sh
#   research run --on vy-nebius-2 --project verity --source . --cwd source --phase cpu --cpus 2 --timeout 30m \
#     --env PORT=7811 --env ROUNDS=131072 -- bash benchmarks/hash/dither/percall_rtt.sh
#   research fetch CHALLENGER_RUN --all; research fetch CLIENT_RUN --all
#   python3 benchmarks/hash/dither/percall_rtt.py check --challenger-out ~/.research/runs/CHALLENGER_RUN/out \
#     --client ~/.research/runs/CLIENT_RUN/out/percall-rtt.json
set -uo pipefail
D=${RESEARCH_RUN_DIR:?}; OUT=$D/out; mkdir -p "$OUT"
PORT=${PORT:-7811}
git rev-parse HEAD 2>/dev/null | tee "$OUT/commit.txt"
echo "percall_rtt: client on $(hostname), waiting for the challenger's tunnel on 127.0.0.1:$PORT at $(date -u +%H:%M:%SZ)"
for _ in $(seq "${CHALLENGER_WAIT_S:-900}"); do
  ss -Hltn "sport = :$PORT" | grep -q . && break
  sleep 1
done
ss -Hltn "sport = :$PORT" | grep -q . || { echo "percall_rtt: NO_CHALLENGER on 127.0.0.1:$PORT"; exit 13; }
echo "percall_rtt: the tunnel is up at $(date -u +%H:%M:%SZ)"
"${PY:-python3}" benchmarks/hash/dither/percall_rtt.py run --challenger "127.0.0.1:$PORT" --rounds "${ROUNDS:-131072}" \
  --out "$OUT/percall-rtt.json" "$@"
rc=$?
echo "percall_rtt: exited $rc at $(date -u +%H:%M:%SZ)"
exit "$rc"
