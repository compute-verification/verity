#!/usr/bin/env bash
# ASK-64's per-step round (percall_rtt.py) on the serving node (vy-nebius-2), against one session of the challenger on its
# auditor host.  Request that session from infra's gate first, under a development period and a stand-in Commit, its
# layout a live window of ROUNDS rounds (default 32768) and its minutes longer than MAX_S:
#   python3 benchmarks/pouw/served_h1/coins.py open --gate USER@HOST --ssh-key KEY --period dev-NAME --commit RANDOM_HEX \
#     --peer vy-nebius-2 --epoch 1 --rounds 32768 --minutes 40
# It returns the session's id (SESSION) and the port its challenger listens on here (PORT), which this waits for with
# `ss -ltn` (never a probe connect: the challenger serves the first connection only).  The client opens rounds for at
# most MAX_S seconds (default 1800), then draws and closes, so the session ends inside its minutes (vy-nebius-2 to
# vy-challenger-1 is about 96 ms a round).  Extra arguments go to `percall_rtt.py run`.
#   research run --on vy-nebius-2 --project verity --source . --cwd source --timeout 45m --declared-output 'out/*' \
#     --env SESSION=ID --env PORT=PORT --env ROUNDS=32768 --env MAX_S=1800 -- bash benchmarks/hash/dither/percall_rtt.sh
#   ssh GATE fetch SESSION | tar xz; python3 benchmarks/hash/dither/percall_rtt.py check --challenger-out SESSION \
#     --client CLIENT_RUN/out/percall-rtt.json --key-record KEY_RECORD
set -uo pipefail
D=${RESEARCH_RUN_DIR:?}; OUT=$D/out; mkdir -p "$OUT"
PORT=${PORT:?the port the session tunnel opens here}; SESSION=${SESSION:?the session id from infra}
git rev-parse HEAD 2>/dev/null | tee "$OUT/commit.txt"
echo "percall_rtt: client on $(hostname), waiting for the challenger's tunnel on 127.0.0.1:$PORT at $(date -u +%H:%M:%SZ)"
for _ in $(seq $((${CHALLENGER_WAIT_S:-900} * 20))); do
  ss -Hltn "sport = :$PORT" | grep -q . && break
  sleep 0.05
done
ss -Hltn "sport = :$PORT" | grep -q . || { echo "percall_rtt: NO_CHALLENGER on 127.0.0.1:$PORT"; exit 13; }
echo "percall_rtt: the tunnel is up at $(date -u +%H:%M:%SZ)"
"${PY:-python3}" benchmarks/hash/dither/percall_rtt.py run --challenger "127.0.0.1:$PORT" --session "$SESSION" \
  --rounds "${ROUNDS:-32768}" --max-s "${MAX_S:-1800}" --out "$OUT/percall-rtt.json" "$@"
rc=$?
echo "percall_rtt: exited $rc at $(date -u +%H:%M:%SZ)"
exit "$rc"
