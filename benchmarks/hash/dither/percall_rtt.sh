#!/usr/bin/env bash
# ASK-64's per-step round (percall_rtt.py) on the serving node (vy-nebius-2), against one session of the challenger on its
# auditor host.  Take that session from infra's gate first with the lane's own gate key, in the lane's open development
# period (period-open --name dev-NAME --cap N --circuit SHA512), its layout a live window of ROUNDS rounds (default 32768)
# and its minutes longer than MAX_S:
#   python3 benchmarks/pouw/served_h1/coins.py open --gate USER@HOST --ssh-key KEY --peer vy-nebius-2 --epoch 1 \
#     --rounds 32768 --minutes 40 --secret-file SECRET
# It reserves the session (SESSION) and opens it on its stand-in root (`coins.stand_in_root`, never a window's), and
# prints the port its challenger listens on here (PORT), which this waits for with `ss -ltn` (never a probe connect: the
# challenger serves the first connection whose hello it accepts), and the root of the coin commitment its challenger
# signed (COMMITMENT), which the client holds the hello's to; the session's secret, which the hello carries, is in
# the new file SECRET (mode 600, never printed), and goes to the node on ssh's stdin, never as an argument, to the file
# this reads (SECRET_FILE, default ~/.local/state/pouw-challenger/SESSION.secret there):
#   research pods ssh vy-nebius-2 -- 'umask 077; mkdir -p ~/.local/state/pouw-challenger &&
#     cat > ~/.local/state/pouw-challenger/SESSION.secret' < SECRET
# The client opens rounds for at most MAX_S seconds (default 1800), then draws and closes, so the session ends inside its
# minutes (vy-nebius-2 to vy-challenger-1 is about 96 ms a round).  Extra arguments go to `percall_rtt.py run`.
#   research run --on vy-nebius-2 --project verity --source . --cwd source --timeout 45m --declared-output 'out/*' \
#     --env SESSION=ID --env PORT=PORT --env COMMITMENT=HEX --env ROUNDS=32768 --env MAX_S=1800 \
#     -- bash benchmarks/hash/dither/percall_rtt.sh
#   ssh GATE fetch SESSION | tar xz; ssh GATE ledger > ledger.jsonl
#   python3 benchmarks/hash/dither/percall_rtt.py check --challenger-out SESSION --ledger ledger.jsonl \
#     --client CLIENT_RUN/out/percall-rtt.json --key-record KEY_RECORD --period dev-NAME
set -uo pipefail
D=${RESEARCH_RUN_DIR:?}; OUT=$D/out; mkdir -p "$OUT"
PORT=${PORT:?the port the session tunnel opens here}; SESSION=${SESSION:?the session id from infra}
COMMITMENT=${COMMITMENT:?the coin commitment root the gate opened the session with}
SECRET_FILE=${SECRET_FILE:-$HOME/.local/state/pouw-challenger/$SESSION.secret}
[ -f "$SECRET_FILE" ] || { echo "percall_rtt: NO_SECRET: the session's secret isn't at $SECRET_FILE on $(hostname)"; exit 13; }
git rev-parse HEAD 2>/dev/null | tee "$OUT/commit.txt"
echo "percall_rtt: client on $(hostname), waiting for the challenger's tunnel on 127.0.0.1:$PORT at $(date -u +%H:%M:%SZ)"
for _ in $(seq $((${CHALLENGER_WAIT_S:-900} * 20))); do
  ss -Hltn "sport = :$PORT" | grep -q . && break
  sleep 0.05
done
ss -Hltn "sport = :$PORT" | grep -q . || { echo "percall_rtt: NO_CHALLENGER on 127.0.0.1:$PORT"; exit 13; }
echo "percall_rtt: the tunnel is up at $(date -u +%H:%M:%SZ)"
"${PY:-python3}" benchmarks/hash/dither/percall_rtt.py run --challenger "127.0.0.1:$PORT" --session "$SESSION" \
  --secret-file "$SECRET_FILE" --commitment "$COMMITMENT" --rounds "${ROUNDS:-32768}" --max-s "${MAX_S:-1800}" \
  --out "$OUT/percall-rtt.json" "$@"
rc=$?
echo "percall_rtt: exited $rc at $(date -u +%H:%M:%SZ)"
exit "$rc"
