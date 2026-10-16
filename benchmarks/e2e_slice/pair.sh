#!/usr/bin/env bash
# The challenger and the prover of one window (window.sh's, under its lease or with ENGINE=fake without one): the challenger
# first, its own process and circuit, then the prover on its port. Reads O, D, S, PY, FLOCK, ENGINE, ROUNDS, CALLS, CADENCE,
# STAGES, COINS, ATTACK, PROVER_ARGS and CHALLENGER_ARGS.
set -uo pipefail
# shellcheck disable=SC2086
$PY $D/challenger.py --flock "$FLOCK" --out "$O/challenger" --scratch "$S/c" --port-file "$S/port" --cadence "$CADENCE" \
  --coins "$COINS" ${CHALLENGER_ARGS:-} > "$O/challenger.log" 2>&1 &
ch=$!
until [ -s "$S/port" ]; do kill -0 $ch 2>/dev/null || break; sleep 1; done
if [ -s "$S/port" ]; then
  # shellcheck disable=SC2086
  $PY $D/prover.py --engine "$ENGINE" --flock "$FLOCK" --port "$(cat "$S/port")" --out "$O/prover" --scratch "$S/p" \
    --rounds "$ROUNDS" --calls "$CALLS" --stages "$STAGES" --coins "$COINS" --attack "$ATTACK" ${PROVER_ARGS:-} \
    > "$O/prover.log" 2>&1
  echo "prover rc $?" >> "$O/rc.txt"
fi
wait $ch; echo "challenger rc $?" >> "$O/rc.txt"
