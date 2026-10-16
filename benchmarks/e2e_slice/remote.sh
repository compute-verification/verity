#!/usr/bin/env bash
# The slice's window with its challenger on another machine (the trusted-device placement): ROLE=challenger on the
# auditor's machine, its link and its C-Flock verifier listening on that machine's own address HOST; ROLE=prover on the
# prover's machine, connecting to HOST:PORT (retrying while the challenger builds and stages). The fake engine, C-Flock's
# --zk proofs across the network (FLOCK=none: the challenger's checks only); a coin as soon as a round's operand roots
# arrive (CADENCE 0), so the prover's coin wait is the round trip. Each role is its own `research run` (cwd = the source
# tree's root). Under $OUT: challenger/{rounds.jsonl,window.json} or prover/{rounds.jsonl,prover.json}, ROLE.log, net.txt.
#
#   research run --on vy-nebius-1 --project verity --campaign cmp-13 --source . --cwd source --env ROLE=challenger \
#     --env HOST=10.80.0.18 -- bash benchmarks/e2e_slice/remote.sh
#   research run --on vy-nebius-2 --project verity --campaign cmp-13 --source . --cwd source --env ROLE=prover \
#     --env HOST=10.80.0.18 -- bash benchmarks/e2e_slice/remote.sh
# PORT (47613), WINDOW (8), FOLD (duo), ROUNDS (4), CADENCE (0), SHAPE (--m 256 --n 256 --k 4096), K (4096), THREADS (48).
# The Nebius nodes' security group admits TCP 22 alone, so between them the prover takes TUNNEL=peer (ssh forwards).
set -uo pipefail
export O=${OUT:?} D=benchmarks/e2e_slice
: "${ROLE:?challenger or prover}" "${HOST:?the address of the challenger machine}"
PORT=${PORT:-47613} WINDOW=${WINDOW:-8} FOLD=${FOLD:-duo} ROUNDS=${ROUNDS:-4} CADENCE=${CADENCE:-0} K=${K:-4096}
SHAPE=${SHAPE:---m 256 --n 256 --k 4096}
git rev-parse HEAD > "$O/commit.txt" 2>&1
bash $D/build.sh > "$O/build.log" 2>&1 || { tail -20 "$O/build.log"; exit 2; }
PY=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["python"])' "$O/builds.json")
[ "${FLOCK:-}" = none ] || FLOCK=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["prove"])' "$O/builds.json")
S=$(mktemp -d /dev/shm/e2e-remote.XXXXXX)
trap 'rm -rf "$S"' EXIT
export PYTHONPATH=$PWD:$PWD/verity/ml/flock/python:$PWD/integrations/vllm:$PWD/tools/verity/circuit_check/src
{ hostname; ip -4 -o addr show eth0; } > "$O/net.txt" 2>&1
if [ "$ROLE" = challenger ]; then
  $PY $D/challenger.py --flock "$FLOCK" --out "$O/challenger" --scratch "$S/c" --port-file "$S/port" --listen "$HOST" \
    --port "$PORT" --cadence "$CADENCE" --served dither --window "$WINDOW" --fold "$FOLD" --k "$K" > "$O/challenger.log" 2>&1
else
  # the security group admits TCP 22 alone (and drops ICMP): a TCP handshake with HOST's sshd is one round trip
  $PY -c 'import socket, sys, time
t = []
for _ in range(20):
    s = time.perf_counter(); socket.create_connection((sys.argv[1], 22), 5).close(); t.append(1e3 * (time.perf_counter() - s))
t.sort(); print(f"tcp connect {sys.argv[1]}:22 ms: min {t[0]:.3f} median {t[10]:.3f} max {t[-1]:.3f}")' "$HOST" >> "$O/net.txt" 2>&1
  # TUNNEL=peer: the link and each round's verifier through ssh forwards, as research/pods/nebius/peers reaches HOST
  [ "${TUNNEL:-}" = peer ] && TUNNEL=$(awk -v h="@$HOST" 'substr($NF, length($NF) - length(h) + 1) == h {$1 = ""; print}' \
    tools/research/src/research/pods/nebius/peers)
  # shellcheck disable=SC2086
  $PY $D/prover.py --engine fake --flock "$FLOCK" --host "$HOST" --port "$PORT" --connect-tries 2400 --out "$O/prover" \
    --tunnel "${TUNNEL:-}" \
    --scratch "$S/p" --rounds "$ROUNDS" --calls 1 --served dither --window "$WINDOW" --fold "$FOLD" $SHAPE \
    --prove-threads "${THREADS:-48}" --workers 8 > "$O/prover.log" 2>&1
fi
rc=$?
echo "$ROLE rc $rc" | tee "$O/rc.txt"
tail -3 "$O/$ROLE.log"
exit $rc
