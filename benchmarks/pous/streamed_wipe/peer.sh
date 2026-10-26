#!/usr/bin/env bash
# demo.sh with the memory challenger on a cluster peer: a rehearsal of the challenger on its own host before the auditor's
# gate serves it (INF-258). This tree goes to a mktemp -d on PEER over ssh (PEER's line in /etc/vy/peers, the vy-cluster
# key), and demo.sh starts the challenger's side there over ssh as the gate's verb will (the request on stdin;
# `{"listening"}` and the report on stdout), the fill and the reads over direct TCP to PEER's address at PORT (47611).
# PEER is the lane's machine, not the auditor's, so the run certifies nothing. cwd = a clone, as demo.sh's:
#
#   research run --on NODE --project verity --source . --cwd clone --timeout 60m -- \
#     bash benchmarks/pous/streamed_wipe/peer.sh PEER [--port P] -- DEMO_OPTIONS
#
# The peer's scratch (the tree, its venv, the challenger's key and session log) is removed on exit.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PEER=${1:?usage: peer.sh PEER [--port P] -- DEMO_OPTIONS}
shift
PORT=47611
[ "${1:-}" = --port ] && { PORT=$2; shift 2; }
[ "${1:-}" = -- ] && shift
LINE=$(grep -E "^$PEER " /etc/vy/peers) || { echo "peer.sh: $PEER has no line in /etc/vy/peers"; exit 2; }
read -r -a TO <<< "${LINE#"$PEER "}"
ADDR=${TO[-1]#*@}
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 "${TO[@]}")
D=$("${SSH[@]}" mktemp -d /tmp/wipe-peer-XXXXXX)
trap '"${SSH[@]}" rm -rf "$D" || true' EXIT
tar -c --exclude=./.git --exclude=./.venv --exclude=./out --exclude=.lake . | "${SSH[@]}" tar -x -C "$D"
"${SSH[@]}" "bash -lc 'cd $D && uv sync -q --all-packages --extra torch-cpu'"
bash benchmarks/pous/streamed_wipe/demo.sh --challenger "$PEER" --challenger-address "$ADDR:$PORT" \
  --challenger-command "${SSH[*]} cd $D && exec $D/.venv/bin/python -m verity.core.protocols.pous.wipe_party --device $D/device --listen 0.0.0.0:$PORT --streamer build" \
  "$@"
