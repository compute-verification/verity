#!/usr/bin/env bash
# demo.sh with the memory challenger on a cluster peer: a rehearsal of the challenger on its own host before the auditor's
# gate serves it (INF-258). This tree goes to a mktemp -d on PEER over ssh (PEER's line in /etc/vy/peers, the vy-cluster
# key), and demo.sh starts the challenger's side there over ssh as the gate's verb will (the request on stdin;
# `{"listening"}` and the report on stdout), the fill and the reads over direct TCP to PEER's address at PORT (47611).
# The cluster's nodes pass only ssh to each other (r20261026-092032-a53b, -ba1e: TCP 47611 times out), so there
# --tunnel carries the fill and the reads in an ssh forward instead (this node's 127.0.0.1 to PEER's), and the record
# names that link. PEER is the lane's machine, not the auditor's, so the run certifies nothing. cwd = a clone, as
# demo.sh's:
#
#   research run --on NODE --project verity --source . --cwd clone --timeout 60m -- \
#     bash benchmarks/pous/streamed_wipe/peer.sh PEER [--port P] [--tunnel] -- DEMO_OPTIONS
#
# The peer's scratch (the tree, its venv, the challenger's key and session log) is removed on exit.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PEER=${1:?usage: peer.sh PEER [--port P] [--tunnel] -- DEMO_OPTIONS}
shift
PORT=47611 TUNNEL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT=$2; shift 2 ;;
    --tunnel) TUNNEL=1; shift ;;
    --) shift; break ;;
    *) break ;;
  esac
done
LINE=$(grep -E "^$PEER " /etc/vy/peers) || { echo "peer.sh: $PEER has no line in /etc/vy/peers"; exit 2; }
read -r -a TO <<< "${LINE#"$PEER "}"
ADDR=${TO[-1]#*@}
SSHO=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20)
SSH=("${SSHO[@]}" "${TO[@]}")
D=$("${SSH[@]}" mktemp -d /tmp/wipe-peer-XXXXXX)
TPID=
trap '[ -n "$TPID" ] && kill "$TPID" 2>/dev/null; "${SSH[@]}" rm -rf "$D" || true' EXIT
tar -c --exclude=./.git --exclude=./.venv --exclude=./out --exclude=.lake . | "${SSH[@]}" tar -x -C "$D"
"${SSH[@]}" "bash -lc 'cd $D && uv sync -q --all-packages --extra torch-cpu'"
LISTEN=0.0.0.0:$PORT TARGET=$ADDR:$PORT LINK=()
if [ "$TUNNEL" = 1 ]; then
  LPORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
  # Not a control master (-M): ssh -N sets TCP_NODELAY on its connection only when it isn't one, and without it a read
  # waits out the peer's delayed ACK (r20261026-092852-4c4d: one round trip of 41.5 ms, past the 20 ms deadline).
  "${SSHO[@]}" -N -o ExitOnForwardFailure=yes -L "127.0.0.1:$LPORT:127.0.0.1:$PORT" "${TO[@]}" &
  TPID=$!
  for _ in $(seq 100); do
    python3 -c 'import socket, sys; socket.create_connection(("127.0.0.1", int(sys.argv[1])), 1).close()' "$LPORT" \
      2>/dev/null && break
    kill -0 "$TPID" || { echo "peer.sh: the ssh forward to $PEER exited"; exit 2; }
    sleep 0.2
  done
  LISTEN=127.0.0.1:$PORT TARGET=127.0.0.1:$LPORT
  LINK=(--link "challenger=$PEER@ssh to $ADDR:22, forwarding this node's 127.0.0.1:$LPORT to its 127.0.0.1:$PORT")
fi
bash benchmarks/pous/streamed_wipe/demo.sh --challenger "$PEER" --challenger-address "$TARGET" \
  --challenger-command "${SSH[*]} cd $D && exec $D/.venv/bin/python -m verity.core.protocols.pous.wipe_party --device $D/device --listen $LISTEN --streamer build" \
  "${LINK[@]}" "$@"
