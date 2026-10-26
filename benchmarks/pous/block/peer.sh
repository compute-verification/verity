#!/usr/bin/env bash
# run.sh with the memory challenger on a cluster peer: a rehearsal of the challenger on its own host before the auditor's
# gate serves it (INF-258). This tree goes to a mktemp -d on PEER over ssh (PEER's line in /etc/vy/peers), the challenger's
# key is made there first (its public key printed, and handed to the check as --trusted, as infra will publish the
# auditor's), and block.py starts `challenger_party` there over ssh as the gate's verb will (the request on stdin;
# `{"listening"}` and the report on stdout). The sessions run over direct TCP to PEER's address at PORT (47610) when a
# probe there connects (--path auto, the default), else in an ssh forward in the form r20261026-094403-bb5f found sound
# (a pty session, which sets TCP_NODELAY, with ObscureKeystrokeTiming=no). PEER is the lane's machine, not the
# auditor's, so the run certifies nothing about Demo B. cwd = a clone, as run.sh's:
#
#   research run --on NODE --project verity --source . --cwd clone --timeout 60m -- \
#     bash benchmarks/pous/block/peer.sh PEER [--port P] [--path auto|direct|tunnel] -- RUN_OPTIONS
#
# The peer's scratch (the tree, its venv, the challenger's key and session log) is removed on exit.
set -euo pipefail
cd "$(dirname "$0")/../../.."
O=${OUT:-${RESEARCH_RUN_DIR:-$PWD}/out}
mkdir -p "$O"
PEER=${1:?usage: peer.sh PEER [--port P] [--path auto|direct|tunnel] -- RUN_OPTIONS}
shift
PORT=47610 MODE=auto
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT=$2; shift 2 ;;
    --path) MODE=$2; shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done
LINE=$(grep -E "^$PEER " /etc/vy/peers) || { echo "peer.sh: $PEER has no line in /etc/vy/peers"; exit 2; }
read -r -a TO <<< "${LINE#"$PEER "}"
ADDR=${TO[-1]#*@}
SSHO=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20)
SSH=("${SSHO[@]}" "${TO[@]}")
D=$("${SSH[@]}" mktemp -d /tmp/pous-peer-XXXXXX)
TPID=
trap '[ -n "$TPID" ] && kill "$TPID" 2>/dev/null; "${SSH[@]}" rm -rf "$D" || true' EXIT
date -u +ship-%H:%M:%SZ
tar -c --exclude=./.git --exclude=./.venv --exclude=./out --exclude=.lake . | "${SSH[@]}" tar -x -C "$D"
"${SSH[@]}" "bash -lc 'cd $D && uv sync -q --all-packages --extra torch-cpu'"
date -u +keygen-%H:%M:%SZ
KEY=$("${SSH[@]}" "cd $D && .venv/bin/python -c 'from verity.core.protocols.pous import period; print(period.challenger(\"$D/device\").public_key.hex())'" | tail -1)
date -u +keyed-%H:%M:%SZ
echo "peer.sh: the challenger's key on $PEER: ${KEY:0:16}..."
probe() {
  "${SSHO[@]}" -n "${TO[@]}" "timeout 25 $D/.venv/bin/python -c 'import socket; s = socket.create_server((\"0.0.0.0\", $PORT)); print(\"up\", flush=True); s.settimeout(20); c, _ = s.accept(); c.sendall(b\"ok\")'" > "$O/probe.txt" 2>&1 &
  local pp=$!
  for _ in $(seq 50); do grep -q up "$O/probe.txt" 2>/dev/null && break; sleep 0.2; done
  local got
  got=$(python3 -c 'import socket, sys; s = socket.create_connection((sys.argv[1], int(sys.argv[2])), 5); print(s.recv(2).decode())' "$ADDR" "$PORT" 2>&1 || true)
  wait "$pp" 2>/dev/null || true
  echo "probe $ADDR:$PORT: $got" >> "$O/probe.txt"
  [ "$got" = ok ]
}
PATHNAME=$MODE
if [ "$MODE" = auto ]; then
  if probe; then PATHNAME=direct; else PATHNAME=tunnel; fi
  echo "peer.sh: direct TCP to $ADDR:$PORT $([ "$PATHNAME" = direct ] && echo connects || echo 'does not connect'): the $PATHNAME path"
fi
LISTEN=0.0.0.0:$PORT TARGET=$ADDR:$PORT
if [ "$PATHNAME" = tunnel ]; then
  LPORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
  # A session with a pty, not -N: sshd sets TCP_NODELAY on a connection only for a session (r20261026-092852-4c4d);
  # ObscureKeystrokeTiming=no: a pty session otherwise sends on a 20 ms clock (r20261026-094115-fb5c).
  "${SSHO[@]}" -n -tt -o ObscureKeystrokeTiming=no -o ExitOnForwardFailure=yes \
    -L "127.0.0.1:$LPORT:127.0.0.1:$PORT" "${TO[@]}" sleep 7200 >/dev/null &
  TPID=$!
  for _ in $(seq 100); do
    python3 -c 'import socket, sys; socket.create_connection(("127.0.0.1", int(sys.argv[1])), 1).close()' "$LPORT" \
      2>/dev/null && break
    kill -0 "$TPID" || { echo "peer.sh: the ssh forward to $PEER exited"; exit 2; }
    sleep 0.2
  done
  LISTEN=127.0.0.1:$PORT TARGET=127.0.0.1:$LPORT
elif [ "$PATHNAME" != direct ]; then
  echo "peer.sh: --path auto, direct or tunnel, not $PATHNAME"; exit 2
fi
echo "{\"peer\": \"$PEER\", \"path\": \"$PATHNAME\", \"target\": \"$TARGET\", \"listen\": \"$LISTEN\"}" > "$O/path.json"
bash benchmarks/pous/block/run.sh --challenger "$PEER" --challenger-address "$TARGET" --trusted "$KEY" \
  --challenger-command "${SSH[*]} cd $D && exec $D/.venv/bin/python -m verity.core.protocols.pous.challenger_party --device $D/device --listen $LISTEN" \
  "$@"
