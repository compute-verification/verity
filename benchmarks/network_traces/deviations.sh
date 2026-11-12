#!/usr/bin/env bash
# deviations.py's tries against a served id front that listens on its own node's loopback, from a node that reaches it
# through one ssh local forward. A `research run` job from the tree's root (--cwd source):
#   PEER=USER@HOST PORT=P KEY=FILE [SSH_PORT=22] [PY=PYTHON] bash benchmarks/network_traces/deviations.sh TRY-FLAG...
#   FAKE=1 [PY=PYTHON] bash benchmarks/network_traces/deviations.sh TRY-FLAG...
# TRY-FLAG... are `deviations.py try`'s own: the classes, the traffic and, after `--`, the rehearsal's flags, the
# schedule's among them. The job gives it the front, --out and --scratch.
# 1. The forward: ssh -N with one -L from this node's loopback to the serving node's 127.0.0.1:PORT, and nothing else
#    asked of that node. ssh's configuration, its log and its known-hosts file are in the job's temporary directory,
#    which is deleted, so the serving node's address is on no command line and in nothing the job keeps. The front is
#    reached when a connect through the forward stays open: ssh takes a local connect whatever is behind it, and
#    closes it at once when nothing listens there. FAKE=1 is the dry run: no ssh, and `try --stand-in` starts the
#    stand-in id endpoint as the front.
# 2. The tries: `deviations.py try`. What each try keeps and the table go to $OUT/tries; each try's run directory, with
#    the relay's records and their salts, is in the temporary directory.
# 3. The guard, on every exit: the job fails (exit 9), naming the file, if a file of $OUT holds an IPv4 address other
#    than loopback's, an `@` before a name, or the serving node's host as PEER gave it. Everything the job prints is
#    also $OUT/job.log.
# The exit is try's: 0 every class did what its check says; 1 one didn't, or honest was refused and the rest weren't
# tried; 2 a row is shown neither way. Else 3, the front wasn't reached, or 9, the guard.
# PY is a python with numpy; by default uv makes one in the temporary directory, at the tree's lock.
set -uo pipefail
RUN=${RESEARCH_RUN_DIR:?run this under research run}; OUT=$RUN/out; mkdir -p "$OUT"

if [ -z "${DEVIATIONS_INNER:-}" ]; then
  # The job proper runs below, as a child whose every line, stdout and stderr, is also job.log; then the guard.
  host=; [ -n "${PEER:-}" ] && host=${PEER#*@}
  trap : TERM INT   # the child ends on the signal; this shell stays for the guard
  DEVIATIONS_INNER=1 bash "$0" "$@" 2>&1 | (trap '' TERM INT; exec tee "$OUT/job.log")
  rc=${PIPESTATUS[0]}
  found=$({ grep -rIlE '@[A-Za-z0-9]' "$OUT"
            grep -rIHoE '([0-9]{1,3}\.){3}[0-9]{1,3}' "$OUT" | grep -vE ':127(\.[0-9]{1,3}){3}$' | sed 's/:[^:]*$//'
            [ -n "$host" ] && grep -rlaF -e "$host" "$OUT"; } 2>/dev/null | sort -u)
  if [ -n "$found" ]; then
    echo "deviations: GUARD FAILED: an address or an @name is in what this job keeps, in:" | tee -a "$OUT/job.log" >&2
    echo "$found" | sed "s|^$OUT/|  out/|" | tee -a "$OUT/job.log" >&2
    echo "deviations: (job exit $rc) exit 9: read that file on the node before anyone cites or fetches this run" >&2
    exit 9
  fi
  echo "deviations: guard: no address in out/ ($(find "$OUT" -type f | wc -l) files); exit $rc" | tee -a "$OUT/job.log"
  exit "$rc"
fi

peer=${PEER:-}; unset PEER   # the serving node's address: ssh's configuration below holds it, and nothing else does
W=$(mktemp -d); TUN=
cleanup() { [ -n "$TUN" ] && kill "$TUN" 2>/dev/null; rm -rf "$W"; }
trap cleanup EXIT
trap 'exit 143' TERM INT
log() { echo "$(date -u +%H:%M:%SZ) deviations: $*"; }
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
if [ -z "${PY:-}" ]; then
  command -v uv > /dev/null || { log "no uv on this node: give PY, a python with numpy"; exit 2; }
  UV_PROJECT_ENVIRONMENT=$W/venv uv sync -q --frozen --no-dev --no-install-project \
    || { log "uv made no environment at the tree's lock: give PY"; exit 2; }
  PY=$W/venv/bin/python
fi
"$PY" -c 'import numpy' || { log "PY has no numpy"; exit 2; }

if [ -n "${FAKE:-}" ]; then
  FRONT=(--stand-in)
  log "the dry run: the stand-in id endpoint is the front, and there is no forward"
else
  [ -n "$peer" ] && [ "$peer" != "${peer#*@}" ] || { log "PEER: USER@HOST, the serving node"; exit 2; }
  PORT=${PORT:?PORT: the port of the id front on the loopback of the serving node}
  KEY=${KEY:?KEY: the file of the tunnel key on this node}
  [ -r "$KEY" ] || { log "UNREACHED: the key file $(basename "$KEY") isn't readable on this node"; exit 3; }
  LOCAL=$("$PY" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
  # ssh's whole configuration, so that the address is on no command line, and a known-hosts file of the job's own.
  cat > "$W/ssh_config" <<EOF
Host serving-node
  HostName ${peer#*@}
  User ${peer%%@*}
  Port ${SSH_PORT:-22}
  IdentityFile $KEY
  IdentitiesOnly yes
  BatchMode yes
  ConnectTimeout 10
  StrictHostKeyChecking accept-new
  UserKnownHostsFile $W/known_hosts
  ExitOnForwardFailure yes
  ServerAliveInterval 15
  RequestTTY no
  ForwardAgent no
  ForwardX11 no
  ControlPath none
EOF
  ssh -F "$W/ssh_config" -N -L "127.0.0.1:$LOCAL:127.0.0.1:$PORT" serving-node 2> "$W/tunnel.log" &
  TUN=$!
  # ssh's log names the serving node, so no line of it is printed or kept.
  "$PY" - "$TUN" "$LOCAL" <<'EOF' || exit 3
import socket, sys, time
pid, port = int(sys.argv[1]), int(sys.argv[2])


def alive() -> bool:
    try:
        return open(f"/proc/{pid}/stat").read().rpartition(")")[2].split()[0] != "Z"
    except OSError:
        return False


def unreached(why: str):
    print(f"deviations: UNREACHED: {why}", flush=True)
    sys.exit(1)


until = time.monotonic() + 30
while True:
    if not alive():
        unreached("ssh ended: the key is refused, the forward is forbidden, or the serving node isn't reached")
    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=2)
        break
    except OSError:
        if time.monotonic() > until:
            unreached("the forward didn't open in 30 s")
        time.sleep(0.25)
s.settimeout(3)
try:
    s.recv(1)
except socket.timeout:
    print("deviations: REACHED: a connect through the forward stayed open 3 s, nothing sent", flush=True)
    sys.exit(0)
except OSError:
    pass
unreached("the forward is up, and nothing listens behind it: its connect was closed at once")
EOF
  export DEVIATIONS_FRONT=127.0.0.1:$LOCAL
  FRONT=(--front-env DEVIATIONS_FRONT)
fi

log "the tries: deviations.py try $*"
"$PY" benchmarks/network_traces/deviations.py try "${FRONT[@]}" --out "$OUT/tries" --scratch "$W/tries" "$@"
rc=$?
log "the tries ended, exit $rc: $(find "$OUT/tries" -type f 2> /dev/null | wc -l) files kept in out/tries"
exit "$rc"
