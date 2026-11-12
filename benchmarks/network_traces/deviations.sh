#!/usr/bin/env bash
# deviations.py's tries against a served id front that listens on its own node's loopback, from a node that reaches it
# through one ssh local forward. A `research run` job from the tree's root (--cwd source):
#   PEER_AUTHORITY=USER@HOST[:SSH_PORT] FRONT_AUTHORITY=HOST:PORT TUNNEL_KEY=FILE [REACH_ONLY=1] [PY=PYTHON] \
#     bash benchmarks/network_traces/deviations.sh TRY-FLAG...
#   FAKE=1 [PY=PYTHON] bash benchmarks/network_traces/deviations.sh TRY-FLAG...
# TRY-FLAG... are `deviations.py try`'s own: the classes, the traffic and, after `--`, the rehearsal's flags, the
# schedule's among them. The job gives it the front, --out and --scratch.
# PEER_AUTHORITY is the serving node as ssh reaches it (HOST a name or an IPv4 address), FRONT_AUTHORITY the id front
# as the serving node reaches it (its loopback and the front's port) and TUNNEL_KEY the key's file on this node (a
# leading ~/ is this node's home). None has a default. Their names are ones `research run --env` records without their
# values.
# 1. The forward: ssh -N with one local forward from this node's loopback to FRONT_AUTHORITY, and nothing else asked
#    of the serving node. ssh's configuration, its log and its known-hosts file are in the job's temporary directory,
#    which is deleted, so the three values are on no command line and in nothing the job keeps. The front is reached
#    when a connect through the forward stays open: ssh takes a local connect whatever is behind it, and closes it
#    at once when nothing listens there. REACH_ONLY=1 ends the job there, with no try: exit 0 reached, 3 not. FAKE=1
#    is the dry run: no ssh, and `try --stand-in` starts the stand-in id endpoint as the front.
# 2. The tries: `deviations.py try`. What each try keeps and the table go to $OUT/tries; each try's run directory, with
#    the relay's records and their salts, is in the temporary directory.
# 3. The guard, on every exit: the job fails (exit 9), naming the file, if a file of $OUT holds an IPv4 address other
#    than loopback's, an `@` before a name, or the serving node's host as PEER_AUTHORITY gave it. Everything the job
#    prints is also $OUT/job.log.
# The exit is try's: 0 every class did what its check says; 1 one didn't, honest was refused and the rest weren't
# tried, or a try wrote no check; 2 a row is shown neither way, as tokens and length are until `table --ids` has the
# server's record; 5 a try's network certifier named a lost release on its watermark, or its watermark stopped with
# an error (or it ran with the watermark and wrote no log), and no class after it was tried. Else 3, the front wasn't
# reached; 4, a variable or the python is wrong; or 9, the guard.
# PY is a python with numpy; by default uv makes one in the temporary directory, at the tree's lock.
set -uo pipefail
RUN=${RESEARCH_RUN_DIR:?run this under research run}; OUT=$RUN/out; mkdir -p "$OUT"

if [ -z "${DEVIATIONS_INNER:-}" ]; then
  # The job proper runs below, as a child whose every line, stdout and stderr, is also job.log; then the guard.
  host=${PEER_AUTHORITY:-}; host=${host#*@}; host=${host%%:*}
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

# the serving node's address, the front's and the key's file: ssh's configuration below holds them, and nothing else
peer=${PEER_AUTHORITY:-}; front=${FRONT_AUTHORITY:-}; key=${TUNNEL_KEY:-}
case $key in "~/"*) key=$HOME/${key#"~/"} ;; esac   # a path under this node's home, as the launching host can't say it
unset PEER_AUTHORITY FRONT_AUTHORITY TUNNEL_KEY
W=$(mktemp -d); TUN=
cleanup() { [ -n "$TUN" ] && kill "$TUN" 2>/dev/null; rm -rf "$W"; }
trap cleanup EXIT
trap 'exit 143' TERM INT
log() { echo "$(date -u +%H:%M:%SZ) deviations: $*"; }
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
if [ -z "${PY:-}" ]; then
  command -v uv > /dev/null || { log "no uv on this node: give PY, a python with numpy"; exit 4; }
  UV_PROJECT_ENVIRONMENT=$W/venv uv sync -q --frozen --no-dev --no-install-project \
    || { log "uv made no environment at the tree's lock: give PY"; exit 4; }
  PY=$W/venv/bin/python
fi
"$PY" -c 'import numpy' || { log "PY has no numpy"; exit 4; }

if [ -n "${FAKE:-}" ]; then
  FRONT=(--stand-in)
  log "the dry run: the stand-in id endpoint is the front, and there is no forward"
else
  user=${peer%%@*}; node=${peer#*@}; ssh_port=22
  [ "$node" = "${node%:*}" ] || { ssh_port=${node##*:}; node=${node%:*}; }
  [ -n "$user" ] && [ -n "$node" ] && [ "$peer" != "${peer#*@}" ] && [[ $ssh_port =~ ^[0-9]+$ ]] \
    || { log "PEER_AUTHORITY: the serving node as ssh reaches it: a user, an at sign, a host, and :SSH_PORT if not 22"
         exit 4; }
  [[ $front =~ ^[^:]+:[0-9]+$ ]] \
    || { log "FRONT_AUTHORITY: HOST:PORT, the id front as the serving node reaches it"; exit 4; }
  [ -n "$key" ] || { log "TUNNEL_KEY: the tunnel key's file on this node"; exit 4; }
  [ -r "$key" ] || { log "UNREACHED: the tunnel key's file isn't readable on this node"; exit 3; }
  LOCAL=$("$PY" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
  # ssh's whole configuration, the forward too, so that no value is on a command line; a known-hosts file of the
  # job's own.
  cat > "$W/ssh_config" <<EOF
Host serving-node
  HostName $node
  User $user
  Port $ssh_port
  IdentityFile $key
  IdentitiesOnly yes
  BatchMode yes
  ConnectTimeout 10
  StrictHostKeyChecking accept-new
  UserKnownHostsFile $W/known_hosts
  LocalForward 127.0.0.1:$LOCAL $front
  ExitOnForwardFailure yes
  ServerAliveInterval 15
  RequestTTY no
  ForwardAgent no
  ForwardX11 no
  ControlPath none
EOF
  ssh -F "$W/ssh_config" -N serving-node 2> "$W/tunnel.log" &
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
  [ -z "${REACH_ONLY:-}" ] || { log "REACH_ONLY: no try"; exit 0; }
  export DEVIATIONS_FRONT=127.0.0.1:$LOCAL
  FRONT=(--front-env DEVIATIONS_FRONT)
fi

log "the tries: deviations.py try $*"
"$PY" benchmarks/network_traces/deviations.py try "${FRONT[@]}" --out "$OUT/tries" --scratch "$W/tries" "$@"
rc=$?
log "the tries ended, exit $rc: $(find "$OUT/tries" -type f 2> /dev/null | wc -l) files kept in out/tries"
exit "$rc"
