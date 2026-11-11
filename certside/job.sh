#!/usr/bin/env bash
# The network certifier's side of a served run in the token-id form with a response cap: warden_serve.py rehearse --ids
# --watermark --max-tokens-cap on this node, in front of the serving node's id-form endpoint (the id front: it takes the
# certifier's id request and answers one io_tie.id_event an output id, then io_tie.TRAILER, and closes), which listens
# on the serving node's loopback and is reached through one ssh local forward. A `research run` job from the shipped
# tree's root (--cwd source); certside/launch.sh launches it:
#   PEER=USER@HOST PORT=P [SSH_PORT=22] [KEY=FILE] [MODEL=NAME] [ID_BOUND=N] [schedule and traffic, below]
#     [TRIES=2 [LANES2=64] [RATE2=1.67] [REQUESTS2=2000]]
#     [REACH_ONLY=1 | FAKE=1 [FIRST_MS=2000] [TOKEN_MS=220] [FAKE_IDS=N]] bash certside/job.sh
# 1. Reach, in under a minute, before anything is sent: KEY readable, the forward up (ssh -N, one -L from this node's
#    127.0.0.1:LOCAL to the serving node's 127.0.0.1:PORT, and nothing else asked of the serving node: no command, no
#    second forward), then the endpoint through it: $OUT/reach.json.
#    - MODEL unset: GET /v1/models, five times, each round trip timed; the endpoint lists one model, the run's.
#    - MODEL=NAME: the endpoint needn't answer /v1/models. Two TCP connects through the forward, nothing sent: ssh
#      accepts a local connect whatever the far end, and closes it at once when the far port refuses, so a connect
#      still open after 3 s, with no refusal in ssh's log, is a listener. No round trip is timed.
#    The exit says which it was: 0 reached; 3 not reached (no key, the key refused, the forward forbidden, no ssh);
#    4 the key is accepted and the forward opens, and nothing listens on PORT yet; 5 reached, and the run can't be
#    sent (the model's name is past the 64 bytes an id request holds for it as JSON). REACH_ONLY=1 stops here.
# 2. The rehearsal, into $OUT/warden: REQUESTS id requests (`id_traffic`: Poisson arrivals at RATE a second, prompts of
#    16..PROMPT_MAX token ids and max_tokens of 1..MAX_TOKENS, both log-uniform, each prompt id uniform below 128,256,
#    the temperature sampled from 0, 0.7 and 1, top_p 1, a seed on each, all fixed by SEED), the model MODEL or the
#    endpoint's one, at the schedule below, with the watermark stream sent to the same port. ID_BOUND=N (at most
#    128,256) scales every prompt id below N and changes nothing else (certside/bounded.py).
# 3. warden_serve.py check on $OUT/warden, from its files alone: $OUT/check.txt (RECHECK=0 leaves it out: rehearse has
#    already run the same check once and written check.json); $OUT/summary.json, with each step's seconds and the
#    memory the run took; then the record's reading, certside/report.py: $OUT/report.txt and $OUT/report.json.
# 4. The guard, on every exit: nothing this job keeps or prints names the serving node. ssh's log and its known-hosts
#    file stay in the job's temporary directory, which is deleted; everything the job prints is also $OUT/job.log; and
#    the job fails (exit 9), naming the file, if any file of $OUT holds an IPv4 address other than loopback's, an `@`
#    before a name, or the serving node's host as PEER gave it.
# TRIES=2 is two tries behind the one forward, for a server that starts once: 2 and 3 as try 1, under $OUT/try1/ in
# place of $OUT, and then, if and only if try 1 is ACCEPTED (rehearse and check both exit 0), at once 2 and 3 again
# as try 2, under $OUT/try2/, with LANES2, RATE2 and REQUESTS2 in place of LANES, RATE and REQUESTS and nothing else
# changed. Each try is a rehearsal of its own: its own certifier process, run id, keys and watermark stream. Each
# has its summary.json, and $OUT/summary.json names both verdicts, the seconds between the tries (between_s, from try
# 1's rehearsal returning to try 2's start; between_buckets_s, from its last bucket to try 2's first), and what the
# job kept. The exit: 0 both ACCEPTED; 1 try 1 is not, so try 2 was not started (the job says why, from try 1's record);
# 6 try 1 is and try 2 is not. TRIES=1, the default, is one try, its files in $OUT itself, exit 0 or 1.
# FAKE=1 is the dry run: no ssh; the job starts a fake endpoint on this node's loopback, in the id form at the
# schedule's P, and then does 1 to 4 against it as against a forward. The fake is certside/paced.py: a request's first
# id leaves FIRST_MS after the request and each next one TOKEN_MS later, as a model streams, its ids below FAKE_IDS
# when set (the fake's own are below 128,256). TOKEN_MS=0 is warden_serve's own fake instead, whose id form has no
# time a token: the whole answer at once, FIRST_MS after.
#
# THE SCHEDULE, each number a variable, for GLM-5.3 behind the id front under a 220 ms tick:
# - CELL_BYTES 4096, SLOTS 1 (r): in the id form a response is one id event a DATA cell, so r = 1 at BUCKET_MS 250
#   releases 4 ids a second on a connection. The server ticks every 220 ms, above that, so a stream whose first id is
#   in time stays ahead of its cells. BUCKET_MS is also the server's bucket: the tick admits by the release bucket.
# - INGRESS_CELLS 4 within INGRESS_BUCKETS 16: an id request's fields but the prompt are in its first cell and each
#   prompt id takes 11 bytes, so PROMPT_MAX 1024 ids are 3 cells, 3 buckets at r = 1.
# - DELAY 60 (D, 15 s) from admission to the first egress cell's bucket: the request's ingress, its release at the
#   bucket's close, the forward's trip, the tick's wait for the watermark's line, the prefill and the first id back.
# - LANES 8, QUEUE 64: a connection holds its lane from admission to its END's bucket, D + its ids + 1 buckets, at
#   most 317 (79 s) at MAX_TOKENS 256. An arrival with every lane taken waits in the queue, before its admission, so
#   that wait is in no D and makes nothing late; past QUEUE waiting it is refused, and its window is not certified.
# - CAP 256 (the response cap; `none`: no cap): a request whose max_tokens is past it is refused (cap) and each
#   response is held to its request's max_tokens (limit). MAX_TOKENS equals it, so this traffic has no cap refusal.
# - EGRESS_CELLS 4096: above the cap's 256 ids and the END. WINDOW 600 buckets a record (150 s).
# - RATE 0.2, REQUESTS 240, SEED 0: 20 minutes of arrivals. TAIL 120 s after the last arrival: its wait for a lane,
#   D (15 s) and its longest response (257 cells, 64 s).
set -uo pipefail
RUN=${RESEARCH_RUN_DIR:?run this under research run}; OUT=$RUN/out; mkdir -p "$OUT"

if [ -z "${CERTSIDE_INNER:-}" ]; then
  # The job proper runs below, as a child whose every line, stdout and stderr, is also job.log; then the guard.
  host=; [ -n "${PEER:-}" ] && host=${PEER#*@}
  trap : TERM INT   # the child ends on the signal; this shell stays for the guard
  CERTSIDE_INNER=1 bash "$0" "$@" 2>&1 | (trap '' TERM INT; exec tee "$OUT/job.log")
  rc=${PIPESTATUS[0]}
  found=$({ grep -rIlE '@[A-Za-z0-9]' "$OUT"
            grep -rIHoE '([0-9]{1,3}\.){3}[0-9]{1,3}' "$OUT" | grep -vE ':127(\.[0-9]{1,3}){3}$' | sed 's/:[^:]*$//'
            [ -n "$host" ] && grep -rlaF -e "$host" "$OUT"; } 2>/dev/null | sort -u)
  if [ -n "$found" ]; then
    echo "certside: GUARD FAILED: an address or an @name is in what this job keeps, in:" | tee -a "$OUT/job.log" >&2
    echo "$found" | sed "s|^$OUT/|  out/|" | tee -a "$OUT/job.log" >&2
    echo "certside: (job rc $rc) exit 9: read that file on the node before anyone cites or fetches this run" >&2
    exit 9
  fi
  echo "certside: guard: no address in out/ ($(find "$OUT" -type f | wc -l) files); exit $rc" | tee -a "$OUT/job.log"
  exit "$rc"
fi

SRC=$PWD; HERE=$(dirname "$0")
CELL_BYTES=${CELL_BYTES:-4096}; SLOTS=${SLOTS:-1}; BUCKET_MS=${BUCKET_MS:-250}; DELAY=${DELAY:-60}
LANES=${LANES:-8}; QUEUE=${QUEUE:-64}; WINDOW=${WINDOW:-600}; INGRESS_CELLS=${INGRESS_CELLS:-4}
INGRESS_BUCKETS=${INGRESS_BUCKETS:-16}; EGRESS_CELLS=${EGRESS_CELLS:-4096}; CAP=${CAP:-256}
RATE=${RATE:-0.2}; REQUESTS=${REQUESTS:-240}; SEED=${SEED:-0}; PROMPT_MAX=${PROMPT_MAX:-1024}
MAX_TOKENS=${MAX_TOKENS:-256}; TAIL=${TAIL:-120}
TRIES=${TRIES:-1}; LANES2=${LANES2:-64}; RATE2=${RATE2:-1.67}; REQUESTS2=${REQUESTS2:-2000}
plan() {   # SCHEDULE and TRAFFIC from the variables as they stand: try 2 changes LANES, RATE and REQUESTS
  SCHEDULE=(--cell-bytes "$CELL_BYTES" --slots "$SLOTS" --bucket-ms "$BUCKET_MS" --delay "$DELAY" --lanes "$LANES"
            --queue "$QUEUE" --window "$WINDOW" --ingress-cells "$INGRESS_CELLS" --ingress-buckets "$INGRESS_BUCKETS"
            --egress-cells "$EGRESS_CELLS")
  [ "$CAP" = none ] || SCHEDULE+=(--max-tokens-cap "$CAP")
  TRAFFIC=(--ids --requests "$REQUESTS" --rate "$RATE" --seed "$SEED" --prompt-max "$PROMPT_MAX"
           --max-tokens "$MAX_TOKENS" --tail "$TAIL")
}
plan
[ "$TRIES" = 1 ] || [ "$TRIES" = 2 ] || { echo "certside: TRIES is 1 or 2, not $TRIES"; exit 2; }
MODEL=${MODEL:-}
peer=${PEER:-}; unset PEER   # the serving node's address: ssh's config file below holds it, and nothing else does
W=$(mktemp -d); export TMPDIR=$W
TUN=; FAKED=
cleanup() { [ -n "$TUN" ] && kill "$TUN" 2>/dev/null; [ -n "$FAKED" ] && kill "$FAKED" 2>/dev/null; rm -rf "$W"; }
trap cleanup EXIT
trap 'exit 143' TERM INT
log() { echo "$(date -u +%H:%M:%SZ) certside: $*"; }
SECONDS=0; MARKS=()
mark() { MARKS+=("$1=$SECONDS"); }   # the seconds since the job began at which step $1 ended: summary.json's `seconds`
# peak FILE CMD...: runs CMD, and writes FILE the peak resident MB of the largest process among CMD and its children
peak() {
  "$PY" -c 'import resource, subprocess, sys
rc = subprocess.call(sys.argv[2:])
open(sys.argv[1], "w").write(str(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss >> 10))
sys.exit(rc)' "$@"
}

venv() {
  if [ -n "${PY:-}" ]; then log "python: PY, as given"; return; fi
  command -v uv > /dev/null || { log "no uv on this node (PY=PYTHON with numpy)"; exit 2; }
  uv venv -q --python 3.12 "$W/venv" && uv pip install -q --python "$W/venv/bin/python" numpy==2.5.3 \
    || { log "couldn't make the venv"; exit 2; }
  PY=$W/venv/bin/python
}

# reach PID: the endpoint at 127.0.0.1:$LOCAL, behind the forward ssh process PID holds (0: no forward, the fake's own
# port); writes reach.json and prints one line, REACHED or UNREACHED and the reason's class in words. Exits 0, 3, 4, 5.
reach() {
  timeout 50 python3 - "$1" "$LOCAL" "$OUT/reach.json" "$W/tunnel.log" "$PORT" "$MODEL" <<'EOF'
import http.client, json, os, socket, sys, time
pid, local, out, tlog, port, model = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
where = "the serving node" if pid else "this node (the fake endpoint)"
# ssh's log names the serving node, so none of its lines is printed or kept: only the class of the first needle found.
CLASSES = (
    ("administratively prohibited", "forward forbidden", 3,
     "the key is accepted, and the serving node forbids a forward to this port: its authorized_keys line must permit "
     "127.0.0.1:PORT"),
    ("open failed", "nothing listens", 4, "the key is accepted and the forward opens; nothing listens on the port yet"),
    ("Permission denied", "key refused", 3,
     "the serving node refuses the key: infra authorizes its public half there (a pod's restart drops it)"),
    ("UNPROTECTED PRIVATE KEY", "key unusable", 3, "ssh won't use the key file: its permissions are too open"),
    ("invalid format", "key unusable", 3, "ssh can't read the key file"),
    ("Address already in use", "local port taken", 3, "this node's local port was taken before ssh bound it: run again"),
    ("timed out", "no tcp", 3,
     "no TCP to the serving node's ssh port: the address or port is stale, or it doesn't admit this node"),
    ("Connection refused", "ssh refused", 3, "the serving node's ssh port refuses: no sshd there, or a stale address or port"),
    ("No route", "no route", 3, "no route to the serving node"),
    ("Network is unreachable", "no route", 3, "no route to the serving node"),
    ("Could not resolve", "no name", 3, "the serving node's name doesn't resolve"),
    ("Connection closed", "ssh closed", 3, "the serving node's sshd closed the connection before the forward"),
    ("Connection reset", "ssh closed", 3, "the serving node's sshd reset the connection before the forward"),
)


def done(code, why, cls, **kw):
    json.dump({"reached": code == 0, "exit": code, "class": cls, "why": why, "port": int(port), "local": local,
               "by": "connect" if model else "/v1/models", **kw}, open(out, "w"), indent=1)
    print(("REACHED: " if code == 0 else "UNREACHED: ") + why, flush=True)
    sys.exit(code)


def ssh_said(wait=1.5):
    """(exit, class, words) of the first needle in ssh's log, read for up to `wait` seconds (a refusal reaches the log
    as the connection it refused closes, not before), or None: no line this check knows."""
    end = time.monotonic() + wait
    while pid:
        try:
            text = open(tlog, errors="replace").read()
        except OSError:
            text = ""
        for needle, cls, code, why in CLASSES:
            if needle in text:
                return code, cls, why
        if time.monotonic() >= end:
            break
        time.sleep(0.1)
    return None


def failed(what, cls, **kw):
    """ssh's reason if its log has one, else `what`, of class `cls` (exit 3)."""
    said = ssh_said()
    if said is None:
        done(3, what, cls, **kw)
    done(said[0], said[2], said[1], **kw)


UNKNOWN = ("; ssh's log has no line this check knows, and it names the serving node, so it is not printed: run the "
           "job's ssh by hand on this node")


start = time.monotonic()
while True:
    if pid:
        try:
            os.kill(pid, 0)
        except OSError:
            failed("ssh exited before the forward was up" + UNKNOWN, "unclassified")
    try:
        socket.create_connection(("127.0.0.1", local), timeout=1).close()
        break
    except OSError:
        if time.monotonic() - start > 25:
            failed("no forward within 25 s" + (UNKNOWN if pid else ": the fake endpoint isn't listening"), "unclassified")
        time.sleep(0.2)
up = {"tunnel_s": round(time.monotonic() - start, 3)}
if model:
    for _ in range(2):
        s = socket.create_connection(("127.0.0.1", local), timeout=5)
        s.settimeout(3)
        try:
            closed = s.recv(1) == b""
        except socket.timeout:
            closed = False
        except OSError:
            closed = True
        s.close()
        if closed:
            failed("the forward is up, and a connect that sent nothing was closed within 3 s", "endpoint", **up)
    said = ssh_said(0.5)
    if said is not None:
        done(said[0], said[2], said[1], **up)
    size = len(json.dumps(model).encode())
    if size > 64:
        done(5, f"a listener is behind the port, and MODEL is {size} bytes as JSON, past the 64 an id request holds: "
                "every connection would fail in the client", "model name", **up)
    done(0, f"a listener at 127.0.0.1:{port} of {where} held two connects open 3 s each (nothing sent, no round trip "
            f"timed); the model is MODEL, {model}", "reached", model=model, **up)
gets, models = [], None
for _ in range(5):
    t = time.monotonic()
    try:
        c = http.client.HTTPConnection("127.0.0.1", local, timeout=10)
        c.request("GET", "/v1/models")
        r = c.getresponse()
        body = r.read()
        c.close()
    except (OSError, http.client.HTTPException) as e:
        failed(f"the forward is up and GET /v1/models failed ({type(e).__name__}): if the endpoint doesn't answer it, "
               "give MODEL=NAME", "endpoint", **up)
    gets.append(round((time.monotonic() - t) * 1000, 1))
    if r.status != 200:
        done(3, f"GET /v1/models answered HTTP {r.status}: if the endpoint doesn't serve it, give MODEL=NAME",
             "endpoint", **up)
    try:
        models = [m.get("id") for m in json.loads(body).get("data", [])]
    except (ValueError, AttributeError):
        done(3, "GET /v1/models answered 200 with no model list: give MODEL=NAME", "endpoint", **up)
if len(models) != 1 or not isinstance(models[0], str):
    done(3, f"GET /v1/models lists {len(models)} models, not one: give MODEL=NAME", "endpoint", **up)
size = len(json.dumps(models[0]).encode())
if size > 64:
    done(5, f"the endpoint's model name is {size} bytes as JSON, past the 64 an id request holds: every connection "
            "would fail in the client", "model name", get_ms=gets, **up)
done(0, f"{models[0]} at 127.0.0.1:{port} of {where}, GET round trips {gets} ms", "reached", get_ms=gets,
     model=models[0], **up)
EOF
  local rc=$?
  [ "$rc" = 124 ] && { log "UNREACHED: the reach check took over 50 s"; rc=3; }
  return "$rc"
}

if [ -z "${FAKE:-}" ]; then
  [ -n "$peer" ] && [ "$peer" != "${peer#*@}" ] || { log "PEER: USER@HOST, the serving node"; exit 2; }
  PORT=${PORT:?PORT: the endpoint port on 127.0.0.1 of the serving node}
  KEY=${KEY:-$HOME/.ssh/vy-tick-tunnel}
  LOCAL=${LOCAL:-$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')}
  how="GET /v1/models"; [ -n "$MODEL" ] && how="a connect (MODEL given)"
  log "reach: the serving node's 127.0.0.1:$PORT through one ssh forward, here 127.0.0.1:$LOCAL, key $(basename "$KEY")," \
      "by $how"
  if [ ! -r "$KEY" ]; then
    log "UNREACHED: no key $(basename "$KEY") on this node: the tunnel key infra keeps here (KEY=FILE)"
    echo '{"reached": false, "exit": 3, "class": "no key", "why": "the key file is not readable on this node"}' > "$OUT/reach.json"
    exit 3
  fi
  # ssh's whole configuration, so that the address is on no command line. A known-hosts file of the run's own: RunPod
  # hands a gone pod's address and port to a new pod, whose host key a shared file refuses.
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
  reach "$TUN" || exit
  mark reach
  [ -n "${REACH_ONLY:-}" ] && exit 0
  venv
  mark venv
else
  venv
  mark venv
  payload=$(PYTHONPATH=$SRC "$PY" -c "import sys
from verity.protocols.certifier.verifier.warden import WardenParams
print(WardenParams(cell_bytes=int(sys.argv[1])).payload)" "$CELL_BYTES") || { log "couldn't read the schedule's P"; exit 2; }
  FIRST_MS=${FIRST_MS:-2000}; TOKEN_MS=${TOKEN_MS:-220}
  fake=("$HERE/paced.py"); how="an id every $TOKEN_MS ms from $FIRST_MS ms after its request (certside/paced.py)"
  if [ "$TOKEN_MS" = 0 ]; then
    [ -z "${FAKE_IDS:-}" ] || { log "FAKE_IDS is paced.py's: not with TOKEN_MS=0"; exit 2; }
    fake=(benchmarks/network_traces/warden_serve.py)
    how="each answer whole $FIRST_MS ms after its request (warden_serve's fake)"
  elif [ -n "${FAKE_IDS:-}" ]; then
    fake+=(--ids-below "$FAKE_IDS"); how="$how, its ids below $FAKE_IDS"
  fi
  PYTHONPATH=$SRC "$PY" "${fake[@]}" fake --listen 127.0.0.1:0 --first-ms "$FIRST_MS" --token-ms "$TOKEN_MS" \
    --id-payload "$payload" > "$W/fake.out" 2> "$W/fake.err" &
  FAKED=$!
  for _ in $(seq 100); do [ -s "$W/fake.out" ] && break; sleep 0.1; done
  LOCAL=$("$PY" -c 'import json, sys; print(json.loads(open(sys.argv[1]).readline())["listening"][1])' "$W/fake.out") \
    || { log "the fake endpoint didn't start: $(tail -c 300 "$W/fake.err")"; exit 2; }
  PORT=$LOCAL
  log "fake: a fake endpoint at 127.0.0.1:$LOCAL, the id form at P = $payload, $how"
  reach 0 || exit
  mark reach
  [ -n "${REACH_ONLY:-}" ] && exit 0
fi
export PYTHONPATH=$SRC

REHEARSE=("$PY" benchmarks/network_traces/warden_serve.py)
[ -n "${ID_BOUND:-}" ] && REHEARSE=("$PY" "$HERE/bounded.py" "$ID_BOUND")
UPSTREAM=(--upstream "127.0.0.1:$LOCAL")
[ -n "$MODEL" ] && UPSTREAM+=(--model "$MODEL")
# try DIR: steps 2 and 3 at SCHEDULE and TRAFFIC as they stand, every file under DIR; 0 when it is ACCEPTED (rehearse
# and check both exit 0). TRY is the lines' label: none for the one try, "try N: " of two. Its body is not indented:
# its two here-documents can't be.
TRY=
try() {
local T=$1 rehearsed checked
mkdir -p "$T"
log "${TRY}rehearse: ${UPSTREAM[*]}, ${TRAFFIC[*]}, ${SCHEDULE[*]}, watermark${ID_BOUND:+, prompt ids below $ID_BOUND}"
peak "$W/rehearse.mb" "${REHEARSE[@]}" rehearse "${UPSTREAM[@]}" --watermark --out "$T/warden" "${TRAFFIC[@]}" \
  "${SCHEDULE[@]}" 2>&1 | tee "$T/rehearse.log"
rehearsed=${PIPESTATUS[0]}
mark rehearse
if [ -n "$TUN" ]; then
  if kill -0 "$TUN" 2>/dev/null; then log "${TRY}forward: up to the end"; else log "${TRY}forward: gone before the end"; fi
fi
log "${TRY}rehearse: rc $rehearsed"
if [ "${RECHECK:-1}" = 0 ]; then
  checked=$rehearsed   # rehearse's own exit is its check's verdict
  log "${TRY}check: not run again (RECHECK=0): check.json is rehearse's own, whose verdict is its last line above"
else
  peak "$W/check.mb" "$PY" benchmarks/network_traces/warden_serve.py check --run "$T/warden" > "$T/check.txt" 2>&1
  checked=$?
  log "${TRY}check: rc $checked: $(head -c 600 "$T/check.txt")"
fi
mark check
"$PY" - "$T" "$rehearsed" "$checked" "$W" "${MARKS[@]}" <<'EOF'
import json, pathlib, sys
out, tmp = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[4])


def cgroup():
    """This run's cgroup's memory, MB (cgroup v2): its peak, page cache included, and its anonymous and file MB now."""
    try:
        path = pathlib.Path("/sys/fs/cgroup" + open("/proc/self/cgroup").read().strip().rpartition("::")[2])
        stat = dict(line.split() for line in (path / "memory.stat").read_text().splitlines())
        return {"peak_mb": int((path / "memory.peak").read_text()) >> 20, "anon_mb": int(stat["anon"]) >> 20,
                "file_mb": int(stat["file"]) >> 20}
    except (OSError, ValueError, KeyError):
        return None


def mb(name):
    try:
        return int((tmp / name).read_text())
    except (OSError, ValueError):
        return None


w = json.loads((out / "warden" / "warden.json").read_text())
check = json.loads((out / "warden" / "check.json").read_text())
events = sorted([(c["admitted"], 1) for c in w["connections"] if c.get("admitted") is not None]
                + [(c["ended"], -1) for c in w["connections"] if c.get("admitted") is not None and c.get("ended") is not None])
open_now = peak = 0
for _, step in events:
    open_now += step
    peak = max(peak, open_now)
size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
summary = {"accepted": check["accepted"], "problems": check["problems"][:10], "summary": check["summary"],
           "released": sum(c.get("release") is not None for c in w["connections"]), "peak_open": peak,
           "watermark": w.get("watermark"), "out_bytes": size,
           "exit": {"rehearse": int(sys.argv[2]), "check": int(sys.argv[3])},
           "seconds": {k: int(v) for k, v in (m.split("=") for m in sys.argv[5:])},
           "max_rss_mb": {"certifier": (w.get("usage") or {}).get("max_rss_mb"),
                          "publisher": (w.get("publisher_usage") or {}).get("max_rss_mb"),
                          "largest_in_rehearse": mb("rehearse.mb"), "check": mb("check.mb")},
           "cgroup": cgroup()}
(out / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({k: summary[k] for k in ("accepted", "released", "peak_open", "watermark", "out_bytes", "seconds",
                                          "max_rss_mb", "cgroup")}))
EOF
"$PY" "$HERE/report.py" "$T" | tee "$T/report.txt"
log "${TRY}done in $SECONDS s: rehearse rc $rehearsed, check rc $checked, report rc ${PIPESTATUS[0]}"
[ "$rehearsed" = 0 ] && [ "$checked" = 0 ]
}

if [ "$TRIES" = 1 ]; then
  try "$OUT"
  exit
fi
# Two tries. `seconds` in a try's summary.json are the job's own, from its start: `start` is the try's.
TRY="try 1: "; mark start
try "$OUT/try1"; first=$?
second=
if [ "$first" = 0 ]; then
  log "try 1 ACCEPTED: try 2 starts, with --lanes $LANES2, --rate $RATE2, --requests $REQUESTS2"
  LANES=$LANES2; RATE=$RATE2; REQUESTS=$REQUESTS2; plan
  TRY="try 2: "; mark start
  try "$OUT/try2"; second=$?
else
  log "try 1 is NOT ACCEPTED: try 2 is not started. Why, from its record (out/try1/report.txt, above, has its counts):"
fi
"$PY" - "$OUT" "$first" "$second" <<'EOF'
import json, pathlib, sys
out, first, second = pathlib.Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]


def read(path):
    try:
        return json.loads((out / path).read_text())
    except (OSError, ValueError):
        return None


def of(name, rc):
    """A try's part of the job's summary: its verdict, its exits, its seconds of the job and what its record counts."""
    s, deployment = read(f"{name}/summary.json"), read(f"{name}/warden/deployment.json") or {}
    if s is None:
        return {"run": True, "accepted": False, "exit": rc, "why": "the try left no summary.json: read its rehearse.log"}
    params, origin = deployment.get("params") or {}, deployment.get("origin_unix")
    return {"run": True, "accepted": bool(s["accepted"]) and rc == 0, "exit": s["exit"], "seconds": s["seconds"],
            "released": s["released"], "problems": len(s["problems"]), "out_bytes": s["out_bytes"],
            "digest": (deployment.get("digest") or "")[:16], "windows": deployment.get("windows"),
            # its buckets' first and last instants, unix seconds: the certifier's origin and its windows after it
            "buckets": None if origin is None else
            [origin, origin + deployment["windows"] * params["window"] * params["bucket_ms"] / 1000]}


tries = {"try1": of("try1", first)}
tries["try2"] = of("try2", int(second)) if second else {"run": False, "accepted": False,
                                                       "why": "try 1 was not ACCEPTED, so try 2 was not started"}
if first:
    for problem in (read("try1/summary.json") or {}).get("problems") or ["no problem is listed: read out/try1/rehearse.log"]:
        print("  - " + str(problem)[:300])
both = [tries[name].get("seconds") for name in ("try1", "try2")]
buckets = [tries[name].get("buckets") for name in ("try1", "try2")]
files = [p for p in out.rglob("*") if p.is_file()]
try:
    path = pathlib.Path("/sys/fs/cgroup" + open("/proc/self/cgroup").read().strip().rpartition("::")[2])
    peak_mb = int((path / "memory.peak").read_text()) >> 20
except (OSError, ValueError):
    peak_mb = None
summary = {"tries": 2, "accepted": tries["try1"]["accepted"] and tries["try2"]["accepted"], **tries,
           # from try 1's rehearsal returning to try 2's being started: try 1's check, summary and report
           "between_s": both[1]["start"] - both[0]["rehearse"] if all(both) else None,
           # from try 1's last bucket's end to try 2's first bucket: what the serving node sees of no certifier
           "between_buckets_s": round(buckets[1][0] - buckets[0][1], 1) if all(buckets) else None,
           "cgroup_peak_mb": peak_mb, "out_bytes": sum(p.stat().st_size for p in files), "out_files": len(files)}
(out / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary))
EOF
[ "$first" = 0 ] || { log "done in $SECONDS s: try 1 NOT ACCEPTED, try 2 not started"; exit 1; }
[ "$second" = 0 ] || { log "done in $SECONDS s: try 1 ACCEPTED, try 2 NOT ACCEPTED"; exit 6; }
log "done in $SECONDS s: try 1 ACCEPTED, try 2 ACCEPTED"
