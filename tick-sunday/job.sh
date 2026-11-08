#!/usr/bin/env bash
# The certified tick's network certifier side: warden_serve.py rehearse --watermark on this host, in front of vllm's
# GLM-5.3 endpoint (serve-openai --admit tick, τ_b 100 ms, the recorder) on the serving node's loopback, reached through an
# ssh local forward. A `research run` job from the shipped tree's root (--cwd source); tick-sunday/launch.sh launches it:
#   PEER=research@HOST PORT=P [KEY=FILE] [RATE=1.67] [REQUESTS=3000] [SEED=0] [REACH_ONLY=1 | FAKE=1] bash tick-sunday/job.sh
# 1. Reach, in under a minute, before anything is sent: KEY readable, the tunnel up (ssh -N -L 127.0.0.1:LOCAL to the peer's
#    127.0.0.1:PORT), the endpoint's /v1/models answered through it, each GET's round trip timed: $OUT/reach.json. A
#    failure says which of those it was and exits 3. REACH_ONLY=1 stops here.
# 2. The rehearsal, into $OUT/warden: RATE Poisson arrivals a second for REQUESTS requests of warden_serve's varied traffic
#    (`varied_traffic`: prompts of 16-1024 tokens of filler words and a question, max_tokens 16-1024 under EOS, both
#    log-uniform), greedy, top_p the endpoint's own, the model the endpoint's one (/v1/models), at SCHEDULE below.
#    warden.json holds each connection's release N and release bucket (`release`, `ingress_end`), the watermark's last
#    line, lost releases and error; beside it the records, their certificates, the cell rows and check.json.
# 3. warden_serve.py check on $OUT/warden, from its files alone: $OUT/check.txt; then $OUT/summary.json.
# FAKE=1 runs 2 and 3 against warden_serve's fake endpoint (FIRST_MS, TOKEN_MS) with no tunnel: the dry run.
#
# SCHEDULE, for GLM-5.3 at TP8 eager under the tick and the recorder, through a tunnel of about 100 ms round trip:
# - B = 256, r = 1: r·P = 247 bytes a bucket each way. A streamed /v1/completions token is about 200-280 bytes (its
#   event and chunk framing), so egress drains about 10 tokens a second, under the slowest per-request stream expected
#   there; the server is read eagerly, so this pacing never slows it, and what it streams faster waits in the buffer.
# - D = 150 buckets (15 s) from admission: ingress of the largest request (about 5 KB, 21 buckets at r·P), the release's
#   bucket close, the tunnel's one-way trip, the tick gate's wait for the watermark line, a queued prefill, the first P
#   bytes back. What the server streams before D is a head start that covers a slower stretch later.
# - lanes 256, queue 256: a connection lives D plus its response's cells, about 40 s on average and 2 min at most
#   (1,024 tokens), so about 70 are open at once at 1.67 a second (Poisson, so rarely past 110); 256 never refuses.
# - ingress 96 cells within 100 buckets, egress 4096 cells: above the largest request (21) and response (about 1,200).
# - tail 150 s: the last arrival's D and longest response; windows of 600 buckets (60 s).
set -uo pipefail
D=${RESEARCH_RUN_DIR:?run this under research run}; OUT=$D/out; mkdir -p "$OUT"
SRC=$PWD
RATE=${RATE:-1.67}; REQUESTS=${REQUESTS:-3000}; SEED=${SEED:-0}
SCHEDULE=(--cell-bytes 256 --slots 1 --bucket-ms 100 --delay 150 --lanes 256 --queue 256 --window 600
          --ingress-cells 96 --ingress-buckets 100 --egress-cells 4096 --every 1)
TRAFFIC=(--rate "$RATE" --requests "$REQUESTS" --seed "$SEED" --temperature 0 --prompt-max 1024 --max-tokens 1024
         --tail "${TAIL:-150}")
W=$(mktemp -d); export TMPDIR=$W
TUN=
cleanup() { [ -n "$TUN" ] && kill "$TUN" 2>/dev/null; rm -rf "$W"; }
trap cleanup EXIT
log() { echo "$(date -u +%H:%M:%SZ) tick: $*" | tee -a "$OUT/job.log"; }

if [ -z "${FAKE:-}" ]; then
  PEER=${PEER:?PEER: research@HOST, the serving node}; PORT=${PORT:?PORT: the endpoint port on 127.0.0.1 of the peer}
  KEY=${KEY:-$HOME/.ssh/vy-tick-tunnel}
  LOCAL=${LOCAL:-$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')}
  log "reach: $PEER's 127.0.0.1:$PORT from $(hostname) as 127.0.0.1:$LOCAL, key $KEY"
  if [ ! -r "$KEY" ]; then
    log "UNREACHED: no key $KEY on $(hostname): the tunnel key @infra makes here and authorizes on the peer (KEY=FILE)"
    echo '{"reached": false, "why": "no key"}' > "$OUT/reach.json"; exit 3
  fi
  ssh -N -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=15 -o IdentitiesOnly=yes -i "$KEY" -L "127.0.0.1:$LOCAL:127.0.0.1:$PORT" "$PEER" \
      2> "$OUT/tunnel.log" &
  TUN=$!
  timeout 50 python3 - "$TUN" "$LOCAL" "$OUT/reach.json" "$OUT/tunnel.log" "$PEER" "$PORT" <<'EOF'
import http.client, json, os, socket, sys, time
pid, local, out, tlog, peer, port = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]

def done(reached, why, **kw):
    json.dump({"reached": reached, "why": why, "peer": peer, "port": int(port), "local": local, **kw}, open(out, "w"),
              indent=1)
    print(("REACHED: " if reached else "UNREACHED: ") + why, flush=True)
    sys.exit(0 if reached else 3)

def ssh_said():
    text = open(tlog, errors="replace").read()
    for needle, why in (("administratively prohibited", "the peer forbids this forward (permitopen must name the port)"),
                        ("open failed", "the tunnel is up but nothing listens on the peer's 127.0.0.1:PORT (is vllm's "
                                        "endpoint up?)"),
                        ("Permission denied", "the peer refuses the key: @infra authorizes its public half for research@"),
                        ("timed out", "no TCP to the peer's port 22: its security group must admit this host"),
                        ("Connection refused", "the peer's port 22 refuses: no sshd there, or the wrong address"),
                        ("No route", "no route to the peer"), ("Could not resolve", "the peer's name doesn't resolve")):
        if needle in text:
            return why
    return text.strip().splitlines()[-1] if text.strip() else "ssh said nothing"

start = time.monotonic()
while True:
    try:
        os.kill(pid, 0)
    except OSError:
        done(False, f"ssh exited: {ssh_said()}")
    try:
        socket.create_connection(("127.0.0.1", local), timeout=1).close()
        break
    except OSError:
        if time.monotonic() - start > 25:
            done(False, f"no tunnel within 25 s: {ssh_said()}")
        time.sleep(0.2)
up_s, gets, models = time.monotonic() - start, [], None
for _ in range(5):
    t = time.monotonic()
    try:
        c = http.client.HTTPConnection("127.0.0.1", local, timeout=10)
        c.request("GET", "/v1/models")
        r = c.getresponse()
        body = r.read()
        c.close()
    except (OSError, http.client.HTTPException) as e:
        done(False, f"tunnel up, /v1/models failed ({e!r}): {ssh_said()}", tunnel_s=round(up_s, 3))
    gets.append(round((time.monotonic() - t) * 1000, 1))
    if r.status != 200:
        done(False, f"/v1/models answered HTTP {r.status}", tunnel_s=round(up_s, 3))
    models = [m.get("id") for m in json.loads(body).get("data", [])]
if len(models) != 1:
    done(False, f"/v1/models lists {models}, not one model (MODEL is the endpoint's one)", tunnel_s=round(up_s, 3))
done(True, f"{models[0]} at {peer}'s 127.0.0.1:{port}, GET round trips {gets} ms", tunnel_s=round(up_s, 3),
     get_ms=gets, model=models[0])
EOF
  rc=$?
  [ "$rc" = 124 ] && { log "UNREACHED: the reach check took over 50 s"; exit 3; }
  [ "$rc" = 0 ] || exit 3
  [ -n "${REACH_ONLY:-}" ] && exit 0
  UPSTREAM=(--upstream "127.0.0.1:$LOCAL")
else
  UPSTREAM=(--fake --first-ms "${FIRST_MS:-800}" --token-ms "${TOKEN_MS:-60}")
fi

if [ -n "${PY:-}" ]; then
  log "python: $PY"
else
  command -v uv > /dev/null || { log "no uv on $(hostname) (PY=PYTHON with numpy)"; exit 2; }
  uv venv -q --python 3.12 "$W/venv" && uv pip install -q --python "$W/venv/bin/python" numpy==2.5.3 \
    || { log "couldn't make the venv"; exit 2; }
  PY=$W/venv/bin/python
fi
export PYTHONPATH=$SRC

log "rehearse: ${UPSTREAM[*]}, ${TRAFFIC[*]}, ${SCHEDULE[*]}, watermark"
"$PY" benchmarks/network_traces/warden_serve.py rehearse "${UPSTREAM[@]}" --watermark --out "$OUT/warden" \
  "${TRAFFIC[@]}" "${SCHEDULE[@]}" 2>&1 | tee "$OUT/rehearse.log"
rehearsed=${PIPESTATUS[0]}
if [ -n "$TUN" ]; then
  if kill -0 "$TUN" 2>/dev/null; then log "tunnel: up to the end"; else log "tunnel: gone before the end (tunnel.log)"; fi
fi
log "rehearse: rc $rehearsed"
"$PY" benchmarks/network_traces/warden_serve.py check --run "$OUT/warden" > "$OUT/check.txt" 2>&1
checked=$?
log "check: rc $checked: $(head -c 600 "$OUT/check.txt")"
"$PY" - "$OUT" <<'EOF'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
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
           "watermark": w.get("watermark"), "out_bytes": size}
(out / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({k: summary[k] for k in ("accepted", "released", "peak_open", "watermark", "out_bytes")}))
EOF
log "done: rehearse rc $rehearsed, check rc $checked"
[ "$rehearsed" = 0 ] && [ "$checked" = 0 ]
