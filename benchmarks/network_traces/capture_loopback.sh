#!/usr/bin/env bash
# capture_flows.py on one machine. Not a served run: the stand-in front, the network certifier and the load on one
# machine's loopback (`warden_serve.py rehearse --fake --watermark`), a capture of the loopback beside them, then the
# reader on the two. A `research run` job from the tree's root (--cwd source), on a machine where the job may capture:
# `tcpdump` there, allowed to open the loopback as the job's user. A pool node's job may not.
#   [PY=PYTHON] [EGRESS=bytes|id] bash benchmarks/network_traces/capture_loopback.sh REHEARSE-FLAG...
# REHEARSE-FLAG... are `warden_serve.py rehearse`'s own, the traffic and the schedule; the job gives it --fake,
# --watermark, --out and the egress form. PY is a python with the tree's packages (python3 by default, as under
# `uv run`). EGRESS is the run's egress form, the launch's to give and no flag's: bytes when it isn't given, text
# requests answered in vLLM's stock framing; id, `rehearse --ids`, id requests each answered an id event a token, the
# form a network certifier holds a served run's server to.
# 1. The capture: `tcpdump -i lo -U -s 130 tcp`, started before the rehearsal and stopped once its file has stopped
#    growing. It holds every TCP flow on the loopback meanwhile, other programs' among them, so it stays in the job's
#    temporary directory, which is deleted.
# 2. The run: the rehearsal, with its offline check; its run directory is $OUT/run.
# 3. What is kept of the capture: $OUT/run.pcap, its packets on the run's two ports, the stand-in front's and the
#    relay's (`deployment.json`), at most each one's first 130 bytes: every header and, where the headers are
#    shorter, the first bytes of its content.
# 4. The reader: `capture_flows.py` on $OUT/run.pcap and $OUT/run. The front and the relay share the loopback's address,
#    so the front's is given with its port; the load's connections to the relay are among the other flows. Its result
#    is $OUT/capture_flows.json and its table $OUT/capture_flows.txt, also printed.
# The exit is the reader's (0 read, 2 refused) where the rehearsal's check accepted its run; 1 where it didn't, the
# run left no capture or its form isn't EGRESS's; 4 where there is no tcpdump, no right to capture or no such python,
# or EGRESS isn't a form or a flag gives one.
set -uo pipefail
RUN=${RESEARCH_RUN_DIR:?run this under research run}; OUT=$RUN/out; mkdir -p "$OUT"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${PY:-python3}
W=$(mktemp -d); CAPTURE_PID=
cleanup() { [ -n "$CAPTURE_PID" ] && kill -INT "$CAPTURE_PID" 2>/dev/null; rm -rf "$W"; }
trap cleanup EXIT
trap 'exit 143' TERM INT
log() { echo "capture-loopback: $*"; }

EGRESS=${EGRESS:-bytes}
case "$EGRESS" in
  bytes) form=() ;;
  id) form=(--ids) ;;
  *) log "EGRESS is bytes or id, not '$EGRESS'"; exit 4 ;;
esac
for flag in "$@"; do
  case "$flag" in
    --ids|--egress|--egress=*) log "the egress form is EGRESS's, not $flag's: EGRESS=id for the id form"; exit 4 ;;
  esac
done
"$PY" -c 'import verity.protocols.certifier.verifier.warden' 2> /dev/null \
  || { log "PY has no verity: give a python with the tree's packages, or run this under uv run"; exit 4; }
command -v tcpdump > /dev/null || { log "no tcpdump on this machine"; exit 4; }
tcpdump -i lo -U -s 130 -w "$W/lo.pcap" tcp 2> "$W/tcpdump.log" &
CAPTURE_PID=$!
# tcpdump says it listens once it captures, and exits before that where the job has no right to
until grep -q "listening on" "$W/tcpdump.log"; do
  kill -0 "$CAPTURE_PID" 2>/dev/null || { CAPTURE_PID=; log "tcpdump couldn't capture the loopback here"; exit 4; }
  sleep 0.2
done

"$PY" "$HERE/warden_serve.py" rehearse --fake --watermark --out "$OUT/run" ${form[@]+"${form[@]}"} "$@"
rehearsed=$?

# tcpdump gets its packets a little after they pass: the last are in the file only once it has stopped growing
size=-1
while now=$(stat -c %s "$W/lo.pcap") && [ "$now" != "$size" ]; do size=$now; sleep 1; done
kill -INT "$CAPTURE_PID"; wait "$CAPTURE_PID"; CAPTURE_PID=

ends=$("$PY" -c 'import json, sys
d = json.load(open(sys.argv[1]))
print(*d["upstream"], *d["listen"], d["params"]["egress"])' "$OUT/run/deployment.json") \
  || { log "the rehearsal left no deployment"; exit 1; }
read -r front port relay listen ran <<< "$ends"
[ "$ran" = "$EGRESS" ] || { log "the run's egress form is $ran, not EGRESS's ($EGRESS)"; exit 1; }
tcpdump -r "$W/lo.pcap" -w "$OUT/run.pcap" "tcp port $port or tcp port $listen" 2> /dev/null \
  || { log "the capture couldn't be read back"; exit 1; }

"$PY" "$HERE/capture_flows.py" --capture "$OUT/run.pcap" --server "$front:$port" --certifier "$relay" \
  --run "$OUT/run" --out "$OUT/capture_flows.json" | tee "$OUT/capture_flows.txt"
read=${PIPESTATUS[0]}
[ "$rehearsed" = 0 ] || { log "the rehearsal's check didn't accept its run (exit $rehearsed)"; exit 1; }
exit "$read"
