# the match gate at the 128k replay's captured steps on node A, every weight composed from the checkpoint: step 5611's long
# rows alone (keys past 2,048), then every call of step 5611 and of step 1980
# Beside a replay on the same node: START_AVAIL_G holds each pass until that much memory is available (until UNTIL), and
# MIN_AVAIL_G stops the gate (its whole process group, never the replay) and skips the passes left once available memory is
# under it.
set -u
: "${UNTIL:?UNTIL: a UTC time}"
CAP=/workspace/spot-capture/r20261107-222136-ea99
CK=/workspace/fs/glm53/aca966e4e02791568aa6a4ced368624b3d897f42
I=$RESEARCH_RUN_DIR/inputs; O=$RESEARCH_RUN_DIR/out; mkdir -p "$O"
PY=${PY:-/workspace/venv312/bin/python}
END=$(date -d "$UNTIL" +%s); STOPPED=
avail() { awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo; }
run() {
  name=$1; shift
  [ -z "$STOPPED" ] || { echo "== $name: not run (the gate was stopped for memory)"; return; }
  if [ -n "${START_AVAIL_G:-}" ]; then
    while [ "$(avail)" -lt "$START_AVAIL_G" ]; do
      [ "$(date +%s)" -lt "$END" ] || { echo "== $name: not run, under ${START_AVAIL_G}G available until $UNTIL"; return; }
      sleep 30
    done
  fi
  echo "== $name: $* $(date -u +%FT%TZ), $(avail)G available"
  setsid nice -n 10 "$PY" "$I/stage.py" --capture "$CAP" --out "$O/$name" --checkpoint "$CK" --jobs "${JOBS:-32}" \
    --deadline "$END" "$@" > "$O/$name.log" 2>&1 &
  pid=$!; low=
  while sleep 10; kill -0 "$pid" 2>/dev/null; do
    if [ -n "${MIN_AVAIL_G:-}" ] && [ "$(avail)" -lt "$MIN_AVAIL_G" ]; then
      low=$(avail); kill -TERM -- "-$pid"; sleep 20; kill -KILL -- "-$pid" 2>/dev/null; STOPPED=1
      echo "== $name: stopped at $(date -u +%FT%TZ): ${low}G available, under ${MIN_AVAIL_G}G"; break
    fi
  done
  wait "$pid"; echo "rc $? $(date -u +%FT%TZ), $(avail)G available"
  grep -a -E "^(gate|rows past)" "$O/$name.log" | cut -c1-500
  "$PY" -c "import json; r=json.load(open('$O/$name/gate.json')); print(json.dumps({k: r.get(k) for k in ('from_checkpoint', 'rows', 'instances', 'deadline', 'schedule_ops')}))" 2>/dev/null
}
run gate-step5611-long --step 5611 --every-call 4 --rows-past-keys 2048
run gate-step5611 --step 5611 --every-call 1
run gate-step1980 --step 1980 --every-call 1
df -BG --output=avail /workspace | tail -1
