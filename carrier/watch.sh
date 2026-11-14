#!/bin/bash
# watch.sh RUN [MAX_S [PATTERN]]: wait until RUN has ended, or its stdout.log has a line matching PATTERN, or MAX_S seconds
# (default 1080, under the 20 minutes a tool call may block) have gone; then print the clock, its state and its last
# lines. Polls `research fetch` every 30 s.
run=$1; max=${2:-1080}; pat=${3:-}
d=~/.research/runs/$run; t0=$(date +%s)
while :; do
  timeout 100 research fetch "$run" --all >/dev/null 2>&1
  st=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('state'), d.get('rc'))" "$d/status.json" 2>/dev/null)
  case "$st" in running*|"") ;; *) echo "ENDED: $st"; break;; esac
  if [ -n "$pat" ] && grep -qE "$pat" "$d/stdout.log" 2>/dev/null; then echo "MATCHED: $pat"; break; fi
  if [ $(( $(date +%s) - t0 )) -ge "$max" ]; then echo "STILL RUNNING after ${max}s: $st"; break; fi
  sleep 30
done
date -u +%H:%M:%SZ
tail -6 "$d/stdout.log" 2>/dev/null | cut -c1-1500
