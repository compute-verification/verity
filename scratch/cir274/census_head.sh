#!/usr/bin/env bash
# CIR-274 item 2: delivery B's census at the source tree's head, at 4 layers and then 78, rank 0, (43, 2). Sent with
# `research run --send census_head.sh --send p3_window.py`; the 78-layer process is killed by its PID if its RSS passes
# 48 GB (the lead's bound), and its peak is written to $OUT/watch_l78.txt either way.
set -u
IN="$RESEARCH_RUN_DIR/inputs"
uv sync -q --all-packages --extra torch-cpu || exit 2
git rev-parse HEAD > "$OUT/source_sha.txt" 2>/dev/null || true
for L in 4 78; do
  .venv/bin/python "$IN/p3_window.py" "$OUT" "$L" 0 &
  pid=$!
  peak=0
  while kill -0 "$pid" 2>/dev/null; do
    rss=$(awk '/VmRSS/ {print $2}' "/proc/$pid/status" 2>/dev/null || echo 0)
    [ -n "$rss" ] && [ "$rss" -gt "$peak" ] && peak=$rss
    if [ "${rss:-0}" -gt $((48 * 1024 * 1024)) ]; then
      echo "killed at rss_kb=$rss" >> "$OUT/watch_l$L.txt"
      kill -9 "$pid"
    fi
    sleep 1
  done
  wait "$pid"; rc=$?
  echo "layers=$L rc=$rc peak_rss_kb=$peak" | tee -a "$OUT/watch_l$L.txt"
  [ "$rc" -eq 0 ] || exit "$rc"
done
