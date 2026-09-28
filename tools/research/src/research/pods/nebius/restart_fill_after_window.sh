#!/usr/bin/env bash
# Restart node 2's fill runner once no timed window runs or waits, so a new fill_runner.py takes effect without touching a window.
# It runs in tmux session pouw-infra-fill without a supervisor loop, so this stops it and starts it again in that pane; the new
# runner adopts the running fill jobs from their .pgid files.
set -u
LOG=/workspace/pouw/infra/logs/fill-runner-restart.log
for i in $(seq 1 720); do
  s=$(head -1 /workspace/pouw/fill/status.txt)
  case "$s" in
    *"timed False, window waiting False"*)
      P=$(pgrep -f "^python3 /workspace/pouw/infra/bin/fill_runner.py$" | head -1)
      [ -n "$P" ] && kill -TERM "$P"
      for j in $(seq 1 60); do [ -n "$P" ] && kill -0 "$P" 2> /dev/null || break; sleep 0.5; done
      tmux send-keys -t pouw-infra-fill "python3 /workspace/pouw/infra/bin/fill_runner.py" C-m
      sleep 20
      N=$(pgrep -f "^python3 /workspace/pouw/infra/bin/fill_runner.py$" | head -1)
      echo "$(date -u +%FT%TZ) pous infra restarted fill_runner ($P -> ${N:-NONE}) for the waiters() fix; now: $(head -1 /workspace/pouw/fill/status.txt)" >> "$LOG"
      exit 0 ;;
  esac
  sleep 5
done
echo "$(date -u +%FT%TZ) pous infra: no quiet moment within an hour; fill_runner not restarted" >> "$LOG"
