#!/usr/bin/env bash
# Restart node 2's fill runner once no timed window runs or waits, so a new fill_runner.py takes effect without touching a window.
# The runner is the user unit pouw-infra-fill (enabled at boot), so this restarts the unit rather than a tmux pane, which would
# leave two runners; the new runner adopts the running fill jobs from their .pgid files.
set -u
LOG=/workspace/pouw/infra/logs/fill-runner-restart.log
for i in $(seq 1 720); do
  s=$(head -1 /workspace/pouw/fill/status.txt)
  case "$s" in
    *"timed False, window waiting False"*)
      P=$(systemctl --user show -p MainPID --value pouw-infra-fill)
      systemctl --user restart pouw-infra-fill
      sleep 20
      N=$(systemctl --user show -p MainPID --value pouw-infra-fill)
      echo "$(date -u +%FT%TZ) restarted pouw-infra-fill ($P -> ${N:-NONE}); now: $(head -1 /workspace/pouw/fill/status.txt)" >> "$LOG"
      exit 0 ;;
  esac
  sleep 5
done
echo "$(date -u +%FT%TZ) no quiet moment within an hour; pouw-infra-fill not restarted" >> "$LOG"
