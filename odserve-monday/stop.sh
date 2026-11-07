#!/usr/bin/env bash
# Stops vy-dc-h200-2 once every run on it has custody (after the serve, the send, the stock control and the check; the
# follow-on runs on its own node and needs nothing from this one):
#  1  the node gate, read on the node itself: every run dir under /workspace/research/runs holds a .custody saying
#     "preserved": true (a marker can record a failed publish) or a .fetched, but those ALLOW names (run dirs that never
#     get one: by default the three of Friday's that predate store run ids), and none is running but one ALLOW names
#     whose status's pid is gone. Ends "NODE GATE: N run dirs, bad 0", else nothing is stopped;
#  2  over ssh, `sudo /usr/local/sbin/vy-nebius-stop <the instance id from Nebius's metadata>`, which prints
#     "TERMINATED ... (nebius compute instance stop)" and "STOP: vy-nebius-stop exit 0". Exit 3 or 4: leave the node as it
#     is and post on INF-527. Never shutdown or poweroff: Nebius restarts a VM that shuts down and keeps billing it.
# A run whose custody key expired before it ended has "preserved": false: repair it first (`research fetch --all RID`,
# then `research data custody RID --publish`). CHECK=1 runs step 1 alone; DRY=1 prints both. Its lines in
# LOGS/node-gate.txt and LOGS/stop.txt.
#   [ALLOW="ID ..."] [CHECK=1] [DRY=1] bash stop.sh
set -uo pipefail
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$H/env.sh" || exit 3
ALLOW=${ALLOW-r20261106-100208-h2n2 r20261106-133857-h2n2 r20261106-134237-h2n2}
[[ $ALLOW =~ ^[A-Za-z0-9\ -]*$ ]] || { echo "REFUSED: ALLOW holds something other than run ids"; exit 3; }
# shellcheck disable=SC2016
gate='cd /workspace/research/runs || exit 2; bad=0
for d in r2*/; do d=${d%/}
  st=$(grep -o -E "\"state\": \"[a-z]+\"" "$d/status.json" 2>/dev/null | tail -1 | cut -d\" -f4)
  ok=; case " '"$ALLOW"' " in *" $d "*) ok=1;; esac
  if [ "$st" = running ]; then
    pid=$(grep -o -E "^ \"pid\": [0-9]+" "$d/status.json" | tail -1 | grep -o -E "[0-9]+$")
    [ -n "$ok" ] && [ -n "$pid" ] && [ ! -e "/proc/$pid" ] && { echo "ALLOWED $d (running, its pid $pid gone)"; continue; }
    echo "RUNNING $d"; bad=1; continue
  fi
  grep -qs "\"preserved\": true" "$d/.custody" || [ -e "$d/.fetched" ] && continue
  [ -n "$ok" ] && echo "ALLOWED $d (${st:-no state})" || { echo "NO-CUSTODY $d (${st:-no state})"; bad=1; }
done
echo "NODE GATE: $(ls -d r2*/ | wc -l) run dirs, bad $bad"; [ "$bad" = 0 ]'
# shellcheck disable=SC2016
stop='id=$(curl -sf -H "Metadata: true" http://metadata.nebius.internal/v1/instance-data/id) && [ -n "$id" ] || { echo "STOP: no instance id from the metadata"; exit 5; }; sudo /usr/local/sbin/vy-nebius-stop "$id"; echo "STOP: vy-nebius-stop exit $?"'
[ -z "${DRY:-}" ] || { echo "bash $H/node_probe.sh research@$SERVED_IP '$gate'"; echo "bash $H/node_probe.sh research@$SERVED_IP '$stop'"; exit 0; }
G=$LOGS/node-gate.txt
timeout 1200 bash "$H/node_probe.sh" "research@$SERVED_IP" "$gate" > "$G" 2>&1
rc=$?
grep -q '^EXIT 0$' "$G" || rc=1
echo "CUSTODY GATE exit $rc $(date -u +%H:%M:%SZ): $(grep -v -E '^(== |EXIT )' "$G" | tail -4 | tr '\n' ' ' | cut -c1-400)"
[ "$rc" = 0 ] || { echo "NOT STOPPED: a run on vy-dc-h200-2 lacks custody or is running (or the gate didn't answer): $G"; exit 1; }
[ -z "${CHECK:-}" ] || exit 0
bash "$H/node_probe.sh" "research@$SERVED_IP" "$stop" 2>&1 | tee "$LOGS/stop.txt"
echo "STOP SENT $(date -u +%H:%M:%SZ)"
