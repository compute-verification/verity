#!/usr/bin/env bash
# The served node's inodes before the serve. Lists each serve dir under SERVE_ROOT on vy-dc-h200-2 that still holds a
# class window (v3c-w*), with its run's custody, whether a send copied it to a follow-on node (a run there whose job
# names NAME=<the serve's id> and printed "REPROOF SEND: PASS") and whether a process names it; then the free inodes.
# With REMOVE, it removes those serves' windows, each only where its run's custody is preserved (or fetched, or ALLOW
# names it), its copy was sent (or NO_SEND names it: a serve no follow-on reads) and no process names its serve dir.
#   bash inodes.sh
#   REMOVE="ID ..." [ALLOW="ID ..."] [NO_SEND="ID ..."] bash inodes.sh
# The serve refuses under MIN_FREE_INODES (500,000), and the window its commit writes takes several hundred thousand more,
# so clear windows until FREE is at least 850,000.
set -uo pipefail
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$H/env.sh" || exit 3
for v in REMOVE ALLOW NO_SEND; do
  [[ ${!v:-} =~ ^[A-Za-z0-9\ -]*$ ]] || { echo "REFUSED: $v holds something other than run ids"; exit 3; }
done
# shellcheck disable=SC2016
bash "$H/node_probe.sh" "research@$SERVED_IP" 'S='"$SERVE_ROOT"' R=/workspace/research/runs
has() { case " $1 " in *" $2 "*) return 0;; esac; return 1; }
for A in $S/*/v3c-w*; do
  [ -d "$A" ] || continue
  d=$(basename "$(dirname "$A")")
  cust=$(grep -o -E "\"preserved\": [a-z]+" "$R/$d/.custody" 2> /dev/null | cut -d" " -f2); [ -e "$R/$d/.fetched" ] && cust=${cust:-none}+fetched
  sent=$(grep -l -F "\"NAME=$d\"" $R/*/job.json 2> /dev/null | while read -r j; do s=$(dirname "$j")
    grep -q "^REPROOF SEND: PASS" "$s/stdout.log" 2> /dev/null && basename "$s"; done | head -1)
  busy=$(ps -eo args | grep -v grep | grep -c -F "$S/$d")
  echo "WINDOW $A custody=${cust:-none} sent=${sent:-no} busy=$busy"
  has "'"${REMOVE:-}"'" "$d" || continue
  { [ "${cust%%+*}" = true ] || [ "${cust#*+}" = fetched ] || has "'"${ALLOW:-}"'" "$d"; } || { echo "KEPT $A: its run has no custody (ALLOW names a run that never gets one)"; continue; }
  { [ -n "$sent" ] || has "'"${NO_SEND:-}"'" "$d"; } || { echo "KEPT $A: no send copied it to a follow-on node (NO_SEND names a serve no follow-on reads)"; continue; }
  [ "$busy" = 0 ] || { echo "KEPT $A: a process names its serve dir"; continue; }
  echo "FILES $(find "$A" -xdev | wc -l) in $A"
  rm -rf -- "$A" && echo "REMOVED $A at $(date -u +%H:%M:%SZ)"
done
df -i /workspace | awk "NR == 2 {print \"FREE INODES\", \$4, \"of\", \$2, \"on /workspace (MIN_FREE_INODES 500000; at least 850000 before the serve)\"}"
df -h /workspace | awk "NR == 2 {print \"FREE DISK\", \$4, \"on /workspace\"}"' | tee "$LOGS/inodes.txt"
