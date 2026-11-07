#!/usr/bin/env bash
# Once the served run RID has ended (exit 0, custody preserved): follow/reproof_send.sh on vy-dc-h200-2 at the run's tree,
# copying the deferral's serve dir to FOLLOW_NODE (env.sh's, vy-nebius-cpu-12 unless set) over infra's write-only path,
# under NETACCT/RID there. Waited for up to 20 minutes (a send still running then is read with runlog.sh h200-2 ID all);
# it prints "REPROOF SEND: PASS" or "REPROOF SEND: REFUSED: why" alone, no name, count or size. Its launch in
# LOGS/send-RID.txt.
#   [FOLLOW_NODE=vy-nebius-cpu-11] [DRY=1] bash send.sh RID
set -uo pipefail
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$H/env.sh" || exit 3
RID=${1:?the served run id}
[[ $RID =~ ^r[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$ ]] || { echo "REFUSED: $RID is not a store run id"; exit 3; }
follow_ip "$FOLLOW_NODE" > /dev/null || exit 3
tree_ok || exit 3
cmd=(research run --on "$SERVED" --project verity --source "$TREE" --cwd clone --timeout 2h --wait --env "D=$SERVE_ROOT/$RID"
  --env "NAME=$RID" --env "COPY_TO=research@$FOLLOW_NODE" --send "$H/follow/reproof_send.sh" --send "$H/follow/reproof_files.py"
  -- bash -c 'exec bash "$RESEARCH_RUN_DIR/inputs/reproof_send.sh"')
[ -z "${DRY:-}" ] || { printf '%q ' "${cmd[@]}"; echo; exit 0; }
(cd "$TREE" && "${cmd[@]}") > "$LOGS/send-$RID.txt" 2>&1
rc=$?
sid=$(grep -o -E 'run r[0-9]{8}-[0-9]{6}-[0-9a-f]{4} launched' "$LOGS/send-$RID.txt" | head -1 | cut -d' ' -f2)
echo "SEND ${sid:-not launched} of $RID to $FOLLOW_NODE: exit $rc$([ "$rc" = 79 ] && echo ', still running: read it with runlog.sh h200-2 '"$sid"' all')"
[ -n "$sid" ] && bash "$H/runlog.sh" h200-2 "$sid" all | grep -E '^(STATE|REPROOF SEND)'
