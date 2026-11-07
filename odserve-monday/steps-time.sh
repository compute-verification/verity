#!/usr/bin/env bash
# Read-only: each step's wall time in a run, from its files' times alone (none is read): every log, record and answer
# under the run dir and the serve dir (depth 4), with its creation and last write, oldest first, and the span between. A
# step's log is created as it starts and last written as it ends; set against the follow-on's caps (13h, 4h, 1h, 6h).
# Names only, no contents.
#   bash steps-time.sh h200-2 RID          # the served run: its run dir and SERVE_ROOT/RID
#   bash steps-time.sh follow RID FRID     # the follow-on FRID on FOLLOW_NODE: its run dir and NETACCT/RID
set -uo pipefail
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$H/env.sh" || exit 3
case ${1:?h200-2 or follow} in
  h200-2) host=research@$SERVED_IP R=/workspace/research/runs/${2:?RID} S=$SERVE_ROOT/$2 ;;
  follow) host=research@$(follow_ip "$FOLLOW_NODE") R=/workspace/research/runs/${3:?FRID} S=$NETACCT/${2:?RID} ;;
  *) echo "REFUSED: h200-2 or follow, not $1"; exit 3 ;;
esac
bash "$H/node_probe.sh" "$host" "for d in $R $S; do
  find \$d -maxdepth 4 -type f \\( -name '*.log' -o -name '*.json' -o -name 'verdicts.txt' -o -name '*.err' \\) \
    -not -path '*/window/*' -not -path '*/tree/*' -printf '%p\n' 2>/dev/null
done | while read -r f; do b=\$(stat -c %W \"\$f\") m=\$(stat -c %Y \"\$f\"); echo \"\$b \$m \$f\"; done | sort -n | \
  awk '{ printf \"%s  %s  %6d s  %s\n\", strftime(\"%H:%M:%S\", \$1, 1), strftime(\"%H:%M:%S\", \$2, 1), \$2 - \$1, \$3 }'"
