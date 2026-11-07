#!/usr/bin/env bash
# Read-only: a run's state, custody and lines on its node, from its run dir's status.json, .custody and stdout.log.
#   bash runlog.sh HOST RID [all|serve|follow]
# HOST is h200-2 (the served node), cpu-11, cpu-12 or USER@HOST. all (the default) prints every line, for the jobs that
# print no count (settings, probe, send, stock control, check), but the settings job's CAP line, which names
# WINDOW_STEPS and MAX_TOKENS, cut after "CAP: unset"; serve and follow print the served run's and the
# follow-on's step lines with every count cut (WINDOW-COMMIT's registration, INFERENCE-ONLY's numbers, the scan's
# sizes), and never the ONSITE line, whose serving field holds the run's counts.
set -uo pipefail
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$H/env.sh" || exit 3
case ${1:?HOST} in
  h200-2) host=research@$SERVED_IP ;;
  cpu-11) host=research@$(follow_ip vy-nebius-cpu-11) ;;
  cpu-12) host=research@$(follow_ip vy-nebius-cpu-12) ;;
  *@*) host=$1 ;;
  *) echo "REFUSED: HOST is h200-2, cpu-11, cpu-12 or USER@HOST, not $1"; exit 3 ;;
esac
RID=${2:?RID} MODE=${3:-all}
[[ $RID =~ ^r[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$ ]] || { echo "REFUSED: $RID is not a store run id"; exit 3; }
case $MODE in
  all) filter='cat' ;;
  serve) filter="grep -E '^(GATE|REGISTERED|ROW|CHALLENGER KEY|KEPT WINDOWS|PREFLIGHT|STATED tree|COINS|PERIOD|H1|CERTIFIER-FRONT|SERVE|SEALED|B-OF-RECORD|SERVED-SEAL|INFERENCE-ONLY|WINDOW-CIRCUIT|WINDOW-COMMIT|DEFER|PIECE|RECORD|REFUSED|STOP|RUN:|PORTS|SCRUB|SHARES)' | grep -v -E '^(INFERENCE-ONLY POLICY|ONSITE)'" ;;
  follow) filter="grep -E '^(VENV|DISK|COINS|CERTIFIER-KEY|SITE-REGISTRATION|ROWS|HOLD|FOLLOW-ON|V3C-AUDIT|V3C-PERIOD|V3C-CIRCUITS|CERTIFIER-CHAIN|CERTIFIER-PERIOD|THE NETWORK|COMPOSITE|RECORD|PUBLISH|PREFLIGHT|REFUSED|RECEIVED-SCAN|WINDOW-CIRCUIT-HELD)' | grep -v '^ONSITE'" ;;
  *) echo "REFUSED: MODE is all, serve or follow, not $MODE"; exit 3 ;;
esac
cut_counts="sed -E 's/^(INFERENCE-ONLY [a-z]+|WINDOW-CIRCUIT class [0-9,]+: [a-z]+|WINDOW-COMMIT[^:]*:? ?[A-Za-z]*|RECEIVED-SCAN KIND-SCAN [a-z]+|SERVE [^(]*listening|V3C-CIRCUITS [a-z]+|CERTIFIER-PERIOD [(]exit [0-9]+[)]: CERTIFIER [{]\"public\": [{]\"K\": [0-9]+).*/\\1 …/' | cut -c1-120"
[ "$MODE" = all ] && cut_counts="sed -E 's/(: CAP: unset).*/\\1 …/'"
bash "$H/node_probe.sh" "$host" "R=/workspace/research/runs/$RID; [ -d \$R ] || { echo 'NO RUN DIR '\$R; exit 2; }
echo \"STATE \$(grep -o -E '\"state\": \"[a-z]+\"' \$R/status.json 2>/dev/null | tail -1 | cut -d'\"' -f4)\" \
  \"CUSTODY \$(grep -o -E '\"preserved\": [a-z]+' \$R/.custody 2>/dev/null || echo none)\" \"FETCHED \$([ -e \$R/.fetched ] && echo yes || echo no)\"
cat \$R/stdout.log | $filter | $cut_counts"
