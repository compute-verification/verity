#!/usr/bin/env bash
# The V3c follow-on of run (i) or (ii) on its node, from the pinned tree, its audit's coins from the live gate:
#   bash follow_launch.sh TREE NODE NAME
# NODE is the run's follow-on node, its settings file's PP_PEER and reproof_send.sh's COPY_TO (vy-nebius-cpu-11 for (i),
# vy-nebius-cpu-12 for (ii)); NAME the served run's id, the directory reproof_send.sh copied the serve dir to under infra's
# write-only root there (ROOT). It writes main.txt beside the kit (main's first-parent line now, which must hold the
# gate's pin, GATE_PIN) and sends it with the key record. The re-proof copy goes to cpu-3 over infra's route where NODE
# holds its key (REPROOF=1, cpu-11's default; REPROOF=0 copies nothing). Just before the launch it removes infra's live
# memory hold for the follow-on, /workspace/ramlock/HOLD on NODE (over ssh with the research key, so run it from a login
# shell; infra's master copy in research's home stays), as infra's launch line does, since a hold names no pid and counts
# whole against the job's admission. The job reserves MEM_GB (650, in research's 10^9-byte GB: a 4,32,4 audit's NVIS 35
# session with one Lean check beside it is about 630 GB, under the run scope's MemoryMax 604G; a reservation, not a cap),
# waiting up to MEM_WAIT seconds for it, and refuses while the live hold reserves less (infra's "gb"), which it then
# leaves in place. Its Lean checks run one at a time (V3C_LEAN_WORKERS 1: two large ones beside that session pass the
# scope's cap), and its --timeout is 30 h (the audit's 21 h, the window commit, chain, period check and composites' limits).
# Its custody key lives 24 h (--custody-ttl, research run's cap), past a follow-on's 10 to 13 h at 8,32,4,4,16: one that
# still outlives it ends with .custody "preserved": false ("the custody key expired"), and before the node gate it is
# repaired from this VM with `research fetch --all RUN`, then `research data custody RUN --publish`.
# The follow-on reads the network certifier's key and
# session from its own host's record of the run (CERTIFIER_GATE, at the commit CERTIFIER_PIN INF-494 hashed, the run's
# peer CERTIFIER_PEER: certifier_key.py), not from the prover's copy; on a tree with site_registration.py it also writes
# the auditor's site registration, at SITE_BETA and SITE_LATENCY where set (reproof_follow.sh's defaults the census's).
# TREE must be at PIN_REF's head (default SERVE_REF, the serve's tree: Thursday's pin unless set); another ref than the
# serve's, a fix's, runs with FOLLOW_ANY_TREE=1, the serve dir's window-circuit record still the serve's.
# V3C_AUDIT_TIMEOUT, CERTIFIER_TIMEOUT, CERTIFIER_PERIOD_TIMEOUT and COMPOSITE_TIMEOUT, where set here, reach the
# follow-on (--env) over reproof_follow.sh's and served_verdicts.sh's defaults; CAPS=FILE (Monday's settings-monday.env,
# which the follow-on doesn't read) sets all four from FILE's assignments and refuses where one is missing.
# DRY=1 prints the command instead of running it.
set -euo pipefail
TREE=${1:?usage: follow_launch.sh TREE NODE NAME} NODE=${2:?NODE} NAME=${3:?NAME}
K=$(cd "$(dirname "$0")" && pwd)
case $NODE in
  vy-nebius-cpu-11) REPROOF=${REPROOF:-1} IP=81.85.4.136 ;;
  vy-nebius-cpu-12) REPROOF=${REPROOF:-0} IP=81.85.1.89 ;;
  *) echo "REFUSED: $NODE is neither (i)'s follow-on node (vy-nebius-cpu-11) nor (ii)'s (vy-nebius-cpu-12)"; exit 3 ;;
esac
SERVE_REF=${SERVE_REF:-cursor/onsite-thu-pin-061a}
PIN_REF=${PIN_REF:-$SERVE_REF} ROOT=${ROOT:-/workspace/incoming/netacct} HOLD=${HOLD:-netacct-inf458-thu.json}
GATE_PIN=${GATE_PIN:-f0c4b1355fb} M=${MAIN_REPO:-/home/ubuntu/wt-na-main}
CERTIFIER_GATE=${CERTIFIER_GATE:-research@vy-auditor-2} CERTIFIER_PEER=${CERTIFIER_PEER:-vy-dc-h200-2}
CERTIFIER_PIN=${CERTIFIER_PIN:-701c8c0073a63fb9d3fdc6d9a993e3d5a5039656}
want=$(git -C "$TREE" ls-remote origin "refs/heads/$PIN_REF" | cut -f1) head=$(git -C "$TREE" rev-parse HEAD)
[ -n "$want" ] && [ "$head" = "$want" ] && git -C "$TREE" diff --quiet HEAD \
  || { echo "REFUSED: TREE is at $head (or dirty), not $PIN_REF's ${want:-?}"; exit 3; }
git -C "$M" fetch -q origin main && git -C "$M" rev-list --first-parent origin/main > "$K/main.txt"
grep -q "^$GATE_PIN" "$K/main.txt" \
  || { echo "REFUSED: the gate's pin $GATE_PIN isn't on main's first-parent line, so the a5 client refuses its pins"; exit 3; }
re=() any=() caps=() capped=
[ "$PIN_REF" = "$SERVE_REF" ] || any=(--env FOLLOW_ANY_TREE=1)
for v in V3C_AUDIT_TIMEOUT CERTIFIER_TIMEOUT CERTIFIER_PERIOD_TIMEOUT COMPOSITE_TIMEOUT; do
  if [ -n "${CAPS:-}" ]; then
    x=$(grep -oE "(^|[[:space:]])$v=[1-9][0-9]*[mh]([[:space:]]|$)" "$CAPS" | tail -1 | tr -d '[:space:]' || true)
    [ -n "$x" ] || { echo "REFUSED: CAPS $CAPS sets no $v (a whole number of minutes or hours)"; exit 3; }
    printf -v "$v" %s "${x#*=}"
  fi
  [ -z "${!v:-}" ] || { caps+=(--env "$v=${!v}"); capped+=" $v=${!v}"; }
done
[ "$REPROOF" = 1 ] && re=(--env "REPROOF_DEST=research@vy-nebius-cpu-3:$NAME/"
  --env "REPROOF_COPY=vy-nebius-cpu-3:/workspace/research/lanes/network-accounting/private/reproof/$NAME/")
echo "TREE $head (= $PIN_REF), NODE $NODE, D $ROOT/$NAME, main.txt $(wc -l < "$K/main.txt") commits from" \
     "$(head -c 11 "$K/main.txt") (holds $GATE_PIN), REPROOF $REPROOF, HOLD $HOLD${any:+, FOLLOW_ANY_TREE=1}${capped:+, caps$capped}"
MEM_GB=${MEM_GB:-650}
cmd=(research run --as network-accounting --on "$NODE" --project verity --source "$TREE" --cwd clone --timeout 108000
  --mem-gb "$MEM_GB" --mem-wait "${MEM_WAIT:-900}" --custody-ttl 24h --env "D=$ROOT/$NAME" --env PP_CHALLENGER=gate:research@204.12.178.84
  --env "PP_PEER=$NODE" --env "FOLLOW_HOLD=/workspace/ramlock/$HOLD" --env "V3C_LEAN_WORKERS=${V3C_LEAN_WORKERS:-1}"
  "${re[@]}" "${any[@]}" "${caps[@]}"
  --env "CERTIFIER_GATE=$CERTIFIER_GATE" --env "CERTIFIER_PIN=$CERTIFIER_PIN" --env "CERTIFIER_PEER=$CERTIFIER_PEER"
  ${SITE_BETA:+--env "SITE_BETA=$SITE_BETA"} ${SITE_LATENCY:+--env "SITE_LATENCY=$SITE_LATENCY"}
  --send "$K/challenger-key.json" --send "$K/main.txt" --send "$K/reproof_follow.sh" --send "$K/reproof_copy.sh"
  --send "$K/reproof_files.py" --send "$K/circuits_out.py" --send "$K/certifier_key.py"
  -- bash -c 'exec bash "$RESEARCH_RUN_DIR/inputs/reproof_follow.sh"')
[ -z "${DRY:-}" ] || { printf '%q ' "${cmd[@]}"; echo; exit 0; }
key=$(cd "$M" && python -c 'from research.pods.runpod import ssh_key; print(ssh_key())')
kh=$(cd "$M" && python -c 'from research.pods.runpod import known_hosts; print(known_hosts())')
held=$(ssh -i "$key" -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=no -o "UserKnownHostsFile=$kh" \
  "research@$IP" "f=/workspace/ramlock/$HOLD; if [ -e \$f ]; then head -c 300 \$f
    g=\$(python3 -c 'import json, sys; print(int(json.load(open(sys.argv[1]))[\"gb\"]))' \$f 2> /dev/null || echo 0)
    if [ \"\$g\" -ge $MEM_GB ]; then rm -f \$f && echo ' removed'; else echo \" SHORT: \$g GB\"; fi; else echo absent; fi") \
  || { echo "REFUSED: couldn't reach $NODE to remove its hold /workspace/ramlock/$HOLD, which would hold the job's admission"; exit 3; }
echo "HOLD on $NODE: /workspace/ramlock/$HOLD: $held"
case $held in *" SHORT: "*) echo "REFUSED: infra's hold reserves less than MEM_GB $MEM_GB GB, so it stays and nothing launches:" \
  "ask infra (INF-458) for $MEM_GB GB with the train slot off, or set MEM_GB knowing the audit's tail needs about 630 GB"; exit 3 ;; esac
exec "${cmd[@]}"
