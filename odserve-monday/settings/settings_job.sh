#!/usr/bin/env bash
# Installs the onsite runs' settings files on the node (0600 in a 0700 directory) and checks, printing no count, each of:
#  - remote's private file it reads is there and sets SLOTS and MAX_TOKENS;
#  - the tree's front takes those counts: CERTIFIER_FRONT runs with "$PY" front_check.py, the front's own code with the
#    gate's session alone replaced, so a refusal the front would make after the lease is made here (why, which names the
#    counts, in NAME.front-check.err beside the settings, 0600);
#  - whether remote's first load (REQUESTS, SLOTS by default) is the front's group's size, which odserve_cost.py's table
#    needs against remote's stock (a warning, not a refusal);
#  - the settings defer V3c's audit to the follow-on, with remote-network's composite unless COMPOSITE is set empty
#    (the V3C line says which): Thursday's PRODUCT=remote-network under H1_TIE at V3C_LAW stratified:*, or a run of
#    record's (OF_RECORD set) PRODUCT=remote-network-no-pouw untied (H1_TIE empty) at V3C_LAW subset:1 with COMPOSITE
#    unset, V3C_TIMEOUT 120m and WC_TIMEOUT 60m (its untied 8-slot window commit's and class record's), CLASSES holding
#    V3C_CLASS, and the follow-on's four caps, V3C_AUDIT_TIMEOUT,
#    CERTIFIER_TIMEOUT, CERTIFIER_PERIOD_TIMEOUT and COMPOSITE_TIMEOUT, which follow_launch.sh's CAPS reads off this file
#    and passes (--env), and none of the follow-on's own PP_CHALLENGER, PP_PEER, CERTIFIER_KEY or SITE_REGISTRATION;
#    a V3C_CLASS, COPY_TO set and naming the file's PP_PEER (its follow-on's node) where it sets one, and this node's MemTotal under
#    V3C_AUDIT_MEM_GB (served_verdicts.sh's short_of_memory), so the served run's preflight plans the deferral (its H1_GATE
#    and H1_CIRCUITS, remote_run.sh's, it checks itself), neither file sets WIPE_CMD, POUS_CMD, BANDWIDTH_CMD or
#    CERTIFIER_CMD, the split's CERTIFIER_CHAIN_CMD and CERTIFIER_PERIOD_CMD are set, and POUW_CMD is set empty, so
#    remote_run.sh's served_pouw.sh gives way to PoUW's development piece;
#  - ℓ_a is advice_cap's at one of the two runs' conditions: CAP unset, ID_BYTES 24, no --prefix-caching, and either
#    WINDOW_STEPS 1 (remote_run.sh's, which prep.sh checks) with MAX_TOKENS 1 (1,328 bits; the window pick refuses a
#    request carried across windows) or WINDOW_STEPS 16 with MAX_TOKENS at most 4 (7,432 bits); whether SLOTS and
#    MAX_MODEL_LEN are within 4 and 512, which both figures are computed at (a warning);
# and once, for every file:
#  - with COPY_TO (USER@HOST ..., the V3c follow-ons' nodes), the node's copy key writes to each over infra's write-only
#    path (a node that takes no copy refuses the job after the gate's checks, so they still run);
#  - the front's gate, which every file names alike, answers from this node: the lane's gate key is here, the gate's
#    read-only `pins` returns its pinned commit (with GATE_PIN, that one; a gate that refuses `pins` isn't the challenger
#    gate the front speaks), and the key gets no shell there (a command the gate doesn't take runs nothing).
#   research run --on vy-dc-h200-2 --project verity --source TREE --cwd clone --timeout 10m [--env GATE_PIN=SHA] [--env COPY_TO='U@H ...'] \
#     --send settings.env --send settings-s16.env --send settings_job.sh --send front_check.py -- \
#     bash -c 'exec bash "$RESEARCH_RUN_DIR/inputs/settings_job.sh"'
# NAMES (default "settings.env settings-s16.env", runs (i) and (ii)) are the files it checks, each installed under its name.
set -euo pipefail
umask 077
D=${ONSITE_DIR:-/workspace/research/lanes/network-accounting/onsite}
R=${REMOTE_SETTINGS:-/workspace/research/vllm-served-verdicts/settings-remote/settings.env}
NAMES=${NAMES:-settings.env settings-s16.env}
I=$RESEARCH_RUN_DIR/inputs KEY=$HOME/.ssh/vy-gate-network-accounting
SSH=(ssh -T -i "$KEY" -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=30)
for NAME in $NAMES; do
  [ -f "$I/$NAME" ] || { echo "REFUSED: no $NAME was sent"; exit 3; }
  grep -q 'GATE_USER@GATE_HOST' "$I/$NAME" && { echo "REFUSED: $NAME still names GATE_USER@GATE_HOST"; exit 3; }
done
install -d -m 0700 "$D"
echo "DISK: $(df -BG --output=avail /workspace | tail -1 | tr -d ' ') free on /workspace, beside served_verdicts.sh's DISK_FLOOR_GB (64 by default); each run's serve dir stays on the node until its copy is checked"
[ -r "$R" ] || { echo "REFUSED: remote's settings file $R isn't readable here"; exit 3; }
n=$(grep -cE '^(export )?(SLOTS|MAX_TOKENS)=' "$R" || true)
echo "REMOTE SETTINGS: readable, sets $n of SLOTS and MAX_TOKENS"
[ "$n" = 2 ] || { echo "REFUSED: remote's file must set both, or the front's group and the period's length are served_verdicts.sh's defaults"; exit 3; }
same=$(set +e -a; . "$R" > /dev/null 2>&1; set +a; [ -n "${SLOTS:-}" ] && [ "${REQUESTS:-$SLOTS}" = "$SLOTS" ] && echo yes || echo no)
echo "COST LOAD: remote's first load (REQUESTS, SLOTS by default) is the front's group's size: $same"
[ "$same" = yes ] || echo "WARNING: odserve_cost.py's table refuses against a remote stock of that load (loads of different sizes)"

py=${FRONT_PY:-/workspace/cp/venv312/bin/python}
[ -x "$py" ] || py=$(command -v python3)
t=$(mktemp -d)
trap 'rm -rf "$t"' EXIT
printf '#!/bin/sh\nexec "%s" "%s" "$@"\n' "$py" "$I/front_check.py" > "$t/py" && chmod 0700 "$t/py"
gate=
for NAME in $NAMES; do
  S=$D/$NAME err=$D/$NAME.front-check.err
  install -m 0600 "$I/$NAME" "$S"
  echo "$NAME: INSTALLED $S ($(stat -c %a "$S"))"
  rc=0 scratch=$t/$NAME
  mkdir -p "$scratch"
  (set +e -a; . "$S" > /dev/null 2>&1; set +a; PY=$t/py bash -c "$CERTIFIER_FRONT" _ 18190 "$scratch/serve" "$scratch/out") > "$err" 2>&1 || rc=$?
  [ "$rc" = 0 ] || { echo "REFUSED ($NAME): the tree's front refuses its counts (exit $rc; why in $err, which names them)"; exit 3; }
  echo "$NAME: FRONT: the tree's front takes its counts (its refusals and the gate's parse, run here with no gate contacted)"
  (set +e -a; . "$S" > /dev/null 2>&1; set +a
   m=$(awk '/^MemTotal:/ {print int($2 / 1048576)}' /proc/meminfo) need=${V3C_AUDIT_MEM_GB:-290}
   run=
   [ "${PRODUCT:-}" = remote-network ] && [ -n "${H1_TIE:-}" ] && [[ ${V3C_LAW:-} == stratified:* ]] && run=tied
   [ -n "${OF_RECORD:-}" ] && [ "${PRODUCT:-}" = remote-network-no-pouw ] && [ -z "${H1_TIE:-}" ] \
     && [ "${V3C_LAW:-}" = subset:1 ] && run=of-record
   [ -n "$run" ] && [ -n "${V3C_CLASS:-}" ] && [ -n "${COPY_TO:-}" ] && [ "$m" -lt "$need" ] \
     || { echo "REFUSED ($NAME): the settings don't defer V3c's audit to the follow-on:" \
               "PRODUCT=${PRODUCT:-} H1_TIE=${H1_TIE:-} OF_RECORD=${OF_RECORD:-} V3C_CLASS=${V3C_CLASS:-}" \
               "V3C_LAW=${V3C_LAW:-} COPY_TO=${COPY_TO:-}, this node $m GB against $need GB; Thursday's is" \
               "remote-network under H1_TIE at stratified:*, a run of record's remote-network-no-pouw untied at subset:1"; exit 3; }
   if [ "$run" = of-record ]; then
     [ "${COMPOSITE+set}" != set ] \
       || { echo "REFUSED ($NAME): a run of record's composite is its verdict, and the settings set COMPOSITE"; exit 3; }
     [ "${V3C_TIMEOUT:-}" = 120m ] && [ "${WC_TIMEOUT:-}" = 60m ] \
       || { echo "REFUSED ($NAME): untied, this node commits the 8-slot window after the serve, at V3C_TIMEOUT 120m and WC_TIMEOUT 60m, not ${V3C_TIMEOUT:-unset} and ${WC_TIMEOUT:-unset}"; exit 3; }
     case " ${CLASSES:-} " in *" $V3C_CLASS "*) ;; *)
       echo "REFUSED ($NAME): CLASSES (${CLASSES:-unset}) doesn't hold V3C_CLASS $V3C_CLASS, so step 3 writes no record at it and v3c_commit refuses"; exit 3 ;; esac
     for v in V3C_AUDIT_TIMEOUT CERTIFIER_TIMEOUT CERTIFIER_PERIOD_TIMEOUT COMPOSITE_TIMEOUT; do
       [[ ${!v:-} =~ ^[1-9][0-9]*[mh]$ ]] \
         || { echo "REFUSED ($NAME): the follow-on's cap $v is ${!v:-unset}, not a whole number of minutes or hours"; exit 3; }
     done
     for v in PP_CHALLENGER PP_PEER CERTIFIER_KEY SITE_REGISTRATION; do
       [ -z "${!v:-}" ] || { echo "REFUSED ($NAME): the settings set $v, which only the follow-on takes"; exit 3; }
     done
     echo "$NAME: OF RECORD: untied remote-network-no-pouw at subset:1, COMPOSITE unset, V3C_TIMEOUT 120m, WC_TIMEOUT 60m, CLASSES holding V3C_CLASS; the follow-on's" \
          "caps: audit $V3C_AUDIT_TIMEOUT, chain $CERTIFIER_TIMEOUT, period $CERTIFIER_PERIOD_TIMEOUT, composite $COMPOSITE_TIMEOUT"
   fi
   [ "${POUW_CMD+set}" = set ] && [ -z "$POUW_CMD" ] \
     || { echo "REFUSED ($NAME): the settings don't set POUW_CMD empty, so remote_run.sh's served_pouw.sh proves PoUW's" \
               "drawn units on this node and writes a PoUW period case"; exit 3; }
   [ -z "${PP_PEER:-}" ] || case " $COPY_TO " in *"@$PP_PEER "*) ;; *)
     echo "REFUSED ($NAME): its follow-on's node, PP_PEER $PP_PEER, isn't among COPY_TO's ($COPY_TO), so its serve dir" \
          "would be copied where its V3c audit doesn't draw"; exit 3 ;; esac
   for v in WIPE_CMD POUS_CMD BANDWIDTH_CMD CERTIFIER_CMD; do
     [ -z "${!v:-}" ] || { echo "REFUSED ($NAME): the settings set $v, which PRODUCT=remote-network's preflight refuses"; exit 3; }
   done
   for v in CERTIFIER_CHAIN_CMD CERTIFIER_PERIOD_CMD; do
     [ -n "${!v:-}" ] || { echo "REFUSED ($NAME): the settings leave $v unset: remote-network's composite reads the" \
                                "network certifier's period case, which the split's chain and period check write"; exit 3; }
   done
   what="with $PRODUCT's composite" && [ -n "${COMPOSITE-1}" ] || what="without a composite (COMPOSITE empty)"
   echo "$NAME: V3C: deferred $what to the follow-on on ${PP_PEER:-$COPY_TO}: class $V3C_CLASS, law" \
        "$V3C_LAW $([ -n "${H1_TIE:-}" ] && echo under H1_TIE || echo untied), this node has $m GB, under the $need GB V3c's audit needs; PoUW's development piece" \
        "(POUW_CMD empty)") || exit 3
  (set +e -a; . "$S" > /dev/null 2>&1; set +a
   s=${WINDOW_STEPS:-1} t=${MAX_TOKENS:-}
   case "$s" in
     1) bits=1,328; [ "$t" = 1 ] ;;
     16) bits=7,432; [ "$t" -ge 1 ] 2> /dev/null && [ "$t" -le 4 ] ;;
     *) bits=; false ;;
   esac || { echo "REFUSED ($NAME): the run is neither (i) nor (ii): WINDOW_STEPS ${WINDOW_STEPS:-unset (1)} with" \
                  "MAX_TOKENS ${t:-unset}, where (i) is WINDOW_STEPS 1 with MAX_TOKENS 1 (the window pick refuses a" \
                  "request carried across windows) and (ii) WINDOW_STEPS 16 with MAX_TOKENS at most 4"; exit 3; }
   [ -z "${CAP:-}" ] && [ "${ID_BYTES:-24}" = 24 ] \
     && case " ${SERVE_ARGS:-} ${SERVE_EXTRA:-} " in *" --prefix-caching "* | *" --prefix-caching="*) false ;; esac \
     || { echo "REFUSED ($NAME): ℓ_a isn't advice_cap's with no prefix cache and 24-byte ids: the settings set CAP" \
               "($([ -n "${CAP:-}" ] && echo set || echo unset)), ID_BYTES other than 24 (${ID_BYTES:-unset}) or" \
               "--prefix-caching in SERVE_ARGS or SERVE_EXTRA"; exit 3; }
   w=no; [ "${SLOTS:-4}" -le 4 ] 2> /dev/null && [ "${MAX_MODEL_LEN:-512}" -le 512 ] 2> /dev/null && w=yes
   echo "$NAME: CAP: unset, so served_verdicts.sh derives ℓ_a (advice_cap) at WINDOW_STEPS $s, MAX_TOKENS $t, no prefix" \
        "cache and 24-byte ids; SLOTS at most 4 and MAX_MODEL_LEN at most 512, which the $bits bits is computed at: $w"
   [ "$w" = yes ] || echo "WARNING ($NAME): ℓ_a is advice_cap's at these counts, not the $bits bits") || exit 3
  g=$(grep -oE -- '--gate [^ ]+' "$S" | head -1 | cut -d' ' -f2)
  [ -n "$g" ] && [ "$g" != local ] || { echo "REFUSED ($NAME): its front names no auditor gate (--gate USER@HOST)"; exit 3; }
  [ -z "$gate" ] || [ "$g" = "$gate" ] || { echo "REFUSED ($NAME): its front's gate $g isn't the other file's, $gate"; exit 3; }
  gate=$g
done

copy_refused=
if [ -n "${COPY_TO:-}" ]; then
  ck=$HOME/.ssh/vy-h200-2-to-prover
  [ -f "$ck" ] || { echo "REFUSED: no key $ck for the V3c follow-on's copy path"; exit 3; }
  mkdir -p "$t/copy" && head -c 4096 /dev/urandom > "$t/copy/probe.bin"
  command -v rsync > /dev/null || echo "COPY PATH: this node has no rsync"
  for c in $COPY_TO; do
    n=probe-$(od -An -N4 -tx1 /dev/urandom | tr -d ' \n')
    if rsync -a -e "ssh -i $ck -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=30" "$t/copy/" "$c:$n/" > /dev/null 2> "$t/copy.err"; then
      echo "COPY PATH: rsync to $c wrote $n/ (4 KiB of random bytes; the path is write-only, so not read back)"
    else
      echo "COPY PATH: rsync to $c with $ck failed (exit $?): $(head -n3 "$t/copy.err" | tr '\n' ' ' | cut -c1-400)"
      copy_refused="$copy_refused $c"
    fi
  done
fi

[ -f "$KEY" ] || { echo "REFUSED: no gate key for lane network-accounting on this node ($KEY)"; exit 3; }
out=$("${SSH[@]}" "$gate" pins 2>&1 | tail -n1 || true)
pin=$(python3 -c 'import json, sys; print(json.loads(sys.argv[1])["commit"])' "$out" 2> /dev/null) || {
  why=$(python3 -c 'import json, sys; print(str(json.loads(sys.argv[1])["refused"])[:300])' "$out" 2> /dev/null || true)
  if [ -z "$out" ]; then
    echo "REFUSED: the gate $gate answers pins with an empty log: it records no pin, so the job can't hold it to GATE_PIN"
  elif [ -n "$why" ]; then
    echo "REFUSED: the gate $gate answers the lane's key but refuses pins, so it isn't the gate the front speaks (vy-challenger-gate: warden-rehearse, pins): $why"
  else
    hn=$(ssh -G "${gate#*@}" 2> /dev/null | awk '$1 == "hostname" {print $2}')
    kh=$(ssh-keygen -F "${hn:-${gate#*@}}" > /dev/null 2>&1 && echo "in known_hosts" || echo "not in known_hosts")
    echo "REFUSED: the gate $gate didn't answer pins with the lane's key, with the front's ssh options (its host key in known_hosts):" \
         "$(printf %s "$out" | cut -c1-300) [ssh resolves it to ${hn:-?}, $kh]"
  fi
  exit 3; }
echo "GATE $gate answers with the lane's key, pinned at $pin"
[ -z "${GATE_PIN:-}" ] || [ "$pin" = "$GATE_PIN" ] || { echo "REFUSED: the gate is pinned at $pin, not GATE_PIN $GATE_PIN"; exit 3; }
a=$(od -An -N8 -tx1 /dev/urandom | tr -d ' \n') b=$(od -An -N8 -tx1 /dev/urandom | tr -d ' \n')
got=$("${SSH[@]}" "$gate" "printf '%s%s\\n' $a $b" 2>&1 || true)
case $got in *"$a$b"*) echo "REFUSED: the lane's gate key runs a shell command on $gate (no forced command), so the gate isn't placed off the prover's control"; exit 3 ;; esac
echo "GATE KEY: no shell on $gate (a command the gate doesn't take runs nothing)"
[ -z "$copy_refused" ] || { echo "REFUSED: the copy path to$copy_refused takes no write from this node's copy key"; exit 3; }
