#!/usr/bin/env bash
# The V3c follow-on with the re-proof copy, on the follow-on's node: `served_verdicts.sh v3c` on D with REPROOF_CMD this
# kit's reproof_copy.sh, which after V3c's audit (and the split certifier's chain and period check) copies the serve dir's
# re-proof inputs to REPROOF_DEST and writes their listing, D/private/reproof-listing.txt; the follow-on then writes the
# onsite record with --reproof-copy REPROOF_COPY and --reproof-listing, before its EXIT trap publishes out/. The copy,
# its listing and their bytes stay off out/. Without REPROOF_DEST it copies nothing and the record names no copy.
# Once that has published out/, circuits_out.py puts V3c's public circuit files and their pins in out/v3c-circuits (a file
# over 1 GiB kept on this node instead, in the serve dir's v3c-circuits/, which pins.json names), each a hard link to the
# audit's own file, so they take no second copy of the window's disk; the job's exit is served_verdicts.sh's.
#   research run --on NODE --project verity --source TREE --cwd clone --timeout 108000 --mem-gb 650 --env D=COPY_OF_D \
#     --env REPROOF_COPY=vy-nebius-cpu-3:/PATH/SERVED_RUN_ID/ --env REPROOF_DEST=USER@HOST:DEST/ [--env REPROOF_KEY=PATH] \
#     [--env PP_CHALLENGER=gate:research@204.12.178.84 --env PP_PEER=NODE --send challenger-key.json --send main.txt] \
#     --send reproof_follow.sh --send reproof_copy.sh --send reproof_files.py --send circuits_out.py -- \
#     bash -c 'exec bash "$RESEARCH_RUN_DIR/inputs/reproof_follow.sh"'
# NODE is the run's follow-on node, its settings file's PP_PEER: vy-nebius-cpu-11 for (i), vy-nebius-cpu-12 for (ii).
# With --env PP_CHALLENGER=gate:research@204.12.178.84 --env PP_PEER=NODE, V3c's audit draws on vy-challenger-1's
# live challenger through its gate with the lane's gate key on this node (PP_GATE_KEY, default ~research's
# vy-gate-network-accounting), under infra's key record (PP_CHALLENGER_KEY, default the sent challenger-key.json, the store's
# `challenger-key=current`) and pinned on main's first-parent line (PP_MAIN, default the sent main.txt, `git rev-list
# --first-parent origin/main` at launch): add --send challenger-key.json --send main.txt. The gate opens a root once, ever.
# The timeout holds V3c's audit (V3C_AUDIT_TIMEOUT, 21h here: at 4,32,4 the audit composes each drawn template's circuit
# after its coins, one after another, for about 5 h, then runs its 63 sessions for about 13 h and their Lean checks
# beside them one at a time, V3C_LEAN_WORKERS, whose last ends about 1 h after the last session), the chain after it
# (CERTIFIER_TIMEOUT, 4h), its period check (60m), two composites (60m each) and the rest. Those composed circuits stay on
# this node's disk until the follow-on ends, so it starts only with FOLLOW_DISK_GB free under D (default 650): a compose
# that finds the disk full, or an audit past its limit, comes after the gate opened the root's one session.
set -uo pipefail
IN=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${D:?the copy of the serve dir on this node}"
export D V3C_AUDIT_TIMEOUT=${V3C_AUDIT_TIMEOUT:-21h}
free=$(df --output=avail -BG "$D" | tail -1 | tr -dc 0-9) need=${FOLLOW_DISK_GB:-650}
[ "${free:-0}" -ge "$need" ] || { echo "PREFLIGHT: ${free:-?} GB free under $D, under FOLLOW_DISK_GB $need GB that V3c's composed circuits take at 4,32,4"; exit 3; }
echo "DISK: $free GB free under $D (FOLLOW_DISK_GB $need)"
case ${PP_CHALLENGER:-local} in gate:*)
  : "${PP_PEER:?PP_CHALLENGER=gate:… needs PP_PEER, this node as /etc/vy/peers on the gate names it}"
  export PP_CHALLENGER PP_PEER PP_CHALLENGER_KEY=${PP_CHALLENGER_KEY:-$IN/challenger-key.json} \
    PP_MAIN=${PP_MAIN:-$IN/main.txt} PP_GATE_KEY=${PP_GATE_KEY:-$HOME/.ssh/vy-gate-network-accounting}
  for f in "$PP_CHALLENGER_KEY" "$PP_MAIN" "$PP_GATE_KEY"; do
    [ -s "$f" ] || { echo "PREFLIGHT: $f is missing or empty (PP_CHALLENGER $PP_CHALLENGER)"; exit 3; }
  done
  echo "COINS: V3c's audit on $PP_CHALLENGER, peer $PP_PEER, key record $PP_CHALLENGER_KEY ($(wc -l < "$PP_MAIN") commits of main's first-parent line, from $(head -c 10 "$PP_MAIN"))";;
esac
# With --env CERTIFIER_GATE=USER@HOST (the network certifier's host, vy-auditor-2), CERTIFIER_PIN (the commit its gate runs,
# INF-494's) and CERTIFIER_PEER (the served node as its peers file names it), and --send certifier_key.py: this node, the
# auditor's side, reads that host's own record of the run the serve dir kept, with a read-only key (CERTIFIER_GATE_KEY,
# default ~research's vy-auditor-2-read), and writes CERTIFIER_KEY (certifier-key/v1, beside out/), the network
# certifier's key and session the network checks take; a check never takes them from the prover's copy. Not read, the
# job goes on with CERTIFIER_KEY unset and the network verdict isn't established.
if [ -n "${CERTIFIER_GATE:-}" ]; then
  CERTIFIER_KEY=${CERTIFIER_KEY:-${RESEARCH_RUN_DIR:-.}/certifier-key.json}
  python3 "$IN/certifier_key.py" "$D" "$CERTIFIER_KEY" --gate "$CERTIFIER_GATE" --pin "${CERTIFIER_PIN:?CERTIFIER_GATE needs CERTIFIER_PIN}" \
    --key "${CERTIFIER_GATE_KEY:-$HOME/.ssh/vy-auditor-2-read}" --peer "${CERTIFIER_PEER:?CERTIFIER_GATE needs CERTIFIER_PEER}"
  case $? in
    0|4) export CERTIFIER_KEY ;;
    *) unset CERTIFIER_KEY; echo "CERTIFIER-KEY: none: the network checks have no auditor's key, so the network verdict isn't established" ;;
  esac
fi
# On a tree whose onsite record checks the auditor's own record of its network certifier (benchmarks/one_stage/
# site_registration.py there), this node, the auditor's side, writes the auditor's site registration (SITE_REGISTRATION,
# beside out/: the certifier's key from CERTIFIER_KEY, placed on CERTIFIER_GATE's host for the served node CERTIFIER_PEER,
# its uplink each way SITE_BETA bits a tick and SITE_LATENCY ticks, a tick 1 ns: default the census's 100 Gb/s and 1 ms,
# 100 and 1000000), and the follow-on is refused without CERTIFIER_GATE and CERTIFIER_KEY, before V3c's audit opens the
# root's one session. Only the follow-on sets these: the
# record's check reads auditor-held inputs alone. An older tree takes neither, and nothing here runs on it.
if [ -f benchmarks/one_stage/site_registration.py ]; then
  if [ -n "${CERTIFIER_KEY:-}" ] && [ -n "${CERTIFIER_GATE:-}" ]; then
    SITE_REGISTRATION=${SITE_REGISTRATION:-${RESEARCH_RUN_DIR:-.}/site-registration.json}
    if (PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH; uv sync -q --all-packages --extra torch-cpu) \
      && .venv/bin/python benchmarks/one_stage/site_registration.py "$CERTIFIER_KEY" "$SITE_REGISTRATION" \
        --node "${CERTIFIER_PEER:-}" --certifier "${CERTIFIER_GATE#*@}" --beta "${SITE_BETA:-100}" \
        --latency "${SITE_LATENCY:-1000000}"; then
      export SITE_REGISTRATION
    else
      unset SITE_REGISTRATION
      echo "SITE-REGISTRATION: none: the onsite record's auditor-record check doesn't hold, so the network verdict isn't established"
    fi
  fi
  unset_=CERTIFIER_GATE
  [ -z "${CERTIFIER_GATE:-}" ] || unset_="CERTIFIER_KEY (CERTIFIER-KEY above says why)"
  [ -n "${CERTIFIER_GATE:-}" ] && [ -n "${CERTIFIER_KEY:-}" ] \
    || { echo "PREFLIGHT: this tree's onsite record checks the auditor's record of its network certifier, read off" \
      "CERTIFIER_GATE's host as CERTIFIER_KEY, and $unset_ is unset"; exit 3; }
fi
case ${REPROOF_DEST:-} in *:*)
  [ -s "${REPROOF_KEY:-$HOME/.ssh/reproof_copy_ed25519}" ] \
    || { echo "PREFLIGHT: no key ${REPROOF_KEY:-$HOME/.ssh/reproof_copy_ed25519} for REPROOF_DEST $REPROOF_DEST on this node"; exit 3; };;
esac
# the window's sealed row dir reproof_send.sh sent (D/private/rows.sha256), held to the served node's sums before V3c's
# audit opens the root's one session
if [ -f "$D/private/rows.sha256" ]; then
  (umask 077; cd "$D" && sha256sum --check --strict --quiet private/rows.sha256 > private/rows-check.log 2>&1) \
    || { echo "PREFLIGHT: the row dir isn't as the served node hashed it (D/private/rows-check.log)"; exit 3; }
  echo "ROWS: the window's sealed row dir as the served node hashed it (D/private/rows.sha256): PASS"
else
  echo "ROWS: no sealed row dir here (no D/private/rows.sha256)"
fi
# FOLLOW_HOLD: infra's live memory hold for this follow-on (/workspace/ramlock/NAME; its master copy in research's home is
# infra's and stays), which follow_launch.sh removes before the launch so this job's --mem-gb fits; still there, the job's
# reservation now holds V3c's memory in its place
case ${FOLLOW_HOLD:-} in
  /workspace/ramlock/*.json) if [ -e "$FOLLOW_HOLD" ]; then
      rm -f -- "$FOLLOW_HOLD" && echo "HOLD: removed $FOLLOW_HOLD, this job's --mem-gb now holding V3c's memory"
    else echo "HOLD: $FOLLOW_HOLD already gone, this job's --mem-gb holding V3c's memory"; fi ;;
  '') ;;
  *) echo "PREFLIGHT: FOLLOW_HOLD $FOLLOW_HOLD isn't a live hold under /workspace/ramlock"; exit 3 ;;
esac
if [ -n "${REPROOF_DEST:-}" ]; then
  : "${REPROOF_COPY:?NODE:/PATH where REPROOF_DEST lands, which the onsite record names}"
  export REPROOF_CMD="bash $(printf %q "$IN/reproof_copy.sh") \"\$@\"" REPROOF_LISTING=$D/private/reproof-listing.txt \
    REPROOF_COPY REPROOF_DEST
else
  unset REPROOF_CMD REPROOF_COPY REPROOF_LISTING
fi
bash benchmarks/one_stage/served_verdicts.sh v3c; rc=$?
# a tree whose follow-on removes the window's circuit files itself (`drop_circuits`, its V3C-CIRCUITS line) leaves none here
grep -q '^drop_circuits()' benchmarks/one_stage/served_verdicts.sh && exit $rc
(set -a; . "$D/private/deferred/deferred.env"; set +a
 python3 "$IN/circuits_out.py" "$D/v3c-w$K-${V3C_CLASS//,/-}" "${OUT:-${RESEARCH_RUN_DIR:-.}/out}") \
  || echo "V3C-CIRCUITS: not all in out/ (out/v3c-circuits/pins.json says why)"
exit $rc
