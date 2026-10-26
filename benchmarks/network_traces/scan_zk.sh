#!/usr/bin/env bash
# NET-68's scan composed at w63's class in zero knowledge (scan_zk.py) on a CPU machine, on hidden_zk's builds (its
# job.sh, sourced: C-Flock's flock-circuit and the Lean flock-verify) with the firewall's Lean frame and the
# sampled-proofs verifier built in the tree. From a clone, in a slot of network-accounting's pool:
#   research run --on MACHINE --source . --project verity --as LANE --cwd clone --timeout 180m --env SC_MEM_GB=128 -- \
#     env CHECK_SLOTS=/workspace/research/locks/network/slots python3 tools/verity/check/slot.py --by LANE -- \
#     bash benchmarks/network_traces/scan_zk.sh
# env: SC_MEM_GB (what job.sh's stages may take, within the slot's)  SC_CACHE (the build cache, kept across runs where
#      writable; hidden_zk's by default)  SC_DRY=1 (the run and the host's read of every round only)
#      SC_WARDEN_RUN=DIR (a warden_serve run's directory, `prove --warden-run`)  SC_REHEARSE=1 (first a rehearsal,
#      `warden_serve.py rehearse --fake --ids` at scan_zk's schedule, SC_REQUESTS requests at SC_RATE a second, and
#      prove its run)  SC_DRAW=K (`prove --draw K`: Lean draw --stratified K on the verifier's coins)
#      SC_CHALLENGER=local|gate:USER@HOST (with SC_DRAW: the honest round's K on the challenger's coins, `prove
#      --challenger`, pinned on main's first-parent history fetched from the clone's origin; gate: also SC_PEER, this
#      node as the gate's /etc/vy/peers names it, SC_CHALLENGER_KEY, the deployment's challenger-key/v1 record, and
#      SC_GATE_KEY, the lane's gate key)
# Kept in $OUT/sczk: the driver's summary and logs and everything the verifier received; never a prover's file (inst-*)
# nor a circuit (its pin is kept), nor a session's archive. A rehearsal's public files go in $OUT/warden, its records
# and salts (the prover's) nowhere.
set -uo pipefail
export HZ_CONFIGS=sczk HZ_MEM_GB=${SC_MEM_GB:?}
SC_CACHE=${SC_CACHE:-/workspace/research/cache/pouw-hidden-zk}
if mkdir -p "$SC_CACHE" 2>/dev/null && [ -w "$SC_CACHE" ]; then export HZ_CACHE=$SC_CACHE; fi
source benchmarks/pouw/hidden_zk/job.sh
export PATH=$HOME/.elan/bin:$PATH
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python
setup
(cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier) > "$OUT/build-sampled-proofs.log" 2>&1 \
  || { tail -40 "$OUT/build-sampled-proofs.log"; exit 7; }
export FC_FIREWALL_LEAN=${LEAN%/*}/flock-firewall
(exec 8>"${LEAN%/.lake/*}.firewall.lock"; flock 8; [ -x "$FC_FIREWALL_LEAN" ] || (cd "${LEAN%/.lake/*}" && lake build flock-firewall)) \
  > "$OUT/build-firewall.log" 2>&1
[ -x "$FC_FIREWALL_LEAN" ] || { tail -20 "$OUT/build-firewall.log"; log "sczk: no firewall frame"; exit 7; }

if [ -n "${SC_REHEARSE:-}" ]; then
  log "sczk: rehearsing the warden at scan_zk's schedule"
  # shellcheck disable=SC2046
  "$PY" benchmarks/network_traces/warden_serve.py rehearse --fake --ids --out "$W/warden" \
    --requests "${SC_REQUESTS:-12}" --rate "${SC_RATE:-2}" --tail "${SC_TAIL:-6}" --start-in 2 \
    $("$PY" benchmarks/network_traces/scan_zk.py schedule) 2>&1 | tee -a "$OUT/warden-rehearse.log" | cut -c1-4000
  log "sczk: rehearsal rc ${PIPESTATUS[0]}"
  mkdir -p "$OUT/warden"
  for f in deployment.json cells.jsonl certified.jsonl check.json client.json warden.json serve.log traffic.json; do
    [ -f "$W/warden/$f" ] && cp "$W/warden/$f" "$OUT/warden/"
  done
  SC_WARDEN_RUN=$W/warden
fi
CH=()
if [ -n "${SC_CHALLENGER:-}" ]; then   # the verifier's own copy of main's first-parent history, which the session's pin must be on
  UNSHALLOW=(); [ "$(git rev-parse --is-shallow-repository)" = true ] && UNSHALLOW=(--unshallow)
  git fetch -q "${UNSHALLOW[@]}" origin +refs/heads/main:refs/remotes/origin/main > "$OUT/fetch-main.log" 2>&1 \
    || git fetch -q "${UNSHALLOW[@]}" origin +refs/remotes/origin/main:refs/remotes/origin/main >> "$OUT/fetch-main.log" 2>&1 \
    && git rev-list --first-parent refs/remotes/origin/main > "$OUT/main.txt" || { tail -5 "$OUT/fetch-main.log"; exit 4; }
  CH=(--challenger "$SC_CHALLENGER" --main "$OUT/main.txt")
  case $SC_CHALLENGER in gate:*)
    [ -n "${SC_PEER:-}" ] && [ -n "${SC_CHALLENGER_KEY:-}" ] || { log "sczk: SC_CHALLENGER=gate:… needs SC_PEER and SC_CHALLENGER_KEY"; exit 4; }
    CH+=(--peer "$SC_PEER" --challenger-key "$SC_CHALLENGER_KEY" ${SC_GATE_KEY:+--gate-key "$SC_GATE_KEY"});;
  esac
fi
log "sczk: proving${SC_DRY:+, dry}${SC_WARDEN_RUN:+, warden run $SC_WARDEN_RUN}${SC_DRAW:+, draw $SC_DRAW}${SC_CHALLENGER:+, the honest draw on the coins of $SC_CHALLENGER}"
"$PY" benchmarks/network_traces/scan_zk.py prove --bin "$BIN" --lean "$LEAN" --out "$W/sczk" ${SC_DRY:+--dry} \
  ${SC_WARDEN_RUN:+--warden-run "$SC_WARDEN_RUN"} ${SC_DRAW:+--draw "$SC_DRAW"} "${CH[@]}" 2>&1 \
  | tee -a "$OUT/sczk.log" | cut -c1-4000
rc=${PIPESTATUS[0]}
mkdir -p "$OUT/sczk"
(cd "$W/sczk" && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -name 'instances.bin' ! -path '*/archive/*' \
  ! -name 'signer-state.json' -print0 | xargs -0 -r -I{} cp --parents {} "$OUT/sczk/")
log "sczk: done, rc $rc"
exit "$rc"
