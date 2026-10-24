#!/usr/bin/env bash
# N2's advised unit in zero knowledge (n2_advised_zk.py) on a CPU machine, on hidden_zk's builds (its job.sh, sourced:
# C-Flock's flock-circuit and the Lean flock-verify) with the firewall's Lean frame, N2's network-records and the
# sampled-proofs verifier built in the tree, as n2_zk.sh runs n2_zk.py. From a clone, in a slot of network-accounting's
# pool on vy-nebius-cpu-1:
#   research run --on vy-nebius-cpu-1 --source . --project verity --as LANE --cwd clone --timeout 180m \
#     --env N2_MEM_GB=56 -- env CHECK_SLOTS=/workspace/research/locks/network/slots python3 tools/verity/check/slot.py \
#     --by LANE -- bash benchmarks/network_traces/n2_advised_zk.sh
# env: N2_MEM_GB (what job.sh's stages may take, within the slot's)  N2_SEED (the advised tests' session, 7 by default)
#      N2_E2E (--e2e: network-e2e's output directory, or a .tgz of it, unpacked into the work directory, e.g. one sent
#      with --send to $RESEARCH_RUN_DIR/inputs/)  N2_WINDOWS (--windows, with N2_E2E)
#      N2_JOBS (--jobs, members proved at once, 2 by default)  N2_PORT (--port)  N2_CACHE (the build cache, kept across
#      runs where writable)  N2_DEBUG_CLEAR=1 (--debug-clear)  N2_DRY=1 (the instances and references only)
# Kept in $OUT/n2adv: the driver's records and logs, the public files and everything the verifier received; never a
# prover's file (inst-*) nor a circuit (its pin is kept), nor a session's archive.
set -uo pipefail
export HZ_CONFIGS=n2adv HZ_MEM_GB=${N2_MEM_GB:?}
if [ -n "${N2_CACHE:-}" ] && mkdir -p "$N2_CACHE" 2>/dev/null && [ -w "$N2_CACHE" ]; then export HZ_CACHE=$N2_CACHE; fi
source benchmarks/pouw/hidden_zk/job.sh
export PATH=$HOME/.elan/bin:$PATH
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python
setup
(cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier) > "$OUT/build-sampled-proofs.log" 2>&1 \
  || { tail -40 "$OUT/build-sampled-proofs.log"; exit 7; }
(cd verity/core/protocols/network/lean && lake build network-records) > "$OUT/build-network-records.log" 2>&1 \
  || { tail -40 "$OUT/build-network-records.log"; exit 7; }
export VERITY_NETWORK_RECORDS_EXE=$SRC/verity/core/protocols/network/lean/.lake/build/bin/network-records
export FC_FIREWALL_LEAN=${LEAN%/*}/flock-firewall
(exec 8>"${LEAN%/.lake/*}.firewall.lock"; flock 8; [ -x "$FC_FIREWALL_LEAN" ] || (cd "${LEAN%/.lake/*}" && lake build flock-firewall)) \
  > "$OUT/build-firewall.log" 2>&1
[ -x "$FC_FIREWALL_LEAN" ] || { tail -20 "$OUT/build-firewall.log"; log "n2adv: no firewall frame"; exit 7; }

E2E=${N2_E2E:-}
if [ -f "$E2E" ]; then
  mkdir -p "$W/e2e" && tar xzf "$E2E" -C "$W/e2e" || exit 7
  E2E=$(dirname "$(find "$W/e2e" -name report.json -print -quit)")
fi
log "n2adv: proving, ${E2E:+e2e $E2E windows ${N2_WINDOWS:-1}, }seed ${N2_SEED:-7}, jobs ${N2_JOBS:-2}${N2_DRY:+, dry}"
"$PY" benchmarks/network_traces/n2_advised_zk.py prove --seed "${N2_SEED:-7}" ${E2E:+--e2e "$E2E"} \
  ${N2_WINDOWS:+--windows "$N2_WINDOWS"} --bin "$BIN" --lean "$LEAN" --out "$W/n2adv" \
  --jobs "${N2_JOBS:-2}" ${N2_PORT:+--port "$N2_PORT"} ${N2_DEBUG_CLEAR:+--debug-clear} ${N2_DRY:+--dry} 2>&1 \
  | tee -a "$OUT/n2adv.log" | cut -c1-4000
rc=${PIPESTATUS[0]}
mkdir -p "$OUT/n2adv"
(cd "$W/n2adv" && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -path '*/archive/*' -print0 | xargs -0 -r -I{} cp --parents {} "$OUT/n2adv/")
log "n2adv: done, rc $rc"
exit "$rc"
