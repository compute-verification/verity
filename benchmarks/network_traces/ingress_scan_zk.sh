#!/usr/bin/env bash
# The scan's ingress bucket unit at w63's class in zero knowledge (ingress_scan_zk.py) on a CPU machine, on hidden_zk's
# builds (its job.sh, sourced: C-Flock's flock-circuit and the Lean flock-verify) with the firewall's Lean frame and the
# sampled-proofs verifier built in the tree. From a clone, in a slot of network-accounting's pool:
#   research run --on MACHINE --source . --project verity --as LANE --cwd clone --timeout 180m --env IS_MEM_GB=64 -- \
#     env CHECK_SLOTS=/workspace/research/locks/network/slots python3 tools/verity/check/slot.py --by LANE -- \
#     bash benchmarks/network_traces/ingress_scan_zk.sh
# env: IS_MEM_GB (what job.sh's stages may take, within the slot's)  IS_CACHE (the build cache, kept across runs where
#      writable; hidden_zk's by default)  IS_DRY=1 (the run, the units and the reference only)
# Kept in $OUT/iszk: the driver's summary and logs and everything the verifier received; never a prover's file (inst-*)
# nor a circuit (its pin is kept), nor a session's archive.
set -uo pipefail
export HZ_CONFIGS=iszk HZ_MEM_GB=${IS_MEM_GB:?}
IS_CACHE=${IS_CACHE:-/workspace/research/cache/pouw-hidden-zk}
if mkdir -p "$IS_CACHE" 2>/dev/null && [ -w "$IS_CACHE" ]; then export HZ_CACHE=$IS_CACHE; fi
source benchmarks/pouw/hidden_zk/job.sh
export PATH=$HOME/.elan/bin:$PATH
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python
setup
(cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier) > "$OUT/build-sampled-proofs.log" 2>&1 \
  || { tail -40 "$OUT/build-sampled-proofs.log"; exit 7; }
export FC_FIREWALL_LEAN=${LEAN%/*}/flock-firewall
(exec 8>"${LEAN%/.lake/*}.firewall.lock"; flock 8; [ -x "$FC_FIREWALL_LEAN" ] || (cd "${LEAN%/.lake/*}" && lake build flock-firewall)) \
  > "$OUT/build-firewall.log" 2>&1
[ -x "$FC_FIREWALL_LEAN" ] || { tail -20 "$OUT/build-firewall.log"; log "iszk: no firewall frame"; exit 7; }

log "iszk: proving${IS_DRY:+, dry}"
"$PY" benchmarks/network_traces/ingress_scan_zk.py prove --bin "$BIN" --lean "$LEAN" --out "$W/iszk" ${IS_DRY:+--dry} 2>&1 \
  | tee -a "$OUT/iszk.log" | cut -c1-4000
rc=${PIPESTATUS[0]}
mkdir -p "$OUT/iszk"
(cd "$W/iszk" && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -path '*/archive/*' -print0 \
  | xargs -0 -r -I{} cp --parents {} "$OUT/iszk/")
log "iszk: done, rc $rc"
exit "$rc"
