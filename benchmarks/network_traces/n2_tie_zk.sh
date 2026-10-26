#!/usr/bin/env bash
# The ingress tie in zero knowledge (n2_tie_zk.py) on a CPU machine, on hidden_zk's builds (its job.sh, sourced:
# C-Flock's flock-circuit and the Lean flock-verify) with the firewall's Lean frame and the sampled-proofs verifier
# built in the tree, as n2_zk.sh runs n2_zk.py. From a clone, in a slot of network-accounting's pool:
#   research run --on MACHINE --source . --project verity --as LANE --cwd clone --timeout 180m --env N2_MEM_GB=56 -- \
#     env CHECK_SLOTS=/workspace/research/locks/network/slots python3 tools/verity/check/slot.py --by LANE -- \
#     bash benchmarks/network_traces/n2_tie_zk.sh
# env: N2_MEM_GB (what job.sh's stages may take, within the slot's)  N2_PORT (--port; the members are proved one after
#      another)  N2_CACHE (the build cache, kept across runs where writable)  N2_DEBUG_CLEAR=1
#      (--debug-clear)  N2_DRY=1 (the instances and references only)
# Kept in $OUT/n2tie: the driver's records and logs, the public files and everything the verifier received; never a
# prover's file (inst-*, prover-instances.bin) nor a circuit (its pin is kept), nor a session's archive.
set -uo pipefail
export HZ_CONFIGS=n2tie HZ_MEM_GB=${N2_MEM_GB:?}
if [ -n "${N2_CACHE:-}" ] && mkdir -p "$N2_CACHE" 2>/dev/null && [ -w "$N2_CACHE" ]; then export HZ_CACHE=$N2_CACHE; fi
source benchmarks/pouw/hidden_zk/job.sh
export PATH=$HOME/.elan/bin:$PATH
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python
setup
(cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier) > "$OUT/build-sampled-proofs.log" 2>&1 \
  || { tail -40 "$OUT/build-sampled-proofs.log"; exit 7; }
export FC_FIREWALL_LEAN=${LEAN%/*}/flock-firewall
(exec 8>"${LEAN%/.lake/*}.firewall.lock"; flock 8; [ -x "$FC_FIREWALL_LEAN" ] || (cd "${LEAN%/.lake/*}" && lake build flock-firewall)) \
  > "$OUT/build-firewall.log" 2>&1
[ -x "$FC_FIREWALL_LEAN" ] || { tail -20 "$OUT/build-firewall.log"; log "n2tie: no firewall frame"; exit 7; }

log "n2tie: proving${N2_DRY:+, dry}"
"$PY" benchmarks/network_traces/n2_tie_zk.py prove --bin "$BIN" --lean "$LEAN" --out "$W/n2tie" \
  ${N2_PORT:+--port "$N2_PORT"} ${N2_DEBUG_CLEAR:+--debug-clear} ${N2_DRY:+--dry} 2>&1 \
  | tee -a "$OUT/n2tie.log" | cut -c1-4000
rc=${PIPESTATUS[0]}
mkdir -p "$OUT/n2tie"
(cd "$W/n2tie" && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -name prover-instances.bin ! -path '*/archive/*' \
  -print0 | xargs -0 -r -I{} cp --parents {} "$OUT/n2tie/")
log "n2tie: done, rc $rc"
exit "$rc"
