#!/usr/bin/env bash
# NET-68's scan composed at w63's class in zero knowledge (scan_zk.py) on a CPU machine, on hidden_zk's builds (its
# job.sh, sourced: C-Flock's flock-circuit and the Lean flock-verify) with the firewall's Lean frame and the
# sampled-proofs verifier built in the tree. From a clone, in a slot of network-accounting's pool:
#   research run --on MACHINE --source . --project verity --as LANE --cwd clone --timeout 180m --env SC_MEM_GB=128 -- \
#     env CHECK_SLOTS=/workspace/research/locks/network/slots python3 tools/verity/check/slot.py --by LANE -- \
#     bash benchmarks/network_traces/scan_zk.sh
# env: SC_MEM_GB (what job.sh's stages may take, within the slot's)  SC_CACHE (the build cache, kept across runs where
#      writable; hidden_zk's by default)  SC_DRY=1 (the run and the host's read of every round only)
# Kept in $OUT/sczk: the driver's summary and logs and everything the verifier received; never a prover's file (inst-*)
# nor a circuit (its pin is kept), nor a session's archive.
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

log "sczk: proving${SC_DRY:+, dry}"
"$PY" benchmarks/network_traces/scan_zk.py prove --bin "$BIN" --lean "$LEAN" --out "$W/sczk" ${SC_DRY:+--dry} 2>&1 \
  | tee -a "$OUT/sczk.log" | cut -c1-4000
rc=${PIPESTATUS[0]}
mkdir -p "$OUT/sczk"
(cd "$W/sczk" && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -name 'instances.bin' ! -path '*/archive/*' \
  ! -name 'signer-state.json' -print0 | xargs -0 -r -I{} cp --parents {} "$OUT/sczk/")
log "sczk: done, rc $rc"
exit "$rc"
