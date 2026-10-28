#!/usr/bin/env bash
# A server block's proof units read against the network certifier's transfer rows (transfer_zk.py) on a CPU machine, on
# hidden_zk's builds (its job.sh, sourced: Flock's flock-circuit and the Lean flock-verify) with the firewall's Lean
# frame. From a clone:
#   research run --on NODE --source . --project verity --as LANE --cwd clone --cpus 16 --mem-gb 64 --timeout 180m \
#     --env TZ_MEM_GB=64 -- env CHECK_SLOTS=SLOTS python3 tools/verity/check/slot.py --by LANE -- \
#     bash benchmarks/network_traces/transfer_zk.sh
# env: TZ_MEM_GB (what job.sh's stages may take, within the slot's)  TZ_UNITS (transfer_zk.py's --units, 8)
#      TZ_SHAPE (its --shape: tiny, or h200)
#      TZ_CACHE (the build cache, kept across runs where writable; hidden_zk's by default)  TZ_DRY=1 (the statements
#      and the host's read of them only)
# Kept in $OUT/tzk: transfer_zk's summary and logs and everything the verifier received; never a prover's file (inst-*),
# the network certifier's signer state, a circuit (its pin is kept), nor a session's archive.
set -uo pipefail
export HZ_CONFIGS=tzk HZ_MEM_GB=${TZ_MEM_GB:?}
TZ_CACHE=${TZ_CACHE:-/workspace/research/cache/pouw-hidden-zk}
if mkdir -p "$TZ_CACHE" 2>/dev/null && [ -w "$TZ_CACHE" ]; then export HZ_CACHE=$TZ_CACHE; fi
source benchmarks/pouw/hidden_zk/job.sh
export PATH=$HOME/.elan/bin:$PATH
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python
setup
(cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier) > "$OUT/build-sampled-proofs.log" 2>&1 \
  || { tail -40 "$OUT/build-sampled-proofs.log"; exit 7; }
export FC_FIREWALL_LEAN=${LEAN%/*}/flock-firewall
(exec 8>"${LEAN%/.lake/*}.firewall.lock"; flock 8; [ -x "$FC_FIREWALL_LEAN" ] || (cd "${LEAN%/.lake/*}" && lake build flock-firewall)) \
  > "$OUT/build-firewall.log" 2>&1
[ -x "$FC_FIREWALL_LEAN" ] || { tail -20 "$OUT/build-firewall.log"; log "tzk: no firewall frame"; exit 7; }

log "tzk: proving ${TZ_SHAPE:-tiny}, units ${TZ_UNITS:-8}${TZ_DRY:+, dry}"
"$PY" benchmarks/network_traces/transfer_zk.py prove --bin "$BIN" --lean "$LEAN" --out "$W/tzk" \
  --units "${TZ_UNITS:-8}" --shape "${TZ_SHAPE:-tiny}" ${TZ_DRY:+--dry} 2>&1 | tee -a "$OUT/tzk.log" | cut -c1-4000
rc=${PIPESTATUS[0]}
mkdir -p "$OUT/tzk"
(cd "$W/tzk" && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -name 'signer-state*' ! -path '*/archive/*' -print0 \
  | xargs -0 -r -I{} cp --parents {} "$OUT/tzk/")
log "tzk: done, rc $rc"
exit "$rc"
