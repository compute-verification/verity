#!/usr/bin/env bash
# N2's egress bucket in zero knowledge (egress_zk.py) on a CPU machine, on hidden_zk's builds (its job.sh, sourced:
# C-Flock's flock-circuit and the Lean flock-verify) with the firewall's Lean frame, as warden_zk.sh runs warden_zk.py.
# From a clone, in one of the network slots, as a job of at most 180 minutes:
#   env EZ_MEM_GB=64 CHECK_SLOTS=/workspace/research/locks/network/slots python3 tools/verity/check/slot.py \
#     --by LANE -- bash benchmarks/network_traces/egress_zk.sh
# env: EZ_MEM_GB (what job.sh's stages may take, within the slot's)  EZ_PORT (--port)  EZ_CACHE (the build cache, kept
#      across runs where writable; hidden_zk's by default)  EZ_DRY=1 (the statements and the host's read of them only)
# Kept in $OUT/ezk: egress_zk's summary and logs and everything the verifier received; never a prover's file (inst-*)
# nor a circuit (its pin is kept), nor a session's archive.
set -uo pipefail
export HZ_CONFIGS=ezk HZ_MEM_GB=${EZ_MEM_GB:?}
EZ_CACHE=${EZ_CACHE:-/workspace/research/cache/pouw-hidden-zk}
if mkdir -p "$EZ_CACHE" 2>/dev/null && [ -w "$EZ_CACHE" ]; then export HZ_CACHE=$EZ_CACHE; fi
source benchmarks/pouw/hidden_zk/job.sh
export PATH=$HOME/.elan/bin:$PATH
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python
setup
(cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier) > "$OUT/build-sampled-proofs.log" 2>&1 \
  || { tail -40 "$OUT/build-sampled-proofs.log"; exit 7; }
export FC_FIREWALL_LEAN=${LEAN%/*}/flock-firewall
(exec 8>"${LEAN%/.lake/*}.firewall.lock"; flock 8; [ -x "$FC_FIREWALL_LEAN" ] || (cd "${LEAN%/.lake/*}" && lake build flock-firewall)) \
  > "$OUT/build-firewall.log" 2>&1
[ -x "$FC_FIREWALL_LEAN" ] || { tail -20 "$OUT/build-firewall.log"; log "ezk: no firewall frame"; exit 7; }

log "ezk: proving${EZ_DRY:+, dry}"
"$PY" benchmarks/network_traces/egress_zk.py prove --bin "$BIN" --lean "$LEAN" --out "$W/ezk" ${EZ_PORT:+--port "$EZ_PORT"} \
  ${EZ_DRY:+--dry} 2>&1 | tee -a "$OUT/ezk.log" | cut -c1-4000
rc=${PIPESTATUS[0]}
mkdir -p "$OUT/ezk"
(cd "$W/ezk" && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -name 'signer-state*' ! -path '*/archive/*' -print0 \
  | xargs -0 -r -I{} cp --parents {} "$OUT/ezk/")
log "ezk: done, rc $rc"
exit "$rc"
