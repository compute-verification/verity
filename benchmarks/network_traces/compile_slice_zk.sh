#!/usr/bin/env bash
# The compile slice in zero knowledge (compile_slice_zk.py) on a CPU machine, on hidden_zk's builds (its job.sh, sourced:
# C-Flock's flock-circuit and the Lean flock-verify) with the firewall's Lean frame and the sampled-proofs verifier built
# in the tree. From a clone, in one of the network pool's pinned slots:
#   research run --on vy-nebius-cpu-1 --source . --project verity --as LANE --cwd clone --mem-gb 64 --timeout 180m \
#     --env CS_MEM_GB=64 -- env CHECK_SLOTS=/workspace/research/locks/network/slots python3 tools/verity/check/slot.py \
#     --by LANE -- bash benchmarks/network_traces/compile_slice_zk.sh
# env: CS_MEM_GB (what job.sh's stages may take, within the slot's)  CS_POINTS (compile_slice_zk.py's --points, 1 by
#      default)  CS_CACHE (the build cache, kept across runs where writable; default pouw-hidden-zk's)  CS_DRY=1 (the
#      slice and its references only)
# Kept in $OUT/cszk: the driver's records and logs, the public files and everything the verifier received; never a
# prover's file (inst-*) nor a circuit (its pin is kept), nor a session's archive.
set -uo pipefail
export HZ_CONFIGS=cszk HZ_MEM_GB=${CS_MEM_GB:?}
CS_CACHE=${CS_CACHE:-/workspace/research/cache/pouw-hidden-zk}
if mkdir -p "$CS_CACHE" 2>/dev/null && [ -w "$CS_CACHE" ]; then export HZ_CACHE=$CS_CACHE; fi
source benchmarks/pouw/hidden_zk/job.sh
export PATH=$HOME/.elan/bin:$PATH
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python
setup
(cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier) > "$OUT/build-sampled-proofs.log" 2>&1 \
  || { tail -40 "$OUT/build-sampled-proofs.log"; exit 7; }
export FC_FIREWALL_LEAN=${LEAN%/*}/flock-firewall
(exec 8>"${LEAN%/.lake/*}.firewall.lock"; flock 8; [ -x "$FC_FIREWALL_LEAN" ] || (cd "${LEAN%/.lake/*}" && lake build flock-firewall)) \
  > "$OUT/build-firewall.log" 2>&1
[ -x "$FC_FIREWALL_LEAN" ] || { tail -20 "$OUT/build-firewall.log"; log "cszk: no firewall frame"; exit 7; }

log "cszk: proving, points ${CS_POINTS:-1}${CS_DRY:+, dry}"
"$PY" benchmarks/network_traces/compile_slice_zk.py prove --bin "$BIN" --lean "$LEAN" --out "$W/cszk" \
  --points "${CS_POINTS:-1}" ${CS_DRY:+--dry} 2>&1 | tee -a "$OUT/cszk.log" | cut -c1-4000
rc=${PIPESTATUS[0]}
mkdir -p "$OUT/cszk"
(cd "$W/cszk" && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -path '*/archive/*' -print0 | xargs -0 -r -I{} cp --parents {} "$OUT/cszk/")
log "cszk: done, rc $rc"
exit "$rc"
