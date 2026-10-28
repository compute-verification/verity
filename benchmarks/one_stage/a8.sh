#!/usr/bin/env bash
# One-stage audit, run A8, on a CPU machine: one window of the remote bandwidth audit's openings proved in zero knowledge
# through the opening unit (a8.py), honest and with a tampered block, and the cost points, on hidden_zk's builds (its
# job.sh, sourced: Flock's flock-circuit and the Lean flock-verify). From a clone, since the verifier of record builds
# the sampled-proofs Lean in the tree:
#   research run --on vy-nebius-cpu-1 --source . --pool research --as LANE --project verity --cwd clone --env A8_MEM_GB=96 -- \
#     bash benchmarks/one_stage/a8.sh
# env: A8_MEM_GB (what job.sh's stages may take, within the slot's)  A8_BLOCKS  A8_PLAN  A8_SAMPLES  A8_POINTS (a8.py's
#      --blocks, --plan, --samples, --points; points 1,4,16 by default)  A8_ARMS (fw,nofw by default; fw builds the
#      ZK server's Lean frame beside flock-verify)  A8_COLD  A8_CACHE (the build cache, kept across runs where writable)
#      A8_NO_AUDIT=1 (a8.py's --no-audit: the honest statement staged and the cost points only)
# Kept in $OUT/a8: a8's records and logs, the public files and everything the verifier received; never a prover's file
# (inst-*) nor a circuit (its pin is kept), nor a session's archive.
set -uo pipefail
export HZ_CONFIGS=a8 HZ_MEM_GB=${A8_MEM_GB:?}
if [ -n "${A8_CACHE:-}" ] && mkdir -p "$A8_CACHE" 2>/dev/null && [ -w "$A8_CACHE" ]; then export HZ_CACHE=$A8_CACHE; fi
source benchmarks/pouw/hidden_zk/job.sh
export PATH=$HOME/.elan/bin:$PATH
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python
setup
(cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier) > "$OUT/build-sampled-proofs.log" 2>&1 \
  || { tail -40 "$OUT/build-sampled-proofs.log"; exit 7; }
ARMS=${A8_ARMS:-fw,nofw}
if [[ ,$ARMS, == *,fw,* ]]; then
  export FC_FIREWALL_LEAN=${LEAN%/*}/flock-firewall
  (exec 8>"${LEAN%/.lake/*}.firewall.lock"; flock 8; [ -x "$FC_FIREWALL_LEAN" ] || (cd "${LEAN%/.lake/*}" && lake build flock-firewall)) \
    > "$OUT/build-firewall.log" 2>&1
  [ -x "$FC_FIREWALL_LEAN" ] || { tail -20 "$OUT/build-firewall.log"; log "a8: no firewall frame, arms nofw only"; ARMS=nofw; }
fi
log "a8: blocks ${A8_BLOCKS:-default}, plan ${A8_PLAN:-default}, samples ${A8_SAMPLES:-default}, points ${A8_POINTS:-1,4,16}, arms $ARMS"
"$PY" benchmarks/one_stage/a8.py --bin "$BIN" --lean "$LEAN" --out "$W/a8" ${A8_BLOCKS:+--blocks "$A8_BLOCKS"} \
  ${A8_PLAN:+--plan "$A8_PLAN"} ${A8_SAMPLES:+--samples "$A8_SAMPLES"} --points "${A8_POINTS:-1,4,16}" --arms "$ARMS" \
  --cold "${A8_COLD:-1}" ${A8_NO_AUDIT:+--no-audit} 2>&1 | tee -a "$OUT/a8.log" | cut -c1-4000
rc=${PIPESTATUS[0]}
mkdir -p "$OUT/a8"
(cd "$W/a8" && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -path '*/archive/*' -print0 | xargs -0 -r -I{} cp --parents {} "$OUT/a8/")
log "a8: done, rc $rc"
exit "$rc"
