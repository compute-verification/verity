#!/usr/bin/env bash
# One-stage audit, run A6, on a CPU machine: the W8A8 coordinates of an int8 decoder layer under the stratified law (a6.py),
# on hidden_zk's builds (its job.sh, sourced: C-Flock's flock-circuit and the Lean flock-verify, cached under $A6_CACHE). From
# a clone, since the verifier of record builds the sampled-proofs Lean in the tree:
#   A6_MEM_GB=96 bash benchmarks/one_stage/a6.sh
#   (a job run from a clone of the tree, 96 GB)
# env: A6_MEM_GB (the job's memory, GB)  A6_MEMBERS (a6.py's --members)  A6_K (its --k)  A6_CHUNK (its --chunk)
#      A6_CACHE (the build cache)
# Kept in $OUT/a6: a6's records and logs, the public files and the sessions (their proofs and records); never a prover's file
# (inst-*) nor a circuit (its pin is kept), nor a session's archive (copies of the circuit and public file).
set -uo pipefail
export HZ_CONFIGS=a6 HZ_MEM_GB=${A6_MEM_GB:?}
[ -n "${A6_CACHE:-}" ] && export HZ_CACHE=$A6_CACHE
source benchmarks/pouw/hidden_zk/job.sh
export PATH=$HOME/.elan/bin:$PATH
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python
setup
(cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier) > "$OUT/build-sampled-proofs.log" 2>&1 \
  || { tail -40 "$OUT/build-sampled-proofs.log"; exit 7; }
log "a6: members ${A6_MEMBERS:-default}, k ${A6_K:-default}"
"$PY" benchmarks/one_stage/a6.py --bin "$BIN" --lean "$LEAN" --out "$W/a6" ${A6_MEMBERS:+--members "$A6_MEMBERS"} \
  ${A6_K:+--k "$A6_K"} ${A6_CHUNK:+--chunk "$A6_CHUNK"} 2>&1 | tee -a "$OUT/a6.log" | cut -c1-4000
rc=${PIPESTATUS[0]}
mkdir -p "$OUT/a6"
(cd "$W/a6" && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -path '*/archive/*' -print0 | xargs -0 -I{} cp --parents {} "$OUT/a6/")
log "a6: done, rc $rc"
exit "$rc"
