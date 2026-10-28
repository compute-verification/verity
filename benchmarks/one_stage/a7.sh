#!/usr/bin/env bash
# One-stage audit, run A7, on a CPU machine: the requirement's plan and memory-map units under the stratified law, wired to
# the remote bandwidth audit's ledger (a7.py), whose check 3 on each drawn receipt is proved by the check-3 units
# (`poub.opening`), on hidden_zk's builds (its job.sh, sourced: C-Flock's flock-circuit and the Lean
# flock-verify, cached under $A7_CACHE). From a clone, since the verifier of record builds the sampled-proofs Lean in the
# tree:
#   research run --on M --source . --pool proofs --as LANE --project verity --cwd clone --env A7_MEM_GB=96 -- \
#     bash benchmarks/one_stage/a7.sh
# (no --mem-gb: the pool's slot holds the memory, and --mem-gb beside it waits outside the slot line for free memory)
# env: A7_MEM_GB (what job.sh's stages may take, within the slot's)  A7_K (a7.py's --k, every unit by default)  A7_STEPS (its --steps)
#      A7_SAMPLES (its --samples, the remote session's draws a window)  A7_CACHE (the build cache)
# Kept in $OUT/a7: a7's records and logs, the public files, the canonical commitments' b || c, the memory map and the sessions
# (their proofs and records); never a prover's file (inst-*) nor a circuit (its pin is kept), nor a session's archive.
set -uo pipefail
export HZ_CONFIGS=a7 HZ_MEM_GB=${A7_MEM_GB:?}
[ -n "${A7_CACHE:-}" ] && export HZ_CACHE=$A7_CACHE
source benchmarks/pouw/hidden_zk/job.sh
export PATH=$HOME/.elan/bin:$PATH
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python
setup
(cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier) > "$OUT/build-sampled-proofs.log" 2>&1 \
  || { tail -40 "$OUT/build-sampled-proofs.log"; exit 7; }
log "a7: k ${A7_K:-default}, steps ${A7_STEPS:-default}, samples ${A7_SAMPLES:-default}"
"$PY" benchmarks/one_stage/a7.py --bin "$BIN" --lean "$LEAN" --out "$W/a7" ${A7_K:+--k "$A7_K"} ${A7_STEPS:+--steps "$A7_STEPS"} \
  ${A7_SAMPLES:+--samples "$A7_SAMPLES"} 2>&1 | tee -a "$OUT/a7.log" | cut -c1-4000
rc=${PIPESTATUS[0]}
mkdir -p "$OUT/a7"
(cd "$W/a7" && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -path '*/archive/*' -print0 | xargs -0 -I{} cp --parents {} "$OUT/a7/")
log "a7: done, rc $rc"
exit "$rc"
