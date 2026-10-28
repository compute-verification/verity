#!/usr/bin/env bash
# The fixture of C-Flock's forged-proof negatives (forged_proofs.py), on a CPU machine, on hidden_zk's builds (its job.sh,
# sourced: C-Flock's flock-circuit and the Lean flock-verify, cached under $FP_CACHE). From a clone:
#   FP_CACHE=/workspace/research/cache/pouw-hidden-zk bash benchmarks/one_stage/forged_proofs.sh
#   (a job run from a clone of the tree, 8 CPUs, 32 GB)
# Kept in $OUT: fixture/ (the tree to put as fixture/v1), forged-proofs.log and work/'s logs and records (never an inst file
# nor a circuit).
set -uo pipefail
export HZ_CONFIGS=forged-proofs HZ_MEM_GB=${FP_MEM_GB:-32}
[ -n "${FP_CACHE:-}" ] && export HZ_CACHE=$FP_CACHE
source benchmarks/pouw/hidden_zk/job.sh
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python
setup
"$PY" benchmarks/one_stage/forged_proofs.py --bin "$BIN" --lean "$LEAN" --work "$W/forged" --fixture "$OUT/fixture" 2>&1 \
  | tee -a "$OUT/forged-proofs.log" | cut -c1-4000
rc=${PIPESTATUS[0]}
mkdir -p "$OUT/work"
(cd "$W/forged" && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -path '*/archive/*' -print0 | xargs -0 -I{} cp --parents {} "$OUT/work/")
log "forged-proofs: done, rc $rc"
exit "$rc"
