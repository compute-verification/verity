#!/usr/bin/env bash
# One-stage audit, run A9, on a CPU machine: GLM-5.3 TP8 B1's routed-expert GEMMs in the static pair form (a9.py), then the
# units that fit M0 through the one-stage audit (a11.py --load-values), on hidden_zk's builds as a11.sh runs them. From a clone:
#   A9_MEM_GB=256 bash benchmarks/one_stage/a9.sh
#   (a job run from a clone of the tree, 32 CPUs, 256 GB, up to 300 min, with a11-values-r0-s01.tar, a9-experts-r0-l3.tar, derive-report.json sent)
# Inputs: a11_slice.py's value slice (inputs/a11-values*.tar), a11_slice.ckpt_experts' share (inputs/a9-experts*.tar), the
# load-time roles (inputs/derive-report.json). Records: $OUT/a9 (a9.json, the saved units) and $OUT/a11 (the audit's records).
# env: A9_MEM_GB (the job's memory, GB)  A9_ARGS (more a9.py arguments)  A11_ARGS (more a11.py arguments)  A11_CACHE (the build cache)
#      A9_SAVED (an input tar of a9.py --save-values' directory: audit those units, with no values stage and no value tars)
set -uo pipefail
export HZ_CONFIGS=a11 HZ_MEM_GB=${A9_MEM_GB:?}
export HZ_CACHE=${A11_CACHE:-$HOME/.cache/verity-a11}
mkdir -p "$HZ_CACHE"
source benchmarks/pouw/hidden_zk/job.sh
export PATH=$HOME/.elan/bin:$PATH
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python
IN=${RESEARCH_RUN_DIR:?}/inputs
command -v research >/dev/null && research keep "$HZ_CACHE" --owner @circuits --expires 4d --priority low --root / \
  --why "A11/A9's builds (flock-circuit, flock-verify, the venv), keyed by their inputs" >> "$OUT/job.log" 2>&1
setup
(cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier) > "$OUT/build-sampled-proofs.log" 2>&1 \
  || { tail -40 "$OUT/build-sampled-proofs.log"; exit 7; }
mkdir -p "$OUT/a9"
if [ -n "${A9_SAVED:-}" ]; then
  mkdir -p "$OUT/a9/saved" && tar -xf "$IN/$A9_SAVED" -C "$OUT/a9/saved" || { log "a9: can't extract $A9_SAVED"; exit 2; }
  log "a9: the saved units of $A9_SAVED, no values stage"
else
  V=$W/values && mkdir -p "$V"
  for t in "$IN"/a11-values*.tar "$IN"/a9-experts*.tar; do tar -xf "$t" -C "$V" || { log "a9: can't extract $t"; exit 2; }; done
  log "a9: values $(du -sh "$V" | cut -f1)"
  # shellcheck disable=SC2086
  "$PY" benchmarks/one_stage/a9.py --values "$V" --roles "$IN/derive-report.json" --out "$OUT/a9" --save-values "$OUT/a9/saved" \
    ${A9_ARGS:-} 2>&1 | tee -a "$OUT/a9.log" | cut -c1-4000
  rc=${PIPESTATUS[0]}
  log "a9: values rc $rc"
fi
[ -f "$OUT/a9/saved/kinds.json" ] || exit "$rc"
# shellcheck disable=SC2086
"$PY" benchmarks/one_stage/a11.py --bin "$BIN" --lean "$LEAN" --load-values "$OUT/a9/saved" --kinds "$OUT/a9/saved/kinds.json" \
  --out "$W/a11" ${A11_ARGS:-} 2>&1 | tee -a "$OUT/a11.log" | cut -c1-4000
rc=${PIPESTATUS[0]}
mkdir -p "$OUT/a11"
(cd "$W/a11" && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -path '*/archive/*' -print0 | xargs -0 -I{} cp --parents {} "$OUT/a11/")
log "a9: audit done, rc $rc"
exit "$rc"
