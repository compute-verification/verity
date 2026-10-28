#!/usr/bin/env bash
# One-stage audit, run A11, on a CPU machine: GLM-5.3 TP8 B1's served values through the one-stage audit (a11.py), or its census
# and lowering probe (a11_kinds.py), on hidden_zk's builds (its job.sh, sourced: C-Flock's flock-circuit and the Lean
# flock-verify, cached under $A11_CACHE). From a clone, since the verifier of record builds the sampled-proofs Lean in the tree:
#   A11_MODE=kinds|audit A11_MEM_GB=256 bash benchmarks/one_stage/a11.sh
#   (a job run from a clone of the tree, 32 CPUs, 256 GB, up to 240 min, with its inputs sent)
# kinds: a11_kinds.py on each sent descriptor (inputs/*.json.gz), JSON in $OUT/kinds-<name>.json.
# audit: a11.py on the sent value slices and checkpoint subset (inputs/a11-values*.tar, inputs/a11-ckpt*.tar, both from
#   a11_slice.py), or on a11.py --save-values' directory (inputs/a11-saved*.tar, which skips the fold), the probe's table
#   (inputs/kinds*.json) and the load-time roles (inputs/derive-report.json); records in $OUT/a11. With A11_REPROBE=K1,K2 the
#   table's kinds K1 and K2 are probed again first (a11_kinds.py --table, $OUT/kinds-reprobe.json) and the audit reads that.
# env: A11_MODE  A11_MEM_GB (the job's memory, GB)  A11_JOBS (a11_kinds' --jobs)  A11_CHILD_GB (its --mem-gb)  A11_ONLY (its --only)
#      A11_K (a11.py's --k)  A11_CACHE (the build cache, kept on the node)  A11_ARGS (more a11.py arguments)  A11_REPROBE
#      A11_BOOLEAN (a11_kinds' --boolean: kinds probed through their Boolean version first)
# Kept in $OUT/a11: a11's records and logs, the public files and the sessions; never a prover's file (inst-*), a circuit (its
# pin is kept) or a session's archive.
set -uo pipefail
export HZ_CONFIGS=a11 HZ_MEM_GB=${A11_MEM_GB:?}
export HZ_CACHE=${A11_CACHE:-$HOME/.cache/verity-a11}
mkdir -p "$HZ_CACHE"
source benchmarks/pouw/hidden_zk/job.sh
export PATH=$HOME/.elan/bin:$PATH
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python
IN=${RESEARCH_RUN_DIR:?}/inputs
command -v research >/dev/null && research keep "$HZ_CACHE" --owner @circuits --expires 4d --priority low --root / \
  --why "A11's builds (flock-circuit, flock-verify, the venv), keyed by their inputs" >> "$OUT/job.log" 2>&1
case ${A11_MODE:?} in
kinds)
  PY=$(cached venv numpy2.5.3-blake3-1.0.9 build_python)/bin/python
  cached flock "$(tree_key verity/ml/flock/live verity/ml/flock/*.patch)" build_circuit > "$W/flock.dir" &
  cached lean "$(tree_key verity/core/service/flock)" build_lean > "$W/lean.dir" &
  rc=0
  for d in "$IN"/*.json.gz; do
    n=$(basename "$d" .json.gz)
    log "a11 kinds: $n"
    "$PY" benchmarks/one_stage/a11_kinds.py --descriptor "$d" --out "$OUT/kinds-$n.json" --jobs "${A11_JOBS:-12}" \
      --mem-gb "${A11_CHILD_GB:-160}" ${A11_ONLY:+--only "$A11_ONLY"} ${A11_BOOLEAN:+--boolean "$A11_BOOLEAN"} 2>&1 \
      | tee -a "$OUT/kinds-$n.log" | cut -c1-2000 || rc=1
  done
  wait
  log "a11 kinds: done, builds $(cat "$W/flock.dir" "$W/lean.dir" 2>/dev/null | tr '\n' ' ')"
  exit $rc
  ;;
audit)
  setup
  (cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier) > "$OUT/build-sampled-proofs.log" 2>&1 \
    || { tail -40 "$OUT/build-sampled-proofs.log"; exit 7; }
  V=$W/values && mkdir -p "$V"
  if compgen -G "$IN/a11-saved*.tar" >/dev/null; then
    tar -xf "$(ls "$IN"/a11-saved*.tar | head -1)" -C "$V" && VARG=(--load-values "$V")
  else
    for t in "$IN"/a11-values*.tar "$IN"/a11-ckpt*.tar; do tar -xf "$t" -C "$V"; done
    VARG=(--values "$V")
  fi
  log "a11: values $(du -sh "$V" | cut -f1) (${VARG[0]}), k ${A11_K:-default}"
  KINDS=$(ls "$IN"/kinds*.json | head -1)
  if [ -n "${A11_REPROBE:-}" ]; then
    log "a11: probing $A11_REPROBE again from $(basename "$KINDS")"
    "$PY" benchmarks/one_stage/a11_kinds.py --table "$KINDS" --only "$A11_REPROBE" --out "$OUT/kinds-reprobe.json" \
      --jobs "${A11_JOBS:-4}" --mem-gb "${A11_CHILD_GB:-160}" ${A11_BOOLEAN:+--boolean "$A11_BOOLEAN"} 2>&1 \
      | tee -a "$OUT/kinds-reprobe.log" | cut -c1-2000
    KINDS=$OUT/kinds-reprobe.json
  fi
  # shellcheck disable=SC2086
  "$PY" benchmarks/one_stage/a11.py --bin "$BIN" --lean "$LEAN" "${VARG[@]}" --kinds "$KINDS" \
    --roles "$IN/derive-report.json" --out "$W/a11" ${A11_K:+--k "$A11_K"} ${A11_ARGS:-} \
    2>&1 | tee -a "$OUT/a11.log" | cut -c1-4000
  rc=${PIPESTATUS[0]}
  mkdir -p "$OUT/a11"
  (cd "$W/a11" && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -path '*/archive/*' -print0 | xargs -0 -I{} cp --parents {} "$OUT/a11/")
  log "a11: done, rc $rc"
  exit "$rc"
  ;;
esac
