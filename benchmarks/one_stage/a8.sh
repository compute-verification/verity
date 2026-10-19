#!/usr/bin/env bash
# One-stage audit, run A8, on a CPU machine: GLM-5.3 TP8 B1's served values through the one-stage audit (a8.py), or its census
# and lowering probe (a8_kinds.py), on hidden_zk's builds (its job.sh, sourced: C-Flock's flock-circuit and the Lean
# flock-verify, cached under $A8_CACHE). From a clone, since the verifier of record builds the sampled-proofs Lean in the tree:
#   research run --on auto --cpus 32 --mem-gb 256 --max-min 240 --source . --cwd clone --send FILE ... \
#     --env A8_MODE=kinds|audit --env A8_MEM_GB=256 -- bash benchmarks/one_stage/a8.sh
# kinds: a8_kinds.py on each sent descriptor (inputs/*.json.gz), JSON in $OUT/kinds-<name>.json.
# audit: a8.py on the sent value slices and checkpoint subset (inputs/a8-values*.tar, inputs/a8-ckpt*.tar, both from
#   a8_slice.py), or on a8.py --save-values' directory (inputs/a8-saved*.tar, which skips the fold), the probe's table
#   (inputs/kinds*.json) and the load-time roles (inputs/derive-report.json); records in $OUT/a8. With A8_REPROBE=K1,K2 the
#   table's kinds K1 and K2 are probed again first (a8_kinds.py --table, $OUT/kinds-reprobe.json) and the audit reads that.
# env: A8_MODE  A8_MEM_GB (the job's --mem-gb)  A8_JOBS (a8_kinds' --jobs)  A8_CHILD_GB (its --mem-gb)  A8_ONLY (its --only)
#      A8_K (a8.py's --k)  A8_CACHE (the build cache, kept on the node)  A8_ARGS (more a8.py arguments)  A8_REPROBE
# Kept in $OUT/a8: a8's records and logs, the public files and the sessions; never a prover's file (inst-*), a circuit (its
# pin is kept) or a session's archive.
set -uo pipefail
export HZ_CONFIGS=a8 HZ_MEM_GB=${A8_MEM_GB:?}
export HZ_CACHE=${A8_CACHE:-$HOME/.cache/verity-a8}
mkdir -p "$HZ_CACHE"
source benchmarks/pouw/hidden_zk/job.sh
export PATH=$HOME/.elan/bin:$PATH
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python
IN=${RESEARCH_RUN_DIR:?}/inputs
command -v research >/dev/null && research keep "$HZ_CACHE" --owner @circuits --expires 4d --priority low --root / \
  --why "A8's builds (flock-circuit, flock-verify, the venv), keyed by their inputs" >> "$OUT/job.log" 2>&1
case ${A8_MODE:?} in
kinds)
  PY=$(cached venv numpy2.5.3-blake3-1.0.9 build_python)/bin/python
  cached flock "$(tree_key verity/ml/flock/live verity/ml/flock/*.patch)" build_circuit > "$W/flock.dir" &
  cached lean "$(tree_key verity/core/service/flock)" build_lean > "$W/lean.dir" &
  rc=0
  for d in "$IN"/*.json.gz; do
    n=$(basename "$d" .json.gz)
    log "a8 kinds: $n"
    "$PY" benchmarks/one_stage/a8_kinds.py --descriptor "$d" --out "$OUT/kinds-$n.json" --jobs "${A8_JOBS:-12}" \
      --mem-gb "${A8_CHILD_GB:-160}" ${A8_ONLY:+--only "$A8_ONLY"} 2>&1 | tee -a "$OUT/kinds-$n.log" | cut -c1-2000 || rc=1
  done
  wait
  log "a8 kinds: done, builds $(cat "$W/flock.dir" "$W/lean.dir" 2>/dev/null | tr '\n' ' ')"
  exit $rc
  ;;
audit)
  setup
  (cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier) > "$OUT/build-sampled-proofs.log" 2>&1 \
    || { tail -40 "$OUT/build-sampled-proofs.log"; exit 7; }
  V=$W/values && mkdir -p "$V"
  if compgen -G "$IN/a8-saved*.tar" >/dev/null; then
    tar -xf "$(ls "$IN"/a8-saved*.tar | head -1)" -C "$V" && VARG=(--load-values "$V")
  else
    for t in "$IN"/a8-values*.tar "$IN"/a8-ckpt*.tar; do tar -xf "$t" -C "$V"; done
    VARG=(--values "$V")
  fi
  log "a8: values $(du -sh "$V" | cut -f1) (${VARG[0]}), k ${A8_K:-default}"
  KINDS=$(ls "$IN"/kinds*.json | head -1)
  if [ -n "${A8_REPROBE:-}" ]; then
    log "a8: probing $A8_REPROBE again from $(basename "$KINDS")"
    "$PY" benchmarks/one_stage/a8_kinds.py --table "$KINDS" --only "$A8_REPROBE" --out "$OUT/kinds-reprobe.json" \
      --jobs "${A8_JOBS:-4}" --mem-gb "${A8_CHILD_GB:-160}" 2>&1 | tee -a "$OUT/kinds-reprobe.log" | cut -c1-2000
    KINDS=$OUT/kinds-reprobe.json
  fi
  # shellcheck disable=SC2086
  "$PY" benchmarks/one_stage/a8.py --bin "$BIN" --lean "$LEAN" "${VARG[@]}" --kinds "$KINDS" \
    --roles "$IN/derive-report.json" --out "$W/a8" ${A8_K:+--k "$A8_K"} ${A8_ARGS:-} \
    2>&1 | tee -a "$OUT/a8.log" | cut -c1-4000
  rc=${PIPESTATUS[0]}
  mkdir -p "$OUT/a8"
  (cd "$W/a8" && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -path '*/archive/*' -print0 | xargs -0 -I{} cp --parents {} "$OUT/a8/")
  log "a8: done, rc $rc"
  exit "$rc"
  ;;
esac
