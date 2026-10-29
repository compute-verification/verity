#!/usr/bin/env bash
# The proof leg (prove_leg.py) on a CPU machine, on hidden_zk's builds (its job.sh, sourced: Flock's flock-circuit and the
# Lean flock-verify, cached under $LEG_CACHE) with the ZK server's Lean frame and the sampled-proofs verifier built in the
# tree. From a clone:
#   research run --on auto --source . --as proofs --project verity --cwd clone --cpus 8 --mem-gb 32 \
#     --env LEG_CACHE=/workspace/research/cache/pouw-hidden-zk -- bash benchmarks/one_stage/prove_leg.sh
# env: LEG_MEM_GB (the job's --mem-gb, 32 by default)  LEG_CACHE (the build cache)  LEG_N (units a member, 4)
#      LEG_KS (the stand-in's K values, "128,256")  LEG_NEGATIVES=0 (honest only)  LEG_FIREWALL=0 (prove without it)
#      LEG_REGISTERED=1 (the registered-weights stand-in at the first K, verified with --registered)
#      LEG_DEFS="gemm:1b:2048 rope:1b ..." (vLLM's served Definitions, prove_leg.py defs, in place of the stand-in)
#      LEG_CENSUS=1 (with LEG_DEFS: size each shape only)  LEG_LOWER_REFUSED=1 (with LEG_DEFS: lower a shape the census
#      refuses, for Flock's own refusal)  LEG_READS=slots|inline (with LEG_DEFS: a unit's MUFU table reads in lookup
#      slots or inline; unset, its template's own, chunk_rows.reads)
# Kept in $OUT/leg: the leg's records (leg.json, audit.json, registration, draw), the public files and the sessions; never a
# prover's file (inst-*) nor a circuit (its pin is kept), nor a session's archive. $OUT/summary.json: the verdicts. They are
# kept however the job ends, a timeout or a cancel included.
set -uo pipefail
export HZ_CONFIGS=prove-leg HZ_MEM_GB=${LEG_MEM_GB:-32}
if [ -n "${LEG_CACHE:-}" ] && mkdir -p "$LEG_CACHE" 2>/dev/null && [ -w "$LEG_CACHE" ]; then export HZ_CACHE=$LEG_CACHE; fi
source benchmarks/pouw/hidden_zk/job.sh
keep_leg() {
  [ -d "$W/leg" ] || return 0
  mkdir -p "$OUT/leg"
  (cd "$W/leg" && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -path '*/archive/*' -print0 | xargs -0 -r -I{} cp --parents {} "$OUT/leg/")
  [ -f "$W/leg/summary.json" ] && cp "$W/leg/summary.json" "$OUT/summary.json"
  return 0
}
# job.sh's EXIT trap removes $W: a timeout's or a cancel's TERM must reach it with the records kept first
trap 'keep_leg; rm -rf "$W"' EXIT
trap 'exit 143' TERM INT HUP
export PATH=$HOME/.elan/bin:$PATH
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python
# the prover's worker threads overflow the default stack on wide units
export RUST_MIN_STACK=${RUST_MIN_STACK:-67108864}
setup
(cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier) > "$OUT/build-sampled-proofs.log" 2>&1 \
  || { tail -40 "$OUT/build-sampled-proofs.log"; exit 7; }
FW=
if [ "${LEG_FIREWALL:-1}" = 1 ]; then
  FW=${LEAN%/*}/flock-firewall
  (exec 8>"${LEAN%/.lake/*}.firewall.lock"; flock 8; [ -x "$FW" ] || (cd "${LEAN%/.lake/*}" && lake build flock-firewall)) \
    > "$OUT/build-firewall.log" 2>&1
  [ -x "$FW" ] || { tail -20 "$OUT/build-firewall.log"; log "leg: no firewall frame"; exit 7; }
fi
NEG=--negatives; [ "${LEG_NEGATIVES:-1}" = 0 ] && NEG=
if [ -n "${LEG_DEFS:-}" ]; then
  SHAPES=(); for s in $LEG_DEFS; do SHAPES+=(--shape "$s"); done
  CENSUS=; [ "${LEG_CENSUS:-0}" = 1 ] && CENSUS=--census
  LR=; [ "${LEG_LOWER_REFUSED:-0}" = 1 ] && LR=--lower-refused
  log "leg: served Definitions ${LEG_DEFS}, reads ${LEG_READS:-chunk_rows.reads}, firewall ${FW:-off}, negatives ${LEG_NEGATIVES:-1}${CENSUS:+, census only}${LR:+, $LR}"
  "$PY" benchmarks/one_stage/prove_leg.py defs --bin "$BIN" --lean "$LEAN" ${FW:+--firewall "$FW"} --out "$W/leg" \
    "${SHAPES[@]}" --mem-gb "$HZ_MEM_GB" ${LEG_READS:+--reads "$LEG_READS"} $NEG $CENSUS $LR 2>&1 | tee -a "$OUT/leg.log" | cut -c1-4000
  rc=${PIPESTATUS[0]}
else
  log "leg: stand-in, n ${LEG_N:-4}, ks ${LEG_KS:-128,256}, firewall ${FW:-off}, negatives ${LEG_NEGATIVES:-1}"
  REG=; [ "${LEG_REGISTERED:-0}" = 1 ] && REG=--registered
  "$PY" benchmarks/one_stage/prove_leg.py standin --bin "$BIN" --lean "$LEAN" ${FW:+--firewall "$FW"} --out "$W/leg" \
    --n "${LEG_N:-4}" --ks "${LEG_KS:-128,256}" $NEG $REG 2>&1 | tee -a "$OUT/leg.log" | cut -c1-4000
  rc=${PIPESTATUS[0]}
fi
log "leg: done, rc $rc"
exit "$rc"
