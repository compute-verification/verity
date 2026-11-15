#!/usr/bin/env bash
# The proof subprotocol (prove_subprotocol.py) on a CPU machine, on hidden_zk's builds (its job.sh, sourced: Flock's flock-circuit and the
# Lean verity-verify-proof, cached under $SUBPROTOCOL_CACHE) with the ZK server's Lean frame and the SPOT verifier built in the
# tree. From a clone:
#   SUBPROTOCOL_CACHE=/workspace/research/cache/pouw-hidden-zk bash benchmarks/one_stage/prove_subprotocol.sh
#   (a job run from a clone of the tree, 8 CPUs, 32 GB)
# env: SUBPROTOCOL_MEM_GB (the job's memory, GB, 32 by default)  SUBPROTOCOL_CACHE (the build cache)  SUBPROTOCOL_N (units a member, 4)
#      SUBPROTOCOL_KS (the stand-in's K values, "128,256")  SUBPROTOCOL_NEGATIVES=0 (honest only)  SUBPROTOCOL_ZK_SERVER=0 (prove without it)
#      SUBPROTOCOL_REGISTERED=1 (the registered-weights stand-in at the first K, verified with --registered)
#      SUBPROTOCOL_DEFS="gemm:1b:2048 rope:1b ..." (vLLM's served Definitions, prove_subprotocol.py defs, in place of the stand-in)
#      SUBPROTOCOL_CENSUS=1 (with SUBPROTOCOL_DEFS: size each shape only)  SUBPROTOCOL_LOWER_REFUSED=1 (with SUBPROTOCOL_DEFS: lower a shape the census
#      refuses, for Flock's own refusal)  SUBPROTOCOL_READS=slots|inline (with SUBPROTOCOL_DEFS: a unit's MUFU table reads in lookup
#      slots or inline; unset, its template's own, chunk_rows.reads)  FLOCK_CHALLENGER=1 (each verifier's coin trees from
#      a challenger of its own, its audit in the subprotocol's verdicts)
# Kept in $OUT/leg: the subprotocol's records (leg.json, audit.json, docket, sample), the public files and the sessions; never a
# prover's file (inst-*) nor a circuit (its pin is kept), nor a session's archive. $OUT/summary.json: the verdicts. They are
# kept however the job ends, a timeout or a cancel included.
set -uo pipefail
# a job written for prove_leg.sh (LEG_*) or before the ZK server's rename sets keys nothing reads: refused by name
for o in LEG_FIREWALL:SUBPROTOCOL_ZK_SERVER SUBPROTOCOL_FIREWALL:SUBPROTOCOL_ZK_SERVER \
    LEG_CACHE:SUBPROTOCOL_CACHE LEG_CENSUS:SUBPROTOCOL_CENSUS LEG_DEFS:SUBPROTOCOL_DEFS LEG_KS:SUBPROTOCOL_KS \
    LEG_LOWER_REFUSED:SUBPROTOCOL_LOWER_REFUSED LEG_MEM_GB:SUBPROTOCOL_MEM_GB LEG_N:SUBPROTOCOL_N \
    LEG_NEGATIVES:SUBPROTOCOL_NEGATIVES LEG_READS:SUBPROTOCOL_READS LEG_REGISTERED:SUBPROTOCOL_REGISTERED; do
  n=${o#*:} o=${o%%:*}; [ -z "${!o+x}" ] || [ -n "${!n+x}" ] \
    || { echo "prove-subprotocol: $o is set and $n is not: it is read as $n now; set $n instead" >&2; exit 2; }; done
export HZ_CONFIGS=prove-leg HZ_MEM_GB=${SUBPROTOCOL_MEM_GB:-32}
if [ -n "${SUBPROTOCOL_CACHE:-}" ] && mkdir -p "$SUBPROTOCOL_CACHE" 2>/dev/null && [ -w "$SUBPROTOCOL_CACHE" ]; then export HZ_CACHE=$SUBPROTOCOL_CACHE; fi
source benchmarks/pouw/hidden_zk/job.sh
keep_subprotocol() {
  [ -d "$W/leg" ] || return 0
  mkdir -p "$OUT/leg"
  (cd "$W/leg" && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -path '*/archive/*' -print0 | xargs -0 -r -I{} cp --parents {} "$OUT/leg/")
  [ -f "$W/leg/summary.json" ] && cp "$W/leg/summary.json" "$OUT/summary.json"
  return 0
}
# job.sh's EXIT trap removes $W: a timeout's or a cancel's TERM must reach it with the records kept first
trap 'keep_subprotocol; rm -rf "$W"' EXIT
trap 'exit 143' TERM INT HUP
export PATH=$HOME/.elan/bin:$PATH
export PYTHONPATH=$SRC:$SRC/verity/protocols/proofs/flock/prover/python:$SRC/adapters/vllm:$SRC/adapters/torch:$SRC/tools/config:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python
# the prover's worker threads overflow the default stack on wide units
export RUST_MIN_STACK=${RUST_MIN_STACK:-67108864}
setup
SPD=; [ "${FLOCK_CHALLENGER:-0}" = 1 ] && SPD=spot-difftest
(cd verity/protocols/proofs/spot/verifier/lean && lake build spot-verifier $SPD) > "$OUT/build-sampled-proofs.log" 2>&1 \
  || { tail -40 "$OUT/build-sampled-proofs.log"; exit 7; }
FW=
if [ "${SUBPROTOCOL_ZK_SERVER:-1}" = 1 ]; then
  FW=${LEAN%/*}/flock-zk-server
  (exec 8>"${LEAN%/.lake/*}.firewall.lock"; flock 8; [ -x "$FW" ] || (cd "${LEAN%/.lake/*}" && lake build flock-zk-server)) \
    > "$OUT/build-zk-server.log" 2>&1
  [ -x "$FW" ] || { tail -20 "$OUT/build-zk-server.log"; log "subprotocol: no ZK server frame"; exit 7; }
fi
NEG=--negatives; [ "${SUBPROTOCOL_NEGATIVES:-1}" = 0 ] && NEG=
if [ -n "${SUBPROTOCOL_DEFS:-}" ]; then
  SHAPES=(); for s in $SUBPROTOCOL_DEFS; do SHAPES+=(--shape "$s"); done
  CENSUS=; [ "${SUBPROTOCOL_CENSUS:-0}" = 1 ] && CENSUS=--census
  LR=; [ "${SUBPROTOCOL_LOWER_REFUSED:-0}" = 1 ] && LR=--lower-refused
  log "subprotocol: served Definitions ${SUBPROTOCOL_DEFS}, reads ${SUBPROTOCOL_READS:-chunk_rows.reads}, ZK server ${FW:-off}, negatives ${SUBPROTOCOL_NEGATIVES:-1}${CENSUS:+, census only}${LR:+, $LR}"
  "$PY" benchmarks/one_stage/prove_subprotocol.py defs --bin "$BIN" --lean "$LEAN" ${FW:+--zk-server "$FW"} --out "$W/leg" \
    "${SHAPES[@]}" --mem-gb "$HZ_MEM_GB" ${SUBPROTOCOL_READS:+--reads "$SUBPROTOCOL_READS"} $NEG $CENSUS $LR 2>&1 | tee -a "$OUT/leg.log" | cut -c1-4000
  rc=${PIPESTATUS[0]}
else
  log "subprotocol: stand-in, n ${SUBPROTOCOL_N:-4}, ks ${SUBPROTOCOL_KS:-128,256}, ZK server ${FW:-off}, negatives ${SUBPROTOCOL_NEGATIVES:-1}, challenger ${FLOCK_CHALLENGER:-0}"
  REG=; [ "${SUBPROTOCOL_REGISTERED:-0}" = 1 ] && REG=--registered
  "$PY" benchmarks/one_stage/prove_subprotocol.py standin --bin "$BIN" --lean "$LEAN" ${FW:+--zk-server "$FW"} --out "$W/leg" \
    --n "${SUBPROTOCOL_N:-4}" --ks "${SUBPROTOCOL_KS:-128,256}" $NEG $REG 2>&1 | tee -a "$OUT/leg.log" | cut -c1-4000
  rc=${PIPESTATUS[0]}
fi
log "subprotocol: done, rc $rc"
exit "$rc"
