#!/usr/bin/env bash
# M1 with a unit's MUFU reads in C-Flock lookup slots (m1_unit.py --reads slots, `read_slots`), on a CPU machine, on hidden_zk's
# builds (its job.sh, sourced: C-Flock's flock-circuit and the Lean flock-verify, cached under $RS_CACHE):
#   RS_CACHE=/workspace/research/cache/pouw-hidden-zk bash benchmarks/one_stage/m1_reads.sh
#   (a job run from the source tree, 16 CPUs, 64 GB, keeping out/*)
# Each spec of RS_PROVE is built with its reads in slots, staged, proved under --zk and verified by the Lean verifier; each
# "MODE SPEC" of RS_BUILD (MODE slots or inline) is only built, for its size (ANDs, nonzeros, k_log). Records: $OUT/<tag>/m1.log.
# The router starts at E=32: M1's statement needs a row longer than one SHA-512 block (compose's pin range).
# env: RS_PROVE, RS_BUILD (';'-separated)  RS_ARGS (more m1_unit.py arguments: --stage-ands N|auto|none, auto by default)  RS_MEM_GB=64 (the job's --mem-gb)  RS_CACHE
set -uo pipefail
ROUTER='MoeRouterSigmoidBias_v1{E=%d,TOPK=%d,SCALE={"f64":"0x1.4000000000000p+1"},KERNEL="block"}'
RS_PROVE=${RS_PROVE-$(printf "$ROUTER" 64 8)}
RS_BUILD=${RS_BUILD-slots $(printf "$ROUTER" 256 8)}
export HZ_CONFIGS=m1-reads HZ_MEM_GB=${RS_MEM_GB:-64}
[ -n "${RS_CACHE:-}" ] && export HZ_CACHE=$RS_CACHE
source benchmarks/pouw/hidden_zk/job.sh
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python
setup
rc=0 i=0
m1() {   # m1 TAG ARGS...: one m1_unit run, its records in $OUT/TAG (never the circuit nor a prover's file)
  local tag=$1; shift
  mkdir -p "$OUT/$tag"
  log "m1 $tag: $*"
  ( ulimit -v $(( STAGE_GB * 1024 * 1024 )); "$PY" benchmarks/one_stage/m1_unit.py --out "$W/$tag" "$@" ) > "$OUT/$tag/m1.log" 2>&1
  local r=$?
  grep '^M1U ' "$OUT/$tag/m1.log" | cut -c1-2000 | tee -a "$OUT/job.log"
  [ $r = 0 ] || { log "m1 $tag: rc $r"; tail -30 "$OUT/$tag/m1.log" | tee -a "$OUT/job.log"; rc=1; }
  (cd "$W/$tag" && find . -type f \( -name '*.json' -o -name '*.log' \) -print0 | xargs -0 -r -I{} cp --parents {} "$OUT/$tag/")
  rm -rf "$W/$tag"
}
IFS=';' read -ra PROVE <<< "$RS_PROVE"
IFS=';' read -ra BUILD <<< "$RS_BUILD"
for s in "${PROVE[@]}"; do m1 "p$i-slots" --spec "$s" --reads slots --bin "$BIN" --lean "$LEAN" --n 1 ${RS_ARGS:-}; i=$((i + 1)); done
for b in "${BUILD[@]}"; do m1 "b$i-${b%% *}" --spec "${b#* }" --reads "${b%% *}" ${RS_ARGS:-}; i=$((i + 1)); done
log "m1-reads: done, rc $rc"
exit "$rc"
