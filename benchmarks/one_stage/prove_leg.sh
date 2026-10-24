#!/usr/bin/env bash
# The proof leg (prove_leg.py) on a CPU machine, on hidden_zk's builds (its job.sh, sourced: C-Flock's flock-circuit and the
# Lean flock-verify, cached under $LEG_CACHE) with the firewall's Lean frame and the sampled-proofs verifier built in the
# tree. From a clone:
#   research run --on auto --source . --as proofs --project verity --cwd clone --cpus 8 --mem-gb 32 \
#     --env LEG_CACHE=/workspace/research/cache/pouw-hidden-zk -- bash benchmarks/one_stage/prove_leg.sh
# env: LEG_MEM_GB (the job's --mem-gb, 32 by default)  LEG_CACHE (the build cache)  LEG_N (units a member, 4)
#      LEG_KS (the stand-in's K values, "128,256")  LEG_NEGATIVES=0 (honest only)  LEG_FIREWALL=0 (prove without it)
#      LEG_REGISTERED=1 (the registered-weights stand-in at the first K, verified with --registered)
# Kept in $OUT/leg: the leg's records (leg.json, audit.json, registration, draw), the public files and the sessions; never a
# prover's file (inst-*) nor a circuit (its pin is kept), nor a session's archive. $OUT/summary.json: the verdicts.
set -uo pipefail
export HZ_CONFIGS=prove-leg HZ_MEM_GB=${LEG_MEM_GB:-32}
if [ -n "${LEG_CACHE:-}" ] && mkdir -p "$LEG_CACHE" 2>/dev/null && [ -w "$LEG_CACHE" ]; then export HZ_CACHE=$LEG_CACHE; fi
source benchmarks/pouw/hidden_zk/job.sh
export PATH=$HOME/.elan/bin:$PATH
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python
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
log "leg: stand-in, n ${LEG_N:-4}, ks ${LEG_KS:-128,256}, firewall ${FW:-off}, negatives ${LEG_NEGATIVES:-1}"
NEG=--negatives; [ "${LEG_NEGATIVES:-1}" = 0 ] && NEG=
REG=; [ "${LEG_REGISTERED:-0}" = 1 ] && REG=--registered
"$PY" benchmarks/one_stage/prove_leg.py standin --bin "$BIN" --lean "$LEAN" ${FW:+--firewall "$FW"} --out "$W/leg" \
  --n "${LEG_N:-4}" --ks "${LEG_KS:-128,256}" $NEG $REG 2>&1 | tee -a "$OUT/leg.log" | cut -c1-4000
rc=${PIPESTATUS[0]}
mkdir -p "$OUT/leg"
(cd "$W/leg" && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -path '*/archive/*' -print0 | xargs -0 -r -I{} cp --parents {} "$OUT/leg/")
[ -f "$W/leg/summary.json" ] && cp "$W/leg/summary.json" "$OUT/summary.json"
log "leg: done, rc $rc"
exit "$rc"
