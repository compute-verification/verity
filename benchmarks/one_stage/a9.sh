#!/usr/bin/env bash
# One-stage audit, run A9, on a CPU machine: GLM-5.3 TP8 B1's routed-expert GEMMs in the static pair form (a9.py), then the
# units that fit M0 through the one-stage audit (a8.py --load-values), on hidden_zk's builds as a8.sh runs them. From a clone:
#   research run --queue --on NODE --cpus 32 --mem-gb 256 --max-min 300 --source . --cwd clone --project verity \
#     --send a8-values-r0-s01.tar --send a9-experts-r0-l3.tar --send derive-report.json --env A9_MEM_GB=256 \
#     -- bash benchmarks/one_stage/a9.sh
# Inputs: a8_slice.py's value slice (inputs/a8-values*.tar), a8_slice.ckpt_experts' share (inputs/a9-experts*.tar), the
# load-time roles (inputs/derive-report.json). Records: $OUT/a9 (a9.json, the saved units) and $OUT/a8 (the audit's records).
# env: A9_MEM_GB (the job's --mem-gb)  A9_ARGS (more a9.py arguments)  A8_ARGS (more a8.py arguments)  A8_CACHE (the build cache)
set -uo pipefail
export HZ_CONFIGS=a8 HZ_MEM_GB=${A9_MEM_GB:?}
export HZ_CACHE=${A8_CACHE:-$HOME/.cache/verity-a8}
mkdir -p "$HZ_CACHE"
source benchmarks/pouw/hidden_zk/job.sh
export PATH=$HOME/.elan/bin:$PATH
export PYTHONPATH=$SRC:$SRC/verity/ml/flock/python:$SRC/integrations/vllm:$SRC/tools/verity/circuit_check/src:$SRC/benchmarks/numerical/python
IN=${RESEARCH_RUN_DIR:?}/inputs
command -v research >/dev/null && research keep "$HZ_CACHE" --owner @circuits --expires 4d --priority low --root / \
  --why "A8/A9's builds (flock-circuit, flock-verify, the venv), keyed by their inputs" >> "$OUT/job.log" 2>&1
setup
(cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier) > "$OUT/build-sampled-proofs.log" 2>&1 \
  || { tail -40 "$OUT/build-sampled-proofs.log"; exit 7; }
V=$W/values && mkdir -p "$V"
for t in "$IN"/a8-values*.tar "$IN"/a9-experts*.tar; do tar -xf "$t" -C "$V" || { log "a9: can't extract $t"; exit 2; }; done
log "a9: values $(du -sh "$V" | cut -f1)"
mkdir -p "$OUT/a9"
# shellcheck disable=SC2086
"$PY" benchmarks/one_stage/a9.py --values "$V" --roles "$IN/derive-report.json" --out "$OUT/a9" --save-values "$OUT/a9/saved" \
  ${A9_ARGS:-} 2>&1 | tee -a "$OUT/a9.log" | cut -c1-4000
rc=${PIPESTATUS[0]}
log "a9: values rc $rc"
[ -f "$OUT/a9/saved/kinds.json" ] || exit "$rc"
# shellcheck disable=SC2086
"$PY" benchmarks/one_stage/a8.py --bin "$BIN" --lean "$LEAN" --load-values "$OUT/a9/saved" --kinds "$OUT/a9/saved/kinds.json" \
  --out "$W/a8" ${A8_ARGS:-} 2>&1 | tee -a "$OUT/a8.log" | cut -c1-4000
rc=${PIPESTATUS[0]}
mkdir -p "$OUT/a8"
(cd "$W/a8" && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -path '*/archive/*' -print0 | xargs -0 -I{} cp --parents {} "$OUT/a8/")
log "a9: audit done, rc $rc"
exit "$rc"
