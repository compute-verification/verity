#!/usr/bin/env bash
# The Commit of a served-compiled workload: vLLM at its serving defaults (the workload's execution label `compiled`: torch.compile
# with vLLM's own CUDA-graph mode, no batch invariance, no compilation_config of ours), every value of every engine step committed
# (acquire/sources/served_compiled_source.py), the served Build of record (`verity-vllm served-match`'s build.json) as the required manifest,
# and the Build's Program checked word for word on the committed values (check/replay/served_check.py) with a seed chosen after commitment.
#
# Two processes with identical engine args, each with its own Inductor and vLLM compile dirs and the compile caches off (the Build reads the
# compile artifact of the process that serves; versions.json `compilation` records each process's generated kernels, compiled_merge refuses
# arms whose kernel sets differ), each Inductor dir seeded with the Build of record's pinned autotune winners (`served-pin seed`,
# acquire/served_artifact.py: a process that timed its own could run another Program):
#   1. control   : commit --only-arm control      --compiled-taps off     (no committer, no hooks; PAIRS runs)
#   2. treatment : commit --only-arm instrumented --compiled-taps served  (PAIRS instrumented runs + the served C2 + the Build rederived
#                  from its own artifact -> build_rebuilt.json, after the warm-ups; it refuses (rc 3) before the measured runs when the
#                  artifact it loaded is not the Build of record's -> artifact_check.json)
# then compiled_merge pairs them by run index -> $D/commit/{summary.json,summary.md,runs.jsonl,verdict.json}, and `compiled verify` holds
# the Build of record (its Program and its artifact) to the rebuilt one -> $D/commit/manifest_verify.json.  Exit: the merge's rc, else the
# first nonzero rc of the control, the treatment and the verify (compiled_merge pairs the runs an arm finished, so an arm cut short by
# COMMIT_TIMEOUT, rc 124, would otherwise pass on fewer pairs).
#
# Usage: served_commit.sh CASE WORKLOAD BUILD PROGRAM_DIGEST CHECKPOINT_DIR OUT
#        PROGRAM_DIGEST: the Build's program digest of record (the Match's match.json build.program_digest; the treatment refuses a
#        BUILD that carries another)
#        env: PY (its directory has `verity-vllm`), PAIRS (2), MAX_NUM_SEQS (8), GPU_UTIL (0.3), KV_GIB (4: the KV cache, kv_cache_memory_bytes), COMMIT_TIMEOUT (2400),
#             FAULT (VERITY_FAULT for the treatment), EXTRA_COMMIT_ARGS
set -uo pipefail
CASE=$1; WL=$(realpath "$2"); BUILD=$(realpath "$3"); PDIG=$4; CKPT=$5; D=$(realpath -m "$6")
PY=${PY:-/workspace/venv-cu129/bin/python}
export HF_HOME=${HF_HOME:-/workspace/hf} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} PATH=$(dirname "$PY"):/usr/local/cuda/bin:$PATH
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1               # the integration's root: the commit's manifests/ defaults
mkdir -p "$D"
PAIRS=${PAIRS:-2}; TO=${COMMIT_TIMEOUT:-2400}
log() { echo "[$(date -u +%FT%TZ)] served_commit $CASE: $*" | tee -a "$D/stages.txt"; }
COMMON=(--case "$CASE" --workload "$WL" --pairs "$PAIRS" --warmup 2 --max-num-seqs "${MAX_NUM_SEQS:-8}"
        --engine-arg "gpu_memory_utilization=${GPU_UTIL:-0.3}" --engine-arg "kv_cache_memory_bytes=$(( ${KV_GIB:-4} << 30 ))"
        --retain host --window-mb "${WINDOW_MB:-256}" --window-slots "${WINDOW_SLOTS:-8}" --label "served-$CASE")
for ca in ${EXTRA_COMMIT_ARGS:-}; do COMMON+=("$ca"); done
TRT=(--committer native_collect_v2b --required-manifest "$BUILD" --program-digest "$PDIG" --checkpoint-dir "$CKPT" --openings 64
     --open-after-release)
rm -rf "$D/commit"
# run_arm <arm> <taps> <outdir>: one commit process, its own compile dirs exported before Python starts (torch reads the FX-cache switches
# at import)
run_arm() {
  local arm=$1 taps=$2 od=$3 rc
  local -a extra=(); [ "$arm" = instrumented ] && extra=("${TRT[@]}")
  rm -rf "$od"
  mkdir -p "$od" && verity-vllm served-pin seed --build "$BUILD" --inductor-dir "$od/inductor" > "$od.log" 2>&1 || return 1
  ( echo "\$ verity-vllm commit ${COMMON[*]} --only-arm $arm --compiled-taps $taps ${extra[*]} --out $od"
    VERITY_FAULT=$([ "$arm" = instrumented ] && echo "${FAULT:-}") TORCHINDUCTOR_CACHE_DIR="$od/inductor" VLLM_CACHE_ROOT="$od/vllm_cache" \
    VERITY_INDUCTOR_DIR_PER_PROCESS=1 VLLM_DISABLE_COMPILE_CACHE=1 TORCHINDUCTOR_FX_GRAPH_CACHE=0 TORCHINDUCTOR_AUTOGRAD_CACHE=0 \
      timeout "$TO" verity-vllm commit "${COMMON[@]}" --only-arm "$arm" --compiled-taps "$taps" "${extra[@]}" --out "$od"
  ) >> "$od.log" 2>&1; rc=$?
  return $rc
}
log "control process (pairs $PAIRS)"
run_arm control off "$D/commit_control"; rcc=$?
log "control rc=$rcc"
log "treatment process (served taps, Build of record ${PDIG:0:16}${FAULT:+, VERITY_FAULT=$FAULT})"
run_arm instrumented served "$D/commit_treatment"; rct=$?
log "treatment rc=$rct"
verity-vllm compiled-merge --control "$D/commit_control" --treatment "$D/commit_treatment" --out "$D/commit" > "$D/commit_merge.log" 2>&1; rcm=$?
cat "$D/commit_merge.log"
{ cat "$D/commit_control.log"; cat "$D/commit_treatment.log"; } > "$D/commit.log"
if [ -f "$D/commit_treatment/build_rebuilt.json" ]; then
  verity-vllm compiled verify --manifest "$BUILD" --rebuilt "$D/commit_treatment/build_rebuilt.json" \
      --out "$D/commit/manifest_verify.json" > "$D/manifest_verify.log" 2>&1; rcv=$?
else
  echo "no build_rebuilt.json from the treatment" > "$D/manifest_verify.log"; rcv=1
fi
log "manifest verify (the Build of record vs the treatment's own) rc=$rcv: $(tail -n 1 "$D/manifest_verify.log")"
for rc in $rcc $rct $rcv; do [ "$rc" -ne 0 ] && [ $rcm -eq 0 ] && rcm=$rc; done
log "merge rc=$rcm -> $D/commit (control rc=$rcc treatment rc=$rct)"
exit $rcm
