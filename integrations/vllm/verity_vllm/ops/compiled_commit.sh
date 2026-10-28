#!/usr/bin/env bash
# Commit + overhead for an Execution=compiled workload: vLLM torch.compile + CUDA graphs (enforce_eager=false), the production default
# compiled configuration.  TAPS=kernels (default): P_compiled's VU boundaries = every non-weight buffer of every
# Inductor kernel launch, acquired through Inductor's wrapper-level hook seam (kernels untouched; captured pieces copy into slots at graph
# capture) -- verity_vllm/acquire/sources/compiled_kernel_source.py; the required manifest is P_compiled x query, built by
# verity_vllm/acquire/compiled.py from the CONTROL process's Inductor output code (so the control runs first in round 1).
# TAPS=on = the withdrawn record-op path, kept for reference.
#
# Two processes with IDENTICAL engine args (recorded in each versions.json `compilation`; compiled_merge refuses a mismatch):
#   1. control   : commit_delta --only-arm control    --compiled-taps off      (no committer, no hooks; PAIRS runs; its Inductor dir -> the manifest)
#   2. treatment : commit_delta --only-arm instrumented --compiled-taps $TAPS  (hooks in the wrappers; PAIRS instrumented runs + verdict)
#   repeated ROUNDS times (default 2) with the order alternating per round (c,t / t,c) -> 2 x PAIRS runs per arm, paired by index
# then compiled_merge.py pairs them by run index -> $D/commit/{summary.json,summary.md,runs.jsonl,verdict.json} (row_pod.sh's Commit shape).
#
# Usage: compiled_commit.sh <ROW> <ROLE> [CUDAGRAPH_MODE]          (CUDAGRAPH_MODE default PIECEWISE: attention = splitting op, runs outside the graphs
#        env: SWEEP_DIR, PAIRS (3), GPU_UTIL (0.3), HIDDEN_SO, EXTRA_ENGINE_ARGS, COMMIT_TIMEOUT (2400), FAULT (VERITY_FAULT for the treatment)
set -uo pipefail
ROW=$1; ROLE=$2; CGM=${3:-PIECEWISE}
PY=${PY:-/workspace/venv-cu129/bin/python}
export PYTHONPATH=. HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 VLLM_BATCH_INVARIANT=1 TOKENIZERS_PARALLELISM=false
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} PATH=$(dirname "$PY"):/usr/local/cuda/bin:$PATH
export NATIVE_COLLECT_BUILD=${NATIVE_COLLECT_BUILD:-/workspace/cp/nc_build}
# the tree that runs is the one this script lives in, never an environment default
TREE_SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
if [ -n "${VERITY_TREE:-}" ] && [ "$(cd "$VERITY_TREE" 2>/dev/null && pwd -P)" != "$TREE_SELF" ]; then
  echo "PRECHECK_FAIL_SOURCE_IDENTITY: compiled_commit.sh at $TREE_SELF but \$VERITY_TREE=$VERITY_TREE names another tree" >&2; exit 3
fi
export VERITY_TREE=$TREE_SELF
cd "$VERITY_TREE" || exit 1
D=${SWEEP_DIR:-/workspace/cp/sweep}/$ROW; mkdir -p "$D"
WL=workloads/$ROW.json
"${PY%/*}/verity-vllm" workload-file "$WL" > /dev/null || exit 2
# Fail closed: a bi-compiled workload must declare the compiled target (target.compilation, from workload.py --target
# <device>:piecewise) -- with no target block target_of_workload() returns None, target_mismatches() checks nothing, and the engine
# (built compiled by the explicit engine args below) would run UNDECLARED: a record without the TargetProfile of record.
DECL=$(python3 -c "import json;print((json.load(open('$WL')).get('target') or {}).get('compilation') or '')" 2>/dev/null)
if [ -z "$DECL" ]; then
  echo "compiled_commit $ROW: REFUSED -- workload $WL declares no target.compilation (the release carries an eager workload for a bi-compiled one)" >&2
  exit 2
fi
B=$(python3 -c "import json;print(json.load(open('$WL')).get('batch') or json.load(open('$WL')).get('max_num_seqs') or 1)" 2>/dev/null || echo 1)
[[ "$ROW" =~ __b([0-9]+)__ ]] && B=${BASH_REMATCH[1]}
PAIRS=${PAIRS:-3}; GPU_UTIL=${GPU_UTIL:-0.3}; TO=${COMMIT_TIMEOUT:-2400}; TAPS=${TAPS:-kernels}
HIDDEN_SO=${HIDDEN_SO:-/workspace/cp/fa2/build/matReq/verity_fa2_matReq.so}
log() { echo "[$(date -u +%FT%TZ)] compiled_commit $ROW: $*" | tee -a "$D/stages.txt"; }
COMMON=(--case "$ROLE" --workload "$WL" --committer native_collect_v2b --hidden-so "$HIDDEN_SO" --openings 64 --pairs "$PAIRS" --warmup 2
        --max-num-seqs "$B" --engine-arg "gpu_memory_utilization=$GPU_UTIL" --engine-arg enforce_eager=false
        # the compilation_config of record is the WORKLOAD's engine.args (workload.py --target: cudagraph_mode +
        # inductor_compile_config {emulate_precision_casts: true} from TargetProfile.engine_args()); an explicit 3rd argument overrides the
        # cudagraph mode only and keeps the declared emulation (compiled_merge refuses arms whose inductor_compile_config differ);
        # the compilation_config engine arg itself is passed per process (run_arm: + the per-process cache_dir)
        --window-mb "${WINDOW_MB:-256}" --window-slots "${WINDOW_SLOTS:-8}" --retain "${RETAIN:-host}" --label "$ROW")
[ "${OPEN_AFTER_RELEASE:-1}" = "0" ] || COMMON+=(--open-after-release)
for ea in ${EXTRA_ENGINE_ARGS:-}; do COMMON+=(--engine-arg "$ea"); done
for ca in ${EXTRA_COMMIT_ARGS:-}; do COMMON+=("$ca"); done
REQUIRED_MANIFEST=${REQUIRED_MANIFEST:-}
if [ -z "$REQUIRED_MANIFEST" ] && [ "$TAPS" != kernels ] && [ -f "$D/build_request/instances.json.gz" ]; then
  if "${PY%/*}/verity-vllm" manifest build --program "$D/build_request" --workload "$WL" --no-norm-scales --no-guarded-max --no-router-softmax \
       --no-vocab-range --out "$D/manifest.json" > "$D/manifest.log" 2>&1; then
    REQUIRED_MANIFEST="$D/manifest.json"; log "required-value manifest: $(tail -n 1 "$D/manifest.log")"
  fi
fi
[ -n "$REQUIRED_MANIFEST" ] && [ -f "$REQUIRED_MANIFEST" ] && COMMON+=(--required-manifest "$REQUIRED_MANIFEST")
# Weights of record: the served parameters are authenticated against the pinned checkpoint shards (an engine-free
# record, verity_vllm/check/weights_of_record.py, built from the model's Program weights struct x the pinned safetensors).  P_compiled
# reads the SAME served parameters as the eager Program (the compiled GEMM/norm/RoPE kernels take the module weights as arguments -- their
# C2 relations replay bit-for-bit with them), so the model's weights-of-record record applies to the compiled workloads verbatim.
WEIGHTS_OF_RECORD=${WEIGHTS_OF_RECORD:-}
[ -n "$WEIGHTS_OF_RECORD" ] && [ -f "$WEIGHTS_OF_RECORD" ] && COMMON+=(--weights-of-record "$WEIGHTS_OF_RECORD")
TRT_EXTRA=()
# P_compiled manifest (TAPS=kernels): from the control process's Inductor output code (wrapper call() launch programs) x the workload's steps
build_compiled_manifest() {
  local cdir=$1 out=${2:-$D/compiled_manifest.json}
  local model pieces steps ptoks kset
  model=$(python3 -c "import json;print(json.load(open('$WL'))['model'])")
  steps=$(python3 -c "import json;print(json.load(open('$WL'))['expected_engine_steps'])")
  ptoks=$(python3 -c "import json;w=json.load(open('$WL'));print(len(w['requests'][0]['prompt_token_ids']))")
  # pieces = num_hidden_layers + 1 (vLLM splits at every attention op); the model's config.json from the offline HF cache (no hub lookup)
  pieces=${PIECES:-$(python3 -c "
import json,glob,sys; m='$model'.replace('/','--')
fs=sorted(glob.glob('$HF_HOME/hub/models--'+m+'/snapshots/*/config.json')) or sorted(glob.glob('$model/config.json'))
print(json.load(open(fs[-1]))['num_hidden_layers']+1) if fs else sys.exit(1)" 2>/dev/null)}
  kset=$(python3 -c "import json;print(json.load(open('$cdir/inductor_kernels.json'))['kernel_set_sha256'])" 2>/dev/null)
  # --workload: the request schedule (B>=2: r<i> arrive_step / prompt / max_tokens -> one identity per request per launch per role)
  local sg; sg=$(ls "$cdir"/vllm_compile_cache/rank_0_0/*/computation_graph.py 2>/dev/null | head -n 1)
  [ -n "$sg" ] || log "WARNING: no split graph (computation_graph.py) under $cdir/vllm_compile_cache -- the manifest's wiring will be undeclared"
  if "${PY%/*}/verity-vllm" compiled build --inductor-dir "$cdir/inductor_cache" --pieces "$pieces" --workload "$WL" --steps "$steps" \
       ${kset:+--kernel-set "$kset"} ${sg:+--split-graph "$sg"} ${WEIGHTS_PROGRAM_DIGEST:+--weights-program-digest "$WEIGHTS_PROGRAM_DIGEST"} \
       --out "$out" > "${out%.json}.log" 2>&1; then
    log "P_compiled manifest from $cdir -> $out: $(tail -n 1 "${out%.json}.log")"
    if [ "$out" = "$D/compiled_manifest.json" ]; then
      # (require_build_program_digest) the Program digest of record for a compiled workload is P_compiled derived from
      # the CONTROL arm (production compiled, uninstrumented) -- the treatment reads it, and compiled_merge refuses the workload unless the
      # treatment's OWN derivation agrees.  There is no separate Build step for a compiled workload: the control's compile is the Build.
      local pdig; pdig=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['program_digest'])" "$out")
      TRT_EXTRA=(--required-manifest "$D/compiled_manifest.json" --program-digest "$pdig")
      log "Program digest of record (P_compiled from the control arm) $pdig"
    fi
  else
    log "P_compiled manifest FAILED ($cdir): $(tail -n 3 "${out%.json}.log" | tr '\n' ' ')"
  fi
}
[ "$TAPS" = kernels ] && [ -f "$D/compiled_manifest.json" ] && [[ ",${ARMS:-control,treatment}," != *,control,* ]] && \
  TRT_EXTRA=(--required-manifest "$D/compiled_manifest.json" --program-digest "$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['program_digest'])" "$D/compiled_manifest.json")")
# The eager Program of record whose served-parameter fields the compiled kernels read = the workload's bi-eager counterpart's Build
# digest (out/gen/sweep/rows/<row with bi-eager>.json program_digest; the digest is per model x dtype x target, so any bi-eager workload of
# the same model/target names it).  Named in the compiled manifest as components.weights_struct; the weights-of-record record must be that Program's.
WEIGHTS_PROGRAM_DIGEST=${WEIGHTS_PROGRAM_DIGEST:-}
if [ -z "$WEIGHTS_PROGRAM_DIGEST" ] && [ "$TAPS" = kernels ]; then
  # sources in order: the workload dir's eager Build summary (row_pod's build step for this workload), then the sweep's `rows/`
  # table (a >=16-hex prefix; the weights pin accepts a prefix of that length -- weights_of_record.check)
  for cand in "$D/build_summary.json" "out/gen/sweep/rows/${ROW/bi-compiled/bi-eager}.json" out/gen/sweep/rows/"${ROW%%__tp*}"__tp1__b1__*__greedy__bi-eager.json; do
    [ -f "$cand" ] || continue
    WEIGHTS_PROGRAM_DIGEST=$($PY -c "
import json,re,sys; d=json.load(open(sys.argv[1])); v=str(d.get('program_digest') or (d.get('build') or {}).get('program_digest') or '')
m=re.match(r'[0-9a-f]{16,}', v); print(m.group(0) if m else '')" "$cand" 2>/dev/null)
    [ -n "$WEIGHTS_PROGRAM_DIGEST" ] && { log "eager Program of record (weights struct) $WEIGHTS_PROGRAM_DIGEST from $cand"; break; }
  done
fi

ARMS=${ARMS:-control,treatment}          # rerun only one arm (e.g. ARMS=control after a shared-GPU memory refusal) and re-merge
rm -rf "$D/commit"
# run_arm <arm> <taps> <outdir> <log>: one commit_delta process; on a shared pod vLLM's KV budget can transiently vanish when another
# process grows between vLLM's baseline snapshot and its profile run ("No available memory for the cache blocks") -> retry up to 3x
run_arm() {
  local arm=$1 taps=$2 od=$3 lf=$4 rc=1 attempt
  for attempt in 1 2 3; do
    rm -rf "$od"
    ( echo "\$ ${PY%/*}/verity-vllm commit ${COMMON[*]} --only-arm $arm --compiled-taps $taps --out $od   (attempt $attempt)"
      # per-process Inductor cache dir + vLLM compile cache OFF in BOTH arms: the Inductor output code of every compiled piece is digested
      # from that dir into versions.json compilation.generated_kernels; a shared or vLLM-cached dir would hide it
      local -a extra=(); [ "$arm" = instrumented ] && extra=("${TRT_EXTRA[@]}")
      # + the vLLM compile cache dir per process (compilation_config.cache_dir): vLLM dumps the split graph (rank_0_0/backbone/computation_graph.py
      # = the piece wiring the manifest's dataflow needs) even with the compile cache disabled
      VERITY_FAULT=${FAULT:-} TORCHINDUCTOR_CACHE_DIR="$od/inductor_cache" VERITY_INDUCTOR_DIR_PER_PROCESS=1 VLLM_DISABLE_COMPILE_CACHE=1 \
        timeout "$TO" "${PY%/*}/verity-vllm" commit "${COMMON[@]}" --only-arm "$arm" --compiled-taps "$taps" "${extra[@]}" \
          --engine-arg "compilation_config={\"cudagraph_mode\":\"$CGM\",\"inductor_compile_config\":{\"emulate_precision_casts\":${EMULATE:-false}},\"cache_dir\":\"$od/vllm_compile_cache\"}" --out "$od" ) > "$lf" 2>&1; rc=$?
    if [ $rc -ne 0 ] && grep -q "No available memory for the cache blocks\|CUDA out of memory\|Error in memory profiling\|Free memory on device .* is less than desired" "$lf"; then
      log "$arm attempt $attempt: engine could not get its memory budget (shared GPU); retry in 90 s"; sleep 90; continue
    fi
    break
  done
  return $rc
}
rcc=0; rct=0
# ROUNDS (default 2): alternate the two processes -- round 1 control then treatment, round 2 treatment then control, ... -- so neither
# arm always runs first (interleaved pairs at process granularity: the record ops live in the treatment process's graphs,
# so one process cannot host both arms). Each process runs PAIRS runs after its own warm-up; runs are paired by global index.
ROUNDS=${ROUNDS:-2}
CTRL_DIRS=(); TRT_DIRS=()
for r in $(seq 1 "$ROUNDS"); do
  if [ $((r % 2)) -eq 1 ]; then order="control treatment"; else order="treatment control"; fi
  for arm in $order; do
    if [ "$arm" = control ] && [[ ",$ARMS," == *,control,* ]]; then
      log "round $r control process (cudagraph_mode=$CGM, pairs=$PAIRS, gpu_util=$GPU_UTIL)"
      t0=$(date +%s)
      FAULT= run_arm control off "$D/commit_control_r$r" "$D/commit_control_r$r.log"; rc=$?; [ $rc -ne 0 ] && rcc=$rc
      log "round $r control rc=$rc wall=$(( $(date +%s) - t0 ))s"
      CTRL_DIRS+=("$D/commit_control_r$r")
      [ "$TAPS" = kernels ] && [ ${#TRT_EXTRA[@]} -eq 0 ] && [ $rc -eq 0 ] && build_compiled_manifest "$D/commit_control_r$r"
    elif [ "$arm" = treatment ] && [[ ",$ARMS," == *,treatment,* ]]; then
      log "round $r treatment process (compiled taps $TAPS${FAULT:+, VERITY_FAULT=$FAULT}; ${TRT_EXTRA[*]:-no manifest})"
      t1=$(date +%s)
      run_arm instrumented "$TAPS" "$D/commit_treatment_r$r" "$D/commit_treatment_r$r.log"; rc=$?; [ $rc -ne 0 ] && rct=$rc
      log "round $r treatment rc=$rc wall=$(( $(date +%s) - t1 ))s"
      TRT_DIRS+=("$D/commit_treatment_r$r")
      # the same derivation from the TREATMENT's Inductor output code: compiled_merge refuses the workload unless the two
      # program_digests (kernel set x launch table x dataflow) are equal -- the Program of record is the executed implementation of both arms
      [ "$TAPS" = kernels ] && [ ! -f "$D/compiled_manifest_treatment.json" ] && [ $rc -eq 0 ] && build_compiled_manifest "$D/commit_treatment_r$r" "$D/compiled_manifest_treatment.json"
    fi
  done
done
[ ${#CTRL_DIRS[@]} -eq 0 ] && CTRL_DIRS=($(ls -d "$D"/commit_control_r[0-9]* | grep -v '\.log$'))      # dirs only (not the .log files)
[ ${#TRT_DIRS[@]} -eq 0 ] && TRT_DIRS=($(ls -d "$D"/commit_treatment_r[0-9]* | grep -v '\.log$'))
MERGE_EXTRA=()
if [ "$TAPS" = kernels ]; then
  [ -f "$D/compiled_manifest_treatment.json" ] || build_compiled_manifest "${TRT_DIRS[0]}" "$D/compiled_manifest_treatment.json"
  MERGE_EXTRA=(--manifest-control "$D/compiled_manifest.json" --manifest-treatment "$D/compiled_manifest_treatment.json")
fi
"${PY%/*}/verity-vllm" compiled-merge $(printf -- "--control %s " "${CTRL_DIRS[@]}") $(printf -- "--treatment %s " "${TRT_DIRS[@]}") "${MERGE_EXTRA[@]}" --out "$D/commit" > "$D/commit_merge.log" 2>&1; rcm=$?
cat "$D/commit_merge.log"
cat "$D"/commit_control_r*.log "$D"/commit_treatment_r*.log > "$D/commit.log"
# the manifest the treatment USED (control-derived) vs the same derivation from the treatment's own output code
if [ "$TAPS" = kernels ] && [ -f "$D/compiled_manifest.json" ] && [ -f "$D/compiled_manifest_treatment.json" ]; then
  "${PY%/*}/verity-vllm" compiled verify --manifest "$D/compiled_manifest.json" --rebuilt "$D/compiled_manifest_treatment.json" \
      --out "$D/commit/manifest_verify.json" > "$D/manifest_verify.log" 2>&1; rcv=$?
  log "manifest verify (the control's vs the treatment's own) rc=$rcv: $(tail -n 1 "$D/manifest_verify.log")"
  [ $rcv -ne 0 ] && [ $rcm -eq 0 ] && rcm=$rcv
fi
# the row_pod-shaped view of the record (what verity_vllm.check.verdict from-record reads): commit/runs.jsonl = the instrumented arm's records (binding,
# value_correspondence components incl. sampled_replay / boundary_linkage / weights_pin) -- compiled_merge already wrote commit/runs.jsonl
# + commit/verdict.json; the release line for stages.txt is the tree's commit
[ -n "${VERITY_TREE_COMMIT:-}" ] && log "release ${VERITY_RELEASE_TAG:-UNRELEASED lane/r15-compiled} commit=$VERITY_TREE_COMMIT"
log "merge rc=$rcm -> $D/commit (control rc=$rcc treatment rc=$rct)"
exit $rcm
