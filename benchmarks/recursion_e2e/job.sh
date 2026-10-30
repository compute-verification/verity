#!/usr/bin/env bash
# The recursion bench's job (README.md): verity/ml/flock/pod/85-rec-reprice.sh's steps in order, as one recorded run, each
# invocation under its own RESEARCH_RUN_DIR ($OUT/steps/<step>/r<i>, so every record 85 writes is kept) and /usr/bin/time -v,
# every CPU step REPEAT times and every GPU step once with its runs; then the result (run.py result) in
# $OUT/recursion_e2e.json and $OUT/table.md. Run from the verity tree; `verity bench recursion_e2e run` prints and launches
# it, a firewalled run (85's CPU prover, GPU=0) as a plain run on a CPU node:
#   research run --on CPU_NODE --project verity --source . --cwd source --mem-gb 128 --timeout 327m -- \
#     taskset -c 0-31 bash benchmarks/recursion_e2e/job.sh K=4096 N=2 GPU=0 FLOCK_WORK=/workspace/cache/rec-bench
# and a GPU one as a GPU host job on a GPU node's queue (85 leases a GPU around each prove itself).
# env: K=4096 N=2 (85's inner statement: N instances of GemmCoordinate_v2{K})
#      STEPS=build,inner,icheck,vstage,vcap,oprove,oreplay,lean,forge-after,negative  REPEAT=3 (each CPU step; REPEAT_<STEP>=n one step's)
#      WARM=1 RUNS=3 (the inner's runs)  OPROVE_WARM=0 OPROVE_RUNS=1 (each V* statement's releases)
#      OPROVE_JOBS=1 (V* statements proved at once; 85's own default is 4)
#      MODES=ZK server (oprove's prover sides; the last one's releases are what oreplay and lean read)
#      GPU_MAX_MIN=75 (each prove's lease, minutes)  LEAN_JOBS=4 (Lean verifies this many V* releases at once)
#      OPROVE_GPU=0 under MODES=ZK server, else GPU (V*'s prover: 85's CPU prover, or the GPU one under a lease)
#      NEG=challenge,query (negative: the inner coins of rep 0 flipped, each verified under every V* statement; empty: none)
#      FORGE= (85's forgeries, staged by vstage and proved by oprove, comma separated: message,sum,comb,top,top-salt,coef,dirs;
#      empty: none)  forge-after: coef, dirs and message made on the honest L0 and alg-p0 releases after proving
#      (forge_after.py), each put to Lean beside its honest control
#      TAG=rb-<run id> (85's data/$TAG, removed at the end unless KEEP_DATA=1; a later job's STEPS=oprove,... takes it with
#      TAG=... KEEP_DATA=1)  FLOCK_WORK=/workspace/jobs/rec-bench
#      and any of 85's own (LEVELS, GPU, PCPUS, GPU_PREEMPTIBLE, ...), passed through
#      CUDA_TK, WARM_FROM: as 85 takes them (default: the rec-reprice unpack and cargo tree where present, read only)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
REPO=$(pwd); O=${OUT:-${RESEARCH_RUN_DIR:-/tmp/rec-bench-$$}/out}; mkdir -p "$O"
# settings.py: an unknown REPEAT_* or OPROVE_*, or a value that doesn't parse, is refused; the value settings given are recorded
SETTINGS=$(python3 benchmarks/recursion_e2e/settings.py record "$O/settings.json") || exit 2
# 85's leases are preemptible unless GPU_PREEMPTIBLE is empty: on node 1 a waiter stops one after about 2 min, a lost
# measurement (85 retries the statement afresh, at most GPU_TRIES times), so the bench's are not. A firewalled V* release
# at K = 4096 takes minutes (L3: 519 s), past 85's default lease of 10 min for the larger statements (run.py's LEASE_MIN)
export GPU_PREEMPTIBLE=${GPU_PREEMPTIBLE-} GPU_MAX_MIN=${GPU_MAX_MIN:-75}
export K=${K:-4096} N=${N:-2}; export TAG=${TAG:-rb-${RESEARCH_RUN_ID:-$$}} WARM=${WARM:-1} RUNS=${RUNS:-3}
export FLOCK_WORK=${FLOCK_WORK:-/workspace/jobs/rec-bench}; mkdir -p "$FLOCK_WORK" || { echo "cannot make $FLOCK_WORK" >&2; exit 1; }
STEPS=${STEPS:-build,inner,icheck,vstage,vcap,oprove,oreplay,lean,negative}; REPEAT=${REPEAT:-3}; FORGE=${FORGE:-}
MODES=${MODES:-firewall}; NEG=${NEG-challenge,query}; NEG=${NEG//,/ }
# OPROVE_GPU=0 (the default under the ZK server): V*'s statements proved by 85's CPU prover, no lease. A lease pool stops a
# lease whose GPU computes nothing for about 10 min, and a firewalled release's GPU waits on the Lean session loop for minutes
[[ ,$MODES, == *,firewall,* ]] && OPROVE_GPU=${OPROVE_GPU:-0} || OPROVE_GPU=${OPROVE_GPU:-${GPU:-1}}
# V*'s statements one at a time, as the ZK server proves outer proofs: 85 proves 4 at once by default
export OPROVE_JOBS=${OPROVE_JOBS:-1}
PYTHONPATH=$REPO/tools/research/src python3 -m research keep "$FLOCK_WORK" --owner @benchmarks --expires 7d --priority low \
  --regenerate "bash benchmarks/recursion_e2e/job.sh K=$K N=$N" \
  --why "the recursion bench's builds (and a KEEP_DATA=1 run's data/TAG): a cache, rebuilt by the next run" > "$O/keep.json" 2>&1 \
  || echo "no retention record on $FLOCK_WORK: $(tail -1 "$O/keep.json")" >&2
# 85 moves HOME where it can't write; its Lean builds (flock-verify, flock-firewall) must find the same elan
[ -w "$HOME" ] || { export HOME=$FLOCK_WORK/home; mkdir -p "$HOME"; }
RR=/workspace/jobs/rec-reprice
[ -z "${CUDA_TK:-}" ] && [ -x $RR/cuda-13.3.1/bin/nvcc ] && export CUDA_TK=$RR/cuda-13.3.1
[ -z "${WARM_FROM:-}" ] && [ -d $RR/flock-gpu-link/flock ] && export WARM_FROM=$RR/flock-gpu-link
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$O/job.log"; }
REC=verity/ml/flock/pod/85-rec-reprice.sh
L=$FLOCK_WORK/cache
LSHA=$(cd verity/core/service/flock && find . -name .lake -prune -o -type f -print | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)
{ echo "commit=${RESEARCH_SOURCE_SHA:-$(git rev-parse HEAD 2>/dev/null)} K=$K N=$N TAG=$TAG WARM=$WARM RUNS=$RUNS REPEAT=$REPEAT settings=$SETTINGS"
  echo "OPROVE_WARM=${OPROVE_WARM:-0} OPROVE_RUNS=${OPROVE_RUNS:-1} GPU_MAX_MIN=$GPU_MAX_MIN GPU=${GPU:-1} OPROVE_GPU=$OPROVE_GPU OPROVE_JOBS=$OPROVE_JOBS" \
    $(for v in $(compgen -v REPEAT_); do echo "$v=${!v}"; done)
  echo "steps=$STEPS modes=$MODES lean_jobs=${LEAN_JOBS:-4} lean_sources=$LSHA neg=${NEG// /+} forge=${FORGE//,/+}"
  echo "cuda_tk=${CUDA_TK:-} warm_from=${WARM_FROM:-}"; nproc; nvidia-smi -L 2>/dev/null; } > "$O/params.txt"

# timed NAME I CMD...: CMD under $O/steps/NAME/r<I> (its RESEARCH_RUN_DIR) and /usr/bin/time -v; its exit status
timed() {
  local name=$1 i=$2; shift 2
  local d=$O/steps/$name/r$i
  mkdir -p "$d"
  log "$name r$i: start"
  printf '%s\t%s\t%s\n' "$name" "$i" "$(date +%s.%N)" >> "$O/phases.tsv"
  RESEARCH_RUN_DIR="$d" /usr/bin/time -v -o "$d/time-step.txt" "$@" > "$d/step.log" 2>&1
  local rc=$?
  echo $rc > "$d/rc"
  printf '%s\t%s\t%s\t%s\n' "$name" "$i" "$(date +%s.%N)" "$rc" >> "$O/phases.tsv"
  log "$name r$i: exit $rc"
  return $rc
}
# step NAME I ENV...: one invocation of 85 (STEP=NAME, unless ENV sets it, and ENV)
step() { local name=$1 i=$2; shift 2; timed "$name" "$i" env STEP=$name "$@" bash $REC; }

result() {
  python3 benchmarks/recursion_e2e/run.py result --dir "$O" --out "$O/recursion_e2e.json" --md "$O/table.md" 2> "$O/result.err" \
    | tee -a "$O/job.log"
  local r=${PIPESTATUS[0]}
  [ $r = 0 ] || tail -20 "$O/result.err"
  [ "${KEEP_DATA:-0}" = 1 ] || rm -rf "$FLOCK_WORK/data/$TAG"
  return $r
}
# the run's --max-min (SIGTERM) writes the result so far; `research cancel` ends the scope first (`run.py table RUN`)
trap 'log "SIGTERM: the result so far"; result; exit 143' TERM

# reps STEP: REPEAT_<STEP> (REPEAT_VSTAGE=1, ...) else REPEAT
reps() { local v=REPEAT_${1^^}; seq ${!v:-$REPEAT}; }
rc=0
for s in ${STEPS//,/ }; do
  case $s in
    build) step build 1 || { log "build failed: nothing later can run"; rc=1; break; } ;;
    inner) step inner 1 CHECKS=0 INNER_LOOPBACK=1 || rc=1 ;;
    oprove)
      M=$MODES
      if [ "$OPROVE_GPU" = 0 ] && [ "${GPU:-1}" != 0 ]; then
        # 85's CPU prover (bin-…-g0), built beside the GPU one; STAGE_FROM links this run's staged statements into a
        # throwaway TAG, so the build's staging doesn't restage the ones the inner run was proved against
        step build 2 GPU=0 TAG=$TAG-g0 STAGE_FROM=$FLOCK_WORK/data/$TAG || { rc=1; continue; }
        rm -rf "$FLOCK_WORK/data/$TAG-g0"
      fi
      # 85 builds flock-firewall (its build step, else oprove's) beside flock-verify, from this tree's Lean sources
      step oprove 1 MODES=$M REPLAY=0 WARM=${OPROVE_WARM:-0} RUNS=${OPROVE_RUNS:-1} GPU=$OPROVE_GPU || rc=1
      fw=$L/lean-$LSHA/flock-firewall; [[ ,$M, != *,firewall,* ]] || [ ! -x $fw ] || sha256sum $fw >> "$O/params.txt" ;;
    icheck|vcap|oreplay) for i in $(reps $s); do step $s $i || { rc=1; break; }; done ;;
    vstage) for i in $(reps vstage); do step vstage $i "FORGE=${FORGE//,/ }" || { rc=1; break; }; done ;;
    lean) for i in $(reps lean); do step lean $i INNER_LEAN=1 JOBS=${LEAN_JOBS:-4} NEG_INNER= || rc=1; done ;;
    # the lean step once more with rep 0's coins flipped (85's NEG_INNER): Lean must refuse each flipped coin
    negative) [ -z "$NEG" ] || step negative 1 STEP=lean INNER_LEAN=0 JOBS=${LEAN_JOBS:-4} "NEG_INNER=$NEG" || rc=1 ;;
    # the forgeries the ZK server stops before release, made on the honest releases after proving and put to Lean with the
    # lean step's own flags (forge_after.py; out/forge-after.json)
    forge-after)
      lo=$O/steps/lean/r1/out; fv=$L/lean-$LSHA/flock-verify
      timed forge-after 1 env OUT=$O/steps/forge-after/r1/out python3 benchmarks/recursion_e2e/forge_after.py \
        --work "$FLOCK_WORK" --tag "$TAG" --run-out "$lo" --kinds "${FORGE_AFTER:-coef,dirs,message}" --jobs "${LEAN_JOBS:-4}" \
        --bin "$([ -f $lo/lean/own-L0.txt ] || [ -f $lo/lean/own-alg-p0.txt ] && echo run || echo lean)=$fv" || rc=1 ;;
    *) log "STEPS: $s is not one of build inner icheck vstage vcap oprove oreplay lean forge-after negative"; rc=2 ;;
  esac
done

result || rc=1
log "done: exit $rc"
exit $rc
