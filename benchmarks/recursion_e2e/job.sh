#!/usr/bin/env bash
# The recursion bench's job (README.md): verity/ml/flock/pod/85-rec-reprice.sh's steps in order, as one research run, each
# invocation under its own RESEARCH_RUN_DIR ($OUT/steps/<step>/r<i>, so every record 85 writes is kept) and /usr/bin/time -v,
# every CPU step REPEAT times and every GPU step once with its sessions; then the result (run.py result) in
# $OUT/recursion_e2e.json and $OUT/table.md. Run from the verity tree on vy-nebius-1 as a GPU host job (85 leases a GPU
# around each prove itself); `verity bench recursion_e2e launch` prints and launches it:
#   research run --queue --on vy-nebius-1 --gpu-host --project verity --source . --cwd source --mem-gb 64 --cpus 32 \
#     --max-min 420 -- bash benchmarks/recursion_e2e/job.sh K=4096 N=2
# env: K=4096 N=2 (85's inner statement: N instances of GemmCoordinate_v2{K})
#      STEPS=build,inner,icheck,vstage,vcap,oprove,oreplay,lean,negative  REPEAT=3 (each CPU step; REPEAT_<STEP>=n one step's)
#      WARM=1 RUNS=3 (the inner's sessions)  OPROVE_WARM=0 OPROVE_RUNS=1 (each V* statement's sessions)
#      MODES=firewall (oprove's prover sides; the last one's sessions are what oreplay and lean read)
#      GPU_MAX_MIN=75 (each prove's lease, minutes)  LEAN_JOBS=4 (Lean verifies this many V* sessions at once)
#      OPROVE_GPU=0 under MODES=firewall, else GPU (V*'s prover: 85's CPU prover, or the GPU one under a lease)
#      NEG=challenge,query (negative: the inner coins of rep 0 flipped, each verified under every V* statement; empty: none)
#      TAG=rb-<run id> (85's data/$TAG, removed at the end unless KEEP_DATA=1; a later job's STEPS=oprove,... takes it with
#      TAG=... KEEP_DATA=1)  FLOCK_WORK=/workspace/jobs/rec-bench
#      and any of 85's own (LEVELS, GPU, PCPUS, GPU_PREEMPTIBLE, ...), passed through
#      CUDA_TK, WARM_FROM: as 85 takes them (default: proofs' rec-reprice unpack and cargo tree where present, read only)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
# 85's leases are preemptible unless GPU_PREEMPTIBLE is empty: on node 1 a waiter stops one after about 2 min, a lost
# measurement (85 retries the statement afresh, at most GPU_TRIES times), so the bench's are not. A firewalled V* session
# at K = 4096 takes minutes (L3: 519 s), past 85's default lease of 10 min for the larger statements (run.py's LEASE_MIN)
export GPU_PREEMPTIBLE=${GPU_PREEMPTIBLE-} GPU_MAX_MIN=${GPU_MAX_MIN:-75}
REPO=$(pwd); O=${OUT:-${RESEARCH_RUN_DIR:-/tmp/rec-bench-$$}/out}; mkdir -p "$O"
export K=${K:-4096} N=${N:-2}; export TAG=${TAG:-rb-${RESEARCH_RUN_ID:-$$}} WARM=${WARM:-1} RUNS=${RUNS:-3}
export FLOCK_WORK=${FLOCK_WORK:-/workspace/jobs/rec-bench}; mkdir -p "$FLOCK_WORK" || { echo "cannot make $FLOCK_WORK" >&2; exit 1; }
STEPS=${STEPS:-build,inner,icheck,vstage,vcap,oprove,oreplay,lean,negative}; REPEAT=${REPEAT:-3}
MODES=${MODES:-firewall}; NEG=${NEG-challenge,query}; NEG=${NEG//,/ }
# OPROVE_GPU=0 (the default under the firewall): V*'s statements proved by 85's CPU prover, no lease. Node 1's lease pool
# stops a lease at about 10 min while its GPU computes nothing (three firewalled L0 tries, 74 min non-preemptible leases,
# exit 143 at 10m18s each: r20261024-075309-69be), and a firewalled session's GPU waits on the Lean session loop for minutes
[[ ,$MODES, == *,firewall,* ]] && OPROVE_GPU=${OPROVE_GPU:-0} || OPROVE_GPU=${OPROVE_GPU:-${GPU:-1}}
PYTHONPATH=$REPO/tools/research/src python3 -m research keep "$FLOCK_WORK" --owner @benchmarks --expires 7d --priority low \
  --regenerate "bash benchmarks/recursion_e2e/job.sh K=$K N=$N" \
  --why "the recursion bench's builds (and a KEEP_DATA=1 run's data/TAG): a cache, rebuilt by the next run" > "$O/keep.json" 2>&1 \
  || echo "no retention record on $FLOCK_WORK: $(tail -1 "$O/keep.json")" >&2
# 85 moves HOME where it can't write; the firewall's Lean build below must find the same elan
[ -w "$HOME" ] || { export HOME=$FLOCK_WORK/home; mkdir -p "$HOME"; }
RR=/workspace/jobs/rec-reprice
[ -z "${CUDA_TK:-}" ] && [ -x $RR/cuda-13.3.1/bin/nvcc ] && export CUDA_TK=$RR/cuda-13.3.1
[ -z "${WARM_FROM:-}" ] && [ -d $RR/flock-gpu-link/flock ] && export WARM_FROM=$RR/flock-gpu-link
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$O/job.log"; }
REC=verity/ml/flock/pod/85-rec-reprice.sh
L=$FLOCK_WORK/cache
LSHA=$(cd verity/core/service/flock && find . -name .lake -prune -o -type f -print | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)
{ echo "commit=${RESEARCH_SOURCE_SHA:-$(git rev-parse HEAD 2>/dev/null)} K=$K N=$N TAG=$TAG WARM=$WARM RUNS=$RUNS REPEAT=$REPEAT"
  echo "OPROVE_WARM=${OPROVE_WARM:-0} OPROVE_RUNS=${OPROVE_RUNS:-1} GPU_MAX_MIN=$GPU_MAX_MIN GPU=${GPU:-1} OPROVE_GPU=$OPROVE_GPU" \
    $(for v in $(compgen -v REPEAT_); do echo "$v=${!v}"; done)
  echo "steps=$STEPS modes=$MODES lean_jobs=${LEAN_JOBS:-4} lean_sources=$LSHA neg=${NEG// /+}"
  echo "cuda_tk=${CUDA_TK:-} warm_from=${WARM_FROM:-}"; nproc; nvidia-smi -L 2>/dev/null; } > "$O/params.txt"

# step NAME I ENV...: one invocation of 85 (STEP=NAME, unless ENV sets it, and ENV) under $O/steps/NAME/r<I>; its exit status
step() {
  local name=$1 i=$2; shift 2
  local d=$O/steps/$name/r$i
  mkdir -p "$d"
  log "$name r$i: start"
  printf '%s\t%s\t%s\n' "$name" "$i" "$(date +%s.%N)" >> "$O/phases.tsv"
  env RESEARCH_RUN_DIR="$d" STEP=$name "$@" /usr/bin/time -v -o "$d/time-step.txt" bash $REC > "$d/step.log" 2>&1
  local rc=$?
  echo $rc > "$d/rc"
  printf '%s\t%s\t%s\t%s\n' "$name" "$i" "$(date +%s.%N)" "$rc" >> "$O/phases.tsv"
  log "$name r$i: exit $rc"
  return $rc
}

# firewall: flock-firewall from the sources 85's flock-verify is built from ($L/lean-src at this tree's LSHA), for oprove's
# firewall mode (FC_FIREWALL_LEAN); refuses rather than mixing two trees' Lean
firewall() {
  local fw=$L/lean-$LSHA/flock-firewall
  [ -x $fw ] || (
    export PATH=$HOME/.elan/bin:$PATH
    exec 9>$L/lean.lock; flock 9
    [ -x $fw ] && exit 0
    src=$(cd $L/lean-src 2>/dev/null && find . -name .lake -prune -o -type f -print | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)
    [ "$src" = "$LSHA" ] || { rm -rf $L/lean-src && mkdir -p $L/lean-src && cp -r verity/core/service/flock/. $L/lean-src/; }
    (cd $L/lean-src && lake build flock-firewall) && mkdir -p $L/lean-$LSHA && cp $L/lean-src/.lake/build/bin/flock-firewall $fw
  ) > "$O/build-firewall.log" 2>&1
  [ -x $fw ] || { tail -20 "$O/build-firewall.log"; return 1; }
  export FC_FIREWALL_LEAN=$fw
  sha256sum $fw >> "$O/params.txt"
}

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
      if [[ ,$M, == *,firewall,* ]]; then
        log "flock-firewall: start"
        if firewall; then log "flock-firewall: $FC_FIREWALL_LEAN"
        else log "no flock-firewall: oprove refused (MODES=$M needs it)"; rc=1; continue; fi
      fi
      if [ "$OPROVE_GPU" = 0 ] && [ "${GPU:-1}" != 0 ]; then
        # 85's CPU prover (bin-…-g0), built beside the GPU one; STAGE_FROM links this run's staged statements into a
        # throwaway TAG, so the build's staging doesn't restage the ones the inner session was proved against
        step build 2 GPU=0 TAG=$TAG-g0 STAGE_FROM=$FLOCK_WORK/data/$TAG || { rc=1; continue; }
        rm -rf "$FLOCK_WORK/data/$TAG-g0"
      fi
      step oprove 1 MODES=$M REPLAY=0 WARM=${OPROVE_WARM:-0} RUNS=${OPROVE_RUNS:-1} GPU=$OPROVE_GPU || rc=1 ;;
    icheck|vcap|oreplay) for i in $(reps $s); do step $s $i || { rc=1; break; }; done ;;
    vstage) for i in $(reps vstage); do step vstage $i FORGE= || { rc=1; break; }; done ;;
    lean) for i in $(reps lean); do step lean $i INNER_LEAN=1 JOBS=${LEAN_JOBS:-4} NEG_INNER= || rc=1; done ;;
    # the lean step once more with rep 0's coins flipped (85's NEG_INNER): Lean must refuse each flipped coin
    negative) [ -z "$NEG" ] || step negative 1 STEP=lean INNER_LEAN=0 JOBS=${LEAN_JOBS:-4} "NEG_INNER=$NEG" || rc=1 ;;
    *) log "STEPS: $s is not one of build inner icheck vstage vcap oprove oreplay lean negative"; rc=2 ;;
  esac
done

result || rc=1
log "done: exit $rc"
exit $rc
