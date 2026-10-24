#!/usr/bin/env bash
# The recursion profile's job (README.md): verity/ml/flock/pod/85-rec-reprice.sh's steps in order, each invocation under its
# own RESEARCH_RUN_DIR ($OUT/steps/<step>/r<i>, so every record 85 writes is kept) and /usr/bin/time -v, every CPU step
# REPEAT times and every GPU step once with WARM + RUNS sessions; then the profile's result (run.py profile result) in $OUT.
# Run from the verity tree on vy-nebius-1 as a GPU host job (85 leases a GPU around each prove itself):
#   research run --queue --on vy-nebius-1 --gpu-host --project verity --source . --cwd source --mem-gb 64 --cpus 32 \
#     --max-min 300 -- bash benchmarks/recursion/job.sh K=4096 N=2
# (`verity bench recursion profile run` prints and launches it.)
# env: K=4096 N=2 (85's inner statement: N instances of GemmCoordinate_v2{K})  TAG=bench-k<K>-n<N>
#      STEPS=build,inner,icheck,vstage,vcap,oprove,oreplay,lean  REPEAT=3 (each CPU step)  WARM=1 RUNS=3 (each GPU step)
#      MODES=firewall,dev (oprove's prover sides; the last one's sessions are what oreplay and lean read)
#      LEAN_JOBS=4 (Lean verifies this many V* sessions at once)  FLOCK_WORK=/workspace/jobs/rec-bench
#      CUDA_TK, WARM_FROM: as 85 takes them (default: proofs' rec-reprice unpack and cargo tree where present, read only)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
REPO=$(pwd); O=${OUT:-${RESEARCH_RUN_DIR:-/tmp/rec-bench-$$}/out}; mkdir -p "$O"
export K=${K:-4096} N=${N:-2}; export TAG=${TAG:-bench-k$K-n$N} WARM=${WARM:-1} RUNS=${RUNS:-3}
export FLOCK_WORK=${FLOCK_WORK:-/workspace/jobs/rec-bench}; mkdir -p "$FLOCK_WORK" || { echo "cannot make $FLOCK_WORK" >&2; exit 1; }
STEPS=${STEPS:-build,inner,icheck,vstage,vcap,oprove,oreplay,lean}; REPEAT=${REPEAT:-3}
PYTHONPATH=$REPO/tools/research/src python3 -m research keep "$FLOCK_WORK" --owner @benchmarks --expires 7d --priority low \
  --regenerate "bash benchmarks/recursion/job.sh K=$K N=$N" \
  --why "the recursion profile's builds and 85-rec-reprice's data/$TAG: a cache, rebuilt by the next profile" > "$O/keep.json" 2>&1 \
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
  echo "steps=$STEPS modes=${MODES:-firewall,dev} lean_jobs=${LEAN_JOBS:-4} lean_sources=$LSHA"
  echo "cuda_tk=${CUDA_TK:-} warm_from=${WARM_FROM:-}"; nproc; nvidia-smi -L 2>/dev/null; } > "$O/params.txt"

# step NAME I ENV...: one invocation of 85 (STEP=NAME and ENV) under $O/steps/NAME/r<I>; its exit status
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
    (cd $L/lean-src && lake build flock-firewall) && cp $L/lean-src/.lake/build/bin/flock-firewall $fw
  ) > "$O/build-firewall.log" 2>&1
  [ -x $fw ] || { tail -20 "$O/build-firewall.log"; return 1; }
  export FC_FIREWALL_LEAN=$fw
  sha256sum $fw >> "$O/params.txt"
}

rc=0
for s in ${STEPS//,/ }; do
  case $s in
    build) step build 1 || { log "build failed: nothing later can run"; rc=1; break; } ;;
    inner) step inner 1 CHECKS=0 INNER_LOOPBACK=1 || rc=1 ;;
    oprove)
      M=${MODES:-firewall,dev}
      if [[ ,$M, == *,firewall,* ]]; then
        log "flock-firewall: start"
        if firewall; then log "flock-firewall: $FC_FIREWALL_LEAN"
        else log "no flock-firewall: oprove without the firewall mode"; M=$(echo ,$M, | sed 's/,firewall,/,/; s/^,//; s/,$//'); fi
      fi
      step oprove 1 MODES=$M REPLAY=0 || rc=1 ;;
    icheck|vcap|oreplay) for i in $(seq $REPEAT); do step $s $i || { rc=1; break; }; done ;;
    vstage) for i in $(seq $REPEAT); do step vstage $i FORGE= || { rc=1; break; }; done ;;
    lean) for i in $(seq $REPEAT); do step lean $i INNER_LEAN=1 JOBS=${LEAN_JOBS:-4} || rc=1; done ;;
    *) log "STEPS: $s is not one of build inner icheck vstage vcap oprove oreplay lean"; rc=2 ;;
  esac
done

python3 benchmarks/recursion/run.py profile result --dir "$O" --out "$O/profile.json" --md "$O/table.md" 2> "$O/result.err" \
  | tee -a "$O/job.log" || { tail -20 "$O/result.err"; rc=1; }
log "done: exit $rc"
exit $rc
