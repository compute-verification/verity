#!/usr/bin/env bash
# The hidden-read kill experiment (read_gadget.py): GemmCoordinate<K> class statements with and without two hidden Merkle reads
# a template instance, each proved in loopback --zk sessions on the CPU under /usr/bin/time, every case with the same instances, cores and
# session counts; then Lean's flock-verify on each case's last session. Run from the verity tree (research run --cwd source);
# CPU only. Outputs under out/ (flat): host.txt, check.json, stage-<case>.json, statement-<case>.txt, live.tsv (one LIVE record
# per proof: case, json), time-{prove,serve}-<case>.txt (peak RSS and wall time), prove-/serve- logs, lean-<case>.log and
# lean.tsv (case, accepted, seconds), the core samplers (others_cores.txt, slice_cores.tsv). Staged statements and session
# records go under FLOCK_WORK/scratch/<run id>, removed at exit.
# env: CASES="today-2304 reads2-2304 reads4-2304 today-4096 reads2-4096" (today-K: --reads none; reads2-K: two binary reads of
#      depth 33, three levels a slot; reads4-K: two 4-ary reads of depth 17)  N=8 (instances; the prover pads to 8 blocks)
#      CPUS=48  WARM=1 RUNS=2  LEAN=1  FLOCK_WORK=/workspace, or /workspace/jobs/pc-read-gadget where /workspace isn't writable
#      (node 1) (the prover build, its venv and cached binaries, as 70-class-sweep.sh)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
if [ -z "${RG_PINNED:-}" ]; then
  . verity/ml/flock/pod/cpu-slices.sh
  hold_slices "" "${CPUS:-48}" || exit 1
  export RG_PINNED=$SLICE_CPUS RAYON_NUM_THREADS=$(echo $SLICE_CPUS | tr ',' '\n' | wc -l)
  exec taskset -c $SLICE_CPUS bash "$0" "$@"
fi
export FC_COINS=${FC_COINS:-os}
REPO=$(pwd); O=${RESEARCH_RUN_DIR:-/tmp/read-gadget}/out; mkdir -p $O
[ -z "${FLOCK_WORK:-}" ] && [ ! -w /workspace ] && export FLOCK_WORK=/workspace/jobs/pc-read-gadget
FW=${FLOCK_WORK:-/workspace}; W=$FW/flock-circuit; mkdir -p $W || exit 1
[ -w "$HOME" ] || { export HOME=$FW/home; mkdir -p $HOME; }
S=$FW/scratch/${RESEARCH_RUN_ID:-local-$$}; rm -rf $S; mkdir -p $S || exit 1
L=$FW/cache/pc-read-gadget; mkdir -p $L || exit 1
unset CARGO_TARGET_DIR
{ lscpu | grep -E 'Model name|^CPU\(s\)'; echo "pinned=$RG_PINNED rayon=$RAYON_NUM_THREADS slices=${SLICES:-} slice_wait_s=${SLICE_WAIT_S:-}"
  free -g | head -2; cat /proc/loadavg; } > $O/host.txt 2>&1
python3 verity/ml/flock/pod/gemm_slowdown.py sample --pgid $(ps -o pgid= $$ | tr -d ' ') --out $O/others_cores.txt --every 10 &
SAMPLER=$!
python3 verity/ml/flock/pod/gemm_hill.py cores --cpus $RG_PINNED --out $O/slice_cores.tsv --every 10 &
CORES=$!
trap 'kill $SAMPLER $CORES 2>/dev/null; rm -rf $S' EXIT
[ -x /usr/bin/time ] || { apt-get install -y -qq time >/dev/null 2>&1 || { apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq time >/dev/null 2>&1; }; }
[ -x /usr/bin/time ] || { echo "no /usr/bin/time: the prover's peak memory is the measurement" >&2; exit 1; }

# Lean's flock-verify (no dependencies), built once per source in this lane's own directory, while the prover builds
LEAN_BIN=
if [ "${LEAN:-1}" = 1 ]; then
  LSHA=$(cd verity/core/service/flock && find . -name .lake -prune -o -type f -print | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)
  LEAN_BIN=$L/lean-$LSHA/flock-verify
  (
    export PATH=$HOME/.elan/bin:$PATH
    command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
    exec 9>$L/lean.lock; flock 9
    if [ ! -x $LEAN_BIN ]; then
      rm -rf $L/lean-src && mkdir -p $L/lean-src && cp -r verity/core/service/flock/. $L/lean-src/ \
        && (cd $L/lean-src && lake build flock-verify) && mkdir -p $L/lean-$LSHA && cp $L/lean-src/.lake/build/bin/flock-verify $LEAN_BIN
    fi
  ) > $O/build-lean.log 2>&1 &
  LP=$!
fi

# the CPU prover, cached per pod under its sources' key like 70-class-sweep.sh's binaries (one build at a time per FLOCK_WORK)
SM=${SM:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d .)}; SM=${SM:-89}
KEY=$(cat verity/ml/flock/pod/20-gpu-link.sh verity/ml/flock/pod/60-circuit.sh verity/ml/flock/*.patch verity/ml/flock/cuda_*_patch.py \
  $(find verity/ml/flock/cuda verity/ml/flock/live -type f | LC_ALL=C sort) | sha256sum | cut -c1-16)
BIN=$W/bin-$KEY-g0-sm$SM; F=$FW/flock-gpu-link/flock
exec 8>$W/build.lock; flock 8
if [ -x $BIN/flock-circuit ] && [[ $(ldd $BIN/flock-circuit 2>&1) == *"not found"* ]]; then
  rm -rf $BIN || { echo "$BIN: built against another glibc, and this job can't remove it" >&2; exit 1; }
fi
if [ ! -x $BIN/flock-circuit ]; then
  RESEARCH_RUN_DIR=$O/build FLOCK_WORK=$FW MODE=build GPU=0 SM=$SM bash verity/ml/flock/pod/60-circuit.sh > $O/build.log 2>&1 || { tail -30 $O/build.log; exit 1; }
  mkdir -p $BIN && cp $F/target/release/flock-circuit $BIN/flock-circuit
fi
exec 8>&-
B=$BIN/flock-circuit; sha256sum $B | tee $O/binary.sha256
UV=$(command -v uv || echo $HOME/.local/bin/uv)
[ -x $UV ] || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; UV=$HOME/.local/bin/uv; }
[ -x $W/py/bin/python ] || { $UV venv -q --python 3.12 $W/py && $UV pip install -q --python $W/py/bin/python numpy blake3; } || exit 1
export PATH=$W/py/bin:$PATH PYTHONPATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/ml/flock/python:$REPO/integrations/vllm
G="python3 verity/ml/flock/pod/read_gadget.py"
$G check > $O/check.json || { cat $O/check.json; exit 1; }

PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
N=${N:-8}
for case in ${CASES:-today-2304 reads2-2304 reads4-2304 today-4096 reads2-4096}; do
  K=${case#*-}
  case ${case%-*} in today) R=none ;; reads2) R=2:33:3:2 ;; reads4) R=4:17:1:2 ;; *) echo "case $case?" >&2; exit 1 ;; esac
  D=$S/$case
  $G stage --k $K --reads $R --n $N --sets $S/sets --out $D > $O/stage-$case.json 2> $O/stage-$case.err || { tail -5 $O/stage-$case.err; exit 1; }
  PIN=$(sha512sum $D/circuit.txt | cut -d' ' -f1)
  $B statement --zk --circuit $D/circuit.txt --instances $D/pub-$N.bin --tables $D/tables > $O/statement-$case.txt 2>&1
  /usr/bin/time -v -o $O/time-serve-$case.txt $B serve --zk --listen 127.0.0.1:$PORT --out $D/sessions --instances $D/pub-$N.bin \
    --circuit $D/circuit.txt --tables $D/tables --pin $PIN --operator loopback > $O/serve-$case.log 2>&1 &
  SP=$!
  for i in $(seq 1800); do grep -q SERVING $O/serve-$case.log && break; kill -0 $SP 2>/dev/null || break; sleep 1; done
  /usr/bin/time -v -o $O/time-prove-$case.txt $B prove --zk --verifier 127.0.0.1:$PORT --instances $D/inst-$N.bin --circuit $D/circuit.txt \
    --tables $D/tables --warm ${WARM:-1} --runs ${RUNS:-2} > $O/prove-$case.txt 2> $O/prove-$case.err
  grep -h '^LIVE' $O/prove-$case.txt | sed "s/^LIVE\t/$case\t/" >> $O/live.tsv
  grep -h panicked $O/prove-$case.err | head -3
  # the verifier is /usr/bin/time's child: stop it, and time reports its peak
  kill $(pgrep -P $SP) 2>/dev/null; wait $SP 2>/dev/null
  if [ -n "$LEAN_BIN" ]; then
    [ -n "${LP:-}" ] && wait $LP; LP=
    if [ -x $LEAN_BIN ]; then
      SESS=$(dirname $(ls -1 $D/sessions/*/session.json | LC_ALL=C sort | tail -1))
      A=$D/archive
      $LEAN_BIN archive-put $A $D/circuit.txt $D/pub-$N.bin $D/tables/* > $D/archive.txt 2> $O/lean-$case.log
      key() { awk -v f="$1" '$2 ~ f"$" {print $1}' $D/archive.txt | head -1; }
      /usr/bin/time -v -o $O/time-lean-$case.txt $LEAN_BIN verify --zk --statement verity/flock-circuit --archive $A \
        --circuit $(key circuit.txt) --public $(key pub-$N.bin) --session $SESS >> $O/lean-$case.log 2>&1
      echo -e "$case\t$?\t$(grep -h 'Elapsed (wall' $O/time-lean-$case.txt | awk '{print $NF}')\t$(grep -h '^VERDICT' $O/lean-$case.log | head -1 | cut -c1-300)" >> $O/lean.tsv
    fi
  fi
  rm -rf $D/sessions $D/archive $D/inst-$N.bin
done
true
