#!/usr/bin/env bash
# The hidden-read kill experiment (read_gadget.py): GemmCoordinate<K> class statements with and without two hidden Merkle reads
# a template instance, each proved in loopback --zk sessions on the CPU under /usr/bin/time, every case with the same instances, cores and
# session counts, each session sampling every unit (--draw subset:N); then Lean's flock-verify on each case's last session,
# which it refuses: the read gadget's circuit is composed here, not a Program's, and a session without the verifier's
# program is outside what the soundness proof covers. Each case's sessions take their coin trees from its own
# challenger (challenger.sh: serve --tree-from), and their records are held to its release log. Run from the verity tree
# (research run --cwd source); CPU only. Outputs under out/ (flat): host.txt, check.json, stage-<case>.json,
# statement-<case>.txt, live.tsv (one LIVE record per proof: case, json), time-{prove,serve}-<case>.txt (peak RSS and wall
# time), prove-/serve- logs, challenger-<case>/ (its record, releases.jsonl and audit.json) and audit.tsv (case, challenger
# exit, audit exit), lean-<case>.log and lean.tsv (case, accepted, seconds), the core samplers (others_cores.txt,
# slice_cores.tsv). Staged statements and session records go under FLOCK_WORK/scratch/<run id>, removed at exit.
# Pins its own slices where it can take slice locks (node 1); on a vy-cpu pod, run it in a check slot (--pool), which pins it.
# env: CASES="today-2304 reads2-2304 reads4-2304 today-4096 reads2-4096" (today-K: --reads none; reads2-K: two binary reads of
#      depth 33, three levels a slot; reads4-K: two 4-ary reads of depth 17)  N=8 (instances; the prover pads to 8 blocks)
#      CPUS=48  WARM=1 RUNS=2  LEAN=1  FLOCK_WORK=/workspace, or where /workspace isn't writable
#      /workspace/jobs/pc-read-gadget (node 1) else /workspace/cache/read-gadget (the prover build, its venv and cached
#      binaries, as 70-class-sweep.sh)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
if [ -z "${RG_PINNED:-}" ]; then
  if mkdir -p "${SLICE_LOCKS:-/workspace/jobs/slices}" 2>/dev/null; then
    . verity/protocols/flock/pod/cpu-slices.sh
    hold_slices "" "${CPUS:-48}" || exit 1
    export RG_PINNED=$SLICE_CPUS RAYON_NUM_THREADS=$(echo $SLICE_CPUS | tr ',' '\n' | wc -l)
    exec taskset -c $SLICE_CPUS bash "$0" "$@"
  fi
  # no slice locks here (a vy-cpu pod): the check slot (research run --pool) has pinned the job to its own cores
  export RG_PINNED=$(taskset -pc $$ | sed 's/.*: //') RAYON_NUM_THREADS=$(nproc)
fi
REPO=$(pwd); O=${RESEARCH_RUN_DIR:-/tmp/read-gadget}/out; mkdir -p $O
if [ -z "${FLOCK_WORK:-}" ] && [ ! -w /workspace ]; then
  if [ -d /workspace/jobs ]; then export FLOCK_WORK=/workspace/jobs/pc-read-gadget; else export FLOCK_WORK=/workspace/cache/read-gadget; fi
fi
FW=${FLOCK_WORK:-/workspace}; W=$FW/flock-circuit; mkdir -p $W || exit 1
[ -w "$HOME" ] || { export HOME=$FW/home; mkdir -p $HOME; }
S=$FW/scratch/${RESEARCH_RUN_ID:-local-$$}; rm -rf $S; mkdir -p $S || exit 1
L=$FW/cache/pc-read-gadget; mkdir -p $L || exit 1
unset CARGO_TARGET_DIR
{ lscpu | grep -E 'Model name|^CPU\(s\)'; echo "pinned=$RG_PINNED rayon=$RAYON_NUM_THREADS slices=${SLICES:-} slice_wait_s=${SLICE_WAIT_S:-}"
  free -g | head -2; cat /proc/loadavg; } > $O/host.txt 2>&1
python3 verity/protocols/flock/pod/gemm_slowdown.py sample --pgid $(ps -o pgid= $$ | tr -d ' ') --out $O/others_cores.txt --every 10 &
SAMPLER=$!
python3 verity/protocols/flock/pod/gemm_hill.py cores --cpus $RG_PINNED --out $O/slice_cores.tsv --every 10 &
CORES=$!
trap 'kill $SAMPLER $CORES 2>/dev/null; rm -rf $S' EXIT
[ -x /usr/bin/time ] || { apt-get install -y -qq time >/dev/null 2>&1 || { apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq time >/dev/null 2>&1; }; }
[ -x /usr/bin/time ] || { echo "no /usr/bin/time: the prover's peak memory is the measurement" >&2; exit 1; }

# Lean's flock-verify (nothing fetched), built once per source in this lane's own directory, while the prover builds
LEAN_BIN=
if [ "${LEAN:-1}" = 1 ]; then
  LSHA=$(python3 verity/protocols/flock/pod/lake_closure.py key verity/protocols/flock/verifier)
  LEAN_BIN=$L/lean-$LSHA/flock-verify
  (
    export PATH=$HOME/.elan/bin:$PATH
    command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
    exec 9>$L/lean.lock; flock 9
    if [ ! -x $LEAN_BIN ]; then
      rm -rf $L/lean-src && LEAN_SRC=$(python3 verity/protocols/flock/pod/lake_closure.py copy verity/protocols/flock/verifier $L/lean-src) \
        && (cd $LEAN_SRC && lake build flock-verify) && mkdir -p $L/lean-$LSHA && cp $LEAN_SRC/.lake/build/bin/flock-verify $LEAN_BIN
    fi
  ) > $O/build-lean.log 2>&1 &
  LP=$!
fi
CHALLENGER_REPO=$REPO CHALLENGER_WORK=$S/challengers
. verity/protocols/flock/prover/pod/challenger.sh
challenger_build

# the CPU prover, cached per pod under its sources' key like 70-class-sweep.sh's binaries (one build at a time per FLOCK_WORK)
SM=${SM:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d .)}; SM=${SM:-89}
KEY=$(cat verity/protocols/flock/pod/20-gpu-link.sh verity/protocols/flock/pod/60-circuit.sh verity/protocols/flock/prover/*.patch verity/protocols/flock/prover/cuda_*_patch.py \
  $(find verity/protocols/flock/prover/cuda verity/protocols/flock/prover/live -type f | LC_ALL=C sort) | sha256sum | cut -c1-16)
BIN=$W/bin-$KEY-g0-sm$SM; F=$FW/flock-gpu-link/flock
exec 8>$W/build.lock; flock 8
if [ -x $BIN/flock-circuit ] && [[ $(ldd $BIN/flock-circuit 2>&1) == *"not found"* ]]; then
  rm -rf $BIN || { echo "$BIN: built against another glibc, and this job can't remove it" >&2; exit 1; }
fi
if [ ! -x $BIN/flock-circuit ]; then
  RESEARCH_RUN_DIR=$O/build FLOCK_WORK=$FW MODE=build GPU=0 SM=$SM bash verity/protocols/flock/pod/60-circuit.sh > $O/build.log 2>&1 || { tail -30 $O/build.log; exit 1; }
  mkdir -p $BIN && cp $F/target/release/flock-circuit $BIN/flock-circuit
fi
exec 8>&-
B=$BIN/flock-circuit; sha256sum $B | tee $O/binary.sha256
UV=$(command -v uv || echo $HOME/.local/bin/uv)
[ -x $UV ] || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; UV=$HOME/.local/bin/uv; }
[ -x $W/py/bin/python ] || { $UV venv -q --python 3.12 $W/py && $UV pip install -q --python $W/py/bin/python numpy blake3; } || exit 1
export PATH=$W/py/bin:$PATH PYTHONPATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/protocols/flock/prover/python:$REPO/adapters/vllm
CHALLENGER_PY=python3
challenger_built || exit 1
G="python3 verity/protocols/flock/pod/read_gadget.py"
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
  challenger_start $O/challenger-$case || exit 1
  /usr/bin/time -v -o $O/time-serve-$case.txt $B serve --zk --listen 127.0.0.1:$PORT --out $D/sessions --instances $D/pub-$N.bin \
    --circuit $D/circuit.txt --tables $D/tables --pin $PIN --operator loopback --draw subset:$N --sessions $((${WARM:-1} + ${RUNS:-2})) \
    "${TREE_FROM[@]}" > $O/serve-$case.log 2>&1 &
  SP=$!
  for i in $(seq 1800); do grep -q SERVING $O/serve-$case.log && break; kill -0 $SP 2>/dev/null || break; sleep 1; done
  /usr/bin/time -v -o $O/time-prove-$case.txt $B prove --zk --verifier 127.0.0.1:$PORT --instances $D/inst-$N.bin --circuit $D/circuit.txt \
    --tables $D/tables --warm ${WARM:-1} --runs ${RUNS:-2} > $O/prove-$case.txt 2> $O/prove-$case.err
  PRC=$?
  grep -h '^LIVE' $O/prove-$case.txt | sed "s/^LIVE\t/$case\t/" >> $O/live.tsv
  grep -h panicked $O/prove-$case.err | head -3
  # the verifier (/usr/bin/time's child, which reports its peak) closes after its last session; stop it if the prover failed
  [ $PRC = 0 ] || kill $(pgrep -P $SP) 2>/dev/null
  wait $SP 2>/dev/null
  challenger_end $O/challenger-$case; CRC=$?
  challenger_audit $O/challenger-$case $(dirname $(ls -1 $D/sessions/*/session.json)) > /dev/null
  echo -e "$case\t$CRC\t$?" >> $O/audit.tsv
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
