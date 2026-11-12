#!/usr/bin/env bash
# Served proof blocks' Flock --zk prove time against the statement's size, on one GPU or on the CPU prover, for each block
# type's proof overhead and its fixed cost a statement. Run from the tree (research run --on NODE --cwd source):
#   1. build   flock-circuit (verity/protocols/flock/pod/60-circuit.sh MODE=build; with PT_DEVICE=gpu its CUDA prover), outside any lease
#   2. stage   each UNIT of PT_UNITS (stage.py's TYPE[:S=V,...] or census:NAME, `+public-weights` for its --public-weights,
#              `+packed-weights` for --public-weights --packed) at each N of
#              PT_NS, or of its own list (UNIT@N,N...; a census type's N may be stage.py's full or full/D), PT_JOBS at a
#              time, outside any lease, into $FLOCK_WORK/data/ (kept, keyed by the staging tree)
#   3. prove   per statement, a loopback --zk session set: a verifier process on VCPUS cores, WARM + RUNS prover sessions on the
#              rest, as benchmarks/flock_s8_block/run.sh runs its sessions (the ZK server's send check off, FC_COINS=os),
#              each with every unit in its sample (`subset:N`: Lean's verifier refuses a session without a sample);
#              with PT_DEVICE=gpu every session under one `gpu-lease 1` of at most LEASE_MIN minutes, the prover with --gpu;
#              out/DEV/NAME/start and done hold the statement's start and end (epoch seconds), and exit the prover's exit
#              status: done is written once the session set ends, whatever that status, so a statement finished only on 0
#   4. prove_time.py: out/summary.json
# env: PT_UNITS="fp8-coord:K=6144 fp8-coord:K=6144+public-weights"  PT_NS="16 64"  PT_DEVICE=gpu (or cpu)  PT_JOBS=4  WARM=1
#      RUNS=3  VCPUS=4  LEASE_MIN=30  SM=120  PT_CARD (gpu: the leased card's, from nvidia-smi)  FLOCK_WORK (default a
#      directory under /workspace/tmp, kept)  STEPS="build stage prove"  PT_BINARY (the prover and verifier binary the
#      sessions run, by default the one the build step builds: an A/B arm on another build)
#      PT_PERF=1: each prover process under `perf record -F PT_PERF_HZ` (default 499), its flat profile by symbol in perf.txt
#      beside prove.tsv (the samples themselves are not kept); upstream's phase traces pass through as arguments
#      (LIGERITO_TRACE=1 PCS_TRACE=1 LINCHECK_TRACE=1, into prove.err)
#      PT_STATEMENTS="DIR ...": statement directories staged elsewhere (stage_window.py's served members), each proved as it
#      is under its directory's name after PT_UNITS' (which then defaults to none)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
REPO=$(pwd); O=${RESEARCH_RUN_DIR:-/tmp/served-units-time}/out; mkdir -p $O
DEV=${PT_DEVICE:-gpu}; case $DEV in gpu | cpu) ;; *) echo "PT_DEVICE is gpu or cpu, not $DEV"; exit 1 ;; esac
FW=${FLOCK_WORK:-/workspace/tmp/served-units-time-$DEV}; mkdir -p $FW || exit 1
STMTS=${PT_STATEMENTS:-}; [ -n "$STMTS" ] && UNITS=${PT_UNITS:-} || UNITS=${PT_UNITS:-"fp8-coord:K=6144 fp8-coord:K=6144+public-weights"}
NS=${PT_NS:-"16 64"}; JOBS=${PT_JOBS:-4}
for s in $STMTS; do for f in circuit.txt pub.bin inst.bin stage.json; do
  [ -s $s/$f ] || { echo "PT_STATEMENTS: $s has no $f (stage_window.py stages one)"; exit 1; }; done; done
WARM=${WARM:-1}; RUNS=${RUNS:-3}; NSESS=$((WARM + RUNS)); SM=${SM:-120}; STEPS=${STEPS:-"build stage prove"}
TIMEV="/usr/bin/time -v -o"  # each step's time-*.txt record, where GNU time is installed
[ -x /usr/bin/time ] || { TIMEV=; echo "no GNU time at /usr/bin/time (apt-get install time): no time-*.txt records"; }
if [ $DEV = gpu ]; then
  . verity/protocols/flock/pod/cuda-13.3.sh
  C13=${CUDA_TK:-/usr/local/cuda-13.3}; export PATH=$C13/bin:$PATH LD_LIBRARY_PATH=$C13/lib64:${LD_LIBRARY_PATH:-}
fi
set -f  # a census name's brackets are not a glob
export PATH=$HOME/.cargo/bin:$PATH FC_COINS=os FC_PIPELINE_DEPTH=${FC_PIPELINE_DEPTH:-4} FC_HOST_PREPIN=${FC_HOST_PREPIN:-1}
CPUS=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))'); NC=$(echo $CPUS | tr , '\n' | wc -l)
VC=${VCPUS:-4}; PCPU=$(echo $CPUS | cut -d, -f1-$((NC - VC))); VCPU=$(echo $CPUS | cut -d, -f$((NC - VC + 1))-)
BUILT=$FW/flock-gpu-link/flock/target/release/flock-circuit; B=${PT_BINARY:-$BUILT}; PY=$FW/py/bin/python
has() { case " $STEPS " in *" $1 "*) return 0 ;; esac; return 1; }
{ echo "device=$DEV cpus=$CPUS prover=$PCPU verifier=$VCPU"; lscpu | grep -E 'Model name|^CPU\(s\)'; free -g | head -2
  [ $DEV = gpu ] && has prove && nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv; } > $O/host.txt 2>&1
echo "device=$DEV units=$UNITS ns=$NS statements=$STMTS warm=$WARM runs=$RUNS" > $O/config.txt
# PAIRS: one UNIT@N a statement, each unit at its own Ns or PT_NS
PAIRS=$(for t in $UNITS; do
  u=${t%@*}; ns=$NS; [ "$u" = "$t" ] || ns=$(echo "${t##*@}" | tr , ' ')
  for n in $ns; do echo "$u@$n"; done
done)

if has build; then
  echo "== build $(date -u +%T)"
  ( flock 9; MODE=build GPU=$([ $DEV = gpu ] && echo 1 || echo 0) SM=$SM FLOCK_WORK=$FW RESEARCH_RUN_DIR=$O/build \
    bash verity/protocols/flock/pod/60-circuit.sh > $O/build.log 2>&1 ) 9> $FW/.build.lock \
    || { tail -40 $O/build.log $O/build/out/build.log 2>/dev/null; exit 1; }
  sha256sum $BUILT | tee $O/binary.sha256
fi
[ $B = $BUILT ] || sha256sum $B | tee $O/binary-sessions.sha256
UV=$(command -v uv || echo $HOME/.local/bin/uv)
[ -x $UV ] || { curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1; UV=$HOME/.local/bin/uv; }
[ -x $PY ] || { $UV venv -q --python 3.12 $FW/py && $UV pip install -q --python $PY numpy blake3; } || exit 1
export PYTHONPATH=$REPO:$REPO/verity/protocols/flock/prover/python:$REPO/tools/verity/circuit_check/src:$REPO/adapters/vllm
# KEY: the staging tree's digest, over the Python under KEY_PATHS, the list window.py's KEY_PATHS repeats. A path the tree no
# longer holds would leave the key without what was under it, and a statement staged before a change there would be proved
# as current: the run stops instead
KEY_PATHS="benchmarks/served_units/stage.py verity/circuits verity/definitions verity/commitments verity/protocols adapters/vllm/verity_vllm tools/verity/circuit_check/src"
for p in $KEY_PATHS; do
  [ -e $REPO/$p ] || { echo "prove_time.sh: no $p in $REPO, a path of the staging key: correct KEY_PATHS in" \
    "benchmarks/served_units/prove_time.sh and window.py together"; exit 1; }
done
KEY=$(cd $REPO && find $KEY_PATHS -name '*.py' -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16) \
  || { echo "prove_time.sh: no staging key: find, xargs or sha256sum failed under KEY_PATHS (its own message is above)"; exit 1; }

# name UNIT N: the statement's directory name, a census name's past 96 characters cut to 80 and its digest's first 12
name() {
  local s=$(echo "$1" | tr 'A-Z' 'a-z' | tr -c 'a-z0-9.\n-' '-' | tr -s '-'); s=${s%-}
  [ ${#s} -gt 96 ] && s="${s:0:80}-$(echo -n "$1" | sha256sum | cut -c1-12)"
  echo "$s-n${2//\//-}"
}
# stage UNIT N: stage.py's statement into $FW/data/NAME-KEY, its record under $O/stage/NAME
stage() {
  local u=${1%+*} flag= nm=$(name "$1" $2)
  case $1 in *+public-weights) flag=--public-weights ;; *+packed-weights) flag="--public-weights --packed" ;; *) u=$1 ;; esac
  local d=$FW/data/$nm-$KEY o=$O/stage/$nm
  mkdir -p $o $FW/data
  (
    flock 9
    [ -e $d/.done ] && exit 0
    rm -rf $d && mkdir -p $d
    $TIMEV ${TIMEV:+$o/time-stage.txt} $PY $REPO/benchmarks/served_units/stage.py "$u" $2 $d $flag > $o/stage.log 2>&1 \
      || { echo "stage $nm FAILED: $(tail -1 $o/stage.log)"; rm -rf $d; exit 1; }
    touch $d/.done
  ) 9> $d.lock || return 1
  cp $d/stage.json $o/
  echo "staged $nm: $(python3 -c "import json; s = json.load(open('$o/stage.json')); print(s['ands'], 'ANDs, k_log', s['k_log'], 'G', s['vus_per_block'], s['seconds'])")"
}
if has stage; then
  echo "== stage $(date -u +%T) key $KEY"
  fail=0
  for p in $PAIRS; do
    while [ "$(jobs -rp | wc -l)" -ge $JOBS ]; do wait -n || fail=1; done
    stage "${p%@*}" "${p##*@}" &
  done
  while [ "$(jobs -rp | wc -l)" -gt 0 ]; do wait -n || fail=1; done
  [ $fail = 0 ] || { echo "a statement failed to stage"; exit 1; }
fi

free_port() { python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
wait_serving() { for i in $(seq 1800); do grep -q SERVING $1 && return 0; kill -0 $2 2>/dev/null || return 1; sleep 1; done; return 1; }
stamp() { while IFS= read -r l; do printf '%s\t%s\n' "$(date +%s.%N)" "$l"; done; }
# gpu_sample DIR: until DIR/stop exists, the GPU's utilization, memory, SM clock and power each ~0.2 s, stamped
gpu_sample() {
  rm -f $1/stop
  while [ ! -e $1/stop ]; do
    printf '%s\t%s\n' "$(date +%s.%N)" "$(nvidia-smi --query-gpu=utilization.gpu,memory.used,clocks.sm,power.draw \
      --format=csv,noheader,nounits -i ${GPU_LEASE_UUID:-0} 2>&1)"
    sleep 0.2
  done > $1/smi.tsv
}
# session UNIT N: stage.py's statement's sessions (session_at)
session() { local nm=$(name "$1" $2); local d=$FW/data/$nm-$KEY; session_at $nm $d; }
# session_at NAME DIR: a verifier process and WARM + RUNS prover sessions against DIR's statement, into $O/$DEV/NAME
session_at() {
  local nm=$1 d=$2 G= GS=; local o=$O/$DEV/$nm
  [ $DEV = gpu ] && G=--gpu
  mkdir -p $o $O/stage/$nm; rm -rf $d/sessions $o/done $o/exit; cp $d/stage.json $O/stage/$nm/; date +%s.%N > $o/start
  local PORT=$(free_port) PIN=$(sha512sum $d/circuit.txt | cut -d' ' -f1)
  local N=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['n'])" $d/stage.json)
  taskset -c $VCPU $B serve --zk --draw subset:$N --listen 127.0.0.1:$PORT --out $d/sessions --instances $d/pub.bin \
    --circuit $d/circuit.txt --pin $PIN --operator loopback --concurrency $((NSESS + 1)) > $o/serve.log 2>&1 &
  local SP=$!
  wait_serving $o/serve.log $SP || { tail -20 $o/serve.log; return 1; }
  [ $DEV = gpu ] && { gpu_sample $o & GS=$!; }
  local PERF=; [ "${PT_PERF:-0}" = 1 ] && PERF="perf record -q -F ${PT_PERF_HZ:-499} -o $o/perf.data --"
  RAYON_NUM_THREADS=$((NC - VC)) FC_VERIFY_AHEAD=$NSESS taskset -c $PCPU $TIMEV ${TIMEV:+$o/time-prove.txt} $PERF $B prove --zk $G \
    --verifier 127.0.0.1:$PORT --instances $d/inst.bin --circuit $d/circuit.txt --warm $WARM --runs $RUNS \
    2> $o/prove.err | stamp > $o/prove.tsv
  local rc=${PIPESTATUS[0]}
  echo "prove $nm exit $rc: $(grep -o '"accepted":[a-z]*' $o/prove.tsv | sort | uniq -c | tr '\n' ' ')$(grep -o '"why":"[^"]*"' $o/prove.tsv | sort -u | head -2 | tr '\n' ' ')"
  if [ -n "$PERF" ]; then
    perf report -q -i $o/perf.data --stdio --no-children --sort symbol --percent-limit 0.2 > $o/perf.txt 2> $o/perf-report.err
    rm -f $o/perf.data
  fi
  [ -n "$GS" ] && { touch $o/stop; wait $GS; }
  kill $SP; wait $SP 2>/dev/null
  echo $rc > $o/exit; date +%s.%N > $o/done
  grep -h panicked $o/prove.err | head -3
  return 0
}
if has prove; then
  echo "== prove ($DEV) $(date -u +%T)"
  if [ $DEV = gpu ]; then
    export -f session session_at name gpu_sample free_port wait_serving stamp
    export B O FW KEY DEV PCPU VCPU NC VC NSESS WARM RUNS PAIRS TIMEV STMTS
    GPU_LEASE_WHO=${GPU_LEASE_WHO:-served-units-time} gpu-lease 1 --wait --max-min ${LEASE_MIN:-30} -- bash -c '
      set -f; nvidia-smi -L > $O/lease-gpu.txt
      for p in $PAIRS; do session "${p%@*}" "${p##*@}"; done
      for s in $STMTS; do session_at $(basename $s) $s; done'
  else
    for p in $PAIRS; do session "${p%@*}" "${p##*@}"; done
    for s in $STMTS; do session_at $(basename $s) $s; done
  fi
fi
echo "== summary $(date -u +%T)"
CARD=${PT_CARD:-}
if [ $DEV = gpu ] && [ -z "$CARD" ]; then
  case "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" in
    *H200*) CARD=h200-sxm-141gb ;; *B200*) CARD=b200-sxm-180gb ;; *"RTX PRO 6000"*) CARD=rtx-pro-6000-server ;;
  esac
fi
python3 $REPO/benchmarks/served_units/prove_time.py $O ${CARD:+--card $CARD} > $O/summary.json
python3 -c "import json, sys; [print(r) for r in json.load(open(sys.argv[1]))['table']]" $O/summary.json
