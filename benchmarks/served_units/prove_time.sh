#!/usr/bin/env bash
# Served proof blocks' C-Flock --zk prove time against the statement's size, on one GPU or on the CPU prover, for each block
# type's proof overhead and its fixed cost a statement. Run from the tree (research run --on NODE --cwd source):
#   1. build   flock-circuit (verity/ml/flock/pod/60-circuit.sh MODE=build; with PT_DEVICE=gpu its CUDA prover), outside any lease
#   2. stage   each UNIT of PT_UNITS (stage.py's TYPE[:S=V,...], `+public-weights` for its --public-weights) at each N of
#              PT_NS, PT_JOBS at a time, outside any lease, into $FLOCK_WORK/data/ (kept, keyed by the staging tree)
#   3. prove   per statement, a loopback --zk session set: a verifier process on VCPUS cores, WARM + RUNS prover sessions on the
#              rest, as benchmarks/flock_s8_block/run.sh runs its sessions (the ZK server's send check off, FC_COINS=os);
#              with PT_DEVICE=gpu every session under one `gpu-lease 1` of at most LEASE_MIN minutes, the prover with --gpu
#   4. prove_time.py: out/summary.json
# env: PT_UNITS="fp8-coord:K=6144 fp8-coord:K=6144+public-weights"  PT_NS="16 64"  PT_DEVICE=gpu (or cpu)  PT_JOBS=4  WARM=1
#      RUNS=3  VCPUS=4  LEASE_MIN=30  SM=120  PT_CARD (gpu: the leased card's, from nvidia-smi)  FLOCK_WORK (default a
#      directory under /workspace/tmp, kept)  STEPS="build stage prove"
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
REPO=$(pwd); O=${RESEARCH_RUN_DIR:-/tmp/served-units-time}/out; mkdir -p $O
DEV=${PT_DEVICE:-gpu}; case $DEV in gpu | cpu) ;; *) echo "PT_DEVICE is gpu or cpu, not $DEV"; exit 1 ;; esac
FW=${FLOCK_WORK:-/workspace/tmp/served-units-time-$DEV}; mkdir -p $FW || exit 1
UNITS=${PT_UNITS:-"fp8-coord:K=6144 fp8-coord:K=6144+public-weights"}; NS=${PT_NS:-"16 64"}; JOBS=${PT_JOBS:-4}
WARM=${WARM:-1}; RUNS=${RUNS:-3}; NSESS=$((WARM + RUNS)); SM=${SM:-120}; STEPS=${STEPS:-"build stage prove"}
if [ $DEV = gpu ]; then
  . verity/ml/flock/pod/cuda-13.3.sh
  export PATH=$CUDA_TK/bin:$PATH LD_LIBRARY_PATH=$CUDA_TK/lib64:${LD_LIBRARY_PATH:-}
fi
export PATH=$HOME/.cargo/bin:$PATH FC_COINS=os FC_PIPELINE_DEPTH=${FC_PIPELINE_DEPTH:-4} FC_HOST_PREPIN=${FC_HOST_PREPIN:-1}
CPUS=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))'); NC=$(echo $CPUS | tr , '\n' | wc -l)
VC=${VCPUS:-4}; PCPU=$(echo $CPUS | cut -d, -f1-$((NC - VC))); VCPU=$(echo $CPUS | cut -d, -f$((NC - VC + 1))-)
B=$FW/flock-gpu-link/flock/target/release/flock-circuit; PY=$FW/py/bin/python
has() { case " $STEPS " in *" $1 "*) return 0 ;; esac; return 1; }
{ echo "device=$DEV cpus=$CPUS prover=$PCPU verifier=$VCPU"; lscpu | grep -E 'Model name|^CPU\(s\)'; free -g | head -2
  [ $DEV = gpu ] && nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv; } > $O/host.txt 2>&1
echo "device=$DEV units=$UNITS ns=$NS warm=$WARM runs=$RUNS" > $O/config.txt

if has build; then
  echo "== build $(date -u +%T)"
  ( flock 9; MODE=build GPU=$([ $DEV = gpu ] && echo 1 || echo 0) SM=$SM FLOCK_WORK=$FW RESEARCH_RUN_DIR=$O/build \
    bash verity/ml/flock/pod/60-circuit.sh > $O/build.log 2>&1 ) 9> $FW/.build.lock \
    || { tail -40 $O/build.log $O/build/out/build.log 2>/dev/null; exit 1; }
  sha256sum $B | tee $O/binary.sha256
fi
UV=$(command -v uv || echo $HOME/.local/bin/uv)
[ -x $UV ] || { curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1; UV=$HOME/.local/bin/uv; }
[ -x $PY ] || { $UV venv -q --python 3.12 $FW/py && $UV pip install -q --python $PY numpy blake3; } || exit 1
export PYTHONPATH=$REPO:$REPO/verity/ml/flock/python:$REPO/tools/verity/circuit_check/src:$REPO/integrations/vllm
KEY=$(cd $REPO && find benchmarks/served_units/stage.py verity/ml integrations/vllm/verity_vllm tools/verity/circuit_check/src \
  -name '*.py' -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)

# name UNIT N: the statement's directory name
name() { echo "$(echo "$1" | tr ':,=+' '----' | tr 'A-Z' 'a-z')-n$2"; }
# stage UNIT N: stage.py's statement into $FW/data/NAME-KEY, its record under $O/stage/NAME
stage() {
  local u=${1%+public-weights} flag= nm=$(name $1 $2); [ "$u" = "$1" ] || flag=--public-weights
  local d=$FW/data/$nm-$KEY o=$O/stage/$nm
  mkdir -p $o $FW/data
  (
    flock 9
    [ -e $d/.done ] && exit 0
    rm -rf $d && mkdir -p $d
    /usr/bin/time -v -o $o/time-stage.txt $PY $REPO/benchmarks/served_units/stage.py $u $2 $d $flag > $o/stage.log 2>&1 \
      || { echo "stage $nm FAILED: $(tail -1 $o/stage.log)"; rm -rf $d; exit 1; }
    touch $d/.done
  ) 9> $d.lock || return 1
  cp $d/stage.json $o/
  echo "staged $nm: $(python3 -c "import json; s = json.load(open('$o/stage.json')); print(s['ands'], 'ANDs, k_log', s['k_log'], 'G', s['vus_per_block'], s['seconds'])")"
}
if has stage; then
  echo "== stage $(date -u +%T) key $KEY"
  fail=0
  for u in $UNITS; do for n in $NS; do
    while [ "$(jobs -rp | wc -l)" -ge $JOBS ]; do wait -n || fail=1; done
    stage $u $n &
  done; done
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
# session UNIT N: a verifier process and WARM + RUNS prover sessions against it, into $O/$DEV/NAME
session() {
  local nm=$(name $1 $2) G= GS=; local d=$FW/data/$nm-$KEY o=$O/$DEV/$nm
  [ $DEV = gpu ] && G=--gpu
  mkdir -p $o; rm -rf $d/sessions
  local PORT=$(free_port) PIN=$(sha512sum $d/circuit.txt | cut -d' ' -f1)
  taskset -c $VCPU $B serve --zk --listen 127.0.0.1:$PORT --out $d/sessions --instances $d/pub.bin --circuit $d/circuit.txt \
    --pin $PIN --operator loopback --concurrency $((NSESS + 1)) > $o/serve.log 2>&1 &
  local SP=$!
  wait_serving $o/serve.log $SP || { tail -20 $o/serve.log; return 1; }
  [ $DEV = gpu ] && { gpu_sample $o & GS=$!; }
  RAYON_NUM_THREADS=$((NC - VC)) FC_VERIFY_AHEAD=$NSESS taskset -c $PCPU /usr/bin/time -v -o $o/time-prove.txt $B prove --zk $G \
    --verifier 127.0.0.1:$PORT --instances $d/inst.bin --circuit $d/circuit.txt --warm $WARM --runs $RUNS \
    2> $o/prove.err | stamp > $o/prove.tsv
  echo "prove $nm exit ${PIPESTATUS[0]}: $(grep -o '"accepted":[a-z]*' $o/prove.tsv | sort | uniq -c | tr '\n' ' ')$(grep -o '"why":"[^"]*"' $o/prove.tsv | sort -u | head -2 | tr '\n' ' ')"
  [ -n "$GS" ] && { touch $o/stop; wait $GS; }
  kill $SP; wait $SP 2>/dev/null
  grep -h panicked $o/prove.err | head -3
  return 0
}
if has prove; then
  echo "== prove ($DEV) $(date -u +%T)"
  if [ $DEV = gpu ]; then
    export -f session name gpu_sample free_port wait_serving stamp
    export B O FW KEY DEV PCPU VCPU NC VC NSESS WARM RUNS UNITS NS
    GPU_LEASE_WHO=${GPU_LEASE_WHO:-served-units-time} gpu-lease 1 --wait --max-min ${LEASE_MIN:-30} -- bash -c '
      nvidia-smi -L > $O/lease-gpu.txt
      for u in $UNITS; do for n in $NS; do session $u $n; done; done'
  else
    for u in $UNITS; do for n in $NS; do session $u $n; done; done
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
