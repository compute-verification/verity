#!/usr/bin/env bash
# flock_s8_block: C-Flock --zk's price for proof blocks of ROWS adjacent cells of one int8 GEMM output row (program.py), each
# instance's rows hashed in the circuit, on one GPU node. Run from the tree (research run --on NODE --cwd source):
#   1. build    flock-circuit with the CUDA prover (60-circuit.sh MODE=build), outside any lease
#   2. stage    each K:N[:R] of STATEMENTS (class_statement --stage-only, N instances of S8Matmul_v1{N=R, K}; R defaults to ROWS),
#               outside any lease, into data/kK-nN-rR
#   3. sanity   a loopback --zk session set on the CPU of a K = 64, N = 2 statement (and on the GPU first thing in step 4;
#               SANITY=0 skips that one)
#   4. gpu      under one `gpu-lease 1` of at most LEASE_MIN minutes: per statement, a loopback --zk session set (a verifier
#               process on VCPUS cores, WARM + RUNS prover sessions with --gpu on the rest), sampling the GPU's utilization,
#               memory and, where dcgmi runs, its tensor-pipe and SM activity
#               and with NSYS=1, a last session set (WARM=0, RUNS=2) of the first statement under Nsight Systems' CUDA trace: its
#               kernel and memcpy time summaries in out/nsys (the GPU's busy time, not its utilization counter's)
#   5. cpu      per statement of CPU_STATEMENTS, the same session set without --gpu (the CPU prover), no lease
# out/summary.json: per statement and mode, the median timed session's prove time, its phases, GPU busy share and peaks.
# env: STATEMENTS="4096:1 4096:8 14336:1"  CPU_STATEMENTS="4096:1"  ROWS=8  WARM=1  RUNS=3  VCPUS=4  LEASE_MIN=20  SM=90
#      CUDA_TK (default: verity/ml/flock/pod/cuda-13.3.sh's)  NSYS=1  NSYS_STATEMENT (default the first)  FLOCK_WORK (default: a directory under /workspace/tmp, kept between steps)  STEPS="build stage sanity gpu cpu"
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
REPO=$(pwd); O=${RESEARCH_RUN_DIR:-/tmp/flock-s8-block}/out; mkdir -p $O
FW=${FLOCK_WORK:-/workspace/tmp/flock-s8-block}; mkdir -p $FW || exit 1
STATEMENTS=${STATEMENTS:-"4096:1 4096:8 14336:1"}; CPU_STATEMENTS=${CPU_STATEMENTS:-"4096:1"}; ROWS=${ROWS:-8}
SANITY=${SANITY:-1}; WARM=${WARM:-1}; RUNS=${RUNS:-3}; NS=$((WARM + RUNS)); SM=${SM:-90}; STEPS=${STEPS:-"build stage sanity gpu cpu"}
export FLOCK_WORK=$FW
. verity/ml/flock/pod/cuda-13.3.sh
export PATH=$CUDA_TK/bin:$HOME/.cargo/bin:$PATH LD_LIBRARY_PATH=$CUDA_TK/lib64:${LD_LIBRARY_PATH:-}
export FC_COINS=os FC_PIPELINE_DEPTH=${FC_PIPELINE_DEPTH:-4} FC_HOST_PREPIN=${FC_HOST_PREPIN:-1}
CPUS=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0)))))'); NC=$(echo $CPUS | tr , '\n' | wc -l)
VC=${VCPUS:-4}; PCPU=$(echo $CPUS | cut -d, -f1-$((NC - VC))); VCPU=$(echo $CPUS | cut -d, -f$((NC - VC + 1))-)
B=$FW/flock-gpu-link/flock/target/release/flock-circuit; PY=$FW/flock-circuit/py/bin/python
has() { case " $STEPS " in *" $1 "*) return 0 ;; esac; return 1; }
{ echo "cpus=$CPUS prover=$PCPU verifier=$VCPU"; lscpu | grep -E 'Model name|^CPU\(s\)'; free -g | head -2
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv; command -v dcgmi nsys ncu; } > $O/host.txt 2>&1

if has build; then
  echo "== build $(date -u +%T)"
  MODE=build GPU=1 SM=$SM FLOCK_WORK=$FW RESEARCH_RUN_DIR=$O/build bash verity/ml/flock/pod/60-circuit.sh > $O/build.log 2>&1 \
    || { tail -40 $O/build.log $O/build/out/build.log 2>/dev/null; exit 1; }
  sha256sum $B | tee $O/binary.sha256
fi
[ -x $B ] || { echo "no flock-circuit at $B (STEPS without build needs a FLOCK_WORK that has one)"; exit 1; }
UV=$(command -v uv || echo $HOME/.local/bin/uv)
[ -x $UV ] || { curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1; UV=$HOME/.local/bin/uv; }
[ -x $PY ] || { $UV venv -q --python 3.12 $(dirname $(dirname $PY)) && $UV pip install -q --python $PY numpy blake3; } || exit 1
export PYTHONPATH=$REPO:$REPO/verity/ml/flock/python:$REPO/tools/verity/circuit_check/src:$REPO/integrations/vllm

# spec K:N[:R]: "K N R NAME"
spec() { local k n r; IFS=: read -r k n r <<< "$1"; r=${r:-$ROWS}; echo "$k $n $r k$k-n$n-r$r"; }
# stage SPEC: the statement of N instances of R rows at K into $FW/data/NAME (circuit.txt, inst-N.bin, pub-N.bin, rec.json)
stage() {
  local k n r name; read -r k n r name <<< "$(spec $1)"
  local d=$FW/data/$name o=$O/stage/$name
  mkdir -p $d $o
  [ -f $d/rec.json ] && [ -f $d/inst-$n.bin ] && { cp $d/rec.json $o/rec.json; return 0; }   # staged by an earlier run
  K=$k ROWS=$r N=$n FLOCK_STAGE_CACHE=$d/cache FLOCK_OWN_ROWS=1 FLOCK_PACK_WORDS=1 /usr/bin/time -v -o $o/time-stage.txt \
    $PY -m verity_flock.class_statement --program-module $REPO/benchmarks/flock_s8_block/program.py:program \
    --partition q-template-instance --out $o/out --stage-only --batch fixed --n $n --max-ands $((1 << 34)) \
    --max-statement-bits $((1 << 35)) --batch-ands $((1 << 35)) --selftest 0 > $o/stage.log 2>&1 || { tail -5 $o/stage.log; return 1; }
  local C=$(ls -td $d/cache/*/ | head -1)
  for f in circuit.txt inst-$n.bin pub-$n.bin rec.json; do ln -f $C/$f $d/$f; done
  cp $C/rec.json $o/rec.json; du -sb $d/circuit.txt $d/inst-$n.bin $d/pub-$n.bin > $o/sizes.txt
}
if has stage; then
  echo "== stage $(date -u +%T)"
  for kn in 64:2 $STATEMENTS; do stage $kn || exit 1; done
fi

free_port() { python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
wait_serving() { for i in $(seq 1800); do grep -q SERVING $1 && return 0; kill -0 $2 2>/dev/null || return 1; sleep 1; done; return 1; }
stamp() { while IFS= read -r l; do printf '%s\t%s\n' "$(date +%s.%N)" "$l"; done; }
# gpu_sample DIR: until DIR/stop exists, the GPU's utilization, memory, SM clock and power (nvidia-smi) each ~0.2 s, stamped,
# then the prover's cores' busy jiffies (/proc/stat over PCPU); and where dcgmi runs its SM-active, tensor-pipe-active and
# DRAM-active shares (fields 1002, 1004, 1005), stamped
gpu_sample() {
  rm -f $1/stop
  local P2=; command -v dcgmi > /dev/null && { dcgmi dmon -e 1002,1004,1005 -d 200 2>&1 | stamp > $1/dcgm.tsv & P2=$!; }
  while [ ! -e $1/stop ]; do
    printf '%s\t%s, %s\n' "$(date +%s.%N)" "$(nvidia-smi --query-gpu=utilization.gpu,memory.used,clocks.sm,power.draw \
      --format=csv,noheader,nounits -i ${GPU_LEASE_UUID:-0} 2>&1)" \
      "$(awk -v c=",$PCPU," '$1 ~ /^cpu[0-9]/ && index(c, "," substr($1, 4) ",") {b += $2 + $3 + $4 + $7 + $8 + $9} END {print b}' /proc/stat)"
    sleep 0.2
  done > $1/smi.tsv
  [ -n "$P2" ] && { kill $P2 2>/dev/null; wait $P2 2>/dev/null; }   # dcgmi ends on its next write's SIGPIPE
}
# session SPEC MODE: a verifier process and WARM + RUNS prover sessions against it, into $O/MODE/NAME; MODE gpu or cpu
session() {
  local k n r name mode=$2 G=; read -r k n r name <<< "$(spec $1)"
  local d=$FW/data/$name o=$O/$2/$name
  case $mode in gpu*) G=--gpu ;; esac
  mkdir -p $o; rm -rf $d/sessions-$mode
  local PORT=$(free_port) PIN=$(sha512sum $d/circuit.txt | cut -d' ' -f1)
  taskset -c $VCPU $B serve --zk --listen 127.0.0.1:$PORT --out $d/sessions-$mode --instances $d/pub-$n.bin --circuit $d/circuit.txt \
    --pin $PIN --operator loopback --concurrency $((NS + 1)) > $o/serve.log 2>&1 &
  local SP=$!
  wait_serving $o/serve.log $SP || { tail -20 $o/serve.log; return 1; }
  gpu_sample $o & local GS=$!
  RAYON_NUM_THREADS=$((NC - VC)) FC_VERIFY_AHEAD=$NS taskset -c $PCPU /usr/bin/time -v -o $o/time-prove.txt ${WRAP:-} $B prove --zk $G \
    --verifier 127.0.0.1:$PORT --instances $d/inst-$n.bin --circuit $d/circuit.txt --warm $WARM --runs $RUNS \
    2> $o/prove.err | stamp > $o/prove.tsv
  echo "prove $name $mode exit ${PIPESTATUS[0]}"
  touch $o/stop; wait $GS
  kill $SP; wait $SP 2>/dev/null
  grep -h panicked $o/prove.err | head -3
  grep -q '"accepted":true' $o/prove.tsv
}

if has sanity; then
  echo "== sanity $(date -u +%T)"
  session 64:2 cpu || exit 1
fi
NR=$FW/nsys-root; NSB=$NR/opt/nvidia/nsight-systems/2026.1.3/bin/nsys
if has gpu && [ "${NSYS:-0}" = 1 ] && [ ! -x $NSB ]; then
  mkdir -p $NR/debs && (cd $NR/debs && { apt-get download nsight-systems-2026.1.3 2>/dev/null || curl -fsSLO \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/nsight-systems-2026.1.3_2026.1.3.425-1_amd64.deb; }) \
    && for f in $NR/debs/*.deb; do dpkg-deb -x $f $NR; done
  [ -x $NSB ] || { echo "nsys unpack failed: NSYS=0"; NSYS=0; }
fi
if has gpu; then
  echo "== gpu $(date -u +%T)"
  export -f session spec gpu_sample free_port wait_serving stamp
  export B O FW PCPU VCPU NC VC NS WARM RUNS STATEMENTS SANITY ROWS NSB NSYS NSYS_STATEMENT
  GPU_LEASE_WHO=${GPU_LEASE_WHO:-flock-s8-block} gpu-lease 1 --wait --max-min ${LEASE_MIN:-20} -- bash -c '
    nvidia-smi -L > $O/lease-gpu.txt
    [ "${SANITY:-1}" = 1 ] && session 64:2 gpu
    for kn in $STATEMENTS; do session $kn gpu; done
    if [ "${NSYS:-0}" = 1 ]; then
      rm -rf $FW/nsys-reps; mkdir -p $FW/nsys-reps
      WARM=0 RUNS=2 NS=2 WRAP="$NSB profile -t cuda --sample=none --cpuctxsw=none -o $FW/nsys-reps/prove-%p" \
        session ${NSYS_STATEMENT:-${STATEMENTS%% *}} gpu-nsys
    fi'
  if [ "${NSYS:-0}" = 1 ]; then
    mkdir -p $O/nsys; ls -la $FW/nsys-reps > $O/nsys/reps.txt 2>&1
    for f in $FW/nsys-reps/*.nsys-rep; do
      $NSB stats -q -r cuda_gpu_kern_sum,cuda_gpu_mem_time_sum -f csv -o $O/nsys/$(basename ${f%.nsys-rep}) $f >> $O/nsys/stats.log 2>&1
    done
  fi
fi
if has cpu; then
  echo "== cpu $(date -u +%T)"
  for kn in $CPU_STATEMENTS; do session $kn cpu; done
fi
echo "== summary $(date -u +%T)"
python3 $REPO/benchmarks/flock_s8_block/summary.py $O | tee $O/summary.json
