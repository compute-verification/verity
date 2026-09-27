#!/usr/bin/env bash
# flock-circuit: verity/flock-circuit on a pod. Builds flock-gpu-link's checkout (20-gpu-link.sh MODE=build), installs the
# circuit prover (cuda_circuit_patch.py) and builds flock-circuit; stages each input set (inputs/*.tar, research run --send)
# with verity_flock.circuit (the prover's inst-N.bin, the verifier's own pub-N.bin); then per set:
#   MODE=selftest   CPU and GPU selftests at SELF_N instances
#   MODE=bench      loopback sessions (a verifier process on 127.0.0.1 serving its own public file) at each of POINTS
#   MODE=build      nothing more
# env: SELF_N=4  POINTS="64 256"  WARM=1 RUNS=3  GPU=1  SM (from nvidia-smi)  CPU_SELF=1 (0: skip the CPU selftest)
#      RECORD=1 (selftest: every case's server record and proofs under out/<set>/records-{cpu,gpu})
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
REPO=$(pwd); I=${RESEARCH_RUN_DIR:-/tmp/fc}/inputs; O=${RESEARCH_RUN_DIR:-/tmp/fc}/out; W=/workspace/flock-circuit; mkdir -p $O $W
MODE=${MODE:-selftest}; G=${GPU:-1}
SM=${SM:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d .)}; SM=${SM:-90}
MODE=build GPU=$G SM=$SM bash backends/flock/pod/20-gpu-link.sh > $O/build.log 2>&1 || { tail -30 $O/build.log; exit 1; }
source $HOME/.cargo/env; C13=${CUDA_TK:-/usr/local/cuda-13.3}; export PATH=$C13/bin:$PATH NVCC=$C13/bin/nvcc LD_LIBRARY_PATH=$C13/compat:$C13/lib64:${LD_LIBRARY_PATH:-}
read -r Q PER_ < /sys/fs/cgroup/cpu.max 2>/dev/null || { Q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo -1); PER_=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 100000); }
TH=$( { [ "$Q" = max ] || [ "$Q" = -1 ]; } && nproc || echo $(( (Q + PER_ - 1) / PER_ )) ); export RAYON_NUM_THREADS=$TH
F=/workspace/flock-gpu-link/flock; FEAT="--features sha512 --features glue"
# SHA-512 Merkle trees, leaves and nodes, every Ligerito level (64-byte digests): the Rust side on every build, the device side
# after the circuit prover is generated (it widens both provers' trees)
(cd $F && rm -f crates/flock-merkle/src/hm96.rs && git apply $REPO/backends/flock/flock-sha512-b684b12.patch \
  && git apply $REPO/backends/flock/flock-glue-b684b12.patch) || exit 1
if [ $G = 1 ]; then (cd $F && python3 $REPO/backends/flock/cuda_circuit_patch.py && python3 $REPO/backends/flock/cuda_sha512_patch.py) || exit 1; FEAT="$FEAT --features gpu"; fi
# the selftest's harness injects seeds (determinism check); a proving build (bench, cells) never has the feature
[ "$MODE" = selftest ] && FEAT="$FEAT --features seed-injection"
(cd $F && cargo build --release -p flock-live $FEAT --bin flock-circuit -j $TH) > $O/build-circuit.txt 2>&1 || { tail -40 $O/build-circuit.txt; exit 1; }
B=$F/target/release/flock-circuit; sha256sum $B | tee $O/binary.sha256
[ "$MODE" = build ] && exit 0
[ -x $HOME/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
[ -x $W/py/bin/python ] || { $HOME/.local/bin/uv venv -q --python 3.12 $W/py && $HOME/.local/bin/uv pip install -q --python $W/py/bin/python numpy blake3; }
export PATH=$W/py/bin:$PATH PYTHONPATH=$REPO/packages/verity/src:$REPO/backends/numerical/python:$REPO/backends/flock/python:$REPO/integrations/vllm
lscpu | grep -E 'Model name|^CPU\(s\)' | tee $O/host.txt; echo threads $TH | tee -a $O/host.txt; nvidia-smi -L 2>/dev/null | tee -a $O/host.txt
for tarf in $I/*.tar; do
  t=$(basename $tarf .tar); S=$W/sets/$t; rm -rf $S && mkdir -p $S && tar -xf $tarf -C $S || exit 1
  SET=$(dirname $(find $S -name manifest.json | head -1)); D=$O/$t; mkdir -p $D
  if [ "$MODE" = selftest ]; then
    N=${SELF_N:-4}
    python3 -m verity_flock.circuit --input-set $SET --n $N --out $D/stage > $D/stage.json 2> $D/stage.err || { tail -3 $D/stage.err; continue; }; cat $D/stage.json
    if [ "${CPU_SELF:-1}" = 1 ]; then
      $B selftest --instances $D/stage/inst-$N.bin --circuit $D/stage/circuit.txt --tables $D/stage/tables --forged $D/stage/circuit-forged.txt ${RECORD:+--record-dir $D/records-cpu} > $D/selftest-cpu.txt 2> $D/selftest-cpu.err
      grep SELFTEST $D/selftest-cpu.txt; grep '"pass":false' $D/selftest-cpu.txt | cut -c1-300
    fi
    [ $G = 1 ] && { $B selftest --gpu --instances $D/stage/inst-$N.bin --circuit $D/stage/circuit.txt --tables $D/stage/tables --forged $D/stage/circuit-forged.txt ${RECORD:+--record-dir $D/records-gpu} > $D/selftest-gpu.txt 2> $D/selftest-gpu.err;
                    grep SELFTEST $D/selftest-gpu.txt; grep '"pass":false' $D/selftest-gpu.txt | cut -c1-300; grep -h GPUPHASE $D/selftest-gpu.err | head -2 | cut -c1-600; }
  fi
  if [ "$MODE" = bench ]; then
    python3 -m verity_flock.circuit --input-set $SET --n ${POINTS:-64} --out $D/stage | tee $D/stage.json || exit 1
    PIN=$(sha512sum $D/stage/circuit.txt | cut -d' ' -f1)
    for n in ${POINTS:-64}; do
      $B serve --listen 127.0.0.1:7501 --out $D/sessions-$n --instances $D/stage/pub-$n.bin --circuit $D/stage/circuit.txt --tables $D/stage/tables --pin $PIN --operator loopback > $D/serve-$n.log 2>&1 &
      SP=$!
      for i in $(seq 900); do grep -q SERVING $D/serve-$n.log && break; sleep 1; done
      G_=; [ $G = 1 ] && G_=--gpu
      $B prove --verifier 127.0.0.1:7501 --instances $D/stage/inst-$n.bin --circuit $D/stage/circuit.txt --tables $D/stage/tables $G_ --warm ${WARM:-1} --runs ${RUNS:-3} > $D/prove-$n.txt 2> $D/prove-$n.err
      grep -h "^LIVE" $D/prove-$n.txt | python3 -c "
import json, sys
for l in sys.stdin:
    v = json.loads(l[5:])
    print('LIVE', '$t', v['instances'], 'warm' if v['warm'] else 'timed', 'acc', v['accepted'], 'e2e %.3f' % v['e2e_s'], 'wit %.3f' % v['witness_s'],
          'prove', ['%.3f' % x for x in v['prove_s']], 'buckets', [{k: round(x, 4) for k, x in b.items() if isinstance(x, float)} for b in v['buckets']][:1])
"
      grep -h "panicked" $D/prove-$n.err | head -3
      kill $SP; wait $SP 2>/dev/null
    done
  fi
done
true
