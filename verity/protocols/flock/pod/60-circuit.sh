#!/usr/bin/env bash
# flock-circuit: verity/flock-circuit on a pod. Builds flock-gpu-link's checkout (20-gpu-link.sh MODE=build), installs the
# circuit prover (cuda_circuit_patch.py) and builds flock-circuit; stages each input set (inputs/*.tar, research run --send)
# with verity_flock.circuit (the prover's inst-N.bin, the verifier's own pub-N.bin); then per set:
#   MODE=selftest   CPU and GPU selftests at SELF_N instances
#   MODE=bench      loopback sessions (a verifier process on 127.0.0.1 serving its own public file) at each of POINTS
#   MODE=build      nothing more
# env: SELF_N=4  POINTS="64 256"  WARM=1 RUNS=3  GPU=1  SM (from nvidia-smi)  CPU_SELF=1 (0: skip the CPU selftest)
#      RECORD=1 (selftest: every case's server record and proofs under out/<set>/records-{cpu,gpu})  TYPED=1 (the typed statement)
#      ZK=1 (selftest: Flock ZK's --zk statement, and zkaudit on the device's proofs)  MODES="m0 zk" (bench: each point in each mode;
#      zkseq: --zk with the host's zk work not overlapped, FC_ZK_SEQUENTIAL)  FC_COINS=tree (bench: m0's sessions on the
#      challenger's keyed coin tree, as every --zk session is: each point's server on its own challenger, bounded to WARM + RUNS
#      sessions, its records audited against the challenger's release log, out/<set>/challenger-N-MODE/audit.json; empty: the
#      binary's default, seed coins; a selftest build always seeds)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
export FC_COINS=${FC_COINS:-tree}
REPO=$(pwd); I=${RESEARCH_RUN_DIR:-/tmp/fc}/inputs; O=${RESEARCH_RUN_DIR:-/tmp/fc}/out; W=${FLOCK_WORK:-/workspace}/flock-circuit; mkdir -p $O $W
MODE=${MODE:-selftest}; G=${GPU:-1}; Z=; [ "${ZK:-0}" = 1 ] && Z=--zk; TY=; [ "${TYPED:-0}" = 1 ] && TY=--typed
SM=${SM:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d .)}; SM=${SM:-90}
MODE=build GPU=$G SM=$SM bash verity/protocols/flock/pod/20-gpu-link.sh > $O/build.log 2>&1 || { tail -30 $O/build.log; exit 1; }
source $HOME/.cargo/env; C13=${CUDA_TK:-/usr/local/cuda-13.3}; export PATH=$C13/bin:$PATH NVCC=$C13/bin/nvcc LD_LIBRARY_PATH=$C13/compat:$C13/lib64:${LD_LIBRARY_PATH:-}
read -r Q PER_ < /sys/fs/cgroup/cpu.max 2>/dev/null || { Q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo -1); PER_=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 100000); }
TH=$( { [ "$Q" = max ] || [ "$Q" = -1 ]; } && nproc || echo $(( (Q + PER_ - 1) / PER_ )) ); export RAYON_NUM_THREADS=$TH
F=${FLOCK_WORK:-/workspace}/flock-gpu-link/flock; FEAT="--features sha512 --features glue"
# SHA-512 Merkle trees, leaves and nodes, every Ligerito level (64-byte digests): the Rust side on every build, the device side
# after the circuit prover is generated (it widens both provers' trees)
(cd $F && rm -f crates/flock-merkle/src/hm96.rs && git apply $REPO/verity/protocols/flock/prover/flock-sha512-b684b12.patch \
  && git apply $REPO/verity/protocols/flock/prover/flock-glue-b684b12.patch) || exit 1
# Flock ZK: Ligerito level 0's hiding (flock-circuit --zk)
(cd $F && git apply $REPO/verity/protocols/flock/prover/flock-zk-b684b12.patch) || exit 1
# the session verifier's phase timers in the opening (session_verify.rs): clocks only, no check changes
(cd $F && git apply $REPO/verity/protocols/flock/prover/flock-vtime-b684b12.patch) || exit 1
if [ $G = 1 ]; then (cd $F && python3 $REPO/verity/protocols/flock/prover/cuda_circuit_patch.py && python3 $REPO/verity/protocols/flock/prover/cuda_sha512_patch.py) || exit 1; FEAT="$FEAT --features gpu"; fi
# the selftest's harness injects seeds (determinism check); a proving build (bench, cells) never has the feature
[ "$MODE" = selftest ] && FEAT="$FEAT --features seed-injection"
(cd $F && cargo build --release -p flock-live $FEAT --bin flock-circuit -j $TH) > $O/build-circuit.txt 2>&1 || { tail -40 $O/build-circuit.txt; exit 1; }
B=$F/target/release/flock-circuit; sha256sum $B | tee $O/binary.sha256
[ "$MODE" = build ] && exit 0
[ -x $HOME/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
[ -x $W/py/bin/python ] || { $HOME/.local/bin/uv venv -q --python 3.12 $W/py && $HOME/.local/bin/uv pip install -q --python $W/py/bin/python numpy blake3; }
export PATH=$W/py/bin:$PATH PYTHONPATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/protocols/flock/prover/python:$REPO/adapters/vllm
# the challengers' Lean, built while the sets stage: bench's --zk sessions, and m0's under FC_COINS=tree, take their coin trees
# from a challenger each (pod/challenger.sh)
CH=0; [ "$MODE" = bench ] && { [ "$FC_COINS" = tree ] || [[ " ${MODES:-m0} " == *" zk"* ]]; } && CH=1
if [ $CH = 1 ]; then
  export CHALLENGER_REPO=$REPO CHALLENGER_PY=python3 CHALLENGER_WORK=$W/challengers/${RESEARCH_RUN_ID:-local-$$}
  . verity/protocols/flock/prover/pod/challenger.sh; challenger_build
fi
lscpu | grep -E 'Model name|^CPU\(s\)' | tee $O/host.txt; echo threads $TH | tee -a $O/host.txt; nvidia-smi -L 2>/dev/null | tee -a $O/host.txt
for tarf in $I/*.tar; do
  t=$(basename $tarf .tar); S=$W/sets/$t; rm -rf $S && mkdir -p $S && tar -xf $tarf -C $S || exit 1
  SET=$(dirname $(find $S -name manifest.json | head -1)); D=$O/$t; mkdir -p $D
  if [ "$MODE" = selftest ]; then
    N=${SELF_N:-4}
    python3 -m verity_flock.circuit $TY --input-set $SET --n $N --out $D/stage > $D/stage.json 2> $D/stage.err || { tail -3 $D/stage.err; continue; }; cat $D/stage.json
    if [ "${CPU_SELF:-1}" = 1 ]; then
      $B selftest $Z --instances $D/stage/inst-$N.bin --circuit $D/stage/circuit.txt --tables $D/stage/tables --forged $D/stage/circuit-forged.txt ${RECORD:+--record-dir $D/records-cpu} > $D/selftest-cpu.txt 2> $D/selftest-cpu.err
      grep SELFTEST $D/selftest-cpu.txt; grep '"pass":false' $D/selftest-cpu.txt | cut -c1-300
    fi
    [ $G = 1 ] && { $B selftest $Z --gpu --instances $D/stage/inst-$N.bin --circuit $D/stage/circuit.txt --tables $D/stage/tables --forged $D/stage/circuit-forged.txt ${RECORD:+--record-dir $D/records-gpu} > $D/selftest-gpu.txt 2> $D/selftest-gpu.err;
                    grep SELFTEST $D/selftest-gpu.txt; grep '"pass":false' $D/selftest-gpu.txt | cut -c1-300; grep -h GPUPHASE $D/selftest-gpu.err | head -2 | cut -c1-600; }
    if [ -n "$Z" ] && [ $G = 1 ]; then
      $B zkaudit --zk --gpu --instances $D/stage/inst-$N.bin --circuit $D/stage/circuit.txt --tables $D/stage/tables > $D/zkaudit-gpu.txt 2> $D/zkaudit-gpu.err
      grep -h '"attributed"\|"complete"' $D/zkaudit-gpu.txt | cut -c1-300
    fi
  fi
  if [ "$MODE" = bench ]; then
    python3 -m verity_flock.circuit $TY --input-set $SET --n ${POINTS:-64} --out $D/stage | tee $D/stage.json || exit 1
    PIN=$(sha512sum $D/stage/circuit.txt | cut -d' ' -f1)
    [ $CH = 1 ] && { challenger_built || exit 1; }
    for n in ${POINTS:-64}; do for md in ${MODES:-m0}; do
      Zb=; [ $md = zk ] && Zb=--zk; ENVP=; [ $md = zkseq ] && { Zb=--zk; ENVP=FC_ZK_SEQUENTIAL=1; }
      # the session's coin tree from its own challenger: every --zk session, and m0's under FC_COINS=tree
      TREE_FROM=(); NS=()
      if [ -n "$Zb" ] || [ "$FC_COINS" = tree ]; then
        challenger_start $D/challenger-$n-$md || exit 1; NS=(--sessions $(( ${WARM:-1} + ${RUNS:-3} )))
      fi
      $B serve $Zb --listen 127.0.0.1:7501 --out $D/sessions-$n-$md --instances $D/stage/pub-$n.bin --circuit $D/stage/circuit.txt --tables $D/stage/tables --pin $PIN --operator loopback "${NS[@]}" "${TREE_FROM[@]}" > $D/serve-$n-$md.log 2>&1 &
      SP=$!
      for i in $(seq 900); do grep -q SERVING $D/serve-$n-$md.log && break; sleep 1; done
      G_=; [ $G = 1 ] && G_=--gpu
      env $ENVP $B prove $Zb --verifier 127.0.0.1:7501 --instances $D/stage/inst-$n.bin --circuit $D/stage/circuit.txt --tables $D/stage/tables $G_ --warm ${WARM:-1} --runs ${RUNS:-3} > $D/prove-$n-$md.txt 2> $D/prove-$n-$md.err
      grep -h "^LIVE" $D/prove-$n-$md.txt | python3 -c "
import json, sys
for l in sys.stdin:
    v = json.loads(l[5:])
    print('LIVE', '$t', '$md', v['instances'], 'warm' if v['warm'] else 'timed', 'acc', v['accepted'], 'e2e %.3f' % v['e2e_s'], 'wit %.3f' % v['witness_s'],
          'prove', ['%.3f' % x for x in v['prove_s']], 'buckets', [{k: round(x, 4) for k, x in b.items() if isinstance(x, float)} for b in v['buckets']][:1])
"
      grep -h "panicked" $D/prove-$n-$md.err | head -3
      # a bounded server exits after its last session's record; one short of its sessions is killed
      if [ ${#NS[@]} -gt 0 ]; then
        [ "$(grep -c '^LIVE' $D/prove-$n-$md.txt)" = "${NS[1]}" ] || kill $SP; wait $SP 2>/dev/null
        challenger_end $D/challenger-$n-$md; echo "challenger $n $md exit $?"
        challenger_audit $D/challenger-$n-$md $(ls -d $D/sessions-$n-$md/*/ | sed 's:/$::')
      else
        kill $SP; wait $SP 2>/dev/null
      fi
    done; done
  fi
done
true
