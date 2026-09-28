#!/usr/bin/env bash
# flock-ir-lowering: flock-ir-block (verity/flock-ir-block/v1) on the GPU. Inputs (research run --send): net-<t>.txt and
# inst-<t>-<n>.bin per template t (verity_flock.templates.<module>.stage, renamed; the verifier's own staging), and for a cut template
# the MUFU tables rsq.u32 / sqrt.u32 / rcp.u32 (verity_flock.ir_lower.write_tables). Builds flock-gpu-link's checkout
# (20-gpu-link.sh MODE=build), then flock-ir-block; then CPU and GPU selftests on the SELF files and loopback sessions
# (WARM warm-ups + RUNS timed) on the SWEEP files.
# env: SELF="rope-32 silu-2"  SWEEP="rope-256 rope-1024 silu-8 silu-32"  WARM=1 RUNS=3  GPU=1  SM=90
set -uxo pipefail
SRC=$(pwd); I=$RESEARCH_RUN_DIR/inputs; O=$RESEARCH_RUN_DIR/out; mkdir -p $O
MODE=build GPU=${GPU:-1} bash verity/ml/flock/pod/20-gpu-link.sh || exit 1
source $HOME/.cargo/env; C13=/usr/local/cuda-13.3; export PATH=$C13/bin:$PATH NVCC=$C13/bin/nvcc LD_LIBRARY_PATH=$C13/compat:$C13/lib64:${LD_LIBRARY_PATH:-}
read -r Q PER < /sys/fs/cgroup/cpu.max 2>/dev/null || { Q=max; PER=100000; }
TH=$([ "$Q" = max ] && nproc || echo $(( Q / PER ))); export RAYON_NUM_THREADS=$TH
F=/workspace/flock-gpu-link/flock
FEAT=; G=; [ "${GPU:-1}" = 1 ] && { FEAT="--features gpu"; G=--gpu; }
(cd $F && cargo build --release -p flock-live $FEAT --bin flock-ir-block -j $TH) > $O/build-ir.txt 2>&1 || { tail -40 $O/build-ir.txt; exit 1; }
P=$F/target/release/flock-ir-block; sha256sum $P $I/* | tee $O/binary-ir.sha256
for s in ${SELF-rope-32 silu-2}; do
  t=${s%%-*}; NET=$I/net-$t.txt
  $P selftest --precheck --tables $I --instances $I/inst-$s.bin --netlist $NET > $O/selftest-cpu-$s.txt 2>&1; grep -h 'PRECHECK\|SELFTEST' $O/selftest-cpu-$s.txt | cut -c1-400
  [ -n "$G" ] && { $P selftest --gpu --tables $I --instances $I/inst-$s.bin --netlist $NET > $O/selftest-gpu-$s.txt 2>&1; grep -h 'SELFTEST\|"pass":false' $O/selftest-gpu-$s.txt | cut -c1-400; }
done
for s in ${SWEEP-rope-256 rope-1024 silu-8 silu-32}; do
  t=${s%%-*}; NET=$I/net-$t.txt; PIN=$(sha256sum $NET | cut -d' ' -f1)
  $P serve --listen 127.0.0.1:7401 --out $O/sessions-loop-$s --tables $I --instances $I/inst-$s.bin --netlist $NET --pin $PIN --operator loopback > $O/serve-loop-$s.log 2>&1 &
  SP=$!
  for i in $(seq 900); do grep -q SERVING $O/serve-loop-$s.log && break; sleep 1; done
  $P prove --verifier 127.0.0.1:7401 --tables $I --instances $I/inst-$s.bin --netlist $NET $G --warm ${WARM:-1} --runs ${RUNS:-3} > $O/prove-loop-$s.txt 2>&1
  grep -h "^LIVE\|panicked" $O/prove-loop-$s.txt | cut -c1-900
  kill $SP; wait $SP 2>/dev/null
done
true
