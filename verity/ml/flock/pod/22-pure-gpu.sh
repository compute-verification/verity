#!/usr/bin/env bash
# flock-gpu-link: flock-pure-gpu (verity/flock-pure-block/v1) on the GPU. Inputs (research run --send): net-<rel>.txt and
# inst-<n>.bin (flock-backend's verity_flock.lowering / .instances). Builds with 20-gpu-link.sh, then: CPU and GPU selftests
# at SELF="8 64", loopback sessions at SWEEP="2048 4096 8192" (WARM warm-ups + RUNS timed each), and, with VERIFIER=host:port,
# REMOTE_RUNS sessions at REMOTE_N against that verifier (which serves the same instance file).
set -uxo pipefail
SRC=$(pwd); I=$RESEARCH_RUN_DIR/inputs; O=$RESEARCH_RUN_DIR/out${TAG:+-$TAG}; mkdir -p $O
REL=${REL:-bf16-hopper}; NET=$I/net-$REL.txt; PIN=$(sha256sum $NET | cut -d' ' -f1)
MODE=build GPU=${GPU:-1} bash verity/ml/flock/pod/20-gpu-link.sh || exit 1
source $HOME/.cargo/env; C13=/usr/local/cuda-13.3; export LD_LIBRARY_PATH=$C13/compat:$C13/lib64:${LD_LIBRARY_PATH:-}
read -r Q PER < /sys/fs/cgroup/cpu.max 2>/dev/null || { Q=max; PER=100000; }
export RAYON_NUM_THREADS=$([ "$Q" = max ] && nproc || echo $(( Q / PER )))
P=/workspace/flock-gpu-link/flock/target/release/flock-pure-gpu
G=; [ "${GPU:-1}" = 1 ] && G=--gpu
if [ "${ROLE:-prover}" = verifier ]; then
  $P serve --listen 0.0.0.0:${PORT:-7300} --out $O/sessions --instances $I/inst-${REMOTE_N:-8192}.bin --netlist $NET --pin $PIN \
    --operator ${OPERATOR:-flock-gpu-link-ver} ${SESSIONS:+--sessions $SESSIONS} 2>&1 | tee $O/serve.txt
  exit 0
fi
for n in ${SELF-8 64}; do
  $P selftest --instances $I/inst-$n.bin --netlist $NET > $O/selftest-cpu-$n.txt 2>&1; grep -h SELFTEST $O/selftest-cpu-$n.txt
  [ -n "$G" ] && { $P selftest --gpu --instances $I/inst-$n.bin --netlist $NET > $O/selftest-gpu-$n.txt 2>&1; grep -h 'SELFTEST\|"pass":false' $O/selftest-gpu-$n.txt | cut -c1-400; }
done
if [ -n "${VERIFIER:-}" ]; then
  python3 -c "
import socket,time,statistics,sys
h,p='${VERIFIER_SSH:-}'.split(':') if '${VERIFIER_SSH:-}' else (None,None)
if h:
    ts=[]
    for _ in range(20):
        t=time.perf_counter(); s=socket.create_connection((h,int(p)),timeout=5); ts.append((time.perf_counter()-t)*1e3); s.close(); time.sleep(0.05)
    print('RTT_TCP_CONNECT_MS', {'median': statistics.median(ts), 'min': min(ts), 'n': len(ts), 'target': '${VERIFIER_SSH:-}'})
" | tee $O/rtt.txt
  $P prove --verifier $VERIFIER --instances $I/inst-${REMOTE_N:-8192}.bin --netlist $NET $G --warm ${WARM:-1} --runs ${REMOTE_RUNS:-4} \
    --dump $O/dump-remote > $O/prove-remote-${REMOTE_N:-8192}.txt 2>&1
  grep -h "^LIVE\|panicked" $O/prove-remote-${REMOTE_N:-8192}.txt | cut -c1-1200
fi
for n in ${SWEEP-2048 4096 8192}; do
  $P serve --listen 127.0.0.1:7301 --out $O/sessions-loop-$n --instances $I/inst-$n.bin --netlist $NET --pin $PIN --operator loopback-selftest > $O/serve-loop-$n.log 2>&1 &
  SP=$!
  for i in $(seq 900); do grep -q SERVING $O/serve-loop-$n.log && break; sleep 1; done
  $P prove --verifier 127.0.0.1:7301 --instances $I/inst-$n.bin --netlist $NET $G --warm ${WARM:-1} --runs ${RUNS:-3} > $O/prove-loop-$n.txt 2>&1
  grep -h "^LIVE\|panicked" $O/prove-loop-$n.txt | cut -c1-1200
  kill $SP; wait $SP 2>/dev/null
done
true
