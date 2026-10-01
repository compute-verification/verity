#!/usr/bin/env bash
# Reads at a private index (value hiding, item 4; hidden_reads.py): each gadget against its native reference, the depth-9 Merkle
# opening and the 120-read mux tree as flock-circuit statements proved in loopback sessions on the CPU, M0 and --zk, at two sizes
# each, then every construction's gate counts. Run from the verity tree (research run --cwd source); CPU only, no GPU build.
# Outputs under out/ (flat): check.json, stage-<unit>.json, live.tsv (one LIVE record per proof: unit, instances, mode, json),
# prove-*/serve-* logs, counts.json, host.txt and the core samplers (others_cores.txt; slice_cores.tsv: load on the job's own
# cores from anything else). Staged statements and session records go under FLOCK_WORK/scratch/<run id>, removed at exit.
# env: CPUS=48 (the first CPUS cores of free 16-core slices, cpu-slices.sh)  MERKLE="8 32" MUX="4 16" (instances: m = 25, 27)
#      MODES="m0 zk"  WARM=1 RUNS=2  HEAVY=1 (counts: today's read of the 2^27 tables too)
#      FLOCK_WORK=/workspace/cache/flock-hidden-reads (the build and its venv, kept for the next attempt)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
if [ -z "${HR_PINNED:-}" ]; then
  . backends/flock/pod/cpu-slices.sh
  hold_slices "" "${CPUS:-48}" || exit 1
  export HR_PINNED=$SLICE_CPUS RAYON_NUM_THREADS=$(echo $SLICE_CPUS | tr ',' '\n' | wc -l)
  exec taskset -c $SLICE_CPUS bash "$0" "$@"
fi
REPO=$(pwd); O=${RESEARCH_RUN_DIR:-/tmp/hidden-reads}/out; mkdir -p $O
export FLOCK_WORK=${FLOCK_WORK:-/workspace/cache/flock-hidden-reads}; mkdir -p $FLOCK_WORK
S=$FLOCK_WORK/scratch/${RESEARCH_RUN_ID:-local-$$}; rm -rf $S; mkdir -p $S
unset CARGO_TARGET_DIR
{ lscpu | grep -E 'Model name|^CPU\(s\)'; echo "pinned=$HR_PINNED rayon=$RAYON_NUM_THREADS slices=${SLICES:-} slice_wait_s=${SLICE_WAIT_S:-}"
  free -g | head -2; cat /proc/loadavg; } > $O/host.txt 2>&1
python3 backends/flock/pod/gemm_slowdown.py sample --pgid $(ps -o pgid= $$ | tr -d ' ') --out $O/others_cores.txt --every 10 &
SAMPLER=$!
python3 backends/flock/pod/gemm_hill.py cores --cpus $HR_PINNED --out $O/slice_cores.tsv --every 10 &
CORES=$!
trap 'kill $SAMPLER $CORES 2>/dev/null; rm -rf $S' EXIT
MODE=build GPU=0 bash backends/flock/pod/60-circuit.sh || exit 1
B=$FLOCK_WORK/flock-gpu-link/flock/target/release/flock-circuit; W=$FLOCK_WORK/flock-circuit; UV=$(command -v uv || echo $HOME/.local/bin/uv)
[ -x $W/py/bin/python ] || { $UV venv -q --python 3.12 $W/py && $UV pip install -q --python $W/py/bin/python numpy blake3; } || exit 1
export PATH=$W/py/bin:$PATH PYTHONPATH=$REPO/packages/verity/src:$REPO/backends/numerical/python:$REPO/backends/flock/python:$REPO/integrations/vllm
H="python3 backends/flock/pod/hidden_reads.py"
$H check > $O/check.json || { cat $O/check.json; exit 1; }
PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
for unit in merkle mux; do
  case $unit in merkle) NS=${MERKLE:-8 32} ;; mux) NS=${MUX:-4 16} ;; esac
  D=$S/$unit
  $H stage --unit $unit --n $NS --out $D > $O/stage-$unit.json 2> $O/stage-$unit.err || { tail -5 $O/stage-$unit.err; exit 1; }
  PIN=$(sha512sum $D/circuit.txt | cut -d' ' -f1)
  for n in $NS; do for md in ${MODES:-m0 zk}; do
    Z=; [ $md = zk ] && Z=--zk
    L=$O/serve-$unit-$n-$md.log
    $B serve $Z --listen 127.0.0.1:$PORT --out $S/sessions-$unit-$n-$md --instances $D/pub-$n.bin --circuit $D/circuit.txt \
      --tables $D/tables --pin $PIN --operator loopback > $L 2>&1 &
    SP=$!
    for i in $(seq 900); do grep -q SERVING $L && break; kill -0 $SP 2>/dev/null || break; sleep 1; done
    $B prove $Z --verifier 127.0.0.1:$PORT --instances $D/inst-$n.bin --circuit $D/circuit.txt --tables $D/tables \
      --warm ${WARM:-1} --runs ${RUNS:-2} > $O/prove-$unit-$n-$md.txt 2> $O/prove-$unit-$n-$md.err
    grep -h '^LIVE' $O/prove-$unit-$n-$md.txt | sed "s/^LIVE\t/$unit\t$n\t$md\t/" >> $O/live.tsv
    grep -h panicked $O/prove-$unit-$n-$md.err | head -3
    kill $SP; wait $SP 2>/dev/null
  done; done
done
HV=; [ "${HEAVY:-1}" = 1 ] && HV=--heavy
$H counts $HV --out $O/counts.json > $O/counts.txt 2>&1 || { tail -5 $O/counts.txt; exit 1; }
true
