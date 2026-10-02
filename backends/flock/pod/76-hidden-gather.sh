#!/usr/bin/env bash
# A read at a private index of a registered table (verity_flock.hidden_gather; hidden_gather.py): its tests, the depth-22
# tanh_rn read included (FLOCK_HIDDEN_GATHER_TANH), each table's opening by part, then one opening per unit as a flock-circuit
# statement proved in loopback sessions on the CPU, M0 and --zk, at three sizes per table, and the summary. Run from the verity
# tree (research run --cwd source); CPU only, no GPU build. Outputs under out/ (flat): pytest.txt, parts.json,
# stage-<table>.json, live.tsv (one LIVE record per proof: table, instances, mode, json), prove-*/serve-* logs, summary.json,
# host.txt and the core samplers (as 75-hidden-reads.sh's). Staged statements and session records go under
# FLOCK_WORK/scratch/<run id>, removed at exit.
# env: CPUS=48 (the first CPUS cores of free 16-core slices, cpu-slices.sh)  GELU="8 32 128" TANH="8 32 128" (instances: one
#      per 2^23-bit block, m = 26, 28, 30)  MODES="m0 zk"  WARM=1 RUNS=2
#      FLOCK_WORK=/workspace/cache/flock-hidden-reads (75-hidden-reads.sh's build and venv, kept for the next attempt)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
if [ -z "${HG_PINNED:-}" ]; then
  . backends/flock/pod/cpu-slices.sh
  hold_slices "" "${CPUS:-48}" || exit 1
  export HG_PINNED=$SLICE_CPUS RAYON_NUM_THREADS=$(echo $SLICE_CPUS | tr ',' '\n' | wc -l)
  exec taskset -c $SLICE_CPUS bash "$0" "$@"
fi
REPO=$(pwd); O=${RESEARCH_RUN_DIR:-/tmp/hidden-gather}/out; mkdir -p $O
export FLOCK_WORK=${FLOCK_WORK:-/workspace/cache/flock-hidden-reads}; mkdir -p $FLOCK_WORK
S=$FLOCK_WORK/scratch/${RESEARCH_RUN_ID:-local-$$}; rm -rf $S; mkdir -p $S
unset CARGO_TARGET_DIR
export PYTHONDONTWRITEBYTECODE=1
{ lscpu | grep -E 'Model name|^CPU\(s\)'; echo "pinned=$HG_PINNED rayon=$RAYON_NUM_THREADS slices=${SLICES:-} slice_wait_s=${SLICE_WAIT_S:-}"
  free -g | head -2; cat /proc/loadavg; } > $O/host.txt 2>&1
python3 backends/flock/pod/gemm_slowdown.py sample --pgid $(ps -o pgid= $$ | tr -d ' ') --out $O/others_cores.txt --every 10 &
SAMPLER=$!
python3 backends/flock/pod/gemm_hill.py cores --cpus $HG_PINNED --out $O/slice_cores.tsv --every 10 &
CORES=$!
trap 'kill $SAMPLER $CORES 2>/dev/null; rm -rf $S' EXIT
MODE=build GPU=0 bash backends/flock/pod/60-circuit.sh || exit 1
B=$FLOCK_WORK/flock-gpu-link/flock/target/release/flock-circuit; W=$FLOCK_WORK/flock-circuit; UV=$(command -v uv || echo $HOME/.local/bin/uv)
[ -x $W/py/bin/python ] || { $UV venv -q --python 3.12 $W/py && $UV pip install -q --python $W/py/bin/python numpy blake3; } || exit 1
[ -x $W/py/bin/pytest ] || $UV pip install -q --python $W/py/bin/python pytest || exit 1
export PATH=$W/py/bin:$PATH PYTHONPATH=$REPO/packages/verity/src:$REPO/backends/numerical/python:$REPO/backends/flock/python:$REPO/integrations/vllm
(cd backends/flock && FLOCK_HIDDEN_GATHER_TANH=1 python3 -m pytest tests/test_hidden_gather.py -q -p no:cacheprovider -rs) > $O/pytest.txt 2>&1 \
  || { tail -40 $O/pytest.txt; exit 1; }
H="python3 backends/flock/pod/hidden_gather.py"
$H parts --out $O/parts.json > $O/parts.txt 2>&1 || { tail -5 $O/parts.txt; exit 1; }
PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
for table in gelu_tanh_bf16 tanh_rn; do
  case $table in gelu_tanh_bf16) NS=${GELU:-8 32 128} ;; tanh_rn) NS=${TANH:-8 32 128} ;; esac
  D=$S/$table
  $H stage --table $table --n $NS --out $D > $O/stage-$table.json 2> $O/stage-$table.err || { tail -5 $O/stage-$table.err; exit 1; }
  PIN=$(sha512sum $D/circuit.txt | cut -d' ' -f1)
  for n in $NS; do for md in ${MODES:-m0 zk}; do
    Z=; [ $md = zk ] && Z=--zk
    L=$O/serve-$table-$n-$md.log
    $B serve $Z --listen 127.0.0.1:$PORT --out $S/sessions-$table-$n-$md --instances $D/pub-$n.bin --circuit $D/circuit.txt \
      --tables $D/tables --pin $PIN --operator loopback > $L 2>&1 &
    SP=$!
    for i in $(seq 900); do grep -q SERVING $L && break; kill -0 $SP 2>/dev/null || break; sleep 1; done
    $B prove $Z --verifier 127.0.0.1:$PORT --instances $D/inst-$n.bin --circuit $D/circuit.txt --tables $D/tables \
      --warm ${WARM:-1} --runs ${RUNS:-2} > $O/prove-$table-$n-$md.txt 2> $O/prove-$table-$n-$md.err
    grep -h '^LIVE' $O/prove-$table-$n-$md.txt | sed "s/^LIVE\t/$table\t$n\t$md\t/" >> $O/live.tsv
    grep -h panicked $O/prove-$table-$n-$md.err | head -3
    kill $SP; wait $SP 2>/dev/null
  done; done
done
$H summary --runs ${RESEARCH_RUN_DIR:-/tmp/hidden-gather} --out $O/summary.json > $O/summary.txt 2>&1 || { tail -5 $O/summary.txt; exit 1; }
true
