#!/usr/bin/env bash
# Stage private circuits in universal-unit classes on a CPU node, one process per class (stage_class.py), each under
# /usr/bin/time -v when the node has it:
#   research run --on vy-cpu-2 --cwd source --project verity -- \
#     bash benchmarks/private_circuit/stage.sh CLS="256,64,32 512,64,32" GATES=inputs/F32MulFtz_v3.gates.json N=2
# env: CLS (space-separated G,N_IN,N_OUT)  GATES (a circuit's description; default a random one per class)  N=2 instances
#      KEEP (non-empty: keep the staged files under $OUT)  STAGE_DIR (stage each class into STAGE_DIR/G-N_IN-N_OUT, kept)
#      MAX_GIB (an address-space cap for each staging process, so it fails alone rather than crowd the node)  FLOCK_WORK=/workspace/jobs/pc-stage (the venv; never the run dir)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
REPO=$(pwd); O=$RESEARCH_RUN_DIR/out; W=${FLOCK_WORK:-/workspace/jobs/pc-stage}; mkdir -p $O $W
nproc; free -g | head -2
[ -x $HOME/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
exec 8>$W/venv.lock; flock 8
[ -x $W/py/bin/python ] || { $HOME/.local/bin/uv venv -q --python 3.12 $W/py && $HOME/.local/bin/uv pip install -q --python $W/py/bin/python numpy blake3; }
exec 8>&-
export PYTHONPATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/ml/flock/python:$REPO/integrations/vllm:$REPO/tools/verity/circuit_check/src
GA=(); [ -n "${GATES:-}" ] && GA=(--gates "$(cd "$RESEARCH_RUN_DIR" && realpath "$GATES")")
TIME=(); [ -x /usr/bin/time ] && TIME=(/usr/bin/time -v)
rc=0
[ -n "${MAX_GIB:-}" ] && ulimit -v $((MAX_GIB * 1024 * 1024))
for c in ${CLS:-256,64,32}; do
  d=$O/G${c//,/-}
  mkdir -p $d
  "${TIME[@]}" $W/py/bin/python benchmarks/private_circuit/stage_class.py --cls "$c" "${GA[@]}" --n ${N:-2} --out $d \
    ${KEEP:+--keep-stage} ${STAGE_DIR:+--stage-dir $STAGE_DIR/${c//,/-}} > $d/stage.log 2> $d/time.txt || rc=1
  tail -3 $d/stage.log; grep -E 'Maximum resident|Elapsed' $d/time.txt
done
exit $rc
