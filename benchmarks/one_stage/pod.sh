#!/usr/bin/env bash
# One-stage audit, run A0, on a CPU pod (research run --on POD --source . --cwd clone --send m0.tar --send lean.tar
# --send set.tar -- bash benchmarks/one_stage/pod.sh). The inputs are `git archive`s of M0's tree and the Lean
# verifier's tree at pinned commits, and one captured input set. Builds M0's `flock-circuit` (CPU, M0's own
# 60-circuit.sh MODE=build) and Lean's `flock-verify` (lake build, no dependencies) once per pod, then runs a0.py.
# env: K=16 DELTA=0.01 BIND=1 (run A1's M0 half: the statement binds the partition, program and unit indices)
#      RUN=a3 (one audit over every template set in set.tar, layer LAYER=0, K=128; DRAW_FILE=1 serves the verifier's Lean draw)
#      RUN=a2 (served.tar from vllm-serving-commit: registration.json, pub-N.bin, inst-N.bin, index.json; K=256;
#              UNBIND=1: run the session on re-headered copies for M0's unbound statement, m0.tar then eb90718f,
#              m0b.tar the bound tree e51e2b86 for the verifier's checks of serving's files)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uxo pipefail
REPO=$(pwd); I=$RESEARCH_RUN_DIR/inputs; O=$RESEARCH_RUN_DIR/out; W=/workspace/one-stage; mkdir -p $O $W/bin
nproc; free -g | head -2
M0SHA=$(sha256sum $I/m0.tar | cut -c1-16)
LEANSHA=$(sha256sum $I/lean.tar | cut -c1-16)
rm -rf $W/m0 $W/lean $W/set && mkdir -p $W/m0 $W/lean $W/set
tar -xf $I/m0.tar -C $W/m0 && tar -xf $I/lean.tar -C $W/lean && tar -xf $I/set.tar -C $W/set || exit 1
SET=$(dirname $(find $W/set -name manifest.json | head -1))
sha256sum $I/*.tar | tee $O/inputs.sha256

# Lean's flock-verify, in the background while M0 builds
(
  command -v lake >/dev/null || [ -x $HOME/.elan/bin/lake ] || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
  export PATH=$HOME/.elan/bin:$PATH
  if [ "$(cat $W/bin/lean.sha 2>/dev/null)" != "$LEANSHA" ]; then
    cd $W/lean/backends/flock/verifier/lean && lake build flock-verify && cp .lake/build/bin/flock-verify $W/bin/ && echo $LEANSHA > $W/bin/lean.sha
  fi
) > $O/build-lean.log 2>&1 &
LP=$!

# M0's flock-circuit, CPU build
if [ "$(cat $W/bin/m0.sha 2>/dev/null)" != "$M0SHA" ]; then
  (cd $W/m0 && RESEARCH_RUN_DIR=$O/m0 MODE=build GPU=0 bash verity/ml/flock/pod/60-circuit.sh) > $O/build-m0.log 2>&1 || { tail -40 $O/build-m0.log; exit 1; }
  cp /workspace/flock-gpu-link/flock/target/release/flock-circuit $W/bin/ && echo $M0SHA > $W/bin/m0.sha
fi
wait $LP || { tail -40 $O/build-lean.log; exit 1; }
[ -x $W/bin/flock-verify ] || { tail -40 $O/build-lean.log; exit 1; }
# the drivers' verifier of record (one_stage.verifier) builds the sampled-proofs Lean in this tree on its first check
export PATH=$HOME/.elan/bin:$PATH
sha256sum $W/bin/flock-circuit $W/bin/flock-verify | tee $O/binaries.sha256

[ -x $HOME/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
[ -x $W/py/bin/python ] || { $HOME/.local/bin/uv venv -q --python 3.12 $W/py && $HOME/.local/bin/uv pip install -q --python $W/py/bin/python numpy blake3; }
# M0's import roots in its own tree: the layout with verity at the root, else the one before it
m0path() { if [ -f $1/packages/verity/pyproject.toml ]; then echo $1/packages/verity/src:$1/backends/numerical/python:$1/verity/ml/flock/python:$1/integrations/vllm
           else echo $1:$1/catalog:$1/kernels:$1/experimental:$1/benchmarks/numerical/python:$1/verity/ml/flock/python:$1/integrations/vllm; fi; }
M0PATH=$(m0path $W/m0)
export RAYON_NUM_THREADS=$(nproc)
export PYTHONPATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/integrations/vllm
if [ "${RUN:-a0}" = a2 ]; then
  rm -rf $W/served && mkdir -p $W/served && tar -xf $I/served.tar -C $W/served || exit 1
  SERVED=$(dirname $(find $W/served -name registration.json | head -1))
  BOUND=$M0PATH
  if [ -n "${UNBIND:-}" ]; then
    rm -rf $W/m0b && mkdir -p $W/m0b && tar -xf $I/m0b.tar -C $W/m0b || exit 1
    BOUND=$(m0path $W/m0b)
  fi
  $W/py/bin/python benchmarks/one_stage/a2.py --served $SERVED --template-set $SET --m0-pythonpath $BOUND --bin $W/bin/flock-circuit \
    --lean $W/bin/flock-verify --out $O/a2 --k ${K:-256} ${DELTA:+--delta $DELTA} ${UNBIND:+--unbind --unbound-m0-pythonpath $M0PATH} ${LEGACY_PARTITION:+--legacy-partition} \
    ${STATEMENT:+--statement $STATEMENT} 2>&1 | tee $O/a2.log | cut -c1-4000
elif [ "${RUN:-a0}" = a3 ]; then
  $W/py/bin/python benchmarks/one_stage/a3.py --sets $W/set --m0-pythonpath $M0PATH --bin $W/bin/flock-circuit \
    --lean $W/bin/flock-verify --out $O/a3 --layer ${LAYER:-0} --k ${K:-128} ${DELTA:+--delta $DELTA} ${DRAW_FILE:+--draw-file} \
    ${STATEMENT:+--statement $STATEMENT} 2>&1 | tee $O/a3.log | cut -c1-4000
else
  $W/py/bin/python benchmarks/one_stage/a0.py --set $SET --m0-pythonpath $M0PATH --bin $W/bin/flock-circuit \
    --lean $W/bin/flock-verify --out $O/a0 --k ${K:-16} --delta ${DELTA:-0.01} ${BIND:+--bind} ${STATEMENT:+--statement $STATEMENT} 2>&1 | tee $O/a0.log | cut -c1-4000
fi
exit ${PIPESTATUS[0]}
