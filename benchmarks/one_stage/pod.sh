#!/usr/bin/env bash
# One-stage audit, run A0, on a CPU pod (research run --on POD --source . --cwd source --send m0.tar --send lean.tar
# --send set.tar -- bash benchmarks/one_stage/pod.sh). The inputs are `git archive`s of M0's tree (PR #83) and the Lean
# verifier's tree (PR #113) at pinned commits, and one captured input set. Builds M0's `flock-circuit` (CPU, M0's own
# 60-circuit.sh MODE=build) and Lean's `flock-verify` (lake build, no dependencies) once per pod, then runs a0.py.
# env: K=16 DELTA=0.01
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
  (cd $W/m0 && RESEARCH_RUN_DIR=$O/m0 MODE=build GPU=0 bash backends/flock/pod/60-circuit.sh) > $O/build-m0.log 2>&1 || { tail -40 $O/build-m0.log; exit 1; }
  cp /workspace/flock-gpu-link/flock/target/release/flock-circuit $W/bin/ && echo $M0SHA > $W/bin/m0.sha
fi
wait $LP || { tail -40 $O/build-lean.log; exit 1; }
[ -x $W/bin/flock-verify ] || { tail -40 $O/build-lean.log; exit 1; }
sha256sum $W/bin/flock-circuit $W/bin/flock-verify | tee $O/binaries.sha256

[ -x $HOME/.local/bin/uv ] || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
[ -x $W/py/bin/python ] || { $HOME/.local/bin/uv venv -q --python 3.12 $W/py && $HOME/.local/bin/uv pip install -q --python $W/py/bin/python numpy blake3; }
M0PATH=$W/m0/packages/verity/src:$W/m0/backends/numerical/python:$W/m0/backends/flock/python:$W/m0/integrations/vllm
export RAYON_NUM_THREADS=$(nproc)
PYTHONPATH=$REPO/packages/verity/src:$REPO/backends/numerical/python:$REPO/integrations/vllm:$REPO/protocols/one_stage \
  $W/py/bin/python benchmarks/one_stage/a0.py --set $SET --m0-pythonpath $M0PATH --bin $W/bin/flock-circuit \
  --lean $W/bin/flock-verify --out $O/a0 --k ${K:-16} --delta ${DELTA:-0.01} 2>&1 | tee $O/a0.log | cut -c1-4000
exit ${PIPESTATUS[0]}
