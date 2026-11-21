#!/usr/bin/env bash
# The programs the sampled-proofs step of the window steps runs, built where the step looks for them: the Lean executables
# verity-verify-proof, flock-zk-server and flock-lower-check in the Flock verifier package's .lake/build/bin, flock-tmpl-unit in
# $FLOCK_WORK/lean-bin, and the CPU flock-circuit in $FLOCK_WORK (pod/60-circuit.sh MODE=build).
#
#   bash benchmarks/one_stage/build_sampled_step.sh        in the tree; SERVE_ROOT and FLOCK_WORK as the runner has them
#
# Lean builds hold the tree's one Lake build lock, and two flock-circuit builds in one work directory break each other's
# crates, so they hold $FLOCK_WORK/.build.lock.
set -euo pipefail
TREE=${TREE:-$(cd "$(dirname "$0")/../.." && pwd)}
FW=${FLOCK_WORK:-${SERVE_ROOT:-/workspace/research/vllm-served-verdicts}/flock-work}
LOCK=$TREE/verity/circuits/difftest/.lake
export PATH=$HOME/.elan/bin:$PATH
command -v lake > /dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none
mkdir -p "$FW" "$LOCK"
cd "$TREE"
(
  flock 9
  (cd verity/protocols/proof/flock/verifier && lake build verity-verify-proof flock-zk-server flock-lower-check)
  rm -rf "$FW/lean-src"
  src=$(python3 verity/protocols/proof/flock/prover/pod/lake_closure.py copy verity/protocols/proof/flock/verifier "$FW/lean-src")
  cp tests/verity/protocols/proof/flock/prover/lean/TmplUnit.lean "$src/"
  printf '\n[[lean_exe]]\nname = "flock-tmpl-unit"\nroot = "TmplUnit"\n' >> "$src/lakefile.toml"
  (cd "$src" && lake build flock-tmpl-unit)
  mkdir -p "$FW/lean-bin" && cp "$src/.lake/build/bin/flock-tmpl-unit" "$FW/lean-bin/"
) 9> "$LOCK/harness-build.lock"
mkdir -p "$FW/build-log"
RESEARCH_RUN_DIR=$FW/build-log FLOCK_WORK=$FW MODE=build GPU=0 flock "$FW/.build.lock" bash verity/protocols/proof/flock/prover/pod/60-circuit.sh
ls -l "$FW/flock-gpu-link/flock/target/release/flock-circuit" "$FW/lean-bin/flock-tmpl-unit"
