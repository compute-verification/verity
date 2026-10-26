#!/usr/bin/env bash
# `verity example toy`: every window, one run of the torch loop (examples/toy, a W8A8 MLP on CPU) audited
# by every protocol's leg (verity_all_protocols.table: legs/<component>.py), each window's report in $OUT/w<W>/report.json and
# $OUT/reports.jsonl, and the monitoring-period report over the windows so far in $OUT/monitoring-period.json
# (verity_all_protocols.runner, .monitoring). Builds what the legs run first, as benchmarks/prove_path/run.sh does: M0's
# flock-circuit, Lean's flock-verify and flock-firewall, and the sampled-proofs verifier of record, in scratch and the clone.
# Its arguments go to the runner (--windows N, --start I, --jobs J, --tamper W,..., --pouw h1); ALLP_FLOCK_WORK=DIR keeps
# M0's build across runs.
#   research run --on vy-nebius-cpu-4 --project verity --source . --cwd clone --mem-gb 64 -- \
#     bash examples/toy/run.sh --windows 4 --jobs 4 --tamper 3
set -uxo pipefail
REPO=$(pwd); O=${OUT:-$RESEARCH_RUN_DIR/out}; W=$(mktemp -d); trap 'rm -rf "$W"' EXIT; mkdir -p $O
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.elan/bin:$PATH
{ nproc; free -g | head -2; git rev-parse HEAD HEAD:verity/ml/flock; } | tee $O/machine.txt
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
uv sync --all-packages --extra torch-cpu > $O/sync.log 2>&1 || { tail -20 $O/sync.log; exit 5; }
PY=$REPO/.venv/bin/python
(cd verity/core/service/flock && lake build flock-verify flock-firewall) > $O/build-lean.log 2>&1 &
LP=$!
mkdir -p $O/m0-build
FW=${ALLP_FLOCK_WORK:-$W}
RESEARCH_RUN_DIR=$O/m0-build FLOCK_WORK=$FW MODE=build GPU=0 bash verity/ml/flock/pod/60-circuit.sh > $O/m0-build/build.log 2>&1 \
  || { tail -40 $O/m0-build/build.log; exit 6; }
BIN=$FW/flock-gpu-link/flock/target/release/flock-circuit
wait $LP || { tail -40 $O/build-lean.log; exit 7; }
LEAN=$REPO/verity/core/service/flock/.lake/build/bin/flock-verify; FWL=${LEAN%/*}/flock-firewall
(cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier) > $O/build-sampled-proofs.log 2>&1 || { tail -40 $O/build-sampled-proofs.log; exit 7; }
sha256sum $BIN $LEAN $FWL | tee $O/binaries.sha256
M0PATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/ml/flock/python:$REPO/integrations/vllm
$PY -c 'import json, sys; print(json.dumps(dict(zip(["flock-circuit", "flock-verify", "flock-firewall", "m0-pythonpath"], sys.argv[1:]))))' \
  $BIN $LEAN $FWL $M0PATH > $O/tools.json
export RAYON_NUM_THREADS=$(nproc) RUST_MIN_STACK=$((64 << 20))
PYTHONPATH=$REPO/benchmarks/all_protocols:$REPO/examples/toy $PY -m verity_all_protocols.runner --out $O --tools $O/tools.json --scratch $W "$@"
