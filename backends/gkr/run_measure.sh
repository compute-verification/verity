#!/usr/bin/env bash
# Measured run for `research run --on vy-cpu --stage gkr.unit`: builds the crate in its own target dir, proves and
# verifies B transition units from the shipped witness (--send honest.bin circuit.txt -> $RESEARCH_RUN_DIR/inputs/),
# writes $RESEARCH_RUN_DIR/result.json per verity_numerical.bench.contract.
#   run_measure.sh B
set -euo pipefail
B=${1:?units}
HERE=$(cd "$(dirname "$0")" && pwd)
export PATH=$HOME/.cargo/bin:$PATH
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/workspace/target-gkr}
RUN_DIR=${RESEARCH_RUN_DIR:-.}
CIRC=${CIRCUIT:-$RUN_DIR/inputs/circuit.txt}
WIT=${WITNESS:-$RUN_DIR/inputs/honest.bin}
(cd "$HERE" && cargo build --release 2>&1 | tail -1)
export RUSTC_VERSION
RUSTC_VERSION=$(rustc --version)
echo "commit=${GKR_COMMIT:-unknown} rustc=$RUSTC_VERSION threads=${VERITY_GKR_THREADS:-auto} B=$B"
"$CARGO_TARGET_DIR/release/verity-gkr" run --circuit "$CIRC" --witness "$WIT" --units "$B" \
    --out "$RUN_DIR/result.json" --commit "${GKR_COMMIT:-unknown}"
sha256sum "$CIRC" "$WIT" > "$RUN_DIR/inputs.sha256"
