#!/usr/bin/env bash
# Measured VU run for `research run --on vy-cpu2 --stage gkr.vu`: builds the crate, exports the chained witness of
# bench-instances/v1 ids [LO, LO+B) (cached under $EXPORT_ROOT, keyed by tier/range), proves + verifies the B VUs
# and writes $RESEARCH_RUN_DIR/result.json per verity_numerical.bench.contract (instances ref, overhead).
#   run_measure_vu.sh B [LO]                         positives (tier vu-k1536)
#   NEGATIVES=1 run_measure_vu.sh                    every vu-k1536-neg instance must be rejected
# Env: INSTANCES_ROOT (built dataset), EXPORT_ROOT (witness cache), VERITY_GKR_THREADS (default 12), PROCS (export).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export PATH=$HOME/.cargo/bin:$HOME/.local/bin:$PATH
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/workspace/target-gkr}
export VERITY_GKR_THREADS=${VERITY_GKR_THREADS:-12}
PROCS=${PROCS:-12}
RUN_DIR=${RESEARCH_RUN_DIR:-.}
INSTANCES_ROOT=${INSTANCES_ROOT:-/workspace/bench-instances/v1}
EXPORT_ROOT=${EXPORT_ROOT:-/workspace/vu-export}
(cd "$HERE" && cargo build --release -j12 2>&1 | tail -1)
export RUSTC_VERSION
RUSTC_VERSION=$(rustc --version)
COMMIT=${GKR_COMMIT:-$(git -C "$HERE" rev-parse --short HEAD 2>/dev/null || echo unknown)}
UV="uv run --package verity-numerical python"
if [ "${NEGATIVES:-0}" = "1" ]; then
    EXP=$EXPORT_ROOT/vu-k1536-neg
    [ -f "$EXP/manifest.json" ] || $UV -m verity_numerical.gkr_export.vu export --root "$INSTANCES_ROOT" --tier vu-k1536-neg --out "$EXP" --procs "$PROCS"
    cp "$EXP/manifest.json" "$RUN_DIR/export_manifest.json"
    echo "commit=$COMMIT rustc=$RUSTC_VERSION threads=$VERITY_GKR_THREADS negatives"
    "$CARGO_TARGET_DIR/release/verity-gkr" negatives-vu --dir "$EXP" | tee "$RUN_DIR/negatives_vu.txt"
    exit 0
fi
B=${1:?VUs}
LO=${2:-0}
HI=$((LO + B))
EXP=$EXPORT_ROOT/vu-k1536-$LO-$HI
if [ ! -f "$EXP/manifest.json" ]; then
    # export a superset once (the whole tier when B is large); the prover takes the first B VUs
    $UV -m verity_numerical.gkr_export.vu export --root "$INSTANCES_ROOT" --tier vu-k1536 --lo "$LO" --hi "$HI" --out "$EXP" --procs "$PROCS"
fi
cp "$EXP/manifest.json" "$RUN_DIR/export_manifest.json"
echo "commit=$COMMIT rustc=$RUSTC_VERSION threads=$VERITY_GKR_THREADS B=$B lo=$LO"
"$CARGO_TARGET_DIR/release/verity-gkr" run-vu --dir "$EXP" --vus "$B" --out "$RUN_DIR/result.json" --commit "$COMMIT"
$UV -m verity_numerical.gkr_export.vu finalize --result "$RUN_DIR/result.json" --export-manifest "$RUN_DIR/export_manifest.json" ${RESEARCH_RUN_ID:+--run-id "$RESEARCH_RUN_ID"}
sha256sum "$EXP"/*.bin "$EXP"/*.txt > "$RUN_DIR/inputs.sha256"
