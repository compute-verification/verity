#!/usr/bin/env bash
# Measured VU run for `research run --on vy-cpu2 --stage gkr.vu`: builds the crate, exports the chained witness of
# bench-instances/v1 ids [LO, LO+B) (cached under $EXPORT_ROOT, keyed by tier/range), proves + verifies the B VUs
# and writes $RESEARCH_RUN_DIR/result.json per verity_numerical.bench.contract (instances ref, overhead).
#   run_measure_vu.sh B [LO]                         positives (tier vu-k1536)
#   NEGATIVES=1 run_measure_vu.sh                    every vu-k1536-neg instance must be rejected
#   GATE=1 run_measure_vu.sh                         cargo test; fixtures negatives (unit + vu_synth); BabyBear: the 83-case battery
#   TRANSCRIPTS=1 run_measure_vu.sh                  dump-transcript for B=4 and B=64 into $RESEARCH_RUN_DIR (FIELD=babybear: the GPU lane's vectors)
# Env: FIELD (goldilocks | babybear: `--features babybear`, its own target dir and export cache), VARIANT (v1 | v2: the
# checker-v2 circuits of gkr_export.v2, PROTOCOL.md 14; its own export cache and fixtures/v2), INSTANCES_ROOT (built
# dataset), EXPORT_ROOT (witness cache), VERITY_GKR_THREADS (default 12), PROCS (export).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export PATH=$HOME/.cargo/bin:$HOME/.local/bin:$PATH
FIELD=${FIELD:-goldilocks}
FEATURES=""
SUFFIX=""
if [ "$FIELD" = "babybear" ]; then FEATURES="--features babybear"; SUFFIX="-bb"; fi
VARIANT=${VARIANT:-v1}
VFLAG=""
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/workspace/target-gkr$SUFFIX}
if [ "$VARIANT" = "v2" ]; then VFLAG="--variant v2"; SUFFIX="$SUFFIX-v2"; fi   # same binary, its own export cache
export VERITY_GKR_THREADS=${VERITY_GKR_THREADS:-12}
PROCS=${PROCS:-12}
RUN_DIR=${RESEARCH_RUN_DIR:-.}
INSTANCES_ROOT=${INSTANCES_ROOT:-/workspace/bench-instances/v1}
EXPORT_ROOT=${EXPORT_ROOT:-/workspace/vu-export$SUFFIX}
# shellcheck disable=SC2086
(cd "$HERE" && cargo build --release -j12 $FEATURES 2>&1 | tail -1)
export RUSTC_VERSION
RUSTC_VERSION=$(rustc --version)
COMMIT=${GKR_COMMIT:-$(git -C "$HERE" rev-parse --short HEAD 2>/dev/null || echo unknown)}
UV="uv run --package verity-numerical python"
BIN=$CARGO_TARGET_DIR/release/verity-gkr
FX=$HERE/fixtures
[ "$FIELD" = "babybear" ] && FX=$FX/babybear
[ "$VARIANT" = "v2" ] && FX=$FX/v2
echo "field=$FIELD variant=$VARIANT commit=$COMMIT rustc=$RUSTC_VERSION threads=$VERITY_GKR_THREADS"
if [ "${GATE:-0}" = "1" ]; then
    # shellcheck disable=SC2086
    (cd "$HERE" && cargo test --release -j12 $FEATURES 2>&1 | grep -E "^test result|FAILED|panicked" | tee "$RUN_DIR/cargo_test.txt")
    if [ "$VARIANT" = "v2" ]; then
        # the v2 transition-unit negatives are generated (not stored): every hint / lookup-output / state mutation
        $UV -m verity_numerical.gkr_export.v2 negatives --out "$RUN_DIR/v2neg" | tee "$RUN_DIR/negatives_v2_gen.txt" | tail -3
        "$BIN" negatives "$RUN_DIR/v2neg" | tee "$RUN_DIR/negatives_unit.txt" | tail -1
        "$BIN" negatives-vu --dir "$RUN_DIR/v2neg/vu" | tee "$RUN_DIR/negatives_vu_synth.txt" | tail -1
        rm -rf "$RUN_DIR/v2neg"
    else
        "$BIN" negatives "$FX" | tee "$RUN_DIR/negatives_unit.txt" | tail -1
    fi
    if [ "$FIELD" = "babybear" ]; then
        (cd "$HERE/.." && $UV "$HERE/tools/battery_adapter.py" --bin "$BIN" --out "$RUN_DIR/battery.json" --work "$RUN_DIR/battery_work" | tee "$RUN_DIR/battery.txt" | tail -14)
        rm -rf "$RUN_DIR/battery_work"
    fi
    exit 0
fi
if [ "${NEGATIVES:-0}" = "1" ]; then
    EXP=$EXPORT_ROOT/vu-k1536-neg
    [ -f "$EXP/manifest.json" ] || $UV -m verity_numerical.gkr_export.vu export --root "$INSTANCES_ROOT" --tier vu-k1536-neg --out "$EXP" --procs "$PROCS" --field "$FIELD" $VFLAG
    cp "$EXP/manifest.json" "$RUN_DIR/export_manifest.json"
    echo "negatives"
    "$BIN" negatives-vu --dir "$EXP" | tee "$RUN_DIR/negatives_vu.txt"
    exit 0
fi
export_range() {  # export_range LO HI -> prints the export dir (cached)
    local exp=$EXPORT_ROOT/vu-k1536-$1-$2
    if [ ! -f "$exp/manifest.json" ]; then
        $UV -m verity_numerical.gkr_export.vu export --root "$INSTANCES_ROOT" --tier vu-k1536 --lo "$1" --hi "$2" --out "$exp" --procs "$PROCS" --field "$FIELD" $VFLAG >&2
    fi
    echo "$exp"
}
if [ "${TRANSCRIPTS:-0}" = "1" ]; then
    EXP=$(export_range 0 64)
    for b in 4 64; do
        "$BIN" dump-transcript --dir "$EXP" --vus "$b" --out "$RUN_DIR/transcript_vu${b}.json"
    done
    cp "$EXP/manifest.json" "$RUN_DIR/export_manifest.json"
    sha256sum "$EXP"/*.bin "$EXP"/*.txt "$RUN_DIR"/transcript_*.json > "$RUN_DIR/inputs.sha256"
    exit 0
fi
B=${1:?VUs}
LO=${2:-0}
HI=$((LO + B))
# export a superset once (the whole tier when B is large); the prover takes the first B VUs
EXP=$(export_range "$LO" "$HI")
cp "$EXP/manifest.json" "$RUN_DIR/export_manifest.json"
echo "B=$B lo=$LO"
"$BIN" run-vu --dir "$EXP" --vus "$B" --out "$RUN_DIR/result.json" --commit "$COMMIT"
$UV -m verity_numerical.gkr_export.vu finalize --result "$RUN_DIR/result.json" --export-manifest "$RUN_DIR/export_manifest.json" ${RESEARCH_RUN_ID:+--run-id "$RESEARCH_RUN_ID"}
sha256sum "$EXP"/*.bin "$EXP"/*.txt > "$RUN_DIR/inputs.sha256"
