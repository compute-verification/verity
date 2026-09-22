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
# dataset), EXPORT_ROOT (witness cache), VERITY_GKR_THREADS (default 12), PROCS (export), RTT_MS (emulated network
# round-trip time for the latency axis: `--rtt-ms`, default 0), LOGUP (per-table | union | both: the lookup argument,
# PROTOCOL.md 15; `both` proves the same B twice in one run -- per-table into result_per_table.json, union into
# result.json -- so the two are matched on box, binary and instances).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export PATH=$HOME/.cargo/bin:$HOME/.local/bin:$PATH
FIELD=${FIELD:-goldilocks}
FEATURES=""
SUFFIX=""
if [ "$FIELD" = "babybear" ]; then FEATURES="--features babybear"; SUFFIX="-bb"; fi
VARIANT=${VARIANT:-v1}
VFLAG=""
# v2: the same crate, but its own target dir (another lane rebuilding /workspace/target-gkr from an older source
# mid-run replaced the binary once: r20260922-074323-12f8) and its own export cache
if [ "$VARIANT" = "v2" ]; then VFLAG="--variant v2"; SUFFIX="$SUFFIX-v2"; fi
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/workspace/target-gkr$SUFFIX}
export VERITY_GKR_THREADS=${VERITY_GKR_THREADS:-12}
PROCS=${PROCS:-12}
RUN_DIR=${RESEARCH_RUN_DIR:-.}
LOGUP=${LOGUP:-per-table}
LFLAG="--logup $LOGUP"; [ "$LOGUP" = "both" ] && LFLAG="--logup per-table"
INSTANCES_ROOT=${INSTANCES_ROOT:-/workspace/bench-instances/v1}
EXPORT_ROOT=${EXPORT_ROOT:-/workspace/vu-export$SUFFIX}
# shellcheck disable=SC2086
(cd "$HERE" && cargo build --release -j12 $FEATURES 2>&1 | tail -1)
export RUSTC_VERSION
RUSTC_VERSION=$(rustc --version)
COMMIT=${GKR_COMMIT:-$(git -C "$HERE" rev-parse --short HEAD 2>/dev/null || echo unknown)}
# vy-cpu has a stale `uv` (no `run` subcommand) ahead of astral's on PATH: pick one that works, else install it
UVBIN=""
for c in "$HOME/.local/bin/uv" "/workspace/uvbin/uv" "$(command -v uv || true)"; do
    if [ -n "$c" ] && [ -x "$c" ] && "$c" run --help >/dev/null 2>&1; then UVBIN=$c; break; fi
done
if [ -z "$UVBIN" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
    UVBIN=$HOME/.local/bin/uv
fi
UV="$UVBIN run --package verity-numerical python"
BIN=$CARGO_TARGET_DIR/release/verity-gkr
FX=$HERE/fixtures
[ "$FIELD" = "babybear" ] && FX=$FX/babybear
[ "$VARIANT" = "v2" ] && FX=$FX/v2
echo "field=$FIELD variant=$VARIANT logup=$LOGUP commit=$COMMIT rustc=$RUSTC_VERSION threads=$VERITY_GKR_THREADS"
if [ "${GATE:-0}" = "1" ]; then
    # shellcheck disable=SC2086
    (cd "$HERE" && cargo test --release -j12 $FEATURES 2>&1 | grep -E "^test result|FAILED|panicked" | tee "$RUN_DIR/cargo_test.txt")
    if [ "$VARIANT" = "v2" ]; then
        # the v2 transition-unit negatives are generated (not stored): every hint / lookup-output / state mutation
        $UV -m verity_numerical.gkr_export.v2 negatives --out "$RUN_DIR/v2neg" | tee "$RUN_DIR/negatives_v2_gen.txt" | tail -3
        # shellcheck disable=SC2086
        "$BIN" negatives "$RUN_DIR/v2neg" $LFLAG | tee "$RUN_DIR/negatives_unit.txt" | tail -1
        # shellcheck disable=SC2086
        "$BIN" negatives-vu --dir "$RUN_DIR/v2neg/vu" $LFLAG | tee "$RUN_DIR/negatives_vu_synth.txt" | tail -1
        if [ "$LOGUP" = "both" ]; then
            "$BIN" negatives "$RUN_DIR/v2neg" --logup union | tee "$RUN_DIR/negatives_unit_union.txt" | tail -1
            "$BIN" negatives-vu --dir "$RUN_DIR/v2neg/vu" --logup union | tee "$RUN_DIR/negatives_vu_synth_union.txt" | tail -1
        fi
        rm -rf "$RUN_DIR/v2neg"
    else
        # shellcheck disable=SC2086
        "$BIN" negatives "$FX" $LFLAG | tee "$RUN_DIR/negatives_unit.txt" | tail -1
    fi
    if [ "$FIELD" = "babybear" ]; then
        (cd "$HERE/.." && $UV "$HERE/tools/battery_adapter.py" --bin "$BIN" --out "$RUN_DIR/battery.json" --work "$RUN_DIR/battery_work" | tee "$RUN_DIR/battery.txt" | tail -14)
        rm -rf "$RUN_DIR/battery_work"
    fi
    exit 0
fi
# a box that has never run the VU tiers (vy-cpu) lacks the frozen instance set: rebuild it from the recipe and check
# it against the pinned manifest (bench-instances/v1 README) before any export
if [ ! -f "$INSTANCES_ROOT/vu-k1536-neg.index.json" ] || [ ! -f "$INSTANCES_ROOT/vu-k1536.index.json" ]; then
    mkdir -p "$INSTANCES_ROOT"
    cp -R "$HERE/../../fixtures/bench-instances/v1/." "$INSTANCES_ROOT/"      # seeds, manifest, indices, y words
    $UV -m verity_numerical.bench.instances build --out "$INSTANCES_ROOT" --procs "$PROCS" 2>&1 | tail -2
    $UV -m verity_numerical.bench.instances verify --root "$INSTANCES_ROOT" 2>&1 | tail -2
fi
cache_ok() {  # cache_ok DIR: a manifest of the right variant (a v1 export once landed in the v2 cache: r20260922-075822-5076)
    [ -f "$1/manifest.json" ] || return 1
    if grep -q '"variant"' "$1/manifest.json"; then grep -q "\"variant\": \"$VARIANT\"" "$1/manifest.json"; else [ "$VARIANT" = "v1" ]; fi
}
if [ "${NEGATIVES:-0}" = "1" ]; then
    EXP=$EXPORT_ROOT/vu-k1536-neg
    cache_ok "$EXP" || { rm -rf "$EXP"; $UV -m verity_numerical.gkr_export.vu export --root "$INSTANCES_ROOT" --tier vu-k1536-neg --out "$EXP" --procs "$PROCS" --field "$FIELD" $VFLAG; }
    cp "$EXP/manifest.json" "$RUN_DIR/export_manifest.json"
    echo "negatives"
    # shellcheck disable=SC2086
    "$BIN" negatives-vu --dir "$EXP" $LFLAG | tee "$RUN_DIR/negatives_vu.txt"
    [ "$LOGUP" = "both" ] && "$BIN" negatives-vu --dir "$EXP" --logup union | tee "$RUN_DIR/negatives_vu_union.txt"
    exit 0
fi
export_range() {  # export_range LO HI -> prints the export dir (cached)
    local exp=$EXPORT_ROOT/vu-k1536-$1-$2
    if ! cache_ok "$exp"; then
        rm -rf "$exp"
        $UV -m verity_numerical.gkr_export.vu export --root "$INSTANCES_ROOT" --tier vu-k1536 --lo "$1" --hi "$2" --out "$exp" --procs "$PROCS" --field "$FIELD" $VFLAG >&2
    fi
    echo "$exp"
}
if [ "${TRANSCRIPTS:-0}" = "1" ]; then
    EXP=$(export_range 0 64)
    for b in 4 64; do
        # shellcheck disable=SC2086
        "$BIN" dump-transcript --dir "$EXP" --vus "$b" --out "$RUN_DIR/transcript_vu${b}.json" $LFLAG
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
RTT_MS=${RTT_MS:-0}   # emulated network round-trip time (ms): the prover sleeps it at every critical-path challenge
echo "rtt_ms=$RTT_MS"
if [ "$LOGUP" = "both" ]; then
    "$BIN" run-vu --dir "$EXP" --vus "$B" --out "$RUN_DIR/result_per_table.json" --commit "$COMMIT" --rtt-ms "$RTT_MS" --logup per-table
    $UV -m verity_numerical.gkr_export.vu finalize --result "$RUN_DIR/result_per_table.json" --export-manifest "$RUN_DIR/export_manifest.json" ${RESEARCH_RUN_ID:+--run-id "$RESEARCH_RUN_ID"}
    "$BIN" run-vu --dir "$EXP" --vus "$B" --out "$RUN_DIR/result.json" --commit "$COMMIT" --rtt-ms "$RTT_MS" --logup union
else
    # shellcheck disable=SC2086
    "$BIN" run-vu --dir "$EXP" --vus "$B" --out "$RUN_DIR/result.json" --commit "$COMMIT" --rtt-ms "$RTT_MS" $LFLAG
fi
$UV -m verity_numerical.gkr_export.vu finalize --result "$RUN_DIR/result.json" --export-manifest "$RUN_DIR/export_manifest.json" ${RESEARCH_RUN_ID:+--run-id "$RESEARCH_RUN_ID"}
sha256sum "$EXP"/*.bin "$EXP"/*.txt > "$RUN_DIR/inputs.sha256"
