#!/usr/bin/env bash
# The b-air-v2 lane's pod script (PROTOCOL.md section 15, V2_AIR.md): build, the v2 tests (incl. the pod-sized
# statement-bound round trip), the layout, the 52 VU negatives, the honest witness of every bench VU against the
# recorded accumulators, the red-team battery on the v2 AIR, then the K=1536 VU gate on BabyBear^EXT.
#   bash backends/direct/run_v2.sh [VUS] [QUERIES] [--ext5] [--skip-tests] [--skip-checks] [--bind-operands]
# Pod etiquette (vy-cpu2 is shared): 12 build jobs, RAYON_NUM_THREADS=12.
set -euo pipefail
VUS=${1:-4096}
QUERIES=${2:-100}
shift 2 || true
SKIP_TESTS=0
SKIP_CHECKS=0
EXT=4
BIND=""
NO_BENCH=0
for a in "$@"; do
  case "$a" in
    --skip-tests) SKIP_TESTS=1 ;;
    --skip-checks) SKIP_CHECKS=1 ;;
    --ext5) EXT=5 ;;
    --ext6) EXT=6 ;;
    --bind-operands) BIND="--bind-operands" ;;
    --no-bench) NO_BENCH=1 ;;
  esac
done
ROOT=${INSTANCES_ROOT:-/workspace/bench-instances/v1}
export PATH="$HOME/.cargo/bin:/usr/local/bin:$PATH"
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/workspace/cargo-target-b-chain}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-12}
export VERITY_RUSTC="$(rustc --version)"
OUT="${RESEARCH_RUN_DIR:-.}"
cd backends/direct
echo "== build (v2, BabyBear^$EXT, $VUS VUs, $QUERIES queries, threads=$RAYON_NUM_THREADS, commit=${VERITY_COMMIT:-unknown})"
cargo build --release -j12 2>&1 | tail -2
BIN="$CARGO_TARGET_DIR/release/direct-unit"
if [ "$SKIP_TESTS" = 0 ]; then
  echo "== v2 tests (cargo test --release --test air_v2, incl. --ignored statement-bound round trip at 2^19 rows)"
  cargo test --release -j12 --test air_v2 -- --include-ignored 2>&1 | grep -E "^test |test result|FAILED|panicked" || true
fi
echo "== layout"
"$BIN" chain-layout --v2 --ext "$EXT"
if [ "$SKIP_CHECKS" = 0 ]; then
  echo "== vu-neg --v2 (52 vu-k1536-neg + 44 controls)"
  "$BIN" vu-neg --v2 --root "$ROOT" --out "$OUT" | tail -3
  echo "== vu-honest --v2 ($VUS VUs against the recorded accumulators; exhaustive row-local evaluation of the first 64)"
  "$BIN" vu-honest --v2 --root "$ROOT" --vus "$VUS" --eval 64 --no-assert 2>&1 | tail -40
  echo "== red-team battery on the v2 AIR (83 + 6 extras + 87 controls)"
  ( cd ../.. && UV_CACHE_DIR=${UV_CACHE_DIR:-/workspace/uv-cache} uv run --frozen --package verity-numerical \
      python backends/direct/tools/battery_adapter.py --bin "$BIN" --extras --v2 --out "$OUT/battery_babybear_v2.json" | grep -v "^  rejected" ) || echo "battery (v2) FAILED"
fi
if [ "$NO_BENCH" = 1 ]; then exit 0; fi
echo "== vu-bench --v2 --ext $EXT $BIND"
if command -v /usr/bin/time >/dev/null; then
  /usr/bin/time -v "$BIN" vu-bench --v2 --ext "$EXT" --root "$ROOT" --vus "$VUS" --queries "$QUERIES" $BIND 2> "$OUT/time_v2_ext$EXT.txt" || { cat "$OUT/time_v2_ext$EXT.txt"; exit 1; }
  grep -E "Maximum resident|Elapsed" "$OUT/time_v2_ext$EXT.txt" || true
else
  "$BIN" vu-bench --v2 --ext "$EXT" --root "$ROOT" --vus "$VUS" --queries "$QUERIES" $BIND
fi
