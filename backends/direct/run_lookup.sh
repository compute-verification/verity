#!/usr/bin/env bash
# The b-lookup lane's pod script (PROTOCOL.md section 14): build, the lookup tests (incl. the pod-sized ignored
# round trip + mutations), the layout, then the K=1536 VU benchmark in the LogUp realisation.
#   bash backends/direct/run_lookup.sh babybear [VUS] [QUERIES] [--skip-tests] [--profile]
# Pod etiquette (vy-cpu2 is shared): 12 build jobs, RAYON_NUM_THREADS=12.
set -euo pipefail
FIELD=${1:-babybear}
VUS=${2:-4096}
QUERIES=${3:-100}
shift 3 || true
SKIP_TESTS=0
PROFILE=0
EXT=4
for a in "$@"; do
  case "$a" in
    --skip-tests) SKIP_TESTS=1 ;;
    --profile) PROFILE=1 ;;
    --ext5) EXT=5 ;;
  esac
done
ROOT=${INSTANCES_ROOT:-/workspace/bench-instances/v1}
export PATH="$HOME/.cargo/bin:/usr/local/bin:$PATH"
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/workspace/cargo-target-b-chain}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-12}
export VERITY_RUSTC="$(rustc --version)"
cd backends/direct
echo "== build ($FIELD, $VUS VUs, $QUERIES queries, threads=$RAYON_NUM_THREADS, commit=${VERITY_COMMIT:-unknown})"
cargo build --release -j12 2>&1 | tail -2
BIN="$CARGO_TARGET_DIR/release/direct-unit"
if [ "$SKIP_TESTS" = 0 ]; then
  echo "== lookup tests (cargo test --release --test lookup, incl. --ignored round trip + mutations at 2^17 rows)"
  cargo test --release -j12 --test lookup -- --include-ignored 2>&1 | grep -E "^test |test result|FAILED|panicked" || true
fi
echo "== layout"
"$BIN" chain-layout --field "$FIELD" --lookup
if [ "$FIELD" = "babybear" ]; then
  echo "== red-team battery in the LogUp realisation (witness-side membership + row-local constraints incl. helpers)"
  ( cd ../.. && UV_CACHE_DIR=${UV_CACHE_DIR:-/workspace/uv-cache} uv run --frozen --package verity-numerical \
      python backends/direct/tools/battery_adapter.py --bin "$BIN" --extras --lookup --out "${RESEARCH_RUN_DIR:-.}/battery_babybear_lookup.json" | grep -v "^  rejected" ) || echo "battery (lookup) FAILED"
fi
echo "== vu-bench --lookup"
if [ "$PROFILE" = 1 ] && command -v perf >/dev/null; then
  perf record -F 200 -g -o "${RESEARCH_RUN_DIR:-.}/perf.data" -- "$BIN" vu-bench --lookup --ext "$EXT" --root "$ROOT" --field "$FIELD" --vus "$VUS" --queries "$QUERIES"
  perf report -i "${RESEARCH_RUN_DIR:-.}/perf.data" --no-children --percent-limit 1 --stdio 2>/dev/null | head -80 > "${RESEARCH_RUN_DIR:-.}/perf_report.txt" || true
  rm -f "${RESEARCH_RUN_DIR:-.}/perf.data"
else
  "$BIN" vu-bench --lookup --ext "$EXT" --root "$ROOT" --field "$FIELD" --vus "$VUS" --queries "$QUERIES"
fi
