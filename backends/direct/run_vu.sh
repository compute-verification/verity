#!/usr/bin/env bash
# The b-chain lane's pod script: build, the unit negatives, the 52 VU negatives, then the K=1536 VU benchmark.
#   bash backends/direct/run_vu.sh goldilocks|babybear [VUS] [QUERIES]
# Pod etiquette (vy-cpu2 is shared): 12 build jobs, RAYON_NUM_THREADS=12.
set -euo pipefail
FIELD=${1:-goldilocks}
VUS=${2:-4096}
QUERIES=${3:-100}
ROOT=${INSTANCES_ROOT:-/workspace/bench-instances/v1}
export PATH="$HOME/.cargo/bin:/usr/local/bin:$PATH"
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/workspace/cargo-target-b-chain}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-12}
export VERITY_RUSTC="$(rustc --version)"
cd backends/direct
echo "== build ($FIELD, $VUS VUs, $QUERIES queries, threads=$RAYON_NUM_THREADS, commit=${VERITY_COMMIT:-unknown})"
cargo build --release -j12 2>&1 | tail -2
BIN="$CARGO_TARGET_DIR/release/direct-unit"
if [ "${SKIP_GATE:-0}" != "1" ]; then   # SKIP_GATE=1: a timing-only rerun (the gate passed on the same commit before)
echo "== unit negatives + chain tests (cargo test --release)"
cargo test --release -j12 2>&1 | grep -E "test result|FAILED|panicked" || true
echo "== layout"
"$BIN" chain-layout --field "$FIELD"
echo "== vu-neg (52 vu-k1536-neg)"
"$BIN" vu-neg --root "$ROOT" --field "$FIELD" | tail -5
fi
RTT_MS=${RTT_MS:-0}   # emulated network round-trip time (ms) for the latency axis: the prover sleeps it at every challenge
echo "== vu-bench (rtt_ms=$RTT_MS)"
"$BIN" vu-bench --root "$ROOT" --field "$FIELD" --vus "$VUS" --queries "$QUERIES" --rtt-ms "$RTT_MS"
