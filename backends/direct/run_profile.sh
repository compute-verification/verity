#!/usr/bin/env bash
# Profile the two realisations side by side with Plonky3's tracing spans (PROTOCOL.md section 14.6):
#   bash backends/direct/run_profile.sh babybear [VUS] [QUERIES]
# Runs `vu-bench` (2-bit limbs) and `vu-bench --lookup` (LogUp) once each with RUST_LOG=info; the span close
# events (commit to trace data / compute quotient polynomial / commit to quotient poly chunks / open, with the
# DFT and Merkle spans inside them) go to stderr and are summarised into spans_{limbs,lookup}.txt.
set -euo pipefail
FIELD=${1:-babybear}
VUS=${2:-4096}
QUERIES=${3:-166}
ROOT=${INSTANCES_ROOT:-/workspace/bench-instances/v1}
OUT=${RESEARCH_RUN_DIR:-.}
export PATH="$HOME/.cargo/bin:/usr/local/bin:$PATH"
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/workspace/cargo-target-b-chain}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-12}
export VERITY_RUSTC="$(rustc --version)"
cd backends/direct
echo "== build ($FIELD, $VUS VUs, $QUERIES queries, threads=$RAYON_NUM_THREADS)"
cargo build --release -j12 2>&1 | tail -2
BIN="$CARGO_TARGET_DIR/release/direct-unit"
for mode in limbs lookup; do
  flag=""; [ "$mode" = lookup ] && flag="--lookup"
  echo "== vu-bench $flag (RUST_LOG=info)"
  RUST_LOG=info "$BIN" vu-bench $flag --root "$ROOT" --field "$FIELD" --vus "$VUS" --queries "$QUERIES" \
    2> "$OUT/spans_$mode.raw" | grep -E "^field=|^lookup stages"
  # one line per span close: depth-agnostic name + busy time
  grep -oE '[a-z_ ]+(\{[^}]*\})?: close time\.busy=[0-9.]+(ms|s|µs)' "$OUT/spans_$mode.raw" \
    | sed -E 's/\{[^}]*\}//; s/: close time\.busy=/ /' > "$OUT/spans_$mode.txt" || true
  echo "-- spans ($mode), summed per name:"
  awk '{n=$NF; v=n; if (n ~ /ms$/) {sub(/ms/,"",v); v/=1000} else if (n ~ /µs$/) {sub(/µs/,"",v); v/=1e6} else {sub(/s/,"",v)}; $NF=""; s[$0]+=v; c[$0]++} END {for (k in s) printf "%9.2f s  x%-4d %s\n", s[k], c[k], k}' "$OUT/spans_$mode.txt" | sort -rn | head -40
  rm -f "$OUT/spans_$mode.raw"
done
