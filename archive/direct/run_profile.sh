#!/usr/bin/env bash
# Profile the two realisations side by side with Plonky3's tracing spans (PROTOCOL.md section 14.6):
#   bash archive/direct/run_profile.sh babybear [VUS] [QUERIES] [MODES="limbs lookup v2"]
# Runs `vu-bench` (2-bit limbs) and `vu-bench --lookup` (LogUp) once each with RUST_LOG=debug; the span close
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
cd archive/direct
echo "== build ($FIELD, $VUS VUs, $QUERIES queries, threads=$RAYON_NUM_THREADS)"
cargo build --release -j12 2>&1 | tail -2
BIN="$CARGO_TARGET_DIR/release/direct-unit"
for mode in ${MODES:-limbs lookup}; do
  flag=""; [ "$mode" = lookup ] && flag="--lookup"; [ "$mode" = v2 ] && flag="--v2"
  echo "== vu-bench $flag (RUST_LOG=debug)"
  RUST_LOG=debug "$BIN" vu-bench $flag --root "$ROOT" --field "$FIELD" --vus "$VUS" --queries "$QUERIES" \
    2> "$OUT/spans_$mode.raw" | grep -E "^field=|^lookup stages"
  echo "-- spans ($mode), busy time summed per span path (raw in spans_$mode.raw):"
  python3 - "$OUT/spans_$mode.raw" <<'EOF' || true
import re, sys, collections
tot = collections.Counter(); cnt = collections.Counter()
unit = {"ns": 1e-9, "\u00b5s": 1e-6, "us": 1e-6, "ms": 1e-3, "s": 1.0}
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    line = re.sub(r"\x1b\[[0-9;]*m", "", line)
    m = re.search(r"(INFO|DEBUG) (.*?): close time\.busy=([0-9.]+)(ns|\u00b5s|us|ms|s)\b", line)
    if not m:
        continue
    path = re.sub(r"\{[^}]*\}", "", m.group(2))
    tot[path] += float(m.group(3)) * unit[m.group(4)]; cnt[path] += 1
for k, v in sorted(tot.items(), key=lambda kv: -kv[1])[:40]:
    print(f"{v:9.2f} s  x{cnt[k]:<5d} {k}")
EOF
done
